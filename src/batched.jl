# Batched spectral radii for many parameter points at once (GPU or CPU threads).
#
# All points of a batch share the discretization (tableau, p, r) and the structure
# (D, K). Every O(N) operation runs in KernelAbstractions kernels, so the same code
# runs on `CPU()` (threads) and on `CUDABackend()`. Arrays use the point index as
# the FASTEST dimension: X[b, i], V[b, i, j], Mp[b, row, col, n], so neighbouring
# threads (= neighbouring points) access neighbouring memory. Design: GPU_DESIGN.md.

using KernelAbstractions
const KA = KernelAbstractions

# ---------------------------------------------------------------------------
# Problem / tableau description (isbits, device-callable)
# ---------------------------------------------------------------------------

"""
    BatchedLDDE{D, K}(A, B, tau, period)

Parametrised linear time-periodic DDE

    ẋ(t) = A(t, θ) x(t) + Σₖ Bₖ(t, θ) x(t − τₖ(t, θ)),   period T(θ),

for the batched solvers. All four callables must be device-compatible when used on a
GPU (plain top-level functions, StaticArrays, no allocation):

- `A(t, θ)      -> SMatrix{D, D}`
- `B(t, θ)      -> NTuple{K, SMatrix{D, D}}`
- `tau(t, θ)    -> NTuple{K, <:Real}` (the lags, constant or time-periodic)
- `period(θ)    -> T` (principal period of the coefficients)

`θ` is one element of the parameter vector passed to [`spectral_radii`](@ref), any
isbits value (e.g. an `SVector` or a `NamedTuple`).
"""
struct BatchedLDDE{D, K, FA, FB, FT, FP}
    A::FA
    B::FB
    tau::FT
    period::FP
end

BatchedLDDE{D, K}(A::FA, B::FB, tau::FT, period::FP) where {D, K, FA, FB, FT, FP} =
    BatchedLDDE{D, K, FA, FB, FT, FP}(A, B, tau, period)

# Non-dimensionalised wrapper: t̃ = ω₀ t, x̃ = S x (S = diag(s)). The monodromy of the
# rescaled system is S Φ S⁻¹ over the same period, so the Floquet multipliers are identical;
# only the magnitudes of the step blocks change (needed for Float16, helps Float32).
struct _ScaledA{F, D, T}; f::F; ω0::T; s::SVector{D, T}; end
struct _ScaledB{F, D, T}; f::F; ω0::T; s::SVector{D, T}; end
struct _ScaledTau{F, T}; f::F; ω0::T; end
struct _ScaledPeriod{F, T}; f::F; ω0::T; end
@inline _sim(M, s, ω0) = SMatrix{length(s), length(s)}(ntuple(Val(length(s) * length(s))) do idx
    i = (idx - 1) % length(s) + 1; j = (idx - 1) ÷ length(s) + 1
    s[i] * M[i, j] / (s[j] * ω0)
end)
(a::_ScaledA)(t, θ) = _sim(a.f(t / a.ω0, θ), a.s, a.ω0)
(b::_ScaledB)(t, θ) = map(M -> _sim(M, b.s, b.ω0), b.f(t / b.ω0, θ))
(τ::_ScaledTau)(t, θ) = map(x -> τ.ω0 * x, τ.f(t / τ.ω0, θ))
(T::_ScaledPeriod)(θ) = T.ω0 * T.f(θ)

"""
    rescale(prob::BatchedLDDE, ω0, s)

The same DDE in non-dimensional form: time `t̃ = ω0·t`, state `x̃ = diag(s)·x`. Floquet
multipliers (and hence ρ) are unchanged; the step blocks become O(1), which Float16 (range
6·10⁻⁵ … 6.5·10⁴) requires and Float32 benefits from. Typical choice for a mechanical
model with states [positions; velocities]: `ω0` = a natural angular frequency,
`s = [1, …, 1, 1/ω0, …, 1/ω0]`.
"""
function rescale(prob::BatchedLDDE{D, K}, ω0::Real, s::AbstractVector) where {D, K}
    w = Float64(ω0); sv = SVector{D, Float64}(s)
    return BatchedLDDE{D, K}(_ScaledA(prob.A, w, sv), _ScaledB(prob.B, w, sv),
                             _ScaledTau(prob.tau, w), _ScaledPeriod(prob.period, w))
end

"""
    LDDEProblem(bp::BatchedLDDE, θ)

The ordinary (CPU) [`LDDEProblem`](@ref) of one parameter point — used as the CPU
reference and for single-point analysis with the full `floquet_analysis` toolbox.
"""
function LDDEProblem(bp::BatchedLDDE{D, K}, θ) where {D, K}
    A_f = t -> SMatrix{D, D, Float64}(bp.A(t, θ))
    Bs = [DelayMX(t -> Float64(bp.tau(t, θ)[k]), t -> SMatrix{D, D, Float64}(bp.B(t, θ)[k])) for k in 1:K]
    return LDDEProblem{D, Float64}(ProportionalMX(A_f), Bs, Additive(t -> zero(SVector{D, Float64})))
end

"Device-side Butcher tableau + Lagrange continuous extension on the nodes {0, c, 1}."
struct DeviceTableau{S, T, S2, L}
    a::SMatrix{S, S, T, L}
    b::SVector{S, T}
    c::SVector{S, T}
    unodes::SVector{S2, T}     # unique interpolation nodes (padded)
    nunique::Int
    slot::SVector{S2, Int}     # slot (y_n, Y_1..Y_S, y_{n+1}) -> unique node index
    mult::SVector{S2, T}       # multiplicity of each unique node
end

function _device_tableau(tab::RKTableau{S}, ::Type{T}) where {S, T}
    raw = vcat(0.0, collect(tab.c), 1.0)
    u = sort(unique(raw)); nu = length(u)
    slot = [findfirst(==(x), u) for x in raw]
    mult = [count(==(k), slot) for k in 1:nu]
    S2 = S + 2
    ud = SVector{S2, T}(ntuple(i -> i <= nu ? T(u[i]) : zero(T), S2))
    md = SVector{S2, T}(ntuple(i -> i <= nu ? T(mult[i]) : one(T), S2))
    return DeviceTableau{S, T, S2, S * S}(SMatrix{S, S, T}(tab.a), SVector{S, T}(tab.b), SVector{S, T}(tab.c),
                                          ud, nu, SVector{S2, Int}(slot), md)
end

function DeviceTableau(tab::RKTableau{S}, ::Type{T}=Float64) where {S, T}
    tab.strategy == endpoint && error("batched path: the `endpoint` interpolation strategy is not supported")
    # The batched kernels implement the Lagrange extension only; refuse anything else
    d64 = _device_tableau(tab, Float64)
    for θ in (0.0, 0.13, 0.5, 0.77, 1.0)
        maximum(abs.(collect(tab.ce(θ)) .- collect(_ce_weights(d64, θ)))) < 1e-12 ||
            error("batched path: the continuous extension of this tableau is not the Lagrange " *
                  "interpolant on {0, c, 1} (e.g. the RK4 Hermite extension); use the CPU path")
    end
    return T === Float64 ? d64 : _device_tableau(tab, T)
end

@inline function _ce_weights(tab::DeviceTableau{S, T, S2}, θ) where {S, T, S2}
    return SVector{S2, T}(ntuple(Val(S2)) do i
        u = tab.slot[i]; xu = tab.unodes[u]; L = one(T)
        for v in 1:tab.nunique
            v == u && continue
            L *= (T(θ) - tab.unodes[v]) / (xu - tab.unodes[v])
        end
        L / tab.mult[u]
    end)
end

# ---------------------------------------------------------------------------
# Device storage of the step operators of one batch
# ---------------------------------------------------------------------------

"""
    BatchedOperators

Per-step transition blocks of a batch of parameter points (device arrays, point index
fastest): `Mp[b, row, col, n]`, `Md[b, row, col, s, k, n]`, delay block index
`Midx[b, s, k, n]` and interpolation weights `Wt[b, slot, s, k, n]`; plus the sweep
work buffer `Hist[b, :]` and per-point status flags (0 = ok, 1 = delayed lookup outside
the history window, i.e. r·h < max lag).
"""
struct BatchedOperators{D, S, K, T, A2, A4, A6, I4, A5, IV}
    Mp::A4
    Md::A6
    Midx::I4
    Wt::A5
    Hist::A2
    flag::IV
    p::Int
    r::Int
    nb::Int
end

bsize(::BatchedOperators{D, S}) where {D, S} = (S + 1) * D
state_size(op::BatchedOperators) = (op.r + 1) * bsize(op)

"Bytes per parameter point needed by operators, sweep buffer and a Krylov basis of `m+1` vectors."
function bytes_per_point(D, S, K, p, r, m, T)
    BS = (S + 1) * D; N = (r + 1) * BS
    nT = BS * D * p * (1 + S * K) + (S + 2) * S * K * p + (p + r + 1) * BS + N * (2m + 4)
    return nT * sizeof(T) + 4 * S * K * p
end

# ---------------------------------------------------------------------------
# Build kernel: one work item per (point, step)
# ---------------------------------------------------------------------------

@kernel function _build_kernel!(Mp, Md, Midx, Wt, flag, @Const(θs), prob, tab, p::Int, r::Int,
                                ::Val{D}, ::Val{S}, ::Val{K}) where {D, S, K}
    b, n = @index(Global, NTuple)
    T = promote_type(eltype(Mp), Float32)   # Float16 storage: build in Float32
    SD = S * D
    @inbounds begin
        θ = θs[b]
        h = T(prob.period(θ)) / p
        tn = (n - 1) * h
        As = ntuple(j -> SMatrix{D, D, T}(prob.A(tn + tab.c[j] * h, θ)), Val(S))

        # Stage matrix M = I − h (a ⊗ A_j), LU with partial pivoting (in place)
        M = MMatrix{SD, SD, T}(undef)
        for j in 1:S, i in 1:S, cd in 1:D, rd in 1:D
            M[(i-1)*D + rd, (j-1)*D + cd] = (i == j && rd == cd ? one(T) : zero(T)) - h * tab.a[i, j] * As[j][rd, cd]
        end
        piv = MVector{SD, Int}(undef)
        for k in 1:SD
            pk = k; mx = abs(M[k, k])
            for i in k+1:SD
                v = abs(M[i, k]); if v > mx; mx = v; pk = i; end
            end
            piv[k] = pk
            if pk != k
                for j in 1:SD; tmp = M[k, j]; M[k, j] = M[pk, j]; M[pk, j] = tmp; end
            end
            inv_d = one(T) / M[k, k]
            for i in k+1:SD
                M[i, k] *= inv_d
                f = M[i, k]
                for j in k+1:SD; M[i, j] -= f * M[k, j]; end
            end
        end

        z = MVector{SD, T}(undef)
        # --- proportional block: RHS = [I; I; …; I] column cd
        for cd in 1:D
            for i in 1:S, rd in 1:D; z[(i-1)*D + rd] = rd == cd ? one(T) : zero(T); end
            _lu_solve!(z, M, piv, Val(SD))
            for rd in 1:D
                acc = rd == cd ? one(T) : zero(T)
                for j in 1:S, q in 1:D
                    acc += h * tab.b[j] * As[j][rd, q] * z[(j-1)*D + q]
                end
                Mp[b, rd, cd, n] = acc
            end
            for q in 1:SD; Mp[b, D + q, cd, n] = z[q]; end
        end

        # --- delay blocks, stage by stage
        for s in 1:S
            ts = tn + tab.c[s] * h
            Bt = prob.B(ts, θ)
            τt = prob.tau(ts, θ)
            for k in 1:K
                Bk = SMatrix{D, D, T}(Bt[k])
                for cd in 1:D
                    for i in 1:S, rd in 1:D; z[(i-1)*D + rd] = h * tab.a[i, s] * Bk[rd, cd]; end
                    _lu_solve!(z, M, piv, Val(SD))
                    for rd in 1:D
                        acc = h * tab.b[s] * Bk[rd, cd]
                        for j in 1:S, q in 1:D
                            acc += h * tab.b[j] * As[j][rd, q] * z[(j-1)*D + q]
                        end
                        Md[b, rd, cd, s, k, n] = acc
                    end
                    for q in 1:SD; Md[b, D + q, cd, s, k, n] = z[q]; end
                end
                # delayed lookup position (same convention as build_system_matrices)
                rel = (ts - T(τt[k])) / h + r + 1
                if rel < 1 - T(1e-6)
                    flag[b] = Int32(1)
                end
                m = unsafe_trunc(Int, floor(rel))
                if m >= p + r + 1
                    m = p + r; th = one(T)
                elseif m < 1
                    m = 1; th = zero(T)
                else
                    th = rel - m
                end
                Midx[b, s, k, n] = Int32(m)
                w = _ce_weights(tab, th)
                for q in 1:S+2; Wt[b, q, s, k, n] = w[q]; end
            end
        end
    end
end

@inline function _lu_solve!(z, M, piv, ::Val{SD}) where {SD}
    @inbounds begin
        for k in 1:SD
            pk = piv[k]
            if pk != k; tmp = z[k]; z[k] = z[pk]; z[pk] = tmp; end
        end
        for i in 2:SD
            acc = z[i]
            for j in 1:i-1; acc -= M[i, j] * z[j]; end
            z[i] = acc
        end
        for i in SD:-1:1
            acc = z[i]
            for j in i+1:SD; acc -= M[i, j] * z[j]; end
            z[i] = acc / M[i, i]
        end
    end
    return z
end

"""
    build_batched_operators(prob, θs, tab, p, r; backend=CPU(), T=Float64, build=:auto)

Assemble the per-step transition blocks of all points `θs` and place them on `backend`.

- `build = :device` — one kernel work item per (point, step) on `backend`; the SD×SD
  stage matrix (SD = S·D) is factorized in thread-local memory.
- `build = :host` — threaded CPU assembly with the reference `build_system_matrices`
  (one point per thread), then a single upload. For large S·D.
- `build = :auto` — `:device` (measured on a T4 for D = 24, S·D = 72: device build
  2.2 s vs threaded host build 30 s for 128 points at p = 300 on the 2-core VM).
"""
function build_batched_operators(prob::BatchedLDDE{D, K}, θs::AbstractVector, tab::RKTableau{S},
                                 p::Int, r::Int; backend=CPU(), T::Type=Float64, build::Symbol=:auto) where {D, K, S}
    nb = length(θs)
    BS = (S + 1) * D
    mode = build === :auto ? :device : build
    mode in (:device, :host) || error("build must be :auto, :device or :host")
    dtab = DeviceTableau(tab, promote_type(T, Float32))
    isbits(prob) || error("BatchedLDDE must be isbits (plain functions / callable structs of plain data) " *
                          "to run in GPU kernels; got $(typeof(prob))")
    isbitstype(eltype(θs)) || error("parameter points θ must be isbits (e.g. SVector, NamedTuple of numbers)")
    Hist = KA.zeros(backend, T, nb, (p + r + 1) * BS)
    if mode === :device
        θd = _to_device(backend, collect(θs))
        Mp = KA.allocate(backend, T, nb, BS, D, p)
        Md = KA.allocate(backend, T, nb, BS, D, S, K, p)
        Midx = KA.allocate(backend, Int32, nb, S, K, p)
        Wt = KA.allocate(backend, T, nb, S + 2, S, K, p)
        flag = KA.zeros(backend, Int32, nb)
        _build_kernel!(backend, (min(nb, 64), 1))(Mp, Md, Midx, Wt, flag, θd, prob, dtab, p, r,
                                                 Val(D), Val(S), Val(K); ndrange=(nb, p))
        KA.synchronize(backend)
    else
        Mp_h = zeros(T, nb, BS, D, p); Md_h = zeros(T, nb, BS, D, S, K, p)
        Midx_h = ones(Int32, nb, S, K, p); Wt_h = zeros(T, nb, S + 2, S, K, p); flag_h = zeros(Int32, nb)
        θh = collect(θs)
        Threads.@threads for b in 1:nb
            θ = θh[b]
            grid = TimeGrid(collect(range(0.0, Float64(prob.period(θ)), length=p + 1)))
            sysm = try
                build_system_matrices(LDDEProblem(prob, θ), grid, tab, r)
            catch err
                err isa ErrorException || rethrow()
                flag_h[b] = 1; nothing
            end
            sysm === nothing && continue
            for n in 1:p
                Mp_h[b, :, :, n] .= sysm.M_prop[n]
                for k in 1:K, s in 1:S
                    Md_h[b, :, :, s, k, n] .= sysm.M_del[k][n][s]
                    Midx_h[b, s, k, n] = sysm.delay_indices[k][n][s]
                    Wt_h[b, :, s, k, n] .= sysm.delay_weights[k][n][s]
                end
            end
        end
        Mp = _to_device(backend, Mp_h); Md = _to_device(backend, Md_h)
        Midx = _to_device(backend, Midx_h); Wt = _to_device(backend, Wt_h); flag = _to_device(backend, flag_h)
    end
    return BatchedOperators{D, S, K, T, typeof(Hist), typeof(Mp), typeof(Md), typeof(Midx), typeof(Wt), typeof(flag)}(
        Mp, Md, Midx, Wt, Hist, flag, p, r, nb)
end

_to_device(backend, x::AbstractArray) = (y = KA.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)

# ---------------------------------------------------------------------------
# Sweep kernel (the matvec Y = Φ X for every point), workgroup (BG, R)
# ---------------------------------------------------------------------------

@kernel function _sweep_kernel!(Y, jy::Int, @Const(X), jx::Int, Hist::AbstractArray{T}, @Const(Mp), @Const(Md), @Const(Midx), @Const(Wt),
                                @Const(mask), p::Int, r::Int, nb::Int, ::Val{D}, ::Val{S}, ::Val{K}, ::Val{BG}) where {T, D, S, K, BG}
    b, lr = @index(Global, NTuple)
    lb, _ = @index(Local, NTuple)
    R = @uniform @groupsize()[2]
    BS = @uniform (S + 1) * D
    NDEL = @uniform S * K * D
    ydel = @localmem T (BG, S * K * D)
    ycur = @localmem T (BG, D)
    @inbounds begin
        # history ← reversed input blocks: Hist block (r − i) = X block i
        if b <= nb && mask[b] != 0
            for e in lr:R:((r + 1) * BS)
                i = (e - 1) ÷ BS; q = e - i * BS
                Hist[b, (r - i) * BS + q] = X[b, e, jx]
            end
        end
        @synchronize
        for n in 1:p
            if b <= nb && mask[b] != 0
                for e in lr:R:(NDEL + D)
                    if e <= NDEL
                        # e ↦ (d, s, k): delayed state of lag k at stage s, component d
                        d = (e - 1) % D + 1; sk = (e - 1) ÷ D
                        s = sk % S + 1; k = sk ÷ S + 1
                        m = Midx[b, s, k, n]
                        base0 = (m - 1) * BS; base1 = m * BS
                        acc = promote_type(T, Float32)(Wt[b, 1, s, k, n]) * Hist[b, base0 + d]
                        for i in 1:S
                            acc += Wt[b, i + 1, s, k, n] * Hist[b, base1 + i * D + d]
                        end
                        acc += Wt[b, S + 2, s, k, n] * Hist[b, base1 + d]
                        ydel[lb, e] = acc
                    else
                        d = e - NDEL
                        ycur[lb, d] = Hist[b, (n + r - 1) * BS + d]
                    end
                end
            end
            @synchronize
            if b <= nb && mask[b] != 0
                for row in lr:R:BS
                    acc = zero(promote_type(T, Float32))
                    for d in 1:D
                        acc += Mp[b, row, d, n] * ycur[lb, d]
                    end
                    for k in 1:K, s in 1:S
                        off = ((k - 1) * S + (s - 1)) * D
                        for d in 1:D
                            acc += Md[b, row, d, s, k, n] * ydel[lb, off + d]
                        end
                    end
                    Hist[b, (n + r) * BS + row] = acc
                end
            end
            @synchronize
        end
        if b <= nb && mask[b] != 0
            for e in lr:R:((r + 1) * BS)
                i = (e - 1) ÷ BS; q = e - i * BS
                Y[b, e, jy] = Hist[b, (p + r - i) * BS + q]
            end
        end
    end
end

"""
    SweepConfig(R, BG)

Thread mapping of the sweep kernel: `R` threads cooperate on one parameter point
(they share the rows of each step block), `BG` points per workgroup.
"""
struct SweepConfig
    R::Int
    BG::Int
end

"Number of threads the device keeps resident (overridden for CUDA by the extension)."
resident_threads(::KA.CPU) = Threads.nthreads()
resident_threads(backend) = 40 * 2048

"Free device memory in bytes (overridden for CUDA by the extension)."
available_memory(::KA.CPU) = Int(Sys.free_memory()) ÷ 2
available_memory(backend) = 4 * 2^30

"""
    auto_sweep_config(backend, nb, D, S, K)

Choose the thread mapping automatically: many points → one thread per point (`R = 1`);
few points → `R` threads per point so that `nb·R` reaches the resident thread count of
the device, capped by the step-block height `(S+1)·D`.
"""
function auto_sweep_config(backend, nb, D, S, K, T)
    BS = (S + 1) * D
    if backend isa KA.CPU
        return SweepConfig(1, 1)
    end
    # T4 measurement (D = 4, 2048 points): R = 4 ≈ R = 16 ≪ R = 1 (4× slower sweep at
    # p = 1000); R ≈ BS/4 threads per point, fewer if the batch alone fills the device.
    want = max(1, cld(resident_threads(backend), 4 * max(nb, 1)))
    R = clamp(prevpow(2, want), 1, max(1, prevpow(2, max(1, BS ÷ 4))))
    R = min(R, 256)
    # shared memory: BG·(S·K·D + D) values; keep it ≤ 24 KiB and the group ≤ 256 threads
    BG = max(1, min(256 ÷ R, (24 * 1024) ÷ ((S * K * D + D) * sizeof(T))))
    BG = prevpow(2, BG)
    return SweepConfig(R, BG)
end

"""
    batched_mul!(Y, jy, X, jx, op::BatchedOperators, cfg::SweepConfig, backend)

`Y[:, :, jy] = Φ_b X[:, :, jx]` for every point `b` of the batch with `mask[b] ≠ 0` (one
forward sweep each; masked-out points are skipped).
"""
function batched_mul!(Y, jy, X, jx, op::BatchedOperators{D, S, K}, cfg::SweepConfig, backend;
                      mask=KA.ones(backend, Int32, op.nb)) where {D, S, K}
    nbpad = cld(op.nb, cfg.BG) * cfg.BG
    _sweep_kernel!(backend, (cfg.BG, cfg.R))(Y, jy, X, jx, op.Hist, op.Mp, op.Md, op.Midx, op.Wt, mask,
                                             op.p, op.r, op.nb, Val(D), Val(S), Val(K), Val(cfg.BG);
                                             ndrange=(nbpad, cfg.R))
    return Y
end

# ---------------------------------------------------------------------------
# Batched Krylov–Schur (dominant eigenvalue per point)
# ---------------------------------------------------------------------------

@kernel function _dots_kernel!(Hc, @Const(V), @Const(W), jw::Int, N::Int, @Const(mask))
    b, l = @index(Global, NTuple)
    T = promote_type(eltype(V), Float32)    # accumulate in ≥ Float32
    acc = zero(T)
    @inbounds if mask[b] != 0
        for i in 1:N
            acc += V[b, i, l] * W[b, i, jw]
        end
    end
    @inbounds Hc[b, l] = acc
end

@kernel function _subtract_kernel!(W, jw::Int, @Const(V), @Const(Hc), j::Int, @Const(mask))
    b, i = @index(Global, NTuple)
    @inbounds if mask[b] != 0
        acc = W[b, i, jw]
        for l in 1:j
            acc -= V[b, i, l] * Hc[b, l]
        end
        W[b, i, jw] = acc
    end
end

@kernel function _norm_kernel!(nrm, @Const(W), jw::Int, N::Int)
    b = @index(Global)
    T = promote_type(eltype(W), Float32)
    acc = zero(T)
    @inbounds for i in 1:N
        acc += W[b, i, jw]^2
    end
    @inbounds nrm[b] = sqrt(acc)
end

@kernel function _setcol_kernel!(V, j::Int, @Const(W), jw::Int, @Const(scale), @Const(mask))
    b, i = @index(Global, NTuple)
    @inbounds if mask[b] != 0
        V[b, i, j] = W[b, i, jw] * scale[b]
    end
end

@kernel function _rotate_kernel!(Vt, @Const(V), @Const(Q), m::Int, kq::Int)
    b, i = @index(Global, NTuple)
    T = promote_type(eltype(V), Float32)
    @inbounds for l in 1:kq
        acc = zero(T)
        for q in 1:m
            acc += V[b, i, q] * Q[b, q, l]
        end
        Vt[b, i, l] = acc
    end
end

@kernel function _restart_copy_kernel!(V, @Const(Vt), @Const(kb), m::Int, kq::Int)
    b, i = @index(Global, NTuple)
    @inbounds begin
        k = kb[b]
        vres = V[b, i, m + 1]
        for l in 1:kq
            V[b, i, l] = Vt[b, i, l]
        end
        V[b, i, k + 1] = vres
    end
end

"""
    BatchedEigResult

`rho` (spectral radius), `mu` (dominant multiplier), `converged` (residual test passed),
`residual` (relative Ritz residual of the dominant pair), `matvecs` (sweeps per point),
`flag` (0 = ok, 1 = history window too short for the lag), `timing` (wall times in s
per phase — `build`, `sweep`, `orth`, `host`; device phases are exact only with
`profile = true`, which synchronizes after every phase).
"""
struct BatchedEigResult
    rho::Vector{Float64}
    mu::Vector{ComplexF64}
    converged::Vector{Bool}
    residual::Vector{Float64}
    matvecs::Int
    flag::Vector{Int32}
    timing::NamedTuple
end

BatchedEigResult(rho, mu, conv, res, nmv, flag) = BatchedEigResult(rho, mu, conv, res, nmv, flag, NamedTuple())

"""
    batched_eigs(op, backend; krylovdim=30, keep=15, tol=1e-13, maxiter=20, sweep=:auto)

Dominant Floquet multiplier of every point of the batch by a synchronous batched
Krylov–Schur iteration: device-side sweeps and CGS2 orthogonalisation, host-side
m×m Schur decompositions (threaded). Converged when the Ritz residual of the dominant
eigenpair satisfies `|h_{m+1}ᵀ s| ≤ tol · |λ₁|` for every point (or `maxiter` restarts).
"""
function batched_eigs(op::BatchedOperators{D, S, K, T}, backend; krylovdim::Int=30, keep::Int=15,
                      tol::Real=(T === Float64 ? 1e-13 : 100 * eps(T)), maxiter::Int=20, sweep=:auto, profile::Bool=false) where {D, S, K, T}
    nb = op.nb; N = state_size(op); m = min(krylovdim, N - 1)
    # profile = true synchronizes after every phase and accumulates wall times (s)
    tsw = 0.0; tor = 0.0; tho = 0.0
    tick() = (profile && KA.synchronize(backend); time())
    keep = clamp(keep, 1, m - 2)
    cfg = sweep === :auto ? auto_sweep_config(backend, nb, D, S, K, T) : sweep
    V = KA.zeros(backend, T, nb, N, m + 1)
    Wb = KA.zeros(backend, T, nb, N, 1)
    Vt = KA.zeros(backend, T, nb, N, keep + 1)
    CT = promote_type(T, Float32)               # coefficients/norms never in Float16
    Hc = KA.zeros(backend, CT, nb, m + 1)
    Hc2 = KA.zeros(backend, CT, nb, m + 1)
    nrm = KA.zeros(backend, CT, nb)
    scale = KA.zeros(backend, CT, nb)
    mask = KA.zeros(backend, Int32, nb)
    mask2 = KA.zeros(backend, Int32, nb)
    kbd = KA.zeros(backend, Int32, nb)
    Qd = KA.zeros(backend, CT, nb, m, keep + 1)

    # deterministic start vector (same as the CPU default), normalised
    x0 = Float64[1.0 + 0.1 * sin(7.3 * i) for i in 1:N]; x0 ./= norm(x0)
    V0 = repeat(reshape(T.(x0), 1, N), nb, 1)
    copyto!(V, 1, V0, 1, nb * N)      # column j = 1 is the first nb·N entries of V

    H = [zeros(Float64, m + 1, m) for _ in 1:nb]
    kb = zeros(Int, nb)                         # kept vectors per point (0 at start)
    rho = zeros(nb); mu = zeros(ComplexF64, nb); res = fill(Inf, nb)
    conv = fill(false, nb)                      # Vector{Bool}: written from threads (a BitVector would race)
    done = fill(false, nb)                      # converged or invariant subspace found: frozen
    nmv = 0
    wg2(a) = (min(a, 64), 1)
    nrm0 = KA.zeros(backend, CT, nb)
    bd_tol = T === Float64 ? 100 * eps(T) : 10 * eps(T)

    for it in 1:maxiter
        jstart = minimum(kb[.!done]) + 1
        for j in jstart:m
            act = Int32.((kb .< j) .& .!done)       # points that expand column j
            any(!=(0), act) || break
            copyto!(mask, act)
            t0 = tick()
            batched_mul!(Wb, 1, V, j, op, cfg, backend; mask=mask); nmv += 1
            t1 = tick(); tsw += t1 - t0
            _norm_kernel!(backend, min(nb, 64))(nrm0, Wb, 1, N; ndrange=nb)
            # Classical Gram–Schmidt against V[:, :, 1:j]; a second pass (DGKS criterion)
            # only for the points whose norm dropped below ‖Φv‖/√2 — the orthogonalisation
            # is the largest memory-traffic item, and frozen points are skipped entirely.
            _dots_kernel!(backend, wg2(nb))(Hc, V, Wb, 1, N, mask; ndrange=(nb, j))
            _subtract_kernel!(backend, wg2(nb))(Wb, 1, V, Hc, j, mask; ndrange=(nb, N))
            _norm_kernel!(backend, min(nb, 64))(nrm, Wb, 1, N; ndrange=nb)
            KA.synchronize(backend)
            nr = Array(nrm); nr0 = Array(nrm0)
            reorth = Int32[(act[b] != 0 && nr[b] < nr0[b] / sqrt(CT(2))) ? 1 : 0 for b in 1:nb]
            copyto!(mask2, reorth)
            _dots_kernel!(backend, wg2(nb))(Hc2, V, Wb, 1, N, mask2; ndrange=(nb, j))
            if any(!=(0), reorth)
                _subtract_kernel!(backend, wg2(nb))(Wb, 1, V, Hc2, j, mask2; ndrange=(nb, N))
                _norm_kernel!(backend, min(nb, 64))(nrm, Wb, 1, N; ndrange=nb)
            end
            KA.synchronize(backend)
            hc = Array(Hc); hc2 = Array(Hc2); nr = Array(nrm)
            sc = zeros(CT, nb)
            for b in 1:nb
                act[b] == 0 && continue
                for l in 1:j; H[b][l, j] = Float64(hc[b, l]) + Float64(hc2[b, l]); end
                β = Float64(nr[b]); H[b][j + 1, j] = β
                if β <= bd_tol * Float64(nr0[b])
                    # invariant subspace: the Ritz values of H[1:j, 1:j] are exact
                    λ = eigvals(H[b][1:j, 1:j]); i1 = argmax(abs.(λ))
                    rho[b] = abs(λ[i1]); mu[b] = λ[i1]; res[b] = 0.0
                    conv[b] = true; done[b] = true; act[b] = 0
                else
                    sc[b] = CT(1 / β)
                end
            end
            copyto!(mask, act)
            copyto!(scale, sc)
            _setcol_kernel!(backend, wg2(nb))(V, j + 1, Wb, 1, scale, mask; ndrange=(nb, N))
            tor += tick() - t1
        end
        t2 = tick()

        # host: Schur form, convergence test, restart data
        Q = zeros(CT, nb, m, keep + 1)
        knew = zeros(Int, nb)
        Threads.@threads for b in 1:nb
            done[b] && continue
            Hm = H[b][1:m, 1:m]
            F = schur(Hm)
            λ = F.values
            order = sortperm(abs.(λ); rev=true)
            λ1 = λ[order[1]]
            rho[b] = abs(λ1); mu[b] = λ1
            # keep the `keep` largest (pairs not split)
            thr = abs(λ[order[keep]])
            k = count(x -> abs(x) >= thr * (1 - 1e-12), λ)
            k = min(k, keep + 1)
            sel = fill(false, m); cnt = 0
            for idx in order
                cnt >= k && break
                sel[idx] = true; cnt += 1
            end
            # complete conjugate pairs
            for idx in 1:m
                if sel[idx] && imag(λ[idx]) != 0
                    pidx = findfirst(i -> !sel[i] && λ[i] ≈ conj(λ[idx]), 1:m)
                    pidx !== nothing && (sel[pidx] = true)
                end
            end
            k = count(sel)
            if k > keep + 1
                # cannot keep that many: drop the trailing pair instead
                k = keep - 1
                sel .= false; cnt = 0
                for idx in order; cnt >= k && break; sel[idx] = true; cnt += 1; end
                for idx in 1:m
                    if sel[idx] && imag(λ[idx]) != 0
                        pidx = findfirst(i -> !sel[i] && λ[i] ≈ conj(λ[idx]), 1:m)
                        pidx !== nothing && (sel[pidx] = true)
                    end
                end
                k = count(sel)
            end
            ordschur!(F, sel)
            knew[b] = k
            Z = F.Z; Tm = F.T
            hrow = H[b][m + 1, 1:m]
            # residual of the dominant Ritz pair from the leading k×k block: its eigenvectors
            # are those of the full quasi-triangular T padded with zeros, so |h_{m+1}ᵀ Z s|
            # needs only a k×k eigenproblem (not an m×m one)
            hz = vec(hrow' * Z[:, 1:k])
            E = eigen(Tm[1:k, 1:k])
            i1 = argmax(abs.(E.values))
            sv = E.vectors[:, i1]
            res[b] = abs(sum(hz .* sv)) / norm(sv) / max(abs(λ1), eps())
            conv[b] = res[b] <= tol
            conv[b] && (done[b] = true)
            for l in 1:k, q in 1:m; Q[b, q, l] = CT(Z[q, l]); end
            Hn = zeros(Float64, m + 1, m)
            Hn[1:k, 1:k] .= Tm[1:k, 1:k]
            Hn[k + 1, 1:k] .= vec(hrow' * Z[:, 1:k])
            H[b] = Hn
        end
        tho += time() - t2
        all(done) && break
        it == maxiter && break
        kq = maximum(knew)
        copyto!(Qd, Q)
        _rotate_kernel!(backend, wg2(nb))(Vt, V, Qd, m, kq; ndrange=(nb, N))
        copyto!(kbd, Int32.(knew))
        _restart_copy_kernel!(backend, wg2(nb))(V, Vt, kbd, m, kq; ndrange=(nb, N))
        KA.synchronize(backend)
        for b in 1:nb; done[b] || (kb[b] = knew[b]); end
    end
    return BatchedEigResult(rho, mu, conv, res, nmv, Array(op.flag), (sweep=tsw, orth=tor, host=tho))
end

# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

"""
    spectral_radii(prob::BatchedLDDE, θs, tableau, p, r; backend=CPU(), T=Float64,
                   batchsize=:auto, krylovdim=30, keep=15, tol=1e-13, maxiter=20, retry=true,
                   sweep=:auto, build=:auto, cpu_mode=:reference, profile=false, verbose=false)

Spectral radius of the one-period monodromy operator for every parameter point in
`θs`, all discretized with the same `tableau`, `p` steps per period and `r` delay
steps (h = T(θ)/p per point; `r·h` must cover the largest lag). Points are processed
in batches sized to the free device memory. A point that does not converge within
`maxiter` restarts is solved again with a basis twice as large (`retry = true`);
`matvecs` then counts the sweeps of all passes. Returns a [`BatchedEigResult`](@ref)
(field `rho` holds the spectral radii, in the order of `θs`).

`backend` is any KernelAbstractions backend: `CUDABackend()` after `using CUDA`, or
`CPU()`. On `CPU()` the default `cpu_mode = :reference` runs the reference SOSD solver
(`floquet_analysis`, sparse map + KrylovKit) with one point per thread — the fast CPU
fallback; `cpu_mode = :kernels` runs the GPU kernels on CPU threads instead (slow; meant
for validating the kernels without a GPU). `build` selects where the step operators are
assembled (see [`build_batched_operators`](@ref)). `T = Float32` halves memory traffic at the cost
of ~10⁻⁵ accuracy on ρ (see GPU_DESIGN.md).
"""
function spectral_radii(prob::BatchedLDDE{D, K}, θs::AbstractVector, tab::RKTableau{S}, p::Int, r::Int;
                        backend=CPU(), T::Type=Float64, batchsize=:auto, krylovdim::Int=30, keep::Int=15,
                        tol::Real=(T === Float64 ? 1e-13 : 100 * eps(T)), maxiter::Int=20, retry::Bool=true, sweep=:auto, build::Symbol=:auto,
                        cpu_mode::Symbol=:reference, profile::Bool=false, verbose::Bool=false) where {D, K, S}
    n = length(θs)
    if backend isa KA.CPU && cpu_mode === :reference
        return _spectral_radii_reference(prob, θs, tab, p, r, tol)
    end
    rho = zeros(n); mu = zeros(ComplexF64, n); conv = fill(false, n); res = zeros(n); flag = zeros(Int32, n)
    nmv = 0
    tb = 0.0; tsw = 0.0; tor = 0.0; tho = 0.0
    # The batch iterates synchronously, so one slowly converging point would hold all the
    # others: first pass with a short restart budget, then the stragglers again with a
    # basis twice as large (clustered dominant multipliers need it).
    passes = retry ? ((krylovdim, keep, maxiter), (2krylovdim, 2keep, 3maxiter)) : ((krylovdim, keep, maxiter),)
    todo = collect(1:n)
    for (pass, (kd, kp, mi)) in enumerate(passes)
        isempty(todo) && break
        bpp = bytes_per_point(D, S, K, p, r, min(kd, (r + 1) * (S + 1) * D - 1), T)
        bs = batchsize === :auto ? max(1, min(length(todo), floor(Int, 0.8 * available_memory(backend) / bpp))) : batchsize
        for lo in 1:bs:length(todo)
            idx = todo[lo:min(length(todo), lo + bs - 1)]
            t0 = time()
            op = build_batched_operators(prob, θs[idx], tab, p, r; backend=backend, T=T, build=build)
            t1 = time()
            er = batched_eigs(op, backend; krylovdim=kd, keep=kp, tol=tol, maxiter=mi, sweep=sweep, profile=profile)
            t2 = time()
            tb += t1 - t0; tsw += er.timing.sweep; tor += er.timing.orth; tho += er.timing.host
            verbose && println("pass $pass, $(length(idx)) points: build $(round(t1 - t0, digits=3)) s, eigs " *
                               "$(round(t2 - t1, digits=3)) s ($(er.matvecs) sweeps, $(count(er.converged)) converged)")
            rho[idx] .= er.rho; mu[idx] .= er.mu; conv[idx] .= er.converged
            res[idx] .= er.residual; flag[idx] .= er.flag
            nmv += er.matvecs
            op = nothing
        end
        todo = findall(i -> !conv[i] && flag[i] == 0, 1:n)
    end
    any(!=(0), flag) && @warn "$(count(!=(0), flag)) point(s) look up delayed states before the stored " *
                              "history window (r·h < lag); their ρ is wrong — increase r" maxlog=1
    isempty(todo) || @warn "$(length(todo)) point(s) did not converge (relative Ritz residual > tol); " *
                           "see `converged` / `residual`" maxlog=1
    return BatchedEigResult(rho, mu, conv, res, nmv, flag, (build=tb, sweep=tsw, orth=tor, host=tho))
end

"CPU fallback: the reference SOSD solver (sparse map + KrylovKit), one point per thread."
function _spectral_radii_reference(prob::BatchedLDDE, θs, tab, p, r, tol)
    n = length(θs)
    rho = fill(NaN, n); mu = fill(ComplexF64(NaN), n); conv = fill(false, n); flag = zeros(Int32, n)
    θh = collect(θs)
    Threads.@threads for i in 1:n
        θ = θh[i]
        grid = TimeGrid(collect(range(0.0, Float64(prob.period(θ)), length=p + 1)))
        try
            sol = floquet_analysis(LDDEProblem(prob, θ), grid, tab, r; nev=1, tol=tol)
            rho[i] = sol.spectral_radius; mu[i] = sol.mu; conv[i] = sol.converged >= 1
        catch err
            err isa ErrorException || rethrow()
            flag[i] = 1
        end
    end
    any(!=(0), flag) && @warn "$(count(!=(0), flag)) point(s) look up delayed states before the stored " *
                              "history window (r·h < lag); their ρ is NaN — increase r" maxlog=1
    return BatchedEigResult(rho, mu, conv, fill(NaN, n), 0, flag)
end

"""
    boundary_multisection(rho_of, xs, ylo, yhi; nsub=15, rounds=4)

Stability boundary `y_lim(x)` for every `x` in `xs` (e.g. depth of cut vs spindle
speed), assuming ρ < 1 at `ylo`. Each round evaluates `nsub` interior values for
*all* x at once through the batched evaluator `rho_of(xs_batch, ys_batch) -> ρ`
and keeps the sub-interval containing the lowest unstable sample, so `rounds`
batched calls reach the resolution `(yhi − ylo)/(nsub + 1)^rounds`. Returns the
upper bracket end (`Inf` where no unstable sample was found in `[ylo, yhi]`).
"""
function boundary_multisection(rho_of, xs::AbstractVector, ylo::Real, yhi::Real; nsub::Int=15, rounds::Int=4)
    nx = length(xs)
    lo = fill(Float64(ylo), nx); hi = fill(Float64(yhi), nx); found = falses(nx)
    for round in 1:rounds
        xb = Vector{eltype(xs)}(undef, nx * nsub); yb = zeros(nx * nsub)
        for i in 1:nx, q in 1:nsub
            xb[(i - 1) * nsub + q] = xs[i]
            yb[(i - 1) * nsub + q] = lo[i] + (hi[i] - lo[i]) * q / (nsub + 1)
        end
        ρ = rho_of(xb, yb)
        for i in 1:nx
            # first unstable sub-point (hi itself is unstable once found, or unknown)
            qf = findfirst(q -> ρ[(i - 1) * nsub + q] >= 1, 1:nsub)
            if qf === nothing
                lo[i] = yb[i * nsub]
            else
                found[i] = true
                hi[i] = yb[(i - 1) * nsub + qf]
                qf > 1 && (lo[i] = yb[(i - 1) * nsub + qf - 1])
            end
        end
    end
    return [found[i] ? hi[i] : Inf for i in 1:nx]
end
