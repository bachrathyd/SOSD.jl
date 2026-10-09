# Forced response of the batched solver: the T-periodic orbit of
#     ẋ(t) = A(t) x(t) + Σ_k B_k(t) x(t − τ_k(t)) + f(t),
# the fixed point x* = Φ x* + g of the affine one-period map (g: one period from a zero history
# under the forcing), solved per parameter point by restarted GMRES on (I − Φ) x = g with the
# batched sweep and the CGS2 kernels of `batched_eigs`; then one forced period along x* gives
# the peak-to-peak amplitude of every state component (over the nodes 0, c_i, 1 of each step).

@kernel function _forcing_kernel!(Fv, @Const(θs), prob, f, tab, p::Int, ::Val{D}, ::Val{S}) where {D, S}
    b, n = @index(Global, NTuple)
    T = promote_type(eltype(Fv), Float32)
    SD = S * D
    @inbounds begin
        θ = θs[b]
        h = T(prob.period(θ)) / p
        tn = (n - 1) * h
        As = ntuple(j -> SMatrix{D, D, T}(prob.A(tn + tab.c[j] * h, θ)), Val(S))
        Fs = ntuple(j -> SVector{D, T}(f(tn + tab.c[j] * h, θ)), Val(S))
        # stage matrix M = I − h (a ⊗ A_j), LU with partial pivoting (as in _build_kernel!)
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
                fk = M[i, k]
                for j in k+1:SD; M[i, j] -= fk * M[k, j]; end
            end
        end
        # stage values of the forcing alone: Z = M⁻¹ h (a ⊗ I) F
        z = MVector{SD, T}(undef)
        for i in 1:S, rd in 1:D
            acc = zero(T)
            for j in 1:S; acc += h * tab.a[i, j] * Fs[j][rd]; end
            z[(i-1)*D + rd] = acc
        end
        _lu_solve!(z, M, piv, Val(SD))
        for rd in 1:D
            acc = zero(T)
            for j in 1:S
                acc += h * tab.b[j] * Fs[j][rd]
                for q in 1:D; acc += h * tab.b[j] * As[j][rd, q] * z[(j-1)*D + q]; end
            end
            Fv[b, rd, n] = acc
        end
        for q in 1:SD; Fv[b, D + q, n] = z[q]; end
    end
end

"""
    with_forcing(op, prob, f, θs, tab; backend=CPU()) -> BatchedOperators

The operators `op` of [`build_batched_operators`](@ref) extended by the forced response of
every step to the forcing `f(t, θ) -> SVector{D}` (or a tuple / vector of D numbers), so that
`batched_mul!(…; forced = true)` applies the affine one-period map `x ↦ Φ x + g`.
"""
function with_forcing(op::BatchedOperators{D, S, K, T}, prob::BatchedLDDE{D, K}, f, θs::AbstractVector,
                      tab::RKTableau{S}; backend=CPU()) where {D, S, K, T}
    isbits(f) || error("the forcing must be isbits (a plain function or a callable struct of plain data)")
    BS = (S + 1) * D
    dtab = DeviceTableau(tab, promote_type(T, Float32))
    Fv = KA.allocate(backend, T, op.nb, BS, op.p)
    θd = _to_device(backend, collect(θs))
    _forcing_kernel!(backend, (min(op.nb, 64), 1))(Fv, θd, prob, f, dtab, op.p, Val(D), Val(S); ndrange=(op.nb, op.p))
    KA.synchronize(backend)
    return BatchedOperators{D, S, K, T, typeof(op.Hist), typeof(op.Mp), typeof(op.Md), typeof(op.Midx), typeof(op.Wt),
                            typeof(op.flag), typeof(Fv)}(op.Mp, op.Md, op.Midx, op.Wt, op.Hist, op.flag, op.p, op.r,
                                                         op.nb, op.colloc, op.ew, Fv, true)
end

@kernel function _isub_kernel!(W, jw::Int, @Const(V), j::Int, @Const(mask))      # W ← V_j − W  ((I − Φ) V_j)
    b, i = @index(Global, NTuple)
    @inbounds if mask[b] != 0
        W[b, i, jw] = V[b, i, j] - W[b, i, jw]
    end
end

@kernel function _combine_kernel!(X, jx::Int, @Const(V), @Const(Yc), m::Int, @Const(mask))   # X += V[:, 1:m] y
    b, i = @index(Global, NTuple)
    T = promote_type(eltype(V), Float32)
    @inbounds if mask[b] != 0
        acc = T(X[b, i, jx])
        for l in 1:m; acc += V[b, i, l] * Yc[b, l]; end
        X[b, i, jx] = acc
    end
end

@kernel function _resid_kernel!(Rr, @Const(G), @Const(X), @Const(W), @Const(mask))      # r = g − x + Φx
    b, i = @index(Global, NTuple)
    @inbounds if mask[b] != 0
        Rr[b, i, 1] = G[b, i, 1] - X[b, i, 1] + W[b, i, 1]
    end
end

@kernel function _peak_kernel!(amp, @Const(Hist), p::Int, r::Int, ::Val{D}, ::Val{S}) where {D, S}
    b, d = @index(Global, NTuple)
    BS = (S + 1) * D
    T = promote_type(eltype(Hist), Float32)
    @inbounds begin
        lo = T(Inf); hi = T(-Inf)
        for n in 1:p, q in 0:S                     # y_{n} and the stage values of every step
            v = T(Hist[b, (n + r) * BS + q * D + d])
            lo = min(lo, v); hi = max(hi, v)
        end
        amp[b, d] = hi - lo
    end
end

"""
    PeriodicOrbitResult

`amp[i, d]`: peak-to-peak amplitude of state component `d` on the periodic orbit of point `i`
(sampled at the step nodes 0, c, 1); `residual[i]`: ‖(I − Φ) x − g‖ / ‖g‖ of the orbit;
`converged`; `matvecs` (sweeps per point); `flag` as in [`BatchedEigResult`](@ref). Meaningful
where the point is stable (ρ < 1); near ρ = 1 the amplitude grows without bound.
"""
struct PeriodicOrbitResult
    amp::Matrix{Float64}
    residual::Vector{Float64}
    converged::Vector{Bool}
    matvecs::Int
    flag::Vector{Int32}
end

"""
    periodic_orbits(prob, f, θs, tableau, p, r; backend=CPU(), T=Float64, krylovdim=30,
                    tol=1e-10, maxrestart=20, batchsize=:auto, sweep=:auto, build=:auto)

Periodic orbit of the forced system `ẋ = A x + Σ B_k x(t − τ_k) + f(t, θ)` for every parameter
point (restarted GMRES on the fixed point of the one-period map, batched like
[`spectral_radii`](@ref); the forcing `f(t, θ)` must be T-periodic and isbits). Returns a
[`PeriodicOrbitResult`](@ref) with the peak-to-peak amplitude of each state component.
"""
function periodic_orbits(prob::BatchedLDDE{D, K}, f, θs::AbstractVector, tab::RKTableau{S}, p::Int, r::Int;
                         backend=CPU(), T::Type=Float64, krylovdim::Int=30,
                         tol::Real=(T === Float64 ? 1e-10 : 1e-5), maxrestart::Int=20, batchsize=:auto,
                         sweep=:auto, build::Symbol=:auto, verbose::Bool=false) where {D, K, S}
    n = length(θs)
    amp = zeros(n, D); res = fill(Inf, n); conv = fill(false, n); flag = zeros(Int32, n); nmv = 0
    m = min(krylovdim, (r + 1) * (S + 1) * D - 1)
    bpp = bytes_per_point(D, S, K, p, r, m, T) + (S + 1) * D * p * sizeof(T)
    bs = batchsize === :auto ? max(1, min(n, floor(Int, 0.8 * available_memory(backend) / bpp))) : batchsize
    for lo in 1:bs:n
        idx = lo:min(n, lo + bs - 1)
        t0 = time()
        op = build_batched_operators(prob, θs[idx], tab, p, r; backend=backend, T=T, build=build)
        op = with_forcing(op, prob, f, θs[idx], tab; backend=backend)
        a, rs, cv, k = _batched_orbits(op, backend; m=m, tol=tol, maxrestart=maxrestart, sweep=sweep)
        amp[idx, :] .= a; res[idx] .= rs; conv[idx] .= cv; flag[idx] .= Array(op.flag); nmv += k
        verbose && println("$(length(idx)) points: $(round(time() - t0, digits=3)) s, $(k) sweeps, $(count(cv)) converged")
        op = nothing
    end
    any(!=(0), flag) && @warn "$(count(!=(0), flag)) point(s) look up delayed states before the stored " *
                              "history window (r·h < lag) — increase r" maxlog=1
    return PeriodicOrbitResult(amp, res, conv, nmv, flag)
end

function _batched_orbits(op::BatchedOperators{D, S, K, T}, backend; m::Int, tol::Real, maxrestart::Int, sweep=:auto) where {D, S, K, T}
    nb = op.nb; N = state_size(op)
    cfg = sweep === :auto ? auto_sweep_config(backend, nb, D, S, K, T) : sweep
    CT = promote_type(T, Float32)
    V = KA.zeros(backend, T, nb, N, m + 1)
    W = KA.zeros(backend, T, nb, N, 1)
    X = KA.zeros(backend, T, nb, N, 1)
    G = KA.zeros(backend, T, nb, N, 1)
    Rr = KA.zeros(backend, T, nb, N, 1)
    Hc = KA.zeros(backend, CT, nb, m + 1); Hc2 = KA.zeros(backend, CT, nb, m + 1)
    nrm = KA.zeros(backend, CT, nb); nrm0 = KA.zeros(backend, CT, nb)
    scale = KA.zeros(backend, CT, nb)
    mask = KA.ones(backend, Int32, nb); mask2 = KA.zeros(backend, Int32, nb)
    Yd = KA.zeros(backend, CT, nb, m)
    wg2(a) = (min(a, 64), 1)
    nmv = 0
    # g: one forced period from a zero history; r₀ = g (x₀ = 0)
    batched_mul!(G, 1, X, 1, op, cfg, backend; forced=true); nmv += 1
    copyto!(Rr, G)
    _norm_kernel!(backend, min(nb, 64))(nrm, G, 1, N; ndrange=nb)
    KA.synchronize(backend)
    gn = Float64.(Array(nrm)); gn .= max.(gn, floatmin(Float64))
    res = fill(Inf, nb); done = fill(false, nb)
    for it in 1:maxrestart
        _norm_kernel!(backend, min(nb, 64))(nrm, Rr, 1, N; ndrange=nb)
        KA.synchronize(backend)
        β = Float64.(Array(nrm))
        for b in 1:nb
            res[b] = β[b] / gn[b]
            (res[b] <= tol || !isfinite(res[b])) && (done[b] = true)
        end
        all(done) && break
        act = Int32.(.!done)
        copyto!(mask, act)
        copyto!(scale, CT.(1 ./ max.(β, floatmin(Float64))))
        _setcol_kernel!(backend, wg2(nb))(V, 1, Rr, 1, scale, mask; ndrange=(nb, N))
        H = [zeros(Float64, m + 1, m) for _ in 1:nb]
        for j in 1:m
            batched_mul!(W, 1, V, j, op, cfg, backend; mask=mask); nmv += 1
            _isub_kernel!(backend, wg2(nb))(W, 1, V, j, mask; ndrange=(nb, N))
            _norm_kernel!(backend, min(nb, 64))(nrm0, W, 1, N; ndrange=nb)
            _dots_kernel!(backend, wg2(nb))(Hc, V, W, 1, N, mask; ndrange=(nb, j))
            _subtract_kernel!(backend, wg2(nb))(W, 1, V, Hc, j, mask; ndrange=(nb, N))
            _norm_kernel!(backend, min(nb, 64))(nrm, W, 1, N; ndrange=nb)
            KA.synchronize(backend)
            nr = Array(nrm); nr0 = Array(nrm0)
            reorth = Int32[(act[b] != 0 && nr[b] < nr0[b] / sqrt(CT(2))) ? 1 : 0 for b in 1:nb]
            copyto!(mask2, reorth)
            _dots_kernel!(backend, wg2(nb))(Hc2, V, W, 1, N, mask2; ndrange=(nb, j))
            if any(!=(0), reorth)
                _subtract_kernel!(backend, wg2(nb))(W, 1, V, Hc2, j, mask2; ndrange=(nb, N))
                _norm_kernel!(backend, min(nb, 64))(nrm, W, 1, N; ndrange=nb)
            end
            KA.synchronize(backend)
            hc = Array(Hc); hc2 = Array(Hc2); nr = Array(nrm)
            sc = zeros(CT, nb)
            for b in 1:nb
                act[b] == 0 && continue
                for l in 1:j; H[b][l, j] = Float64(hc[b, l]) + (reorth[b] != 0 ? Float64(hc2[b, l]) : 0.0); end
                hb = Float64(nr[b]); H[b][j + 1, j] = hb
                sc[b] = hb > 1e-300 ? CT(1 / hb) : zero(CT)          # breakdown: exact in the subspace
            end
            copyto!(scale, sc)
            _setcol_kernel!(backend, wg2(nb))(V, j + 1, W, 1, scale, mask; ndrange=(nb, N))
        end
        # least squares min ‖β e₁ − H y‖ per point (host), x += V y
        Y = zeros(CT, nb, m)
        Threads.@threads for b in 1:nb
            act[b] == 0 && continue
            rhs = zeros(m + 1); rhs[1] = β[b]
            y = H[b] \ rhs
            all(isfinite, y) || continue
            for l in 1:m; Y[b, l] = CT(y[l]); end
        end
        copyto!(Yd, Y)
        _combine_kernel!(backend, wg2(nb))(X, 1, V, Yd, m, mask; ndrange=(nb, N))
        # true residual r = g − (I − Φ) x
        batched_mul!(W, 1, X, 1, op, cfg, backend; mask=mask); nmv += 1
        _resid_kernel!(backend, wg2(nb))(Rr, G, X, W, mask; ndrange=(nb, N))
    end
    _norm_kernel!(backend, min(nb, 64))(nrm, Rr, 1, N; ndrange=nb)
    KA.synchronize(backend)
    res .= Float64.(Array(nrm)) ./ gn
    # one forced period along the orbit: the history buffer holds it -> peak-to-peak per component
    copyto!(mask, ones(Int32, nb))
    batched_mul!(W, 1, X, 1, op, cfg, backend; forced=true); nmv += 1
    ampd = KA.zeros(backend, CT, nb, D)
    _peak_kernel!(backend, (min(nb, 64), 1))(ampd, op.Hist, op.p, op.r, Val(D), Val(S); ndrange=(nb, D))
    KA.synchronize(backend)
    return Float64.(Array(ampd)), res, res .<= tol, nmv
end
