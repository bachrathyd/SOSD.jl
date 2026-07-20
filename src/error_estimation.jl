# Embedded-pair error estimation via matrix perturbation analysis.
#
# The monodromy operator is assembled twice on the SAME grid: once with the
# user's tableau (Φ) and once with the embedded lower-order companion (Φ̂,
# see embedded.jl). ΔΦ = Φ − Φ̂ is the classical embedded local-error signal
# propagated consistently through the mapping matrix; first-order perturbation
# theory turns it into error bars:
#
#   eigenvalue    δμ = yᵀ ΔΦ x / (yᵀ x)      (right x, left y, bilinear pairing)
#   mode shape    ‖δx‖/‖x‖ ≤ ‖ΔΦx − δμ·x‖ / gap
#   fixed point   δy* = (I − Φ)⁻¹ (ΔΦ y* + ΔV)
#
# As with ode23, the pair difference estimates the error of the LOWER-order
# companion, so the bar is conservative for the returned higher-order result.

# ---------------------------------------------------------------------------
# Transpose action of SparseMonodromyMap (needed for the left eigenvector)
# ---------------------------------------------------------------------------

_ldiv_transposed(F, w::AbstractVector{<:Real}) = F' \ w
_ldiv_transposed(F, w::AbstractVector{<:Complex}) = (F' \ real.(w)) .+ im .* (F' \ imag.(w))

function LinearMaps._unsafe_mul!(y_out::AbstractVector,
                                 tm::LinearMaps.TransposeMap{T, SparseMonodromyMap{T, LS, RM, BSIZE, R}},
                                 x_in::AbstractVector) where {T, LS, RM, BSIZE, R}
    m = tm.lmap
    p = size(m.R, 1) ÷ BSIZE
    TE = eltype(x_in)
    # Forward: y[i] = (L⁻¹ R P x)[p−i−1] for k = p−i ≥ 1, plus y[i] = x[i−p] for i ≥ p
    # (P = block-reversal permutation, symmetric). Transpose runs the chain backwards.
    w = zeros(TE, p * BSIZE)
    for i in 0:R
        k = p - i
        k >= 1 || continue
        src = i * BSIZE + 1; dst = (k - 1) * BSIZE + 1
        @views w[dst:dst + BSIZE - 1] .+= x_in[src:src + BSIZE - 1]
    end
    z = _ldiv_transposed(m.L_solver, w)
    u = transpose(m.R) * z
    for i in 0:R
        src = (R - i) * BSIZE + 1; dst = i * BSIZE + 1
        @views y_out[dst:dst + BSIZE - 1] .= u[src:src + BSIZE - 1]
    end
    for i in p:R
        src = i * BSIZE + 1; dst = (i - p) * BSIZE + 1
        @views y_out[dst:dst + BSIZE - 1] .+= x_in[src:src + BSIZE - 1]
    end
    return y_out
end

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

"""
    FloquetSolution

Output of [`floquet_analysis`](@ref): dominant multiplier `mu`, its magnitude
`spectral_radius`, all computed `multipliers` and `modes` (right eigenvectors
of the monodromy operator), the periodic `fixpoint` (or `nothing`), and the
number of `converged` Ritz values.
"""
struct FloquetSolution
    mu::ComplexF64
    spectral_radius::Float64
    multipliers::Vector{ComplexF64}
    modes::Vector{Vector{ComplexF64}}
    fixpoint::Union{Nothing, Vector{Float64}}
    converged::Int
end

"""
    FloquetErrorEstimate

The separate error output of [`floquet_analysis`](@ref) (`error_estimation=true`).

- `mu_error`      — combined error bar `safety_factor · (|δμ_Q| + |δμ_I|)` on
  the dominant multiplier; also bounds the spectral-radius error (|δρ| ≤ |δμ|).
- `delta_mu`      — signed eigenvalue shift (complex, unscaled).
- `quadrature_error`, `interpolation_error` — the raw (unscaled) channels.
  NaN = channel unavailable: collocation tableaux report their whole
  cross-family pair difference in the quadrature channel (their reduced-CE
  perturbation is null on the solution manifold, see `embedded.jl`), so the
  interpolation channel is NaN there. For non-collocation tableaux the channel
  measures the delayed-state interpolation uncertainty at the actual lookup
  positions (lookups landing exactly on grid endpoints contribute nothing).
- `eigenvalue_condition` — κ(μ) = ‖x‖‖y‖/|yᵀx| (sensitivity of μ to ANY
  operator perturbation of unit norm).
- `spectral_gap`, `mode_error` — gap to the next Ritz value and the first-order
  mode-shape bound ‖δx‖/‖x‖ ≤ ‖ΔΦx − δμx‖/gap.
- `fixpoint_error`, `fixpoint_delta` — relative bar and full first-order
  correction vector for the periodic solution (NaN/`nothing` if not requested).
- `mu_embedded_quadrature`, `mu_embedded_interpolation` — dominant multiplier
  of the embedded operators (diagnostics, only with `embedded_eigs=true`).

RESOLUTION GUARD: a bar exceeding ~10% of the spectral radius signals a
below-resolution discretization (cf. the Shannon sample recipe) — there BOTH
the multiplier and the bar are untrustworthy; refine (s, p) instead of
interpreting them. Validated coverage refers to resolved points.
"""
struct FloquetErrorEstimate
    mu_error::Float64
    delta_mu::ComplexF64
    quadrature_error::Float64
    interpolation_error::Float64
    eigenvalue_condition::Float64
    spectral_gap::Float64
    mode_error::Float64
    fixpoint_error::Float64
    fixpoint_delta::Union{Nothing, Vector{Float64}}
    mu_embedded_quadrature::ComplexF64
    mu_embedded_interpolation::ComplexF64
end

function Base.show(io::IO, ::MIME"text/plain", e::FloquetErrorEstimate)
    println(io, "FloquetErrorEstimate")
    println(io, "  |δμ| bar          : ", e.mu_error)
    println(io, "    quadrature      : ", e.quadrature_error)
    println(io, "    interpolation   : ", e.interpolation_error)
    println(io, "  cond(μ)           : ", e.eigenvalue_condition)
    println(io, "  mode-shape bar    : ", e.mode_error, "  (gap ", e.spectral_gap, ")")
    print(io,   "  fixpoint rel. bar : ", e.fixpoint_error)
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"Deterministic start vector (no RNG in library paths)."
_det_start(n) = normalize!([1.0 + 0.1 * sin(7.3 * i) for i in 1:n])

"Bilinear (transpose, non-conjugating) pairing — analytic in both arguments."
_blin(a, b) = transpose(a) * b

"""
    floquet_analysis(prob, grid, tableau, r; kwargs...)

Compute the dominant Floquet multiplier(s) of the time-periodic DDE via the
sparse SOSD monodromy operator.

Returns a [`FloquetSolution`](@ref); with `error_estimation = true` returns
`(FloquetSolution, FloquetErrorEstimate)` — the error bar is a **separate
output**, the solution object is unchanged.

Keywords:
- `nev = 3`               number of Ritz values (≥ 2 enables the spectral gap
                          / mode-shape bound)
- `tol = 1e-11`           eigensolver tolerance
- `x0 = nothing`          start vector (deterministic default)
- `error_estimation = false`  embedded-pair error bars (≈ 2–3× cost)
- `periodic_solution = false` also solve the periodic fixed point (and its
                          error bar when estimation is on)
- `safety_factor = 2.0`   multiplies the reported `mu_error` / `mode_error` /
                          `fixpoint_error` bars (the raw channel values are
                          kept unscaled); 1.0 gives the raw pair difference
- `embedded_eigs = false` additionally eigsolve the embedded operators
                          (diagnostics: `mu_embedded_*`)
- `mass_matrix`, `static_threshold` — forwarded to the assembly
"""
function floquet_analysis(prob::LDDEProblem{D, T}, grid::TimeGrid{T}, tableau::RKTableau{S}, r::Int;
                          nev::Int=3, tol::Real=1e-11, x0=nothing,
                          error_estimation::Bool=false,
                          periodic_solution::Bool=false,
                          safety_factor::Real=2.0,
                          embedded_eigs::Bool=false,
                          mass_matrix::Union{Nothing, AbstractMatrix}=nothing,
                          static_threshold::Int=32) where {D, T, S}
    p = length(grid.h)
    BSIZE = (S + 1) * D
    state_size = (r + 1) * BSIZE
    nev = min(nev, state_size - 1)
    kdim = min(max(30, nev + 10), state_size)

    build = tab -> begin
        sysm = build_system_matrices(prob, grid, tab, r;
                                     static_threshold=static_threshold, mass_matrix=mass_matrix)
        lazy = MonodromyMap(prob, grid, tab, sysm, p, r, state_size)
        (lazy, SparseMonodromyMap(lazy))
    end

    lazy, Phi = build(tableau)
    x0v = x0 === nothing ? _det_start(state_size) : copy(x0)
    vals, vecs, info = eigsolve(Phi, x0v, nev, :LM; tol=tol, krylovdim=kdim)
    mu = ComplexF64(vals[1])
    multipliers = ComplexF64.(vals)
    modes = [ComplexF64.(v) for v in vecs]

    fix = nothing; V = nothing
    if periodic_solution
        V = inhomogeneous_sweep(lazy)
        fix, _ = linsolve(Phi, V, zeros(T, state_size), 1, -1; tol=Float64(tol))
    end
    sol = FloquetSolution(mu, abs(mu), multipliers, modes, fix, info.converged)
    error_estimation || return sol

    # ---- embedded-pair error analysis ------------------------------------
    x = modes[1]
    lvals, lvecs, _ = eigsolve(transpose(Phi), x0v, nev, :LM; tol=tol, krylovdim=kdim)
    y = ComplexF64.(lvecs[argmin(abs.(ComplexF64.(lvals) .- mu))])
    denom = _blin(y, x)
    kappa = norm(x) * norm(y) / abs(denom)
    Phix = Phi * x
    gap = length(multipliers) >= 2 ?
          minimum(abs(mu - multipliers[k]) for k in 2:length(multipliers)) : NaN
    node_idx = [b * BSIZE + d for b in 0:r for d in 1:D]

    # Q companion priority: classical embedded pair > cross-family collocation
    # companion (one order below) > drop-node quadrature rule (conservative).
    # For same-(a,c) companions ΔΦ is a genuinely small perturbation and the
    # first-order formula δμ = yᵀΔΦx/yᵀx applies. A cross-family companion has
    # different abscissae, so its stage rows differ at O(h) ("gauge" difference
    # that cancels only in the eigenvalues) — there the exact difference μ − μ̂
    # is used instead, and mode/fixpoint distances are measured on the node
    # (physical) components of the state only.
    tabQ = quadrature_companion_tableau(tableau)
    # I channel only for non-collocation tableaux: for collocation the block
    # data lie ON the collocation polynomial, so every reduced interpolant
    # reproduces them identically (null perturbation); there the cross-family
    # Q companion carries the interpolation uncertainty as well.
    tabI = tableau.strategy == collocation ? nothing :
           embedded_tableau(tableau; quadrature=false, interpolation=true)
    if tabQ === nothing && tabI === nothing
        error("error_estimation: no embedded companion exists for this tableau " *
              "(single-stage method with endpoint interpolation strategy?)")
    end
    tabQ === nothing && @warn "quadrature channel unavailable for this tableau; " *
                              "the bar uses the interpolation channel only" maxlog=1

    dmuQ = ComplexF64(NaN, NaN); modeQ = NaN; muhatQ = ComplexF64(NaN, NaN)
    lazyQ = nothing; PhiQ = nothing; same_ac = false
    if tabQ !== nothing
        same_ac = tabQ.c == tableau.c && tabQ.a == tableau.a
        lazyQ, PhiQ = build(tabQ)
        if same_ac
            dPhixQ = Phix .- PhiQ * x
            dmuQ = _blin(y, dPhixQ) / denom
            modeQ = isnan(gap) ? NaN : norm(dPhixQ .- dmuQ .* x) / (gap * norm(x))
            if embedded_eigs
                v, _, _ = eigsolve(PhiQ, x0v, 1, :LM; tol=tol, krylovdim=kdim)
                muhatQ = ComplexF64(v[1])
            end
        else
            v, vc, _ = eigsolve(PhiQ, x0v, min(2, state_size - 1), :LM; tol=tol, krylovdim=kdim)
            j = argmin(abs.(ComplexF64.(v) .- mu))    # complex pair: match the partner
            muhatQ = ComplexF64(v[j])
            dmuQ = mu - muhatQ
            modeQ = _node_sin_angle(x, ComplexF64.(vc[j]), node_idx)
        end
    end

    dmuI = ComplexF64(NaN, NaN); modeI = NaN; muhatI = ComplexF64(NaN, NaN)
    if tabI !== nothing
        lazyI, PhiI = build(tabI)
        dPhixI = Phix .- PhiI * x
        dmuI = _blin(y, dPhixI) / denom
        modeI = isnan(gap) ? NaN : norm(dPhixI .- dmuI .* x) / (gap * norm(x))
        if embedded_eigs
            v, _, _ = eigsolve(PhiI, x0v, 1, :LM; tol=tol, krylovdim=kdim)
            muhatI = ComplexF64(v[1])
        end
    end

    barQ = isnan(abs(dmuQ)) ? NaN : abs(dmuQ)
    barI = isnan(abs(dmuI)) ? NaN : abs(dmuI)
    bar = safety_factor * ((isnan(barQ) ? 0.0 : barQ) + (isnan(barI) ? 0.0 : barI))
    delta_mu = (isnan(abs(dmuQ)) ? zero(ComplexF64) : dmuQ) +
               (isnan(abs(dmuI)) ? zero(ComplexF64) : dmuI)
    mode_err = (isnan(modeQ) && isnan(modeI)) ? NaN :
               safety_factor * ((isnan(modeQ) ? 0.0 : modeQ) + (isnan(modeI) ? 0.0 : modeI))

    # fixed point: exact two-method difference on the node components
    fix_err = NaN; fix_delta = nothing
    if periodic_solution && fix !== nothing
        lazyF = nothing; PhiF = nothing
        if !same_ac && PhiQ !== nothing
            lazyF, PhiF = lazyQ, PhiQ         # cross-family companion is a full method
        else
            tabF = companion_tableau(tableau)
            tabF === nothing || ((lazyF, PhiF) = build(tabF))
        end
        if PhiF !== nothing
            V_F = inhomogeneous_sweep(lazyF)
            fixF, _ = linsolve(PhiF, V_F, zeros(T, state_size), 1, -1; tol=Float64(tol))
            fix_delta = fix .- fixF
            fix_err = safety_factor * norm(fix_delta[node_idx]) / max(norm(fix[node_idx]), eps())
        end
    end

    est = FloquetErrorEstimate(bar, delta_mu, barQ, barI, kappa, gap, mode_err,
                               fix_err, fix_delta, muhatQ, muhatI)
    return sol, est
end

"Sine of the principal angle between two vectors restricted to the node slots."
function _node_sin_angle(x::AbstractVector, xh::AbstractVector, node_idx::Vector{Int})
    a = view(x, node_idx); b = view(xh, node_idx)
    c = min(abs(dot(a, b)) / (norm(a) * norm(b)), 1.0)
    return sqrt(1 - c^2)
end

"""
    spectral_radius(prob, grid, tableau, r; error_estimation=false, kwargs...)

Convenience wrapper: the spectral radius of the monodromy operator. With
`error_estimation = true` returns `(rho, rho_error_bar)` — the bar arrives as
a separate output, so the plain call keeps the original interface.
"""
function spectral_radius(prob::LDDEProblem, grid::TimeGrid, tableau::RKTableau, r::Int;
                         error_estimation::Bool=false, kwargs...)
    if error_estimation
        sol, est = floquet_analysis(prob, grid, tableau, r; error_estimation=true, kwargs...)
        return sol.spectral_radius, est.mu_error
    end
    sol = floquet_analysis(prob, grid, tableau, r; kwargs...)
    return sol.spectral_radius
end
