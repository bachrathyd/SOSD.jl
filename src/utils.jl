using ForwardDiff
using StaticArrays
using LinearAlgebra

"""
    extract_SDM_system(rhs, params, ::Val{D}; delays=nothing, t_test=0.0) where D

Automatically extracts the linear system matrices `A(t)` and `B_k(t)` from a DDE
RHS function with the DifferentialEquations.jl signature `rhs(u, h, p, t)`.

The delay lags are taken from `delays` — a vector whose elements are either
constants or functions `t -> τ(t)`. If `delays === nothing`, the (constant) lags
are auto-detected by probing the history calls made by `rhs` at `t = t_test`.
Time-dependent delays must always be passed explicitly.

Each `B_k` is extracted by differentiating `rhs` with respect to the delayed
state of lag `τ_k` only (all other lags are masked to zero), so multiple delay
terms are separated correctly.
"""
function extract_SDM_system(rhs, params, ::Val{D}; delays=nothing, t_test=0.0) where D
    u_test = zeros(MMatrix{D, 1, Float64})

    # --- Determine the delay lag functions ---
    tau_funcs = if delays === nothing
        lags = Float64[]
        h_probe = (p, t_hist) -> begin
            lag = t_test - t_hist
            if lag > 0 && !any(l -> isapprox(l, lag; atol=1e-12, rtol=1e-8), lags)
                push!(lags, lag)
            end
            return zeros(SVector{D, Float64})
        end
        rhs(zeros(SVector{D, Float64}), h_probe, params, t_test)
        sort!(lags)
        Function[let lag_c = lag; (t) -> lag_c; end for lag in lags]
    else
        Function[tau isa Number ? (let val = float(tau); (t) -> val; end) : tau for tau in delays]
    end

    # --- A(t): Jacobian w.r.t. the current state (history masked to zero) ---
    A_f = (t) -> begin
        ForwardDiff.jacobian(u -> rhs(u, (p, t_hist) -> zeros(SVector{D, Float64}), params, t), u_test)
    end
    AMx = ProportionalMX(A_f)

    # --- B_k(t): Jacobian w.r.t. the delayed state of lag τ_k only ---
    BMxs = [begin
        B_f = let tau_f = tau_f
            (t) -> begin
                tau_val = tau_f(t)
                ForwardDiff.jacobian(u_test) do u_tau
                    h_masked = (p, t_hist) -> begin
                        mask = isapprox(t - t_hist, tau_val; atol=1e-10, rtol=1e-8) ?
                               one(eltype(u_tau)) : zero(eltype(u_tau))
                        return u_tau .* mask
                    end
                    rhs(u_test, h_masked, params, t)
                end
            end
        end
        DelayMX(tau_f, B_f)
    end for tau_f in tau_funcs]

    # --- c(t): additive part (zero state, zero history) ---
    c_f = (t) -> begin
        rhs(zeros(SVector{D, Float64}), (p, t_hist) -> zeros(SVector{D, Float64}), params, t)
    end
    cVec = Additive(c_f)

    B_final = isempty(BMxs) ? DelayMX[] : BMxs
    return LDDEProblem{D, Float64}(AMx, B_final, cVec)
end

"""
    build_system_matrices(problem, grid, tableau, r; static_threshold=32)

Precompute the per-step transition blocks. Uses the StaticArrays fast path when
the stage-coupled dimension `S*D ≤ static_threshold`, and automatically falls
back to the heap-allocated [`build_system_matrices_dense`](@ref) above that
(StaticArrays inversion of large `SMatrix` types is prohibitively slow to compile).
"""
function build_system_matrices(problem::LDDEProblem{D, T}, grid::TimeGrid{T}, tableau::RKTableau{S, T, CE}, r::Int; static_threshold::Int=32) where {D, T, S, CE}
    # Dense fallback when either the stage-coupled dimension is large or the
    # state dimension alone is (StaticArrays inv/unrolling compile cost grows
    # steeply with D even for few stages).
    if S * D > static_threshold || D > 12
        return build_system_matrices_dense(problem, grid, tableau, r)
    end
    n_steps = length(grid.h)
    BSIZE = (S + 1) * D
    W = (tableau.strategy == endpoint) ? tableau.order : S + 2
    
    M_prop = Vector{SMatrix{BSIZE, D, T}}(undef, n_steps)
    M_del = [Vector{SVector{S, SMatrix{BSIZE, D, T}}}(undef, n_steps) for _ in 1:length(problem.B)]
    delay_indices = [Vector{SVector{S, Int}}(undef, n_steps) for _ in 1:length(problem.B)]
    delay_weights = [Vector{SVector{S, SVector{W, T}}}(undef, n_steps) for _ in 1:length(problem.B)]
    c_vector = Vector{SVector{BSIZE, T}}(undef, n_steps)
    
    t_start = grid.t[1]; h_const = grid.h[1]; SD = S * D

    for n in 1:n_steps
        t_n = grid.t[n]; h_n = grid.h[n]
        M_local = MMatrix{SD, SD, T}(undef)
        for j in 1:S
            A_j = problem.A.f(t_n + tableau.c[j] * h_n)
            for i in 1:S
                for row_d in 1:D, col_d in 1:D
                    val = (i == j && row_d == col_d ? 1.0 : 0.0) - h_n * tableau.a[i, j] * A_j[row_d, col_d]
                    M_local[(i-1)*D + row_d, (j-1)*D + col_d] = val
                end
            end
        end
        M_inv = inv(SMatrix{SD, SD, T}(M_local))
        
        RHS_y = MMatrix{SD, D, T}(undef)
        for i in 1:S, d in 1:D; for col_d in 1:D; RHS_y[(i-1)*D + d, col_d] = (d == col_d ? 1.0 : 0.0); end; end
        Y_from_y = M_inv * SMatrix{SD, D, T}(RHS_y) 
        
        M_p = MMatrix{BSIZE, D, T}(undef); y_next_from_y = MMatrix{D, D, T}(I(D))
        for j in 1:S
            A_j = problem.A.f(t_n + tableau.c[j] * h_n)
            Y_j_from_y = Y_from_y[(j-1)*D+1 : j*D, :]
            y_next_from_y += h_n * tableau.b[j] * (A_j * Y_j_from_y)
        end
        M_p[1:D, : ] = y_next_from_y; M_p[D+1:end, :] = Y_from_y; M_prop[n] = SMatrix{BSIZE, D, T}(M_p)

        for k in 1:length(problem.B)
            Ms_k = Vector{SMatrix{BSIZE, D, T}}(undef, S); Idxs = Vector{Int}(undef, S); Ws = Vector{SVector{W, T}}(undef, S)
            for stage_idx in 1:S
                RHS_del_stage = zeros(MMatrix{SD, D, T}); B_stage = problem.B[k].f(t_n + tableau.c[stage_idx] * h_n)
                for i in 1:S
                    coeff = h_n * tableau.a[i, stage_idx]
                    for row_d in 1:D, col_d in 1:D; RHS_del_stage[(i-1)*D + row_d, col_d] = coeff * B_stage[row_d, col_d]; end
                end
                Y_from_del_stage = M_inv * SMatrix{SD, D, T}(RHS_del_stage)
                M_d_stage = MMatrix{BSIZE, D, T}(zeros(BSIZE, D)); y_next_from_del_stage = MMatrix{D, D, T}(zeros(D, D))
                for j in 1:S
                    A_j = problem.A.f(t_n + tableau.c[j] * h_n); Y_j_from_del_stage = Y_from_del_stage[(j-1)*D+1 : j*D, :]
                    term = A_j * Y_j_from_del_stage; if j == stage_idx; term += B_stage; end
                    y_next_from_del_stage += h_n * tableau.b[j] * term
                end
                M_d_stage[1:D, :] = y_next_from_del_stage; M_d_stage[D+1:end, :] = Y_from_del_stage; Ms_k[stage_idx] = SMatrix{BSIZE, D, T}(M_d_stage)
                
                t_ni = t_n + tableau.c[stage_idx] * h_n; tau_val = problem.B[k].tau(t_ni); t_del = t_ni - tau_val
                rel_idx = (t_del - t_start) / h_const + r + 1
                if rel_idx < 1 - 1e-6
                    error("Delayed state lookup falls before the stored history window: " *
                          "t = $(t_ni), lag τ = $(tau_val) ⇒ t−τ = $(t_del), but the history only " *
                          "reaches back to t = $(t_start - r*h_const) (r = $r, h = $h_const). " *
                          "Increase r so that r·h ≥ max lag, or fix the delay definition of the problem.")
                end
                m_idx = floor(Int, rel_idx)
                if m_idx >= (n_steps + r + 1); m_idx = n_steps + r; theta = 1.0; elseif m_idx < 1; m_idx = 1; theta = 0.0; else; theta = rel_idx - m_idx; end
                Idxs[stage_idx] = m_idx; Ws[stage_idx] = tableau.ce(theta)
            end
            M_del[k][n] = SVector{S}(Ms_k); delay_indices[k][n] = SVector{S}(Idxs); delay_weights[k][n] = SVector{S}(Ws)
        end
        
        RHS_c = zeros(MMatrix{SD, 1, T}); cs = [problem.c.f(t_n + tableau.c[j] * h_n) for j in 1:S]
        for i in 1:S
            val = zeros(MVector{D, T}); for j in 1:S; val .+= h_n * tableau.a[i, j] .* cs[j]; end
            for d in 1:D; RHS_c[(i-1)*D + d, 1] = val[d]; end
        end
        Y_from_c = M_inv * SMatrix{SD, 1, T}(RHS_c); y_next_from_c = zeros(MVector{D, T})
        for j in 1:S
            A_j = problem.A.f(t_n + tableau.c[j] * h_n); Y_j_from_c = Y_from_c[(j-1)*D+1 : j*D, 1]
            y_next_from_c .+= h_n * tableau.b[j] .* (A_j * Y_j_from_c .+ cs[j])
        end
        c_vec = MVector{BSIZE, T}(undef); c_vec[1:D] = y_next_from_c; c_vec[D+1:end] = Y_from_c; c_vector[n] = SVector{BSIZE, T}(c_vec)
    end
    return SystemMatrices{D, S, T, W, BSIZE}(M_prop, M_del, delay_indices, delay_weights, c_vector)
end

"""
    build_system_matrices_dense(problem, grid, tableau, r)

Heap-allocated (`Matrix`-based) variant of [`build_system_matrices`](@ref) for
systems whose stage-coupled dimension `S*D` is too large for StaticArrays
(high-dimensional FEM models, very high collocation orders). Mathematically
identical to the static path; uses an LU factorization of the stage-coupling
matrix instead of an explicit `SMatrix` inverse.
"""
function build_system_matrices_dense(problem::LDDEProblem{D, T}, grid::TimeGrid{T}, tableau::RKTableau{S, T, CE}, r::Int) where {D, T, S, CE}
    n_steps = length(grid.h)
    BSIZE = (S + 1) * D
    W = (tableau.strategy == endpoint) ? tableau.order : S + 2
    SD = S * D
    n_del = length(problem.B)

    M_prop = Vector{Matrix{T}}(undef, n_steps)
    M_del = [Vector{Vector{Matrix{T}}}(undef, n_steps) for _ in 1:n_del]
    delay_indices = [Vector{SVector{S, Int}}(undef, n_steps) for _ in 1:n_del]
    delay_weights = [Vector{SVector{S, SVector{W, T}}}(undef, n_steps) for _ in 1:n_del]
    c_vector = Vector{Vector{T}}(undef, n_steps)

    t_start = grid.t[1]; h_const = grid.h[1]
    A_stages = Vector{Matrix{T}}(undef, S)

    for n in 1:n_steps
        t_n = grid.t[n]; h_n = grid.h[n]

        # Stage-coupling matrix M_local = I - h * (a ⊗ A) and its factorization
        for j in 1:S
            A_stages[j] = Matrix{T}(problem.A.f(t_n + tableau.c[j] * h_n))
        end
        M_local = zeros(T, SD, SD)
        for j in 1:S, i in 1:S
            aij = h_n * tableau.a[i, j]
            if aij != 0
                @views M_local[(i-1)*D+1:i*D, (j-1)*D+1:j*D] .= (-aij) .* A_stages[j]
            end
        end
        for idx in 1:SD; M_local[idx, idx] += one(T); end
        F = lu!(M_local)

        # Proportional block: response of [y_{n+1}; Y_1..Y_S] to y_n
        RHS_y = zeros(T, SD, D)
        for i in 1:S, d in 1:D; RHS_y[(i-1)*D + d, d] = one(T); end
        Y_from_y = F \ RHS_y

        M_p = zeros(T, BSIZE, D)
        y_next_from_y = Matrix{T}(I, D, D)
        for j in 1:S
            @views y_next_from_y .+= (h_n * tableau.b[j]) .* (A_stages[j] * Y_from_y[(j-1)*D+1:j*D, :])
        end
        M_p[1:D, :] .= y_next_from_y
        M_p[D+1:end, :] .= Y_from_y
        M_prop[n] = M_p

        # Delay blocks: response to the delayed state fetched at each stage
        for k in 1:n_del
            Ms_k = Vector{Matrix{T}}(undef, S)
            Idxs = Vector{Int}(undef, S)
            Ws = Vector{SVector{W, T}}(undef, S)
            for stage_idx in 1:S
                B_stage = Matrix{T}(problem.B[k].f(t_n + tableau.c[stage_idx] * h_n))
                RHS_del = zeros(T, SD, D)
                for i in 1:S
                    coeff = h_n * tableau.a[i, stage_idx]
                    if coeff != 0
                        @views RHS_del[(i-1)*D+1:i*D, :] .= coeff .* B_stage
                    end
                end
                Y_from_del = F \ RHS_del
                y_next_from_del = zeros(T, D, D)
                for j in 1:S
                    term = A_stages[j] * @view Y_from_del[(j-1)*D+1:j*D, :]
                    if j == stage_idx; term .+= B_stage; end
                    y_next_from_del .+= (h_n * tableau.b[j]) .* term
                end
                M_d = zeros(T, BSIZE, D)
                M_d[1:D, :] .= y_next_from_del
                M_d[D+1:end, :] .= Y_from_del
                Ms_k[stage_idx] = M_d

                # Delayed-lookup index and interpolation weights
                t_ni = t_n + tableau.c[stage_idx] * h_n
                tau_val = problem.B[k].tau(t_ni)
                t_del = t_ni - tau_val
                rel_idx = (t_del - t_start) / h_const + r + 1
                if rel_idx < 1 - 1e-6
                    error("Delayed state lookup falls before the stored history window: " *
                          "t = $(t_ni), lag τ = $(tau_val) ⇒ t−τ = $(t_del), but the history only " *
                          "reaches back to t = $(t_start - r*h_const) (r = $r, h = $h_const). " *
                          "Increase r so that r·h ≥ max lag, or fix the delay definition of the problem.")
                end
                m_idx = floor(Int, rel_idx)
                if m_idx >= (n_steps + r + 1); m_idx = n_steps + r; theta = one(T); elseif m_idx < 1; m_idx = 1; theta = zero(T); else; theta = T(rel_idx - m_idx); end
                Idxs[stage_idx] = m_idx
                Ws[stage_idx] = tableau.ce(theta)
            end
            M_del[k][n] = Ms_k
            delay_indices[k][n] = SVector{S}(Idxs)
            delay_weights[k][n] = SVector{S}(Ws)
        end

        # Additive (inhomogeneous) block
        cs = [Vector{T}(problem.c.f(t_n + tableau.c[j] * h_n)) for j in 1:S]
        RHS_c = zeros(T, SD)
        for i in 1:S
            for j in 1:S
                @views RHS_c[(i-1)*D+1:i*D] .+= (h_n * tableau.a[i, j]) .* cs[j]
            end
        end
        Y_from_c = F \ RHS_c
        y_next_from_c = zeros(T, D)
        for j in 1:S
            y_next_from_c .+= (h_n * tableau.b[j]) .* (A_stages[j] * @view(Y_from_c[(j-1)*D+1:j*D]) .+ cs[j])
        end
        c_vec = zeros(T, BSIZE)
        c_vec[1:D] .= y_next_from_c
        c_vec[D+1:end] .= Y_from_c
        c_vector[n] = c_vec
    end
    return SystemMatricesDense{D, S, T, W, BSIZE}(M_prop, M_del, delay_indices, delay_weights, c_vector)
end
