using ForwardDiff
using StaticArrays
using LinearAlgebra

"""
    extract_SDM_system(rhs, params, ::Val{D}; t_test=0.0) where D

Automatically extracts the linear system matrices A and B from a DDE RHS function.
"""
function extract_SDM_system(rhs, params, ::Val{D}; t_test=0.0) where D
    u_test = zeros(MMatrix{D, 1, Float64})
    
    # Extract A (proportional part)
    A_f = (t) -> begin
        ForwardDiff.jacobian(u -> rhs(u, (p, t_hist) -> zeros(SVector{D, Float64}), params, t), u_test)
    end
    
    # Extract B_i (delay parts)
    # We assume one delay for now, but LDDEProblem supports multiple
    # Find tau by looking at the h calls (this is a bit manual, but works for Mathieu/Milling)
    # For now, we assume a single delay τ = 2π as per Mathieu defaults if not specified.
    # To be more general, we can inspect the rhs or pass delays explicitly.
    
    B_f = (t) -> begin
        # Jacobian with respect to the delayed state
        ForwardDiff.jacobian(u_tau -> rhs(u_test, (p, t_hist) -> u_tau, params, t), u_test)
    end
    
    tau_f = (t) -> 2π # Default for Mathieu
    
    AMx = ProportionalMX(A_f)
    BMx = DelayMX(tau_f, B_f)
    
    # Extract c (additive part)
    c_f = (t) -> begin
        rhs(zeros(SVector{D, Float64}), (p, t_hist) -> zeros(SVector{D, Float64}), params, t)
    end
    cVec = Additive(c_f)
    
    return LDDEProblem{D, Float64}(AMx, [BMx], cVec)
end

function build_system_matrices(problem::LDDEProblem{D, T}, grid::TimeGrid{T}, tableau::RKTableau{S, T, CE}, r::Int) where {D, T, S, CE}
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
                rel_idx = (t_del - t_start) / h_const + r + 1; m_idx = floor(Int, rel_idx)
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
