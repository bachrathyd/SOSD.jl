struct SparseMonodromyMap{T, LS, RM, BSIZE, R} <: LinearMaps.LinearMap{T}
    L_solver::LS
    R::RM
    state_size::Int
end

Base.size(m::SparseMonodromyMap) = (m.state_size, m.state_size)

function LinearMaps._unsafe_mul!(y_out::AbstractVector, m::SparseMonodromyMap{T, LS, RM, BSIZE, R}, x_in::AbstractVector) where {T, LS, RM, BSIZE, R}
    # x_in: [v_0, ..., v_{-r}]
    # We need v_hist: [v_{-r}, ..., v_0] for the sparse system
    # Use eltype(x_in) to support complex vectors if x_in is complex
    TE = eltype(x_in)
    v_hist = Vector{TE}(undef, (R + 1) * BSIZE)
    for i in 0:R
        src = i * BSIZE + 1
        dst = (R - i) * BSIZE + 1
        v_hist[dst : dst + BSIZE - 1] .= x_in[src : src + BSIZE - 1]
    end
    
    # Solve system L * V_period = R * v_hist
    rhs = m.R * v_hist
    v_period = m.L_solver \ rhs
    
    # Extract final history [v_p, ..., v_{p-r}]
    # v_period contains [v_1, ..., v_p]
    p = length(v_period) ÷ BSIZE
    
    # We need [v_p, v_{p-1}, ..., v_{p-r}]
    for i in 0:R
        # Block p-i
        k = p - i
        dst = i * BSIZE + 1
        if k >= 1
            src = (k - 1) * BSIZE + 1
            y_out[dst : dst + BSIZE - 1] .= v_period[src : src + BSIZE - 1]
        else
            # From initial history (overlap)
            src = (k + R) * BSIZE + 1
            y_out[dst : dst + BSIZE - 1] .= v_hist[src : src + BSIZE - 1]
        end
    end
    return y_out
end

function SparseMonodromyMap(m::MonodromyMap{D, S, T, W, BSIZE}) where {D, S, T, W, BSIZE}
    Q_hist, Q_period = build_explicit_matrices(m)
    # L = Q_period, R = -Q_hist
    L_solver = lu(Q_period)
    return SparseMonodromyMap{T, typeof(L_solver), typeof(Q_hist), BSIZE, m.r}(L_solver, -Q_hist, m.state_size)
end
