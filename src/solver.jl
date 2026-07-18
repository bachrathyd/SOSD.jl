function LinearMaps._unsafe_mul!(y_out::AbstractVector{T}, m::MonodromyMap{D, S, T, W, BSIZE, CE}, x_in::AbstractVector{T}) where {D, S, T, W, BSIZE, CE}
    return base_sweep!(y_out, m, x_in, false)
end

function base_sweep!(y_out::AbstractVector{T}, m::MonodromyMap{D, S, T, W, BSIZE, CE}, x_in::AbstractVector{T}, include_additive::Bool) where {D, S, T, W, BSIZE, CE}
    p = m.p; r = m.r; sys_mats = m.sys_mats; history = m.history_buffer 
    strategy = m.tableau.strategy
    
    for i in 0:r
        src_idx = i * BSIZE + 1; dst_idx = (r - i) * BSIZE + 1
        @views history[dst_idx : dst_idx + BSIZE - 1] .= x_in[src_idx : src_idx + BSIZE - 1]
    end
    
    n_delays = length(sys_mats.M_del)
    for n in 1:p
        v_prev_idx = (n + r - 1) * BSIZE + 1
        y_n = SVector{D, T}(ntuple(i -> history[v_prev_idx + i - 1], Val(D)))
        v_curr = sys_mats.M_prop[n] * y_n
        
        for k in 1:n_delays
            indices_kn = sys_mats.delay_indices[k][n]; weights_kn = sys_mats.delay_weights[k][n]; M_del_kn = sys_mats.M_del[k][n]
            for j in 1:S
                y_del_kj = fetch_delayed_state(indices_kn[j], weights_kn[j], history, BSIZE, Val(D), Val(S), Val(strategy))
                v_curr += M_del_kn[j] * y_del_kj
            end
        end
        if include_additive; v_curr += sys_mats.c_vector[n]; end
        v_curr_idx = (n + r) * BSIZE + 1
        for i in 1:BSIZE; history[v_curr_idx + i - 1] = v_curr[i]; end
    end
    for i in 0:r
        dst_idx = i * BSIZE + 1; src_idx = (p + r - i) * BSIZE + 1
        @views y_out[dst_idx : dst_idx + BSIZE - 1] .= history[src_idx : src_idx + BSIZE - 1]
    end
    return y_out
end

@inline function fetch_delayed_state(m_idx::Int, weights::SVector{W, T}, history::Vector{T}, BSIZE::Int, ::Val{D}, ::Val{S}, ::Val{SOSD.collocation}) where {W, T, D, S}
    # Stage-based interpolation (Collocation)
    idx_m = (m_idx - 1) * BSIZE + 1; idx_next = m_idx * BSIZE + 1
    res = zero(SVector{D, T})
    y_m = SVector{D, T}(ntuple(d -> history[idx_m + d - 1], Val(D)))
    res += weights[1] * y_m
    for i in 1:S
        Yi = SVector{D, T}(ntuple(d -> history[idx_next + D + (i-1)*D + d - 1], Val(D)))
        res += weights[i+1] * Yi
    end
    y_next = SVector{D, T}(ntuple(d -> history[idx_next + d - 1], Val(D)))
    res += weights[S+2] * y_next
    return res
end

@inline function fetch_delayed_state(m_idx::Int, weights::SVector{W, T}, history::Vector{T}, BSIZE::Int, ::Val{D}, ::Val{S}, ::Val{SOSD.denseoutput}) where {W, T, D, S}
    # Stage-based mapping for Dense Output (same mapping as collocation)
    return fetch_delayed_state(m_idx, weights, history, BSIZE, Val(D), Val(S), Val(SOSD.collocation))
end

@inline function fetch_delayed_state(m_idx::Int, weights::SVector{W, T}, history::Vector{T}, BSIZE::Int, ::Val{D}, ::Val{S}, ::Val{SOSD.endpoint}) where {W, T, D, S}
    # Centered endpoint interpolation
    res = zero(SVector{D, T}); offset = (W - 1) ÷ 2
    for i in 1:W
        block_idx = m_idx + (i - 1 - offset)
        if block_idx < 1; block_idx = 1; end 
        idx = (block_idx - 1) * BSIZE + 1
        yi = SVector{D, T}(ntuple(d -> history[idx + d - 1], Val(D)))
        res += weights[i] * yi
    end
    return res
end

function inhomogeneous_sweep(m::MonodromyMap{D, S, T, W, BSIZE, CE}) where {D, S, T, W, BSIZE, CE}
    y_out = zeros(T, m.state_size); x_zero = zeros(T, m.state_size)
    return base_sweep!(y_out, m, x_zero, true)
end

function solve_periodic_solution(m::MonodromyMap{D, S, T, W, BSIZE, CE}) where {D, S, T, W, BSIZE, CE}
    V = inhomogeneous_sweep(m); initial_guess = zeros(T, m.state_size)
    y_fixed, info = linsolve(m, V, initial_guess, 1, -1; tol=1e-10)
    return y_fixed
end
