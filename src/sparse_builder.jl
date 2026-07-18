function build_explicit_matrices(m::MonodromyMap{D, S, T, W, BSIZE, CE}) where {D, S, T, W, BSIZE, CE}
    p = m.p; r = m.r; sys_mats = m.sys_mats; strategy = m.tableau.strategy
    I_L, J_L, V_L = Int[], Int[], T[]; I_R, J_R, V_R = Int[], Int[], T[]
    for n in 1:p
        row_off = (n - 1) * BSIZE
        for i in 1:BSIZE; push!(I_L, row_off + i); push!(J_L, row_off + i); push!(V_L, 1.0); end
        if n == 1
            for col_d in 1:D, row_b in 1:BSIZE
                val = -sys_mats.M_prop[n][row_b, col_d]
                if val != 0; push!(I_R, row_off + row_b); push!(J_R, r * BSIZE + col_d); push!(V_R, val); end
            end
        else
            for col_d in 1:D, row_b in 1:BSIZE
                val = -sys_mats.M_prop[n][row_b, col_d]
                if val != 0; push!(I_L, row_off + row_b); push!(J_L, (n - 2) * BSIZE + col_d); push!(V_L, val); end
            end
        end
        for k in 1:length(sys_mats.M_del)
            for j in 1:S
                m_idx = sys_mats.delay_indices[k][n][j]; weights = sys_mats.delay_weights[k][n][j]; M_kj = sys_mats.M_del[k][n][j]
                if strategy == MFCM.collocation || strategy == MFCM.denseoutput
                    for block_rel in [0, 1]
                        b_idx = m_idx + block_rel - (r + 1)
                        if block_rel == 0
                            w1 = weights[1]
                            if w1 != 0
                                M_w = M_kj * w1
                                for col_d in 1:D, row_b in 1:BSIZE
                                    val = -M_w[row_b, col_d]
                                    if val != 0
                                        if b_idx <= 0; push!(I_R, row_off + row_b); push!(J_R, (b_idx + r) * BSIZE + col_d); push!(V_R, val);
                                        else; push!(I_L, row_off + row_b); push!(J_L, (b_idx - 1) * BSIZE + col_d); push!(V_L, val); end
                                    end
                                end
                            end
                        else
                            for stage_s in 1:S
                                w = weights[stage_s + 1]
                                if w != 0
                                    M_w = M_kj * w
                                    for col_d in 1:D, row_b in 1:BSIZE
                                        val = -M_w[row_b, col_d]
                                        if val != 0
                                            if b_idx <= 0; push!(I_R, row_off + row_b); push!(J_R, (b_idx + r) * BSIZE + D + (stage_s - 1) * D + col_d); push!(V_R, val);
                                            else; push!(I_L, row_off + row_b); push!(J_L, (b_idx - 1) * BSIZE + D + (stage_s - 1) * D + col_d); push!(V_L, val); end
                                        end
                                    end
                                end
                            end
                            w_end = weights[S+2]
                            if w_end != 0
                                M_w = M_kj * w_end
                                for col_d in 1:D, row_b in 1:BSIZE
                                    val = -M_w[row_b, col_d]
                                    if val != 0
                                        if b_idx <= 0; push!(I_R, row_off + row_b); push!(J_R, (b_idx + r) * BSIZE + col_d); push!(V_R, val);
                                        else; push!(I_L, row_off + row_b); push!(J_L, (b_idx - 1) * BSIZE + col_d); push!(V_L, val); end
                                    end
                                end
                            end
                        end
                    end
                else # endpoint
                    offset = (W - 1) ÷ 2
                    for i in 1:W
                        block_idx = m_idx + (i - 1 - offset); if block_idx < 1; block_idx = 1; end
                        b_idx = block_idx - (r + 1); w = weights[i]
                        if w != 0
                            M_w = M_kj * w
                            for col_d in 1:D, row_b in 1:BSIZE
                                val = -M_w[row_b, col_d]
                                if val != 0
                                    if b_idx <= 0; push!(I_R, row_off + row_b); push!(J_R, (b_idx + r) * BSIZE + col_d); push!(V_R, val);
                                    else; push!(I_L, row_off + row_b); push!(J_L, (b_idx - 1) * BSIZE + col_d); push!(V_L, val); end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    L = sparse(I_L, J_L, V_L, p * BSIZE, p * BSIZE); R = sparse(I_R, J_R, V_R, p * BSIZE, (r + 1) * BSIZE)
    return R, L
end
