using MFCM
using Plots
using LinearAlgebra

function verify_interpolation()
    # Use GL2Tableau (Order 4)
    tableau = GL2Tableau()
    h = 0.1
    t_n = 0.0
    
    # Test function: x(t) = exp(t)
    # x'(t) = x(t)
    # Stage values Y_i = x(t_n + c_i * h)
    y_n = exp(t_n)
    Y1 = exp(t_n + tableau.c[1] * h)
    Y2 = exp(t_n + tableau.c[2] * h)
    
    # We also have y_{n+1} = exp(t_n + h)
    y_next = exp(t_n + h)
    
    # The state vector v_n stores [y_{n+1}, Y1, Y2] (Wait, check solver.jl)
    # In solver.jl: v_curr = vcat(Vector(y_next), Y_vec)
    # So v_n = [y_{n+1}, Y1, Y2] ?
    # Let's check history_v usage:
    # v_prev = history_v[n + r]
    # y_n = SVector{D, T}(v_prev[1:D])
    # This means v_n stores the START of the next step, which is the END of the current step.
    # So history_v[m_idx] corresponds to step m_idx.
    # v[m_idx] = [y(t_{m_idx}), Y_{1, m_idx-1}, Y_{2, m_idx-1}] ?
    
    # Actually, let's look at Step 5 in base_sweep!:
    # v_curr = vcat(Vector(y_next), Y_vec)
    # history_v[n + r + 1] = v_curr
    # This stores y_{n+1} and the stages of step n.
    
    # To interpolate in step n (t in [t_n, t_{n+1}]):
    # we need y_n and Y_{1,n}, Y_{2,n}.
    # y_n is in history_v[n+r][1:D].
    # Y_{i,n} are in history_v[n+r+1][D+1 : end].
    
    # Current interpolation in solver.jl:
    # nodes = [0, c1, c2]
    # vals = [y_n, Y1, Y2]
    
    nodes = vcat(0.0, [tableau.c[1], tableau.c[2]])
    vals = [y_n, Y1, Y2]
    
    # Reference function for comparison
    f_ref(t) = exp(t)
    
    # Lagrange interpolation
    function lagrange(theta, nodes, vals)
        res = 0.0
        for i in 1:length(nodes)
            w = 1.0
            for j in 1:length(nodes)
                if i != j
                    w *= (theta - nodes[j]) / (nodes[i] - nodes[j])
                end
            end
            res += w * vals[i]
        end
        return res
    end

    thetas = 0.0:0.01:1.0
    t_vals = t_n .+ thetas .* h
    y_ref = f_ref.(t_vals)
    
    # 3-point Lagrange interpolation (y_n, Y1, Y2)
    nodes3 = vcat(0.0, [tableau.c[1], tableau.c[2]])
    vals3 = [y_n, Y1, Y2]
    y_interp3 = [lagrange(theta, nodes3, vals3) for theta in thetas]
    err3 = abs.(y_ref .- y_interp3)

    # 4-point Lagrange interpolation (y_n, Y1, Y2, y_next)
    nodes4 = vcat(0.0, [tableau.c[1], tableau.c[2]], 1.0)
    vals4 = [y_n, Y1, Y2, y_next]
    y_interp4 = [lagrange(theta, nodes4, vals4) for theta in thetas]
    err4 = abs.(y_ref .- y_interp4)
    
    # Plotting
    p1 = plot(thetas, y_ref, label="Reference", title="Interpolation Verification (GL2)")
    plot!(p1, thetas, y_interp3, label="3-pt (y_n, Y1, Y2)", linestyle=:dash)
    plot!(p1, thetas, y_interp4, label="4-pt (y_n, Y1, Y2, y_next)", linestyle=:dot)
    
    p2 = plot(thetas, err3, label="Error 3-pt", title="Interpolation Error", yscale=:log10)
    plot!(p2, thetas, err4, label="Error 4-pt")
    
    p = plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig(p, "interpolation_verification.png")
    
    println("Max error (3-pt): ", maximum(err3))
    println("Max error (4-pt): ", maximum(err4))
end

verify_interpolation()
