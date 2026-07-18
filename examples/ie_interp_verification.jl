using MFCM
using Plots
using LinearAlgebra
using StaticArrays

function verify_ie_interpolation()
    tableau = ImplicitEuler()
    h = 0.1
    r = 1
    
    f_ref(t) = exp(t)
    D = 1
    S = 1
    BSIZE = (S + 1) * D
    
    # Step 0:
    y0 = f_ref(0.0)
    y1 = f_ref(h)
    
    # Step 1:
    y2 = f_ref(2*h)
    
    grid = MFCM.TimeGrid([h, 2*h, 3*h])
    prob = LDDEProblem{1, Float64}(ProportionalMX(t->zeros(1,1)), DelayMX[], Additive(t->zeros(1)))
    sys_mats = build_system_matrices(prob, grid, tableau, r)
    
    # Construction of MonodromyMap needs correct type params or let Julia infer
    m = MonodromyMap(prob, grid, tableau, sys_mats, 1, r, (r+1)*BSIZE)
    
    # history in flat format [v_0, v_{-1}, ..., v_{-r}]?
    # No, base_sweep uses [v_0, v_{-1}, ..., v_{-r}] and converts to [v_{-r}, ..., v_0] in history buffer.
    # Actually, get_delayed_state_from_flat_history uses the history buffer format.
    
    # Let's use the helper fetch_delayed_state but we need a history buffer.
    # history buffer in solver: [v_{-r}, ..., v_p]
    # For IE s=1, BSIZE=2. v_n = [y_{n+1}, Y_{1,n}]
    # Y_{1,n} = y_{n+1} for IE.
    
    v_m1 = [y0, y0]
    v_0 = [y1, y1]
    v_1 = [y2, y2]
    
    # history buffer for get_delayed_state_from_flat_history: [v_{-1}, v_0, v_1, v_2]
    y3 = f_ref(3*h)
    v_2 = [y3, y3]
    history = [v_m1..., v_0..., v_1..., v_2...]
    
    thetas = 0.0:0.01:1.0
    t_vals = h .+ thetas .* h
    y_ref = f_ref.(t_vals)
    
    y_interp = Float64[]
    for t in t_vals
        # rel_idx calculation
        t_start = grid.t[1]
        rel_idx = (t - t_start) / h + r + 1
        m_idx = floor(Int, rel_idx)
        theta = rel_idx - m_idx
        
        weights = m.tableau.ce(theta)
        y = MFCM.fetch_delayed_state(m_idx, weights, history, BSIZE, Val(1), Val(1))[1]
        push!(y_interp, y)
    end
    
    err = abs.(y_ref .- y_interp)
    
    p1 = plot(thetas, y_ref, label="Reference", title="IE Interpolation Verification")
    plot!(p1, thetas, y_interp, label="IE Interp", linestyle=:dash)
    p2 = plot(thetas, err, label="Error", title="Interpolation Error", yscale=:log10)
    savefig(plot(p1, p2, layout=(2,1)), "ie_interp_verification.png")
    
    println("Max error (IE): ", maximum(err))
end

verify_ie_interpolation()
