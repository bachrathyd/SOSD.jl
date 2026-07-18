using SOSD
using Plots
using LinearAlgebra
using StaticArrays

function verify_mfcm_interpolation()
    tableau = GL2Tableau()
    h = 0.1
    r = 1
    p = 1
    
    f_ref(t) = exp(t)
    D = 1
    S = 2
    BSIZE = (S + 1) * D
    
    # history_v size p+r+1 = 1+1+1 = 3
    # v_{-1}, v_0, v_1
    
    # Step -1 (t in [-h, 0]):
    y_m1 = f_ref(-h)
    Y1_m1 = f_ref(-h + tableau.c[1]*h)
    Y2_m1 = f_ref(-h + tableau.c[2]*h)
    y0 = f_ref(0.0)
    v_m1 = [y0, Y1_m1, Y2_m1]
    
    # Step 0 (t in [0, h]):
    Y1_0 = f_ref(0.0 + tableau.c[1]*h)
    Y2_0 = f_ref(0.0 + tableau.c[2]*h)
    y1 = f_ref(h)
    v_0 = [y1, Y1_0, Y2_0]
    
    # Step 1 (t in [h, 2h]):
    Y1_1 = f_ref(h + tableau.c[1]*h)
    Y2_1 = f_ref(h + tableau.c[2]*h)
    y2 = f_ref(2*h)
    v_1 = [y2, Y1_1, Y2_1]
    
    # Step 2:
    y3 = f_ref(3*h)
    v_2 = [y3, y3, y3] # filler
    
    history = [v_m1..., v_0..., v_1..., v_2...]
    
    grid = SOSD.TimeGrid([h, 2*h, 3*h]) # t_1=h
    prob = LDDEProblem{1, Float64}(ProportionalMX(t->zeros(1,1)), DelayMX[], Additive(t->zeros(1)))
    sys_mats = build_system_matrices(prob, grid, tableau, r)
    
    m = MonodromyMap(prob, grid, tableau, sys_mats, 1, 1, (r+1)*BSIZE)
    
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
        y = SOSD.fetch_delayed_state(m_idx, weights, history, BSIZE, Val(D), Val(S))[1]
        push!(y_interp, y)
    end
    
    err = abs.(y_ref .- y_interp)
    
    p1 = plot(thetas, y_ref, label="Reference", title="SOSD fetch_delayed_state Verification")
    plot!(p1, thetas, y_interp, label="SOSD Interp", linestyle=:dash)
    
    p2 = plot(thetas, err, label="Error", title="Interpolation Error", yscale=:log10)
    
    p = plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig(p, "mfcm_interp_verification.png")
    
    println("Max error: ", maximum(err))
end

verify_mfcm_interpolation()
