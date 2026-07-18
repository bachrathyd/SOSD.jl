using MFCM
using Plots
using StaticArrays
using KrylovKit
using MDBM

# Seasonal Maturation Model (Linearized)
# y'(t) = -(a + b cos(t)) y(t - tau)

function bio_rhs(u, h, p, t)
    # p = [a, b, tau]
    a, b, tau = p
    hist = h(p, t - tau)
    y_del = hist[1]
    
    du = -(a + b * cos(t)) * y_del
    return @SVector [du]
end

function bio_mumax(a, b; tau=2.0, p_steps=60)
    T = 2π # Period of coefficients
    prob = extract_SDM_system(bio_rhs, [a, b, tau], Val(1))
    tableau = GL2Tableau()
    
    grid = TimeGrid(collect(range(0.0, T, length=p_steps+1)))
    
    # r = tau / h
    h = T / p_steps
    r_steps = round(Int, tau / h)
    
    sys_mats = build_system_matrices(prob, grid, tableau, r_steps)
    
    S = 2 # GL2
    D = 1
    BSIZE = (S + 1) * D
    state_size = (r_steps + 1) * BSIZE
    m = MonodromyMap(prob, grid, tableau, sys_mats, p_steps, r_steps, state_size)
    
    vals, _ = eigsolve(m, rand(m.state_size), 1, :LM)
    return abs(vals[1])
end

function generate_bio_chart()
    println("Generating Biological Stability Chart...")
    
    # a in [0, 2], b in [0, 2]
    axis = [Axis(0.0:0.5:2.0), Axis(0.0:0.5:2.0)]
    
    f_mdbm(a, b) = bio_mumax(a, b) - 1.0
    
    mdbm_prob = MDBM_Problem(f_mdbm, axis)
    MDBM.solve!(mdbm_prob, 3)
    
    p = plot(title="Biological Stability Chart (Seasonal Maturation)", xlabel="Average Growth Rate (a)", ylabel="Seasonal Amplitude (b)")
    points = getinterpolatedsolution(mdbm_prob)
    if !isempty(points)
        scatter!(p, points[1, :], points[2, :], markersize=1, label="Stability Boundary", color=:green)
    end
    
    savefig(p, "bio_stability_chart.png")
    println("Biological chart saved to bio_stability_chart.png")
end

generate_bio_chart()
