using SOSD
using MDBM
using Plots
using StaticArrays
using KrylovKit

# Fixed parameters
const a1 = 0.01
const τ = 2π
const T = 2π
const b = -0.0

function mathieu_mumax(delta, epsilon; p_steps=60)
    # Define RHS for specific delta, epsilon
    rhs = (u, h, p, t) -> begin
        x1, x2 = u
        hist = h(p, t - τ)
        du1 = x2
        du2 = -a1 * x2 - (delta + epsilon * cos(t)) * x1 + b * hist[1]
        return @SVector [du1, du2]
    end
    
    prob = extract_SDM_system(rhs, nothing, Val(2))
    tableau = GL(2)
    grid = TimeGrid(collect(range(0.0, T, length=p_steps+1)))
    sys_mats = build_system_matrices(prob, grid, tableau, p_steps)
    S = 2; D = 2
    state_size = (p_steps + 1) * (S + 1) * D
    m = MonodromyMap(prob, grid, tableau, sys_mats, p_steps, p_steps, state_size)
    
    vals, _ = eigsolve(m, rand(m.state_size), 1, :LM)
    return abs(vals[1])
end

function generate_stability_chart()
    println("Starting MDBM stability chart generation...")
    
    # Axis definition
    axis = [Axis(-2.0:1.0:5.0), Axis(-0.01:1.0:5.0)]
    
    # Function for MDBM: stable if < 0, unstable if > 0
    f_mdbm(delta, epsilon) = mathieu_mumax(delta, epsilon,p_steps=30) - 1.0
    
    iteration = 5 # number of refinements
    
    # MDBM call
    mdbm_prob = MDBM_Problem(f_mdbm, axis)
    MDBM.solve!(mdbm_prob, iteration)
    
    # Plotting
    p = plot(title="Stability Chart (Delayed Mathieu)", xlabel="delta", ylabel="epsilon")
    
    # Extract points from MDBM
    points = getinterpolatedsolution(mdbm_prob)
    
    if !isempty(points)
        # points is a vector of vectors or a matrix?
        # getinterpolatedsolution usually returns a matrix with 2 rows for 2D
        scatter!(p, points[1, :], points[2, :], markersize=1, label="Stability Boundary", color=:black)
    end
    
    savefig(p, "mathieu_stability_chart.png")
    println("Stability chart saved to mathieu_stability_chart.png")
end

generate_stability_chart()
