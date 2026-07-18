using MFCM
using Plots
using StaticArrays
using KrylovKit
using MDBM

# 1-DOF Milling Model
# M x'' + C x' + K x = -K_c * a * g(t) * (x(t) - x(t-tau))
# Here g(t) is the directional factor (periodic)

const M = 1.0
const K = (2π * 100.0)^2 # 100 Hz
const zeta = 0.02
const C = 2 * zeta * sqrt(M * K)
const Kc = 1e8 # N/m^2 (approx)
const fz = 2 # 2 teeth

function milling_rhs(u, h, p, t)
    # p = [Omega, a]
    Omega_rpm, a = p
    tau = 60.0 / (fz * Omega_rpm)
    T = 60.0 / Omega_rpm # Spindle period
    
    # g(t) for milling (simple approximation for half-immersion)
    # phase = (t % T) / T
    # g = (phase < 0.5) ? 1.0 : 0.0
    
    # More realistic: g(t) is sum of directional factors of teeth
    g = 0.0
    for i in 1:fz
        phi = 2π * Omega_rpm / 60.0 * t + (i-1) * 2π / fz
        phi_mod = mod(phi, 2π)
        # Entry/Exit angles (e.g. 0 to pi)
        if 0 <= phi_mod <= π
            g += sin(phi_mod) * cos(phi_mod) # directional factor approx
        end
    end

    x, v = u
    hist = h(p, t - tau)
    x_del = hist[1]
    
    du1 = v
    du2 = (1/M) * (-C * v - K * x - Kc * a * g * (x - x_del))
    return @SVector [du1, du2]
end

function milling_mumax(Omega_rpm, a; p_steps=80)
    T = 60.0 / Omega_rpm
    tau = 60.0 / (fz * Omega_rpm)
    # T / tau = fz. So p / r = fz.
    # If r = p_steps, p = fz * p_steps.
    
    prob = extract_SDM_system(milling_rhs, [Omega_rpm, a], Val(2))
    tableau = GL2Tableau()
    
    # p steps per spindle period T
    grid = TimeGrid(collect(range(0.0, T, length=p_steps+1)))
    
    # r = number of steps in one tau
    # r = p_steps / fz
    r_steps = round(Int, p_steps / fz)
    sys_mats = build_system_matrices(prob, grid, tableau, r_steps)
    
    S = 2 # GL2
    D = 2
    BSIZE = (S + 1) * D
    state_size = (r_steps + 1) * BSIZE
    m = MonodromyMap(prob, grid, tableau, sys_mats, p_steps, r_steps, state_size)
    
    vals, _ = eigsolve(m, rand(m.state_size), 1, :LM)
    return abs(vals[1])
end

function generate_milling_chart()
    println("Generating Milling Stability Chart...")
    
    # Omega in [2000, 10000] RPM, a in [0, 0.01] m
    axis = [Axis(2000.0:2000.0:10000.0), Axis(0.0:0.002:0.01)]
    
    f_mdbm(Omega, a) = milling_mumax(Omega, a) - 1.0
    
    mdbm_prob = MDBM_Problem(f_mdbm, axis)
    MDBM.solve!(mdbm_prob, 3)
    
    p = plot(title="Milling Stability Chart", xlabel="Spindle Speed [RPM]", ylabel="Depth of Cut [m]")
    points = getinterpolatedsolution(mdbm_prob)
    if !isempty(points)
        scatter!(p, points[1, :], points[2, :], markersize=1, label="Stability Boundary", color=:red)
    end
    
    savefig(p, "milling_stability_chart.png")
    println("Milling chart saved to milling_stability_chart.png")
end

generate_milling_chart()
