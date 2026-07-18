using SOSD
using Plots
using LinearAlgebra
using StaticArrays
using KrylovKit
using BenchmarkTools
using Printf

# Mathieu equation parameters
const ζ = 0.2
const δ = 3.0
const ε = 3.0
const b = -0.5
const τ = 2π
const T = 2π

function mathieu_rhs(u, h, p, t)
    x1, x2 = u
    hist = h(p, t - τ)
    x1_delayed = hist[1]
    du1 = x2
    du2 = -2ζ * x2 - (δ + ε * cos(2π * t / T)) * x1 + b * x1_delayed
    return @SVector [du1, du2]
end

function get_mumax(p_steps, gl_order)
    prob = extract_SDM_system(mathieu_rhs, nothing, Val(2))
    grid = TimeGrid(collect(range(0.0, T, length=p_steps+1)))
    tableau = GL(gl_order)
    r_steps = p_steps # T = τ
    
    sys_mats = build_system_matrices(prob, grid, tableau, r_steps)
    S = gl_order
    D = 2
    state_size = (r_steps + 1) * (S + 1) * D
    m = MonodromyMap(prob, grid, tableau, sys_mats, p_steps, r_steps, state_size)
    
    # Use Sparse solver for consistency and speed at high orders
    sm = SparseMonodromyMap(m)
    
    x0 = rand(ComplexF64, sm.state_size)
    vals, _, _ = eigsolve(sm, x0, 1, :LM; tol=1e-12)
    return abs(vals[1])
end

println("Computing high-precision reference value...")
# Use GL(20) with p=100 as a reference (should be very precise)
mu_ref = get_mumax(100, 20)
@printf("Reference mu: %.16f\n", mu_ref)

ps = 1:60
ss = 1:20 # Start with 20 to avoid too long runs, can expand later

errors = zeros(length(ps), length(ss))
times = zeros(length(ps), length(ss))

println("Starting grid search (p x s)...")
for (j, s) in enumerate(ss)
    print("s = $s: ")
    for (i, p) in enumerate(ps)
        t = @elapsed begin
            mu = get_mumax(p, s)
        end
        errors[i, j] = abs(mu - mu_ref) / mu_ref
        times[i, j] = t
        print(".")
    end
    println(" Done.")
end

# Find optimums
target_errors = [1e-3, 1e-5, 1e-8, 1e-12]
println("\nOptimal Configurations (min time for target error):")
for te in target_errors
    best_time = Inf
    best_config = (0, 0)
    for j in 1:length(ss), i in 1:length(ps)
        if errors[i, j] <= te && times[i, j] < best_time
            best_time = times[i, j]
            best_config = (ps[i], ss[j])
        end
    end
    if best_config != (0, 0)
        @printf("Target Error %.0e: p=%d, s=%d (Time: %.4fs)\n", te, best_config[1], best_config[2], best_time)
    else
        @printf("Target Error %.0e: Not reached in grid\n", te)
    end
end

# Visualization
p1 = heatmap(ss, ps, log10.(errors .+ 1e-16), 
             title="Log10 Relative Error", xlabel="GL Order (s)", ylabel="Resolution (p)",
             clims=(-15, 0))

p2 = heatmap(ss, ps, times, 
             title="CPU Time (s)", xlabel="GL Order (s)", ylabel="Resolution (p)")

plot(p1, p2, layout=(1, 2), size=(1000, 400))
savefig("optimal_gl_search.png")
println("Results saved to optimal_gl_search.png")
