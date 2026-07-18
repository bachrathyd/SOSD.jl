using SOSD
using StaticArrays
using KrylovKit
using Printf
using LinearAlgebra

# Mathieu Problem
function create_mathieu(δ, ε, b0, a1; T=2π)
    AMx = SOSD.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = SOSD.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = SOSD.Additive(t -> @SVector [0.0, sin(4π / T * t)])
    SOSD.LDDEProblem{2, Float64}(AMx, [BMx1], cVec)
end

δ, ε, b0, a1 = 3.0, 0.2, -0.15, 0.1
T = 2π
prob = create_mathieu(δ, ε, b0, a1, T=T)

function get_mu(p, tab)
    grid = TimeGrid(collect(range(0.0, T, length=p+1)))
    sys = build_system_matrices(prob, grid, tab, p)
    m = MonodromyMap(prob, grid, tab, sys, p, p, (p+1)*(size(tab.a,1)+1)*2)
    vals, _ = eigsolve(m, rand(m.state_size), 1, :LM; tol=1e-14)
    return abs(vals[1])
end

println("Calculating Reference Values...")
mu_ref1 = get_mu(500, GL(10))
mu_ref2 = get_mu(800, GL(10))
mu_ref3 = get_mu(1000, GL(10))
mu_ref4 = get_mu(500, GL(12))

@printf("GL(10) p=500:  %.18f\n", mu_ref1)
@printf("GL(10) p=800:  %.18f\n", mu_ref2)
@printf("GL(10) p=1000: %.18f\n", mu_ref3)
@printf("GL(12) p=500:  %.18f\n", mu_ref4)

# Use mu_ref3 as ground truth for now
ref = mu_ref3

ps = [20, 40, 80, 160, 320]
methods = [
    ("RK4", RK4(), 4),
    ("RK5", RK5(), 5)
]

for (name, tab, target_k) in methods
    println("\nConvergence for $name:")
    errors = []
    for p in ps
        mu = get_mu(p, tab)
        err = abs(mu - ref)
        push!(errors, err)
        @printf("p=%3d, mu=%.12f, err=%.2e\n", p, mu, err)
    end
    
    log_ps = log.(ps)
    log_errs = log.(errors)
    coeffs = [log_ps ones(length(ps))] \ log_errs
    k = -coeffs[1]
    @printf("Calculated k: %.2f (Target: %.1f)\n", k, Float64(target_k))
end
