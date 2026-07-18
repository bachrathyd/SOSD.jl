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
    sm = SparseMonodromyMap(m)
    x0 = rand(sm.state_size)
    vals, _, _ = eigsolve(sm, x0, 1, :LM; tol=1e-12)
    return abs(vals[1])
end

ref = 0.3514172614
ps = [40, 80, 160, 320]

println("Verifying Endpoint Strategy:")
for name in ["RK3", "RK4"]
    tab = (name == "RK3") ? RK3(strategy=SOSD.endpoint) : RK4(strategy=SOSD.endpoint)
    println("\nConvergence for $name (Endpoint):")
    errors = []
    for p in ps
        mu = get_mu(p, tab)
        err = abs(mu - ref)
        push!(errors, err)
        @printf("p=%3d, mu=%.12f, err=%.2e\n", p, mu, err)
    end
    log_ps = log.(ps); log_errs = log.(errors)
    k = -([log_ps ones(length(ps))] \ log_errs)[1]
    @printf("Calculated k: %.2f\n", k)
end
