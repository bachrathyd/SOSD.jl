using SOSD
using StaticArrays
using KrylovKit
using Printf
using LinearAlgebra
using RungeKutta

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
    # Use Sparse solver for more stability in this test
    sm = SparseMonodromyMap(m)
    x0 = rand(sm.state_size)
    vals, _, _ = eigsolve(sm, x0, 1, :LM; tol=1e-13)
    return abs(vals[1])
end

println("Check RK Orders in RungeKutta.jl:")
@printf("RK4 order: %d\n", RungeKutta.order(TableauRK4()))
@printf("RK5 order: %d\n", RungeKutta.order(TableauRK5()))

println("\nInternal Convergence of SOSD GL(2):")
# Use p=1000 as reference for GL(2)
ref_gl2 = get_mu(1000, GL(2))
ps = [50, 100, 200, 400, 800]
for p in ps
    mu = get_mu(p, GL(2))
    err = abs(mu - ref_gl2)
    @printf("p=%4d | mu=%.10f | err=%.2e\n", p, mu, err)
end

println("\nInternal Convergence of SOSD RK4:")
ref_rk4 = get_mu(2000, RK4())
for p in ps
    mu = get_mu(p, RK4())
    err = abs(mu - ref_rk4)
    @printf("p=%4d | mu=%.10f | err=%.2e\n", p, mu, err)
end

println("\nInternal Convergence of SOSD RK5:")
ref_rk5 = get_mu(2000, RK5())
for p in ps
    mu = get_mu(p, RK5())
    err = abs(mu - ref_rk5)
    @printf("p=%4d | mu=%.10f | err=%.2e\n", p, mu, err)
end

println("\nInternal Convergence of SOSD RK8:")
ref_rk8 = get_mu(2000, RK8())
for p in ps
    mu = get_mu(p, RK8())
    err = abs(mu - ref_rk8)
    @printf("p=%4d | mu=%.10f | err=%.2e\n", p, mu, err)
end
