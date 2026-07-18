using MFCM
using StaticArrays
using KrylovKit
using Printf
using LinearAlgebra

# Mathieu Problem
function create_mathieu(δ, ε, b0, a1; T=2π)
    AMx = MFCM.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = MFCM.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = MFCM.Additive(t -> @SVector [0.0, sin(4π / T * t)])
    MFCM.LDDEProblem{2, Float64}(AMx, [BMx1], cVec)
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

println("Checking GL2 and RK4 convergence to the same limit:")
ps = [40, 80, 160, 320, 640]
for p in ps
    mu_gl2 = get_mu(p, GL(2))
    mu_rk4 = get_mu(p, RK4())
    @printf("p=%3d | GL2: %.10f | RK4: %.10f | Diff: %.2e\n", p, mu_gl2, mu_rk4, abs(mu_gl2 - mu_rk4))
end
