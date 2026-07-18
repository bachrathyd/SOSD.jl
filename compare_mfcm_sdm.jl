using MFCM
using SemiDiscretizationMethod
using StaticArrays
using KrylovKit
using Printf
using LinearAlgebra

# Mathieu Problem Parameters
δ, ε, b0, a1 = 3.0, 0.2, -0.15, 0.1
T = 2π
τ = 2π

# MFCM Problem
function create_mathieu_mfcm(δ, ε, b0, a1; T=2π)
    AMx = MFCM.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = MFCM.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = MFCM.Additive(t -> @SVector [0.0, sin(4π / T * t)])
    MFCM.LDDEProblem{2, Float64}(AMx, [BMx1], cVec)
end

# SDM Problem
function create_mathieu_sdm(δ, ε, b0, a1; T=2π)
    AMx = SemiDiscretizationMethod.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = SemiDiscretizationMethod.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = SemiDiscretizationMethod.Additive(t -> @SVector [0.0, sin(4π / T * t)])
    SemiDiscretizationMethod.LDDEProblem(AMx, [BMx1], cVec)
end

prob_mfcm = create_mathieu_mfcm(δ, ε, b0, a1, T=T)
prob_sdm = create_mathieu_sdm(δ, ε, b0, a1, T=T)

function get_mu_mfcm(p, tab)
    grid = TimeGrid(collect(range(0.0, T, length=p+1)))
    sys = build_system_matrices(prob_mfcm, grid, tab, p)
    m = MonodromyMap(prob_mfcm, grid, tab, sys, p, p, (p+1)*(size(tab.a,1)+1)*2)
    vals, _ = eigsolve(m, rand(m.state_size), 1, :LM; tol=1e-12)
    return abs(vals[1])
end

function get_mu_sdm(p)
    h = T / p
    method = SemiDiscretization(2, h)
    return spectralRadiusOfMapping(DiscreteMapping_LR(prob_sdm, method, τ, n_steps=p))
end

println("Comparing MFCM and SDM Results:")
ps = [40, 80, 160, 320, 640]

for p in ps
    mu_sdm = get_mu_sdm(p)
    mu_gl2 = get_mu_mfcm(p, GL(2))
    mu_gl4 = get_mu_mfcm(p, GL(4))
    @printf("p=%3d | SDM: %.10f | MFCM GL2: %.10f | MFCM GL4: %.10f\n", p, mu_sdm, mu_gl2, mu_gl4)
end
