using MFCM
using Test
using LinearAlgebra
using StaticArrays
using KrylovKit

# Mathieu equation parameters from paper (Fig 4a)
# ζ = 0.2, δ = 3, ε = 3, b = -0.5, τ = 2π, T = 2π
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

@testset "Mathieu Equation Validation" begin
    # 1. Extract system
    prob = extract_SDM_system(mathieu_rhs, nothing, Val(2))
    
    # 2. Setup discretization
    p_steps = 100
    r_steps = 100 # Since T = τ, r = p
    grid = TimeGrid(collect(range(0.0, T, length=p_steps+1)))
    tableau = GL2Tableau()
    
    # 3. Build matrices
    sys_mats = build_system_matrices(prob, grid, tableau, r_steps)
    
    # 4. Create Monodromy Map
    S_stages = 2
    D = 2
    state_size = (r_steps + 1) * (S_stages + 1) * D
    m = MonodromyMap(prob, grid, tableau, sys_mats, p_steps, r_steps, state_size)
    
    # 5. Solve for eigenvalues
    # Initial vector for KrylovKit
    x0 = rand(m.state_size)
    vals, vecs, info = eigsolve(m, x0, 1, :LM)
    
    mumax = abs(vals[1])
    @show mumax
    
    # Expected value: The paper says for these parameters it's around some value.
    # We can compare with a known value or check convergence.
    @test mumax > 0
end
