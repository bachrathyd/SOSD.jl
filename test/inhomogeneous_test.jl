using SOSD
using Test
using StaticArrays
using LinearAlgebra

@testset "Inhomogeneous Periodic Solution" begin
    # x'(t) = -x(t) + sin(t)
    # The periodic solution is x(t) = 0.5 sin(t) - 0.5 cos(t)
    # x(0) = -0.5
    
    function rhs(u, h, p, t)
        return @SVector [ -u[1] + sin(t) ]
    end
    
    D = 1
    T = 2π
    p = 100
    prob = extract_SDM_system(rhs, nothing, Val(D))
    grid = TimeGrid(collect(range(0.0, T, length=p+1)))
    tableau = GL2Tableau()
    
    r = 0 # No delay
    sys_mats = build_system_matrices(prob, grid, tableau, r)
    S = size(tableau.a, 1)
    m = MonodromyMap(prob, grid, tableau, sys_mats, p, r, (r+1)*(S+1)*D)
    
    y_fixed = solve_periodic_solution(m)
    @test y_fixed[1] ≈ -0.5 atol=1e-8
end
