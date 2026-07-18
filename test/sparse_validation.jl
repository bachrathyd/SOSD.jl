using SOSD
using Test
using LinearAlgebra
using SparseArrays

@testset "Sparse Matrix Validation" begin
    # Simple DDE: y'(t) = -y(t) + 0.5 * y(t-1)
    prob = extract_SDM_system((u,h,p,t)->[-u[1] + 0.5*h(p,t-1.0)[1]], nothing, Val(1))
    tableau = GL2Tableau()
    
    p = 10
    h = 1.0 / p
    grid = TimeGrid(collect(range(0.0, 1.0, length=p+1)))
    sys = build_system_matrices(prob, grid, tableau, p)
    m = MonodromyMap(prob, grid, tableau, sys, p, p, (p+1)*3)
    
    # 1. Get lazy transition
    # We can get the full matrix by applying to unit vectors
    T_lazy = zeros(m.state_size, m.state_size)
    for i in 1:m.state_size
        e_i = zeros(m.state_size)
        e_i[i] = 1.0
        T_lazy[:, i] = m * e_i
    end
    
    # 2. Get explicit transition
    sm = SparseMonodromyMap(m)
    T_explicit = zeros(m.state_size, m.state_size)
    for i in 1:m.state_size
        e_i = zeros(m.state_size)
        e_i[i] = 1.0
        T_explicit[:, i] = sm * e_i
    end
    
    # Compare
    diff = norm(T_lazy - T_explicit)
    @show diff
    @test diff < 1e-10
end
