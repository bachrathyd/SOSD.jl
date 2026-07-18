using SOSD
using Test
using BenchmarkTools
using LinearAlgebra
using StaticArrays

function mathieu_rhs(u, h, p, t)
    x1, x2 = u
    hist = h(p, t - 2π)
    du1 = x2
    du2 = -0.4 * x2 - (3.0 + 3.0 * cos(t)) * x1 - 0.5 * hist[1]
    return @SVector [du1, du2]
end

@testset "Complexity Analysis" begin
    prob = extract_SDM_system(mathieu_rhs, nothing, Val(2))
    tableau = GL2Tableau()
    
    ps = [10, 20, 40, 80, 160]
    times = []
    
    for p in ps
        h = 2π / p
        grid = TimeGrid(collect(range(0.0, 2π, length=p+1)))
        sys = build_system_matrices(prob, grid, tableau, p)
        m = MonodromyMap(prob, grid, tableau, sys, p, p, (p+1)*6)
        x0 = rand(m.state_size)
        
        # Benchmark mul!
        y = zeros(m.state_size)
        t = @belapsed mul!($y, $m, $x0)
        push!(times, t)
    end
    
    # log(t) ~ log(p)
    log_ps = log.(ps)
    log_ts = log.(times)
    coeffs = [log_ps ones(length(ps))] \ log_ts
    k = coeffs[1]
    @show k
    
    @test k ≈ 1.0 atol=0.2
end
