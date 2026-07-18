using MFCM
using Test
using LinearAlgebra
using StaticArrays
using KrylovKit

@testset "GL3 Convergence Analysis" begin
    # ODE: y' = -y. T = 1.0. Phi = exp(-1.0)
    prob = extract_SDM_system((u,h,p,t)->[-u[1]], nothing, Val(1))
    tableau = GL3Tableau()
    
    ps = [4, 6, 8, 12, 16]
    errors = []
    mu_exact = exp(-1.0)
    
    for p in ps
        h = 1.0 / p
        grid = TimeGrid(collect(range(0.0, 1.0, length=p+1)))
        sys = build_system_matrices(prob, grid, tableau, 0)
        # S=3. state size = (0+1)*(3+1)*1 = 4
        m = MonodromyMap(prob, grid, tableau, sys, p, 0, 4) 
        vals, _ = eigsolve(m, rand(m.state_size), 1, :LM)
        push!(errors, abs(abs(vals[1]) - mu_exact))
    end
    
    log_ps = log.(ps)
    log_errs = log.(errors)
    coeffs = [log_ps ones(length(ps))] \ log_errs
    k = -coeffs[1]
    @show k
    
    @test k > 5.5 # GL3 should be 6th order for ODE
end
