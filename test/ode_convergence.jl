using MFCM
using Test
using LinearAlgebra
using StaticArrays
using KrylovKit

@testset "ODE Convergence Analysis" begin
    # y' = -y. T = 1.0. Phi = exp(-1.0)
    prob = extract_SDM_system((u,h,p,t)->[-u[1]], nothing, Val(1))
    tableau = GL2Tableau()
    
    ps = [4, 8, 16, 32, 64]
    errors = []
    mu_exact = exp(-1.0)
    
    for p in ps
        h = 1.0 / p
        grid = TimeGrid(collect(range(0.0, 1.0, length=p+1)))
        sys = build_system_matrices(prob, grid, tableau, 0)
        # GL2 s=2. State block size (s+1)*D = 3.
        m = MonodromyMap(prob, grid, tableau, sys, p, 0, 3) 
        vals, _ = eigsolve(m, rand(m.state_size), 1, :LM)
        push!(errors, abs(abs(vals[1]) - mu_exact))
    end
    
    log_ps = log.(ps)
    log_errs = log.(errors)
    coeffs = [log_ps ones(length(ps))] \ log_errs
    k = -coeffs[1]
    @show k
    
    @test k > 3.8 # GL2 should be 4th order for ODE
end
