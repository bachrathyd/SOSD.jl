using SOSD
using Test
using LinearAlgebra
using StaticArrays
using KrylovKit

# Scalar DDE: y'(t) = -y(t) + 0.5 * y(t - 1.0)
function simple_dde_rhs(u, h, p, t)
    hist = h(p, t - 1.0)
    return -u[1] + 0.5 * hist[1]
end

@testset "Convergence Analysis" begin
    prob = extract_SDM_system((u,h,p,t)->[-u[1] + 0.5*h(p,t-1.0)[1]], nothing, Val(1))
    
    # Reference value with high resolution
    p_ref = 512
    h_ref = 1.0 / p_ref
    grid_ref = TimeGrid(collect(range(0.0, 1.0, length=p_ref+1)))
    tableau = GL2Tableau()
    sys_ref = build_system_matrices(prob, grid_ref, tableau, p_ref)
    m_ref = MonodromyMap(prob, grid_ref, tableau, sys_ref, p_ref, p_ref, (p_ref+1)*3)
    vals_ref, _ = eigsolve(m_ref, rand(m_ref.state_size), 1, :LM)
    mu_ref = abs(vals_ref[1])
    @show mu_ref
    
    ps = [16, 32, 64, 128]
    errors = []
    for p in ps
        h = 1.0 / p
        grid = TimeGrid(collect(range(0.0, 1.0, length=p+1)))
        sys = build_system_matrices(prob, grid, tableau, p)
        m = MonodromyMap(prob, grid, tableau, sys, p, p, (p+1)*3)
        vals, _ = eigsolve(m, rand(m.state_size), 1, :LM)
        push!(errors, abs(abs(vals[1]) - mu_ref))
    end
    
    # Calculate convergence rate
    # error ~ p^(-k) => log(error) ~ -k * log(p)
    log_ps = log.(ps)
    log_errs = log.(errors)
    coeffs = [log_ps ones(length(ps))] \ log_errs
    k = -coeffs[1]
    @show k
    
    @test k > 3.0 # Expect around 4.0 for GL2
end
