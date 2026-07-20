# Tests for the embedded-pair error estimation (error_estimation = true).

using Test
using SOSD
using LinearAlgebra
using StaticArrays
using LinearMaps

# Delayed Mathieu, benchmark parameters (mesh-commensurate, mu ≈ 0.35141726095)
function _mathieu_prob()
    A_f = t -> @SMatrix [0.0 1.0; -3.0-0.2*cos(t) -0.1]
    B_f = t -> @SMatrix [0.0 0.0; -0.15 0.0]
    c_f = t -> @SVector [0.0, sin(2.0*t)]
    return SOSD.LDDEProblem{2, Float64}(ProportionalMX(A_f), [DelayMX(t -> 2π, B_f)], Additive(c_f))
end

# Seasonal biological model (non-commensurate delay, mu ≈ 0.62602193)
function _bio_prob()
    A_f = t -> SMatrix{1,1}(0.0)
    B_f = t -> SMatrix{1,1}(-(1.0 + cos(t)))
    c_f = t -> SVector{1}(0.0)
    return SOSD.LDDEProblem{1, Float64}(ProportionalMX(A_f), [DelayMX(t -> 2.0, B_f)], Additive(c_f))
end

const MU_REF_MATHIEU = 0.35141726095
const MU_REF_BIO = 0.6260219314919767

@testset "Embedded companion construction" begin
    # generic drop-node weights: order conditions up to s−2 hold, s−1 violated
    for tab in (GL(3), GL(5), RK4())
        b_hat = embedded_weights(tab)
        @test b_hat !== nothing
        s = length(tab.b)
        @test abs(sum(b_hat) - 1.0) < 1e-12
        @test any(abs.(collect(b_hat) .- collect(tab.b)) .> 1e-8)
    end
    # BS3 carries the classical Bogacki–Shampine order-2 weights
    bs = BS3()
    @test bs.b_embedded !== nothing
    @test collect(bs.b_embedded) ≈ [7/24, 1/4, 1/3, 1/8]
    @test abs(sum(bs.b_embedded) - 1.0) < 1e-14
    # order-2 quadrature condition holds, order-3 condition is violated
    @test abs(sum(bs.b_embedded .* bs.c) - 1/2) < 1e-14
    @test abs(sum(bs.b_embedded .* bs.c .^ 2) - 1/3) > 1e-3

    # cross-family collocation companions live in the same state space
    for s in (1, 2, 3)
        comp = SOSD.order_reduced_collocation_companion(GL(s))
        @test comp !== nothing
        @test length(comp.b) == s
        @test maximum(abs.(collect(comp.c) .- collect(GL(s).c))) > 1e-6
    end

    # reduced CE reproduces low-degree data exactly (weights may split between
    # slots that hold identical values, e.g. c₁ = 0 stages) and differs from
    # the full CE at interior evaluation points
    bs = BS3()
    ce_red = SOSD.reduced_continuous_extension(bs)
    @test ce_red !== nothing
    g = x -> 0.3 + 0.7x                       # degree ≤ reduced degree
    slots = vcat(g(0.0), [g(c) for c in bs.c], g(1.0))
    for θ in (0.0, 0.35, 1.0)
        @test abs(sum(ce_red(θ) .* slots) - g(θ)) < 1e-12
    end
    @test maximum(abs.(ce_red(0.35) .- bs.ce(0.35))) > 1e-3
end

@testset "SparseMonodromyMap transpose action" begin
    # r = p (Mathieu, τ = T), r < p (bio, short delay), and an explicit tableau
    for (prob, D, p, r, tab) in ((_mathieu_prob(), 2, 12, 12, GL(2)),
                                 (_bio_prob(),     1, 16,  6, GL(2)),
                                 (_mathieu_prob(), 2, 12, 12, BS3()))
        S = length(tab.b); BSIZE = (S + 1) * D
        state = (r + 1) * BSIZE
        grid = TimeGrid(collect(range(0.0, 2π, length=p+1)))
        sysm = build_system_matrices(prob, grid, tab, r)
        m = MonodromyMap(prob, grid, tab, sysm, p, r, state)
        Phi = SparseMonodromyMap(m)
        a = [sin(3.1i) for i in 1:state]; b = [cos(1.7i) for i in 1:state]
        @test abs(dot(b, Phi * a) - dot(transpose(Phi) * b, a)) < 1e-12
        @test norm(Matrix(Phi)' - Matrix(transpose(Phi))) < 1e-11
    end
end

@testset "Interface compatibility" begin
    prob = _mathieu_prob()
    p = 40; r = 40
    grid = TimeGrid(collect(range(0.0, 2π, length=p+1)))
    tab = GL(2)
    # plain call: single output, unchanged behavior
    sol = floquet_analysis(prob, grid, tab, r)
    @test sol isa FloquetSolution
    @test abs(sol.spectral_radius - MU_REF_MATHIEU) < 1e-3
    rho = spectral_radius(prob, grid, tab, r)
    @test rho == sol.spectral_radius
    # with estimation: the bar arrives as a SEPARATE output
    sol2, est = floquet_analysis(prob, grid, tab, r; error_estimation=true)
    @test sol2 isa FloquetSolution && est isa FloquetErrorEstimate
    @test sol2.mu ≈ sol.mu atol=1e-12
    rho2, bar = spectral_radius(prob, grid, tab, r; error_estimation=true)
    @test rho2 == sol2.spectral_radius && bar == est.mu_error
end

@testset "Coverage: bar contains the true error" begin
    # commensurate (Mathieu) and non-commensurate (bio), several tableaux
    prob_m = _mathieu_prob(); prob_b = _bio_prob()
    for (prob, T, tau, mu_ref, tabs) in (
            (prob_m, 2π, 2π, MU_REF_MATHIEU, (GL(2), GL(3), BS3())),
            (prob_b, 2π, 2.0, MU_REF_BIO, (GL(2), GL(3), BS3())))
        for tab in tabs, p in (24, 48)
            r = ceil(Int, tau / (T / p))
            grid = TimeGrid(collect(range(0.0, T, length=p+1)))
            sol, est = floquet_analysis(prob, grid, tab, r; error_estimation=true)
            err_true = abs(sol.spectral_radius - mu_ref)
            @test est.mu_error > 0
            @test est.mu_error >= err_true   # the bar must contain the truth
            @test est.mu_error < 0.5         # ... and stay meaningful
            @test est.eigenvalue_condition >= 1.0
        end
    end
end

@testset "First-order perturbation consistency (same-stage pair)" begin
    # for a same-(a,c) companion, δμ from the perturbation formula must agree
    # with the true eigenvalue difference μ − μ̂ to leading order
    prob = _bio_prob()
    tab = BS3()
    for p in (40, 80)
        r = ceil(Int, 2.0 / (2π / p))
        grid = TimeGrid(collect(range(0.0, 2π, length=p+1)))
        sol, est = floquet_analysis(prob, grid, tab, r; error_estimation=true, embedded_eigs=true)
        dmu_exact = abs(sol.mu - est.mu_embedded_quadrature)
        @test est.quadrature_error ≈ dmu_exact rtol=0.35
    end
end

@testset "Interpolation channel semantics" begin
    prob_m = _mathieu_prob(); prob_b = _bio_prob()
    p = 32; grid = TimeGrid(collect(range(0.0, 2π, length=p+1)))
    # non-collocation tableau: the I channel is a genuine finite contribution
    # (interior-stage delayed lookups use the CE even for commensurate delays)
    _, est_m = floquet_analysis(prob_m, grid, BS3(), p; error_estimation=true)
    @test isfinite(est_m.interpolation_error) && est_m.interpolation_error >= 0
    r_b = ceil(Int, 2.0 / (2π / p))
    _, est_b = floquet_analysis(prob_b, grid, BS3(), r_b; error_estimation=true)
    @test est_b.interpolation_error > 1e-8
    # collocation: I channel folded into the cross-family Q channel (NaN)
    _, est_gl = floquet_analysis(prob_b, grid, GL(2), r_b; error_estimation=true)
    @test isnan(est_gl.interpolation_error)
    @test est_gl.quadrature_error > 0
end

@testset "Fixed-point error bar" begin
    prob = _mathieu_prob()
    tab = GL(2); T = 2π
    # reference fixed point from a fine grid
    p_ref = 320
    grid_ref = TimeGrid(collect(range(0.0, T, length=p_ref+1)))
    sol_ref = floquet_analysis(prob, grid_ref, GL(5), p_ref; periodic_solution=true)
    y_ref = sol_ref.fixpoint[1:2]              # y(0), node component of block 0
    for p in (24, 48)
        grid = TimeGrid(collect(range(0.0, T, length=p+1)))
        sol, est = floquet_analysis(prob, grid, tab, p; error_estimation=true, periodic_solution=true)
        @test sol.fixpoint !== nothing && est.fixpoint_delta !== nothing
        err_true = norm(sol.fixpoint[1:2] - y_ref)
        S = length(tab.b); BSIZE = (S + 1) * 2
        node_idx = [b * BSIZE + d for b in 0:p for d in 1:2]
        pred = norm(est.fixpoint_delta[node_idx])
        @test est.fixpoint_error > 0
        @test 2 * pred >= err_true    # predicted bar covers the true fixpoint error
    end
end
