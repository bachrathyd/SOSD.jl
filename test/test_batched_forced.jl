# Batched solver: end value from the collocation polynomial (no update rows in the sweep) and
# the forced response (periodic orbit by GMRES). KernelAbstractions CPU backend.
using SOSD, StaticArrays, LinearAlgebra, Test
using KernelAbstractions: CPU
import KernelAbstractions as KA
include(joinpath(@__DIR__, "batched_models.jl"))

# Mathieu with the forcing f₀ cos t on the velocity equation, θ = (δ, ε, b0, a1, f0)
mathieu5_A(t, θ) = @SMatrix [0.0 1.0; -θ[1]-θ[2]*cos(t) -θ[4]]
mathieu5_B(t, θ) = (@SMatrix([0.0 0.0; θ[3] 0.0]),)
const MATHIEU5 = BatchedLDDE{2, 1}(mathieu5_A, mathieu5_B, mathieu_tau, mathieu_T)
mathieu5_f(t, θ) = SVector(0.0, θ[5] * cos(t))

@testset "Batched: collocation end value, forced response" begin
    # --- the sweep with the end value from the stage values equals the one with the update rows
    for tab in (GL(3), GL(2), SOSD.from_rkjl(SOSD.RungeKutta.TableauRadauIIA(3)))
        @test SOSD._collocation_endweights(tab)[1]
        θs = [SVector(3.0, 0.2), SVector(1.0, 2.0)]
        p, r = 30, 30
        opc = build_batched_operators(MATHIEU, θs, tab, p, r; backend=CPU())
        opm = build_batched_operators(MATHIEU, θs, tab, p, r; backend=CPU(), endrows=:matrix)
        @test opc.colloc && !opm.colloc
        N = SOSD.state_size(opc)
        X = randn(2, N, 1)
        Yc = zeros(2, N, 1); Ym = zeros(2, N, 1)
        cfg = SOSD.SweepConfig(1, 1)
        SOSD.batched_mul!(Yc, 1, X, 1, opc, cfg, CPU())
        SOSD.batched_mul!(Ym, 1, X, 1, opm, cfg, CPU())
        @test maximum(abs.(Yc .- Ym)) < 1e-11 * maximum(abs.(Ym))
    end

    # --- forced response: ε = 0, b0 = 0 is ẍ + a1 ẋ + δ x = f0 cos t, periodic orbit
    #     x = f0 cos(t − φ) / |δ − 1 + i a1|  ->  peak-to-peak 2 f0 / |δ − 1 + i a1|
    tab = GL(3)
    δs = [0.5, 0.9, 1.0, 2.0, 5.0]
    θs = [SVector(δ, 0.0, 0.0, 0.1, 1.0) for δ in δs]
    res = periodic_orbits(MATHIEU5, mathieu5_f, θs, tab, 40, 41; backend=CPU())
    exact = [2 / abs(complex(δ - 1, 0.1)) for δ in δs]
    rel = abs.(res.amp[:, 1] .- exact) ./ exact
    println("forced Mathieu (ε = 0): max rel. error of the peak-to-peak amplitude = ", maximum(rel),
            "   (node sampling; residual ≤ ", maximum(res.residual), ")")
    @test all(res.converged)
    @test maximum(rel) < 2e-3            # the peak is sampled at the step nodes only

    # --- with the delay and the parametric term: the orbit is the fixed point of the one-period
    #     map; check it by a long forced simulation from a zero history (stable point, ρ < 1)
    θ = SVector(3.0, 1.0, -0.15, 0.2, 1.0)
    p = 40; r = 41
    ρ = spectral_radii(MATHIEU5, [θ], tab, p, r; backend=CPU(), cpu_mode=:kernels).rho[1]
    @test ρ < 0.9
    op = with_forcing(build_batched_operators(MATHIEU5, [θ], tab, p, r; backend=CPU()), MATHIEU5, mathieu5_f, [θ], tab)
    N = SOSD.state_size(op)
    X = zeros(1, N, 1); Y = zeros(1, N, 1)
    cfg = SOSD.SweepConfig(1, 1)
    for k in 1:ceil(Int, log(1e-13) / log(ρ))            # transient decays like ρᵏ
        SOSD.batched_mul!(Y, 1, X, 1, op, cfg, CPU(); forced=true); X .= Y
    end
    SOSD.batched_mul!(Y, 1, X, 1, op, cfg, CPU(); forced=true)
    BS = 4 * 2; hist = Array(op.Hist)
    vals = [hist[1, (n + r) * BS + q * 2 + 1] for n in 1:p for q in 0:3]
    amp_sim = maximum(vals) - minimum(vals)
    res2 = periodic_orbits(MATHIEU5, mathieu5_f, [θ], tab, p, r; backend=CPU())
    println("forced delayed Mathieu: amplitude GMRES ", res2.amp[1, 1], "  simulation ", amp_sim)
    @test abs(res2.amp[1, 1] - amp_sim) < 1e-8 * amp_sim
end
