# Batched solver vs the CPU reference (`floquet_analysis`: sparse map + KrylovKit) on
# textbook models. Runs on the KernelAbstractions CPU backend; gpu/test_gpu.jl repeats
# the same comparison on CUDA.
using SOSD, StaticArrays, LinearAlgebra, Test
using KernelAbstractions: CPU
include(joinpath(@__DIR__, "batched_models.jl"))

function reference_rho(bp, θ, tab, p, r)
    grid = TimeGrid(collect(range(0.0, bp.period(θ), length=p + 1)))
    return floquet_analysis(SOSD.LDDEProblem(bp, θ), grid, tab, r; nev=1, tol=1e-13).spectral_radius
end

@testset "Batched spectral radii (CPU backend)" begin
    tab = GL(3)
    cases = [
        ("delayed Mathieu", MATHIEU, [SVector(3.0, 0.2), SVector(1.0, 2.0), SVector(6.0, 1.0)], 40, 40),
        ("turning SSV (τ(t), p≠r)", TURNING_SSV, [SVector(0.3, 0.2), SVector(0.5, 0.1), SVector(1.0, 0.05)], 200, ssv_r(200)),
        ("2-DOF milling", milling_model(1; yratio=1.05),
         [SVector(10000.0, 2e-4), SVector(15000.0, 5e-4), SVector(8000.0, 1e-3)], 60, 60),
    ]
    for (name, bp, θs, p, r) in cases
        ρref = [reference_rho(bp, θ, tab, p, r) for θ in θs]
        res = spectral_radii(bp, θs, tab, p, r; backend=CPU(), cpu_mode=:kernels)
        rel = maximum(abs.(res.rho .- ρref) ./ ρref)
        println(rpad(name, 26), " max rel. diff = ", rel, "   sweeps = ", res.matvecs)
        @test all(res.converged)
        @test rel < 1e-8
        # host-assembled operators give the same multipliers
        resh = spectral_radii(bp, θs, tab, p, r; backend=CPU(), cpu_mode=:kernels, build=:host)
        @test maximum(abs.(resh.rho .- ρref) ./ ρref) < 1e-8
        # default CPU fallback = reference solver threaded over points
        @test maximum(abs.(spectral_radii(bp, θs, tab, p, r).rho .- ρref) ./ ρref) < 1e-8
    end

    # other collocation families and a 4-DOF (D = 8) model
    for tabx in (SOSD.from_rkjl(SOSD.RungeKutta.TableauRadauIIA(3)), GL(2))
        θs = [SVector(3.0, 0.2), SVector(6.0, 1.0)]
        ρref = [reference_rho(MATHIEU, θ, tabx, 40, 40) for θ in θs]
        @test maximum(abs.(spectral_radii(MATHIEU, θs, tabx, 40, 40; cpu_mode=:kernels).rho .- ρref) ./ ρref) < 1e-8
    end
    bp8 = milling_model(2; yratio=1.05)
    θs = [SVector(12000.0, 3e-4), SVector(20000.0, 1e-4)]
    ρref = [reference_rho(bp8, θ, tab, 40, 40) for θ in θs]
    @test maximum(abs.(spectral_radii(bp8, θs, tab, 40, 40; cpu_mode=:kernels).rho .- ρref) ./ ρref) < 1e-8

    # history window too short is flagged, not silently wrong
    @test (@test_logs (:warn,) match_mode=:any spectral_radii(TURNING_SSV, [SVector(0.3, 0.2)], tab, 200, 10; cpu_mode=:kernels)).flag[1] == 1
    @test (@test_logs (:warn,) match_mode=:any spectral_radii(TURNING_SSV, [SVector(0.3, 0.2)], tab, 200, 10)).flag[1] == 1

    # multisection boundary of the delayed Mathieu equation in ε at fixed δ
    rho_of(xs, ys) = spectral_radii(MATHIEU, [SVector(x, y) for (x, y) in zip(xs, ys)], tab, 30, 30).rho
    εlim = boundary_multisection(rho_of, [3.0], 0.0, 10.0; nsub=7, rounds=3)[1]
    @test isfinite(εlim)
    @test reference_rho(MATHIEU, SVector(3.0, εlim), tab, 30, 30) >= 1
    @test reference_rho(MATHIEU, SVector(3.0, εlim - 10 / 8^3), tab, 30, 30) < 1
end
