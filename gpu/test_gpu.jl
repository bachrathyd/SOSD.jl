# CPU vs GPU agreement of the batched solver (run on a machine with an NVIDIA GPU):
#   julia --project=gpu gpu/test_gpu.jl
using CUDA, SOSD, StaticArrays, LinearAlgebra, Test, Printf
using KernelAbstractions: CPU
include(joinpath(@__DIR__, "..", "test", "batched_models.jl"))
CUDA.functional() || error("CUDA not functional on this machine")
println("GPU: ", CUDA.name(CUDA.device()), "   Julia threads: ", Threads.nthreads())
const GPU = CUDABackend()

reference_rho(bp, θ, tab, p, r) = floquet_analysis(SOSD.LDDEProblem(bp, θ),
    TimeGrid(collect(range(0.0, bp.period(θ), length=p + 1))), tab, r; nev=1, tol=1e-13).spectral_radius
reldiff(a, b) = maximum(abs.(a .- b) ./ abs.(b))

@testset "GPU vs CPU reference" begin
    tab = GL(3)
    cases = [
        ("delayed Mathieu (D=2)", MATHIEU, [SVector(3.0, 0.2), SVector(1.0, 2.0), SVector(6.0, 1.0)], 40, 40),
        ("turning SSV (D=2, τ(t))", TURNING_SSV, [SVector(0.3, 0.2), SVector(0.5, 0.1), SVector(1.0, 0.05)], 200, ssv_r(200)),
        ("2-DOF milling (D=4)", milling_model(1; yratio=1.05),
         [SVector(10000.0, 2e-4), SVector(15000.0, 5e-4), SVector(8000.0, 1e-3)], 100, 100),
        ("4-DOF milling (D=8)", milling_model(2; yratio=1.05), [SVector(12000.0, 3e-4), SVector(20000.0, 1e-4)], 60, 60),
    ]
    for (name, bp, θs, p, r) in cases
        ρref = [reference_rho(bp, θ, tab, p, r) for θ in θs]
        for build in (:device, :host)
            res = spectral_radii(bp, θs, tab, p, r; backend=GPU, build=build)
            d = reldiff(res.rho, ρref)
            @printf("%-26s build=%-6s  max rel. diff = %.2e  sweeps = %d\n", name, build, d, res.matvecs)
            @test all(res.converged)
            @test d < 1e-8
        end
        # every thread mapping gives the same result
        for cfg in (SweepConfig(1, 32), SweepConfig(8, 8), SweepConfig(32, 4))
            res = spectral_radii(bp, θs, tab, p, r; backend=GPU, sweep=cfg)
            @test reldiff(res.rho, ρref) < 1e-8
        end
        # Float32: accuracy report only
        res32 = spectral_radii(bp, θs, tab, p, r; backend=GPU, T=Float32, tol=1e-6)
        @printf("%-26s Float32: max rel. diff = %.2e\n", name, reldiff(res32.rho, ρref))
    end

    # large system: 12-DOF milling (D = 24, S·D = 72)
    bp24 = milling_model(6; yratio=1.05); θs = [SVector(12000.0, 2e-4), SVector(18000.0, 1e-4)]
    ρref = [reference_rho(bp24, θ, tab, 30, 30) for θ in θs]
    res = spectral_radii(bp24, θs, tab, 30, 30; backend=GPU, build=:host)
    @printf("%-26s build=host    max rel. diff = %.2e\n", "12-DOF milling (D=24)", reldiff(res.rho, ρref))
    @test reldiff(res.rho, ρref) < 1e-8
    try
        resd = spectral_radii(bp24, θs, tab, 30, 30; backend=GPU, build=:device)
        @printf("%-26s build=device  max rel. diff = %.2e\n", "12-DOF milling (D=24)", reldiff(resd.rho, ρref))
        @test reldiff(resd.rho, ρref) < 1e-8
    catch err
        println("12-DOF milling, build=:device not available on this GPU: ", sprint(showerror, err)[1:min(end, 300)])
    end
end
