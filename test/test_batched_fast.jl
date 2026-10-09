# Fast chart path (one Arnoldi pass per point on the device) vs the converged batched solver.
using SOSD, StaticArrays, LinearAlgebra, Test
using KernelAbstractions: CPU
include(joinpath(@__DIR__, "batched_models.jl"))
mathieu5_A(t, θ) = @SMatrix [0.0 1.0; -θ[1]-θ[2]*cos(t) -θ[4]]
mathieu5_B(t, θ) = (@SMatrix([0.0 0.0; θ[3] 0.0]),)
const MATHIEU5F = BatchedLDDE{2, 1}(mathieu5_A, mathieu5_B, mathieu_tau, mathieu_T)
mathieu5_ff(t, θ) = SVector(0.0, θ[5] * cos(t))

@testset "Batched fast path" begin
    tab = GL(3)
    θs = [SVector(δ, ε) for δ in range(-1, 10, length=12) for ε in range(0, 10, length=6)]
    ref = spectral_radii(MATHIEU, θs, tab, 16, 18; backend=CPU(), cpu_mode=:kernels).rho
    for T in (Float64, Float32)
        f = fast_spectral_radii(MATHIEU, θs, tab, 16, 18; m=8, backend=CPU(), T=T)
        rel = abs.(f.rho .- ref) ./ ref
        mis = count((f.rho .>= 1) .!= (ref .>= 1))
        println("fast path $T, m = 8: median rel ", sort(rel)[end÷2], ", max ", maximum(rel), ", misclassified ", mis)
        @test mis == 0
        @test sort(rel)[end÷2] < (T === Float64 ? 1e-6 : 1e-4)
    end
    # forced orbit amplitude: analytic case (ε = 0, b0 = 0)
    δs = [0.5, 1.0, 2.0, 5.0]
    θf = [SVector(δ, 0.0, 0.0, 0.1, 1.0) for δ in δs]
    f = fast_spectral_radii(MATHIEU5F, θf, tab, 40, 41; m=8, backend=CPU(), T=Float64, forcing=mathieu5_ff)
    exact = [2 / abs(complex(δ - 1, 0.1)) for δ in δs]
    println("fast forced amplitude rel err ", maximum(abs.(f.amp .- exact) ./ exact))
    @test maximum(abs.(f.amp .- exact) ./ exact) < 3e-3
end
