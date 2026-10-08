# Full stability chart of one time-periodic DDE with the batched solver: 2-DOF milling
# (textbook tool, down-milling a/D = 0.05), spindle speed n × axial depth w.
#   julia -t auto --project=gpu examples/batched_chart_demo.jl [cpu|gpu] [F64|F32|F16] [nx] [ny] [p]
# Writes chart_grid.csv (n, w, ρ) and chart_boundary.csv (n, w_lim) to the current directory,
# and prints the timing. The ρ = 1 boundary is the multisection result (15 sub-points × 4 rounds).
using SOSD, StaticArrays, Printf
using KernelAbstractions: CPU
include(joinpath(@__DIR__, "..", "test", "batched_models.jl"))

const DEV = get(ARGS, 1, "gpu")
const PREC = Dict("F64" => Float64, "F32" => Float32, "F16" => Float16)[get(ARGS, 2, "F64")]
const NX = parse(Int, get(ARGS, 3, "200")); const NY = parse(Int, get(ARGS, 4, "100"))
const P = parse(Int, get(ARGS, 5, "60"))
backend = if DEV == "gpu"
    @eval using CUDA
    CUDABackend()
else
    CPU()
end
prob = PREC === Float64 ? milling_model(1; yratio=1.05) : milling_model_nd(1; yratio=1.05)   # nd for low precision
ns = collect(range(5000.0, 25000.0, length=NX)); ws = collect(range(0.0, 5e-3, length=NY + 1))[2:end]
θs = [SVector(n, w) for w in ws for n in ns]
kw = PREC === Float64 ? (;) : (retry=false, maxiter=8)

spectral_radii(prob, θs[1:min(end, 64)], GL(3), 10, 10; backend=backend, T=PREC, kw...)   # compile
t = @elapsed res = spectral_radii(prob, θs, GL(3), P, P; backend=backend, T=PREC, kw...)
@printf("grid %d×%d (%d points), p = r = %d, %s on %s: %.2f s = %.1f µs/ρ, %d converged\n",
        NX, NY, length(θs), P, PREC, DEV, t, 1e6t / length(θs), count(res.converged))
open("chart_grid.csv", "w") do io
    println(io, "n_rpm,w_m,rho")
    foreach(i -> println(io, θs[i][1], ",", θs[i][2], ",", res.rho[i]), eachindex(θs))
end

rho_of(xs, ys) = spectral_radii(prob, SVector.(xs, ys), GL(3), P, P; backend=backend, T=PREC, kw...).rho
t = @elapsed wlim = boundary_multisection(rho_of, ns, 0.0, 5e-3; nsub=15, rounds=4)
@printf("boundary: %d speeds × 60 ρ (multisection 15×4): %.2f s\n", NX, t)
open("chart_boundary.csv", "w") do io
    println(io, "n_rpm,w_lim_m")
    foreach(i -> println(io, ns[i], ",", wlim[i]), eachindex(ns))
end
