# Benchmark: CPU (existing SOSD path, threaded over points) vs batched solver on CPU threads
# vs batched solver on an NVIDIA GPU. Writes CSV files to the output directory.
#   julia --project=gpu gpu/bench_gpu.jl <outdir> [quick]
# Timing: wall clock of complete calls (build + eigensolve + transfers), after a warm-up
# call that compiles every kernel. Points of one call are independent parameter points.
using CUDA, SOSD, StaticArrays, LinearAlgebra, Printf
using KernelAbstractions: CPU
include(joinpath(@__DIR__, "..", "test", "batched_models.jl"))
BLAS.set_num_threads(1)

const OUT = length(ARGS) >= 1 ? ARGS[1] : "."
const QUICK = length(ARGS) >= 2 && ARGS[2] == "quick"
mkpath(OUT)
const HAS_GPU = CUDA.functional()
const GPUNAME = HAS_GPU ? CUDA.name(CUDA.device()) : "none"
const NTH = Threads.nthreads()
const CPUNAME = Sys.cpu_info()[1].model
println("GPU: $GPUNAME | CPU: $CPUNAME, $(Sys.CPU_THREADS) logical cores, Julia threads: $NTH")

const ROWS = String[]
function record(case, method, prec, D, p, npts, t, extra="")
    tp = t / npts
    @printf("%-22s %-26s %-8s D=%-3d p=r=%-5d n=%-6d  total %9.3f s   per ρ %10.3f ms  %s\n",
            case, method, prec, D, p, npts, t, 1e3 * tp, extra)
    push!(ROWS, join((case, method, prec, D, p, npts, t, tp, NTH, GPUNAME, extra), ","))
    open(joinpath(OUT, "bench_results.csv"), "w") do io
        println(io, "case,method,precision,D,p,npoints,total_s,per_rho_s,julia_threads,gpu,note")
        foreach(r -> println(io, r), ROWS)
    end
end

reference_rho(bp, θ, tab, p, r) = floquet_analysis(SOSD.LDDEProblem(bp, θ),
    TimeGrid(collect(range(0.0, bp.period(θ), length=p + 1))), tab, r; nev=1, tol=1e-11).spectral_radius

"Existing CPU path: one point per thread (build + sparse LU + KrylovKit)."
function cpu_reference(bp, θs, tab, p, r)
    ρ = zeros(length(θs))
    Threads.@threads for i in eachindex(θs)
        ρ[i] = reference_rho(bp, θs[i], tab, p, r)
    end
    return ρ
end

milling_points(n) = [SVector(5000.0 + 20000.0 * mod(0.618034 * i, 1.0), 5e-4 * (0.2 + mod(0.414214 * i, 1.0))) for i in 1:n]

tab = GL(3)

# ---------------------------------------------------------------------------
# 1. Time per ρ, 2-DOF milling (D = 4), GL3, p = r
# ---------------------------------------------------------------------------
bp = milling_model(1; yratio=1.05)
for p in (QUICK ? (100,) : (100, 300, 1000))
    r = p
    ncpu = max(NTH, 2) * (p <= 300 ? 4 : 1)
    θc = milling_points(ncpu)
    cpu_reference(bp, θc[1:1], tab, p, r)
    t = @elapsed ρc = cpu_reference(bp, θc, tab, p, r)
    record("milling2dof", "CPU SOSD (threads/points)", "Float64", 4, p, ncpu, t)


    HAS_GPU || continue
    spectral_radii(bp, θc[1:2], tab, p, r; backend=CUDABackend())
    for nb in (QUICK ? (256,) : (64, 512, 4096))
        θg = milling_points(nb)
        t = @elapsed res = spectral_radii(bp, θg, tab, p, r; backend=CUDABackend())
        nconv = count(res.converged)
        ref = ρc[1:min(ncpu, nb)]
        record("milling2dof", "GPU batched", "Float64", 4, p, nb, t,
               @sprintf("conv=%d/%d maxrel=%.1e sweeps=%d", nconv, nb, maximum(abs.(res.rho[1:length(ref)] .- ref) ./ ref), res.matvecs))
    end
    spectral_radii(bp, θc[1:2], tab, p, r; backend=CUDABackend(), T=Float32, tol=1e-6)
    nb = QUICK ? 256 : 4096
    θg = milling_points(nb)
    res64 = spectral_radii(bp, θg, tab, p, r; backend=CUDABackend())
    t = @elapsed res32 = spectral_radii(bp, θg, tab, p, r; backend=CUDABackend(), T=Float32, tol=1e-6)
    record("milling2dof", "GPU batched", "Float32", 4, p, nb, t,
           @sprintf("maxrel_vs_F64=%.1e median=%.1e", maximum(abs.(res32.rho .- res64.rho) ./ res64.rho),
                    sort(abs.(res32.rho .- res64.rho) ./ res64.rho)[nb ÷ 2]))
end

# ---------------------------------------------------------------------------
# 2. Large system: 12-DOF milling (D = 24), GL3
# ---------------------------------------------------------------------------
bp24 = milling_model(6; yratio=1.05)
for p in (QUICK ? (50,) : (100, 300))
    r = p
    ncpu = max(NTH, 2)
    θc = milling_points(ncpu)
    cpu_reference(bp24, θc[1:1], tab, p, r)
    t = @elapsed ρc = cpu_reference(bp24, θc, tab, p, r)
    record("milling12dof", "CPU SOSD (threads/points)", "Float64", 24, p, ncpu, t)
    HAS_GPU || continue
    for build in (:host, :device)
        try
            spectral_radii(bp24, θc[1:2], tab, p, r; backend=CUDABackend(), build=build)
        catch err
            println("D=24 build=$build failed: ", sprint(showerror, err)[1:min(end, 200)]); continue
        end
        for nb in (QUICK ? (16,) : (1, 16, 128))
            θg = milling_points(nb)
            t = @elapsed res = spectral_radii(bp24, θg, tab, p, r; backend=CUDABackend(), build=build)
            ref = ρc[1:min(ncpu, nb)]
            record("milling12dof", "GPU batched build=$build", "Float64", 24, p, nb, t,
                   @sprintf("maxrel=%.1e sweeps=%d", maximum(abs.(res.rho[1:length(ref)] .- ref) ./ ref), res.matvecs))
        end
    end
end

# ---------------------------------------------------------------------------
# 3. Stability chart, 2-DOF milling, n ∈ [5000, 25000] rpm, w ∈ [0, 5] mm
# ---------------------------------------------------------------------------
if HAS_GPU
    p = r = QUICK ? 60 : 200
    nn = QUICK ? 50 : 200
    ns = collect(range(5000.0, 25000.0, length=nn))
    rho_of(xs, ys) = spectral_radii(bp, [SVector(x, y) for (x, y) in zip(xs, ys)], tab, p, r;
                                    backend=CUDABackend()).rho
    boundary_multisection(rho_of, ns[1:2], 0.0, 5e-3; nsub=3, rounds=1)
    t = @elapsed wlim = boundary_multisection(rho_of, ns, 0.0, 5e-3; nsub=15, rounds=4)
    record("chart_milling2dof", "GPU multisection 15x4", "Float64", 4, p, nn * 15 * 4, t, "boundary points=$nn")
    open(joinpath(OUT, "chart_boundary.csv"), "w") do io
        println(io, "n_rpm,w_lim_m")
        foreach(i -> println(io, ns[i], ",", wlim[i]), eachindex(ns))
    end
    nw = QUICK ? 30 : 100
    ws = collect(range(0.0, 5e-3, length=nw + 1))[2:end]
    θgrid = [SVector(n, w) for w in ws for n in ns]
    t = @elapsed resg = spectral_radii(bp, θgrid, tab, p, r; backend=CUDABackend())
    record("chart_milling2dof", "GPU brute-force grid", "Float64", 4, p, length(θgrid), t, "grid=$(nn)x$(nw)")
    open(joinpath(OUT, "chart_grid.csv"), "w") do io
        println(io, "n_rpm,w_m,rho")
        foreach(i -> println(io, θgrid[i][1], ",", θgrid[i][2], ",", resg.rho[i]), eachindex(θgrid))
    end
    # CPU estimate for the same chart with per-speed bisection (16 steps): measured CPU time per ρ
    θc = milling_points(max(NTH, 2) * 4)
    cpu_reference(bp, θc[1:1], tab, p, r)
    t = @elapsed cpu_reference(bp, θc, tab, p, r)
    record("chart_milling2dof", "CPU SOSD (threads/points)", "Float64", 4, p, length(θc), t,
           @sprintf("bisection chart estimate %d speeds x 16 = %.1f s on %d threads", nn, t / length(θc) * nn * 16, NTH))
end
println("results -> ", joinpath(OUT, "bench_results.csv"))
