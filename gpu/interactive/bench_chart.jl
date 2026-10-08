# Chart benchmark with the interactive server's code path: time per chart and per ρ, and
# misclassified points (ρ ≷ 1) versus the Float64 chart, for precision × grid × steps.
#   julia -t auto --project=gpu gpu/interactive/bench_chart.jl <outdir> [examples...]
include(joinpath(@__DIR__, "server.jl"))
using Printf

const OUTDIR = get(ARGS, 1, ".")
const EXS = length(ARGS) > 1 ? ARGS[2:end] : ["milling", "mathieu"]
mkpath(OUTDIR)
const CSV = joinpath(OUTDIR, "chart_bench.csv")
open(CSV, "w") do io
    println(io, "device,example,nx,ny,p,precision,mode,krylov,t_ms,us_per_rho,t_build,t_sweep,t_orth,t_host,converged,misclassified,median_rel,max_rel")
end
readrho(f, n) = (a = Vector{Float32}(undef, n); read!(f, a); a)

function run_chart(ex, nx, ny, p, prec, mode, kd)
    e = EXBYKEY[ex]
    out = tempname()
    q = Dict("ex" => ex, "nx" => string(nx), "ny" => string(ny), "x0" => string(e.xr[1]), "x1" => string(e.xr[2]),
             "y0" => string(e.yr[1]), "y1" => string(e.yr[2]), "p" => string(p), "kd" => string(kd),
             "prec" => prec, "mode" => mode, "out" => out)
    chart(q)                                  # compile / warm this shape
    best = nothing; tbest = Inf
    for _ in 1:3                              # min of 3 (the GPU is otherwise idle)
        js = chart(q)
        t = parse(Float64, match(r"\"t_total\":([0-9.eE+-]+)", js).captures[1])
        t < tbest && (tbest = t; best = js)
    end
    return best, readrho(out, nx * ny)
end
num(js, k) = (m = match(Regex("\"$k\":([0-9.eE+-]+|null)"), js); m === nothing || m.captures[1] == "null" ? NaN : parse(Float64, m.captures[1]))

for ex in EXS, (nx, ny) in ((128, 64), (256, 128), (512, 256)), p in (40, 100)
    js64, ρ64 = run_chart(ex, nx, ny, p, "F64", "accurate", 30)
    for (prec, mode, kd) in (("F64", "accurate", 30), ("F64", "fast", 16), ("F32", "fast", 16), ("F32", "fast", 10), ("F16", "fast", 16))
        js, ρ = (prec, mode, kd) == ("F64", "accurate", 30) ? (js64, ρ64) : run_chart(ex, nx, ny, p, prec, mode, kd)
        rel = abs.(ρ .- ρ64) ./ ρ64
        mis = count((ρ .>= 1) .!= (ρ64 .>= 1))
        line = join((device_name(), ex, nx, ny, p, prec, mode, kd, num(js, "t_total"), num(js, "us_per_rho"),
                     num(js, "t_build"), num(js, "t_sweep"), num(js, "t_orth"), num(js, "t_host"), Int(num(js, "converged")),
                     mis, sort(rel)[end ÷ 2], maximum(rel)), ",")
        open(io -> println(io, line), CSV, "a")
        @printf("%-8s %4d×%-4d p=%-4d %-4s %-8s kd=%-3d %8.0f ms %7.1f µs/ρ  misclassified %5d  median %.1e  max %.1e\n",
                ex, nx, ny, p, prec, mode, kd, num(js, "t_total"), num(js, "us_per_rho"), mis, sort(rel)[end ÷ 2], maximum(rel))
    end
end
println("results -> ", CSV)
