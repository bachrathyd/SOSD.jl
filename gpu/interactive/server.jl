# SOSD interactive stability-chart server: a persistent Julia process that keeps the GPU
# kernels compiled and answers one request per line on stdin.
#
#   julia -t auto --project=gpu gpu/interactive/server.jl [--cpu]
#
# Protocol (one line each, whitespace-separated key=value pairs):
#   meta                                   -> META {json: device, examples, precisions}
#   chart ex=milling nx=200 ny=100 x0=.. x1=.. y0=.. y1=.. p=60 kd=16 prec=F32 mode=fast
#         k=0.05,0.011,2,1.05 out=/dev/shm/sosd_rho.f32
#                                          -> OK {json: timings, statistics}
#   quit
# The chart is written as raw Float32 ρ values, index ix + nx·iy (iy = 0 is the bottom row).
# Errors: ERR message. Everything else the process prints goes to stderr.

using SOSD, StaticArrays, LinearAlgebra, Printf
using KernelAbstractions
const USE_GPU = !("--cpu" in ARGS)
if USE_GPU
    using CUDA
    CUDA.functional() || error("CUDA not functional (use --cpu)")
end
include(joinpath(@__DIR__, "..", "..", "test", "batched_models.jl"))
BLAS.set_num_threads(1)

const BACKEND = USE_GPU ? CUDABackend() : KernelAbstractions.CPU()
device_name() = USE_GPU ? CUDA.name(CUDA.device()) : "CPU ($(Threads.nthreads()) threads)"

# --------------------------------------------------------------------------
# Examples: one problem each, a 2-D chart over θ[1:2], the knobs set the other parameters
# --------------------------------------------------------------------------
struct Example
    key::String
    title::String
    xl::String
    yl::String
    xr::NTuple{2, Float64}
    yr::NTuple{2, Float64}
    yscale::Float64                     # chart y value × yscale = model parameter
    knobs::Vector{NamedTuple}
    make::Function                      # knob values -> (problem, θ-builder, r-of-p)
end

const EXAMPLES = [
    Example("milling", "2-DOF milling (down-milling, periodic cutting-force coefficient)",
            "spindle speed n [rpm]", "axial depth of cut w [mm]", (5000.0, 25000.0), (0.0, 5.0), 1e-3,
            [(name = "radial immersion a/D", lo = 0.02, hi = 1.0, value = 0.05, step = 0.01),
             (name = "damping ratio ζ", lo = 0.002, hi = 0.05, value = 0.011, step = 0.001),
             (name = "number of teeth z", lo = 1, hi = 8, value = 2, step = 1),
             (name = "y/x frequency ratio", lo = 0.8, hi = 1.3, value = 1.05, step = 0.01)],
            k -> (milling_model_nd(1; aD = k[1], ζ = k[2], z = round(Int, k[3]), yratio = k[4]),
                  (x, y) -> SVector(x, y), p -> p)),
    Example("mathieu", "delayed damped Mathieu equation  ẍ + a₁ẋ + (δ + ε cos t)x = b₀x(t − 2π)",
            "δ [–]", "ε [–]", (-1.0, 10.0), (0.0, 10.0), 1.0,
            [(name = "delayed gain b₀", lo = -1.0, hi = 1.0, value = -0.15, step = 0.01),
             (name = "damping a₁", lo = 0.0, hi = 1.0, value = 0.1, step = 0.01)],
            k -> (MATHIEU4, (x, y) -> SVector(x, y, k[1], k[2]), p -> p)),
    Example("turning_ssv", "turning with spindle-speed variation (time-periodic delay, T = 10·τ₀)",
            "spindle speed Ω [–]", "cutting stiffness k_w [–]", (0.2, 2.0), (0.0, 0.6), 1.0,
            [(name = "SSV amplitude A", lo = 0.0, hi = 0.3, value = 0.1, step = 0.01),
             (name = "damping ζ", lo = 0.01, hi = 0.3, value = 0.1, step = 0.01)],
            k -> (TURNING_SSV4, (x, y) -> SVector(x, y, k[1], k[2]), p -> ssv4_r(p, k[1]))),
]
const EXBYKEY = Dict(e.key => e for e in EXAMPLES)
const PRECISIONS = [("F64", "Float64 (reference)"), ("F32", "Float32"), ("F16", "Float16 (storage; build in Float32)")]
const PRECTYPE = Dict("F64" => Float64, "F32" => Float32, "F16" => Float16)

jstr(s) = "\"" * replace(string(s), "\\" => "\\\\", "\"" => "\\\"") * "\""
jnum(x) = isfinite(x) ? string(Float64(x)) : "null"

function meta_json()
    exs = map(EXAMPLES) do e
        knobs = join(["{\"name\":$(jstr(k.name)),\"lo\":$(k.lo),\"hi\":$(k.hi),\"value\":$(k.value),\"step\":$(k.step)}"
                      for k in e.knobs], ",")
        "{\"key\":$(jstr(e.key)),\"title\":$(jstr(e.title)),\"xl\":$(jstr(e.xl)),\"yl\":$(jstr(e.yl))," *
        "\"xr\":[$(e.xr[1]),$(e.xr[2])],\"yr\":[$(e.yr[1]),$(e.yr[2])],\"knobs\":[$knobs]}"
    end
    precs = join(["[$(jstr(k)),$(jstr(l))]" for (k, l) in PRECISIONS], ",")
    return "{\"device\":$(jstr(device_name())),\"threads\":$(Threads.nthreads()),\"examples\":[$(join(exs, ","))],\"precisions\":[$precs]}"
end

parse_kv(words) = Dict(split(w, "=", limit = 2)[1] => split(w, "=", limit = 2)[2] for w in words if occursin('=', w))

function chart(q)
    ex = EXBYKEY[q["ex"]]
    nx = parse(Int, q["nx"]); ny = parse(Int, q["ny"])
    x0, x1, y0, y1 = (parse(Float64, q[k]) for k in ("x0", "x1", "y0", "y1"))
    p = parse(Int, q["p"]); kd = parse(Int, get(q, "kd", "16"))
    T = PRECTYPE[get(q, "prec", "F64")]
    fast = get(q, "mode", "fast") == "fast"
    knobs = [parse(Float64, v) for v in split(get(q, "k", ""), ",", keepempty = false)]
    length(knobs) == length(ex.knobs) || (knobs = [k.value for k in ex.knobs])
    t0 = time()
    prob, θof, rof = ex.make(knobs)
    xs = range(x0, x1, length = nx); ys = range(y0, y1, length = ny)
    θs = [θof(x, y * ex.yscale) for y in ys for x in xs]       # ix fastest, iy = 0 bottom
    kw = fast ? (krylovdim = kd, keep = max(2, kd ÷ 2), maxiter = 8, retry = false) :
                (krylovdim = max(kd, 30), keep = max(15, kd ÷ 2))
    res = spectral_radii(prob, θs, GL(3), p, rof(p); backend = BACKEND, T = T, kw...)
    t1 = time()
    write(q["out"], Float32.(res.rho))
    tm = res.timing
    nconv = count(res.converged); nunst = count(>(1), res.rho)
    return "{\"n\":$(length(θs)),\"t_total\":$(jnum(1e3 * (t1 - t0))),\"t_build\":$(jnum(1e3 * get(tm, :build, NaN)))," *
           "\"t_sweep\":$(jnum(1e3 * get(tm, :sweep, NaN))),\"t_orth\":$(jnum(1e3 * get(tm, :orth, NaN)))," *
           "\"t_host\":$(jnum(1e3 * get(tm, :host, NaN))),\"us_per_rho\":$(jnum(1e6 * (t1 - t0) / length(θs)))," *
           "\"sweeps\":$(res.matvecs),\"converged\":$nconv,\"unstable\":$nunst,\"flagged\":$(count(!=(0), res.flag))," *
           "\"rmin\":$(jnum(minimum(res.rho))),\"rmax\":$(jnum(maximum(res.rho)))}"
end

function serve()
    println(stderr, "device: ", device_name(), "  threads: ", Threads.nthreads())
    # compile every kernel variant of the default view (all precisions) before answering
    for prec in (USE_GPU ? ("F64", "F32", "F16") : ("F64",)), ex in ("milling",)
        e = EXBYKEY[ex]
        q = Dict("ex" => ex, "nx" => (USE_GPU ? "160" : "8"), "ny" => (USE_GPU ? "80" : "4"), "x0" => string(e.xr[1]), "x1" => string(e.xr[2]),
                 "y0" => string(e.yr[1]), "y1" => string(e.yr[2]), "p" => "8", "kd" => "16", "prec" => prec,
                 "mode" => "fast", "out" => tempname())
        t = @elapsed chart(q)
        println(stderr, "warm-up $ex $prec: ", round(t, digits = 1), " s")
    end
    println("READY")
    flush(stdout)
    for line in eachline(stdin)
        words = split(strip(line))
        isempty(words) && continue
        cmd = words[1]
        try
            if cmd == "meta"
                println("META ", meta_json())
            elseif cmd == "chart"
                println("OK ", chart(parse_kv(words[2:end])))
            elseif cmd == "quit"
                break
            else
                println("ERR unknown command $cmd")
            end
        catch err
            println("ERR ", replace(sprint(showerror, err), '\n' => ' ')[1:min(end, 500)])
        end
        flush(stdout)
    end
end

serve()
