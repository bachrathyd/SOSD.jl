# GPU worker of the streamed stability-chart app (gpu/stream/app.py): a persistent Julia process
# that keeps the compiled kernels and the chart of the current view, and answers one JSON request
# per line on stdin with one JSON line on stdout.
#
#   julia -t auto --project=gpu/webui gpu/stream/worker.jl [--cpu]
#
# The chart is a W×H pixel grid (pixel (0, 0) top left). It is refined on nested lattices: the
# level of stride s computes the nodes (i·s, j·s) and fills the s×s block centred on each node, so
# a finer level overwrites a coarser one and only the new nodes of a level are computed. The ρ (and
# the forced-response amplitude) of every pixel stays here; the app gets coloured images only.
#
# {"cmd":"level", view..., "s":8, "j0":0, "j1":20, "img":path}
#     computes the lattice rows j0 ≤ j < j1 of stride s (all of them if j1 is missing) and writes
#     the stride-s image (RGB bytes, row-major from the top) to path.
#     view: code (webgpu/expr.js modelJulia), values, xi, yi, x0, x1, y0, y1, W, H, S, p, m, prec
#           ("F32" | "F64" | "mixed": Float64 for the last level), forced; colour: cr, bnd
#     -> {"iw","ih","s","n","t_kernel","t_colour","r","done","unstable","flagged","alo","ahi","forced"}
# {"cmd":"image", "s":1, "img":path, colour...} -> the current chart at stride s, no computation
# {"cmd":"warm", view...}                       -> compiles the kernels of a model (4 points)
# {"cmd":"quit"}

using SOSD, StaticArrays, KernelAbstractions, JSON3, LinearAlgebra
const USE_GPU = !("--cpu" in ARGS)
const HAS_CUDA = USE_GPU && try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
const BACKEND = HAS_CUDA ? CUDA.CUDABackend() : KernelAbstractions.CPU()
device_name() = HAS_CUDA ? CUDA.name(CUDA.device()) : "CPU ($(Threads.nthreads()) threads)"
BLAS.set_num_threads(1)

# ---------------------------------------------------------------------------------------------
# models: the Julia text of the page's model, compiled once per text
# ---------------------------------------------------------------------------------------------
const MODELS = Dict{String, Any}()
function model_of(code::String)
    get!(MODELS, code) do
        mod = Module(:SOSDModel)
        Base.include_string(mod, code)
        g(s) = Base.invokelatest(getproperty, mod, s)
        D = g(:D)
        (prob = BatchedLDDE{D, 1}(g(:A), g(:B), g(:tau), g(:period)), forcing = g(:forcing),
         NP = g(:NP), forced_ok = g(:HAS_FORCING), tau = g(:tau), period = g(:period))
    end
end

# ---------------------------------------------------------------------------------------------
# the chart of the current view
# ---------------------------------------------------------------------------------------------
mutable struct Chart
    key::Any
    W::Int
    H::Int
    rho::Vector{Float32}          # W·H, index px + W·py + 1
    amp::Vector{Float32}
    done::Int                     # finest completed stride (typemax: nothing yet)
    r::Int
    alo::Float64                  # amplitude colour scale (fixed per view after its first level)
    ahi::Float64
    forced::Bool
end
const CH = Chart(nothing, 0, 0, Float32[], Float32[], typemax(Int), 1, 1.0, 10.0, false)

viewkey(q) = (String(q.code), Float64.(q.values), Int(q.xi), Int(q.yi), Float64(q.x0), Float64(q.x1),
              Float64(q.y0), Float64(q.y1), Int(q.W), Int(q.H), Int(q.S), Int(q.p), Int(q.m),
              String(q.prec), Bool(q.forced))

# pixel centre -> parameter point
px2x(q, px) = q.x0 + (px + 0.5) * (q.x1 - q.x0) / q.W
py2y(q, py) = q.y1 - (py + 0.5) * (q.y1 - q.y0) / q.H
function theta(mdl, q)
    vals = Float64.(q.values); xi = Int(q.xi) + 1; yi = Int(q.yi) + 1; NP = mdl.NP
    (x, y) -> (v = copy(vals); v[xi] = x; v[yi] = y; SVector{NP, Float64}(v))
end

# delay steps r (as the page's delaySteps): the largest τ/h + 1 over a 21×21 sample of the view
function delay_steps(mdl, q)
    θ = theta(mdl, q); p = Int(q.p); rmax = 1
    for x in range(Float64(q.x0), Float64(q.x1), length = 21), y in range(Float64(q.y0), Float64(q.y1), length = 21)
        t = θ(x, y)
        T = Base.invokelatest(mdl.period, t); h = T / p
        for k in 0:23
            τ = Base.invokelatest(mdl.tau, T * k / 24, t)[1]
            isfinite(τ) && isfinite(h) && h > 0 && (rmax = max(rmax, ceil(Int, τ / h + 1e-9) + 1))
        end
    end
    return min(rmax, 4096)
end

function reset!(q, mdl)
    W = Int(q.W); H = Int(q.H)
    if length(CH.rho) != W * H
        CH.rho = zeros(Float32, W * H); CH.amp = zeros(Float32, W * H)
    end
    CH.key = viewkey(q); CH.W = W; CH.H = H; CH.done = typemax(Int)
    CH.r = delay_steps(mdl, q)
    CH.forced = Bool(q.forced) && mdl.forced_ok
    CH.alo = NaN; CH.ahi = NaN
end

"compute the lattice rows j0 ≤ j < j1 of stride s (only the nodes new to the previous level)"
function compute_rows!(q, mdl, s, j0, j1)
    W, H = CH.W, CH.H
    last = String(q.prec) == "mixed" && s == 1
    T = (String(q.prec) == "F64" || last) ? Float64 : Float32
    reuse = CH.done == 2s && !last              # the even nodes are those of the previous level
    nxs = cld(W, s)
    pxs = Int[]; pys = Int[]
    for j in j0:j1-1, i in 0:nxs-1
        reuse && iseven(i) && iseven(j) && continue
        push!(pxs, i * s); push!(pys, j * s)
    end
    n = length(pxs)
    n == 0 && return (n = 0, t = 0.0, flagged = 0)
    θ = theta(mdl, q)
    θs = [θ(px2x(q, pxs[k]), py2y(q, pys[k])) for k in 1:n]
    t0 = time()
    fr = Base.invokelatest(fast_spectral_radii, mdl.prob, θs, GL(Int(q.S)), Int(q.p), CH.r; m = Int(q.m),
                           backend = BACKEND, T = T, forcing = CH.forced ? mdl.forcing : nothing)
    t = time() - t0
    h = s ÷ 2
    Threads.@threads for k in 1:n
        ρ = Float32(isfinite(fr.rho[k]) ? fr.rho[k] : 3f38); a = Float32(fr.amp[k])
        for py in max(0, pys[k] - h):min(H - 1, pys[k] - h + s - 1), px in max(0, pxs[k] - h):min(W - 1, pxs[k] - h + s - 1)
            CH.rho[px + W * py + 1] = ρ; CH.amp[px + W * py + 1] = a
        end
    end
    if CH.forced && isnan(CH.alo)
        v = sort!([Float64(fr.amp[k]) for k in 1:n if fr.rho[k] < 1 && 0 < fr.amp[k] < 1e30])
        if length(v) >= 16 || j1 * s >= H
            lo, hi = 1.0, 10.0
            if !isempty(v)
                hi = v[min(length(v), floor(Int, 0.98 * length(v)) + 1)]
                lo = max(v[floor(Int, 0.02 * length(v)) + 1], hi * 1e-4)
                hi > 1.5lo || (lo = hi / 3)
            end
            CH.alo, CH.ahi = lo, hi
        end
    end
    return (n = n, t = t, flagged = count(!=(0), fr.flag))
end

# ---------------------------------------------------------------------------------------------
# colours (as the page): log10 ρ on RdBu over [-cr, cr]; with the periodic orbit the stable
# points by the peak-to-peak amplitude, log scale over [alo, ahi], white -> dark green
# ---------------------------------------------------------------------------------------------
const STOPS = [(5, 48, 97), (33, 102, 172), (67, 147, 195), (146, 197, 222), (247, 247, 247),
               (244, 165, 130), (214, 96, 77), (178, 24, 43), (103, 0, 31)]
const GREENS = [(247, 252, 245), (229, 245, 224), (199, 233, 192), (161, 217, 155), (116, 196, 118),
                (65, 171, 93), (35, 139, 69), (0, 109, 44), (0, 68, 27)]
function ramp(stops, t)
    t = clamp(isfinite(t) ? t : 1.0, 0.0, 1.0) * (length(stops) - 1)
    i = min(length(stops) - 2, floor(Int, t)); w = t - i
    a = stops[i + 1]; b = stops[i + 2]
    return ntuple(k -> round(UInt8, a[k] + w * (b[k] - a[k])), 3)
end
const LUTN = 1024
const LUT_RDBU = [ramp(STOPS, k / (LUTN - 1)) for k in 0:LUTN-1]
const LUT_GREEN = [ramp(GREENS, k / (LUTN - 1)) for k in 0:LUTN-1]
lut(L, t) = L[clamp(round(Int, (isfinite(t) ? t : 1.0) * (LUTN - 1)), 0, LUTN - 1) + 1]

function image(s, path, q)
    W, H = CH.W, CH.H
    iw = cld(W, s); ih = cld(H, s)
    cr = Float64(get(q, :cr, 0.5)); bnd = Bool(get(q, :bnd, true))
    la, lh = log(CH.alo), log(CH.ahi)
    rgb = Vector{UInt8}(undef, 3 * iw * ih)
    unst = Threads.Atomic{Int}(0)
    Threads.@threads for j in 0:ih-1
        nu = 0
        for i in 0:iw-1
            k = i * s + W * (j * s) + 1
            ρ = CH.rho[k]
            u = ρ >= 1; nu += u
            c = if CH.forced && !u
                lut(LUT_GREEN, (log(max(Float64(CH.amp[k]), 1e-30)) - la) / (lh - la))
            else
                lut(LUT_RDBU, (log10(max(Float64(ρ), 1e-30)) / cr + 1) / 2)
            end
            if bnd && ((i + 1 < iw && (CH.rho[k + s] >= 1) != u) || (j + 1 < ih && (CH.rho[k + W * s] >= 1) != u))
                c = (0x40, 0x40, 0x40)
            end
            o = 3 * (i + iw * j)
            rgb[o + 1] = c[1]; rgb[o + 2] = c[2]; rgb[o + 3] = c[3]
        end
        Threads.atomic_add!(unst, nu)
    end
    write(path, rgb)
    return (iw = iw, ih = ih, unstable = unst[] / (iw * ih))
end

# ---------------------------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------------------------
function level(q)
    mdl = model_of(String(q.code))
    viewkey(q) == CH.key || reset!(q, mdl)
    s = Int(q.s)
    nys = cld(CH.H, s)
    j0 = Int(get(q, :j0, 0)); j1 = min(nys, Int(something(get(q, :j1, nothing), nys)))
    cached = CH.done <= s
    c = cached ? (n = 0, t = 0.0, flagged = 0) : compute_rows!(q, mdl, s, j0, j1)
    if !cached && j1 >= nys
        CH.done = s
    end
    t0 = time()
    im = image(cached ? CH.done : s, String(q.img), q)
    return (; im..., s = cached ? CH.done : s, n = c.n, t_kernel = 1e3 * c.t, t_colour = 1e3 * (time() - t0),
            r = CH.r, done = CH.done == typemax(Int) ? 0 : CH.done, flagged = c.flagged, rows = j1, nys,
            alo = isnan(CH.alo) ? nothing : CH.alo, ahi = isnan(CH.ahi) ? nothing : CH.ahi,
            forced = CH.forced, cached)
end

function warm(q)
    mdl = model_of(String(q.code))
    θ = theta(mdl, q)
    θs = [θ(Float64(q.x0) + (q.x1 - q.x0) * a, Float64(q.y0) + (q.y1 - q.y0) * b) for a in (0.3, 0.7), b in (0.3, 0.7)][:]
    T = String(get(q, :prec, "F32")) == "F64" ? Float64 : Float32
    t0 = time()
    Base.invokelatest(fast_spectral_radii, mdl.prob, θs, GL(Int(q.S)), Int(q.p), 4; m = Int(q.m), backend = BACKEND, T = T,
                      forcing = Bool(q.forced) && mdl.forced_ok ? mdl.forcing : nothing)
    return (t = time() - t0,)
end

function serve()
    println(stderr, "device: ", device_name(), ", Julia threads: ", Threads.nthreads())
    # compile the kernels of the first example (Mathieu, Float32, with and without the orbit)
    code = join(Iterators.drop(eachline(joinpath(@__DIR__, "..", "webui", "warmup_model.jl"), keep = true), 1))
    for forced in (false, true)
        try
            t = @elapsed warm((code = code, values = [3.0, 2.0, -0.15, 0.1, 1.0], xi = 0, yi = 1, x0 = -1.0, x1 = 10.0,
                               y0 = 0.0, y1 = 10.0, S = 3, p = 16, m = 6, prec = "F32", forced = forced))
            println(stderr, "warm-up (forced = $forced): ", round(t, digits = 1), " s")
        catch err
            println(stderr, "warm-up failed: ", sprint(showerror, err))
        end
    end
    println(JSON3.write((ready = true, device = device_name(), threads = Threads.nthreads(), cuda = HAS_CUDA)))
    flush(stdout)
    for line in eachline(stdin)
        isempty(strip(line)) && continue
        out = try
            q = JSON3.read(line)
            cmd = String(q.cmd)
            cmd == "quit" && break
            r = cmd == "level" ? level(q) :
                cmd == "image" ? (CH.key === nothing ? error("no chart yet") : image(Int(q.s), String(q.img), q)) :
                cmd == "warm" ? warm(q) : error("unknown command $cmd")
            JSON3.write(r)
        catch err
            JSON3.write((error = first(replace(sprint(showerror, err), '\n' => ' '), 600),))
        end
        println(out)
        flush(stdout)
    end
end

serve()
