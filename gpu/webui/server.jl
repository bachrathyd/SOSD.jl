# The browser user interface of webgpu/ (the same page) with the computation on the Colab GPU:
# this server serves the page and answers its evaluation requests with the batched SOSD solver of
# SOSD.jl (CUDA). In a Colab cell:
#     !nohup julia -t auto --project=gpu/webui gpu/webui/server.jl 8800 > webui.log 2>&1 &
#     from google.colab import output; output.serve_kernel_port_as_iframe(8800, height=1100)
# The page detects the server (GET api/info) and uses it instead of WebGPU.
#
# POST api/eval  (JSON) {code, values, xi, yi, xy (base64 Float32, interleaved) | grid {nx, ny, box},
#                        S, p, r, final, forced}
#   code: Julia source of the model (webgpu/expr.js modelJulia), compiled once per text;
#   final = false: Float32, short Krylov–Schur (the coarse levels while dragging);
#   final = true:  Float64, converged Krylov–Schur (tol 1e-10) — the accurate chart.
# -> application/octet-stream: Float32 ρ[n], Float32 amp[n] (peak-to-peak of x₁ on the periodic
#    orbit; 0 unless forced), Float32 flag[n]

using HTTP, JSON3, Base64, StaticArrays, SOSD, KernelAbstractions
const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
const BACKEND = HAS_CUDA ? CUDA.CUDABackend() : KernelAbstractions.CPU()
const ROOT = normpath(joinpath(@__DIR__, "..", "..", "webgpu"))
const MODELS = Dict{String, Any}()
const LOCK = ReentrantLock()

function model_of(code::String)
    get!(MODELS, code) do
        m = Module(:SOSDModel)
        Base.include_string(m, code)
        g(s) = Base.invokelatest(getproperty, m, s)
        D = g(:D)
        prob = BatchedLDDE{D, 1}(g(:A), g(:B), g(:tau), g(:period))
        (prob = prob, forcing = g(:forcing), D = D, NP = g(:NP), forced_ok = g(:HAS_FORCING))
    end
end

function points(req, NP)
    vals = Float64.(req.values); xi = Int(req.xi) + 1; yi = Int(req.yi) + 1
    θ = (x, y) -> (v = copy(vals); v[xi] = x; v[yi] = y; SVector{NP, Float64}(v))
    if haskey(req, :grid)
        g = req.grid; nx = Int(g.nx); ny = Int(g.ny); b = g.box
        dx = (b.x1 - b.x0) / max(1, nx - 1); dy = (b.y1 - b.y0) / max(1, ny - 1)
        return [θ(b.x0 + i * dx, b.y0 + j * dy) for j in 0:ny-1 for i in 0:nx-1]
    end
    xy = reinterpret(Float32, base64decode(String(req.xy)))
    return [θ(Float64(xy[2k - 1]), Float64(xy[2k])) for k in 1:length(xy) ÷ 2]
end

function evaluate(req)
    mdl = model_of(String(req.code))
    θs = points(req, mdl.NP)
    n = length(θs)
    S = Int(req.S); p = Int(req.p); r = Int(req.r)
    final = Bool(get(req, :final, false)); forced = Bool(get(req, :forced, false)) && mdl.forced_ok
    tab = GL(S)
    T = final ? Float64 : Float32
    kw = final ? (krylovdim=20, keep=10, tol=1e-10, maxiter=20, retry=true) :
                 (krylovdim=10, keep=5, tol=1e-5, maxiter=4, retry=false)
    t0 = time()
    res = Base.invokelatest(spectral_radii, mdl.prob, θs, tab, p, r; backend=BACKEND, T=T,
                            cpu_mode=:kernels, kw...)
    amp = zeros(Float32, n)
    stable = findall(<(1), res.rho)                  # the orbit is shown (and exists) only there
    if forced && !isempty(stable)
        orb = Base.invokelatest(periodic_orbits, mdl.prob, mdl.forcing, θs[stable], tab, p, r; backend=BACKEND, T=T,
                                krylovdim=final ? 30 : 12, tol=final ? 1e-10 : 1e-5, maxrestart=final ? 20 : 3)
        amp[stable] .= Float32.(orb.amp[:, 1])
    end
    ms = 1000 * (time() - t0)
    out = Vector{Float32}(undef, 3n)
    out[1:n] .= Float32.(ifelse.(isfinite.(res.rho), res.rho, 3f38))
    out[n+1:2n] .= amp
    out[2n+1:3n] .= Float32.(res.flag)
    return out, ms
end

const MIME = Dict(".html" => "text/html; charset=utf-8", ".js" => "text/javascript; charset=utf-8",
                  ".mjs" => "text/javascript; charset=utf-8", ".wgsl" => "text/plain; charset=utf-8",
                  ".json" => "application/json", ".md" => "text/plain; charset=utf-8", ".png" => "image/png")

# the request body: a byte vector in HTTP.jl 1.x, a BytesBody (field data) in 2.x
bodybytes(req) = (b = req.body; b isa AbstractVector{UInt8} ? b : b.data)

function handle(req::HTTP.Request)
    path = HTTP.unescapeuri(HTTP.URI(req.target).path)
    if endswith(path, "/api/info")
        name = HAS_CUDA ? CUDA.name(CUDA.device()) : "CPU (no CUDA)"
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                             JSON3.write((device = "Colab: " * name, threads = Threads.nthreads(), cuda = HAS_CUDA)))
    elseif endswith(path, "/api/eval") && req.method == "POST"
        try
            out, ms = lock(LOCK) do
                evaluate(JSON3.read(bodybytes(req)))
            end
            return HTTP.Response(200, ["Content-Type" => "application/octet-stream", "X-Compute-Ms" => string(round(ms, digits=1)),
                                       "Access-Control-Expose-Headers" => "X-Compute-Ms"], collect(reinterpret(UInt8, out)))
        catch err
            @error "eval failed" exception = (err, catch_backtrace())
            return HTTP.Response(500, ["Content-Type" => "text/plain"], first(sprint(showerror, err), 2000))
        end
    end
    # static files of the page (webgpu/)
    rel = lstrip(path, '/')
    full = normpath(joinpath(ROOT, isempty(rel) ? "index.html" : rel))
    (startswith(full, ROOT) && isfile(full)) || return HTTP.Response(404, "not found")
    ext = lowercase(splitext(full)[2])
    return HTTP.Response(200, ["Content-Type" => get(MIME, ext, "application/octet-stream"), "Cache-Control" => "no-cache"],
                         read(full))
end

port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8800
println("SOSD web UI: device ", HAS_CUDA ? CUDA.name(CUDA.device()) : "CPU", ", serving $ROOT on port $port")
# warm-up: compile the kernels of the first example (Mathieu) before the page asks
let code = read(joinpath(@__DIR__, "warmup_model.jl"), String)
    try
        req = JSON3.read(JSON3.write((code = code, values = [3.0, 2.0, -0.15, 0.1, 1.0], xi = 0, yi = 1,
                                      grid = (nx = 8, ny = 4, box = (x0 = -1.0, x1 = 10.0, y0 = 0.0, y1 = 10.0)),
                                      S = 3, p = 16, r = 18, final = false, forced = true)))
        evaluate(req); println("warm-up (Float32) done")
        evaluate(JSON3.read(JSON3.write(merge(copy(req), Dict(:final => true))))); println("warm-up (Float64) done")
    catch err
        @warn "warm-up failed" exception = err
    end
end
println("ready")
flush(stdout)
HTTP.serve(handle, "0.0.0.0", port)
