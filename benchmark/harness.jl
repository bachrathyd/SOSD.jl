# Shared benchmark harness for the MFCM paper experiments.
#
# Fair-comparison rules (paper Section: methodology):
#   - one machine, BLAS threads pinned to 1, environment reported
#   - log-log axes everywhere, wide p ranges
#   - timing: warmup + repeated runs, mean and std reported
#   - eigenvalue error measured against a two-resolution-verified reference
#   - deterministic start vectors (no RNG in timed paths)

using MFCM
using SemiDiscretizationMethod
const SDM = SemiDiscretizationMethod
using LinearAlgebra
using SparseArrays
using StaticArrays
using KrylovKit
using Printf
using Statistics
using DelimitedFiles
using Dates

BLAS.set_num_threads(1)
ENV["GKSwstype"] = "100"   # headless GR

const RESULTS_DIR = joinpath(@__DIR__, "results")
const FIGURES_DIR = joinpath(@__DIR__, "figures")
mkpath(RESULTS_DIR); mkpath(FIGURES_DIR)

# ---------------------------------------------------------------------------
# Environment report
# ---------------------------------------------------------------------------

function write_env_report()
    open(joinpath(RESULTS_DIR, "environment.txt"), "w") do io
        println(io, "Timestamp: ", Dates.now())
        println(io, "Julia: ", VERSION)
        println(io, "OS: ", Sys.KERNEL, " ", Sys.MACHINE)
        println(io, "CPU: ", Sys.cpu_info()[1].model, " x ", Sys.CPU_THREADS, " threads")
        println(io, "BLAS: ", BLAS.get_config(), "  threads = ", BLAS.get_num_threads())
        println(io, "Julia threads: ", Threads.nthreads())
        for pkg in ("MFCM", "SemiDiscretizationMethod", "KrylovKit", "StaticArrays", "RungeKutta")
            println(io, pkg, " loaded")
        end
    end
end

# ---------------------------------------------------------------------------
# Test systems: (name, MFCM problem, SDM problem, T, taumax, D, r_of_p)
# Each system defines r(p) so that r*h >= taumax on the grid h = T/p.
# ---------------------------------------------------------------------------

struct BenchSystem
    name::String
    prob                 # MFCM LDDEProblem
    sdm_prob             # SemiDiscretizationMethod LDDEProblem (or nothing)
    T::Float64
    taumax::Float64
    D::Int
    r_of_p::Function     # p -> r
end

# --- Delayed Mathieu (benchmark parameters, mu ~ 0.35143) ---
function make_mathieu(; δ=3.0, ε=0.2, b0=-0.15, a1=0.1, T=2π)
    A_f = t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1]
    B_f = t -> @SMatrix [0.0 0.0; b0 0.0]
    c_f = t -> @SVector [0.0, sin(4π / T * t)]
    prob = MFCM.LDDEProblem{2, Float64}(MFCM.ProportionalMX(A_f),
        [MFCM.DelayMX(t -> 2π, B_f)], MFCM.Additive(c_f))
    sdm_prob = SDM.LDDEProblem(SDM.ProportionalMX(A_f), [SDM.DelayMX(t -> 2π, B_f)], SDM.Additive(c_f))
    BenchSystem("mathieu", prob, sdm_prob, T, 2π, 2, p -> p)
end

# --- Seasonal scalar biological model: y' = -(a + b cos t) y(t - tau) ---
function make_bio(; a=1.0, b=1.0, τ=2.0, T=2π)
    A_f = t -> SMatrix{1,1}(0.0)
    B_f = t -> SMatrix{1,1}(-(a + b*cos(t)))
    c_f = t -> SVector{1}(0.0)
    prob = MFCM.LDDEProblem{1, Float64}(MFCM.ProportionalMX(A_f),
        [MFCM.DelayMX(t -> τ, B_f)], MFCM.Additive(c_f))
    sdm_prob = SDM.LDDEProblem(SDM.ProportionalMX(A_f), [SDM.DelayMX(t -> τ, B_f)], SDM.Additive(c_f))
    BenchSystem("bio", prob, sdm_prob, T, τ, 1, p -> ceil(Int, τ / (T / p)))
end

# --- Turning with spindle speed variation (time-periodic delay, T/tau ~ 10) ---
function make_turning_ssv(; kw=0.2, ζ=0.1, Ω=0.3, ASSV=0.1, NT=10)
    T = 2π / Ω * NT
    τ_f = t -> 2π / Ω * (1 + ASSV * sin(t / T * 2π))
    τmax = 2π / Ω * (1 + ASSV)
    A_f = t -> @SMatrix [0.0 1.0; -1-kw -ζ]
    B_f = t -> @SMatrix [0.0 0.0; kw 0.0]
    c_f = t -> @SVector [0.0, cos(t * 2π)]
    prob = MFCM.LDDEProblem{2, Float64}(MFCM.ProportionalMX(A_f),
        [MFCM.DelayMX(τ_f, B_f)], MFCM.Additive(c_f))
    sdm_prob = SDM.LDDEProblem(SDM.ProportionalMX(A_f), [SDM.DelayMX(τ_f, B_f)], SDM.Additive(c_f))
    BenchSystem("turning_ssv", prob, sdm_prob, T, τmax, 2, p -> ceil(Int, τmax / (T / p)))
end

# --- FEM beam with delayed boundary feedback (D = 28, tau/T = 1/2) ---
include(joinpath(@__DIR__, "..", "examples", "beam_delay_feedback.jl"))

function make_beam(; act_and_wait=false)
    prob, D, τ, Tper = beam_problem(act_and_wait=act_and_wait)

    # SDM twin
    E=210e9; A=1e-4; ρ=7800.0; L=100.0; η=0.01; N=15; P=0.2
    M, C, K = beam_matrices(E, A, ρ, L, η, N)
    Nm1 = N - 1
    Z = zeros(Nm1, Nm1); In = Matrix(I, Nm1, Nm1)
    A_sys = [Z In; -M\K -M\C]
    B = zeros(2Nm1, 2Nm1); B[2Nm1, 1] = P * E / (L / N)
    F = zeros(Nm1); F[Nm1] = 1.0
    Ffirst = vcat(zeros(Nm1), F)
    B_f = act_and_wait ? (t -> B .* (mod(t, Tper) < 0.8 * Tper)) : (t -> B)
    sdm_prob = SDM.LDDEProblem(SDM.ProportionalMX(t -> A_sys),
        [SDM.DelayMX(t -> τ, B_f)], SDM.Additive(t -> Ffirst))

    name = act_and_wait ? "beam_aaw" : "beam"
    BenchSystem(name, prob, sdm_prob, Tper, τ, D, p -> p ÷ 2)
end

# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------

"Deterministic start vector for the eigensolver."
det_x0(n) = normalize!([1.0 + 0.1 * sin(7.3i) for i in 1:n])

"""
    mfcm_mu(sys, p, tableau; solver=:sparse, tol=1e-11)

Dominant Floquet multiplier magnitude via MFCM. Returns NaN on failure.
"""
function mfcm_mu(sys::BenchSystem, p::Int, tableau; solver::Symbol=:sparse, tol=1e-11)
    r = sys.r_of_p(p)
    S = size(tableau.a, 1)
    BSIZE = (S + 1) * sys.D
    state_size = (r + 1) * BSIZE
    grid = TimeGrid(collect(range(0.0, sys.T, length=p+1)))
    sysm = build_system_matrices(sys.prob, grid, tableau, r)
    m_lazy = MonodromyMap(sys.prob, grid, tableau, sysm, p, r, state_size)
    m = solver === :sparse ? SparseMonodromyMap(m_lazy) : m_lazy
    vals, _, info = eigsolve(m, det_x0(state_size), 1, :LM; tol=tol)
    return abs(vals[1])
end

"SDM baseline (order k semi-discretization)."
function sdm_mu(sys::BenchSystem, p::Int; order::Int=2)
    method = SemiDiscretization(order, sys.T / p)
    mapping = DiscreteMapping_LR(sys.sdm_prob, method, sys.taumax, n_steps=p)
    return spectralRadiusOfMapping(mapping)
end

"""
    matrix_stats(sys, p, tableau)

Structure metrics of the sparse pair (R, L): sizes, nnz, density, bandwidth of L.
"""
function matrix_stats(sys::BenchSystem, p::Int, tableau)
    r = sys.r_of_p(p)
    S = size(tableau.a, 1)
    BSIZE = (S + 1) * sys.D
    state_size = (r + 1) * BSIZE
    grid = TimeGrid(collect(range(0.0, sys.T, length=p+1)))
    sysm = build_system_matrices(sys.prob, grid, tableau, r)
    m = MonodromyMap(sys.prob, grid, tableau, sysm, p, r, state_size)
    R, L = build_explicit_matrices(m)
    rows = rowvals(L)
    bw = 0
    for j in 1:size(L, 2), k in nzrange(L, j)
        bw = max(bw, abs(rows[k] - j))
    end
    return (n_L = size(L, 1), nnz_L = nnz(L), nnz_R = nnz(R),
            density_L = nnz(L) / (size(L,1) * size(L,2)), bandwidth_L = bw)
end

# ---------------------------------------------------------------------------
# Timing with repetitions (mean +- std)
# ---------------------------------------------------------------------------

"""
    time_stats(f; min_reps=5, max_reps=15, budget=3.0)

Warm up once, then repeat `f()` until `budget` seconds or `max_reps` reps
(at least `min_reps` unless a single rep exceeds the budget).
Returns (value, t_mean, t_std, n_reps).
"""
function time_stats(f; min_reps::Int=5, max_reps::Int=15, budget::Float64=3.0)
    value = f()                     # warmup (also the returned value)
    times = Float64[]
    total = 0.0
    while length(times) < max_reps
        t = @elapsed f()
        push!(times, t)
        total += t
        (total > budget && length(times) >= min_reps) && break
        (total > 4 * budget) && break   # single very slow rep: stop early
    end
    return value, mean(times), (length(times) > 1 ? std(times) : 0.0), length(times)
end

# ---------------------------------------------------------------------------
# Reference values (two-resolution agreement check, cached)
# ---------------------------------------------------------------------------

const REF_CACHE_FILE = joinpath(RESULTS_DIR, "reference_values.csv")

function load_ref_cache()
    cache = Dict{String, Float64}()
    if isfile(REF_CACHE_FILE)
        for row in eachrow(readdlm(REF_CACHE_FILE, ',', Any; comments=true))
            cache[String(row[1])] = Float64(row[2])
        end
    end
    return cache
end

function save_ref_cache(cache)
    open(REF_CACHE_FILE, "w") do io
        println(io, "# system, mu_reference")
        for (k, v) in sort(collect(cache))
            @printf(io, "%s,%.16e\n", k, v)
        end
    end
end

"""
    reference_mu(sys; s, p1, p2, solver, agree_tol=1e-10)

High-accuracy reference: GL(s) at two resolutions, must agree to `agree_tol`
(relative). Cached in results/reference_values.csv. Large systems (beam) use
the lazy operator — a GL10 sparse assembly at p≈900 would need >1e8 nonzeros,
which is itself part of the accuracy–sparsity story, not a reference tool.
"""
function reference_mu(sys::BenchSystem; s::Int = sys.D > 10 ? 5 : 10,
                      p1::Int = 600, p2::Int = 900,
                      solver::Symbol = sys.D > 10 ? :lazy : :sparse,
                      agree_tol = 1e-10)
    cache = load_ref_cache()
    haskey(cache, sys.name) && return cache[sys.name]
    @printf("[ref] computing reference for %s (GL%d, p=%d & %d, %s)...\n", sys.name, s, p1, p2, solver)
    tab = GL(s)
    mu1 = mfcm_mu(sys, iseven(p1) ? p1 : p1 + 1, tab; tol=1e-13, solver=solver)
    mu2 = mfcm_mu(sys, iseven(p2) ? p2 : p2 + 1, tab; tol=1e-13, solver=solver)
    rel = abs(mu1 - mu2) / abs(mu2)
    @printf("[ref] %s: mu(p1)=%.15g  mu(p2)=%.15g  rel.diff=%.2e\n", sys.name, mu1, mu2, rel)
    rel < agree_tol || @warn "Reference for $(sys.name) not converged to $agree_tol (rel=$rel)"
    cache[sys.name] = mu2
    save_ref_cache(cache)
    return mu2
end

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

"Fit a log-log slope over points with err in (lo, hi)."
function fit_slope(xs, errs; lo=1e-13, hi=1e-2)
    idx = findall(i -> lo < errs[i] < hi && isfinite(errs[i]) && xs[i] > 0, eachindex(errs))
    length(idx) < 2 && return NaN
    A = [log10.(xs[idx]) ones(length(idx))]
    coeffs = A \ log10.(errs[idx])
    return coeffs[1]
end

"Append rows to a CSV (creating it with a header if absent)."
function append_csv(path, header::String, rows::Vector{<:Tuple})
    isnew = !isfile(path)
    open(path, "a") do io
        isnew && println(io, header)
        for row in rows
            println(io, join(row, ","))
        end
    end
end

"The standard method set for the comparison studies."
function method_set(; large_system::Bool=false)
    methods = Vector{Tuple{String, Any, Int}}()   # (label, tableau or :sdm..., nominal order)
    push!(methods, ("SDM-O2", :sdm2, 2))
    push!(methods, ("RK1", ExplicitEuler(), 1))
    push!(methods, ("RK2", Heun(), 2))
    push!(methods, ("RK4", MFCM.RK4(), 4))
    push!(methods, ("RK5", MFCM.RK5(), 5))
    push!(methods, ("GL1", GL(1), 2))
    push!(methods, ("GL2", GL(2), 4))
    push!(methods, ("GL3", GL(3), 6))
    if !large_system
        # high orders are only affordable on small systems: the per-step block
        # density grows ~ S^2 D^2 — that growth is measured in the sweet-spot study
        push!(methods, ("GL5", GL(5), 10))
        push!(methods, ("GL8", GL(8), 16))
    end
    return methods
end

println("[harness] loaded. Systems: mathieu, bio, turning_ssv, beam(, beam_aaw)")
