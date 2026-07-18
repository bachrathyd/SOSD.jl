using SOSD
using SemiDiscretizationMethod
using Plots
using StaticArrays
using KrylovKit
using BenchmarkTools
using LaTeXStrings
using LinearAlgebra
using Printf
using DifferentialEquations
using DataInterpolations
using LinearMaps
using Serialization

# --- Mathieu Problem Definition (SOSD) ---
function createMathieuProblem_SOSD(δ, ε, b0, a1; T=2π)
    AMx = SOSD.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = SOSD.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = SOSD.Additive(t -> @SVector [0.0, 0.0])
    SOSD.LDDEProblem{2, Float64}(AMx, [BMx1], cVec)
end

# --- Mathieu Problem Definition (SDM) ---
function createMathieuProblem_SDM(δ, ε, b0, a1; T=2π)
    AMx = SemiDiscretizationMethod.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = SemiDiscretizationMethod.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = SemiDiscretizationMethod.Additive(t -> @SVector [0.0, 0.0])
    SemiDiscretizationMethod.LDDEProblem(AMx, [BMx1], cVec)
end

# --- Mathieu Problem Definition (DDE Solver) ---
function DelayMathieu!(du, u, h, p, t)
    δ, ε, b0, a1, τ, T = p
    du[1] = u[2]
    # IMPORTANT: h(p, t-τ) returns the history at t-τ
    # Our history is defined for t in [-τ, 0]
    hist = h(p, t - τ)
    du[2] = -(δ + ε * cos(2π / T * t)) * u[1] - a1 * u[2] + b0 * hist[1]
end

# --- DDE Solver-based Period Mapping ---
struct DDEPeriodMapping{P, PR, SA, TF, V} <: LinearMaps.LinearMap{ComplexF64}
    p_params::P
    prob::PR
    solv_args::SA
    T::Float64
    t_fixed::TF
    u_template::V
    state_size::Int
    interp_degree::Int
end

Base.size(m::DDEPeriodMapping) = (m.state_size, m.state_size)

function LinearMaps._unsafe_mul!(y_out::AbstractVector, m::DDEPeriodMapping, x_in::AbstractVector)
    d = length(m.u_template); p_pts = length(m.t_fixed); t_hist = m.t_fixed
    
    # Pre-extract real parts to avoid repeated calls
    x_real = real.(x_in)
    
    # Build interpolation for history
    # Note: x_in is [v_0, v_{-1}, ..., v_{-r}]
    # Chronological order for interpolation: [v_{-r}, ..., v_0]
    # v_{-r} is at index (p_pts-1)*d + 1
    
    # We'll use a more efficient way to build the interpolation
    vals1 = zeros(p_pts); vals2 = zeros(p_pts)
    for i in 1:p_pts
        # i=1 is t=-tau, which is v_{-r}
        idx_in = (p_pts - i) * d
        vals1[i] = x_real[idx_in + 1]
        vals2[i] = x_real[idx_in + 2]
    end
    
    deg = min(m.interp_degree, p_pts - 1)
    if deg < 1; deg = 1; end
    
    itp1 = BSplineInterpolation(vals1, t_hist, deg, :ArcLen, :Average)
    itp2 = BSplineInterpolation(vals2, t_hist, deg, :ArcLen, :Average)
    
    # History function for the DDE solver
    h_local(p, t) = @MVector [itp1(t), itp2(t)]
    
    u0 = h_local(m.p_params, 0.0)
    # Use the proper DDEProblem remake
    new_prob = remake(m.prob; u0=u0, h=h_local)
    
    sol = solve(new_prob, m.solv_args.alg; m.solv_args...)
    
    if sol.retcode != :Success && sol.retcode != :Terminated
        @warn "DDE Solver failed with retcode: $(sol.retcode)"
    end

    for (i, t_rel) in enumerate(m.t_fixed)
        t_eval = t_rel + m.T
        # Clamp t_eval to [0, T] to be safe
        if t_eval < 0.0; t_eval = 0.0; end
        if t_eval > m.T; t_eval = m.T; end
        
        # sol(t) uses dense output if available
        v = sol(t_eval)
        y_out[(i-1)*d + 1] = v[1]
        y_out[(i-1)*d + 2] = v[2]
    end
    return y_out
end

# --- Benchmarking Helpers ---
function run_sdm_bench(p_val, prob, meth, T)
    mapping = DiscreteMapping_LR(prob, meth, T, n_steps=p_val)
    return spectralRadiusOfMapping(mapping)
end

function run_mfcm_bench(p_val, r_val, prb, grd, tb, BSIZE)
    sys = build_system_matrices(prb, grd, tb, r_val)
    m_lazy = MonodromyMap(prb, grd, tb, sys, p_val, r_val, (r_val+1)*BSIZE)
    m = SparseMonodromyMap(m_lazy)
    x0 = ones(ComplexF64, m.state_size)
    vals, _ = eigsolve(m, x0, 1, :LM, tol=1e-14)
    return abs(vals[1])
end

function run_dde_bench(p_val, params, alg, s_order, is_implicit)
    τ, T = params[5], params[6]; dt = T / p_val
    t_fixed = collect(range(-τ, 0.0, length=p_val+1))
    u0_template = @MVector [0.0, 0.0]
    h_init(p, t) = @MVector [1.0, 0.0]
    prob = DDEProblem{true}(DelayMathieu!, u0_template, h_init, (0.0, T), params; constant_lags=[τ])
    
    interp_degree = is_implicit ? 2 * s_order : s_order
    # Ensure dense output is enabled for evaluations at t_fixed + T
    solv_args = (alg=MethodOfSteps(alg), adaptive=false, dt=dt, verbose=false, dense=true)
    mapping = DDEPeriodMapping(params, prob, solv_args, T, t_fixed, u0_template, length(t_fixed)*2, interp_degree)
    
    x0 = ones(ComplexF64, mapping.state_size)
    vals, _ = eigsolve(mapping, x0, 1, :LM, tol=1e-10)
    return abs(vals[1])
end

function run_comprehensive_benchmark()
    δ, ε, b0, a1, τ, T = 3.0, 0.2, -0.15, 0.1, 2π, 2π
    params = (δ, ε, b0, a1, τ, T); D = 2
    filename = "matrix_free_benchmark.png"
    main_title = "Progressive Benchmark: DDE Solvers vs SOSD vs SDM"

    println("Computing high-precision reference values...")
    p_ref = 1000; tableau_ref = GL(10)
    prob_ref = createMathieuProblem_SOSD(δ, ε, b0, a1, T=T); grid_ref = TimeGrid(collect(range(0.0, T, length=p_ref+1)))
    sys_ref = build_system_matrices(prob_ref, grid_ref, tableau_ref, p_ref)
    m_ref = SparseMonodromyMap(MonodromyMap(prob_ref, grid_ref, tableau_ref, sys_ref, p_ref, p_ref, (p_ref+1)*11*D))
    vals_ref, _ = eigsolve(m_ref, rand(ComplexF64, m_ref.state_size), 1, :LM)
    mu_ref = abs(vals_ref[1]); @printf("Reference mu: %.16f\n", mu_ref)

    ps = unique(sort([10, 20, 30, 50, round.(Int, 10 .^ (1.8:0.4:4.5))...]))
    
    all_solvers = [
        ("SDM (O2)", :SDM, 2, false),
        ("SOSD GL1 (O1)", :SOSD_GL1, 1, true),
        ("SOSD GL3 (O6)", :SOSD_GL3, 6, true),
        ("SOSD RK4 (O4)", :SOSD_RK4, 4, false),
        # DDE Explicit
        ("DDE Euler (O1)", DifferentialEquations.Euler(), 1, false),
        ("DDE Heun (O2)", DifferentialEquations.Heun(), 2, false),
        ("DDE BS3 (O3)", DifferentialEquations.BS3(), 3, false),
        ("DDE RK4 (O4)", DifferentialEquations.RK4(), 4, false),
        ("DDE BS5 (O5)", DifferentialEquations.BS5(), 5, false),
        ("DDE Vern6 (O6)", DifferentialEquations.Vern6(), 6, false),
        ("DDE Vern7 (O7)", DifferentialEquations.Vern7(), 7, false),
        ("DDE Vern8 (O8)", DifferentialEquations.Vern8(), 8, false),
        ("DDE Vern9 (O9)", DifferentialEquations.Vern9(), 9, false),
        ("DDE Feagin10 (O10)", DifferentialEquations.Feagin10(), 10, false),
        ("DDE Feagin12 (O12)", DifferentialEquations.Feagin12(), 12, false),
        ("DDE Feagin14 (O14)", DifferentialEquations.Feagin14(), 14, false),
        # DDE Implicit
        ("DDE ImpEuler (O1)", DifferentialEquations.ImplicitEuler(), 1, true),
        ("DDE Trap (O2)", DifferentialEquations.Trapezoid(), 2, true),
        ("DDE Rosen23 (O2)", DifferentialEquations.Rosenbrock23(), 2, true),
        ("DDE Rodas4 (O4)", DifferentialEquations.Rodas4(), 4, true),
        ("DDE Radau5 (O5)", DifferentialEquations.RadauIIA5(), 5, true)
    ]

    results = Dict(name => (Int[], Float64[], Float64[], Float64[]) for (name, _, _, _) in all_solvers)
    
    # --- DIAGNOSTIC TEST ---
    println("\nRunning diagnostic check for DDE mapping...")
    try
        val_test = run_dde_bench(20, params, DifferentialEquations.RK4(), 4, false)
        println("Diagnostic RK4 (p=20) mu = $val_test (Target ≈ 0.35)")
    catch e
        println("Diagnostic failed: $e")
    end

    for p in ps
        println("\n--- Testing Resolution p = $p ---")
        for (name, alg, s_order, is_implicit) in all_solvers
            if (name == "SDM (O2)" && p > 1500) || (startswith(name, "DDE") && p > 1000)
                continue
            end
            
            print("  $name: ")
            try
                local t, val, bts
                if alg == :SDM
                    prob_s = createMathieuProblem_SDM(δ, ε, b0, a1, T=T); method = SemiDiscretization(2, T/p)
                    run_sdm_bench(p, prob_s, method, T) 
                    t = @belapsed run_sdm_bench($p, $prob_s, $method, $T)
                    stats = @timed run_sdm_bench(p, prob_s, method, T)
                    val = stats.value; bts = stats.bytes
                elseif alg == :SOSD_GL1
                    prob = createMathieuProblem_SOSD(δ, ε, b0, a1, T=T); grid = TimeGrid(collect(range(0.0, T, length=p+1)))
                    run_mfcm_bench(p, p, prob, grid, GL(1), 4) 
                    t = @belapsed run_mfcm_bench($p, $p, $prob, $grid, $(GL(1)), 4)
                    stats = @timed run_mfcm_bench(p, p, prob, grid, GL(1), 4)
                    val = stats.value; bts = stats.bytes
                elseif alg == :SOSD_GL3
                    prob = createMathieuProblem_SOSD(δ, ε, b0, a1, T=T); grid = TimeGrid(collect(range(0.0, T, length=p+1)))
                    run_mfcm_bench(p, p, prob, grid, GL(3), 8)
                    t = @belapsed run_mfcm_bench($p, $p, $prob, $grid, $(GL(3)), 8)
                    stats = @timed run_mfcm_bench(p, p, prob, grid, GL(3), 8)
                    val = stats.value; bts = stats.bytes
                elseif alg == :SOSD_RK4
                    prob = createMathieuProblem_SOSD(δ, ε, b0, a1, T=T); grid = TimeGrid(collect(range(0.0, T, length=p+1)))
                    run_mfcm_bench(p, p, prob, grid, SOSD.RK4(), 10)
                    t = @belapsed run_mfcm_bench($p, $p, $prob, $grid, $(SOSD.RK4()), 10)
                    stats = @timed run_mfcm_bench(p, p, prob, grid, SOSD.RK4(), 10)
                    val = stats.value; bts = stats.bytes
                else
                    run_dde_bench(p, params, alg, s_order, is_implicit) 
                    t = @belapsed run_dde_bench($p, $params, $alg, $s_order, $is_implicit)
                    stats = @timed run_dde_bench(p, params, alg, s_order, is_implicit)
                    val = stats.value; bts = stats.bytes
                end
                
                err = abs(val - mu_ref)
                push!(results[name][1], p); push!(results[name][2], t); push!(results[name][3], err); push!(results[name][4], bts)
                @printf("Time=%.3fs, Err=%.2e\n", t, err)
            catch e
                println("Failed: $e")
            end
        end
        
        # Real-time Plotting
        default(fontfamily="Computer Modern", titlefontsize=12, guidefontsize=10, tickfontsize=9, legendfontsize=6)
        p1 = plot(title="p vs Error", xscale=:log10, yscale=:log10, xlabel="p", ylabel="Error", grid=true)
        p2 = plot(title="p vs Time", xscale=:log10, yscale=:log10, xlabel="p", ylabel="Time [s]", grid=true)
        p3 = plot(title="Work-Precision", xscale=:log10, yscale=:log10, xlabel="Time [s]", ylabel="Error", grid=true)
        p4 = plot(title="p vs Memory", xscale=:log10, yscale=:log10, xlabel="p", ylabel="Allocations [MB]", grid=true)
        
        color_palette = palette(:turbo, length(all_solvers))
        for (idx, (name, _, _, _)) in enumerate(all_solvers)
            ps_v, ts_v, es_v, ms_v = results[name]
            if isempty(ps_v); continue; end
            c = color_palette[idx]; lw = 1.0; ls = :solid
            if name == "SDM (O2)"; c = :black; lw = 3.0; end
            if contains(name, "SOSD"); lw = 2.0; end
            if contains(name, "Imp") || contains(name, "Radau") || contains(name, "Rodas") || contains(name, "Trap") || contains(name, "Rosen"); ls = :dash; end
            v_idx = findall(x -> x > 0, es_v); ps_p, ts_p, es_p, ms_p = ps_v[v_idx], ts_v[v_idx], max.(es_v[v_idx], 1e-16), ms_v[v_idx]
            label = name
            v_fit = findall(i -> 1e-13 < es_p[i] < 1e-1, 1:length(es_p))
            if length(v_fit) > 1
                k = -([log10.(ps_p[v_fit]) ones(length(v_fit))] \ log10.(es_p[v_fit]))[1]
                label = "$name (k=$(round(k, digits=1)))"
            end
            plot!(p1, ps_p, es_p, m=:circle, ms=2, c=c, lw=lw, ls=ls, label=label)
            plot!(p2, ps_p, ts_p, m=:circle, ms=2, c=c, lw=lw, ls=ls, label=name)
            plot!(p3, ts_p, es_p, m=:circle, ms=2, c=c, lw=lw, ls=ls, label=label)
            plot!(p4, ps_p, ms_p ./ (1024^2), m=:circle, ms=2, c=c, lw=lw, ls=ls, label=name)
        end
        savefig(plot(p1, p2, p3, p4, layout=(2,2), size=(1300, 1100), plot_title=main_title), filename)
        serialize("benchmark_data.jls", results)
        println("Updated plot.")
    end
end

run_comprehensive_benchmark()
