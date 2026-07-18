
using Plots
using StaticArrays
using KrylovKit
using DifferentialEquations
using DataInterpolations
using BenchmarkTools
using LinearAlgebra
using SemiDiscretizationMethod
using Printf

# --- Mathieu Problem Definition (Synchronized with exhaustive_benchmark.jl) ---
# Equation: x'' + a1*x' + (δ + ε*cos(2π/T*t))x = b0*x(t-τ)
function DelayMathieu!(du, u, h, p, t)
    δ, ε, b0, a1, τ, T = p
    du[1] = u[2]
    du[2] = -(δ + ε * cos(2π / T * t)) * u[1] - a1 * u[2] + b0 * h(p, t - τ)[1]
end

# --- Operator Structure ---
struct DDEPeriodMapping{P, PR, SA, TF, V}
    p_params::P
    prob::PR
    solv_args::SA
    T::Float64
    t_fixed::TF
    u_template::V
end

function (m::DDEPeriodMapping)(u_vec)
    d = length(m.u_template)
    # Use order 5 B-Splines for high resolution, fallback to lower for small p
    order = min(5, length(m.t_fixed) - 1)
    if order < 1
        return u_vec # Should not happen with p >= 6
    end
    
    interps = ntuple(i -> BSplineInterpolation(getindex.(u_vec, i), m.t_fixed, order, :ArcLen, :Average), d)
    h_func(p, t_eval) = typeof(m.u_template)(ntuple(i -> interps[i](t_eval), Val(d)))
    
    u0 = h_func(m.p_params, m.t_fixed[end])
    new_prob = remake(m.prob; u0=u0, h=h_func, p=m.p_params, tspan=(0.0, m.T))
    
    sol = solve(new_prob; m.solv_args...)
    t_save = m.t_fixed .+ m.T
    return [sol(t) for t in t_save]
end

# --- Grid Synchronization ---
function setup_synchronized_grid(params, p_target)
    τ, T = params[5], params[6]
    dt = T / p_target
    N_T = Int(round(T / dt))
    dt_sync = T / N_T
    N_tau = Int(ceil(τ / dt_sync))
    tau_max = N_tau * dt_sync
    t_fixed = collect(range(-tau_max, 0.0, length=N_tau + 1))
    return t_fixed, dt_sync
end

# --- Benchmark Suite ---
function run_exhaustive_matrix_free_benchmark()
    println("Starting Exhaustive Matrix-Free DDE Benchmark...")
    
    # Parameters from exhaustive_benchmark.jl
    params = (3.0, 0.2, -0.15, 0.1, 2π, 2π)
    τ, T = params[5], params[6]
    u0_template = @MArray [0.0, 0.0]
    
    mu_ref = 0.3514172609535511
    println("Synchronized Reference mu: $mu_ref")

    # Start from p=6 to satisfy BSplineInterpolation(order=5)
    ps_target = unique(sort([6:9..., round.(Int, 10 .^ (1:0.15:4))...]))
    
    solvers = [
        ("Euler (O1)", Euler(), false),
        ("Heun (O2)", Heun(), false),
        ("BS3 (O3)", BS3(), false),
        ("RK4 (O4)", RK4(), false),
        ("Vern6 (O6)", Vern6(), false),
        ("ImplicitEuler (O1)", ImplicitEuler(), true),
        ("Trapezoid (O2)", Trapezoid(), true),
    ]

    results = Dict()
    execution_order = []

    # 1. Benchmark SDM
    println("\nBenchmarking SDM (Order 2) [Reference]...")
    sdm_ps, sdm_times, sdm_errors, sdm_mems = [], [], [], []
    push!(execution_order, "SDM (O2)")
    for p in ps_target
        if p > 1000; break; end 
        t_fixed, dt = setup_synchronized_grid(params, p)
        prob_sdm = LDDEProblem(ProportionalMX(t -> [0.0 1.0; -(params[1] + params[2] * cos(2π * t / T)) -params[4]]),
                               [DelayMX(τ, t -> [0.0 0.0; params[3] 0.0])])
        try
            # Skip very low p if SDM fails (SDM often needs p >= r)
            if p < 5; continue; end
            
            method = SemiDiscretization(2, dt)
            m_sdm = DiscreteMapping_LR(prob_sdm, method, T, n_steps=p)
            
            # Warmup
            spectralRadiusOfMapping(m_sdm)
            
            t_bench = @belapsed spectralRadiusOfMapping($m_sdm)
            stats = @timed spectralRadiusOfMapping(m_sdm)
            err = abs(stats.value - mu_ref)
            
            push!(sdm_ps, length(t_fixed))
            push!(sdm_times, t_bench)
            push!(sdm_errors, err)
            push!(sdm_mems, stats.bytes)
            @printf("p=%-5d Time=%-8.2fms Err=%-8.2e\n", p, t_bench*1000, err)
        catch e
            println("\nSDM error at p=$p: $e")
            break
        end
    end
    results["SDM (O2)"] = (Float64.(sdm_ps), Float64.(sdm_times), Float64.(sdm_errors), Float64.(sdm_mems))

    # 2. Benchmark MFCM (Matrix-Free DiffEq) Solvers
    for (name, alg, is_implicit) in solvers
        println("\nBenchmarking $name...")
        times, errors, p_vals, mems = [], [], [], []
        push!(execution_order, name)
        
        for p_req in ps_target
            t_fixed, dt = setup_synchronized_grid(params, p_req)
            p_actual = length(t_fixed)
            
            h_init(p_args, t) = @MArray [1.0, 0.0]
            prob = DDEProblem{true}(DelayMathieu!, u0_template, h_init, (0.0, T), params; constant_lags=[τ])
            solv_args = (alg=MethodOfSteps(alg), adaptive=false, dt=dt, verbose=false, dense=true)
            mapping = DDEPeriodMapping(params, prob, solv_args, T, t_fixed, u0_template)
            
            s_initial = [MVector{2}(rand(2)) for _ in 1:p_actual]
            Neig = 1
            krylov_solver = KrylovKit.Arnoldi(tol=1e-10, krylovdim=15, verbosity=0, maxiter=20)

            try
                schursolve(mapping, s_initial, Neig, :LM, krylov_solver)
                t_bench = @belapsed schursolve($mapping, $s_initial, $Neig, :LM, $krylov_solver)
                stats = @timed schursolve(mapping, s_initial, Neig, :LM, krylov_solver)
                
                vals = stats.value[3]
                err = abs(abs(vals[1]) - mu_ref)
                
                push!(times, t_bench)
                push!(errors, err)
                push!(p_vals, p_actual)
                push!(mems, stats.bytes)
                @printf("p=%-5d Time=%-8.2fms Err=%-8.2e\n", p_req, t_bench*1000, err)

                if t_bench > 3.0
                    println("(3s limit reached)")
                    break
                end
                if err < 1e-14 && p_req > 100
                    println("(Noise floor reached)")
                    break
                end
            catch e
                println("Failed for p=$p_req: $e")
                break
            end
        end
        results[name] = (Float64.(p_vals), Float64.(times), Float64.(errors), Float64.(mems))
    end

    # --- Plotting ---
    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=8)
    
    p1 = plot(title="p vs Eigenvalue Error", xlabel="p (Grid Points)", ylabel="Error", xscale=:log10, yscale=:log10, grid=true, minorgrid=true)
    p2 = plot(title="p vs CPU Time", xlabel="p (Grid Points)", ylabel="Time [s]", xscale=:log10, yscale=:log10, grid=true, minorgrid=true)
    p3 = plot(title="Work-Precision: Time vs Error", xlabel="Time [s]", ylabel="Error", xscale=:log10, yscale=:log10, grid=true, minorgrid=true)
    p4 = plot(title="p vs Memory", xlabel="p (Grid Points)", ylabel="Allocations [MB]", xscale=:log10, yscale=:log10, grid=true, minorgrid=true)

    colors = palette(:auto)
    for (idx, name) in enumerate(execution_order)
        ps_plot, ts, es, ms = results[name]
        if isempty(ps_plot); continue; end
        c = colors[mod1(idx, length(colors))]
        if name == "SDM (O2)"; c = :black; end
        
        label = name
        valid_fit = findall(i -> 1e-12 < es[i] < 1e-2, 1:length(es))
        if length(valid_fit) > 1
            coeffs = [log10.(ps_plot[valid_fit]) ones(length(valid_fit))] \ log10.(es[valid_fit])
            k = -coeffs[1]
            label = "$name (k=$(round(k, digits=1)))"
        end

        plot!(p1, ps_plot, es, marker=:circle, markersize=3, label=label, color=c)
        plot!(p2, ps_plot, ts, marker=:circle, markersize=3, label=name, color=c)
        plot!(p3, ts, es, marker=:circle, markersize=3, label=label, color=c)
        plot!(p4, ps_plot, ms ./ (1024^2), marker=:circle, markersize=3, label=name, color=c)
    end

    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1100), plot_title="Matrix-Free DDE Solver Benchmark (Synchronized Parameters)")
    savefig(final_plot, "matrix_free_benchmark_synchronized.png")
    println("\nBenchmark complete. Final plot saved to matrix_free_benchmark_synchronized.png")
end

run_exhaustive_matrix_free_benchmark()
