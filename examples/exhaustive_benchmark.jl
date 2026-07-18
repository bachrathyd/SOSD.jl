using SOSD
using SemiDiscretizationMethod
using Plots
using StaticArrays
using KrylovKit
using BenchmarkTools
using LaTeXStrings
using LinearAlgebra
using Printf

# --- Mathieu Problem Definition ---
function createMathieuProblem_SOSD(δ, ε, b0, a1; T=2π)
    AMx = SOSD.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = SOSD.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = SOSD.Additive(t -> @SVector [0.0, sin(4π / T * t)])
    SOSD.LDDEProblem{2, Float64}(AMx, [BMx1], cVec)
end

function createMathieuProblem_SDM(δ, ε, b0, a1; T=2π)
    AMx = SemiDiscretizationMethod.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = SemiDiscretizationMethod.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = SemiDiscretizationMethod.Additive(t -> @SVector [0.0, sin(4π / T * t)])
    SemiDiscretizationMethod.LDDEProblem(AMx, [BMx1], cVec)
end

function run_full_benchmark(bench_mode=:sparse)
    δ, ε, b0, a1 = 3.0, 0.2, -0.15, 0.1
    T = 2π
    D = 2
    
    filename = bench_mode == :sparse ? "mathieu_work_precision_sparse_full.png" : "mathieu_work_precision_lazy_full.png"
    main_title = bench_mode == :sparse ? "Exhaustive Sparse Benchmark" : "Turbo-Lazy Benchmark"

    println("Computing high-precision reference values...")
    p_ref = 800
    r_ref = p_ref
    tableau_ref = GL(10)
    prob_ref = createMathieuProblem_SOSD(δ, ε, b0, a1, T=T)
    grid_ref = TimeGrid(collect(range(0.0, T, length=p_ref+1)))
    sys_ref = build_system_matrices(prob_ref, grid_ref, tableau_ref, r_ref)
    m_ref = MonodromyMap(prob_ref, grid_ref, tableau_ref, sys_ref, p_ref, r_ref, (r_ref+1)*(11)*D)
    
    # Warmup and solve
    eigsolve(m_ref, rand(m_ref.state_size), 1, :LM)
    vals_ref, _ = eigsolve(m_ref, rand(m_ref.state_size), 1, :LM)
    mu_ref = abs(vals_ref[1])
    @show mu_ref

    # Resolution range: [1:9, 10^(1:0.25:5)]
    ps = [1:9..., round.(Int, 10 .^ (1:0.25:5))...]
    ps = sort(unique(ps))
    
    solvers = []
  #  # GL 1-10
  #  for s in 1:10
  #      push!(solvers, ("GL$s", GL(s), s))
  #  end
    # Explicit 1-5
    push!(solvers, ("Euler (O1)", ExplicitEuler(), 1))
    push!(solvers, ("Heun (O2)", Heun(), 2))
    push!(solvers, ("RK3 (O3)", RK3(), 3))
    push!(solvers, ("RK4 (O4)", RK4(), 4))
    push!(solvers, ("RK5 (O5)", RK5(), 6))
    
    results = Dict()
    execution_order = []
    
    default(fontfamily="Computer Modern", titlefontsize=12, guidefontsize=10, tickfontsize=9, legendfontsize=7)
    
    function save_benchmark_plot()
        p1 = plot(title="N_eval vs Eigenvalue Error", xlabel="N_eval (p * S)", ylabel="Error", xscale=:log10, yscale=:log10, grid=true, minorgrid=true, legend=:topright)
        p2 = plot(title="N_eval vs CPU Time", xlabel="N_eval (p * S)", ylabel="Time [s]", xscale=:log10, yscale=:log10, grid=true, minorgrid=true, legend=:bottomright)
        p3 = plot(title="Work-Precision: Time vs Error", xlabel="Time [s]", ylabel="Error", xscale=:log10, yscale=:log10, grid=true, minorgrid=true, legend=:topright)
        p4 = plot(title="N_eval vs Memory", xlabel="N_eval (p * S)", ylabel="Allocations [MB]", xscale=:log10, yscale=:log10, grid=true, minorgrid=true, legend=:bottomright)
        
        # Color palette
        color_palette = palette(:auto)

        # Plot SDM first if exists
        if haskey(results, "SDM (O2)")
            xs, ts, errs, mems = results["SDM (O2)"]
            valid = errs .> 1e-17
            label = "SDM (O2)"
            if sum(valid) > 1
                calc_idx = findall(i -> 1e-15 < errs[i] < 1e-3, 1:length(errs))
                if length(calc_idx) > 1
                    coeffs = [log10.(xs[calc_idx]) ones(length(calc_idx))] \ log10.(errs[calc_idx])
                    label = "SDM (O2) (k=$(round(-coeffs[1], digits=1)))"
                end
            end
            plot!(p1, xs[valid], errs[valid], marker=:circle, markersize=3, label=label, lw=3.0, color=:black)
            plot!(p2, xs, ts, marker=:circle, markersize=3, label="SDM (O2)", lw=3.0, color=:black)
            plot!(p3, ts[valid], errs[valid], marker=:circle, markersize=3, label=label, lw=3.0, color=:black)
            plot!(p4, xs, mems ./ (1024^2), marker=:circle, markersize=3, label="SDM (O2)", lw=3.0, color=:black)
        end

        for (idx, name) in enumerate(execution_order)
            if name == "SDM (O2)"; continue; end
            xs, ts, errs, mems = results[name]
            
            color = color_palette[mod1(idx, length(color_palette))]
            valid = errs .> 1e-17
            
            label = name
            if sum(valid) > 1
                # Drop error > 1e-3 and error < 1e-13 for fitting
                calc_idx = findall(i -> 1e-13 < errs[i] < 1e-3, 1:length(errs))
                if length(calc_idx) > 1
                    coeffs = [log10.(xs[calc_idx]) ones(length(calc_idx))] \ log10.(errs[calc_idx])
                    k_slope = -coeffs[1]
                    label = "$name (k=$(round(k_slope, digits=1)))"
                end
            end
            
            plot!(p1, xs[valid], errs[valid], marker=:circle, markersize=2, label=label, lw=1.5, color=color)
            plot!(p2, xs, ts, marker=:circle, markersize=2, label=name, lw=1.5, color=color)
            plot!(p3, ts[valid], errs[valid], marker=:circle, markersize=2, label=label, lw=1.5, color=color)
            plot!(p4, xs, mems ./ (1024^2), marker=:circle, markersize=2, label=name, lw=1.5, color=color)
        end
        
        plt = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1100), plot_title=main_title)
        savefig(plt, filename)
    end

    # --- 1. Benchmark SDM (Reference) ---
    println("\nBenchmarking SDM (Order 2) [Reference]...")
    sdm_times, sdm_errors, sdm_evals, sdm_mems = [], [], [], []
    push!(execution_order, "SDM (O2)")
    for p in ps[ps .<= 1000]
        print("$p ")
        try
            h = T / p
            prob_s = createMathieuProblem_SDM(δ, ε, b0, a1, T=T)
            method = SemiDiscretization(2, h)
            # Warmup
            spectralRadiusOfMapping(DiscreteMapping_LR(prob_s, method, 2π, n_steps=p))
            # Measure
            t = @belapsed spectralRadiusOfMapping(DiscreteMapping_LR($prob_s, $method, 2π, n_steps=$p))
            stats = @timed spectralRadiusOfMapping(DiscreteMapping_LR(prob_s, method, 2π, n_steps=p))
            
            err = abs(stats.value - mu_ref)
            push!(sdm_times, t)
            push!(sdm_errors, err)
            push!(sdm_evals, p * 2)
            push!(sdm_mems, stats.bytes)
        catch e
            println("\nSDM failed for p=$p: $e")
        end
    end
    println()
    results["SDM (O2)"] = (Float64.(sdm_evals), Float64.(sdm_times), Float64.(sdm_errors), Float64.(sdm_mems))
    save_benchmark_plot()

    # --- 2. Benchmark SOSD Solvers ---
    for (name, tab, S) in solvers
        full_name = bench_mode == :sparse ? "Sparse $name" : "Lazy $name"
        println("\nBenchmarking $full_name...")
        times, errors, evals, mems = [], [], [], []
        push!(execution_order, full_name)
        
        floor_count = 0
        for p in ps
            print("$p ")
            try
                h = T / p; r = p
                prob = createMathieuProblem_SOSD(δ, ε, b0, a1, T=T)
                grid = TimeGrid(collect(range(0.0, T, length=p+1)))
                BSIZE = (S + 1) * D
                state_size = (r + 1) * BSIZE
                
                # Setup Map
                sys = build_system_matrices(prob, grid, tab, r)
                m_lazy = MonodromyMap(prob, grid, tab, sys, p, r, state_size)
                m = bench_mode == :sparse ? SparseMonodromyMap(m_lazy) : m_lazy
                x0 = ones(state_size) / sqrt(state_size)
                
                # Warmup
                eigsolve(m, x0, 1, :LM)
                # Measure Time
                t = @belapsed eigsolve($m, $x0, 1, :LM)
                # Measure Memory
                stats = @timed eigsolve(m, x0, 1, :LM)
                
                vals, _ = stats.value
                err = abs(abs(vals[1]) - mu_ref)
                
                push!(times, t)
                push!(errors, err)
                push!(evals, p * S)
                push!(mems, stats.bytes)
                
                # Stop if last 3 points reached noise floor
                if err < 1e-13
                    floor_count += 1
                else
                    floor_count = 0
                end
                
                if floor_count >= 3 && p > 50
                    println("(Reached floor)")
                    break
                end
                
                # Stop if last execution time > 1.0s
                if t > 1.0 && p > 10
                    println("(Time limit 1.0s reached)")
                    break
                end
            catch e
                println("\nFailed for $full_name p=$p: $e")
            end
        end
        println()
        results[full_name] = (Float64.(evals), Float64.(times), Float64.(errors), Float64.(mems))
        save_benchmark_plot()
    end
    
    println("\nBenchmarking complete. Final plot saved to $filename")
end

# Run both if needed, but user asked for Sparse first then Lazy
if length(ARGS) > 0 && ARGS[1] == "lazy"
    run_full_benchmark(:lazy)
else
    run_full_benchmark(:sparse)
end
