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
    cVec = SOSD.Additive(t -> @SVector [0.0, 0.0]) # No additive term for stability
    SOSD.LDDEProblem{2, Float64}(AMx, [BMx1], cVec)
end

function createMathieuProblem_SDM(δ, ε, b0, a1; T=2π)
    AMx = SemiDiscretizationMethod.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = SemiDiscretizationMethod.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = SemiDiscretizationMethod.Additive(t -> @SVector [0.0, 0.0])
    SemiDiscretizationMethod.LDDEProblem(AMx, [BMx1], cVec)
end

# --- Helper Functions for Benchmarking (at top-level for macro scope) ---
function run_sdm_bench(p_val, prob, meth)
    mapping = DiscreteMapping_LR(prob, meth, 2π, n_steps=p_val)
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

function run_final_benchmark()
    δ, ε, b0, a1 = 3.0, 0.2, -0.15, 0.1
    T = 2π
    D = 2
    
    filename = "matrix_free_benchmark.png"
    main_title = "SOSD vs SDM Performance & Complexity Analysis"

    println("Computing high-precision reference values...")
    p_ref = 1000
    r_ref = p_ref
    tableau_ref = GL(10)
    prob_ref = createMathieuProblem_SOSD(δ, ε, b0, a1, T=T)
    grid_ref = TimeGrid(collect(range(0.0, T, length=p_ref+1)))
    sys_ref = build_system_matrices(prob_ref, grid_ref, tableau_ref, r_ref)
    m_ref_lazy = MonodromyMap(prob_ref, grid_ref, tableau_ref, sys_ref, p_ref, r_ref, (r_ref+1)*(11)*D)
    m_ref = SparseMonodromyMap(m_ref_lazy)
    
    eigsolve(m_ref, rand(ComplexF64, m_ref.state_size), 1, :LM)
    vals_ref, _ = eigsolve(m_ref, rand(ComplexF64, m_ref.state_size), 1, :LM)
    mu_ref = abs(vals_ref[1])
    @printf("Reference mu: %.16f\n", mu_ref)

    ps = unique(sort([10, 15, 20, 25, 30, round.(Int, 10 .^ (1.5:0.25:5))...]))
    
    solvers = [
        ("SOSD RK4", RK4(), 4),
        ("SOSD GL1 (IE)", GL(1), 1),
        ("SOSD GL2", GL(2), 2),
        ("SOSD GL3", GL(3), 3),
        ("SOSD GL5", GL(5), 5),
    ]
    
    results = Dict()
    execution_order = []
    
    # --- 1. Benchmark SDM ---
    println("\nBenchmarking SDM (Order 2)...")
    sdm_times, sdm_errors, sdm_ps, sdm_mems = [], [], [], []
    push!(execution_order, "SDM (O2)")
    for p in ps[ps .<= 2000] 
        print("$p ")
        try
            h = T / p
            prob_s = createMathieuProblem_SDM(δ, ε, b0, a1, T=T)
            method = SemiDiscretization(2, h)
            
            # Warmup
            run_sdm_bench(p, prob_s, method)
            
            t = @belapsed run_sdm_bench($p, $prob_s, $method)
            stats = @timed run_sdm_bench(p, prob_s, method)
            
            err = abs(stats.value - mu_ref)
            push!(sdm_times, t)
            push!(sdm_errors, err)
            push!(sdm_ps, p)
            push!(sdm_mems, stats.bytes)
            
            if t > 20.0; break; end 
        catch e
            println("\nSDM failed for p=$p: $e")
            break
        end
    end
    println()
    results["SDM (O2)"] = (Float64.(sdm_ps), Float64.(sdm_times), Float64.(sdm_errors), Float64.(sdm_mems))

    # --- 2. Benchmark SOSD Solvers ---
    for (name, tab, S) in solvers
        println("\nBenchmarking $name...")
        times, errors, p_vals, mems = [], [], [], []
        push!(execution_order, name)
        
        floor_count = 0
        for p in ps
            print("$p ")
            try
                h = T / p; r = p
                prob = createMathieuProblem_SOSD(δ, ε, b0, a1, T=T)
                grid = TimeGrid(collect(range(0.0, T, length=p+1)))
                BSIZE = (S + 1) * D
                
                # Warmup
                run_mfcm_bench(p, r, prob, grid, tab, BSIZE)
                
                t = @belapsed run_mfcm_bench($p, $r, $prob, $grid, $tab, $BSIZE)
                stats = @timed run_mfcm_bench(p, r, prob, grid, tab, BSIZE)
                
                err = abs(stats.value - mu_ref)
                
                push!(times, t)
                push!(errors, err)
                push!(p_vals, p)
                push!(mems, stats.bytes)
                
                if err < 2e-15
                    floor_count += 1
                else
                    floor_count = 0
                end
                
                if floor_count >= 2 && p > 100
                    println("(Floor reached)")
                end
                
                if t > 10.0 && p > 100
                    println("(Time limit 10s reached)")
                    break
                end
                if p > 100000; break; end
            catch e
                println("\nFailed for $name p=$p: $e")
                break
            end
        end
        println()
        results[name] = (Float64.(p_vals), Float64.(times), Float64.(errors), Float64.(mems))
    end

    # --- Plotting ---
    default(fontfamily="Computer Modern", titlefontsize=12, guidefontsize=10, tickfontsize=9, legendfontsize=8)
    
    p1 = plot(title="Resolution (p) vs Eigenvalue Error", xlabel="p (Grid Points)", ylabel="Error", xscale=:log10, yscale=:log10, grid=true, minorgrid=true)
    p2 = plot(title="Resolution (p) vs CPU Time", xlabel="p (Grid Points)", ylabel="Time [s]", xscale=:log10, yscale=:log10, grid=true, minorgrid=true)
    p3 = plot(title="Work-Precision: Time vs Error", xlabel="Time [s]", ylabel="Error", xscale=:log10, yscale=:log10, grid=true, minorgrid=true)
    p4 = plot(title="Resolution (p) vs Memory", xlabel="p (Grid Points)", ylabel="Allocations [MB]", xscale=:log10, yscale=:log10, grid=true, minorgrid=true)
colors = palette(:auto)
for (idx, name) in enumerate(execution_order)
    ps_vals, ts_vals, es_vals, ms_vals = results[name]
    if isempty(ps_vals); continue; end

    # Filter non-positive values for log scales
    valid_idx = findall(x -> x > 0, es_vals)
    if isempty(valid_idx); continue; end

    ps_plot = ps_vals[valid_idx]
    ts = ts_vals[valid_idx]
    es = es_vals[valid_idx]
    ms = ms_vals[valid_idx]

    # Clamp errors to machine precision for better visualization on log scale
    es_clamped = max.(es, 1e-16)

    c = colors[mod1(idx, length(colors))]
    lw = 1.5
    msize = 3
    if name == "SDM (O2)"
        c = :black
        lw = 3.0
        msize = 4
    end

    label = name
    # Fit convergence rate for p1
    valid_fit = findall(i -> 1e-13 < es[i] < 1e-2, 1:length(es))
    if length(valid_fit) > 1
        coeffs = [log10.(ps_plot[valid_fit]) ones(length(valid_fit))] \ log10.(es[valid_fit])
        k = -coeffs[1]
        label = "$name (k=$(round(k, digits=1)))"
    end

    plot!(p1, ps_plot, es_clamped, marker=:circle, markersize=msize, label=label, color=c, lw=lw)
    plot!(p2, ps_plot, ts, marker=:circle, markersize=msize, label=name, color=c, lw=lw)
    plot!(p3, ts, es_clamped, marker=:circle, markersize=msize, label=label, color=c, lw=lw)
    plot!(p4, ps_plot, ms ./ (1024^2), marker=:circle, markersize=msize, label=name, color=c, lw=lw)
end


    p_theory = [10.0, 100000.0]
    t_ref = results[execution_order[2]][2][1]
    p_ref_val = results[execution_order[2]][1][1]
    plot!(p2, p_theory, t_ref .* (p_theory ./ p_ref_val), label="O(p) Linear", color=:gray, linestyle=:dash, lw=1)

    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1100), plot_title=main_title)
    savefig(final_plot, filename)
    println("\nBenchmark complete. Final plot saved to $filename")
end

run_final_benchmark()
