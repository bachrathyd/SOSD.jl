# Work-precision and time-complexity study (handoff §4.1 + §4.2):
# for every system and method, sweep p and record eigenvalue error and CPU time
# (warmup + repeated runs, mean ± std). All data goes to CSV; figures are made
# separately by make_figures.jl.

(@isdefined SOSD_HARNESS_LOADED) || (include(joinpath(@__DIR__, "harness.jl")); SOSD_HARNESS_LOADED = true)

"Round to even (beam needs r = p/2)."
even_p(x) = max(2, 2 * round(Int, x / 2))

function run_work_precision(sys::BenchSystem; p_max_time::Float64=20.0,
                            ps::Vector{Int}=unique(round.(Int, 10 .^ (0.8:0.2:3.6))))
    mu_ref = reference_mu(sys)
    csv = joinpath(RESULTS_DIR, "work_precision_$(sys.name).csv")
    isfile(csv) && rm(csv)
    large = sys.D > 10

    for (label, tab, nominal) in method_set(large_system=large)
        println("\n[$(sys.name)] $label")
        floor_count = 0
        for p in ps
            print("  p=$p ")
            local mu, tm, ts, nrep
            try
                f = tab === :sdm2 ? (() -> sdm_mu(sys, p; order=2)) : (() -> sosd_mu(sys, p, tab))
                mu, tm, ts, nrep = time_stats(f; budget=2.0, min_reps=4, max_reps=10)
            catch e
                println("FAILED: ", sprint(showerror, e)[1:min(end,200)])
                break
            end
            err = abs(mu - mu_ref)
            rel_err = err / abs(mu_ref)
            @printf("rel_err=%.2e t=%.3gs(±%.1g,n=%d)\n", rel_err, tm, ts, nrep)
            append_csv(csv, "method,nominal_order,p,abs_error,rel_error,t_mean,t_std,n_reps",
                       [(label, nominal, p, err, rel_err, tm, ts, nrep)])
            # stopping rules
            rel_err < 5e-15 ? (floor_count += 1) : (floor_count = 0)
            if floor_count >= 3 && p > 30
                println("  (floor reached)"); break
            end
            if tm > p_max_time
                println("  (time cap)"); break
            end
        end
    end
    println("[work-precision] $(sys.name) done -> $csv")
end

run_work_precision(make_mathieu())
run_work_precision(make_bio())
# turning: T ≈ 209 ⇒ meaningful resolutions start around p ≈ 100
run_work_precision(make_turning_ssv(); ps=unique(round.(Int, 10 .^ (1.8:0.2:3.8))))
run_work_precision(make_beam(); ps=unique(even_p.(2 .^ (2.5:0.5:9.5))))
