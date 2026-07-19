# Re-measure turning + beam work-precision with minimum-of-repeats timing.
include(joinpath(@__DIR__, "harness.jl"))
SOSD_HARNESS_LOADED = true
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
            local mu, tm, ts, nrep, tmin
            try
                f = tab === :sdm2 ? (() -> sdm_mu(sys, p; order=2)) : (() -> sosd_mu(sys, p, tab))
                mu, tm, ts, nrep, tmin = time_stats(f; budget=2.0, min_reps=4, max_reps=10)
            catch e
                println("FAILED"); break
            end
            err = abs(mu - mu_ref); rel_err = err / abs(mu_ref)
            @printf("rel=%.1e t_min=%.3g\n", rel_err, tmin)
            append_csv(csv, "method,nominal_order,p,abs_error,rel_error,t_mean,t_std,n_reps,t_min",
                       [(label, nominal, p, err, rel_err, tm, ts, nrep, tmin)])
            rel_err < 5e-15 ? (floor_count += 1) : (floor_count = 0)
            (floor_count >= 3 && p > 30) && (println("(floor)"); break)
            tmin > p_max_time && (println("(cap)"); break)
        end
    end
end
run_work_precision(make_turning_ssv(); ps=unique(round.(Int, 10 .^ (1.8:0.2:3.8))))
run_work_precision(make_beam(); ps=unique(even_p.(2 .^ (2.5:0.5:9.5))))
println("WP LARGE RERUN DONE")
