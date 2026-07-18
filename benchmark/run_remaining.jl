# Remaining studies after the main suite (restarted with the D>12 dense-path
# fix): beam sweet-spot, non-smooth stress test, turning sweet-spot showcase.

include(joinpath(@__DIR__, "harness.jl"))
SOSD_HARNESS_LOADED = true

t0 = time()

println("="^70, "\n SWEET SPOT — BEAM\n", "="^70)
even_p(x) = max(2, 2 * round(Int, x / 2))
# reuse the sweep from run_sweet_spot.jl without re-running mathieu:
function sweep_ps(sys::BenchSystem, ss::Vector{Int}, ps_of_s::Function; time_cap=25.0)
    mu_ref = reference_mu(sys)
    csv = joinpath(RESULTS_DIR, "sweet_spot_$(sys.name).csv")
    isfile(csv) && rm(csv)
    for s in ss
        tab = GL(s)
        println("\n[$(sys.name)] GL$s (order $(2s))")
        for p in ps_of_s(s)
            print("  p=$p ")
            local mu, tm, ts, nrep, stats
            try
                mu, tm, ts, nrep = time_stats(() -> sosd_mu(sys, p, tab); budget=1.5, min_reps=3, max_reps=8)
                stats = matrix_stats(sys, p, tab)
            catch e
                println("FAILED: ", sprint(showerror, e)[1:min(end,160)])
                break
            end
            rel_err = abs(mu - mu_ref) / abs(mu_ref)
            @printf("rel_err=%.2e t=%.3gs nnzL=%d nnzR=%d\n", rel_err, tm, stats.nnz_L, stats.nnz_R)
            append_csv(csv, "s,order,p,rel_error,t_mean,t_std,n_reps,n_L,nnz_L,nnz_R,density_L,bandwidth_L",
                       [(s, 2s, p, rel_err, tm, ts, nrep, stats.n_L, stats.nnz_L, stats.nnz_R,
                         stats.density_L, stats.bandwidth_L)])
            tm > time_cap && (println("  (time cap)"); break)
            rel_err < 1e-14 && p > 4 && break
        end
    end
    println("[sweet-spot] $(sys.name) done -> $csv")
end

sweep_ps(make_beam(), collect(1:6), s -> unique(even_p.(2 .^ (1.5:0.5:8.5))); time_cap=4.0)

println("\n", "="^70, "\n NON-SMOOTH STRESS TEST\n", "="^70)
include(joinpath(@__DIR__, "run_nonsmooth.jl"))

println("\n", "="^70, "\n SWEET SPOT — TURNING (large-p showcase)\n", "="^70)
include(joinpath(@__DIR__, "run_sweet_spot_turning.jl"))

@printf("\nRemaining studies finished in %.1f minutes.\n", (time() - t0) / 60)
