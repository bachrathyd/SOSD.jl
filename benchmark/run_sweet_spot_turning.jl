# Large-p CPU showcase (Daniel's request): the turning-SSV system needs at
# least several hundred points per period (T/T_osc ≈ 36), in contrast to the
# delayed Mathieu equation where p ≈ 10–30 suffices. Sweep (p, s) exactly as
# in run_sweet_spot.jl; the banded moderate-order route should win decisively
# here, while the single-step spectral corner is not even feasible.

(@isdefined SOSD_HARNESS_LOADED) || (include(joinpath(@__DIR__, "harness.jl")); SOSD_HARNESS_LOADED = true)

function sweep_ps_turning(sys::BenchSystem, ss::Vector{Int}, ps_of_s::Function; time_cap=30.0)
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
            @printf("rel_err=%.2e t=%.3gs nnzL=%d\n", rel_err, tm, stats.nnz_L)
            append_csv(csv, "s,order,p,rel_error,t_mean,t_std,n_reps,n_L,nnz_L,density_L,bandwidth_L",
                       [(s, 2s, p, rel_err, tm, ts, nrep, stats.n_L, stats.nnz_L,
                         stats.density_L, stats.bandwidth_L)])
            tm > time_cap && (println("  (time cap)"); break)
            rel_err < 1e-14 && p > 200 && break
        end
    end
    println("[sweet-spot] $(sys.name) done -> $csv")
end

sweep_ps_turning(make_turning_ssv(),
                 collect(1:8),
                 s -> unique(round.(Int, 10 .^ (2.0:0.2:3.8))))
