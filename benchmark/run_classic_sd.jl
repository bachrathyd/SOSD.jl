# Baseline parity with the predecessor (handoff §10, item 1):
# reproduce the SD-classic (traditional monodromy build, ~O(p^2.8)) versus
# MFSD (banded LR mapping, O(p^1)) time-complexity contrast on the delayed
# Mathieu equation, in the same environment as all other measurements.

(@isdefined SOSD_HARNESS_LOADED) || (include(joinpath(@__DIR__, "harness.jl")); SOSD_HARNESS_LOADED = true)

function run_classic_sd()
    sys = make_mathieu()
    csv = joinpath(RESULTS_DIR, "classic_sd_vs_mfsd.csv")
    isfile(csv) && rm(csv)

    ps = unique(round.(Int, 10 .^ (1.0:0.25:6.0)))   # up to p = 1e6 (predecessor range)
    classic_alive = true
    for p in ps
        h = sys.T / p
        method = SemiDiscretization(2, h)

        # MFSD (banded LR) — linear complexity, full range to 1e6
        mu_lr, tm_lr, ts_lr, n_lr, t_lr = time_stats(
            () -> spectralRadiusOfMapping(DiscreteMapping_LR(sys.sdm_prob, method, sys.taumax, n_steps=p));
            budget=2.0, min_reps=2, max_reps=6)

        # SD-classic (full one-period mapping) — stop once it exceeds the 2 s cap
        t_cl = NaN; mu_cl = NaN
        if classic_alive
            try
                mu_cl, tm_cl, ts_cl, n_cl, t_cl = time_stats(
                    () -> spectralRadiusOfMapping(DiscreteMapping(sys.sdm_prob, method, sys.taumax, n_steps=p));
                    budget=2.0, min_reps=2, max_reps=6)
                t_cl > 2.0 && (classic_alive = false)
            catch e
                @warn "SD-classic failed at p=$p" exception=(e,)
                classic_alive = false
            end
        end

        @printf("p=%7d  MFSD %.4gs   SD-classic %.4gs  |Δμ|=%.2e\n",
                p, t_lr, t_cl, abs(mu_lr - mu_cl))
        append_csv(csv, "p,t_mfsd,t_classic,mu_gap",
                   [(p, t_lr, t_cl, abs(mu_lr - mu_cl))])
    end
    println("[classic-sd] done -> $csv")
end

run_classic_sd()
