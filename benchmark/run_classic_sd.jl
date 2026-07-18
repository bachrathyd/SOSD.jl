# Baseline parity with the predecessor (handoff §10, item 1):
# reproduce the SD-classic (traditional monodromy build, ~O(p^2.8)) versus
# MFSD (banded LR mapping, O(p^1)) time-complexity contrast on the delayed
# Mathieu equation, in the same environment as all other measurements.

(@isdefined SOSD_HARNESS_LOADED) || (include(joinpath(@__DIR__, "harness.jl")); SOSD_HARNESS_LOADED = true)

function run_classic_sd()
    sys = make_mathieu()
    csv = joinpath(RESULTS_DIR, "classic_sd_vs_mfsd.csv")
    isfile(csv) && rm(csv)

    ps = unique(round.(Int, 10 .^ (1.0:0.25:4.0)))
    for p in ps
        h = sys.T / p
        method = SemiDiscretization(2, h)

        # MFSD (banded LR) — linear complexity
        mu_lr, t_lr, s_lr, n_lr = time_stats(
            () -> spectralRadiusOfMapping(DiscreteMapping_LR(sys.sdm_prob, method, sys.taumax, n_steps=p));
            budget=2.0, min_reps=3, max_reps=8)

        # SD-classic (full one-period mapping) — polynomial complexity; cap runtime
        t_cl = NaN; s_cl = 0.0; n_cl = 0; mu_cl = NaN
        if p <= 3000
            try
                mu_cl, t_cl, s_cl, n_cl = time_stats(
                    () -> spectralRadiusOfMapping(DiscreteMapping(sys.sdm_prob, method, sys.taumax, n_steps=p));
                    budget=2.0, min_reps=3, max_reps=8)
            catch e
                @warn "SD-classic failed at p=$p" exception=(e,)
            end
        end

        @printf("p=%5d  MFSD %.4gs(±%.1g)   SD-classic %.4gs(±%.1g)  |Δμ|=%.2e\n",
                p, t_lr, s_lr, t_cl, s_cl, abs(mu_lr - mu_cl))
        append_csv(csv, "p,t_mfsd,t_mfsd_std,n_mfsd,t_classic,t_classic_std,n_classic,mu_gap",
                   [(p, t_lr, s_lr, n_lr, t_cl, s_cl, n_cl, abs(mu_lr - mu_cl))])
        (isfinite(t_cl) && t_cl > 60) && break
    end
    println("[classic-sd] done -> $csv")
end

run_classic_sd()
