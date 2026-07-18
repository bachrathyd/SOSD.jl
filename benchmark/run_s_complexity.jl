# s-complexity study (Fig-6 redesign): at FIXED step counts p = 1, 10, 100,
# sweep the stage number s continuously up to 100 and record BOTH the
# eigenvalue error and the (minimum-of-repeats) CPU time, for the easy
# (Mathieu) and hard (turning SSV) systems. This shows (a) the
# superconvergence cliff position scaling with the resolution demand and
# (b) the empirical CPU-vs-s complexity exponent, including where the
# O((sd)^3) stage-elimination term takes over.

include(joinpath(@__DIR__, "harness.jl"))
SOSD_HARNESS_LOADED = true

function run_s_complexity(sys::BenchSystem; time_cap=5.0)
    mu_ref = reference_mu(sys)
    csv = joinpath(RESULTS_DIR, "s_complexity_$(sys.name).csv")
    isfile(csv) && rm(csv)
    svals = vcat(1:20, 22:2:40, 45:5:100)
    for p in (1, 10, 100)
        println("\n[$(sys.name)] fixed p = $p")
        for s in svals
            local mu, tmin
            try
                tab = GL(s)
                mu, tmean, tstd, nrep, tmin = time_stats(() -> sosd_mu(sys, p, tab);
                                                         budget=1.0, min_reps=3, max_reps=7)
            catch e
                println("  s=$s FAILED: ", sprint(showerror, e)[1:min(end,120)])
                continue
            end
            rel = abs(mu - mu_ref) / abs(mu_ref)
            N = p * (s + 1)
            @printf("  s=%3d N=%5d rel_err=%.2e t_min=%.4gs\n", s, N, rel, tmin)
            append_csv(csv, "p,s,order,N,rel_error,t_min",
                       [(p, s, 2s, N, rel, tmin)])
            tmin > time_cap && (println("  (cap)"); break)
        end
    end
    println("[s-complexity] $(sys.name) done -> $csv")
end

run_s_complexity(make_mathieu())
run_s_complexity(make_turning_ssv())
println("S COMPLEXITY DONE")
