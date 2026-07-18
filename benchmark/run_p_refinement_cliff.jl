# p-refinement cliff study (Daniel's superconvergence caveat):
# error versus stage number s AT FIXED step count p, for the easy (Mathieu)
# and hard (turning SSV) systems. Superconvergence means the error falls
# steeply with s — but only AFTER the total sample count N = p*(s+1) crosses
# the Shannon-type resolution bound N ~ c * omega_max*T/(2pi). The cliff
# position therefore scales with the natural-oscillation content of the
# system: "a handful of stages" is enough only for easy systems.

(@isdefined SOSD_HARNESS_LOADED) || (include(joinpath(@__DIR__, "harness.jl")); SOSD_HARNESS_LOADED = true)

function run_cliff(sys::BenchSystem, p_list::Vector{Int}, s_list::Vector{Int}; time_cap=4.0)
    mu_ref = reference_mu(sys)
    csv = joinpath(RESULTS_DIR, "p_refinement_cliff_$(sys.name).csv")
    isfile(csv) && rm(csv)
    for p in p_list
        println("\n[$(sys.name)] fixed p = $p")
        for s in s_list
            print("  s=$s ")
            local mu, tm
            try
                tm = @elapsed mu = sosd_mu(sys, p, GL(s))
            catch e
                println("FAILED: ", sprint(showerror, e)[1:min(end,120)])
                continue
            end
            rel_err = abs(mu - mu_ref) / abs(mu_ref)
            N = p * (s + 1)
            @printf("N=%d rel_err=%.2e t=%.3gs\n", N, rel_err, tm)
            append_csv(csv, "p,s,order,N,rel_error,t",
                       [(p, s, 2s, N, rel_err, tm)])
            tm > time_cap && (println("  (time cap)"); break)
        end
    end
    println("[cliff] $(sys.name) done -> $csv")
end

# easy: cliff expected at tiny s even for p = 1..4
run_cliff(make_mathieu(), [1, 2, 4, 8], collect(1:24))

# hard: N must cross ~200-400 => cliff at s ~ N_min/p; single-step corner
# would need s ~ hundreds (uncomputable within budget -> visible as absence)
run_cliff(make_turning_ssv(), [10, 25, 50, 100, 200], vcat(1:12, 14:2:30, 34:4:50))
