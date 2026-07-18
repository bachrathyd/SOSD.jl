# Order verification for the other collocation families (handoff §3.2):
# Radau IIA (order 2s-1) and Lobatto IIIA (order 2s-2) on the delayed Mathieu
# equation. Complements run_order_verification.jl (Gauss + explicit families).

(@isdefined SOSD_HARNESS_LOADED) || (include(joinpath(@__DIR__, "harness.jl")); SOSD_HARNESS_LOADED = true)
using SOSD.RungeKutta

function run_order_implicit_families()
    sys = make_mathieu()
    mu_ref = reference_mu(sys)
    csv = joinpath(RESULTS_DIR, "order_verification_implicit_families.csv")
    isfile(csv) && rm(csv)

    families = [
        ("RadauIIA-2",   SOSD.from_rkjl(TableauRadauIIA(2)),   3),
        ("RadauIIA-3",   SOSD.from_rkjl(TableauRadauIIA(3)),   5),
        ("LobattoIIIA-3", SOSD.from_rkjl(TableauLobattoIIIA(3)), 4),
        ("LobattoIIIA-4", SOSD.from_rkjl(TableauLobattoIIIA(4)), 6),
    ]
    for (label, tab, nominal) in families
        errs = Float64[]; used = Int[]
        for p in unique(round.(Int, 2 .^ (3:0.5:9)))
            mu = try
                sosd_mu(sys, p, tab)
            catch e
                @warn "$label failed at p=$p"; break
            end
            rel = abs(mu - mu_ref) / abs(mu_ref)
            push!(errs, rel); push!(used, p)
            append_csv(csv, "method,nominal_order,p,rel_error", [(label, nominal, p, rel)])
            rel < 1e-14 && length(errs) >= 4 && break
        end
        k = fit_slope(used, errs; lo=1e-14, hi=1e-1)
        @printf("%-13s nominal %d  fitted slope %.2f\n", label, nominal, -k)
    end
    println("[order-implicit] done -> $csv")
end

run_order_implicit_families()
