# Order verification (handoff §4.3): eigenvalue error vs p on the delayed
# Mathieu equation, one curve per method, fitted log-log slopes.
# Verifies that each integrator achieves its theoretical convergence order and
# that the delay interpolation does not cap it.

(@isdefined MFCM_HARNESS_LOADED) || (include(joinpath(@__DIR__, "harness.jl")); MFCM_HARNESS_LOADED = true)

function run_order_verification()
    sys = make_mathieu()
    mu_ref = reference_mu(sys)

    ps = unique(round.(Int, 2 .^ (3:0.5:11)))
    csv = joinpath(RESULTS_DIR, "order_verification_mathieu.csv")
    isfile(csv) && rm(csv)

    methods = method_set()
    slopes = Dict{String, Float64}()

    for (label, tab, nominal) in methods
        errs = Float64[]
        used_p = Int[]
        for p in ps
            mu = NaN
            try
                mu = tab === :sdm2 ? sdm_mu(sys, p; order=2) : mfcm_mu(sys, p, tab)
            catch e
                @warn "order-verification $label failed at p=$p" exception=e
                break
            end
            err = abs(mu - mu_ref)
            push!(errs, err); push!(used_p, p)
            # stop once safely below the noise floor (keep 2 floor points)
            if length(errs) >= 3 && all(e -> e < 1e-14, errs[end-1:end])
                break
            end
        end
        k = fit_slope(used_p, errs)
        slopes[label] = k
        @printf("%-8s nominal order %2d  fitted slope %5.2f\n", label, nominal, -k)
        append_csv(csv, "method,nominal_order,p,abs_error",
                   [(label, nominal, used_p[i], errs[i]) for i in eachindex(used_p)])
    end

    open(joinpath(RESULTS_DIR, "order_verification_slopes.csv"), "w") do io
        println(io, "method,nominal_order,fitted_slope")
        for (label, _, nominal) in methods
            @printf(io, "%s,%d,%.3f\n", label, nominal, -get(slopes, label, NaN))
        end
    end
    println("[order-verification] done -> $csv")
end

run_order_verification()
