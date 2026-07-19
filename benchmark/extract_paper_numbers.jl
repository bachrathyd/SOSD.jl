# Single source of truth: extract every number quoted in the manuscript from
# the FINAL results CSVs. Run after any benchmark rerun; update the paper from
# this output only.

using DelimitedFiles, Printf, Statistics

const R = joinpath(@__DIR__, "results")

cols(f) = begin
    raw, hdr = readdlm(joinpath(R, f), ','; header=true)
    Dict(String(h) => raw[:, j] for (j, h) in enumerate(vec(hdr)))
end
fitp(x, y) = ([log10.(Float64.(x)) ones(length(x))] \ log10.(Float64.(y)))[1]

println("="^72)
println("CLASSIC PARITY (classic_sd_vs_mfsd.csv)")
let d = cols("classic_sd_vs_mfsd.csv")
    p = Float64.(d["p"]); tl = Float64.(d["t_mfsd"]); tc = Float64.(d["t_classic"])
    ok = isfinite.(tc)
    @printf("  SD-classic exponent (all valid pts): %.2f\n", fitp(p[ok], tc[ok]))
    @printf("  MFSD exponent (p >= 100):            %.2f\n",
            fitp(p[p .>= 100], tl[p .>= 100]))
    ilast = findlast(ok)
    @printf("  last classic point: p=%d  speedup vs MFSD there: %.0fx\n",
            Int(p[ilast]), tc[ilast]/tl[ilast])
    @printf("  MFSD at p=1e6: %.1f s\n", tl[end])
end

println("\nWP H-EXPONENTS (upper half of p-range, t_min)")
for sys in ("mathieu", "bio", "turning_ssv", "beam")
    d = cols("work_precision_$sys.csv")
    m = String.(d["method"]); p = Float64.(d["p"]); t = Float64.(d["t_min"])
    ks = Float64[]
    for mm in unique(m)
        idx = findall(==(mm), m)
        length(idx) < 5 && continue
        pp = p[idx]; tt = t[idx]
        half = pp .>= sqrt(maximum(pp)*minimum(pp))
        sum(half) > 2 && push!(ks, fitp(pp[half], tt[half]))
    end
    @printf("  %-12s exponents: min %.2f  median %.2f  max %.2f\n",
            sys, minimum(ks), median(ks), maximum(ks))
end

println("\nBEAM WP CROSSOVER (min time to reach target)")
let d = cols("work_precision_beam.csv")
    m = String.(d["method"]); rel = Float64.(d["rel_error"]); t = Float64.(d["t_min"])
    for tol in (1e-4, 1e-6, 1e-8)
        entries = String[]
        for mm in ("SDM-O2", "RK4", "RK5", "GL2", "GL3")
            idx = findall(i -> m[i]==mm && rel[i] <= tol, eachindex(m))
            isempty(idx) && continue
            push!(entries, @sprintf("%s %.3fs", mm, minimum(t[idx])))
        end
        println("  tol=$(tol): ", join(entries, " | "))
    end
end

println("\nSWEET-SPOT OPTIMA (per target, min t_mean cell)")
for sys in ("mathieu", "turning_ssv", "beam")
    f = "sweet_spot_$sys.csv"
    isfile(joinpath(R, f)) || continue
    d = cols(f)
    s = Int.(d["s"]); p = Int.(d["p"]); rel = Float64.(d["rel_error"]); t = Float64.(d["t_mean"])
    keep = s .<= 30
    s = s[keep]; p = p[keep]; rel = rel[keep]; t = t[keep]
    print("  $sys: ")
    for tol in (1e-4, 1e-6, 1e-8, 1e-12)
        bi = 0; bt = Inf
        for i in eachindex(s)
            rel[i] <= tol && t[i] < bt && (bt = t[i]; bi = i)
        end
        bi == 0 ? print("[$(tol): -] ") :
                  @printf("[%.0e: s=%d p=%d %.3fs] ", tol, s[bi], p[bi], bt)
    end
    println()
end

println("\nS-LAW AT FIXED p (largest p with >=5 s-values, s<=30)")
for sys in ("mathieu", "turning_ssv", "beam")
    f = "sweet_spot_$sys.csv"
    isfile(joinpath(R, f)) || continue
    d = cols(f)
    s = Int.(d["s"]); p = Int.(d["p"]); t = Float64.(d["t_mean"])
    keep = s .<= 30; s = s[keep]; p = p[keep]; t = t[keep]
    for pv in sort(unique(p), rev=true)
        idx = findall(==(pv), p)
        if length(idx) >= 5
            @printf("  %-12s p=%5d: t ~ s^%.2f\n", sys, pv, fitp(s[idx], t[idx]))
            break
        end
    end
end

println("\nS-COMPLEXITY (s_complexity CSVs, if present)")
for sys in ("mathieu", "turning_ssv")
    f = "s_complexity_$sys.csv"
    isfile(joinpath(R, f)) || continue
    d = cols(f)
    p = Int.(d["p"]); s = Int.(d["s"]); t = Float64.(d["t_min"])
    for pv in sort(unique(p))
        idx = findall(i -> p[i]==pv && s[i] >= 8, eachindex(p))
        length(idx) < 3 && continue
        @printf("  %-12s p=%3d: large-s CPU law t ~ s^%.2f\n", sys, pv, fitp(s[idx], t[idx]))
    end
end
println("="^72)
