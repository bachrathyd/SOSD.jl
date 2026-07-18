# Print the headline numbers for the manuscript from the results CSVs:
# per-target sweet-spot optima, spectral-corner behaviour, non-smooth slopes,
# and beam work-precision crossovers.

using DelimitedFiles
using Printf

const RESULTS = joinpath(@__DIR__, "results")

function cols(path)
    raw, hdr = readdlm(path, ','; header=true)
    Dict(String(h) => raw[:, j] for (j, h) in enumerate(vec(hdr)))
end

fit_slope(xs, ys; lo=1e-14, hi=1e-1) = begin
    idx = findall(i -> lo < ys[i] < hi && isfinite(ys[i]), eachindex(ys))
    length(idx) < 2 ? NaN : ([log10.(Float64.(xs[idx])) ones(length(idx))] \ log10.(Float64.(ys[idx])))[1]
end

println("="^70)
for sys in ("mathieu", "beam")
    path = joinpath(RESULTS, "sweet_spot_$sys.csv")
    isfile(path) || continue
    d = cols(path)
    s = Int.(d["s"]); p = Int.(d["p"]); err = Float64.(d["rel_error"])
    t = Float64.(d["t_mean"]); nnz = Float64.(d["nnz_L"]); bw = Float64.(d["bandwidth_L"])
    println("SWEET SPOT — $sys")
    @printf("%10s  %4s  %6s  %10s  %12s  %8s\n", "target", "s*", "p*", "time [s]", "nnz(L)", "bw(L)")
    for tol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
        best_t = Inf; best = 0
        for i in eachindex(s)
            if err[i] <= tol && t[i] < best_t
                best_t = t[i]; best = i
            end
        end
        best == 0 && (println("  tol=$tol not reached"); continue)
        @printf("%10.0e  %4d  %6d  %10.4f  %12d  %8d\n", tol, s[best], p[best], t[best], nnz[best], bw[best])
    end
    # fill growth: nnz per step at max p for each s
    println("  fill growth (nnz/p): ", join(["s=$sv:" * string(round(Int, maximum(nnz[(s .== sv)] ./ p[(s .== sv)]))) for sv in sort(unique(s))], "  "))
    println()
end

let path = joinpath(RESULTS, "spectral_corner_mathieu.csv")
    if isfile(path)
        d = cols(path)
        println("SPECTRAL CORNER (p=1) — mathieu")
        s = Int.(d["s"]); err = Float64.(d["rel_error"]); t = Float64.(d["t_mean"])
        for i in eachindex(s)
            @printf("  s=%3d  rel_err=%.2e  t=%.4fs\n", s[i], err[i], t[i])
        end
        println()
    end
end

let path = joinpath(RESULTS, "nonsmooth_beam_aaw.csv")
    if isfile(path)
        d = cols(path)
        println("NON-SMOOTH — beam act-and-wait")
        s = Int.(d["s"]); fam = String.(d["family"]); p = Float64.(d["p"]); err = Float64.(d["rel_error"])
        for sv in sort(unique(s)), f in ("aligned", "offgrid")
            idx = findall(i -> s[i] == sv && fam[i] == f, eachindex(s))
            isempty(idx) && continue
            k = -fit_slope(p[idx], err[idx])
            @printf("  GL%d %-8s slope %.2f (nominal %d)\n", sv, f, k, 2sv)
        end
        println()
    end
end

let path = joinpath(RESULTS, "work_precision_beam.csv")
    if isfile(path)
        d = cols(path)
        println("BEAM WP — best time to reach targets, per method")
        m = String.(d["method"]); err = Float64.(d["rel_error"]); t = Float64.(d["t_mean"])
        for tol in (1e-4, 1e-6, 1e-8)
            entries = String[]
            for mm in unique(m)
                idx = findall(i -> m[i] == mm && err[i] <= tol, eachindex(m))
                isempty(idx) && continue
                push!(entries, @sprintf("%s:%.3fs", mm, minimum(t[idx])))
            end
            println("  tol=$(tol):  ", join(entries, "  "))
        end
    end
end
