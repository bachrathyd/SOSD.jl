# Generate all paper figures from benchmark/results/*.csv.
# Runs independently of the studies, so plots can be iterated without re-running.
# Outputs PNG (review) + PDF (paper) into benchmark/figures/, PDFs copied to paper/figures/.

using Plots
using DelimitedFiles
using Printf
using Statistics

ENV["GKSwstype"] = "100"
gr()
default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=10,
        tickfontsize=8, legendfontsize=7, grid=true, minorgrid=true, dpi=300)

const RESULTS = joinpath(@__DIR__, "results")
const FIGS = joinpath(@__DIR__, "figures")
const PAPER_FIGS = joinpath(@__DIR__, "..", "paper", "figures")
mkpath(FIGS); mkpath(PAPER_FIGS)

"Load a CSV with a header into a Dict of column name => Vector{String}."
function load_csv(path)
    raw, header = readdlm(path, ','; header=true)
    cols = Dict{String, Vector}()
    for (j, name) in enumerate(vec(header))
        cols[String(name)] = raw[:, j]
    end
    return cols
end

fnum(v) = Float64.(v)

function saveboth(plt, name)
    savefig(plt, joinpath(FIGS, "$name.png"))
    savefig(plt, joinpath(FIGS, "$name.pdf"))
    cp(joinpath(FIGS, "$name.pdf"), joinpath(PAPER_FIGS, "$name.pdf"); force=true)
    println("  -> $name")
end

function fit_slope_window(xs, errs; lo=1e-14, hi=1e-1)
    idx = findall(i -> lo < errs[i] < hi && isfinite(errs[i]) && xs[i] > 0, eachindex(errs))
    length(idx) < 2 && return NaN
    A = [log10.(xs[idx]) ones(length(idx))]
    return (A \ log10.(errs[idx]))[1]
end

const METHOD_COLORS = Dict(
    "SDM-O2" => :black, "RK1" => :gray60, "RK2" => :gray40,
    "RK4" => :darkorange, "RK5" => :chocolate,
    "GL1" => :steelblue, "GL2" => :royalblue, "GL3" => :mediumblue,
    "GL5" => :navy, "GL8" => :indigo)
mcolor(m) = get(METHOD_COLORS, m, :auto)
mstyle(m) = startswith(m, "GL") ? :solid : (m == "SDM-O2" ? :solid : :dash)
mwidth(m) = m == "SDM-O2" ? 3.0 : 1.8

# ---------------------------------------------------------------------------
# 1. Order verification
# ---------------------------------------------------------------------------
function fig_order()
    path = joinpath(RESULTS, "order_verification_mathieu.csv")
    isfile(path) || (println("skip fig_order"); return)
    d = load_csv(path)
    methods = unique(String.(d["method"]))
    plt = plot(xscale=:log10, yscale=:log10, xlabel="resolution  p",
               ylabel="|Δμ| (dominant multiplier error)", legend=:bottomleft,
               title="Order verification — delayed Mathieu")
    for m in methods
        idx = findall(==(m), String.(d["method"]))
        ps = fnum(d["p"][idx]); errs = max.(fnum(d["abs_error"][idx]), 1e-17)
        k = -fit_slope_window(ps, errs)
        lbl = isfinite(k) ? "$m (slope $(round(k, digits=2)))" : m
        plot!(plt, ps, errs, marker=:circle, ms=3, lw=mwidth(m), ls=mstyle(m),
              color=mcolor(m), label=lbl)
    end
    saveboth(plt, "order_verification")
end

# ---------------------------------------------------------------------------
# 2. Work-precision (per system): p-err | p-time(+ribbon,slope) | time-err
# ---------------------------------------------------------------------------
function fig_work_precision(sysname)
    path = joinpath(RESULTS, "work_precision_$sysname.csv")
    isfile(path) || (println("skip wp $sysname"); return)
    d = load_csv(path)
    methods = unique(String.(d["method"]))
    p1 = plot(xscale=:log10, yscale=:log10, xlabel="p", ylabel="relative error", legend=:bottomleft)
    p2 = plot(xscale=:log10, yscale=:log10, xlabel="p", ylabel="CPU time [s]", legend=:topleft)
    p3 = plot(xscale=:log10, yscale=:log10, xlabel="CPU time [s]", ylabel="relative error", legend=:bottomleft)
    for m in methods
        idx = findall(==(m), String.(d["method"]))
        ps = fnum(d["p"][idx]); errs = max.(fnum(d["rel_error"][idx]), 1e-16)
        tm = fnum(d["t_mean"][idx]); ts = fnum(d["t_std"][idx])
        k = -fit_slope_window(ps, errs)
        lble = isfinite(k) ? "$m (k=$(round(k, digits=1)))" : m
        # time-scaling exponent from the upper half of the p range
        half = ps .>= sqrt(maximum(ps) * minimum(ps))
        kt = sum(half) > 1 ? ([log10.(ps[half]) ones(sum(half))] \ log10.(tm[half]))[1] : NaN
        lblt = isfinite(kt) ? "$m (t~p^$(round(kt, digits=1)))" : m
        plot!(p1, ps, errs, marker=:circle, ms=2.5, lw=mwidth(m), ls=mstyle(m), color=mcolor(m), label=lble)
        plot!(p2, ps, tm, ribbon=ts, fillalpha=0.25, marker=:circle, ms=2.5,
              lw=mwidth(m), ls=mstyle(m), color=mcolor(m), label=lblt)
        plot!(p3, tm, errs, marker=:circle, ms=2.5, lw=mwidth(m), ls=mstyle(m), color=mcolor(m), label=lble)
    end
    plt = plot(p1, p2, p3, layout=(1, 3), size=(1500, 460),
               plot_title="Work-precision — $sysname", left_margin=8Plots.mm, bottom_margin=8Plots.mm)
    saveboth(plt, "work_precision_$sysname")
end

# ---------------------------------------------------------------------------
# 3. Sweet spot: U-curves + structure panel
# ---------------------------------------------------------------------------
function fig_sweet_spot(sysname; tols=[1e-4, 1e-6, 1e-8, 1e-10])
    path = joinpath(RESULTS, "sweet_spot_$sysname.csv")
    isfile(path) || (println("skip sweetspot $sysname"); return)
    d = load_csv(path)
    ss = Int.(d["s"]); ps = Int.(d["p"]); errs = fnum(d["rel_error"])
    tms = fnum(d["t_mean"]); nnzs = fnum(d["nnz_L"]); bws = fnum(d["bandwidth_L"])
    svals = sort(unique(ss))

    pU = plot(xlabel="Gauss stages  s  (order 2s)", ylabel="CPU time to target [s]",
              yscale=:log10, legend=:topright, title="Sweet spot — $sysname")
    for tol in tols
        xs = Int[]; ys = Float64[]; popt = Int[]
        for s in svals
            idx = findall(i -> ss[i] == s && errs[i] <= tol, eachindex(ss))
            isempty(idx) && continue
            best = idx[argmin(tms[idx])]
            push!(xs, s); push!(ys, tms[best]); push!(popt, ps[best])
        end
        isempty(xs) && continue
        lbl = @sprintf("ε = %.0e", tol)
        plot!(pU, xs, ys, marker=:circle, ms=4, lw=2, label=lbl)
        # annotate the optimum
        i0 = argmin(ys)
        annotate!(pU, [(xs[i0], ys[i0] * 0.7, Plots.text("p=$(popt[i0])", 6, :gray30))])
    end

    pS = plot(xlabel="Gauss stages  s", ylabel="nnz(Φ_L) per step", yscale=:log10,
              legend=:topleft, title="Fill growth — $sysname")
    nnz_per_step = Float64[]; bw_s = Float64[]
    for s in svals
        idx = findall(==(s), ss)
        # nnz per step is p-independent: take the largest-p entry
        best = idx[argmax(ps[idx])]
        push!(nnz_per_step, nnzs[best] / ps[best])
        push!(bw_s, bws[best])
    end
    plot!(pS, svals, nnz_per_step, marker=:circle, ms=4, lw=2, color=:royalblue, label="nnz/p (measured)")
    plot!(pS, svals, nnz_per_step[end] .* (svals ./ svals[end]).^3, ls=:dash, color=:gray,
          label="∝ s³")
    plot!(twinx(pS), svals, bw_s, marker=:diamond, ms=3, lw=1.5, color=:darkred,
          yscale=:log10, ylabel="bandwidth(Φ_L)", label="bandwidth", legend=:bottomright)

    plt = plot(pU, pS, layout=(1, 2), size=(1100, 440), left_margin=8Plots.mm, bottom_margin=8Plots.mm)
    saveboth(plt, "sweet_spot_$sysname")
end

# ---------------------------------------------------------------------------
# 4. Spectral corner (single step): error & time vs s
# ---------------------------------------------------------------------------
function fig_spectral_corner()
    path = joinpath(RESULTS, "spectral_corner_mathieu.csv")
    isfile(path) || (println("skip spectral corner"); return)
    d = load_csv(path)
    ss = Int.(d["s"]); errs = max.(fnum(d["rel_error"]), 1e-16); tms = fnum(d["t_mean"])
    p1 = plot(ss, errs, yscale=:log10, xlabel="stages s (single step, p = 1)",
              ylabel="relative error", marker=:circle, lw=2, color=:mediumblue,
              label="error", title="Single-step (global collocation) limit")
    p2 = plot(ss, tms, yscale=:log10, xlabel="stages s", ylabel="CPU time [s]",
              marker=:circle, lw=2, color=:darkred, label="time")
    plt = plot(p1, p2, layout=(1, 2), size=(1000, 420), left_margin=8Plots.mm, bottom_margin=8Plots.mm)
    saveboth(plt, "spectral_corner")
end

# ---------------------------------------------------------------------------
# 5. Non-smoothness stress test
# ---------------------------------------------------------------------------
function fig_nonsmooth()
    path = joinpath(RESULTS, "nonsmooth_beam_aaw.csv")
    isfile(path) || (println("skip nonsmooth"); return)
    d = load_csv(path)
    ss = Int.(d["s"]); fams = String.(d["family"]); ps = fnum(d["p"]); errs = max.(fnum(d["rel_error"]), 1e-16)
    plt = plot(xscale=:log10, yscale=:log10, xlabel="p", ylabel="relative error",
               legend=:bottomleft, title="Act-and-wait beam: grid-aligned vs off-grid switching")
    colors = Dict(1 => :steelblue, 2 => :royalblue, 3 => :navy)
    for s in sort(unique(ss)), fam in ("aligned", "offgrid")
        idx = findall(i -> ss[i] == s && fams[i] == fam, eachindex(ss))
        isempty(idx) && continue
        k = -fit_slope_window(ps[idx], errs[idx])
        lbl = "GL$s $(fam) (k=$(round(k, digits=1)))"
        plot!(plt, ps[idx], errs[idx], marker=fam == "aligned" ? :circle : :xcross,
              ms=4, lw=fam == "aligned" ? 2 : 1.5, ls=fam == "aligned" ? :solid : :dash,
              color=colors[s], label=lbl)
    end
    saveboth(plt, "nonsmooth_beam")
end

# ---------------------------------------------------------------------------
println("Generating figures...")
fig_order()
for s in ("mathieu", "bio", "turning_ssv", "beam")
    fig_work_precision(s)
end
fig_sweet_spot("mathieu")
fig_sweet_spot("beam")
fig_spectral_corner()
fig_nonsmooth()
println("Done.")
