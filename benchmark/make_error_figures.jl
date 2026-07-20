# Figures for the embedded-pair error-estimation validation, from
# results/error_estimation.csv (run_error_estimation.jl).
# Validation-only figures: written to benchmark/figures/, NOT copied to paper/.

using Plots
using DelimitedFiles
using Printf
using Statistics
using LaTeXStrings

ENV["GKSwstype"] = "100"
gr()
default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=10,
        tickfontsize=8, legendfontsize=7, grid=true, minorgrid=false, dpi=300)

const RESULTS = joinpath(@__DIR__, "results")
const FIGS = joinpath(@__DIR__, "figures")
mkpath(FIGS)

function load_csv(path)
    raw, header = readdlm(path, ','; header=true)
    cols = Dict{String, Vector}()
    for (j, name) in enumerate(vec(header))
        cols[String(name)] = raw[:, j]
    end
    return cols
end
fnum(v) = Float64.(v)

function save_local(plt, name)
    savefig(plt, joinpath(FIGS, "$name.png"))
    savefig(plt, joinpath(FIGS, "$name.pdf"))
    println("  -> $name")
end

const MCOLOR = Dict("GL2" => :royalblue, "GL3" => :mediumblue, "GL5" => :navy,
                    "BS3" => :forestgreen, "RK4" => :darkorange)
mcolor(m) = get(MCOLOR, m, :gray)
const SYS_TITLE = Dict("mathieu" => "delayed Mathieu (commensurate)",
                       "bio" => "biological (non-commensurate)",
                       "turning_ssv" => "turning SSV (time-varying delay)")
const SYS_MARK = Dict("mathieu" => :circle, "bio" => :utriangle, "turning_ssv" => :diamond)

d = load_csv(joinpath(RESULTS, "error_estimation.csv"))
sys_v = String.(d["system"]); met_v = String.(d["method"])
p_v = fnum(d["p"]); errt_v = fnum(d["err_true"]); bar_v = fnum(d["bar"])
barQ_v = fnum(d["barQ"]); barI_v = fnum(d["barI"])
rho_v = fnum(d["rho"]); muref_v = fnum(d["mu_ref"])
fixt_v = fnum(d["fix_true"]); fixp_v = fnum(d["fix_pred"])

cap(e) = clamp(e, 1e-16, 3.0)

function fit_slope_window(xs, errs; lo=1e-13, hi=1e-1)
    idx = findall(i -> lo < errs[i] < hi && isfinite(errs[i]) && xs[i] > 0, eachindex(errs))
    length(idx) < 3 && return NaN
    span = log10(maximum(errs[idx])) - log10(minimum(errs[idx]))
    span < 1.0 && return NaN
    A = [log10.(xs[idx]) ones(length(idx))]
    return (A \ log10.(errs[idx]))[1]
end

# ---------------------------------------------------------------------------
# 1. Predicted bar vs true error, per system (one panel per method)
# ---------------------------------------------------------------------------
function fig_prediction(sysname)
    methods = ["GL2", "GL3", "GL5", "BS3", "RK4"]
    panels = []
    for m in methods
        idx = findall(i -> sys_v[i] == sysname && met_v[i] == m, eachindex(sys_v))
        isempty(idx) && continue
        ps = p_v[idx]; et = cap.(errt_v[idx]); bb = cap.(bar_v[idx])
        kt = -fit_slope_window(ps, errt_v[idx]); kb = -fit_slope_window(ps, bar_v[idx])
        plt = plot(xscale=:log10, yscale=:log10, xlabel=L"p", ylabel=L"\varepsilon",
                   title=m, ylims=(1e-16, 3.0), legend=:bottomleft)
        lt = isfinite(kt) ? @sprintf("true (slope %.1f)", kt) : "true"
        lb = isfinite(kb) ? @sprintf("predicted bar (%.1f)", kb) : "predicted bar"
        q = cap.(barQ_v[idx]); qi = cap.(barI_v[idx])
        any(isfinite, barI_v[idx]) &&
            plot!(plt, ps, qi, ls=:dot, lw=1.0, color=:gray55, label=L"|\delta\mu_I|")
        plot!(plt, ps, q, ls=:dot, lw=1.0, color=:gray30, label=L"|\delta\mu_Q|")
        plot!(plt, ps, bb, marker=:square, ms=3, lw=1.8, ls=:dash, color=mcolor(m), label=lb)
        plot!(plt, ps, et, marker=:circle, ms=3, lw=1.8, color=mcolor(m), label=lt)
        push!(panels, plt)
    end
    plt = plot(panels..., layout=(2, 3), size=(1150, 640),
               plot_title="Error prediction — " * SYS_TITLE[sysname], plot_titlefontsize=13,
               left_margin=5Plots.mm, bottom_margin=4Plots.mm)
    save_local(plt, "error_prediction_$(sysname)")
end

# ---------------------------------------------------------------------------
# 2. Coverage: bar / true-error ratio (points >= 1 ⇒ bar contains the root)
# ---------------------------------------------------------------------------
function fig_coverage()
    # "resolved" ⇔ the bar itself is small relative to ρ; a bar above ~10% of ρ
    # flags a below-resolution discretization whose μ AND bar are untrustworthy
    plt = plot(xscale=:log10, yscale=:log10, xlabel=L"p",
               ylabel=L"\Delta_{\mathrm{pred}} / \varepsilon_{\mathrm{true}}",
               title="Coverage of the predicted error bar", legend=:outerright,
               ylims=(1e-3, 1e7), size=(820, 480))
    n_ok = 0; n_tot = 0; n_unres = 0; n_unres_ok = 0
    for sysname in ("mathieu", "bio", "turning_ssv"), m in ("GL2", "GL3", "GL5", "BS3", "RK4")
        idx = findall(i -> sys_v[i] == sysname && met_v[i] == m &&
                           errt_v[i] > 1e-9 && isfinite(bar_v[i]) && bar_v[i] > 0,
                      eachindex(sys_v))
        isempty(idx) && continue
        resolved = [bar_v[i] < 0.1 * abs(rho_v[i]) for i in idx]
        ratio = clamp.(bar_v[idx] ./ errt_v[idx], 1.5e-3, 0.7e7)
        ir = idx[resolved]; iu = idx[.!resolved]
        n_ok += count(>=(1.0), bar_v[ir] ./ errt_v[ir]); n_tot += length(ir)
        n_unres += length(iu); n_unres_ok += count(>=(1.0), bar_v[iu] ./ errt_v[iu])
        scatter!(plt, p_v[ir], ratio[resolved], marker=SYS_MARK[sysname], ms=4.5,
                 color=mcolor(m), msw=0.3, label=(sysname == "mathieu" ? m : ""))
        isempty(iu) || scatter!(plt, p_v[iu], ratio[.!resolved], marker=:xcross, ms=4.5,
                                color=mcolor(m), msw=1.8, label="")
    end
    for (sysname, mk) in SYS_MARK
        scatter!(plt, [NaN], [NaN], marker=mk, color=:gray50, ms=4.5, msw=0.3,
                 label=split(SYS_TITLE[sysname], " (")[1])
    end
    scatter!(plt, [NaN], [NaN], marker=:xcross, color=:gray50, ms=4.5, msw=1.8,
             label=L"unresolved ($\Delta > 0.1\rho$)")
    hline!(plt, [1.0], color=:black, lw=2, ls=:dash, label="ratio = 1")
    annotate!(plt, [(11, 1e6, Plots.text(@sprintf("coverage, resolved points: %d/%d (%.1f%%)",
              n_ok, n_tot, 100n_ok/n_tot), 9, :left)),
                    (11, 2.2e5, Plots.text(@sprintf("unresolved (flagged by the bar itself): %d, %d covered",
              n_unres, n_unres_ok), 8, :left))])
    save_local(plt, "error_coverage")
end

# ---------------------------------------------------------------------------
# 3. The user-visible product: mu with error bars vs the reference
# ---------------------------------------------------------------------------
function fig_bars_demo()
    panels = []
    for (sysname, m) in (("mathieu", "GL2"), ("bio", "GL2"), ("turning_ssv", "GL3"))
        idx = findall(i -> sys_v[i] == sysname && met_v[i] == m, eachindex(sys_v))
        isempty(idx) && continue
        mu_ref = muref_v[idx[1]]
        keep = [i for i in idx if bar_v[i] < 0.3 * abs(rho_v[i])]
        isempty(keep) && (keep = idx)
        ps = p_v[keep]; rh = rho_v[keep]; bb = bar_v[keep]
        half = 1.35 * maximum(abs.(rh .- mu_ref) .+ bb)
        plt = plot(xscale=:log10, xlabel=L"p", ylabel=L"\mu_{\max}",
                   title=SYS_TITLE[sysname] * " — " * m,
                   ylims=(mu_ref - half, mu_ref + half), legend=:topright)
        hline!(plt, [mu_ref], color=:black, ls=:dash, lw=1.6, label=L"\mu_{\mathrm{ref}}")
        scatter!(plt, ps, rh, yerror=bb, marker=:circle, ms=4, color=mcolor(m),
                 msw=1.2, label=L"\mu(p) \pm \Delta_{\mathrm{pred}}")
        push!(panels, plt)
    end
    plt = plot(panels..., layout=(1, length(panels)), size=(1150, 360),
               plot_title="Spectral radius with predicted error bars",
               plot_titlefontsize=13, left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    save_local(plt, "error_bars_demo")
end

# ---------------------------------------------------------------------------
# 4. Fixed-point (periodic solution) error prediction
# ---------------------------------------------------------------------------
function fig_fixpoint()
    panels = []
    for sysname in ("mathieu", "turning_ssv")
        plt = plot(xscale=:log10, yscale=:log10, xlabel=L"p",
                   ylabel=L"\|\,\delta y(0)\,\|", title=SYS_TITLE[sysname],
                   ylims=(1e-16, 3.0), legend=:bottomleft)
        for m in ("GL2", "GL3", "BS3")
            idx = findall(i -> sys_v[i] == sysname && met_v[i] == m &&
                               isfinite(fixt_v[i]) && isfinite(fixp_v[i]), eachindex(sys_v))
            isempty(idx) && continue
            ps = p_v[idx]
            plot!(plt, ps, cap.(fixt_v[idx]), marker=:circle, ms=3, lw=1.8,
                  color=mcolor(m), label="$m true")
            plot!(plt, ps, cap.(2 .* fixp_v[idx]), marker=:square, ms=3, lw=1.8,
                  ls=:dash, color=mcolor(m), label="$m predicted")
        end
        push!(panels, plt)
    end
    plt = plot(panels..., layout=(1, 2), size=(900, 400),
               plot_title="Fixed-point error prediction (node value at t = 0)",
               plot_titlefontsize=13, left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    save_local(plt, "error_fixpoint")
end

fig_prediction("mathieu")
fig_prediction("bio")
fig_prediction("turning_ssv")
fig_coverage()
fig_bars_demo()
fig_fixpoint()
println("[make_error_figures] done.")
