# Brute-force stability charts at the sweet-spot-optimal (s*, p*) settings,
# demonstrating the wall-clock cost of a HIGH-RESOLUTION chart:
#   - delayed Mathieu: (delta, b) plane, 301 x 301 points, optimal (s,p) = (7,2)
#   - SSV turning: (Omega, kw) plane, 151 x 151 points, optimal (s,p) = (4,158)
# Colour: log10 of the spectral radius; black contour: the stability boundary
# rho = 1. Total wall time is written into paper/figures/chart_times.tex.

include(joinpath(@__DIR__, "harness.jl"))
SOSD_HARNESS_LOADED = true
using Plots
ENV["GKSwstype"] = "100"
gr()
default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=10,
        tickfontsize=8, legendfontsize=8, dpi=300)

const PAPER_FIGS = joinpath(@__DIR__, "..", "paper", "figures")
mkpath(PAPER_FIGS)

function mu_point(sys::BenchSystem, p::Int, tab)
    try
        return sosd_mu(sys, p, tab; tol=1e-8)
    catch
        return NaN
    end
end

function chart(make_sys, xs, ys, s, p, name, xlab, ylab, ttl)
    tab = GL(s)
    # warm-up (compile) outside the timed region
    mu_point(make_sys(xs[1], ys[1]), p, tab)
    R = Matrix{Float64}(undef, length(ys), length(xs))
    wall = @elapsed for (j, x) in enumerate(xs), (i, y) in enumerate(ys)
        R[i, j] = mu_point(make_sys(x, y), p, tab)
    end
    n = length(xs) * length(ys)
    @printf("%s: %d points in %.1f s  (%.2f ms/point)\n", name, n, wall, 1000wall/n)

    L = log10.(max.(R, 1e-12))
    plt = heatmap(xs, ys, L, c=:RdBu, colorbar_title="\nlog10 rho(mu)",
                  xlabel=xlab, ylabel=ylab, title=ttl, clims=(-2, 2))
    contour!(plt, xs, ys, L, levels=[0.0], lc=:black, lw=2.5, cbar=false)
    for ext in ("png", "pdf")
        savefig(plt, joinpath(@__DIR__, "figures", "stability_chart_$name.$ext"))
    end
    cp(joinpath(@__DIR__, "figures", "stability_chart_$name.pdf"),
       joinpath(PAPER_FIGS, "stability_chart_$name.pdf"); force=true)
    return wall, n
end

# --- delayed Mathieu: (delta, b) plane, eps = 1, optimal (7, 2) -------------
w1, n1 = chart((δ, b) -> make_mathieu(δ=δ, ε=1.0, b0=b, a1=0.1),
               collect(range(-1.0, 5.0, length=301)),
               collect(range(-2.0, 1.5, length=301)),
               7, 2, "mathieu", "delta", "b",
               "Delayed Mathieu — 301x301, GL7, p=2")

# --- SSV turning: (Omega, kw) plane, optimal (4, 158) -----------------------
w2, n2 = chart((Ω, kw) -> make_turning_ssv(kw=kw, Ω=Ω),
               collect(range(0.15, 0.55, length=151)),
               collect(range(0.001, 0.35, length=151)),
               4, 158, "turning", "Omega", "k_w",
               "SSV turning — 151x151, GL4, p=158")

open(joinpath(PAPER_FIGS, "chart_times.tex"), "w") do io
    @printf(io, "\\newcommand{\\chartTimeMathieu}{%.0f}\n", w1)
    @printf(io, "\\newcommand{\\chartPtsMathieu}{%d}\n", n1)
    @printf(io, "\\newcommand{\\chartMsMathieu}{%.1f}\n", 1000w1/n1)
    @printf(io, "\\newcommand{\\chartTimeTurning}{%.0f}\n", w2)
    @printf(io, "\\newcommand{\\chartPtsTurning}{%d}\n", n2)
    @printf(io, "\\newcommand{\\chartMsTurning}{%.1f}\n", 1000w2/n2)
end
println("STABILITY CHARTS DONE")
