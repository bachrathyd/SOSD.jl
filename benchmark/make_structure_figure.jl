# Sparsity-pattern figure for the paper: the banded pair (Φ_R, Φ_L) of the
# solution operator on the delayed Mathieu equation at small resolution,
# for an explicit (RK4) and an implicit collocation (GL3) tableau.
# Shows: unit block-lower-triangular Φ_L (trivially solvable by forward
# substitution), the sub-diagonal step chain, and the delay band whose
# per-block fill grows with the stage number.

using SOSD
using SparseArrays
using StaticArrays
using Plots
using LinearAlgebra

ENV["GKSwstype"] = "100"
gr()
default(fontfamily="Computer Modern", titlefontsize=8, guidefontsize=7,
        tickfontsize=6, legendfontsize=6, dpi=300)

const FIGS = joinpath(@__DIR__, "figures")
const PAPER_FIGS = joinpath(@__DIR__, "..", "paper", "figures")
mkpath(FIGS); mkpath(PAPER_FIGS)

function mathieu_prob(; δ=3.0, ε=0.2, b0=-0.15, a1=0.1, T=2π)
    SOSD.LDDEProblem{2, Float64}(
        SOSD.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1]),
        [SOSD.DelayMX(t -> 2π, t -> @SMatrix [0.0 0.0; b0 0.0])],
        SOSD.Additive(t -> @SVector [0.0, sin(4π / T * t)]))
end

function pattern_panel(tab, name; p=8)
    prob = mathieu_prob()
    T = 2π; r = p; D = 2
    S = size(tab.a, 1); BSIZE = (S + 1) * D
    grid = TimeGrid(collect(range(0.0, T, length=p+1)))
    sysm = build_system_matrices(prob, grid, tab, r)
    m = MonodromyMap(prob, grid, tab, sysm, p, r, (r + 1) * BSIZE)
    R, L = build_explicit_matrices(m)

    function spy_plot(M, ttl)
        I_, J_, _ = findnz(M)
        plt = scatter(J_, I_, ms=2.8, msw=0, color=:royalblue, label=false,
                      title=ttl, yflip=true, aspect_ratio=:equal,
                      xlim=(0, size(M, 2) + 1), ylim=(0, size(M, 1) + 1),
                      xlabel="column", ylabel="row")
        # block grid lines
        for b in 0:BSIZE:size(M, 2); vline!(plt, [b + 0.5], lc=:gray85, lw=0.4, label=false); end
        for b in 0:BSIZE:size(M, 1); hline!(plt, [b + 0.5], lc=:gray85, lw=0.4, label=false); end
        return plt
    end
    return spy_plot(L, "Φ_L — $name (nnz = $(nnz(L)))"),
           spy_plot(R, "Φ_R — $name (nnz = $(nnz(R)))")
end

pL1, pR1 = pattern_panel(SOSD.RK4(), "explicit RK4 (s = 4)")
pL2, pR2 = pattern_panel(GL(3), "Gauss GL3 (s = 3)")

plt = plot(pL1, pR1, pL2, pR2, layout=(2, 2), size=(680, 640),
           left_margin=5Plots.mm, bottom_margin=5Plots.mm)
savefig(plt, joinpath(FIGS, "structure_patterns.png"))
savefig(plt, joinpath(FIGS, "structure_patterns.pdf"))
cp(joinpath(FIGS, "structure_patterns.pdf"), joinpath(PAPER_FIGS, "structure_patterns.pdf"); force=true)
println("structure_patterns saved")
