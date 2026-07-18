# Brute-force stability charts at sweet-spot-optimal settings, demonstrating
# the wall-clock cost of HIGH-RESOLUTION charts:
#   - delayed Mathieu: (delta, b) plane, 301 x 301, fixed (s,p) = (7,2)
#   - SSV turning (intricate parameters: zeta = 0.02, A_SSV = 0.3):
#     (Omega, kw) plane, 151 x 151, s = 4 with p chosen PER POINT from the
#     Shannon recipe p = ceil(N_min/(s+1)), N_min ~ 7 * NT * omega_max / Omega.
# Colour: log10 spectral radius, blue = stable, red = unstable; black: rho = 1.
# The measured total CPU time is printed in each chart title and stored in
# paper/figures/chart_times.tex; raw mu-grids are cached in results/.

include(joinpath(@__DIR__, "harness.jl"))
SOSD_HARNESS_LOADED = true
using Plots
using DelimitedFiles
using LaTeXStrings
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

using MDBM

"p may be an Int or a function of the x-coordinate (adaptive resolution)."
function chart(make_sys, xs, ys, s, p, name, xlab, ylab, ttl_base)
    Rfile = joinpath(RESULTS_DIR, "chart_$(name)_R.csv")
    Mfile = joinpath(RESULTS_DIR, "chart_$(name)_meta.csv")
    p_of = p isa Function ? p : (_ -> p)
    tab = GL(s)
    local R, wall
    if isfile(Rfile) && isfile(Mfile)
        R = readdlm(Rfile, ',')
        meta = readdlm(Mfile, ',')
        wall = Float64(meta[1])
        println("$name: replotting heatmap from cached data")
    else
        mu_point(make_sys(xs[1], ys[1]), p_of(xs[1]), tab)   # warm-up (compile)
        R = Matrix{Float64}(undef, length(ys), length(xs))
        # multithreaded over columns: every grid point is independent, and each
        # sosd_mu call builds its own matrices (no shared mutable state);
        # BLAS stays at 1 thread so there is no oversubscription
        wall = @elapsed Threads.@threads for j in eachindex(xs)
            x = xs[j]
            pj = p_of(x)
            for (i, y) in enumerate(ys)
                R[i, j] = mu_point(make_sys(x, y), pj, tab)
            end
        end
        writedlm(Rfile, R, ',')
        writedlm(Mfile, [wall length(xs)*length(ys)], ',')
        @printf("%s: %d points in %.1f s wall on %d threads (%.2f ms/point/thread)\n", name,
                length(xs)*length(ys), wall, Threads.nthreads(),
                1000wall*Threads.nthreads()/(length(xs)*length(ys)))
    end
    n = length(xs) * length(ys)

    # --- MDBM-refined stability boundary (interpolated, higher-order) -------
    # the ::Float64 declaration is required: MDBM types its containers from the
    # inferred return type, and the try/catch inside mu_point widens it to Any
    foo(x, y)::Float64 = log10(max(mu_point(make_sys(x, y), p_of(x), tab), 1e-12))
    ax = [MDBM.Axis(range(xs[1], xs[end], length=13), :x),
          MDBM.Axis(range(ys[1], ys[end], length=13), :y)]
    mdbm_prob = MDBM_Problem(foo, ax)
    t_mdbm = @elapsed MDBM.solve!(mdbm_prob, 4)
    bp = getinterpolatedsolution(mdbm_prob)
    @printf("%s: MDBM boundary in %.1f s (%d points)\n", name, t_mdbm, length(bp[1]))

    L = log10.(max.(R, 1e-12))
    ttl = @sprintf("%s — map %.1f s wall (%d thr) + MDBM %.1f s", ttl_base, wall,
                   Threads.nthreads(), t_mdbm)
    # reversed RdBu: blue = stable (rho < 1), red = unstable
    plt = heatmap(xs, ys, L, c=cgrad(:RdBu, rev=true), colorbar_title=L"\log_{10}\rho",
                  xlabel=xlab, ylabel=ylab, title=ttl, clims=(-2, 2))
    scatter!(plt, bp[1], bp[2], ms=1.6, msw=0, color=:black,
             label=L"\rho = 1" * " (MDBM)", legend=:topright)
    for ext in ("png", "pdf")
        savefig(plt, joinpath(@__DIR__, "figures", "stability_chart_$name.$ext"))
    end
    cp(joinpath(@__DIR__, "figures", "stability_chart_$name.pdf"),
       joinpath(PAPER_FIGS, "stability_chart_$name.pdf"); force=true)
    return wall, n, t_mdbm
end

# --- delayed Mathieu: (delta, b), eps = 1, optimal (7, 2) -------------------
w1, n1, m1 = chart((δ, b) -> make_mathieu(δ=δ, ε=1.0, b0=b, a1=0.1),
               collect(range(-1.0, 5.0, length=100)),
               collect(range(-2.0, 1.5, length=100)),
               7, 2, "mathieu", L"\delta", L"b",
               "Delayed Mathieu, GL7, p=2")

# --- SSV turning, intricate: zeta=0.02, A_SSV=0.3, adaptive p(Omega) --------
p_ssv(Ω) = clamp(ceil(Int, 16.0 / Ω), 60, 400)
w2, n2, m2 = chart((Ω, kw) -> make_turning_ssv(kw=kw, Ω=Ω, ζ=0.02, ASSV=0.3),
               collect(range(0.08, 0.40, length=100)),
               collect(range(0.001, 0.40, length=100)),
               4, p_ssv, "turning", L"\Omega", L"k_\mathrm{w}",
               "SSV turning (ζ=0.02, A=0.3), GL4, p=p(Ω)")

open(joinpath(PAPER_FIGS, "chart_times.tex"), "w") do io
    @printf(io, "\\newcommand{\\chartTimeMathieu}{%.0f}\n", w1)
    @printf(io, "\\newcommand{\\chartPtsMathieu}{%d}\n", n1)
    @printf(io, "\\newcommand{\\chartMsMathieu}{%.1f}\n", 1000w1/n1)
    @printf(io, "\\newcommand{\\chartMdbmMathieu}{%.0f}\n", m1)
    @printf(io, "\\newcommand{\\chartTimeTurning}{%.0f}\n", w2)
    @printf(io, "\\newcommand{\\chartPtsTurning}{%d}\n", n2)
    @printf(io, "\\newcommand{\\chartMsTurning}{%.1f}\n", 1000w2/n2)
    @printf(io, "\\newcommand{\\chartMdbmTurning}{%.0f}\n", m2)
end
println("STABILITY CHARTS DONE")
