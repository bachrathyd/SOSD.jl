# Stability chart with the embedded-pair error estimation as a TRUST MAP:
# delayed Mathieu (delta, b0) plane at deliberately coarse resolution (GL2,
# small p). Yellow overlay = points whose predicted bar straddles rho = 1
# ("stability undecided at this resolution"); black = the rho = 1 boundary
# from the fine GL7 reference chart (cached chart_mathieu_R.csv). Refining p
# shrinks the undecided band around the true boundary.
#
# Multithreaded (chart convention): run with  julia -t 16 --project=benchmark

include(joinpath(@__DIR__, "harness.jl"))
using Plots
using DelimitedFiles
using LaTeXStrings
using Printf
ENV["GKSwstype"] = "100"
gr()
default(fontfamily="Computer Modern", titlefontsize=10, guidefontsize=10,
        tickfontsize=8, legendfontsize=7, dpi=300)

const δs = collect(range(-1.0, 5.0, length=100))
const bs = collect(range(-2.0, 1.5, length=100))

function point(δ, b, p, tab)
    sys = make_mathieu(δ=δ, ε=1.0, b0=b, a1=0.1)
    grid = TimeGrid(collect(range(0.0, sys.T, length=p+1)))
    try
        sol, est = floquet_analysis(sys.prob, grid, tab, p;
                                    error_estimation=true, nev=2, tol=1e-8)
        return sol.spectral_radius, est.mu_error
    catch
        return NaN, NaN
    end
end

panels = []
for p in (8, 24)
    tab = GL(2)
    point(δs[1], bs[1], p, tab)   # warm-up (compile)
    R = Matrix{Float64}(undef, length(bs), length(δs))
    E = similar(R)
    wall = @elapsed Threads.@threads for j in eachindex(δs)
        for (i, b) in enumerate(bs)
            R[i, j], E[i, j] = point(δs[j], b, p, tab)
        end
    end
    @printf("p=%d: %d points in %.1f s wall on %d threads\n",
            p, length(R), wall, Threads.nthreads())
    L = log10.(max.(R, 1e-12))
    ttl = @sprintf("GL2, p=%d — %.1f s wall (%d thr)", p, wall, Threads.nthreads())
    plt = heatmap(δs, bs, L, c=cgrad(:RdBu, rev=true), clims=(-2, 2),
                  colorbar_title=L"\log_{10}\rho", xlabel=L"\delta", ylabel=L"b_0",
                  title=ttl)
    # undecided: the predicted bar straddles the stability boundary rho = 1
    und = [abs(R[i, j] - 1) <= E[i, j] for i in eachindex(bs), j in eachindex(δs)]
    xs_u = [δs[j] for i in eachindex(bs), j in eachindex(δs) if und[i, j]]
    ys_u = [bs[i] for i in eachindex(bs), j in eachindex(δs) if und[i, j]]
    scatter!(plt, xs_u, ys_u, marker=:square, ms=1.9, msw=0, color=:yellow,
             alpha=0.85, label=L"|\rho - 1| \leq \Delta_{\mathrm{pred}}")
    # fine reference boundary (GL7 chart, same grid) — must lie inside the band
    Rref_file = joinpath(RESULTS_DIR, "chart_mathieu_R.csv")
    if isfile(Rref_file)
        Rref = readdlm(Rref_file, ',')
        contour!(plt, δs, bs, log10.(max.(Rref, 1e-12)), levels=[0.0],
                 color=:black, lw=2, colorbar_entry=false)
        plot!(plt, [NaN], [NaN], color=:black, lw=2, label=L"\rho = 1" * " (GL7 ref)")
    end
    plot!(plt, legend=:bottomright, legend_background_color=RGBA(1, 1, 1, 0.75))
    push!(panels, plt)
end

plt = plot(panels..., layout=(1, 2), size=(1150, 420),
           plot_title="Where can the coarse chart be trusted? — undecided band from the error bar",
           plot_titlefontsize=12, left_margin=5Plots.mm, bottom_margin=5Plots.mm)
savefig(plt, joinpath(FIGURES_DIR, "error_chart_mathieu.png"))
savefig(plt, joinpath(FIGURES_DIR, "error_chart_mathieu.pdf"))
println("[run_error_chart] done -> error_chart_mathieu")
