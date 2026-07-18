using SOSD
using Plots
using LinearAlgebra
using StaticArrays
using KrylovKit
using BenchmarkTools
using Printf

# Mathieu equation parameters
const ζ = 0.2; const δ = 3.0; const ε = 3.0; const b = -0.5; const τ = 2π; const T = 2π

# Pre-extract system once
function get_prob()
    rhs = (u, h, p_params, t) -> begin
        x, v = u
        x_tau = h(p_params, t - τ)[1]
        return @SVector [v, -2*ζ*v - (δ + ε*cos(2π/T*t))*x + b*x_tau]
    end
    return SOSD.extract_SDM_system(rhs, nothing, Val(2))
end

const global_prob = get_prob()

function mathieu_mumax(p, s; tab=nothing)
    if isnothing(tab); try tab = SOSD.GL(s); catch; return NaN, 0.0; end; end
    grid = SOSD.TimeGrid(collect(range(0.0, T, length=p+1)))
    BSIZE = (s + 1) * 2; state_size = (p + 1) * BSIZE
    mu_val = NaN
    t = @elapsed begin
        try
            sys = SOSD.build_system_matrices(global_prob, grid, tab, p)
            m = SOSD.SparseMonodromyMap(SOSD.MonodromyMap(global_prob, grid, tab, sys, p, p, state_size))
            vals, _ = eigsolve(m, rand(state_size), 1, :LM; tol=1e-11)
            mu_val = abs(vals[1])
        catch e
            mu_val = NaN
        end
    end
    return mu_val, t
end

println("Computing reference...")
mu_ref = 0.8675382878830239 

println("\nFixed p=10, s=1:200:")
ss_line = [1:2:50..., 60:10:200...]; errs_l = []; times_l = []
for s in ss_line
    mu, t = mathieu_mumax(10, s)
    if t > 10.0 || isnan(mu); println("\nBreak at s=$s"); break; end
    push!(errs_l, max(1e-16, abs(mu - mu_ref))); push!(times_l, t)
    @printf("%d(%.1fs) ", s, t)
end

println("\n\nGrid (p, s):")
ps_g = [10, 20, 40, 60, 80]; ss_g = [1, 2, 4, 8, 12, 16, 20, 30]
err_g = fill(NaN, length(ps_g), length(ss_g)); time_g = fill(NaN, length(ps_g), length(ss_g))
for (i, p) in enumerate(ps_g)
    @printf("p=%d: ", p)
    for (j, s) in enumerate(ss_g)
        mu, t = mathieu_mumax(p, s)
        time_g[i, j] = t
        if t > 10.0 || isnan(mu); @printf("T/O "); break; end
        err_g[i, j] = log10(max(1e-16, abs(mu - mu_ref)))
        @printf("%d(%.1fs) ", s, t)
    end
    println()
end

p1 = plot(ss_line[1:length(errs_l)], errs_l, yscale=:log10, title="Error p=10", marker=:circle, label="Error")
p2 = plot(ss_line[1:length(times_l)], times_l, yscale=:log10, title="Time p=10", marker=:square, label="Time", color=:red)
p3 = heatmap(ss_g, ps_g, err_g, title="log10(Err) + Time Contours", fill=true, xlabel="s", ylabel="p")
contour!(p3, ss_g, ps_g, time_g, levels=6, color=:white, lw=1.5)
plt = plot(p1, p2, p3, layout=@layout([a b; c]), size=(1000, 800))
savefig("high_order_gl_analysis.png")
println("Done.")
