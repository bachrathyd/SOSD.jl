# Top-up: turning cliff curves at p = 20, 30, 40 (appends to the existing CSV,
# which already has p = 1 and 10).
include(joinpath(@__DIR__, "harness.jl"))
SOSD_HARNESS_LOADED = true

function mu_fast(sys::BenchSystem, p::Int, tab)
    r = sys.r_of_p(p); S = size(tab.a, 1); BSIZE = (S + 1) * sys.D
    state_size = (r + 1) * BSIZE
    grid = TimeGrid(collect(range(0.0, sys.T, length=p+1)))
    sysm = build_system_matrices(sys.prob, grid, tab, r)
    m = SparseMonodromyMap(MonodromyMap(sys.prob, grid, tab, sysm, p, r, state_size))
    vals, _, _ = eigsolve(m, det_x0(state_size), 1, :LM; tol=1e-8, maxiter=60)
    return abs(vals[1])
end

sys = make_turning_ssv()
mu_ref = reference_mu(sys)
csv = joinpath(RESULTS_DIR, "s_complexity_$(sys.name).csv")
svals = vcat(1:2:25, 28:4:40)
for p in (20, 30, 40)
    println("\n[turning] fixed p = $p")
    for s in svals
        local mu, tmin
        try
            tab = GL(s)
            mu_fast(sys, p, tab)
            t1 = @elapsed mu = mu_fast(sys, p, tab)
            t2 = @elapsed mu_fast(sys, p, tab)
            tmin = min(t1, t2)
        catch e
            println("  s=$s FAILED"); continue
        end
        rel = abs(mu - mu_ref) / abs(mu_ref)
        @printf("  s=%3d N=%5d rel=%.1e t=%.3gs\n", s, p*(s+1), rel, tmin)
        append_csv(csv, "p,s,order,N,rel_error,t_min",
                   [(p, s, 2s, p*(s+1), rel, tmin)])
        tmin > 5.0 && (println("  (cap)"); break)
    end
end
println("TOPUP DONE")
