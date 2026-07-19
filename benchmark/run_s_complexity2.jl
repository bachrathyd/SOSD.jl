# Bounded s-complexity study (Fig-6): p in {1,10,100}, s up to 40.
# Beyond s ~ 30-35 the product-form Lagrange continuous-extension weights lose
# accuracy (condition ~ 2^s) - a stated implementation limit; optima lie at
# s <= 10 so the range is more than sufficient. Looser eigsolve tolerance
# (1e-8) avoids grinding against unreachable tolerances in the pre-resolution
# ("nonsense") regime.
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

function run_sc(sys::BenchSystem; time_cap=5.0)
    mu_ref = reference_mu(sys)
    csv = joinpath(RESULTS_DIR, "s_complexity_$(sys.name).csv")
    isfile(csv) && rm(csv)
    svals = vcat(1:2:25, 28:4:40)
    for p in (1, 10, 100)
        println("\n[$(sys.name)] fixed p = $p")
        for s in svals
            local mu, tmin
            try
                tab = GL(s)
                mu_fast(sys, p, tab)                     # warm-up
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
            tmin > time_cap && (println("  (cap)"); break)
        end
    end
end

run_sc(make_mathieu())
run_sc(make_turning_ssv())
println("S COMPLEXITY DONE")
