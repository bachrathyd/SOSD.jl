# Non-smoothness stress test (handoff §5): the act-and-wait switching of the
# beam feedback (switch at t = 0.8 T) degrades high-order convergence whenever
# the switching instant does NOT coincide with a grid point.
#   - p ≡ 0 (mod 5): switch lands exactly on the grid  -> full order retained
#   - p powers of two: switch falls inside a step       -> order collapses
# This is measured for GL1/GL2/GL3 and is a core argument for the sweet spot:
# moderate order + many steps is more robust for non-smooth engineering systems.

(@isdefined MFCM_HARNESS_LOADED) || (include(joinpath(@__DIR__, "harness.jl")); MFCM_HARNESS_LOADED = true)

function run_nonsmooth()
    sys = make_beam(act_and_wait=true)

    # Reference: fine grid with ALIGNED switching point (p multiple of 10)
    cache = load_ref_cache()
    mu_ref = get(cache, sys.name, NaN)
    if !isfinite(mu_ref)
        println("[nonsmooth] computing aligned-grid reference (GL3, p=1500/2000, lazy)...")
        mu1 = mfcm_mu(sys, 1500, GL(3); tol=1e-13, solver=:lazy)
        mu2 = mfcm_mu(sys, 2000, GL(3); tol=1e-13, solver=:lazy)
        @printf("[nonsmooth] ref agree: %.3e\n", abs(mu1 - mu2) / abs(mu2))
        mu_ref = mu2
        cache[sys.name] = mu_ref
        save_ref_cache(cache)
    end

    csv = joinpath(RESULTS_DIR, "nonsmooth_beam_aaw.csv")
    isfile(csv) && rm(csv)

    aligned_ps  = [20, 40, 80, 160, 320, 640]            # multiples of 10: 0.8T on grid
    offgrid_ps  = [16, 32, 64, 128, 256, 512]            # powers of two: 0.8T inside a step

    for s in (1, 2, 3)
        tab = GL(s)
        for (family, ps) in (("aligned", aligned_ps), ("offgrid", offgrid_ps))
            println("\n[nonsmooth] GL$s, $family grids")
            errs = Float64[]; used = Int[]
            for p in ps
                local mu
                try
                    mu = mfcm_mu(sys, p, tab)
                catch e
                    println("  p=$p FAILED"); break
                end
                rel_err = abs(mu - mu_ref) / abs(mu_ref)
                @printf("  p=%4d rel_err=%.2e\n", p, rel_err)
                push!(errs, rel_err); push!(used, p)
                append_csv(csv, "s,order,family,p,rel_error",
                           [(s, 2s, family, p, rel_err)])
            end
            k = fit_slope(used, errs; lo=1e-14, hi=1e-1)
            @printf("[nonsmooth] GL%d %s: fitted slope %.2f (nominal %d)\n", s, family, -k, 2s)
        end
    end
    println("[nonsmooth] done -> $csv")
end

run_nonsmooth()
