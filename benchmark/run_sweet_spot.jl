# Sweet-spot study (handoff §4.4, the headline result):
# sweep per-step order s and resolution p on two contrasting systems
# (small Mathieu vs high-dimensional beam), recording error, CPU time and
# matrix structure (nnz, density, bandwidth). Post-processing then finds, for
# each accuracy target, the CPU-time-optimal (p, s) — the U-curve over s.
# Also includes the single-step "spectral corner" (p = 1, s large): the
# degenerate case where one collocation step spans the whole period.

(@isdefined SOSD_HARNESS_LOADED) || (include(joinpath(@__DIR__, "harness.jl")); SOSD_HARNESS_LOADED = true)

even_p(x) = max(2, 2 * round(Int, x / 2))

function sweep_ps(sys::BenchSystem, ss::Vector{Int}, ps_of_s::Function; time_cap=25.0)
    mu_ref = reference_mu(sys)
    csv = joinpath(RESULTS_DIR, "sweet_spot_$(sys.name).csv")
    isfile(csv) && rm(csv)

    for s in ss
        tab = try
            GL(s)
        catch e
            @warn "GL($s) construction failed"; continue
        end
        println("\n[$(sys.name)] GL$s (order $(2s))")
        for p in ps_of_s(s)
            print("  p=$p ")
            local mu, tm, ts, nrep, stats
            try
                mu, tm, ts, nrep = time_stats(() -> sosd_mu(sys, p, tab); budget=1.5, min_reps=3, max_reps=8)
                stats = matrix_stats(sys, p, tab)
            catch e
                println("FAILED: ", sprint(showerror, e)[1:min(end,160)])
                break
            end
            rel_err = abs(mu - mu_ref) / abs(mu_ref)
            @printf("rel_err=%.2e t=%.3gs nnzL=%d bw=%d\n", rel_err, tm, stats.nnz_L, stats.bandwidth_L)
            append_csv(csv, "s,order,p,rel_error,t_mean,t_std,n_reps,n_L,nnz_L,density_L,bandwidth_L",
                       [(s, 2s, p, rel_err, tm, ts, nrep, stats.n_L, stats.nnz_L,
                         stats.density_L, stats.bandwidth_L)])
            tm > time_cap && (println("  (time cap)"); break)
            rel_err < 1e-14 && p > 4 && break   # already at the floor: higher p adds nothing
        end
    end
    println("[sweet-spot] $(sys.name) done -> $csv")
end

# --- Mathieu: s = 1..20 (+ spectral corner p=1, s up to 60) -----------------
sys_m = make_mathieu()
sweep_ps(sys_m,
         collect(1:12) ∪ [16, 20],
         s -> unique(round.(Int, 10 .^ (0.0:0.2:3.2))))

# spectral corner: a single step spanning the whole period (tau = T => r = 1)
let sys = sys_m
    mu_ref = reference_mu(sys)
    csv = joinpath(RESULTS_DIR, "spectral_corner_mathieu.csv")
    isfile(csv) && rm(csv)
    println("\n[spectral corner] single step p=1, s = 4..60")
    for s in [4, 6, 8, 10, 14, 18, 24, 30, 40, 50, 60]
        local mu, tm, ts, nrep
        try
            tab = GL(s)
            mu, tm, ts, nrep = time_stats(() -> sosd_mu(sys, 1, tab); budget=1.5, min_reps=3, max_reps=8)
        catch e
            println("  s=$s FAILED: ", sprint(showerror, e)[1:min(end,160)])
            continue
        end
        rel_err = abs(mu - mu_ref) / abs(mu_ref)
        @printf("  s=%2d rel_err=%.2e t=%.3gs\n", s, rel_err, tm)
        append_csv(csv, "s,order,rel_error,t_mean,t_std,n_reps",
                   [(s, 2s, rel_err, tm, ts, nrep)])
    end
end

# --- Beam: s = 1..6, even p ---------------------------------------------------
sweep_ps(make_beam(),
         collect(1:6),
         s -> unique(even_p.(2 .^ (1.5:0.5:8.5)));
         time_cap=40.0)
