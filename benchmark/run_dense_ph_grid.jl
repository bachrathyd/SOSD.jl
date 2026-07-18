# Dense (s, p) grid for the p-h map figure (Daniel's Fig-5 spec):
#   - p from 1 (single step), log grid with 3x the previous density
#   - stages s = 1:50
#   - 5 s CPU cap per point (cells beyond it are left uncomputed -> NaN)
# Replaces sweet_spot_turning_ssv.csv wholesale.

include(joinpath(@__DIR__, "harness.jl"))
SOSD_HARNESS_LOADED = true

function run_dense_grid(sys::BenchSystem; time_cap=5.0)
    mu_ref = reference_mu(sys)
    csv = joinpath(RESULTS_DIR, "sweet_spot_$(sys.name).csv")
    isfile(csv) && rm(csv)
    ps = unique(round.(Int, 10 .^ (0.0:0.0667:3.6)))   # p = 1 ... ~4000, 3x density
    for s in 1:50
        tab = try GL(s) catch; continue; end
        print("s=$s: ")
        for p in ps
            local mu, tm, ts, nrep, stats
            try
                mu, tm, ts, nrep = time_stats(() -> sosd_mu(sys, p, tab); budget=1.0, min_reps=2, max_reps=5)
                stats = matrix_stats(sys, p, tab)
            catch e
                print("x"); continue
            end
            rel = abs(mu - mu_ref) / abs(mu_ref)
            print(".")
            append_csv(csv, "s,order,p,rel_error,t_mean,t_std,n_reps,n_L,nnz_L,density_L,bandwidth_L",
                       [(s, 2s, p, rel, tm, ts, nrep, stats.n_L, stats.nnz_L, stats.density_L, stats.bandwidth_L)])
            if tm > time_cap
                println(" (cap at p=$p)")
                @goto next_s
            end
            # deep past the floor: finer p adds nothing for the map
            rel < 1e-15 && p > 400 && break
        end
        println()
        @label next_s
    end
    println("[dense-grid] $(sys.name) done -> $csv")
end

run_dense_grid(make_turning_ssv())
println("DENSE GRID DONE")
