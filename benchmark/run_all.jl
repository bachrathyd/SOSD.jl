# Master benchmark runner: executes every study in order and records the
# environment. Results go to benchmark/results/*.csv; figures are generated
# separately with make_figures.jl (so plots can be tweaked without re-running).
#
# Usage:  julia --project=benchmark benchmark/run_all.jl

include(joinpath(@__DIR__, "harness.jl"))
MFCM_HARNESS_LOADED = true

write_env_report()
t0 = time()

println("\n", "="^70, "\n ORDER VERIFICATION\n", "="^70)
include(joinpath(@__DIR__, "run_order_verification.jl"))

println("\n", "="^70, "\n WORK-PRECISION / TIME-COMPLEXITY\n", "="^70)
include(joinpath(@__DIR__, "run_work_precision.jl"))

println("\n", "="^70, "\n SWEET SPOT\n", "="^70)
include(joinpath(@__DIR__, "run_sweet_spot.jl"))

println("\n", "="^70, "\n NON-SMOOTHNESS STRESS TEST\n", "="^70)
include(joinpath(@__DIR__, "run_nonsmooth.jl"))

@printf("\nAll studies finished in %.1f minutes.\n", (time() - t0) / 60)
