# Validation study for the embedded-pair error estimation
# (floquet_analysis(...; error_estimation=true)).
#
# For each (system, method, p): the computed spectral radius, its predicted
# error bar (with the raw channels), and the TRUE error against the cached
# two-resolution reference — so the figures can answer: does the predicted
# range really contain the real root, and how conservative is it?
#
# Output: results/error_estimation.csv  (figures: make_error_figures.jl)

include(joinpath(@__DIR__, "harness.jl"))
using Printf

const OUT = joinpath(RESULTS_DIR, "error_estimation.csv")

systems = [make_mathieu(), make_bio(), make_turning_ssv()]
methods = [("GL2", GL(2)), ("GL3", GL(3)), ("GL5", GL(5)),
           ("BS3", SOSD.BS3()), ("RK4", SOSD.RK4())]
PS = Dict(
    "mathieu"     => [10, 16, 24, 36, 54, 80, 120, 180, 270, 400],
    "bio"         => [10, 16, 24, 36, 54, 80, 120, 180, 270, 400],
    "turning_ssv" => [46, 68, 100, 150, 226, 340, 510])

# Fixed-point references: y(0) node value from a fine GL5 grid
println("[fixref] computing fixed-point references...")
fixref = Dict{String, Vector{Float64}}()
for sys in systems
    sys.name == "bio" && continue          # zero forcing → trivial fixed point
    p_ref = sys.name == "mathieu" ? 800 : 1800
    r_ref = sys.r_of_p(p_ref)
    grid = TimeGrid(collect(range(0.0, sys.T, length=p_ref+1)))
    sol = floquet_analysis(sys.prob, grid, GL(5), r_ref; periodic_solution=true, nev=1)
    fixref[sys.name] = sol.fixpoint[1:sys.D]
    println("[fixref] $(sys.name): y(0) = ", fixref[sys.name])
end

nan_or(x) = isnan(x) ? NaN : x

open(OUT, "w") do io
    println(io, "system,method,p,s,rho,err_true,bar,barQ,barI,kappa,gap,mode_err,dmu_hatQ,dmu_hatI,fix_true,fix_pred,mu_ref")
    for sys in systems
        mu_ref = reference_mu(sys)
        for (label, tab) in methods
            S = length(tab.b)
            for p in PS[sys.name]
                r = sys.r_of_p(p)
                grid = TimeGrid(collect(range(0.0, sys.T, length=p+1)))
                do_fix = haskey(fixref, sys.name)
                sol = nothing; est = nothing
                try
                    sol, est = floquet_analysis(sys.prob, grid, tab, r;
                                                error_estimation=true,
                                                periodic_solution=do_fix,
                                                embedded_eigs=true)
                catch err
                    @warn "run failed" sys.name label p err
                    continue
                end
                err_true = abs(sol.spectral_radius - mu_ref)
                dhatQ = abs(sol.mu - est.mu_embedded_quadrature)
                dhatI = abs(sol.mu - est.mu_embedded_interpolation)
                fix_t = NaN; fix_p = NaN
                if do_fix && sol.fixpoint !== nothing && est.fixpoint_delta !== nothing
                    fix_t = norm(sol.fixpoint[1:sys.D] .- fixref[sys.name])
                    fix_p = norm(est.fixpoint_delta[1:sys.D])
                end
                @printf(io, "%s,%s,%d,%d,%.16e,%.6e,%.6e,%.6e,%.6e,%.4e,%.4e,%.6e,%.6e,%.6e,%.6e,%.6e,%.16e\n",
                        sys.name, label, p, S, sol.spectral_radius, err_true,
                        est.mu_error, nan_or(est.quadrature_error), nan_or(est.interpolation_error),
                        est.eigenvalue_condition, est.spectral_gap, nan_or(est.mode_error),
                        dhatQ, dhatI, fix_t, fix_p, mu_ref)
                flush(io)
            end
            println("[done] $(sys.name)  $label")
        end
    end
end
println("[run_error_estimation] finished -> $OUT")
