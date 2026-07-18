
using Plots
using StaticArrays
using KrylovKit
using DifferentialEquations
using DataInterpolations
using BenchmarkTools
using LinearAlgebra
using SemiDiscretizationMethod

# 1. Governing Equation for MFCM (In-place)
function DelayMathieu!(du, u, h, p, t)
    ζ, δ, ϵ, b, τ, T = p
    du[1] = u[2]
    du[2] = -(δ + ϵ * cos(2π * t / T)) * u[1] - 2ζ * u[2] + b * h(p, t - τ)[1]
end

# 2. Operator Structure
struct DDEPeriodMapping{P, PR, SA, TF, V}
    p::P
    prob::PR
    solv_args::SA
    T::Float64
    t_fixed::TF
    u_template::V
end

function (m::DDEPeriodMapping)(u_vec)
    d = length(m.u_template)
    interps = ntuple(i -> BSplineInterpolation(getindex.(u_vec, i), m.t_fixed, 5, :ArcLen, :Average), d)
    h_func(p, t_eval) = typeof(m.u_template)(ntuple(i -> interps[i](t_eval), Val(d)))
    
    u0 = h_func(m.p, m.t_fixed[end])
    new_prob = remake(m.prob; u0=u0, h=h_func, p=m.p, tspan=(0.0, m.T))
    
    sol = solve(new_prob; m.solv_args..., saveat=m.t_fixed .+ m.T)
    return sol.u
end

# 3. Grid Synchronization Logic
function setup_synchronized_grid(p, N_target)
    ζ, δ, ϵ, b, τ, T = p
    dt = T / N_target
    N_T = Int(round(T / dt))
    dt_sync = T / N_T
    N_tau = Int(ceil(τ / dt_sync))
    tau_max = N_tau * dt_sync
    t_fixed = collect((-tau_max):dt_sync:0.0)
    return t_fixed, dt_sync, tau_max
end

# 4. Benchmarking Engine
function run_comprehensive_benchmark()
    println("Initializing Comprehensive DDE Benchmark...")
    
    # Parameters
    p = (0.005, 1.5, 0.15, 0.05, 2π, 2π) # ζ, δ, ϵ, b, τ, T
    τ, T = p[5], p[6]
    u0_template = @MArray [0.0, 0.0]

    # Reference Value from SDM
    println("Calculating SDM Reference...")
    AM(t) = [0.0 1.0; -(p[2] + p[3] * cos(2π * t / T)) -2p[1]]
    BM(t) = [0.0 0.0; p[4] 0.0]
    # Ensure τ is Float64 to avoid eps(Int) error
    τ_float = Float64(τ)
    sdm_prob = LDDEProblem(ProportionalMX(AM), [DelayMX(τ_float, BM)])
    
    # Using DiscreteMapping with numeric order 1 (SDM)
    # The second argument to SemiDiscretization is the time step dt_sdm
    dt_sdm = T / 500
    sdm_mapping = DiscreteMapping(sdm_prob, SemiDiscretization(1, dt_sdm), T)
    mu_ref = spectralRadiusOfMapping(sdm_mapping)
    println("SDM Reference (mu_max): $mu_ref")

    # Solvers to test
    solvers = [
        ("Euler (O1)", Euler(), false),
        ("Heun (O2)", Heun(), false),
        ("RK4 (O4)", RK4(), false),
        ("Vern6 (O6)", Vern6(), false),
        ("ImplicitEuler (O1)", ImplicitEuler(), true),
        ("Trapezoid (O2)", Trapezoid(), true),
        ("SDIRK2 (O2)", SDIRK2(), true),
    ]

    N_steps_list = [40, 80, 160, 320]
    results = Dict()

    for (name, alg, is_implicit) in solvers
        println("Testing Solver: $name")
        res_list = []
        for N in N_steps_list
            t_fixed, dt, tau_max = setup_synchronized_grid(p, N)
            
            h_init(p, t) = @MArray [1.0, 0.0]
            prob = DDEProblem{true}(DelayMathieu!, u0_template, h_init, (0.0, T), p; constant_lags=[τ])
            
            solv_args = (alg=MethodOfSteps(alg), adaptive=false, dt=dt, verbose=false)
            mapping = DDEPeriodMapping(p, prob, solv_args, T, t_fixed, u0_template)
            s_initial = [MVector{2}(rand(2)) for _ in 1:length(t_fixed)]
            Neig = 1
            krylov_solver = KrylovKit.Arnoldi(tol=1e-10, krylovdim=20, verbosity=0)

            # Warm-up
            schursolve(mapping, s_initial, Neig, :LM, krylov_solver)

            # Benchmark
            b_res = @benchmark schursolve($mapping, $s_initial, $Neig, :LM, $krylov_solver)
            avg_time = mean(b_res.times) / 1e6 # ms
            mem = b_res.memory / 1024^2 # MiB
            
            # Error
            _, _, vals, _ = schursolve(mapping, s_initial, Neig, :LM, krylov_solver)
            mu_calc = abs(vals[1])
            err = abs(mu_calc - mu_ref)
            
            push!(res_list, (N=length(t_fixed), time=avg_time, err=err, mem=mem))
            println("  N=$(length(t_fixed)), Error=$(err), Time=$(avg_time)ms")
        end
        results[name] = res_list
    end

    # 5. Plotting
    p1 = plot(title="Work-Precision", xlabel="CPU Time [ms]", ylabel="Error", xscale=:log10, yscale=:log10)
    p2 = plot(title="Convergence", xlabel="Nsteps", ylabel="Error", xscale=:log10, yscale=:log10)
    p3 = plot(title="Memory", xlabel="Nsteps", ylabel="Memory [MiB]", xscale=:log10, yscale=:log10)

    for (name, res) in results
        ns = [r.N for r in res]
        ts = [r.time for r in res]
        es = [r.err for r in res]
        ms = [r.mem for r in res]
        
        plot!(p1, ts, es, label=name, marker=:circle)
        plot!(p2, ns, es, label=name, marker=:square)
        plot!(p3, ns, ms, label=name, marker=:triangle)
    end

    final_plot = plot(p1, p2, p3, layout=(3,1), size=(1000, 1200))
    savefig(final_plot, "dde_benchmark_results.png")
    println("Benchmark complete. Results saved to dde_benchmark_results.png")
end

run_comprehensive_benchmark()
