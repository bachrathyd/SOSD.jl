
using Plots
using StaticArrays
using KrylovKit
using DifferentialEquations
using DataInterpolations
using BenchmarkTools
using LinearAlgebra

# 1. Define the Governing Equation (In-place)
function DelayMathieu!(du, u, h, p, t)
    ζ, δ, ϵ, b, τ, T = p
    # F = 0.1 * (cos(2π * t / T)^10) # Optional periodic forcing
    du[1] = u[2]
    du[2] = -(δ + ϵ * cos(2π * t / T)) * u[1] - 2ζ * u[2] + b * h(p, t - τ)[1]
end

# Struct-based implementation to avoid globals and ensure type stability.
# This encapsulates the mapping from one period's history to the next.
# By using a struct with parametric types, we ensure the compiler can optimize the mapping.
struct DDEPeriodMapping{P, PR, SA, TF, V}
    p::P
    prob::PR
    solv_args::SA
    T::Float64
    t_fixed::TF
    u_template::V
end

# Functor that performs the one-period mapping: y(t) for t ∈ [-tau_max, 0]  ->  y(t+T) for t ∈ [-tau_max, 0]
function (m::DDEPeriodMapping)(u_vec)
    # u_vec is Vector{MVector{d, Float64}} representing the history on t_fixed
    d = length(m.u_template)
    
    # 1. Create high-order history interpolation (B-Splines)
    # We use order 5 for high accuracy as in the original code.
    # getindex.(u_vec, i) extracts the i-th component of all MVectors in the history.
    interps = ntuple(i -> BSplineInterpolation(getindex.(u_vec, i), m.t_fixed, 5, :ArcLen, :Average), d)
    
    # Closure for the history function used by DifferentialEquations.jl
    h_func(p, t_eval) = typeof(m.u_template)(ntuple(i -> interps[i](t_eval), Val(d)))
    
    # 2. Remake the DDE problem with the new history and corresponding initial condition
    # u0 must match the history at the end of the previous interval (t=0 relative to the new start)
    u0 = h_func(m.p, m.t_fixed[end])
    
    # Efficiently reuse the problem structure to avoid full re-allocation
    new_prob = remake(m.prob; u0=u0, h=h_func, p=m.p, tspan=(0.0, m.T))
    
    # 3. Solve for one period
    # Using saveat ensures we get the solution exactly on the shifted grid [T-tau_max, T]
    # If m.solv_args specifies a fixed dt that aligns with saveat, this is very efficient.
    sol = solve(new_prob; m.solv_args..., saveat=m.t_fixed .+ m.T)
    
    # Return the state vector at the saved points.
    # We return the vector directly; KrylovKit will handle it.
    return sol.u
end

function run_dde_test()
    println("Setting up Refined DDE Period Mapping Test...")

    # Parameters
    ζ = 0.005
    δ_init = 1.5
    ϵ_init = 0.15
    τ = 2π
    b = 0.05
    T = 2π
    p = (ζ, δ_init, ϵ_init, b, τ, T)

    # Grid Configuration
    # Increasing Nsteps to 200 for better discretization accuracy
    Nsteps = 200
    τ_max = max(τ, T)
    t_fixed = collect(LinRange(-τ_max, 0, Nsteps))
    
    # Initial DDE Problem Setup
    u0 = @MArray [0.0, 0.0] # Template state
    h_init(p, t) = @MArray [1.0, 0.0]
    prob = DDEProblem{true}(DelayMathieu!, u0, h_init, (0.0, T), p; constant_lags=[τ])

    # Solver Configuration (Fixed Step)
    # dt is chosen to align perfectly with the history grid
    dt = T / (Nsteps - 1)
    solv_args = (alg=MethodOfSteps(RK4()), adaptive=false, dt=dt, verbose=false)

    # Initial guess for KrylovKit (random history)
    s_initial = [MVector{2}(rand(2)) for _ in 1:Nsteps]

    # Create the mapping object (The "Matrix-Free Operator")
    mapping = DDEPeriodMapping(p, prob, solv_args, T, t_fixed, u0)

    # KrylovKit Configuration
    Neig = 3
    krylov_solver = KrylovKit.Arnoldi(tol=1e-9, krylovdim=(Neig + 15), verbosity=0, maxiter=10)

    println("Starting Matrix-Free Eigenvalue Calculation (KrylovKit)...")
    println("Using RK4 fixed-step solver with N=$Nsteps steps per period.")
    
    # First call includes compilation time
    @time TT, vecs, vals, info = schursolve(mapping, s_initial, Neig, :LM, krylov_solver)
    
    println("\nResults:")
    println("Floquet Multipliers (Top $Neig):")
    for (i, v) in enumerate(vals[1:Neig])
        println("  μ_$i = $v, |μ_$i| = $(abs(v))")
    end

    # Reference value from a high-accuracy solve
    mu_ref = 0.8322455410056632
    err = abs(vals[1]) - mu_ref 
    println("\nDifference from reference: $err")
    
    if abs(err) < 1e-4
        println("Verification SUCCESSFUL (within 1e-4)")
    else
        println("Verification FAILED (check Nsteps or solver order)")
    end

    println("\nPerformance Benchmark (one mapping call):")
    @btime $mapping($s_initial)
    
    println("\nFull Krylov solve benchmark:")
    @benchmark schursolve($mapping, $s_initial, $Neig, :LM, $krylov_solver)
end

# Execute the test
if abspath(PROGRAM_FILE) == @__FILE__
    run_dde_test()
end
