
using Plots
using StaticArrays
using KrylovKit


using DifferentialEquations

using DataInterpolations
# 1. Define the Governing Equation (In-place)

function DelayMathieu!(du, u, h, p, t)
    ζ, δ, ϵ, b, τ, T = p
    F = 0.1 * (cos(2π * t / T)^10)
    du[1] = u[2]
    du[2] = -(δ + ϵ * cos(2π * t / T)) * u[1] - 2ζ * u[2] + b * h(p, t - τ)[1] #+ F
end

# 2. Setup Parameters
const ζ = 0.005
const δ_init = 1.5
const ϵ_init = 0.15
const τ = 2π
const b = 0.05
const T = 2π
const p_init = (ζ, δ_init, ϵ_init, b, τ, T)

# 3. Create Figure Layout
# 4. Long Simulation
u0 = @MArray [1.0, 0.0]
h1(p, t) = @MArray [1.0, 0.0]
const prob_long = DDEProblem{true}(DelayMathieu!, u0, h1, (0.0, T * 8.0), p_init; constant_lags=[τ])
const Solver_args = Dict(:alg => MethodOfSteps(Tsit5()), :verbose => false, :reltol => 1e-5)

println("Running long simulation...")
@time sol_long = solve(prob_long; Solver_args...)

plot(sol_long)

## -------------------------------------------------

# The Implementaion of the Diff.EQ.based implementation of the one-period mapping starts here
# this can be treated as a lazy implementation, or allocation free, hance matrix free eigen calculation is used!

function create_bspline(u::AbstractVector{<:AbstractVector}, t::AbstractVector; order::Int=3)
    d = length(u[1]) # Number of dimensions (e.g., 3 for a 3D MVector)
    interps = ntuple(i -> BSplineInterpolation([val[i] for val in u], t, order, :ArcLen, :Average), d)
    return (p, t_eval) -> typeof(u[1])(ntuple(i -> interps[i](t_eval), Val(d)))
end

function state_mapping(u_loc, t_f, p_loc, prob, solv_args, T)
    h_remake = create_bspline(u_loc, t_f, order=5)
    u0_remake = h_remake(p_loc, t_f[end])
    new_prob = remake(prob; u0=u0_remake, h=h_remake, p=p_loc, tspan=(0, T))
    #    sol = solve(new_prob; solv_args..., adaptive=false, dt=T/Nsteps)#, saveat=t_fixed .+ T)
    sol_loc = solve(new_prob; solv_args..., saveat=t_f .+ T)#adaptive=false, dt=T/Nsteps,
    return deepcopy(sol_loc.u)
end

function linearmapping(u::TT)::TT where TT
    state_mapping(u, t_fixed, p_init, prob_long, Solver_args, T)
end

function mf_dde_mumax(s_initial, KrylovKit_arg)
   TT, vecs, vals, info = schursolve(linearmapping, s_initial, KrylovKit_arg...)
   return abs(vals[1])
end

Nsteps = 100
#Δt = T / Nsteps
τ_max = max(τ, T)#Tmust be larger than τ_max
t_fixed = collect(LinRange(-τ_max, 0, Nsteps))
d = 2
#TODO : create a fixed timestep solver....
#const Solver_args = Dict(:alg => MethodOfSteps(RK()), :verbose => false, adaptive=false, dt=0.01)

s_initial = [MVector{d}(rand(d)) for _ in 1:Nsteps]

Neig = 3
Krylov_arg = (Neig, :LM, KrylovKit.Arnoldi(tol=1e-9, krylovdim=(Neig + 15), verbosity=0, maxiter=10));

@time TT, vecs, vals, info = schursolve(linearmapping, s_initial, Krylov_arg...);
abs.(vals[1])
mf_dde_mumax(s_initial, Krylov_arg)

#@time vals, vecs, info = eigsolve(linearmapping, s_initial, Krylov_arg...);
#abs.(vals[1])

mu_ref=0.8322455410056632
error=mu_ref -abs.(vals[1]) 



using BenchmarkTools
@benchmark TT, vecs, vals, info = schursolve(linearmapping, s_initial, Krylov_arg...)
#15 ms





