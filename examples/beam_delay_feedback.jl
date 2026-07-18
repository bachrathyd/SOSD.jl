# FEM longitudinal beam with delayed boundary feedback — predecessor paper, Appendix C.
#
# N = 15 linear elements, fixed at node 1  ⇒  D = 2*(N-1) = 28 first-order states.
# Delayed displacement feedback from the first free node to the last node,
# with act-and-wait switching (feedback active only for t < 0.8 T) and
# constant excitation at the free end.
#
# This is the "large-p / high-dimensional" demonstration case of the paper:
# the banded moderate-order route should win decisively here.
# It exercises the dense (heap-allocated) assembly path of MFCM, since
# S*D = 28*S is far beyond the StaticArrays fast-path threshold.
#
# Parameters follow SemiDiscretizationMethod.jl/examples/beam_delay_feedback.jl:
#   E = 210 GPa, A = 1e-4 m², ρ = 7800 kg/m³, L = 100 m, η = 0.01, N = 15
#   c = sqrt(E/ρ), Tspeed = L/c, τ = 0.2*Tspeed, T = 0.4*Tspeed  (τ/T = 1/2)

using MFCM
using LinearAlgebra
using KrylovKit
using Printf

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

function beam_matrices(E, A, ρ, L, η, N)
    dx = L / N
    m = ρ * A * dx
    k = E * A / dx

    M = Diagonal(fill(m, N))
    K = zeros(N, N)
    for i in 1:N
        if i > 1
            K[i, i] += k
            K[i, i-1] -= k
        end
        if i < N
            K[i, i] += k
            K[i, i+1] -= k
        end
    end

    # Fixed boundary condition at the first node
    K = K[2:end, 2:end]
    M = M[2:end, 2:end]
    C = K * η
    return M, C, K
end

"""
    beam_problem(; E, A, ρ, L, η, N, P, τpTspeed, TperpTspeed, act_and_wait)

Build the MFCM `LDDEProblem` for the delayed-feedback beam and return
`(prob, D, τ, Tper)`. With `act_and_wait = true` the feedback matrix switches
off for t ≥ 0.8*Tper (non-smooth stress-test configuration).
"""
function beam_problem(; E=210e9, A=1e-4, ρ=7800.0, L=100.0, η=0.01, N=15,
                       P=0.2, τpTspeed=0.2, TperpTspeed=0.4, act_and_wait=true)
    c = sqrt(E / ρ)
    Tspeed = L / c
    τ = τpTspeed * Tspeed
    Tper = TperpTspeed * Tspeed

    M, C, K = beam_matrices(E, A, ρ, L, η, N)
    Nm1 = N - 1
    D = 2 * Nm1

    Z = zeros(Nm1, Nm1)
    In = Matrix(I, Nm1, Nm1)
    A_sys = [Z In; -M\K -M\C]

    B = zeros(D, D)
    B[D, 1] = P * E / (L / N)   # displacement of first free node → last velocity equation

    F = zeros(Nm1); F[Nm1] = 1.0
    Ffirst = vcat(zeros(Nm1), F)

    AMx = ProportionalMX(t -> A_sys)
    τ1 = t -> τ
    BMx1 = if act_and_wait
        DelayMX(τ1, t -> B .* (mod(t, Tper) < 0.8 * Tper))
    else
        DelayMX(τ1, t -> B)
    end
    cVec = Additive(t -> Ffirst)

    prob = LDDEProblem{D, Float64}(AMx, [BMx1], cVec)
    return prob, D, τ, Tper
end

# ---------------------------------------------------------------------------
# Spectral radius via the sparse multiplication-free route
# ---------------------------------------------------------------------------

"""
    beam_mumax(p; s=2, act_and_wait=true, solver=:sparse)

Dominant Floquet multiplier magnitude for the beam with `p` steps per period
and `GL(s)` collocation. τ/T = 1/2, so r = p ÷ 2 (`p` must be even).
"""
function beam_mumax(p::Int; s::Int=2, act_and_wait::Bool=true, solver::Symbol=:sparse)
    iseven(p) || error("p must be even (τ/T = 1/2 ⇒ r = p ÷ 2)")
    prob, D, τ, Tper = beam_problem(act_and_wait=act_and_wait)
    r = p ÷ 2
    tableau = GL(s)
    grid = TimeGrid(collect(range(0.0, Tper, length=p+1)))
    BSIZE = (s + 1) * D
    state_size = (r + 1) * BSIZE

    sys = build_system_matrices(prob, grid, tableau, r)
    m_lazy = MonodromyMap(prob, grid, tableau, sys, p, r, state_size)
    m = solver === :sparse ? SparseMonodromyMap(m_lazy) : m_lazy

    x0 = ones(Float64, state_size) ./ sqrt(state_size)
    vals, _, _ = eigsolve(m, x0, 1, :LM; tol=1e-11)
    return abs(vals[1])
end

# ---------------------------------------------------------------------------
# Demonstration run (executed when called as a script)
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    println("FEM beam with delayed boundary feedback (D = 28)")
    println("=" ^ 60)

    println("\nSmooth configuration (act-and-wait off):")
    for p in [20, 40, 80, 160]
        t = @elapsed mu = beam_mumax(p; s=2, act_and_wait=false)
        @printf("  GL2, p=%4d:  mu = %.12f   (%.2fs)\n", p, mu, t)
    end

    println("\nNon-smooth configuration (act-and-wait ON, switch at 0.8T):")
    for p in [20, 40, 80, 160]
        t = @elapsed mu = beam_mumax(p; s=2, act_and_wait=true)
        @printf("  GL2, p=%4d:  mu = %.12f   (%.2fs)\n", p, mu, t)
    end
end
