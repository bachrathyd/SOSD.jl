# Delay differential-algebraic (singular mass matrix) example — paper appendix.
#
# Descriptor system  M ẋ = A(t) x + B x(t-τ),  M = diag(1, ε):
#     ẋ₁ = -x₁ + x₂ + 0.5 x₁(t-1)
#   ε ẋ₂ = sin(t) x₁ - x₂
# For ε = 0 (index-1 DDAE) the algebraic row gives x₂ = sin(t) x₁, so the
# system reduces to the scalar LDDE
#     ẋ₁ = (-1 + sin t) x₁ + 0.5 x₁(t-1),
# which serves as the reference. A singular M requires a stiffly accurate
# tableau (Radau IIA); the update row x_{n+1} = Y_s then avoids M⁻¹ entirely.

using SOSD
using SOSD.RungeKutta
using StaticArrays
using LinearAlgebra
using KrylovKit
using Printf

const τ = 1.0
const T = 2π

# --- descriptor form (D = 2) -----------------------------------------------
function dae_problem()
    A_f = t -> [-1.0 1.0; sin(t) -1.0]
    B_f = t -> [0.5 0.0; 0.0 0.0]
    c_f = t -> [0.0, 0.0]
    SOSD.LDDEProblem{2, Float64}(SOSD.ProportionalMX(A_f),
        [SOSD.DelayMX(t -> τ, B_f)], SOSD.Additive(c_f))
end

# --- reduced scalar reference ----------------------------------------------
function reduced_problem()
    SOSD.LDDEProblem{1, Float64}(
        SOSD.ProportionalMX(t -> SMatrix{1,1}(-1.0 + sin(t))),
        [SOSD.DelayMX(t -> τ, t -> SMatrix{1,1}(0.5))],
        SOSD.Additive(t -> SVector{1}(0.0)))
end

function mu_dae(p; ε=0.0, s=3)
    prob = dae_problem()
    Mmass = [1.0 0.0; 0.0 ε]
    tab = SOSD.from_rkjl(TableauRadauIIA(s))   # stiffly accurate
    r = ceil(Int, τ / (T / p))
    grid = TimeGrid(collect(range(0.0, T, length=p+1)))
    sys = build_system_matrices(prob, grid, tab, r; mass_matrix=Mmass)
    S = s; BSIZE = (S + 1) * 2
    m = MonodromyMap(prob, grid, tab, sys, p, r, (r + 1) * BSIZE)
    sm = SparseMonodromyMap(m)
    vals, _ = eigsolve(sm, ones(sm.state_size) ./ sqrt(sm.state_size), 1, :LM; tol=1e-12)
    return abs(vals[1])
end

function mu_reduced(p; s=5)
    prob = reduced_problem()
    tab = GL(s)
    r = ceil(Int, τ / (T / p))
    grid = TimeGrid(collect(range(0.0, T, length=p+1)))
    sys = build_system_matrices(prob, grid, tab, r)
    BSIZE = (s + 1) * 1
    m = MonodromyMap(prob, grid, tab, sys, p, r, (r + 1) * BSIZE)
    sm = SparseMonodromyMap(m)
    vals, _ = eigsolve(sm, ones(sm.state_size) ./ sqrt(sm.state_size), 1, :LM; tol=1e-12)
    return abs(vals[1])
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Reference (reduced scalar LDDE, GL5, p=400):")
    mu_ref = mu_reduced(400)
    @printf("  mu_ref = %.14f\n\n", mu_ref)

    println("Singular M = diag(1, 0), Radau IIA s=3 (index-1 DDAE):")
    for p in (25, 50, 100, 200)
        mu = mu_dae(p; ε=0.0)
        @printf("  p=%4d  mu = %.14f   rel.err = %.2e\n", p, mu, abs(mu - mu_ref)/mu_ref)
    end

    println("\nSingular perturbation M = diag(1, 1e-6), Radau IIA s=3:")
    for p in (50, 200)
        mu = mu_dae(p; ε=1e-6)
        @printf("  p=%4d  mu = %.14f   rel.err = %.2e\n", p, mu, abs(mu - mu_ref)/mu_ref)
    end
end
