# SOSD.jl — Solution-Operator Semi-Discretization

A high-performance Julia package for the stability analysis of time-periodic Delayed Differential Equations (DDEs).

SOSD generalizes the multiplication-free semi-discretization method (MFSD) to
arbitrary per-step order: the one-period **solution operator** is represented as a
sparse, banded pair (Φ_R, Φ_L) whose per-step blocks embed any Runge–Kutta or
collocation scheme through its Butcher tableau — from the classical
piecewise-constant semi-discretization step up to superconvergent Gauss stages —
while keeping O(p) time complexity.

*(Package formerly developed under the working name MFCM.)*

## Features
- **O(p^1) Time Complexity:** Avoids explicit construction of dense monodromy matrices
  (verified vs. the traditional monodromy build, which measures ~O(p^2.9)).
- **Arbitrary-order collocation steps:** Gauss–Legendre `GL(s)` with verified
  superconvergent order `2s` (GL5 → observed order 9.96), plus Radau IIA (`2s−1`)
  and Lobatto IIIA (`2s−2`) via `from_rkjl`. Explicit RK1–RK8 are available, but
  their observed order is capped by their dense-output (continuous-extension) order.
- **Time-periodic delays** `τ(t)` supported (e.g. spindle-speed variation).
- **Large systems:** automatic heap-allocated assembly path for `S*D > 32`
  (`build_system_matrices_dense`) — FEM-scale models (d ≈ 30+) work out of the box.
- **Lazy Operator:** `LinearMaps.jl` + `KrylovKit.jl` memory-efficient route.
- **Explicit Sparse Support:** banded `(Φ_R, Φ_L)` pair with unit block-lower-triangular
  `Φ_L` (`SparseMonodromyMap`), validated against the lazy operator to ~1e-16.
- **Automatic Linearization:** extract `A(t)`, `B_k(t)` from a DDE RHS function; delay
  lags are auto-detected from the history calls, or passed explicitly via
  `extract_SDM_system(rhs, p, Val(D); delays=[τ1, ...])`. Out-of-window delayed
  lookups raise an error instead of silently clamping.
- **MDBM Integration:** Ready for multi-dimensional stability chart generation.
- **Embedded-pair error estimation** (`error_estimation = true`): ode23-style
  error bars for the spectral radius / dominant multiplier, the mode shape and
  the periodic fixed point, from a lower-order companion of the mapping matrix
  (matrix perturbation analysis + cross-family collocation pairs). The bar is
  returned as a **separate output**, so the plain interface is unchanged:

  ```julia
  rho             = spectral_radius(prob, grid, GL(3), r)
  rho, rho_bar    = spectral_radius(prob, grid, GL(3), r; error_estimation=true)
  sol, est        = floquet_analysis(prob, grid, GL(3), r; error_estimation=true,
                                     periodic_solution=true)
  # est.mu_error, est.mode_error, est.fixpoint_error, est.eigenvalue_condition, ...
  ```

  Validated on all benchmark systems (`benchmark/run_error_estimation.jl` +
  `make_error_figures.jl`); design notes in `ERROR_ESTIMATION_PLAN.md`.

## Benchmark suite & paper
`benchmark/` contains the full fair-comparison harness (order verification,
work-precision, sweet-spot, non-smooth stress test, SD-classic parity) used by the
manuscript in `paper/` — see `benchmark/run_all.jl` and `benchmark/make_figures.jl`.
Baselines are cross-validated against
[SemiDiscretizationMethod.jl](https://github.com/bachrathyd/SemiDiscretizationMethod.jl)
on all four test systems (delayed Mathieu, seasonal scalar model, SSV turning,
FEM beam with delayed boundary feedback).

## Installation
```julia
using Pkg
Pkg.activate(".")
```

## Quick Start
```julia
using SOSD
using StaticArrays
using KrylovKit

# Define a Delayed Mathieu Equation
function mathieu_rhs(u, h, p, t)
    x1, x2 = u
    hist = h(p, t - 2π) # Delay tau = 2pi
    du1 = x2
    du2 = -0.1 * x2 - (3.0 + 1.5 * cos(t)) * x1 - 0.5 * hist[1]
    return @SVector [du1, du2]
end

# 1. Extract linear system
prob = extract_SDM_system(mathieu_rhs, nothing, Val(2))

# 2. Setup discretization
p_steps = 100
T = 2π
grid = TimeGrid(collect(range(0.0, T, length=p_steps+1)))
tableau = GL2Tableau() # 4th order

# 3. Precompute system matrices
sys_mats = build_system_matrices(prob, grid, tableau, p_steps)

# 4. Create Monodromy Map
m = MonodromyMap(prob, grid, tableau, sys_mats, p_steps, p_steps, (p_steps+1)*6)

# 5. Solve for Floquet Multipliers
vals, _ = eigsolve(m, rand(m.state_size), 1, :LM)
println("Max Multiplier: ", abs(vals[1]))
```

## Engineering Case Studies
Verified implementations and stability charts for:
1. **Delayed Mathieu Equation**
2. **1-DOF Regenerative Milling**
3. **Seasonal Maturation (Biological Model)**

See `examples/` for details.

## Verification
- **Convergence:** Verified $O(h^4)$ for GL2 and $O(h^6)$ for GL3.
- **Complexity:** Verified $O(p^1)$ scaling.
- **Accuracy:** Verified against explicit sparse matrix solutions (error < 1e-15).

## References
1. Bachrathy, D., & Stepan, G. (2012). Improved semi-discretization method for periodic systems with delay.
2. Multiplication-Free Collocation Method for stability analysis of DDEs.
