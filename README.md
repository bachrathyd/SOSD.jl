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

## Batched stability charts on GPU (or CPU threads)

For parameter sweeps (one spectral radius per point, thousands of points) use the
batched API. All points of one call share the discretization (tableau, `p` steps per
period, `r` delay steps); each point has its own period `T(θ)`, lags and coefficients.
The coefficient functions take `(t, θ)` and return StaticArrays, so they compile for
the GPU — use plain top-level functions or callable structs, no global mutable state.

```julia
using SOSD, StaticArrays
using CUDA                       # optional; without it use backend = CPU()

# ẍ + 2ζẋ + (1 + ε cos t) x = −w (x(t) − x(t − τ)),  θ = (τ, w)
A(t, θ)   = @SMatrix [0.0 1.0; -1 - 0.5cos(t) - θ[2]  -0.1]
B(t, θ)   = (@SMatrix([0.0 0.0; θ[2] 0.0]),)          # one delay term (K = 1)
tau(t, θ) = (θ[1],)                                   # lags at time t
period(θ) = 2π                                        # coefficient period
prob = BatchedLDDE{2, 1}(A, B, tau, period)           # D = 2 states, K = 1 delays

θs = [SVector(τ, w) for w in range(0, 1, 200) for τ in range(0.5, 2π, 300)]
p  = 200                                              # steps per period, h = T(θ)/p
r  = p                                                # delay steps, r·h ≥ max lag
res = spectral_radii(prob, θs, GL(3), p, r; backend=CUDABackend())
res.rho          # spectral radii, same order as θs (also res.mu, res.converged)
```

- `backend = CUDABackend()` runs the batched kernels on the GPU; `backend = CPU()`
  (default) runs the reference solver threaded over points — the CPU fallback.
- `T = Float32` halves the memory traffic; ρ then carries ~10⁻⁵ relative error (fine
  for chart pictures, not for refinement) — see `GPU_DESIGN.md`.
- Batch size, thread mapping (one thread per point for many small systems, a thread
  block per point for few large ones) and operator assembly (`build = :device` for
  S·D ≤ 32, else threaded CPU assembly + upload) are chosen automatically.
- `res.flag[i] == 1` marks points whose lag exceeds `r·h`: increase `r`.

**Boundary curves.** `boundary_multisection` replaces per-speed bisection: every round
evaluates `nsub` depths for *all* speeds in one batched call.

```julia
rho_of(ns, ws) = spectral_radii(prob, SVector.(ns, ws), GL(3), p, r; backend=CUDABackend()).rho
w_lim = boundary_multisection(rho_of, n_grid, 0.0, w_max; nsub=15, rounds=4)  # 16⁴ resolution
```

A full example (textbook milling, delayed Mathieu, turning with spindle-speed
variation) is in `test/batched_models.jl`; `gpu/test_gpu.jl` and `gpu/bench_gpu.jl`
are the CUDA test and benchmark, and `colab/SOSD_GPU_Colab.ipynb` runs both on a
Colab GPU. For one point with error bars, convert with `LDDEProblem(prob, θ)` and use
`floquet_analysis`.

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
