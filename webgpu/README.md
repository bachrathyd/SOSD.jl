# SOSD in the browser (WebGPU)

A self-contained static page that computes stability charts of linear time-periodic DDEs

    ẋ(t) = A(t) x(t) + B(t) x(t − τ(t)),   A, B, τ periodic with period T

**on the viewer's own GPU** via WebGPU, in Float32, with live sliders on the model parameters.
Port of the batched solver of SOSD.jl (`src/batched.jl`): one GPU thread per parameter point
runs p Gauss–Legendre collocation steps over one period (the delayed state from the continuous
extension of the stored steps), m Arnoldi steps on the monodromy operator and Francis QR on the
m×m Hessenberg matrix → ρ. No build step, no server-side computation, no external CDN.

Two ways to get the chart (both can be on):
* **Brute force** — ρ on an nx × ny grid: the colour map of log₁₀ ρ and the boundary from it.
* **MDBM** (multi-dimensional bisection method) — a small initial grid, then only the cells
  whose corners change the sign of log ρ are split (iterations 0–6), plus a neighbour check:
  the ρ = 1 boundary at the resolution of (initial grid × 2^iterations) with a fraction of the
  evaluations. A lobe narrower than the initial grid spacing can be missed — refine the initial grid.

| file | content |
|---|---|
| `index.html` | page, layout, styles |
| `app.js` | UI: model text, parameters (sliders, X / Y axes), method, discretization, chart, `?validate=1`, `?bench=1` |
| `expr.js` | the model language (parameters `ζ = 0:0.05 @ 0.011`, constants, helpers, `T`, `τ`, `A = [..; ..]`, `B`), Float64 host evaluator, WGSL generation |
| `engine.js` | WebGPU host code: Gauss–Legendre tableaux, pipelines (cached per model), banded dispatches below the GPU watchdog |
| `sosd.wgsl` | the per-point solver: collocation sweep, Arnoldi, Francis QR (`hqr`) |
| `hqr.js` | the same Francis QR on the host (unit-tested against Julia `eigvals`: `validate/hqr_test.mjs`) |
| `mdbm.js` | MDBM boundary refinement, batched per iteration |
| `examples.js` | delayed Mathieu, 2-DOF milling, turning with spindle-speed variation — written like their Julia twins in `test/batched_models.jl` |
| `stats.js` | optional anonymous usage counts (GoatCounter); off unless configured |
| `validate/reference.jl` | Float64 CPU reference (SOSD.jl) on 16×10 grids → `validate/ref.json`, compared by `?validate=1` |

Validation (AMD Radeon iGPU, Chrome, Float32, Krylov m = 24) vs the Float64 CPU solver:
median |Δρ|/ρ 3·10⁻⁷ (Mathieu, milling), 1·10⁻⁶ (turning SSV), worst 3·10⁻⁵, **0 misclassified**
points out of 3 × 160.

Run locally: `python -m http.server` in this folder, open http://localhost:8000
(WebGPU needs a secure context: https or localhost; Chrome / Edge, or Firefox / Safari with WebGPU).
