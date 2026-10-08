# SOSD in the browser (WebGPU)

**Live: https://bachrathyd.github.io/SOSD.jl/webgpu/**

A self-contained static page that computes stability charts of linear time-periodic DDEs

    ẋ(t) = A(t) x(t) + B(t) x(t − τ(t)),   A, B, τ periodic with period T

**on the viewer's own GPU** via WebGPU, in Float32, with live sliders on the model parameters.
Port of the batched solver of SOSD.jl (`src/batched.jl`): per parameter point, p Gauss–Legendre
collocation steps over one period (the delayed state from the continuous extension of the stored
steps), m Arnoldi steps on the monodromy operator and Francis QR on the m×m Hessenberg matrix → ρ.
No build step, no server-side computation, no external CDN.

GPU layout (`sosd.wgsl`): a `prep` kernel (one thread per point and step) builds every step
matrix W_n ([y_{n+1}; stages] = W_n [y_n; delayed values]) and the delayed-lookup records once;
the Arnoldi kernel then needs only p small mat-vecs per matvec. It runs LPP lanes per point
(chosen per batch: several for the small MDBM batches, one for large grids), and every per-point
array is stored in chunks of points, element-major inside a chunk (a point-fastest layout over
the whole batch was up to 9× slower: TLB misses). AMD Vega 8 iGPU (Ryzen 7 5800H), Chrome:
Mathieu 128×64 grid ≈ 0.3 s (≈ 31 µs/ρ), 2-DOF milling ≈ 1.3 s (≈ 160 µs/ρ); a 4-point batch
10–13 ms.

Two ways to get the chart (both can be on):
* **Brute force** — ρ on an nx × ny grid: the colour map of log₁₀ ρ and the boundary from it.
* **MDBM** (multi-dimensional bisection method) — a small initial grid, then only the cells
  whose corners change the sign of log ρ are split (iterations 0–6), plus a neighbour check:
  the ρ = 1 boundary at the resolution of (initial grid × 2^iterations) with a fraction of the
  evaluations. A lobe narrower than the initial grid spacing can be missed — refine the initial grid.

### Speed (AMD Vega 8 iGPU of a Ryzen 7 5800H, Chrome, GPU time per ρ from timestamp queries, 256×128 grid)

| example, default (s, p, m) | µs per ρ | of which step-matrix build |
|---|---|---|
| delayed Mathieu, GL3, p = 16, m = 6 | 3.7 | 0.4 |
| 2-DOF milling, GL3, p = 40, m = 8 | 42 | 14 |
| turning SSV, GL3, p = 200, m = 6 | 18 | 5 |

The defaults come from a scan against high-resolution references (`validate/bench.mjs`, `scan`):
for Mathieu, GL3 with p = 16, m = 6 is as fast as GL2 with p = 20, m = 7 but has 0 instead of 15
misclassified points of 8192; m = 6 gives the turning-SSV chart of m = 12 at half the cost; milling
needs m = 8. The Krylov dimension matters: the Gram–Schmidt cost grows like m² N and dominates at
small p.

**Float16.** The Krylov basis can be stored in Float16 (option; arithmetic stays Float32): ρ to
about 1e-3, 5–25 % faster depending on the model. Float16 for the sweep history or the step
matrices was tested and dropped: the rounding of every one of the p steps accumulates in the
monodromy (median |Δρ|/ρ 1–4 %, up to 6 % of the points misclassified). The Nyquist count of
InterpolatedNyquist tolerates Float16 because an integer winding number is robust to small phase
errors; ρ near 1 is not.

| file | content |
|---|---|
| `index.html` | page, layout, styles |
| `app.js` | UI: model text, parameters (sliders, X / Y axes), method, discretization, chart, `?validate=1`, `?bench=1` |
| `expr.js` | the model language (parameters `ζ = 0:0.05 @ 0.011`, constants, helpers, `T`, `τ`, `A = [..; ..]`, `B`), Float64 host evaluator, WGSL generation |
| `engine.js` | WebGPU host code: Gauss–Legendre tableaux, pipelines (cached per model), banded dispatches below the GPU watchdog |
| `sosd.wgsl` | the per-point solver: collocation sweep, Arnoldi, Francis QR (`hqr`) |
| `hqr.js` | the same Francis QR on the host (unit-tested against Julia `eigvals`: `validate/hqr_test.mjs`) |
| `mdbm.js` | MDBM boundary refinement, batched per iteration, neighbour tracing |
| `cpu.js`, `cpuworker.js`, `cpusolver.js` | Float64 CPU solver (twin of `sosd.wgsl`) in a web-worker pool |
| `examples.js` | delayed Mathieu, 2-DOF milling, turning with spindle-speed variation — written like their Julia twins in `test/batched_models.jl` |
| `validate/bench.mjs` | console benchmark: GPU µs per ρ (prep / main), accuracy vs a kept run, (s, p, m) scans against a reference |
| `stats.js` | anonymous usage counts (GoatCounter site `sosdgpu`: page views, examples and features used; no cookies, never the model text; `GOATCOUNTER = ''` turns it off) |
| `validate/reference.jl` | Float64 CPU reference (SOSD.jl) on 16×10 grids → `validate/ref.json`, compared by `?validate=1` |

Validation (AMD Radeon iGPU, Chrome, Float32, Krylov m = 24) vs the Float64 CPU solver:
median |Δρ|/ρ 4·10⁻⁷ (Mathieu), 5·10⁻⁷ (milling), 2·10⁻⁶ (turning SSV), worst 3·10⁻⁵,
**0 misclassified** points out of 3 × 160.

Interaction: a running chart is finished (not restarted) while a slider moves, then the newest
values are computed, so the chart keeps updating while dragging; a structural change or Stop
cancels it, and so does the time limit (default 5 s, editable). The old chart stays until the new
one is complete. The MDBM neighbour check runs once at the end, by default on the CPU: a Float64
twin of the kernel (`cpusolver.js`, equal to SOSD.jl to 1e-15: `node validate/cpu_test.mjs`) in a
pool of web workers (`cpu.js`), because those few, almost sequential evaluations are faster there
than GPU round trips. *High-resolution image*: HD … 8K PNG with one ρ per plot pixel (points are
generated band by band on the host, so 8K = 25 M ρ fits), with the expected time estimated from
the last brute-force chart.

Chart: mouse wheel zooms about the cursor, left-drag zooms into a box, middle- or shift-drag
pans, two fingers pinch / pan on touch screens, double-click resets; the new range is written
back into the model text and recomputed. The side panel width is draggable.

Run locally: `python -m http.server` in this folder, open http://localhost:8000
(WebGPU needs a secure context: https or localhost; Chrome / Edge, or Firefox / Safari with WebGPU).

Publishing: the branch `gh-pages` holds a copy of this folder as `webgpu/` (without the Julia
validation project, with `validate/ref.json`) plus a redirecting `index.html`; update it by
copying the files over and committing on `gh-pages`.
