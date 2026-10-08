# GPU design note — batched SOSD spectral radii

Branch `gpu` (from `error-estimation` @ ed1424d). Goal: stability charts of linear
time-periodic DDEs (one spectral radius ρ per parameter point) on an NVIDIA GPU, with a
CPU fallback. The GPU kernels also run on CPU threads (`cpu_mode = :kernels`, for
validation without a GPU); the default CPU fallback is the reference solver threaded
over points, because the KernelAbstractions CPU backend measured ~50× slower.

## 1. Where the time goes (measured, CPU, before any GPU work)

2-DOF milling (D = 4), GL(3), T = τ, r = p, single thread, existing CPU path
(local workstation shared with other jobs that day — the ratios are what matter)
(`build_system_matrices` → `SparseMonodromyMap` (sparse LU) → KrylovKit `eigsolve`):

| r = p | N = (r+1)(S+1)D | build | sparse assemble + LU | eigsolve (30 matvecs) | total / ρ |
|---|---|---|---|---|---|
| 100  | 1 616  | 15 ms  | 10 ms  | 3 ms  | ~28 ms  |
| 300  | 4 816  | 91 ms  | 50 ms  | 11 ms | ~150 ms |
| 1000 | 16 016 | 220–520 ms | 210–400 ms | 80–90 ms | ~0.6–0.8 s |

Facts this establishes:

1. **The Krylov iteration is not the bottleneck** (≈ 10–15 %). Assembly of the per-step
   operators (≈ 1000 heap allocations per step) and the sparse LU dominate.
2. The sparse LU is unnecessary: Φ_L is unit block-lower-triangular, so
   `Φ_L⁻¹ Φ_R x` *is* the forward sweep over the stored step blocks (the lazy
   `MonodromyMap`). The GPU path never factorizes anything globally.
3. The eigensolver needs only ~30–60 operator applications; a dominant-eigenvalue
   Krylov–Schur iteration with a 20–30 vector basis is enough.

## 2. What parallelism exists

| stage | independent work items | dependency |
|---|---|---|
| step-operator build | (point b, step n): B·p items | none — embarrassingly parallel |
| one matvec (forward sweep) | rows of one step block (BSIZE = (S+1)D) | **sequential in n** (causal recursion through y_{n−1} and the delayed history) |
| Krylov orthogonalisation | (point, vector) dot products | per Arnoldi step |
| Ritz extraction | (point): small m×m Schur | none (host) |

The sweep has depth p and only BSIZE·D·(1+S·K) flops per step (≈ 640 flops for
D = 4, S = 3). A *single* parameter point therefore cannot fill a GPU: one sweep runs
inside one thread block on one SM, with a barrier per step. Launching one kernel per
step would cost p launches (≈ 5 µs each) per matvec — slower than the CPU (≈ 1 ms per
matvec at r = 1000). **Parallelising one monodromy on the GPU does not pay off at these
sizes; batching many parameter points does.**

## 3. Decision

```
grid level   : parameter points (thousands per launch)            -> blocks
block level  : BG points × R threads per point                     -> workgroup (BG, R)
thread level : the R threads of one point share the rows of each
               step block and synchronise once per step             -> @synchronize
```

* **Build kernel** — one work item per (point, step). Evaluates `A(t,θ)`, `B_k(t,θ)`,
  `τ_k(t,θ)` on the device, factorizes the SD×SD stage matrix in registers/local
  memory (partial pivoting), writes `M_prop`, `M_del`, delay index and interpolation
  weights. Same algebra as `build_system_matrices`.
* **Sweep kernel** (the matvec) — workgroup `(BG, R)`. With R = 1 every thread owns a
  whole point (no barriers, best when there are ≥ ~10⁴ points). With R > 1 the R
  threads of one point first interpolate the delayed states into shared memory, then
  split the BSIZE rows; two barriers per step. R is chosen **automatically** so that
  `B·R` reaches the number of resident threads of the device (CUDA: SMs × 2048),
  capped at BSIZE — many small points → R = 1; few large points → R ≈ BSIZE.
* **Batched Krylov–Schur** — basis `V[b, i, j]` (point index fastest → coalesced
  access for R = 1), CGS2 orthogonalisation, thick restart keeping the dominant Schur
  vectors (conjugate pairs never split), per-point convergence test
  `|h_{m+1}ᵀ s| ≤ tol·|λ₁|`. The m×m Schur/eigen problems run on the host (threaded),
  everything O(N) runs on the device.
* **Batch size** is chosen from the free device memory and the bytes per point
  (operators + history + basis), so one call can take any number of points.
* **Large systems.** The build kernel keeps the SD×SD stage matrix in thread-local
  memory. That is fine for SD ≲ 30 (D ≤ 8 with GL3). For D = 24 (SD = 72) it is
  thread-local *global* memory; it still runs but is the part to watch in the
  benchmark. Fallback option: `build=:cpu` (threaded CPU build with the existing
  dense assembly, then upload) — the user-requested "CPU builds, GPU iterates" mode.

**Common discretization across the batch.** All points of one batch share
(p, r, S, D, K). Typical chart usage with T = (a/b)·τ and a fixed number of steps per
delay gives exactly that: h = T(θ)/p varies per point, p and r do not. Time-varying
delays are fine (indices are per point and per step).

## 4. Float32 vs Float64 (measured — Float32 not recommended)

Consumer/inference GPUs (T4, L4) run FP64 at a fraction of the FP32 rate (T4 GEMM
measured on Colab: 2.89 TFLOPS FP32, 0.25 TFLOPS FP64); A100/H100 run FP64 at 1/2.
The a-priori estimate (round-off ~√p·6·10⁻⁸ per sweep) suggested ~10⁻⁵ on ρ. The
measurement is worse: relative ρ error vs Float64 on the T4 was 4.6·10⁻⁷ (Mathieu),
1.1·10⁻⁵ (SSV), 1.2·10⁻⁴ (2-DOF milling), 1.1·10⁻³ (4-DOF milling); over 4096 milling
points the median was 7·10⁻⁴ with single points off by O(1) (the Krylov–Schur
residual test is meaningless near the Float32 round-off floor of these strongly
non-normal operators). Speed gain: only 1.3× on the T4 (the batch is not FP64-bound).
Decision: Float64 everywhere; `T = Float32` stays available but is documented as
not recommended.

## 5. Stability-chart drivers

* `spectral_radii(prob, θs, tab, p, r; backend)` — brute-force grid (contour / MDBM
  post-processing on the host).
* `boundary_multisection(...)` — replaces per-speed bisection (15 sequential ρ
  evaluations) by k-section: every round evaluates `nsub` depths for *all* speeds in
  one batch; 4 rounds of 15 sub-points give the 2⁻¹⁶ resolution of 16 bisection steps
  with 4 sequential GPU calls instead of 16, and it cannot jump over a narrow lobe that
  bisection would miss between its sample points.

## 6. Scope / non-goals

* Continuous extensions supported in the batched path: Lagrange on the nodes
  {0, c₁…c_S, 1} (all collocation tableaux: Gauss, Radau, Lobatto; generic dense
  output). The special RK4 Hermite extension and the `endpoint` strategy stay CPU-only
  (checked at construction, fails loud).
* No mass matrix / DAE, no additive term (ρ only), no error estimation in the batched
  path (v1).

## 7. Verification (Colab, Tesla T4, Float64)

`gpu/test_gpu.jl`, 30/30 pass — max relative ρ difference to `floquet_analysis`
(sparse map + KrylovKit, tol 1e-13), both operator builds:

| model | D | build on GPU | build on CPU + upload |
|---|---|---|---|
| delayed Mathieu | 2 | 2.1·10⁻¹⁵ | 2.5·10⁻¹⁵ |
| turning, SSV (τ(t), p ≠ r) | 2 | 9.5·10⁻¹⁵ | 1.6·10⁻¹⁵ |
| 2-DOF milling | 4 | 4.5·10⁻¹¹ | 1.1·10⁻¹⁰ |
| 4-DOF milling | 8 | 7.1·10⁻¹¹ | 7.6·10⁻¹¹ |
| 12-DOF milling | 24 | 1.6·10⁻¹⁰ | 8.7·10⁻¹¹ |

The 10⁻¹¹–10⁻¹⁰ level for milling is the conditioning of the dominant multiplier
(non-normal monodromy, nearly coincident x/y modes): KrylovKit itself moves by
~10⁻¹⁰ between Krylov dimensions 30/60/120 on the same operator.

Bugs the GPU run exposed (all fixed, regression-tested on the CPU backend): a
non-isbits tableau struct (CUDA refuses the launch; now checked on the host), a
`BitVector` written from `@threads` (data race), and a too-small Krylov basis for
clustered complex pairs (D = 24: 5 % wrong ρ from an unconverged Ritz value → basis
30/15, straggler retry with 60/30, and a warning if still unconverged).
* CPU path is untouched; the batched path is additive (`src/batched.jl`).
