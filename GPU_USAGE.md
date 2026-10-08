# Using the batched (GPU / multi-point) SOSD solver — quick guide for downstream projects

Branch: `gpu` of https://github.com/bachrathyd/SOSD.jl (design + measurements: `GPU_DESIGN.md`).

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/bachrathyd/SOSD.jl", rev = "gpu")
Pkg.add("CUDA")            # only on a machine with an NVIDIA GPU
```

## 1. Describe the problem once, with all parameters in θ

The coefficient functions take `(t, θ)` and return StaticArrays (so they compile for the GPU):
plain top-level functions or callable structs, no globals that change, no allocation.

```julia
using SOSD, StaticArrays
# ẍ + 2ζẋ + (1 + ε cos t) x = w (x(t − τ) − x(t)),   θ = (τ, w)
A(t, θ)   = @SMatrix [0.0 1.0; -1 - 0.5cos(t) - θ[2]  -0.1]
B(t, θ)   = (@SMatrix([0.0 0.0; θ[2] 0.0]),)      # one delay term (K = 1)
tau(t, θ) = (θ[1],)                                # lags at time t (may depend on t)
period(θ) = 2π                                     # coefficient period T(θ)
prob = BatchedLDDE{2, 1}(A, B, tau, period)        # D = 2 states, K = 1 delays
```

All points of one call share `p` (steps per period, h = T(θ)/p) and `r` (delay steps,
`r·h ≥ max lag`). With T = (a/b)·τ and a fixed number of steps per delay this holds
automatically. Points violating `r·h ≥ lag` come back with `flag = 1`.

## 2. Evaluate many points at once

```julia
θs  = [SVector(τ, w) for w in range(0, 1, 200) for τ in range(0.5, 2π, 300)]
res = spectral_radii(prob, θs, GL(3), p, r; backend = CUDABackend())   # or backend = CPU()
res.rho          # ρ per point (same order as θs); res.converged, res.mu, res.flag
```

`backend = CPU()` (default, no GPU) runs the reference solver threaded over points —
start Julia with `-t auto`.

## 3. Pick the precision by the job

| job | settings | accuracy |
|---|---|---|
| boundary refinement, validation | defaults (`T = Float64`) | ρ to ~1e-10 vs the CPU solver |
| **chart pictures (fast)** | `prob = rescale(prob, ω0, s)`, `T = Float32, krylovdim = 10, keep = 5, retry = false, maxiter = 8` | same stable/unstable classification as Float64 (0 of 131 072 points differed on the milling test chart; median \|Δρ\|/ρ ≈ 1e-7) |
| not recommended | `T = Float16` | ≥ 0.4 % of chart points misclassified, not faster |

**Low precision needs a non-dimensional model.** `rescale(prob, ω0, s)` gives the same
multipliers with time t̃ = ω0·t and state x̃ = diag(s)·x. For a mechanical model with
states [positions; velocities] use ω0 = a natural angular frequency and
`s = [ones(n); fill(1/ω0, n)]`. In physical units (ω² ~ 1e7) Float32 was unreliable.

## 4. Boundaries

```julia
rho_of(xs, ys) = spectral_radii(prob, SVector.(xs, ys), GL(3), p, r; backend = CUDABackend()).rho
w_lim = boundary_multisection(rho_of, n_grid, 0.0, w_max; nsub = 15, rounds = 4)
```

All speeds advance together (one batched call per round), 16⁴ resolution in 4 rounds.
For 2-D boundaries with far fewer evaluations see the MDBM option of the web app
(`gpu/web/`).

## Measured speed (2-DOF milling chart, 256×128 points, p = 40)

16 CPU threads (existing solver): 65 s · A100 Float64: 2.4 s · A100 Float32 chart mode:
0.54 s · T4 Float32 chart mode: 1.15 s. Per ρ on the A100: 16–40 µs (Float32),
75–125 µs (Float64) vs 1.8–4.8 ms on 16 CPU threads.

## Colab

`colab/SOSD_GPU_Colab.ipynb`; in the Colab terminal:
`curl -fsSL https://raw.githubusercontent.com/bachrathyd/SOSD.jl/gpu/gpu/colab_run.sh | bash -s -- setup`
then the interactive cell (`gpu/colab/interactive_app.py`).
