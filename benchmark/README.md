# Benchmark suite — reproducibility notes

This directory contains every experiment behind the manuscript in `../paper/`.

## How to reproduce everything

```bash
# 1. one-time environment setup (from the repository root)
julia -e 'using Pkg; Pkg.activate("benchmark"); Pkg.develop(path="."); Pkg.instantiate()'

# 2. run all studies (several hours; CSVs appear incrementally in results/)
julia --project=benchmark benchmark/run_all.jl

# 3. render all figures (PNG for review, PDF for the paper)
julia --project=benchmark benchmark/make_figures.jl

# extras not included in run_all.jl:
julia --project=benchmark benchmark/run_classic_sd.jl              # SD-classic vs MFSD parity
julia --project=benchmark benchmark/run_order_implicit_families.jl # Radau IIA / Lobatto IIIA orders
```

## Fair-comparison rules enforced by the harness (`harness.jl`)

- one machine, **BLAS pinned to 1 thread**, environment recorded in
  `results/environment.txt`;
- CPU times are **means over repeated warm-started runs**, standard deviations
  reported and plotted as bands;
- **deterministic start vectors** for all Krylov eigensolves (no RNG in timed paths);
- eigenvalue errors are measured against a reference accepted only when **two
  resolutions of a higher-order method agree to 1e-10 relative**
  (`results/reference_values.csv`), and the reference method is cross-validated
  against `SemiDiscretizationMethod.jl` on every system;
- log–log axes and wide resolution ranges everywhere; fitted slopes are computed
  on the pre-floor error window.

## Studies

| Script | Paper section | Output |
|---|---|---|
| `run_order_verification.jl` | Order verification | `order_verification_mathieu.csv`, `..._slopes.csv` |
| `run_order_implicit_families.jl` | Order verification (Radau/Lobatto) | `order_verification_implicit_families.csv` |
| `run_work_precision.jl` | Work-precision / time-complexity | `work_precision_<system>.csv` |
| `run_sweet_spot.jl` | Accuracy–sparsity sweet spot | `sweet_spot_<system>.csv`, `spectral_corner_mathieu.csv` |
| `run_nonsmooth.jl` | Non-smoothness stress test | `nonsmooth_beam_aaw.csv` |
| `run_classic_sd.jl` | Baseline parity with predecessor | `classic_sd_vs_mfsd.csv` |

Test systems (`harness.jl`): delayed Mathieu (d=2), seasonal scalar model (d=1),
turning with spindle-speed variation (d=2, time-periodic delay, T/τ ≈ 9),
FEM beam with delayed boundary feedback (d=28, τ/T = 1/2, optional act-and-wait
switching). Exact parameters are in the paper's Appendix A and in `harness.jl`.
