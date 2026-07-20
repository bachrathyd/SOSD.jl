# Embedded-pair error estimation for SOSD — design plan

Branch: `error-estimation`. Goal: an `error_estimation = true/false` option that
returns, **in a separate output** (original interface untouched), an approximated
error bar for the spectral radius / dominant Floquet multiplier, the mode shape,
and the periodic fixed point — so the user can decide whether to restart with a
higher resolution.

## 1. Concept — the ode23 idea transplanted to the mapping matrix

An embedded RK pair (e.g. Bogacki–Shampine in MATLAB's `ode23`) advances the
same stages with two weight vectors `b` (order q) and `b̂` (order q−1); the
difference estimates the local error. Here we do **not** adapt the step size.
Instead we assemble the one-period solution operator **twice on the same grid**:

- `Φ`  — the user's tableau `(a, b, c)` with its continuous extension (CE),
- `Φ̂`  — the *embedded companion*: same stages `(a, c)`, lower-order weights
  `b̂` and/or a lower-order CE,

and treat `ΔΦ = Φ − Φ̂` as a **matrix perturbation**. Because only the update
row (`b`) and the delay-interpolation weights (CE) change, `Φ̂` shares the whole
stage structure of `Φ`; the perturbation is exactly the classical embedded local
error, propagated consistently through the monodromy operator.

### Companion choice per tableau family (as implemented)

| tableau | Q companion | I companion |
|---|---|---|
| carries a classical pair (BS3 = ode23) | the pair's `b̂` (order q−1), same stages | CE with one interior node dropped |
| collocation (Gauss / Radau IIA / Lobatto IIIA) | **cross-family same-s companion**: Gauss(s) → Radau IIA(s) (order 2s−1), Radau IIA(s) → Lobatto IIIA(s); s = 1 → implicit Euler | — (folded into Q, see below) |
| other explicit RK | drop-node interpolatory weights (order s̃−1) — conservative; prefer a real pair | CE with one interior node dropped |

Two findings shaped this (both verified numerically during development):

1. **No same-node weight swap can be tight for Gauss.** Any distinct `b̂` on
   the Gauss nodes is capped at quadrature order s−1, and every added-sample
   interpolatory rule degenerates (`∫ℓ_new ∝ ∫Pₛ = 0` by Legendre
   orthogonality). The cross-family companion keeps the *same stage count* —
   hence the same stage-augmented state space — while being a complete
   collocation method only ONE order below, which makes the bar tight.
2. **The drop-node CE perturbation is exactly null for collocation**: the
   block data (y_m, Y₁…Yₛ, y_{m+1}) lie ON the degree-s collocation
   polynomial, so any interpolant through ≥ s+1 of those points reproduces it
   identically. The interpolation uncertainty of collocation is therefore
   carried by the cross-family companion (whose abscissae and CE differ),
   not by a separate I channel.

The combined bar is `mu_error = safety_factor · (|δμ_Q| + |δμ_I|)` with
`safety_factor = 2` by default (the standard embedded-estimate margin; the raw
channels are reported unscaled).

## 2. Matrix perturbation analysis

With right/left eigenvectors `Φx = μx`, `yᵀΦ = μyᵀ` (left vector from the new
transpose action of `SparseMonodromyMap`; bilinear pairing, no conjugation):

- **Eigenvalue / spectral radius**, same-(a,c) companions: first-order shift
  `δμ = yᵀ ΔΦ x / (yᵀ x)` per channel (ΔΦ applied operator-wise, `Φx − Φ̂x`).
  Cross-family companions have O(h) stage-row ("gauge") differences that only
  cancel in the eigenvalues, so there the **exact difference** `δμ = μ − μ̂`
  is used (one extra eigsolve). Since `ρ = |μ|`, `|δρ| ≤ |δμ|`. We also
  report the **eigenvalue condition number** κ(μ) = ‖x‖‖y‖/|yᵀx|.
- **Mode shape**: same-(a,c) → first-order bound
  `‖δx‖/‖x‖ ≤ ‖ΔΦx − δμ·x‖ / gap` (gap = min |μ − μ_k| over the other Ritz
  values); cross-family → sine of the principal angle between the dominant
  eigenvectors **restricted to the node (physical) components**.
- **Fixed point** (periodic solution): exact two-method difference — the
  companion's fixed point is solved with its own operator (one extra Krylov
  linsolve) and compared on the node components.

The embedded maps reuse `build_system_matrices` + `SparseMonodromyMap`
verbatim — no new assembly code. Cost with estimation on: ~2–3 extra
assemblies + 1–2 extra eigsolves (roughly 3× a plain run; opt-in).

### What the bar means (honesty clause)

Exactly like ode23, the pair difference estimates the error of the **lower**
order companion; the returned higher-order μ is used ("local extrapolation"),
so the bar is a *conservative* estimate that should upper-bound the true
error. For collocation the companion is only one order class below (Gauss 2s
vs Radau 2s−1), so the over-estimation factor grows only ~h⁻¹–h⁻² even in the
superconvergent regime; where the two methods sit on the same [s+1] floor the
×2 safety factor supplies the coverage margin. The validation study
quantifies all of this.

## 3. API (backwards compatible)

New exported driver, existing workflow untouched:

```julia
sol = floquet_analysis(prob, grid, tableau, r)                    # FloquetSolution
sol, err = floquet_analysis(prob, grid, tableau, r;
                            error_estimation = true,              # ← the option
                            periodic_solution = true, nev = 3,
                            safety_factor = 2.0)                  # default margin
rho          = spectral_radius(prob, grid, tableau, r)            # convenience
rho, bar     = spectral_radius(prob, grid, tableau, r; error_estimation = true)
```

`FloquetSolution`: μ, ρ, multipliers, modes, fixpoint, solver info.
`FloquetErrorEstimate` (the separate output): `mu_error` (combined bar),
`quadrature_error`, `interpolation_error`, signed `delta_mu`,
`eigenvalue_condition`, `mode_error`, `spectral_gap`, `fixpoint_error` +
`fixpoint_delta`, and the embedded-map eigenvalues (diagnostics, opt-in).

Building blocks also exported: `embedded_tableau(tab; quadrature, interpolation)`,
`BS3()` (the ode23 tableau with its true embedded weights), transpose action for
`SparseMonodromyMap`.

Limitations (v1): sparse solver path only (`:lazy` beam-scale later);
single-stage *explicit* Euler has no companion at all (errors out with an
explanation — GL1/implicit Euler are fine via the cross-family ladder);
`endpoint` interpolation strategy unsupported for the I channel.

## 4. Validation study (`benchmark/run_error_estimation.jl`)

Systems: Mathieu (commensurate — I channel off), bio (non-commensurate),
turning SSV (time-varying delay). Methods: GL2, GL3, BS3, RK4. Sweep p
log-spaced; references from the cached two-resolution values.

Figures (`benchmark/make_error_figures.jl`):
1. `error_prediction_<sys>` — true |μ−μ_ref| vs predicted bar vs p, per method.
2. `error_coverage` — ratio bar/true across everything; points ≥ 1 ⇒ the bar
   truly contains the real root.
3. `error_bars_demo` — μ ± bar with the reference line (the user-visible product).
4. `error_fixpoint` — predicted vs true fixed-point error (Mathieu, turning).

Tests (`test/test_error_estimation.jl`): embedded-weight order conditions,
transpose correctness (⟨Φx,y⟩ = ⟨x,Φᵀy⟩ and dense Matrix check), first-order
consistency δμ ≈ μ − μ̂, coverage smoke test, interface compatibility
(flag off ⇒ identical output to the plain path).
