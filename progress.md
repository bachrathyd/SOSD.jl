# Project Progress Tracking

## Phase 2 — verification, completion and paper (Claude, 2026-07-18)

### Fixed
- [x] **Critical delay bug**: `extract_SDM_system` hardcoded `τ = 2π`; the
      milling and biological charts and two tests were silently computed with a
      clamped, wrong delay. Delays are now auto-detected from the RHS history
      calls (or passed via `delays=[...]`), each `B_k` extracted with per-lag
      masking, and out-of-window delayed lookups now raise an error instead of
      clamping.
- [x] Cross-validation vs `SemiDiscretizationMethod.jl` on ALL systems:
      mathieu 9.5e-4 @ p=50, bio 2.6e-4, beam 1.2e-4, turning SSV
      converging jointly to μ = 4.47617 (gap 1.5e-7 @ p=4000).

### Added
- [x] `build_system_matrices_dense` + `SystemMatricesDense`: heap-allocated
      assembly path (auto-selected for S·D > 32) — FEM-scale systems and very
      high collocation orders now work without StaticArrays compile blowup.
      Verified identical to the static path to ~1e-16.
- [x] `examples/beam_delay_feedback.jl` — predecessor Appendix C system
      (D = 28, act-and-wait), the missing 4th test case.
- [x] `benchmark/` — full fair-comparison suite: harness (BLAS pinned,
      deterministic vectors, two-resolution-verified references, mean±σ
      timing), order verification, work-precision × 4 systems, sweet-spot
      (p×s + matrix structure), single-step spectral corner, non-smooth
      stress test, SD-classic parity.
- [x] `paper/` — SAGE `sagej` manuscript per the handoff plan; compiles clean.

### Verified results (so far)
- Test suite 7/7 on Julia 1.12.4 (with correct delays).
- Gauss superconvergence 2s: GL2/GL3/GL5 slopes 3.996/5.985/9.962; GL8 at
  1e-16 floor from p = 16. RadauIIA → 2s−1; LobattoIIIA → 2s−2.
- Explicit RK capped by continuous-extension order (RK4 → 3.3, RK5 → 2.1):
  the delay-interpolation order requirement demonstrated; structural argument
  for collocation steps (matching-order CE for free).
- Explicit steps unstable on the stiff FEM beam until a CFL-like p bound
  (errors 1e20–1e43); A-stable Gauss stages unaffected.
- All MFCM variants: fitted time exponent 1.0 (linear complexity preserved).
- SD-classic ≈ O(p^2.86) vs MFSD O(p^1.0) — predecessor parity; μ agreement
  1e-15; 2500× speedup at p = 1778.
- Single-step GL20 (p = 1) reproduces converged spectrum to 3e-9 —
  collocation-equivalence demonstrated.

### Running / pending
- [ ] Benchmark suite completion (beam WP → sweet-spot → non-smooth).
- [ ] Regenerate all figures from final CSVs; fill red placeholders in paper.
- [ ] Final proofread pass + reproducibility appendix with commit hash.

## Phase 1 — initial development (Gemini, earlier)
(See git history of this file for the original Phase 1 log: lazy + sparse
monodromy operators, tableau library, three engineering cases, initial
work-precision studies. Note: Phase 1 milling/bio stability charts were
affected by the delay bug fixed above and must be regenerated before use.)
