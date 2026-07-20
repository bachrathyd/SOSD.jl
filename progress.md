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
- All SOSD variants: fitted time exponent 1.0 (linear complexity preserved).
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

## Phase 3 — review rounds and submission readiness (2026-07-19)
- Three full review rounds completed: Round 1 (mathematician 22 + engineer 20
  + editor 75 findings) — caught and fixed a real overclaim (2s superconvergence
  restricted to mesh-commensurate delays; measured s+1..2s bracket on
  non-aligned systems), the GEVP statement, recipe and scaling inconsistencies.
  Round 2 (fresh trio) — number synchronization to regenerated figures,
  classic-parity verification (1200x at p=1e3 is real), abstract cut to ~190
  words, 29 consolidated fixes applied. Round 3 — 11-point acceptance
  checklist: ALL PASS; 5 peripheral defects fixed (16-thread chart titles,
  ph-map terminology, wp legend declutter, 2 wording items).
- Paper: 12 pages, 0 errors, JVC header, 18-entry verified bibliography,
  submission kit under paper/submission/ (cover letter, title page,
  highlights, checklist). Remaining before submission: commit hash into the
  reproducibility appendix, ORCID verification, funding AG-mark confirmation.

## 2026-07-20 (phase 4): off-mesh convergence figure + section numbering

- New Figure (order_offmesh, appendix B): controlled visualization of the
  [s+1, 2s] order bracket for non-mesh-commensurate delays. Two panels from
  existing WP data: seasonal model (GL1/GL2/GL3, fits 2.2/2.9/3.8 -- presses
  the s+1 interpolation floor) and SSV turning (GL2/GL3/GL5, fits
  4.0/5.3/7.2 -- up to the 2s cap). Shaded guide bands s+1..2s anchored at
  the finest pre-floor point; every curve stays inside its band. SSV fits
  exclude the ~1e-13 two-resolution-reference plateau (fit window lo=1e-11);
  GL1 dropped from the SSV panel (pre-asymptotic over the whole p-range).
- All fitted slopes match the numbers already quoted in Section 5.2 text.
- Defect found and fixed while verifying the PDF: sagej/SageH sets
  secnumdepth=-2 (unnumbered headings), so all 28 Section~\ref/Appendix~\ref
  cross-references rendered EMPTY ("Section .") -- pre-existing since the
  first draft, missed by all three review rounds. Restored numbering via
  \setcounter{secnumdepth}{2} in the preamble; verified "Section 5.2",
  "Appendix B", lettered appendices all resolve. 12 pages, 0 errors.
