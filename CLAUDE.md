# CLAUDE.md — SOSD.jl working guide

Handoff/context file for Claude Code. Read this first every session. It captures the
project, the paper, **how Daniel likes to work**, the benchmark methodology, and the
hard-won gotchas so a fresh session (after `/clear`) is productive immediately.

---

## 1. What this project is

**SOSD = Solution-Operator Semi-Discretization.** A Julia package + manuscript on
**general-order, multiplication-free stability analysis of time-periodic delay
differential equations (DDEs)**. It computes the dominant Floquet multiplier(s) of the
one-period solution operator (monodromy) in **O(p) time** (p = number of steps), with an
arbitrary Runge–Kutta / collocation integrator embedded per step through its Butcher
tableau.

- **Repo:** `C:\Users\Bachrathy\Documents\git\Integration_based_stab_general_order`
  (the folder name is historical; the *package* is `SOSD`).
- **GitHub:** `https://github.com/bachrathyd/SOSD.jl` (public, branch `main`).
- **Predecessor paper (JVC, in press):** `../BD_lin_time_complex_DDE_spectrum_modified.tex`
  — the MFSD method (multiplication-free semi-discretization). **This paper is the
  sequel.** Match its style, grammar, and figure conventions.
- **Single author:** Daniel Bachrathy, Dept. of Applied Mechanics, BME, Budapest.
  bachrathy@mm.bme.hu. Funding line (exact): *"This work was supported by the Hungarian
  Scientific Research Fund (grant number NKFIH-AG-152125)."*

### The core scientific story (know this cold)
- The per-step integrator is a **free design choice**: any explicit RK or implicit
  collocation scheme embeds via `Â = [a; bᵀ]`, `K = Â⊗I_d`, giving a stage-augmented
  residual system with a banded sparse pair `(Φ_L, Φ_R)`; Floquet multipliers come from
  the GEVP `μ Φ_L v = Φ_R v` (Φ_L unit block-lower-triangular → forward-solvable, O(p)).
- **Accuracy–sparsity trade-off / "sweet spot":** cost ∝ p·sᵏ (k≈2–3, the dense
  per-step solve is cubic-ish in stage count s); error ∝ (Δt/T_c)^(order). For a target
  ε there is a **problem-dependent CPU optimum** at *moderate order, many steps* — not
  single-step-super-high-order (its cubic cost kills it).
- **Shannon/Nyquist bound:** you need `N = p(s+1) ≳ c_s · ω_max·T/2π` total samples
  (c_s ≈ 5–10) before superconvergence can even begin. This is the *a priori* recipe for
  choosing (s, p). Below the band the discrete spectrum is O(1) wrong.
- **Convergence order depends on delay/mesh commensurability** (CRITICAL — see §6):
  - **Mesh-commensurate** delay (τ = T = p·Δt, delay lands on grid nodes): Gauss
    collocation gives full **superconvergent 2s**; Radau IIA 2s−1; Lobatto IIIA 2s−2.
  - **Non-commensurate** (delay falls inside steps): the continuous-extension
    interpolation caps the order at **s+1** (Bellen–Zennaro), unless error cancellation
    lifts it. Measured behaviour is the **bracket [s+1, 2s]** — s+1 is the guaranteed
    floor, 2s the ceiling; where a problem lands is a property of the problem.
- **Scope:** smooth systems. Non-smooth coefficients (act-and-wait switching, milling)
  collapse *every* method to first order unless the discontinuity is grid-aligned — so
  nothing is lost by the sparse moderate-order variant. Solving non-smoothness properly
  is explicitly out of scope.

### Four benchmark systems (in `benchmark/harness.jl`)
| name | d | delay | role |
|---|---|---|---|
| `make_mathieu()` | 2 | τ = T = 2π (commensurate) | easy/low-demand smooth validation |
| `make_bio()` | 1 | τ = 2, T = 2π (non-commensurate, "seasonal") | cheap scalar, high order excels |
| `make_turning_ssv()` | 2 | time-varying, T/τ≈9.1 | hard, high-resolution, p≫r |
| `make_beam()` | 28 | τ/T = 1/2 | FEM beam, dense, stiff, act-and-wait option |

---

## 2. Repository layout

```
src/                      SOSD package (module SOSD.jl, utils.jl, RKTableau, MonodromyMap …)
  utils.jl                extract_SDM_system (delay AUTO-detection — see §6), dense path
examples/                 beam_delay_feedback.jl, dae_mass_matrix.jl, …
test/                     package tests (7/7 passing; cross-checked vs SemiDiscretizationMethod.jl)
benchmark/
  Project.toml/Manifest   the benchmark environment (pin!) — run with --project=benchmark
  harness.jl              BenchSystem, make_*, sosd_mu, sdm_mu, reference_mu (cached),
                          time_stats (min-of-reps), matrix_stats, det_x0
  run_*.jl                one study each (see §4)
  make_figures.jl         ALL figures from results/*.csv (independent of the studies)
  extract_paper_numbers.jl SINGLE SOURCE OF TRUTH for every number quoted in the paper
  results/                *.csv (data), *.png (review), reference_values.csv (cache)
  figures/                *.png + *.pdf  (PDFs copied to paper/figures/)
paper/
  main.tex                the manuscript (sagej class, JVC)
  sagej.cls, SageH.bst    SAGE class + bib style
  references.bib          bibliography
  figures/                *.pdf included by main.tex
  main.pdf                compiled (12 pages)
  submission/             cover_letter, title_page, highlights, submission_checklist.md
progress.md               phase log (append a dated entry each work session)
CLAUDE.md                 this file
```

---

## 3. Build / run commands (Windows, PowerShell primary)

**Figures** (fast, CSV-driven, safe to iterate):
```
julia --project=benchmark benchmark\make_figures.jl
```
Regenerates every figure from `results/*.csv`. GKS "glyph missing 179" warnings are
harmless (a superscript ³ in a title). PNGs are for review, PDFs go to `paper/figures/`.

**Build the PDF** — **latexmk is BROKEN here** (MiKTeX can't find perl). Use pdflatex
directly, twice (refs/labels need two passes; the `.bbl` is committed so no bibtex step
needed unless references.bib changed):
```
cd paper; pdflatex -interaction=nonstopmode main.tex; pdflatex -interaction=nonstopmode main.tex
```
Check `main.log` for `^!` errors and `Output written on main.pdf (NN pages …)`.

**Recompute a benchmark study** (slow; see §5 for the timing rule):
```
julia -t 1 --project=benchmark benchmark\run_work_precision.jl
```

**After any benchmark rerun:** run `extract_paper_numbers.jl` and re-sync every quoted
number in `main.tex` to its output. Do not hand-edit numbers.

---

## 4. Benchmark studies (`benchmark/run_*.jl`)

- `run_order_verification.jl` — order vs p on Mathieu (Gauss + explicit), fitted slopes.
- `run_order_implicit_families.jl` — Radau IIA / Lobatto IIIA orders on Mathieu.
- `run_work_precision.jl` — ε and CPU vs p, per system → the 3-panel Fig 3/9/10/11.
  **CPU-capped at 2 s on t_min, p up to 1e6** (beam capped 2¹⁶ for RAM).
- `run_sweet_spot*.jl` — (s,p) grid, CPU-to-target U-curves + structure.
- `run_p_refinement_cliff.jl` / `run_s_complexity*.jl` / `run_sc_topup*.jl` — error vs
  N=p(s+1) at fixed p (the cliff, Fig 6). `run_sc_topup_mathieu_small.jl` adds p=2,4,6,8.
- `run_nonsmooth.jl` — act-and-wait beam, grid-aligned vs off-grid.
- `run_stability_charts.jl` — brute-force charts (Mathieu, SSV turning), **multithreaded**
  (`julia -t 16`), colored by log spectral radius, MDBM boundary refinement.
- `run_dense_ph_grid.jl` — the (s,p) CPU/error maps (Fig 8 etc.).

---

## 5. THE TIMING RULE (do not violate)

The work-precision and sweet-spot studies **measure CPU time**. For fair, low-scatter
numbers:
- BLAS is pinned to **1 thread**; run Julia with **`-t 1`**.
- `time_stats` uses **min-of-repeats** (BenchmarkTools practice), robust to GC/OS spikes.
- **Never run two CPU-heavy Julia processes at once during a timing study**, and don't
  run `make_figures.jl` or `pdflatex` while a timing study is measuring — it pollutes the
  numbers. Kill/queue other work first. (This is why the Fig 6 top-up was run *before*,
  not during, the work-precision recompute.)

Stability-chart generation is the opposite — those are **multithreaded** (`-t 16`) and the
title reports the true wall time. Keep the two worlds separate.

References are computed by two-resolution agreement and **cached** in
`results/reference_values.csv` — don't delete it; recomputation is expensive.

---

## 6. Correctness lessons / bugs already fixed (don't regress)

- **Delay was hardcoded** (`tau_f = t -> 2π`) in `extract_SDM_system` — milling/bio charts
  were computed with the wrong delay. Now **auto-detected** by probing the rhs with a
  recording history function, per-lag masked, with a fail-loud bounds check. Never
  reintroduce a hardcoded delay.
- **2s superconvergence overclaim** — caught by the round-1 mathematician. 2s holds
  ONLY for mesh-commensurate delays; the honest claim is the **[s+1, 2s] bracket**.
  Fig 7 (Appendix B, `order_offmesh`) visualizes it with guide bands.
- **GEVP orientation** (`μ` on which side) was misstated twice — verify the direction
  against the code whenever you touch that equation.
- **StaticArrays compile-time blowup** for high-stage GL tableaux (GL(s) with s ≳ 24 is
  pathologically slow to compile). Cap s in cliff/s-complexity top-ups at ~18–20.
- **Dense path** auto-switches when `S*D > threshold` or **`D > 12`** or a mass matrix is
  present (heap `Matrix` + LU instead of `SMatrix` inverse) — needed for the beam (D=28).
- **secnumdepth** — `sagej.cls` sets `\setcounter{secnumdepth}{-2}` (unnumbered
  headings), which makes every `Section~\ref`/`Appendix~\ref` render EMPTY. The preamble
  now forces `\setcounter{secnumdepth}{2}`. Keep it.
- Stale figure-derived numbers → the reason `extract_paper_numbers.jl` exists. Sync after
  every rerun.

---

## 7. HOW DANIEL LIKES TO WORK (read this — it saves round-trips)

### Communication style
- Bundles **many change requests in one message**; emphatic punctuation (`!!!`, `RIGHT?!`).
  Fast typing with typos (e.g. "dealy"=delay, "resiolution", "figrue") — **parse intent
  generously**, don't ask for spelling clarification.
- Wants **honest status**, not optimism. If something isn't done (e.g. a recompute still
  running), say so plainly and explain why. He values the reasoning, not just the result.
- Cares about **cost/efficiency**: clear context at phase boundaries, keep bulk output
  (PNG reads, long Julia logs) out of the transcript (redirect to a log, grep it),
  delegate read-heavy work to subagents. See §9.

### Figures — his standing conventions (apply by default)
- **All axis labels as LaTeX strings** (`L"..."`) — everywhere, every panel.
- **Cap nonsense error values at `3·10⁰`** on ε axes (pre-resolution explicit-method
  blowups reach 1e43 and carry no information beyond "unstable"). `ylims=(1e-16, 3.0)`.
- **Page-wide (`figure*`)** when a figure is too small in two-column; Fig 1 (matrix
  sparsity) definitely page-wide.
- **Colorbar + labeled contour lines** for 2D maps; when contrasting two variables use
  color for one and labeled contours for the other (e.g. CPU color + ε contour).
- **Timing via BenchmarkTools / min-of-reps; NO uncertainty ribbons** on CPU curves —
  state in the caption that timing is accurate (<~5%). He explicitly dislikes the ribbons.
- **Stability charts:** brute-force, colored by **log spectral radius**, **blue = stable,
  red = unstable** (`cgrad(:RdBu, rev=true)`), **100×100** grid, a **higher-order MDBM
  boundary** overlaid (pattern from the `InterpolatedNyquist.jl` gallery, ~5–10 s), and
  the **full CPU time shown in the title** (with the real thread count).
- Mark a **single optimum with a star** when showing a CPU-optimal setup.
- When he gives a **specific p-set** ("use p=1,2,4,6,8,10"), honour it exactly and update
  the caption to match.
- He iterates on figures visually — expect "make it page-wide", "the dots too small",
  "I can't see the colorscale", "it ends too early". Read the PNG, diagnose, adjust.

### Paper / documentation
- **Short paper**; push extra examples (stability charts, extra work-precision, DAE) into
  the **appendix**. Main text stays focused.
- **Honest about prior art** — explicitly state what is *not* new (MFSD assembly,
  collocation itself). The contributions are the general-order embedding, the fair
  order-vs-step benchmarking, the measured accuracy–sparsity trade-off, the practical
  recipe.
- **Match the predecessor paper's style/grammar/voice** (`BD_lin_time_complex_DDE…`).
- Journal target: **Journal of Vibration and Control (SAGE)**, `sagej` class — Q1, fast
  review, accepts near-published sequel content. **No predatory journals, ever.**
- Every quoted number must trace to `extract_paper_numbers.jl`.

### Review process (when he asks to "review")
- **Multiple independent adversarial rounds** (≥3), each with distinct personas
  (mathematician, engineer, …), thorough on math, style, figures, grammar, the big
  picture. Then a **cleanup pass**, then the **GitHub update**. Report findings per round.

---

## 8. Git conventions

- Branch `main`, remote `bachrathyd/SOSD.jl`. Commit + push when he says so.
- **Multi-line commit messages: use a message file** (`git commit -F <file>`). PowerShell
  5.1 mangles inline `@'…'@` here-strings passed to `git -m` (splits on spaces/quotes).
- Regenerating figures rewrites every PDF with a new embedded timestamp → **byte churn
  even when the plot is identical**. Before committing, `git restore` the figure PDFs that
  are timestamp-only diffs; commit only the genuinely changed figures.
- End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- `.gitignore` should keep LaTeX aux files out; watch for CRLF/LF warnings (harmless).

---

## 9. Cost-efficiency habits (his session was ~$1k-equivalent of API tokens)

- **Context grows and never shrinks** — cache-read cost is ~97% of a long session. Clear
  at phase boundaries; `progress.md` + this file are the handoff artifacts.
- **Keep bulk output out of the transcript:** redirect Julia to a log
  (`… > results/x.log 2>&1`) and grep it; don't let 20k-token benchmark stdout or big PNGs
  land in context uncounted.
- **Delegate read-heavy fan-out to subagents** (their context dies with them; only the
  report returns) — reviews, "read all files and report", log analysis.
- **Batch** figure/paper changes into fewer, larger turns.

---

## 10. Current state & open items (as of 2026-07-21)

- Package verified (tests pass, cross-validated). Paper: **12 pages, 0 errors**, JVC
  header, verified bibliography, submission kit under `paper/submission/`.
- Figures: order verification, off-mesh convergence bracket (Fig 7/App B), work-precision
  ×4, sweet-spot ×3, (s,p)-maps, cliff (Fig 6), spectral corner, non-smooth beam, two
  brute-force stability charts.
- **In flight this session:** work-precision recompute with the **2 s CPU cap + p→1e6**
  retune (so cheap methods extend to the 2 s wall in the ε–T_CPU panel instead of ending
  early). When it finishes: regenerate figures, verify the WP panels reach ~2 s, rebuild
  PDF, commit + push.
- **Before submission** (`paper/submission/submission_checklist.md`): insert final commit
  hash into the Reproducibility appendix (remove `\TODOnum`); verify ORCID
  (0000-0002-7268-2999) on the title page; decide repo visibility; tag a release
  (`v0.2.0-submission`).
