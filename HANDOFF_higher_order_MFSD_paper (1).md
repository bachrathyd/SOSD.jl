# Handoff Spec — Follow-up Paper: Higher-Order, Multiplication-Free Stability Analysis of Time-Periodic DDEs

**Author:** Daniel Bachrathy, Department of Applied Mechanics, Budapest University of Technology and Economics (BME)
**Target package:** `SemiDiscretizationMethod.jl` (extend the existing MFSD implementation)
**This document is the full brief for an autonomous coding+writing agent (Claude Code / Fable 5).** It should be sufficient to: implement the methods, run the numerical tests, produce the figures, draw the conclusions, and draft the full journal manuscript for human review.

---

## 0. How to use this document

You (the agent) are picking up a research program that already has one published paper. That paper — the **MFSD paper** — is the *predecessor*. This new paper is the *sequel*. Read Section 1 to understand what already exists and must NOT be re-claimed as novel. Read Section 2 for the actual thesis of the new paper. Sections 3–7 are the implementation, experiments, comparisons, and writing plan. Section 8 lists the traps that make this kind of paper fail review.

Work in this order:
1. Reproduce the predecessor's baseline (SD and MFSD) so all comparisons are apples-to-apples in one codebase.
2. Implement the new integrator families (explicit RK, implicit collocation/IRK) inside the multiplication-free assembly.
3. Build the work-precision and time-complexity harness (log-log, with variance).
4. Run the two contrasting demonstration regimes (low-order-suffices vs high-order-needed).
5. Only then write the manuscript.

Do not write prose before the numbers exist. Every claim in the paper must be backed by a generated figure or table.

---

## 1. What already exists (the predecessor MFSD paper) — DO NOT re-claim

The prior paper is: *"Linear Time Complexity Analysis of Time-Periodic Delayed Systems with Multiplication Free Semi-Discretization Method"* (Bachrathy, Journal of Vibration and Control, 2026). Its established contributions — treat these as **prior art / given foundation**:

- **The system.** Linear time-periodic DDE in first-order form:
  `ẋ(t) = A(t) x(t) + B(t) x(t−τ)`, with `A(t), B(t)` T-periodic, `x ∈ ℝ^d`, constant (or periodically modulated) delay `τ`.
- **Semi-discretization (SD) basics.** Uniform grid `Δt = T/p`; `r = ⌈τ/Δt⌉`; discretized state segment `y_i = [x_i, x_{i−1}, …, x_{i−r}]ᵀ`. One-step map `x_{i+1} = P_i x_i + R_i x_{i−r}`, with `P_i = exp(A(t_i)Δt)`, `R_i = (exp(A(t_i)Δt) − I) A(t_i)^{-1} B(t_i)` for zeroth-order; higher-order forms per Insperger 2008.
- **The MFSD trick (already published).** Instead of forming the monodromy matrix `Φ` by multiplying `p` one-step matrices `C_i` (the expensive part), write every step as a residual `0 = I·x_{i+1} − P_i x_i − R_i x_{i−r}`, stack all `p` steps into one big sparse banded system over the extended vector `q` spanning `[t−τ, t+T]`, split into left/right blocks `Φ_L y_p = Φ_R y_0`, and get the spectrum from the **generalized eigenvalue problem** `eigs(Φ_R, Φ_L)` — no explicit matrix products, no explicit inverse.
- **Established result:** `Φ_L, Φ_R` are sparse/banded; assembly + sparse generalized eig gives **linear time complexity O(p¹)**, versus ≈O(p^2.8) for traditional SD. Validated to `p = 10^6` on the delayed Mathieu equation plus three engineering cases (seasonal biology, multi-cutter turning w/ SSV, FEM beam w/ delayed boundary feedback).
- **Handling of p−1=r, p−1>r, p−1<r** partitioning cases is already worked out.
- **Already flagged as future work in the predecessor:** distributed delays; neutral/advanced-type DDEs; stochastic SD extension. (These are NOT this paper — leave them.)

**Critical framing rule:** The MFSD *assembly* (multiplication-free banded generalized eigenproblem) is the *inherited machinery*. This new paper must not present multiplication-free assembly as its own novelty. Its novelty is what you put *inside* each step.

---

## 2. The thesis of the NEW paper

**Working title (pick/refine):**
*"Higher-Order Multiplication-Free Stability Analysis of Time-Periodic Delayed Systems: Runge–Kutta and Collocation Steps, and the Accuracy–Sparsity Trade-off"*

**One-sentence thesis:** Once stability analysis of periodic DDEs is written in the multiplication-free banded form, the per-step integrator becomes a free design choice — and by inserting Runge–Kutta / implicit collocation steps (rather than the piecewise-constant SD step) you can raise the convergence order arbitrarily, *but* there is a genuine, quantifiable sweet spot between per-step order and matrix sparsity that decides real-world CPU time, and previous "collocation beats semi-discretization" comparisons were unfair because they conflated p-refinement with h-refinement.

**The chain of ideas to convey (this is the narrative spine — preserve its logic):**

1. **SD is just fixed-step time integration.** The semi-discretization one-step map is structurally a fixed-step integrator whose only approximation is treating `A(t), B(t)` and the delayed term as constant across `Δt`. That piecewise-constant assumption is what caps the convergence order (2nd-order optimal for the classic scheme, per Insperger 2008).

2. **So swap the integrator.** Because the multiplication-free form only needs the *per-step residual equation*, we can replace the piecewise-constant step with any one-step integrator: a Runge–Kutta scheme. Place the method's coefficients (Butcher tableau) into the block rows. Higher order per step → higher convergence order of the eigenvalues.

3. **Explicit vs implicit is a matrix-structure choice.**
   - **Explicit RK** → the added stage couplings are strictly lower-triangular → the assembled left matrix stays triangular / very cheaply invertible → linear solves are trivial (forward substitution), so cost per `p` stays minimal. Order is capped (classical order barriers) but you can still reach ~10th order.
   - **Implicit collocation / IRK (Gauss, Radau, Lobatto)** → `s`-stage Gauss collocation gives order `2s` (super-convergence). This requires storing the *intermediate stage states*, not just the step endpoints — i.e. extra state rows in the block system. You get spectral-like accuracy per step, but the per-step block becomes denser.

4. **In the high-order limit, one step = the full collocation/spectral method.** Push the per-step collocation order very high (e.g. ~100) with a single step and you have literally reconstructed the global collocation / pseudospectral method (strong-form here vs the weak/Galerkin form some references use, but equivalent in the limit). *This is the honest admission that the extreme is not new — it is known collocation.* The paper's value is mapping the continuum between the two extremes.

5. **Therefore the old "collocation vs SD" comparisons were unfair.** They pitted a p-refinement method (collocation, raising polynomial order) against an h-refinement method (traditional SD, shrinking Δt). That is like comparing p-type vs h-type FEM refinement and declaring one universally superior. A *fair* comparison holds the refinement philosophy fixed or reports both on the same work–precision axes.

6. **The real engineering result: the accuracy–sparsity sweet spot.** A single ultra-high-order step yields a (near) full/dense matrix that is expensive to factor/solve; many moderate-order steps yield a banded matrix that is far cheaper per solve. For a given accuracy target, the CPU-time optimum is usually *moderate order × many steps*, not *max order × one step*. The location of the sweet spot depends on the problem: for systems that need only ~10–20 collocation points, the matrix is small enough that density does not matter; for systems needing ~200 points (long period/delay, high fidelity), the banded moderate-order route wins decisively on CPU time.

7. **Deliverable:** a fair, log–log, variance-reported work–precision + CPU-time study across the integrator families, on multiple test systems chosen to sit on opposite sides of the sweet spot, plus an open-source Julia implementation so practitioners can pick the right point themselves.

**What "novel" means here (state this honestly in the paper):** not the invention of collocation, and not the invention of multiplication-free assembly — but (a) the systematic embedding of arbitrary RK/collocation steps into the multiplication-free banded DDE stability framework, (b) the explicit-vs-implicit structure/CPU trade-off analysis, (c) the identification and characterization of the accuracy–sparsity sweet spot, and (d) the fair benchmarking methodology that corrects the p-vs-h unfairness in prior comparisons.

---

## 3. Methods to implement (inside `SemiDiscretizationMethod.jl`)

Implement all of these behind a common interface so a single benchmark harness can call any of them and produce `Φ_L, Φ_R` (or the equivalent generalized eigenproblem) for the same test system.

### 3.1 Baselines (must reproduce, for fair reference lines)
- **SD-classic (h-refinement):** the traditional repeated-multiplication monodromy build. Needed so we can plot the ≈O(p^2.8) reference and confirm we match the predecessor paper.
- **MFSD (piecewise-constant step):** the published linear-complexity banded generalized-eig version. This is order-2-limited but O(p¹). It is the "moderate accuracy, maximal sparsity" corner.

### 3.2 New: RK-stepped multiplication-free assembly
General principle: for a one-step integrator advancing `x_i → x_{i+1}` over `Δt`, write **all stage equations and the update equation as residual block-rows** in the big sparse system, exactly as MFSD does for the single piecewise-constant residual. The delayed term `x(t−τ)` at each stage/abscissa is supplied from the appropriate earlier stored state (interpolated if the stage time does not land on a grid node — see 3.4).

Implement, parameterized by a **Butcher tableau (A_bt, b, c)**:

- **Explicit RK family (lower-triangular A_bt):**
  - RK4 (classical), and a configurable higher-order explicit scheme (e.g. order 5–10). Because stages depend only on earlier stages, the stage-coupling blocks are strictly lower triangular ⇒ the left matrix stays triangular/banded and cheaply solvable. Emphasize: *no dense fill*.
  - Expected eigenvalue convergence order ≈ the RK order (verify numerically; delay interpolation order can cap it — see 3.4).

- **Implicit RK / collocation family (full or structured A_bt):**
  - **Gauss–Legendre collocation**, `s` stages, order `2s` (super-convergence — the "two times s" result Daniel highlighted).
  - **Radau IIA** (order `2s−1`, good stiff/decay behavior, stage at endpoint — convenient for the delayed coupling and for act-and-wait-type problems).
  - **Lobatto IIIA** (order `2s−2`, includes both endpoints).
  - These require storing **intermediate stage states** as additional rows in the extended vector `q`. Document exactly how the state vector grows: for `s` internal stages per step, the per-step block grows accordingly and the matrix bandwidth widens with `s`. This widening is the mechanism behind the sparsity cost of high order — measure it.

### 3.3 The "single high-order step = collocation" degenerate case
Provide a mode that uses **one step with N collocation points over the whole period/delay window**, to demonstrate empirically that it reproduces the global collocation/pseudospectral spectrum (match eigenvalues to a reference within tolerance). Use this to make the equivalence argument concrete and to show the dense-matrix cost blow-up.

### 3.4 Delay handling across stages (important correctness detail)
When an integrator evaluates the vector field at stage abscissa `t_i + c_k Δt`, the delayed argument `t_i + c_k Δt − τ` generally does not fall on a stored node. Implement consistent interpolation of the delayed state:
- The interpolation order must be **≥ the integrator order**, otherwise the delay term caps the observed convergence (this is a classic and easy-to-miss failure — call it out). Insperger 2008's remark ("first-order approximation of the delayed term is optimal for piecewise-constant coefficients → 2nd-order convergence") is the low-order analogue; generalize it.
- Prefer using the same collocation/interpolation basis for the delayed lookup as for the step, so order is consistent.
- For explicit schemes, delayed values needed at a stage are already known (past states) — cheap. For implicit schemes with stages near the current step, ensure the coupling is placed in the correct (left vs right) block.

### 3.5 Assembly + solve
- Keep everything in `SparseArrays`. Assemble `Φ_L, Φ_R` directly (linear-time assembly). 
- Get the dominant eigenvalues via `Arpack.eigs` / `ArnoldiMethod` / `KrylovKit` generalized interface on `(Φ_R, Φ_L)` (avoid forming `Φ_L^{-1}Φ_R`; exploit that `Φ_L` is well-conditioned with identity blocks on the diagonal, so sparse forward/backward substitution is stable and fast).
- For the dense single-step collocation mode, fall back to dense `eigen`.
- Return: the leading eigenvalues (for stability = spectral radius / rightmost exponent) and, where relevant, the particular solution / fixed point (the predecessor computes periodic solutions for excited systems — keep that capability).

---

## 4. Numerical experiments (generate ALL of these before writing)

For every method above, on every test system below, produce:

### 4.1 Work–precision diagrams (log–log, mandatory)
- **x-axis:** CPU time (wall-clock, averaged over repeated runs). **y-axis:** eigenvalue error (error in dominant characteristic multiplier / spectral radius vs a high-accuracy reference). 
- Plot every method as its own curve. This is the central figure type of the paper.
- Reference "truth": use the highest-order collocation at very high resolution, or an analytic/semi-analytic result where available (delayed Mathieu has well-characterized stability boundaries).

### 4.2 Time-complexity diagrams (log–log, mandatory)
- **x-axis:** discretization parameter `p` (and/or per-step order `s`). **y-axis:** CPU time.
- Show scaling exponents by fitting straight lines on the log–log plot. Reproduce the predecessor's ≈O(p^2.8) for SD-classic and O(p¹) for MFSD, then place the new methods.
- **Report variance:** mean line + ±1 std shaded band, matching the predecessor's figure style (they use mean lines with ±1σ colored bands and dashed fitted lines). Use ≥ (say) 10 repetitions per point; report machine/BLAS thread settings.

### 4.3 Order-verification diagrams
- Error vs `Δt` (or vs number of steps) at fixed per-step order, on a log–log scale, to confirm each integrator achieves its theoretical convergence order (RK order; Gauss `2s`; etc.). This validates correctness of the implementation and of the delay-interpolation order (4.3 will expose it if delay interpolation is under-ordered).

### 4.4 The sweet-spot study (the headline result)
- For a fixed target accuracy, sweep **per-step order `s`** from low to very high while adjusting the number of steps to hold accuracy roughly constant; plot **CPU time vs per-step order**. Expect a U-shape (or minimum) — the sweet spot. Do this for both demonstration regimes below and show the minimum lands in *different* places.
- Also plot **matrix density / bandwidth vs per-step order** to explain *why* the sweet spot exists (rising order → rising fill → rising solve cost).

### 4.5 Explicit-vs-implicit CPU study
- Directly compare an explicit high-order scheme (cheap triangular solves, capped order) against implicit collocation (dense-ish blocks, super-convergent) on the same work–precision axes. Show the regime where explicit wins on wall-clock despite lower order, and the regime where implicit wins because it needs far fewer points.

---

## 5. Test systems (choose to straddle the sweet spot)

Reuse the predecessor's cases for continuity and add contrast cases. Two systems must be chosen specifically so that **one needs only ~10–20 collocation points (small matrix, density irrelevant)** and **one needs ~200 collocation points (banded moderate-order wins)**.

1. **Delayed Mathieu equation** — the canonical validation system (well-known stability chart; use for order verification and work–precision). Smooth, moderate resolution needs. Good "low-order-suffices / small-matrix" candidate at modest accuracy, and a controllable way to demand high resolution by increasing period/delay ratio or accuracy target.
2. **Seasonal single-species biological model** (Appendix A of predecessor): scalar DDE, `d=1`, smooth periodic coefficients + external forcing. Cheap — good for high-order-per-step demonstration where a small dense matrix is fine.
3. **Multi-cutter turning with spindle-speed variation** (Appendix B): `d=4`, two periodically modulated delays, Mathieu-type. Mid-size; good for explicit-vs-implicit trade-off.
4. **FEM longitudinal beam with delayed boundary feedback + act-and-wait + harmonic excitation** (Appendix C): `d=30` (15 linear elements), high-dimensional, constant delay, time-periodic excitation. This is the **high-resolution / large-`p`** case where the banded moderate-order route should win decisively — the natural "~200 points needed" candidate. Note act-and-wait introduces switching (non-smooth) behavior — good stress test for high-order methods (which, like collocation, struggle with discontinuities; discuss honestly).

For each system, pull exact parameters from the predecessor's figure captions (they are fully specified there) so results are reproducible and comparable.

**Non-smoothness caveat to test and report:** high-order per-step methods (RK and collocation alike) lose order across discontinuities (tooth entry/exit in milling, act-and-wait switching). Show this, and note it as a reason moderate-order-many-steps can be more robust — reinforcing the sweet-spot message.

---

## 6. Fair-comparison methodology (make this an explicit section)

This is a core intellectual contribution — write it as a named methodology section, not a footnote.

- **State the p-vs-h confound.** Prior "collocation ≫ semi-discretization" claims compared a p-refinement method to an h-refinement method. Analogy: p-type vs h-type FEM refinement. Declaring a winner from that is category error.
- **Prescribe fair axes.** Always compare on **work–precision (CPU vs error)** and **time-complexity (CPU vs p)** in **log–log** with **variance bands**. Never compare at a single low resolution (<100) or on a linear scale — that hides true asymptotic rates and is where misleading conclusions come from (the predecessor already criticizes sub-100, non-log-log comparisons — extend that critique).
- **Hold the refinement philosophy explicit.** When comparing methods, report both what happens under h-refinement (more steps) and p-refinement (higher order) so readers see the whole surface.
- **Same solver, same sparsity treatment, same machine, same BLAS threads** for every method. Report all of it.

---

## 7. Manuscript plan (draft after numbers exist)

Target a Q1 venue (see Section 9). Structure:

1. **Abstract** — problem (accuracy limit of SD despite MFSD's speed), idea (integrator is a free choice inside multiplication-free assembly; explicit-triangular vs implicit-collocation; sweet spot), method, fair-comparison contribution, headline numbers, open-source package.
2. **Introduction** — DDE stability importance; SD vs collocation/pseudospectral landscape; the unfair-comparison problem; state contributions crisply (Section 2's honest novelty list). Cite predecessor as the foundation.
3. **Background** — recap MFSD banded assembly (brief; cite predecessor for full derivation); SD as fixed-step integration; convergence-order limit of piecewise-constant step.
4. **Higher-order multiplication-free stepping** — general Butcher-tableau embedding; explicit (triangular, cheap) vs implicit collocation (stage states, order 2s); delay interpolation order requirement; the single-step→collocation/spectral equivalence (strong vs weak form remark).
5. **Fair comparison methodology** (Section 6).
6. **Numerical results** — order verification; work–precision; time-complexity with variance; the sweet-spot U-curves + density/bandwidth explanation; explicit-vs-implicit; non-smoothness stress test. Two contrasting regimes (small-matrix-high-order vs large-matrix-banded).
7. **Discussion** — when to use which; practical guidance / decision rule for practitioners (given problem size, delay/period ratio, accuracy target → recommended order & step count).
8. **Conclusion** — the honest "this unifies and clarifies rather than reinvents" message; the sweet spot as the actionable takeaway; open-source Julia package.
9. **Appendices** — full model parameters (reuse predecessor); reproducibility notes (versions, hardware, thread counts).

**Tone:** honest and clarifying. The paper's strength is rigor and fairness, not a claim of a brand-new operator. Explicitly acknowledge the "solution operator / monodromy operator" viewpoint and prior collocation/pseudospectral work; position this as the fair, unifying, practically-actionable treatment with a reusable tool.

---

## 8. Failure modes to avoid (reviewers will check these)

- **Do NOT re-claim MFSD assembly or collocation as novel.** Novelty = embedding + trade-off analysis + fair methodology + tool. Say so plainly.
- **Do NOT under-order the delay interpolation** — it silently caps convergence and will make high-order methods look broken. Verify order (Section 4.3).
- **Do NOT show sub-100-resolution or linear-scale comparisons** as evidence of scaling. Log–log, wide `p` range, variance bands — always.
- **Do NOT hide the dense-matrix cost** of high single-step order. That cost is the whole point of the sweet spot; measure and show it (bandwidth/density vs order).
- **Do NOT ignore non-smooth cases.** Show where high order degrades; it strengthens, not weakens, the sweet-spot argument.
- **Do NOT compare across different machines/BLAS settings.** Fix and report the environment.
- **Reproducibility:** pin Julia + package versions; fix RNG seeds where timing uses randomized repetition ordering; report hardware (note: prior related work used an NVIDIA P4000 GPU and CUDA.jl for a sister stochastic package — CPU sparse is the main track here, but if any GPU timing is included, isolate and label it).

---

## 9. Journal targets (Q1 candidates)

Primary fit (author has history here, method + application balance): **Journal of Vibration and Control** (SAGE) — and the predecessor uses the SAGE `sagej` class, so the template is already in hand. Strong alternatives: **Nonlinear Dynamics** (Springer) for dynamics+method; **Mechanical Systems and Signal Processing** if the engineering/chatter angle is emphasized. For a more numerical-analysis-forward framing: **Numerical Algorithms** or **Journal of Computational and Applied Mathematics**. Recommendation: draft to SAGE `sagej` (JVC) first; it maximizes reuse of the predecessor's LaTeX setup and reviewer familiarity.

---

## 10. Concrete first actions for the agent

1. Clone/enter `SemiDiscretizationMethod.jl`; reproduce the delayed-Mathieu SD-classic vs MFSD time-complexity plot from the predecessor to confirm baseline parity.
2. Add a `ButcherTableau` abstraction and an `assemble_banded(system, integrator, p, s)` function returning sparse `Φ_L, Φ_R`.
3. Implement explicit RK4 + one high-order explicit scheme; verify order on delayed Mathieu (Section 4.3).
4. Implement Gauss and Radau IIA collocation with stage-state storage + consistent delay interpolation; verify order 2s / 2s−1.
5. Implement the single-step-N-point mode; confirm it matches a reference collocation spectrum.
6. Build the benchmark harness (work–precision, time-complexity, sweet-spot, explicit-vs-implicit) with mean±1σ over repetitions, log–log output, and fitted slopes.
7. Run all four test systems; save every figure and a results table.
8. Only then: draft the manuscript per Section 7, inserting the generated figures, and hand back for Daniel's review.

**Deliverables back to Daniel:** (a) extended package code + tests, (b) all figures + a results table, (c) the drafted manuscript (SAGE `sagej` LaTeX), (d) a short reproducibility README.
