# Multiplication-Free Collocation Method (MFCM)

## Project Goals & Standards

*   **Language:** All code, comments, documentation, and strings must be in **English**.
*   **Documentation:** Every public function and structure must have a **docstring** following Julia conventions for future documentation generation.
*   **Package Structure:** The project must be structured as a standard Julia package (`src/`, `test/`, `Project.toml`, `README.md`, etc.) ready for registration.
*   **Progress Tracking:** Maintain a `progress.md` file in the root directory. This file must track completed tasks and list upcoming steps.
*   **Verification & Visualization:** Every implementation step must be accompanied by a verification script (e.g., in `test/` or a dedicated `examples/` folder).
    *   Verification results must be visible (e.g., printing matrices, convergence tables).
    *   If applicable, graphical results (plots, work-precision diagrams) must be saved as image files (e.g., `.png` or `.pdf`) to allow for offline review.
*   **Verification Criteria:** Implementation is not complete until the following are verified:
    *   Order of convergence (spectral accuracy).
    *   Linear time complexity ($O(p^1)$).
    *   Fixed-point/periodic solution error convergence.
    *   Work-precision analysis (CPU time vs. eigenvalue error).

## 1. Theoretical Summary

### 1.1. Core Concept
Traditional spectral and collocation methods for Delayed Differential Equations (DDEs) build a dense monodromy matrix with $O(p^3)$ time complexity. MFCM (Multiplication-Free Collocation Method) avoids explicit construction of the transition matrix. Instead, it strings together all intermediate states (Runge-Kutta stages) of all time steps into a single large state vector ($\mathbf{q}$) and solves the global system: $0 = \mathbf{Q}\mathbf{q}$.

The trick lies in the structure of the $\mathbf{Q}$ operator, which is an extremely sparse, banded block matrix constructed via dyadic (Kronecker-like) decomposition. For stability analysis, this global equation is decomposed into a one-period transition.

### 1.2. Global State Vector
To avoid redundancy (ensuring corners of adjacent blocks match), the $n$-th step state vector for an $s$-stage RK method:
$$v_n = \begin{bmatrix} y_n \\ Y_{1,n} \\ \vdots \\ Y_{s,n} \end{bmatrix}$$
The global vector is the sequence: $\mathbf{q} = [\dots, v_{n-1}^T, v_n^T, v_{n+1}^T, \dots]^T$.

### 1.3. Dyadic Local Mapping ($\mathcal{M}_n$)
A single step mapping decomposes into the dyadic combination of three independent components:
1. **Topological Skeleton ($\mathcal{T}$):** Constant matrix describing the left side ($-y_n + Y_{i,n} = \dots$) and connecting steps.
2. **Butcher Tableau ($\mathcal{B}$):** Constant matrix containing $a_{ij}$ and $b_j$ weights.
3. **System Matrices ($\mathcal{A}_n$):** Diagonal block of system matrices evaluated at $t_n + c_i h$.

### 1.4. Avoiding Order Reduction
Retrieving $B(t)y(t-\tau)$ is critical. Simple interpolation (e.g., splines) must NOT be used as it destroys high-order accuracy. **Solution:** Use the **Continuous Extension (Dense Output polynomials)** associated with the chosen Butcher tableau to calculate delayed terms.

### 1.5. Handling $T$ and $\tau_{max}$ Ratios
The Floquet multipliers are the eigenvalues of the one-period transition: $\Phi_L y_p = \Phi_R y_0$.
The decomposition of $\mathbf{Q}$ depends on the ratio of period ($T = p \cdot h$) and delay ($\tau = r \cdot h$):
*   **$p-1 = r$ (No Overlap):** $\mathbf{Q}$ splits into two square $\Phi_L$ and $\Phi_R$ operators.
*   **$p-1 > r$ (Padding):** Period is longer than delay. $y_0$ is padded with zeros.
*   **Progress Tracking:** Maintain a `progress.md` file in the root directory. This file must track completed tasks and list upcoming steps. **This file must be kept up-to-date after every significant milestone.**

## 3. Benchmarking & Cross-Validation Requirements

*   **Comparison against Reference:** Cross-validate results with [SemiDiscretizationMethod.jl](https://github.com/HTSykora/SemiDiscretizationMethod.jl).
    *   Plot stability boundaries from both packages on the same chart for the Mathieu equation.
    *   Verify that Floquet multipliers and fixed-points converge to the same values.
*   **Solver Suite:** Implement and compare the following methods:
    *   **Explicit:** Euler, Trapezoidal (Heun), RK3, RK4.
    *   **Implicit:** Implicit Euler, Implicit Trapezoidal.
    *   **Spectral (Collocation):** Gauss-Legendre (GL) orders 1 through 7.
*   **Performance Metrics (Work-Precision):** Generate three diagrams for each solver (and a consolidated one with 3 subplots):
    1.  **Resolution ($p$) vs. Error:** Log-log scale with fitted convergence exponents.
    2.  **Resolution ($p$) vs. CPU Time:** Log-log scale.
    3.  **CPU Time vs. Error:** Work-Precision diagram (Work-Time vs Error).
    *   **Range:** $p \in [10, 10^5]$.
*   **Analysis Targets:**
    *   Convergence of the spectral radius (max Floquet multiplier).
    *   Convergence of the periodic fixed-point solution.
    *   Computational scaling verification ($O(p^1)$).

## 4. Performance Optimization Tasks
*   **Profiling & Bottleneck Identification:** Use the Julia profiler to determine which parts of the `MonodromyMap` and `base_sweep!` are most time-consuming.
*   **Lazy vs. Explicit Sparse:** Compare the performance of the lazy operator (LinearMaps.jl) against a precomputed explicit sparse matrix implementation. While lazy uses less memory, explicit sparse might be significantly faster for eigenvalue calculations.
*   **Type Stability Analysis:** Audit the codebase for type instabilities (using `@code_warntype`) that might be causing excessive allocations and slowdowns.
*   **SDM Implementation Parity:** Analyze `SemiDiscretizationMethod.jl` to ensure MFCM has comparable or better efficiency in system matrix evaluations and transition matrix structure.
*   **General Optimization:** Improve memory layout, minimize allocations in the inner sweep loop, and leverage specialized solvers where applicable.
*   **DDE-based Memory Efficiency Comparison:** Compare MFCM/SDM memory usage with a pure DDE solver approach (e.g., using `DifferentialEquations.jl`). 
    *   **Test Case:** Keep $dt$ and $\tau$ fixed, and increase the period $T$. 
    *   **Hypothesis:** MFCM and SDM memory usage should grow linearly with $T$ (since they store the full state history over the period), whereas a DDE solver should only need to store the solution over the delay interval $\tau$, leading to constant memory usage regardless of $T$. Verify this behavior and document the trade-offs (e.g., DDE solver being slower due to interpolation and step management).

