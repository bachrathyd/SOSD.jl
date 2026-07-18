# Multiplication-Free Collocation Method (MFCM)

A high-performance Julia package for the stability analysis of time-periodic Delayed Differential Equations (DDEs).

## Features
- **O(p^1) Time Complexity:** Avoids explicit construction of dense monodromy matrices.
- **Spectral Accuracy:** Supports high-order Gauss-Legendre collocation (up to 6th order and beyond).
- **Lazy Operator:** Uses `LinearMaps.jl` for memory-efficient Floquet multiplier calculation via `KrylovKit.jl`.
- **Automatic Linearization:** Extract system matrices $A(t)$ and $B_i(t)$ automatically from DDE RHS functions.
- **Explicit Sparse Support:** Includes tools to build explicit sparse representations for validation or custom solvers.
- **MDBM Integration:** Ready for multi-dimensional stability chart generation.

## Installation
```julia
using Pkg
Pkg.activate(".")
```

## Quick Start
```julia
using MFCM
using StaticArrays
using KrylovKit

# Define a Delayed Mathieu Equation
function mathieu_rhs(u, h, p, t)
    x1, x2 = u
    hist = h(p, t - 2π) # Delay tau = 2pi
    du1 = x2
    du2 = -0.1 * x2 - (3.0 + 1.5 * cos(t)) * x1 - 0.5 * hist[1]
    return @SVector [du1, du2]
end

# 1. Extract linear system
prob = extract_SDM_system(mathieu_rhs, nothing, Val(2))

# 2. Setup discretization
p_steps = 100
T = 2π
grid = TimeGrid(collect(range(0.0, T, length=p_steps+1)))
tableau = GL2Tableau() # 4th order

# 3. Precompute system matrices
sys_mats = build_system_matrices(prob, grid, tableau, p_steps)

# 4. Create Monodromy Map
m = MonodromyMap(prob, grid, tableau, sys_mats, p_steps, p_steps, (p_steps+1)*6)

# 5. Solve for Floquet Multipliers
vals, _ = eigsolve(m, rand(m.state_size), 1, :LM)
println("Max Multiplier: ", abs(vals[1]))
```

## Engineering Case Studies
Verified implementations and stability charts for:
1. **Delayed Mathieu Equation**
2. **1-DOF Regenerative Milling**
3. **Seasonal Maturation (Biological Model)**

See `examples/` for details.

## Verification
- **Convergence:** Verified $O(h^4)$ for GL2 and $O(h^6)$ for GL3.
- **Complexity:** Verified $O(p^1)$ scaling.
- **Accuracy:** Verified against explicit sparse matrix solutions (error < 1e-15).

## References
1. Bachrathy, D., & Stepan, G. (2012). Improved semi-discretization method for periodic systems with delay.
2. Multiplication-Free Collocation Method for stability analysis of DDEs.
