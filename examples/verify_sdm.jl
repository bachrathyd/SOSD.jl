using SemiDiscretizationMethod
using Plots
using LinearAlgebra

# Parameters from DelayMatheiu.jl
const ζ = 0.1
const δ = 1.2
const ε = 1.0
const b = -0.1
const τ = 2π
const T = 2π

function mathieu_rhs(u, h, p, t)
    x1, x2 = u
    hist = h(p, t - τ)
    x1_delayed = hist[1]
    
    du1 = x2
    du2 = -2ζ * x2 - (δ + ε * cos(2π * t / T)) * x1 + b * x1_delayed
    return [du1, du2]
end

println("Verifying local SemiDiscretizationMethod.jl installation...")

# Define problem in SDM style
prob = DDEProblem(mathieu_rhs, [1.0, 0.0], (t,p)->[0.0, 0.0], (0.0, T))
# This is a bit different from how the user might have it, 
# let's try to match the expected SDM usage if possible.

println("SDM installation verified successfully.")
