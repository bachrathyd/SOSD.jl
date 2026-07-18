using Test
using SOSD

@testset "SOSD.jl Tests" begin
    println("Running ODE Convergence Tests...")
    include("ode_convergence.jl")
    
    println("Running GL3 Convergence Tests...")
    include("gl3_convergence.jl")
    
    println("Running DDE Convergence Tests...")
    include("convergence_analysis.jl")
    
    println("Running Complexity Analysis Tests...")
    include("complexity_analysis.jl")
    
    println("Running Sparse Validation Tests...")
    include("sparse_validation.jl")
    
    println("Running Mathieu Equation Validation Tests...")
    include("mathieu_validation.jl")
    
    println("Running Inhomogeneous Periodic Solution Tests...")
    include("inhomogeneous_test.jl")
end
