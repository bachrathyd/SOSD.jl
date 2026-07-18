using MFCM
using SemiDiscretizationMethod
using Plots
using StaticArrays
using KrylovKit
using BenchmarkTools
using LaTeXStrings
using LinearAlgebra
using Printf

# --- Mathieu Problem Definition ---
function createMathieuProblem_MFCM(δ, ε, b0, a1; T=2π)
    AMx = MFCM.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = MFCM.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = MFCM.Additive(t -> @SVector [0.0, sin(4π / T * t)])
    MFCM.LDDEProblem{2, Float64}(AMx, [BMx1], cVec)
end

function createMathieuProblem_SDM(δ, ε, b0, a1; T=2π)
    AMx = SemiDiscretizationMethod.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = SemiDiscretizationMethod.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = SemiDiscretizationMethod.Additive(t -> @SVector [0.0, sin(4π / T * t)])
    SemiDiscretizationMethod.LDDEProblem(AMx, [BMx1], cVec)
end

function run_explicit_benchmark()
    δ, ε, b0, a1 = 3.0, 0.2, -0.15, 0.1
    T = 2π
    D = 2
    
    println("Computing high-precision reference values...")
    p_ref = 800
    r_ref = p_ref
    tableau_ref = GL(10)
    prob_ref = createMathieuProblem_MFCM(δ, ε, b0, a1, T=T)
    grid_ref = TimeGrid(collect(range(0.0, T, length=p_ref+1)))
    sys_ref = build_system_matrices(prob_ref, grid_ref, tableau_ref, r_ref)
    m_ref = MonodromyMap(prob_ref, grid_ref, tableau_ref, sys_ref, p_ref, r_ref, (r_ref+1)*(11)*D)
    
    vals_ref, _ = eigsolve(m_ref, rand(m_ref.state_size), 1, :LM)
    mu_ref = abs(vals_ref[1])
    @show mu_ref

    # Resolution range: 20 to 1000 for better slope calculation
    ps = round.(Int, 10 .^ range(log10(20), log10(1000), length=15))
    ps = sort(unique(ps))
    
    solvers = [
        ("Euler (O1)", ExplicitEuler()),
        ("Heun (O2)", Heun()),
        ("RK3 (O3)", RK3()),
        ("RK4 (O4)", RK4()),
        ("RK5 (O5)", RK5()),
        ("RK8 (O8)", RK8())
    ]
    
    results = Dict()
    execution_order = []
    
    # --- 2. Explicit MFCM ---
    for (name, tab) in solvers
        S = size(tab.a, 1)
        full_name = "Sparse $name"
        println("\nBenchmarking $full_name...")
        times, errors, evals, mems = [], [], [], []
        push!(execution_order, full_name)
        
        for p in ps
            print("$p ")
            try
                h = T / p; r = p
                prob = createMathieuProblem_MFCM(δ, ε, b0, a1, T=T)
                grid = TimeGrid(collect(range(0.0, T, length=p+1)))
                BSIZE = (S + 1) * D
                state_size = (r + 1) * BSIZE
                
                sys = build_system_matrices(prob, grid, tab, r)
                m_lazy = MonodromyMap(prob, grid, tab, sys, p, r, state_size)
                m = SparseMonodromyMap(m_lazy)
                x0 = ones(state_size) / sqrt(state_size)
                
                vals, _ = eigsolve(m, x0, 1, :LM)
                err = abs(abs(vals[1]) - mu_ref)
                
                push!(times, 0.0); push!(errors, err); push!(evals, p); push!(mems, 0.0)
            catch e; println("\nFailed for $full_name p=$p: $e"); end
        end
        
        # Calculate slope
        errs = Float64.(errors)
        xs = Float64.(ps)
        calc_idx = findall(i -> 1e-12 < errs[i] < 1e-2, 1:length(errs))
        if length(calc_idx) > 1
            coeffs = [log10.(xs[calc_idx]) ones(length(calc_idx))] \ log10.(errs[calc_idx])
            k_val = -coeffs[1]
            @printf("\n  - %s: Calculated k = %.2f\n", name, k_val)
        end
    end
end

run_explicit_benchmark()
