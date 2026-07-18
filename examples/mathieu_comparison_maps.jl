using Pkg
Pkg.activate(".")

using SOSD
using SemiDiscretizationMethod
using Plots
using StaticArrays
using KrylovKit
using MDBM
using LaTeXStrings

# 1. Utility to create problem (SOSD style)
function createMathieuProblem_SOSD(δ, ε, b0, a1; T=2π)
    AMx = SOSD.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = SOSD.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = SOSD.Additive(t -> @SVector [0.0, sin(4π / T * t)])
    SOSD.LDDEProblem{2, Float64}(AMx, [BMx1], cVec)
end

# Utility to create problem (SDM style)
function createMathieuProblem_SDM(δ, ε, b0, a1; T=2π)
    AMx = SemiDiscretizationMethod.ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π
    BMx1 = SemiDiscretizationMethod.DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = SemiDiscretizationMethod.Additive(t -> @SVector [0.0, sin(4π / T * t)])
    SemiDiscretizationMethod.LDDEProblem(AMx, [BMx1], cVec)
end

# 2. Function for SOSD mumax
function mfcm_mumax(δ, ε, b0, a1; T=2π, p_steps=10, tableau=GL(2))::Float64
    τmax = 2π
    h = T / p_steps
    r_steps = round(Int, τmax / h)
    
    prob = createMathieuProblem_SOSD(δ, ε, b0, a1, T=T)
    grid = TimeGrid(collect(range(0.0, T, length=p_steps+1)))
    sys_mats = build_system_matrices(prob, grid, tableau, r_steps)
    
    S = size(tableau.a, 1)
    D = 2
    BSIZE = (S + 1) * D
    state_size = (r_steps + 1) * BSIZE
    m = MonodromyMap(prob, grid, tableau, sys_mats, p_steps, r_steps, state_size)
    
    vals, _ = eigsolve(m, rand(m.state_size), 1, :LM)
    return abs(vals[1])
end

# 3. Function for SDM mumax (Reference)
function sdm_mumax(δ, ε, b0, a1; T=2π, p_steps=10)::Float64
    τmax = 2π
    τmax = 2π
    prob = createMathieuProblem_SDM(δ, ε, b0, a1, T=T)
    method = SemiDiscretization(2, T / p_steps)
    mapping = DiscreteMapping_LR(prob, method, τmax, n_steps=p_steps)
    return spectralRadiusOfMapping(mapping)
end

function run_comparison_maps()
    println("Generating Comparison Maps...")
    
    # Map 1: delta vs b0
    a1 = 0.01; ε = 1.0; T_period = π;  p_steps_SD = 40;p_steps_SOSD = 10
    foo_mfcm(δ, b0) = log(mfcm_mumax(δ, ε, b0, a1; T=T_period, p_steps=p_steps_SOSD, tableau=GL(2)))
    τmax = 2π
    foo_sdm(δ, b0) = log(sdm_mumax(δ, ε, b0, a1; T=T_period, p_steps=p_steps_SD))
    τmax = 2π
    
    axis = [Axis(-1.0:1.0:5.0, :δ), Axis(-2.0:0.5:1.5, :b0)]
    
    println("Running MDBM for Map 1 (SOSD)...")
    mdbm_mfcm = MDBM_Problem(foo_mfcm, axis)
    MDBM.solve!(mdbm_mfcm, 4)
    pts_mfcm = getinterpolatedsolution(mdbm_mfcm)
    
    println("Running MDBM for Map 1 (SDM)...")
    mdbm_sdm = MDBM_Problem(foo_sdm, axis)
    MDBM.solve!(mdbm_sdm, 4)
    pts_sdm = getinterpolatedsolution(mdbm_sdm)
    
    p1 = plot(title="Stability Map 1: δ vs b0 (T=π, τ=2π)", xlabel=L"\delta", ylabel=L"b_0")
    if !isempty(pts_sdm)
        scatter!(p1, pts_sdm[1, :], pts_sdm[2, :], markersize=5, label="SDM (Ref)", color=:blue)
    end
    if !isempty(pts_mfcm)
        scatter!(p1, pts_mfcm[1, :], pts_mfcm[2, :], markersize=2, label="SOSD (GL2)", color=:red)
    end
    
    savefig(p1, "mathieu_map_comparison_1.png")
    println("Map 1 saved.")
    
    # Map 2: delta vs epsilon (No delay)
    b0 = 0.0; T_period = 2π; p_steps_SD = 40;p_steps_SOSD = 10
    
    foo_mfcm2(δ, ε_val) = log(mfcm_mumax(δ, ε_val, b0, a1; T=T_period, p_steps=p_steps_SOSD, tableau=GL(2)))
    foo_sdm2(δ, ε_val) = log(sdm_mumax(δ, ε_val, b0, a1; T=T_period, p_steps=p_steps_SD))
    
    axis2 = [Axis(-2.0:1.0:5.0, :δ), Axis(-0.01:1.0:5.0, :ε)]
    
    println("Running MDBM for Map 2 (SOSD)...")
    mdbm_mfcm2 = MDBM_Problem(foo_mfcm2, axis2)
    MDBM.solve!(mdbm_mfcm2, 4)
    pts_mfcm2 = getinterpolatedsolution(mdbm_mfcm2)
    
    println("Running MDBM for Map 2 (SDM)...")
    mdbm_sdm2 = MDBM_Problem(foo_sdm2, axis2)
    MDBM.solve!(mdbm_sdm2, 4)
    pts_sdm2 = getinterpolatedsolution(mdbm_sdm2)
    
    p2 = plot(title="Stability Map 2: δ vs ε (No delay)", xlabel=L"\delta", ylabel=L"\epsilon")
    if !isempty(pts_sdm2)
        scatter!(p2, pts_sdm2[1, :], pts_sdm2[2, :], markersize=5, label="SDM (Ref)", color=:blue)
    end
    if !isempty(pts_mfcm2)
        scatter!(p2, pts_mfcm2[1, :], pts_mfcm2[2, :], markersize=2, label="SOSD (GL2)", color=:red)
    end
    
    savefig(p2, "mathieu_map_comparison_2.png")
    println("Map 2 saved.")

    # Map 3: 3D delta vs b0 vs epsilon
    T_period = 2π; ; p_steps_SD = 40;p_steps_SOSD = 10
    
    foo_mfcm3(δ, b0_val, ε_val) = log(mfcm_mumax(δ, ε_val, b0_val, a1; T=T_period, p_steps=p_steps_SOSD, tableau=GL(2)))
    foo_sdm3(δ, b0_val, ε_val) = log(sdm_mumax(δ, ε_val, b0_val, a1; T=T_period, p_steps=p_steps_SD))
    
    axis3 = [Axis(-2.0:1.0:5.0, :δ), Axis(-2.0:0.5:1.5, :b0), Axis(-0.01:1.0:5.0, :ε)]
    
    println("Running MDBM for Map 3 (SOSD)...")
    mdbm_mfcm3 = MDBM_Problem(foo_mfcm3, axis3)
    MDBM.solve!(mdbm_mfcm3, 2) # Coarser for 3D
    pts_mfcm3 = getinterpolatedsolution(mdbm_mfcm3)
    
    println("Running MDBM for Map 3 (SDM)...")
    mdbm_sdm3 = MDBM_Problem(foo_sdm3, axis3)
    MDBM.solve!(mdbm_sdm3, 2)
    pts_sdm3 = getinterpolatedsolution(mdbm_sdm3)
    
    p3 = plot(title="Stability Map 3: 3D (δ, b0, ε)", xlabel=L"\delta", ylabel=L"b_0", zlabel=L"\epsilon")
    if !isempty(pts_sdm3)
        scatter!(p3, pts_sdm3[1, :], pts_sdm3[2, :], pts_sdm3[3, :], markersize=2, label="SDM (Ref)", color=:blue)
    end
    if !isempty(pts_mfcm3)
        scatter!(p3, pts_mfcm3[1, :], pts_mfcm3[2, :], pts_mfcm3[3, :], markersize=0.5, label="SOSD (GL2)", color=:red)
    end
    
    savefig(p3, "mathieu_map_comparison_3.png")
    println("Map 3 saved.")
end

run_comparison_maps()
run_comparison_maps()
