# Generic textbook models for the batched (GPU/CPU) solver, all device-compatible.
# θ is the per-point parameter (isbits); every coefficient function is (t, θ) -> SMatrix.
using SOSD, StaticArrays

# --- Delayed damped Mathieu equation, θ = (δ, ε): ẍ + a1 ẋ + (δ + ε cos t) x = b0 x(t − 2π)
mathieu_A(t, θ) = @SMatrix [0.0 1.0; -θ[1]-θ[2]*cos(t) -0.1]
mathieu_B(t, θ) = (@SMatrix([0.0 0.0; -0.15 0.0]),)
mathieu_tau(t, θ) = (2π,)
mathieu_T(θ) = 2π
const MATHIEU = BatchedLDDE{2, 1}(mathieu_A, mathieu_B, mathieu_tau, mathieu_T)

# --- Turning with sinusoidal spindle speed variation (time-periodic delay), θ = (Ω, kw)
#     ẍ + ζ ẋ + x = kw (x(t − τ(t)) − x(t)),  τ(t) = 2π/Ω (1 + A sin(2π t / T)),  T = NT · 2π/Ω
const SSV_ζ = 0.1; const SSV_A = 0.1; const SSV_NT = 10
ssv_A(t, θ) = @SMatrix [0.0 1.0; -1-θ[2] -SSV_ζ]
ssv_B(t, θ) = (@SMatrix([0.0 0.0; θ[2] 0.0]),)
ssv_T(θ) = 2π / θ[1] * SSV_NT
ssv_tau(t, θ) = (2π / θ[1] * (1 + SSV_A * sin(2π * t / ssv_T(θ))),)
const TURNING_SSV = BatchedLDDE{2, 1}(ssv_A, ssv_B, ssv_tau, ssv_T)
"r needed for p steps per period (max lag 2π/Ω·(1+A), T = NT·2π/Ω)"
ssv_r(p) = ceil(Int, p * (1 + SSV_A) / SSV_NT) + 1

# --- Milling, NM modes in each of x and y (D = 4 NM), θ = (spindle speed n [rpm], depth w [m])
#     Insperger–Stépán textbook tool: 2 teeth, down-milling a/D = 0.05, Kt = 6e8, Kn = 2e8 N/m²,
#     first mode 922 Hz, ζ = 0.011, m = 0.03993 kg; higher modes generic (×1.7, ×2.4, …).
#     `yratio` scales the y-direction frequencies (1.0 = symmetric tool, whose x/y modes
#     coincide and make the dominant multiplier a near-double, ill-conditioned eigenvalue).
#     Modal coordinate i ∈ 1:2NM: x modes first, then y modes; NM2 = 2NM.
struct MillingTool{NM, NM2}
    z::Int
    φen::Float64
    φex::Float64
    Kt::Float64
    Kn::Float64
    ω::SVector{NM2, Float64}
    ζ::SVector{NM2, Float64}
    m::SVector{NM2, Float64}
end

function MillingTool(NM::Int; z=2, aD=0.05, Kt=6e8, Kn=2e8, f1=922.0, ζ=0.011, m1=0.03993, yratio=1.0)
    f = [f1 * (1 + 0.7 * (j - 1)) for j in 1:NM]
    m = [m1 * (1 + 0.5 * (j - 1)) for j in 1:NM]
    ω = vcat(2π .* f, 2π .* f .* yratio)
    MillingTool{NM, 2NM}(z, acos(2aD - 1), π, Kt, Kn, SVector{2NM}(ω), SVector{2NM}(fill(ζ, 2NM)), SVector{2NM}(vcat(m, m)))
end

@inline function milling_H(tool::MillingTool, t, n_rpm)
    Ω = 2π * n_rpm / 60
    hxx = 0.0; hxy = 0.0; hyx = 0.0; hyy = 0.0
    for j in 0:tool.z-1
        φ = mod(Ω * t + j * 2π / tool.z, 2π)
        if tool.φen < φ < tool.φex
            s, c = sincos(φ)
            hxx += (tool.Kt * c + tool.Kn * s) * s;  hxy += (tool.Kt * c + tool.Kn * s) * c
            hyx += (-tool.Kt * s + tool.Kn * c) * s; hyy += (-tool.Kt * s + tool.Kn * c) * c
        end
    end
    return SMatrix{2, 2}(hxx, hyx, hxy, hyy)
end

# State q = [x_1..x_NM, y_1..y_NM, ẋ_1.., ẏ_1..]; tool-tip displacement = Σ modes per direction.
struct MillingA{NM, NM2}; tool::MillingTool{NM, NM2}; end
struct MillingB{NM, NM2}; tool::MillingTool{NM, NM2}; end

function (a::MillingA{NM})(t, θ) where {NM}
    tool = a.tool; n2 = 2NM; D = 4NM
    H = milling_H(tool, t, θ[1])
    return SMatrix{D, D, Float64}(ntuple(Val(D * D)) do idx
        row = (idx - 1) % D + 1; col = (idx - 1) ÷ D + 1
        if row <= n2
            (col == row + n2) ? 1.0 : 0.0
        else
            i = row - n2; di = (i - 1) ÷ NM + 1
            if col <= n2
                dj = (col - 1) ÷ NM + 1
                -(col == i ? tool.ω[i]^2 : 0.0) - θ[2] * H[di, dj] / tool.m[i]
            else
                (col - n2 == i) ? -2 * tool.ζ[i] * tool.ω[i] : 0.0
            end
        end
    end)
end

function (bm::MillingB{NM})(t, θ) where {NM}
    tool = bm.tool; n2 = 2NM; D = 4NM
    H = milling_H(tool, t, θ[1])
    return (SMatrix{D, D, Float64}(ntuple(Val(D * D)) do idx
        row = (idx - 1) % D + 1; col = (idx - 1) ÷ D + 1
        if row > n2 && col <= n2
            i = row - n2; di = (i - 1) ÷ NM + 1; dj = (col - 1) ÷ NM + 1
            θ[2] * H[di, dj] / tool.m[i]
        else
            0.0
        end
    end),)
end

struct MillingTau{NM}; z::Int; end
(mt::MillingTau)(t, θ) = (60 / (mt.z * θ[1]),)
struct MillingPeriod{NM}; z::Int; end
(mp::MillingPeriod)(θ) = 60 / (mp.z * θ[1])

"Milling model with `NM` modes per direction (state dimension D = 4NM); coefficient period = tooth pass = τ."
function milling_model(NM::Int; kw...)
    tool = MillingTool(NM; kw...)
    return BatchedLDDE{4NM, 1}(MillingA(tool), MillingB(tool), MillingTau{NM}(tool.z), MillingPeriod{NM}(tool.z))
end
