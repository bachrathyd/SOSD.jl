# Julia model of the Mathieu example (webgpu/expr.js modelJulia), for the server warm-up
using StaticArrays
const D = 2
const NP = 5
@inline _step(x) = x >= zero(x) ? one(x) : zero(x)
@inline function period(θ)
    t = 0.0

    return (2.0 * 3.141592653589793)
end
@inline function tau(t, θ)

    return ((2.0 * 3.141592653589793),)
end
@inline function A(t, θ)

    return @SMatrix([0.0 1.0; (-(θ[1] + (θ[2] * cos(t)))) (-θ[4])])
end
@inline function B(t, θ)

    return (@SMatrix([0.0 0.0; θ[3] 0.0]),)
end
@inline function forcing(t, θ)

    return SVector(0.0, (θ[5] * cos(t)))
end
const HAS_FORCING = true
