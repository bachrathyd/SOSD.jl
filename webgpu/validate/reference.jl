# Reference spectral radii for the browser examples (?validate=1 of the page):
# Float64 CPU solver of SOSD on the Julia twins of webgpu/examples.js (test/batched_models.jl),
# a 16×10 grid over each example's chart axes at the default values of the other parameters.
#   julia --project=webgpu/validate webgpu/validate/reference.jl
using SOSD, StaticArrays, JSON3
include(joinpath(@__DIR__, "..", "..", "test", "batched_models.jl"))

const NX, NY = 16, 10
grid(x0, x1, y0, y1) = [(x, y) for y in range(y0, y1, length=NY) for x in range(x0, x1, length=NX)]

cases = [
    # key, problem, θ builder from chart (x, y), x range, y range, p, r(p)
    ("mathieu", MATHIEU4, (x, y) -> SVector(x, y, -0.15, 0.1), (-1.0, 10.0), (0.0, 10.0), 40, p -> p + 1),
    ("milling", milling_model_nd(1; aD=0.05, ζ=0.011, yratio=1.05), (x, y) -> SVector(x, y * 1e-3),
        (5000.0, 25000.0), (0.0, 5.0), 40, p -> p + 1),
    ("turning_ssv", TURNING_SSV4, (x, y) -> SVector(x, y, 0.1, 0.1), (0.2, 2.0), (0.0, 0.6), 200,
        p -> ceil(Int, 2π / 1 * (1 + 0.1) / (10 * 2π / p)) + 1),
]
out = Dict{String, Any}()
for (key, prob, θof, xr, yr, p, rof) in cases
    pts = grid(xr..., yr...)
    θs = [θof(x, y) for (x, y) in pts]
    r = rof(p)
    t = @elapsed res = spectral_radii(prob, θs, GL(3), p, r; tol=1e-12)
    println(rpad(key, 12), " p = $p, r = $r: ", length(pts), " points in ", round(t, digits=1), " s; unstable ",
            count(>(1), res.rho), "; converged ", count(res.converged))
    out[key] = Dict("p" => p, "S" => 3, "nx" => NX, "ny" => NY, "x" => [xr...], "y" => [yr...],
                    "pts" => [[x, y] for (x, y) in pts], "rho" => res.rho)
end
open(joinpath(@__DIR__, "ref.json"), "w") do io
    JSON3.write(io, out)
end
println("-> ", joinpath(@__DIR__, "ref.json"))
