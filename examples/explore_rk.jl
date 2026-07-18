using RungeKutta

println("Available methods in RungeKutta.jl:")
# For example, Gauss(3)
rk = Gauss(3)
println("Gauss(3) nodes: ", rk.c)
println("Gauss(3) A matrix: ", rk.A)
println("Gauss(3) b weights: ", rk.b)

# Check if there is a dense output method
# In RungeKutta.jl, it might be separate.
