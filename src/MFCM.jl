module MFCM

using LinearAlgebra
using StaticArrays
using SparseArrays
using KrylovKit
using LinearMaps

export RKTableau, TimeGrid, SystemMatrices, MonodromyMap
export LDDEProblem, ProportionalMX, DelayMX, Additive
export extract_SDM_system, build_system_matrices
export GL, GL2Tableau, GL3Tableau, ExplicitEuler, Heun, RK3, RK4, RK5, RK8, ImplicitEuler, ImplicitTrapezoidal
export solve_periodic_solution, inhomogeneous_sweep, SparseMonodromyMap
export build_explicit_matrices, get_explicit_transition_matrix

@enum InterpStrategy collocation endpoint denseoutput

"""
    RKTableau{S, T, CE}
"""
struct RKTableau{S, T, CE}
    a::SMatrix{S, S, T}
    b::SVector{S, T}
    c::SVector{S, T}
    ce::CE
    interp_nodes::Vector{T}
    order::Int
    strategy::InterpStrategy
end

"""
    TimeGrid{T}
"""
struct TimeGrid{T}
    t::Vector{T}
    h::Vector{T}
end

function TimeGrid(t::Vector{T}) where T
    h = diff(t)
    return TimeGrid(t, h)
end

"""
    ProportionalMX{F}
"""
struct ProportionalMX{F}
    f::F
end

"""
    DelayMX{F1, F2}
"""
struct DelayMX{F1, F2}
    tau::F1
    f::F2
end

"""
    Additive{F}
"""
struct Additive{F}
    f::F
end

"""
    LDDEProblem{D, T, FA, FB, FC}
"""
struct LDDEProblem{D, T, FA, FB, FC}
    A::ProportionalMX{FA}
    B::Vector{FB}
    c::Additive{FC}
end

function LDDEProblem{D, T}(A::ProportionalMX{FA}, B::Vector{FB}, c::Additive{FC}) where {D, T, FA, FB, FC}
    return LDDEProblem{D, T, FA, FB, FC}(A, B, c)
end

"""
    SystemMatrices{D, S, T, W, BSIZE}
"""
struct SystemMatrices{D, S, T, W, BSIZE}
    M_prop::Vector{SMatrix{BSIZE, D, T}}
    M_del::Vector{Vector{SVector{S, SMatrix{BSIZE, D, T}}}}
    delay_indices::Vector{Vector{SVector{S, Int}}}
    delay_weights::Vector{Vector{SVector{S, SVector{W, T}}}}
    c_vector::Vector{SVector{BSIZE, T}}
end

struct MonodromyMap{D, S, T, W, BSIZE, CE} <: LinearMaps.LinearMap{T}
    problem::LDDEProblem{D, T}
    grid::TimeGrid{T}
    tableau::RKTableau{S, T, CE}
    sys_mats::SystemMatrices{D, S, T, W, BSIZE}
    p::Int
    r::Int
    state_size::Int
    history_buffer::Vector{T}
end

function MonodromyMap(prob, grid, tableau::RKTableau{S, T, CE}, sys_mats, p, r, state_size) where {S, T, CE}
    D, W, BSIZE = typeof(sys_mats).parameters[1], typeof(sys_mats).parameters[4], typeof(sys_mats).parameters[5]
    h_buf = Vector{T}(undef, (p + r + 1) * BSIZE)
    return MonodromyMap{D, S, T, W, BSIZE, CE}(prob, grid, tableau, sys_mats, p, r, state_size, h_buf)
end

Base.size(m::MonodromyMap) = (m.state_size, m.state_size)

include("utils.jl")
include("tableau_library.jl")
include("solver.jl")
include("sparse_builder.jl")
include("sparse_map.jl")

end # module
