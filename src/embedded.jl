# Embedded lower-order companions of an RKTableau, used by the error
# estimation (ode23-style pair difference expressed as a perturbation of the
# mapping matrix). Two independent channels:
#   Q — quadrature: same stages (a, c), lower-order weights b̂
#   I — interpolation: continuous extension with one interior node dropped

"Group stage indices by (approximately) equal abscissa value."
function _distinct_node_groups(c::AbstractVector{T}; atol=1e-12) where T
    nodes = T[]
    groups = Vector{Vector{Int}}()
    for (i, ci) in enumerate(c)
        found = false
        for (g, node) in enumerate(nodes)
            if isapprox(node, ci; atol=atol)
                push!(groups[g], i); found = true; break
            end
        end
        if !found
            push!(nodes, ci); push!(groups, [i])
        end
    end
    return nodes, groups
end

"Interpolatory quadrature weights on `nodes` for ∫₀¹ (Vandermonde moment solve)."
function _interpolatory_weights(nodes::AbstractVector{T}) where T
    n = length(nodes)
    M = [nodes[i]^(k - 1) for k in 1:n, i in 1:n]
    m = [one(T) / k for k in 1:n]
    return M \ m
end

"""
    embedded_weights(tab::RKTableau) -> SVector or nothing

Lower-order companion weights `b̂` on the same stages. Uses the classical
embedded pair when the tableau carries one (`tab.b_embedded`); otherwise builds
the interpolatory quadrature rule on the stage abscissae with the node closest
to 1/2 removed — the maximal-order (s−1) distinct rule available on the same
stage data. Returns `nothing` when no companion exists (single distinct node).
"""
function embedded_weights(tab::RKTableau{S, T}) where {S, T}
    tab.b_embedded === nothing || return tab.b_embedded
    nodes, groups = _distinct_node_groups(tab.c)
    length(nodes) < 2 && return nothing
    # drop the interior-most node group (max leverage on the top moment)
    drop = argmin([(abs(nodes[g] - 0.5), -nodes[g]) for g in eachindex(nodes)])
    keep = setdiff(eachindex(nodes), drop)
    w = _interpolatory_weights(nodes[keep])
    b_hat = zeros(T, S)
    for (wi, g) in zip(w, keep)
        for i in groups[g]
            b_hat[i] = wi / length(groups[g])
        end
    end
    return SVector{S, T}(b_hat)
end

"""
    reduced_continuous_extension(tab::RKTableau) -> function or nothing

Continuous-extension weight function of one polynomial degree lower: the
Lagrange basis on the interpolation nodes (0, c…, 1) with the interior node
closest to 1/2 removed. Nodes 0 and 1 are always kept, so the reduced CE stays
exact at θ ∈ {0, 1} and delayed lookups that land exactly on grid endpoints
contribute nothing to the channel. Returns `nothing` for the `endpoint`
strategy or when no interior node exists.

NOTE — only meaningful for NON-collocation tableaux. For collocation methods
the block data (y_m, Y₁…Yₛ, y_{m+1}) lie ON the degree-s collocation
polynomial, so any interpolant through ≥ s+1 of those points reproduces it
identically and this perturbation is exactly null on the solution manifold;
there the cross-family companion carries the interpolation uncertainty instead.
"""
function reduced_continuous_extension(tab::RKTableau{S, T}) where {S, T}
    tab.strategy == endpoint && return nothing
    nodes_raw = vcat(zero(T), Vector(tab.c), one(T))
    nodes = sort!(unique(nodes_raw))
    interior = [n for n in nodes if !isapprox(n, 0.0; atol=1e-12) && !isapprox(n, 1.0; atol=1e-12)]
    isempty(interior) && return nothing
    drop_node = interior[argmin([(abs(n - 0.5), -n) for n in interior])]
    kept = [n for n in nodes if !isapprox(n, drop_node; atol=1e-12)]
    n_nodes = length(kept)
    nodes_static = SVector{n_nodes, T}(kept)
    # slot map: slot 1 = y_n (θ=0), slots 2..S+1 = stages, slot S+2 = y_{n+1} (θ=1)
    occ_map = ntuple(Val(n_nodes)) do idx
        node = nodes_static[idx]; occ = Int[]
        isapprox(node, 0.0; atol=1e-12) && push!(occ, 1)
        for i in 1:S
            isapprox(tab.c[i], node; atol=1e-12) && push!(occ, i + 1)
        end
        isapprox(node, 1.0; atol=1e-12) && push!(occ, S + 2)
        SVector{length(occ), Int}(occ)
    end
    ce = (theta) -> begin
        w = ntuple(Val(n_nodes)) do i
            wi = one(T)
            for j in 1:n_nodes
                if i != j
                    wi *= (theta - nodes_static[j]) / (nodes_static[i] - nodes_static[j])
                end
            end
            wi
        end
        full = MVector{S + 2, T}(undef)
        for i in 1:(S + 2); full[i] = zero(T); end
        for i in 1:n_nodes
            occs = occ_map[i]; val = w[i] / length(occs)
            for idx in occs; full[idx] = val; end
        end
        return SVector{S + 2, T}(full)
    end
    return ce
end

"""
    order_reduced_collocation_companion(tab::RKTableau) -> RKTableau or nothing

Same-stage-count companion for collocation tableaux, one order class lower on
the maximal-order ladder Gauss (2s) → Radau IIA (2s−1) → Lobatto IIIA (2s−2):
Gauss and every non-Radau collocation scheme get Radau IIA(s); Radau IIA
itself gets Lobatto IIIA(s) (GL(1) for s = 1). Because the stage count is
unchanged, the companion operator lives in the SAME stage-augmented state
space, so ΔΦ = Φ − Φ̂ is a genuine matrix perturbation — and being a complete
collocation method, the companion is only ONE order below the user's method,
which keeps the bar tight (unlike any same-node weight swap, which is capped
at quadrature order s−1: on Gauss nodes every added-sample interpolatory rule
degenerates, ∫ℓ_new ∝ ∫Pₛ = 0 by Legendre orthogonality).

Note the companion has its own abscissae and continuous extension, so this
channel responds to every error mechanism of the pair, not just the update
quadrature.
"""
function order_reduced_collocation_companion(tab::RKTableau{S, T}) where {S, T}
    tab.strategy == collocation || return nothing
    if S == 1
        ie = ImplicitEuler()   # = Radau IIA(1); RungeKutta.jl has no s=1 Radau
        return maximum(abs.(collect(ie.c) .- collect(tab.c))) > 1e-10 ? ie : GL(1)
    end
    radau = from_rkjl(TableauRadauIIA(S), strategy=collocation)
    if maximum(abs.(collect(radau.c) .- collect(tab.c))) > 1e-10
        return radau
    end
    return from_rkjl(TableauLobattoIIIA(S), strategy=collocation)
end

"""
    quadrature_companion_tableau(tab) -> RKTableau or nothing

The tableau whose monodromy operator serves as the quadrature-channel
companion Φ̂: the classical embedded pair when the tableau carries one, the
cross-family collocation companion for collocation schemes, and the drop-node
weight rule otherwise.
"""
function quadrature_companion_tableau(tab::RKTableau)
    tab.b_embedded !== nothing && return embedded_tableau(tab; quadrature=true, interpolation=false)
    comp = order_reduced_collocation_companion(tab)
    comp !== nothing && return comp
    return embedded_tableau(tab; quadrature=true, interpolation=false)
end

"""
    companion_tableau(tab) -> RKTableau or nothing

Full lower-order companion used for the fixed-point (periodic solution)
channel — both the update weights and the interpolation lowered, or the
cross-family collocation companion.
"""
function companion_tableau(tab::RKTableau)
    tab.b_embedded !== nothing && return embedded_tableau(tab; quadrature=true, interpolation=true)
    comp = order_reduced_collocation_companion(tab)
    comp !== nothing && return comp
    return embedded_tableau(tab; quadrature=true, interpolation=true)
end

"""
    embedded_tableau(tab; quadrature=true, interpolation=false) -> RKTableau or nothing

The embedded companion tableau: same stage structure `(a, c)`, with the update
weights and/or the continuous extension replaced by their lower-order
counterparts. Assembling the monodromy operator with the companion and
subtracting gives the mapping-matrix perturbation `ΔΦ` used by the error
estimation. Returns `nothing` when the requested channel is unavailable.
"""
function embedded_tableau(tab::RKTableau{S, T, CE}; quadrature::Bool=true, interpolation::Bool=false) where {S, T, CE}
    (quadrature || interpolation) || throw(ArgumentError("select at least one channel"))
    b_hat = tab.b
    if quadrature
        be = embedded_weights(tab)
        be === nothing && return nothing
        b_hat = be
    end
    ce_hat = tab.ce
    if interpolation
        ce = reduced_continuous_extension(tab)
        ce === nothing && return nothing
        ce_hat = ce
    end
    return RKTableau{S, T, typeof(ce_hat)}(tab.a, b_hat, tab.c, ce_hat,
                                           tab.interp_nodes, tab.order, tab.strategy, nothing)
end
