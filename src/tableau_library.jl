using RungeKutta
using StaticArrays
using LinearAlgebra

"""
    from_rkjl(tab::Tableau; strategy=nothing, b_embedded=nothing)

Converts a RungeKutta.jl tableau to an SOSD.RKTableau. `b_embedded` optionally
attaches the classical lower-order companion weights of an embedded pair.
"""
function from_rkjl(tab::Tableau; strategy=nothing, b_embedded=nothing)
    s = tab.s
    a_mat = SMatrix{s, s}(tab.a)
    b_vec = SVector{s}(tab.b)
    c_vec = SVector{s}(tab.c)
    order = RungeKutta.order(tab)
    
    if isnothing(strategy)
        method_name = String(tab.name)
        if occursin("Gauss", method_name) || occursin("Lobatto", method_name) || occursin("Radau", method_name) || occursin("CrankNicolson", method_name) || occursin("BackwardEuler", method_name)
            strategy = collocation
        else
            strategy = denseoutput
        end
    end

    ce = nothing
    nodes_raw = Float64[]

    if strategy == collocation
        nodes_raw = vcat(0.0, tab.c, 1.0)
        nodes = unique(nodes_raw); sort!(nodes); n_nodes = length(nodes); nodes_static = SVector{n_nodes}(nodes)
        occ_map = ntuple(Val(n_nodes)) do idx
            node = nodes_static[idx]; occ_indices = Int[]
            if node == 0.0; push!(occ_indices, 1); end
            for i in 1:s; if tab.c[i] == node; push!(occ_indices, i + 1); end; end
            if node == 1.0; push!(occ_indices, s + 2); end
            SVector{length(occ_indices), Int}(occ_indices)
        end
        ce = (theta) -> begin
            w = ntuple(Val(n_nodes)) do i
                wi = 1.0; for j in 1:n_nodes; if i != j; wi *= (theta - nodes_static[j]) / (nodes_static[i] - nodes_static[j]); end; end; wi
            end
            full_weights = MVector{s + 2, Float64}(undef); for i in 1:(s+2); full_weights[i] = 0.0; end
            for i in 1:n_nodes; occs = occ_map[i]; val = w[i] / length(occs); for idx in occs; full_weights[idx] = val; end; end
            return SVector{s+2, Float64}(full_weights)
        end

    elseif strategy == endpoint
        W_val = order
        offset = (W_val - 1) ÷ 2
        ce = (theta) -> begin
            w = ntuple(Val(W_val)) do i
                t_j = float(i - 1 - offset); wi = 1.0
                for j in 1:W_val; if i != j; wi *= (theta - float(j - 1 - offset)) / (t_j - float(j - 1 - offset)); end; end; wi
            end
            return SVector{W_val, Float64}(w)
        end

    elseif strategy == denseoutput
        name = String(tab.name)
        if name == "RK4"
            ce = (theta) -> begin
                w1 = (1-theta)*(1-2*theta)^2
                w23 = 4*theta*(1-theta)
                w4 = theta^2*(2*theta-1)
                return SVector{6, Float64}(w1, 0.0, w23/2, w23/2, 0.0, w4)
            end
        else
            nodes_raw = vcat(0.0, tab.c, 1.0)
            nodes = unique(nodes_raw); sort!(nodes); n_nodes = length(nodes); nodes_static = SVector{n_nodes}(nodes)
            occ_map = ntuple(Val(n_nodes)) do idx
                node = nodes_static[idx]; occ_indices = Int[]
                if node == 0.0; push!(occ_indices, 1); end
                for i in 1:s; if tab.c[i] == node; push!(occ_indices, i + 1); end; end
                if node == 1.0; push!(occ_indices, s + 2); end
                SVector{length(occ_indices), Int}(occ_indices)
            end
            ce = (theta) -> begin
                w = ntuple(Val(n_nodes)) do i
                    wi = 1.0; for j in 1:n_nodes; if i != j; wi *= (theta - nodes_static[j]) / (nodes_static[i] - nodes_static[j]); end; end; wi
                end
                full_weights = MVector{s + 2, Float64}(undef); for i in 1:(s+2); full_weights[i] = 0.0; end
                for i in 1:n_nodes; occs = occ_map[i]; val = w[i] / length(occs); for idx in occs; full_weights[idx] = val; end; end
                return SVector{s+2, Float64}(full_weights)
            end
        end
    end
    
    b_emb = b_embedded === nothing ? nothing : SVector{s, Float64}(b_embedded)
    return RKTableau{s, Float64, typeof(ce)}(a_mat, b_vec, c_vec, ce, nodes_raw, order, strategy, b_emb)
end

"""
    BS3(; strategy=denseoutput)

Bogacki–Shampine 3(2) pair — the method behind MATLAB's `ode23`. The order-2
embedded weights are attached, so `error_estimation` uses the classical pair.
"""
function BS3(; strategy=denseoutput)
    a = [0 0 0 0; 1/2 0 0 0; 0 3/4 0 0; 2/9 1/3 4/9 0]
    b = [2/9, 1/3, 4/9, 0]
    c = [0, 1/2, 3/4, 1]
    tab = Tableau(:BS3, 3, 4, float.(a), float.(b), float.(c))
    return from_rkjl(tab, strategy=strategy, b_embedded=[7/24, 1/4, 1/3, 1/8])
end

function RK8(; strategy=denseoutput)
    s_val = 11; a = zeros(s_val, s_val); c = zeros(s_val); b = zeros(s_val)
    c[2] = 1/2; a[2,1] = 1/2; c[3] = 1/2; a[3,1] = 1/4; a[3,2] = 1/4; c[4] = 1/2; a[4,2] = -1/2; a[4,3] = 1
    c[5] = 1/4; a[5,1] = 7/32; a[5,2] = 0; a[5,3] = 5/32; a[5,4] = -1/16; c[6] = 1/4; a[6,1] = 1/64; a[6,2] = 0; a[6,3] = 0; a[6,4] = 9/64; a[6,5] = 3/32
    c[7] = 1/2; a[7,1] = 0; a[7,2] = 0; a[7,3] = 0; a[7,4] = 3/16; a[7,5] = -3/4; a[7,6] = 17/16
    c[8] = 3/4; a[8,1] = 0; a[8,2] = 0; a[8,3] = 0; a[8,4] = 23/64; a[8,5] = -9/8; a[8,6] = 15/16; a[8,7] = 87/64
    c[9] = 1.0; a[9,1] = 1/16; a[9,2] = 0; a[9,3] = 0; a[9,4] = -3/8; a[9,5] = 2; a[9,6] = -11/8; a[9,7] = -1; a[9,8] = 3/2
    c[10] = 1.0; a[10,1] = 0; a[10,2] = 0; a[10,3] = 0; a[10,4] = -3/10; a[10,5] = 9/5; a[10,6] = -17/20; a[10,7] = -17/20; a[10,8] = 9/10; a[10,9] = 1/4
    c[11] = 1.0; a[11,1] = 1/20; a[11,2] = 0; a[11,3] = 0; a[11,4] = 0; a[11,5] = 0; a[11,6] = 1/4; a[11,7] = 1/4; a[11,8] = 0; a[11,9] = 1/4; a[11,10] = 1/5
    b[1] = 1/20; b[6] = 1/4; b[7] = 1/4; b[9] = 1/4; b[10] = 1/5
    tab = Tableau(:VernerRK8, 8, s_val, a, b, c)
    return from_rkjl(tab, strategy=strategy)
end

GL(s) = from_rkjl(TableauGauss(s), strategy=collocation)
GL2Tableau() = GL(2)
GL3Tableau() = GL(3)
ExplicitEuler(; strategy=denseoutput) = from_rkjl(TableauForwardEuler(), strategy=strategy)
Heun(; strategy=denseoutput) = from_rkjl(TableauHeun2(), strategy=strategy)
RK3(; strategy=denseoutput) = from_rkjl(TableauKutta3(), strategy=strategy)
RK4(; strategy=denseoutput) = from_rkjl(TableauRK4(), strategy=strategy)
RK5(; strategy=denseoutput) = from_rkjl(TableauRK5(), strategy=strategy)
ImplicitEuler() = from_rkjl(TableauBackwardEuler(), strategy=collocation)
ImplicitTrapezoidal() = from_rkjl(TableauCrankNicolson(), strategy=collocation)
