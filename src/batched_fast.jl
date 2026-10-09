# Fast chart path of the batched solver (the algorithm of the WebGPU page, webgpu/sosd.wgsl):
# one GPU thread per parameter point runs a single Arnoldi pass of m steps on the monodromy
# (classical Gram–Schmidt with one reorthogonalization when needed), the eigenvalues of the m×m
# Hessenberg matrix by Francis QR on the device, and — with a forcing — GMRES in the same Krylov
# space (started from the forced one-period response g) for the periodic orbit and its
# peak-to-peak amplitude. Nothing returns to the host until the chart is done: no restarts, no
# convergence test, so the accuracy is that of the m-dimensional Krylov space (validated on the
# textbook examples: m = 6…8 classify like a converged Krylov–Schur iteration; use
# `spectral_radii` for converged values).

"Francis double-shift QR (EISPACK hqr) of the upper Hessenberg n×n block of `a` → (max |λ|, λ)"
@inline function _hqr_dev!(a::MMatrix{MM, MM, CT}, n::Int, ::Type{CT}) where {MM, CT}
    wr = zero(MVector{MM, CT}); wi = zero(MVector{MM, CT})
    sgn(x, y) = y >= 0 ? abs(x) : -abs(x)
    anorm = zero(CT)
    @inbounds for i in 1:n, j in max(i - 1, 1):n; anorm += abs(a[i, j]); end
    nn = n; t = zero(CT); ok = true
    p = q = r = s = w = x = y = z = zero(CT)
    guard = 0
    @inbounds while nn >= 1 && guard < 100_000
        its = 0
        while true
            guard += 1
            l = nn
            while l >= 2
                s = abs(a[l-1, l-1]) + abs(a[l, l])
                s == 0 && (s = anorm)
                if abs(a[l, l-1]) + s == s
                    a[l, l-1] = zero(CT); break
                end
                l -= 1
            end
            x = a[nn, nn]
            if l == nn
                wr[nn] = x + t; wi[nn] = zero(CT); nn -= 1; break
            end
            y = a[nn-1, nn-1]; w = a[nn, nn-1] * a[nn-1, nn]
            if l == nn - 1
                p = CT(0.5) * (y - x); q = p * p + w; z = sqrt(abs(q)); x += t
                if q >= 0
                    z = p + sgn(z, p)
                    wr[nn-1] = x + z; wr[nn] = x + z
                    z != 0 && (wr[nn] = x - w / z)
                    wi[nn-1] = zero(CT); wi[nn] = zero(CT)
                else
                    wr[nn-1] = x + p; wr[nn] = x + p; wi[nn-1] = -z; wi[nn] = z
                end
                nn -= 2; break
            end
            if its == 60
                ok = false; nn = 0; break
            end
            if its == 10 || its == 20
                t += x
                for i in 1:nn; a[i, i] -= x; end
                s = abs(a[nn, nn-1]) + abs(a[nn-1, nn-2])
                x = CT(0.75) * s; y = x; w = CT(-0.4375) * s * s
            end
            its += 1
            m = nn - 2
            while m >= l
                z = a[m, m]; r = x - z; s = y - z
                p = (r * s - w) / a[m+1, m] + a[m, m+1]
                q = a[m+1, m+1] - z - r - s
                r = a[m+2, m+1]
                s = abs(p) + abs(q) + abs(r)
                p /= s; q /= s; r /= s
                m == l && break
                u = abs(a[m, m-1]) * (abs(q) + abs(r))
                v = abs(p) * (abs(a[m-1, m-1]) + abs(z) + abs(a[m+1, m+1]))
                u + v == v && break
                m -= 1
            end
            for i in m+2:nn
                a[i, i-2] = zero(CT)
                i != m + 2 && (a[i, i-3] = zero(CT))
            end
            for k in m:nn-1
                if k != m
                    p = a[k, k-1]; q = a[k+1, k-1]; r = zero(CT)
                    k != nn - 1 && (r = a[k+2, k-1])
                    x = abs(p) + abs(q) + abs(r)
                    if x != 0; p /= x; q /= x; r /= x; end
                end
                s = sgn(sqrt(p * p + q * q + r * r), p)
                if s != 0
                    if k == m
                        l != m && (a[k, k-1] = -a[k, k-1])
                    else
                        a[k, k-1] = -s * x
                    end
                    p += s; x = p / s; y = q / s; z = r / s; q /= p; r /= p
                    for j in k:nn
                        p = a[k, j] + q * a[k+1, j]
                        if k != nn - 1
                            p += r * a[k+2, j]; a[k+2, j] -= p * z
                        end
                        a[k+1, j] -= p * y; a[k, j] -= p * x
                    end
                    for i in l:min(nn, k + 3)
                        p = x * a[i, k] + y * a[i, k+1]
                        if k != nn - 1
                            p += z * a[i, k+2]; a[i, k+2] -= p * r
                        end
                        a[i, k+1] -= p * q; a[i, k] -= p
                    end
                end
            end
        end
    end
    best = zero(CT); br = zero(CT); bi = zero(CT)
    @inbounds for i in 1:n
        mg = sqrt(wr[i]^2 + wi[i]^2)
        if mg > best; best = mg; br = wr[i]; bi = wi[i]; end
    end
    return best, br, bi, ok
end

# one period for point b: V[b, :, jout] = Φ V[b, :, jin] (+ g when forced); track: min / max of
# the first state component over the step nodes
@inline function _fsweep!(V, jin, jout, Hist, Mp, Md, Midx, Wt, Fv, ew, b, p, r, ::Val{D}, ::Val{S}, ::Val{K},
                          ::Val{COLL}, forced::Bool, track::Bool, ::Type{CT}) where {D, S, K, COLL, CT}
    BS = (S + 1) * D
    lo = CT(Inf); hi = CT(-Inf)
    @inbounds begin
        for e in 1:((r + 1) * BS)
            i = (e - 1) ÷ BS; q = e - i * BS
            Hist[b, (r - i) * BS + q] = V[b, e, jin]
        end
        ycur = MVector{D, CT}(undef)
        ydel = MVector{S * K * D, CT}(undef)
        for n in 1:p
            for d in 1:D; ycur[d] = Hist[b, (n + r - 1) * BS + d]; end
            for k in 1:K, s in 1:S
                m = Midx[b, s, k, n]
                base0 = (m - 1) * BS; base1 = m * BS
                off = ((k - 1) * S + (s - 1)) * D
                for d in 1:D
                    acc = CT(Wt[b, 1, s, k, n]) * Hist[b, base0 + d]
                    for i in 1:S; acc += Wt[b, i + 1, s, k, n] * Hist[b, base1 + i * D + d]; end
                    ydel[off + d] = acc + Wt[b, S + 2, s, k, n] * Hist[b, base1 + d]
                end
            end
            for row in 1:BS
                COLL && row <= D && continue
                acc = zero(CT)
                for d in 1:D; acc += Mp[b, row, d, n] * ycur[d]; end
                for k in 1:K, s in 1:S
                    off = ((k - 1) * S + (s - 1)) * D
                    for d in 1:D; acc += Md[b, row, d, s, k, n] * ydel[off + d]; end
                end
                forced && (acc += Fv[b, row, n])
                Hist[b, (n + r) * BS + row] = acc
                if track && (row - 1) % D == 0
                    lo = min(lo, acc); hi = max(hi, acc)
                end
            end
            if COLL
                for d in 1:D
                    acc = CT(ew[1]) * ycur[d]
                    for i in 1:S; acc += ew[i + 1] * Hist[b, (n + r) * BS + i * D + d]; end
                    Hist[b, (n + r) * BS + d] = acc
                    if track && d == 1
                        lo = min(lo, acc); hi = max(hi, acc)
                    end
                end
            elseif track
                v = CT(Hist[b, (n + r) * BS + 1]); lo = min(lo, v); hi = max(hi, v)
            end
        end
        for e in 1:((r + 1) * BS)
            i = (e - 1) ÷ BS; q = e - i * BS
            V[b, e, jout] = Hist[b, (p + r - i) * BS + q]
        end
    end
    return lo, hi
end

@kernel function _fast_kernel!(out, V, Hist, @Const(Mp), @Const(Md), @Const(Midx), @Const(Wt), @Const(Fv), ew,
                               nb::Int, p::Int, r::Int, ::Val{D}, ::Val{S}, ::Val{K}, ::Val{M},
                               ::Val{COLL}, ::Val{FORCED}) where {D, S, K, M, COLL, FORCED}
    b = @index(Global)
    @inbounds if b <= nb
        CT = promote_type(eltype(V), Float32)
        BS = (S + 1) * D
        N = (r + 1) * BS
        args = (Hist, Mp, Md, Midx, Wt, Fv, ew, b, p, r, Val(D), Val(S), Val(K), Val(COLL))
        # Krylov start: forced -> the response g from a zero history (one space for ρ and the orbit)
        β0 = zero(CT)
        if FORCED
            for i in 1:N; V[b, i, 1] = zero(eltype(V)); end
            _fsweep!(V, 1, 1, args..., true, false, CT)
            for i in 1:N; β0 += CT(V[b, i, 1])^2; end
            β0 = sqrt(β0)
        end
        useg = FORCED && β0 > CT(1e-30)
        nrm = zero(CT)
        for i in 1:N
            v = useg ? CT(V[b, i, 1]) : CT(1) + CT(0.1) * sin(CT(7.3) * i)
            V[b, i, 1] = v; nrm += v * v
        end
        nrm = 1 / sqrt(nrm)
        for i in 1:N; V[b, i, 1] *= nrm; end
        H = zero(MMatrix{M + 1, M, CT})
        h = MVector{M, CT}(undef); h2 = MVector{M, CT}(undef)
        mm = M; ovf = false
        brk = CT === Float64 ? CT(1e-12) : CT(1e-6)
        for j in 1:M
            _fsweep!(V, j, j + 1, args..., false, false, CT)
            for l in 1:j; h[l] = zero(CT); h2[l] = zero(CT); end
            n0 = zero(CT)
            for i in 1:N
                w = CT(V[b, i, j + 1]); n0 += w * w
                for l in 1:j; h[l] += V[b, i, l] * w; end
            end
            b2 = zero(CT)
            for i in 1:N
                w = CT(V[b, i, j + 1])
                for l in 1:j; w -= h[l] * V[b, i, l]; end
                V[b, i, j + 1] = w; b2 += w * w
                for l in 1:j; h2[l] += V[b, i, l] * w; end
            end
            hh = zero(CT)
            for l in 1:j; hh += h2[l]^2; end
            if hh > CT(1e-12) * b2                    # reorthogonalization
                b2 = zero(CT)
                for i in 1:N
                    w = CT(V[b, i, j + 1])
                    for l in 1:j; w -= h2[l] * V[b, i, l]; end
                    V[b, i, j + 1] = w; b2 += w * w
                end
                for l in 1:j; h[l] += h2[l]; end
            end
            for l in 1:j; H[l, j] = h[l]; end
            β = sqrt(b2)
            if !(isfinite(β) && isfinite(n0))
                ovf = true; mm = j; break
            end
            H[j + 1, j] = β
            if β <= brk * sqrt(n0)
                mm = j; break
            end
            for i in 1:N; V[b, i, j + 1] = V[b, i, j + 1] / β; end
        end
        if ovf
            out[b, 1] = CT(Inf); out[b, 2] = zero(CT)
        else
            a = zero(MMatrix{M, M, CT})
            for j in 1:mm, i in 1:mm; a[i, j] = H[i, j]; end
            ρ, _, _, _ = _hqr_dev!(a, mm, CT)
            out[b, 1] = ρ
            out[b, 2] = zero(CT)
            if FORCED && useg && ρ < 1
                # GMRES: min ‖β₀ e₁ − (Ĩ − H) y‖ by Givens rotations, x = V y -> column M + 1
                G = zero(MMatrix{M + 1, M, CT})
                for j in 1:mm, i in 1:mm+1; G[i, j] = (i == j ? one(CT) : zero(CT)) - H[i, j]; end
                rhs = zero(MVector{M + 1, CT}); rhs[1] = β0
                for j in 1:mm
                    x = G[j, j]; z = G[j + 1, j]; rr = sqrt(x * x + z * z)
                    c = rr > 0 ? x / rr : one(CT); s = rr > 0 ? z / rr : zero(CT)
                    for k in j:mm
                        p0 = G[j, k]; p1 = G[j + 1, k]
                        G[j, k] = c * p0 + s * p1; G[j + 1, k] = -s * p0 + c * p1
                    end
                    r0 = rhs[j]; r1 = rhs[j + 1]
                    rhs[j] = c * r0 + s * r1; rhs[j + 1] = -s * r0 + c * r1
                end
                y = zero(MVector{M, CT})
                for i in mm:-1:1
                    acc = rhs[i]
                    for k in i+1:mm; acc -= G[i, k] * y[k]; end
                    y[i] = acc / G[i, i]
                end
                for i in 1:N
                    acc = zero(CT)
                    for l in 1:mm; acc += y[l] * V[b, i, l]; end
                    V[b, i, M + 1] = acc
                end
                lo, hi = _fsweep!(V, M + 1, 1, args..., true, true, CT)
                out[b, 2] = hi - lo
            end
        end
    end
end

"""
    fast_spectral_radii(prob, θs, tab, p, r; m=8, backend=CPU(), T=Float32, forcing=nothing,
                        batchsize=:auto) -> (rho, amp, flag)

The fast chart path (see the header of this file): ρ of every point from one m-step Arnoldi pass
on the device; with `forcing = f(t, θ)` also the peak-to-peak amplitude of the first state on
the periodic orbit (`amp`, 0 where ρ ≥ 1). `flag` as in [`BatchedEigResult`](@ref).
"""
function fast_spectral_radii(prob::BatchedLDDE{D, K}, θs::AbstractVector, tab::RKTableau{S}, p::Int, r::Int;
                             m::Int=8, backend=CPU(), T::Type=Float32, forcing=nothing, batchsize=:auto) where {D, K, S}
    n = length(θs)
    rho = zeros(n); amp = zeros(n); flag = zeros(Int32, n)
    N = (r + 1) * (S + 1) * D
    m = clamp(m, 2, min(24, N - 1))
    bpp = bytes_per_point(D, S, K, p, r, m, T) + (S + 1) * D * p * sizeof(T)
    bs = batchsize === :auto ? max(1, min(n, floor(Int, 0.8 * available_memory(backend) / bpp))) : batchsize
    CT = promote_type(T, Float32)
    for lo in 1:bs:n
        idx = lo:min(n, lo + bs - 1); nb = length(idx)
        op = build_batched_operators(prob, θs[idx], tab, p, r; backend=backend, T=T)
        forced = forcing !== nothing
        forced && (op = with_forcing(op, prob, forcing, θs[idx], tab; backend=backend))
        V = KA.allocate(backend, T, nb, N, m + 1)
        out = KA.zeros(backend, CT, nb, 2)
        ew = SVector{S + 1, CT}(ntuple(i -> CT(op.ew[i]), Val(S + 1)))
        _fast_kernel!(backend, backend isa KA.CPU ? 1 : 64)(out, V, op.Hist, op.Mp, op.Md, op.Midx, op.Wt, op.Fv, ew,
            nb, p, r, Val(D), Val(S), Val(K), Val(m), Val(op.colloc), Val(forced); ndrange=nb)
        KA.synchronize(backend)
        o = Array(out)
        rho[idx] .= Float64.(o[:, 1]); amp[idx] .= Float64.(o[:, 2]); flag[idx] .= Array(op.flag)
        op = nothing; V = nothing
    end
    return (rho = rho, amp = amp, flag = flag)
end
