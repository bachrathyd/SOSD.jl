// SOSD in the browser: spectral radius of the one-period monodromy operator of
//     ẋ(t) = A(t) x(t) + B(t) x(t − τ(t)),
// Float32, port of src/batched.jl. One period = p collocation steps (Gauss–Legendre, S stages)
// of length h = T/p; the delayed state comes from the continuous extension (Lagrange on
// {0, c, 1}) of the stored steps; history window r steps, operator state (r+1) blocks
// [y, Y_1..Y_S]. Two entry points:
//   prep — one thread per (point, step): the step map does not depend on the Krylov vector, so
//          it is built ONCE: the stage system is solved for all right-hand sides (Gaussian
//          elimination, partial pivoting), giving the SD×BS step matrix of the stage values
//              [Y_1..Y_S] = W_n [y_n; yd_1..yd_S]      (yd_s = x(t_s − τ(t_s)))
//          and per stage the delayed-lookup record (block index, flag, Lagrange weights);
//          the end value is the collocation polynomial at 1: y_{n+1} = EW·[y_n; Y_1..Y_S]
//          (exactly the Runge–Kutta update of a collocation method, without its D rows of W);
//   main — LPP threads per point (G = 64/LPP points per workgroup): Arnoldi (m steps,
//          classical Gram–Schmidt with DGKS reorthogonalization) on the monodromy, every
//          matvec = p small mat-vecs with the stored W_n, rows shared by the lanes; eigenvalues
//          of the m×m Hessenberg matrix by Francis QR (hqr, as webgpu/hqr.js) -> ρ = max |λ|.
//          Several lanes per point cut the latency of a small batch (MDBM stages), one lane per
//          point is best for large grids; the host chooses.
// The host replaces the two marker lines below by the model (D, NP, m_period, m_tau, m_AB from
// expr.js) and the tableau (S, MMAX = Krylov dimension, LPP, AT, BT, CT, interpolation nodes XN).

//@MODEL@
//@TABLEAU@

const BS: u32 = (S + 1u) * D;          // block: [y, Y_1 .. Y_S]
const SD: u32 = S * D;
const DD: u32 = D * D;
const NN: u32 = S + 2u;                // interpolation nodes {0, c_1..c_S, 1}
const LR: u32 = NN + 2u;               // lookup record: block index, flag, NN weights
const WW: u32 = SD * BS;              // step matrix: stage rows only
const G: u32 = WG / LPP;              // points per workgroup = chunk size of the storage layout

struct U {
  npts: u32, p: u32, r: u32, m: u32,
  xi: u32, yi: u32, off: u32, pad: u32,
  P0: array<vec4<f32>, 4>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pts: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> hist: array<SH>;            // ST: f32, or f16 (fast mode)
@group(0) @binding(3) var<storage, read_write> V: array<SV>;
@group(0) @binding(4) var<storage, read_write> outv: array<vec4<f32>>;   // ρ, Re μ, Im μ, flags
@group(0) @binding(5) var<storage, read_write> Wb: array<SW>;             // step matrices − identity part
@group(0) @binding(6) var<storage, read_write> Lb: array<f32>;            // lookup records

var<private> b: u32;        // this point (local index inside the band)
var<private> npts: u32;
var<private> cb: u32;       // its chunk of G points
var<private> lane: u32;     // its slot in the chunk

// Every per-point array is stored in chunks of G points, element-major inside a chunk:
// neighbouring threads touch neighbouring words AND one point's data stays on a few pages
// (a plain point-fastest layout puts consecutive elements npts·4 bytes apart, and the TLB misses
// then dominate as soon as the working set grows). W_n is stored column-major, so the lanes of a
// point (consecutive rows) read consecutive words.
fn Hx(k: u32) -> u32 { return (cb * ((u.p + u.r + 1u) * BS) + k) * G + lane; }
fn Vx(j: u32, i: u32, N: u32) -> u32 { return (cb * ((min(u.m, MMAX) + 1u) * N) + j * N + i) * G + lane; }
fn Wx(n: u32, k: u32) -> u32 { return (cb * (u.p * WW) + n * WW + k) * G + lane; }      // n = 0..p-1
fn Fx(n: u32, k: u32) -> u32 { return (cb * (u.p * SD) + n * SD + k) * G + lane; }
fn Lx(n: u32, s: u32, k: u32) -> u32 { return (cb * (u.p * S * LR) + (n * S + s) * LR + k) * G + lane; }

// storage accessors: values are stored as ST (f16 in the fast mode), computed in f32
fn hR(k: u32) -> f32 { return f32(hist[Hx(k)]); }
fn hS(k: u32, v: f32) { hist[Hx(k)] = SH(v); }
fn vR(j: u32, i: u32, N: u32) -> f32 { return f32(V[Vx(j, i, N)]); }
fn vS(j: u32, i: u32, N: u32, v: f32) { V[Vx(j, i, N)] = SV(v); }

fn params() -> array<f32, NP> {
  var P: array<f32, NP>;
  for (var k = 0u; k < NP; k++) { P[k] = u.P0[k / 4u][k % 4u]; }
  let xy = pts[b];
  P[u.xi] = xy.x; P[u.yi] = xy.y;
  return P;
}

@compute @workgroup_size(64)
fn prep(@builtin(global_invocation_id) gid: vec3<u32>) {
  npts = u.npts;
  let g = gid.x;
  lane = g % G;
  let rest = g / G;
  let n = rest % u.p;                    // step 0..p-1
  cb = rest / u.p;
  b = cb * G + lane;
  if (b >= npts) { return; }
  let P = params();
  let h = m_period(P) / f32(u.p);
  let r = u.r;
  let tn = f32(n) * h;
  var As: array<f32, S * DD>; var Bs: array<f32, S * DD>;
  var Am: array<f32, DD>; var Bm: array<f32, DD>;
  var Fs: array<f32, SD>; var Fm: array<f32, D>;
  for (var s = 0u; s < S; s++) {
    let ts = tn + CT[s] * h;
    m_AB(ts, P, &Am, &Bm);
    if (FORCED) { m_F(ts, P, &Fm); for (var d = 0u; d < D; d++) { Fs[s * D + d] = Fm[d]; } }
    for (var k = 0u; k < DD; k++) { As[s * DD + k] = Am[k]; Bs[s * DD + k] = Bm[k]; }
    // delayed state at ts − τ(ts): block index (1-based) and position inside the step
    let rel = (ts - m_tau(ts, P)) / h + f32(r) + 1.0;
    var fl = 0.0;
    if (rel < 1.0 - 1e-4) { fl = 1.0; }
    var mi = i32(floor(rel)); var th = rel - f32(mi);
    if (mi >= i32(u.p + r + 1u)) { mi = i32(u.p + r); th = 1.0; }
    if (mi < 1) { mi = 1; th = 0.0; }
    Lb[Lx(n, s, 0u)] = f32(mi);
    Lb[Lx(n, s, 1u)] = fl;
    for (var i = 0u; i < NN; i++) {
      var l = 1.0;
      for (var k = 0u; k < NN; k++) { if (k != i) { l *= (th - XN[k]) / (XN[i] - XN[k]); } }
      Lb[Lx(n, s, 2u + i)] = l;
    }
  }
  // stage system  Y_i − h Σ_j a_ij A_j Y_j = y_n + h Σ_j a_ij B_j yd_j  for all BS right-hand
  // sides at once: R = [1⊗I_D | h (a⊗I) blkdiag(B_j)]
  var M: array<f32, SD * SD>; var R: array<f32, SD * BS>;
  var zf: array<f32, SD>;                  // forcing: h Σ_j a_ij f(t_j)
  for (var i = 0u; i < S; i++) {
    for (var ri = 0u; ri < D; ri++) {
      let row = i * D + ri;
      for (var c = 0u; c < BS; c++) { R[row * BS + c] = 0.0; }
      R[row * BS + ri] = 1.0;
      for (var j = 0u; j < S; j++) {
        let ha = h * AT[i * S + j];
        if (FORCED) { zf[row] += ha * Fs[j * D + ri]; }
        for (var c = 0u; c < D; c++) {
          var v = -ha * As[j * DD + ri * D + c];
          if (i == j && ri == c) { v += 1.0; }
          M[row * SD + j * D + c] = v;
          R[row * BS + D + j * D + c] = ha * Bs[j * DD + ri * D + c];
        }
      }
    }
  }
  for (var k = 0u; k < SD; k++) {
    var pk = k; var mx = abs(M[k * SD + k]);
    for (var i = k + 1u; i < SD; i++) { let v = abs(M[i * SD + k]); if (v > mx) { mx = v; pk = i; } }
    if (pk != k) {
      for (var j = 0u; j < SD; j++) { let t = M[k * SD + j]; M[k * SD + j] = M[pk * SD + j]; M[pk * SD + j] = t; }
      for (var j = 0u; j < BS; j++) { let t = R[k * BS + j]; R[k * BS + j] = R[pk * BS + j]; R[pk * BS + j] = t; }
      if (FORCED) { let t = zf[k]; zf[k] = zf[pk]; zf[pk] = t; }
    }
    let inv = 1.0 / M[k * SD + k];
    for (var i = k + 1u; i < SD; i++) {
      let f = M[i * SD + k] * inv;
      if (f != 0.0) {
        for (var j = k + 1u; j < SD; j++) { M[i * SD + j] -= f * M[k * SD + j]; }
        for (var j = 0u; j < BS; j++) { R[i * BS + j] -= f * R[k * BS + j]; }
        if (FORCED) { zf[i] -= f * zf[k]; }
      }
    }
  }
  for (var ii = 0u; ii < SD; ii++) {
    let i = SD - 1u - ii;
    let inv = 1.0 / M[i * SD + i];
    for (var c = 0u; c < BS; c++) {
      var acc = R[i * BS + c];
      for (var j = i + 1u; j < SD; j++) { acc -= M[i * SD + j] * R[j * BS + c]; }
      R[i * BS + c] = acc * inv;
    }
    if (FORCED) {
      var acc = zf[i];
      for (var j = i + 1u; j < SD; j++) { acc -= M[i * SD + j] * zf[j]; }
      zf[i] = acc * inv;
    }
  }
  if (FORCED) { for (var k = 0u; k < SD; k++) { fbW(Fx(n, k), zf[k]); } }
  // column-major, minus the identity part Y_i ≈ y_n (the O(h) remainder keeps its accuracy in f16)
  for (var k = 0u; k < SD * BS; k++) {
    let row = k / BS; let c = k % BS;
    Wb[Wx(n, c * SD + row)] = SW(R[k] - select(0.0, 1.0, c == row % D));
  }
}

// Workgroup = G points × LPP lanes per point. The lanes of a point share every loop over the
// state (row / element index ≡ lane mod LPP); the sweep's step-to-step dependency and the
// Gram–Schmidt sums are synchronized with barriers. All loop bounds come from the uniform
// buffer, so every invocation reaches every barrier (a finished point idles along).
var<workgroup> Xs: array<f32, G * BS>;                     // [y_n; yd_1..yd_S] per point
var<workgroup> red: array<f32, WG * (MMAX + 1u)>;         // per-lane partial sums

var<private> ln: u32;       // lane of this invocation inside its point
var<private> gp: u32;       // point inside the workgroup

/** sum over the LPP lanes of this point of red[lane][0..cnt) -> out (identical in every lane) */
fn reduce(cnt: u32, out: ptr<function, array<f32, MMAX + 1u>>) {
  if (LPP > 1u) { workgroupBarrier(); }
  for (var l = 0u; l < cnt; l++) {
    var acc = 0.0;
    for (var q = 0u; q < LPP; q++) { acc += red[(gp * LPP + q) * (MMAX + 1u) + l]; }
    (*out)[l] = acc;
  }
  if (LPP > 1u) { workgroupBarrier(); }
}

/** barrier between the lanes of a point (none with one lane per point: program order suffices) */
fn bar() { if (LPP > 1u) { storageBarrier(); workgroupBarrier(); } }

var<private> tmin: f32;     // peak tracking of the first state component (periodic orbit)
var<private> tmax: f32;

// Y[jout] = Φ Y[jin] (monodromy applied to basis column jin, result in column jout); forced: with
// the forcing term (the affine one-period map); track: min / max of the first state component
fn sweep(jin: u32, jout: u32, forced: bool, track: bool) {
  let p = u.p; let r = u.r; let N = (r + 1u) * BS;
  for (var e = ln; e < N; e += LPP) {
    let i = e / BS; let q = e % BS;
    hS((r - i) * BS + q, vR(jin, e, N));
  }
  bar();
  for (var n = 0u; n < p; n++) {
    let bc = (n + r) * BS;
    for (var k = ln; k < D; k += LPP) { Xs[gp * BS + k] = hR(bc + k); }
    for (var st = 0u; st < S; st++) {
      // the delayed state of stage st: one lookup record, D interpolations
      let mi = u32(Lb[Lx(n, st, 0u)]);
      var w: array<f32, NN>;
      for (var i = 0u; i < NN; i++) { w[i] = Lb[Lx(n, st, 2u + i)]; }
      let b0 = (mi - 1u) * BS; let b1 = mi * BS;
      for (var d = 0u; d < D; d++) {
        let k = D + st * D + d;
        if (k % LPP == ln) {
          var x = w[0] * hR(b0 + d);
          for (var i = 0u; i < S; i++) { x += w[i + 1u] * hR(b1 + (i + 1u) * D + d); }
          Xs[gp * BS + k] = x + w[S + 1u] * hR(b1 + d);
        }
      }
    }
    if (LPP > 1u) { workgroupBarrier(); }
    let bn = (n + r + 1u) * BS;
    for (var row = ln; row < SD; row += LPP) {
      var acc = Xs[gp * BS + row % D];                 // identity part: Y_i ≈ y_n
      for (var c = 0u; c < BS; c++) { acc += f32(Wb[Wx(n, c * SD + row)]) * Xs[gp * BS + c]; }
      if (FORCED && forced) { acc += fbR(Fx(n, row)); }
      if (track && row % D == 0u) { tmin = min(tmin, acc); tmax = max(tmax, acc); }
      hS(bn + D + row, acc);
    }
    bar();
    // end value: the collocation polynomial at θ = 1
    for (var d = ln; d < D; d += LPP) {
      var y = EW[0] * Xs[gp * BS + d];
      for (var i = 0u; i < S; i++) { y += EW[i + 1u] * hR(bn + D + i * D + d); }
      if (track && d == 0u) { tmin = min(tmin, y); tmax = max(tmax, y); }
      hS(bn + d, y);
    }
    bar();
  }
  for (var e = ln; e < N; e += LPP) {
    let i = e / BS; let q = e % BS;
    vS(jout, e, N, hR((p + r - i) * BS + q));
  }
  bar();
}

// eigenvalues of the n×n upper Hessenberg matrix in a (row-major, stride MMAX): max |λ| and λ
var<private> a: array<f32, MMAX * MMAX>;
fn A_(i: u32, j: u32) -> f32 { return a[i * MMAX + j]; }
fn S_(i: u32, j: u32, v: f32) { a[i * MMAX + j] = v; }
fn sgn(x: f32, y: f32) -> f32 { return select(-abs(x), abs(x), y >= 0.0); }

fn hqr(n: i32) -> vec4<f32> {
  var wr: array<f32, MMAX>; var wi: array<f32, MMAX>;
  var anorm = 0.0;
  for (var i = 0; i < n; i++) { for (var j = max(i - 1, 0); j < n; j++) { anorm += abs(A_(u32(i), u32(j))); } }
  var nn = n - 1; var t = 0.0; var ok = 1.0;
  var p = 0.0; var q = 0.0; var r = 0.0; var s = 0.0; var w = 0.0; var x = 0.0; var y = 0.0; var z = 0.0;
  var guard = 0;
  while (nn >= 0 && guard < 100000) {
    var its = 0; var l = 0;
    loop {
      guard++;
      l = nn;
      while (l >= 1) {
        s = abs(A_(u32(l - 1), u32(l - 1))) + abs(A_(u32(l), u32(l)));
        if (s == 0.0) { s = anorm; }
        if (abs(A_(u32(l), u32(l - 1))) + s == s) { S_(u32(l), u32(l - 1), 0.0); break; }
        l--;
      }
      x = A_(u32(nn), u32(nn));
      if (l == nn) { wr[nn] = x + t; wi[nn] = 0.0; nn--; break; }
      y = A_(u32(nn - 1), u32(nn - 1));
      w = A_(u32(nn), u32(nn - 1)) * A_(u32(nn - 1), u32(nn));
      if (l == nn - 1) {
        p = 0.5 * (y - x); q = p * p + w; z = sqrt(abs(q)); x += t;
        if (q >= 0.0) {
          z = p + sgn(z, p);
          wr[nn - 1] = x + z; wr[nn] = x + z;
          if (z != 0.0) { wr[nn] = x - w / z; }
          wi[nn - 1] = 0.0; wi[nn] = 0.0;
        } else {
          wr[nn - 1] = x + p; wr[nn] = x + p;
          wi[nn - 1] = -z; wi[nn] = z;
        }
        nn -= 2; break;
      }
      if (its == 60) { ok = 0.0; nn = -1; break; }
      if (its == 10 || its == 20) {
        t += x;
        for (var i = 0; i <= nn; i++) { S_(u32(i), u32(i), A_(u32(i), u32(i)) - x); }
        s = abs(A_(u32(nn), u32(nn - 1))) + abs(A_(u32(nn - 1), u32(nn - 2)));
        x = 0.75 * s; y = x; w = -0.4375 * s * s;
      }
      its++;
      var m = nn - 2;
      while (m >= l) {
        z = A_(u32(m), u32(m));
        r = x - z; s = y - z;
        p = (r * s - w) / A_(u32(m + 1), u32(m)) + A_(u32(m), u32(m + 1));
        q = A_(u32(m + 1), u32(m + 1)) - z - r - s;
        r = A_(u32(m + 2), u32(m + 1));
        s = abs(p) + abs(q) + abs(r);
        p /= s; q /= s; r /= s;
        if (m == l) { break; }
        let uu = abs(A_(u32(m), u32(m - 1))) * (abs(q) + abs(r));
        let vv = abs(p) * (abs(A_(u32(m - 1), u32(m - 1))) + abs(z) + abs(A_(u32(m + 1), u32(m + 1))));
        if (uu + vv == vv) { break; }
        m--;
      }
      for (var i = m + 2; i <= nn; i++) {
        S_(u32(i), u32(i - 2), 0.0);
        if (i != m + 2) { S_(u32(i), u32(i - 3), 0.0); }
      }
      for (var k = m; k <= nn - 1; k++) {
        if (k != m) {
          p = A_(u32(k), u32(k - 1)); q = A_(u32(k + 1), u32(k - 1)); r = 0.0;
          if (k != nn - 1) { r = A_(u32(k + 2), u32(k - 1)); }
          x = abs(p) + abs(q) + abs(r);
          if (x != 0.0) { p /= x; q /= x; r /= x; }
        }
        s = sgn(sqrt(p * p + q * q + r * r), p);
        if (s != 0.0) {
          if (k == m) { if (l != m) { S_(u32(k), u32(k - 1), -A_(u32(k), u32(k - 1))); } }
          else { S_(u32(k), u32(k - 1), -s * x); }
          p += s; x = p / s; y = q / s; z = r / s; q /= p; r /= p;
          for (var j = k; j <= nn; j++) {
            p = A_(u32(k), u32(j)) + q * A_(u32(k + 1), u32(j));
            if (k != nn - 1) { p += r * A_(u32(k + 2), u32(j)); S_(u32(k + 2), u32(j), A_(u32(k + 2), u32(j)) - p * z); }
            S_(u32(k + 1), u32(j), A_(u32(k + 1), u32(j)) - p * y);
            S_(u32(k), u32(j), A_(u32(k), u32(j)) - p * x);
          }
          let mmin = min(nn, k + 3);
          for (var i = l; i <= mmin; i++) {
            p = x * A_(u32(i), u32(k)) + y * A_(u32(i), u32(k + 1));
            if (k != nn - 1) { p += z * A_(u32(i), u32(k + 2)); S_(u32(i), u32(k + 2), A_(u32(i), u32(k + 2)) - p * r); }
            S_(u32(i), u32(k + 1), A_(u32(i), u32(k + 1)) - p * q);
            S_(u32(i), u32(k), A_(u32(i), u32(k)) - p);
          }
        }
      }
    }
  }
  var best = 0.0; var br = 0.0; var bi = 0.0;
  for (var i = 0; i < n; i++) {
    let mag = sqrt(wr[i] * wr[i] + wi[i] * wi[i]);
    if (mag > best) { best = mag; br = wr[i]; bi = wi[i]; }
  }
  return vec4<f32>(best, br, bi, ok);
}

// Arnoldi on the monodromy from the (normalized) vector in column 0 of V: m steps, Hessenberg
// matrix in H; -> (steps done, flags). Classical Gram–Schmidt with one reorthogonalization,
// fused (ORTH = 1): pass A the dots h = Vᵀw; pass B the update w −= V h together with the second
// dots h' = Vᵀw (from the same loads of V); pass C the correction w −= V h' only if h' is not
// negligible. ORTH = 0: pass A and B only (no reorthogonalization).
var<private> H: array<f32, (MMAX + 1u) * MMAX>;
var<private> li: u32;       // local invocation index

fn arnoldi(m: u32) -> vec2<u32> {
  let N = (u.r + 1u) * BS;
  for (var i = 0u; i < (MMAX + 1u) * MMAX; i++) { H[i] = 0.0; }
  var mm = m; var flags = 0u; var alive = true;
  for (var j = 0u; j < m; j++) {
    sweep(j, j + 1u, false, false);
    var hc: array<f32, MMAX + 1u>;
    {
      var hp: array<f32, MMAX + 1u>;
      var w2 = 0.0;
      for (var i = ln; i < N; i += LPP) {
        let w = vR(j + 1u, i, N);
        w2 += w * w;
        for (var l = 0u; l < MMAX; l++) { if (l <= j) { hp[l] += vR(l, i, N) * w; } }
      }
      for (var l = 0u; l < MMAX; l++) { if (l <= j) { red[li * (MMAX + 1u) + l] = hp[l]; } }
      red[li * (MMAX + 1u) + MMAX] = w2;
    }
    reduce(MMAX + 1u, &hc);
    let n0 = hc[MMAX];
    var h2: array<f32, MMAX + 1u>;
    {
      var hp: array<f32, MMAX + 1u>;
      var b2 = 0.0;
      for (var i = ln; i < N; i += LPP) {
        var vl: array<f32, MMAX>;
        var w = vR(j + 1u, i, N);
        for (var l = 0u; l < MMAX; l++) { if (l <= j) { vl[l] = vR(l, i, N); w -= hc[l] * vl[l]; } }
        vS(j + 1u, i, N, w);
        b2 += w * w;
        if (ORTH == 1u) { for (var l = 0u; l < MMAX; l++) { if (l <= j) { hp[l] += vl[l] * w; } } }
      }
      for (var l = 0u; l < MMAX; l++) { if (l <= j) { red[li * (MMAX + 1u) + l] = hp[l]; } }
      red[li * (MMAX + 1u) + MMAX] = b2;
    }
    reduce(MMAX + 1u, &h2);
    var beta = h2[MMAX];
    for (var l = 0u; l < MMAX; l++) { if (l <= j) { H[l * MMAX + j] = hc[l]; } }
    if (ORTH == 1u) {
      var hh = 0.0;
      for (var l = 0u; l < MMAX; l++) { if (l <= j) { hh += h2[l] * h2[l]; } }
      let fix = hh > 1e-12 * beta;           // the same in every lane of the point
      var b3 = 0.0;
      if (fix) {
        for (var i = ln; i < N; i += LPP) {
          var w = vR(j + 1u, i, N);
          for (var l = 0u; l < MMAX; l++) { if (l <= j) { w -= h2[l] * vR(l, i, N); } }
          vS(j + 1u, i, N, w);
          b3 += w * w;
        }
      }
      red[li * (MMAX + 1u)] = b3;
      var bb: array<f32, MMAX + 1u>;
      reduce(1u, &bb);
      if (fix) {
        beta = bb[0];
        for (var l = 0u; l < MMAX; l++) { if (l <= j) { H[l * MMAX + j] += h2[l]; } }
      }
    }
    beta = sqrt(beta);
    if (alive) {
      if (!(beta == beta) || beta > 3.0e38 || !(n0 < 3.0e38)) { flags |= 2u; mm = j + 1u; alive = false; }   // overflow
      else {
        H[(j + 1u) * MMAX + j] = beta;
        if (beta <= 1e-6 * sqrt(n0)) { mm = j + 1u; alive = false; }                                      // invariant subspace
      }
    }
    // a finished point keeps sweeping (barriers) on a harmless normalized vector
    let ib = select(1.0 / max(sqrt(n0), 1e-30), 1.0 / beta, alive);
    for (var i = ln; i < N; i += LPP) { vS(j + 1u, i, N, vR(j + 1u, i, N) * ib); }
    bar();
  }
  return vec2<u32>(mm, flags);
}

/** normalize column 0 of V (sum over the lanes) -> its norm */
fn normalize0() -> f32 {
  let N = (u.r + 1u) * BS;
  var nrm = 0.0;
  for (var i = ln; i < N; i += LPP) { let v = vR(0u, i, N); nrm += v * v; }
  red[li * (MMAX + 1u)] = nrm;
  var sv: array<f32, MMAX + 1u>;
  reduce(1u, &sv);
  let beta = sqrt(sv[0]);
  let ib = 1.0 / max(beta, 1e-30);
  for (var i = ln; i < N; i += LPP) { vS(0u, i, N, vR(0u, i, N) * ib); }
  bar();
  return beta;
}

// GMRES for the periodic orbit: (I − Φ) x = g in the Krylov space of Φ from g (V, H of arnoldi):
// min ‖β e₁ − (Ĩ − H) y‖ by Givens rotations; -> y (identical in every lane)
var<private> ga: array<f32, (MMAX + 1u) * MMAX>;
fn gmres_y(mm: u32, beta: f32) -> array<f32, MMAX> {
  for (var i = 0u; i <= mm; i++) {
    for (var j = 0u; j < mm; j++) { ga[i * MMAX + j] = select(0.0, 1.0, i == j) - H[i * MMAX + j]; }
  }
  var rhs: array<f32, MMAX + 1u>;
  rhs[0] = beta;
  for (var j = 0u; j < mm; j++) {
    let x = ga[j * MMAX + j]; let z = ga[(j + 1u) * MMAX + j];
    let rr = sqrt(x * x + z * z);
    let c = select(1.0, x / rr, rr > 0.0); let s = select(0.0, z / rr, rr > 0.0);
    for (var k = j; k < mm; k++) {
      let p0 = ga[j * MMAX + k]; let p1 = ga[(j + 1u) * MMAX + k];
      ga[j * MMAX + k] = c * p0 + s * p1; ga[(j + 1u) * MMAX + k] = -s * p0 + c * p1;
    }
    let r0 = rhs[j]; let r1 = rhs[j + 1u];
    rhs[j] = c * r0 + s * r1; rhs[j + 1u] = -s * r0 + c * r1;
  }
  var y: array<f32, MMAX>;
  for (var ii = 0u; ii < mm; ii++) {
    let i = mm - 1u - ii;
    var acc = rhs[i];
    for (var k = i + 1u; k < mm; k++) { acc -= ga[i * MMAX + k] * y[k]; }
    y[i] = acc / ga[i * MMAX + i];
  }
  return y;
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  npts = u.npts;
  li = lid;
  gp = lid / LPP; ln = lid % LPP;
  cb = wid.x; lane = gp;
  b = cb * G + gp;
  let valid = b < npts;                  // a padding point computes garbage, its result is not written
  let N = (u.r + 1u) * BS;
  let m = min(u.m, MMAX);
  // start vector (deterministic, as SOSD._det_start)
  for (var i = ln; i < N; i += LPP) { vS(0u, i, N, 1.0 + 0.1 * sin(7.3 * f32(i + 1u))); }
  bar();
  _ = normalize0();
  var flags = 0u;
  // delay outside the stored window at some stage (flag 1)
  if (ln == 0u) {
    for (var n = 0u; n < u.p; n++) { for (var st = 0u; st < S; st++) { if (Lb[Lx(n, st, 1u)] != 0.0) { flags |= 1u; } } }
  }
  let ar = arnoldi(m);
  let mm = ar.x; flags |= ar.y;
  var ev = vec4<f32>(3.0e38, 0.0, 0.0, 0.0);
  if (ln == 0u && (flags & 2u) == 0u) {
    for (var i = 0u; i < MMAX * MMAX; i++) { a[i] = 0.0; }
    for (var i = 0u; i < mm; i++) { for (var j = 0u; j < mm; j++) { a[i * MMAX + j] = H[i * MMAX + j]; } }
    ev = hqr(i32(mm));
    if (ev.w == 0.0) { flags |= 4u; }
  }
  var amp = 0.0;
  if (FORCED) {
    // the response to the forcing over one period from a zero history: g
    for (var i = ln; i < N; i += LPP) { vS(0u, i, N, 0.0); }
    bar();
    sweep(0u, 0u, true, false);
    let beta = normalize0();
    // periodic orbit: (I − Φ) x = g by GMRES in the Krylov space of Φ from g
    let ag = arnoldi(m);
    let y = gmres_y(ag.x, beta);
    for (var i = ln; i < N; i += LPP) {
      var x = 0.0;
      for (var j = 0u; j < MMAX; j++) { if (j < ag.x) { x += y[j] * vR(j, i, N); } }
      vS(m, i, N, x);
    }
    bar();
    // one period along the orbit: peak-to-peak of the first state component
    tmin = 3.0e38; tmax = -3.0e38;
    sweep(m, 0u, true, true);
    red[li * (MMAX + 1u)] = tmin; red[li * (MMAX + 1u) + 1u] = tmax;
    if (LPP > 1u) { workgroupBarrier(); }
    var lo = 3.0e38; var hi = -3.0e38;
    for (var q = 0u; q < LPP; q++) {
      lo = min(lo, red[(gp * LPP + q) * (MMAX + 1u)]); hi = max(hi, red[(gp * LPP + q) * (MMAX + 1u) + 1u]);
    }
    if (LPP > 1u) { workgroupBarrier(); }
    amp = hi - lo;
    if (!(amp == amp)) { amp = 3.0e38; }
  }
  if (ln != 0u || !valid) { return; }
  if ((flags & 2u) != 0u) { outv[b] = vec4<f32>(3.0e38, 0.0, 0.0, f32(flags)); return; }
  outv[b] = vec4<f32>(ev.x, ev.y, select(ev.z, amp, FORCED), f32(flags));
}
