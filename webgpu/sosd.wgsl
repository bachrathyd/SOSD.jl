// SOSD in the browser: spectral radius of the one-period monodromy operator of
//     ẋ(t) = A(t) x(t) + B(t) x(t − τ(t)),
// Float32, port of src/batched.jl. One period = p collocation steps (Gauss–Legendre, S stages)
// of length h = T/p; the delayed state comes from the continuous extension (Lagrange on
// {0, c, 1}) of the stored steps; history window r steps, operator state (r+1) blocks
// [y, Y_1..Y_S]. Two entry points:
//   prep — one thread per (point, step): the step map does not depend on the Krylov vector, so
//          it is built ONCE: the stage system is solved for all right-hand sides (Gaussian
//          elimination, partial pivoting), giving the BS×BS step matrix
//              [y_{n+1}; Y_1..Y_S] = W_n [y_n; yd_1..yd_S]      (yd_s = x(t_s − τ(t_s)))
//          and per stage the delayed-lookup record (block index, flag, Lagrange weights);
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
const WW: u32 = BS * BS;
const G: u32 = 64u / LPP;             // points per workgroup = chunk size of the storage layout

struct U {
  npts: u32, p: u32, r: u32, m: u32,
  xi: u32, yi: u32, off: u32, pad: u32,
  P0: array<vec4<f32>, 4>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pts: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> hist: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> outv: array<vec4<f32>>;   // ρ, Re μ, Im μ, flags
@group(0) @binding(5) var<storage, read_write> Wb: array<f32>;            // step matrices
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
fn Lx(n: u32, s: u32, k: u32) -> u32 { return (cb * (u.p * S * LR) + (n * S + s) * LR + k) * G + lane; }

fn params() -> array<f32, NP> {
  var P: array<f32, NP>;
  for (var k = 0u; k < NP; k++) { P[k] = u.P0[k / 4u][k % 4u]; }
  let xy = pts[u.off + b];
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
  for (var s = 0u; s < S; s++) {
    let ts = tn + CT[s] * h;
    m_AB(ts, P, &Am, &Bm);
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
  for (var i = 0u; i < S; i++) {
    for (var ri = 0u; ri < D; ri++) {
      let row = i * D + ri;
      for (var c = 0u; c < BS; c++) { R[row * BS + c] = 0.0; }
      R[row * BS + ri] = 1.0;
      for (var j = 0u; j < S; j++) {
        let ha = h * AT[i * S + j];
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
    }
    let inv = 1.0 / M[k * SD + k];
    for (var i = k + 1u; i < SD; i++) {
      let f = M[i * SD + k] * inv;
      if (f != 0.0) {
        for (var j = k + 1u; j < SD; j++) { M[i * SD + j] -= f * M[k * SD + j]; }
        for (var j = 0u; j < BS; j++) { R[i * BS + j] -= f * R[k * BS + j]; }
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
  }
  // rows of y_{n+1} = y_n + h Σ_j b_j (A_j Y_j + B_j yd_j); then the stage rows
  for (var ri = 0u; ri < D; ri++) {
    for (var c = 0u; c < BS; c++) {
      var acc = 0.0;
      if (c == ri) { acc = 1.0; }
      for (var j = 0u; j < S; j++) {
        var ay = 0.0;
        for (var q = 0u; q < D; q++) { ay += As[j * DD + ri * D + q] * R[(j * D + q) * BS + c]; }
        if (c >= D + j * D && c < D + (j + 1u) * D) { ay += Bs[j * DD + ri * D + (c - D - j * D)]; }
        acc += h * BT[j] * ay;
      }
      Wb[Wx(n, c * BS + ri)] = acc;
    }
  }
  for (var k = 0u; k < SD * BS; k++) { Wb[Wx(n, (k % BS) * BS + D + k / BS)] = R[k]; }
}

// Workgroup = G points × LPP lanes per point. The lanes of a point share every loop over the
// state (row / element index ≡ lane mod LPP); the sweep's step-to-step dependency and the
// Gram–Schmidt sums are synchronized with barriers. All loop bounds come from the uniform
// buffer, so every invocation reaches every barrier (a finished point idles along).
var<workgroup> Xs: array<f32, G * BS>;                     // [y_n; yd_1..yd_S] per point
var<workgroup> red: array<f32, 64u * (MMAX + 1u)>;         // per-lane partial sums

var<private> ln: u32;       // lane of this invocation inside its point
var<private> gp: u32;       // point inside the workgroup

/** sum over the LPP lanes of this point of red[lane][0..cnt) -> out (identical in every lane) */
fn reduce(cnt: u32, out: ptr<function, array<f32, MMAX + 1u>>) {
  workgroupBarrier();
  for (var l = 0u; l < cnt; l++) {
    var acc = 0.0;
    for (var q = 0u; q < LPP; q++) { acc += red[(gp * LPP + q) * (MMAX + 1u) + l]; }
    (*out)[l] = acc;
  }
  workgroupBarrier();
}

// Y[jout] = Φ Y[jin] (monodromy applied to basis column jin, result in column jout)
fn sweep(jin: u32, jout: u32) -> u32 {
  let p = u.p; let r = u.r; let N = (r + 1u) * BS;
  var flags = 0u;
  for (var e = ln; e < N; e += LPP) {
    let i = e / BS; let q = e % BS;
    hist[Hx((r - i) * BS + q)] = V[Vx(jin, e, N)];
  }
  storageBarrier();
  for (var n = 0u; n < p; n++) {
    let bc = (n + r) * BS;
    for (var k = ln; k < BS; k += LPP) {
      var x = 0.0;
      if (k < D) {
        x = hist[Hx(bc + k)];
      } else {
        let s = (k - D) / D; let d = (k - D) % D;
        let mi = u32(Lb[Lx(n, s, 0u)]);
        if (Lb[Lx(n, s, 1u)] != 0.0) { flags |= 1u; }
        let b0 = (mi - 1u) * BS; let b1 = mi * BS;
        x = Lb[Lx(n, s, 2u)] * hist[Hx(b0 + d)];
        for (var i = 0u; i < S; i++) { x += Lb[Lx(n, s, 3u + i)] * hist[Hx(b1 + (i + 1u) * D + d)]; }
        x += Lb[Lx(n, s, 2u + S + 1u)] * hist[Hx(b1 + d)];
      }
      Xs[gp * BS + k] = x;
    }
    workgroupBarrier();
    let bn = (n + r + 1u) * BS;
    for (var row = ln; row < BS; row += LPP) {
      var acc = 0.0;
      for (var c = 0u; c < BS; c++) { acc += Wb[Wx(n, c * BS + row)] * Xs[gp * BS + c]; }
      hist[Hx(bn + row)] = acc;
    }
    storageBarrier();
    workgroupBarrier();
  }
  for (var e = ln; e < N; e += LPP) {
    let i = e / BS; let q = e % BS;
    V[Vx(jout, e, N)] = hist[Hx((p + r - i) * BS + q)];
  }
  storageBarrier();
  return flags;
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

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  npts = u.npts;
  gp = lid / LPP; ln = lid % LPP;
  cb = wid.x; lane = gp;
  b = cb * G + gp;
  let valid = b < npts;                  // a padding point computes garbage, its result is not written
  let N = (u.r + 1u) * BS;
  let m = min(u.m, MMAX);
  // start vector (deterministic, as SOSD._det_start)
  var sv: array<f32, MMAX + 1u>;
  var nrm = 0.0;
  for (var i = ln; i < N; i += LPP) { let v = 1.0 + 0.1 * sin(7.3 * f32(i + 1u)); nrm += v * v; }
  red[lid * (MMAX + 1u)] = nrm;
  reduce(1u, &sv);
  nrm = 1.0 / sqrt(sv[0]);
  for (var i = ln; i < N; i += LPP) { V[Vx(0u, i, N)] = (1.0 + 0.1 * sin(7.3 * f32(i + 1u))) * nrm; }
  storageBarrier();
  var H: array<f32, (MMAX + 1u) * MMAX>;
  var mm = m; var flags = 0u; var alive = true;
  for (var j = 0u; j < m; j++) {
    flags |= sweep(j, j + 1u);
    // classical Gram–Schmidt: one pass for all the dots, one for the update; repeated only when
    // the norm dropped by more than 1/√2 (DGKS criterion), which keeps the basis orthogonal in f32
    var n0 = 0.0; var beta = 0.0; var nprev = 0.0; var again = true;
    for (var pss = 0u; pss < 2u; pss++) {
      var hp: array<f32, MMAX + 1u>;
      var w2 = 0.0;
      if (again) {
        for (var i = ln; i < N; i += LPP) {
          let w = V[Vx(j + 1u, i, N)];
          w2 += w * w;
          for (var l = 0u; l <= j; l++) { hp[l] += V[Vx(l, i, N)] * w; }
        }
      }
      for (var l = 0u; l <= j; l++) { red[lid * (MMAX + 1u) + l] = hp[l]; }
      red[lid * (MMAX + 1u) + MMAX] = w2;
      var hc: array<f32, MMAX + 1u>;
      reduce(MMAX + 1u, &hc);
      var b2 = 0.0;
      if (again) {
        if (pss == 0u) { n0 = hc[MMAX]; nprev = n0; }
        for (var i = ln; i < N; i += LPP) {
          var w = V[Vx(j + 1u, i, N)];
          for (var l = 0u; l <= j; l++) { w -= hc[l] * V[Vx(l, i, N)]; }
          V[Vx(j + 1u, i, N)] = w;
          b2 += w * w;
        }
      }
      red[lid * (MMAX + 1u)] = b2;
      var bs: array<f32, MMAX + 1u>;
      reduce(1u, &bs);
      if (again) {
        for (var l = 0u; l <= j; l++) { H[l * MMAX + j] += hc[l]; }
        beta = bs[0];
        if (beta > 0.5 * nprev) { again = false; }
        nprev = beta;
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
    for (var i = ln; i < N; i += LPP) { V[Vx(j + 1u, i, N)] *= ib; }
    storageBarrier();
  }
  if (ln != 0u || !valid) { return; }
  if ((flags & 2u) != 0u) { outv[u.off + b] = vec4<f32>(3.0e38, 0.0, 0.0, f32(flags)); return; }
  for (var i = 0u; i < MMAX * MMAX; i++) { a[i] = 0.0; }
  for (var i = 0u; i < mm; i++) { for (var j = 0u; j < mm; j++) { a[i * MMAX + j] = H[i * MMAX + j]; } }
  let ev = hqr(i32(mm));
  if (ev.w == 0.0) { flags |= 4u; }
  outv[u.off + b] = vec4<f32>(ev.x, ev.y, ev.z, f32(flags));
}
