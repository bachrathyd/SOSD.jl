// SOSD in the browser: spectral radius of the one-period monodromy operator of
//     ẋ(t) = A(t) x(t) + B(t) x(t − τ(t)),
// one GPU thread per parameter point (Float32). Port of src/batched.jl:
//   * one period = p collocation steps (Gauss–Legendre, S stages) of length h = T/p, the
//     delayed state from the continuous extension (Lagrange on {0, c, 1}) of the stored steps,
//     history window r steps (state of the operator: (r+1) blocks [y, Y_1..Y_S]);
//   * the stage system of a step is solved on the fly (Gaussian elimination, partial
//     pivoting), so no per-step operators are stored — only the sweep history and the basis;
//   * Arnoldi (m steps, two-pass Gram–Schmidt) on the monodromy, eigenvalues of the m×m
//     Hessenberg matrix by Francis QR (hqr, as webgpu/hqr.js) -> ρ = max |λ|.
// The host replaces the two marker lines below by the model (D, NP, m_period, m_tau, m_AB from
// expr.js) and the tableau (S, MMAX, AT, BT, CT, interpolation nodes XN).

//@MODEL@
//@TABLEAU@

const BS: u32 = (S + 1u) * D;          // block: [y, Y_1 .. Y_S]
const SD: u32 = S * D;
const DD: u32 = D * D;
const NN: u32 = S + 2u;                // interpolation nodes {0, c_1..c_S, 1}

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

var<private> b: u32;        // this point (local index inside the band)
var<private> npts: u32;

fn Hx(k: u32) -> u32 { return k * npts + b; }                 // hist[k] of this point
fn Vx(j: u32, i: u32, N: u32) -> u32 { return (j * N + i) * npts + b; }

fn lagrange(th: f32) -> array<f32, NN> {
  var w: array<f32, NN>;
  for (var i = 0u; i < NN; i++) {
    var l = 1.0;
    for (var k = 0u; k < NN; k++) {
      if (k != i) { l *= (th - XN[k]) / (XN[i] - XN[k]); }
    }
    w[i] = l;
  }
  return w;
}

// Y[jout] = Φ Y[jin] (monodromy applied to basis column jin, result in column jout)
fn sweep(jin: u32, jout: u32, P: array<f32, NP>, h: f32) -> u32 {
  let p = u.p; let r = u.r; let N = (r + 1u) * BS;
  var flags = 0u;
  for (var e = 0u; e < N; e++) {
    let i = e / BS; let q = e % BS;
    hist[Hx((r - i) * BS + q)] = V[Vx(jin, e, N)];
  }
  var Am: array<f32, DD>; var Bm: array<f32, DD>;
  var As: array<f32, S * DD>; var BY: array<f32, SD>;
  var y0: array<f32, D>; var M: array<f32, SD * SD>; var z: array<f32, SD>;
  for (var n = 1u; n <= p; n++) {
    let tn = f32(n - 1u) * h;
    let bc = (n + r - 1u) * BS;
    for (var d = 0u; d < D; d++) { y0[d] = hist[Hx(bc + d)]; }
    for (var s = 0u; s < S; s++) {
      let ts = tn + CT[s] * h;
      m_AB(ts, P, &Am, &Bm);
      for (var k = 0u; k < DD; k++) { As[s * DD + k] = Am[k]; }
      // delayed state at ts − τ(ts): block index (1-based) and position inside the step
      let rel = (ts - m_tau(ts, P)) / h + f32(r) + 1.0;
      if (rel < 1.0 - 1e-4) { flags |= 1u; }
      var mi = i32(floor(rel)); var th = rel - f32(mi);
      if (mi >= i32(p + r + 1u)) { mi = i32(p + r); th = 1.0; }
      if (mi < 1) { mi = 1; th = 0.0; }
      let w = lagrange(th);
      let b0 = u32(mi - 1) * BS; let b1 = u32(mi) * BS;
      var yd: array<f32, D>;
      for (var d = 0u; d < D; d++) {
        var acc = w[0] * hist[Hx(b0 + d)];
        for (var i = 0u; i < S; i++) { acc += w[i + 1u] * hist[Hx(b1 + (i + 1u) * D + d)]; }
        acc += w[S + 1u] * hist[Hx(b1 + d)];
        yd[d] = acc;
      }
      for (var i = 0u; i < D; i++) {
        var acc = 0.0;
        for (var d = 0u; d < D; d++) { acc += Bm[i * D + d] * yd[d]; }
        BY[s * D + i] = acc;
      }
    }
    // stage system  Y_i − h Σ_j a_ij A_j Y_j = y_n + h Σ_j a_ij B_j y_del,j
    for (var i = 0u; i < S; i++) {
      for (var ri = 0u; ri < D; ri++) {
        let row = i * D + ri;
        var rhs = y0[ri];
        for (var j = 0u; j < S; j++) {
          rhs += h * AT[i * S + j] * BY[j * D + ri];
          for (var c = 0u; c < D; c++) {
            var v = -h * AT[i * S + j] * As[j * DD + ri * D + c];
            if (i == j && ri == c) { v += 1.0; }
            M[row * SD + j * D + c] = v;
          }
        }
        z[row] = rhs;
      }
    }
    for (var k = 0u; k < SD; k++) {                    // Gaussian elimination, partial pivoting
      var pk = k; var mx = abs(M[k * SD + k]);
      for (var i = k + 1u; i < SD; i++) { let v = abs(M[i * SD + k]); if (v > mx) { mx = v; pk = i; } }
      if (pk != k) {
        for (var j = 0u; j < SD; j++) { let tmp = M[k * SD + j]; M[k * SD + j] = M[pk * SD + j]; M[pk * SD + j] = tmp; }
        let tz = z[k]; z[k] = z[pk]; z[pk] = tz;
      }
      let inv = 1.0 / M[k * SD + k];
      for (var i = k + 1u; i < SD; i++) {
        let f = M[i * SD + k] * inv;
        if (f != 0.0) {
          for (var j = k + 1u; j < SD; j++) { M[i * SD + j] -= f * M[k * SD + j]; }
          z[i] -= f * z[k];
        }
      }
    }
    for (var ii = 0u; ii < SD; ii++) {
      let i = SD - 1u - ii;
      var acc = z[i];
      for (var j = i + 1u; j < SD; j++) { acc -= M[i * SD + j] * z[j]; }
      z[i] = acc / M[i * SD + i];
    }
    let bn = (n + r) * BS;
    for (var ri = 0u; ri < D; ri++) {
      var acc = y0[ri];
      for (var j = 0u; j < S; j++) {
        var ay = BY[j * D + ri];
        for (var c = 0u; c < D; c++) { ay += As[j * DD + ri * D + c] * z[j * D + c]; }
        acc += h * BT[j] * ay;
      }
      hist[Hx(bn + ri)] = acc;
    }
    for (var k = 0u; k < SD; k++) { hist[Hx(bn + D + k)] = z[k]; }
  }
  for (var e = 0u; e < N; e++) {
    let i = e / BS; let q = e % BS;
    V[Vx(jout, e, N)] = hist[Hx((p + r - i) * BS + q)];
  }
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  b = gid.x; npts = u.npts;
  if (b >= npts) { return; }
  var P: array<f32, NP>;
  for (var k = 0u; k < NP; k++) { P[k] = u.P0[k / 4u][k % 4u]; }
  let xy = pts[u.off + b];
  P[u.xi] = xy.x; P[u.yi] = xy.y;
  let T = m_period(P);
  let h = T / f32(u.p);
  let N = (u.r + 1u) * BS;
  let m = min(u.m, MMAX);
  // start vector (deterministic, as SOSD._det_start)
  var nrm = 0.0;
  for (var i = 0u; i < N; i++) { let v = 1.0 + 0.1 * sin(7.3 * f32(i + 1u)); V[Vx(0u, i, N)] = v; nrm += v * v; }
  nrm = 1.0 / sqrt(nrm);
  for (var i = 0u; i < N; i++) { V[Vx(0u, i, N)] *= nrm; }
  var H: array<f32, (MMAX + 1u) * MMAX>;
  var mm = m; var flags = 0u;
  for (var j = 0u; j < m; j++) {
    flags |= sweep(j, j + 1u, P, h);
    var n0 = 0.0;
    for (var i = 0u; i < N; i++) { let v = V[Vx(j + 1u, i, N)]; n0 += v * v; }
    for (var pss = 0u; pss < 2u; pss++) {
      for (var l = 0u; l <= j; l++) {
        var dt = 0.0;
        for (var i = 0u; i < N; i++) { dt += V[Vx(l, i, N)] * V[Vx(j + 1u, i, N)]; }
        H[l * MMAX + j] += dt;
        for (var i = 0u; i < N; i++) { V[Vx(j + 1u, i, N)] -= dt * V[Vx(l, i, N)]; }
      }
    }
    var beta = 0.0;
    for (var i = 0u; i < N; i++) { let v = V[Vx(j + 1u, i, N)]; beta += v * v; }
    beta = sqrt(beta);
    if (!(beta == beta) || beta > 3.0e38 || !(n0 < 3.0e38)) { flags |= 2u; mm = j + 1u; break; }   // overflow
    H[(j + 1u) * MMAX + j] = beta;
    if (beta <= 1e-6 * sqrt(n0)) { mm = j + 1u; break; }                                          // invariant subspace
    let ib = 1.0 / beta;
    for (var i = 0u; i < N; i++) { V[Vx(j + 1u, i, N)] *= ib; }
  }
  if ((flags & 2u) != 0u) { outv[u.off + b] = vec4<f32>(3.0e38, 0.0, 0.0, f32(flags)); return; }
  for (var i = 0u; i < MMAX * MMAX; i++) { a[i] = 0.0; }
  for (var i = 0u; i < mm; i++) { for (var j = 0u; j < mm; j++) { a[i * MMAX + j] = H[i * MMAX + j]; } }
  let ev = hqr(i32(mm));
  if (ev.w == 0.0) { flags |= 4u; }
  outv[u.off + b] = vec4<f32>(ev.x, ev.y, ev.z, f32(flags));
}
