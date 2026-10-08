// Float64 CPU twin of sosd.wgsl (prep + sweep + Arnoldi + Francis QR) for one parameter point.
// Used by the web workers of cpu.js for the small, almost sequential batches of the MDBM
// neighbour check, where the GPU latency per call is larger than the work.

import { hqr } from './hqr.js';

/**
 * ρ of the monodromy operator for the parameter vector P.
 * f: { T, tau, AB } compiled by expr.js modelJS; tab: Gauss–Legendre tableau {s, a, b, c};
 * p steps per period, m Krylov dimension, r delay window (steps), D state dimension.
 */
export function rhoPoint(f, P, tab, p, m, r, D) {
  const S = tab.s, BS = (S + 1) * D, SD = S * D, DD = D * D, NN = S + 2;
  const XN = [0, ...tab.c, 1];
  const h = f.T(P) / p;
  // step matrices W_n (row-major BS×BS) and delayed lookups (block index, weights)
  const W = new Float64Array(p * BS * BS), Lmi = new Int32Array(p * S), Lw = new Float64Array(p * S * NN);
  const As = new Float64Array(S * DD), Bs = new Float64Array(S * DD), Am = new Float64Array(DD), Bm = new Float64Array(DD);
  const M = new Float64Array(SD * SD), R = new Float64Array(SD * BS);
  for (let n = 0; n < p; n++) {
    const tn = n * h;
    for (let s = 0; s < S; s++) {
      const ts = tn + tab.c[s] * h;
      f.AB(ts, P, Am, Bm);
      As.set(Am, s * DD); Bs.set(Bm, s * DD);
      const rel = (ts - f.tau(ts, P)) / h + r + 1;
      let mi = Math.floor(rel), th = rel - mi;
      if (mi >= p + r + 1) { mi = p + r; th = 1; }
      if (mi < 1) { mi = 1; th = 0; }
      Lmi[n * S + s] = mi;
      for (let i = 0; i < NN; i++) {
        let l = 1;
        for (let k = 0; k < NN; k++) if (k !== i) l *= (th - XN[k]) / (XN[i] - XN[k]);
        Lw[(n * S + s) * NN + i] = l;
      }
    }
    M.fill(0); R.fill(0);
    for (let i = 0; i < S; i++) for (let ri = 0; ri < D; ri++) {
      const row = i * D + ri;
      R[row * BS + ri] = 1;
      for (let j = 0; j < S; j++) {
        const ha = h * tab.a[i * S + j];
        for (let c = 0; c < D; c++) {
          M[row * SD + j * D + c] = -ha * As[j * DD + ri * D + c] + (i === j && ri === c ? 1 : 0);
          R[row * BS + D + j * D + c] = ha * Bs[j * DD + ri * D + c];
        }
      }
    }
    for (let k = 0; k < SD; k++) {
      let pk = k, mx = Math.abs(M[k * SD + k]);
      for (let i = k + 1; i < SD; i++) { const v = Math.abs(M[i * SD + k]); if (v > mx) { mx = v; pk = i; } }
      if (pk !== k) {
        for (let j = 0; j < SD; j++) { const t = M[k * SD + j]; M[k * SD + j] = M[pk * SD + j]; M[pk * SD + j] = t; }
        for (let j = 0; j < BS; j++) { const t = R[k * BS + j]; R[k * BS + j] = R[pk * BS + j]; R[pk * BS + j] = t; }
      }
      const inv = 1 / M[k * SD + k];
      for (let i = k + 1; i < SD; i++) {
        const q = M[i * SD + k] * inv;
        if (q !== 0) {
          for (let j = k + 1; j < SD; j++) M[i * SD + j] -= q * M[k * SD + j];
          for (let j = 0; j < BS; j++) R[i * BS + j] -= q * R[k * BS + j];
        }
      }
    }
    for (let i = SD - 1; i >= 0; i--) {
      const inv = 1 / M[i * SD + i];
      for (let c = 0; c < BS; c++) {
        let acc = R[i * BS + c];
        for (let j = i + 1; j < SD; j++) acc -= M[i * SD + j] * R[j * BS + c];
        R[i * BS + c] = acc * inv;
      }
    }
    const w0 = n * BS * BS;
    for (let ri = 0; ri < D; ri++) for (let c = 0; c < BS; c++) {
      let acc = c === ri ? 1 : 0;
      for (let j = 0; j < S; j++) {
        let ay = 0;
        for (let q = 0; q < D; q++) ay += As[j * DD + ri * D + q] * R[(j * D + q) * BS + c];
        if (c >= D + j * D && c < D + (j + 1) * D) ay += Bs[j * DD + ri * D + (c - D - j * D)];
        acc += h * tab.b[j] * ay;
      }
      W[w0 + ri * BS + c] = acc;
    }
    W.set(R, w0 + D * BS);
  }
  // Arnoldi on the monodromy (state: r+1 blocks), CGS with DGKS reorthogonalization
  const N = (r + 1) * BS, hist = new Float64Array((p + r + 1) * BS), X = new Float64Array(BS);
  const V = new Float64Array((m + 1) * N), H = new Float64Array((m + 1) * m);
  const sweep = (jin, jout) => {
    for (let e = 0; e < N; e++) hist[(r - Math.floor(e / BS)) * BS + e % BS] = V[jin * N + e];
    for (let n = 0; n < p; n++) {
      const bc = (n + r) * BS;
      for (let d = 0; d < D; d++) X[d] = hist[bc + d];
      for (let s = 0; s < S; s++) {
        const mi = Lmi[n * S + s], w = (n * S + s) * NN, b0 = (mi - 1) * BS, b1 = mi * BS;
        for (let d = 0; d < D; d++) {
          let acc = Lw[w] * hist[b0 + d];
          for (let i = 0; i < S; i++) acc += Lw[w + i + 1] * hist[b1 + (i + 1) * D + d];
          X[D + s * D + d] = acc + Lw[w + S + 1] * hist[b1 + d];
        }
      }
      const bn = (n + r + 1) * BS, w0 = n * BS * BS;
      for (let row = 0; row < BS; row++) {
        let acc = 0;
        for (let c = 0; c < BS; c++) acc += W[w0 + row * BS + c] * X[c];
        hist[bn + row] = acc;
      }
    }
    for (let e = 0; e < N; e++) V[jout * N + e] = hist[(p + r - Math.floor(e / BS)) * BS + e % BS];
  };
  let nrm = 0;
  for (let i = 0; i < N; i++) { const v = 1 + 0.1 * Math.sin(7.3 * (i + 1)); V[i] = v; nrm += v * v; }
  nrm = 1 / Math.sqrt(nrm);
  for (let i = 0; i < N; i++) V[i] *= nrm;
  let mm = m;
  const hc = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    sweep(j, j + 1);
    const o = (j + 1) * N;
    let n0 = 0, beta = 0, nprev = 0;
    for (let pass = 0; pass < 2; pass++) {
      hc.fill(0);
      let w2 = 0;
      for (let i = 0; i < N; i++) {
        const w = V[o + i];
        w2 += w * w;
        for (let l = 0; l <= j; l++) hc[l] += V[l * N + i] * w;
      }
      if (pass === 0) { n0 = w2; nprev = w2; }
      beta = 0;
      for (let i = 0; i < N; i++) {
        let w = V[o + i];
        for (let l = 0; l <= j; l++) w -= hc[l] * V[l * N + i];
        V[o + i] = w; beta += w * w;
      }
      for (let l = 0; l <= j; l++) H[l * m + j] += hc[l];
      if (beta > 0.5 * nprev) break;
      nprev = beta;
    }
    beta = Math.sqrt(beta);
    if (!isFinite(beta) || !isFinite(n0)) return Infinity;
    H[(j + 1) * m + j] = beta;
    if (beta <= 1e-12 * Math.sqrt(n0)) { mm = j + 1; break; }
    for (let i = 0; i < N; i++) V[o + i] /= beta;
  }
  const a = new Float64Array(mm * mm);
  for (let i = 0; i < mm; i++) for (let j = 0; j < mm; j++) a[i * mm + j] = H[i * m + j];
  const { wr, wi } = hqr(a, mm);
  let best = 0;
  for (let i = 0; i < mm; i++) best = Math.max(best, Math.hypot(wr[i], wi[i]));
  return best;
}
