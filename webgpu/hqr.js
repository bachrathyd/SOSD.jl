// Eigenvalues of a real upper Hessenberg matrix (Francis double-shift QR, EISPACK hqr).
// Host reference of the WGSL routine in sosd.wgsl (same structure, line by line), used by the
// validation page and the unit test (validate/hqr_test.mjs).
//
// a: Float64Array n*n (row-major, a[i*n + j]); overwritten. Returns {wr, wi, ok}.
export function hqr(a, n) {
  const A = (i, j) => a[i * n + j];
  const S = (i, j, v) => { a[i * n + j] = v; };
  const sign = (x, y) => (y >= 0 ? Math.abs(x) : -Math.abs(x));
  const wr = new Float64Array(n), wi = new Float64Array(n);
  let anorm = 0;
  for (let i = 0; i < n; i++) for (let j = Math.max(i - 1, 0); j < n; j++) anorm += Math.abs(A(i, j));
  let nn = n - 1, t = 0, ok = true;
  let p = 0, q = 0, r = 0, s = 0, w = 0, x = 0, y = 0, z = 0;
  while (nn >= 0) {
    let its = 0, l = 0;
    for (;;) {
      for (l = nn; l >= 1; l--) {
        s = Math.abs(A(l - 1, l - 1)) + Math.abs(A(l, l));
        if (s === 0) s = anorm;
        if (Math.abs(A(l, l - 1)) + s === s) { S(l, l - 1, 0); break; }
      }
      x = A(nn, nn);
      if (l === nn) {                       // one root found
        wr[nn] = x + t; wi[nn] = 0; nn--; break;
      }
      y = A(nn - 1, nn - 1);
      w = A(nn, nn - 1) * A(nn - 1, nn);
      if (l === nn - 1) {                   // two roots found
        p = 0.5 * (y - x); q = p * p + w; z = Math.sqrt(Math.abs(q)); x += t;
        if (q >= 0) {
          z = p + sign(z, p);
          wr[nn - 1] = wr[nn] = x + z;
          if (z) wr[nn] = x - w / z;
          wi[nn - 1] = wi[nn] = 0;
        } else {
          wr[nn - 1] = wr[nn] = x + p;
          wi[nn - 1] = -z; wi[nn] = z;
        }
        nn -= 2; break;
      }
      if (its === 60) { ok = false; nn = -1; break; }   // no convergence
      if (its === 10 || its === 20) {       // exceptional shift
        t += x;
        for (let i = 0; i <= nn; i++) S(i, i, A(i, i) - x);
        s = Math.abs(A(nn, nn - 1)) + Math.abs(A(nn - 1, nn - 2));
        y = x = 0.75 * s;
        w = -0.4375 * s * s;
      }
      its++;
      let m;
      for (m = nn - 2; m >= l; m--) {
        z = A(m, m);
        r = x - z; s = y - z;
        p = (r * s - w) / A(m + 1, m) + A(m, m + 1);
        q = A(m + 1, m + 1) - z - r - s;
        r = A(m + 2, m + 1);
        s = Math.abs(p) + Math.abs(q) + Math.abs(r);
        p /= s; q /= s; r /= s;
        if (m === l) break;
        const u = Math.abs(A(m, m - 1)) * (Math.abs(q) + Math.abs(r));
        const v = Math.abs(p) * (Math.abs(A(m - 1, m - 1)) + Math.abs(z) + Math.abs(A(m + 1, m + 1)));
        if (u + v === v) break;
      }
      for (let i = m + 2; i <= nn; i++) {
        S(i, i - 2, 0);
        if (i !== m + 2) S(i, i - 3, 0);
      }
      for (let k = m; k <= nn - 1; k++) {
        if (k !== m) {
          p = A(k, k - 1); q = A(k + 1, k - 1); r = 0;
          if (k !== nn - 1) r = A(k + 2, k - 1);
          x = Math.abs(p) + Math.abs(q) + Math.abs(r);
          if (x !== 0) { p /= x; q /= x; r /= x; }
        }
        s = sign(Math.sqrt(p * p + q * q + r * r), p);
        if (s !== 0) {
          if (k === m) { if (l !== m) S(k, k - 1, -A(k, k - 1)); }
          else S(k, k - 1, -s * x);
          p += s; x = p / s; y = q / s; z = r / s; q /= p; r /= p;
          for (let j = k; j <= nn; j++) {
            p = A(k, j) + q * A(k + 1, j);
            if (k !== nn - 1) { p += r * A(k + 2, j); S(k + 2, j, A(k + 2, j) - p * z); }
            S(k + 1, j, A(k + 1, j) - p * y);
            S(k, j, A(k, j) - p * x);
          }
          const mmin = nn < k + 3 ? nn : k + 3;
          for (let i = l; i <= mmin; i++) {
            p = x * A(i, k) + y * A(i, k + 1);
            if (k !== nn - 1) { p += z * A(i, k + 2); S(i, k + 2, A(i, k + 2) - p * r); }
            S(i, k + 1, A(i, k + 1) - p * q);
            S(i, k, A(i, k) - p);
          }
        }
      }
      if (l >= nn - 1) break;
    }
  }
  return { wr, wi, ok };
}
