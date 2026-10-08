// Web worker of the CPU pool (cpu.js): ρ for a slice of points with the Float64 solver.
import { rhoPoint } from './cpusolver.js';

let cacheSrc = null, cacheFn = null;

self.onmessage = (ev) => {
  const { id, src, tab, p, m, r, D, values, xi, yi, xy } = ev.data;
  if (src !== cacheSrc) { cacheFn = new Function(src)(); cacheSrc = src; }
  const n = xy.length / 2, rho = new Float32Array(n), P = Float64Array.from(values);
  for (let k = 0; k < n; k++) {
    P[xi] = xy[2 * k]; P[yi] = xy[2 * k + 1];
    let v;
    try { v = rhoPoint(cacheFn, P, tab, p, m, r, D); } catch (e) { v = NaN; }
    rho[k] = isFinite(v) ? v : 3e38;
  }
  self.postMessage({ id, rho }, [rho.buffer]);
};
