// Pool of web workers running the Float64 CPU solver (cpusolver.js). For a handful of points
// the CPU answers sooner than a GPU round trip; the MDBM neighbour check uses it.

import { modelJS } from './expr.js';
import { gaussTableau } from './engine.js';

export class CpuPool {
  constructor(size = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1))) {
    this.size = size;
    this.workers = [];
    this.next = 1;
    this.waiting = new Map();
  }

  worker(k) {
    if (!this.workers[k]) {
      const w = new Worker(new URL('./cpuworker.js', import.meta.url), { type: 'module' });
      w.onmessage = (ev) => { const cb = this.waiting.get(ev.data.id); this.waiting.delete(ev.data.id); cb?.(ev.data.rho); };
      this.workers[k] = w;
    }
    return this.workers[k];
  }

  /** ρ for the points xy (Float32Array, interleaved), split over the workers */
  async evaluate(model, values, xi, yi, xy, { S = 3, p = 40, m = 12, r }) {
    const n = xy.length / 2;
    if (!n) return new Float32Array(0);
    const src = modelJS(model), t = gaussTableau(S);
    const tab = { s: t.s, a: t.a, b: t.b, c: t.c };
    const parts = Math.min(this.size, n), per = Math.ceil(n / parts);
    const jobs = [];
    for (let k = 0; k < parts; k++) {
      const lo = k * per, hi = Math.min(n, lo + per);
      if (lo >= hi) break;
      const id = this.next++;
      jobs.push(new Promise((res) => {
        this.waiting.set(id, res);
        this.worker(k).postMessage({ id, src, tab, p, m, r, D: model.D, values: Array.from(values), xi, yi,
                                     xy: xy.slice(2 * lo, 2 * hi) });
      }).then((rho) => [lo, rho]));
    }
    const out = new Float32Array(n);
    for (const [lo, rho] of await Promise.all(jobs)) out.set(rho, lo);
    return out;
  }
}
