// The computation on a server instead of WebGPU: the Colab notebook runs gpu/webui/server.jl
// (SOSD.jl's batched solver on the Colab GPU) and serves this page; the page then sends its
// points there. Same interface as engine.js Engine: evaluate(), delaySteps(), deviceName.
// Coarse levels (dragging) in Float32 with a short Krylov–Schur, the final level (opt.final) in
// Float64 with a converged one (tol 1e-10).

import { Engine } from './engine.js';
import { modelJulia } from './expr.js';

const b64 = (f32) => {
  const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
};

export class RemoteEngine {
  /** the server, if this page was served by it (GET api/info); else null */
  static async detect() {
    try {
      const r = await fetch('api/info', { cache: 'no-store' });
      if (!r.ok || !(r.headers.get('content-type') || '').includes('json')) return null;
      return await r.json();
    } catch (e) { return null; }
  }

  static async create(log = () => {}, info = null) {
    info = info || await RemoteEngine.detect();
    if (!info) throw new Error('no SOSD server (api/info) behind this page');
    return new RemoteEngine(info, log);
  }

  constructor(info, log) {
    this.info = { vendor: 'server', description: info.device };
    this.server = info;
    this.log = log;
    this.hasF16 = false; this.hasTS = false;
    this.remote = true;
    this.rate = new Map();          // model code -> points per ms (per precision)
    this.codes = new WeakMap();
  }

  get deviceName() { return this.server.device; }
  release() {}
  lanesPerPoint() { return 1; }

  code(model) {
    let c = this.codes.get(model);
    if (!c) { c = modelJulia(model); this.codes.set(model, c); }
    return c;
  }

  /** as Engine.evaluate; the points go to the server in chunks (progress, cancel between them) */
  async evaluate(model, values, xi, yi, pts, opt) {
    const { S = 3, p = 40, onProgress = null, cancel = null } = opt;
    const final = !!opt.final, forced = !!opt.forced;
    const grid = !(pts instanceof Float32Array);
    const n = grid ? pts.nx * pts.ny : pts.length / 2;
    const at = (i) => {
      if (!grid) return [pts[2 * i], pts[2 * i + 1]];
      const { nx, ny, box } = pts, ii = i % nx, j = (i - ii) / nx;
      return [box.x0 + ii * (box.x1 - box.x0) / Math.max(1, nx - 1), box.y0 + j * (box.y1 - box.y0) / Math.max(1, ny - 1)];
    };
    const r = Engine.prototype.delaySteps.call(this, model, values, xi, yi, n, at, p);
    const code = this.code(model);
    const key = code + '|' + final + '|' + forced + '|' + S + '|' + p;
    const rho = new Float32Array(n), amp = forced ? new Float32Array(n) : null, flags = new Uint8Array(n);
    const t0 = performance.now();
    let off = 0, gpuMs = 0;
    let chunk = Math.min(n, this.rate.has(key) ? Math.max(256, Math.floor(this.rate.get(key) * 400)) : 4096);
    while (off < n) {
      if (cancel && cancel()) break;
      const cnt = Math.min(chunk, n - off);
      const xy = new Float32Array(2 * cnt);
      for (let q = 0; q < cnt; q++) { const [x, y] = at(off + q); xy[2 * q] = x; xy[2 * q + 1] = y; }
      const body = JSON.stringify({ code, values: Array.from(values), xi, yi, xy: b64(xy), S, p, r, final, forced,
        m: opt.m ?? 8, accurate: !!opt.accurate });
      const tc = performance.now();
      const resp = await fetch('api/eval', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body });
      if (!resp.ok) throw new Error('server: ' + (await resp.text()).slice(0, 400));
      const out = new Float32Array(await resp.arrayBuffer());
      gpuMs += +(resp.headers.get('X-Compute-Ms') || 0);
      rho.set(out.subarray(0, cnt), off);
      if (amp) amp.set(out.subarray(cnt, 2 * cnt), off);
      for (let q = 0; q < cnt; q++) flags[off + q] = out[2 * cnt + q];
      const dt = performance.now() - tc;
      this.rate.set(key, cnt / Math.max(dt, 1));
      off += cnt;
      chunk = Math.max(256, Math.floor(this.rate.get(key) * 400));      // ~0.4 s per request
      onProgress?.(off / n);
    }
    const ms = performance.now() - t0;
    return { rho, mu: null, flags, amp, r, ms, lpp: 1, done: off >= n, gpuPrep: 0, gpuMain: gpuMs };
  }

  delaySteps(...a) { return Engine.prototype.delaySteps.apply(this, a); }
}
