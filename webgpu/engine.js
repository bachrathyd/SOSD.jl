// WebGPU host side of the SOSD browser port: device, one compute pipeline per model (sosd.wgsl
// with the generated coefficient functions and the tableau spliced in; cached by its code),
// banded dispatches (each submission stays far below the browser / OS GPU watchdog; the band
// adapts to the measured time), read-back of ρ.

import { modelWGSL, hostEvaluator } from './expr.js';

export class WebGPUUnavailable extends Error {}

// ---------------------------------------------------------------------------------------------
// collocation tableaux (Gauss–Legendre): nodes c, a_ij = ∫₀^{c_i} ℓ_j, b_j = ∫₀¹ ℓ_j
// ---------------------------------------------------------------------------------------------
const GAUSS = {
  1: [0.5],
  2: [0.5 - Math.sqrt(3) / 6, 0.5 + Math.sqrt(3) / 6],
  3: [0.5 - Math.sqrt(15) / 10, 0.5, 0.5 + Math.sqrt(15) / 10],
  4: [0.5 - Math.sqrt(525 + 70 * Math.sqrt(30)) / 70, 0.5 - Math.sqrt(525 - 70 * Math.sqrt(30)) / 70,
      0.5 + Math.sqrt(525 - 70 * Math.sqrt(30)) / 70, 0.5 + Math.sqrt(525 + 70 * Math.sqrt(30)) / 70],
};

/** Lagrange basis polynomial ℓ_j on nodes c as coefficients (ascending powers) */
function lagrangeCoeffs(c, j) {
  let poly = [1];
  for (let k = 0; k < c.length; k++) {
    if (k === j) continue;
    const d = c[j] - c[k], next = new Array(poly.length + 1).fill(0);
    for (let i = 0; i < poly.length; i++) { next[i] += -c[k] * poly[i] / d; next[i + 1] += poly[i] / d; }
    poly = next;
  }
  return poly;
}
const integ = (poly, x) => poly.reduce((acc, co, i) => acc + co * Math.pow(x, i + 1) / (i + 1), 0);

export function gaussTableau(s) {
  const c = GAUSS[s];
  const a = [], b = [];
  for (let j = 0; j < s; j++) b.push(integ(lagrangeCoeffs(c, j), 1));
  for (let i = 0; i < s; i++) for (let j = 0; j < s; j++) a.push(integ(lagrangeCoeffs(c, j), c[i]));
  return { s, a, b, c, order: 2 * s };
}

function tableauWGSL(tab, MMAX, LPP) {
  const f = (v) => { let t = v.toPrecision(9); if (!/[.eE]/.test(t)) t += '.0'; return t; };
  const arr = (v) => `array<f32, ${v.length}>(${v.map(f).join(', ')})`;
  const nodes = [0, ...tab.c, 1];
  return `const S: u32 = ${tab.s}u;\nconst MMAX: u32 = ${MMAX}u;\nconst LPP: u32 = ${LPP}u;\n` +
    `const AT = ${arr(tab.a)};\nconst BT = ${arr(tab.b)};\nconst CT = ${arr(tab.c)};\nconst XN = ${arr(nodes)};\n`;
}

// ---------------------------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------------------------
const OUT_BYTES = 16;            // vec4<f32>: ρ, Re μ, Im μ, flags
const U_BYTES = 96;
const MMAX = 24;                 // largest Krylov dimension (each m is compiled as its own pipeline)

export class Engine {
  static async create(log = () => {}) {
    if (!('gpu' in navigator)) {
      throw new WebGPUUnavailable(window.isSecureContext
        ? 'This browser does not expose WebGPU (navigator.gpu is missing).'
        : 'WebGPU needs a secure context: open the page via https:// or http://localhost.');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new WebGPUUnavailable('WebGPU is present, but no GPU adapter is available.');
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 1 << 30),
        maxBufferSize: Math.min(lim.maxBufferSize, 1 << 30),
      },
    });
    const info = adapter.info || {};
    const r = await fetch('sosd.wgsl', { cache: 'no-cache' });
    if (!r.ok) throw new Error('cannot load sosd.wgsl — serve the folder over http (e.g. python -m http.server)');
    return new Engine(device, info, await r.text(), log);
  }

  constructor(device, info, src, log) {
    this.device = device; this.info = info; this.src = src; this.log = log;
    this.pipelines = new Map();
    this.maxBuf = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    this.lost = null;
    device.lost.then((e) => { this.lost = e; log('GPU device lost: ' + (e.message || e.reason)); });
    device.addEventListener?.('uncapturederror', (ev) => log('WebGPU error: ' + ev.error.message));
    this.ubuf = device.createBuffer({ size: U_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bandTarget = 300;     // ms per submitted band (well below the ~2 s watchdog; fewer syncs)
    this.ptsPerMs = null;      // measured throughput, per pipeline
  }

  get deviceName() {
    const i = this.info;
    return [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(' ') || 'GPU';
  }

  async pipeline(model, S, m, lpp) {
    const code = this.src.replace('//@MODEL@', modelWGSL(model)).replace('//@TABLEAU@', tableauWGSL(gaussTableau(S), m, lpp));
    let p = this.pipelines.get(code);
    if (p) return p;
    const module = this.device.createShaderModule({ code, label: 'sosd' });
    const ci = await module.getCompilationInfo?.();
    const errs = ci ? ci.messages.filter((m) => m.type === 'error') : [];
    if (errs.length) throw new Error('WGSL: ' + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
    const [prep, main] = await Promise.all(['prep', 'main'].map((entryPoint) =>
      this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint } })));
    p = { prep, main, rate: null };
    if (this.pipelines.size > 30) this.pipelines.delete(this.pipelines.keys().next().value);
    this.pipelines.set(code, p);
    return p;
  }

  /**
   * ρ for every (x, y) in `xy` (Float32Array, interleaved): parameters `values` (Float64 array
   * in model.params order) with params[xi] = x, params[yi] = y. Options: S (stages), p (steps
   * per period), m (Krylov dimension), onProgress(fraction).
   * Returns { rho, flags, r, ms (GPU wall time) }.
   */
  async evaluate(model, values, xi, yi, xy, opt) {
    const { S = 3, p = 40, onProgress = null, cancel = null } = opt;
    const m = Math.min(Math.max(opt.m ?? 12, 2), MMAX);
    const n = xy.length / 2;
    const D = model.D, BS = (S + 1) * D;
    const r = this.delaySteps(model, values, xi, yi, xy, p);
    const N = (r + 1) * BS;
    const wFloats = p * BS * BS, lFloats = p * S * (S + 4);     // step matrices, lookup records
    const bytesPerPt = 4 * ((p + r + 1) * BS + (m + 1) * N + wFloats + lFloats);
    // lanes per point: several for a small batch (latency of one point), one for a large grid
    const lpp = opt.lpp ?? this.lanesPerPoint(n, BS);
    const G = 64 / lpp;
    const pl = await this.pipeline(model, S, m, lpp);
    const dev = this.device;
    const ptsBuf = dev.createBuffer({ size: Math.max(16, xy.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(ptsBuf, 0, xy);
    const outBuf = dev.createBuffer({ size: Math.max(16, n * OUT_BYTES), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    // per-buffer size limit, and the prep dispatch (one thread per point and step) ≤ 65535 groups
    const perPt = Math.max((p + r + 1) * BS, (m + 1) * N, wFloats, lFloats) * 4;
    const maxBand = Math.max(64, Math.min(Math.floor(65535 / p) * 64, 65535 * G, Math.floor(this.maxBuf / perPt / 64) * 64 - 64,
                                          Math.floor(1.5e9 / bytesPerPt)));
    let band = Math.min(maxBand, pl.rate ? Math.max(64, Math.floor(pl.rate * this.bandTarget)) : 256);
    let hist = null, V = null, Wb = null, Lb = null, cap = 0;
    const t0 = performance.now();
    for (let off = 0; off < n;) {
      if (cancel && cancel()) break;
      const nb = Math.min(band, n - off);
      if (nb > cap) {
        [hist, V, Wb, Lb].forEach((bfr) => bfr?.destroy());
        cap = nb;
        const mk = (floats) => dev.createBuffer({ size: 4 * floats * Math.ceil(cap / 64) * 64, usage: GPUBufferUsage.STORAGE });
        hist = mk((p + r + 1) * BS); V = mk((m + 1) * N); Wb = mk(wFloats); Lb = mk(lFloats);
      }
      const ub = new ArrayBuffer(U_BYTES), u32 = new Uint32Array(ub), f = new Float32Array(ub);
      u32.set([nb, p, r, m, xi, yi, off, 0]);
      values.forEach((v, k) => { f[8 + k] = v; });
      dev.queue.writeBuffer(this.ubuf, 0, ub);
      const all = { 0: this.ubuf, 1: ptsBuf, 2: hist, 3: V, 4: outBuf, 5: Wb, 6: Lb };
      const group = (pipe, ids) => dev.createBindGroup({ layout: pipe.getBindGroupLayout(0),
        entries: ids.map((k) => ({ binding: k, resource: { buffer: all[k] } })) });
      const enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      if (opt.only !== 'main') {
        pass.setPipeline(pl.prep); pass.setBindGroup(0, group(pl.prep, [0, 1, 5, 6]));
        pass.dispatchWorkgroups(Math.ceil(Math.ceil(nb / G) * G * p / 64));
      }
      if (opt.only !== 'prep') {
        pass.setPipeline(pl.main); pass.setBindGroup(0, group(pl.main, [0, 2, 3, 4, 5, 6]));
        pass.dispatchWorkgroups(Math.ceil(nb / G));
      }
      pass.end();
      const tb = performance.now();
      dev.queue.submit([enc.finish()]);
      await dev.queue.onSubmittedWorkDone();
      const dt = performance.now() - tb;
      pl.rate = nb / Math.max(dt, 1);                            // points per ms
      off += nb;
      band = Math.min(maxBand, Math.max(64, Math.floor(pl.rate * this.bandTarget)));
      onProgress?.(off / n);
    }
    const read = dev.createBuffer({ size: Math.max(16, n * OUT_BYTES), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, read, 0, Math.max(16, n * OUT_BYTES));
    dev.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    const ms = performance.now() - t0;
    [ptsBuf, outBuf, read, hist, V, Wb, Lb].forEach((bfr) => bfr?.destroy());
    const rho = new Float32Array(n), flags = new Uint32Array(n), mu = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
      rho[i] = out[4 * i]; mu[2 * i] = out[4 * i + 1]; mu[2 * i + 1] = out[4 * i + 2]; flags[i] = out[4 * i + 3];
    }
    return { rho, mu, flags, r, ms, bytesPerPt, lpp };
  }

  /** lanes per point: enough threads in flight (≈ 16384) for a batch of n points, at most one
   *  lane per row of the step matrix (BS) and 16 (measured on an AMD Vega iGPU: a 4-point batch
   *  3–8× faster than one lane per point, an 8192-point grid as fast) */
  lanesPerPoint(n, BS) {
    const cap = Math.min(16, 1 << Math.floor(Math.log2(Math.max(1, BS))));
    let l = 1;
    while (l < cap && n * l < 16384) l *= 2;
    return l;
  }

  /** delay window r: the largest τ(t)/h over the points (Float64 on the host, 24 samples per period) */
  delaySteps(model, values, xi, yi, xy, p) {
    const ev = hostEvaluator(model);
    const P = Float64Array.from(values);
    const n = xy.length / 2;
    const stride = Math.max(1, Math.floor(n / 4000));            // a subsample is enough for smooth τ, T
    let rmax = 1;
    const check = (i) => {
      P[xi] = xy[2 * i]; P[yi] = xy[2 * i + 1];
      const T = ev.T(P), h = T / p;
      for (let k = 0; k < 24; k++) {
        const tau = ev.tau(T * k / 24, P);
        if (isFinite(tau) && isFinite(h) && h > 0) rmax = Math.max(rmax, Math.ceil(tau / h + 1e-9) + 1);
      }
    };
    for (let i = 0; i < n; i += stride) check(i);
    if (n > 0) check(n - 1);
    return Math.min(rmax, 4096);
  }
}
