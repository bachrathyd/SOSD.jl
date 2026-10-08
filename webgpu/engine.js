// WebGPU host side of the SOSD browser port: device, one compute pipeline per model (sosd.wgsl
// with the generated coefficient functions and the tableau spliced in; cached by its code),
// banded dispatches (each submission stays far below the browser / OS GPU watchdog; the band
// adapts to the measured time), read-back of ρ.

import { modelWGSL, modelJS } from './expr.js';

const compiled = new WeakMap();                    // model -> { T, tau, AB } (JavaScript)

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

function tableauWGSL(tab, MMAX, LPP, f16 = false, ORTH = 1) {
  const f = (v) => { let t = v.toPrecision(9); if (!/[.eE]/.test(t)) t += '.0'; return t; };
  const arr = (v) => `array<f32, ${v.length}>(${v.map(f).join(', ')})`;
  const nodes = [0, ...tab.c, 1];
  // end value of the collocation polynomial: Lagrange weights at θ = 1 on the nodes {0, c}
  const n0 = [0, ...tab.c];
  const EW = n0.map((xi, i) => n0.reduce((l, xk, k) => (k === i ? l : l * (1 - xk) / (xi - xk)), 1));
  const t = (k) => (f16 && (f16 === true || f16.includes(k)) ? 'f16' : 'f32');
  return `alias SW = ${t('W')};\nalias SH = ${t('H')};\nalias SV = ${t('V')};\nconst S: u32 = ${tab.s}u;\nconst MMAX: u32 = ${MMAX}u;\nconst LPP: u32 = ${LPP}u;\n` +
    `const AT = ${arr(tab.a)};\nconst BT = ${arr(tab.b)};\nconst CT = ${arr(tab.c)};\nconst XN = ${arr(nodes)};\n` +
    `const EW = ${arr(EW)};\nconst ORTH: u32 = ${ORTH}u;\n`;
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
    const features = ['timestamp-query', 'shader-f16'].filter((x) => adapter.features.has(x));
    const device = await adapter.requestDevice({
      requiredFeatures: features,
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
    this.hasTS = device.features.has('timestamp-query');
    this.hasF16 = device.features.has('shader-f16');
    if (this.hasTS) {          // GPU time of the prep and main passes of a band
      this.qset = device.createQuerySet({ type: 'timestamp', count: 4 });
      this.qres = device.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.qread = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    this.pool = new Map();     // work buffers kept between calls (grow-only; release() frees them)
  }

  /** a pooled buffer of at least `bytes` (contents undefined) */
  buffer(name, bytes, usage) {
    const b = this.pool.get(name);
    if (b && b.size >= bytes) return b;
    b?.destroy();
    const nb = this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 256) * 256), usage });
    this.pool.set(name, nb);
    return nb;
  }

  release() { for (const b of this.pool.values()) b.destroy(); this.pool.clear(); }

  get deviceName() {
    const i = this.info;
    return [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(' ') || 'GPU';
  }

  async pipeline(model, S, m, lpp, f16 = false, orth = 1) {
    const code = (f16 ? 'enable f16;\n' : '') +
      this.src.replace('//@MODEL@', modelWGSL(model)).replace('//@TABLEAU@', tableauWGSL(gaussTableau(S), m, lpp, f16, orth));
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
   * ρ for every point of `pts`: a Float32Array of interleaved (x, y), or a regular grid
   * { nx, ny, box: {x0, x1, y0, y1} } (point k = i + nx·j, generated band by band, so even an
   * 8K image needs no full point list). Parameters `values` (in model.params order) with
   * params[xi] = x, params[yi] = y. Options: S (stages), p (steps per period), m (Krylov
   * dimension), lpp (lanes per point), onProgress(fraction), cancel() -> true stops between bands.
   * Points, results and work buffers live per band. Returns { rho, mu, flags, r, ms, done }.
   */
  async evaluate(model, values, xi, yi, pts, opt) {
    const { S = 3, p = 40, onProgress = null, cancel = null } = opt;
    const m = Math.min(Math.max(opt.m ?? 12, 2), MMAX);
    const grid = !(pts instanceof Float32Array);
    const n = grid ? pts.nx * pts.ny : pts.length / 2;
    const fill = (dst, off, cnt) => {                       // band points -> dst (interleaved)
      if (!grid) { dst.set(pts.subarray(2 * off, 2 * (off + cnt))); return; }
      const { nx, ny, box } = pts, dx = (box.x1 - box.x0) / Math.max(1, nx - 1), dy = (box.y1 - box.y0) / Math.max(1, ny - 1);
      for (let q = 0; q < cnt; q++) {
        const k = off + q, i = k % nx, j = (k - i) / nx;
        dst[2 * q] = box.x0 + i * dx; dst[2 * q + 1] = box.y0 + j * dy;
      }
    };
    const D = model.D, BS = (S + 1) * D;
    const r = this.delaySteps(model, values, xi, yi, n, (i) => { const t = new Float32Array(2); fill(t, i, 1); return t; }, p);
    const N = (r + 1) * BS;
    const wFloats = p * S * D * BS, lFloats = p * S * (S + 4);  // step matrices (stage rows), lookup records
    const bytesPerPt = 4 * ((p + r + 1) * BS + (m + 1) * N + wFloats + lFloats) + 8 + OUT_BYTES;
    // lanes per point: several for a small batch (latency of one point), one for a large grid
    const lpp = opt.lpp ?? this.lanesPerPoint(n, BS);
    const G = 64 / lpp;
    // fast mode: hist, V and W stored as f16 (half the memory traffic), arithmetic in f32
    const f16 = this.hasF16 && opt.f16 ? opt.f16 : false;   // true, or a subset of 'WHV'
    const eb = (k) => (f16 && (f16 === true || f16.includes(k)) ? 2 : 4);
    const pl = await this.pipeline(model, S, m, lpp, f16, opt.orth ?? 1);
    const dev = this.device;
    // per-buffer size limit, and the prep dispatch (one thread per point and step) ≤ 65535 groups
    const perPt = Math.max((p + r + 1) * BS, (m + 1) * N, wFloats, lFloats) * 4;
    const maxBand = Math.max(64, Math.min(Math.floor(65535 / p) * 64, 65535 * G, Math.floor(this.maxBuf / perPt / 64) * 64 - 64,
                                          Math.floor(1.5e9 / bytesPerPt)));
    // first band of a new pipeline: from the work per point (p·BS²·m), so that it stays well below
    // the GPU watchdog even for an expensive model; later bands follow the measured rate
    // (the measured rate is kept per pipeline as work units per ms: p and r are uniforms, so one
    // pipeline serves models of very different cost per point)
    const work = p * BS * BS * m + m * m * N;
    const band0 = Math.max(64, Math.min(4096, Math.floor(3e8 / work)));
    let band = Math.min(maxBand, n, pl.rate ? Math.max(64, Math.floor(pl.rate * this.bandTarget / work)) : band0);
    let hist = null, V = null, Wb = null, Lb = null, ptsBuf = null, outBuf = null, read = null, cap = 0;
    let gpuPrep = 0, gpuMain = 0;
    const S_ = GPUBufferUsage.STORAGE;
    const rho = new Float32Array(n), flags = new Uint8Array(n), mu = n <= 2e6 ? new Float32Array(2 * n) : null;
    const t0 = performance.now();
    let off = 0;
    while (off < n) {
      if (cancel && cancel()) break;
      const nb = Math.min(band, n - off);
      if (nb > cap) {
        cap = Math.min(maxBand, Math.max(nb, Math.min(n - off, 2 * nb)));
        const slots = Math.ceil(cap / 64) * 64;
        hist = this.buffer('hist', eb('H') * (p + r + 1) * BS * slots, S_);
        V = this.buffer('V', eb('V') * (m + 1) * N * slots, S_);
        Wb = this.buffer('W', eb('W') * wFloats * slots, S_);
        Lb = this.buffer('L', 4 * lFloats * slots, S_);
        ptsBuf = this.buffer('pts', 8 * slots, S_ | GPUBufferUsage.COPY_DST);
        outBuf = this.buffer('out', OUT_BYTES * slots, S_ | GPUBufferUsage.COPY_SRC);
        read = this.buffer('read', OUT_BYTES * slots, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      }
      const xy = new Float32Array(2 * nb);
      fill(xy, off, nb);
      dev.queue.writeBuffer(ptsBuf, 0, xy);
      const ub = new ArrayBuffer(U_BYTES), u32 = new Uint32Array(ub), f = new Float32Array(ub);
      u32.set([nb, p, r, m, xi, yi, 0, 0]);
      values.forEach((v, k) => { f[8 + k] = v; });
      dev.queue.writeBuffer(this.ubuf, 0, ub);
      const all = { 0: this.ubuf, 1: ptsBuf, 2: hist, 3: V, 4: outBuf, 5: Wb, 6: Lb };
      const group = (pipe, ids) => dev.createBindGroup({ layout: pipe.getBindGroupLayout(0),
        entries: ids.map((k) => ({ binding: k, resource: { buffer: all[k] } })) });
      const enc = dev.createCommandEncoder();
      const ts = (k) => (this.hasTS ? { timestampWrites: { querySet: this.qset, beginningOfPassWriteIndex: k, endOfPassWriteIndex: k + 1 } } : {});
      let pass = enc.beginComputePass(ts(0));
      pass.setPipeline(pl.prep); pass.setBindGroup(0, group(pl.prep, [0, 1, 5, 6]));
      pass.dispatchWorkgroups(Math.ceil(Math.ceil(nb / G) * G * p / 64));
      pass.end();
      pass = enc.beginComputePass(ts(2));
      pass.setPipeline(pl.main); pass.setBindGroup(0, group(pl.main, [0, 2, 3, 4, 5, 6]));
      pass.dispatchWorkgroups(Math.ceil(nb / G));
      pass.end();
      enc.copyBufferToBuffer(outBuf, 0, read, 0, OUT_BYTES * nb);
      if (this.hasTS) { enc.resolveQuerySet(this.qset, 0, 4, this.qres, 0); enc.copyBufferToBuffer(this.qres, 0, this.qread, 0, 32); }
      const tb = performance.now();
      dev.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ, 0, OUT_BYTES * nb);
      if (this.hasTS) {
        await this.qread.mapAsync(GPUMapMode.READ);
        const t = new BigUint64Array(this.qread.getMappedRange());
        if (t[1] > t[0]) gpuPrep += Number(t[1] - t[0]) / 1e6;
        if (t[3] > t[2]) gpuMain += Number(t[3] - t[2]) / 1e6;
        this.qread.unmap();
      }
      const out = new Float32Array(read.getMappedRange(0, OUT_BYTES * nb));
      for (let i = 0; i < nb; i++) {
        rho[off + i] = out[4 * i]; flags[off + i] = out[4 * i + 3];
        if (mu) { mu[2 * (off + i)] = out[4 * i + 1]; mu[2 * (off + i) + 1] = out[4 * i + 2]; }
      }
      read.unmap();
      const dt = performance.now() - tb;
      pl.rate = nb * work / Math.max(dt, 1);                     // work units per ms
      off += nb;
      band = Math.min(maxBand, Math.max(64, Math.floor(pl.rate * this.bandTarget / work)));
      onProgress?.(off / n);
    }
    const ms = performance.now() - t0;
    return { rho, mu, flags, r, ms, bytesPerPt, lpp, done: off >= n, gpuPrep, gpuMain };
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

  /** delay window r: the largest τ(t)/h over the points (Float64 on the host, 24 samples per
   *  period, at most ~400 of the points: enough for smooth τ, T); at(i) -> [x, y]. T and τ are
   *  compiled to JavaScript once per model (a tree walk cost tens of ms per call). */
  delaySteps(model, values, xi, yi, n, at, p) {
    let ev = compiled.get(model);
    if (!ev) { ev = new Function(modelJS(model))(); compiled.set(model, ev); }
    const P = Float64Array.from(values);
    const stride = Math.max(1, Math.floor(n / 400));
    let rmax = 1;
    const check = (i) => {
      const q = at(i);
      P[xi] = q[0]; P[yi] = q[1];
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
