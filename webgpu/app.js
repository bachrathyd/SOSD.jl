// UI of the SOSD WebGPU page: model text -> parameters (sliders, X / Y axes) -> brute-force grid
// and / or MDBM boundary on the GPU -> chart. ?validate=1 compares the examples with the Float64
// Julia reference (validate/ref.json), ?bench=1 runs the benchmark once.

import { Engine, WebGPUUnavailable } from './engine.js';
import { RemoteEngine } from './engine-remote.js';
import { parseModel, ParseError } from './expr.js';
import { EXAMPLES } from './examples.js';
import { mdbmBoundary } from './mdbm.js';
import { initStats, countEvent, countOnce, countModel } from './stats.js';
import { CpuPool } from './cpu.js';

const $ = (id) => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const STRIDES = [1, 2, 4, 8, 16, 32, 64];        // start resolution = final grid / stride
const MAX_PTS = 2.5e6;                            // cap of the final (pixel) grid
const MD_GRIDS = [[6, 4], [8, 5], [12, 7], [16, 8], [24, 8], [24, 12], [32, 16], [48, 12], [64, 16], [96, 24]];
const CR = 0.5;                                   // colour range of log10 ρ: [-CR, CR]
const STOPS = [[5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222], [247, 247, 247],
               [244, 165, 130], [214, 96, 77], [178, 24, 43], [103, 0, 31]];

let engine = null, model = null, modelText = '', ex = null;
const cpu = new CpuPool();
let values = [], axes = [0, 1];
let result = null;                                // { box, bf: {nx, ny, rho}, md: {...}, stats }
let busy = false, pending = false, generation = 0;

function log(s) { $('log').textContent += s + '\n'; }
function showMsg(s) { $('msg').style.display = s ? 'block' : 'none'; $('msg').innerHTML = s || ''; }

// ---------------------------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------------------------
async function main() {
  initStats();
  EXAMPLES.forEach((e) => $('ex').add(new Option(e.title, e.key)));
  STRIDES.forEach((st) => $('bfStart').add(new Option(`1/${st}`, st)));
  $('bfStart').value = 8;
  EXPORTS.forEach((e, i) => $('exRes').add(new Option(`${e.name} ${e.w} × ${e.h}`, i)));
  $('mdGrid').add(new Option('from the brute-force grid', -1));
  MD_GRIDS.forEach(([a, b], i) => $('mdGrid').add(new Option(`${a} × ${b}`, i)));
  try {
    // served by the Colab notebook (gpu/webui/server.jl): compute there; else WebGPU here
    const server = await RemoteEngine.detect();
    engine = server ? await RemoteEngine.create(log, server) : await Engine.create(log);
    if (server) {
      $('tLimit').value = 60; document.title = 'SOSD on ' + server.device;
      document.querySelector('header .sub').innerHTML = `Computed by SOSD.jl's batched solver on <b>${server.device}</b>: ` +
        'Float32 while dragging and for the coarse levels, Float64 with a converged Krylov–Schur iteration (and GMRES for the ' +
        'periodic orbit) for the final level, MDBM and the export. <span id="device"></span>';
    }
    $('device').textContent = 'GPU: ' + engine.deviceName;
    countOnce('gpu/' + (engine.info.vendor || 'unknown'));
  } catch (e) {
    showMsg((e instanceof WebGPUUnavailable ? '<b>No WebGPU.</b> ' : '<b>Error.</b> ') + e.message +
      ' Use a current Chrome / Edge (or Firefox / Safari with WebGPU enabled).');
    countOnce('no-webgpu');
    return;
  }
  $('ex').onchange = () => loadExample($('ex').value);
  $('reset').onclick = () => loadExample(ex.key);
  $('go').onclick = () => schedule(true, true);
  $('stop').onclick = stop;
  $('exGo').onclick = exportImage;
  $('bench').onclick = () => bench();
  $('mdIt').oninput = () => { $('mdItV').textContent = $('mdIt').value; schedule(); };
  if (!engine.hasF16) { $('f16').disabled = true; $('f16').parentElement.title = 'this GPU / browser has no shader-f16'; }
  for (const id of ['bfOn', 'mdOn', 'mdGrid', 'mdNb', 'S', 'p', 'm', 'f16', 'forcedOn']) $(id).onchange = () => { updateEtas(); schedule(true, true); };
  $('bfStart').onchange = () => { master = 'res'; updateEtas(); };
  $('fps').onchange = () => { master = 'fps'; updateEtas(); };
  $('mdPts').onchange = draw;
  let eqTimer = null;
  $('eq').oninput = () => { clearTimeout(eqTimer); eqTimer = setTimeout(() => setModelText($('eq').value, true), 400); };
  $('helpBtn').onclick = () => { $('helpBg').hidden = false; countOnce('help-open'); };
  $('helpClose').onclick = $('helpBg').onclick = (ev) => { if (ev.target === $('helpBg') || ev.target === $('helpClose')) $('helpBg').hidden = true; };
  sizeCanvas();
  bindPointer(); bindSplit();
  $('resetAxes').onclick = resetAxes;
  $('savePng').onclick = savePng;
  let rz = null;
  window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { sizeCanvas(); draw(); updateEtas(); schedule(); }, 150); });
  if (Q.has('validate')) { await validate(); return; }
  loadExample(EXAMPLES[0].key);
  if (Q.has('bench')) bench();
}

function loadExample(key) {
  ex = EXAMPLES.find((e) => e.key === key);
  $('ex').value = key;
  $('formula').innerHTML = ex.formula;
  $('note').textContent = ex.note || ''; $('note').style.display = ex.note ? '' : 'none';
  $('eq').value = ex.text;
  $('p').value = ex.p; $('m').value = ex.m; $('S').value = ex.S;
  $('bfOn').checked = ex.bf !== false; $('mdOn').checked = ex.md !== false;
  $('forcedOn').checked = !!ex.forced;
  $('mdGrid').value = ex.mdbm[0] === 'bf' ? -1 : MD_GRIDS.findIndex(([a, b]) => a === ex.mdbm[0] && b === ex.mdbm[1]);
  $('mdIt').value = ex.mdbm[2]; $('mdItV').textContent = ex.mdbm[2];
  setModelText(ex.text, false);
  countEvent('example/' + key);
}

function setModelText(text, edited) {
  try {
    const mdl = parseModel(text);
    const old = model;
    model = mdl; modelText = text;
    // keep values / axes of parameters with the same name
    values = mdl.params.map((q) => {
      const k = old ? old.params.findIndex((o) => o.name === q.name) : -1;
      return k >= 0 && !(!edited) ? values[k] : q.value;
    });
    const names = mdl.params.map((q) => q.name);
    if (!edited && ex.axes) axes = ex.axes.map((n) => names.indexOf(n));
    if (axes[0] < 0 || axes[0] >= names.length) axes[0] = 0;
    if (axes[1] < 0 || axes[1] >= names.length || axes[1] === axes[0]) axes[1] = axes[0] === 0 ? 1 : 0;
    $('eqStatus').className = 'status';
    $('eqStatus').textContent = `${mdl.D} states, ${mdl.params.length} parameters, ${mdl.helpers.length} helpers`;
    if (edited) countModel(text);
    rememberHome();
    $('forcedOn').disabled = !mdl.f;
    $('forcedOn').parentElement.title = mdl.f ? '' : 'the model has no forcing  f = [ ... ]';
    buildParams();
    updateEtas();
    schedule(true, true);
  } catch (e) {
    $('eqStatus').className = 'status err';
    $('eqStatus').textContent = (e instanceof ParseError && e.line >= 0 ? `line ${e.line + 1}: ` : '') + e.message;
  }
}

function buildParams() {
  const tb = $('par');
  tb.innerHTML = '';
  model.params.forEach((q, i) => {
    const fmt = (v) => (+v).toPrecision(4);
    const tr = document.createElement('tr'), tr2 = document.createElement('tr');
    tr.innerHTML = `<td class="nm">${q.name}</td><td class="v"></td>` +
      `<td><input type="number" step="any" class="lo" title="min"></td><td><input type="number" step="any" class="hi" title="max"></td>` +
      `<td class="ax"><button class="bx">X</button><button class="by">Y</button></td>`;
    tr2.innerHTML = `<td colspan="5" style="padding-top:0"><input type="range"></td>`;
    const r = tr2.querySelector('input[type=range]'), v = tr.querySelector('.v');
    const lo = tr.querySelector('.lo'), hi = tr.querySelector('.hi');
    const setRange = () => { r.min = q.lo; r.max = q.hi; r.step = (q.hi - q.lo) / 400 || 1e-6; r.value = values[i]; lo.value = q.lo; hi.value = q.hi; };
    setRange();
    v.textContent = fmt(values[i]);
    r.oninput = () => { values[i] = +r.value; v.textContent = fmt(values[i]); noteInput(); schedule(); };
    lo.onchange = () => { q.lo = +lo.value; setRange(); schedule(true, true); };
    hi.onchange = () => { q.hi = +hi.value; setRange(); schedule(true, true); };
    const bx = tr.querySelector('.bx'), by = tr.querySelector('.by');
    const isAx = axes.includes(i);
    r.disabled = isAx; tr2.style.display = isAx ? 'none' : '';
    if (isAx) v.textContent = axes[0] === i ? 'X axis' : 'Y axis';
    bx.classList.toggle('on', axes[0] === i); by.classList.toggle('on', axes[1] === i);
    bx.onclick = () => { if (axes[1] === i) axes[1] = axes[0]; axes[0] = i; buildParams(); updateEtas(); schedule(true, true); };
    by.onclick = () => { if (axes[0] === i) axes[0] = axes[1]; axes[1] = i; buildParams(); updateEtas(); schedule(true, true); };
    tb.appendChild(tr); tb.appendChild(tr2);
  });
}

// ---------------------------------------------------------------------------------------------
// computing. A running computation is finished, not restarted, when a slider moves: the newest
// values are computed next (so the chart keeps updating while dragging); a structural change
// (model, axes, method) or Stop cancels it, and so does the time limit. The old chart stays on
// screen until the new one is complete (brute force: the new map with the old MDBM boundary
// until the new boundary arrives).
// ---------------------------------------------------------------------------------------------
function schedule(force = false, restart = false) {
  if (!force && !$('auto').checked) return;
  if (busy) { pending = true; if (restart) generation++; return; }
  compute();
}

function opts() {
  return { S: +$('S').value, p: Math.max(4, +$('p').value | 0), m: Math.min(24, Math.max(3, +$('m').value | 0)),
           f16: $('f16').checked ? 'V' : false };
}

function timeLimit() { const v = parseFloat($('tLimit').value); return v > 0 ? 1000 * v : Infinity; }

// Brute force is computed progressively on nested lattices of the final grid: stride s0 (a power
// of 2), then s0/2, …, 1; every level evaluates only its new points in one GPU call and is drawn
// at once, so the picture sharpens step by step for barely more work than one full call.
// While a slider is being dragged only the first level runs, its stride chosen from the measured
// throughput and the frames-per-second target (start resolution); MDBM waits until the drag stops.
const CALL_MS = 12;                               // latency of one GPU call (measured, AMD iGPU)
let master = 'fps';                               // which of start resolution / fps the user set last
let lastInput = -Infinity, idleTimer = null;
const interacting = () => performance.now() - lastInput < 200;
function noteInput() {
  lastInput = performance.now();
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => schedule(), 220);    // the full chart once the drag stops
}

/** the final brute-force grid: one point per device pixel of the plot area (capped) */
function finalGrid() {
  const { W, H } = geom(), dpr = Math.min(2, window.devicePixelRatio || 1);
  let nx = Math.max(16, Math.round(W * dpr)), ny = Math.max(8, Math.round(H * dpr));
  const f = Math.sqrt(Math.min(1, MAX_PTS / (nx * ny)));
  return [Math.round(nx * f), Math.round(ny * f)];
}
const levelCount = (nx, ny, s) => Math.ceil(nx / s) * Math.ceil(ny / s);
/** expected frame time of a level with stride s [ms] */
const frameMs = (nx, ny, s, us) => CALL_MS + levelCount(nx, ny, s) * us / 1000;
/** stride of the first level: the finest power of 2 whose frame fits the fps target */
function strideForFps(nx, ny, us, fps) {
  let s = 1;
  while (s < STRIDES[STRIDES.length - 1] && frameMs(nx, ny, s, us) > 1000 / fps) s *= 2;
  return s;
}

/** the computed sub-lattice of stride s as a grid result (its own box: the nodes i·s ≤ nx−1) */
function levelGrid(nx, ny, rho, s, box, amp = null) {
  const cw = Math.floor((nx - 1) / s) + 1, ch = Math.floor((ny - 1) / s) + 1;
  const sub = new Float32Array(cw * ch), sa = amp ? new Float32Array(cw * ch) : null;
  for (let j = 0; j < ch; j++) for (let i = 0; i < cw; i++) {
    sub[j * cw + i] = rho[j * s * nx + i * s];
    if (sa) sa[j * cw + i] = amp[j * s * nx + i * s];
  }
  const dx = (box.x1 - box.x0) / (nx - 1), dy = (box.y1 - box.y0) / (ny - 1);
  return { nx: cw, ny: ch, rho: sub, amp: sa, stride: s,
           box: { x0: box.x0, x1: box.x0 + (cw - 1) * s * dx, y0: box.y0, y1: box.y0 + (ch - 1) * s * dy } };
}

let dragCache = null;                             // { key, s, rho } of the last drag frame

async function compute() {
  if (!engine || !model) return;
  busy = true; pending = false;
  setBusyUI(true);
  const gen = ++generation;
  const t0 = performance.now(), limit = timeLimit();
  let timedOut = false;
  const cancel = () => {
    if (gen !== generation) return true;
    if (performance.now() - t0 > limit) { timedOut = true; return true; }
    return false;
  };
  const [xi, yi] = axes, px = model.params[xi], py = model.params[yi];
  const box = { x0: px.lo, x1: px.hi, y0: py.lo, y1: py.hi };
  const o = opts(), vals = values.slice(), mdl = model;
  const drag = interacting();
  const bfOn = $('bfOn').checked, mdOn = $('mdOn').checked;
  const res = { box, xname: px.name, yname: py.name, bf: null, md: null, n: 0, ms: 0, r: 0, N: 0, nCpu: 0, msCpu: 0 };
  const progress = (f) => { $('progress').firstChild.style.width = (100 * f).toFixed(1) + '%'; };
  // newer values waiting: refinement of these ones is stale (MDBM too, when a map is shown)
  const stale = () => pending && bfOn;
  const forced = $('forcedOn').checked && !!mdl.f;
  let lastAmp = null;
  // final: the accurate pass (server: Float64, converged Krylov–Schur); coarse levels are not
  const evalBatch = async (pts, extraCancel = () => false, withOrbit = false, final = true) => {
    const out = await engine.evaluate(mdl, vals, xi, yi, pts, { ...o, onProgress: progress, forced: withOrbit, final,
      cancel: () => cancel() || extraCancel() });
    if (!out.done) throw new Cancelled();
    res.n += out.rho.length; res.ms += out.ms; res.r = out.r; res.N = (out.r + 1) * (o.S + 1) * mdl.D;
    lastAmp = out.amp;
    return out.rho;
  };
  const evalCpu = async (xy) => {
    if (cancel() || stale()) throw new Cancelled();
    const tc = performance.now();
    const r = engine.delaySteps(mdl, vals, xi, yi, xy.length / 2, (i) => [xy[2 * i], xy[2 * i + 1]], o.p);
    const rho = await cpu.evaluate(mdl, vals, xi, yi, xy, { ...o, r });
    res.nCpu += rho.length; res.msCpu += performance.now() - tc;
    return rho;
  };
  try {
    if (bfOn) {
      countOnce('brute-force');
      const [nx, ny] = finalGrid();
      const key = JSON.stringify([modelText, vals, box, o, nx, ny, axes, forced]);
      const s0 = +$('bfStart').value;
      const rho = new Float32Array(nx * ny), amp = forced ? new Float32Array(nx * ny) : null;
      let s = s0, from = s0;
      if (!drag && dragCache && dragCache.key === key && dragCache.s >= s0) {
        rho.set(dragCache.rho); from = s = dragCache.s;    // the last drag frame is level one
        if (amp) amp.set(dragCache.amp);
        s /= 2;
      }
      for (; s >= 1; s /= 2) {
        // new nodes of this level: on the stride-s lattice, not on the stride-2s one (except the first)
        const idx = [];
        for (let j = 0; j < ny; j += s) for (let i = 0; i < nx; i += s) {
          // (on a server the final level recomputes every node: all of it in Float64)
          if (s < from && i % (2 * s) === 0 && j % (2 * s) === 0 && !(engine.remote && s === 1)) continue;
          idx.push(j * nx + i);
        }
        const xy = new Float32Array(2 * idx.length), dx = (box.x1 - box.x0) / (nx - 1), dy = (box.y1 - box.y0) / (ny - 1);
        idx.forEach((k, q) => { xy[2 * q] = box.x0 + (k % nx) * dx; xy[2 * q + 1] = box.y0 + Math.floor(k / nx) * dy; });
        const ms0 = res.ms;
        const r = await evalBatch(xy, s < from ? stale : () => false, forced, s === 1);
        idx.forEach((k, q) => { rho[k] = r[q]; });
        if (amp) idx.forEach((k, q) => { amp[k] = lastAmp[q]; });
        // throughput without the latency of the call (small levels would overestimate it)
        if (idx.length >= 16384) { lastRate = { key: rateKey(), us: 1000 * Math.max(0.2 * (res.ms - ms0), res.ms - ms0 - CALL_MS) / idx.length }; updateEtas(); }
        res.bf = s === 1 ? { nx, ny, rho, amp, box } : levelGrid(nx, ny, rho, s, box, amp);
        // the map at once; the previous boundary stays on it until the new one is ready
        result = { ...res, md: mdOn && !drag && result ? result.md : null };
        draw(); stats(res);
        if (drag) { dragCache = { key, s, rho: rho.slice(), amp: amp && amp.slice() }; break; }
      }
      if (!res.bf) {                                    // everything came from the drag frame
        res.bf = from === 1 ? { nx, ny, rho, amp, box } : levelGrid(nx, ny, rho, from, box, amp);
        result = { ...res, md: mdOn && result ? result.md : null };
        draw(); stats(res);
      }
    }
    if (mdOn && !(drag && bfOn)) {
      countOnce('mdbm');
      const nb = $('mdNb').value, it = +$('mdIt').value;
      const mdCancel = () => { if (cancel() || stale()) throw new Cancelled(); return false; };
      const onStage = (st, nc, part) => { if (!cancel()) { result = { ...res, md: part }; draw(); stats({ ...res, md: part }); } };
      const common = { neighbour: nb, evalNeighbour: nb === 'end' ? evalCpu : evalBatch, cancel: mdCancel, onStage };
      const g = +$('mdGrid').value;
      if (g < 0 && res.bf) {
        // start from the brute-force grid: its sign-changing cells are the initial cells
        res.md = await mdbmBoundary(evalBatch, res.bf.box, res.bf.nx, res.bf.ny, it, { ...common, init: res.bf.rho });
      } else {
        const [a, b] = MD_GRIDS[g >= 0 ? g : MD_GRIDS.findIndex(([u, v]) => u === 16 && v === 8)];   // 16 × 8 without a map
        res.md = await mdbmBoundary(evalBatch, box, a, b, it, common);
      }
    }
    if (!(drag && bfOn)) { result = res; draw(); stats(res); }   // a drag frame is already drawn
    showMsg('');
  } catch (e) {
    if (e instanceof Cancelled) {
      if (timedOut) showMsg(`Stopped after the time limit (${(limit / 1000).toFixed(1)} s): the chart shows the last complete level. ` +
                            'Raise the limit, or use a coarser grid / smaller p, m.');
    } else {
      log('error: ' + e.message);
      showMsg('<b>Error:</b> ' + e.message);
    }
  }
  busy = false;
  setBusyUI(false);
  progress(0);
  if (pending) compute();
}

class Cancelled extends Error {}

function setBusyUI(on) {
  $('stop').disabled = !on;
}

function stop() { generation++; pending = false; }

function stats(res) {
  const nMd = res.md ? res.md.points.length / 2 | 0 : 0;
  $('sN').innerHTML = (res.n + res.nCpu).toLocaleString() + (res.md ? ` <small>(MDBM ${nMd}${res.nCpu ? `, ${res.nCpu} on CPU` : ''})</small>` : '');
  const tot = res.ms + res.msCpu;
  $('sT').innerHTML = tot < 1000 ? `${tot.toFixed(0)} <small>ms</small>` : `${(tot / 1000).toFixed(2)} <small>s</small>`;
  $('sU').innerHTML = res.n ? `${(1000 * res.ms / res.n).toFixed(1)} <small>µs</small>` : '–';
  if (res.bf) {
    let u = 0; for (const v of res.bf.rho) if (v >= 1) u++;
    $('sUn').innerHTML = `${(100 * u / res.bf.rho.length).toFixed(1)} <small>%</small>`;
  } else $('sUn').textContent = '–';
  $('sR').innerHTML = `${res.r} <small>· N = ${res.N}</small>`;
}

// ---------------------------------------------------------------------------------------------
// expected times (from the last brute-force throughput of the same model and discretization)
// ---------------------------------------------------------------------------------------------
let lastRate = null;
function rateKey() { const o = opts(); return `${modelText.length}:${modelText.slice(0, 200)}|${o.S}|${o.p}|${o.m}|${o.f16}|${axes}`; }
function fmtTime(ms) {
  if (ms < 1000) return `${Math.max(10, Math.round(ms / 10) * 10)} ms`;
  if (ms < 120e3) return `${(ms / 1000).toPrecision(2)} s`;
  return `${(ms / 60e3).toPrecision(2)} min`;
}
function updateEtas() {
  const us = lastRate && lastRate.key === rateKey() ? lastRate.us : 20;
  const known = lastRate && lastRate.key === rateKey();
  const [nx, ny] = finalGrid();
  if (master === 'fps') $('bfStart').value = strideForFps(nx, ny, us, Math.max(1, +$('fps').value || 15));
  const s0 = +$('bfStart').value;
  [...$('bfStart').options].forEach((op) => {
    const st = +op.value;
    op.text = `${Math.ceil(nx / st)} × ${Math.ceil(ny / st)}` + (known ? `  (≈ ${Math.min(99, 1000 / frameMs(nx, ny, st, us)).toFixed(0)} fps)` : '');
  });
  if (master === 'res') $('fps').value = Math.max(1, Math.round(1000 / frameMs(nx, ny, s0, us)));
  $('bfInfo').textContent = `final: ${nx} × ${ny} = ${(nx * ny).toLocaleString()} ρ (one per pixel)` +
    (known ? `, ≈ ${fmtTime(us * nx * ny / 1000)}` : '');
  [...$('exRes').options].forEach((op, i) => {
    const e = EXPORTS[i], [ex_, ey] = exportGrid(e.w, e.h);
    op.text = `${e.name} ${e.w} × ${e.h}` + (known ? `  (≈ ${fmtTime(us * ex_ * ey / 1000)})` : '');
  });
}

// ---------------------------------------------------------------------------------------------
// drawing: the view box is the axis parameters' current range; a result computed for another box
// (during zoom / pan, until the recompute arrives) is drawn mapped into the view
// ---------------------------------------------------------------------------------------------
const PAD = { l: 72, r: 96, t: 34, b: 56 };
// stable region with the periodic orbit: peak-to-peak amplitude, white -> dark green (ColorBrewer Greens)
const GREENS = [[247, 252, 245], [229, 245, 224], [199, 233, 192], [161, 217, 155], [116, 196, 118],
                [65, 171, 93], [35, 139, 69], [0, 109, 44], [0, 68, 27]];
let greenCache = null;
function greenLut() {
  if (greenCache) return greenCache;
  greenCache = new Uint32Array(LUT_N);
  for (let k = 0; k < LUT_N; k++) {
    const t = k / (LUT_N - 1) * (GREENS.length - 1), i = Math.min(GREENS.length - 2, Math.floor(t)), w = t - i;
    const c = GREENS[i].map((v, q) => Math.round(v + w * (GREENS[i + 1][q] - v)));
    greenCache[k] = (255 << 24) | (c[2] << 16) | (c[1] << 8) | c[0];
  }
  return greenCache;
}
/** logarithmic colour scale of the amplitude over the stable points: [2nd, 98th percentile]
 *  (near the boundary the amplitude grows without bound; a linear scale would wash out the rest) */
function ampScale(bf) {
  if (bf.ampLo !== undefined) return [bf.ampLo, bf.ampHi];
  const v = [];
  for (let k = 0; k < bf.rho.length; k++) if (bf.rho[k] < 1 && bf.amp[k] > 0 && bf.amp[k] < 1e30) v.push(bf.amp[k]);
  v.sort((a, b) => a - b);
  let lo = 1, hi = 10;
  if (v.length) {
    hi = v[Math.min(v.length - 1, Math.floor(0.98 * v.length))];
    lo = Math.max(v[Math.floor(0.02 * v.length)], hi * 1e-4);
    if (!(hi > lo * 1.5)) { lo = hi / 3; }
  }
  bf.ampLo = lo; bf.ampHi = hi;
  return [lo, hi];
}
const ampT = (a, lo, hi) => (Math.log(Math.max(a, 1e-30)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
const fmtA = (x) => +x.toPrecision(2);
let zoomRect = null;                              // [x0, y0, x1, y1] css px while dragging a zoom box
function cmap(v) {
  const t = Math.min(1, Math.max(0, (v / CR + 1) / 2)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t)), w = t - i;
  return STOPS[i].map((c, k) => Math.round(c + w * (STOPS[i + 1][k] - c)));
}
const LUT_N = 1024;
let lutCache = null, barCache = null;
/** colour map as packed RGBA (little-endian Uint32) over [-CR, CR] */
function colourLut() {
  if (lutCache) return lutCache;
  lutCache = new Uint32Array(LUT_N);
  for (let k = 0; k < LUT_N; k++) {
    const c = cmap(CR * (2 * k / (LUT_N - 1) - 1));
    lutCache[k] = (255 << 24) | (c[2] << 16) | (c[1] << 8) | c[0];
  }
  return lutCache;
}
/** the colour bar as a 1×256 image (top = +CR) */
function colourBar() {
  if (barCache) return barCache;
  const img = new ImageData(1, 256), px = new Uint32Array(img.data.buffer), lut = colourLut();
  for (let k = 0; k < 256; k++) px[k] = lut[Math.round((1 - k / 255) * (LUT_N - 1))];
  barCache = new OffscreenCanvas(1, 256); barCache.getContext('2d').putImageData(img, 0, 0);
  return barCache;
}
function nice(a, b, n) {
  const span = (b - a) / n, mag = Math.pow(10, Math.floor(Math.log10(span)));
  const s = [1, 2, 2.5, 5, 10].map((x) => x * mag).find((x) => x >= span);
  const out = []; for (let v = Math.ceil(a / s) * s; v <= b + 1e-9 * s; v += s) out.push(+v.toPrecision(10));
  return out;
}
function viewBox() {
  if (!model) return result ? result.box : { x0: 0, x1: 1, y0: 0, y1: 1 };
  const px = model.params[axes[0]], py = model.params[axes[1]];
  return { x0: px.lo, x1: px.hi, y0: py.lo, y1: py.hi };
}
/** css size of the chart and of its plot area */
function geom() {
  const cv = $('chart');
  const w = cv.clientWidth || 800, h = Math.round(cv.clientHeight || 480);
  return { w, h, W: w - PAD.l - PAD.r, H: h - PAD.t - PAD.b };
}
function sizeCanvas() {
  const cv = $('chart'), dpr = window.devicePixelRatio || 1;
  const w = cv.parentElement.clientWidth - 16;
  const h = Math.max(280, Math.min(Math.round(w * 0.6), window.innerHeight - 150));
  cv.style.height = h + 'px';
  const W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
}

function draw() {
  const cv = $('chart'), cx = cv.getContext('2d'), dpr = cv.width / (cv.clientWidth || cv.width);
  const { w, h } = geom();
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render(cx, w, h, 1, result, viewBox(), $('mdPts').checked);
  if (zoomRect) {
    const [a, b, c, d] = zoomRect;
    cx.fillStyle = 'rgba(47, 111, 219, 0.12)'; cx.strokeStyle = '#2f6fdb'; cx.lineWidth = 1.2;
    cx.fillRect(Math.min(a, c), Math.min(b, d), Math.abs(c - a), Math.abs(d - b));
    cx.strokeRect(Math.min(a, c), Math.min(b, d), Math.abs(c - a), Math.abs(d - b));
  }
}

/** the chart of `res` in the data box `view` on a w×h context; sc scales fonts, pads and lines */
function render(cx, w, h, sc, res, view, showPts = false, ink = null, muted = null) {
  const css = getComputedStyle(document.documentElement);
  ink = ink || css.getPropertyValue('--ink').trim() || '#222';
  muted = muted || css.getPropertyValue('--muted').trim() || '#666';
  const L = PAD.l * sc, R = PAD.r * sc, T = PAD.t * sc, B = PAD.b * sc;
  const W = w - L - R, H = h - T - B;
  cx.clearRect(0, 0, w, h);
  cx.fillStyle = '#f7f7f7'; cx.fillRect(L, T, W, H);
  if (!res) return;
  const X = (x) => L + (x - view.x0) / (view.x1 - view.x0) * W;
  const Y = (y) => T + H - (y - view.y0) / (view.y1 - view.y0) * H;
  cx.save(); cx.beginPath(); cx.rect(L, T, W, H); cx.clip();
  if (res.bf) {
    const { nx, ny, rho } = res.bf, box = res.bf.box || res.box;
    if (!res.bf.img) {
      const img = new ImageData(nx, ny), px32 = new Uint32Array(img.data.buffer), lut = colourLut();
      const amp = res.bf.amp, gl = greenLut(), [alo, ahi] = amp ? ampScale(res.bf) : [1, 10];
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (amp && rho[k] < 1) {
          const t = Math.min(LUT_N - 1, Math.max(0, Math.round(ampT(amp[k], alo, ahi) * (LUT_N - 1))));
          px32[i + nx * (ny - 1 - j)] = gl[isFinite(t) ? t : LUT_N - 1];
          continue;
        }
        const v = Math.log10(Math.max(rho[k], 1e-30));
        const t = Math.min(LUT_N - 1, Math.max(0, Math.round((v / CR + 1) / 2 * (LUT_N - 1))));
        px32[i + nx * (ny - 1 - j)] = lut[t];
      }
      res.bf.img = new OffscreenCanvas(nx, ny); res.bf.img.getContext('2d').putImageData(img, 0, 0);
    }
    cx.imageSmoothingEnabled = true;
    // grid nodes at the box edges: the image spans half a cell beyond them
    const hx = (box.x1 - box.x0) / (nx - 1) / 2, hy = (box.y1 - box.y0) / (ny - 1) / 2;
    const ax = X(box.x0 - hx), ay = Y(box.y1 + hy);
    cx.drawImage(res.bf.img, ax, ay, X(box.x1 + hx) - ax, Y(box.y0 - hy) - ay);
    // boundary from the grid (marching squares on log ρ), only without an MDBM boundary
    cx.strokeStyle = '#000'; cx.lineWidth = 1.2 * sc; cx.beginPath();
    if (!res.md) {
    const f = (i, j) => Math.log(Math.max(rho[j * nx + i], 1e-30));
    const px = (i) => X(box.x0 + i * 2 * hx), py = (j) => Y(box.y0 + j * 2 * hy);
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      const v = [f(i, j), f(i + 1, j), f(i + 1, j + 1), f(i, j + 1)];
      if ((v[0] >= 0) === (v[1] >= 0) && (v[1] >= 0) === (v[2] >= 0) && (v[2] >= 0) === (v[3] >= 0)) continue;
      const P = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], pts = [];
      for (let e = 0; e < 4; e++) { const a = v[e], b = v[(e + 1) % 4];
        if ((a >= 0) !== (b >= 0)) { const t = a / (a - b), A = P[e], Bp = P[(e + 1) % 4];
          pts.push([px(A[0] + t * (Bp[0] - A[0])), py(A[1] + t * (Bp[1] - A[1]))]); } }
      if (pts.length >= 2) { cx.moveTo(...pts[0]); cx.lineTo(...pts[1]); }
      if (pts.length === 4) { cx.moveTo(...pts[2]); cx.lineTo(...pts[3]); }
    }
    }
    cx.stroke();
  }
  if (res.md) {
    if (showPts) {
      const { points, rho } = res.md, s = 1.5 * sc;
      for (let k = 0; k < rho.length; k++) {
        cx.fillStyle = rho[k] >= 1 ? '#b2182b' : '#2166ac';
        cx.fillRect(X(points[2 * k]) - s, Y(points[2 * k + 1]) - s, 2 * s, 2 * s);
      }
    }
    cx.strokeStyle = '#333'; cx.lineWidth = 2 * sc;
    cx.lineCap = 'round'; cx.beginPath();
    for (const s of res.md.segments) { cx.moveTo(X(s[0]), Y(s[1])); cx.lineTo(X(s[2]), Y(s[3])); }
    cx.stroke();
  }
  cx.restore();
  // axes, ticks, labels
  cx.strokeStyle = muted; cx.lineWidth = sc; cx.strokeRect(L, T, W, H);
  cx.fillStyle = ink; cx.font = `${13 * sc}px system-ui, sans-serif`; cx.textAlign = 'center';
  for (const v of nice(view.x0, view.x1, Math.max(3, Math.round(W / 100 / sc)))) {
    const x = X(v); cx.fillRect(x, T + H, sc, 5 * sc); cx.fillText(+v.toPrecision(6), x, T + H + 19 * sc);
  }
  cx.fillText(axisLabel(res.xname), L + W / 2, T + H + 44 * sc);
  cx.textAlign = 'right';
  for (const v of nice(view.y0, view.y1, Math.max(3, Math.round(H / 70 / sc)))) {
    const y = Y(v); cx.fillRect(L - 5 * sc, y, 5 * sc, sc); cx.fillText(+v.toPrecision(6), L - 8 * sc, y + 4 * sc);
  }
  cx.save(); cx.translate(20 * sc, T + H / 2); cx.rotate(-Math.PI / 2); cx.textAlign = 'center';
  cx.fillText(axisLabel(res.yname), 0, 0); cx.restore();
  // title: what the colours show
  const orbit = res.bf && res.bf.amp;
  cx.textAlign = 'left'; cx.font = `600 ${13 * sc}px system-ui, sans-serif`;
  cx.fillText(orbit
    ? `stable (green): peak-to-peak of ${stateName(0)} on the periodic orbit  ·  unstable (red): log₁₀ ρ`
    : 'colour: log₁₀ ρ, spectral radius of the monodromy operator (ρ < 1 stable, blue)', L, T - 12 * sc);
  cx.font = `${13 * sc}px system-ui, sans-serif`;
  // colour bar
  const bx = L + W + 18 * sc, bw = 16 * sc;
  cx.imageSmoothingEnabled = true;
  if (orbit) {
    // upper half: log10 ρ in [0, CR] (unstable), lower half: amplitude 0 .. a_max (stable)
    const [alo, ahi] = ampScale(res.bf);
    cx.drawImage(colourBar(), 0, 0, 1, 128, bx, T, bw, H / 2);
    const g = new ImageData(1, 256), gp32 = new Uint32Array(g.data.buffer), gl = greenLut();
    for (let k = 0; k < 256; k++) gp32[k] = gl[Math.round((1 - k / 255) * (LUT_N - 1))];
    const gc = new OffscreenCanvas(1, 256); gc.getContext('2d').putImageData(g, 0, 0);
    cx.drawImage(gc, bx, T + H / 2, bw, H / 2);
    cx.strokeRect(bx, T, bw, H); cx.fillStyle = ink; cx.textAlign = 'left';
    for (const v of [CR, CR / 2]) cx.fillText(v.toFixed(2), bx + bw + 4 * sc, T + (1 - v / CR) / 2 * H / 2 + 4 * sc);
    cx.fillText('ρ = 1', bx + bw + 4 * sc, T + H / 2 + 4 * sc);
    // log scale, lower half: alo (bottom) .. ahi (middle)
    cx.fillText(fmtA(alo) + '−', bx + bw + 4 * sc, T + H + 4 * sc);
    cx.fillText(fmtA(Math.sqrt(alo * ahi)), bx + bw + 4 * sc, T + H * 0.75 + 4 * sc);
    cx.fillText(fmtA(ahi) + '+', bx + bw + 4 * sc, T + H / 2 + 18 * sc);
    cx.save(); cx.translate(bx + bw + 52 * sc, T + H / 2); cx.rotate(-Math.PI / 2); cx.textAlign = 'center';
    cx.fillText(`peak-to-peak ${stateName(0)} (log)   |   log₁₀ ρ`, 0, 0); cx.restore();
  } else {
    cx.drawImage(colourBar(), bx, T, bw, H);
    cx.strokeRect(bx, T, bw, H); cx.fillStyle = ink; cx.textAlign = 'left';
    for (const v of [-CR, -CR / 2, 0, CR / 2, CR]) cx.fillText(v.toFixed(2), bx + bw + 4 * sc, T + (1 - v / CR) / 2 * H + 4 * sc);
    cx.save(); cx.translate(bx + bw + 46 * sc, T + H / 2); cx.rotate(-Math.PI / 2); cx.textAlign = 'center';
    cx.fillText('log₁₀ ρ   (ρ < 1 stable, blue)', 0, 0); cx.restore();
  }
}

/** a name for state component k: from the example's declaration, else x₁, x₂, ... */
function stateName(k) {
  const nm = ex && ex.states ? ex.states[k] : null;
  return nm || 'x' + '₁₂₃₄₅₆'[k];
}

// ---------------------------------------------------------------------------------------------
// high-resolution image: one ρ per pixel of the plot area, PNG with axes (not time limited)
// ---------------------------------------------------------------------------------------------
const EXPORTS = [{ name: 'HD', w: 1280, h: 720 }, { name: 'Full HD', w: 1920, h: 1080 },
                 { name: '4K', w: 3840, h: 2160 }, { name: '8K', w: 7680, h: 4320 }];
const exportScale = (w, h) => Math.min(w / 1100, h / 640);
function exportGrid(w, h) {
  const sc = exportScale(w, h);
  return [Math.round(w - (PAD.l + PAD.r) * sc), Math.round(h - (PAD.t + PAD.b) * sc)];
}

let exporting = null;                             // { cancelled } while an export runs
async function exportImage() {
  if (exporting) { exporting.cancelled = true; return; }
  if (!engine || !model) return;
  const e = EXPORTS[+$('exRes').value];
  const job = exporting = { cancelled: false };
  countOnce('export/' + e.name);
  stop();
  while (busy) await new Promise((r) => setTimeout(r, 50));
  busy = true;
  const btn = $('exGo'), label = btn.textContent;
  btn.textContent = 'Cancel';
  const [nx, ny] = exportGrid(e.w, e.h), sc = exportScale(e.w, e.h);
  const [xi, yi] = axes, px = model.params[xi], py = model.params[yi];
  const box = { x0: px.lo, x1: px.hi, y0: py.lo, y1: py.hi };
  try {
    const out = await engine.evaluate(model, values.slice(), xi, yi, { nx, ny, box }, { ...opts(), final: true, forced: $('forcedOn').checked && !!model.f,
      cancel: () => job.cancelled,
      onProgress: (f) => { $('progress').firstChild.style.width = (100 * f).toFixed(1) + '%'; btn.textContent = `Cancel (${(100 * f).toFixed(0)} %)`; } });
    if (out.done) {
      const same = result && result.box && ['x0', 'x1', 'y0', 'y1'].every((k) => result.box[k] === box[k]);
      const res = { box, xname: px.name, yname: py.name, bf: { nx, ny, rho: out.rho, amp: out.amp }, md: same ? result.md : null };
      const cv = new OffscreenCanvas(e.w, e.h), cx = cv.getContext('2d');
      render(cx, e.w, e.h, sc, res, box, false, '#1d2230', '#5d6475');
      cx.globalCompositeOperation = 'destination-over';
      cx.fillStyle = '#ffffff'; cx.fillRect(0, 0, e.w, e.h);
      const blob = await cv.convertToBlob({ type: 'image/png' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `sosd_${ex ? ex.key : 'chart'}_${e.w}x${e.h}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      log(`${e.name} image: ${(nx * ny).toLocaleString()} ρ in ${(out.ms / 1000).toFixed(1)} s`);
      engine.release();                         // the export buffers can be large
    }
  } catch (err) {
    showMsg('<b>Export failed:</b> ' + err.message);
  }
  btn.textContent = label;
  $('progress').firstChild.style.width = '0%';
  exporting = null;
  busy = false;
  if (pending) compute();
}

function axisLabel(name) {
  // the comment of the declaration line, if any, as the axis label: "n = ... # spindle speed [rpm]"
  const line = modelText.split('\n').find((l) => l.replace(/\s/g, '').startsWith(name + '=') && l.includes('#'));
  const c = line ? line.split('#')[1].trim() : '';
  return c ? `${name}: ${c}` : name;
}

/** pointer event -> css px on the canvas, position in the plot area, data coordinates */
function locate(ev) {
  const rc = $('chart').getBoundingClientRect(), { W, H } = geom(), v = viewBox();
  const cx = ev.clientX - rc.left, cy = ev.clientY - rc.top, px = cx - PAD.l, py = cy - PAD.t;
  return { cx, cy, px, py, W, H, inside: px >= 0 && py >= 0 && px <= W && py <= H,
           x: v.x0 + px / W * (v.x1 - v.x0), y: v.y1 - py / H * (v.y1 - v.y0) };
}

function hover(ev) {
  if (!result) return;
  const q = locate(ev);
  if (!q.inside) { $('hover').textContent = ''; return; }
  let s = `${result.xname} = ${q.x.toPrecision(5)}, ${result.yname} = ${q.y.toPrecision(5)}`;
  if (result.bf) {
    const { nx, ny, rho } = result.bf, box = result.bf.box || result.box;
    const i = Math.round((q.x - box.x0) / (box.x1 - box.x0) * (nx - 1)), j = Math.round((q.y - box.y0) / (box.y1 - box.y0) * (ny - 1));
    if (i >= 0 && j >= 0 && i < nx && j < ny) {
      s += `   ρ ≈ ${rho[j * nx + i].toPrecision(5)}`;
      const a = result.bf.amp;
      if (a && rho[j * nx + i] < 1) s += `, peak-to-peak ${stateName(0)} ≈ ${a[j * nx + i].toPrecision(4)}`;
      s += ' (nearest grid point)';
    }
  }
  $('hover').textContent = s;
}

// ---------------------------------------------------------------------------------------------
// zoom / pan (as the InterpolatedNyquist page): wheel about the cursor, left-drag zoom box,
// middle- or shift-drag pan, two-finger pinch + pan, double-click reset. The view is the axis
// parameters' range: it is written into the model text and the chart is recomputed.
// ---------------------------------------------------------------------------------------------
let home = new Map();                             // parameter name -> [lo, hi] as last typed / loaded
let recomputeTimer = null;
function rememberHome() { home = new Map(model.params.map((q) => [q.name, [q.lo, q.hi]])); }

function setView(xr, yr) {
  if (!model) return;
  const ok = (r) => isFinite(r[0]) && isFinite(r[1]) && r[1] - r[0] > 1e-9 * Math.max(1, Math.abs(r[0]), Math.abs(r[1]));
  if (!ok(xr) || !ok(yr)) return;
  const px = model.params[axes[0]], py = model.params[axes[1]];
  [px.lo, px.hi] = xr; [py.lo, py.hi] = yr;
  draw();
  countOnce('zoom-pan');
  clearTimeout(recomputeTimer);
  recomputeTimer = setTimeout(() => { commitView(); schedule(true); }, 250);
}

/** rounded ranges into the model text and the parameter table */
function commitView() {
  for (const k of axes) {
    const q = model.params[k], d = Math.pow(10, Math.floor(Math.log10(q.hi - q.lo)) - 4);
    q.lo = Math.round(q.lo / d) * d; q.hi = Math.round(q.hi / d) * d;
    const fmt = (v) => String(+v.toPrecision(8));
    const esc = q.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(`^(\\s*${esc}\\s*=\\s*)([^@#\\n]*?)(\\s*(?:[@#]|$))`, 'mu');
    modelText = modelText.replace(re, (_, a, b, c) => `${a}${fmt(q.lo)}:${fmt(q.hi)}${c}`);
  }
  $('eq').value = modelText;
  buildParams();
}

function resetAxes() {
  if (!model) return;
  const h = (k) => (home.get(model.params[k].name) || [model.params[k].lo, model.params[k].hi]).slice();
  setView(h(axes[0]), h(axes[1]));
}

function bindPointer() {
  const cv = $('chart');
  const touches = new Map();
  let drag = null, pinch = null;
  cv.addEventListener('pointerdown', (ev) => {
    if (ev.pointerType === 'touch') {
      touches.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      cv.setPointerCapture(ev.pointerId);
      if (touches.size === 2) {
        const [a, b] = [...touches.values()];
        pinch = { view: viewBox(), m: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, d: Math.hypot(a.x - b.x, a.y - b.y) || 1 };
      }
      return;
    }
    const q = locate(ev);
    if (!q.inside) return;
    if (ev.button === 1 || (ev.button === 0 && ev.shiftKey)) {
      drag = { mode: 'pan', q, view: viewBox() }; cv.classList.add('panning');
    } else if (ev.button === 0) {
      drag = { mode: 'box', q };
    } else return;
    ev.preventDefault();
    cv.setPointerCapture(ev.pointerId);
  });
  cv.addEventListener('pointermove', (ev) => {
    if (ev.pointerType === 'touch') {
      if (!touches.has(ev.pointerId)) return;
      touches.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      if (pinch && touches.size === 2) {
        const [a, b] = [...touches.values()], { W, H } = geom(), v0 = pinch.view, rc = cv.getBoundingClientRect();
        const m = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, s = pinch.d / (Math.hypot(a.x - b.x, a.y - b.y) || 1);
        // the data point under the old finger midpoint stays under the new one; the span scales by s
        const fx0 = (pinch.m.x - rc.left - PAD.l) / W, fy0 = (pinch.m.y - rc.top - PAD.t) / H;
        const fx1 = (m.x - rc.left - PAD.l) / W, fy1 = (m.y - rc.top - PAD.t) / H;
        const sx = (v0.x1 - v0.x0) * s, sy = (v0.y1 - v0.y0) * s;
        const dx = v0.x0 + fx0 * (v0.x1 - v0.x0), dy = v0.y1 - fy0 * (v0.y1 - v0.y0);
        const x0 = dx - fx1 * sx, y1 = dy + fy1 * sy;
        setView([x0, x0 + sx], [y1 - sy, y1]);
      }
      return;
    }
    hover(ev);
    if (!drag) return;
    const q = locate(ev);
    if (drag.mode === 'pan') {
      const v = drag.view, ddx = (q.px - drag.q.px) / q.W * (v.x1 - v.x0), ddy = (q.py - drag.q.py) / q.H * (v.y1 - v.y0);
      setView([v.x0 - ddx, v.x1 - ddx], [v.y0 + ddy, v.y1 + ddy]);
    } else {
      const cl = (z, lo, hi) => Math.min(hi, Math.max(lo, z));
      zoomRect = [drag.q.cx, drag.q.cy, cl(q.cx, PAD.l, PAD.l + q.W), cl(q.cy, PAD.t, PAD.t + q.H)];
      draw();
    }
  });
  const end = (ev) => {
    if (ev.pointerType === 'touch') {
      touches.delete(ev.pointerId);
      if (touches.size < 2) pinch = null;
      return;
    }
    if (!drag) return;
    cv.classList.remove('panning');
    if (drag.mode === 'box' && zoomRect) {
      const [a, b, c, d] = zoomRect;
      zoomRect = null;
      if (Math.abs(c - a) > 6 && Math.abs(d - b) > 6) {
        const { W, H } = geom(), v = viewBox();
        const xd = (px) => v.x0 + (px - PAD.l) / W * (v.x1 - v.x0), yd = (py) => v.y1 - (py - PAD.t) / H * (v.y1 - v.y0);
        setView([xd(Math.min(a, c)), xd(Math.max(a, c))], [yd(Math.max(b, d)), yd(Math.min(b, d))]);
      } else draw();
    }
    drag = null;
  };
  cv.addEventListener('pointerup', end);
  cv.addEventListener('pointercancel', end);
  cv.addEventListener('wheel', (ev) => {
    const q = locate(ev);
    if (!q.inside || !result) return;
    ev.preventDefault();
    const dy = ev.deltaMode === 1 ? ev.deltaY * 33 : ev.deltaMode === 2 ? ev.deltaY * 400 : ev.deltaY;
    const f = Math.exp(Math.max(-1, Math.min(1, dy * 0.0015)));
    const v = viewBox();
    setView([q.x + (v.x0 - q.x) * f, q.x + (v.x1 - q.x) * f], [q.y + (v.y0 - q.y) * f, q.y + (v.y1 - q.y) * f]);
  }, { passive: false });
  cv.addEventListener('dblclick', (ev) => { ev.preventDefault(); resetAxes(); });
  cv.addEventListener('pointerleave', () => { $('hover').textContent = ''; });
}

/** draggable divider between the side panel and the chart (its width is kept in localStorage) */
function bindSplit() {
  const sp = $('split'), mainEl = document.querySelector('main'), KEY = 'sosdgpu.split.v1';
  const apply = (w) => {
    w = Math.max(260, Math.min(w, 0.7 * window.innerWidth));
    mainEl.style.setProperty('--side', w + 'px');
    sizeCanvas(); draw();
  };
  try { const w = +localStorage.getItem(KEY); if (w) apply(w); } catch (e) { /* no storage */ }
  let x0 = null, w0 = 0;
  sp.addEventListener('pointerdown', (ev) => {
    x0 = ev.clientX; w0 = document.querySelector('aside').getBoundingClientRect().width;
    sp.setPointerCapture(ev.pointerId); sp.classList.add('drag'); ev.preventDefault();
  });
  sp.addEventListener('pointermove', (ev) => { if (x0 !== null) apply(w0 + ev.clientX - x0); });
  const up = () => {
    if (x0 === null) return;
    x0 = null; sp.classList.remove('drag');
    try { localStorage.setItem(KEY, String(parseFloat(mainEl.style.getPropertyValue('--side')))); } catch (e) { /* ignore */ }
  };
  sp.addEventListener('pointerup', up); sp.addEventListener('pointercancel', up);
  sp.addEventListener('dblclick', () => {
    mainEl.style.removeProperty('--side');
    try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ }
    sizeCanvas(); draw();
  });
}

function savePng() {
  countOnce('save-png');
  $('chart').toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `sosd_${ex ? ex.key : 'chart'}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }, 'image/png');
}

// ---------------------------------------------------------------------------------------------
// benchmark and validation
// ---------------------------------------------------------------------------------------------
async function bench() {
  countOnce('bench-run');
  const t = [];
  for (let k = 0; k < 3; k++) { generation++; await compute(); if (result) t.push(result.ms); }
  t.sort((a, b) => a - b);
  log(`benchmark (${engine.deviceName}): median ${t[1].toFixed(0)} ms for ${result.n} ρ = ${(1000 * t[1] / result.n).toFixed(1)} µs/ρ`);
}

async function validate() {
  countOnce('validate-run');
  $('report').innerHTML = '<p>Running the examples on the reference grids…</p>';
  const ref = await (await fetch('validate/ref.json', { cache: 'no-cache' })).json();
  const rows = [];
  for (const e of EXAMPLES) {
    const r = ref[e.key]; if (!r) continue;
    const mdl = parseModel(e.text);
    const names = mdl.params.map((q) => q.name);
    const xi = names.indexOf(e.axes[0]), yi = names.indexOf(e.axes[1]);
    const xy = Float32Array.from(r.pts.flat());
    // Krylov m = 24, and the example's own m (its default), each also with the Float16 basis
    for (const [m, f16] of [[24, false], [e.m, false], [e.m, 'V']]) {
      const out = await engine.evaluate(mdl, mdl.params.map((q) => q.value), xi, yi, xy, { S: r.S, p: r.p, m, f16 });
      let worst = 0, med = [], mis = 0;
      r.rho.forEach((v, k) => {
        const d = Math.abs(out.rho[k] - v) / v; med.push(d); worst = Math.max(worst, d);
        if ((out.rho[k] >= 1) !== (v >= 1)) mis++;
      });
      med.sort((a, b) => a - b);
      rows.push({ key: e.key + (f16 ? ' (f16 basis)' : ''), n: r.rho.length, med: med[med.length >> 1], worst, mis, ms: out.ms, p: r.p, r: out.r, m });
    }
  }
  $('report').innerHTML = '<table><tr><th>example</th><th>points</th><th>p</th><th>r</th><th>m</th><th>median |Δρ|/ρ</th><th>max |Δρ|/ρ</th>' +
    '<th>misclassified</th><th>GPU ms</th></tr>' + rows.map((w) =>
      `<tr><td>${w.key}</td><td>${w.n}</td><td>${w.p}</td><td>${w.r}</td><td>${w.m}</td><td>${w.med.toExponential(1)}</td><td>${w.worst.toExponential(1)}</td>` +
      `<td class="${w.mis ? 'fail' : 'pass'}">${w.mis}</td><td>${w.ms.toFixed(0)}</td></tr>`).join('') + '</table>' +
    '<p class="muted">Reference: Float64 CPU solver of SOSD.jl (validate/reference.jl), same discretization (s, p); here Float32, Krylov m = 24 and the default m of the example.</p>';
  console.log('VALIDATE', JSON.stringify(rows));
}

main();
