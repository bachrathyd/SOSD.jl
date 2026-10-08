// UI of the SOSD WebGPU page: model text -> parameters (sliders, X / Y axes) -> brute-force grid
// and / or MDBM boundary on the GPU -> chart. ?validate=1 compares the examples with the Float64
// Julia reference (validate/ref.json), ?bench=1 runs the benchmark once.

import { Engine, WebGPUUnavailable } from './engine.js';
import { parseModel, ParseError } from './expr.js';
import { EXAMPLES } from './examples.js';
import { mdbmBoundary } from './mdbm.js';
import { initStats, countEvent, countOnce, countModel } from './stats.js';
import { CpuPool } from './cpu.js';

const $ = (id) => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const BF_GRIDS = [[32, 16], [64, 32], [128, 64], [192, 96], [256, 128], [384, 192], [512, 256], [768, 384], [1024, 512]];
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
  BF_GRIDS.forEach(([a, b], i) => $('bfGrid').add(new Option(`${a} × ${b} = ${(a * b).toLocaleString()}`, i)));
  EXPORTS.forEach((e, i) => $('exRes').add(new Option(`${e.name} ${e.w} × ${e.h}`, i)));
  MD_GRIDS.forEach(([a, b], i) => $('mdGrid').add(new Option(`${a} × ${b}`, i)));
  try {
    engine = await Engine.create(log);
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
  for (const id of ['bfOn', 'bfGrid', 'mdOn', 'mdGrid', 'mdNb', 'S', 'p', 'm']) $(id).onchange = () => { updateEtas(); schedule(true, true); };
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
  window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { sizeCanvas(); draw(); }, 60); });
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
  $('bfGrid').value = BF_GRIDS.findIndex(([a, b]) => a === ex.brute[0] && b === ex.brute[1]);
  $('mdGrid').value = MD_GRIDS.findIndex(([a, b]) => a === ex.mdbm[0] && b === ex.mdbm[1]);
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
    r.oninput = () => { values[i] = +r.value; v.textContent = fmt(values[i]); schedule(); };
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
  return { S: +$('S').value, p: Math.max(4, +$('p').value | 0), m: Math.min(24, Math.max(3, +$('m').value | 0)) };
}

function timeLimit() { const v = parseFloat($('tLimit').value); return v > 0 ? 1000 * v : Infinity; }

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
  const res = { box, xname: px.name, yname: py.name, bf: null, md: null, n: 0, ms: 0, r: 0, N: 0, nCpu: 0, msCpu: 0 };
  const progress = (f) => { $('progress').firstChild.style.width = (100 * f).toFixed(1) + '%'; };
  const evalBatch = async (pts) => {
    const out = await engine.evaluate(mdl, vals, xi, yi, pts, { ...o, cancel, onProgress: progress });
    if (!out.done) throw new Cancelled();
    res.n += out.rho.length; res.ms += out.ms; res.r = out.r; res.N = (out.r + 1) * (o.S + 1) * mdl.D;
    return out.rho;
  };
  const evalCpu = async (xy) => {
    if (cancel()) throw new Cancelled();
    const tc = performance.now();
    const r = engine.delaySteps(mdl, vals, xi, yi, xy.length / 2, (i) => [xy[2 * i], xy[2 * i + 1]], o.p);
    const rho = await cpu.evaluate(mdl, vals, xi, yi, xy, { ...o, r });
    res.nCpu += rho.length; res.msCpu += performance.now() - tc;
    return rho;
  };
  try {
    const prevMd = result && result.md;
    if ($('bfOn').checked) {
      countOnce('brute-force');
      const [nx, ny] = BF_GRIDS[+$('bfGrid').value];
      const rho = await evalBatch({ nx, ny, box });
      res.bf = { nx, ny, rho };
      lastRate = { key: rateKey(), us: 1000 * res.ms / (nx * ny) };
      updateEtas();
      // the new map at once, the old boundary on it until the new one is ready
      result = { ...res, md: $('mdOn').checked ? prevMd : null };
      draw(); stats(res);
    }
    if ($('mdOn').checked) {
      countOnce('mdbm');
      const [a, b] = MD_GRIDS[+$('mdGrid').value];
      const nb = $('mdNb').value;
      res.md = await mdbmBoundary(evalBatch, box, a, b, +$('mdIt').value,
        { neighbour: nb, evalNeighbour: nb === 'end' ? evalCpu : evalBatch, cancel: () => { if (cancel()) throw new Cancelled(); return false; } });
    }
    result = res; draw(); stats(res);
    showMsg('');
  } catch (e) {
    if (e instanceof Cancelled) {
      if (timedOut) showMsg(`Stopped after the time limit (${(limit / 1000).toFixed(1)} s): the chart shows the last complete result. ` +
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
function rateKey() { const o = opts(); return `${modelText.length}:${modelText.slice(0, 200)}|${o.S}|${o.p}|${o.m}|${axes}`; }
function fmtTime(ms) {
  if (ms < 1000) return `${Math.max(10, Math.round(ms / 10) * 10)} ms`;
  if (ms < 120e3) return `${(ms / 1000).toPrecision(2)} s`;
  return `${(ms / 60e3).toPrecision(2)} min`;
}
function updateEtas() {
  const us = lastRate && lastRate.key === rateKey() ? lastRate.us : null;
  [...$('bfGrid').options].forEach((op, i) => {
    const [a, b] = BF_GRIDS[i];
    op.text = `${a} × ${b} = ${(a * b).toLocaleString()}` + (us ? `  (≈ ${fmtTime(us * a * b / 1000)})` : '');
  });
  [...$('exRes').options].forEach((op, i) => {
    const e = EXPORTS[i], [nx, ny] = exportGrid(e.w, e.h);
    op.text = `${e.name} ${e.w} × ${e.h}` + (us ? `  (≈ ${fmtTime(us * nx * ny / 1000)})` : '');
  });
}

// ---------------------------------------------------------------------------------------------
// drawing: the view box is the axis parameters' current range; a result computed for another box
// (during zoom / pan, until the recompute arrives) is drawn mapped into the view
// ---------------------------------------------------------------------------------------------
const PAD = { l: 72, r: 96, t: 14, b: 56 };
let zoomRect = null;                              // [x0, y0, x1, y1] css px while dragging a zoom box
function cmap(v) {
  const t = Math.min(1, Math.max(0, (v / CR + 1) / 2)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t)), w = t - i;
  return STOPS[i].map((c, k) => Math.round(c + w * (STOPS[i + 1][k] - c)));
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
    const { box } = res, { nx, ny, rho } = res.bf;
    if (!res.bf.img) {
      const img = new ImageData(nx, ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const c = cmap(Math.log10(Math.max(rho[j * nx + i], 1e-30))), o = 4 * (i + nx * (ny - 1 - j));
        img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
      }
      res.bf.img = new OffscreenCanvas(nx, ny); res.bf.img.getContext('2d').putImageData(img, 0, 0);
    }
    cx.imageSmoothingEnabled = true;
    // grid nodes at the box edges: the image spans half a cell beyond them
    const hx = (box.x1 - box.x0) / (nx - 1) / 2, hy = (box.y1 - box.y0) / (ny - 1) / 2;
    const ax = X(box.x0 - hx), ay = Y(box.y1 + hy);
    cx.drawImage(res.bf.img, ax, ay, X(box.x1 + hx) - ax, Y(box.y0 - hy) - ay);
    // boundary from the grid (marching squares on log ρ)
    cx.strokeStyle = '#000'; cx.lineWidth = 1.2 * sc; cx.beginPath();
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
    cx.strokeStyle = res.bf ? '#ffd400' : '#000'; cx.lineWidth = (res.bf ? 2.2 : 2) * sc;
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
  // colour bar
  const bx = L + W + 18 * sc, bw = 16 * sc;
  for (let k = 0; k < H; k++) { const c = cmap(CR * (1 - 2 * k / H)); cx.fillStyle = `rgb(${c})`; cx.fillRect(bx, T + k, bw, 1.5); }
  cx.strokeRect(bx, T, bw, H); cx.fillStyle = ink; cx.textAlign = 'left';
  for (const v of [-CR, -CR / 2, 0, CR / 2, CR]) cx.fillText(v.toFixed(2), bx + bw + 4 * sc, T + (1 - v / CR) / 2 * H + 4 * sc);
  cx.save(); cx.translate(bx + bw + 46 * sc, T + H / 2); cx.rotate(-Math.PI / 2); cx.textAlign = 'center';
  cx.fillText('log₁₀ ρ   (ρ < 1 stable, blue)', 0, 0); cx.restore();
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
    const out = await engine.evaluate(model, values.slice(), xi, yi, { nx, ny, box }, { ...opts(),
      cancel: () => job.cancelled,
      onProgress: (f) => { $('progress').firstChild.style.width = (100 * f).toFixed(1) + '%'; btn.textContent = `Cancel (${(100 * f).toFixed(0)} %)`; } });
    if (out.done) {
      const same = result && result.box && ['x0', 'x1', 'y0', 'y1'].every((k) => result.box[k] === box[k]);
      const res = { box, xname: px.name, yname: py.name, bf: { nx, ny, rho: out.rho }, md: same ? result.md : null };
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
    const { nx, ny, rho } = result.bf, { box } = result;
    const i = Math.round((q.x - box.x0) / (box.x1 - box.x0) * (nx - 1)), j = Math.round((q.y - box.y0) / (box.y1 - box.y0) * (ny - 1));
    if (i >= 0 && j >= 0 && i < nx && j < ny) s += `   ρ ≈ ${rho[j * nx + i].toPrecision(5)} (nearest grid point)`;
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
    const out = await engine.evaluate(mdl, mdl.params.map((q) => q.value), xi, yi, xy, { S: r.S, p: r.p, m: 24 });
    let worst = 0, med = [], mis = 0;
    r.rho.forEach((v, k) => {
      const d = Math.abs(out.rho[k] - v) / v; med.push(d); worst = Math.max(worst, d);
      if ((out.rho[k] >= 1) !== (v >= 1)) mis++;
    });
    med.sort((a, b) => a - b);
    rows.push({ key: e.key, n: r.rho.length, med: med[med.length >> 1], worst, mis, ms: out.ms, p: r.p, r: out.r });
  }
  $('report').innerHTML = '<table><tr><th>example</th><th>points</th><th>p</th><th>r</th><th>median |Δρ|/ρ</th><th>max |Δρ|/ρ</th>' +
    '<th>misclassified</th><th>GPU ms</th></tr>' + rows.map((w) =>
      `<tr><td>${w.key}</td><td>${w.n}</td><td>${w.p}</td><td>${w.r}</td><td>${w.med.toExponential(1)}</td><td>${w.worst.toExponential(1)}</td>` +
      `<td class="${w.mis ? 'fail' : 'pass'}">${w.mis}</td><td>${w.ms.toFixed(0)}</td></tr>`).join('') + '</table>' +
    '<p class="muted">Reference: Float64 CPU solver of SOSD.jl (validate/reference.jl), same discretization; here Float32, Krylov m = 24.</p>';
  console.log('VALIDATE', JSON.stringify(rows));
}

main();
