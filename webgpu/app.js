// UI of the SOSD WebGPU page: model text -> parameters (sliders, X / Y axes) -> brute-force grid
// and / or MDBM boundary on the GPU -> chart. ?validate=1 compares the examples with the Float64
// Julia reference (validate/ref.json), ?bench=1 runs the benchmark once.

import { Engine, WebGPUUnavailable } from './engine.js';
import { parseModel, ParseError } from './expr.js';
import { EXAMPLES } from './examples.js';
import { mdbmBoundary } from './mdbm.js';
import { initStats, countEvent, countOnce, countModel } from './stats.js';

const $ = (id) => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const BF_GRIDS = [[32, 16], [64, 32], [96, 48], [128, 64], [192, 96], [256, 128], [384, 192], [512, 256]];
const MD_GRIDS = [[6, 4], [8, 5], [12, 7], [16, 8], [24, 8], [24, 12], [32, 16], [48, 12], [64, 16], [96, 24]];
const CR = 0.5;                                   // colour range of log10 ρ: [-CR, CR]
const STOPS = [[5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222], [247, 247, 247],
               [244, 165, 130], [214, 96, 77], [178, 24, 43], [103, 0, 31]];

let engine = null, model = null, modelText = '', ex = null;
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
  $('go').onclick = () => compute();
  $('bench').onclick = () => bench();
  $('mdIt').oninput = () => { $('mdItV').textContent = $('mdIt').value; schedule(); };
  for (const id of ['bfOn', 'bfGrid', 'mdOn', 'mdGrid', 'mdNb', 'S', 'p', 'm']) $(id).onchange = () => schedule(true);
  $('mdPts').onchange = draw;
  let eqTimer = null;
  $('eq').oninput = () => { clearTimeout(eqTimer); eqTimer = setTimeout(() => setModelText($('eq').value, true), 400); };
  $('helpBtn').onclick = () => { $('helpBg').hidden = false; countOnce('help-open'); };
  $('helpClose').onclick = $('helpBg').onclick = (ev) => { if (ev.target === $('helpBg') || ev.target === $('helpClose')) $('helpBg').hidden = true; };
  $('chart').onmousemove = hover;
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
    buildParams();
    schedule(true);
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
    lo.onchange = () => { q.lo = +lo.value; setRange(); schedule(true); };
    hi.onchange = () => { q.hi = +hi.value; setRange(); schedule(true); };
    const bx = tr.querySelector('.bx'), by = tr.querySelector('.by');
    const isAx = axes.includes(i);
    r.disabled = isAx; tr2.style.display = isAx ? 'none' : '';
    if (isAx) v.textContent = axes[0] === i ? 'X axis' : 'Y axis';
    bx.classList.toggle('on', axes[0] === i); by.classList.toggle('on', axes[1] === i);
    bx.onclick = () => { if (axes[1] === i) axes[1] = axes[0]; axes[0] = i; buildParams(); schedule(true); };
    by.onclick = () => { if (axes[0] === i) axes[0] = axes[1]; axes[1] = i; buildParams(); schedule(true); };
    tb.appendChild(tr); tb.appendChild(tr2);
  });
}

// ---------------------------------------------------------------------------------------------
// computing
// ---------------------------------------------------------------------------------------------
function schedule(force = false) {
  if (!force && !$('auto').checked) return;
  if (busy) { pending = true; generation++; return; }
  compute();
}

function opts() {
  return { S: +$('S').value, p: Math.max(4, +$('p').value | 0), m: Math.min(24, Math.max(4, +$('m').value | 0)) };
}

async function compute() {
  if (!engine || !model) return;
  busy = true; pending = false;
  const gen = ++generation;
  const cancel = () => gen !== generation;
  const [xi, yi] = axes, px = model.params[xi], py = model.params[yi];
  const box = { x0: px.lo, x1: px.hi, y0: py.lo, y1: py.hi };
  const o = opts(), vals = values.slice();
  const res = { box, xname: px.name, yname: py.name, bf: null, md: null, n: 0, ms: 0, r: 0, N: 0 };
  const evalBatch = async (xy) => {
    const out = await engine.evaluate(model, vals, xi, yi, xy, { ...o, cancel,
      onProgress: (f) => { $('progress').firstChild.style.width = (100 * f).toFixed(1) + '%'; } });
    res.n += xy.length / 2; res.ms += out.ms; res.r = out.r; res.N = (out.r + 1) * (o.S + 1) * model.D;
    return out.rho;
  };
  try {
    if ($('bfOn').checked) {
      countOnce('brute-force');
      const [nx, ny] = BF_GRIDS[+$('bfGrid').value];
      const xy = new Float32Array(2 * nx * ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        xy[2 * k] = box.x0 + (box.x1 - box.x0) * i / (nx - 1);
        xy[2 * k + 1] = box.y0 + (box.y1 - box.y0) * j / (ny - 1);
      }
      const rho = await evalBatch(xy);
      if (cancel()) return finish();
      res.bf = { nx, ny, rho };
      result = res; draw(); stats(res);
    }
    if ($('mdOn').checked) {
      countOnce('mdbm');
      const [a, b] = MD_GRIDS[+$('mdGrid').value];
      res.md = await mdbmBoundary(evalBatch, box, a, b, +$('mdIt').value,
        { neighbour: $('mdNb').checked, cancel, onStage: () => { if (!cancel()) { result = res; draw(); stats(res); } } });
      if (cancel()) return finish();
    }
    result = res; draw(); stats(res);
  } catch (e) {
    log('error: ' + e.message);
    showMsg('<b>Error:</b> ' + e.message);
  }
  finish();
  function finish() {
    busy = false;
    $('progress').firstChild.style.width = '0%';
    if (pending) compute();
  }
}

function stats(res) {
  $('sN').innerHTML = res.n.toLocaleString() + (res.md ? ` <small>(MDBM ${res.md.points.length / 2 | 0})</small>` : '');
  $('sT').innerHTML = res.ms < 1000 ? `${res.ms.toFixed(0)} <small>ms</small>` : `${(res.ms / 1000).toFixed(2)} <small>s</small>`;
  $('sU').innerHTML = res.n ? `${(1000 * res.ms / res.n).toFixed(1)} <small>µs</small>` : '–';
  if (res.bf) {
    let u = 0; for (const v of res.bf.rho) if (v >= 1) u++;
    $('sUn').innerHTML = `${(100 * u / res.bf.rho.length).toFixed(1)} <small>%</small>`;
  } else $('sUn').textContent = '–';
  $('sR').innerHTML = `${res.r} <small>· N = ${res.N}</small>`;
}

// ---------------------------------------------------------------------------------------------
// drawing
// ---------------------------------------------------------------------------------------------
const PAD = { l: 72, r: 96, t: 14, b: 56 };
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

function draw() {
  const cv = $('chart'), cx = cv.getContext('2d');
  const W = cv.width - PAD.l - PAD.r, H = cv.height - PAD.t - PAD.b;
  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue('--ink').trim() || '#222', muted = css.getPropertyValue('--muted').trim() || '#666';
  cx.clearRect(0, 0, cv.width, cv.height);
  cx.fillStyle = '#f7f7f7'; cx.fillRect(PAD.l, PAD.t, W, H);
  if (!result) return;
  const { box } = result;
  const X = (x) => PAD.l + (x - box.x0) / (box.x1 - box.x0) * W;
  const Y = (y) => PAD.t + H - (y - box.y0) / (box.y1 - box.y0) * H;
  if (result.bf) {
    const { nx, ny, rho } = result.bf;
    const img = new ImageData(nx, ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = cmap(Math.log10(Math.max(rho[j * nx + i], 1e-30))), o = 4 * (i + nx * (ny - 1 - j));
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
    }
    const off = new OffscreenCanvas(nx, ny); off.getContext('2d').putImageData(img, 0, 0);
    cx.imageSmoothingEnabled = true;
    // grid nodes at the box edges: the image spans half a cell beyond them
    const dx = W / (nx - 1), dy = H / (ny - 1);
    cx.save(); cx.beginPath(); cx.rect(PAD.l, PAD.t, W, H); cx.clip();
    cx.drawImage(off, PAD.l - dx / 2, PAD.t - dy / 2, W + dx, H + dy);
    // boundary from the grid (marching squares on log ρ)
    cx.strokeStyle = '#000'; cx.lineWidth = 1.2; cx.beginPath();
    const f = (i, j) => Math.log(Math.max(rho[j * nx + i], 1e-30));
    const px = (i) => PAD.l + i * dx, py = (j) => PAD.t + H - j * dy;
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      const v = [f(i, j), f(i + 1, j), f(i + 1, j + 1), f(i, j + 1)], P = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], pts = [];
      for (let e = 0; e < 4; e++) { const a = v[e], b = v[(e + 1) % 4];
        if ((a >= 0) !== (b >= 0)) { const t = a / (a - b), A = P[e], B = P[(e + 1) % 4];
          pts.push([px(A[0] + t * (B[0] - A[0])), py(A[1] + t * (B[1] - A[1]))]); } }
      if (pts.length >= 2) { cx.moveTo(...pts[0]); cx.lineTo(...pts[1]); }
      if (pts.length === 4) { cx.moveTo(...pts[2]); cx.lineTo(...pts[3]); }
    }
    cx.stroke(); cx.restore();
  }
  if (result.md) {
    cx.save(); cx.beginPath(); cx.rect(PAD.l, PAD.t, W, H); cx.clip();
    if ($('mdPts').checked) {
      const { points, rho } = result.md;
      for (let k = 0; k < rho.length; k++) {
        cx.fillStyle = rho[k] >= 1 ? '#b2182b' : '#2166ac';
        cx.fillRect(X(points[2 * k]) - 1.5, Y(points[2 * k + 1]) - 1.5, 3, 3);
      }
    }
    cx.strokeStyle = result.bf ? '#ffd400' : '#000'; cx.lineWidth = result.bf ? 2.2 : 2;
    cx.lineCap = 'round'; cx.beginPath();
    for (const s of result.md.segments) { cx.moveTo(X(s[0]), Y(s[1])); cx.lineTo(X(s[2]), Y(s[3])); }
    cx.stroke(); cx.restore();
  }
  // axes, ticks, labels
  cx.strokeStyle = muted; cx.lineWidth = 1; cx.strokeRect(PAD.l, PAD.t, W, H);
  cx.fillStyle = ink; cx.font = '13px system-ui, sans-serif'; cx.textAlign = 'center';
  for (const v of nice(box.x0, box.x1, 8)) { const x = X(v); cx.fillRect(x, PAD.t + H, 1, 5); cx.fillText(+v.toPrecision(6), x, PAD.t + H + 19); }
  cx.fillText(axisLabel(result.xname), PAD.l + W / 2, PAD.t + H + 44);
  cx.textAlign = 'right';
  for (const v of nice(box.y0, box.y1, 6)) { const y = Y(v); cx.fillRect(PAD.l - 5, y, 5, 1); cx.fillText(+v.toPrecision(6), PAD.l - 8, y + 4); }
  cx.save(); cx.translate(20, PAD.t + H / 2); cx.rotate(-Math.PI / 2); cx.textAlign = 'center';
  cx.fillText(axisLabel(result.yname), 0, 0); cx.restore();
  // colour bar
  const bx = PAD.l + W + 18, bw = 16;
  for (let k = 0; k < H; k++) { const c = cmap(CR * (1 - 2 * k / H)); cx.fillStyle = `rgb(${c})`; cx.fillRect(bx, PAD.t + k, bw, 1); }
  cx.strokeRect(bx, PAD.t, bw, H); cx.fillStyle = ink; cx.textAlign = 'left';
  for (const v of [-CR, -CR / 2, 0, CR / 2, CR]) cx.fillText(v.toFixed(2), bx + bw + 4, PAD.t + (1 - v / CR) / 2 * H + 4);
  cx.save(); cx.translate(bx + bw + 46, PAD.t + H / 2); cx.rotate(-Math.PI / 2); cx.textAlign = 'center';
  cx.fillText('log₁₀ ρ   (ρ < 1 stable, blue)', 0, 0); cx.restore();
}

function axisLabel(name) {
  // the comment of the declaration line, if any, as the axis label: "n = ... # spindle speed [rpm]"
  const line = modelText.split('\n').find((l) => l.replace(/\s/g, '').startsWith(name + '=') && l.includes('#'));
  const c = line ? line.split('#')[1].trim() : '';
  return c ? `${name}: ${c}` : name;
}

function hover(ev) {
  if (!result) return;
  const cv = $('chart'), rc = cv.getBoundingClientRect();
  const W = cv.width - PAD.l - PAD.r, H = cv.height - PAD.t - PAD.b;
  const px = (ev.clientX - rc.left) * cv.width / rc.width - PAD.l, py = (ev.clientY - rc.top) * cv.height / rc.height - PAD.t;
  if (px < 0 || py < 0 || px > W || py > H) { $('hover').textContent = ''; return; }
  const { box } = result;
  const x = box.x0 + px / W * (box.x1 - box.x0), y = box.y1 - py / H * (box.y1 - box.y0);
  let s = `${result.xname} = ${x.toPrecision(5)}, ${result.yname} = ${y.toPrecision(5)}`;
  if (result.bf) {
    const { nx, ny, rho } = result.bf;
    const i = Math.round(px / W * (nx - 1)), j = Math.round((H - py) / H * (ny - 1));
    s += `   ρ ≈ ${rho[j * nx + i].toPrecision(5)} (nearest grid point)`;
  }
  $('hover').textContent = s;
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
