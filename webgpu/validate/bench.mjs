// Kernel benchmark for the browser console (served page, any URL of this folder):
//   const B = await import('/validate/bench.mjs?v=' + Date.now());
//   await B.run()                 -> table: GPU µs per ρ (prep / main, timestamp queries), wall
//   B.keep('base'); ... B.compare('base')   -> accuracy of a changed kernel vs a kept run
// Cases: the examples at their defaults and at the validation discretization, a 256×128 grid
// over the example's axes (best of `reps`), and a 16×16 batch (latency).

const v = Date.now();
const { Engine } = await import(`../engine.js?v=${v}`);
const { EXAMPLES } = await import(`../examples.js?v=${v}`);
const { parseModel } = await import(`../expr.js?v=${v}`);

export const CASES = [
  { key: 'mathieu', S: 2, p: 20, m: 7 },
  { key: 'mathieu', S: 3, p: 40, m: 12 },
  { key: 'milling', S: 3, p: 40, m: 12 },
  { key: 'turning_ssv', S: 3, p: 200, m: 12 },
];

let eng = null;
const results = new Map();
const kept = (globalThis.__benchKept ??= new Map());     // survives re-imports of this module

export async function engine() { if (!eng) eng = await Engine.create(() => {}); return eng; }

function setup(c, nx, ny) {
  const e = EXAMPLES.find((q) => q.key === c.key), mdl = parseModel(e.text);
  const names = mdl.params.map((q) => q.name), xi = names.indexOf(e.axes[0]), yi = names.indexOf(e.axes[1]);
  const px = mdl.params[xi], py = mdl.params[yi];
  return { mdl, xi, yi, vals: mdl.params.map((q) => q.value), grid: { nx, ny, box: { x0: px.lo, x1: px.hi, y0: py.lo, y1: py.hi } } };
}

export async function one(c, nx = 256, ny = 128, reps = 3, extra = {}) {
  const E = await engine();
  const { mdl, xi, yi, vals, grid } = setup(c, nx, ny);
  let best = null;
  for (let k = 0; k < reps; k++) {
    const out = await E.evaluate(mdl, vals, xi, yi, grid, { S: c.S, p: c.p, m: c.m, ...extra });
    const g = out.gpuPrep + out.gpuMain;
    if (!best || g < best.g) best = { g, out };
  }
  const n = nx * ny, o = best.out;
  return { o, row: `${c.key} GL${c.S} p${c.p} m${c.m} n=${n}: prep ${(1000 * o.gpuPrep / n).toFixed(2)} + main ${(1000 * o.gpuMain / n).toFixed(2)} = ${(1000 * best.g / n).toFixed(2)} µs/ρ GPU, wall ${o.ms.toFixed(0)} ms, lpp ${o.lpp}` };
}

export async function run(extra = {}, cases = CASES) {
  const rows = [];
  for (const c of cases) {
    const big = await one(c, 256, 128, 3, extra);
    results.set(JSON.stringify(c), big.o.rho);
    rows.push(big.row);
    rows.push((await one(c, 16, 16, 3, extra)).row);
  }
  return rows;
}

export function keep(name) { kept.set(name, new Map([...results].map(([k, r]) => [k, r.slice()]))); return [...results.keys()]; }

/** accuracy of the last run vs a kept one: median / max relative difference, misclassified */
export function compare(name) {
  const base = kept.get(name), rows = [];
  for (const [k, r] of results) {
    const b = base && base.get(k);
    if (!b) continue;
    const d = []; let mis = 0;
    r.forEach((x, i) => { d.push(Math.abs(x - b[i]) / b[i]); if ((x >= 1) !== (b[i] >= 1)) mis++; });
    d.sort((a, c) => a - c);
    rows.push(`${k}: median ${d[d.length >> 1].toExponential(1)}, p99 ${d[Math.floor(0.99 * d.length)].toExponential(1)}, max ${d[d.length - 1].toExponential(1)}, misclassified ${mis}/${d.length}`);
  }
  return rows;
}

/** reference chart (high resolution in s, p, m) on an nx × ny grid, kept as `ref:<key>` */
export async function reference(key, c, nx = 128, ny = 64) {
  const E = await engine();
  const { mdl, xi, yi, vals, grid } = setup({ key }, nx, ny);
  const out = await E.evaluate(mdl, vals, xi, yi, grid, c);
  kept.set('ref:' + key, out.rho.slice());
  return `${key} reference ${JSON.stringify(c)}: ${(1000 * (out.gpuPrep + out.gpuMain) / (nx * ny)).toFixed(1)} µs/ρ`;
}

/** misclassified points vs the reference and GPU µs/ρ for each discretization in `list` */
export async function scan(key, list, nx = 128, ny = 64, extra = {}) {
  const E = await engine(), ref = kept.get('ref:' + key), rows = [];
  const { mdl, xi, yi, vals, grid } = setup({ key }, nx, ny);
  for (const c of list) {
    const out = await E.evaluate(mdl, vals, xi, yi, grid, { ...c, ...extra });
    let mis = 0; const d = [];
    out.rho.forEach((x, i) => { if ((x >= 1) !== (ref[i] >= 1)) mis++; d.push(Math.abs(x - ref[i]) / ref[i]); });
    d.sort((a, b) => a - b);
    rows.push({ ...c, us: +(1000 * (out.gpuPrep + out.gpuMain) / (nx * ny)).toFixed(2), mis, med: +d[d.length >> 1].toExponential(1), p99: +d[Math.floor(0.99 * d.length)].toExponential(1) });
  }
  return rows;
}
