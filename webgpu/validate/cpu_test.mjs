// Float64 CPU solver (cpusolver.js) vs the SOSD.jl reference (validate/ref.json):
//   node validate/cpu_test.mjs
import fs from 'fs';
import { parseModel, modelJS } from '../expr.js';
import { gaussTableau, Engine } from '../engine.js';
import { rhoPoint } from '../cpusolver.js';
import { EXAMPLES } from '../examples.js';

const ref = JSON.parse(fs.readFileSync(new URL('./ref.json', import.meta.url), 'utf8'));
let fail = 0;
for (const e of EXAMPLES) {
  const r0 = ref[e.key]; if (!r0) continue;
  const mdl = parseModel(e.text), f = new Function(modelJS(mdl))();
  const names = mdl.params.map((q) => q.name), xi = names.indexOf(e.axes[0]), yi = names.indexOf(e.axes[1]);
  const r = Engine.prototype.delaySteps.call(null, mdl, mdl.params.map((q) => q.value), xi, yi, r0.pts.length, (i) => r0.pts[i], r0.p);
  const tab = gaussTableau(r0.S), P = Float64Array.from(mdl.params.map((q) => q.value));
  const d = []; let mis = 0; const t0 = Date.now();
  r0.rho.forEach((v, k) => {
    P[xi] = r0.pts[k][0]; P[yi] = r0.pts[k][1];
    const x = rhoPoint(f, P, tab, r0.p, 24, r, mdl.D);
    d.push(Math.abs(x - v) / v); if ((x >= 1) !== (v >= 1)) mis++;
  });
  d.sort((a, b) => a - b);
  fail += mis;
  console.log(`${e.key}: r = ${r}, median |Δρ|/ρ ${d[d.length >> 1].toExponential(1)}, max ${d[d.length - 1].toExponential(1)}, ` +
              `misclassified ${mis}/${d.length}, ${((Date.now() - t0) / d.length).toFixed(2)} ms per point`);
}
process.exit(fail ? 1 : 0);
