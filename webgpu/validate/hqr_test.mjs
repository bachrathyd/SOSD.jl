// node webgpu/validate/hqr_test.mjs <cases.json>: hqr.js vs Julia eigvals (spectral radius)
import { hqr } from '../hqr.js';
import { readFileSync } from 'fs';
const cases = JSON.parse(readFileSync(process.argv[2], 'utf8'));
let worst = 0, fail = 0;
for (const c of cases) {
  const { wr, wi, ok } = hqr(Float64Array.from(c.a), c.n);
  let rho = 0; for (let i = 0; i < c.n; i++) rho = Math.max(rho, Math.hypot(wr[i], wi[i]));
  const rel = Math.abs(rho - c.rho) / c.rho;
  worst = Math.max(worst, rel); if (!ok || rel > 1e-10) fail++;
}
console.log(`${cases.length} matrices, worst relative error of rho ${worst.toExponential(2)}, failures ${fail}`);
