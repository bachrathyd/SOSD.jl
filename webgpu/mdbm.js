// Stability boundary ρ = 1 by the multi-dimensional bisection method (MDBM, Bachrathy & Stépán),
// 2-D, one function f = log ρ: evaluate f on a small initial grid, keep the cells whose corner
// values change sign (the boundary passes through them), split each into 4, evaluate only the new
// corner points, repeat. A neighbour check adds cells next to bracketing cells that also bracket
// (so a branch entering through a discarded cell is not lost). The boundary inside each final
// cell is drawn from the edge crossings (linear interpolation of f, as MDBM's first-order
// interpolation). Every stage's new points are evaluated in ONE batched GPU call.

/**
 * @param evalBatch async (Float32Array xy) -> Float32Array rho
 * @returns { segments: [[x0,y0,x1,y1], ...], points: Float32Array xy, rho: Float32Array, stages: [...] }
 */
export async function mdbmBoundary(evalBatch, box, nx0, ny0, iters, opt = {}) {
  const { neighbour = true, cancel = () => false, onStage = null } = opt;
  const F = 1 << iters;                       // fine lattice units per initial cell
  const NXf = (nx0 - 1) * F, NYf = (ny0 - 1) * F;
  const key = (i, j) => i * (NYf + 1) + j;
  const X = (i) => box.x0 + (box.x1 - box.x0) * i / NXf;
  const Y = (j) => box.y0 + (box.y1 - box.y0) * j / NYf;
  const val = new Map();                      // key -> f = log ρ (+∞ for overflow)
  const stages = [];
  const allPts = [], allRho = [];

  async function evalMissing(list) {          // list of [i, j] lattice points
    const need = [], seen = new Set();
    for (const [i, j] of list) {
      const k = key(i, j);
      if (!val.has(k) && !seen.has(k)) { seen.add(k); need.push([i, j]); }
    }
    if (!need.length) return 0;
    const xy = new Float32Array(2 * need.length);
    need.forEach(([i, j], q) => { xy[2 * q] = X(i); xy[2 * q + 1] = Y(j); });
    const rho = await evalBatch(xy);
    need.forEach(([i, j], q) => {
      const r = rho[q];
      val.set(key(i, j), r > 0 && isFinite(r) ? Math.log(r) : (r > 0 ? 50 : -50));
      allPts.push(xy[2 * q], xy[2 * q + 1]); allRho.push(r);
    });
    return need.length;
  }
  const f = (i, j) => val.get(key(i, j));
  const corners = (c) => [[c.i, c.j], [c.i + c.s, c.j], [c.i + c.s, c.j + c.s], [c.i, c.j + c.s]];
  const brackets = (c) => {
    let pos = false, neg = false;
    for (const [i, j] of corners(c)) { const v = f(i, j); if (v >= 0) pos = true; else neg = true; }
    return pos && neg;
  };

  // initial grid
  const init = [];
  for (let a = 0; a < nx0; a++) for (let b = 0; b < ny0; b++) init.push([a * F, b * F]);
  stages.push({ stage: 'initial grid', n: await evalMissing(init) });
  let cells = [];
  for (let a = 0; a < nx0 - 1; a++) for (let b = 0; b < ny0 - 1; b++) cells.push({ i: a * F, j: b * F, s: F });
  cells = cells.filter(brackets);
  onStage?.(stages[stages.length - 1], cells.length);

  for (let it = 1; it <= iters; it++) {
    if (cancel()) break;
    const s = cells.length ? cells[0].s / 2 : 1;
    const kids = [];
    for (const c of cells) for (const [di, dj] of [[0, 0], [1, 0], [0, 1], [1, 1]]) kids.push({ i: c.i + di * s, j: c.j + dj * s, s });
    const n = await evalMissing(kids.flatMap(corners));
    cells = kids.filter(brackets);
    stages.push({ stage: `iteration ${it}`, n });
    onStage?.(stages[stages.length - 1], cells.length);
  }

  if (neighbour && cells.length && !cancel()) {
    // neighbour check at the final cell size: add bracketing neighbours until none is new
    const s = cells[0].s;
    const have = new Set(cells.map((c) => key(c.i, c.j)));
    let front = cells, added = 0, nEval = 0;
    for (let round = 0; round < 8 && front.length; round++) {
      const cand = [];
      for (const c of front) for (const [di, dj] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        const i = c.i + di * s, j = c.j + dj * s;
        if (i < 0 || j < 0 || i + s > NXf || j + s > NYf) continue;
        const k = key(i, j);
        if (!have.has(k)) { have.add(k); cand.push({ i, j, s }); }
      }
      nEval += await evalMissing(cand.flatMap(corners));
      front = cand.filter(brackets);
      added += front.length;
      cells = cells.concat(front);
    }
    stages.push({ stage: 'neighbour check', n: nEval, added });
    onStage?.(stages[stages.length - 1], cells.length);
  }

  // boundary segments: edge crossings of f inside each final cell
  const segments = [];
  for (const c of cells) {
    const P = corners(c), v = P.map(([i, j]) => f(i, j)), pts = [];
    for (let e = 0; e < 4; e++) {
      const a = v[e], b = v[(e + 1) % 4];
      if ((a >= 0) !== (b >= 0)) {
        const t = a / (a - b), A = P[e], B = P[(e + 1) % 4];
        pts.push([X(A[0] + t * (B[0] - A[0])), Y(A[1] + t * (B[1] - A[1]))]);
      }
    }
    if (pts.length >= 2) segments.push([...pts[0], ...pts[1]]);
    if (pts.length === 4) segments.push([...pts[2], ...pts[3]]);
  }
  return { segments, points: Float32Array.from(allPts), rho: Float32Array.from(allRho), stages,
           cells: cells.length, finest: [NXf + 1, NYf + 1] };
}
