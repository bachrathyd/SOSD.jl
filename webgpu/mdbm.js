// Stability boundary ρ = 1 by the multi-dimensional bisection method (MDBM, Bachrathy & Stépán),
// 2-D, one function f = log ρ: evaluate f on a small initial grid, keep the cells whose corner
// values change sign (zeroth-order bracketing, as MDBM.jl interpolationorder = 0), split each into
// 4, evaluate only the new corner points, repeat. Neighbour check (MDBM.jl checkneighbour!,
// directional): a face of a kept cell whose two corners change sign is crossed by the boundary,
// so the cell behind that face must be kept too — it is added and traced further until no new
// cell appears. This recovers branches that were lost because a coarser cell did not show a sign
// change. Default: once, at the end (few points per round, almost sequential: evalNeighbour, the
// CPU pool); optionally after every iteration. Only at the end is the boundary drawn by
// first-order (linear) interpolation of f along the cell edges. Every refinement stage's new
// points are evaluated in ONE batched GPU call.

/**
 * @param evalBatch async (Float32Array xy) -> Float32Array rho
 * @returns { segments: [[x0,y0,x1,y1], ...], points: Float32Array xy, rho: Float32Array, stages: [...] }
 */
export async function mdbmBoundary(evalBatch, box, nx0, ny0, iters, opt = {}) {
  const { neighbour = 'end', cancel = () => false, onStage = null, evalNeighbour = evalBatch, init = null } = opt;
  const F = 1 << iters;                       // fine lattice units per initial cell
  const NXf = (nx0 - 1) * F, NYf = (ny0 - 1) * F;
  const key = (i, j) => i * (NYf + 1) + j;
  const X = (i) => box.x0 + (box.x1 - box.x0) * i / NXf;
  const Y = (j) => box.y0 + (box.y1 - box.y0) * j / NYf;
  const val = new Map();                      // key -> f = log ρ (+∞ for overflow)
  const stages = [];
  const allPts = [], allRho = [];

  async function evalMissing(list, ev = evalBatch) {   // list of [i, j] lattice points
    const need = [], seen = new Set();
    for (const [i, j] of list) {
      const k = key(i, j);
      if (!val.has(k) && !seen.has(k)) { seen.add(k); need.push([i, j]); }
    }
    if (!need.length) return 0;
    const xy = new Float32Array(2 * need.length);
    need.forEach(([i, j], q) => { xy[2 * q] = X(i); xy[2 * q + 1] = Y(j); });
    const rho = await ev(xy);
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

  // initial grid: evaluated here, or given (init: ρ on the nx0 × ny0 grid, row-major in y, e.g.
  // the brute-force chart)
  if (init) {
    for (let a = 0; a < nx0; a++) for (let b = 0; b < ny0; b++) {
      const r = init[b * nx0 + a];
      val.set(key(a * F, b * F), r > 0 && isFinite(r) ? Math.log(r) : (r > 0 ? 50 : -50));
    }
    stages.push({ stage: 'initial grid (given)', n: 0 });
  } else {
    const pts0 = [];
    for (let a = 0; a < nx0; a++) for (let b = 0; b < ny0; b++) pts0.push([a * F, b * F]);
    stages.push({ stage: 'initial grid', n: await evalMissing(pts0) });
  }
  let cells = [];
  for (let a = 0; a < nx0 - 1; a++) for (let b = 0; b < ny0 - 1; b++) cells.push({ i: a * F, j: b * F, s: F });
  cells = cells.filter(brackets);
  onStage?.(stages[stages.length - 1], cells.length);

  // neighbour tracing at the current cell size (cells of equal size s)
  const FACES = [[0, 3, -1, 0], [1, 2, 1, 0], [0, 1, 0, -1], [3, 2, 0, 1]];   // corner pair, direction
  async function traceNeighbours(label, ev) {
    if (!cells.length) return;
    const s = cells[0].s;
    const have = new Set(cells.map((c) => key(c.i, c.j)));
    let front = cells, added = 0, nEval = 0, rounds = 0;
    while (front.length && !cancel() && cells.length < 400000) {
      const cand = [];
      for (const c of front) {
        const v = corners(c).map(([i, j]) => f(i, j) >= 0);
        for (const [a, b, di, dj] of FACES) {
          if (v[a] === v[b]) continue;
          const i = c.i + di * s, j = c.j + dj * s;
          if (i < 0 || j < 0 || i + s > NXf || j + s > NYf) continue;
          const k = key(i, j);
          if (!have.has(k)) { have.add(k); cand.push({ i, j, s }); }
        }
      }
      if (!cand.length) break;
      nEval += await evalMissing(cand.flatMap(corners), ev);
      front = cand.filter(brackets);
      added += front.length; rounds++;
      cells = cells.concat(front);
    }
    if (nEval || added) {
      stages.push({ stage: label, n: nEval, added, rounds });
      onStage?.(stages[stages.length - 1], cells.length);
    }
  }

  for (let it = 1; it <= iters; it++) {
    if (cancel()) break;
    const s = cells.length ? cells[0].s / 2 : 1;
    const kids = [];
    for (const c of cells) for (const [di, dj] of [[0, 0], [1, 0], [0, 1], [1, 1]]) kids.push({ i: c.i + di * s, j: c.j + dj * s, s });
    const n = await evalMissing(kids.flatMap(corners));
    cells = kids.filter(brackets);
    stages.push({ stage: `iteration ${it}`, n });
    onStage?.(stages[stages.length - 1], cells.length);
    if (neighbour === 'every') await traceNeighbours(`neighbour check ${it}`, evalBatch);
  }
  if (neighbour === 'end' || (neighbour === 'every' && iters === 0)) await traceNeighbours('neighbour check', evalNeighbour);

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
