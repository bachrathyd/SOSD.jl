// The model language of the page: a linear time-periodic DDE in first-order form
//
//     ẋ(t) = A(t) x(t) + B(t) x(t − τ(t)),   A, B, τ periodic with period T
//
// written as text. Lines (`#` starts a comment):
//
//   ζ = 0:0.05 @ 0.011       parameter: range [0, 0.05], initial value 0.011 (default: midpoint)
//   z = 2                    constant (no slider)
//   ω = 2*pi*n/60            helper (may use t, parameters, constants, earlier helpers)
//   T = 2*pi                 period of the coefficients (no t)
//   τ = 2*pi                 delay (may depend on t; also `tau =`)
//   A = [0, 1; -(δ + ε*cos(t)), -a]       rows separated by `;`, entries by `,`
//   B = [0, 0; b, 0]
//
// Expressions: numbers, + − * / ^ (also ** and · and the unicode minus), unary minus,
// parentheses, functions sin cos tan asin acos atan exp log sqrt abs floor mod(a, b) min(a, b) max(a, b)
// pow(a, b) atan2(y, x) sign step(x) (1 for x ≥ 0, else 0) tanh sinh cosh, constants pi (π),
// the variable t. Names may contain Greek letters, digits, subscripts and _.
//
// Output: { params: [{name, lo, hi, value}], D, T, tau, A, B, helpers } with expression trees,
// a Float64 evaluator for the host, and WGSL source for the coefficient functions.

const FUNCS1 = new Set(['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'floor', 'sign', 'step',
                        'tanh', 'sinh', 'cosh', 'asin', 'acos', 'atan']);
const FUNCS2 = new Set(['mod', 'min', 'max', 'pow', 'atan2']);
const CONSTS = { pi: Math.PI, 'π': Math.PI };

export class ParseError extends Error {
  constructor(msg, line, col) { super(msg); this.line = line; this.col = col; }
}

// ---------------------------------------------------------------------------------------------
// tokenizer / parser (precedence climbing)
// ---------------------------------------------------------------------------------------------
const isIdStart = (c) => /[\p{L}_]/u.test(c);
const isIdChar = (c) => /[\p{L}\p{N}_₀-₉′']/u.test(c);

function tokenize(s, line) {
  const out = [];
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === ' ' || c === '\t') { i++; continue; }
    if (/[0-9.]/.test(c)) {
      const m = s.slice(i).match(/^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/);
      if (!m) throw new ParseError('bad number', line, i);
      out.push({ t: 'num', v: parseFloat(m[0]), col: i });
      i += m[0].length;
      continue;
    }
    if (isIdStart(c)) {
      let j = i + 1;
      while (j < s.length && isIdChar(s[j])) j++;
      out.push({ t: 'id', v: s.slice(i, j), col: i });
      i = j;
      continue;
    }
    if (s.startsWith('**', i)) { out.push({ t: 'op', v: '^', col: i }); i += 2; continue; }
    const map = { '·': '*', '−': '-', '×': '*' };
    const cc = map[c] || c;
    if ('+-*/^(),;[]:@='.includes(cc)) { out.push({ t: 'op', v: cc, col: i }); i++; continue; }
    throw new ParseError(`unexpected character '${c}'`, line, i);
  }
  return out;
}

class Parser {
  constructor(toks, line) { this.toks = toks; this.i = 0; this.line = line; }
  peek() { return this.toks[this.i]; }
  next() { return this.toks[this.i++]; }
  expectOp(v) {
    const t = this.next();
    if (!t || t.t !== 'op' || t.v !== v) throw new ParseError(`expected '${v}'`, this.line, t ? t.col : -1);
  }
  atEnd() { return this.i >= this.toks.length; }
  // expression := term (('+'|'-') term)*
  expr() {
    let a = this.term();
    for (;;) {
      const t = this.peek();
      if (t && t.t === 'op' && (t.v === '+' || t.v === '-')) { this.next(); a = { k: t.v, a, b: this.term() }; }
      else return a;
    }
  }
  term() {
    let a = this.unary();
    for (;;) {
      const t = this.peek();
      if (t && t.t === 'op' && (t.v === '*' || t.v === '/')) { this.next(); a = { k: t.v, a, b: this.unary() }; }
      else return a;
    }
  }
  unary() {
    const t = this.peek();
    if (t && t.t === 'op' && t.v === '-') { this.next(); return { k: 'neg', a: this.unary() }; }
    if (t && t.t === 'op' && t.v === '+') { this.next(); return this.unary(); }
    return this.power();
  }
  power() {
    const a = this.atom();
    const t = this.peek();
    if (t && t.t === 'op' && t.v === '^') { this.next(); return { k: '^', a, b: this.unary() }; }
    return a;
  }
  atom() {
    const t = this.next();
    if (!t) throw new ParseError('unexpected end of line', this.line, -1);
    if (t.t === 'num') return { k: 'num', v: t.v };
    if (t.t === 'op' && t.v === '(') { const e = this.expr(); this.expectOp(')'); return e; }
    if (t.t === 'id') {
      const n = this.peek();
      if (n && n.t === 'op' && n.v === '(') {
        this.next();
        const args = [this.expr()];
        while (this.peek() && this.peek().t === 'op' && this.peek().v === ',') { this.next(); args.push(this.expr()); }
        this.expectOp(')');
        if (FUNCS1.has(t.v) && args.length === 1) return { k: 'f1', f: t.v, a: args[0] };
        if (FUNCS2.has(t.v) && args.length === 2) return { k: 'f2', f: t.v, a: args[0], b: args[1] };
        throw new ParseError(`unknown function ${t.v}/${args.length}`, this.line, t.col);
      }
      if (t.v in CONSTS) return { k: 'num', v: CONSTS[t.v] };
      return { k: 'id', v: t.v, col: t.col };
    }
    throw new ParseError(`unexpected '${t.v}'`, this.line, t.col);
  }
  // [a, b; c, d]
  matrix() {
    this.expectOp('[');
    const rows = [[this.expr()]];
    for (;;) {
      const t = this.next();
      if (!t) throw new ParseError("missing ']'", this.line, -1);
      if (t.t === 'op' && t.v === ',') rows[rows.length - 1].push(this.expr());
      else if (t.t === 'op' && t.v === ';') rows.push([this.expr()]);
      else if (t.t === 'op' && t.v === ']') break;
      else throw new ParseError(`unexpected '${t.v}' in matrix`, this.line, t.col);
    }
    return rows;
  }
}

function idsOf(e, out = new Set()) {
  if (!e) return out;
  if (e.k === 'id') out.add(e.v);
  for (const c of [e.a, e.b]) if (c) idsOf(c, out);
  return out;
}

function evalConst(e) {
  // a constant expression (numbers and functions only)
  return evalExpr(e, (name) => { throw new Error(name); });
}

/** Float64 evaluation; `look(name)` resolves identifiers */
export function evalExpr(e, look) {
  switch (e.k) {
    case 'num': return e.v;
    case 'id': return look(e.v);
    case 'neg': return -evalExpr(e.a, look);
    case '+': return evalExpr(e.a, look) + evalExpr(e.b, look);
    case '-': return evalExpr(e.a, look) - evalExpr(e.b, look);
    case '*': return evalExpr(e.a, look) * evalExpr(e.b, look);
    case '/': return evalExpr(e.a, look) / evalExpr(e.b, look);
    case '^': return Math.pow(evalExpr(e.a, look), evalExpr(e.b, look));
    case 'f1': {
      const x = evalExpr(e.a, look);
      switch (e.f) {
        case 'step': return x >= 0 ? 1 : 0;
        case 'sign': return Math.sign(x);
        default: return Math[e.f](x);
      }
    }
    case 'f2': {
      const a = evalExpr(e.a, look), b = evalExpr(e.b, look);
      switch (e.f) {
        case 'mod': return a - b * Math.floor(a / b);
        case 'pow': return Math.pow(a, b);
        default: return Math[e.f](a, b);
      }
    }
  }
  throw new Error('bad node ' + e.k);
}

// ---------------------------------------------------------------------------------------------
// model text -> model
// ---------------------------------------------------------------------------------------------
const SPECIAL = { T: 'T', τ: 'tau', tau: 'tau', A: 'A', B: 'B', f: 'f' };

export function parseModel(text) {
  const params = [], consts = new Map(), helpers = [], warn = [];
  const model = { params, consts, helpers, T: null, tau: null, A: null, B: null, f: null, warn };
  // logical lines: comments removed, a line continues while ( or [ is open
  const lines = [], starts = [];
  let buf = '', depth = 0, first = 0;
  text.split('\n').forEach((raw, li) => {
    const s = raw.replace(/#.*$/, '');
    if (!buf.trim()) first = li;
    buf += (buf ? ' ' : '') + s;
    for (const c of s) { if (c === '(' || c === '[') depth++; else if (c === ')' || c === ']') depth--; }
    if (depth <= 0) { lines.push(buf); starts.push(first); buf = ''; depth = 0; }
  });
  if (buf.trim()) { lines.push(buf); starts.push(first); }
  lines.forEach((s, k) => {
    const li = starts[k];
    if (!s.trim()) return;
    const toks = tokenize(s, li);
    if (toks.length < 3 || toks[0].t !== 'id' || toks[1].t !== 'op' || toks[1].v !== '=')
      throw new ParseError('expected  name = ...', li, toks[0] ? toks[0].col : 0);
    const name = toks[0].v;
    const p = new Parser(toks.slice(2), li);
    const sp = SPECIAL[name];
    if (sp === 'A' || sp === 'B' || sp === 'f') {
      model[sp] = p.matrix();
      if (!p.atEnd()) throw new ParseError('unexpected text after the matrix', li, p.peek().col);
      return;
    }
    const e = p.expr();
    // range declaration  lo:hi  /  lo:step:hi  (optionally  @ value)
    if (!p.atEnd() && p.peek().t === 'op' && p.peek().v === ':') {
      p.next();
      let hi = p.expr(), step = null;
      if (!p.atEnd() && p.peek().t === 'op' && p.peek().v === ':') { p.next(); step = hi; hi = p.expr(); }
      let val = null;
      if (!p.atEnd() && p.peek().t === 'op' && p.peek().v === '@') { p.next(); val = p.expr(); }
      if (!p.atEnd()) throw new ParseError('unexpected text after the range', li, p.peek().col);
      const lo_ = evalConst(e), hi_ = evalConst(hi);
      if (sp) throw new ParseError(`${name} cannot be a parameter`, li, 0);
      params.push({ name, lo: lo_, hi: hi_, value: val ? evalConst(val) : 0.5 * (lo_ + hi_),
                    step: step ? evalConst(step) : null });
      return;
    }
    if (!p.atEnd()) throw new ParseError(`unexpected '${p.peek().v}'`, li, p.peek().col);
    if (sp) { model[sp] = e; return; }
    const ids = idsOf(e);
    if (ids.size === 0) { consts.set(name, evalConst(e)); return; }
    helpers.push({ name, e });
  });
  if (!model.A) throw new ParseError('missing  A = [ ... ]', lines.length - 1, 0);
  if (!model.B) throw new ParseError('missing  B = [ ... ]', lines.length - 1, 0);
  if (!model.T) throw new ParseError('missing  T = ...  (period of the coefficients)', lines.length - 1, 0);
  if (!model.tau) throw new ParseError('missing  τ = ...  (delay)', lines.length - 1, 0);
  const D = model.A.length;
  for (const [nm, M] of [['A', model.A], ['B', model.B]]) {
    if (M.length !== D || M.some((r) => r.length !== D))
      throw new ParseError(`${nm} must be ${D}×${D} (square, same size as A)`, lines.length - 1, 0);
  }
  if (D > 6) throw new ParseError('at most 6 states in the browser version', lines.length - 1, 0);
  if (model.f) {      // forcing: a column [f1; f2; ...] or a row [f1, f2, ...] of D entries
    const v = model.f.length === 1 ? model.f[0] : model.f.map((r) => r[0]);
    if (v.length !== D || (model.f.length > 1 && model.f.some((r) => r.length !== 1)))
      throw new ParseError(`f must have ${D} entries (the forcing of each state equation)`, lines.length - 1, 0);
    model.f = v;
  }
  model.D = D;
  // identifiers must be known
  const known = new Set(['t', ...params.map((q) => q.name), ...consts.keys()]);
  for (const h of helpers) {
    for (const id of idsOf(h.e)) if (!known.has(id)) throw new ParseError(`unknown name ${id} in ${h.name}`, 0, 0);
    known.add(h.name);
  }
  const all = [model.T, model.tau, ...model.A.flat(), ...model.B.flat(), ...(model.f || [])];
  for (const e of all) for (const id of idsOf(e)) if (!known.has(id)) throw new ParseError(`unknown name ${id}`, 0, 0);
  const tdep = (e) => {
    const seen = new Set();
    const rec = (x) => { for (const id of idsOf(x)) { if (id === 't') return true;
      const h = helpers.find((q) => q.name === id); if (h && !seen.has(id)) { seen.add(id); if (rec(h.e)) return true; } } return false; };
    return rec(e);
  };
  if (tdep(model.T)) throw new ParseError('T must not depend on t', 0, 0);
  if (params.length > 16) throw new ParseError('at most 16 parameters', 0, 0);
  return model;
}

/** Float64 evaluator: returns {T(P), tau(t, P), A(t, P), B(t, P)} with P a parameter array */
export function hostEvaluator(model) {
  const pidx = new Map(model.params.map((q, i) => [q.name, i]));
  const env = (t, P) => {
    const vals = new Map();
    const look = (n) => {
      if (n === 't') return t;
      if (pidx.has(n)) return P[pidx.get(n)];
      if (model.consts.has(n)) return model.consts.get(n);
      if (vals.has(n)) return vals.get(n);
      throw new Error('unknown ' + n);
    };
    for (const h of model.helpers) {
      try { vals.set(h.name, evalExpr(h.e, look)); } catch (e) { /* t-dependent helper with t undefined */ }
    }
    return look;
  };
  return {
    T: (P) => evalExpr(model.T, env(0, P)),
    tau: (t, P) => evalExpr(model.tau, env(t, P)),
    A: (t, P) => { const l = env(t, P); return model.A.map((r) => r.map((e) => evalExpr(e, l))); },
    B: (t, P) => { const l = env(t, P); return model.B.map((r) => r.map((e) => evalExpr(e, l))); },
  };
}

// ---------------------------------------------------------------------------------------------
// WGSL generation
// ---------------------------------------------------------------------------------------------
const f32 = (v) => {
  if (!isFinite(v)) throw new Error('non-finite constant');
  let s = v.toPrecision(9);
  if (!/[.eE]/.test(s)) s += '.0';
  return s;
};

function wgsl(e, name) {
  switch (e.k) {
    case 'num': return f32(e.v);
    case 'id': return name(e.v);
    case 'neg': return `(-${wgsl(e.a, name)})`;
    case '+': case '-': case '*': case '/': return `(${wgsl(e.a, name)} ${e.k} ${wgsl(e.b, name)})`;
    case '^': {
      if (e.b.k === 'num' && Number.isInteger(e.b.v) && Math.abs(e.b.v) <= 4) {
        const n = Math.abs(e.b.v), x = `xp_${Math.random().toString(36).slice(2, 7)}`;
        const base = wgsl(e.a, name);
        if (n === 0) return '1.0';
        const prod = Array(n).fill(`(${base})`).join(' * ');
        return e.b.v < 0 ? `(1.0 / (${prod}))` : `(${prod})`;
      }
      return `pow(${wgsl(e.a, name)}, ${wgsl(e.b, name)})`;
    }
    case 'f1': {
      const a = wgsl(e.a, name);
      switch (e.f) {
        case 'step': return `select(0.0, 1.0, ${a} >= 0.0)`;
        case 'log': return `log(${a})`;
        default: return `${e.f}(${a})`;
      }
    }
    case 'f2': {
      const a = wgsl(e.a, name), b = wgsl(e.b, name);
      switch (e.f) {
        case 'mod': return `fmod_(${a}, ${b})`;
        default: return `${e.f}(${a}, ${b})`;
      }
    }
  }
  throw new Error('bad node');
}

/**
 * WGSL for the coefficient functions of `model`:
 *   fn m_period(P) -> f32, fn m_tau(t, P) -> f32,
 *   fn m_AB(t, P, A: ptr<function, array<f32, DD>>, B: ...)   (row-major, A[i*D + j])
 */
export function modelWGSL(model, forced = false) {
  const D = model.D, NP = Math.max(1, model.params.length);
  const pidx = new Map(model.params.map((q, i) => [q.name, i]));
  const hname = (n) => 'h_' + [...n].map((c) => (/[A-Za-z0-9]/.test(c) ? c : 'u' + c.codePointAt(0))).join('');
  const name = (n) => {
    if (n === 't') return 't';
    if (pidx.has(n)) return `P[${pidx.get(n)}]`;
    if (model.consts.has(n)) return f32(model.consts.get(n));
    return hname(n);
  };
  const lets = (needT) => model.helpers
    .filter((h) => needT || !usesT(h, model))
    .map((h) => `  let ${hname(h.name)} = ${wgsl(h.e, name)};`).join('\n');
  const tparam = 't: f32, P: array<f32, NP>';
  let s = `const D: u32 = ${D}u;\nconst NP: u32 = ${NP}u;\n`;
  s += `fn fmod_(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }\n`;
  s += `fn m_period(P: array<f32, NP>) -> f32 {\n  let t = 0.0;\n${lets(false)}\n  return ${wgsl(model.T, name)};\n}\n`;
  s += `fn m_tau(${tparam}) -> f32 {\n${lets(true)}\n  return ${wgsl(model.tau, name)};\n}\n`;
  s += `fn m_AB(${tparam}, A: ptr<function, array<f32, ${D * D}>>, B: ptr<function, array<f32, ${D * D}>>) {\n${lets(true)}\n`;
  for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) {
    s += `  (*A)[${i * D + j}] = ${wgsl(model.A[i][j], name)};\n`;
    s += `  (*B)[${i * D + j}] = ${wgsl(model.B[i][j], name)};\n`;
  }
  s += '}\n';
  // forcing f(t) (the periodic-orbit option); FORCED switches the forced kernels on
  s += `const FORCED: bool = ${forced && model.f ? 'true' : 'false'};\n`;
  // the forcing part of the stage values lives in binding 7, declared only when it is used
  s += forced && model.f
    ? '@group(0) @binding(7) var<storage, read_write> Fb: array<f32>;\nfn fbW(i: u32, v: f32) { Fb[i] = v; }\nfn fbR(i: u32) -> f32 { return Fb[i]; }\n'
    : 'fn fbW(i: u32, v: f32) { }\nfn fbR(i: u32) -> f32 { return 0.0; }\n';
  s += `fn m_F(${tparam}, F: ptr<function, array<f32, ${D}>>) {\n${lets(true)}\n`;
  for (let i = 0; i < D; i++) s += `  (*F)[${i}] = ${model.f ? wgsl(model.f[i], name) : '0.0'};\n`;
  s += '}\n';
  return s;
}

function usesT(h, model) {
  const seen = new Set();
  const rec = (e) => {
    for (const id of idsOf(e)) {
      if (id === 't') return true;
      const g = model.helpers.find((q) => q.name === id);
      if (g && !seen.has(id)) { seen.add(id); if (rec(g.e)) return true; }
    }
    return false;
  };
  return rec(h.e);
}

// ---------------------------------------------------------------------------------------------
// JavaScript generation (Float64 CPU path: web workers of cpu.js)
// ---------------------------------------------------------------------------------------------
function js(e, name) {
  switch (e.k) {
    case 'num': return String(e.v);
    case 'id': return name(e.v);
    case 'neg': return `(-${js(e.a, name)})`;
    case '+': case '-': case '*': case '/': return `(${js(e.a, name)} ${e.k} ${js(e.b, name)})`;
    case '^': return `Math.pow(${js(e.a, name)}, ${js(e.b, name)})`;
    case 'f1': {
      const a = js(e.a, name);
      return e.f === 'step' ? `((${a}) >= 0 ? 1 : 0)` : `Math.${e.f}(${a})`;
    }
    case 'f2': {
      const a = js(e.a, name), b = js(e.b, name);
      return e.f === 'mod' ? `fmod(${a}, ${b})` : `Math.${e.f}(${a}, ${b})`;
    }
  }
  throw new Error('bad node');
}

/**
 * Source of a function body returning { T(P), tau(t, P), AB(t, P, A, B) } (A, B row-major
 * Float64Array D*D), for `new Function(src)()`.
 */
export function modelJS(model) {
  const D = model.D;
  const pidx = new Map(model.params.map((q, i) => [q.name, i]));
  const hname = (n) => 'h_' + [...n].map((c) => (/[A-Za-z0-9]/.test(c) ? c : 'u' + c.codePointAt(0))).join('');
  const name = (n) => {
    if (n === 't') return 't';
    if (pidx.has(n)) return `P[${pidx.get(n)}]`;
    if (model.consts.has(n)) return `(${model.consts.get(n)})`;
    return hname(n);
  };
  const lets = (needT) => model.helpers
    .filter((h) => needT || !usesT(h, model))
    .map((h) => `  const ${hname(h.name)} = ${js(h.e, name)};`).join('\n');
  let s = `'use strict';\nconst fmod = (a, b) => a - b * Math.floor(a / b);\n`;
  s += `const T = (P) => {\n  const t = 0;\n${lets(false)}\n  return ${js(model.T, name)};\n};\n`;
  s += `const tau = (t, P) => {\n${lets(true)}\n  return ${js(model.tau, name)};\n};\n`;
  s += `const AB = (t, P, A, B) => {\n${lets(true)}\n`;
  for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) {
    s += `  A[${i * D + j}] = ${js(model.A[i][j], name)};\n  B[${i * D + j}] = ${js(model.B[i][j], name)};\n`;
  }
  s += '};\nreturn { T, tau, AB };\n';
  return s;
}
