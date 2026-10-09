# SOSD stability charts on the Colab GPU, streamed as images (the pattern of the InterpolatedNyquist
# T4 app). The Julia worker (gpu/stream/worker.jl) computes the chart level by level on the GPU and
# colours it; this kernel encodes every level as a JPEG and streams it to the page over one
# long-lived response, and the page only draws (axes, colour bar, zoom / pan). The page has the
# model text, examples and parameters of the WebGPU page (it imports webgpu/expr.js and
# examples.js), so a model is written the same way.
#
# In Colab (after the setup cell of colab/SOSD_Stream_Colab.ipynb):
#   exec(open('/content/SOSD.jl/gpu/stream/app.py').read())
# Local test without CUDA:
#   SOSD_DIR=<repo> SOSD_WORKER_ARGS=--cpu python gpu/stream/app.py      -> http://localhost:8790/
import os, io, json, math, time, struct, threading, subprocess, tempfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from PIL import Image as PImage

REPO = os.environ.get('SOSD_DIR', '/content/SOSD.jl')
WORKER_ARGS = os.environ.get('SOSD_WORKER_ARGS', '').split()
PROJECT = os.environ.get('SOSD_PROJECT', 'gpu/webui')       # the Julia environment of the worker
PORT = int(os.environ.get('SOSD_PORT', globals().get('PORT', 8790)))
_TMP = '/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir()
IMG = os.path.join(_TMP, 'sosd_stream.rgb')
IMG_FULL = os.path.join(_TMP, 'sosd_full.rgb')
OUT_DIR = '/content' if os.path.isdir('/content') else _TMP
LOG = os.path.join(OUT_DIR, 'sosd_stream_worker.log')
STATIC = os.path.join(REPO, 'webgpu')
os.environ['PATH'] = '/root/.juliaup/bin' + os.pathsep + os.environ['PATH']


class Worker:
    "The Julia GPU worker: one JSON request per line on stdin, one JSON answer line on stdout."
    def __init__(self):
        self.lock = threading.Lock()
        self.log = open(LOG, 'w')
        self.p = subprocess.Popen(['julia', '-t', 'auto', '--project=' + PROJECT, 'gpu/stream/worker.jl'] + WORKER_ARGS,
                                  cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                  text=True, encoding='utf-8', bufsize=1)
        t0 = time.time()
        print('starting the Julia GPU worker (the first start compiles the kernels, a few minutes) ...', flush=True)
        self.info = self._read()
        print(f"worker ready after {time.time() - t0:.0f} s: {self.info.get('device')}", flush=True)

    def _read(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('the Julia worker exited; see ' + LOG)
            if line.startswith('{'):
                return json.loads(line)

    def alive(self):
        return self.p.poll() is None

    def ask(self, req):
        with self.lock:
            self.p.stdin.write(json.dumps(req) + '\n')
            self.p.stdin.flush()
            r = self._read()
        if 'error' in r:
            raise RuntimeError(r['error'])
        return r


VIEW = ('code', 'values', 'xi', 'yi', 'x0', 'x1', 'y0', 'y1', 'W', 'H', 'S', 'p', 'm', 'prec', 'forced', 'cr', 'bnd')
RATE = {}                    # (model, S, p, m, precision, forced) -> nodes per ms (GPU kernel)
STRIDES = [1, 2, 4, 8, 16, 32, 64]
OVERHEAD_MS = 35             # colouring, encoding and transfer of one frame (coarse levels)


def _ratekey(q):
    return (hash(q['code']), q['S'], q['p'], q['m'], q['prec'], q['forced'])


def _nodes(W, H, s):
    "new nodes of the stride-s level (after the stride-2s level)"
    n = math.ceil(W / s) * math.ceil(H / s)
    return n - (math.ceil(W / (2 * s)) * math.ceil(H / (2 * s)) if s < 64 else 0)


def _encode(info, q):
    a = PImage.frombytes('RGB', (info['iw'], info['ih']), open(IMG, 'rb').read(3 * info['iw'] * info['ih']))
    dw = int(q.get('dispW', 0) or 0)
    if dw and a.width > 1.25 * dw:              # the final levels of a grid finer than the screen
        a = a.resize((dw, max(1, round(a.height * dw / a.width))), PImage.BOX)
    buf = io.BytesIO()
    if q.get('enc') == 'png':
        a.save(buf, 'PNG', compress_level=1)
        return buf.getvalue(), 'image/png'
    a.save(buf, 'JPEG', quality=90, subsampling=0)
    return buf.getvalue(), 'image/jpeg'


class _Stream:
    "The newest state posted by the page (n counts the accepted states), Stop, models to compile while idle."
    def __init__(self):
        self.cv = threading.Condition()
        self.q, self.n, self.sid, self.seq, self.gen, self.stop, self.warm = None, 0, None, 0, 0, 0, []


def _refine(st, q, send):
    "the progressive levels of state q; returns early when a newer state arrives or on Stop"
    W, H, seq, me = q['W'], q['H'], q['seq'], q['_n']
    key = _ratekey(q)
    budget = 1000 / max(1, float(q.get('fps', 15))) - OVERHEAD_MS
    rate = RATE.get(key)
    s0 = 16
    if rate:
        s0 = next((s for s in STRIDES if math.ceil(W / s) * math.ceil(H / s) / rate <= budget), 64)
    tlimit = float(q.get('tlimit', 60)) * 1e3
    t0 = time.time()
    req = {k: q[k] for k in VIEW}
    req.update(cmd='level', img=IMG)
    s, first = s0, True
    while True:
        nys, nxs = math.ceil(H / s), math.ceil(W / s)
        j = 0
        while j < nys:
            with st.cv:
                newer, stopped = st.n > me, st.stop >= me
            if newer:
                return
            el = (time.time() - t0) * 1e3
            if stopped or el > tlimit:
                send({'stopped': True, 'timeout': not stopped, 'seq': seq, 's': s, 'elapsed': el}, b'', None)
                return
            rate = RATE.get(key)
            # the first level in one piece, the others in pieces of ~0.3 s (a new state waits at most that)
            per_row = nxs if (first or s == 1 and q['prec'] == 'mixed') else max(1, nxs * 3 // 4)
            rows = nys - j if first or not rate else max(1, int(rate * 300 / per_row))
            req.update(s=s, j0=j, j1=min(nys, j + rows))
            info = st_worker.ask(req)
            if info['n'] >= 256 and info['t_kernel'] > 0:
                # a first call may include compiling the kernel: a much faster rate replaces it
                r = info['n'] / info['t_kernel']
                RATE[key] = r if key not in RATE or r > 3 * RATE[key] else 0.5 * RATE[key] + 0.5 * r
            img, mime = _encode(info, q)
            el = (time.time() - t0) * 1e3
            rate = RATE.get(key)
            if info['cached']:                  # this view is already computed down to info['done']
                s = info['s']
                j = nys = math.ceil(H / s)
            else:
                j = info['rows']
            left = (nys - j) * nxs * (0.75 if s < s0 else 1) + sum(_nodes(W, H, t) for t in STRIDES if t < s)
            info.update(seq=seq, mime=mime, kb=len(img) / 1024, elapsed=el, s0=s0,
                        eta=left / rate if rate else None, us=1e3 / rate if rate else None,
                        box=[q['x0'], q['x1'], q['y0'], q['y1']], W=W, H=H, final=(s == 1 and j >= nys))
            send(info, img, mime)
            first = False
        if s == 1:
            return
        s //= 2


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ct='application/json', code=200):
        self.send_response(code)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0].strip('/')
        if path == '':
            return self._send(_PAGE.encode(), 'text/html; charset=utf-8')
        if path == 'info':
            return self._send(json.dumps(st_worker.info).encode())
        if path == 'stream':
            return self._stream()
        if path.startswith('webgpu/') and path.endswith('.js') and '..' not in path:
            f = os.path.join(REPO, *path.split('/'))
            if os.path.isfile(f):
                return self._send(open(f, 'rb').read(), 'text/javascript; charset=utf-8')
        self._send(b'not found', 'text/plain', 404)

    def do_POST(self):
        path = self.path.split('?')[0].strip('/')
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        if path == 'savefile':                    # the composed PNG (with axes) from the page
            name = os.path.basename(self.headers.get('X-Name', 'sosd_chart.png'))
            fn = os.path.join(OUT_DIR, name)
            open(fn, 'wb').write(body)
            return self._send(json.dumps({'file': fn, 'mb': len(body) / 1e6}).encode())
        q = json.loads(body or b'{}')
        st = _ST
        if path == 'state':                       # newest state wins (posts may overtake each other)
            with st.cv:                           # a reloaded page (new sid) starts its seq again
                if q.get('sid') != st.sid or q.get('seq', 0) > st.seq:
                    st.n += 1
                    q['_n'] = st.n
                    st.q, st.seq, st.sid = q, q.get('seq', 0), q.get('sid')
                    st.cv.notify_all()
            return self._send(b'{}')
        if path == 'stop':
            with st.cv:
                st.stop = st.n
            return self._send(b'{}')
        if path == 'warm':
            with st.cv:
                st.warm.extend(q.get('list', []))
                st.cv.notify_all()
            return self._send(b'{}')
        if path == 'full':                        # the whole chart (one pixel per grid point) as PNG
            try:
                info = st_worker.ask(dict(cmd='image', s=1, img=IMG_FULL, cr=q.get('cr', 0.5), bnd=q.get('bnd', True)))
                a = PImage.frombytes('RGB', (info['iw'], info['ih']), open(IMG_FULL, 'rb').read(3 * info['iw'] * info['ih']))
                buf = io.BytesIO()
                a.save(buf, 'PNG', compress_level=3)
                return self._send(buf.getvalue(), 'image/png')
            except Exception as err:
                return self._send(str(err).encode(), 'text/plain', 500)
        self._send(b'not found', 'text/plain', 404)

    def _stream(self):
        # frames: [header length][image length] (big-endian uint32), header JSON, image bytes
        st = _ST
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Accel-Buffering', 'no')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        with st.cv:
            st.gen += 1
            gen = st.gen
            st.cv.notify_all()                    # an older stream (reloaded page) stops

        def send(info, img, mime):
            h = json.dumps(info).encode()
            self.wfile.write(struct.pack('>II', len(h), len(img)) + h + img)
            self.wfile.flush()

        done = 0
        try:
            while True:
                with st.cv:
                    st.cv.wait_for(lambda: st.gen != gen or st.n > done or st.warm, timeout=10)
                    if st.gen != gen or st is not _ST:
                        return
                    q = w = None
                    if st.n > done:
                        q, done = st.q, st.n
                    elif st.warm:
                        w = st.warm.pop(0)
                if w is not None:                 # compile another example's kernels while idle
                    try:
                        r = st_worker.ask(dict(w, cmd='warm'))
                        send({'warm': w.get('name', ''), 't': r['t']}, b'', None)
                    except Exception as err:
                        send({'warm': w.get('name', ''), 'error': str(err)}, b'', None)
                    continue
                if q is None:                     # keep-alive for the proxy
                    send({}, b'', None)
                    continue
                try:
                    _refine(st, q, send)
                except (BrokenPipeError, ConnectionResetError):
                    raise
                except Exception as err:
                    send({'error': str(err), 'seq': q.get('seq', 0)}, b'', None)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


_PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>SOSD stability charts (Colab GPU)</title>
<style>
 body {margin: 6px 8px; font: 13px system-ui, sans-serif; background: #fff; color: #222}
 .row {display: flex; flex-wrap: wrap; gap: 5px 16px; align-items: center; margin: 4px 0}
 label {display: inline-flex; align-items: center; gap: 5px}
 select, input[type=number] {font: inherit; padding: 1px 3px}
 input[type=number] {width: 78px}
 input.n {width: 56px}
 input[type=range] {width: 150px}
 button {font: inherit; padding: 2px 9px; cursor: pointer}
 .val {display: inline-block; min-width: 46px; font-variant-numeric: tabular-nums}
 #st, #st2 {font: 12px ui-monospace, monospace; margin: 3px 0; min-height: 1.3em; white-space: pre-wrap}
 #err {color: #c22; font: 12px ui-monospace, monospace; white-space: pre-wrap}
 textarea {width: 100%; max-width: 1100px; font: 12px ui-monospace, monospace}
 #cv {display: block; touch-action: none; cursor: grab}
 h1 {font-size: 15px; margin: 2px 0 6px}
 #formula, #note {opacity: .8}
</style></head><body>
<h1>SOSD stability charts on the Colab GPU <span id="dev" style="font-weight: 400; opacity: .7"></span></h1>
<div class="row">
 <label>example <select id="ex"></select></label> <span id="formula"></span>
</div>
<details id="mdl"><summary>model text (the language of the WebGPU page; edits apply after a short pause)</summary>
 <textarea id="txt" rows="16" spellcheck="false"></textarea><div id="err"></div></details>
<div class="row"><label>x axis <select id="ax"></select></label><label>y axis <select id="ay"></select></label>
 <span id="knobs" class="row" style="margin: 0"></span></div>
<div class="row">
 <label>Gauss–Legendre stages s <input class="n" type="number" id="S" min="1" max="8" step="1"></label>
 <label>steps p <input class="n" type="number" id="p" min="2" max="2000" step="1"></label>
 <label>Krylov m <input class="n" type="number" id="m" min="3" max="24" step="1"></label>
 <label>precision <select id="prec"><option value="F32">Float32</option><option value="mixed">Float32, last level Float64</option><option value="F64">Float64</option></select></label>
 <label><input type="checkbox" id="forced"> forced response (periodic orbit)</label>
</div>
<div class="row">
 <label>final grid <select id="grid"><option value="plot">plot area (one ρ per screen pixel)</option>
  <option value="1280x720">HD 1280×720</option><option value="1920x1080">Full HD 1920×1080</option>
  <option value="3840x2160">4K 3840×2160</option><option value="7680x4320">8K 7680×4320</option></select></label>
 <label>while moving <input class="n" type="number" id="fps" min="1" max="60" value="15"> fps</label>
 <label>time limit <input class="n" type="number" id="tl" min="1" value="60"> s</label>
 <label>colour range <select id="cr"><option value="0.25">ρ ∈ [0.56, 1.8]</option><option value="0.5" selected>ρ ∈ [0.32, 3.2]</option><option value="1">ρ ∈ [0.1, 10]</option></select></label>
 <label><input type="checkbox" id="bnd" checked> boundary</label>
 <label>image <select id="enc"><option value="jpeg">JPEG</option><option value="png">PNG</option></select></label>
</div>
<div class="row">
 <label>x <input type="number" id="x0" step="any"> … <input type="number" id="x1" step="any"></label>
 <label>y <input type="number" id="y0" step="any"> … <input type="number" id="y1" step="any"></label>
 <button id="rst">reset view</button><button id="stop">stop</button><button id="save">save PNG (final grid, with axes)</button>
 <span style="opacity: .7">wheel: zoom, drag: pan</span>
</div>
<canvas id="cv"></canvas>
<div id="st">connecting…</div><div id="st2"></div><div id="note"></div>
<script type="module">
import { parseModel, modelJulia } from './webgpu/expr.js';
import { EXAMPLES } from './webgpu/examples.js';
const $ = (id) => document.getElementById(id);
const PAD = {l: 76, r: 100, t: 30, b: 50};
const STOPS = [[5,48,97],[33,102,172],[67,147,195],[146,197,222],[247,247,247],[244,165,130],[214,96,77],[178,24,43],[103,0,31]];
const GREENS = [[247,252,245],[229,245,224],[199,233,192],[161,217,155],[116,196,118],[65,171,93],[35,139,69],[0,109,44],[0,68,27]];
const SID = Math.random().toString(36).slice(2);
let ex = null, model = null, code = '', text = '', values = [], axes = [0, 1], view = null, frame = null, seq = 0, info = {};

const ramp = (S, t) => { t = Math.min(1, Math.max(0, t)) * (S.length - 1); const i = Math.min(S.length - 2, Math.floor(t)), w = t - i;
  return `rgb(${S[i].map((c, k) => Math.round(c + w * (S[i + 1][k] - c)))})`; };
function axisLabel(name) {
  const line = text.split('\n').find((l) => l.replace(/\s/g, '').startsWith(name + '=') && l.includes('#'));
  const c = line ? line.split('#')[1].trim() : '';
  return c ? `${name}: ${c}` : name;
}
function stateName() { return (ex && ex.states && ex.states[0]) || 'x₁'; }
function nice(a, b, n) {
  const span = (b - a) / n, mag = Math.pow(10, Math.floor(Math.log10(span)));
  const s = [1, 2, 2.5, 5, 10].map((x) => x * mag).find((x) => x >= span), out = [];
  for (let v = Math.ceil(a / s) * s; v <= b + 1e-9 * s; v += s) out.push(+v.toPrecision(10));
  return out;
}

// --- geometry: the canvas fills the width; the final grid is the plot area or a preset ---
const cv = $('cv'), ctx = cv.getContext('2d');
function plotSize() { const w = Math.max(320, Math.min(window.innerWidth - 24, 1500)); return [w - PAD.l - PAD.r, Math.round(0.5 * (w - PAD.l - PAD.r))]; }
function grid() {
  if ($('grid').value !== 'plot') return $('grid').value.split('x').map(Number);
  const [w, h] = plotSize(); return [Math.round(w), Math.round(h)];
}
function layout() {
  const [w, h] = plotSize(), dpr = window.devicePixelRatio || 1, cw = w + PAD.l + PAD.r, ch = h + PAD.t + PAD.b;
  if (cv.width !== Math.round(cw * dpr)) { cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr); cv.style.width = cw + 'px'; cv.style.height = ch + 'px'; }
  return {w, h, dpr};
}

// --- the chart: the newest frame mapped into the current view, axes, title, colour bar ---
function drawChart(c, sc, w, h, fr, v) {
  const L = PAD.l * sc, T = PAD.t * sc, W = w * sc, H = h * sc, ink = '#222';
  c.fillStyle = '#fff'; c.fillRect(0, 0, (w + PAD.l + PAD.r) * sc, (h + PAD.t + PAD.b) * sc);
  c.fillStyle = '#f2f2f2'; c.fillRect(L, T, W, H);
  if (fr) {
    // frame pixel i covers grid px [i·s + ½ − s/2, …) of a W×H grid over fr.box
    const [bx0, bx1, by0, by1] = fr.info.box, s = fr.info.s, G = fr.info.W, GH = fr.info.H;
    const gx = (g) => bx0 + g / G * (bx1 - bx0), gy = (g) => by1 - g / GH * (by1 - by0);
    const X = (x) => L + (x - v.x0) / (v.x1 - v.x0) * W, Y = (y) => T + (v.y1 - y) / (v.y1 - v.y0) * H;
    const e0 = 0.5 - s / 2, e1 = e0 + fr.info.iw * s, f1 = e0 + fr.info.ih * s;
    c.save(); c.beginPath(); c.rect(L, T, W, H); c.clip();
    c.imageSmoothingEnabled = s === 1 && fr.bmp.width < fr.info.iw;
    c.drawImage(fr.bmp, X(gx(e0)), Y(gy(e0)), X(gx(e1)) - X(gx(e0)), Y(gy(f1)) - Y(gy(e0)));
    c.restore();
  }
  c.strokeStyle = '#555'; c.lineWidth = sc; c.strokeRect(L, T, W, H);
  c.fillStyle = ink; c.font = `${12 * sc}px system-ui, sans-serif`; c.textAlign = 'center';
  for (const x of nice(v.x0, v.x1, 8)) { const px = L + (x - v.x0) / (v.x1 - v.x0) * W;
    c.fillRect(px, T + H, sc, 5 * sc); c.fillText(+x.toPrecision(6), px, T + H + 18 * sc); }
  c.fillText(axisLabel(model.params[axes[0]].name), L + W / 2, T + H + 40 * sc);
  c.textAlign = 'right';
  for (const y of nice(v.y0, v.y1, 6)) { const py = T + H - (y - v.y0) / (v.y1 - v.y0) * H;
    c.fillRect(L - 5 * sc, py, 5 * sc, sc); c.fillText(+y.toPrecision(6), L - 8 * sc, py + 4 * sc); }
  c.save(); c.translate(18 * sc, T + H / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center';
  c.fillText(axisLabel(model.params[axes[1]].name), 0, 0); c.restore();
  const forced = fr && fr.info.forced, CR = +$('cr').value;
  c.textAlign = 'left'; c.font = `600 ${13 * sc}px system-ui, sans-serif`;
  c.fillText(forced ? `stable (green): peak-to-peak of ${stateName()} on the periodic orbit  ·  unstable (red): log₁₀ ρ`
                    : 'colour: log₁₀ ρ, spectral radius of the monodromy operator (ρ < 1 stable, blue)', L, T - 10 * sc);
  c.font = `${12 * sc}px system-ui, sans-serif`;
  const bx = L + W + 16 * sc, bw = 16 * sc;
  if (forced) {
    const alo = fr.info.alo || 1, ahi = fr.info.ahi || 10, fa = (x) => +x.toPrecision(2);
    for (let k = 0; k < 128; k++) { c.fillStyle = ramp(STOPS, 0.5 + 0.5 * (1 - k / 127)); c.fillRect(bx, T + k / 128 * H / 2, bw, H / 256 + sc); }
    for (let k = 0; k < 128; k++) { c.fillStyle = ramp(GREENS, 1 - k / 127); c.fillRect(bx, T + H / 2 + k / 128 * H / 2, bw, H / 256 + sc); }
    c.strokeStyle = '#555'; c.strokeRect(bx, T, bw, H); c.fillStyle = ink;
    for (const v2 of [CR, CR / 2]) c.fillText(v2.toFixed(2), bx + bw + 4 * sc, T + (1 - v2 / CR) / 2 * H / 2 + 4 * sc);
    c.fillText('ρ = 1', bx + bw + 4 * sc, T + H / 2 + 4 * sc);
    c.fillText(fa(ahi) + '+', bx + bw + 4 * sc, T + H / 2 + 18 * sc);
    c.fillText(fa(Math.sqrt(alo * ahi)), bx + bw + 4 * sc, T + 0.75 * H + 4 * sc);
    c.fillText(fa(alo) + '−', bx + bw + 4 * sc, T + H + 4 * sc);
    c.save(); c.translate(bx + bw + 56 * sc, T + H / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center';
    c.fillText(`peak-to-peak ${stateName()} (log)   |   log₁₀ ρ`, 0, 0); c.restore();
  } else {
    for (let k = 0; k < 256; k++) { c.fillStyle = ramp(STOPS, 1 - k / 255); c.fillRect(bx, T + k / 256 * H, bw, H / 256 + sc); }
    c.strokeStyle = '#555'; c.strokeRect(bx, T, bw, H); c.fillStyle = ink;
    for (const v2 of [-CR, -CR / 2, 0, CR / 2, CR]) c.fillText(v2.toFixed(2), bx + bw + 4 * sc, T + (1 - v2 / CR) / 2 * H + 4 * sc);
    c.save(); c.translate(bx + bw + 50 * sc, T + H / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center';
    c.fillText('log₁₀ ρ   (ρ < 1 stable, blue)', 0, 0); c.restore();
  }
}
function redraw() {
  if (!model || !view) return;
  const {w, h, dpr} = layout();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawChart(ctx, 1, w, h, frame, view);
}

// --- model, parameters, view ---
function setText(t) {
  let m;
  try { m = parseModel(t); } catch (err) { $('err').textContent = String(err.message || err); return false; }
  $('err').textContent = (m.warn || []).join('\n');
  const old = model ? new Map(model.params.map((q, i) => [q.name, values[i]])) : new Map();
  const oldAxes = model ? axes.map((i) => model.params[i].name) : null;
  model = m; text = t; code = modelJulia(m);
  values = m.params.map((q) => old.has(q.name) ? old.get(q.name) : q.value);
  const idx = (n) => m.params.findIndex((q) => q.name === n);
  const want = oldAxes || (ex ? ex.axes : []);
  axes = [Math.max(0, idx(want[0])), idx(want[1]) >= 0 ? idx(want[1]) : Math.min(1, m.params.length - 1)];
  for (const id of ['ax', 'ay']) { $(id).innerHTML = ''; m.params.forEach((q, i) => $(id).add(new Option(q.name, i))); }
  $('ax').value = axes[0]; $('ay').value = axes[1];
  $('forced').disabled = !m.f;
  buildKnobs();
  if (!oldAxes || !view) resetView(false);
  return true;
}
function buildKnobs() {
  $('knobs').innerHTML = '';
  model.params.forEach((q, i) => {
    if (axes.includes(i)) return;
    const lab = document.createElement('label'), r = document.createElement('input'), v = document.createElement('span');
    Object.assign(r, {type: 'range', min: q.lo, max: q.hi, step: q.step || (q.hi - q.lo) / 1000});
    r.value = values[i]; v.className = 'val'; v.textContent = +(+values[i]).toPrecision(4);
    r.oninput = () => { values[i] = +r.value; v.textContent = +(+r.value).toPrecision(4); go(); };
    lab.append(q.name + ' ', r, v); $('knobs').append(lab);
  });
}
function showView() { for (const k of ['x0', 'x1', 'y0', 'y1']) $(k).value = +view[k].toPrecision(8); }
function resetView(send = true) {
  const a = model.params[axes[0]], b = model.params[axes[1]];
  view = {x0: a.lo, x1: a.hi, y0: b.lo, y1: b.hi}; showView(); if (send) go();
}
function setExample(key) {
  ex = EXAMPLES.find((e) => e.key === key);
  model = null; view = null; frame = null;
  $('txt').value = ex.text; $('formula').innerHTML = ex.formula || ''; $('note').textContent = ex.note || '';
  $('S').value = ex.S; $('p').value = ex.p; $('m').value = ex.m; $('forced').checked = !!ex.forced;
  setText(ex.text); go();
}

// --- the stream: post the state, frames come back on one long-lived response ---
function state() {
  const [W, H] = grid(), [w] = plotSize();
  return {code, values, xi: axes[0], yi: axes[1], ...view, W, H, S: +$('S').value, p: +$('p').value, m: +$('m').value,
          prec: $('prec').value, forced: $('forced').checked && !!model.f, cr: +$('cr').value, bnd: $('bnd').checked,
          fps: +$('fps').value || 15, tlimit: +$('tl').value || 60, enc: $('enc').value, dispW: Math.round(w * (window.devicePixelRatio || 1))};
}
let posting = 0, pdirty = false, slow = null;
const sentAt = new Map();
function go() {
  if (!model || !view) return;
  redraw();
  if (posting >= 3) { pdirty = true; return; }
  const q = state(); q.seq = ++seq; q.sid = SID; sentAt.set(q.seq, performance.now());
  posting++;
  clearTimeout(slow);
  slow = setTimeout(() => { $('st').textContent = 'computing… (a new model, s, m or precision compiles its GPU kernel first: 10–60 s)'; }, 1500);
  fetch('state', {method: 'POST', body: JSON.stringify(q)}).catch(() => {})
    .finally(() => { posting--; if (pdirty) { pdirty = false; go(); } });
}
const fmtT = (ms) => ms == null ? '?' : ms < 1000 ? ms.toFixed(0) + ' ms' : (ms / 1000).toFixed(1) + ' s';
async function onFrame(r, img) {
  if (r.warm !== undefined) { $('st2').textContent = r.error ? `compiling ${r.warm} failed: ${r.error}` : `compiled the kernels of ${r.warm} (${r.t.toFixed(1)} s)`; return; }
  if (r.seq === undefined) return;                       // keep-alive
  clearTimeout(slow);
  if (r.error) { $('st').textContent = 'error: ' + r.error; return; }
  if (r.stopped) { $('st').textContent += r.timeout ? `  — time limit reached (${fmtT(r.elapsed)})` : '  — stopped'; return; }
  if (r.seq < seq - 50) return;
  const bmp = await createImageBitmap(new Blob([img], {type: r.mime}));
  if (frame) frame.bmp.close();
  frame = {bmp, info: r}; info = r;
  redraw();
  const t0 = sentAt.get(r.seq);
  for (const k of [...sentAt.keys()]) if (k < r.seq) sentAt.delete(k);
  const lev = `${Math.ceil(r.W / r.s)}×${Math.ceil(r.H / r.s)}` + (r.rows < r.nys ? ` (rows ${r.rows}/${r.nys})` : '');
  $('st').textContent = `level ${lev} of ${r.W}×${r.H}` + (r.final ? ' — final' : '') + ` | ${r.n.toLocaleString()} new ρ, GPU ${fmtT(r.t_kernel)}` +
    (r.us ? ` (${r.us.toFixed(2)} µs/ρ)` : '') + ` | colour ${fmtT(r.t_colour)}, ${r.mime.slice(6).toUpperCase()} ${r.kb.toFixed(0)} KB` +
    (t0 ? ` | ${fmtT(performance.now() - t0)} since the change` : '') + (r.final ? '' : r.eta != null ? ` | to the final grid ≈ ${fmtT(r.eta)}` : '') +
    ` | r = ${r.r} delay steps, unstable ${(100 * r.unstable).toFixed(1)} %` + (r.flagged ? `, flagged ${r.flagged}` : '');
}
async function readStream() {
  const cat = (cs) => { const o = new Uint8Array(cs.reduce((s, c) => s + c.length, 0)); let k = 0; for (const c of cs) { o.set(c, k); k += c.length; } return o; };
  for (;;) {
    try {
      const rd = (await fetch('stream', {cache: 'no-store'})).body.getReader();
      let chunks = [], have = 0;
      const take = async (n) => {
        while (have < n) { const {done, value} = await rd.read(); if (done) throw new Error('closed'); chunks.push(value); have += value.length; }
        const all = chunks.length === 1 ? chunks[0] : cat(chunks), rest = all.subarray(n);
        chunks = rest.length ? [rest] : []; have = rest.length;
        return all.subarray(0, n);
      };
      for (;;) {
        const hd = await take(8), dv = new DataView(hd.buffer, hd.byteOffset, 8), hl = dv.getUint32(0), il = dv.getUint32(4);
        const r = JSON.parse(new TextDecoder().decode(await take(hl)));
        await onFrame(r, il > 0 ? (await take(il)).slice() : null);
      }
    } catch (err) {
      $('st2').textContent = `frame stream: ${err} — reconnecting`;
      await new Promise((res) => setTimeout(res, 1000));
      go();
    }
  }
}

// --- zoom (wheel at the cursor) and pan (drag) ---
function at(ev) {
  const rc = cv.getBoundingClientRect(), [w, h] = plotSize();
  return [(ev.clientX - rc.left - PAD.l) / w, (ev.clientY - rc.top - PAD.t) / h];
}
cv.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const [fx, fy] = at(ev), f = Math.exp(0.0015 * ev.deltaY);
  const cx = view.x0 + fx * (view.x1 - view.x0), cy = view.y1 - fy * (view.y1 - view.y0);
  view = {x0: cx - fx * (view.x1 - view.x0) * f, x1: cx + (1 - fx) * (view.x1 - view.x0) * f,
          y0: cy - (1 - fy) * (view.y1 - view.y0) * f, y1: cy + fy * (view.y1 - view.y0) * f};
  showView(); go();
}, {passive: false});
let drag = null;
cv.onpointerdown = (ev) => { drag = {p: at(ev), v: {...view}}; cv.setPointerCapture(ev.pointerId); cv.style.cursor = 'grabbing'; };
cv.onpointermove = (ev) => {
  if (!drag) return;
  const [fx, fy] = at(ev), dx = (fx - drag.p[0]) * (drag.v.x1 - drag.v.x0), dy = (fy - drag.p[1]) * (drag.v.y1 - drag.v.y0);
  view = {x0: drag.v.x0 - dx, x1: drag.v.x1 - dx, y0: drag.v.y0 + dy, y1: drag.v.y1 + dy};
  showView(); go();
};
cv.onpointerup = cv.onpointercancel = () => { drag = null; cv.style.cursor = 'grab'; };
cv.ondblclick = () => resetView();

// --- save: the whole final grid with axes, as PNG (downloaded, and kept in /content) ---
$('save').onclick = async () => {
  $('st2').textContent = 'saving…';
  try {
    const r = await fetch('full', {method: 'POST', body: JSON.stringify({cr: +$('cr').value, bnd: $('bnd').checked})});
    if (!r.ok) throw new Error(await r.text());
    const bmp = await createImageBitmap(await r.blob()), [w] = plotSize(), sc = bmp.width / w, h = bmp.height / sc;
    const oc = new OffscreenCanvas(Math.round((w + PAD.l + PAD.r) * sc), Math.round((h + PAD.t + PAD.b) * sc));
    drawChart(oc.getContext('2d'), sc, w, h, {bmp, info: {...info, s: 1, iw: bmp.width, ih: bmp.height}}, frame ? {x0: info.box[0], x1: info.box[1], y0: info.box[2], y1: info.box[3]} : view);
    const blob = await oc.convertToBlob({type: 'image/png'});
    const name = `sosd_${ex.key}_${bmp.width}x${bmp.height}.png`;
    const s = await (await fetch('savefile', {method: 'POST', headers: {'X-Name': name}, body: blob})).json();
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click();
    $('st2').textContent = `saved ${s.file} (${s.mb.toFixed(1)} MB; Files panel of Colab) and offered as a download` + (info.final ? '' : ' — NOTE: the chart was not refined to the final grid yet');
  } catch (err) { $('st2').textContent = 'save failed: ' + err; }
};

// --- controls ---
EXAMPLES.forEach((e) => $('ex').add(new Option(e.title, e.key)));
$('ex').onchange = () => setExample($('ex').value);
let typing = null;
$('txt').oninput = () => { clearTimeout(typing); typing = setTimeout(() => { if (setText($('txt').value)) go(); }, 600); };
$('ax').onchange = $('ay').onchange = () => {
  const a = +$('ax').value, b = +$('ay').value;
  if (a === b) { $('ax').value = axes[0]; $('ay').value = axes[1]; return; }
  axes = [a, b]; buildKnobs(); resetView();
};
for (const id of ['S', 'p', 'm', 'prec', 'forced', 'grid', 'fps', 'tl', 'cr', 'bnd', 'enc']) $(id).onchange = go;
for (const k of ['x0', 'x1', 'y0', 'y1']) $(k).onchange = () => { view[k] = +$(k).value; go(); };
$('rst').onclick = () => resetView();
$('stop').onclick = () => fetch('stop', {method: 'POST', body: JSON.stringify({seq})});
let rsz = null;
window.addEventListener('resize', () => { clearTimeout(rsz); rsz = setTimeout(go, 300); });

const srv = await (await fetch('info')).json();
$('dev').textContent = '— ' + srv.device + (srv.cuda ? '' : ' (no CUDA)');
readStream();
setExample(EXAMPLES[0].key);
// compile the other examples' kernels while idle
const warm = EXAMPLES.slice(1).map((e) => { const m = parseModel(e.text), a = m.params.findIndex((q) => q.name === e.axes[0]), b = m.params.findIndex((q) => q.name === e.axes[1]);
  return {name: e.key, code: modelJulia(m), values: m.params.map((q) => q.value), xi: a, yi: b, x0: m.params[a].lo, x1: m.params[a].hi,
          y0: m.params[b].lo, y1: m.params[b].hi, S: e.S, p: e.p, m: e.m, prec: 'F32', forced: !!(e.forced && m.f)}; });
fetch('warm', {method: 'POST', body: JSON.stringify({list: warm})});
</script></body></html>'''

# --- start (or reuse) the worker and the HTTP server, then show the page ----------------------
try:
    st_worker
    if not st_worker.alive():
        raise NameError
except NameError:
    st_worker = Worker()
_ST = _Stream()                                   # a re-run: older streams stop
try:
    _httpd.shutdown()
    _httpd.server_close()
except NameError:
    pass
_httpd = ThreadingHTTPServer(('', PORT), _Handler)
_httpd.daemon_threads = True
threading.Thread(target=_httpd.serve_forever, daemon=True).start()
try:
    from google.colab import output
    output.serve_kernel_port_as_iframe(PORT, height=1250)
except ImportError:
    print(f'open http://localhost:{PORT}/  (Ctrl+C to stop)')
    if __name__ == '__main__':
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            st_worker.p.terminate()
