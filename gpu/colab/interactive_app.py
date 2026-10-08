# SOSD interactive stability charts on a Colab GPU.
# Run in a Colab cell (after the gpu branch is cloned and gpu/ instantiated, e.g. by gpu/colab_run.sh):
#   import urllib.request
#   exec(urllib.request.urlopen('https://raw.githubusercontent.com/bachrathyd/SOSD.jl/gpu/gpu/colab/interactive_app.py').read().decode())
# Starts gpu/interactive/server.jl (a persistent Julia process, kernels compiled once), a small
# HTTP server in this kernel, and shows the page through Colab's port proxy. Re-running the cell
# reuses a running server.
import os, json, time, struct, threading, subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Outside Colab (local test): SOSD_DIR=<repo> SOSD_SERVER_ARGS=--cpu python gpu/colab/interactive_app.py
import tempfile
REPO_DIR = os.environ.get('SOSD_DIR', '/content/SOSD.jl')
SERVER_ARGS = os.environ.get('SOSD_SERVER_ARGS', '').split()
PORT = int(os.environ.get('SOSD_PORT', '8800'))
_TMP = '/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir()
RHO_FILE = os.path.join(_TMP, 'sosd_rho.f32')
LOG = os.path.join('/content' if os.path.isdir('/content') else _TMP, 'sosd_server.log')
os.environ['PATH'] = '/root/.juliaup/bin' + os.pathsep + os.environ['PATH']


class ChartServer:
    "The Julia GPU server: one request per line on stdin, one answer line on stdout."
    def __init__(self):
        self.lock = threading.Lock()
        self.log = open(LOG, 'w')
        self.p = subprocess.Popen(['julia', '-t', 'auto', '--project=gpu', 'gpu/interactive/server.jl'] + SERVER_ARGS,
                                  cwd=REPO_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, encoding='utf-8', bufsize=1)
        t0 = time.time()
        print('starting the GPU chart server (first start compiles the kernels, ~2-5 min) ...', flush=True)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('chart server exited during start-up; see ' + LOG)
            if line.strip() == 'READY':
                break
        print(f'server ready after {time.time() - t0:.0f} s', flush=True)
        self.meta = json.loads(self.ask('meta').split(' ', 1)[1])

    def alive(self):
        return self.p.poll() is None

    def ask(self, cmd):
        with self.lock:
            self.p.stdin.write(cmd + '\n')
            self.p.stdin.flush()
            line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('chart server died; see ' + LOG)
        return line.rstrip('\n')


def _chart(q):
    keys = ('ex', 'nx', 'ny', 'x0', 'x1', 'y0', 'y1', 'p', 'kd', 'prec', 'mode')
    cmd = 'chart ' + ' '.join(f'{k}={q[k]}' for k in keys) + ' k=' + ','.join(str(float(v)) for v in q['k']) + \
          f' out={RHO_FILE}'
    t0 = time.time()
    ans = srv.ask(cmd)
    if not ans.startswith('OK '):
        return {'error': ans}, b''
    info = json.loads(ans[3:])
    data = open(RHO_FILE, 'rb').read()
    info['t_server_py'] = (time.time() - t0) * 1e3
    return info, data


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ct):
        self.send_response(200)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0].strip('/')
        if path == '':
            self._send(_PAGE.encode(), 'text/html; charset=utf-8')
        elif path == 'meta':
            self._send(json.dumps(srv.meta).encode(), 'application/json')
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.split('?')[0].strip('/') != 'chart':
            return self.send_error(404)
        q = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        try:
            info, data = _chart(q)
        except Exception as err:
            info, data = {'error': str(err)}, b''
        h = json.dumps(info).encode()
        self._send(struct.pack('>I', len(h)) + h + data, 'application/octet-stream')


_PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>SOSD stability chart</title>
<style>
 body {margin: 0; font: 13px system-ui, sans-serif; background: #fff; color: #222}
 #app {padding: 8px 10px; max-width: 1180px}
 .row {display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; margin: 4px 0}
 label {display: inline-flex; align-items: center; gap: 5px}
 select, input[type=number] {font: inherit; padding: 2px 3px}
 input[type=number] {width: 76px}
 input[type=range] {width: 170px}
 button {font: inherit; padding: 3px 10px; cursor: pointer}
 .val {display: inline-block; min-width: 48px; font-variant-numeric: tabular-nums}
 #st, #bench {font: 12px ui-monospace, monospace; white-space: pre-wrap; margin: 4px 0}
 #title {font-weight: 600; margin: 4px 0}
 canvas {display: block}
</style></head><body><div id="app">
 <div class="row">
  <label>example <select id="ex"></select></label>
  <label>precision <select id="prec"></select></label>
  <label>mode <select id="mode"><option value="fast">fast (no retry, 8 restarts)</option><option value="accurate">accurate (retry, KrylovKit-like)</option></select></label>
  <label>grid <select id="grid"></select></label>
  <label>steps p <input type="range" id="p" min="10" max="400" step="10" value="60"><span class="val" id="pv"></span></label>
  <label>Krylov dim <select id="kd"><option>10</option><option>12</option><option selected>16</option><option>20</option><option>30</option></select></label>
 </div>
 <div class="row" id="knobs"></div>
 <div class="row">
  <label>x from <input type="number" id="x0"></label><label>to <input type="number" id="x1"></label>
  <label>y from <input type="number" id="y0"></label><label>to <input type="number" id="y1"></label>
  <button id="reset">reset view</button>
  <label>colour range <select id="cr"><option value="0.1">ρ ∈ [0.79, 1.26]</option><option value="0.3" selected>ρ ∈ [0.5, 2]</option><option value="1">ρ ∈ [0.1, 10]</option></select></label>
  <label><input type="checkbox" id="smooth"> smooth pixels</label>
  <button id="bn">benchmark ×5</button>
  <button id="cmp">compare with Float64</button>
 </div>
 <div id="title"></div>
 <canvas id="cv" width="1100" height="470" style="width:100%;max-width:1100px"></canvas>
 <div id="st">connecting ...</div><div id="bench"></div>
</div>
<script>
const $ = id => document.getElementById(id);
let META = null, EX = null, inflight = false, pending = false, last = null, seq = 0;
const GRIDS = [[64,32],[128,64],[200,100],[320,160],[400,200],[640,320],[800,400]];
const STOPS = [[5,48,97],[33,102,172],[67,147,195],[146,197,222],[247,247,247],[244,165,130],[214,96,77],[178,24,43],[103,0,31]];
function cmap(v) {   // v = log10(rho) / range in [-1, 1] -> RdBu (blue = stable, red = unstable)
  let t = Math.min(1, Math.max(0, (v + 1) / 2)) * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(t)), w = t - i;
  return STOPS[i].map((c, k) => Math.round(c + w * (STOPS[i + 1][k] - c)));
}
function state() {
  const g = GRIDS[+$('grid').value];
  return {ex: EX.key, nx: g[0], ny: g[1], x0: +$('x0').value, x1: +$('x1').value, y0: +$('y0').value, y1: +$('y1').value,
          p: +$('p').value, kd: +$('kd').value, prec: $('prec').value, mode: $('mode').value,
          k: EX.knobs.map((k, i) => +$('k' + i).value)};
}
async function post(q) {
  const t0 = performance.now();
  const r = await fetch('chart', {method: 'POST', body: JSON.stringify(q)});
  const buf = await r.arrayBuffer(), dv = new DataView(buf), hl = dv.getUint32(0);
  const info = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  info.t_round = performance.now() - t0;
  const rho = info.error ? null : new Float32Array(buf.slice(4 + hl));
  return {q, info, rho};
}
function request() {
  if (!META) return;
  if (inflight) { pending = true; return; }
  inflight = true; pending = false;
  const q = state();
  $('st').textContent = 'computing ' + q.nx + '×' + q.ny + ' = ' + q.nx * q.ny + ' points ...';
  post(q).then(res => { last = res; draw(); status(); }).catch(e => { $('st').textContent = 'error: ' + e; })
         .finally(() => { inflight = false; if (pending) request(); });
}
function status() {
  const i = last.info;
  if (i.error) { $('st').textContent = 'server error: ' + i.error; return; }
  const f = x => x == null ? '–' : x.toFixed(0);
  $('st').textContent = META.device + ' | ' + i.n + ' points, ' + last.q.prec + ', p = ' + last.q.p + ', Krylov ' + last.q.kd +
    ' | solve ' + f(i.t_total) + ' ms = ' + i.us_per_rho.toFixed(1) + ' µs/ρ' +
    '  (build ' + f(i.t_build) + ', sweeps ' + f(i.t_sweep) + ', orth ' + f(i.t_orth) + ', host ' + f(i.t_host) + ')' +
    ' | round trip ' + f(i.t_round) + ' ms\n' +
    'converged ' + i.converged + '/' + i.n + ', unstable ' + (100 * i.unstable / i.n).toFixed(1) + ' %, ' +
    'ρ ∈ [' + i.rmin.toPrecision(3) + ', ' + i.rmax.toPrecision(3) + '], sweeps ' + i.sweeps + (i.flagged ? ', FLAGGED ' + i.flagged : '');
}
const PAD = {l: 70, r: 90, t: 10, b: 50};
function nice(a, b, n) {
  const span = (b - a) / n, mag = Math.pow(10, Math.floor(Math.log10(span))), s = [1, 2, 2.5, 5, 10].map(x => x * mag).find(x => x >= span);
  const out = []; for (let v = Math.ceil(a / s) * s; v <= b + 1e-9 * s; v += s) out.push(+v.toPrecision(10)); return out;
}
function draw() {
  if (!last || !last.rho) return;
  const CR = +$('cr').value, cv = $('cv'), cx = cv.getContext('2d'), q = last.q, nx = q.nx, ny = q.ny, rho = last.rho;
  const W = cv.width - PAD.l - PAD.r, H = cv.height - PAD.t - PAD.b;
  cx.fillStyle = '#fff'; cx.fillRect(0, 0, cv.width, cv.height);
  const img = new ImageData(nx, ny);
  for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
    const c = cmap(Math.log10(rho[ix + nx * iy]) / CR), o = 4 * (ix + nx * (ny - 1 - iy));
    img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
  }
  const off = new OffscreenCanvas(nx, ny); off.getContext('2d').putImageData(img, 0, 0);
  cx.imageSmoothingEnabled = $('smooth').checked;
  cx.drawImage(off, PAD.l, PAD.t, W, H);
  // ρ = 1 boundary: marching squares on the cell centres
  const X = ix => PAD.l + (ix + 0.5) * W / nx, Y = iy => PAD.t + H - (iy + 0.5) * H / ny, f = (ix, iy) => rho[ix + nx * iy] - 1;
  cx.strokeStyle = '#000'; cx.lineWidth = 1.6; cx.beginPath();
  for (let iy = 0; iy < ny - 1; iy++) for (let ix = 0; ix < nx - 1; ix++) {
    const v = [f(ix, iy), f(ix + 1, iy), f(ix + 1, iy + 1), f(ix, iy + 1)];
    const P = [[ix, iy], [ix + 1, iy], [ix + 1, iy + 1], [ix, iy + 1]], pts = [];
    for (let e = 0; e < 4; e++) { const a = v[e], b = v[(e + 1) % 4];
      if ((a < 0) !== (b < 0)) { const t = a / (a - b), A = P[e], B = P[(e + 1) % 4];
        pts.push([X(A[0] + t * (B[0] - A[0])), Y(A[1] + t * (B[1] - A[1]))]); } }
    if (pts.length >= 2) { cx.moveTo(pts[0][0], pts[0][1]); cx.lineTo(pts[1][0], pts[1][1]); }
    if (pts.length === 4) { cx.moveTo(pts[2][0], pts[2][1]); cx.lineTo(pts[3][0], pts[3][1]); }
  }
  cx.stroke();
  // axes
  cx.strokeStyle = '#444'; cx.lineWidth = 1; cx.strokeRect(PAD.l, PAD.t, W, H);
  cx.fillStyle = '#222'; cx.font = '12px system-ui'; cx.textAlign = 'center';
  for (const v of nice(q.x0, q.x1, 8)) { const x = PAD.l + (v - q.x0) / (q.x1 - q.x0) * W;
    cx.fillRect(x, PAD.t + H, 1, 5); cx.fillText(+v.toPrecision(6), x, PAD.t + H + 18); }
  cx.fillText(EX.xl, PAD.l + W / 2, PAD.t + H + 40);
  cx.textAlign = 'right';
  for (const v of nice(q.y0, q.y1, 6)) { const y = PAD.t + H - (v - q.y0) / (q.y1 - q.y0) * H;
    cx.fillRect(PAD.l - 5, y, 5, 1); cx.fillText(+v.toPrecision(6), PAD.l - 8, y + 4); }
  cx.save(); cx.translate(18, PAD.t + H / 2); cx.rotate(-Math.PI / 2); cx.textAlign = 'center'; cx.fillText(EX.yl, 0, 0); cx.restore();
  // colour bar
  const bx = PAD.l + W + 18, bw = 16;
  for (let k = 0; k < H; k++) { const c = cmap(1 - 2 * k / H); cx.fillStyle = `rgb(${c})`; cx.fillRect(bx, PAD.t + k, bw, 1); }
  cx.strokeRect(bx, PAD.t, bw, H); cx.fillStyle = '#222'; cx.textAlign = 'left';
  for (const v of [-1, -0.5, 0, 0.5, 1]) cx.fillText((v * CR).toFixed(CR < 0.5 ? 2 : 1), bx + bw + 4, PAD.t + (1 - v) / 2 * H + 4);
  cx.save(); cx.translate(bx + bw + 40, PAD.t + H / 2); cx.rotate(-Math.PI / 2); cx.textAlign = 'center';
  cx.fillText('log₁₀ ρ  (blue stable, red unstable)', 0, 0); cx.restore();
}
function setExample(key) {
  EX = META.examples.find(e => e.key === key);
  $('title').textContent = EX.title;
  $('knobs').innerHTML = EX.knobs.map((k, i) =>
    `<label>${k.name} <input type="range" id="k${i}" min="${k.lo}" max="${k.hi}" step="${k.step}" value="${k.value}"><span class="val" id="kv${i}">${k.value}</span></label>`).join('');
  EX.knobs.forEach((k, i) => $('k' + i).oninput = () => { $('kv' + i).textContent = $('k' + i).value; request(); });
  resetView();
}
function resetView() { $('x0').value = EX.xr[0]; $('x1').value = EX.xr[1]; $('y0').value = EX.yr[0]; $('y1').value = EX.yr[1]; request(); }
async function bench() {
  const q = state(), ts = [], us = [];
  for (let i = 0; i < 5; i++) { const r = await post(q); if (r.info.error) { $('bench').textContent = r.info.error; return; }
    ts.push(r.info.t_total); us.push(r.info.us_per_rho); }
  ts.sort((a, b) => a - b); us.sort((a, b) => a - b);
  $('bench').textContent = `benchmark ${q.nx}×${q.ny} ${q.prec} p=${q.p} Krylov ${q.kd} ${q.mode}: median ${ts[2].toFixed(0)} ms (min ${ts[0].toFixed(0)}) = ${us[2].toFixed(1)} µs/ρ`;
}
async function compare() {
  if (!last || !last.rho) return;
  const q = Object.assign({}, last.q, {prec: 'F64', mode: 'accurate'}), mine = last.rho;
  $('bench').textContent = 'computing the Float64 reference ...';
  const r = await post(q); if (r.info.error) { $('bench').textContent = r.info.error; return; }
  let mis = 0, rel = []; for (let i = 0; i < mine.length; i++) {
    if ((mine[i] >= 1) !== (r.rho[i] >= 1)) mis++; rel.push(Math.abs(mine[i] - r.rho[i]) / r.rho[i]); }
  rel.sort((a, b) => a - b);
  $('bench').textContent = `${last.q.prec} vs Float64 (${mine.length} points): misclassified ${mis} (${(100 * mis / mine.length).toFixed(2)} %), ` +
    `|Δρ|/ρ median ${rel[rel.length >> 1].toExponential(1)}, 99 % ${rel[Math.floor(0.99 * rel.length)].toExponential(1)}, max ${rel[rel.length - 1].toExponential(1)}` +
    ` | Float64 took ${r.info.t_total.toFixed(0)} ms`;
}
$('cv').onmousemove = ev => {
  if (!last || !last.rho) return; const q = last.q, cv = $('cv'), rc = cv.getBoundingClientRect();
  const W = cv.width - PAD.l - PAD.r, H = cv.height - PAD.t - PAD.b;
  const px = (ev.clientX - rc.left) * cv.width / rc.width - PAD.l, py = (ev.clientY - rc.top) * cv.height / rc.height - PAD.t;
  if (px < 0 || py < 0 || px > W || py > H) return;
  const ix = Math.min(q.nx - 1, Math.floor(px / W * q.nx)), iy = Math.min(q.ny - 1, Math.floor((H - py) / H * q.ny));
  const x = q.x0 + (ix + 0.5) / q.nx * (q.x1 - q.x0), y = q.y0 + (iy + 0.5) / q.ny * (q.y1 - q.y0);
  cv.title = `x = ${x.toPrecision(5)}, y = ${y.toPrecision(5)}, ρ = ${last.rho[ix + q.nx * iy].toPrecision(6)}`;
};
fetch('meta').then(r => r.json()).then(m => {
  META = m;
  $('ex').innerHTML = m.examples.map(e => `<option value="${e.key}">${e.key}</option>`).join('');
  $('prec').innerHTML = m.precisions.map(p => `<option value="${p[0]}">${p[1]}</option>`).join('');
  $('prec').value = 'F32';
  $('grid').innerHTML = GRIDS.map((g, i) => `<option value="${i}">${g[0]}×${g[1]} = ${g[0] * g[1]}</option>`).join('');
  $('grid').value = 1;
  $('pv').textContent = $('p').value;
  ['prec', 'mode', 'grid', 'kd', 'x0', 'x1', 'y0', 'y1'].forEach(id => $(id).onchange = request);
  $('p').oninput = () => { $('pv').textContent = $('p').value; request(); };
  $('ex').onchange = () => setExample($('ex').value);
  $('reset').onclick = resetView; $('smooth').onchange = draw; $('cr').onchange = draw; $('bn').onclick = bench; $('cmp').onclick = compare;
  setExample(m.examples[0].key);
}).catch(e => { $('st').textContent = 'meta failed: ' + e; });
</script></body></html>'''

# --- start (or reuse) the Julia server and the HTTP server, then show the page ----------------
try:
    srv
    if not srv.alive():
        raise NameError
except NameError:
    srv = ChartServer()
try:
    _httpd
except NameError:
    _httpd = ThreadingHTTPServer(('0.0.0.0', PORT), _Handler)
    threading.Thread(target=_httpd.serve_forever, daemon=True).start()
print('device:', srv.meta['device'], '| Julia threads:', srv.meta['threads'])
try:
    from google.colab import output
    output.serve_kernel_port_as_iframe(PORT, height=900)
except ImportError:
    print(f'open http://localhost:{PORT}/  (Ctrl+C to stop)')
    if __name__ == '__main__':
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            srv.ask('quit') if srv.alive() else None
