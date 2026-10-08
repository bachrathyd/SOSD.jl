// Optional anonymous usage counts with GoatCounter (https://www.goatcounter.com: no cookies, no
// personal data), as in the InterpolatedNyquist WebGPU page. OFF unless GOATCOUNTER is set; then
// only these are sent: one page view (the path, never the query or hash), example/<key> when an
// example is opened, custom-model once per distinct edited model text (a local hash decides what
// is new; the text is never sent), and features once per page session (brute-force, mdbm,
// validate-run, bench-run), the GPU vendor class and no-webgpu.

export const GOATCOUNTER = 'https://sosdgpu.goatcounter.com/count';
export const SHOW_PUBLIC_COUNTER = false;

const queue = [];
const sent = new Set();
let ready = false;

function send(o) {
  try { window.goatcounter.count(o); } catch (e) { /* blocked or offline */ }
}

export function initStats() {
  if (!GOATCOUNTER) return;
  try {
    const s = document.createElement('script');
    s.async = true;
    s.src = 'https://gc.zgo.at/count.js';
    s.dataset.goatcounter = GOATCOUNTER;
    s.dataset.goatcounterSettings = '{"no_onload": true}';
    s.addEventListener('load', () => {
      ready = !!(window.goatcounter && window.goatcounter.count);
      if (!ready) return;
      send({ path: location.pathname });
      while (queue.length) send(queue.shift());
    });
    document.head.appendChild(s);
  } catch (e) { /* ignore */ }
}

export function countEvent(path) {
  if (!GOATCOUNTER) return;
  const o = { path, title: path, event: true };
  if (ready) send(o); else if (queue.length < 50) queue.push(o);
}

export function countOnce(path) {
  if (!GOATCOUNTER || sent.has(path)) return;
  sent.add(path);
  countEvent(path);
}

const counted = new Set();
/** an edited model text, counted once per distinct text (only a hash is kept, the text is never sent) */
export function countModel(text) {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
  if (counted.has(h)) return;
  counted.add(h);
  countEvent('custom-model');
}
