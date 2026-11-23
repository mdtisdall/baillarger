// The display window of the brick pane, with a histogram of what is in view.
//
// A plug-in of the brick pane (window.BVP hooks, api window.BVAPI), as static/overlays.js.
//
// The window: the values drawn black (lower) and white (upper), linear between; no gamma, no log. Each volume has its own:
// its default is the volume's meta window (baillarger.viewer.bricks.window), and a window set here is kept by the server in
// the shared state (`windows`: {volume: [lower, upper]}, only the volumes not at their default), so every open window of
// the viewer shows the same one, and switching echo shows that echo's window. A volume is named as the page asks for it:
// its /volume/list name, or "echo" for the default volume (/volume). A new window changes two numbers in the shader and
// redraws: it fetches no bricks.
//
// The histogram is of what is in view, read in the page from the loaded bricks: in each tile of the pane, the value that
// the shader draws (the finest loaded level from the tile's level, nearest voxel) at the centre of each CSS pixel, on a
// grid of at most HIST_SIDE x HIST_SIDE points per tile (a coarser stride on a larger tile). So a voxel counts by the
// screen area it covers. Points outside the volume or where nothing is loaded yet do not count; label volumes are not part
// of it. HIST_BINS bins from the lowest to the highest value in view (widened to hold the window), heights log(1 + count).
// It is read again after each change of the view (pan, zoom, slice, layout, echo) and each brick load, at most every
// HIST_WAIT_MS. The window's two limits are marked on it, and dragging a mark moves that limit.
const HIST_SIDE = 256, HIST_BINS = 128, HIST_WAIT_MS = 120, SHARE_MS = 60, HW = 180, HH = 34;
const W = window.WIN = { api: null, windows: {}, hist: null, sig: "", timer: 0, drag: null, sent: 0, sendTimer: 0 };

const volName = () => { const b = W.api.BV.base || "/volume"; return b === "/volume" ? "echo" : b.slice("/volume/".length); };
const dflt = () => W.api.BV.M.window;
const current = () => W.windows[volName()] || dflt();
const fmt = x => Math.abs(x) >= 1000 || Number.isInteger(x) ? String(Math.round(x)) : x.toPrecision(4);

// the window of the shown volume to the shader and the controls
function applyWin() {
  const api = W.api; if (!api || !api.BV.M) return;
  const w = current(), $ = api.$; api.BV.win = w;
  if (document.activeElement !== $("wlo")) $("wlo").value = fmt(w[0]);
  if (document.activeElement !== $("whi")) $("whi").value = fmt(w[1]);
  $("wdef").disabled = !W.windows[volName()];
  $("wdef").title = `back to the volume's default window: ${fmt(dflt()[0])} to ${fmt(dflt()[1])} (percentiles 0.5 and 99.5)`;
  api.draw(); drawHist();
}
// a new window for the shown volume (null: its default), here at once and to the other windows (at most every SHARE_MS
// while a mark is dragged; the last one always)
function setWin(w, now = true) {
  const name = volName();
  if (w) { if (!w.every(Number.isFinite) || !(w[0] < w[1])) { applyWin(); return; } W.windows[name] = [w[0], w[1]]; }
  else delete W.windows[name];
  applyWin();
  clearTimeout(W.sendTimer);
  const send = () => { W.sent = performance.now(); W.api.share({ windows: { [name]: w ? [w[0], w[1]] : null } }); };
  if (now || performance.now() - W.sent >= SHARE_MS) send(); else W.sendTimer = setTimeout(send, SHARE_MS);
}

function controls() {
  const api = W.api, bar = api.$("vfit").parentElement, box = document.createElement("span"); box.id = "bvwin";
  box.style.cssText = "display:inline-flex;align-items:center;gap:4px";
  box.innerHTML = `<label title="the display window: values at or below the lower limit are black, at or above the upper limit white, linear between (each volume its own, the same in every viewer window)">window
      <input id="wlo" type="number" step="any" style="width:68px"></label>
    <canvas id="whist" width="${HW}" height="${HH}" style="width:${HW}px;height:${HH}px;cursor:ew-resize;background:#111317;border:1px solid #343840;border-radius:3px;touch-action:none"
      title="the values in view in the slices (log of the count); the marks are the window's limits (blue lower, orange upper): drag one to move it"></canvas>
    <input id="whi" type="number" step="any" style="width:68px">
    <button id="wdef">default</button>`;
  bar.appendChild(box);
  const $ = api.$, c = $("whist");
  const typed = () => setWin([Number($("wlo").value), Number($("whi").value)]);
  $("wlo").onchange = typed; $("whi").onchange = typed;
  $("wdef").onclick = () => setWin(null);
  const val = e => { const r = c.getBoundingClientRect(), R = W.drag.range; return R[0] + (e.clientX - r.left) / r.width * (R[1] - R[0]); };
  c.addEventListener("pointerdown", e => {
    if (!W.hist || !W.hist.range) return; try { c.setPointerCapture(e.pointerId); } catch { /* a pointer the page did not start */ }
    W.drag = { k: 0, range: W.hist.range.slice() };
    const v = val(e), w = current(); W.drag.k = Math.abs(v - w[0]) <= Math.abs(v - w[1]) ? 0 : 1;   // the nearer mark
  });
  c.addEventListener("pointermove", e => {
    if (!W.drag) return; const w = current().slice(), v = val(e), k = W.drag.k, eps = (W.drag.range[1] - W.drag.range[0]) * 1e-3;
    w[k] = k === 0 ? Math.min(v, w[1] - eps) : Math.max(v, w[0] + eps); setWin(w, false);
  });
  const up = () => { if (!W.drag) return; W.drag = null; setWin(current().slice()); later(); };
  c.addEventListener("pointerup", up); c.addEventListener("pointercancel", up);
}

// ------------------------------------------------------------------ the histogram of the view
// the value that the shader draws at world point w in a tile of level lev, or null (outside, or nothing loaded)
function valueAt(BV, aff, w, lev) {
  const M = BV.M, B = M.brick, sh0 = M.levels[0].shape;
  const i0 = aff(BV.inv[0], w).map(x => Math.floor(x + 0.5));
  if (i0[0] < 0 || i0[1] < 0 || i0[2] < 0 || i0[0] >= sh0[0] || i0[1] >= sh0[1] || i0[2] >= sh0[2]) return null;
  for (let L = lev; L < BV.nl; L++) {
    const sh = M.levels[L].shape, iv = L === 0 ? i0 : aff(BV.inv[L], w).map((x, k) => Math.min(sh[k] - 1, Math.max(0, Math.floor(x + 0.5))));
    const b = iv.map(i => Math.floor(i / B)), s = BV.pages[((b[2] + BV.pz[L]) * BV.gy + b[1]) * BV.gx + b[0]];
    const c = s ? BV.cpu[s - 1] : null; if (!c) continue;
    return c.data[(iv[0] - b[0] * B) + c.n[0] * ((iv[1] - b[1] * B) + c.n[1] * (iv[2] - b[2] * B))];
  }
  return null;
}
function histogram() {
  const api = W.api, BV = api.BV; if (!BV.M || !BV.tiles.length || !api.cross) return null;
  const vals = [];
  for (const t of BV.tiles) {
    if (!(t.w > 0 && t.h > 0)) continue;
    const st = Math.max(1, Math.max(t.w, t.h) / HIST_SIDE);
    for (let y = st / 2; y < t.h; y += st) for (let x = st / 2; x < t.w; x += st) {
      const v = valueAt(BV, api.affApply, api.px2w(t, x, y), t.L ?? BV.nl - 1); if (v !== null) vals.push(v);
    }
  }
  if (!vals.length) return { range: null, counts: null, n: 0 };
  let lo = Infinity, hi = -Infinity; for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const vlo = lo, vhi = hi, w = current(); lo = Math.min(lo, w[0]); hi = Math.max(hi, w[1]); if (!(hi > lo)) hi = lo + 1;
  const pad = (hi - lo) * 0.02; lo -= pad; hi += pad;
  const counts = new Uint32Array(HIST_BINS), k = HIST_BINS / (hi - lo);
  for (const v of vals) counts[Math.min(HIST_BINS - 1, Math.floor((v - lo) * k))]++;
  return { range: [lo, hi], counts, n: vals.length, vlo, vhi };
}
function drawHist() {
  const c = W.api && W.api.$("whist"); if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== HW * dpr) { c.width = HW * dpr; c.height = HH * dpr; }
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, HW, HH);
  const h = W.hist; if (!h || !h.counts) { g.fillStyle = "#9aa0a8"; g.font = "10px system-ui"; g.fillText("nothing in view yet", 6, 20); return; }
  const R = W.drag ? W.drag.range : h.range, X = v => (v - R[0]) / (R[1] - R[0]) * HW, w = current();
  g.fillStyle = "rgba(255, 255, 255, 0.07)"; g.fillRect(X(w[0]), 0, X(w[1]) - X(w[0]), HH);   // the window's span
  let m = 0; for (const n of h.counts) m = Math.max(m, Math.log1p(n));
  g.fillStyle = "#8a93a0"; const x0 = X(h.range[0]), bw = (X(h.range[1]) - x0) / HIST_BINS;
  h.counts.forEach((n, i) => { if (!n) return; const y = Math.log1p(n) / m * (HH - 2); g.fillRect(x0 + i * bw, HH - y, Math.max(1, bw), y); });
  for (const [k, col] of [[0, "#4da3ff"], [1, "#ffb84d"]]) {
    const x = Math.round(X(w[k])) + 0.5; g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, HH); g.stroke();
  }
  c.dataset.help = c.dataset.help || c.title;
  c.title = c.dataset.help + `\n${h.n.toLocaleString()} points in view, values ${fmt(h.vlo)} to ${fmt(h.vhi)}`;
}
function later() {   // read the histogram again soon (once, after a burst of changes; not while a mark is dragged)
  clearTimeout(W.timer);
  W.timer = setTimeout(() => { if (W.drag) { later(); return; } W.hist = histogram(); drawHist(); }, HIST_WAIT_MS);
}
// what the histogram depends on: the volume, its loads, the view
function signature() {
  const BV = W.api.BV, c = W.api.cross || [];
  return [BV.base, BV.gen, BV.loads, BV.s, ...BV.C, ...c, ...BV.tiles.map(t => `${t.name}${t.w}x${t.h}L${t.L}`)].join("|");
}

// ------------------------------------------------------------------ the hooks
const hooks = {
  init(api) {
    W.api = api; controls();
    window.VIEWER.hooks.state.push(st => {   // the first state from the server, and each change from another viewer window
      if (st && st.windows && typeof st.windows === "object") { W.windows = { ...st.windows }; applyWin(); later(); }
    });
    applyWin();
  },
  update() { if (W.api) applyWin(); },   // after each change of the view, and after an echo switch (its meta reset BV.win)
  render() { if (!W.api) return; const s = signature(); if (s !== W.sig) { W.sig = s; later(); } },
};
(window.BVP = window.BVP || []).push(hooks);
