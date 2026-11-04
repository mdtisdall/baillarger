// Overlays on the brick pane (phase 4.0.5): label volumes and the outlines of the true white surface.
//
// A plug-in of the brick pane: index.html calls the hooks of each object in window.BVP (init, update, render, draw2d,
// text, pointer) with its API (window.BVAPI). This file adds no code to index.html beyond those hooks.
//
// Label layers. Each label volume (/volume/list, kind "labels") is its own brick pyramid on its own grid: a page table and
// an atlas (R16I or R32I: labels are never float) of its own, with its own level for each tile (its voxels at least one CSS
// pixel, coarser when its bricks do not fit the atlas) and the same fallback to coarser loaded levels as the echo. Labels
// are sampled nearest, never interpolated; a colour table (a small RGBA texture, value -> colour, from the volume's meta)
// colours them; value 0 is transparent. The outline mode draws a label's edge (a pixel whose label differs from one of
// its four neighbours one CSS pixel away), not its fill. The second pass blends over the echo in each tile.
//
// Surface outlines. Per tile, GET /outline?axis=&mm=&box= gives the segments where the true mesh crosses the tile's plane,
// in world mm (float32, end a then end b). The box is the tile's view widened by half a view on each side and snapped to a
// grid of half a view, so small pans reuse the cache. The server cuts the overview mesh above its face limit and says so
// (X-Outline-Source: overview); the tile's text line then says "outline: overview".
const ATLAS_BYTES = 64 * 2 ** 20, BATCH = 64, INFLIGHT = 3, OUTLINE_CACHE = 48, OUTLINE_WAIT_MS = 60;
const LFS = sampler => `#version 300 es
precision highp float; precision highp int; precision highp usampler3D; precision highp ${sampler};
uniform ${sampler} atlas; uniform usampler3D pages; uniform highp sampler2D lut;
uniform mat4 w2v[16]; uniform ivec3 shp[16]; uniform int pz[16]; uniform int nlev, lev, B, lutw, edge; uniform ivec3 slots;
uniform vec3 w0, du, dv; uniform vec2 org; uniform float alpha, ek; out vec4 o;
int lab(vec3 w) {   // the label at world point w: the finest loaded level from lev; 0 outside; -1 nothing loaded
  ivec3 i0 = ivec3(floor((w2v[0] * vec4(w, 1.0)).xyz + 0.5));
  if (any(lessThan(i0, ivec3(0))) || any(greaterThanEqual(i0, shp[0]))) return 0;
  for (int L = lev; L < nlev; L++) {
    ivec3 iv = L == 0 ? i0 : clamp(ivec3(floor((w2v[L] * vec4(w, 1.0)).xyz + 0.5)), ivec3(0), shp[L] - 1);
    ivec3 b = iv / B; uint s = texelFetch(pages, ivec3(b.xy, b.z + pz[L]), 0).r;
    if (s == 0u) continue;
    int k = int(s) - 1; ivec3 sl = ivec3(k % slots.x, (k / slots.x) % slots.y, k / (slots.x * slots.y));
    return int(texelFetch(atlas, sl * B + iv - b * B, 0).r);
  }
  return -1;
}
vec4 colour(int v) { return v <= 0 ? vec4(0.0) : texelFetch(lut, ivec2(v % lutw, v / lutw), 0); }
void main() {
  vec2 f = gl_FragCoord.xy - org; vec3 w = w0 + f.x * du + f.y * dv;
  int v = lab(w); vec4 c = colour(v); if (c.a == 0.0) discard;
  if (edge == 1) {
    int a = lab(w + ek * du), b = lab(w - ek * du), d = lab(w + ek * dv), e = lab(w - ek * dv);
    bool on = (a >= 0 && a != v) || (b >= 0 && b != v) || (d >= 0 && d != v) || (e >= 0 && e != v);
    if (!on) discard;
  }
  o = vec4(c.rgb, c.a * alpha);
}`;

class Layer {   // one label volume on the brick pane
  constructor(api, name, M) {
    this.api = api; this.name = name; this.M = M; this.nl = M.levels.length; this.B = M.brick;
    this.i32 = M.dtype === "int32"; if (M.dtype === "float32") throw new Error(`${name}: labels sent as float`);
    this.inv = M.levels.map(lv => api.affInv(lv.affine));
    const A = M.levels[0].affine; this.vs0 = Math.min(...[0, 1, 2].map(k => Math.hypot(A[0][k], A[1][k], A[2][k])));
    this.table = new Map(Object.entries(M.table).map(([v, e]) => [Number(v), e]));
    this.cache = new Map(); this.cpu = new Map(); this.pending = new Set(); this.want = new Set(); this.inflight = 0; this.tileL = new Map();
    const g = this.g = api.BV.g, B = this.B;
    this.prog = api.program(g, api.VS, LFS("isampler3D"));
    const grid = M.levels.map(lv => lv.shape.map(n => Math.ceil(n / B)));
    this.gx = grid[0][0]; this.gy = grid[0][1]; this.pz = []; let z = 0; for (const gr of grid) { this.pz.push(z); z += gr[2]; }
    this.pages = new Uint32Array(this.gx * this.gy * z);
    g.activeTexture(g.TEXTURE5);
    this.ptex = g.createTexture(); g.bindTexture(g.TEXTURE_3D, this.ptex); g.texStorage3D(g.TEXTURE_3D, 1, g.R32UI, this.gx, this.gy, z);
    for (const p of [g.TEXTURE_MIN_FILTER, g.TEXTURE_MAG_FILTER]) g.texParameteri(g.TEXTURE_3D, p, g.NEAREST);
    const mx = g.getParameter(g.MAX_3D_TEXTURE_SIZE), side = Math.min(512, mx), bpv = this.i32 ? 4 : 2;
    this.sx = this.sy = Math.floor(side / B); this.sz = Math.max(1, Math.floor(Math.min(mx, ATLAS_BYTES / bpv / (this.sx * B) ** 2) / B));
    this.nslots = this.sx * this.sy * this.sz; this.free = Array.from({ length: this.nslots }, (_, k) => this.nslots - 1 - k);
    this.atlas = g.createTexture(); g.bindTexture(g.TEXTURE_3D, this.atlas);
    g.texStorage3D(g.TEXTURE_3D, 1, this.i32 ? g.R32I : g.R16I, this.sx * B, this.sy * B, this.sz * B);
    for (const p of [g.TEXTURE_MIN_FILTER, g.TEXTURE_MAG_FILTER]) g.texParameteri(g.TEXTURE_3D, p, g.NEAREST);
    // the colour table: value v at texel (v % lutw, v / lutw)
    const vmax = Math.max(0, ...this.table.keys()); this.lutw = Math.min(4096, vmax + 1); const lh = Math.ceil((vmax + 1) / this.lutw);
    const lut = new Uint8Array(4 * this.lutw * lh);
    for (const [v, e] of this.table) lut.set(e.rgba, 4 * v);
    this.lut = g.createTexture(); g.bindTexture(g.TEXTURE_2D, this.lut);
    g.texImage2D(g.TEXTURE_2D, 0, g.RGBA8, this.lutw, lh, 0, g.RGBA, g.UNSIGNED_BYTE, lut);
    for (const p of [g.TEXTURE_MIN_FILTER, g.TEXTURE_MAG_FILTER]) g.texParameteri(g.TEXTURE_2D, p, g.NEAREST);
    g.activeTexture(g.TEXTURE0);
  }
  vox(L, w) { return this.api.affApply(this.inv[L], w); }
  key(L, b) { return `${L},${b[0]},${b[1]},${b[2]}`; }
  page(L, b) { return ((b[2] + this.pz[L]) * this.gy + b[1]) * this.gx + b[0]; }
  slot(s) { return [s % this.sx, Math.floor(s / this.sx) % this.sy, Math.floor(s / (this.sx * this.sy))]; }
  range(t, L) {   // as bvRange, on this volume's grid
    const sh = this.M.levels[L].shape, B = this.B, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const [px, py] of [[0, 0], [t.w, 0], [0, t.h], [t.w, t.h]]) {
      const v = this.vox(L, this.api.px2w(t, px, py)); for (let k = 0; k < 3; k++) { const i = Math.floor(v[k] + 0.5); lo[k] = Math.min(lo[k], i); hi[k] = Math.max(hi[k], i); } }
    for (let k = 0; k < 3; k++) {
      if (L === 0 && (hi[k] < 0 || lo[k] >= sh[k])) return null;
      lo[k] = Math.min(sh[k] - 1, Math.max(0, lo[k])); hi[k] = Math.min(sh[k] - 1, Math.max(0, hi[k]));
    }
    return { lo: lo.map(i => Math.floor(i / B)), hi: hi.map(i => Math.floor(i / B)) };
  }
  update(tiles, s) {
    const budget = Math.floor((this.nslots - 1) / Math.max(1, tiles.length)), n = r => r ? (r.hi[0] - r.lo[0] + 1) * (r.hi[1] - r.lo[1] + 1) * (r.hi[2] - r.lo[2] + 1) : 0;
    this.want = new Set([this.key(this.nl - 1, [0, 0, 0])]);
    for (const t of tiles) {
      if (!this.range(t, 0)) { this.tileL.set(t.name, this.nl - 1); continue; }
      let L = Math.min(this.nl - 1, Math.max(0, Math.ceil(Math.log2(1 / (s * this.vs0))))), r;
      for (;; L++) { r = this.range(t, L); if (n(r) <= budget || L === this.nl - 1) break; }
      this.tileL.set(t.name, L); if (!r) continue;
      for (let i = r.lo[0]; i <= r.hi[0]; i++) for (let j = r.lo[1]; j <= r.hi[1]; j++) for (let k = r.lo[2]; k <= r.hi[2]; k++) this.want.add(this.key(L, [i, j, k]));
    }
    for (const k of this.want) if (this.cache.has(k)) { const v = this.cache.get(k); this.cache.delete(k); this.cache.set(k, v); }
    this.fetch();
  }
  fetch() {
    while (this.inflight < INFLIGHT) {
      const miss = [...this.want].filter(k => !this.cache.has(k) && !this.pending.has(k)).map(k => k.split(",").map(Number));
      if (!miss.length) break;
      const L = Math.max(...miss.map(m => m[0])), batch = miss.filter(m => m[0] === L).slice(0, BATCH), keys = batch.map(m => this.key(L, m.slice(1)));
      for (const k of keys) this.pending.add(k); this.inflight++;
      this.load(L, batch.map(m => m.slice(1)), keys).finally(() => { this.inflight--; for (const k of keys) this.pending.delete(k); this.fetch(); });
    }
  }
  alloc(key) {
    if (this.cache.has(key)) return -1;
    let s = this.free.pop();
    if (s === undefined && this.want.has(key))
      for (const [k, v] of this.cache) if (!this.want.has(k)) { const m = k.split(",").map(Number); this.cache.delete(k); this.cpu.delete(k); this.pages[this.page(m[0], m.slice(1))] = 0; s = v; break; }
    if (s === undefined) return -1;
    this.cache.set(key, s); return s;
  }
  async load(L, bs, keys) {
    let buf;
    try { const r = await fetch(`/volume/${this.name}/bricks?level=${L}&b=${bs.map(b => b.join(",")).join(";")}`);
          if (!r.ok) throw new Error(`${r.status} ${await r.text()}`); buf = await r.arrayBuffer(); }
    catch (e) { this.api.log(`${this.name} bricks: ${e.message}`); return; }
    const g = this.g, B = this.B, sh = this.M.levels[L].shape, T = this.i32 ? Int32Array : Int16Array; let o = 0;
    g.activeTexture(g.TEXTURE5); g.bindTexture(g.TEXTURE_3D, this.atlas);
    bs.forEach((b, m) => {
      const n = b.map((x, k) => Math.min(B, sh[k] - x * B)), len = n[0] * n[1] * n[2], data = new T(buf.slice(o, o + len * T.BYTES_PER_ELEMENT)); o += len * T.BYTES_PER_ELEMENT;
      const s = this.alloc(keys[m]); if (s < 0) return; const sl = this.slot(s);
      g.texSubImage3D(g.TEXTURE_3D, 0, sl[0] * B, sl[1] * B, sl[2] * B, n[0], n[1], n[2], g.RED_INTEGER, this.i32 ? g.INT : g.SHORT, data);
      this.pages[this.page(L, b)] = s + 1; this.cpu.set(keys[m], { data, n });
    });
    g.bindTexture(g.TEXTURE_3D, this.ptex);
    g.texSubImage3D(g.TEXTURE_3D, 0, 0, 0, 0, this.gx, this.gy, this.pages.length / (this.gx * this.gy), g.RED_INTEGER, g.UNSIGNED_INT, this.pages);
    g.activeTexture(g.TEXTURE0);
    this.api.draw();
  }
  // the label at world point w, as the shader reads it: {v, L} (v 0 outside; null: nothing loaded there)
  at(w, lev = 0) {
    const sh0 = this.M.levels[0].shape, i0 = this.vox(0, w).map(x => Math.floor(x + 0.5));
    if (i0.some((i, k) => i < 0 || i >= sh0[k])) return { v: 0, L: 0 };
    for (let L = lev; L < this.nl; L++) {
      const sh = this.M.levels[L].shape, iv = L === 0 ? i0 : this.vox(L, w).map((x, k) => Math.min(sh[k] - 1, Math.max(0, Math.floor(x + 0.5))));
      const b = iv.map(i => Math.floor(i / this.B)), c = this.cpu.get(this.key(L, b)); if (!c) continue;
      const r = iv.map((i, k) => i - b[k] * this.B); return { v: c.data[r[0] + c.n[0] * (r[1] + c.n[1] * r[2])], L, voxel: iv };
    }
    return null;
  }
  name_of(v) { return v === 0 ? "outside" : (this.table.get(v)?.name ?? `label ${v} (not in the table)`); }
  render(geoms, alpha, edge, dpr) {
    const g = this.g, u = n => g.getUniformLocation(this.prog, n), nl = this.nl;
    g.useProgram(this.prog);
    g.activeTexture(g.TEXTURE2); g.bindTexture(g.TEXTURE_3D, this.atlas); g.uniform1i(u("atlas"), 2);
    g.activeTexture(g.TEXTURE3); g.bindTexture(g.TEXTURE_3D, this.ptex); g.uniform1i(u("pages"), 3);
    g.activeTexture(g.TEXTURE4); g.bindTexture(g.TEXTURE_2D, this.lut); g.uniform1i(u("lut"), 4);
    g.activeTexture(g.TEXTURE0);
    const w2v = new Float32Array(16 * nl), shp = new Int32Array(3 * nl);
    for (let L = 0; L < nl; L++) { const I = this.inv[L]; for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) w2v[16 * L + 4 * col + row] = I[row][col];
      shp.set(this.M.levels[L].shape, 3 * L); }
    g.uniformMatrix4fv(u("w2v"), false, w2v); g.uniform3iv(u("shp"), shp); g.uniform1iv(u("pz"), new Int32Array(this.pz));
    g.uniform1i(u("nlev"), nl); g.uniform1i(u("B"), this.B); g.uniform3i(u("slots"), this.sx, this.sy, this.sz);
    g.uniform1i(u("lutw"), this.lutw); g.uniform1i(u("edge"), edge ? 1 : 0); g.uniform1f(u("alpha"), alpha); g.uniform1f(u("ek"), Math.max(1, Math.round(dpr)));
    for (const G of geoms) {
      g.viewport(G.X, G.Y, G.W, G.H); g.scissor(G.X, G.Y, G.W, G.H);
      g.uniform3fv(u("w0"), G.w0); g.uniform3fv(u("du"), G.du); g.uniform3fv(u("dv"), G.dv); g.uniform2f(u("org"), G.x0, G.y0);
      g.uniform1i(u("lev"), this.tileL.get(G.t.name) ?? nl - 1);
      g.drawArrays(g.TRIANGLES, 0, 3);
    }
  }
}

const OVL = window.OVL = { api: null, vols: [], layers: new Map(), show: [], alpha: 0.5, edge: false, outlines: true,
                           ocache: new Map(), otimer: 0, cursor: null, ostat: new Map() };
const CHOICES = { none: [], labels: ["labels"], fixes: ["fixes"], both: ["labels", "fixes"] };

async function layer(name) {
  if (!OVL.layers.has(name)) {
    const M = await OVL.api.get(`/volume/${name}/meta`);
    OVL.layers.set(name, new Layer(OVL.api, name, M));
  }
  return OVL.layers.get(name);
}
async function choose(v) {
  const names = (CHOICES[v] || []).filter(n => OVL.vols.some(x => x.name === n));
  const t0 = performance.now(); OVL.show = [];
  try { for (const n of names) await layer(n); }
  catch (e) { OVL.api.log(`overlay: ${e.message}`); return; }
  if (names.some(n => !OVL.layers.has(n))) return;
  OVL.show = names.map(n => OVL.layers.get(n));
  if (names.length) OVL.api.log(`overlay ${names.join(" + ")} ready (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  hooks.update(); OVL.api.draw();
}
function controls() {
  const box = document.createElement("span"); box.id = "bvovl";
  const opt = OVL.vols.length ? Object.keys(CHOICES).filter(k => CHOICES[k].every(n => OVL.vols.some(x => x.name === n))) : ["none"];
  box.innerHTML = `<label title="the label volume over the echo (the most common label of each 2 × 2 × 2 block on coarse levels)">labels
      <select id="ovsel">${opt.map(k => `<option value="${k}">${k === "both" ? "labels + fixes" : k}</option>`).join("")}</select></label>
    <label>opacity <input type="range" id="ovop" min="0" max="1" step="0.05" value="${OVL.alpha}" style="width:60px"></label>
    <label title="draw each label's edge, not its fill"><input type="checkbox" id="ovedge"> edges only</label>
    <label title="where the true white surface crosses each slice (the overview when the view holds too many faces)"><input type="checkbox" id="ovout" checked> surface outline</label>`;
  const bar = OVL.api.$("vfit").parentElement; bar.appendChild(box);
  const $ = OVL.api.$;
  $("ovsel").value = opt.includes("labels") ? "labels" : "none";
  $("ovsel").onchange = () => choose($("ovsel").value);
  $("ovop").oninput = () => { OVL.alpha = Number($("ovop").value); OVL.api.draw(); };
  $("ovedge").onchange = () => { OVL.edge = $("ovedge").checked; OVL.api.draw(); };
  $("ovout").onchange = () => { OVL.outlines = $("ovout").checked; hooks.update(); OVL.api.draw(); };
  return $("ovsel").value;
}

// ------------------------------------------------------------------ surface outlines
function oview(t) {   // the tile's view in world mm: lo, hi (the plane's axis at the crosshair)
  const api = OVL.api, a = api.px2w(t, 0, t.h), b = api.px2w(t, t.w, 0), lo = a.slice(), hi = b.slice();
  for (let k = 0; k < 3; k++) { lo[k] = Math.min(a[k], b[k]); hi[k] = Math.max(a[k], b[k]); }
  return { lo, hi };
}
function okey(t) {   // the box to ask for and its cache key: the view widened by half a view, snapped to half a view
  const { lo, hi } = oview(t), mm = OVL.api.cross[t.n], L = [...lo], H = [...hi];
  for (const k of [t.r, t.u]) {
    const ext = Math.max(hi[k] - lo[k], 1e-3), q = 2 ** Math.ceil(Math.log2(ext / 2));
    L[k] = Math.floor((lo[k] - ext / 2) / q) * q; H[k] = Math.ceil((hi[k] + ext / 2) / q) * q;
  }
  L[t.n] = H[t.n] = mm;
  return { key: `${t.n}|${mm.toFixed(4)}|${L.map(x => x.toFixed(3))}|${H.map(x => x.toFixed(3))}`, lo: L, hi: H, mm };
}
async function oload(t, k) {
  const c = { seg: null, source: "", pending: true }; OVL.ocache.set(k.key, c);
  if (OVL.ocache.size > OUTLINE_CACHE) OVL.ocache.delete(OVL.ocache.keys().next().value);
  try {
    const r = await fetch(`/outline?axis=${t.n}&mm=${k.mm}&box=${[...k.lo, ...k.hi].join(",")}`);
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    c.seg = new Float32Array(await r.arrayBuffer()); c.source = r.headers.get("X-Outline-Source"); c.faces = Number(r.headers.get("X-Faces-Read"));
  } catch (e) { OVL.ocache.delete(k.key); OVL.api.log(`outline: ${e.message}`); return; }
  c.pending = false; OVL.api.draw();
}
function oupdate() {
  clearTimeout(OVL.otimer);
  if (!OVL.outlines || !OVL.api.BV.tiles.length) return;
  OVL.otimer = setTimeout(() => {
    for (const t of OVL.api.BV.tiles) { const k = okey(t); t.okey = k.key; if (!OVL.ocache.has(k.key)) oload(t, k); else { const c = OVL.ocache.get(k.key); OVL.ocache.delete(k.key); OVL.ocache.set(k.key, c); } }
  }, OUTLINE_WAIT_MS);
}
function odraw(g, t) {   // the segments of the tile's plane: the newest loaded cut of this plane, else nothing
  const mm = OVL.api.cross[t.n]; let c = t.okey && OVL.ocache.get(t.okey);
  if (!c || !c.seg) { c = null; for (const [k, v] of OVL.ocache) if (v.seg && k.startsWith(`${t.n}|${mm.toFixed(4)}|`)) c = v; }
  OVL.ostat.set(t.name, c ? c.source : ""); if (!c) return;
  const s = c.seg, w2px = OVL.api.w2px; g.strokeStyle = c.source === "overview" ? "rgba(255, 210, 63, 0.75)" : "rgba(255, 210, 63, 0.95)";
  g.lineWidth = 1.25; g.setLineDash(c.source === "overview" ? [4, 3] : []); g.beginPath();
  const a = [0, 0, 0], b = [0, 0, 0];
  for (let i = 0; i < s.length; i += 6) {
    a[0] = s[i]; a[1] = s[i + 1]; a[2] = s[i + 2]; b[0] = s[i + 3]; b[1] = s[i + 4]; b[2] = s[i + 5];
    const p = w2px(t, a), q = w2px(t, b); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]);
  }
  g.stroke(); g.setLineDash([]);
}

// ------------------------------------------------------------------ the hooks
const hooks = {
  async init(api) {
    OVL.api = api;
    try { OVL.vols = (await api.get("/volume/list")).filter(v => v.kind === "labels"); } catch { OVL.vols = []; }
    const v = controls();
    api.$("bvo").addEventListener("pointermove", e => {
      const r = api.$("bvo").getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const t = api.BV.tiles.find(t => x >= t.x && x < t.x + t.w && y >= t.y && y < t.y + t.h);
      OVL.cursor = t ? { t, mm: api.px2w(t, x - t.x, y - t.y) } : null; api.draw();
    });
    api.$("bvo").addEventListener("pointerleave", () => { OVL.cursor = null; api.draw(); });
    await choose(v);
  },
  update() { if (!OVL.api) return; const BV = OVL.api.BV; for (const l of OVL.show) l.update(BV.tiles, BV.s); oupdate(); },
  render(g, geoms, dpr) {
    if (!OVL.show.length || OVL.alpha <= 0) return;
    g.enable(g.BLEND); g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA);
    for (const l of OVL.show) l.render(geoms, OVL.alpha, OVL.edge, dpr);
    g.disable(g.BLEND);
  },
  draw2d(g, t) { if (OVL.outlines) odraw(g, t); },
  text(t) {   // the tile's text line: what is under the cursor, and the outline's source
    let s = "";
    if (OVL.cursor && OVL.cursor.t === t)
      for (const l of OVL.show) { const a = l.at(OVL.cursor.mm, l.tileL.get(t.name) ?? 0); s += ` · ${l.name}: ${a ? `${a.v} ${l.name_of(a.v)}` + (a.L ? ` (level ${a.L})` : "") : "loading"}`; }
    if (OVL.outlines && OVL.ostat.get(t.name) === "overview") s += " · outline: overview";
    return s;
  },
};
// for checks in the browser: the label of each shown layer at world point mm, at level 0 when loaded
OVL.labelAt = mm => Object.fromEntries(OVL.show.map(l => { const a = l.at(mm); return [l.name, a && { value: a.v, name: l.name_of(a.v), level: a.L, voxel: a.voxel }]; }));
(window.BVP = window.BVP || []).push(hooks);
