// The columns pane (phase 4.3): the columns near the WM selection, from the published samplers (ray-sampler/1 and
// column-volume/1, through baillarger.viewer.columns), for what is on screen.
//
// - GET /columns?v=&radius=&every=&e_in=&e_out= gives the columns whose roots lie within the radius of the selected true
//   vertex along the surface; 202 means the server computes them (the first request on a superblock builds its local
//   graph): the pane asks again. GET /columns/profiles?(the same) gives their echoes.
// - The 3D surface view (VIEWER.hooks.draw3d): each column as a line, coloured by its derived class or by the class of the
//   label interval under each sample. Only on the true surface (level 0): an inflated level has another geometry.
// - The brick tiles (a plug-in of window.BVP, draw2d on #bvo): a dot where a column crosses the tile's plane.
// - Which columns the 3D view and the tiles show: a menu in the surface pane's bar and one in the volume pane's bar,
//   "columns: off | section | all". section (the default): the columns of the section (one a step; mean across, the
//   columns of each step's mean); all: every column in the radius (the cap applies). The volume pane's menu acts on the
//   bricks viewer only (NiiVue keeps its .tck), so it is disabled with NiiVue.
// - The profile section that follows the surface: GET /columns/section (baillarger.viewer.section) cuts the true white
//   mesh with the plane through the picked column's root that holds its normal and the section axis u, and gives one step
//   for each mean spacing of the roots along that curve, in order of arc length, with the column nearest along the surface
//   ("nearest") or the columns within one voxel along the surface ("mean across": their mean echo). Side by side, the
//   echo along s (s = 0, the root, at the dashed line; outward is down); in "nearest", a tick at each label-interval
//   boundary in the colour of the interval that starts there.
// - The section's roots (a "section roots" checkbox in the surface pane's bar and one in the volume pane's bar, each
//   apart from the columns menu): the roots of the columns that the section shows (nearest: one a step; mean across: every
//   column of a step's mean), coloured by their step's place along the section (ARC, plasma from the left end of the
//   section to the right end; the same colours in a bar along the top of the section), with a dark rim; the picked
//   column's root in white. 3D view (VIEWER.points): on the true surface (level 0) at the roots themselves; on an inflated
//   level at the overview vertex of each root (GET /vertex/overview, as the WM selection marker). Bricks: the roots
//   within half a voxel of a tile's slice, as discs on that slice.
// - One column's details in a tooltip over the section (its true vertex, arc length, distance from the selection, |n̄|,
//   derived class, label intervals). A click on the section picks the column; one line under the section names it.
// - "every" thins only the displayed columns ("columns: all"): the section chooses from every root within the radius (the
//   server ignores every), so with every > 1 the pane asks /columns twice: with every for the display (C.all), and with
//   every=1 and cols= for the section's columns (C.data).
// - Over the cap (more than MAX_COLUMNS columns in the radius): /columns refuses, but the section does not depend on it.
//   The pane asks /columns/section, then /columns and /columns/profiles with cols= (the section's columns only), and says
//   so in red above the section: the 3D lines, the tiles and the .tck then show only the section's columns.
// - The selection drives everything: the WM selection of the shared state, and the pane's settings and picked column
//   (state key "columns": the picked column's true vertex and the selection it belongs to, `at`), so every window shows
//   the same columns. A new WM selection resets the picked column to the selection's; a picked column of another
//   selection is ignored, and one that is not in the request falls back to the selection's (the server's `fallback`).
const V = window.VIEWER;
const $ = V.$;
// the colours of the interval classes (OUTSIDE, WM, CORTEX, OTHER, OFF_GRID) and the derived classes (OPEN, IN_WM, NEITHER)
const ICOL = [[150, 150, 150], [235, 235, 235], [214, 60, 50], [60, 130, 230], [241, 196, 15]];
const INAME = ["outside", "WM", "cortex", "other", "off grid"];
const DCOL = { 1: [60, 200, 120], 2: [245, 140, 40], 3: [170, 110, 230] };
const DNAME = { 1: "open", 2: "in WM", 3: "neither" };
// the colours of the section's roots along the section: plasma (matplotlib's, CC0) at 9 points, from ARC[0] to ARC[1] of
// it (not its darkest end, which the dark slices would hide); sequential, readable with colour-blindness, neither white
// (the picked root) nor cyan (the WM selection)
const PLASMA = ["0d0887", "4c02a1", "7e03a8", "a92395", "cc4778", "e56b5d", "f89540", "fdc527", "f0f921"]
  .map(h => [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255));
const ARC = [0.12, 0.95];
function plasma(t) {
  const x = (ARC[0] + Math.min(1, Math.max(0, t)) * (ARC[1] - ARC[0])) * (PLASMA.length - 1);
  const k = Math.min(PLASMA.length - 2, Math.floor(x)), f = x - k, a = PLASMA[k], b = PLASMA[k + 1];
  return [0, 1, 2].map(i => a[i] + f * (b[i] - a[i]));
}
const css = c => `rgb(${c.map(x => Math.round(255 * x)).join(",")})`;
const esc = x => String(x).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const C = { meta: null, data: null, prof: null, key: null, req: null, pick: -1, sec: null, sseq: 0, seq: 0, gl: null, nvMesh: null,
  slice: null,   // { v: the selection's true vertex, n: the slice's normal } after a shift-click in a slice view
  pickV: null,   // the picked column's true vertex (the section's centre)
  vidx: new Map(),   // true vertex -> index in C.data
  secIdx: [],   // the section's columns at each step, as indices in C.data
  secSet: new Set(),   // the section's columns (indices in C.data)
  stepRGB: [],   // the colour of each step (plasma along the arc)
  capped: null,   // the server's refusal when the whole request is over the cap
  all: null,   // with every > 1: the displayed columns ({ head, data }; "columns: all"); C.data then holds the section's
               // columns only, with every = 1 (cols=). null: C.data is the whole request (every = 1) or over the cap
  loading: false,
  hl: null };   // the section's roots: { sec, cols, xyz (level 0), rgb, rootRGB (true vertex -> rgb), ov: overview vertices or null, pickOv }

// ------------------------------------------------------------------ the pane's DOM
function build() {
  const p = $("p-columns");
  p.innerHTML = `
  <h2><span>Columns (ray sampler, column volume)</span><span id="cstat"></span></h2>
  <div class="pbar">
    <label>radius <select id="crad"><option>0.5</option><option>1</option><option selected>2</option><option>3</option><option>4</option><option>5</option></select> mm</label>
    <label title="every n-th root: thins the displayed columns (columns: all, in the 3D view and the slices); the section always chooses from every root within the radius">every <select id="cevery"><option>1</option><option>2</option><option>4</option><option>8</option></select></label>
    <label>inward <select id="cein"><option>0.5</option><option selected>1</option><option>2</option></select> mm</label>
    <label>outward <select id="ceout"><option>2</option><option selected>4</option><option>6</option><option>8</option></select> mm</label>
    <label>colour <select id="ccol"><option value="interval">label of each interval</option><option value="derived">derived class</option></select></label>
    <a id="ctck" href="#" download style="color:var(--mute)">.tck</a>
  </div>
  <div class="pbar" id="ccap" style="display:none;color:#ff6b5b;font-size:12px"></div>
  <div class="pbar">
    <label title="nearest: at each step the column nearest along the surface; mean across: the mean of the columns within one voxel along the surface (it hides their own variation)">section
      <select id="cmode"><option value="nearest">nearest</option><option value="mean">mean across</option></select></label>
    <label>echo <select id="cecho"><option value="0">1</option><option value="1" selected>2</option><option value="2">3</option></select></label>
    <label>show <select id="cshow"><option value="echo">echo</option><option value="class">label class</option></select></label>
    <span id="ckey" style="font-size:11px"></span>
  </div>
  <div class="pbar" id="cplane" style="font-size:12px;color:var(--mute)"></div>
  <div class="fill" style="min-height:160px"><canvas id="csec"></canvas>
    <div id="ctip" style="display:none;position:fixed;z-index:50;pointer-events:none;background:#0d0f12f0;border:1px solid var(--line);
      border-radius:4px;padding:4px 7px;font-size:12px;white-space:nowrap"></div></div>
  <div id="cdet" style="flex:none;font-size:12px;padding:2px 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"></div>`;
  // the surface pane's bar and the volume pane's bar: which columns, and the section's roots
  const menu = (id, what) => `<label title="the columns ${what}: section, the columns of the profile section (one a step; mean across, the columns of each step's mean); all, every column within the radius (at most the cap); off, none">columns
    <select id="${id}"><option value="off">off</option><option value="section" selected>section</option><option value="all">all</option></select></label>`;
  const sb = document.createElement("span"); sb.style.display = "contents";
  sb.innerHTML = menu("c3d", "in the 3D view (the true surface only)") +
    `<label title="the roots of the profile section's columns, coloured by their place along the section (the bar at the top of the section, left to right); the picked column's root white"><input type="checkbox" id="chl" checked> section roots</label>
    <span id="chlnote" style="font-size:11px;color:var(--mute)"></span>`;
  $("sbar").appendChild(sb);
  const vb = document.createElement("span"); vb.style.display = "contents";
  vb.innerHTML = menu("ctile", "where they cross the slices") +
    `<label title="the roots of the profile section's columns within half a voxel of each slice, coloured by their place along the section; the picked column's root white"><input type="checkbox" id="chlv" checked> section roots</label>`;
  $("vwhite").closest(".pbar").appendChild(vb);
  if (!V.bricks) for (const id of ["ctile", "chlv"]) { const el = $(id); el.disabled = true;
    el.closest("label").title = "these act on the bricks viewer (viewer: bricks (ours)); NiiVue shows the columns as a .tck"; el.closest("label").style.opacity = 0.5; }
  $("ckey").innerHTML = ICOL.map((c, k) => `<span class="sw" style="background:rgb(${c})"></span>${INAME[k]}`).join(" ");
  for (const id of ["crad", "cevery", "cein", "ceout"]) $(id).onchange = () => { shareSettings(); load(); };
  $("ccol").onchange = () => { redraw(); };
  for (const id of ["c3d", "ctile", "chl", "chlv"]) $(id).onchange = () => { shareSettings(); redraw(); };
  $("cmode").onchange = () => { shareSettings(); section(); };
  for (const id of ["cecho", "cshow"]) $(id).onchange = () => drawSection();
  $("csec").addEventListener("click", e => {
    const S = C.sec; if (!S) return;
    const k = Math.floor(e.offsetX / ($("csec").clientWidth / Math.max(1, S.arc.length)));
    const c = k >= 0 && k < S.arc.length ? stepColumn(k) : -1;
    if (c >= 0) { setPick(c); shareSettings(); }
  });
  $("csec").addEventListener("mousemove", e => tooltip(e));
  $("csec").addEventListener("mouseleave", () => { $("ctip").style.display = "none"; });
}
// the display settings that every window shares (the columns menus, the roots checkboxes, the section's mode)
const display = () => ({ show3d: $("c3d").value, showTiles: $("ctile").value, roots3d: $("chl").checked, rootsVol: $("chlv").checked,
  mode: $("cmode").value });
const settings = () => ({ radius: +$("crad").value, every: +$("cevery").value, e_in: +$("cein").value, e_out: +$("ceout").value });
function shareSettings() {
  const sel = V.selection; V.share({ columns: { ...settings(), ...display(), column: C.pickV, at: sel ? sel.true_vertex : null } });
}

// ------------------------------------------------------------------ the binary answers (baillarger.viewer.columns.encode)
const TYPES = { int64: BigInt64Array, int32: Int32Array, uint8: Uint8Array, float32: Float32Array, float64: Float64Array };
function decode(buf) {
  const n = new Uint32Array(buf, 0, 1)[0], head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  const base = 4 + n, out = {};
  for (const t of head.arrays) {
    const T = TYPES[t.dtype], count = t.shape.reduce((a, b) => a * b, 1);
    out[t.name] = new T(buf, base + t.offset, count);
  }
  return [head, out];
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
// GET u: the binary answer, after asking again while the server answers 202 (a job); null when a newer request started
async function fetchBinary(u, seq) {
  const t0 = performance.now();
  for (;;) {
    const r = await fetch(u);
    if (seq !== C.seq) return null;
    if (r.status === 200) return { buf: await r.arrayBuffer(), ms: performance.now() - t0 };
    const j = await r.json();
    if (r.status !== 202) throw new Error(j.error || `HTTP ${r.status}`);
    $("cstat").textContent = j.status === "busy" ? "waiting for another request…" : `computing (${((performance.now() - t0) / 1000).toFixed(1)} s)…`;
    await sleep(400);
  }
}

// ------------------------------------------------------------------ loading
const vertexOf = c => Number(C.data.vertex[c]);
// every: the request's "every" (default: the menu's)
const query = (sel, every) => { const s = settings(); return `v=${sel.true_vertex}&radius=${s.radius}&every=${every ?? s.every}&e_in=${s.e_in}&e_out=${s.e_out}`; };
// a /columns answer as the pane's (section's) data
function setData(head, d) {
  C.head = head; C.data = d; C.prof = null; prepare(head, d);
  C.vidx = new Map(); for (let c = 0; c < head.n; c++) C.vidx.set(Number(d.vertex[c]), c);
}
function setTck(head) {   // the .tck of the displayed columns
  C.key = head.key; $("ctck").href = `/columns/${head.key}.tck`; $("ctck").setAttribute("download", `columns_${head.request.vertex}.tck`);
}
async function load() {
  const sel = V.selection;
  if (!sel || sel.true_vertex === undefined || sel.true_vertex === null) return;
  const s = settings(), seq = ++C.seq, q = query(sel);
  C.loading = true; C.data = null; C.all = null; C.prof = null; C.sec = null; C.hl = null; C.capped = null; C.pick = -1; C.vidx = new Map(); C.secSet = new Set();
  $("ccap").style.display = "none"; $("cstat").textContent = "loading…"; drawSection(); details(); redraw();
  try {
    let a;
    try { a = await fetchBinary(`/columns?${q}`, seq); }
    catch (e) { if (!/more than \d+/.test(e.message)) throw e; a = "capped"; C.capped = e.message.replace(/^(\w+Error: )+/, ""); }
    if (!a || seq !== C.seq) return;
    let st;
    if (a === "capped") {
      st = `radius ${s.radius} mm: over the cap`;
      $("ccap").textContent = `Radius ${s.radius} mm: ${C.capped}. The section still follows the surface; the 3D lines, the tiles ` +
        `and the .tck show only the section's columns.`;
      $("ccap").style.display = "";
    } else {
      const [head, d] = decode(a.buf);
      if (s.every === 1) setData(head, d); else { prepare(head, d); C.all = { head, data: d }; }
      setTck(head);
      st = `${head.n} columns, ${head.samples.toLocaleString()} samples · ${(a.ms / 1000).toFixed(2)} s ` +
        `(server: graph ${head.seconds.graph} s, sampler ${head.seconds.sample} s on ${head.seconds.calls.toLocaleString()} roots of superblock ${head.superblock.join(",")})`;
      loadTck();
    }
    $("cstat").textContent = st;
    C.loading = false;
    await section();
    if (C.capped || C.all || seq !== C.seq) return;   // else the section's columns came with their profiles
    const p = await fetchBinary(`/columns/profiles?${q}`, seq); if (!p) return;
    C.prof = decode(p.buf)[1];
    $("cstat").textContent = st + ` · profiles ${(p.ms / 1000).toFixed(2)} s`;
    V.log(`columns at vertex ${sel.true_vertex}: ${C.head.n} columns in ${(a.ms / 1000).toFixed(2)} s, profiles in ${(p.ms / 1000).toFixed(2)} s`);
    drawSection();
  } catch (e) { if (seq === C.seq) { $("cstat").textContent = e.message; V.log("columns: " + e.message, true); } }
  finally { if (seq === C.seq) C.loading = false; }
}
// per sample: its column, and the class of the label interval that holds it; the roots in the 3D pane's frame (box mm)
function prepare(head, d) {
  const n = head.n, N = head.samples, M = head.to_view;
  d.col = new Int32Array(N); d.cls = new Uint8Array(N);
  for (let c = 0; c < n; c++) {
    const a = Number(d.sample_start[c]), b = Number(d.sample_start[c + 1]), q0 = Number(d.interval_start[c]), q1 = Number(d.interval_start[c + 1]);
    let q = q0;
    for (let i = a; i < b; i++) { d.col[i] = c; while (q < q1 - 1 && d.s[i] >= d.interval_s1[q]) q++; d.cls[i] = d.interval_class[q]; }
  }
  d.view = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) { const x = d.position[3*i], y = d.position[3*i+1], z = d.position[3*i+2];
    for (let k = 0; k < 3; k++) d.view[3*i+k] = M[k][0] * x + M[k][1] * y + M[k][2] * z + M[k][3]; }
}
const colourOf = (d, i) => $("ccol").value === "derived" ? DCOL[d.derived_class[d.col[i]]] : ICOL[d.cls[i]];

// ------------------------------------------------------------------ 3D: the columns as lines
const LVS = `#version 300 es
in vec3 pos; in vec3 col; uniform mat4 mvp; out vec3 vcol; void main() { gl_Position = mvp * vec4(pos, 1.0); vcol = col; }`;
const LFS = `#version 300 es
precision mediump float; in vec3 vcol; uniform float alpha; out vec4 o; void main() { o = vec4(vcol, alpha); }`;
// the columns that a menu ("off", "section", "all") asks for: { d: their data, cols: indices into it }. all: the displayed
// columns (every n-th; over the cap, the section's)
function shown(mode) {
  if (mode === "all" && C.all) return { d: C.all.data, cols: Array.from({ length: C.all.head.n }, (_, c) => c) };
  if (!C.data || mode === "off") return { d: C.data, cols: [] };
  if (mode === "section") return { d: C.data, cols: [...C.secSet] };
  return { d: C.data, cols: Array.from({ length: C.head.n }, (_, c) => c) };
}
const isPick = (d, c) => Number(d.vertex[c]) === C.pickV;
function lines3d() {   // the segments between consecutive samples of each column, with the colour of the first
  const { d, cols } = shown($("c3d").value); let m = 0;
  for (const c of cols) m += Number(d.sample_start[c + 1]) - Number(d.sample_start[c]) - 1;
  const pos = new Float32Array(6 * m), col = new Float32Array(6 * m); let o = 0;
  for (const c of cols) for (let i = Number(d.sample_start[c]); i < Number(d.sample_start[c + 1]) - 1; i++) {
    const rgb = colourOf(d, i), w = isPick(d, c) ? 1 : 0.85;
    for (let e = 0; e < 2; e++) for (let k = 0; k < 3; k++) { pos[6*o+3*e+k] = d.view[3*(i+e)+k]; col[6*o+3*e+k] = w * rgb[k] / 255; }
    o++;
  }
  return { pos, col, m };
}
function upload3d() {
  const g = V.GL.g; if (!g || !(C.data || C.all)) return;
  if (!C.gl) { const p = V.program(g, LVS, LFS); C.gl = { p, vao: g.createVertexArray(), bp: g.createBuffer(), bc: g.createBuffer(), m: 0 };
    g.bindVertexArray(C.gl.vao);
    for (const [b, name] of [[C.gl.bp, "pos"], [C.gl.bc, "col"]]) { const loc = g.getAttribLocation(p, name); g.bindBuffer(g.ARRAY_BUFFER, b);
      g.enableVertexAttribArray(loc); g.vertexAttribPointer(loc, 3, g.FLOAT, false, 0, 0); }
    g.bindVertexArray(null); }
  const L = lines3d();
  g.bindBuffer(g.ARRAY_BUFFER, C.gl.bp); g.bufferData(g.ARRAY_BUFFER, L.pos, g.DYNAMIC_DRAW);
  g.bindBuffer(g.ARRAY_BUFFER, C.gl.bc); g.bufferData(g.ARRAY_BUFFER, L.col, g.DYNAMIC_DRAW);
  C.gl.m = L.m;
}
V.hooks.draw3d.push((g, M) => {   // the section's roots
  if (!C.meta) return;
  const hl = C.hl, on = hl && $("chl").checked, G = V.GL, rim = [0.04, 0.04, 0.05];
  const note = !on ? "" : G.level === 0 ? `${hl.xyz.length / 3} roots` :
    `${hl.sec.roots.length} roots, shown at their overview vertices (inflated surface)`;
  if ($("chlnote").textContent !== note) $("chlnote").textContent = note;
  if (!on) return;
  // the picked root first, then the coloured discs, then their dark rims: at equal depth a later disc fails the depth test
  if (G.level === 0) { V.points(g, M, hl.pick, [1, 1, 1], 9); V.points(g, M, hl.xyz, hl.rgb, 6); V.points(g, M, hl.xyz, rim, 8, 6); return; }
  if (!hl.ov) { if (!hl.asked) highlightOverview(hl); return; }
  const at = v => [G.pos[3*v], G.pos[3*v+1], G.pos[3*v+2]];
  // an overview vertex with several roots takes the colour of the first along the section
  const first = new Map(); hl.cols.forEach((c, i) => { const v = hl.ovOf.get(vertexOf(c)); if (v !== undefined && !first.has(v)) first.set(v, i); });
  const xyz = new Float32Array(3 * first.size), rgb = new Float32Array(3 * first.size); let i = 0;
  for (const [v, k] of first) { xyz.set(at(v), 3 * i); rgb.set(hl.rgb.subarray(3 * k, 3 * k + 3), 3 * i); i++; }
  V.points(g, M, new Float32Array(at(hl.pickOv)), [1, 1, 1], 9); V.points(g, M, xyz, rgb, 6); V.points(g, M, xyz, rim, 8, 6);
});
V.hooks.draw3d.push((g, M) => {
  if (!C.gl || !C.gl.m || V.GL.level !== 0) return;
  g.useProgram(C.gl.p); g.bindVertexArray(C.gl.vao); g.uniformMatrix4fv(g.getUniformLocation(C.gl.p, "mvp"), false, M);
  g.uniform1f(g.getUniformLocation(C.gl.p, "alpha"), 1); g.drawArrays(g.LINES, 0, 2 * C.gl.m);
  // the parts behind the surface, faded, as the points of the selection
  g.depthFunc(g.GREATER); g.depthMask(false); g.enable(g.BLEND); g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA);
  g.uniform1f(g.getUniformLocation(C.gl.p, "alpha"), 0.25); g.drawArrays(g.LINES, 0, 2 * C.gl.m);
  g.depthFunc(g.LESS); g.depthMask(true); g.disable(g.BLEND); g.bindVertexArray(null);
});

// ------------------------------------------------------------------ the brick tiles: where each column crosses the plane
const BVP = window.BVP = window.BVP || [];
let BVAPI = null;
BVP.push({
  init(api) { BVAPI = api; },
  draw2d(g, t) {
    if (!(C.data || C.all) || !BVAPI) return;
    const { d, cols } = shown($("ctile").value), n = t.n, c0 = BVAPI.cross[n];
    for (const c of cols) {
      const P = d.position;
      const a = Number(d.sample_start[c]), b = Number(d.sample_start[c + 1]);
      for (let i = a; i < b - 1; i++) {
        const u = P[3*i+n] - c0, v = P[3*(i+1)+n] - c0;
        if ((u > 0) === (v > 0) && u !== 0) continue;
        const f = u === v ? 0 : u / (u - v), w = [0, 1, 2].map(k => P[3*i+k] + f * (P[3*(i+1)+k] - P[3*i+k]));
        const [x, y] = BVAPI.w2px(t, w), rgb = colourOf(d, i), pk = isPick(d, c), r = pk ? 3.5 : 2;
        g.fillStyle = `rgb(${rgb})`; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill();
        if (pk) { g.strokeStyle = "#fff"; g.lineWidth = 1; g.stroke(); }
      }
    }
    // the section's roots within half a voxel of the slice; the picked root last, in white
    const hl = C.hl; t.nroots = 0; if (!hl || !C.data || !$("chlv").checked) return;
    const half = BVAPI.BV.step[n] / 2, R = C.data.root;
    g.lineWidth = 1.5; g.strokeStyle = "rgba(10, 10, 12, 0.95)";
    const disc = (c, fill, r) => {
      if (Math.abs(R[3*c+n] - c0) > half) return false;
      const [x, y] = BVAPI.w2px(t, [R[3*c], R[3*c+1], R[3*c+2]]);
      g.fillStyle = fill; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill(); g.stroke(); return true;
    };
    hl.cols.forEach((c, i) => { if (c !== C.pick && disc(c, css(hl.rgb.subarray(3 * i, 3 * i + 3)), 4)) t.nroots++; });
    if (C.pick >= 0 && disc(C.pick, "#fff", 4.5)) t.nroots++;
  },
  text(t) {
    const n = shown($("ctile").value).cols.length;
    return (n ? ` · ${n} columns` : "") + (C.hl && $("chlv").checked ? ` · ${t.nroots || 0} section roots in this slice` : "");
  },
});
function redrawTiles() { if (BVAPI) BVAPI.draw(); }

// ------------------------------------------------------------------ NiiVue: the columns as .tck (when NiiVue is the volume pane)
async function loadTck() {
  const nv = V.NV && V.NV.nv; if (!nv || !C.key || !nv.addMeshFromUrl) return;
  try {
    if (C.nvMesh && nv.meshes.includes(C.nvMesh)) nv.removeMesh(C.nvMesh);
    C.nvMesh = await nv.addMeshFromUrl({ url: `/columns/${C.key}.tck`, name: `columns_${C.key}.tck`, rgba255: [255, 120, 60, 255] });
    nv.drawScene();
  } catch (e) { V.log("columns .tck in NiiVue: " + e.message); }
}

// ------------------------------------------------------------------ the profile section
// the section axis u: the 3D view's screen right at the time of the selection, in the plane normal to the picked column
// (world mm); the server makes the curve and picks the columns (GET /columns/section)
function axisU() {
  const fr = V.frame, M = C.meta.to_view;
  let u = [1, 0, 0];
  // a direction of the 3D frame in world mm: to_view's 3 x 3 part is diag(vox) A^-1 = R^T for A = R diag(vox), a rotation
  if (fr && fr.right) { const R = fr.right; u = [0, 1, 2].map(j => M[0][j] * R[0] + M[1][j] * R[1] + M[2][j] * R[2]); }
  const c = C.vidx.get(C.pickV); if (!C.data || c === undefined) return u;   // the server makes u normal to n
  const d = C.data, n0 = [0, 1, 2].map(k => d.normal[3*c+k]);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  let k = dot(u, n0); u = u.map((x, i) => x - k * n0[i]);
  if (Math.hypot(...u) < 1e-3) { u = Math.abs(n0[2]) < 0.9 ? [-n0[1], n0[0], 0] : [0, -n0[2], n0[1]]; k = dot(u, n0); u = u.map((x, i) => x - k * n0[i]); }
  const L = Math.hypot(...u); return u.map(x => x / L);
}
// the section through the picked column (C.pickV; the selection's when none); over the cap, the section's columns are
// loaded with cols=
async function section() {
  const sel = V.selection; if (!sel || C.loading || !C.meta) return;
  const seq = ++C.sseq, lseq = C.seq; C.sec = null; C.hl = null; drawSection(); $("ctip").style.display = "none";
  if (C.pickV === null || C.pickV === undefined) C.pickV = sel.true_vertex;
  const q = query(sel), u = axisU();
  let j;
  try {
    // a shift-click in a slice view: the slice through the selection's root, while the section is centred there
    const sl = C.slice && C.slice.v === sel.true_vertex && C.pickV === sel.true_vertex ? `&plane=${C.slice.n.join(",")}` : "";
    const r = await fetch(`/columns/section?${q}&u=${u.map(x => x.toFixed(6)).join(",")}&mode=${$("cmode").value}&at=${C.pickV}${sl}`);
    j = await r.json(); if (seq !== C.sseq || lseq !== C.seq) return;
    if (r.status !== 200) throw new Error(j.error || `HTTP ${r.status}`);
    if (j.fallback) V.log(`section: ${j.fallback}; centred on the selection's column`);
    C.pickV = j.vertex;
    const need = [...new Set([...j.roots, j.vertex])];
    if ((C.capped || C.all) && need.some(v => !C.vidx.has(v))) {   // the section's columns alone, from every root
      const cq = `${query(sel, 1)}&cols=${need.join(",")}`;
      const a = await fetchBinary(`/columns?${cq}`, lseq); if (!a || seq !== C.sseq) return;
      const [head, d] = decode(a.buf); setData(head, d); if (C.capped) { setTck(head); loadTck(); }
      const p = await fetchBinary(`/columns/profiles?${cq}`, lseq); if (!p || seq !== C.sseq) return;
      C.prof = decode(p.buf)[1];
      const what = C.capped ? `radius ${settings().radius} mm: over the cap` : `${C.all.head.n} columns shown (every ${settings().every})`;
      $("cstat").textContent = `${what} · the section's ${head.n} columns in ${((a.ms + p.ms) / 1000).toFixed(2)} s`;
    }
  } catch (e) { if (seq === C.sseq) { V.log("section: " + e.message, true); $("cstat").textContent = "section: " + e.message; } return; }
  const S = j.slice, ax = S ? ["sagittal", "coronal", "axial"][S.plane.findIndex(x => Math.abs(x) > 0.5)] ?? "slice" : "";
  $("cplane").textContent = !S ? "section plane: the root's normal and the 3D view's screen right" :
    S.used ? `section plane: the ${ax} slice through the root · the root's normal is ${S.angle.toFixed(1)}° out of the slice` :
    `section plane: not the ${ax} slice (${S.fallback}); the root's normal and the screen right instead · the normal is ${S.angle.toFixed(1)}° out of the slice`;
  C.sec = j; C.secIdx = j.step_vertices.map(vs => vs.map(v => C.vidx.get(v)));
  C.secSet = new Set(C.secIdx.flat().filter(c => c !== undefined));
  const a0 = j.arc[0], a1 = j.arc[j.arc.length - 1];
  C.stepRGB = j.arc.map(a => plasma(a1 > a0 ? (a - a0) / (a1 - a0) : 0.5));
  C.pick = C.vidx.get(j.vertex) ?? -1;
  V.log(`section: ${j.arc.length} steps of ${j.step.toFixed(3)} mm (${j.mode}), the curve ${j.curve_length.map(x => x.toFixed(2)).join(" + ")} mm, ` +
    `${j.n.toLocaleString()} columns in the radius${j.capped ? " (over the cap)" : ""}, ${j.seconds} s`);
  drawSection(); highlight(); details(); redraw();
}
// the section's roots: in the order of the steps, each with the colour of the first step that holds it; their positions
// on the true surface (box mm); the overview vertices on demand
function highlight() {
  const S = C.sec, d = C.data; C.hl = null; if (!S || !d) { V.draw3d(); return; }
  const cols = [], ks = [], seen = new Set();
  C.secIdx.forEach((cs, k) => { for (const c of cs) if (c !== undefined && !seen.has(c)) { seen.add(c); cols.push(c); ks.push(k); } });
  const M = C.head.to_view, xyz = new Float32Array(3 * cols.length), rgb = new Float32Array(3 * cols.length);
  const view = c => [0, 1, 2].map(k => M[k][0] * d.root[3*c] + M[k][1] * d.root[3*c+1] + M[k][2] * d.root[3*c+2] + M[k][3]);
  cols.forEach((c, i) => { xyz.set(view(c), 3 * i); rgb.set(C.stepRGB[ks[i]], 3 * i); });
  C.hl = { sec: S, cols, xyz, rgb, pick: new Float32Array(C.pick >= 0 ? view(C.pick) : []), ov: null, ovOf: null, pickOv: -1, asked: false };
  V.draw3d();
}
async function highlightOverview(hl) {   // the overview vertex of each root and of the picked root (inflated levels)
  hl.asked = true;
  try {
    const j = await V.get(`/vertex/overview?v=${[...hl.sec.roots, hl.sec.vertex].join(",")}`);
    if (C.hl !== hl) return;
    hl.ov = j.vertex.slice(0, -1); hl.pickOv = j.vertex[j.vertex.length - 1];
    hl.ovOf = new Map(hl.sec.roots.map((v, i) => [v, hl.ov[i]])); V.draw3d();
  } catch (e) { V.log("section roots: " + e.message, true); }
}
// the column a step stands for: its own (nearest), or the one whose root is nearest the step's point (mean across)
function stepColumn(k) {
  const cs = C.secIdx[k], p = C.sec.point[k], d = C.data; if (!cs.length) return -1;
  let best = cs[0], bd = Infinity;
  for (const c of cs) { const r = Math.hypot(d.root[3*c] - p[0], d.root[3*c+1] - p[1], d.root[3*c+2] - p[2]); if (r < bd) { bd = r; best = c; } }
  return best;
}
function drawSection() {
  const cv = $("csec"); if (!cv) return;   // a resize before the pane is built
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.fillStyle = "#000"; g.fillRect(0, 0, W, H);
  const S = C.sec; if (!C.data || !S || !S.arc.length) return;
  const d = C.data, s = settings(), top = 21, bot = H - 16, ys = v => top + (v + s.e_in) / (s.e_in + s.e_out) * (bot - top);
  const nc = S.arc.length, cw = W / nc, e = +$("cecho").value, show = $("cshow").value, prof = C.prof, step = C.head.step;
  const all = [...new Set(C.secIdx.flat())];
  let lo = 0, hi = 1;
  if (prof && show === "echo") {   // the window: the 1st and 99th percentiles of the samples of the section's columns
    const v = []; for (const c of all) for (let i = Number(d.sample_start[c]); i < Number(d.sample_start[c + 1]); i++) if (prof.valid[i]) v.push(prof.echo[3*i+e]);
    v.sort((a, b) => a - b); if (v.length) { lo = v[Math.floor(0.01 * (v.length - 1))]; hi = v[Math.floor(0.99 * (v.length - 1))]; }
  }
  const dy = (bot - top) / ((s.e_in + s.e_out) / step);
  const grey = x => { const t = Math.round(255 * Math.min(1, Math.max(0, (x - lo) / (hi - lo || 1)))); return [t, t, t]; };
  C.secIdx.forEach((cs, x) => {
    if (!cs.length) return;
    // per sample k (s = k step): the mean of the valid echoes of the step's columns, and their classes
    const acc = new Map();
    for (const c of cs) for (let i = Number(d.sample_start[c]); i < Number(d.sample_start[c + 1]); i++) {
      const k = Math.round(d.s[i] / step); let a = acc.get(k); if (!a) acc.set(k, a = { sum: 0, n: 0, m: 0, cls: d.cls[i], mixed: false });
      a.m++; if (d.cls[i] !== a.cls) a.mixed = true;
      if (prof && prof.valid[i]) { a.sum += prof.echo[3*i+e]; a.n++; }
    }
    for (const [k, a] of acc) {
      let rgb;
      if (show === "class") rgb = a.mixed ? [90, 90, 90] : ICOL[a.cls];
      else if (!prof) continue;
      else if (!a.n) rgb = [40, 0, 0];
      else rgb = grey(a.sum / a.n);
      g.fillStyle = `rgb(${rgb})`; g.fillRect(x * cw, ys(k * step) - dy / 2, Math.ceil(cw), Math.ceil(dy));
    }
    if (cs.length !== 1) return;
    // the label-interval boundaries: a tick in the colour of the interval that starts there
    const c = cs[0];
    for (let q = Number(d.interval_start[c]) + 1; q < Number(d.interval_start[c + 1]); q++) {
      const v = d.interval_s0[q]; if (v < -s.e_in || v > s.e_out) continue;
      g.fillStyle = `rgb(${ICOL[d.interval_class[q]]})`; g.fillRect(x * cw, ys(v) - 1, Math.max(1, cw), 2);
    }
  });
  g.strokeStyle = "#ff2bd6"; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(0, ys(0)); g.lineTo(W, ys(0)); g.stroke(); g.setLineDash([]);
  const k0 = S.arc.findIndex(a => Math.abs(a) < 1e-9);   // the step at the picked root
  if (k0 >= 0) { g.strokeStyle = "#fff"; g.strokeRect(k0 * cw - 0.5, top - 2, cw + 1, bot - top + 4); }
  // the colours of the section's roots in the other panes, step by step, left end to right end
  C.stepRGB.forEach((c, x) => { g.fillStyle = css(c); g.fillRect(x * cw, 0, Math.ceil(cw), 5); });
  g.fillStyle = "#9aa0a8"; g.font = "11px system-ui"; g.textBaseline = "top";
  g.fillText(`s = −${s.e_in} mm (inward)`, 4, 7); g.textBaseline = "bottom";
  g.fillText(`s = ${s.e_out} mm · ${nc} steps of ${S.step.toFixed(3)} mm along the surface, arc ${S.arc[0].toFixed(2)} to ${S.arc[nc - 1].toFixed(2)} mm, ` +
    `${all.length} columns (${S.mode === "mean" ? "mean within one voxel" : "nearest"})` +
    (show === "echo" ? ` · echo ${e + 1}, window ${lo.toFixed(0)}–${hi.toFixed(0)}` + (prof ? "" : " (loading)") : ""), 4, H - 1);
}

// ------------------------------------------------------------------ one column's details: a tooltip over the section
function details() {   // one line under the section: the picked column
  const el = $("cdet"); if (!C.data || C.pick < 0) { el.textContent = ""; return; }
  const d = C.data, c = C.pick;
  el.innerHTML = `picked: the column of true vertex <b>${vertexOf(c)}</b> (its root white in the other panes), ` +
    `${d.distance[c].toFixed(3)} mm from the selection · the mouse over the section shows each column's details`;
}
function tooltip(e) {
  const tip = $("ctip"), S = C.sec, cv = $("csec");
  if (!S || !C.data || !S.arc.length) { tip.style.display = "none"; return; }
  const k = Math.floor(e.offsetX / (cv.clientWidth / S.arc.length)), c = k >= 0 && k < S.arc.length ? stepColumn(k) : -1;
  if (c < 0) { tip.style.display = "none"; return; }
  const d = C.data, n = C.secIdx[k].length;
  let t = `<span class="sw" style="background:${css(C.stepRGB[k])};width:10px;height:10px"></span><b>true vertex ${vertexOf(c)}</b>` +
    (c === C.pick ? " (picked)" : "") + `<br>arc ${S.arc[k].toFixed(3)} mm · ${d.distance[c].toFixed(3)} mm from the selection` +
    `<br>|n̄| ${d.normal_length[c].toFixed(3)} · derived class <span style="color:rgb(${DCOL[d.derived_class[c]]})">${esc(DNAME[d.derived_class[c]])}</span>` +
    (n > 1 ? `<br><span style="color:var(--mute)">the step is the mean of ${n} columns; this one's root is nearest the step's point</span>` : "") +
    `<table style="border-collapse:collapse;margin-top:2px"><tr style="color:var(--mute)"><th style="text-align:left">label</th><th>s from</th><th>s to (mm)</th></tr>`;
  for (let q = Number(d.interval_start[c]); q < Number(d.interval_start[c + 1]); q++)
    t += `<tr><td style="color:rgb(${ICOL[d.interval_class[q]]});padding-right:8px">${d.interval_label[q]} ${esc(INAME[d.interval_class[q]])}</td>` +
      `<td style="text-align:right;padding:0 6px">${d.interval_s0[q].toFixed(3)}</td><td style="text-align:right;padding:0 6px">${d.interval_s1[q].toFixed(3)}</td></tr>`;
  tip.innerHTML = t + "</table>"; tip.style.display = "";
  // next to the cursor, inside the window (the section may be lower than the tooltip)
  const W = window.innerWidth, H = window.innerHeight, w = tip.offsetWidth, h = tip.offsetHeight;
  const x = e.clientX + 14 + w > W ? e.clientX - 14 - w : e.clientX + 14, y = Math.min(Math.max(0, H - h), e.clientY + 14);
  tip.style.left = `${Math.max(0, x)}px`; tip.style.top = `${Math.max(0, y)}px`;
}
function setPick(c) { C.pick = c; C.pickV = vertexOf(c); section(); details(); redraw(); }
function redraw() { if (C.data || C.all) upload3d(); else if (C.gl) C.gl.m = 0; V.draw3d(); redrawTiles(); }

// ------------------------------------------------------------------ the shared state
// a new WM selection (a click, the box, ?v=, another window): the picked column is the selection's
V.hooks.selection.push((sel, frame, how) => {
  C.pickV = sel ? sel.true_vertex : null;
  C.slice = sel && how && how.slice ? { v: sel.true_vertex, n: how.slice } : null;   // any other selection: the usual plane
  if (C.meta) load();
});
// the controls from the shared state's "columns"
function setControls(s) {
  for (const [id, k] of [["crad", "radius"], ["cevery", "every"], ["cein", "e_in"], ["ceout", "e_out"], ["c3d", "show3d"],
    ["ctile", "showTiles"], ["cmode", "mode"]]) if (k in s && s[k] !== null && s[k] !== undefined) $(id).value = String(s[k]);
  for (const [id, k] of [["chl", "roots3d"], ["chlv", "rootsVol"]]) if (typeof s[k] === "boolean") $(id).checked = s[k];
}
V.hooks.state.push(st => {
  if (!("columns" in st) || !st.columns || !C.meta) return;
  const s = st.columns, old = settings(), sel = V.selection, oldMode = $("cmode").value, oldDisp = JSON.stringify(display());
  setControls(s);
  const modeChanged = $("cmode").value !== oldMode, dispChanged = JSON.stringify(display()) !== oldDisp;
  const same = Object.keys(old).every(k => old[k] === settings()[k]);
  // a picked column counts only for the selection it was picked in (an older one is stale)
  const col = sel && s.at === sel.true_vertex && s.column !== null && s.column !== undefined ? s.column : null;
  if (!same) { if (col !== null) C.pickV = col; load(); return; }
  if (col !== null && col !== C.pickV) { C.pickV = col; if (C.sec) section(); return; }
  if (modeChanged) { if (C.sec) section(); return; }
  if (dispChanged) redraw();
});
V.hooks.resize.push(() => drawSection());
(async () => {
  await V.ready;
  // the pane's settings drive the 3D view and the tiles too, so it is built also when it is not shown (?panes=volume)
  if (!(await fetch("/columns/meta")).ok) return;
  build();
  C.meta = await V.get("/columns/meta");
  const r = $("crad"); for (const o of [...r.options]) if (+o.value > C.meta.max_radius) o.remove();
  // a new window: the settings of the others (the first state came before the pane was built)
  try { const st = (await V.get("/state")).state; if (st.columns) setControls(st.columns); } catch (e) { /* get() logs it */ }
  V.log(`columns pane: ${C.meta.spec}, radius up to ${C.meta.max_radius} mm, at most ${C.meta.max_columns} columns a request, ` +
    (C.meta.echoes.length ? `${C.meta.echoes.length} echoes` : "no echoes (no profiles: start with --echo-dir)"));
  drawSection();
  if (V.selection) load();
})();
