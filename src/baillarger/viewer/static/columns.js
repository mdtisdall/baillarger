// The columns pane (phase 4.3): the columns near the WM selection, from the published samplers (ray-sampler/1 and
// column-volume/1, through baillarger.viewer.columns), for what is on screen.
//
// - GET /columns?v=&radius=&every=&e_in=&e_out= gives the columns whose roots lie within the radius of the selected true
//   vertex along the surface; 202 means the server computes them (the first request on a superblock builds its local
//   graph): the pane asks again. GET /columns/profiles?(the same) gives their echoes.
// - The 3D surface view (VIEWER.hooks.draw3d): each column as a line, coloured by its derived class or by the class of the
//   label interval under each sample. Only on the true surface (level 0): an inflated level has another geometry.
// - The brick tiles (a plug-in of window.BVP, draw2d on #bvo): a dot where a column crosses the tile's plane.
// - The profile section: the columns whose roots lie within a half width of the plane through the selected root that holds
//   its normal and the section axis u, side by side in order of u, the echo along s (s = 0, the root, at the dashed line;
//   outward is down), with a tick at each label-interval boundary in the colour of the interval that starts there.
// - One column's details: its label intervals, |n̄|, its derived class. A click on the section picks the column.
// - The selection drives everything: the WM selection of the shared state, and the pane's settings and picked column
//   (state key "columns"), so every window shows the same columns.
const V = window.VIEWER;
const $ = V.$;
// the colours of the interval classes (OUTSIDE, WM, CORTEX, OTHER, OFF_GRID) and the derived classes (OPEN, IN_WM, NEITHER)
const ICOL = [[150, 150, 150], [235, 235, 235], [214, 60, 50], [60, 130, 230], [241, 196, 15]];
const INAME = ["outside", "WM", "cortex", "other", "off grid"];
const DCOL = { 1: [60, 200, 120], 2: [245, 140, 40], 3: [170, 110, 230] };
const DNAME = { 1: "open", 2: "in WM", 3: "neither" };
const C = { meta: null, data: null, prof: null, key: null, req: null, pick: -1, order: [], seq: 0, gl: null, nvMesh: null };

// ------------------------------------------------------------------ the pane's DOM
function build() {
  const p = $("p-columns");
  p.innerHTML = `
  <h2><span>Columns (ray sampler, column volume)</span><span id="cstat"></span></h2>
  <div class="pbar">
    <label>radius <select id="crad"><option>0.5</option><option>1</option><option selected>2</option><option>3</option><option>4</option><option>5</option></select> mm</label>
    <label>every <select id="cevery"><option>1</option><option>2</option><option>4</option><option>8</option></select></label>
    <label>inward <select id="cein"><option>0.5</option><option selected>1</option><option>2</option></select> mm</label>
    <label>outward <select id="ceout"><option>2</option><option selected>4</option><option>6</option><option>8</option></select> mm</label>
    <label>colour <select id="ccol"><option value="interval">label of each interval</option><option value="derived">derived class</option></select></label>
    <label><input type="checkbox" id="c3d" checked> 3D</label>
    <label><input type="checkbox" id="ctile" checked> tiles</label>
    <a id="ctck" href="#" download style="color:var(--mute)">.tck</a>
  </div>
  <div class="pbar">
    <label>section: half width <select id="chalf"><option>0.1</option><option selected>0.25</option><option>0.5</option><option value="1e9">all</option></select> mm</label>
    <label>echo <select id="cecho"><option value="0">1</option><option value="1" selected>2</option><option value="2">3</option></select></label>
    <label>show <select id="cshow"><option value="echo">echo</option><option value="class">label class</option></select></label>
    <span id="ckey" style="font-size:11px"></span>
  </div>
  <div class="fill" style="min-height:160px"><canvas id="csec"></canvas></div>
  <div id="cdet" style="flex:none;max-height:40%;overflow-y:auto;font-size:12px;padding:2px 8px"></div>`;
  $("ckey").innerHTML = ICOL.map((c, k) => `<span class="sw" style="background:rgb(${c})"></span>${INAME[k]}`).join(" ");
  for (const id of ["crad", "cevery", "cein", "ceout"]) $(id).onchange = () => { shareSettings(); load(); };
  for (const id of ["ccol", "c3d", "ctile"]) $(id).onchange = () => { redraw(); };
  for (const id of ["chalf", "cecho", "cshow"]) $(id).onchange = () => { section(); drawSection(); };
  $("csec").addEventListener("click", e => {
    const k = Math.floor(e.offsetX / ($("csec").clientWidth / Math.max(1, C.order.length)));
    if (k >= 0 && k < C.order.length) { setPick(C.order[k]); V.share({ columns: { ...settings(), column: vertexOf(C.pick) } }); }
  });
}
const settings = () => ({ radius: +$("crad").value, every: +$("cevery").value, e_in: +$("cein").value, e_out: +$("ceout").value });
function shareSettings() { V.share({ columns: { ...settings(), column: C.pick >= 0 ? vertexOf(C.pick) : null } }); }

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
async function load() {
  const sel = V.selection;
  if (!sel || sel.true_vertex === undefined || sel.true_vertex === null) return;
  const s = settings(), seq = ++C.seq;
  const q = `v=${sel.true_vertex}&radius=${s.radius}&every=${s.every}&e_in=${s.e_in}&e_out=${s.e_out}`;
  $("cstat").textContent = "loading…";
  try {
    const a = await fetchBinary(`/columns?${q}`, seq); if (!a) return;
    const [head, d] = decode(a.buf);
    C.head = head; C.data = d; C.key = head.key; C.prof = null;
    prepare();
    const want = C.wantVertex ?? sel.true_vertex; C.wantVertex = null;
    let pick = -1; for (let c = 0; c < head.n; c++) if (Number(d.vertex[c]) === want) pick = c;
    if (pick < 0) for (let c = 0; c < head.n; c++) if (pick < 0 || d.distance[c] < d.distance[pick]) pick = c;
    C.pick = pick;
    $("ctck").href = `/columns/${head.key}.tck`; $("ctck").setAttribute("download", `columns_${sel.true_vertex}.tck`);
    const st = `${head.n} columns, ${head.samples.toLocaleString()} samples · ${(a.ms / 1000).toFixed(2)} s ` +
      `(server: graph ${head.seconds.graph} s, sampler ${head.seconds.sample} s on ${head.seconds.calls.toLocaleString()} roots of superblock ${head.superblock.join(",")})`;
    $("cstat").textContent = st;
    section(); redraw(); details(); loadTck();
    const p = await fetchBinary(`/columns/profiles?${q}`, seq); if (!p) return;
    C.prof = decode(p.buf)[1];
    $("cstat").textContent = st + ` · profiles ${(p.ms / 1000).toFixed(2)} s`;
    V.log(`columns at vertex ${sel.true_vertex}: ${head.n} columns in ${(a.ms / 1000).toFixed(2)} s, profiles in ${(p.ms / 1000).toFixed(2)} s`);
    drawSection();
  } catch (e) { if (seq === C.seq) { $("cstat").textContent = e.message; V.log("columns: " + e.message); } }
}
// per sample: its column, and the class of the label interval that holds it; the roots in the 3D pane's frame (box mm)
function prepare() {
  const d = C.data, n = C.head.n, N = C.head.samples, M = C.head.to_view;
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
const colourOf = i => $("ccol").value === "derived" ? DCOL[C.data.derived_class[C.data.col[i]]] : ICOL[C.data.cls[i]];

// ------------------------------------------------------------------ 3D: the columns as lines
const LVS = `#version 300 es
in vec3 pos; in vec3 col; uniform mat4 mvp; out vec3 vcol; void main() { gl_Position = mvp * vec4(pos, 1.0); vcol = col; }`;
const LFS = `#version 300 es
precision mediump float; in vec3 vcol; uniform float alpha; out vec4 o; void main() { o = vec4(vcol, alpha); }`;
function lines3d() {   // the segments between consecutive samples of each column, with the colour of the first
  const d = C.data, n = C.head.n; let m = 0;
  for (let c = 0; c < n; c++) m += Number(d.sample_start[c + 1]) - Number(d.sample_start[c]) - 1;
  const pos = new Float32Array(6 * m), col = new Float32Array(6 * m); let o = 0;
  for (let c = 0; c < n; c++) for (let i = Number(d.sample_start[c]); i < Number(d.sample_start[c + 1]) - 1; i++) {
    const rgb = colourOf(i), w = c === C.pick ? 1 : 0.85;
    for (let e = 0; e < 2; e++) for (let k = 0; k < 3; k++) { pos[6*o+3*e+k] = d.view[3*(i+e)+k]; col[6*o+3*e+k] = w * rgb[k] / 255; }
    o++;
  }
  return { pos, col, m };
}
function upload3d() {
  const g = V.GL.g; if (!g || !C.data) return;
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
V.hooks.draw3d.push((g, M) => {
  if (!C.gl || !C.gl.m || !$("c3d").checked || V.GL.level !== 0) return;
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
    if (!C.data || !$("ctile").checked || !BVAPI) return;
    const d = C.data, n = t.n, c0 = BVAPI.cross[n], P = d.position;
    for (let c = 0; c < C.head.n; c++) {
      const a = Number(d.sample_start[c]), b = Number(d.sample_start[c + 1]);
      for (let i = a; i < b - 1; i++) {
        const u = P[3*i+n] - c0, v = P[3*(i+1)+n] - c0;
        if ((u > 0) === (v > 0) && u !== 0) continue;
        const f = u === v ? 0 : u / (u - v), w = [0, 1, 2].map(k => P[3*i+k] + f * (P[3*(i+1)+k] - P[3*i+k]));
        const [x, y] = BVAPI.w2px(t, w), rgb = colourOf(i), r = c === C.pick ? 3.5 : 2;
        g.fillStyle = `rgb(${rgb})`; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill();
        if (c === C.pick) { g.strokeStyle = "#fff"; g.lineWidth = 1; g.stroke(); }
      }
    }
  },
  text() { return C.data && $("ctile").checked ? ` · ${C.head.n} columns` : ""; },
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
function section() {
  C.order = []; if (!C.data || C.pick < 0) return;
  const d = C.data, p = C.pick, n0 = [0, 1, 2].map(k => d.normal[3*p+k]), r0 = [0, 1, 2].map(k => d.root[3*p+k]);
  const fr = V.frame, M = C.head.to_view;
  let u = [1, 0, 0];
  // a direction of the 3D frame in world mm: to_view's 3 x 3 part is diag(vox) A^-1 = R^T for A = R diag(vox), a rotation
  if (fr && fr.right) { const R = fr.right; u = [0, 1, 2].map(j => M[0][j] * R[0] + M[1][j] * R[1] + M[2][j] * R[2]); }
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  let k = dot(u, n0); u = u.map((x, i) => x - k * n0[i]);
  if (Math.hypot(...u) < 1e-3) { u = Math.abs(n0[2]) < 0.9 ? [-n0[1], n0[0], 0] : [0, -n0[2], n0[1]]; k = dot(u, n0); u = u.map((x, i) => x - k * n0[i]); }
  const L = Math.hypot(...u); u = u.map(x => x / L);
  const w = [n0[1] * u[2] - n0[2] * u[1], n0[2] * u[0] - n0[0] * u[2], n0[0] * u[1] - n0[1] * u[0]], half = +$("chalf").value;
  const rows = [];
  for (let c = 0; c < C.head.n; c++) {
    const q = [0, 1, 2].map(i => d.root[3*c+i] - r0[i]);
    if (Math.abs(dot(q, w)) <= half) rows.push([dot(q, u), c]);
  }
  rows.sort((a, b) => a[0] - b[0]); C.order = rows.map(r => r[1]); C.u = rows.map(r => r[0]);
}
function drawSection() {
  const cv = $("csec"), dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.fillStyle = "#000"; g.fillRect(0, 0, W, H);
  if (!C.data || !C.order.length) return;
  const d = C.data, s = settings(), top = 14, bot = H - 16, ys = v => top + (v + s.e_in) / (s.e_in + s.e_out) * (bot - top);
  const nc = C.order.length, cw = W / nc, e = +$("cecho").value, show = $("cshow").value, prof = C.prof;
  let lo = 0, hi = 1;
  if (prof && show === "echo") {   // the window: the 1st and 99th percentiles of the shown samples
    const v = []; for (const c of C.order) for (let i = Number(d.sample_start[c]); i < Number(d.sample_start[c + 1]); i++) if (prof.valid[i]) v.push(prof.echo[3*i+e]);
    v.sort((a, b) => a - b); if (v.length) { lo = v[Math.floor(0.01 * (v.length - 1))]; hi = v[Math.floor(0.99 * (v.length - 1))]; }
  }
  const dy = (bot - top) / ((s.e_in + s.e_out) / C.head.step);
  C.order.forEach((c, x) => {
    for (let i = Number(d.sample_start[c]); i < Number(d.sample_start[c + 1]); i++) {
      let rgb;
      if (show === "class") rgb = ICOL[d.cls[i]];
      else if (!prof) continue;
      else if (!prof.valid[i]) rgb = [40, 0, 0];
      else { const t = Math.round(255 * Math.min(1, Math.max(0, (prof.echo[3*i+e] - lo) / (hi - lo || 1)))); rgb = [t, t, t]; }
      g.fillStyle = `rgb(${rgb})`; g.fillRect(x * cw, ys(d.s[i]) - dy / 2, Math.ceil(cw), Math.ceil(dy));
    }
    // the label-interval boundaries: a tick in the colour of the interval that starts there
    for (let q = Number(d.interval_start[c]) + 1; q < Number(d.interval_start[c + 1]); q++) {
      const v = d.interval_s0[q]; if (v < -s.e_in || v > s.e_out) continue;
      g.fillStyle = `rgb(${ICOL[d.interval_class[q]]})`; g.fillRect(x * cw, ys(v) - 1, Math.max(1, cw), 2);
    }
  });
  g.strokeStyle = "#ff2bd6"; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(0, ys(0)); g.lineTo(W, ys(0)); g.stroke(); g.setLineDash([]);
  const k = C.order.indexOf(C.pick);
  if (k >= 0) { g.strokeStyle = "#fff"; g.strokeRect(k * cw - 0.5, top - 2, cw + 1, bot - top + 4); }
  g.fillStyle = "#9aa0a8"; g.font = "11px system-ui"; g.textBaseline = "top";
  g.fillText(`s = −${s.e_in} mm (inward)`, 4, 1); g.textBaseline = "bottom";
  g.fillText(`s = ${s.e_out} mm · ${nc} columns, u from ${C.u[0].toFixed(2)} to ${C.u[nc - 1].toFixed(2)} mm` +
    (show === "echo" ? ` · echo ${e + 1}, window ${lo.toFixed(0)}–${hi.toFixed(0)}` + (prof ? "" : " (loading)") : ""), 4, H - 1);
}

// ------------------------------------------------------------------ one column's details
function details() {
  const el = $("cdet"); if (!C.data || C.pick < 0) { el.textContent = ""; return; }
  const d = C.data, c = C.pick;
  let t = `<b>column of vertex ${vertexOf(c)}</b> · edge key ${d.edge_key[c]} · ${d.owned[c] ? "owned" : "halo"} column of superblock ` +
    `${C.head.superblock.join(",")} · ${d.distance[c].toFixed(3)} mm from the selection · |n̄| ${d.normal_length[c].toFixed(3)} · ` +
    `derived class <span style="color:rgb(${DCOL[d.derived_class[c]]})">${DNAME[d.derived_class[c]]}</span>` +
    `<table style="border-collapse:collapse;margin-top:3px"><tr style="color:var(--mute)"><th>s from (mm)</th><th>s to</th><th>label</th><th>class</th></tr>`;
  for (let q = Number(d.interval_start[c]); q < Number(d.interval_start[c + 1]); q++)
    t += `<tr><td style="text-align:right;padding:0 8px">${d.interval_s0[q].toFixed(3)}</td><td style="text-align:right;padding:0 8px">${d.interval_s1[q].toFixed(3)}</td>` +
      `<td style="text-align:right;padding:0 8px">${d.interval_label[q]}</td><td style="color:rgb(${ICOL[d.interval_class[q]]})">${INAME[d.interval_class[q]]}</td></tr>`;
  el.innerHTML = t + "</table>";
}
function setPick(c) { C.pick = c; section(); drawSection(); details(); redraw(); }
function redraw() { if (C.data) upload3d(); V.draw3d(); redrawTiles(); }

// ------------------------------------------------------------------ the shared state
V.hooks.selection.push(() => { if (C.meta) load(); });
V.hooks.state.push(st => {
  if (!("columns" in st) || !st.columns || !C.meta) return;
  const s = st.columns, old = settings();
  for (const [id, k] of [["crad", "radius"], ["cevery", "every"], ["cein", "e_in"], ["ceout", "e_out"]]) if (k in s) $(id).value = String(s[k]);
  const same = Object.keys(old).every(k => old[k] === settings()[k]);
  if (same && C.data && s.column !== null && s.column !== undefined) {
    for (let c = 0; c < C.head.n; c++) if (vertexOf(c) === s.column) { setPick(c); return; }
  }
  C.wantVertex = s.column; if (!same) load();
});
V.hooks.resize.push(() => drawSection());
(async () => {
  await V.ready;
  // the pane's settings drive the 3D view and the tiles too, so it is built also when it is not shown (?panes=volume)
  if (!(await fetch("/columns/meta")).ok) return;
  build();
  C.meta = await V.get("/columns/meta");
  const r = $("crad"); for (const o of [...r.options]) if (+o.value > C.meta.max_radius) o.remove();
  V.log(`columns pane: ${C.meta.spec}, radius up to ${C.meta.max_radius} mm, at most ${C.meta.max_columns} columns a request, ` +
    (C.meta.echoes.length ? `${C.meta.echoes.length} echoes` : "no echoes (no profiles: start with --echo-dir)"));
  drawSection();
  if (V.selection) load();
})();
