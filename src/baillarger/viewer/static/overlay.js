// The overlay (wave 2.1, Part B): any per-vertex scalar map on the white surface (a GIFTI file given with --overlay,
// served by baillarger.viewer.overlay) as the colours of the overview in the 3D surface pane. Nothing here knows what
// the maps are.
//
// - GET /overlay/meta: the files and their maps, with the 1st, 50th and 99th percentiles of each map's finite values.
//   A 404 (no --overlay) hides the overlay controls.
// - GET /overlay/<file>/<map>?agg=max|min: one float32 for each overview vertex: the largest (or smallest) value of the
//   overview vertex and of the true vertices that its faces carry, so that one true vertex is not lost in the overview.
//   The colours go to the 3D pane through the page's colour hook (VIEWER.hooks.colour); "none" gives the grey of before.
// - GET /overlay/<file>/<map>/sites?n=&spacing=&order=: the ranked list. A row click makes the site's true vertex the
//   WM selection (the shared state: every window, and the columns pane, follow) and turns the 3D view to it.
// - GET /overlay/at?v=: the values of every map at the selected true vertex, in a small table.
const V = window.VIEWER;
const $ = V.$;
// viridis (matplotlib's, CC0), at 9 points; linear between them
const VIRIDIS = ["440154", "472d7b", "3b528b", "2c728e", "21918c", "28ae80", "5ec962", "addc30", "fde725"]
  .map(h => [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255));
const NAN_RGB = [110 / 255, 110 / 255, 110 / 255];   // the grey of the overview with no overlay
const O = { meta: null, maps: [], cur: null, values: null, lo: 0, hi: 1, sites: [], seq: 0 };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = x => x === null || x === undefined || !Number.isFinite(x) ? "—" : Math.abs(x) >= 1e4 || (x !== 0 && Math.abs(x) < 1e-3) ? x.toExponential(3) : (+x.toPrecision(4)).toString();
const url = (m, tail = "") => `/overlay/${encodeURIComponent(m.file)}/${encodeURIComponent(m.map)}${tail}`;

function viridis(t) {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1), k = Math.min(VIRIDIS.length - 2, Math.floor(x)), f = x - k;
  const a = VIRIDIS[k], b = VIRIDIS[k + 1];
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
}

// ------------------------------------------------------------------ the controls (a bar in the surface pane) and the tables
function build() {
  const bar = document.createElement("div"); bar.className = "pbar"; bar.id = "ovbar";
  bar.innerHTML = `
    <label>overlay <select id="ovmap"><option value="">none</option></select></label>
    <span id="ovctl" style="display:contents"><label title="each overview vertex shows the largest (or smallest) value of its true vertex and of the true vertices its faces carry">
      <select id="ovagg"><option value="max">max</option><option value="min">min</option></select></label>
    <label>range <input id="ovlo" type="number" step="any" style="width:72px"> to <input id="ovhi" type="number" step="any" style="width:72px"></label>
    <button id="ovreset" title="the 1st to the 99th percentile">1–99 %</button>
    <canvas id="ovbarc" width="160" height="12" style="width:160px;height:12px;vertical-align:middle"></canvas>
    <span id="ovstat" style="color:var(--mute);font-size:11px"></span></span>`;
  $("sbar").after(bar);
  const panel = document.createElement("div"); panel.id = "ovpanel";
  panel.style.cssText = "flex:none;max-height:32%;display:none;gap:10px;padding:2px 8px 4px;font-size:12px;min-height:0";
  panel.innerHTML = `
    <div style="flex:1;min-width:0;display:flex;flex-direction:column;min-height:0">
      <div class="pbar" style="padding:0 0 2px"><b>ranked list</b>
        <label>order <select id="ovorder"><option value="high">high first</option><option value="low">low first</option></select></label>
        <label>n <select id="ovn"><option>50</option><option selected>200</option><option>500</option></select></label>
        <label title="drop the true vertices within this distance (mm) of a site">spacing <select id="ovsp"><option>0.5</option><option selected>1</option><option>2</option><option>5</option></select> mm</label>
        <span id="ovsstat" style="color:var(--mute)"></span></div>
      <div id="ovsites" style="overflow-y:auto;min-height:0;flex:1"></div>
    </div>
    <div style="flex:none;overflow-y:auto;min-height:0"><b>at the WM selection</b><div id="ovat"></div></div>`;
  $("p-surface").appendChild(panel);
  $("ovmap").onchange = choose;
  $("ovagg").onchange = choose;
  for (const id of ["ovlo", "ovhi"]) $(id).onchange = () => { O.lo = +$("ovlo").value; O.hi = +$("ovhi").value; paint(); };
  $("ovreset").onclick = () => { if (O.cur) { range(O.cur); paint(); } };
  for (const id of ["ovorder", "ovn", "ovsp"]) $(id).onchange = loadSites;
  $("ovsites").addEventListener("click", e => { const tr = e.target.closest("tr[data-k]"); if (tr) openSite(O.sites[+tr.dataset.k]); });
}

function range(m) {
  O.lo = m.p1 ?? 0; O.hi = m.p99 ?? 1; if (!(O.hi > O.lo)) O.hi = O.lo + 1;
  $("ovlo").value = fmt(O.lo); $("ovhi").value = fmt(O.hi);
}

function colourBar() {
  const c = $("ovbarc"), g = c.getContext("2d");
  for (let x = 0; x < c.width; x++) { const [r, gg, b] = viridis(x / (c.width - 1)); g.fillStyle = `rgb(${r * 255},${gg * 255},${b * 255})`; g.fillRect(x, 0, 1, c.height); }
}

// ------------------------------------------------------------------ a map on the overview
async function choose() {
  const k = $("ovmap").value, seq = ++O.seq;
  O.cur = k === "" ? null : O.maps[+k];
  $("ovpanel").style.display = O.cur ? "flex" : "none";
  $("ovctl").style.display = O.cur ? "contents" : "none";
  if (!O.cur) { O.values = null; $("ovstat").textContent = ""; V.recolour(); return; }
  range(O.cur);
  const t0 = performance.now();
  const r = await fetch(url(O.cur, `?agg=${$("ovagg").value}`));
  if (seq !== O.seq) return;
  if (!r.ok) { V.log(`overlay: ${r.status} for ${O.cur.file} ${O.cur.map}`); return; }
  O.values = new Float32Array(await r.arrayBuffer());
  const ms = performance.now() - t0;
  $("ovstat").textContent = `${O.cur.finite.toLocaleString()} of ${O.cur.n.toLocaleString()} true vertices finite · median ${fmt(O.cur.p50)}`;
  V.log(`overlay ${O.cur.file} ${O.cur.map} (${$("ovagg").value}): ${O.values.length.toLocaleString()} overview values in ${ms.toFixed(0)} ms`);
  paint();
  loadSites();
}

function paint() { V.recolour(); }

V.hooks.colour.push(nv => {
  if (!O.values || O.values.length !== nv) return null;
  const out = new Float32Array(3 * nv), d = O.hi - O.lo || 1;
  for (let v = 0; v < nv; v++) {
    const x = O.values[v], c = Number.isFinite(x) ? viridis((x - O.lo) / d) : NAN_RGB;
    out[3 * v] = c[0]; out[3 * v + 1] = c[1]; out[3 * v + 2] = c[2];
  }
  return out;
});

// ------------------------------------------------------------------ the ranked list
async function loadSites() {
  if (!O.cur) return;
  const seq = O.seq, m = O.cur, t0 = performance.now();
  $("ovsstat").textContent = "computing…"; $("ovsites").innerHTML = "";
  let j;
  try { j = await V.get(url(m, `/sites?n=${$("ovn").value}&spacing=${$("ovsp").value}&order=${$("ovorder").value}`)); }
  catch (e) { $("ovsstat").textContent = e.message; return; }
  if (seq !== O.seq || m !== O.cur) return;
  O.sites = j.sites;
  $("ovsstat").textContent = `${O.sites.length} sites (${(performance.now() - t0).toFixed(0)} ms)`;
  const sel = V.selection ? V.selection.true_vertex : -1;
  $("ovsites").innerHTML = `<table style="border-collapse:collapse;width:100%"><tr style="color:var(--mute)"><th>rank</th><th>value</th><th>vertex</th><th>mm</th></tr>` +
    O.sites.map((s, k) => `<tr data-k="${k}" style="cursor:pointer${s.vertex === sel ? ";background:#2f3a48" : ""}"><td>${s.rank}</td><td>${esc(fmt(s.value))}</td>` +
      `<td>${s.vertex}</td><td style="color:var(--mute)">${s.world_mm.map(x => x.toFixed(1)).join(", ")}</td></tr>`).join("") + "</table>";
}

// a site: the WM selection at its true vertex (the marker at a nearby overview vertex), the 3D view turned to face it
async function openSite(s) {
  // the marker's overview vertex: the server's, else the overview vertex nearest the site
  const ov = s.overview_vertex ?? (await V.get(`/snap?mm=${s.world_mm.join(",")}`)).vertex;
  const sel = { vertex: ov, true_vertex: s.vertex, world_mm: s.world_mm, voxel: s.voxel, trace: null, root_mm: null };
  V.turnTo(ov);
  await V.select(sel, V.screenFrame());
  for (const tr of $("ovsites").querySelectorAll("tr[data-k]")) tr.style.background = O.sites[+tr.dataset.k] === s ? "#2f3a48" : "";
}

// ------------------------------------------------------------------ the values at the WM selection
async function showAt(sel) {
  const el = $("ovat"); if (!el) return;
  if (!sel) { el.textContent = "no selection"; return; }
  let j; try { j = await V.get(`/overlay/at?v=${sel.true_vertex}`); } catch (e) { el.textContent = e.message; return; }
  let rows = "";
  for (const [file, maps] of Object.entries(j.values))
    for (const [map, vals] of Object.entries(maps))
      rows += `<tr${O.cur && O.cur.file === file && O.cur.map === map ? ' style="color:var(--acc)"' : ""}><td style="color:var(--mute)">${esc(file)}</td><td>${esc(map)}</td><td style="text-align:right;padding-left:8px">${esc(fmt(vals[0]))}</td></tr>`;
  el.innerHTML = `<div style="color:var(--mute)">true vertex ${sel.true_vertex}</div><table style="border-collapse:collapse">${rows}</table>`;
}
V.hooks.selection.push(sel => { if (O.meta) showAt(sel); });

// ------------------------------------------------------------------ start
(async () => {
  await V.ready;
  if (!V.has("surface")) return;
  const r = await fetch("/overlay/meta");
  if (!r.ok) return;   // no --overlay: no controls
  O.meta = await r.json();
  build(); colourBar();
  for (const f of O.meta.files) for (const m of f.maps) {
    const k = O.maps.push({ file: f.file, ...m }) - 1;
    const o = document.createElement("option"); o.value = String(k); o.textContent = `${f.file}: ${m.map}`; $("ovmap").appendChild(o);
  }
  await choose();
  V.log(`overlay: ${O.maps.length} maps in ${O.meta.files.length} files`);
  showAt(V.selection);
})();
