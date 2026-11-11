// The stream 1 review pane (phase 4.2): stream 1's queue of sites, the site answers with their previews, and scribbles,
// written to the edit log (wm-edits/1) by the server (baillarger.viewer.review); and the paint mode with its undo. A site
// answer and a paint write their labels into the label edits (label-edits/1), the diff that the dseg reads; a scribble does
// not. The pane holds no stream 1 logic: the server calls phase 1.3's code for the queue, the random sites, the choices,
// their previews, the log and the edits.
// "accept the fix" (accept-fix) shows at an override site only: it writes the site's fix's labels into the edits.
//
// The brush has two modes. paint: the labels of the menu (outside, WM, cortex and the eraser "input label", which gives the
// voxels their input label again); its submit writes into the edits (/review/paint), and undo (/review/undo) takes back the
// last paint of this viewer run. scribble: the classes of the label set; its submit is a record of the log only. At a site,
// "manual paint" opens the paint mode, and Submit sends the strokes as the site's answer. With --edits alone the server
// has no queue: the pane is paint only.
//
// A plug-in of the brick pane (window.BVP, as static/overlays.js): it draws the previews and the strokes on #bvo, takes the
// pointer in brush mode, and puts its own pane in place of the surface pane when the server has a review (/review/queue).
//
// Timing: opening a site starts its reviewer time; submit sends the seconds from opening to submitting. A scribble's time
// runs from its first stroke to its submit. A random site does not show that it is random until it is answered, so that
// its kind does not change how the reviewer looks (a coordinator decision for review).
const CHOICES = [   // [name, key, label]
  ["remove-strand", "1", "remove strand"], ["thicken", "2", "thicken"], ["fill-gap", "3", "fill gap"],
  ["widen-gap", "4", "widen gap"], ["leave", "5", "leave"], ["manual-paint", "6", "manual paint"],
  ["accept-fix", "7", "accept the fix"],
];
const SITE_PX = 10;   // px per voxel (at least) when a site opens
const COL = { answer: "rgba(64, 224, 255, 0.85)", correction: "rgba(255, 140, 0, 0.85)", site: "rgba(255, 255, 255, 0.9)" };
const STROKE_RGB = { wm: [245, 245, 245], cortex: [255, 70, 90], outside: [70, 140, 255], input: [255, 0, 255] };
const PREVIEW_POLL_MS = 400;
// the most preview voxels drawn on one tile; above it the tile draws the first MAX_DRAWN and says so. Measured with R.bench
// on the laptop (Chrome, 1200 x 800): 13,189 squares in 3.3 ms, so 50,000 take about 12.5 ms, within one frame at 60 fps
const MAX_DRAWN = 50000;
const R = window.REVIEW = {
  api: null, on: false, meta: null, sites: [], session: null, cur: null, t0: 0, choice: null, params: {}, preview: null,
  previewSeq: 0, brush: false, mode: "paint", cls: null, radius: 1, undoDepth: 0, strokes: [], stroke: null, s0: 0, inv: null, A: null, revealed: new Map(),
};

// ------------------------------------------------------------------ helpers
const $ = id => document.getElementById(id);
async function post(u, body) {
  const r = await fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json(); if (!r.ok || j.error) throw new Error(j.error || `${r.status}`); return j;
}
async function getj(u) { const r = await fetch(u); const j = await r.json(); if (!r.ok || j.error) throw new Error(j.error || `${r.status}`); return j; }
function say(m, bad) { const el = $("rvmsg"); el.textContent = m; el.style.color = bad ? "#ff6b6b" : "var(--mute)"; if (bad) R.api.log("review: " + m); }
const mm2vox = w => R.api.affApply(R.inv, w).map(x => Math.round(x));
const vox2mm = v => R.api.affApply(R.A, v);
// the voxel axis along world axis n (the grid's axis nearest it), and the other two
function vaxis(n) { let best = 0, k = 0; for (let a = 0; a < 3; a++) { const c = Math.abs(R.A[n][a]); if (c > best) { best = c; k = a; } } return k; }
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ------------------------------------------------------------------ the pane
function pane() {
  const p = document.createElement("div"); p.className = "pane"; p.id = "p-review";
  p.innerHTML = `<h2><span>WM review (stream 1)</span><span id="rvcount"></span></h2>
  <div class="pbar" id="rvsession"></div>
  <div class="pbar" id="rvsite" style="display:none">
    <b id="rvsname"></b>
    ${CHOICES.map(([c, k, l]) => `<button data-choice="${c}" title="key ${k}">${k} ${l}</button>`).join("")}
    <label title="key t: 1 or 2">radius <select id="rvthick"><option>1</option><option>2</option></select></label>
    <button id="rvsubmit" title="Enter">submit ⏎</button><button id="rvskip" title="key s">skip (s)</button>
  </div>
  <div class="pbar" id="rvprev" style="white-space:pre-wrap"></div>
  <div class="pbar" id="rvbrush">
    <label title="key b"><input type="checkbox" id="rvbon"> brush (b)</label>
    <label title="key p: paint writes into the label edits; scribble is a record of the log only">mode <select id="rvmode">
      <option value="paint">paint</option><option value="scribble">scribble</option></select></label>
    <label>label <select id="rvcls"></select></label>
    <label title="keys [ and ]">radius <select id="rvrad"><option>0</option><option selected>1</option><option>2</option><option>3</option><option>4</option></select> voxels</label>
    <button id="rvssub" title="key x">submit (x)</button><button id="rvsclr" title="Escape">clear</button>
    <button id="rvundo" title="key u: undo the last paint of this viewer run">undo (u)</button><span id="rvsn"></span>
  </div>
  <div class="pbar"><span id="rvmsg" style="color:var(--mute)"></span></div>
  <div id="rvtable" style="overflow-y:auto;flex:1;min-height:0;font-size:12px"></div>
  <div class="help" style="position:static;padding:2px 8px">keys: 1–7 choices · t thicken radius · Enter submit · s skip · n next ·
    b brush on/off · p paint/scribble · w WM · c cortex · o outside · e eraser (input label) · [ ] brush radius · x submit strokes ·
    u undo the last paint · Esc clear strokes. Row: open that site.
    Preview: <span style="color:${COL.answer}">■ the answer</span> <span style="color:${COL.correction}">■ the correction</span></div>`;
  const surf = $("p-surface"); surf.style.display = "none"; surf.parentElement.insertBefore(p, surf.nextSibling);
  for (const b of p.querySelectorAll("button[data-choice]")) b.onclick = () => choose(b.dataset.choice);
  $("rvthick").onchange = () => R.choice === "thicken" && choose("thicken");
  $("rvsubmit").onclick = submit; $("rvskip").onclick = skip;
  $("rvbon").onchange = () => { R.brush = $("rvbon").checked; R.api.draw(); };
  $("rvmode").onchange = () => setMode($("rvmode").value);
  $("rvcls").onchange = () => { R.cls = $("rvcls").value; };
  $("rvrad").onchange = () => { R.radius = Number($("rvrad").value); };
  $("rvssub").onclick = submitStrokes; $("rvsclr").onclick = clearStrokes; $("rvundo").onclick = undo;
}
function clearStrokes() { R.strokes = []; R.stroke = null; strokesInfo(); R.api.draw(); }
// the brush mode: paint (the menu of the paint mode) or scribble (the classes); strokes of the other mode must go first
function setMode(m, quiet) {
  if (m === "scribble" && R.meta.paint_only) m = "paint";
  if (m !== R.mode && (R.strokes.length || R.stroke)) {
    $("rvmode").value = R.mode; if (!quiet) say("submit or clear the strokes first (Esc)", true); return false;
  }
  R.mode = m; $("rvmode").value = m;
  const list = m === "paint" ? R.meta.menu : R.meta.classes;
  $("rvcls").innerHTML = list.map(c => `<option value="${esc(c.value)}" data-key="${esc(c.key || "")}">${esc(c.name)}${typeof c.value === "number" ? ` (${c.value})` : ""}</option>`).join("");
  pick("wm"); R.cls = $("rvcls").value; strokesInfo(); R.api.draw(); return true;
}
function sessionBar() {
  const s = R.session, el = $("rvsession");
  if (R.meta.paint_only) { el.innerHTML = `<b>Paint only</b> (no queue): the paint mode writes into the label edits ${esc(R.meta.edits)}`; return; }
  if (!s) {
    el.innerHTML = `<b>No session.</b> <label title="the share of random, unranked sites in this session (0 to 1; no default)">random share
      <input type="number" id="rvshare" min="0" max="1" step="0.05" placeholder="0–1" style="width:64px"></label>
      <label>seed <input type="number" id="rvseed" step="1" style="width:90px" value="${Math.floor(Math.random() * 2 ** 31)}"></label>
      <button id="rvstart">start session</button>`;
    $("rvstart").onclick = start; return;
  }
  el.innerHTML = `<b>Session ${esc(s.id)}</b> · random share ${s.random_share} · seed ${s.seed} · ${s.n_random} random + ${s.n_ranked} ranked sites
    <button id="rvnext" title="key n">next site (n)</button>`;
  $("rvnext").onclick = next;
}
function table() {
  if (R.meta.paint_only) { $("rvcount").textContent = ""; $("rvtable").innerHTML = ""; return; }
  const n = { "not shown": 0, shown: 0, answered: 0, skipped: 0 };
  for (const s of R.sites) n[s.state] = (n[s.state] || 0) + 1;
  $("rvcount").textContent = Object.entries(n).map(([k, v]) => `${v} ${k}`).join(" · ");
  const rows = R.sites.map(s => {
    const kind = s.kind === null ? "site" : s.kind;   // a random site before its answer: no kind
    return `<tr data-id="${esc(s.id)}" class="${R.cur && R.cur.id === s.id ? "cur" : ""}"><td>${esc(s.id)}</td><td style="text-align:left">${esc(kind)}</td>
      <td>${s.rank ?? ""}</td><td>${s.score === null || s.score === undefined ? "" : Number(s.score).toFixed(3)}</td><td class="st-${s.state.replace(" ", "")}">${s.state}</td></tr>`;
  }).join("");
  $("rvtable").innerHTML = `<table><tr><th>site</th><th style="text-align:left">kind</th><th>rank</th><th>score</th><th>state</th></tr>${rows}</table>`;
  for (const tr of $("rvtable").querySelectorAll("tr[data-id]")) tr.onclick = () => open(tr.dataset.id);
}
async function refresh() {
  const q = await getj("/review/queue");
  R.sites = q.sites; R.session = q.session; R.meta = q; R.undoDepth = q.undo_depth;
  sessionBar(); table(); strokesInfo();
}

// ------------------------------------------------------------------ a session, its sites, the answers
async function start() {
  const v = $("rvshare").value.trim(), share = Number(v), seed = Number($("rvseed").value);
  if (v === "" || !(share >= 0 && share <= 1)) { say("give the random share: a number from 0 to 1", true); return; }
  if (!Number.isInteger(seed)) { say("the seed must be an integer", true); return; }
  try { await post("/review/session", { random_share: share, seed }); } catch (e) { say(e.message, true); return; }
  await refresh(); say(`session started: random share ${share}, seed ${seed}`); await next();
}
async function next() {
  if (!R.session) { say("start a session first", true); return; }
  let r; try { r = await getj("/review/next"); } catch (e) { say(e.message, true); return; }
  if (!r.site) {
    say("no more sites in this session"); R.cur = null; R.choice = null; $("rvsite").style.display = "none"; $("rvprev").textContent = "";
    await refresh(); R.api.draw(); return;
  }
  show(r.site);
}
async function open(id) {
  if (!R.session) { say("start a session first", true); return; }
  let r; try { r = await getj(`/review/next?site=${encodeURIComponent(id)}`); } catch (e) { say(e.message, true); return; }
  show(r.site);
}
function show(site) {   // the site is shown (the server recorded it): move there, start its time
  R.cur = site; R.t0 = performance.now(); R.choice = null; R.preview = null; R.previewSeq++;
  const mm = site.mm, BV = R.api.BV; for (const t of BV.tiles) { BV.C[t.r] = mm[t.r]; BV.C[t.u] = mm[t.u]; }
  BV.s = Math.max(BV.s, SITE_PX / R.vs);   // at least SITE_PX px per voxel, so that the preview squares show; a closer zoom stays
  R.api.crossHere(mm); R.api.update();
  $("rvsite").style.display = ""; $("rvsname").textContent = `site ${site.id}` + (site.kind ? ` (${site.kind})` : "");
  for (const b of document.querySelectorAll("#rvsite button[data-choice]")) {
    b.style.outline = ""; if (b.dataset.choice === "accept-fix") b.style.display = site.kind === "override" ? "" : "none";
  }
  $("rvprev").textContent = "pick a choice to see its preview"; $("rvprev").style.color = "";
  const s = R.sites.find(x => x.id === site.id); if (s && s.state === "not shown") s.state = "shown"; table();
}
function params(c) { return c === "thicken" ? { radius: Number($("rvthick").value) } : {}; }
async function choose(c) {
  if (!R.cur) { say("open a site first", true); return; }
  if (c === "manual-paint" && !setMode("paint")) return;
  R.choice = c; R.params = params(c); const seq = ++R.previewSeq;
  for (const b of document.querySelectorAll("#rvsite button[data-choice]")) b.style.outline = b.dataset.choice === c ? "2px solid var(--acc)" : "";
  if (c === "manual-paint") {   // the paint mode: the strokes are the answer; there is no preview of strokes yet
    R.preview = null; $("rvbon").checked = true; R.brush = true; R.api.draw();
    $("rvprev").textContent = "manual paint: paint the strokes (left drag), then Submit (Enter) writes them as this site's answer" +
      " into the label edits and the log. Undo (u) takes them back from the edits."; $("rvprev").style.color = ""; return;
  }
  $("rvprev").textContent = `${c}: computing the preview…`;
  let r; try { r = await post("/review/preview", { site: R.cur.id, choice: c, params: R.params }); } catch (e) { if (seq === R.previewSeq) say(e.message, true); return; }
  while (r.status === "running") {   // a long preview runs as a job on the server
    await new Promise(ok => setTimeout(ok, PREVIEW_POLL_MS)); if (seq !== R.previewSeq) return;
    try { r = await getj(`/review/preview?job=${encodeURIComponent(r.job)}`); } catch (e) { say(e.message, true); return; }
  }
  if (seq !== R.previewSeq) return;
  if (r.status !== "done") { say(`preview: ${r.error || r.status}`, true); return; }
  R.preview = index(r); R.api.draw();
  const w = r.overrides ? "\n⚠ the correction overrides this answer: the automatic fix changes the voxels that the answer sets" : "";
  const cut = r.cut_by_box ? "\nthe preview's box cuts WM: the correction outside the box is not shown (approximate)" : "";
  $("rvprev").textContent = `${c}: the answer changes ${r.answer.length} voxels, then the correction changes ${r.correction.length}` +
    ` (${(r.seconds ?? 0).toFixed(2)} s)` + w + cut;
  $("rvprev").style.color = r.overrides ? "#ffd23f" : "";
}
// the preview voxels by slice: for each voxel axis a, a map from the voxel's index along a to its voxels
function index(r) {
  const out = { overrides: r.overrides, n: r.answer.length + r.correction.length, by: [new Map(), new Map(), new Map()] };
  for (const [kind, vs] of [["correction", r.correction], ["answer", r.answer]])
    for (const v of vs) for (let a = 0; a < 3; a++) { const m = out.by[a]; if (!m.has(v[a])) m.set(v[a], []); m.get(v[a]).push([kind, v]); }
  return out;
}
async function submit() {
  if (!R.cur || !R.choice) { say("open a site and pick a choice first", true); return; }
  const manual = R.choice === "manual-paint";
  if (manual && !R.strokes.length) { say("manual paint: paint at least one stroke first", true); return; }
  if (!manual && !R.preview) { say("wait for the preview", true); return; }
  const secs = (performance.now() - R.t0) / 1000, body = { site: R.cur.id, choice: R.choice, params: R.params, reviewer_seconds: secs };
  if (manual) body.strokes = paintStrokes();
  let r; try { r = await post("/review/answer", body); }
  catch (e) { say(e.message, true); return; }
  if (manual) { R.strokes = []; R.undoDepth = r.undo_depth; strokesInfo(); }
  const how = r.how === "random" ? "a random site" : `ranked ${r.rank ?? ""}`;
  say(`answer ${r.answer} written: ${R.choice} at site ${R.cur.id} (${r.kind}, ${how}), ${secs.toFixed(1)} s;` +
    ` ${r.edited} voxels written into the label edits`);
  const s = R.sites.find(x => x.id === R.cur.id); if (s) { s.state = "answered"; s.kind = r.kind; }
  R.preview = null; R.api.draw(); await refresh(); await next();
}
async function skip() {
  if (!R.cur) return;
  try { await post("/review/skip", { site: R.cur.id }); } catch (e) { say(e.message, true); return; }
  say(`site ${R.cur.id} skipped`); R.preview = null; await refresh(); await next();
}

// ------------------------------------------------------------------ scribbles
function strokesInfo() {
  const n = R.strokes.reduce((a, s) => a + s.voxels.size, 0);
  $("rvsn").textContent = (R.strokes.length ? `${R.strokes.length} ${R.mode} strokes, ${n} voxels · ` : "") + `undo: ${R.undoDepth} paints`;
  $("rvundo").disabled = !R.undoDepth;
}
const paintStrokes = () => R.strokes.map(s => ({ label: s.value, voxels: [...s.voxels.values()] }));
function dab(t, x, y) {   // the voxels within the brush radius of the cursor, in the tile's slice
  const c = mm2vox(R.api.px2w(t, x, y)), n = vaxis(t.n), ax = [0, 1, 2].filter(a => a !== n), r = R.radius;
  for (let i = -r; i <= r; i++) for (let j = -r; j <= r; j++) {
    if (i * i + j * j > r * r + 0.25) continue;
    const v = c.slice(); v[ax[0]] += i; v[ax[1]] += j; R.stroke.voxels.set(v.join(","), v);
  }
}
function paint(t, x, y) {   // dabs from the last point to (x, y), at most half a voxel apart, so a fast drag leaves no gaps
  const last = R.stroke.last || [x, y], step = Math.max(0.5, R.api.BV.s * R.vs / 2), d = Math.hypot(x - last[0], y - last[1]);
  const k = Math.max(1, Math.ceil(d / step));
  for (let i = 1; i <= k; i++) dab(t, last[0] + (x - last[0]) * i / k, last[1] + (y - last[1]) * i / k);
  R.stroke.last = [x, y];
}
// the brush's submit: at a site with manual paint, the site's answer; else a paint (paint mode) or a scribble
async function submitStrokes() {
  if (R.mode === "scribble") return scribble();
  if (R.cur && R.choice === "manual-paint") return submit();
  if (!R.strokes.length) { say("no strokes", true); return; }
  const strokes = paintStrokes();
  let r; try { r = await post("/review/paint", { strokes }); } catch (e) { say(e.message, true); return; }
  R.undoDepth = r.undo_depth;
  say(`paint written: ${r.voxels} of ${strokes.reduce((a, s) => a + s.voxels.length, 0)} voxels changed their label in the label edits` +
    " (the labels volume still shows the dseg)");
  R.strokes = []; strokesInfo(); R.api.draw();
}
async function undo() {
  if (R.strokes.length || R.stroke) { say("submit or clear the strokes first (Esc)", true); return; }
  let r; try { r = await post("/review/undo", {}); } catch (e) { say(e.message, true); return; }
  R.undoDepth = r.undo_depth; strokesInfo(); say(`undo: ${r.voxels} voxels of the label edits written back`);
}
async function scribble() {
  if (!R.strokes.length) { say("no strokes", true); return; }
  const secs = (performance.now() - R.s0) / 1000;
  const strokes = R.strokes.map(s => ({ class: s.value, voxels: [...s.voxels.values()] }));
  let r; try { r = await post("/review/scribble", { strokes, reviewer_seconds: secs }); } catch (e) { say(e.message, true); return; }
  say(`scribble ${r.answer} written: ${strokes.length} strokes, ${strokes.reduce((a, s) => a + s.voxels.length, 0)} voxels, ${secs.toFixed(1)} s`);
  R.strokes = []; strokesInfo(); R.api.draw();
}

// ------------------------------------------------------------------ drawing on #bvo
function squares(g, t, list, colour, cap = MAX_DRAWN) {   // list: [kind, voxel] or voxels; one square per voxel in the tile's slice
  const px = Math.max(1, R.api.BV.s * R.vs);
  let drawn = 0;
  for (const e of list) {
    if (drawn >= cap) return drawn;
    const [kind, v] = Array.isArray(e[1]) ? e : [null, e];
    const [x, y] = R.api.w2px(t, vox2mm(v)); if (x < -px || y < -px || x > t.w + px || y > t.h + px) continue;
    g.fillStyle = colour || COL[kind]; g.fillRect(x - px / 2, y - px / 2, px, px); drawn++;
  }
  return drawn;
}
function draw2d(g, t) {
  if (!R.on || !R.A) return;
  const n = vaxis(t.n), k = mm2vox(R.api.cross)[n];
  if (R.preview) {
    const list = R.preview.by[n].get(k) || [];
    const d = squares(g, t, list); t.rvcap = list.length > d && d >= MAX_DRAWN ? list.length : 0;
  }
  for (const s of [...R.strokes, ...(R.stroke ? [R.stroke] : [])]) {
    const rgb = STROKE_RGB[s.key] || [255, 210, 63], vs = [...s.voxels.values()].filter(v => v[n] === k);
    squares(g, t, vs, `rgba(${rgb.join(",")}, 0.6)`);
  }
  if (R.cur) {   // the site: a ring round its centre
    const [x, y] = R.api.w2px(t, R.cur.mm); g.strokeStyle = COL.site; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 9, 0, 2 * Math.PI); g.stroke();
  }
}
function text(t) {
  let s = "";
  if (R.brush) s += ` · ${R.mode}: ${$("rvcls").selectedOptions[0]?.textContent ?? R.cls} r ${R.radius}`;
  if (t.rvcap) s += ` · preview: ${MAX_DRAWN} of ${t.rvcap} voxels drawn`;
  return s;
}
function pointer(type, e, h) {   // brush mode: a left drag paints one stroke of the current label
  if (!R.on || !R.brush || e.button > 0 || e.shiftKey) return false;
  if (type === "down" && h) {
    if (!R.strokes.length && !R.stroke) R.s0 = performance.now();
    const c = $("rvcls"), opt = c.options[c.selectedIndex];
    const value = c.value === "input" ? "input" : Number(c.value);
    R.stroke = { value, key: opt.dataset.key, voxels: new Map(), tile: h.t }; paint(h.t, h.x, h.y); R.api.draw(); return true;
  }
  if (!R.stroke) return false;
  const t = R.stroke.tile, r = $("bvo").getBoundingClientRect(), x = e.clientX - r.left - t.x, y = e.clientY - r.top - t.y;
  if (type === "move") { paint(t, x, y); R.api.draw(); return true; }
  if (type === "up") { if (R.stroke.voxels.size) R.strokes.push(R.stroke); R.stroke = null; strokesInfo(); R.api.draw(); return true; }
  return false;
}

// ------------------------------------------------------------------ keys
function keys(e) {
  if (!R.on || e.metaKey || e.ctrlKey || e.altKey || ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  const c = CHOICES.find(x => x[1] === e.key && (x[0] !== "accept-fix" || (R.cur && R.cur.kind === "override")));
  const act = c ? () => choose(c[0]) : {
    Enter: () => (R.cur ? submit() : submitStrokes()), s: skip, n: next, x: submitStrokes, u: undo,
    p: () => setMode(R.mode === "paint" ? "scribble" : "paint"),
    o: () => pick("outside"), e: () => pick("input"),
    t: () => { $("rvthick").value = $("rvthick").value === "1" ? "2" : "1"; if (R.choice === "thicken") choose("thicken"); },
    b: () => { $("rvbon").checked = !$("rvbon").checked; $("rvbon").onchange(); },
    w: () => pick("wm"), c: () => pick("cortex"),
    "[": () => { $("rvrad").value = String(Math.max(0, R.radius - 1)); $("rvrad").onchange(); },
    "]": () => { $("rvrad").value = String(Math.min(4, R.radius + 1)); $("rvrad").onchange(); },
    Escape: () => $("rvsclr").onclick(),
  }[e.key];
  if (!act) return;
  e.preventDefault(); e.stopPropagation(); act();
}
function pick(key) { const o = [...$("rvcls").options].find(o => o.dataset.key === key); if (o) { $("rvcls").value = o.value; R.cls = o.value; R.api.draw(); } }

// ------------------------------------------------------------------ the hooks
const hooks = {
  async init(api) {
    if (R.api) return;   // once
    R.api = api;
    let q; try { q = await getj("/review/queue"); } catch { return; }   // no review on this server
    R.on = true; R.A = q.affine; R.inv = api.affInv(q.affine); R.meta = q; pane();
    R.vs = Math.min(...[0, 1, 2].map(k => Math.hypot(R.A[0][k], R.A[1][k], R.A[2][k])));   // the finest voxel spacing, mm
    if (q.paint_only) { $("rvmode").querySelector('option[value="scribble"]').remove(); $("rvsite").style.display = "none"; }
    setMode("paint", true);
    document.addEventListener("keydown", keys, true);
    await refresh();
    say(q.paint_only ? `paint only: label edits ${q.edits}` : `edit log: ${q.log} · label edits: ${q.edits}`);
  },
  draw2d, text, pointer,
};
// for the check of MAX_DRAWN: the time of one draw of n squares, all in tile 0's view (ms; the mean of 10 draws)
R.bench = (n = MAX_DRAWN) => {
  const g = $("bvo").getContext("2d"), t = R.api.BV.tiles[0], v0 = mm2vox(R.api.cross), m = vaxis(t.n), ax = [0, 1, 2].filter(a => a !== m);
  const side = Math.ceil(Math.sqrt(n)), list = [];
  for (let i = 0; i < n; i++) { const v = v0.slice(); v[ax[0]] += (i % side) - (side >> 1); v[ax[1]] += Math.floor(i / side) - (side >> 1); list.push(["answer", v]); }
  const t0 = performance.now(); let d = 0; for (let k = 0; k < 10; k++) d = squares(g, t, list, null, Infinity); return { ms: (performance.now() - t0) / 10, drawn: d };
};
(window.BVP = window.BVP || []).push(hooks);
