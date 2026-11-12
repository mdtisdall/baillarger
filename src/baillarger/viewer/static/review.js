// The stream 1 review pane (phase 4.2): stream 1's queue of sites, the site answers with their previews, and scribbles,
// written to the edit log (wm-edits/1) by the server (baillarger.viewer.review); and the paint mode with its undo. A site
// answer and a paint write their labels into the label edits (label-edits/1), the diff that the dseg reads; a scribble does
// not. The pane holds no stream 1 logic: the server calls phase 1.3's code for the queue, the random sites, the choices,
// their previews, the log and the edits.
// "accept the fix" (accept-fix) shows at an override site only: it writes the site's fix's labels into the edits.
//
// The brush has two modes. paint: the labels of the menu (outside, WM, cortex and the eraser "input label", which gives the
// voxels their input label again); its submit writes into the edits (/review/paint). Undo (/review/undo) takes back the last
// write into the edits of this viewer run: a paint or a site answer. scribble: the classes of the label set; its submit is a record of the log only. At a site,
// "manual paint" opens the paint mode, and Submit sends the strokes as the site's answer. With --edits alone the server
// has no queue: the pane is paint only.
//
// Preview of strokes: with manual paint at a site, the pane asks /review/preview {site, choice: manual-paint, strokes}
// when the choice opens and after each stroke, clear and undo (a newer request wins). The server previews the
// pending edits near the site and the strokes, with the strokes not yet submitted. The pane draws the change and the
// correction with the preview drawing, and says when the correction would change an edited voxel and when the box cuts the
// WM (the preview is then approximate). Submit stays allowed.
//
// Pending edits (/review/pending): the edits that are not yet in the dseg on the screen (the labels volume), each with
// the label that it gives (an erased voxel: its input label). The pane draws them on the slice tiles with the preview
// drawing, each square in its label's colour of the label overlay (/volume/labels/meta; outside: no fill) with an
// outline. It asks for them at the start and after each write into the edits (a paint, an undo, a site answer); no brick
// volume changes. The input labels are the volume "input" of the brick pane, to compare with the labels.
//
// A plug-in of the brick pane (window.BVP, as static/overlays.js): it draws the previews and the strokes on #bvo, takes the
// pointer in brush mode, and puts its own pane in place of the surface pane when the server has a review (/review/queue).
//
// Regenerate: the button starts the regenerate command on the server (POST /review/regenerate), a subprocess under
// scripts/memguard. The pane polls its state (GET /review/regenerate) and shows its log lines. The paint mode is closed
// while it runs (the command read the label edits at its start). When it ends, the pane shows the overrides (each fix that
// changes edited voxels: the fix, its voxel, the number of edited voxels) and asks for a restart of the viewer; the server
// refuses each write into the edits until the restart. A failed run shows its error; the old outputs stay.
//
// Surface preview (/review/surface-preview): after each stroke in paint mode, clear, paint, site answer and undo, and at
// the start, the pane asks the server for the white surface after the label edits and the paint strokes not yet submitted
// (a newer request wins). The server keeps a working antialiased field: the stored field with a local solve of each change
// of the edits written in (the radius of docs/preview-radius.md). It answers with the mesh blocks that the solves
// re-marched: the blocks of the edits (working blocks; the pane asks only for those that changed after its revision) and
// the blocks of the strokes (preview blocks). The 3D view shows them in place of the surface's blocks at the level "true
// surface" (VIEWER.hide hides the surface there): working blocks blue, preview blocks green. The slices draw their cut
// (/review/surface-preview/cut, viewer.outlines.plane_cut) in green over the surface's yellow outline. The preview is
// approximate; regenerate is exact. Key v (or the button) shows the 3D view in place of this pane, and back.
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
const COL = { answer: "rgba(64, 224, 255, 0.85)", correction: "rgba(255, 140, 0, 0.85)", site: "rgba(255, 255, 255, 0.9)",
  pending: "rgba(255, 0, 255, 0.95)" };
const STROKE_RGB = { wm: [245, 245, 245], cortex: [255, 70, 90], outside: [70, 140, 255], input: [255, 0, 255] };
const PREVIEW_POLL_MS = 400;
const SURF_COL = { cut: "rgba(64, 255, 128, 0.95)", working: [0.45, 0.7, 1.0], preview: [0.3, 1.0, 0.5] };
const REGEN_POLL_MS = 1000;
// the most preview voxels drawn on one tile; above it the tile draws the first MAX_DRAWN and says so. Measured with R.bench
// on the laptop (Chrome, 1200 x 800): 13,189 squares in 3.3 ms, so 50,000 take about 12.5 ms, within one frame at 60 fps
const MAX_DRAWN = 50000;
const R = window.REVIEW = {
  api: null, on: false, meta: null, sites: [], session: null, cur: null, t0: 0, choice: null, params: {}, preview: null,
  previewSeq: 0, sp: { on: true, off: "", seq: 0, rev: -1, job: null, cells: 0, working: new Map(), preview: new Map(), cuts: new Map(), gl: null, show3d: false },
  regen: null, pending: null, pendingSeq: 0, labelRGBA: new Map(), brush: false, mode: "paint", cls: null, radius: 1, undoDepth: 0, strokes: [], stroke: null, s0: 0, inv: null, A: null, revealed: new Map(),
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
    <button id="rvundo" title="key u: undo the last write into the label edits (a paint or a site answer) of this viewer run">undo (u)</button><span id="rvsn"></span>
  </div>
  <div class="pbar" id="rvsurf"><label title="the white surface after the edits and the strokes: a local solve of the antialiased field, approximate">
    <input type="checkbox" id="rvspon" checked> surface preview</label>
    <button id="rv3d" title="key v: the 3D view in place of this pane, and back">3D view (v)</button><span id="rvsps" style="white-space:pre-wrap"></span></div>
  <div class="pbar" id="rvregen"><button id="rvrgo" title="run the regenerate command on the label edits (about 20 min and 22 GB on the laptop)">regenerate the outputs</button>
    <span id="rvrstate" style="white-space:pre-wrap"></span></div>
  <pre id="rvrlog" style="display:none;max-height:120px;overflow-y:auto;margin:0 8px;font-size:11px;white-space:pre-wrap"></pre>
  <div id="rvrover" style="display:none;max-height:160px;overflow-y:auto;font-size:12px"></div>
  <div class="pbar"><span id="rvmsg" style="color:var(--mute)"></span></div>
  <div id="rvtable" style="overflow-y:auto;flex:1;min-height:0;font-size:12px"></div>
  <div class="help" style="position:static;padding:2px 8px">keys: 1–7 choices · t thicken radius · Enter submit · s skip · n next ·
    b brush on/off · p paint/scribble · w WM · c cortex · o outside · e eraser (input label) · [ ] brush radius · x submit strokes ·
    u undo the last write into the edits · Esc clear strokes · v 3D view. Row: open that site.
    Preview: <span style="color:${COL.answer}">■ the answer</span> <span style="color:${COL.correction}">■ the correction</span> ·
    <span style="color:${COL.pending}">□</span> pending edits (not yet in the dseg), filled in their label's colour ·
    <span style="color:${SURF_COL.cut}">—</span> the surface preview's cut</div>`;
  const surf = $("p-surface"); surf.style.display = "none"; surf.parentElement.insertBefore(p, surf.nextSibling);
  for (const b of p.querySelectorAll("button[data-choice]")) b.onclick = () => choose(b.dataset.choice);
  $("rvthick").onchange = () => R.choice === "thicken" && choose("thicken");
  $("rvsubmit").onclick = submit; $("rvskip").onclick = skip;
  $("rvbon").onchange = () => {
    const why = closed(); if (why && $("rvbon").checked) { $("rvbon").checked = false; say(why, true); }
    R.brush = $("rvbon").checked; R.api.draw();
  };
  $("rvmode").onchange = () => setMode($("rvmode").value);
  $("rvcls").onchange = () => { R.cls = $("rvcls").value; };
  $("rvrad").onchange = () => { R.radius = Number($("rvrad").value); };
  $("rvssub").onclick = submitStrokes; $("rvsclr").onclick = clearStrokes; $("rvundo").onclick = undo;
  $("rvrgo").onclick = regenerate;
  $("rvspon").onchange = () => { R.sp.on = $("rvspon").checked; if (R.sp.on) surfacePreview(); else { R.sp.seq++; showSurface(); } };
  $("rv3d").onclick = () => show3d(!R.sp.show3d);
}
function clearStrokes() { R.strokes = []; R.stroke = null; strokesInfo(); R.api.draw(); strokesPreview(); surfacePreview(); }
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
  if (c === "manual-paint" && closed()) { say(closed(), true); return; }
  if (c === "manual-paint" && !setMode("paint")) return;
  R.choice = c; R.params = params(c); const seq = ++R.previewSeq;
  for (const b of document.querySelectorAll("#rvsite button[data-choice]")) b.style.outline = b.dataset.choice === c ? "2px solid var(--acc)" : "";
  if (c === "manual-paint") {   // the paint mode: the strokes are the answer, previewed with the pending edits
    R.preview = null; $("rvbon").checked = true; R.brush = true; R.api.draw(); strokesPreview(); return;
  }
  $("rvprev").textContent = `${c}: computing the preview…`;
  const r = await askPreview({ site: R.cur.id, choice: c, params: R.params }, seq); if (!r) return;
  R.preview = index(r); R.api.draw();
  const w = r.overrides ? "\n⚠ the correction overrides this answer: the automatic fix changes the voxels that the answer sets" : "";
  $("rvprev").textContent = `${c}: the answer changes ${r.answer.length} voxels, then the correction changes ${r.correction.length}` +
    ` (${(r.seconds ?? 0).toFixed(2)} s)` + w + cutNote(r);
  $("rvprev").style.color = r.overrides ? "#ffd23f" : "";
}
const cutNote = r => r.cut_by_box ? "\nthe preview's box cuts WM: the correction outside the box is not shown (approximate)" : "";
// a preview request, polled while it runs as a job on the server; null if it failed or a newer request (seq) replaced it
async function askPreview(body, seq) {
  let r; try { r = await post("/review/preview", body); } catch (e) { if (seq === R.previewSeq) say(e.message, true); return null; }
  while (r.status === "running") {   // a long preview runs as a job on the server
    await new Promise(ok => setTimeout(ok, PREVIEW_POLL_MS)); if (seq !== R.previewSeq) return null;
    try { r = await getj(`/review/preview?job=${encodeURIComponent(r.job)}`); } catch (e) { if (seq === R.previewSeq) say(e.message, true); return null; }
  }
  if (seq !== R.previewSeq) return null;
  if (r.status !== "done") { say(`preview: ${r.error || r.status}`, true); return null; }
  return r;
}
// the preview of the strokes not yet submitted and the pending edits (the module comment), with manual paint at a site
async function strokesPreview() {
  if (!R.cur || R.choice !== "manual-paint" || closed()) return;
  const seq = ++R.previewSeq, n = R.strokes.length, how = "manual paint: paint the strokes (left drag), then Submit (Enter) writes them" +
    " as this site's answer into the label edits and the log. Undo (u) takes them back from the edits (the log keeps the answer).";
  $("rvprev").textContent = `manual paint: computing the preview of ${n} strokes and the pending edits…\n${how}`; $("rvprev").style.color = "";
  const r = await askPreview({ site: R.cur.id, choice: "manual-paint", params: {}, strokes: paintStrokes() }, seq); if (!r) return;
  R.preview = index(r); R.api.draw();
  const w = r.overrides ? "\n⚠ the correction would change edited voxels (of these strokes or of the pending edits): Submit is allowed" : "";
  $("rvprev").textContent = `manual paint: the strokes and the pending edits near them change ${r.answer.length} voxels, then the` +
    ` correction changes ${r.correction.length} (${(r.seconds ?? 0).toFixed(2)} s)` + w + cutNote(r) + "\n" + how;
  $("rvprev").style.color = r.overrides ? "#ffd23f" : "";
}
// [kind, voxel] entries by slice: for each voxel axis a, a map from the voxel's index along a to its entries
function bySlice(entries) {
  const by = [new Map(), new Map(), new Map()];
  for (const e of entries) for (let a = 0; a < 3; a++) { const m = by[a], k = e[1][a]; if (!m.has(k)) m.set(k, []); m.get(k).push(e); }
  return by;
}
// the preview voxels by slice
function index(r) {
  const entries = [...r.correction.map(v => ["correction", v]), ...r.answer.map(v => ["answer", v])];
  return { overrides: r.overrides, n: entries.length, by: bySlice(entries) };
}
// the pending edits (the module comment), asked again after each write into the edits; a newer request wins
async function pending() {
  const seq = ++R.pendingSeq;
  let r; try { r = await getj("/review/pending"); } catch (e) { if (seq === R.pendingSeq) { R.pending = null; strokesInfo(); say(`pending edits: ${e.message}`, true); R.api.draw(); } return; }
  if (seq !== R.pendingSeq) return;
  R.pending = { n: r.voxels.length, by: bySlice(r.voxels.map((v, i) => [r.labels[i], v])) };
  strokesInfo(); R.api.draw();
}
// the label overlay's colour of a label: its colour table (/volume/labels/meta); outside and a label not in it: no fill
const labelFill = lab => { const c = R.labelRGBA.get(lab); return c ? `rgba(${c[0]}, ${c[1]}, ${c[2]}, 0.85)` : null; };
async function submit() {
  if (!R.cur || !R.choice) { say("open a site and pick a choice first", true); return; }
  const manual = R.choice === "manual-paint";
  if (manual && !R.strokes.length) { say("manual paint: paint at least one stroke first", true); return; }
  if (!manual && !R.preview) { say("wait for the preview", true); return; }
  const secs = (performance.now() - R.t0) / 1000, body = { site: R.cur.id, choice: R.choice, params: R.params, reviewer_seconds: secs };
  if (manual) body.strokes = paintStrokes();
  let r; try { r = await post("/review/answer", body); }
  catch (e) { say(e.message, true); return; }
  if (manual) R.strokes = [];
  R.undoDepth = r.undo_depth; strokesInfo(); pending(); surfacePreview();
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

// ------------------------------------------------------------------ regenerate
// why the writes into the edits are closed (a run in progress, or a run that replaced the outputs), or "" when they are open
function closed() {
  const st = R.regen && R.regen.status;
  if (st === "running") return "regenerate runs: the paint mode is closed until it ends";
  if (st === "done" || (st === "failed" && R.regen.outputs_changed)) return "the outputs were regenerated: restart the viewer";
  return "";
}
async function regenerate() {
  if (R.strokes.length || R.stroke) { say("submit or clear the strokes first (Esc)", true); return; }
  if (!confirm("Regenerate the dseg, the white surface, its overview and the sites from the label edits?\n" +
    "It takes about 20 min and about 22 GB on the laptop. The paint mode is closed while it runs, " +
    "and the viewer must restart after it.")) return;
  let r; try { r = await post("/review/regenerate", {}); } catch (e) { say(e.message, true); return; }
  showRegen(r); say("regenerate started"); pollRegen();
}
async function pollRegen() {
  while (R.regen && R.regen.status === "running") {
    await new Promise(ok => setTimeout(ok, REGEN_POLL_MS));
    try { showRegen(await getj("/review/regenerate")); } catch (e) { say(e.message, true); return; }
  }
}
function showRegen(r) {
  const was = R.regen && R.regen.status; R.regen = r;
  const st = r.status, min = r.seconds === null ? "" : ` ${(r.seconds / 60).toFixed(1)} min`, why = closed();
  $("rvrgo").disabled = st === "running" || !!why;
  const cap = r.cap_gb === null ? "" : ` · memory cap ${r.cap_gb} GB (the viewer used ${r.viewer_gb} GB at the start)`;
  $("rvrstate").textContent = {
    idle: "",
    running: `running${min}${cap}. The paint mode is closed: the command read the label edits at its start.`,
    done: `done in${min}. Restart the viewer to see the new outputs. Until the restart, each paint, answer, accept-fix and undo is refused.`,
    failed: `failed after${min}: ${r.error}. ` + (r.outputs_changed
      ? "Some outputs were replaced: restart the viewer, then run regenerate again (it finishes the moves first)."
      : "The old outputs stay."),
  }[st] || st;
  $("rvrstate").style.color = st === "failed" ? "#ff6b6b" : st === "done" ? "#ffd23f" : "";
  const log = $("rvrlog"); log.style.display = r.log.length ? "" : "none";
  const end = log.scrollTop + log.clientHeight >= log.scrollHeight - 4; log.textContent = r.log.join("\n"); if (end) log.scrollTop = log.scrollHeight;
  const ov = $("rvrover");
  if (st === "done") {
    ov.style.display = "";
    ov.innerHTML = r.overrides.length
      ? `<div style="padding:2px 8px"><b>${r.overrides.length} fixes change edited voxels</b> (the correction overrides these edits). Row: go to the voxel.</div>
        <table><tr><th>fix</th><th style="text-align:left">kind</th><th>voxel</th><th>edited voxels</th></tr>${r.overrides.map((o, k) =>
          `<tr data-k="${k}"><td>${o.fix}</td><td style="text-align:left">${esc(o.kind)}</td><td>${o.voxel.join(",")}</td><td>${o.voxels}</td></tr>`).join("")}</table>`
      : `<div style="padding:2px 8px">No fix changes edited voxels.</div>`;
    for (const tr of ov.querySelectorAll("tr[data-k]")) tr.onclick = () => { R.api.crossHere(vox2mm(r.overrides[Number(tr.dataset.k)].voxel)); R.api.update(); };
  } else ov.style.display = "none";
  if (why) {   // the paint mode is closed: the brush off, its strokes dropped
    R.brush = false; $("rvbon").checked = false; R.strokes = []; R.stroke = null; R.preview = null; strokesInfo(); R.api.draw();
  }
  for (const id of ["rvbon", "rvssub", "rvsubmit"]) $(id).disabled = !!why;
  if (why) $("rvundo").disabled = true;
  if (was === "running" && st !== "running") { say(st === "done" ? "regenerate done: restart the viewer" : `regenerate failed: ${r.error}`, st !== "done"); refresh(); }
}

// ------------------------------------------------------------------ scribbles
function strokesInfo() {
  const n = R.strokes.reduce((a, s) => a + s.voxels.size, 0);
  $("rvsn").textContent = (R.strokes.length ? `${R.strokes.length} ${R.mode} strokes, ${n} voxels · ` : "") + `undo: ${R.undoDepth} writes` +
    (R.pending ? ` · pending: ${R.pending.n} voxels` : "");
  $("rvundo").disabled = !R.undoDepth || !!closed();
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
    " (drawn as pending edits: the labels volume still shows the dseg)");
  R.strokes = []; strokesInfo(); R.api.draw(); pending(); surfacePreview();
}
async function undo() {
  if (R.strokes.length || R.stroke) { say("submit or clear the strokes first (Esc)", true); return; }
  let r; try { r = await post("/review/undo", {}); } catch (e) { say(e.message, true); return; }
  R.undoDepth = r.undo_depth; strokesInfo(); say(`undo: ${r.voxels} voxels of the label edits written back`); pending(); strokesPreview(); surfacePreview();
}
async function scribble() {
  if (!R.strokes.length) { say("no strokes", true); return; }
  const secs = (performance.now() - R.s0) / 1000;
  const strokes = R.strokes.map(s => ({ class: s.value, voxels: [...s.voxels.values()] }));
  let r; try { r = await post("/review/scribble", { strokes, reviewer_seconds: secs }); } catch (e) { say(e.message, true); return; }
  say(`scribble ${r.answer} written: ${strokes.length} strokes, ${strokes.reduce((a, s) => a + s.voxels.length, 0)} voxels, ${secs.toFixed(1)} s`);
  R.strokes = []; strokesInfo(); R.api.draw();
}

// ------------------------------------------------------------------ the surface preview (the module comment)
// the binary answers of the server (baillarger.viewer.columns.encode: a JSON header, then the arrays)
const TYPES = { int32: Int32Array, float32: Float32Array };
function decode(buf) {
  const n = new Uint32Array(buf, 0, 1)[0], head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n))), out = {};
  for (const t of head.arrays) out[t.name] = new TYPES[t.dtype](buf, 4 + n + t.offset, t.shape.reduce((a, b) => a * b, 1));
  return [head, out];
}
const sps = (m, bad) => { const el = $("rvsps"); el.textContent = m ? " " + m : ""; el.style.color = bad ? "#ff6b6b" : "var(--mute)"; if (bad) R.api.log("surface preview: " + m); };
async function surfacePreview() {
  const S = R.sp; if (!R.on || !S.on || S.off || closed()) return;
  const seq = ++S.seq, strokes = R.mode === "paint" ? paintStrokes() : [], t0 = performance.now();
  sps(`computing the surface after the edits${strokes.length ? ` and ${strokes.length} strokes` : ""}…`);
  let r; try { r = await post("/review/surface-preview", { strokes, since: S.rev }); } catch (e) { if (seq === S.seq) sps(e.message, true); return; }
  while (r.status === "running") {
    await new Promise(ok => setTimeout(ok, PREVIEW_POLL_MS)); if (seq !== S.seq) return;
    try { r = await getj(`/review/surface-preview?job=${encodeURIComponent(r.job)}&since=${S.rev}`); } catch (e) { if (seq === S.seq) sps(e.message, true); return; }
  }
  if (seq !== S.seq) return;
  if (r.status !== "done") {
    sps(r.error || r.status, true);
    if (/no antialiased field|no white surface|not the dseg's/.test(r.error || "")) S.off = r.error;   // asking again does not help
    return;
  }
  let buf; try { const q = await fetch(`/review/surface-preview/mesh?job=${encodeURIComponent(r.job)}&since=${r.since}`); if (!q.ok) throw new Error(`${q.status}`); buf = await q.arrayBuffer(); }
  catch (e) { if (seq === S.seq) sps(`mesh: ${e.message}`, true); return; }
  if (seq !== S.seq) return;
  const [head, a] = decode(buf), preview = new Map(); let kv = 0, kf = 0;
  for (const [i, j, k, nv, nf, kind] of head.blocks) {
    const m = { id: [i, j, k], kind, v: a.v.subarray(3 * kv, 3 * (kv + nv)), f: a.f.subarray(3 * kf, 3 * (kf + nf)) };
    (kind ? preview : S.working).set(`${i},${j},${k}`, m); kv += nv; kf += nf;
  }
  S.preview = preview; S.rev = r.revision; S.job = r.job; S.cells = r.cells; S.cuts.clear(); S.last = r;
  showSurface();
  const sec = ((performance.now() - t0) / 1000).toFixed(2);
  sps(`${S.working.size} blocks changed by the edits` + (r.radius === null ? "; the strokes change no block" :
    `; the strokes re-march ${r.blocks.length} blocks (R ${r.radius} voxels, ${r.solves.length} solves)`) + ` · ${sec} s · approximate`);
}
// the blocks shown in place of the surface's: the working blocks, the preview's over them
const shownBlocks = () => new Map([...R.sp.working, ...R.sp.preview]);
// the meshes in the 3D view (box mm: voxel coordinates times the voxel size), and the surface's blocks under them hidden
const SVS = `#version 300 es
in vec3 pos; in vec3 nrm; in vec3 col; uniform mat4 mvp; uniform mat3 rot; out vec3 vcol; out float vlit;
void main() { gl_Position = mvp * vec4(pos, 1.0); vlit = abs((rot * nrm).z); vcol = col; }`;
const SFS = `#version 300 es
precision mediump float; in vec3 vcol; in float vlit; out vec4 o;
void main() { o = vec4(vcol * (0.35 + 0.65 * vlit), 1.0); }`;
function showSurface() {
  R.api.draw();
  const V = window.VIEWER, g = V && V.GL.g, S = R.sp; if (!g) return;
  const shown = R.sp.on ? shownBlocks() : new Map(), vox = [0, 1, 2].map(k => Math.hypot(R.A[0][k], R.A[1][k], R.A[2][k]));
  if (!S.gl) {
    const prog = V.program(g, SVS, SFS), vao = g.createVertexArray(); g.bindVertexArray(vao);
    const bufs = {}; for (const [name, k] of [["pos", 3], ["nrm", 3], ["col", 3]]) {
      const b = bufs[name] = g.createBuffer(), loc = g.getAttribLocation(prog, name);
      g.bindBuffer(g.ARRAY_BUFFER, b); g.enableVertexAttribArray(loc); g.vertexAttribPointer(loc, k, g.FLOAT, false, 0, 0);
    }
    const ib = g.createBuffer(); g.bindBuffer(g.ELEMENT_ARRAY_BUFFER, ib); g.bindVertexArray(null);
    S.gl = { prog, vao, bufs, ib, n: 0 };
  }
  let nv = 0, nf = 0; for (const m of shown.values()) { nv += m.v.length / 3; nf += m.f.length / 3; }
  const pos = new Float32Array(3 * nv), nrm = new Float32Array(3 * nv), col = new Float32Array(3 * nv), idx = new Uint32Array(3 * nf);
  let o = 0, q = 0; const top = [0, 0, 0];
  for (const m of shown.values()) {
    const n = m.v.length / 3, c = m.kind ? SURF_COL.preview : SURF_COL.working;
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { pos[3 * (o + i) + k] = m.v[3 * i + k] * vox[k]; col[3 * (o + i) + k] = c[k]; }
    for (let t = 0; t < m.f.length; t += 3) {   // vertex normals: the sum of the faces' normals (area-weighted)
      const a = o + m.f[t], b = o + m.f[t + 1], d = o + m.f[t + 2]; idx[q++] = a; idx[q++] = b; idx[q++] = d;
      const u = [0, 1, 2].map(k => pos[3 * b + k] - pos[3 * a + k]), w = [0, 1, 2].map(k => pos[3 * d + k] - pos[3 * a + k]);
      const x = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
      for (const p of [a, b, d]) for (let k = 0; k < 3; k++) nrm[3 * p + k] += x[k];
    }
    for (let k = 0; k < 3; k++) top[k] = Math.max(top[k], m.id[k] + 1);
    o += n;
  }
  for (const [name, arr] of [["pos", pos], ["nrm", nrm], ["col", col]]) { g.bindBuffer(g.ARRAY_BUFFER, S.gl.bufs[name]); g.bufferData(g.ARRAY_BUFFER, arr, g.DYNAMIC_DRAW); }
  g.bindVertexArray(S.gl.vao); g.bindBuffer(g.ELEMENT_ARRAY_BUFFER, S.gl.ib); g.bufferData(g.ELEMENT_ARRAY_BUFFER, idx, g.DYNAMIC_DRAW); g.bindVertexArray(null);
  S.gl.n = idx.length;
  // the surface's blocks under the shown blocks are not drawn
  const flags = new Uint8Array(Math.max(1, top[0] * top[1] * top[2]));
  for (const m of shown.values()) flags[(m.id[2] * top[1] + m.id[1]) * top[0] + m.id[0]] = 1;
  V.hide(shown.size ? { n: top, step: vox.map(x => x * S.cells), flags } : null);   // it draws the 3D view again
}
function draw3d(g, M) {
  const S = R.sp, V = window.VIEWER; if (!S.gl || !S.gl.n || !S.on || V.GL.level !== 0) return;
  const Q = V.GL.R; g.useProgram(S.gl.prog); g.bindVertexArray(S.gl.vao);
  g.uniformMatrix4fv(g.getUniformLocation(S.gl.prog, "mvp"), false, M);
  g.uniformMatrix3fv(g.getUniformLocation(S.gl.prog, "rot"), false, new Float32Array([Q[0], Q[3], Q[6], Q[1], Q[4], Q[7], Q[2], Q[5], Q[8]]));
  g.drawElements(g.TRIANGLES, S.gl.n, g.UNSIGNED_INT, 0); g.bindVertexArray(null);
}
// the 3D view in place of the review pane (on), at the level "true surface" (where the preview's blocks are), turned to
// the last re-marched blocks; or the review pane again
async function show3d(on) {
  const V = window.VIEWER, surf = $("p-surface"); if (!V || !V.GL.g) { say("no 3D view on this page", true); return; }
  R.sp.show3d = on; $("p-review").style.display = on ? "none" : ""; surf.style.display = on ? "" : "none";
  let back = $("rvback");
  if (!back) { back = document.createElement("button"); back.id = "rvback"; back.textContent = "review pane (v)"; back.onclick = () => show3d(false); $("sbar").appendChild(back); }
  back.style.display = on ? "" : "none";
  if (!on) return;
  if ($("level").value !== "0") { $("level").value = "0"; await $("level").onchange(); }
  const S = R.sp, ids = [...(S.preview.size ? S.preview : S.working).values()].map(m => m.id);
  if (ids.length) {   // the centre of those blocks, box mm
    const vox = [0, 1, 2].map(k => Math.hypot(R.A[0][k], R.A[1][k], R.A[2][k]));
    V.GL.target = [0, 1, 2].map(k => (ids.reduce((a, b) => a + b[k], 0) / ids.length + 0.5) * S.cells * vox[k]);
    V.GL.dist = Math.min(V.GL.dist, 3 * S.cells * Math.max(...vox));
  }
  V.draw3d();
}
// the cut of the shown blocks by the tile's plane, from the server (asked once per plane and preview); drawn in SURF_COL.cut
function surfaceCut(g, t) {
  const S = R.sp; if (!S.on || !S.job || !(S.working.size || S.preview.size)) return;
  const mm = R.api.cross[t.n], key = `${S.job}|${t.n}|${mm.toFixed(4)}`;
  let c = S.cuts.get(key);
  if (!c) {
    c = { seg: null }; S.cuts.set(key, c); const job = S.job;
    fetch(`/review/surface-preview/cut?job=${encodeURIComponent(job)}&axis=${t.n}&mm=${mm}`)
      .then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.arrayBuffer(); })
      .then(b => { c.seg = new Float32Array(b); if (S.job === job) R.api.draw(); })
      .catch(e => { S.cuts.delete(key); R.api.log(`surface preview cut: ${e.message}`); });
    return;
  }
  if (!c.seg || !c.seg.length) return;
  const s = c.seg, a = [0, 0, 0], b = [0, 0, 0]; g.strokeStyle = SURF_COL.cut; g.lineWidth = 1.5; g.beginPath();
  for (let i = 0; i < s.length; i += 6) {
    a[0] = s[i]; a[1] = s[i + 1]; a[2] = s[i + 2]; b[0] = s[i + 3]; b[1] = s[i + 4]; b[2] = s[i + 5];
    const p = R.api.w2px(t, a), q = R.api.w2px(t, b); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]);
  }
  g.stroke();
}

// ------------------------------------------------------------------ drawing on #bvo
// list: [kind, voxel] or voxels; one square per voxel in the tile's slice. colour: a fill, or a function of the kind (null:
// no fill); default COL[kind]. outline: the colour of a 1 px outline of each square, if given
function squares(g, t, list, colour, cap = MAX_DRAWN, outline = null) {
  const px = Math.max(1, R.api.BV.s * R.vs);
  let drawn = 0;
  if (outline) { g.strokeStyle = outline; g.lineWidth = 1; }
  for (const e of list) {
    if (drawn >= cap) return drawn;
    const [kind, v] = Array.isArray(e[1]) ? e : [null, e];
    const [x, y] = R.api.w2px(t, vox2mm(v)); if (x < -px || y < -px || x > t.w + px || y > t.h + px) continue;
    const fill = typeof colour === "function" ? colour(kind) : colour || COL[kind];
    if (fill) { g.fillStyle = fill; g.fillRect(x - px / 2, y - px / 2, px, px); }
    if (outline) g.strokeRect(x - px / 2 + 0.5, y - px / 2 + 0.5, px - 1, px - 1);
    drawn++;
  }
  return drawn;
}
function draw2d(g, t) {
  if (!R.on || !R.A) return;
  const n = vaxis(t.n), k = mm2vox(R.api.cross)[n];
  t.rvpcap = 0;
  if (R.pending) {   // under the preview and the strokes
    const list = R.pending.by[n].get(k) || [];
    const d = squares(g, t, list, labelFill, MAX_DRAWN, COL.pending); t.rvpcap = list.length > d && d >= MAX_DRAWN ? list.length : 0;
  }
  if (R.preview) {
    const list = R.preview.by[n].get(k) || [];
    const d = squares(g, t, list); t.rvcap = list.length > d && d >= MAX_DRAWN ? list.length : 0;
  }
  for (const s of [...R.strokes, ...(R.stroke ? [R.stroke] : [])]) {
    const rgb = STROKE_RGB[s.key] || [255, 210, 63], vs = [...s.voxels.values()].filter(v => v[n] === k);
    squares(g, t, vs, `rgba(${rgb.join(",")}, 0.6)`);
  }
  surfaceCut(g, t);
  if (R.cur) {   // the site: a ring round its centre
    const [x, y] = R.api.w2px(t, R.cur.mm); g.strokeStyle = COL.site; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 9, 0, 2 * Math.PI); g.stroke();
  }
}
function text(t) {
  let s = "";
  if (R.brush) s += ` · ${R.mode}: ${$("rvcls").selectedOptions[0]?.textContent ?? R.cls} r ${R.radius}`;
  if (t.rvcap) s += ` · preview: ${MAX_DRAWN} of ${t.rvcap} voxels drawn`;
  if (t.rvpcap) s += ` · pending: ${MAX_DRAWN} of ${t.rvpcap} voxels drawn`;
  return s;
}
function pointer(type, e, h) {   // brush mode: a left drag paints one stroke of the current label
  if (!R.on || !R.brush || e.button > 0 || e.shiftKey || closed()) return false;
  if (type === "down" && h) {
    if (!R.strokes.length && !R.stroke) R.s0 = performance.now();
    const c = $("rvcls"), opt = c.options[c.selectedIndex];
    const value = c.value === "input" ? "input" : Number(c.value);
    R.stroke = { value, key: opt.dataset.key, voxels: new Map(), tile: h.t }; paint(h.t, h.x, h.y); R.api.draw(); return true;
  }
  if (!R.stroke) return false;
  const t = R.stroke.tile, r = $("bvo").getBoundingClientRect(), x = e.clientX - r.left - t.x, y = e.clientY - r.top - t.y;
  if (type === "move") { paint(t, x, y); R.api.draw(); return true; }
  if (type === "up") {
    const added = R.stroke.voxels.size > 0; if (added) R.strokes.push(R.stroke);
    R.stroke = null; strokesInfo(); R.api.draw(); if (added) { strokesPreview(); surfacePreview(); } return true;
  }
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
    v: () => show3d(!R.sp.show3d),
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
    try {   // the label overlay's colours, for the pending edits (no labels volume: outlines only)
      const M = await getj("/volume/labels/meta");
      R.labelRGBA = new Map(Object.entries(M.table || {}).map(([v, e]) => [Number(v), e.rgba]));
    } catch { /* no labels volume */ }
    pending();
    try { showRegen(await getj("/review/regenerate")); pollRegen(); } catch (e) { say(e.message, true); }
    if (window.VIEWER) window.VIEWER.hooks.draw3d.push(draw3d);
    surfacePreview();   // the working field: the stored field and the pending edits
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
