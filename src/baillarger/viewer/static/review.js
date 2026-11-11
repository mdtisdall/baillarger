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
const REGEN_POLL_MS = 1000;
// the most preview voxels drawn on one tile; above it the tile draws the first MAX_DRAWN and says so. Measured with R.bench
// on the laptop (Chrome, 1200 x 800): 13,189 squares in 3.3 ms, so 50,000 take about 12.5 ms, within one frame at 60 fps
const MAX_DRAWN = 50000;
const R = window.REVIEW = {
  api: null, on: false, meta: null, sites: [], session: null, cur: null, t0: 0, choice: null, params: {}, preview: null,
  previewSeq: 0, regen: null, pending: null, pendingSeq: 0, labelRGBA: new Map(), brush: false, mode: "paint", cls: null, radius: 1, undoDepth: 0, strokes: [], stroke: null, s0: 0, inv: null, A: null, revealed: new Map(),
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
  <div class="pbar" id="rvregen"><button id="rvrgo" title="run the regenerate command on the label edits (about 20 min and 22 GB on the laptop)">regenerate the outputs</button>
    <span id="rvrstate" style="white-space:pre-wrap"></span></div>
  <pre id="rvrlog" style="display:none;max-height:120px;overflow-y:auto;margin:0 8px;font-size:11px;white-space:pre-wrap"></pre>
  <div id="rvrover" style="display:none;max-height:160px;overflow-y:auto;font-size:12px"></div>
  <div class="pbar"><span id="rvmsg" style="color:var(--mute)"></span></div>
  <div id="rvtable" style="overflow-y:auto;flex:1;min-height:0;font-size:12px"></div>
  <div class="help" style="position:static;padding:2px 8px">keys: 1–7 choices · t thicken radius · Enter submit · s skip · n next ·
    b brush on/off · p paint/scribble · w WM · c cortex · o outside · e eraser (input label) · [ ] brush radius · x submit strokes ·
    u undo the last write into the edits · Esc clear strokes. Row: open that site.
    Preview: <span style="color:${COL.answer}">■ the answer</span> <span style="color:${COL.correction}">■ the correction</span> ·
    <span style="color:${COL.pending}">□</span> pending edits (not yet in the dseg), filled in their label's colour</div>`;
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
  if (c === "manual-paint" && closed()) { say(closed(), true); return; }
  if (c === "manual-paint" && !setMode("paint")) return;
  R.choice = c; R.params = params(c); const seq = ++R.previewSeq;
  for (const b of document.querySelectorAll("#rvsite button[data-choice]")) b.style.outline = b.dataset.choice === c ? "2px solid var(--acc)" : "";
  if (c === "manual-paint") {   // the paint mode: the strokes are the answer; there is no preview of strokes yet
    R.preview = null; $("rvbon").checked = true; R.brush = true; R.api.draw();
    $("rvprev").textContent = "manual paint: paint the strokes (left drag), then Submit (Enter) writes them as this site's answer" +
      " into the label edits and the log. Undo (u) takes them back from the edits (the log keeps the answer)."; $("rvprev").style.color = ""; return;
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
  R.undoDepth = r.undo_depth; strokesInfo(); pending();
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
  R.strokes = []; strokesInfo(); R.api.draw(); pending();
}
async function undo() {
  if (R.strokes.length || R.stroke) { say("submit or clear the strokes first (Esc)", true); return; }
  let r; try { r = await post("/review/undo", {}); } catch (e) { say(e.message, true); return; }
  R.undoDepth = r.undo_depth; strokesInfo(); say(`undo: ${r.voxels} voxels of the label edits written back`); pending();
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
    try {   // the label overlay's colours, for the pending edits (no labels volume: outlines only)
      const M = await getj("/volume/labels/meta");
      R.labelRGBA = new Map(Object.entries(M.table || {}).map(([v, e]) => [Number(v), e.rgba]));
    } catch { /* no labels volume */ }
    pending();
    try { showRegen(await getj("/review/regenerate")); pollRegen(); } catch (e) { say(e.message, true); }
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
