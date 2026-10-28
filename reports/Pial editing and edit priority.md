# Pial editing and edit priority: plan

**Purpose.** This plan offers a second way to get the top of the cortex: start from a pial surface that is nearly right, and fix it by hand where needed. It does not try to predict the top everywhere with a forward model. The plan is self-contained: a session that has not seen the design discussion should be able to do the work from this document and the files it points to. Results go in *Results* below, written as current state.

**Why.** The goal of the seismic-flattening work is to find and study laminar contrast (main plan, `reports/Seismic flattening for cortical layers.md`). The top of the cortex is an input to that, not the goal. A forward model that places the top correctly everywhere, kissing faces included, is a hard problem of its own (main plan, *The contact as an inference problem*, findings 11–27). Several existing models already give pial surfaces that are close. A few manual fixes, in the right places, are enough to go on with.

**Two deliverables:**
- **(a) An editor.** A simple local web tool that shows the 3D image with the white and pial meshes over it. You place control points, then ask for a remesh and see what fixing those locations does to their neighbourhoods. A wait of seconds to about half a minute is fine: this tests whether the design works, not how fast it can be (the user, 2026-10-27).
- **(b) An edit-priority diagnostic.** Maps on the pial surface, and a worklist, that point to where edits are likely needed.

**Not in scope:** changes to the contact model, the thickness field or the solver in `baillarger.fronts`. Edits are stored so that a later model can use them as evidence (§3), but this plan does not change any model.

## 1. Decision open for the user: the base pial models

The editor starts from one base model. The disagreement diagnostic (§4, signal 1) compares several. Every base is turned into the same form, a thickness t_m(r) along each column (§2), so any of them can be used. **Which models to use, and which is the default base, is open.**

Candidates found in the main plan:

| Model | Where it stands | Covers |
|---|---|---|
| **Label boundary** | The aseg's cortex-to-fluid boundary. In fused cortex the column runs on until the next stop (the contact of another model, or the ridge of arrival) | Open cortex only, on its own |
| **One pass, equal weights** | `baillarger.fronts`, one pass from all WM; finding 27 on the whole patches | Whole patches; contact = equal arrival |
| **One pass, field weights** | The same pass, weights from the thickness field (σ form), with finding 26's guards; findings 25, 27 | Whole patches; contact within 0.08–0.26 mm of the points at three of four faces, 0.40 mm at the V1V2 banks |
| **Per-front posterior, walls from roots** | Findings 11, 18, 19 | The marked faces only; failed at PCC piece 1. Not a whole-patch model |
| **Level-set filling, τ = 1 and its seam** | `reports/Level-set filling test plan.md` | Not run yet |

Things to decide:
1. Which models go into the comparison.
2. Which model is the default base the editor opens with. It may differ by patch.
3. Whether the label boundary is the top everywhere in open cortex, whichever model is used. All the models above agree there by construction, apart from their column directions.

Until this is decided, the work in §6 builds and tests with **one pass, equal weights** as a placeholder. It needs no thickness field and runs on the whole patches in under 10 s per pass (finding 27). That choice is for the build only, not a recommendation.

## 2. Representation: the pial surface as a height over the white mesh

**Current state (step 5):** the editor's base is now a **contoured surface** (step 5), edited by 3D points. The height-over-columns form below was built and tested (steps 1–4), but on real data its columns cross, so as a pial surface it tangles in fused cortex (step 4 follow-up). It stays the way to carry laminar depth along columns, and the editor still loads column sessions.

**The pial surface is the white mesh with each vertex moved up its own column, to that column's top.** It is not edited as a free mesh.

- **White mesh:** finding 28's recommended surface (`prototypes/scripts/mesh_smoothing.py`): the WM label contoured by marching tetrahedra, relaxed tangentially, then smoothed along the mesh by twicing at about 0.32 mm. Not `baillarger.frame.white_mesh` (volume smoothing loses thin WM).
- **Column of a white vertex r:** the integral curve of the arrival gradient from r, as a polyline sampled every 0.04–0.08 mm. Arrival comes from the unweighted march from the white mesh (`march_band`, which also carries each voxel's root position and arc length). This is the "geometry" piece of the main plan's *The fronts as PDE problems*: no thickness beliefs enter it, so a column cannot bend because a neighbour is believed thicker.
- **Thickness t(r):** the arc length along r's column to its top. The pial vertex for r is the point at arc length t(r) on r's column.
- **The top of a base model m** on column r is the first point where the column enters fluid or the territory of another front under model m. That gives t_m(r) for every model on the same columns, so models can be compared vertex by vertex.

**Why this form:**
1. **Edits are 1D.** Dragging a pial vertex slides it along its own column. The pial surface always has the white surface's topology, and every pial vertex keeps its root.
2. **An edit is thickness evidence.** A clicked top is a visible-top observation at one root, the same kind of evidence the thickness field already takes (`prototypes/scripts/prob_thickness.py`; main plan, *Thickness and where the contact lies*). A later model can use the edits directly.
3. **Kissing faces.** A click on a contact is a top for the column on each bank that passes nearest it. That matches what the user's marked contact points mean (layer I against layer I, finding 17), and it ties the two banks' sheets together.
4. **Two-surface equal volume.** The two-surface estimator (main plan, *The fallback*, role 1: untested) needs the pial area in correspondence with the white surface. This form gives it per vertex.

**Columns must reach past every candidate top.** The thick bank's top at a kissing face lies beyond the equal-arrival ridge (finding 12), and an open column's true top can lie above the label boundary. So each column is extended:
- **Past the ridge of arrival:** along the data's layer normal, signed by continuity with the column's last direction, up to 1.5 × the column's expected thickness or until it enters WM. The data orientation stays coherent across fused contacts (finding 1), so it can carry a column across where the arrival gradient reverses. The point where the extension starts is recorded with the column.
- **Into fluid:** along its last direction, up to 1 mm.

[inference: the extension rule is a design choice. Test 7 in §5 checks it on phantoms, and its error at the marked faces is reported.]

**Free edits (the fallback).** For a click x, the column is found from the march's carried root position and arc length, interpolated at x (no search over polylines). The **miss distance** is the distance from x to that root's polyline. If it is more than one voxel (0.16 mm), the columns are probably wrong there: the click is stored as a free position constraint, flagged, and the pial vertices near it move in 3D (§3). The miss distances are also a diagnostic (§4, signal 6).

**Coordinates.** Meshes, columns and edits are stored in world mm (scanner space, through the patch's crop affine; `baillarger.data.crop_affine`), so they are independent of the box. The editor also shows 0-based crop voxel coordinates and ITK-SNAP's 1-based numbering.

## 3. The editor (deliverable a)

### Architecture

- **A local Python server** on 127.0.0.1 (stdlib `http.server`, JSON endpoints, one request at a time). It holds the box, the columns, the factored correction-field solver and the list of points. The full volume is about 600 MB and stays on the machine. It is not a hosted page.
- **Batch, not live.** Placing, deleting or undoing a point changes only the list. **Remesh** rebuilds every edit from the list and sends the whole pial mesh back. The result depends only on the list (its order included), so a saved list reproduces the same surface.
- **A browser front end built on Niivue** (WebGL NIfTI viewer, loaded from a CDN at a pinned version). It loads the volume and meshes, shows three orthogonal slices with the meshes' cross-sections plus a 3D view, and reports click positions in mm.
- **To check in the spike** (§6, step 2) before relying on them:
  - Niivue's 2D mesh cross-sections: are they clear enough at 160 µm?
  - Swapping a mesh's vertex positions in place: how fast is it at about 150k vertices?
  - Which mesh format to send (VTK from `baillarger.mesh.write_vtk`, or GIfTI).

  If Niivue fails one of these, the fallback is a plain canvas slice view with the mesh cross-sections computed by the server.
- **Endpoints:** add or delete a point; undo and redo; set the range or the base; remesh (binary float32 positions of every pial vertex); save and load the edits file; the meshes and the volume. To come with §4: the priority maps and the worklist.

### What the user does

- **Views:** echo 2 with the trend removed (the default display), three orthogonal slices and a 3D view. White mesh and pial mesh as contour lines in the slices; pial coloured by a chosen priority map in 3D.
- **Place points at the crosshair:** a click moves the crosshair, then a key adds a *top* point (t), a *contact* point (k: a top for both banks' nearest columns) or a *free* point (f). Points show hollow until the next remesh.
- **Remesh** (m): the server maps each point to its column (root, arc length), solves, and returns the new pial surface. Each point's miss distance is listed, and points that fell back to free are shown in red.
- **Delete** a point from the list; undo and redo.
- **Range:** the correction's influence length (default 2 mm), used at the next remesh.
- **No vertex dragging.** A drag observes one vertex, but a saved drag replays as a click, which observes the triangle the click maps to. Where the base thickness jumps between neighbours, those give different surfaces (up to 2.7 voxels, measured in step 4). Points only.
- **Worklist:** "next" moves the slices to the next priority patch; the user marks it *fixed* or *OK* (§4).

### The re-solve: a correction field on the white mesh

The pial thickness is t = t_base + Δt. Δt is the posterior mean of a Gaussian field on the white mesh's vertices, conditioned on the edits.
- **Prior:** mean 0. Precision Q = τ (κ²M + K)ᵀ M⁻¹ (κ²M + K): K is the cotangent Laplacian, M the lumped mass matrix (the SPDE form of a Matérn field on a mesh, Lindgren, Rue and Lindström 2011). The range ρ = √8/κ is the slider's influence length. τ sets the prior SD of the correction (default 0.5 mm). The κ² term is the level term: the correction returns to 0 far from edits.
- **Observations:** each top edit gives y_k = arc length at the click, at its root (a barycentric row a_k on the white mesh triangle that holds the root), with SD σ_edit (default 0.03 mm, a fraction of a voxel).
- **Posterior mean:** Δt = Q⁻¹Aᵀ (A Q⁻¹Aᵀ + Σ)⁻¹ (y − A t_base).
  - Q is factored once when a patch loads (sparse Cholesky).
  - Each edit adds one column Q⁻¹a_k (one back-substitution, cached), and the k × k system is re-solved. This is expected to take tens of milliseconds at about 150k vertices [estimate; measured in the spike].
- **Why a correction, not a new thickness field:** where there are no edits, the base model's own detail stays. Near an edit, the change falls off smoothly over the influence length.
- **Contact edits:** two observations, one for each bank's root, each at the arc length to the clicked point along its own column.
- **Free edits:** a 3D displacement field on the pial vertices with the same prior per component, applied after the height. It is kept separate so that column-based edits stay 1D. A free point is observed at the nearest point of the pial surface *without* free displacement, so free points do not depend on each other's order.
- **Order of a rebuild:** every top and contact point first, then the free points and the tops that fell back to free, against the pial surface the tops give.
- **After each solve:** a self-intersection check of the changed pial region only (`baillarger.mesh_intersect.faces_hitting`). Intersecting faces are shown, not fixed.
- **Areal borders.** Thickness can kink at an areal border, and a correction should not spread across it. The first version has only the slider. A drawn barrier (edges of K cut along a stroke) is later work.

### What is saved

**The edits, not the mesh.** One JSON file per patch, `prototypes/data/pial_edits_<patch>.json`:
- per edit: kind (top, contact, free), world mm, crop voxel coordinates, root(s) and arc length(s) at the time of the edit, miss distance, base model, influence length, time;
- per worklist patch reviewed: *fixed* or *OK*, and the time.

The pial mesh is rebuilt from the base model plus the edits. So:
- **edits replay** onto a later or different base model (the roots and arc lengths are recomputed from the world positions);
- **edits are ground truth.** They join the points already marked in ITK-SNAP (23 at M1/S1, 35 at the V1V2 banks, and the PCC pieces). Every model can be scored against them.

The rebuilt pial mesh is also exported for other tools: VTK in world mm, and an ITK-SNAP overlay of the cortex it bounds, in `prototypes/data/pial_<patch>_*`.

## 4. The edit-priority diagnostic (deliverable b)

The diagnostic is a set of maps, read by eye. It has no pass threshold. Each signal is a per-vertex map on the pial mesh, shown on its own. A combined score orders the worklist only.

1. **Disagreement between the base models.** The spread of t_m(r) across the chosen models (§1), per vertex: range, and SD where there are three or more. This is the main signal. Places where all models agree and are all wrong can be found only by the other signals.
2. **Contact classes** from finding 27, at the boundary where each column's top lies: contact-like, fold-like, overlap-like (both sides already past their expected thickness: a fused sulcus or a label error), mixed.
3. **Thickness surprise and uncertainty.** |t − μ|/σ at each root against the thickness field fitted without that root's evidence, and σ itself. A large σ marks the fused stretches with no evidence (finding 16).
4. **No faults in the bands.** Where a band is visible (the V1 stria, by finding 10's definition first), its depth relative to the current top should vary smoothly between neighbouring columns. A jump means the top or the column is wrong (main plan, *Rules for the work*: no faults within a stack). This is the signal closest to the laminar goal.
5. **Image fit at the top.** Along the column at the top:
   - in open cortex, is there an edge to fluid, and how far from the top is the strongest one?
   - at a contact, are there two thin superficial layers meeting, with no fluid gap (finding 17)?
6. **Geometry.** Self-intersections; thickness outside about 1.2–4.5 mm; column extension used (§2); miss distances of nearby free edits.
7. **Context, shown greyed:** columns within 2 mm of a box face, and hidden-gap or label-error flags. They are left out of the worklist.

**The worklist.** Vertices whose combined score lies in the top few percent are joined into connected patches on the mesh. Patches are ordered by area × mean score. The combination is a weighted sum of the signals, each scaled by its own spread over the patch. The weights are displayed and can be changed, and they decide only the order.

**How the diagnostic is judged.** Over time, compare where the user made edits, and where they marked *OK*, with each signal's map. Report it as views and tables:
- how much each signal is raised where edits were made, against where patches were marked *OK*;
- how far each base model's top was from each edit, by region and contact class. This also shows which base to start from where.

## 5. Tests

Each piece gets exact-answer tests in `prototypes/scripts/test_pial_edit.py` before it is used on data. Diagnose failures, write a failing test, then fix. Never tune constants to hide a symptom.

1. **No edits:** the pial surface equals the base exactly.
2. **One edit, small σ_edit:** t at the edited root equals the edit to within σ_edit.
3. **The correction against a dense solve:** on a small flat mesh, Δt equals the posterior computed with dense matrices. On a large flat mesh, away from its edges, its shape follows the 2D Matérn (ν = 1) kernel κr K₁(κr) within the discretisation error, and its range scales with the slider.
4. **Adding edits one at a time** gives the same Δt as solving for all of them at once.
5. **Click mapping:** on finding 28's phantoms (the wavy body, the plane 4° off the grid), the (root, arc length) from the carried fields equals the exact foot point and distance, within the march's measured error (finding 28's table).
6. **Contact edit** on apposed slabs (`baillarger.phantoms` P4: banks 12 and 18 voxels, known contact): one click at the contact gives both banks' tops at it.
7. **Column extension** on P4 and on a wedge of two banks at 45°: extended columns carry on along the known normal, and the thick bank's column reaches its true top beyond the equal-arrival ridge.
8. **Replay:** edits replayed onto a perturbed base give the same tops at the edited roots, and the same Δt shape.
9. **Free edits:** the displacement equals the click at the clicked point, and it is 0 far away.
10. **Timing,** measured, not a pass criterion: the load time (factorisation) and the per-edit time on the M1S1 and V1V2 patch meshes.

Then, on real data:
11. **The four marked faces:** place the user's existing ITK-SNAP points as contact edits. Report:
    - each point's miss distance and whether it fell back to a free edit;
    - the base model's top error at each point before the edit;
    - the new pial surface over echo 2 with the trend removed, on the faces' slices.

## 6. Order of work

1. **Pial-as-height builder** (`prototypes/scripts/pial_height.py`): white mesh, the march from it, the columns and their extensions, t_m(r) per base model. Export VTK and an ITK-SNAP overlay per base model. Tests 5 and 7.
2. **Niivue spike** (`prototypes/scripts/pial_edit/static/`): load one box with its meshes; check the three points in §3, *Architecture*. Record what was found in *Results*.
3. **Correction-field solver** (`prototypes/scripts/pial_edit/solver.py`): tests 1–4, 6, 8–10.
4. **Server and editor** (`prototypes/scripts/pial_edit/server.py`): edits, undo, saving, replay. Check it in the Browser pane through a `.claude/launch.json` entry (the dev-workflow `browser-check-localhost` skill).
5. **Priority maps and worklist** (`prototypes/scripts/pial_priority.py`), signals 1–7. Then test 11.
6. **The user edits a patch.** Then report how the diagnostic did (§4, *How the diagnostic is judged*).

Steps 1 and 3 do not depend on each other and can run in parallel. Step 2 can start at once with any mesh.

If the tools prove useful, the builder and solver move into `src/baillarger/` through a branch and a PR (`CLAUDE.md`). That is written up as a proposal first.

## 7. Rules

- **Read first:**
  - `CLAUDE.md`;
  - the main plan, `reports/Seismic flattening for cortical layers.md`: the top summary, *Rules for the work*, *Method*, *The fronts as PDE problems*, findings 17, 25–28;
  - the memory notes listed in `MEMORY.md`.
- **Prototype code only,** in `prototypes/scripts/`. Do not edit `src/`, commit, branch or open PRs. The prototypes and reports are untracked by design.
- **Heavy runs:** `scripts/memguard <GB> <command>`, one at a time. Size the cap at about 1.5× the run's measured or expected peak; measure a small case first. memguard runs its command in the background, so pass scripts as files. The editor's server is not a heavy run once a patch is loaded, but loading a patch (march, columns, factorisation) is measured once under memguard.
- **Always 3D.** Columns, fields and the correction are computed in 3D or on the 3D mesh. Slices only show results and take clicks.
- **Display echo 2 with the trend removed.**
- **Stay away from box faces:** whole patches loaded from the full volume; columns within 2 mm of a face are greyed and kept out of the worklist.
- **Never label banks by geometry.** A column's identity is its root.
- **Ground truth from the user.** Edits and *OK* marks are the reference.

## Open questions

- **The base models** (§1).
- **The column extension rule** past the ridge of arrival (§2): whether the sign-continued data normal is good enough at kissing faces, or whether per-front columns are needed there.
- **The contoured base (step 5):** do its slits follow the contacts in the user's reading? What to do in the fused cortex near the box faces, where nothing is carved? Where does its genus of 92 come from?
- **Contact clicks:** on real data a click at a ridge is often on a shock where one bank meets itself, not on a contact, and the contact search can pair two columns of the same bank (step 4 follow-up). It needs a stronger pairing rule.
- **The influence length:** one length for the whole patch, or shorter near contacts, where the two banks' tops are tied.
- **Barriers at areal borders:** whether they are needed in practice.
- **Using the edits as evidence:** whether to feed them back into the thickness field and the contact rule (as visible tops and contact sums t₁ + t₂ = L), or keep the editor as a correction layer only. Out of scope here (§ *Not in scope*), but the edits file carries what that would need.

## Sources

- **Matérn fields on meshes (SPDE approach):** Lindgren, Rue and Lindström 2011. From memory.
- **Precedent for control-point editing of surfaces:** FreeSurfer's control points and pial edits in Freeview, followed by a re-run of the pipeline. Slow (minutes or more) and not local. From memory.
- **Niivue:** the NiiVue WebGL viewer (github.com/niivue/niivue). Its mesh features are to be checked in the spike.

## Results

### Step 1: the pial-as-height builder (done, on M1S1; two stand-in base models)

**Code.** `prototypes/scripts/pial_height.py` (library), `pial_height_run.py` (one patch), `test_pial_height.py` (14 tests, all passing: `nix develop --command uv run pytest prototypes/scripts/test_pial_height.py`, 95 s, peak 0.8 GB).

**How it is built** (plan §2 as specified, with these choices made):
- **The march** is `mesh_smoothing.march_band` from a 2.5-voxel band round the twicing 0.32 mm mesh, through cortex and fluid (missing tissue excluded), metric as finding 27 (data normal signed by the WM-distance gradient, weight coherence, plus 0.1 × that gradient; cost ratio 3).
- **The columns follow D∇T, not ∇T** (D = M⁻¹, the dual metric): under an anisotropic metric that is the characteristic direction. The test on the wavy body confirms it at ratio 3.
- **Tracing:** RK2, step 0.25 voxel, a sample kept every 0.5 voxel. Columns are traced again on demand rather than stored for the editor: tracing all 439k columns of M1S1 takes 16 s.
- **The ridge** is where the characteristic direction vanishes or turns back, or arrival stops increasing. In cortex the column then follows the layer normal (signed by continuity); in fluid it goes straight on. Entering fluid always switches to straight on, up to 1 mm past the entry.
- **The length cap is a fixed 5 mm**, not plan §2's 1.5 × expected thickness: there is no thickness field yet. Most extended columns stop at the cap before reaching the other bank's WM.
- **The two stand-in base models** (plan §1 is still open): *label*: the trilinear fluid label's 0.5 crossing, or the ridge of arrival where no fluid comes first. *One pass, equal weights*: finding 27's pass (root patches of 6 voxels, ratio 3); the top is the first sample in fluid, or in the territory of a patch more than 18 voxels from the column's own patch along the white surface, placed midway between samples (±0.25 voxel).

**Tests against exact answers** (mm; medians, p95 in brackets):

| | Ratio 1 | Ratio 3 | Tolerance (finding 28 plus interpolation) |
|---|---|---|---|
| Click mapping, wavy body: arc length error | 0.007 (0.025) | 0.006 (0.025) | 0.02 (0.06) |
| Click mapping: root error | 0.034 (0.110) | 0.009 (0.028) | 0.06 (0.20); 0.015 (0.04) |
| Column at 5 / 10 voxels from foot + s n | 0.020 / 0.035 (0.076 / 0.137) | 0.005 / 0.007 (0.020 / 0.028) | as the root, + 0.01 (0.03) for the vertex's own error |
| Label top − T (voxels; staircase of the label) | +0.008, IQR 0.32 | −0.005, IQR 0.36 | median < 0.25 |
| P4, bank B (95 columns): ridge met at | 23.00 (exact 23) | 23.00 | ± 1 |
| P4: x at arc length 30; off its own line | 25.500 (exact 25.5); ≤ 0.046 voxel | 25.500; ≤ 0.006 | ± 0.1 voxel |

The 45° wedge (plan test 7, second case) is **not done**. Its answer depends on which version of the data normal is used near the bisector (the level-set plan's versions A–C), and that is not decided.

**M1S1, the whole patch** (151³; 439,447 white vertices, 272,504 of them ≥ 2 mm from the box faces, which is what the counts below cover; 109 s; peak 7.0 GB under `scripts/memguard`):
- **Columns:** 171,575 reach fluid; 103,886 meet a ridge in cortex and are extended along the layer normal. Stops: 5 mm cap 128,220; fluid extension 102,477; entered WM 21,786; left the domain 20,021; no direction 0.
- **Label model:** top at fluid 160,669 (thickness median 2.61 mm, p5–p95 1.98–3.73); at the ridge 103,751 (2.84; 2.24–3.96); none 8,084. Of the ridge tops, 94,241 have no fluid anywhere along the column: fused or self-meeting cortex, not near-misses of the fluid.
- **One pass, equal weights:** fluid 122,663 (2.52); another front 135,649 (2.36; 1.72–3.00); none 14,192.
- **Where both models place the top at fluid** they agree to 0.024 mm (median; p99 0.10). That is the difference between the two placement rules alone.
- **Where they disagree.** Overall |difference| median 0.075 mm, p75 0.56, p90 1.37; 38 % of vertices differ by more than 0.16 mm and 26 % by more than 0.5 mm.
  - **Of the columns that reach open fluid, 26 % (41,736) get an earlier top from the one pass** (another front first, at median 0.77 of the label's thickness, range 0.46–0.95). This looks like fronts meeting themselves at fundi (fold sheets), which the separation rule takes for contacts: the open problem of findings 20 and 27 [inference: not checked view by view]. **It is a defect of that stand-in base model, not of the builder.**
  - Where the label model's top is the ridge and the one pass's is another front (91,940), the one pass is earlier by median 0.21 mm (p75 0.84).
- **The pial meshes tangle where neighbouring tops jump.** In fused cortex and at fundi, neighbouring columns' tops switch between fluid, ridge and another front, and the triangles between them cross (`pial_M1S1_sections.png`: spikes across the fused zones, the dark contact line in plane k, and the fused top of plane i). The disagreement and the switching of top kind between neighbours are what §4's priority maps should pick up. Self-intersections were not counted.

**Views:** `prototypes/figures/pial_M1S1_sections.png` (three planes through the patch centre: echo 2 trend removed; white surface green, label top blue, one-pass top orange). In ITK-SNAP: `pial_M1S1_overlay.nii.gz` with `pial_labels.txt` over `pial_M1S1_echo2_detrended.nii.gz`. Meshes (world mm, line up with the volume): `pial_M1S1_white.vtk` (near_box_face, ridge_mm, fluid_mm, stop_why, extended), `pial_M1S1_label.vtk` and `pial_M1S1_onepass_equal.vtk` (t_mm, kind, diff_mm). Numbers: `pial_M1S1.json`.

**Run:** `nix develop --command scripts/memguard 12 uv run --with matplotlib python prototypes/scripts/pial_height_run.py prototypes/figures prototypes/data M1S1`.

### Step 2: the Niivue spike (done)

**Setup.** `prototypes/scripts/pial_edit/server.py` (stdlib, 127.0.0.1) and `pial_edit/static/index.html`. M1S1 whole patch (151³): `fronts_eval_M1S1_echo2_detrended.nii.gz` and finding 28's white mesh `mesh_M1S1_twice_0p32mm.vtk` (439,447 vertices, 872,882 triangles). Niivue 0.69.0 (May 2026), loaded from jsdelivr's bundled build (`@niivue/niivue@0.69.0/+esm`; the plain `build/niivue/index.js` imports bare module names a browser cannot resolve). Measured in the Claude desktop app's browser pane (Chromium) on the user's Mac. The stand-in for the re-solve is a Gaussian bump along the normals (`POST /bump`); it measures the round trip only.

**The three questions.**
1. **Niivue's 2D mesh display is not usable for this. The editor draws its own sections.**
   - Niivue shows a mesh on a 2D slice as the part of the 3D surface within a slab of the slice (`meshThicknessOn2D`), drawn with depth testing against the opaque slice. Most of it is hidden: at 0.08–0.16 mm only fragments show, at 0.5 mm a band to one side of the true crossing. X-ray mode (`meshXRay`) did not change this.
   - **Replacement:** the exact plane section (triangles crossing the slice plane, cut linearly) for each of the three tiles, drawn on a 2D canvas over the WebGL canvas, placed with Niivue's own tile geometry (`screenSlices`: `leftTopWidthHeight`, `leftTopMM`, `fovMM`; screen y runs against mm). Plain JavaScript over all 873k triangles: **16–22 ms** for three planes (about 8,300 segments).
   - **Checked:** the overlay's position of the crosshair, sent back through Niivue's `canvasPos2frac`, returns the crosshair to 0.01 mm in all three tiles. At 4× zoom the white mesh's section runs along the cortex-to-WM boundary of the image at sub-voxel precision in all three tiles. Niivue's and `nibabel`'s mm-to-voxel mappings agree for this file (sform, diagonal).
2. **Swapping vertex positions in place is fast enough.** `mesh.pts` is changed, then `mesh.updateMesh(gl)` and `drawScene()`.
   - Per edit, once the page is warm: server bump < 1 ms; transfer 4–10 ms (all 5.3 MB of positions, or only the moved vertices at 65–90 kB, about the same here because it is local); Niivue's rebuild 21–45 ms; sections 17 ms. **Total 25–50 ms.**
   - About one edit in ten took 200–330 ms, all of it in `updateMesh` straight after the fetch. With a short pause after the fetch it never did (37 ms each time). The likely cause is garbage collection or the browser's own work, not the mesh size [inference].
   - `updateGLVolume()` is not needed after a mesh change and adds about 30 ms; `drawScene()` is enough.
3. **Format: binary arrays, not VTK.** The ASCII VTK file is 34 MB and is parsed once by the server (cached as `.npz`, 0.5 s). The page gets `uint32` counts, `float32` positions in world mm and `uint32` triangles (fetch 40–70 ms, building the `NVMesh` 150–700 ms). Volume load: 0.4–0.5 s.

**Gotchas for step 4.**
- `screenSlices` is updated only by `drawScene()`. After a resize, call `drawScene()` before reading it, and keep the overlay canvas at the GL canvas's own CSS size (not the box's).
- `setPan2Dxyzmm([x, y, z, zoom])`: the tile centre is (c₀ − pan)/zoom, with c₀ the centre at zoom 1. So to centre on X at zoom z, pan = c₀ − z·X (measured on all three tiles).
- When the browser pane is hidden, its screenshots can show an old frame. Read the canvas's pixels or front the tab.
- Niivue's ESM build needs the `+esm` URL. Pin the version.

**Run:** `.venv/bin/python prototypes/scripts/pial_edit/server.py 8765 <volume.nii.gz> <mesh.vtk> ...`, then open `http://127.0.0.1:8765/` (in the desktop app's browser pane through a temporary `.claude/launch.json` entry, removed after use). The server caches each parsed VTK as `<mesh>.spike.npz` beside it. The page's "Bump at crosshair" (key b) moves the first mesh; `window.spikeLog` holds the timings.

### Step 3: the correction-field solver (done)

**Code.** `prototypes/scripts/pial_edit/solver.py`; tests in `prototypes/scripts/test_pial_solver.py` (13, all passing, 13 s, peak 0.9 GB); timing in `pial_edit/solver_timing.py` (test 10).

**How it works** (§3 as specified, with these choices made):
- **The prior is never formed.** Q = τ² L M⁻¹ L with L = κ²M + K, so Q⁻¹ = τ⁻² L⁻¹ M L⁻¹. Only L, a one-ring sparse matrix, is factored, once per range. Each edit's column Q⁻¹a is two solves with that factor, cached.
- **Parameters:** range ρ = √8/κ, prior SD σ₀ (τ² = 1/(4πκ²σ₀²)), defaults 2 mm, 0.5 mm, σ_edit 0.03 mm. The box faces' mesh edges are natural boundaries.
- **The factorisation must pivot on the diagonal.** SuperLU (`scipy.sparse.linalg.splu`) with `diag_pivot_thresh=0` and `SymmetricMode`, ordering MMD on AᵀA + A. With the default partial pivoting, the 320k-vertex PCC mesh had not finished after 10 minutes at 11 GB; with diagonal pivots it took 2.4 s and 1.3 GB. No CHOLMOD is installed.
- **The session** (`Session`) ties the field to step 1's geometry:
  - *Top edit:* the root and arc length carried by the march at the click; the root's face and barycentric weights on the white mesh. The miss distance is the click's distance from the root's column (the barycentric blend of its vertices' polylines). Above one voxel, the edit falls back to a free edit.
  - *Contact edit:* the top edit, plus the column that passes nearest the click from the other side: the root more than 1 mm away, its direction at the click opposed.
  - *Free edit:* a 3D displacement field per component (same prior and factor), observed at the pial point nearest the click.
  - *Saving:* edits are saved as world mm with their roots, arc lengths and miss distances, and replayed from the world positions onto any base.
  - The base thickness is kept in voxels and only the correction is added, so with no edits the pial surface is the base exactly.

**Tests against exact answers.**

| Test (plan §5) | Result |
|---|---|
| Cotangent stiffness | rows sum to 0; zero on linear functions inside; mass = area; positive semidefinite |
| 1. No edits | the field, the session's thickness and the pial points are the base exactly |
| 2. One edit (σ_edit 0.03 mm) | within σ_edit at the edit; base 1.4 mm away to 0.1 mm |
| 3. Against the dense formula (4 edits, small mesh) | equal to rtol 10⁻⁸ |
| 3. Matérn shape, 16 mm flat mesh at 0.08 mm spacing (the white meshes' edge) | covariance against σ₀² κr K₁(κr) over 0.1–1.5 ranges: max error 0.0125 σ₀² (ρ 1 mm), 0.0034 (ρ 2 mm); prior SD 0.512 and 0.504 (0.5) |
| 3. The error is discretisation | ρ 1 mm, spacing 0.16 → 0.08 → 0.04 mm: error 0.023 → 0.0125 → 0.0034 σ₀²; SD 0.531 → 0.512 → 0.504 |
| 4. Order | edits added in reverse, or removed and added again: the same field to 10⁻¹⁰ |
| Top edit, P4 open bank A | thickness at the root 10.505 voxels (exact 10.5); miss 0.000 voxel; the column's point at it within 0.1 voxel of the click |
| 6. Contact on P4 (click on the contact x = 25.5) | arc lengths 15.99996 and 29.99994 voxels (exact 16, 30); thickness at both roots within 0.05 voxel; both columns' tops at the click (bank B's miss 0.14 voxel, the vertex spacing) |
| 8. Replay onto a perturbed base (smooth, up to ±1.8 voxels) | same roots and arc lengths; tops at the edited roots within 0.05 voxel; 40 voxels away the perturbation shows through to 0.05 voxel |
| 9. Free edit | the displaced surface passes 0.003 voxel from the click; nothing more than 0.001 mm moves beyond 3 ranges; removing it restores every point exactly |

A first version of the Matérn test divided by the vertex's own discrete variance instead of σ₀². That doubled the apparent error at ρ = 1 mm (0.036). The test now compares the covariance itself and checks that the error falls under refinement.

**Timing** (test 10; a measurement, not a criterion; finding 28's twicing meshes; 20 edits at random points; `pial_solver_timing.json`):

| Mesh | Vertices | Load (factor) | Peak | Per edit (two solves) | Correction at 1 / 10 / 20 edits | Vertices moved > 0.001 mm per edit | Change of range, 20 edits |
|---|---|---|---|---|---|---|---|
| PCC_SPL | 319,946 | 2.7 s | 1.7 GB | 63 ms | 0.7 / 5 / 22 ms | 10,283 (max 15,627) | 3.6 s |
| M1S1 | 439,447 | 5.8 s | 3.1 GB | 94 ms | 0.9 / 9 / 37 ms | 9,101 (max 15,403) | 7.4 s |
| V1V2 | 652,269 | **66 s** | **7.6 GB** | **306 ms** | 1.3 / 16 / 58 ms | 11,318 (max 19,031) | **70 s** |

- With step 2's page round trip (25–50 ms), an edit on M1S1 shows in about 0.15–0.2 s: fine for clicks, about 5–6 updates a second while dragging. This is slower than §3's estimate of "tens of milliseconds".
- **V1V2: the time is in SuperLU's ordering, and its first timing came from a broken factor** (checked after step 3; `v1v2_why` runs, not kept):
  - V1V2's mesh had **124 triangles with a zero-length edge** (vertices closer than the VTK's 0.1 µm rounding). Their cotangent weights reached 10²⁹⁴. A 440k-vertex piece of V1V2 then failed as "exactly singular", and the full mesh's 66 s factor was never checked. **Fix:** the prior welds vertices closer than 10⁻³ × the median edge and drops the triangles that collapse (75 vertices on V1V2; none on M1S1 or PCC_SPL). V1V2's solve residual is now 8 × 10⁻¹¹ (the others 2 × 10⁻¹⁵). There is a test with a duplicated vertex and a zero-area triangle.
  - **Welding did not change the time** (64 s, same fill). Pieces of V1V2 cut along its long axis took 1.2 s at 163k vertices, 5.9 s at 326k and 16 s at 437k, against 5.6 s for M1S1 at 439k. **With COLAMD ordering instead of MMD, the full V1V2 factors in 11 s**, though with twice the fill (125M) and 5× the work estimate (Σ column count²). So the time goes into the multiple-minimum-degree ordering, which grows steeply on this mesh, not into the numerical factorisation. Why it does on V1V2 and not M1S1 is not known: V1V2 has genus about 18 against M1S1's 0, and thin blades [inference: not tested].
  - Vertex degree is not the cause (median 6, max 10 on both).
  - Speed is not a goal (the editor is a batch feasibility check), so neither the ordering nor a local factorisation is pursued. If V1V2's 64 s load matters, use COLAMD for it.
- The correction is a dense n × k product and grows linearly with the number of edits. Each column is negligible beyond a few ranges, so it could be kept sparse.
- The session on a real patch is step 4.

### Step 4: the server and editor (done, batch; on M1S1)

**Code.** `prototypes/scripts/pial_edit/editor_server.py` (server), `pial_edit/static/editor.html` (page), `pial_edit/editor_timing.py` (the check below). `pial_height_run.py` now also writes `pial_<patch>_session.npz` (587 MB for M1S1: white mesh, columns, carried fields, both stand-in bases), which the server loads in 4.6 s with the factor. The spike's `server.py` and `index.html` are kept as they were.

**Run:** `.venv/bin/python prototypes/scripts/pial_edit/editor_server.py 8766 prototypes/data/pial_M1S1_session.npz prototypes/data/pial_M1S1_echo2_detrended.nii.gz`, then open `http://127.0.0.1:8766/`. Edits are saved to `pial_M1S1_edits.json` beside the session file.

**The design changed to batch** at the user's direction (§3): points are only listed, and Remesh rebuilds from the list. A first, incremental version (each edit solved at once, vertex drags, undo by removing an observation) worked and was timed (top edit 115–150 ms server side, free and contact edits about 600 ms), but replaying its saved edits did **not** reproduce the session: up to 2.7 voxels of thickness and 30 voxels of position. The cause was the vertex drag: it observes one vertex, while its saved form replays as a click, which observes the triangle the click maps to, and the base thickness jumps there. Batch rebuilding from points only removes the problem by construction.

**Bugs found on the real patch, fixed with tests:**
- **Clicks on a ridge got a root on neither bank.** The click mapping interpolated the carried root position over the 8 surrounding voxels, which at a ridge belong to two fronts: the average of two roots millimetres apart. Now only corners whose roots agree (within 3 voxels of the root at the heaviest corner) are blended, each carrying its arc length to the click along its own column (the extra terms cancel when all 8 corners are kept). Tests: a click on P4's ridge maps to one bank, at 23.000 voxels on both sides.
- **A non-finite click (a drag past a column's end) entered the field as NaN**, made the thickness NaN and broke replay. Edits now reject non-finite positions and observations. Test added.

**Check on M1S1** (`editor_timing.py`; `pial_editor_M1S1.json`; 16 points: 8 tops at a base top moved 0.1–0.4 mm along its column, 5 contacts at the ridge point of columns whose label top is a ridge, 3 free points 0.2 mm off the surface):

| | Result |
|---|---|
| Load (session file, factor) | 4.6 s; peak 4.6 GB |
| Remesh, 16 points | 5.1–5.9 s (positions 0.16 s); in the browser, 1 point: 1.1 s including transfer and drawing |
| Remesh again | identical (max difference 0) |
| The same points shuffled | max difference 3 × 10⁻¹⁴ voxel |
| Save, then a fresh server loads and remeshes | identical (0); 5.3 s |
| Range 1 mm and back; base switched and back | 8.0 s each for the range (refactor 2.9 s), 5.0 s for the base; back to identical |
| Tops: fell back to free | 1 of 8 (a click above the top, miss 0.17 mm, just over the one-voxel limit); the others' misses 0.02–0.08 mm |
| Contacts: both sides found | 3 of 5 (misses 0.06–0.15 mm on the first side, ≤ 0.03 on the second); 2 fell back to free (first-side misses 0.23 and 0.70 mm) |
| Free points | the surface passes through them (P4 test: 0.003 voxel) |

- **In the browser** (the desktop app's pane): a top point placed 0.3 mm below the current top, then Remesh. The nearest pial vertex ends 0.028 mm from the click. The point list, its miss distance, the marker and the new section all show.
- **Contacts on real data miss more often than tops.** Two of five ridge clicks were more than a voxel from the first side's column, even with the ridge fix. At a ridge the columns turn into their extensions along the layer normal, and the march's carried root and the traced column need not agree there [inference: not examined point by point]. Open.
- **Not done:** the pial self-intersection check after a remesh (§3); priority maps and worklist (step 5); V1V2 and PCC_SPL sessions.
- **The user's reading (2026-10-27), after using the editor on M1S1:** the interface is usable; the default (base) surface is still quite bad. See the step 4 follow-up for why.

### Step 4 follow-up: why clicks miss, and why the base pial is bad

**The failing clicks, one by one** (`pial_edit/diag_click_miss.py`; views `prototypes/figures/pial_click_<i>.png`; points of step 4's check):
- **All three failures (contacts 9 and 12, top 3) sit on a shock where one bank's front meets itself.** The 8 voxels round the click carry roots 2–5 mm apart: at point 9 four corners carry one root and four another, 4.6 mm apart. Point 3's click had been moved 0.21 mm past its column's ridge, onto such a shock.
- **The carried root is then from whichever part of the front arrived first, and that part's traced columns pass beside the click, not through it** (0.3–0.95 mm at point 9, 0.13–0.32 mm at point 12). The traced columns stop at their own ridges and turn onto the layer normal, while the march's characteristics end at the shock. The two need not agree there.
- **The contact search paired two columns of the same bank.** At point 9 the click's own column and the carried root's columns both start on the left bank's WM and cross the dark contact line together (`pial_click_9.png`). The "opposed directions" test let a nearly parallel pair through. The successful contacts 8 and 10 are not checked for this (point 8's two roots are only 1.2 mm apart).
- So on real data, a click at a ridge is often not at a contact, and the fallback to a free edit was the right outcome there. The pairing rule needs a stronger test than opposed directions.

**The base pial is bad mostly because of the columns, not the tops** (`pial_base_quality.py`; `pial_base_quality_M1S1.json`; views `pial_base_label.png`, `pial_base_onepass_equal.png`):

| | Label base | One pass, equal weights |
|---|---|---|
| Thickness roughness, \|t − 1-ring median\| p50 / p90 / p99 (mm) | 0.001 / 0.08 / 1.16 | 0.000 / 0.08 / 2.08 |
| Vertices rougher than 0.16 mm: all; top at fluid; at a ridge or another front; no top | 6.9 %; 1.8 %; 13.6 %; 20.1 % | 4.7 %; 1.0 %; 5.3 %; 30.7 % |
| Intersecting triangles (of 872,882); ≥ 2 mm from the box faces | 336,640; 257,464 | 273,492; 203,893 |
| After robust smoothing of t (1-ring medians × 3, then twicing at 0.32 mm): roughness p99 | 0.012 | 0.012 |
| After smoothing: intersecting triangles; inside | 308,460; 231,872 | 182,575; — |

- **Open cortex is fine.** Tops at fluid are smooth (p90 0.02–0.08 mm), the two bases agree there to 0.024 mm (step 1), and the smoothed top traces the fluid edge.
- **In fused cortex the tops jump,** because a column's top is its first ridge or foreign front, and those lie at scattered depths. Smoothing the thickness removes the jumps.
- **But a smooth thickness still gives a tangled surface: a quarter to a third of all triangles intersect.** The columns themselves cross: they are traced from a noisy ratio-3 data metric and extended past their ridges. A height over crossing columns cannot be a clean surface, whatever the height.
- **Why earlier results looked better:** findings 11–25 scored where a contact *lies* at the user's marked points (within 0.08–0.3 mm). They never built the whole surface, where every column has to agree with its neighbours.

**What this means for the representation** (§2) [inference; for the user's decision]: the pial surface should start from a surface, not from per-column events. Contour it as the white surface is: the fluid boundary of the label where the cortex is open, and the competition's label boundary between far roots (the contact) where it is fused. A contour cannot cross itself. Edits could then be 3D (the free edits that already exist), and columns kept for laminar depth only. The alternative, columns that do not cross (an isotropic metric, or no extension past ridges), keeps 1D edits but is untested.

### Step 5: the contoured base and 3D edits (done, on M1S1)

**Code.** `prototypes/scripts/pial_contour_base.py` (the base), `pial_edit/solver.py` (`SurfaceSession`), `editor_server.py` (loads either kind of session file), `test_pial_solver.py` (4 tests on a slit phantom; all 36 tests of both files pass).

**How the base is built** (all 3D):
- **Open cortex:** the boundary of the tissue label (WM, cortex, missing tissue) against fluid.
- **Fused cortex:** the one-pass contact with equal weights (finding 27's pass: root patches of 6 voxels, cost ratio 3, label boundaries between patches more than 18 voxels apart along the white surface, sub-voxel positions).
  - Left out: pairs within 2 mm of a box face, and fold-like pairs (both sides reach the boundary before 0.8 of their patch's expected thickness).
  - The expectation is cruder than finding 27's field: each patch's median arc length at its visible tops (`open_thickness`, columns that run into fluid), the nearest patch's where it has none (65 % of patches have none).
- **The kept contacts are carved out of the tissue as a slit,** then the tissue is contoured and smoothed as the white surface (binary marching tetrahedra, tangential relaxation, twicing at 0.32 mm). Two slits were compared:
  - **slit1:** the voxel of each pair nearest the sub-voxel contact (sheets about ±0.08 mm from it);
  - **slit2:** both voxels (about ±0.16 mm from the label boundary).

**M1S1** (`pial_M1S1_contour.json`; 67 s, peak 3.5 GB):
- **Contact pairs:** 37,772. Of these, 17,006 (45 %) lie within 2 mm of a box face. Away from the faces: contact-like 14,329, mixed 5,819, fold-like 601, overlap-like 17. Carved: 20,165 pairs.

| | Label base (columns) | One pass (columns) | **Contour, slit2** | Contour, slit1 |
|---|---|---|---|---|
| Vertices; triangles | 439,447; 872,882 | same | 297,253; 591,762 | 288,539; 580,810 |
| Intersecting triangles | 336,640 | 273,492 | **0** | 32 |
| Topology | white mesh's | white mesh's | 17 pieces, genus 92, manifold | 48 pieces, genus 1,742 |

- **slit1 leaks:** a one-voxel slit that steps diagonally leaves tissue bridges across the contact, so its contour has handles (genus 1,742).
- **slit2 is the base:** no intersections and a manifold surface. Its genus of 92 is not explained yet. It may be the label's own handles, or slits left partly open where contacts were not carved [inference: not checked].
- **Views** (`pial_contour_M1S1_slit2.png`, the same three planes as `pial_base_label.png`): where the cortex is open, the contour follows the fluid edge; in fused cortex it continues as thin slits along the dark contact lines (plane k lower right, plane j centre, plane i). In plane k's centre it carves a pocket beside the fluid, to be judged by eye. Where the contacts are near a box face nothing is carved, so there is no surface in the fused cortex there (plane i, top right).
- **In ITK-SNAP:** `pial_M1S1_contour_overlay.nii.gz` with `pial_contour_labels.txt` (contact voxels carved by slit1 and slit2, slit2 only, fold-like not carved, near a face not carved) over `pial_M1S1_echo2_detrended.nii.gz`. Meshes: `pial_M1S1_contour_slit2.vtk` and `_slit1.vtk`.

**3D edits** (`SurfaceSession`):
- Every point says "the surface passes here". The displacement is a 3D field on the pial mesh (step 3's prior, one field per component), observed at the point of the *unedited* surface nearest the click, so the result depends only on the list.
- A **contact** point also observes the nearest point of a facing sheet (normals opposed) within 1 mm, so both sheets of a slit move to the click.
- *top* and *free* are the same on this base.
- **Tests on a slit phantom:** no edits gives the base exactly; the surface passes 0.001–0.003 voxel from each point; a contact moves both sheets; a free point moves only its sheet; reversing the list changes nothing.

**In the browser** (session `pial_M1S1_contour_slit2_session.npz`, base "contour"): the session loads; a remesh with a contact and a free point takes 0.49 s in total; the contact observed two sheets. Fixed on the way: the page found the axial and coronal tiles by position in Niivue's tile list, which in the 2×2 layout also holds the 3D render tile, so zooming went to NaN. It now finds them by orientation.

**The user's reading (2026-10-27), in the editor:** this default surface is much better than the column bases.

**Vertex dragging** (added after that reading; on surface sessions only):
- **Use:** the "Drag vertices" button or key d. Press on a pial line in a 2D tile: the nearest pial vertex within 0.4 mm of the click, and within 0.3 mm of the slice, is picked. Drag it within the slice's plane (its position across the slice stays). While dragging, only that vertex and the overlay move (a local preview, no server call). On release the page adds a drag point and remeshes.
- **What is saved:** a drag is the vertex's position on the unedited surface (`from_mm`) and the target (`mm`). A remesh observes that base vertex (one vertex, weight 1) with the whole displacement, so a saved drag rebuilds the same surface, in any order with the other points.
- **The same vertex dragged twice:** the last drag counts; earlier ones are listed as superseded. Undoing the last brings back the one before.
- **Why this does not repeat step 4's drag problem:** there, a drag observed one vertex but was saved as a click that replayed onto a triangle. Here the vertex itself is saved.
- **Tests** (`test_pial_solver.py`, all 39 of both files pass):
  - a dragged vertex lands 0.006 voxel from its target (within σ_edit);
  - drags and contacts in another order give the same surface (10⁻⁹);
  - last drag wins, and without it the earlier one applies;
  - the server saves and a fresh server replays drags identically;
  - undoing everything gives back the base exactly.
- **In the browser** (M1S1 contour session, the page's own mouse handlers): a vertex dragged 0.46 mm in the axial plane, then released. The saved base position is the vertex's own, the target moved only within the plane, and after the remesh (0.4 s) the vertex is 0.0016 mm from the target.

**Crossings, touching, and drags that refuse to cross** (after the user asked how the interface stops sheets crossing: before this, it did not):
- **Crossings,** after every remesh: pial triangles that cross another pial triangle or the white surface (`baillarger.mesh_intersect.intersecting_faces`, whole meshes: about 1–1.5 s on M1S1). Drawn thick red in the sections, counted, and grouped into places (0.5 mm), with key x jumping to the next. A place within 1 mm of a contact point is labelled "at a contact": there two sheets are pulled together on purpose.
- **Touching,** in violet (the user's definition: two surfaces within δ of each other; "Touch mm", default 0.1 mm). **One set of definitions** (`solver.py`, *touching*), used by the touching map, the contact search and the "at a contact" labels alike:
  - *Facing:* two places on the pial surface face each other when their unit normals point against each other (dot < −0.5). A point's normal is its triangle's (turned to the base surface's outward side); a vertex's is the area-weighted mean of its triangles'.
  - *Distance:* from a point to a triangle's surface (its closest point), never vertex to vertex.
  - *Touching:* a pial vertex within δ of a facing pial triangle, or within δ of the white surface (a different surface: no facing test), on a triangle that does not cross. Crossing takes precedence.
  - *A contact point's other sheet:* the nearest facing pial triangle within the contact radius (1 mm): the same facing test and distance, searched further. *At a contact* means within that radius of a contact point.
  - The base slits' gap (0.32 mm) is above the default δ, so unedited slits are not touching. **But smoothing narrows a slit near its open edges:** on the phantom, a one-voxel slit (0.149–0.160 mm in its interior) closes to 0.10 mm within about three smoothing widths of its edges, and those parts do touch. The real slits' ends presumably narrow the same way [inference: not measured].
  - *Tests:* the phantom's slit interior touches at 0.165 mm but not at 0.145 mm (surface distance; vertex-to-vertex would read longer); the contact search and the touching test find the same facing triangle at the same distance.
- **A kissing (contact) point and σ_edit:** each sheet is pulled to the click as an observation with σ_edit = 0.03 mm, the scatter the fit allows. The actual miss is the needed move × σ_edit² / (σ_edit² + σ₀²), about 0.4 % of it: on the phantom 0.0014 voxel. Only edits that pull against each other make it larger (up to about σ_edit per sheet, so about 2σ_edit = 0.06 mm between the sheets). That is below δ, so a contact point always shows as touching.
- **Keys:** the editor takes its keys (t, k, f, m, u, r, s, z, d, v, x, Ctrl/Cmd+Z) in the capture phase and stops them there, so Niivue's own canvas keys never see them (its k moves the crosshair one voxel along j, which placed contact points one voxel off; its m cycles the drag mode). Niivue's h, j, l and Ctrl+U/D still move the crosshair. Checked in the page: with the image focused, k put the contact point at the crosshair and the crosshair stayed; m remeshed and left the drag mode alone.
- **Crosshair boxes:** x, y and z boxes in the toolbar show the crosshair in world mm and follow it as it moves. Editing one (Return or Tab) moves the crosshair there and zooms about it, redrawing the sections with it.
- **Vertex dots** (for dragging): the pial vertices within half the median pial edge of a tile's plane (about 0.04 mm on M1S1), drawn as dots on the pial line; "Vertices" checkbox, key v, on by default. These are exactly the draggable vertices. In drag mode a press grabs the dot nearest the pointer within 10 screen pixels (it was 0.4 mm), and the dot a press would grab is ringed on hover. Checked in the browser: the press picked the vertex under the dot, and the drag was placed (no crossings).
- **Speed after the touching change:** that drag took 53 s on the server. One crossing-and-touching check took 24 s, and a first drag runs it twice (its initial check, then its result); the pull-back took 0.47 s and the edit 0.17 s. The cost is the surface-distance touching map over all 297k pial vertices (vertex pairs within δ plus an edge, then point-to-triangle distances), done after every drag on the whole meshes. The page's timing sum also undercounts it (the initial check is not added).
- **Faster drags** (at the user's request; their tolerance: a drag may differ from a full remesh by up to 0.1 voxel, since a full remesh comes later):
  - *Changes:* a drag's observation is added without a rebuild, with the same "last drag wins" rule. The three displacement components share one solved column, which makes full remeshes three times cheaper on that step too.
  - *Re-check near what moved only:* every vertex within δ plus two edges of a vertex that moved more than 0.01 voxel (old or new position), and the triangles on them. Crossing and touching are recomputed there and kept elsewhere.
  - *Smaller answers:* only the vertices that moved more than 0.01 voxel are sent. The server's positions are always computed in full.
  - *Memory:* the facing-triangle search now runs in chunks. The whole-mesh version had passed memguard's 10 GB cap on M1S1.
  - *The threshold matters:* the correction field never reaches exactly zero, so at 10⁻⁶ voxel most of the surface counts as moved. On M1S1, three drags (`dragtime.py`, a scratch script, not kept):

    | A vertex has moved above | Vertices moved | Re-checked | Server time per drag |
    |---|---|---|---|
    | 10⁻⁶ voxel | 58–74k | 63–80k | 12–15 s |
    | 10⁻⁴ | 35–38k | 39–52k | 8–11 s |
    | 10⁻³ | 22–24k | 27–37k | 5.3–7.9 s |
    | **0.01 (chosen)** | 11–13k | 14–23k | **3.2–5.3 s** (4 threads) |

  - At every threshold, a full remesh of the same list gave the same positions, crossing and touching triangles in these runs. The tests (all pass) include fast drags against a full remesh on the slit phantom.
  - **Profiled and split in two** (one drag profiled on M1S1, 4 threads). The 2.9 s had gone to:
    - the touching re-check (1.35 s), of which 0.6 s was a per-vertex Python loop and 0.36 s was de-duplicating candidate pairs before the facing filter;
    - four crossing checks (0.69 s);
    - the longest edge recomputed over all 600k triangles in three functions (0.45 s);
    - finding the re-check region with per-point Python lists (0.9–1.4 s once the rest was faster).

    **Fixes:** vectorised filters; the facing filter before de-duplicating; the longest edge computed once and passed down; neighbour pairs from SciPy's `sparse_distance_matrix`, and the region from one distance query against a small tree of the moved points.
  - **Two phases:** a drag answers with positions and crossings (all the pull-back needs), and the page then asks for touching (`POST /touching`), which rechecks the region the drag left pending. Regions of drags not yet followed up are merged.
  - **M1S1, 4 threads, four drags** (`dragtime2.py`, a scratch script, not kept):
    - drag phase **0.90–0.99 s** (edit 0.05, pull-back 0.36–0.38, region 0.13–0.15, crossing re-check 0.35–0.40);
    - touching phase 0.88–1.95 s (14–23k vertices rechecked);
    - a full remesh 13 s.

    After the four drags, a full remesh of the same list gave the same positions and the same crossing and touching triangles (0 differing).
  - **In the browser:** the drag shows after 1.08 s (status "touching updating…"), and the touching update arrives 1.6 s later, 2.7 s in all.
  - **Tests** (all 35 solver tests pass) follow each drag with its touching request, as the page does; one skips it, so that the pending region carries over. Fast drags with a contact point equal a full remesh, and the one crossing place left is at the contact (both sheets pulled together, within 0.0002 mm); with the 0.01-voxel threshold, movements below it there are not checked.
- **Pulling touching sheets apart** (the user's reading, 2026-10-27: drags refused to pull touching surfaces apart and only dragged their touching point around; in their session, 12 of 54 drags were pulled back, five to under 2 % of their path). Two causes, each now handled:
  - *Contact points pin the sheets together* (reproduced on the slit phantom). With sheet A pushed onto B and nothing else nearby, pulling either sheet away is placed in full. With a contact point 1 voxel away, pulling B away is refused at 8–16 % and A at 2 %: the contact point holds both sheets to one spot, and pulling one away bends it round that spot and through the other. One contact point 1.6–1.9 mm away holds the same drag alone. The pinning is what a contact point means, so it stays, and a refused drag now names the contact points that hold it: `Editor._holders` takes the contact points within twice the prior's range out (nearest first, by popping their observations with the cached columns kept, so nothing is solved again) until the full drag makes no new crossing, then puts back each one the drag does not need, and finally puts all back. The status line says "held by contact point #i (d mm away): delete it to pull these sheets apart", the point list highlights them, and the drag's entry says "held by #i". When taking out every nearby contact point does not free the drag, it names none. Tests: one contact point 1 voxel away is named and one 2.2 mm away is not; with a second holder 1.6 mm away both are named; after deleting them the drag is placed in full; and the positions after the search equal a full remesh. Cost on the phantom: 0.13 s per refused drag.
  - *The press could take the other sheet.* Where two sheets touch, their dots overlap, and a press took the dot nearer the pointer, whichever sheet it was on; "pulling apart" then pushed that vertex into the sheet meant to move. Now, when a dot of a facing sheet (outward normals dot < −0.5, the touching test's rule) is also within 10 px, the choice waits for 4 px of movement and takes the sheet whose outward normal opposes the movement: the one being pulled away from the other. The page computes vertex normals from the triangles with the outward signs of `GET /mesh/pial_sign`. Checked in the page on the phantom: with B's dot nearer the press, pulling −x moved A and pulling +x moved B, both placed in full.
  - **Load:** the earlier threshold runs used all cores for several minutes while another session's level-set run was also going, and pinned the user's machine. Timing runs now use 4 threads, one at a time.
- **Drags refuse to cross in a new place:** on release (`POST /drag`), a drag is refused if it makes triangles cross (pial against pial or against the white surface) that did not cross before and are more than 0.5 mm (`PLACE_MM`, the spacing of the crossing places) from every triangle that crossed before. A crossing that only shifts at its edge does not count: counting every newly crossing triangle refused harmless drags 2–4 mm from the crossing at a contact point, as in the user's session (drags 15 and 21: the first refused triangles were 3.7 and 2.4 mm from the vertex, at the edge of the 400-triangle crossing at contact points 0–2; on the phantom, pulling a sheet away 1.3 mm from a crossing was stopped at 5 % before and is placed now). A refused drag is pulled back along its path by bisection (8 steps) to the furthest point that makes none; the status line says where the first refused crossing was (position, distance from the vertex, against the pial or the white surface), and which contact points hold it (below). Only the drag's observed values change during the search (no new solves). Each trial checks the triangles the drag moved (`faces_hitting`, about 0.1–0.25 s). Points (top, contact, free) are not refused; their crossings are shown.
- **Pushing the other sheet** ("Push" checkbox; Shift on release flips it for one drag): the white surface still stops the drag (the same bisection against the white surface only). Then, up to 3 rounds, where the drag crosses another pial sheet that faces it (vertex normals dot < −0.5 against the dragged vertex's), that sheet's vertices less than a gap (twice the touch distance, 0.2 mm) in front of the dragged sheet are pushed to the gap, along the dragged sheet's outward normal: push points 0.3 mm apart, deepest first, at most 60 for a drag, their columns in one solve per round; a push made in an earlier round is moved further if needed. If crossings remain after the rounds (the pushes can make their own, where the pushed sheet meets a third), the pushes are dropped and the drag is a plain one, pulled back as without push, so pushing never places a drag less far (scaling the drag and its pushes back together instead placed two of the user's drags at 4 % and 0.4 %, against 62 % and 81 % plain). The pushes are saved with the drag (`pushes`: each pushed vertex's base position and target, as a drag is saved), shown as small squares, superseded with it by a later drag of the same vertex, and undone with it. Tests: on the phantom, pushing sheet A 2 voxels through B is placed in full with B pushed ahead and nothing crossing; a full remesh and a save and reload give the same surface; undo restores the base; pushing toward the white surface stops at it with no pushes. Checked in the page on the phantom (60 pushes, 0.5 s of pushing). On M1S1, the user's refused drags 21, 15, 9 and 11 (they saw 7, 16, 38 and 43 %) replayed from their edits: plain, 92, 62, 81 and 100 %, each stopped by a real crossing 0.6–0.7 mm from the vertex; with push, drag 21 is placed in full (26 pushes, 3.8 s) and the others as plain. A refused drag takes 4–9 s on M1S1, a placed one about 1 s.
- **At a contact** the two sheets are not merged and not kept δ apart: both are pulled to the clicked point (each within σ_edit). They coincide there, fall back to the base gap over the influence range, and show as touching around the point. Between contact points the sheets keep the base gap (±0.16 mm) unless edited, and the topology never changes. A true zero-thickness contact along its length would need snapping facing sheets within δ together, or one shared (non-manifold) sheet. Neither is built.
- **Tests** (all 46 of both files pass):
  - on the slit phantom with a white surface inside its left block, a drag across the slit added without pull-back shows 292 crossing triangles;
  - through `/drag` the same drag stops at x = 30.496 (the other sheet is at 30.5) with no crossings, and ends as touching;
  - a drag into the white surface stops at x = 23.521 (its face is at 23.5);
  - a drag that crosses nothing is placed in full;
  - the slit's sheets (0.16 mm apart) are not touching at 0.1 mm and are at 0.2 mm;
  - a contact point's sheets touch at the contact.
- **In the browser** (M1S1): a pial vertex dragged 1 mm past its nearest white-surface point (3.44 mm requested). It was pulled back to 71 % (moved 2.44 mm, stopping 0.003 mm short of the white surface; 91 triangles would have crossed), leaving no crossings, with touching shown. About 9 s per drag: pull-back 3.9 s, whole-mesh checks 1.4 s each.
- **Limits:**
  - A pulled-back drag stops at the crossing itself, with no margin, so it ends touching.
  - Bisection assumes crossings appear once along the path.
  - Drag timings undercount a first drag's initial rebuild.

**Run:** `.venv/bin/python prototypes/scripts/pial_edit/editor_server.py 8767 prototypes/data/pial_M1S1_contour_slit2_session.npz prototypes/data/pial_M1S1_echo2_detrended.nii.gz --base contour`.

**Open:**
- the fused cortex near the box faces (nothing carved);
- whether the carved slits follow the contacts in your reading (the views and the overlay);
- the cause of genus 92;
- fold-like filtering is only as good as the crude expected thickness.

### Contact clicks as evidence for the base (tested on M1S1; not in the editor)

The proposal (from the session "Contour mesh to laminar analysis workflow"): a contact click tells, for the root on each bank, how far that bank's front goes to the contact. Added to the thickness field's evidence, it should move the whole weighted contact, not only the 2 mm an edit reaches. Tested with the 23 marked M1S1 contact points (`M1S1_test_points.json`, "gap_all") as clicks; the editor's code is unchanged.

- **Method** (`click_evidence.py`, finding 25's setup: a box of 141 voxels centred on the points, every point ≥ 7.2 mm from a face; cost ratio 3; root patches of 6 voxels; the thickness field from column tops with finding 26's guards). A click's two fronts are the end labels of the column line through it; each side's arrival at the point, S, comes from the equal-arrival pass; log S replaces the field's evidence on the nodes of that root patch's seeds. Arrival, not arc length (the competition compares arrival with the belief). Each point is scored (finding 25's local two-sides score) only by a model that did not see it, except "in sample".

  | Weights | \|offset\| median (mm) | Median, IQR | Worst | Distance from a scored point to the nearest click |
  |---|---|---|---|---|
  | Equal arrival | 0.68 | +0.68, 0.17 | 0.87 | — |
  | Field from tops (finding 25) | 0.26 | −0.26, 0.23 | 0.66 | — |
  | Field + clicks, interleaved (even points score odd, and the reverse) | **0.11** | −0.10, 0.14 | 0.49 | median 1.1 mm |
  | Field + clicks, three spatial blocks (each scored by the other two) | 0.22 | −0.22, 0.24 | 0.52 | 2.1–2.8 mm (median per block) |
  | Field + clicks, in sample | 0.06 | −0.03, 0.14 | 0.21 | 0 |

- **The evidence reaches about a millimetre, not the whole contact, and the prior's scale is not why** (`click_reach.py`, same box, field and clicks):
  - *Distances in the field's own terms.* In the block split the scored roots lie 1.7 mm (median; 90th 4.6) along the white surface from the nearest click root, closer than the points (2.6 mm), and the z field's prior correlates them at 0.65 (median; 0.82 in the interleaved split). The field does link them.
  - *Not the tops pulling back.* |μ − S| at the scored roots (median, mm): interleaved, tops only 0.26 → tops + clicks 0.15 → clicks only, no tops 0.15; blocks, 0.26 → 0.27 → 0.25. Without any tops, clicks still do not predict roots 2–3 mm away. The mean bias shrinks (+0.11 → +0.07 mm) but the scatter stays.
  - *The prior's scale, chosen on the visible tops* (5-fold cross-validation over 4 mm spatial blocks of the tops' nodes; held-out log predictive density, the rule fixed before the run): ELL_Z 4 mm, SIG_Z 0.4 (0.861) and the current 2 mm, 0.2 (0.855) are tied, both calibrated (z SD 0.95, 0.90); 1 mm and 8 mm do worse (best 0.76 each). Rerun with 4 mm, 0.4, the click test gives the same: tops only 0.28, interleaved 0.12, blocks 0.22 mm.
  - *So the error varies from point to point at 2–3 mm.* With the tops-only field, μ − S at the end label on one side of the column line is +0.5 to +0.8 mm at points 2–15 and −0.1 to +0.3 at points 16–23; on the other side it ranges −0.6 to +0.4. A field smooth over millimetres cannot carry one click's correction to points whose errors differ. [inference: possible causes, untested: noise in S itself (arrival from the equal pass, through a metric steered by a locally unreliable normal); the units (tops in arc length, clicks in arrival); root identity (about 1 mm with 6-voxel patches); a real step in thickness, as at an areal border along the central sulcus (no faults in cortex, but facies changes). Which side is which bank was not checked per point, so the per-side pattern is only indicative.]
- **Field weights alone fix most of the base's miss, with no clicks** (`click_base.py`: the contoured base rebuilt on the M1S1 patch exactly as `pial_contour_base.py`, slit2, from the weighted pass). Distance from each point to the carved contact's midline (halfway between the two sheets):

  | Base | Carved at | Median | 90th | Worst | Mesh |
  |---|---|---|---|---|---|
  | Equal arrival (the editor's current base) | 22 of 23 | 0.66 mm | 0.81 | 0.87 | genus 92 |
  | Field weights | 22 | **0.23** | 0.48 | 0.55 | 2 crossing triangles; genus 111 |
  | Field + all 23 clicks (in sample) | 21 | 0.07 | 0.20 | 0.22 | 0 crossing; genus 86 |

  Point 7 is under 1 mm from the patch face and never carved; with clicks, point 6 (2.1 mm from it) is not carved either. The current base sits 0.5–0.9 mm off at every point, all toward the thick side (finding 25's equal-arrival miss). Views: `prototypes/figures/click_base_M1S1.png` (the three bases on the planes holding most points), `click_evidence_M1S1.png`. Editor sessions: `pial_M1S1_contour_field_session.npz` and `pial_M1S1_contour_clicks_session.npz` (open one with the editor's session argument; the clicks base saw the points it is scored on).
- **The same points placed as editor contact points** (`click_edits.py`: the editor's own `Editor`, unchanged; each point added as a contact point, key k, and remeshed). Distance from each point to the carved contact's midline (median; worst):

  | Base | Before | All 23 placed | Half placed, the other half scored |
  |---|---|---|---|
  | Equal arrival (current) | 0.66; 0.87 mm | 0.00; 0.22 | 0.24; 0.41 |
  | Field weights | 0.23; 0.55 mm | 0.00; 0.31 | **0.10**; 0.40 |

  With all 23 placed, both sheets reach every point (two sides found at 21–22 of 23; one at point 7, under 1 mm from the patch face). Crossings after the edits: 1,602 triangles in 10 places on the current base and 1,392 in 11 on the field base, all at contact points except one place on the field base (the two sheets pulled together at a point cross within it, as before). The surface moved by up to 1.3 mm, with 25,000–34,000 vertices moving more than 0.05 mm. Points 1.1 mm from a placed point (half placed) end up 0.24 mm off on the current base and 0.10 mm off on the field base: an edit corrects its neighbourhood, but how far off the base is decides what remains between edits. View: `prototypes/figures/click_edits_M1S1.png`.
- **Reading** [inference]: the larger, cheaper gain is the base's weights: the field-weighted pass instead of equal arrival takes the M1S1 kissing face from 0.66 to 0.23 mm with no user input. Clicks then help within about a millimetre of each, as edits do, and a longer prior does not change that. Open: the field-weighted base elsewhere in the patch (only this face has marked truth; finding 25's V1V2 banks missed with field weights at ratio 3), its two crossing triangles and higher genus, and the cause of the point-to-point scatter.

### Edits on the competition's inputs, the surface read off the owners (feasibility, M1S1)

The aim: restore the fronts' guarantee to editing. Every cortex voxel has one owner (a root patch), so a surface read off the boundaries between owners cannot cross itself. Edits change the competition's inputs, the pass is rerun, and the surface is extracted again; the mesh is never edited. `owner_edit.py`, on finding 25's box (141 voxels on the 23 marked points; cost ratio 3; root patches of 6 voxels; the field's weights from column tops as the start). The editor's code is not used.

- **An edit.** A contact click at x: the two fronts at x (the end labels of the column line through it) get their expected thickness set to their arrival at x, so both reach their belief exactly there; σ unchanged; 2 rounds (the territories move). Variant "spread 1 mm": the same log correction also given to the roots around each edited one, weighted exp(−d²/2(0.5 mm)²) along the white surface, up to 1.5 mm.
- **The surface, with no slit.** The tissue's exposed voxel faces against fluid, plus one face per contact pair (roots > 18 voxels apart, not fold-like, ≥ 2 mm from a box face), shared by the two banks: the contact is one sheet with a bank on each side. Contact faces are moved to the sub-voxel contact (the two fronts' arrivals), then the mesh is relaxed and smoothed as the base (twicing at 2 voxels).
- **Crossings: none from the owners; all from the smoothing.** The surface read off the owners has 0 crossing triangles, as raw voxel faces and with the sub-voxel contact. After smoothing: 20 before the edits and 24 after, every one within 0.11 mm of a junction (an edge with three triangles, where the contact sheet meets the open surface; 2,268 such edges). Untested fix: keep the junction lines as features in the smoothing (smoothed only along themselves).
- **Edits by weights are less exact than mesh edits.** |offset| of the boundary from the points (median; worst; finding 25's score):

  | | Before | All 23 placed | Half placed, the other half scored |
  |---|---|---|---|
  | Two roots per click | 0.26; 0.66 mm | 0.14; 0.37 | 0.16; 0.62 |
  | Spread 1 mm | 0.26; 0.66 mm | 0.11; 0.35 | 0.17; 0.61 |

  Distance from the points to the extracted contact sheet: 0.25 mm before, 0.11 after all 23. Mesh edits on the field-weighted base (above) put the contact on placed points (0.00) and 0.10 mm off between them (a different score and domain, so only roughly comparable). [inference: why weights fall short: a root patch is about 1 mm across, so neighbouring clicks share roots and ask them for different thicknesses; the click sets thickness at one place, not the boundary's shape across the patch; S is read from a pass whose boundary is still off.]
- **Time** (box of 1.7 M domain voxels, 4 threads): one pass 6.8 s, extraction 4.8 s. An edit with 2 rounds is 3 passes plus an extraction, about 25 s: within "remesh and wait 30 s", without a local rerun.
- **Seen in the views** (`prototypes/figures/owner_edit_M1S1.png`): besides the kissing face, contact sheets run through cortex where there is no sulcus (finding 27's overlap-like boundaries, kept as in the slit base). They are the base's contact set, not the edits'.
- **What a shared sheet changes downstream.** The surface is no longer a manifold: each bank's pial is its open part plus its side of the shared sheet. The editor's solver (a field on a manifold mesh) and the laminar step's columns would need that pairing.
- **Reading** [inference]: the guarantee holds by construction until smoothing, and the smoothing's crossings are at junctions, a known kind of fix. The cost is precision: weight edits move a root patch's territory, not the boundary's shape, so they land within about 0.1–0.15 mm where mesh edits land on the point. Carrying the root position through the pass (open decision 4 of the main plan's Next step 0) would make the owners finer than 1 mm patches.
