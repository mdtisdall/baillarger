# Plan: label editing in the viewer

Status: plan. Nothing is built.

## Goal

The user corrects the segmentation in the viewer. The user sees each correction at once
and an approximate effect of the correction on the white surface. When the user accepts
the corrections, one job regenerates the dseg and the white surface exactly.

## Decisions

- Labels: the paint menu has three classes only: outside, WM and cortex (right
  hemisphere 0, 41, 42; left hemisphere 0, 2, 3; `docs/formats/label-set.md`). The other
  labels stay as the segmentation gives them.
- Brush: a 2D disc on the slice under the pointer, as the scribble brush now. A 3D
  correction is painted slice by slice.
- Edits outside the queue: each one is an answer at a free site, in a session of the
  mode "free". A count of answers can then include or exclude them.
- Undo: the user can undo a submitted answer. The undo is a new answer (the log stays
  append-only).
- Topology: the pane warns when pending strokes change the topology of the WM mask.
  Submit is still allowed; the automatic correction of the regeneration handles it.
- Overrides: when the automatic correction must change a painted voxel, it does so (the
  rule of `white_dseg` now: last, and recorded). The regeneration reports each override.
- Scope: all the parts below.
- The preview is approximate; the regeneration is exact.

## What exists

- The answer log, `wm-edits/1` (`baillarger.wm_edits`). An answer gives voxels and their
  new labels. `white_dseg --edits LOG` replays the answers, in log order, on the input
  labels, before the automatic topology correction. The input labels are the source of
  the wm-dseg (its sidecar's `Sources`: the excised dseg).
- `EditLog.answer` accepts an answer only at a site that the session showed (a `shown`
  record). An answer record already has a field `Kind` ("site" or "scribble").
- The choice `manual-paint` exists; `wm_paint import` writes it after a paint in
  ITK-SNAP. In the pane, the choice gives the ITK-SNAP commands.
- The answers that `white_dseg` replays are verified: handle removal changes them last,
  and the fix table records the answer that a fix overrides (`overrides`,
  `docs/formats/wm-dseg.md`). `wm_sites` makes a queue site of the kind `override` for
  each such fix.
- The wm-dseg sidecar lists the log's files in `Sources`, but not which answers it
  replayed.
- The review pane (`baillarger.viewer.review`, `static/review.js`), a plug-in of the
  brick pane: the queue of sites, five fixed choices with previews, and scribbles. A
  scribble is a training label only; dseg does not apply it. The viewer starts the review
  only with both `--sites` and `--edits`. The pane's labels are the dseg's (after the
  correction), not the input labels.
- The label overlay of the brick pane shows the labels of the dseg files. It does not show
  an answer until the dseg is regenerated.
- `white_surf` computes the antialiased field on the mask's box, but it does not keep it.
- The radius of a local field solve round an edit: `docs/preview-radius.md`.

## Part 1: paint answers and undo

- The viewer reads the input labels (the wm-dseg's source) for the review.
- The pane gets a paint mode next to the scribble mode. Menu: outside, WM, cortex, and
  "input label" (an eraser: it paints each voxel back to its input label). Keys as the
  scribble keys. The brush radius keys stay.
- The strokes stay in the page until the user submits them. Escape clears them.
- Submit: `POST /review/paint`. The server resolves "input label" from the input labels,
  checks that each label is one of the three classes and that each voxel is on the grid,
  and appends one `manual-paint` answer. A voxel in two strokes keeps the later stroke's
  label. `Params`: `{"Tool": "viewer-paint", "Radius": r}`. `Params.Crop` stays for answers
  from ITK-SNAP.
- At a queue site, the choice `manual-paint` opens the paint mode, and Submit writes the
  paint as the site's answer. The ITK-SNAP commands stay in the choice's preview, as a
  second way.
- Undo: `POST /review/undo {answer}`. The server appends a `manual-paint` answer at the
  same site, with `Supersedes: [answer]` and `Params: {"Tool": "viewer-undo"}`. It covers
  the answer's voxels where that answer is still the last answer to cover them; a later
  answer keeps its voxels. Its labels are the labels that those voxels had before the
  answer: the input labels with the answers before it replayed (`Log.before`).
- Until Part 3, the pane shows the answers submitted from the page, so that an answer does
  not go from view when the strokes clear.
- Format: `docs/formats/wm-edits.md` (version 1, edited in place) gets the `Tool` and
  `Radius` parameters of `manual-paint`.

Tests: the route writes one answer with the painted voxels and labels; a label that is not
one of the three is refused, and nothing is written; "input label" gives the input label;
a voxel off the grid is refused; replay of the log gives the painted labels; an undo
restores the labels from before the answer, also when a later answer covers some of its
voxels (the later answer is not undone there: undo writes only the voxels where the
undone answer is still the last).

## Part 2: free sites

- A session has a mode: "queue" (now) or "free". `EditLog.start_session` gets the mode;
  the session record gets the field `Mode`; a record without it reads as "queue". Format:
  `wm-edits.md`, version 1, edited in place.
- The viewer starts the review with `--edits` alone (no queue: free editing only), or with
  `--sites` and `--edits` as now.
- A free session starts at the first paint with no queue site open, one for each viewer
  run. For each free answer, the server writes the `shown` record of its site (the first
  voxel painted), then the answer.
- `Log.sessions()` gives the mode of each session. No count of answers exists yet; a
  later count reads the mode.

Tests: a free session and its answers read back with their mode; an old log reads as
"queue"; the review pane's queue state does not show free answers as queue sites; the
viewer starts with `--edits` alone.

## Part 3: show the edits

- `white_dseg` records the last answer it replayed: `EditsLastAnswer` (an answer Id, or
  null) in the wm-dseg sidecar. Format: `wm-dseg.md`, version 1, edited in place.
- A label layer "input-edits": the input labels with the log's answers replayed. The
  server builds it per brick: a brick without an answer voxel is the input labels' brick.
  After each answer, the server rebuilds the bricks that hold the answer's voxels, and
  their coarse levels (the label pyramid's `mode` reducer).
- A fill layer "edited voxels": the voxels of the answers after `EditsLastAnswer`, in
  the label overlay's colours.
- The current overlay (the dseg after correction) stays, so the user can compare.

Tests: a brick with an answer voxel shows the new label; a brick without one is the input
brick, unchanged; after an answer, only the bricks that hold its voxels and their coarse
levels change; `EditsLastAnswer` is the last answer's Id after `white_dseg --edits`.

## Part 4: the effect on the WM mask and the surface

The preview's mask is the dseg's WM mask with the filled-mask changes of the answers
after `EditsLastAnswer` and of the pending strokes. It does not have the automatic
correction.

### 4a: topology check

For the pending strokes, and for each answer:

- The filled mask changes where a voxel moves between a filled and a not-filled label.
- Simple points: the changed voxels are applied one at a time, in a fixed order. If each
  one is a simple point when it is applied (`baillarger.topology`, the connectivity pair
  of `white_dseg`), the topology does not change. This answer is exact.
- Otherwise: the change of the Euler characteristic (exact, from the changed voxels and
  their 2 x 2 x 2 neighbourhoods), and the merges of pieces of the mask or of the
  background, from a labelling of their pieces that the server makes one time when it
  starts. A split cannot be proven from a box; the check reports "may split".
- The pane shows the result on the strokes: "no change", or what changes. Submit is
  allowed.

- A filled cavity is the case where the changed voxels hold every voxel of a background
  piece: the labelling shows it.

Tests: on phantoms, a bump on a flat sheet gives "no change" (all simple); a bridge that
closes a loop gives a change of the Euler characteristic of -1 and no merge (a handle); a
bridge between two pieces gives -1 and one merge; a filled cavity gives -1 and a
background piece that goes; a cut through a piece gives "may split".

### 4b: surface preview

- `white_surf` writes the antialiased field as a derivative: float32 on the box of the
  mask, in the chunked store (`baillarger.blocks`), with a JSON sidecar and a spec in
  `docs/formats/` (version 1). A BIDS name with `desc-antialias`.
- The server keeps a working field: the stored field, updated by the local solve of each
  answer (and each undo) after `EditsLastAnswer`. A preview of pending strokes starts from
  the working field and does not change it.
- The local solve: the box of the changed voxels plus R, from the working field, with the
  voxels farther than R from the changed voxels held (the method of
  `docs/preview-radius.md`). R follows the rule there: c measured at the edit,
  delta0 = the larger of 1 and half the largest side of the changed voxels' box (voxels),
  tau = 0.01 voxel; its smallest value is 16 voxels. Outside the field's box the field is
  -(BAND + 3) (the value outside the mask).
- Marching tetrahedra re-marches the mesh blocks (`BLOCK` voxels a side) that the solved
  box touches, from the working field. In the 3D view of the true surface, these blocks
  replace the surface's blocks. The slices draw their outline in a second colour.
- The work is a box of about 10^5 voxels and a few hundred iterations, on the GPU.

Tests: on a phantom, the preview of an edit equals the full solve within 0.01 voxel near
the edit; an edit with no filled-mask change gives no new blocks; the stored field reads
back equal to the field that `white_surf` used; a preview leaves the working field
unchanged, and an answer changes it only in its box.

## Part 5: regenerate

- One job (`baillarger.viewer.jobs`, one at a time) or one command runs `white_dseg
  --edits LOG` on the input labels, then `white_surf`, then the overview, then
  `wm_sites`, into a new derivatives root. It does not overwrite the old root.
- The job runs under `scripts/memguard`. On the laptop it takes about 20 min, with a
  peak of about 22 GB, so the viewer's own memory counts in the cap.
- When the job ends, the pane shows the new root, the overrides of painted voxels (from
  the fix table's `overrides`: the answer, its site and the voxels), and how to restart
  the viewer on the new root. The overridden sites are queue sites of the kind
  `override` in the new queue.

Tests: the command on a phantom gives the same dseg as `white_dseg --edits` run by hand;
the old root is unchanged; an answer that the correction must override is reported.

## Order of the PRs

The joins merge in this order. The cores can merge earlier (see Parallel work).

1. Parts 1 and 2 (they change the same format and the same pane).
2. Part 3.
3. Part 4a.
4. Part 4b.
5. Part 5.

## Parallel work

Each part has a core and a join. The cores do not depend on each other. The joins put
the cores into the routes and the pane, and they meet in a few shared files.

| Part | Core (files) | Join (files) | The join needs |
|---|---|---|---|
| 1 + 2 | `wm_edits.py` (mode, `Log.before`), `docs/formats/wm-edits.md`, their tests | `viewer/review.py` (`/review/paint`, `/review/undo`, `--edits` alone), `static/review.js` (paint mode, undo) | the `EditLog` contract |
| 3 | `white_dseg.py` and `docs/formats/wm-dseg.md` (`EditsLastAnswer`); `viewer/volumes.py` (the label layer); their tests | `static/overlays.js` (the layer menu) | the layer's name |
| 4a | a new module beside `topology.py`, its tests | `viewer/review.py` (route), `static/review.js` (warning) | the topology contract |
| 4b | `white_surf.py` (the field derivative), its format doc; a new module for the local solve; their tests | `viewer/review.py` (route, working field), the 3D view and the slices (blocks) | the preview contract |
| 5 | a new `regenerate` command, its tests | `baillarger.viewer.jobs` (the job), `static/review.js` (message) | the command's arguments |

Waves:

1. Wave 1: the five cores, at the same time. They change different files.
2. Wave 2: the joins, one at a time, because they share `review.py`, `review.js` and
   `index.html`. Order: 1 + 2, 3, 4a, 4b, 5.

Who does the work:

- One coordinator session holds this plan. It starts one sub-session for each top-level
  task (a core, then a join), at the same time where the waves allow. It keeps the plan
  and the contracts current, reviews each sub-session's PR, and asks the user to merge.
- Each sub-session owns its task: its branch, its worktree, its own steps, and its own
  sub-agents (the `parallel-agents` skill: each sub-agent owns different files). It
  reports its PR to the coordinator.
- A sub-session that needs a change to a contract asks the coordinator. The coordinator
  changes the plan first, then tells the other sub-sessions that use the contract.

Rules:

- One part, one branch, one PR (CLAUDE.md). Parts that run at the same time each have
  their own worktree.
- A core's PR can merge before its join. Then the join's PR is small.
- Heavy runs (more than a few GB) stay one at a time, under `scripts/memguard`: the
  hemisphere checks of 4b and the full regeneration of 5 (about 22 GB). Agents can write
  and unit-test these parts at the same time, but their hemisphere runs must not overlap.

### Contracts

The joins use these interfaces. A core must keep them, or change this plan first.

- `EditLog.start_session(random_share, random_seed, dseg_sha256, queue_sha256,
  time=None, mode="queue")`. The session record gets the field `Mode`, "queue" or
  "free". `Log.sessions()` gives `Mode` for each session; a record without the field
  reads as "queue".
- `Log.before(answer_id) -> Log`: the log of the answers before that answer, in memory
  (for undo).
- A paint answer: `EditLog.answer(session, site, "manual-paint", voxels, labels,
  decided_against=<dseg SHA-256>, reviewer_seconds=s, params={"Tool": "viewer-paint",
  "Radius": r})`; an undo: the same with `params={"Tool": "viewer-undo"}` and
  `supersedes=[answer]`.
- `POST /review/paint`. Request: `{site: "i,j,k" or null, strokes: [{label: 0 | 41 | 42
  (left: 0 | 2 | 3) | "input", voxels: [[i, j, k], ...]}], reviewer_seconds}`. With
  `site` null, the server uses the free session of this viewer run (it starts it if
  needed), writes the `shown` record of the first voxel of the first stroke, and uses that
  voxel as the site. Response: `{answer, session, mode, voxels}`, or `{error}` with
  nothing written.
- `POST /review/undo {answer}`. Response: `{answer, supersedes, voxels}`, or `{error}`.
- The wm-dseg sidecar: `EditsLastAnswer`, the Id of the last answer replayed, or null.
- The label layer: the volume `input-edits` in the volume list (`GET /volume/list`),
  served as bricks like the other label volumes. It changes when the log has a new
  answer.
- The topology check: `local_topology(labels_box, changed, filled, pieces) ->
  {"simple": bool, "euler": dE, "merges": n, "may_split": bool}`, on a box round the
  changed voxels; `pieces` is the server's labelling of the pieces of the mask and the
  background. `POST /review/topology {voxels, labels}` gives the same object.
- The field derivative: the antialiased field on the mask's box, float32, in the chunked
  store, with `desc-antialias` in its BIDS name and a spec in `docs/formats/` (version 1).
  The local solve: `preview_patch(field, mask, changed, radius) -> (box, values)`, the box
  (slices of the grid) and the solved field in it. The server re-marches the blocks that
  the box touches. `POST /review/surface-preview {voxels, labels}` gives those blocks'
  vertices and faces (voxel coordinates of the grid) and the radius used.
- The command: `python -m baillarger.regenerate INPUT --edits LOG --out ROOT`. INPUT is
  the input labels (the excised dseg). It writes the dseg, the white surface, the
  overview and the queue of sites into ROOT, refuses a ROOT that exists, and prints the
  overrides of answers.

## Not in this plan

- Denser roots in the profile section.
- A 3D brush, and labels other than outside, WM and cortex.
- The excision map: corrections to it are paint answers with the label outside.
- A count of answers by error type.
