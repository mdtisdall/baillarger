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
- Edits outside the queue: each one is an answer at a free site, in a session of the kind
  "free". The error-type counts can then include or exclude them.
- Scope: all the steps below, built and merged in order, one PR for each part.
- The preview is approximate; the regeneration is exact (memory: edit preview, then
  regenerate).

## What exists

- The answer log, `wm-edits/1` (`baillarger.wm_edits`). An answer gives voxels and their
  new labels. `white_dseg --edits LOG` replays the answers, in log order, on the input
  labels, before the automatic topology correction. The choice `manual-paint` exists;
  `wm_paint import` writes it after a paint in ITK-SNAP.
- The review pane (`baillarger.viewer.review`, `static/review.js`), a plug-in of the
  brick pane: the queue of sites, five fixed choices with previews, and scribbles. A
  scribble is a training label only; dseg does not apply it.
- The label overlay of the brick pane shows the labels of the dseg files. It does not show
  an answer until the dseg is regenerated.
- `white_surf` computes the antialiased field on the mask's box, but it does not keep it.
- The radius of a local field solve round an edit: `docs/preview-radius.md`.

## Part 1: paint answers

The pane gets a paint mode next to the scribble mode.

- Menu: outside, WM, cortex, and "input label" (an eraser: it paints each voxel back to
  its label in the input labels). Keys as the scribble keys. The brush radius keys stay.
- The strokes stay in the page until the user submits them. Escape clears them.
- Submit: `POST /review/paint {site, strokes: [{label, voxels}], reviewer_seconds}`.
  The server resolves "input label" from the input labels, checks that each label is one
  of the three classes and that each voxel is on the grid, and appends one `manual-paint`
  answer (`EditLog.answer`). A voxel in two strokes keeps the later stroke's label.
  `Params`: `{"Tool": "viewer-paint", "Radius": r}`. `Params.Crop` stays for answers from
  ITK-SNAP.
- `manual-paint` stays a choice of the queue: at a queue site, Submit writes the paint as
  the site's answer.
- Format: `docs/formats/wm-edits.md` (version 1, edited in place) gets the `Tool` and
  `Radius` parameters of `manual-paint`.

Tests: the route writes one answer with the painted voxels and labels; a label that is not
one of the three is refused, and nothing is written; "input label" gives the input label;
replay of the log gives the painted labels; a voxel off the grid is refused.

## Part 2: free sites

- A session has a kind: "queue" (now) or "free". `EditLog.start_session` gets the kind;
  a log without the field reads as "queue". Format: `wm-edits.md`, version 1, edited in
  place.
- When the user paints with no queue site open, the pane starts a free session if none is
  open. The answer's site is the first voxel painted.
- `wm_sites` and the error-type counts read the kind, so that a count can include or
  exclude free answers.

Tests: a free session and its answers read back with their kind; an old log reads as
"queue"; the review pane's queue state does not show free answers as queue sites.

## Part 3: show the edits

- A label layer "input + edits": the input labels with the log's answers replayed. The
  server builds it per brick: a brick without an answer voxel is the input labels' brick.
  After each answer, the server rebuilds only the bricks that hold the answer's voxels.
- A fill layer "edited voxels": the voxels of the answers since the dseg was generated, in
  one colour per class (the label overlay's colours).
- The current overlay (the dseg after correction) stays, so the user can compare.

Tests: a brick with an answer voxel shows the new label; a brick without one is the input
brick, unchanged; after an answer, only the bricks that hold its voxels change.

## Part 4: the effect on the WM mask and the surface

### 4a: topology check

For the strokes before submit, and for each answer:

- The filled mask changes where a voxel moves between a filled and a not-filled label.
- The change of the Euler characteristic of the filled mask is computed from the changed
  voxels and their 2 x 2 x 2 neighbourhoods (exact and local; `baillarger.topology`, the
  same connectivity pair as the checks of `white_dseg`).
- In a box round the edit, the change in the number of pieces of the mask and of the
  background is counted.
- The pane shows the result: "no change", or "adds or removes a handle, a cavity or a
  piece". It does not fix the change. The automatic correction of the regeneration decides.

Tests: on phantoms, a bridge that closes a loop gives one handle; a filled cavity gives
one cavity less; a cut that splits a piece gives one piece more; a bump on a flat sheet
gives no change.

### 4b: surface preview

- `white_surf` writes the antialiased field as a derivative: float32 on the box of the
  mask, in the chunked store (`baillarger.blocks`), with a JSON sidecar and a spec in
  `docs/formats/` (version 1). A BIDS name with `desc-antialias`.
- The preview mask is the current WM mask with the edit's filled-mask changes. It does not
  have the automatic topology correction.
- The field is solved in the box of the changed voxels plus R, from the stored field,
  with the voxels farther than R from the changed voxels held (the method of
  `docs/preview-radius.md`). R follows the rule there with tau = 0.01 voxel; its smallest
  value is 16 voxels.
- Marching tetrahedra on that box gives a patch. The 3D view draws the patch on the true
  surface. The slices draw the patch's outline in a second colour.
- The work is a box of about 10^5 voxels and a few hundred iterations, on the GPU.

Tests: on a phantom, the patch of an edit equals the full solve within 0.01 voxel near the
edit; an edit with no filled-mask change gives no patch; the stored field reads back equal
to the field that `white_surf` used.

## Part 5: regenerate

- One job (`baillarger.viewer.jobs`, one at a time) or one command runs `white_dseg
  --edits LOG` on the input labels, then `white_surf`, then the overview, into a new
  derivatives root. It does not overwrite the old root.
- The job runs under `scripts/memguard`. On the laptop it takes about 20 min, with a
  peak of about 22 GB, so the viewer's own memory counts in the cap.
- When the job ends, the pane shows the new root and how to restart the viewer on it. The
  queue of sites is rebuilt from the new surface and dseg (`wm_sites`).

Tests: the command on a phantom gives the same dseg as `white_dseg --edits` run by hand;
the old root is unchanged.

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
| 1 + 2 | `wm_edits.py`, `docs/formats/wm-edits.md`, their tests | `viewer/review.py` (`/review/paint`), `static/review.js` (paint mode) | the `EditLog` contract |
| 3 | `viewer/volumes.py` (the label layer), its tests | `static/overlays.js` (the layer menu) | the layer's name |
| 4a | a new module beside `topology.py`, its tests | `viewer/review.py` (route), `static/review.js` (warning) | the topology contract |
| 4b | `white_surf.py` (the field derivative), its format doc; a new module for the local solve; their tests | `viewer/review.py` (route), the 3D view and the slices (patch) | the preview contract |
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
  time=None, kind="queue")`. The session record gets the field `Kind`, "queue" or
  "free". `Log.sessions()` gives `Kind` for each session; a record without the field
  reads as "queue".
- A paint answer: `EditLog.answer(session, site, "manual-paint", voxels, labels,
  decided_against=<dseg SHA-256>, reviewer_seconds=s, params={"Tool": "viewer-paint",
  "Radius": r})`.
- `POST /review/paint`. Request: `{site: "i,j,k" or null, strokes: [{label: 0 | 41 | 42
  (left: 0 | 2 | 3) | "input", voxels: [[i, j, k], ...]}], reviewer_seconds}`. With
  `site` null, the server uses a free session (it starts one if none is open) and the
  first voxel of the first stroke as the site. Response: `{answer, session, kind,
  voxels}`, or `{error}` with nothing written.
- The label layer: the volume `input-edits` in the volume list (`/volumes`), served as
  bricks like the other label volumes. It changes when the log has a new answer.
- The topology check: `local_topology(labels_box, changed, filled) -> {"euler": dE,
  "pieces": dP, "background_pieces": dB}`, on a box round the changed voxels.
  `POST /review/topology {voxels, labels}` gives the same object.
- The field derivative: the antialiased field on the mask's box, float32, in the chunked
  store, with `desc-antialias` in its BIDS name and a spec in `docs/formats/` (version 1).
  The local solve: `preview_patch(field, mask, changed, radius) -> (vertices, faces)`, in
  voxel coordinates of the grid. `POST /review/surface-preview {voxels, labels}` gives the
  patch and the radius used.
- The command: `python -m baillarger.regenerate INPUT --edits LOG --out ROOT`. It writes
  the dseg, the white surface, the overview and the queue of sites into ROOT, and refuses
  a ROOT that exists.

## Not in this plan

- Denser roots in the profile section.
- A 3D brush, and labels other than outside, WM and cortex.
- The excision map: corrections to it are paint answers with the label outside.
