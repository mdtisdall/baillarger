# Plan: label editing in the viewer

Status: plan. Nothing is built.

## Goal

The user corrects the segmentation in the viewer. The user sees each correction at once,
what the automatic correction would do after it, and an approximate effect on the white
surface. When the user accepts the corrections, one job regenerates the dseg and the
white surface exactly.

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
- Topology: the pane shows what the automatic correction would do after the pending
  strokes. Submit is still allowed; the regeneration's correction decides.
- Overrides: when the automatic correction must change a painted voxel, it does so (the
  rule of `white_dseg` now: last, and recorded). The regeneration reports each override.
- Scope: all the parts below.
- The preview is approximate; the regeneration is exact.
- Reuse first: each part uses the code below where it exists, and adds a parameter to an
  existing function before it adds a new function.

## What exists, and what each part reuses

| Need | Existing code | Used by |
|---|---|---|
| An answer log, replay on the input labels | `wm_edits` (`EditLog.answer`, `replay`, `Log.without`) | 1, 2, 3 |
| A brush on the slice tiles, strokes held in the page | `static/review.js` (scribble mode: `paint`, the dabs, the radius keys, Escape) | 1 |
| Strokes to voxels: a voxel in two strokes keeps the later one; voxels on the grid | `Review.scribble` in `viewer/review.py` | 1 |
| Only the voxels that change go into an answer | `wm_paint.changes` (the ITK-SNAP import) | 1 |
| What the automatic correction does after an answer, on a box round it, with the answer's voxels verified; whether it overrides; whether the box cuts the WM | `wm_choices.preview` (it takes a `Change`, for example a manual paint); `/review/preview` with its job for a slow preview | 1 (4a below) |
| Drawing voxel lists on the slice tiles (answer and correction colours, the cap `MAX_DRAWN`) | `static/review.js` (the preview drawing) | 3 |
| A label image as a brick volume in the pane | `Volumes.add_nifti` in `viewer/volumes.py` | 3 |
| The verified answers and the overrides in the regeneration | `white_dseg.wm_topo` (`ManualVoxels`, the fix table's `overrides`); `wm_sites` (sites of the kind `override`) | 5 |
| The antialiased field | `antialias.thin_plate`, `antialias.signed_distance` | 4b |
| Marching on a crop, read on the whole grid, in the mesh blocks | `blocks.EmbeddedSource`, `mesh.marching_tetrahedra_blocks` (as `white_surf` uses them) | 4b |
| A chunked store for a whole-grid array | `baillarger.blocks` (the dseg writes its store directly) | 4b |
| The cut of a mesh by a slice plane | `viewer.outlines.plane_cut` | 4b |
| The steps of the regeneration | the commands `white_dseg`, `white_surf`, `white_overview`, `wm_sites` (their `write_*` functions) | 5 |
| The radius of a local field solve | `docs/preview-radius.md` | 4b |

Facts that the plan must respect:

- `EditLog.answer` accepts an answer only at a site that the session showed (a `shown`
  record). An answer record already has a field `Kind` ("site" or "scribble").
- The viewer starts the review only with both `--sites` and `--edits`.
- The review's labels are the dseg's (after the correction). The input labels are the
  source of the wm-dseg (its sidecar's `Sources`: the excised dseg).
- The wm-dseg sidecar does not record which answers it replayed.
- `ArrayPyramid` has no way to change a brick after it is built, and the page caches
  bricks. So the plan does not change a brick volume while the viewer runs.
- `baillarger.viewer.jobs` runs a job in a thread of the viewer. A run of about 22 GB
  must not run in the viewer's process.
- `white_surf` computes the antialiased field, but it does not keep it.

## Part 1: paint answers and undo

- The viewer reads the input labels (the wm-dseg's source) for the review.
- Paint mode: the scribble mode of `review.js` with a label menu in place of the stroke
  classes: outside, WM, cortex, and "input label" (an eraser: it paints each voxel back
  to its input label). The brush, the dabs, the radius keys and Escape are the scribble
  mode's.
- Submit: `POST /review/paint`. The server turns the strokes into voxels with the code of
  `Review.scribble` (moved into one function that both routes call), resolves "input
  label" from the input labels, checks that each label is one of the three classes, keeps
  only the voxels whose label changes (as `wm_paint.changes`, moved into `wm_edits` so
  that both use it), and appends one `manual-paint` answer. `Params`:
  `{"Tool": "viewer-paint", "Radius": r}`. `Params.Crop` stays for answers from ITK-SNAP.
- At a queue site, the choice `manual-paint` opens the paint mode, and Submit writes the
  paint as the site's answer. The ITK-SNAP commands stay in the choice's preview, as a
  second way.
- Undo: `POST /review/undo {answer}`. The server appends a `manual-paint` answer at the
  same site, with `Supersedes: [answer]` and `Params: {"Tool": "viewer-undo"}`. It covers
  the answer's voxels where that answer is still the last answer to cover them; a later
  answer keeps its voxels. Its labels are the labels that those voxels had before the
  answer: `replay` of the input labels with `Log.before(answer)`, a sibling of
  `Log.without` (the same filter of records).
- Format: `docs/formats/wm-edits.md` (version 1, edited in place) gets the `Tool` and
  `Radius` parameters of `manual-paint`.

Tests: the route writes one answer with the painted voxels and labels; a label that is not
one of the three is refused, and nothing is written; "input label" gives the input label;
a voxel whose label does not change is not in the answer; a voxel off the grid is
refused; replay of the log gives the painted labels; an undo restores the labels from
before the answer, and leaves the voxels of a later answer; the scribble route gives the
same strokes as before the move.

## Part 2: free sites

- A session has a mode: "queue" (now) or "free". `EditLog.start_session` gets the mode;
  the session record gets the field `Mode`; a record without it reads as "queue". Format:
  `wm-edits.md`, version 1, edited in place.
- The viewer starts the review with `--edits` alone (no queue: free editing only), or with
  `--sites` and `--edits` as now.
- A free session starts at the first paint with no queue site open, one for each viewer
  run. For each free answer, the server writes the `shown` record of its site (the first
  voxel painted), then the answer.
- `Log.sessions()` gives the mode of each session.

Tests: a free session and its answers read back with their mode; an old log reads as
"queue"; the review pane's queue state does not show free answers as queue sites; the
viewer starts with `--edits` alone.

## Part 3: show the edits

- `white_dseg` records the last answer it replayed: `EditsLastAnswer` (an answer Id, or
  null) in the wm-dseg sidecar. Format: `wm-dseg.md`, version 1, edited in place.
- The input labels are a volume of the pane, "input" (`Volumes.add_nifti`), so the user
  can compare them with the dseg ("labels").
- The pending answers are the answers after `EditsLastAnswer`. `GET /review/pending`
  gives their voxels and labels, after replay (the last answer at each voxel). The pane
  draws them on the slice tiles with the preview drawing of `review.js`, in the label
  overlay's colours. After each answer or undo the pane asks again. No brick volume
  changes.

Tests: `EditsLastAnswer` is the last answer's Id after `white_dseg --edits`, and null
without a log; `/review/pending` gives the voxels of the answers after it, with the last
answer's label where two cover a voxel; after an undo, the undone voxels show their
earlier labels.

## Part 4: the effect on the WM mask and the surface

### 4a: what the automatic correction would do

- The pending strokes go to the existing preview: `/review/preview` with the choice
  `manual-paint` and the strokes as a `Change` (`wm_choices.preview` takes one). It runs
  `white_dseg.wm_topo` on a box round the strokes, with the strokes' voxels verified.
- The pane draws the answer and the correction (the existing preview drawing), and says
  when the correction would override a painted voxel and when the box cuts the WM (the
  preview is then approximate). Submit is allowed.
- No new topology code.

Tests: the preview of paint strokes that close a loop gives a correction (a cut or a fill)
round them; strokes on a flat sheet give no correction; a preview of strokes equals the
preview of the same `Change` given directly to `wm_choices.preview`.

### 4b: surface preview

- `white_surf` writes the antialiased field as a derivative: float32 on the box of the
  mask, in the chunked store (`baillarger.blocks`), with a JSON sidecar and a spec in
  `docs/formats/` (version 1). A BIDS name with `desc-antialias`.
- `antialias.thin_plate` gets two optional parameters: a start field (default: the signed
  distance, as now) and a mask of held voxels (default: none). The local solve is this
  function on a crop; there is no second solver. (The loop of
  `prototypes/scripts/antialias_preview_radius.py` shows the change.)
- The preview's mask is the dseg's WM mask with the pending answers and their
  corrections (4a) applied.
- The server keeps a working field: the stored field, updated by the local solve of each
  pending answer and each undo. A preview of pending strokes starts from the working
  field and does not change it.
- The local solve: the box of the changed voxels plus R, from the working field, with the
  voxels farther than R from the changed voxels held (`docs/preview-radius.md`). R
  follows the rule there: c measured at the edit, delta0 = the larger of 1 and half the
  largest side of the changed voxels' box (voxels), tau = 0.01 voxel; its smallest value
  is 16 voxels. Outside the field's box the field is -(BAND + 3) (as `white_surf`'s
  `EmbeddedSource`).
- `marching_tetrahedra_blocks` on an `EmbeddedSource` of the working field re-marches the
  mesh blocks that the solved box touches (`white_surf`'s steps, on those blocks only). In
  the 3D view of the true surface these blocks replace the surface's blocks. The slices
  draw their cut (`plane_cut`) in a second colour.
- The work is a box of about 10^5 voxels and a few hundred iterations, on the GPU.

Tests: `thin_plate` with no new parameters gives the same field as now; on a phantom, the
preview of an edit equals the full solve within 0.01 voxel near the edit; an edit with no
change of the WM mask gives no new blocks; the stored field reads back equal to the field
that `white_surf` used; a preview leaves the working field unchanged, and an answer
changes it only in its box.

## Part 5: regenerate

- A command `regenerate` calls the `write_*` functions of `white_dseg` (with the log),
  `white_surf`, `white_overview` and `wm_sites`, in that order, into a new derivatives
  root. It does not overwrite the old root.
- The viewer starts the command as a subprocess under `scripts/memguard` (not in a
  thread of `baillarger.viewer.jobs`), and shows its state and log lines. On the laptop it
  takes about 20 min, with a peak of about 22 GB, so the viewer's own memory counts in the
  cap.
- When it ends, the pane shows the new root, the overrides of painted voxels (from the fix
  table's `overrides`: the answer, its site and the voxels), and how to restart the viewer
  on the new root. The overridden sites are queue sites of the kind `override` in the new
  queue (`wm_sites`, as now).

Tests: the command on a phantom gives the same files as the four commands run by hand;
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
| 1 + 2 | `wm_edits.py` (mode, `Log.before`, the changed-voxels function from `wm_paint`), `wm_paint.py` (uses it), `docs/formats/wm-edits.md`, their tests | `viewer/review.py` (`/review/paint`, `/review/undo`, the shared strokes function, `--edits` alone), `static/review.js` (paint mode, undo) | the `EditLog` contract |
| 3 | `white_dseg.py`, `docs/formats/wm-dseg.md` (`EditsLastAnswer`), its tests | `viewer/volumes.py` ("input"), `viewer/review.py` (`/review/pending`), `static/review.js` (drawing) | `EditsLastAnswer` |
| 4a | none | `viewer/review.py` (`/review/preview` takes strokes), `static/review.js` | the paint request |
| 4b | `antialias.py` (start, held), `white_surf.py` (the field derivative, the blocks' re-march), its format doc, their tests | `viewer/review.py` (route, working field), the 3D view and the slices (blocks) | the preview contract |
| 5 | a new `regenerate` command, its tests | `viewer/review.py` (the subprocess), `static/review.js` (state, message) | the command's arguments |

Waves:

1. Wave 1: the cores of 1 + 2, 3, 4b and 5, at the same time. They change different
   files.
2. Wave 2: the joins, one at a time, because they share `review.py` and `review.js`.
   Order: 1 + 2, 3, 4a, 4b, 5.

Who does the work:

- One coordinator session holds this plan. It starts one sub-session for each top-level
  task (a core, then a join), at the same time where the waves allow. It keeps the plan
  and the contracts current, reviews each sub-session's PR, and asks the user to merge.
- Each sub-session owns its task: its branch, its worktree, its own steps, and its own
  sub-agents (the `parallel-agents` skill: each sub-agent owns different files). It
  reports its PR to the coordinator.
- A sub-session that needs a change to a contract asks the coordinator. The coordinator
  changes the plan first, then tells the other sub-sessions that use the contract.
- Before it writes a new function, a sub-session looks for an existing one (the table
  "What exists"), and says in its PR what it reused.

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
- `Log.before(answer_id) -> Log`: the log without that answer and the answers after it,
  in memory (for undo).
- `wm_edits.changed_voxels(labels, voxels, new) -> (voxels, labels)`: only the voxels
  whose label changes (from `wm_paint.changes`).
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
- `POST /review/preview` also takes `{site, choice: "manual-paint", strokes}` (the paint
  request's strokes); the response is the existing preview's.
- The wm-dseg sidecar: `EditsLastAnswer`, the Id of the last answer replayed, or null.
- `GET /review/pending`: `{voxels: [[i, j, k], ...], labels: [...], answers: [ids]}`.
- `antialias.thin_plate(mask, dev, band, eps, tol, max_iter, start=None, held=None)`.
- The field derivative: the antialiased field on the mask's box, float32, in the chunked
  store, with `desc-antialias` in its BIDS name and a spec in `docs/formats/` (version 1).
  `POST /review/surface-preview {strokes}` gives the re-marched blocks' vertices and faces
  (voxel coordinates of the grid), their block ids and the radius used.
- The command: `python -m baillarger.regenerate INPUT --edits LOG --out ROOT`. INPUT is
  the input labels (the excised dseg). It writes the dseg, the white surface, the
  overview and the queue of sites into ROOT, refuses a ROOT that exists, and prints the
  overrides of answers.

## Not in this plan

- Denser roots in the profile section.
- A 3D brush, and labels other than outside, WM and cortex.
- The excision map: corrections to it are paint answers with the label outside.
- A count of answers by error type.
- A brick volume that changes while the viewer runs.
