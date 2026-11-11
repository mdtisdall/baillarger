# Plan: label editing in the viewer

Status: plan. Nothing is built.

## Goal

The user corrects the segmentation in the viewer. The user sees each edit at once, what
the automatic correction would do after it, and an approximate effect on the white
surface. When the user is satisfied, one command regenerates the dseg and the white
surface exactly.

## What editing needs, and what it does not

An edit changes labels. Editing needs:

1. the edits, applied to the input labels before the automatic correction;
2. undo;
3. which edits are not yet in the dseg on the screen (pending);
4. which edited voxels the automatic correction changes (overrides);
5. a preview of the effect, and a regeneration.

For these, the current state of the edits is sufficient: each edited voxel and its
label. Editing does not keep a history of edits. These are not parts of editing, and
this plan does not serve them:

- labels to train the network (checking labels is not editing);
- the success measure, and the other records of the review queue (sessions, sites
  shown and skipped, rank, reviewer time).

The review queue keeps its answer log (`wm-edits/1`) as its own record. The edits do
not come from that log.

## Decisions

- The edits are one diff: each edited voxel and its label, a sparse file next to the
  outputs (`label-edits/1`, below). `white_dseg --edits` reads the diff. Nothing else
  applies edits.
- Labels: the paint menu has three classes: outside, WM and cortex (right hemisphere 0,
  41, 42; left hemisphere 0, 2, 3; `docs/formats/label-set.md`), and "input label" (an
  eraser: it removes the voxels from the diff, so they take the input label again).
- Brush: a 2D disc on the slice under the pointer, as the scribble brush now. A 3D
  edit is painted slice by slice.
- An edit records only the voxels whose label it changes (from the label with the
  current diff applied). Painting a voxel with its current label is not an edit.
- Undo: a stack in the viewer, for the edits of this viewer run. It is not saved. To
  undo an older edit, the user uses the eraser.
- The automatic correction may change an edited voxel (the rule of `white_dseg` now:
  edited voxels change last). The pane shows it in the preview, Submit is allowed, and
  the regeneration reports each override.
- A regeneration replaces the outputs in their root. It keeps a copy of the diff that it
  used. An older output can be made again from its input labels, its copy of the diff
  and its commit.
- The preview is approximate; the regeneration is exact.
- Reuse first: each part uses the code in "What exists" where it can, and adds a
  parameter to an existing function before it adds a new function.

## What exists, and what each part reuses

| Need | Existing code | Used by |
|---|---|---|
| Edits on the input labels, edited voxels changed last by the correction, fixes that change them | `white_dseg.wm_topo` (now with a `Log`: `replay`, `remove_handles(..., verified)`, `Fix.overrides`) | 1 |
| A brush on the slice tiles, strokes held in the page | `static/review.js` (scribble mode: `paint`, the dabs, the radius keys, Escape) | 2 |
| Strokes to voxels: a voxel in two strokes keeps the later one; voxels on the grid | `Review.scribble` in `viewer/review.py` | 2 |
| Only the voxels that change | `wm_paint.changes` (the ITK-SNAP import) | 1, 2 |
| What the automatic correction does after a change, on a box round it; whether it overrides; whether the box cuts the WM | `wm_choices.preview` (it takes a `Change`); `/review/preview` with its job for a slow preview | 4a |
| Drawing voxel lists on the slice tiles (the cap `MAX_DRAWN`) | `static/review.js` (the preview drawing) | 3, 4a |
| A label image as a brick volume in the pane | `Volumes.add_nifti` in `viewer/volumes.py` | 3 |
| The antialiased field | `antialias.thin_plate`, `antialias.signed_distance` | 4b |
| Marching on a crop, in the mesh blocks | `blocks.EmbeddedSource`, `mesh.marching_tetrahedra_blocks` (as `white_surf` uses them) | 4b |
| A volume read box by box (a memory map of an uncompressed `.nii`) | `blocks.open_volume` | 4b |
| The cut of a mesh by a slice plane | `viewer.outlines.plane_cut` | 4b |
| The steps of the regeneration | the `write_*` functions of `white_dseg`, `white_surf`, `white_overview`, `wm_sites` | 5 |
| The radius of a local field solve | `docs/preview-radius.md` | 4b |

Facts that the plan must respect:

- The review's labels are the dseg's (after the correction). The input labels are the
  dseg's source (its sidecar's `Sources`: the excised dseg).
- `ArrayPyramid` has no way to change a brick after it is built, and the page caches
  bricks. So the plan does not change a brick volume while the viewer runs.
- `baillarger.viewer.jobs` runs a job in a thread of the viewer. A run of about 22 GB
  must not run in the viewer's process.
- `white_surf` computes the antialiased field, but it does not keep it.
- There are no real answer logs yet (only test logs), so no edits need conversion.

## Part 1: the diff

- A module `baillarger.label_edits`: read, write, apply, and change a diff; compare two
  diffs.
- The format `label-edits/1` (`docs/formats/label-edits.md`, version 1):
  `sub-<sub>_ses-<ses>_hemi-<h>_desc-labeledits_edits.npy`, a NumPy `.npy` file (no
  pickle) of structured records (`index` int64, the linear index in C order of the grid;
  `label` int32), sorted by index, each index once; and a JSON sidecar
  (`SpecVersion`, `Description`, `Hemisphere`, `Shape`, `Affine`, `Count`, `SHA256` of
  the `.npy` file, `Provenance`). Each label is in `label-set/1`.
- A write replaces both files: it writes new files in the same directory, then renames
  them over the old ones, with a lock (`fcntl.flock`, as `EditLog`). A reader checks the
  sidecar's `SHA256`.
- `white_dseg --edits EDITS` reads the diff in place of the answer log. The edited
  voxels are the diff's voxels: handle removal changes them last, as now. Each connected
  piece of edited voxels (26-adjacency) whose label differs from the input is a fix of the
  kind `manual_edit`. The fix table's `overrides` gives the number of edited voxels that
  the fix changes (`n/a` if none). The output keeps a copy of the diff that it used
  (`desc-wmtopo` with the suffix `edits`, the same format), and its sidecar lists it in
  `Sources`.
- `wm_sites` makes an `override` site for each fix with `overrides`, as now. To accept
  the fix, the user writes the fix's labels into the diff (a choice of the override site,
  Part 2); then the next regeneration has no override there. `Accepts`,
  `accepted_fixes` and the match of kept fixes go.
- The answers of the review queue change the labels by a write into the diff. The answer
  log still records them, for the queue.
- The ITK-SNAP import (`wm_paint`) writes into the diff.
- Formats edited in place (version 1): `wm-dseg.md` (`--edits`, `overrides`, `Sources`,
  the copy of the diff), `wm-sites.md` (override sites), `wm-edits.md` (the log is the
  queue's record; `dseg` does not read it; `Accepts` goes).

Tests: write and read give the same diff; a write replaces the files, and a reader sees
the old or the new diff, not a mix; the eraser removes voxels; apply gives the labels;
`white_dseg --edits` on a phantom gives the same dseg as now with the same edits as a
log; a fix that changes an edited voxel gives `overrides`; the copy of the diff is in the
output; a written fix's labels in the diff remove the override site at the next run; a
queue answer writes into the diff.

## Part 2: paint and undo

- Paint mode: the scribble mode of `review.js` with the label menu (Decisions). The
  brush, the dabs, the radius keys and Escape are the scribble mode's.
- Submit: `POST /review/paint {strokes}`. The server turns the strokes into voxels with
  the code of `Review.scribble` (moved into one function that both routes call), checks
  each label, keeps only the voxels whose label changes (`wm_paint.changes`, moved into
  `label_edits`), writes them into the diff, and pushes their earlier diff entries (a
  label, or none) on the undo stack.
- Undo: `POST /review/undo` pops the stack and writes the earlier entries back.
- At a queue site, the choice `manual-paint` opens the paint mode. Its Submit writes into
  the diff, and the queue's log records the answer as now. At an override site, the
  choice "accept the fix" writes the fix's labels into the diff.
- The viewer can start without a queue (`--edits` alone): paint only.

Tests: a paint writes the painted voxels and labels into the diff; a label that is not
in the menu is refused, and nothing is written; a voxel with its current label is not
written; a voxel off the grid is refused; undo restores the diff exactly, also after two
paints that overlap; the scribble route gives the same strokes as before the move.

## Part 3: show the edits

- The input labels are a volume of the pane, "input" (`Volumes.add_nifti`), so the user
  can compare them with the dseg ("labels").
- Pending edits: the voxels where the current diff and the dseg's copy of the diff
  differ. `GET /review/pending` gives them with their labels. The pane draws them on the
  slice tiles with the preview drawing of `review.js`, in the label overlay's colours.
  After each paint or undo the pane asks again. No brick volume changes.

Tests: `/review/pending` gives the voxels that changed since the dseg's copy, with their
labels, and voxels that the eraser removed (with the input label); after an undo of all
paints there are none.

## Part 4: the effect on the WM mask and the surface

### 4a: what the automatic correction would do

- The pending edits and the strokes not yet submitted go to the existing preview:
  `/review/preview` with the choice `manual-paint` and the voxels as a `Change`
  (`wm_choices.preview` takes one). It runs `white_dseg.wm_topo` on a box round them,
  with the edited voxels changed last.
- The pane draws the edit and the correction (the existing preview drawing), and says
  when the correction would change an edited voxel and when the box cuts the WM (the
  preview is then approximate). Submit is allowed.
- No new topology code.

Tests: the preview of strokes that close a loop gives a correction round them; strokes
on a flat sheet give no correction; a preview of strokes equals the preview of the same
`Change` given directly to `wm_choices.preview`.

### 4b: surface preview

- `white_surf` writes the antialiased field as a derivative: an uncompressed float32
  NIfTI (`.nii`) on the box of the mask (its affine moved to the box), with a JSON
  sidecar and a spec in `docs/formats/` (version 1). A BIDS name with `desc-antialias`.
  The viewer reads it with `blocks.open_volume`, which reads only the voxels of a box.
  There is no chunked store yet: NIfTI-Zarr comes with `docs/scaling.md`, item 3, and
  this plan does not make it.
- `antialias.thin_plate` gets two optional parameters: a start field (default: the signed
  distance, as now) and a mask of held voxels (default: none). The local solve is this
  function on a crop; there is no second solver. (The loop of
  `prototypes/scripts/antialias_preview_radius.py` shows the change.)
- The preview's mask is the dseg's WM mask with the pending edits and their corrections
  (4a) applied.
- The server keeps a working field in memory: the stored field, updated by the local
  solve of each paint and undo. At start it is made from the stored field and the
  pending edits. A preview of strokes not yet submitted starts from the working field
  and does not change it.
- The local solve: the box of the changed voxels plus R, from the working field, with the
  voxels farther than R from the changed voxels held (`docs/preview-radius.md`). R
  follows the rule there: c measured at the edit, delta0 = the larger of 1 and half the
  largest side of the changed voxels' box (voxels), tau = 0.01 voxel; its smallest value
  is 16 voxels. Outside the field's box the field is -(BAND + 3) (as `white_surf`'s
  `EmbeddedSource`).
- `marching_tetrahedra_blocks` on an `EmbeddedSource` of the working field re-marches the
  mesh blocks that the solved box touches. In the 3D view these blocks replace the
  surface's blocks. The slices draw their cut (`plane_cut`) in a second colour.
- The work is a box of about 10^5 voxels and a few hundred iterations, on the GPU.

Tests: `thin_plate` with no new parameters gives the same field as now; on a phantom, the
preview of an edit equals the full solve within 0.01 voxel near the edit; an edit with no
change of the WM mask gives no new blocks; the stored field reads back equal to the field
that `white_surf` used; a preview leaves the working field unchanged, and a paint changes
it only in its box.

## Part 5: regenerate

- A command `regenerate` calls the `write_*` functions of `white_dseg` (with the diff),
  `white_surf`, `white_overview` and `wm_sites`, in that order. It writes into a new
  directory in the root, then replaces the old outputs with the new ones. It does not
  keep the old outputs.
- The viewer starts the command as a subprocess under `scripts/memguard` (not in a
  thread of `baillarger.viewer.jobs`), and shows its state and log lines. On the laptop it
  takes about 20 min, with a peak of about 22 GB, so the viewer's own memory counts in the
  cap. The paint mode is closed while it runs: the command reads the diff once, at its
  start.
- When it ends, the pane shows the overrides (the fix table's `overrides`: the fix, its
  voxel and the number of edited voxels) and asks the user to restart the viewer, which
  then shows the new outputs.

Tests: the command on a phantom gives the same files as the four commands run by hand;
if a step fails, the old outputs are unchanged; an edit that the correction must change
is reported.

## Order of the PRs

The joins merge in this order. The cores can merge earlier (see Parallel work).

1. Part 1.
2. Part 2.
3. Part 3.
4. Part 4a.
5. Part 4b.
6. Part 5.

## Parallel work

Each part has a core and a join. The cores do not depend on each other. The joins put
the cores into the routes and the pane, and they meet in a few shared files.

| Part | Core (files) | Join (files) | The join needs |
|---|---|---|---|
| 1 | `label_edits.py` (new), `white_dseg.py`, `wm_sites.py`, `wm_paint.py`, `docs/formats/label-edits.md`, `wm-dseg.md`, `wm-sites.md`, `wm-edits.md`, their tests | `viewer/review.py` (queue answers write into the diff) | the `label_edits` contract |
| 2 | none (Part 1's core has the shared functions) | `viewer/review.py` (`/review/paint`, `/review/undo`, the shared strokes function, "accept the fix", `--edits` alone), `static/review.js` (paint mode, undo) | the `label_edits` contract |
| 3 | none | `viewer/volumes.py` ("input"), `viewer/review.py` (`/review/pending`), `static/review.js` (drawing) | the copy of the diff in the dseg's output |
| 4a | none | `viewer/review.py` (`/review/preview` takes voxels), `static/review.js` | the paint request |
| 4b | `antialias.py` (start, held), `white_surf.py` (the field derivative, the blocks' re-march), its format doc, their tests | `viewer/review.py` (route, working field), the 3D view and the slices (blocks) | the preview contract |
| 5 | a new `regenerate` command, its tests | `viewer/review.py` (the subprocess), `static/review.js` (state, message) | the command's arguments |

Waves:

1. Wave 1: the cores of 1, 4b and 5, at the same time. They change different files.
   The core of 5 uses `white_dseg --edits EDITS` from the contract; its tests that run
   `white_dseg` with edits wait until the core of 1 merges.
2. Wave 2: the joins, one at a time, because they share `review.py` and `review.js`.
   Order: 1, 2, 3, 4a, 4b, 5.

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
  hemisphere checks of 1 and 4b, and the full regeneration of 5 (about 22 GB). Agents can
  write and unit-test these parts at the same time, but their hemisphere runs must not
  overlap.

### Contracts

The joins use these interfaces. A core must keep them, or change this plan first.

- `label_edits.Edits`: `index` (int64, sorted, unique) and `label` (int32) arrays.
- `label_edits.read(path) -> Edits`; `label_edits.write(path, edits, *, hemisphere,
  shape, affine)` (replaces the files, under the lock); `label_edits.empty() -> Edits`.
- `label_edits.apply(labels, edits) -> labels` (a copy with the edits written).
- `label_edits.paint(edits, labels, voxels, new) -> (Edits, undo)`: `labels` are the input
  labels; `new` a label per voxel, or -1 for the eraser. It keeps only the voxels whose
  label changes. `undo` holds the earlier entries of those voxels.
- `label_edits.restore(edits, undo) -> Edits`.
- `label_edits.differ(a, b) -> (index, label)`: the voxels where two diffs differ, with
  the label in `a` (-1 where `a` has none: the input label).
- `python -m baillarger.white_dseg ASEG --edits EDITS --out DERIV`; the fix table's
  `overrides` is the number of edited voxels that a fix changes; the output keeps the
  diff that it used.
- `POST /review/paint {strokes: [{label: 0 | 41 | 42 (left: 0 | 2 | 3) | "input",
  voxels: [[i, j, k], ...]}]}`. Response: `{voxels, undo_depth}`, or `{error}` with
  nothing written.
- `POST /review/undo`. Response: `{voxels, undo_depth}`, or `{error}` if the stack is
  empty.
- `POST /review/preview` also takes `{site: "i,j,k", choice: "manual-paint", strokes}`;
  the response is the existing preview's.
- `GET /review/pending`: `{voxels: [[i, j, k], ...], labels: [...]}`.
- `antialias.thin_plate(mask, dev, band, eps, tol, max_iter, start=None, held=None)`.
- The field derivative: the antialiased field on the mask's box, an uncompressed float32
  `.nii`, with `desc-antialias` in its BIDS name and a spec in `docs/formats/` (version 1).
  `POST /review/surface-preview {strokes}` gives the re-marched blocks' vertices and faces
  (voxel coordinates of the grid), their block ids and the radius used.
- The command: `python -m baillarger.regenerate INPUT --edits EDITS --root ROOT`. INPUT
  is the input labels (the excised dseg). It writes the dseg, the white surface, the
  overview and the queue of sites, replaces the outputs in ROOT, and prints the
  overrides.

## Not in this plan

- Labels to train the network, and which labels train it.
- The success measure, and the queue's answer log apart from what Part 1 changes.
- A history of edits.
- Denser roots in the profile section.
- A 3D brush, and labels other than outside, WM and cortex.
- The excision map: corrections to it are paint edits with the label outside.
- A brick volume that changes while the viewer runs.
