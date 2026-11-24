# Plan: label editing in the viewer

Status: Parts 1 to 6 are built and merged: Part 1 (#168, #175), Part 2 (#181), Part 3
(#195), Part 4a (#197), Part 4b (core #171, join #198), Part 5 (core #167, #172,
#177, join #196) and Part 6 (core #219, join #220, #223, the grow log #224). Part 7
(the standard edits file) is planned, not built.

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
label. Editing does not keep a history of edits. The edits file is a file of a BIDS
hierarchy: a user who wants its history puts it under version control, with any tool,
as for any other file there. That is the user's choice. It is not a function of the
tool, and not a step of editing (Part 7). These are not parts of editing, and this
plan does not serve them:

- labels to train the network (checking labels is not editing);
- the success measure, and the other records of the review queue (sessions, sites
  shown and skipped, rank, reviewer time).

The review queue keeps its answer log (`wm-edits/1`) as its own record. The edits do
not come from that log.

## Decisions

- The edits of a hemisphere are one diff: the difference between the input labels and
  the labels after the user's edits. It has each edited voxel and its label, in one
  sparse file (`label-edits/1`, below) at a standard BIDS path (Part 7). Part 7
  makes its records a TSV.
  `white_dseg --edits` reads the diff. Nothing else applies edits.
- The tool keeps the edits file whole: the hash, the write that replaces the file, the
  lock, and the undo of one viewer run. It keeps no history of the file. Version
  control of the file is the user's choice (User, 2026-11-23).
- Labels: the paint menu has three classes: outside, WM and cortex (right hemisphere 0,
  41, 42; left hemisphere 0, 2, 3; `docs/formats/label-set.md`), and "input label" (an
  eraser: it removes the voxels from the diff, so they take the input label again).
- Brush: a 2D disc on the slice under the pointer, as the scribble brush now. A 3D
  edit is painted slice by slice.
- The grow brush (Part 6): the strokes are seeds. The server grows them in 3D from the
  echoes, smooths the WM boundary near the change, and shows the result as a proposal.
  The user accepts it (it is then a plain paint) or discards it. Plain paint stays
  exact, voxel for voxel. (User, 2026-11-17, from the replay of site 1.)
- An edit records only the voxels whose label it changes. A voxel's current label is
  its entry in the diff if it has one, else its label in the dseg (the labels that the
  user sees, after the correction), not its input label. Painting a voxel with its
  current label is not an edit. Painting a voxel back to its input label, against the
  correction, is an edit: the voxel gets an entry, so the next regeneration does not
  make the same fix there. The eraser removes the entries of the voxels that have one
  and changes no other voxel.
- Undo: a stack in the viewer, for every write into the diff of this viewer run (paints
  and site answers). It is not saved. To undo an older edit, the user uses the eraser.
- The automatic correction may change an edited voxel (the rule of `white_dseg` now:
  edited voxels change last). The pane shows it in the preview, Submit is allowed, and
  the regeneration reports each override.
- A regeneration replaces the outputs in their root. It keeps a copy of the diff that it
  used. An older output can be made again from its input labels, its copy of the diff
  and its commit.
- The preview is approximate; the regeneration is exact.
- Editing does not need ITK-SNAP. It was a bootstrap only, and Part 2 removed it. The
  writer of the diff is the viewer's review pane: its answers, paints and undos are each
  one `label_edits.update`, which holds the diff's file lock (across processes) for the
  read-change-write.
- Reuse first: each part uses the code in "What exists" where it can, and adds a
  parameter to an existing function before it adds a new function.

## What exists, and what each part reuses

| Need | Existing code | Used by |
|---|---|---|
| Edits on the input labels, edited voxels changed last by the correction, fixes that change them | `white_dseg.wm_topo` (now with a `Log`: `replay`, `remove_handles(..., verified)`, `Fix.overrides`) | 1 |
| A brush on the slice tiles, strokes held in the page | `static/review.js` (scribble mode: `paint`, the dabs, the radius keys, Escape) | 2 |
| Strokes to voxels: a voxel in two strokes keeps the later one; voxels on the grid | `Review.scribble` in `viewer/review.py` | 2 |
| Only the voxels that change | `label_edits.paint` | 1, 2 |
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
  them over the old ones, with a lock (`fileops.flock`, as `EditLog`). A reader checks the
  sidecar's `SHA256`. A sidecar that does not agree with the records is older than them
  (a write that stopped between its renames): the reader checks the records and writes a
  new sidecar (`docs/formats/label-edits.md`, Recovery).
- `white_dseg --edits EDITS` reads the diff in place of the answer log. The edited
  voxels are the diff's voxels: handle removal changes them last, as now. Each connected
  piece of edited voxels (26-adjacency) whose label differs from the input is a fix of the
  kind `manual_edit`. The fix table's `overrides` gives the number of edited voxels that
  the fix changes (`n/a` if none). The output keeps a copy of the diff that it used
  (`desc-wmtopo` with the suffix `edits`, the same format), and its sidecar lists it in
  `Sources`.
- `wm_sites` makes an `override` site for each fix with `overrides`, as now. To accept
  the fix, the user writes the fix's labels into the diff: the choice `accept-fix` (key 7)
  of the review pane, at override sites only. It takes the fix's voxels and labels from
  the dseg's fix diff (`fix_diff.fix_voxels`). Then the next regeneration has no override
  there. `leave` writes nothing. `Accepts`, `accepted_fixes` and the match of kept fixes
  go.
- The answers of the review queue change the labels by a write into the diff
  (`label_edits.paint` in one `label_edits.update`): only the voxels whose label changes
  from the current label (Decisions: the diff's entry, else the dseg's label). The answer
  log still records them, for the queue: the diff is written first, then the log, in the
  same `label_edits.update` (its `after` step), so the diff's lock is held across both
  writes and no other writer writes between them. If the log refuses the answer, the
  diff is written back under that lock, no undo entry is pushed, and the error is the
  log's. A scribble is a record of the log only. The viewer
  takes the log as `--log` and the diff as `--edits`; with `--sites`, both are needed.
- The ITK-SNAP import (`wm_paint`) was a bootstrap only. Part 2 removed it.
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

- Paint mode: the brush of `review.js` has two modes, paint and scribble (key p). Paint
  has the label menu (Decisions; `review.menu`: outside, WM, cortex and the eraser
  "input"); scribble keeps the classes of the label set and is a record of the log
  only. The brush, the dabs, the radius keys and Escape are shared. Keys: w WM,
  c cortex, o outside, e eraser, x submit, u undo.
- Submit: `POST /review/paint {strokes}`. The server turns the strokes into voxels with
  `review.stroke_voxels` (the function that `/review/scribble` also calls: a voxel in two
  strokes keeps the later one), checks each label against the menu and each voxel
  against the grid (one bad label or voxel refuses the whole paint), keeps only the
  voxels whose label changes (`label_edits.paint`), writes them into the diff, and
  pushes their earlier diff entries (a label, or none) on the undo stack. A paint that
  changes no voxel writes nothing and pushes nothing. `wm_paint.changes` was not moved:
  `label_edits.paint` already keeps only the voxels that change.
- Undo: every write into the diff of this viewer run (a paint, and a site answer of any
  choice: `accept-fix` and `manual-paint` too) pushes its earlier entries on one stack.
  `POST /review/undo` writes the top entries back (`label_edits.restore`) and pops them,
  so an undo always reverses the last write and never brings back labels under a later
  answer. An undo is refused if the diff on disk is not the one that the top write left
  (another writer changed it). The stack is in memory, for this viewer run. An undo does not change the log:
  an undone answer stays in it.
- At a queue site, the choice `manual-paint` opens the paint mode. Its Submit is
  `POST /review/answer` with the strokes: it writes into the diff as a paint, and the
  queue's log records the answer: the voxels of the strokes
  and the label each takes (the eraser's voxels: their input label). Its preview has no
  voxels until Part 4a.
- The viewer can start without a queue (`--edits` alone, with no `--sites` and no
  `--log`): paint only. The hemisphere then comes from the dseg's sidecar.
- This part removed the ITK-SNAP path: `wm_paint`, its format
  (`docs/formats/wm-paint.md`), its tests, and the ITK-SNAP commands of the review pane.
  The prototype `bent_box --paint` now takes label-edits diffs in place of painted
  crops.

Tests: a paint writes the painted voxels and labels into the diff; a label that is not
in the menu is refused, and nothing is written; a voxel with its current label is not
written; a voxel off the grid is refused; undo restores the diff exactly, also after two
paints that overlap; a paint, then a queue answer on an overlapping voxel, then undo
gives the diff after the paint; the scribble route gives the same strokes as before the
move.

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
  then shows the new outputs. Until the restart, the review refuses each paint, answer,
  accept-fix and undo, and writes nothing: it compares the identity (`os.stat`: device,
  inode, mtime) of the dseg, the queue and the fix diff with their identity at its start.

Tests: the command on a phantom gives the same files as the four commands run by hand;
if a step fails, the old outputs are unchanged; an edit that the correction must change
is reported.

## Part 6: the grow brush

**Why.** A 2D brush, slice by slice, leaves a spiky WM boundary: islands and notches of
one voxel between the painted slices. At site 1 (near vertex 12403800), the WM within 4
voxels of the edit had 604 thin voxels (removed by a 6-connected opening) against 74
before the edit, and the regeneration kept 324. An offline replay of that edit
(`scratch/smart-brush/`) compared four forms. The user chose "grow from seeds, then
smooth the WM" (26 thin voxels, Dice 0.87 against the user's own change).

**The mode.**

- Paint mode gets a sub-mode, grow (key g toggles it). The label menu is the same:
  outside, WM and cortex. The eraser is not in grow mode.
- The user paints strokes on a few slices, in any view, as in paint mode. Submit (x or
  Enter) sends them to `POST /review/grow`. The server returns a proposal: voxels and
  their labels.
- The pane draws the proposal on the slice tiles, in its own colour. The 4a and 4b
  previews take the proposal as their strokes.
- Accept (key a) sends the proposal to `POST /review/paint`, as strokes. So an accepted
  proposal is one paint: it goes into the diff, and one undo removes it. Escape
  discards the proposal and the strokes. A new stroke discards the proposal and keeps
  the strokes.

**The proposal** (`label_grow.grow`, a new core module):

1. **Seeds.** The stroke voxels with their labels (`review.stroke_voxels`: a voxel in
   two strokes keeps the later one).
2. **The current labels.** The dseg's labels with the diff's entries (the labels that
   the user sees).
3. **The area.** The strokes' region and the voxels within BAND = 3 voxels of it. The
   region fills the slices between stroked slices: between two stroked slices of one
   view that are next to each other, the 2D signed distances of their strokes are
   interpolated linearly. (A stroke is on a slice of a view when all its voxels lie
   on one slice of exactly one axis.) The box is the strokes' bounding box with a
   margin. The area is refused (an error, nothing proposed) above 2·10⁶ voxels.
4. **Fixed seeds.** Outside the area, each voxel of the box keeps its current label.
5. **The growth.** A random walker (`skimage.segmentation.random_walker`, BSD licence)
   on echoes 2 and 3 in the box: the log intensity of each echo, scaled by its 2nd and
   98th percentiles in the area. β = 130, mode `cg_j`, tolerance 10⁻⁵. The classes are the labels of
   the seeds. Each voxel of the area takes its most probable class.
6. **The smoothing.** The WM mask after the growth is smoothed: a Gaussian of
   σ = 1 voxel on its signed distance, threshold 0. The smoothing changes only voxels
   of the area within ZONE = 4 voxels of a voxel that the growth changed. A voxel that leaves the
   WM takes its most probable non-WM class from the growth. A voxel that joins the WM
   takes the WM label.
7. **The proposal.** The voxels of the box whose new label is not their current label.

BAND, β, σ and ZONE are first values from site 1. The response gives them. A scan of a
value is a finding to report, not a setting to tune.

**The echoes.** The echoes that the viewer shows (`--echo-dir`), read for the box only,
at full resolution. Without echoes 2 and 3, grow mode is off, and the pane says why.

**The cost.** The area of site 1 was about 10⁵ voxels, and the growth and smoothing took
about 1 s on the CPU. The growth runs as a viewer job (`baillarger.viewer.jobs`), with a
progress state, so a large area does not block the pane.

Tests (on a synthetic image, not the site):

- Two classes of different contrast, with a bump of the WM label into the cortex. Cortex
  strokes on one slice in four, over the bump: the proposal moves the WM boundary to the
  true boundary, within one voxel, and changes no voxel outside the area.
- After the proposal, the WM within ZONE of the change has no voxel that a 6-connected
  opening removes, where the true boundary has none.
- Strokes on voxels that already have their label, in a region where the image agrees
  with the labels: an empty proposal.
- The same strokes give the same proposal (determinism).
- An area above the limit is refused, and nothing is written.
- An accepted proposal gives the same diff as `/review/paint` with the same voxels, and
  one undo restores the diff before it.

## Part 7: the standard edits file

**Why.** The path of the edits file is an argument of the viewer and of the
regeneration (`--edits`), so each launch and each regenerate command names it by hand.
The hashes, the kept fix diffs, the answer log and the grow log look like version
tracking, but none of them gives back an earlier state of the edits. The user's
decision (2026-11-23): each ex vivo hemisphere has one standard, BIDS-compliant edits
file, the diff between the input labels and the user's edits. A user who wants to
version-control this file does so as for any other file of a BIDS hierarchy.

**The rule.**

- A hemisphere has one edits file. Its path comes from the input labels:

  `<derivatives>/labeledits/sub-<sub>/ses-<ses>/anat/sub-<sub>_ses-<ses>_hemi-<h>_desc-labeledits_edits.tsv`

  with its sidecar. `<derivatives>` is the directory that holds the derivatives root
  of the input labels (the root is the directory above the input's
  `sub-<sub>/ses-<ses>/anat/`). So `labeledits` is a derivatives root of its own,
  beside the root of the input labels. `sub` and `ses` are the input's entities, and
  `<h>` is the hemisphere of the input labels (`white_dseg.hemisphere`).
- One function gives the path: `label_edits.standard_path(inp, hemisphere)`, from
  `label_edits.bids_path`. The viewer and the regeneration call it. No other code
  builds the path.
- The first write makes the `labeledits` root and its `dataset_description.json`
  (`bids.dataset_description`).
- The tool does not version the file. No command, launch entry or document of the
  project gives version control as a step of editing.

**The records are a TSV** (User, 2026-11-23). `label-edits/1` is edited in place
(version 1).

- The records file is a BIDS tabular file, `..._desc-labeledits_edits.tsv`: a header
  line, then one line for each edited voxel, with the columns `i`, `j`, `k` (the
  voxel's indices on the grid) and `label`, all integers, separated by tabs, with
  `\n` line ends. The lines are sorted by `i`, then `j`, then `k` (the order of the
  linear index), and each voxel is in the file one time. The name follows BIDS; the
  suffix `edits` is ours.
- The sidecar keeps its keys. It also describes the four columns, as BIDS does for a
  tabular file. `SHA256` is the hash of the `.tsv` file. The writer is deterministic:
  the same diff gives the same bytes.
- In memory the diff is `label_edits.Edits`, as now (`index`, `label`). Only
  `label_edits` reads and writes the file. The writes, the lock, the check of the
  hash and the repair of a stopped write do not change.
- The dseg's copy of the edits has the same form: `..._desc-wmtopo_edits.tsv`.
- A version control tool shows a change of the file voxel by voxel. The cost is the
  size: about 0.2 MB for 10,425 voxels, in proportion to the edited voxels. A write
  replaces the whole file, as now.
- The one `.npy` diff that exists (the user's) is converted one time:
  `python -m baillarger.label_edits convert PATH.npy` writes the `.tsv` and its
  sidecar next to it, reads them back, and checks that the diff is the same. It does
  not delete the `.npy` files. Then a regeneration gives the output set a `.tsv`
  copy. After that, no code reads a `.npy` diff, and a later PR removes the command.

**The viewer.** Without `--edits`, the viewer finds the file from the surface: the
surface's dseg, then the dseg's input labels (`white_dseg.input_labels`), then the
standard path. A hemisphere with no edits file starts with the empty diff. `--edits
PATH` stays, for another file: the tests, and a run on a scratch copy. `--sites` needs
`--log`, as now; the answer log has no standard path in this part.

**Editing is switched on in the pane** (User, 2026-11-23). With the rule, each viewer
on an output set of the same input labels finds the same file: an old set that was
renamed, and a scratch set made from the real input labels. The explicit `--edits`
was the guard against a write from the wrong viewer. The switch replaces it.

- Without `--edits`, the viewer starts with editing off. The pane shows the path of
  the edits file and a switch. `POST /review/editing {on}` sets it, for this viewer
  run only. The page never sends a path.
- While editing is off, each write is refused and nothing is written: a paint, an
  answer, an accept-fix and an undo. The edits file and its root are not made. The reads work: the pending edits, the previews and the grow proposal.
- With `--edits PATH`, editing is on at start, as now: the path in the command is the
  deliberate act.
- The viewer does not switch editing on if the dseg has no copy of the edits that it
  can read (an output set from before the TSV). The pane says that the set needs a
  regeneration.

**The regeneration.** `python -m baillarger.regenerate INPUT --root ROOT` uses the
standard file, if it is there. `--edits PATH` gives another file, and `--no-edits`
runs with no edits. The command logs the edits file that it uses.

**The regeneration is a command-line call only** (User, 2026-11-23). The pane's
regenerate button goes, with all that serves it: the routes `/review/regenerate`, the
subprocess under `scripts/memguard` (`review.Regeneration`, `regen_cap_gb` and the
`REGEN_*` values), the closing of the paint mode while it runs, and the button, the
state and the log lines in `review.js`.

- The user runs the command in a terminal, under `scripts/memguard` (about 20 min, a
  peak of about 22 GB on the laptop).
- The pane shows the command for its hemisphere as text to copy
  (`Review.regenerate_argv`, which stays): `GET /review/queue` gives it as
  `regenerate_command`. With an edits file at the standard path, the command has no
  `--edits`.
- The check of the outputs' identity stays. When the command has replaced the
  outputs, the review refuses each write until the viewer restarts, and the pane says
  so. The overrides are in the command's output.
- A write into the edits while the command runs is a valid edit. The command read
  the diff once, at its start, so this edit is pending after the restart.

**The grow log.** It is a debugging log of the grow brush, and it is off by default
(User, 2026-11-23). The viewer option `--grow-log PATH` switches it on and gives its
file. Without the option, the viewer writes no grow log, and nothing is written next
to the edits file but the edits. The log is not a record of the edits, and no code
reads it.

**Not changed.** The sidecar's keys of `label-edits/1`, and `label_edits.Edits`; the
undo stack (memory, one viewer run); the dseg's copy of the edits apart from its
form (the pending edits need it); the check of the outputs' identity after a
regeneration.

Tests: `standard_path` of input labels in a derivatives root gives the path above, and
refuses input labels that are not in `sub-<sub>/ses-<ses>/anat/` of a root; a viewer
without `--edits` on a phantom, with editing switched on, writes its paint into the
standard file, and makes the root's `dataset_description.json`; a viewer with `--edits` writes into that file and
makes no standard file; the regeneration without `--edits` gives the same files as
with `--edits` on the standard file; `--no-edits` gives the same files as a root with
no standard file; a viewer without `--grow-log` writes no grow log after a grow, a
paint and an undo, and a viewer with it writes their lines into that file.

Tests of the TSV: write and read give the same diff; the same diff gives the same
bytes; the file has the header and the columns above, sorted; a reader refuses a file
that is not sorted, has a voxel two times or outside the grid, or has a column
missing; the tests of Part 1 for the stopped write and the lock pass on the TSV; the
converter's `.tsv` reads back equal to the `.npy` diff, and the `.npy` files are
still there.

Tests of the switch: without `--edits`, a paint, an answer and an undo before the
switch are refused, and no file and no root are made; after the switch a
paint writes; the pending edits and a preview work with editing off; with `--edits`,
editing is on at start; on an output set with no readable copy of the edits, the
switch is refused.

Tests of the regeneration from the command line: `/review/regenerate` is not a route;
`GET /review/queue` gives the command, without `--edits` for a standard edits file
and with it for another file; after the outputs are replaced, a paint is refused
until a new viewer starts (the test of Part 5, kept).

## Order of the PRs

The joins merge in this order. The cores can merge earlier (see Parallel work).

1. Part 1.
2. Part 2.
3. Part 3.
4. Part 4a.
5. Part 4b.
6. Part 5.
7. Part 6: its core, then its join.
8. Part 7: its core, then its join. Then the one-time conversion of the user's edits
   file and a regeneration, on the user's instruction.

## Parallel work

Each part has a core and a join. The cores do not depend on each other. The joins put
the cores into the routes and the pane, and they meet in a few shared files.

| Part | Core (files) | Join (files) | The join needs |
|---|---|---|---|
| 1 | `label_edits.py` (new), `white_dseg.py`, `wm_sites.py`, `wm_paint.py`, `docs/formats/label-edits.md`, `wm-dseg.md`, `wm-sites.md`, `wm-edits.md`, their tests | `viewer/review.py` (queue answers write into the diff; the choice `accept-fix`; `--log` and `--edits`), `static/review.js` (the `accept-fix` button) | the `label_edits` contract |
| 2 | none (Part 1's core has the shared functions) | `viewer/review.py` (`/review/paint`, `/review/undo`, the shared strokes function, `--edits` alone), `static/review.js` (paint mode, undo); then the removal of `wm_paint.py`, `docs/formats/wm-paint.md`, `tests/test_wm_paint.py` and the ITK-SNAP commands | the `label_edits` contract |
| 3 | none | `viewer/volumes.py` ("input"), `viewer/review.py` (`/review/pending`), `static/review.js` (drawing) | the copy of the diff in the dseg's output |
| 4a | none | `viewer/review.py` (`/review/preview` takes voxels), `static/review.js` | the paint request |
| 4b | `antialias.py` (start, held), `white_surf.py` (the field derivative, the blocks' re-march), its format doc, their tests | `viewer/review.py` (route, working field), the 3D view and the slices (blocks) | the preview contract |
| 5 | a new `regenerate` command, its tests | `viewer/review.py` (the subprocess), `static/review.js` (state, message) | the command's arguments |
| 6 | `label_grow.py` (new), `tests/test_label_grow.py` | `viewer/review.py` (`/review/grow`, the echoes), `static/review.js` (grow mode, the proposal, accept) | the `label_grow` contract |
| 7 | `label_edits.py` (the TSV, `standard_path`, the converter), `white_dseg.py` (the copy as a TSV), `regenerate.py` (`--edits` optional, `--no-edits`), `docs/formats/label-edits.md`, `wm-dseg.md`, their tests | `viewer/review.py` (`--edits` optional; `/review/editing`; `--grow-log`; the regenerate button goes), `static/review.js` (the path and the switch; the regenerate button goes) | `standard_path`, the `/review/editing` contract |

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
- `label_edits.read(path, *, shape=None, affine=None, hemisphere=None) -> Edits`: the one
  reader; it refuses a diff of another grid or hemisphere (each when given) and repairs
  a stopped write. `label_edits.write(path, edits, *, hemisphere, shape, affine)`
  (replaces the files, under the lock); `label_edits.empty() -> Edits`.
- `label_edits.update(path, fn, *, hemisphere, shape, affine, known=None,
  labels_checked=False, after=None) -> Current`: one read-change-write under the
  exclusive lock; `fn(current)` returns the new diff, or None for no write.
  `after(current)`, if given, runs after the write (after the read, for no write) with
  the lock still held; if it raises, the diff before the update is written back under
  the same lock (the empty diff, if there was no file) and its error propagates. `known` (a `Current`: the
  edits, the SHA-256 and the records file that the caller last wrote) skips the read of
  the records while it agrees with the sidecar. `labels_checked` skips the check of the
  labels against label-set/1, for a caller that has checked them (`label_edits.paint`).
- `label_edits.apply(labels, edits, *, offset=(0, 0, 0), shape=None) -> labels` (a copy
  with the edits written). `labels` are the whole grid, or a crop of the
  grid of this `shape` whose voxel [0, 0, 0] is the grid voxel `offset`; the diff's voxels
  outside the crop are left out.
- `label_edits.paint(edits, labels, voxels, new, *, dseg) -> (Edits, undo)`: `labels` are
  the input labels and `dseg` the dseg's labels, of one shape (`dseg` omitted: the dseg is
  the input labels); `new` a label per voxel, or -1 for the eraser. A paint keeps only the
  voxels whose new label is not their current label (the diff's entry, else the dseg's
  label); the eraser keeps only the voxels with an entry. `undo` holds the earlier entries
  of those voxels.
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
- `POST /review/answer {site, choice: "manual-paint", strokes, reviewer_seconds}` (the
  strokes as for `/review/paint`); the response is the other answers' with
  `undo_depth`. No other choice takes strokes.
- `GET /review/queue` also gives `menu` (the paint labels), `paint_only` and
  `undo_depth`.
- `POST /review/preview` also takes `{site: "i,j,k", choice: "manual-paint", strokes}`;
  the response is the existing preview's.
- `GET /review/pending`: `{voxels: [[i, j, k], ...], labels: [...]}`.
- `antialias.thin_plate(mask, dev, band, eps, tol, max_iter, start=None, held=None)`.
- The field derivative: the antialiased field on the mask's box, an uncompressed float32
  `.nii`, with `desc-antialias` in its BIDS name and a spec in `docs/formats/` (version 1).
  `POST /review/surface-preview {strokes, since}` brings the working field to the edits
  and previews the strokes on a copy; it answers as a job (`{status, job}`), polled with
  `GET /review/surface-preview?job=&since=`. When done: the revision of the working
  blocks, the preview's block ids, the working blocks changed after `since`, and the
  radius used. `GET /review/surface-preview/mesh?job=&since=` gives those blocks'
  vertices (voxel coordinates of the grid) and faces (`viewer.columns.encode`);
  `GET /review/surface-preview/cut?job=&axis=&mm=` gives their cut by a slice plane, as
  `/outline` does.
- The input labels are the volume "input" of the pane (`Volumes.add_nifti`).
- `POST /review/regenerate` starts the command below under `scripts/memguard`;
  `GET /review/regenerate` gives its state. Both: `{status (idle | running | done |
  failed), log, overrides, error, outputs_changed, cap_gb, viewer_gb, command,
  seconds}`. While it runs, each write into the edits is refused.
- `label_grow.grow(strokes, current, echoes, *, shape, wm, settings=DEFAULT,
  box_limit=2_000_000) -> Proposal`: `strokes` a list of (label, voxels (N, 3)) in
  stroke order; `current(box)` the current labels of a box; `echoes(box)` echoes 2 and
  3 of a box; a box is ((lo), (hi)), hi exclusive; `shape` the grid; `wm` the WM label.
  `Proposal`: `index` (int64, sorted, unique), `label` (int32), `box` (lo, hi), and
  `settings` (BAND, β, σ, ZONE). It raises `ValueError` for an area above the limit.
- `POST /review/grow {strokes}` (the strokes as for `/review/paint`, no eraser) answers
  as a job (`{status, job}`), polled with `GET /review/grow?job=`. When done:
  `{voxels, labels, box, settings}`, or `{error}`. It writes nothing.
- The grow log (`review.grow_log_path`: next to the edits, suffix `growlog`, `.jsonl`;
  sidecar `grow-log/1`): with the grow brush on, one JSON line for each grow (strokes,
  proposal or error), paint (strokes, and the accepted proposal's `grow_job`), undo and
  surface preview (the summaries of its solves). It is for diagnosis and replay only.
  An accept is `POST /review/paint {strokes, grow_job}`.
- The command: `python -m baillarger.regenerate INPUT --edits EDITS --root ROOT`. INPUT
  is the input labels (the excised dseg). It writes the dseg, the white surface, the
  overview and the queue of sites, replaces the outputs in ROOT, and prints the
  overrides.
- Part 7: `label_edits.standard_path(inp, hemisphere) -> Path`: the edits file of the
  hemisphere of the input labels `inp` (Part 7, The rule). It raises `ValueError` for
  input labels that are not in `sub-<sub>/ses-<ses>/anat/` of a derivatives root.
- Part 7: `python -m baillarger.regenerate INPUT [--edits EDITS | --no-edits] --root
  ROOT`. With neither option, the edits are the standard file, if it is there.
- Part 7: the records of a diff are a `.tsv` (`i`, `j`, `k`, `label`); the functions
  of `label_edits` above keep their arguments and results.
- Part 7: `POST /review/editing {on: true | false}`. Response: `{editing, edits}`
  (`edits` is the path of the edits file), or `{error}` if the dseg has no readable
  copy of the edits. `GET /review/queue` also gives `editing` and `edits`. With
  editing off, each write route answers `{error}` and writes nothing.
- Part 7: `POST /review/regenerate` and `GET /review/regenerate` go. `GET
  /review/queue` also gives `regenerate_command` (the command line, a string).
- Part 7: the grow log is written only with the viewer option `--grow-log PATH`, into
  that file (`.jsonl`, with its sidecar `grow-log/1`). `review.grow_log_path` goes.

## Not in this plan

- Labels to train the network, and which labels train it.
- The success measure, and the queue's answer log apart from what Part 1 changes.
- A history of edits, and version control of the edits file: the user's choice, outside
  the tool.
- The review as an ordinary pane of the viewer, next to the surface pane. With Part 7,
  each viewer with `--white` can find the edits file, so this needs no new argument.
  It is a change of the viewer, for a plan of its own.
- Denser roots in the profile section.
- A 3D brush, and labels other than outside, WM and cortex. (The grow brush grows in
  3D from 2D strokes.)
- The excision map: corrections to it are paint edits with the label outside.
- A brick volume that changes while the viewer runs.
