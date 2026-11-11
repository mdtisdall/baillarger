# label-edits: the label edits of a hemisphere

SpecVersion `label-edits/1`. Written by `baillarger.label_edits.write` and
`baillarger.label_edits.update` (the viewer's review pane, `baillarger.viewer.review` with `--edits`: the answers
of the review queue, the paint mode and its undo; and the copy that
`python -m baillarger.white_dseg ASEG --edits EDITS --out DERIV` keeps). Read
by `baillarger.label_edits.read` and by the dseg command (`--edits`,
`docs/formats/wm-dseg.md`). The sidecar's keys are enforced by
`baillarger.label_edits.EDITS_SPEC`, and `tests/test_bids.py` checks that the
table below lists the same keys, types and requirements.

## What the diff is

The label edits are one diff: each edited voxel and its label. The diff holds
the current state of the edits only. It does not keep a history of edits.
The dseg command applies the diff to the input labels before the automatic
correction. Nothing else applies edits.

- An edit records only the voxels whose label it changes. The label before
  the edit is the voxel's entry in the diff if it has one, else its label in
  the dseg (the labels that the user sees, after the automatic correction).
  To paint a voxel with that label is not an edit
  (`baillarger.label_edits.paint`). To paint a voxel back to its input label,
  where the correction changed it, is an edit: the voxel gets an entry with
  its input label.
- The eraser removes voxels from the diff. Then they take their input label
  again. The eraser does not change a voxel that has no entry.
- A voxel of the diff can have its input label. It is still an edited
  voxel: the correction changes it last (`wm-dseg/1`, Manual edits).
- The diff applies to any input labels on its grid (`Shape`, `Affine`).

The answer log of the review queue (`wm-edits/1`) is a separate record. The
diff does not come from it: a site answer of the queue writes its labels into
the diff (`baillarger.label_edits.paint`), and then the log records the
answer. Both writes are in one `baillarger.label_edits.update` (the log
append is its step after the write), so the diff's lock is held across both,
and no other writer writes between them. If the log refuses the answer, the
diff is written back as it was, under the same lock.
At an override site, the choice `accept-fix` writes the fix's labels into the
diff. A scribble is a record of the log only.

The paint mode of the review pane (`/review/paint`) writes its strokes into
the diff in the same way (`baillarger.label_edits.paint`); its labels are
outside, WM and cortex of the hemisphere, and the eraser. At a queue site,
the choice `manual-paint` is a paint whose strokes are the site's answer.
Every write into the diff of a viewer run (a paint, and a site answer of
any choice) pushes the earlier entries of the voxels that it changes on one
undo stack. Undo (`/review/undo`) writes the top entries back
(`baillarger.label_edits.restore`), so it always reverses the last write.
An undo is refused if the diff on disk is not the one that the top write
left: another writer changed it since. The undo stack is in the viewer's memory only, for one viewer run; the diff
keeps no history. An undo does not change the answer log.

## Files

| File | Content |
|---|---|
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-labeledits_edits.npy` | The records (below) |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-labeledits_edits.json` | Sidecar, `label-edits/1` |

The files are in `sub-<sub>/ses-<ses>/anat/` of a BIDS derivatives root
(`baillarger.label_edits.bids_path`). The copy that the dseg command keeps
has the same format and the name `sub-<sub>_ses-<ses>_desc-wmtopo_edits.npy`
(`wm-dseg/1`). The `edits` suffix and the `desc` values are ours.

### The records

A NumPy `.npy` file (no pickle) of one-dimensional structured records,
12 bytes each (`baillarger.label_edits.EDITS_DTYPE`):

| Field | Type | Meaning |
|---|---|---|
| `index` | `<i8` | The voxel's linear index on the grid (C order of `Shape`) |
| `label` | `<i4` | The voxel's label, in `label-set/1` |

The records are sorted by `index`, and each index is in the file one time.

## Writes and reads

A write replaces both files. It writes new files in the same directory,
flushes each to the disk, then renames them over the old ones: the records
first, then the sidecar. After each rename it flushes the directory
(`baillarger.fileops.write_atomic`). It holds an exclusive lock
(`fcntl.flock` on the directory, `baillarger.fileops.flock`) while it does
this. A read holds a shared lock on the directory, reads the sidecar and the
records, and checks the records against the sidecar's `SHA256` and `Count`.
So a reader sees the old diff or the new diff, never a mix. A write refuses
(ValueError, and writes nothing) a diff that is not sorted, has an index
two times or outside the grid, or has a label that is not in `label-set/1`.

A reader can also check the diff against a grid (`Shape` and `Affine`) and a
hemisphere: `baillarger.label_edits.read(path, shape=, affine=,
hemisphere=)` refuses (ValueError) a diff of another grid or hemisphere. The
viewer, the dseg command and `bent_box` use this one reader.

A change of the diff (a paint, an answer, an undo) is one
`baillarger.label_edits.update`: it holds the exclusive lock across the
read, the change and the write. So two writers (two viewer processes on one
diff) do not lose each other's voxels. An update can run a step after the
write with the lock still held (the answer log's append); if that step
fails, the update writes the diff before it back (the empty diff, if there
was no file) under the same lock. A writer can keep the diff that it
last wrote in memory: if the sidecar's `SHA256` and the records file are
the ones that it wrote, the update reads only the sidecar.

Recovery. A writer that stops between the two renames (a signal, the memory
cap, a power loss) leaves the new records with the old sidecar. A reader
that finds a `SHA256` or `Count` that does not agree with the records takes
the sidecar as older than the records. It checks the records on the
sidecar's `Shape` (sorted, each index once, inside the grid, labels in
`label-set/1`), then writes a new sidecar with the records' `Count` and
`SHA256` and the old sidecar's other facts, under the exclusive lock, and
logs a warning. Records that are not a valid diff are refused (ValueError).
A first write that stops between the renames leaves records with no
sidecar: an update writes the sidecar from its own grid and hemisphere.

## Sidecar keys: label-edits/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `label-edits/1` |
| `Description` | string | yes | What the file is |
| `Hemisphere` | string | yes | `R` or `L` |
| `Shape` | array of integers | yes | The grid's shape (3 integers) |
| `Affine` | array of numbers | yes | The grid's voxel-to-mm affine, 4 × 4, by rows (16 numbers) |
| `Count` | integer | yes | The number of records (edited voxels) |
| `SHA256` | string | yes | The SHA-256 of the `.npy` file |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
