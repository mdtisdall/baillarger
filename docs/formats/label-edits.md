# label-edits: the label edits of a hemisphere

SpecVersion `label-edits/1`. Written by `baillarger.label_edits.write` (the
viewer's review pane, `baillarger.viewer.review` with `--edits`: the answers
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
the diff (`baillarger.label_edits.paint`), and the log records the answer.
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
The undo stack is in the viewer's memory only, for one viewer run; the diff
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

A write replaces both files. It writes new files in the same directory, then
renames them over the old ones: the records first, then the sidecar. It
holds an exclusive lock (`fcntl.flock` on the directory) while it does this.
A read holds a shared lock on the directory, reads the sidecar and the
records, and checks the records against the sidecar's `SHA256` and `Count`.
So a reader sees the old diff or the new diff, never a mix. A write refuses
(ValueError, and writes nothing) a diff that is not sorted, has an index
two times or outside the grid, or has a label that is not in `label-set/1`.

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
