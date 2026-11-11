# label-edits: the label edits of a hemisphere

SpecVersion `label-edits/1`. Written by `baillarger.label_edits.write` (the
ITK-SNAP import `python -m baillarger.wm_paint import`, and the copy that
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
  the edit is the input label with the diff applied. To paint a voxel with
  that label is not an edit (`baillarger.label_edits.paint`).
- The eraser removes voxels from the diff. Then they take their input label
  again.
- A voxel of the diff can have its input label. It is still an edited
  voxel: the correction changes it last (`wm-dseg/1`, Manual edits).
- The diff applies to any input labels on its grid (`Shape`, `Affine`).

The answer log of the review queue (`wm-edits/1`) is a separate record. The
diff does not come from it.

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
