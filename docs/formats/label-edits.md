# label-edits: the label edits of a hemisphere

SpecVersion `label-edits/1`. Written by `baillarger.label_edits.write` and
`baillarger.label_edits.update` (the viewer's review pane, `baillarger.viewer.review`: the answers
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

The pending edits are the voxels where the diff and the dseg's copy of the
diff (below) differ (`baillarger.label_edits.differ`): the edits that are
not yet in the dseg. The review pane gets them from `/review/pending`, each
with the label in the diff (an erased voxel: its input label), and draws
them on the slice tiles after each paint and undo.

## Files

| File | Content |
|---|---|
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-labeledits_edits.tsv` | The records (below) |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-labeledits_edits.json` | Sidecar, `label-edits/1` |

The files are in `sub-<sub>/ses-<ses>/anat/` of a BIDS derivatives root
(`baillarger.label_edits.bids_path`). The copy that the dseg command keeps
has the same format and the name `sub-<sub>_ses-<ses>_desc-wmtopo_edits.tsv`
(`wm-dseg/1`). The `edits` suffix and the `desc` values are ours.

### The standard file

A hemisphere has one edits file. Its path comes from the input labels
(`baillarger.label_edits.standard_path(inp, hemisphere)`): for input labels at
`<derivatives>/<root>/sub-<sub>/ses-<ses>/anat/<file>`, it is

`<derivatives>/labeledits/sub-<sub>/ses-<ses>/anat/sub-<sub>_ses-<ses>_hemi-<h>_desc-labeledits_edits.tsv`

So `labeledits` is a derivatives root of its own, beside the root of the input
labels; `<h>` is the hemisphere of the input labels
(`baillarger.white_dseg.hemisphere`). Input labels that are not in
`sub-<sub>/ses-<ses>/anat/` of a root have no standard file (ValueError). The
first write into a BIDS root (a diff in `sub-<sub>/ses-<ses>/anat/` of it)
makes the root's `dataset_description.json` (`baillarger.bids.dataset_description`).
`python -m baillarger.regenerate INPUT --root ROOT` uses the standard file of
INPUT if it is there (`--edits PATH` gives another file, `--no-edits` none).
The tool does not version the file; a user who wants a history of it uses
version control as for any other file.

### The records

A BIDS tabular file (`.tsv`): a header line, then one line for each edited
voxel. The values are integers in decimal, with no sign for a value of 0 or
more and no leading zeros, separated by one tab; each line ends with `\n`
(also the last one).

| Column | Meaning |
|---|---|
| `i` | The voxel's index on the grid's first axis (0 ≤ `i` < `Shape[0]`) |
| `j` | The voxel's index on the grid's second axis |
| `k` | The voxel's index on the grid's third axis |
| `label` | The voxel's label, in `label-set/1` |

The header is `i`, `j`, `k`, `label`, in this order. The lines are sorted by
`i`, then `j`, then `k` (the order of the linear index, C order of `Shape`),
and each voxel is in the file one time. The writer is deterministic: the same
diff gives the same bytes (`baillarger.label_edits.tsv_bytes`). A reader
refuses (ValueError) a file with a column missing or another column, a line
with another number of fields, a value that is not an integer, a voxel
outside the grid, lines that are not sorted, a voxel two times, a label not
in `label-set/1`, or any other form than the writer's. The file grows with
the edited voxels (about 20 bytes a voxel), and a write replaces the whole
file.

The earlier form of the records, a NumPy `.npy` of (`index` `<i8`, `label`
`<i4`) records, is read only by the one-time converter:
`python -m baillarger.label_edits convert PATH.npy` checks the `.npy`
against its sidecar, writes the `.tsv` and its sidecar next to it, reads
them back and checks that the diff is the same. Both files have the same
sidecar path, so the converter first copies the `.npy`'s sidecar to
`PATH.npy.json`; it deletes nothing.

## Writes and reads

A write replaces both files. It writes new files in the same directory,
flushes each to the disk, then renames them over the old ones: the records
first, then the sidecar. After each rename it flushes the directory
(`baillarger.fileops.write_atomic`). It holds an exclusive lock
(`fcntl.flock` on the directory, `baillarger.fileops.flock`) while it does
this. A read holds a shared lock on the directory, reads the sidecar and the
records, and checks the records against the sidecar's `SHA256` and `Count`
(a `Count` that does not agree with records of the right `SHA256` is refused).
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
that finds a `SHA256` that does not agree with the records takes
the sidecar as older than the records. It checks the records on the
sidecar's `Shape` (as a read does: the form, sorted, each voxel once,
inside the grid, labels in `label-set/1`), then writes a new sidecar with the records' `Count` and
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
| `SHA256` | string | yes | The SHA-256 of the `.tsv` file |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `i` | object | yes | The column `i`, as BIDS describes a column of a tabular file (`Description`) |
| `j` | object | yes | The column `j` (`Description`) |
| `k` | object | yes | The column `k` (`Description`) |
| `label` | object | yes | The column `label` (`Description`) |
