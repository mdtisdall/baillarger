# wm-edits: the answer log of the WM review

SpecVersion `wm-edits/1`. Written by `baillarger.wm_edits.EditLog` (the
review pane of stream 4, `baillarger.viewer.review`); read by
`baillarger.wm_edits.read_log`. The sidecar's keys are enforced by
`baillarger.wm_edits.EDITS_SPEC`, and `tests/test_bids.py` checks that the
table below lists the same keys, types and requirements.

The log holds the answers of the review in the order that they were made,
and the records of each review session. It is the review queue's own record.
It holds raw facts only, and no measure.

## The log does not change the labels

`dseg` does not read the log. An answer changes the labels by a write into
the label edits (`label-edits/1`, `docs/formats/label-edits.md`: one diff,
each edited voxel and its label), and the log records the answer. The
`dseg` command reads the diff (`--edits EDITS`, `docs/formats/wm-dseg.md`).

An answer holds voxels and their labels, with no old label and no check
against an input, on the log's grid (`Shape`, `Affine`).

There are two kinds of answer:

- **A site answer** is a choice at a site that the review showed
  (`docs/formats/wm-choices.md`): `remove-strand`, `thicken`, `fill-gap`,
  `widen-gap`, `leave`, `manual-paint` or `accept-fix`. `accept-fix` is a
  choice of an override site (`wm-sites/1`): its voxels and labels are the
  site's fix's in the fix diff (`wm-dseg-fixes/1`), and its `Params` are
  `{"Fix": <the fix's id>}`. `manual-paint`'s voxels are the strokes of the
  pane's paint mode, each with the label that it takes (the eraser's
  voxels: their input label). The review pane writes the labels of each site
  answer into the label edits; a scribble changes no label. A paint of the
  paint mode outside a site answer, and its undo, are not records of the log.
- **A scribble** is a set of strokes, each with one class (a label of
  `label-set/1`). It can be at a site that was shown, or at no site.

`leave` has no voxels. It is recorded, so that the site counts as answered.

## The log is append-only

A correction is a new answer. An answer can name the answers that it
supersedes (`Supersedes`); that is a record only. Nothing in the log is
changed or removed. `Log.without(id)` gives the log
without one answer, in memory, for inspection; it does not change the file.

## Files

In the BIDS derivatives root `DERIV`, for an input with entities `sub` and
`ses` (no `hemi`, as for `wm-dseg/1`):

| File (in `sub-<sub>/ses-<ses>/anat/`) | Content |
|---|---|
| `sub-<sub>_ses-<ses>_desc-wmedits_edits.jsonl` | The log: one JSON object per line, one record per line |
| `sub-<sub>_ses-<ses>_desc-wmedits_edits.bin` | The voxels of the answers |
| `sub-<sub>_ses-<ses>_desc-wmedits_edits.json` | Sidecar, `wm-edits/1`, written one time, when the log is made |

`desc-wmeditstest` names a test log. Its answers are not real answers. The `edits` suffix is ours (BIDS has
none for this).

Why this layout:

- JSON Lines appends one line per event. A reader can read a file that is
  being written: it ignores a last line with no line end.
- The voxels of all answers are in one binary file, not one file per
  answer, so that the number of files stays small.
- The writer (`EditLog`) holds an exclusive lock (`fcntl.flock` on the
  `.jsonl`) for each append. It appends the voxels first, then the line,
  and flushes both to the disk. A line gives the offset, count and SHA-256
  of its voxels, so bytes of the `.bin` file that no line names (an append
  that did not finish) are ignored, and the next append writes over them.

### The voxels

The `.bin` file is a sequence of little-endian records of 16 bytes
(`baillarger.wm_edits.VOXEL_DTYPE`), with no header:

| Field | Type | Meaning |
|---|---|---|
| `index` | int64 | The voxel's linear index on the grid (C order of `Shape`) |
| `label` | int32 | For a site answer, the voxel's new label; for a scribble, the stroke's class |
| `stroke` | int32 | For a scribble, the stroke (0, 1, ...); -1 for a site answer |

A voxel is at most once in one answer. Each label is in `label-set/1`.

## Records

Each line is one JSON object. `Record` gives its kind. `Time` is ISO 8601
with the time zone. `Site` is a voxel `[i, j, k]` (0-based) on the grid.

### `session`

A review session starts.

| Field | Type | Meaning |
|---|---|---|
| `Session` | integer | The session's id: 1, 2, ... in the log |
| `Time` | string | When it started |
| `RandomShare` | number | The share of random sites that the reviewer chose at its start (0 to 1) |
| `RandomSeed` | integer or null | The seed of the random sites (`wm-sites/1`, Random sites); null with no random sites |
| `DsegSHA256` | string | The SHA-256 of the `desc-wmtopo_dseg.nii.gz` shown |
| `QueueSHA256` | string | The SHA-256 of the queue file shown (`wm-sites/1`) |

### `shown` and `skipped`

A site was shown in a session, or the reviewer skipped it.

| Field | Type | Meaning |
|---|---|---|
| `Session` | integer | The session |
| `Time` | string | When |
| `Site` | array of 3 integers | The site |
| `SiteKind` | string | `shown` only: its `kind` in the queue (`wm-sites/1`) |
| `How` | string | `shown` only: `random` for a random site, `ranked` for another |
| `Rank` | integer or null | `shown` only: its rank in the queue; null for a random site |
| `Score` | number or null | `shown` only: its score in the queue; null if the queue has none |

A site is skipped only after it was shown in the same session.

### `answer`

A site answer or a scribble.

| Field | Type | Meaning |
|---|---|---|
| `Id` | integer | 1, 2, ... in the log, unique |
| `Session` | integer | The session in which it was made |
| `Kind` | string | `site` or `scribble` |
| `Site` | array of 3 integers or null | The site. A site answer has one; a scribble can have none |
| `SiteKind`, `How`, `Rank`, `Score` | | From the `shown` record of that site in the session (the last one); for no site: null, `none`, null, null |
| `Choice` | string | A site answer's choice (above); `scribble` for a scribble |
| `Params` | object | The choice's parameters, for example `radius` and `reach` |
| `Supersedes` | array of integers | The Ids of earlier answers that it corrects |
| `DecidedAgainst` | string or null | The SHA-256 of the `desc-wmtopo_dseg.nii.gz` that the reviewer looked at. Provenance, not a gate |
| `Time` | string | When it was submitted |
| `ReviewerSeconds` | number | The reviewer time, from opening the site to submitting the answer |
| `Voxels` | object | `Offset` and `Count` (records of the `.bin` file) and `SHA256` (of those bytes) |
| `Strokes` | array of objects or null | A scribble's strokes, `{"Stroke", "Class"}`; null for a site answer |

The log refuses (ValueError, and writes nothing) an answer at a site that
was not shown in its session, a site answer with no site, `leave` with
voxels, an unknown choice, and
`Supersedes` with an Id that is not before it.

Whether an answer changes voxels of the `dseg` is computed from the answer
and the `dseg` at its `DecidedAgainst`. The log does not record it.

## Sidecar keys: wm-edits/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-edits/1` |
| `Description` | string | yes | What the files are |
| `Sources` | array of strings | yes | The segmentation whose grid the log takes, when the log was made (absolute path). Provenance only |
| `SourcesSHA256` | object | yes | Its SHA-256, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `Shape` | array of integers | yes | The grid's shape (3 integers) |
| `Affine` | array of numbers | yes | The grid's voxel-to-mm affine, 4 × 4, by rows (16 numbers) |
