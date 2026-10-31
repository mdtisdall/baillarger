# wm-edits: manual label edits for the WM topology correction

SpecVersion `wm-edits/1`. Written by `baillarger.wm_edits.write_edits` (for
example by a review queue: one answer per site), read by
`python -m baillarger.white dseg ASEG --edits EDITS --out DERIV`
(`baillarger.white.write_wm_topo`); the keys are enforced by
`baillarger.wm_edits.EDITS_SPEC`, and `tests/test_bids.py` checks that the
table below lists the same keys, types and requirements.

The edits are diffs against one input segmentation (the aseg), which is never
changed. `dseg` applies them first, then makes its automatic correction
(`docs/formats/wm-dseg.md`), so no edit can leave a handle or a cavity in the
WM mask; and since the edits are kept apart from the automatic changes, they
apply again when the correction changes. Each answer is one entry, so it can
be reviewed, reverted or replayed on its own.

## Files

In the BIDS derivatives root `DERIV`, for an input with entities `sub` and
`ses`:

| File (in `sub-<sub>/ses-<ses>/anat/`) | Content |
|---|---|
| `sub-<sub>_ses-<ses>_desc-wmedits_edits.tsv` | One row per changed voxel |
| `sub-<sub>_ses-<ses>_desc-wmedits_edits.json` | Sidecar, `wm-edits/1`: the answers |

The `edits` suffix is ours (BIDS has none for this), in BIDS's entity order;
`dseg` reads the files and never writes them.

## The table

Tab-separated, with a header row, one row per voxel; no voxel appears twice.

| Column | Type | Meaning |
|---|---|---|
| `i`, `j`, `k` | integer | The voxel (0-based) on the input segmentation's grid |
| `old` | integer | The input's label there; `dseg` refuses the edits if the input has another |
| `new` | integer | The label to set (not `old`) |
| `answer` | integer | The `Id` of the answer the row belongs to |

## Answers

`Answers` in the sidecar lists every answer, in the order given. Each is an
object:

| Field | Type | Meaning |
|---|---|---|
| `Id` | integer | Unique within the file; rows refer to it |
| `Site` | array of 3 integers | The voxel (0-based) the answer is about |
| `Choice` | string | What was chosen, for example `remove-strand`, `thicken`, `fill-gap`, `leave` |
| `DecidedAgainst` | string or null | SHA-256 of the `desc-wmtopo_dseg.nii.gz` that was looked at when choosing, so that a later rerun can list the answers whose context changed; null if none |
| `Time` | string | When (ISO 8601) |

An answer may have no rows (for example `leave`): it is recorded so that the
site counts as answered.

## Sidecar keys: wm-edits/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-edits/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | The input segmentation the edits are against (absolute path) |
| `SourcesSHA256` | object | yes | Its SHA-256, by path; `dseg` refuses the edits for another input |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Answers` | array of objects | yes | The answers (above) |
