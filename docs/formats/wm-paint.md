# wm-paint: a crop for ITK-SNAP

SpecVersion `wm-paint/1`. Written by `python -m baillarger.wm_paint export`
(`baillarger.wm_paint.export`). Read by `baillarger.wm_paint.changes` and by
`python -m baillarger.wm_paint import` (`baillarger.wm_paint.import_paint`),
which appends to the answer log (`docs/formats/wm-edits.md`). The sidecar's
keys are enforced by `baillarger.wm_paint.PAINT_SPEC`, and
`tests/test_bids.py` checks that the table below lists the same keys, types
and requirements.

A crop is a cube of the `desc-wmtopo` segmentation round a review site. The
reviewer paints the labels of the crop in ITK-SNAP. The painted change
becomes one answer of the log.

## Files

In the crop directory that `export` is given:

| File | Content |
|---|---|
| `labels.nii.gz` | The crop of the segmentation. The reviewer paints this file and saves it in place |
| `exported.nii.gz` | An unchanged copy of the same crop. Not to edit. `import` compares it with `labels.nii.gz` |
| `echo-<n>.nii.gz` | The crop of each echo given to `export`, for n = 1, 2, ... in the order given |
| `crop.json` | Sidecar, `wm-paint/1`, written one time, by `export` |

`export` refuses (FileExistsError) a directory that has `labels.nii.gz`.

## Steps

1. Run `export` with the segmentation, the site (`--site I J K`), the
   crop directory (`--out`), the echoes (`--echo`, repeated) and the half-width
   (`--half`, default 32 voxels).
2. Open `labels.nii.gz` in ITK-SNAP, with the echoes.
3. Paint with labels of `label-set/1`.
4. Save `labels.nii.gz` in place.
5. Run `import` with the crop directory, the log (`--log`), the session
   (`--session`) and the reviewer time in seconds (`--seconds`). It prints
   the Id of the answer.

## What `export` does

- The crop is the cube of half-width `--half` round the site, cut by the
  edges of the grid (`baillarger.wm_choices.box_round`).
- Each echo must have the same shape (first three axes) and the same affine
  as the segmentation. Otherwise `export` refuses (ValueError) and writes
  nothing.
- `Offset` is the grid voxel of the crop's voxel `[0, 0, 0]`. The crop's
  affine maps that voxel to the same world point as the grid voxel `Offset`.

## What `import` does

1. It refuses (ValueError) a log whose `Shape` is not the crop's `Shape`.
2. It compares `labels.nii.gz` with `exported.nii.gz`. Both must have the
   same shape and affine. The voxels where they differ are the change. The
   new label of each is the label in `labels.nii.gz`.
3. It appends one answer to the log. The answer's `Choice` is
   `manual-paint`, and its `Site` is the crop's `Site`. Its voxels are the
   changed voxels, in grid coordinates (crop index plus `Offset`), with
   their new labels. Its `DecidedAgainst` is the crop's `DsegSHA256`. Its
   `Params` has `Crop`, the absolute path of the crop directory.
4. An answer with no changed voxels is valid. Its `Count` is 0.

The site must have been shown in the session. Otherwise the log refuses the
answer (ValueError) and writes nothing. The log also refuses a label that is
not in `label-set/1`.

## Sidecar keys: wm-paint/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-paint/1` |
| `Description` | string | yes | What the files are |
| `Sources` | array of strings | yes | The segmentation, then the echoes, that were cropped (absolute paths) |
| `SourcesSHA256` | object | yes | Their SHA-256, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Site` | array of integers | yes | The site `[i, j, k]` on the grid |
| `Offset` | array of integers | yes | The grid voxel of the crop's voxel `[0, 0, 0]` |
| `Shape` | array of integers | yes | The grid's shape |
| `DsegSHA256` | string | yes | SHA-256 of the `desc-wmtopo` segmentation that was cropped |
