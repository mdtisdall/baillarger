# wm-dseg: the segmentation with a topologically correct WM mask

SpecVersion `wm-dseg/1` (the corrected segmentation) and `wm-dseg-fixes/1`
(the diagnostic image of the changes). Written by
`python -m baillarger.white_dseg ASEG [--edits EDITS] --out DERIV`
(`baillarger.white_dseg.write_wm_topo`); the keys are enforced by
`baillarger.white_dseg.DSEG_SPEC` and `FIXES_SPEC`, and `tests/test_bids.py` checks
that the tables below list the same keys, types and requirements.

## Files

In the BIDS derivatives root `DERIV`, for an input with entities `sub` and
`ses`:

| File (in `sub-<sub>/ses-<ses>/anat/`) | Content |
|---|---|
| `sub-<sub>_ses-<ses>_desc-wmtopo_dseg.nii.gz` | The corrected segmentation: the input's grid, affine, header and data type |
| `sub-<sub>_ses-<ses>_desc-wmtopo_dseg.tsv` | `index` and `name` (FreeSurfer LUT) of every label in it |
| `sub-<sub>_ses-<ses>_desc-wmtopo_dseg.json` | Sidecar, `wm-dseg/1` |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.nii.gz` | uint8, the changed voxels: 1 handle cut, 2 handle fill, 3 cavity fill, 4 manual edit, 5 small piece; 0 elsewhere. A voxel that two steps change shows the value of the later step (the order: 4, then 1 to 3, then 5) |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.tsv` | `index` and `name` of those values |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.json` | Sidecar, `wm-dseg-fixes/1` |

`DERIV/dataset_description.json` (`DatasetType` `derivative`) is written if
it is not there. BIDS's `desc` entity and `dseg` suffix are standard; the
`desc` values are ours.

## The correction

The hemisphere is the one whose WM label is in the input (41: R, 2: L; both
or neither is refused). The WM mask is the voxels with a filled label:

The filled labels are in `docs/formats/label-set.md` (`baillarger.labels.FILLED_LABELS`).
The hippocampus (53 / 17) is not filled: it is archicortex, whose laminar
axis does not run out from the WM below it. Phase 1.8 decides whether this
stays.

1. Cavities of the mask (background components, Kuhn adjacency, that do not
   reach the border of the mask's bounding box plus 2 voxels) are filled.
2. Handles are removed (`baillarger.handles.remove_handles`): each is cut or
   filled, whichever changes fewer voxels.
3. Cavity and handle fills get the WM label. Each cut voxel gets the label
   of the nearest voxel (Euclidean, mm) that is outside the corrected mask
   and not cut.
4. Small pieces are removed. A small piece is a piece of the mask after
   steps 1 to 3 (Kuhn adjacency) whose volume (voxels × the voxel volume)
   is less than `CrumbMM3` (1 mm³, `baillarger.labels.CRUMB_MM3`; the
   same value as in `wm-excisions/1`). Each voxel of a small piece gets the
   most common label of its 26-neighbours (in the image) that do not have a
   filled label, in the labels after step 3. If two labels have the same
   count, the smaller label value wins. If a voxel has no such neighbour,
   the command writes nothing and stops with an error. A speck that touches
   a larger piece only across an edge or corner diagonal is a piece of its
   own (Kuhn adjacency does not join them), so it is removed and not joined.

Guarantee: the corrected segmentation's voxels with a filled label are the
corrected mask exactly, and its Betti numbers (Kuhn adjacency) are
(b0, 0, 0), and each of its b0 pieces has a volume of `CrumbMM3` or more.
Voxels outside the mask's bounding box are unchanged. Each piece
of the mask is made handle- and cavity-free by construction, with the other
pieces as background; a fill of one piece can touch another and make a
handle or a cavity of the union, so the command checks the Betti numbers and
writes nothing if b1 or b2 is not 0.

## Manual edits

With `--edits`, the manual edits of the input in a `wm-edits/1` table
(`docs/formats/wm-edits.md`) are made first, and the correction above runs on
the edited segmentation: the filled labels, the mask and `BettiBefore` are
those of the edited input, so no edit can leave a handle or a cavity. `dseg`
refuses edits made against another input (their `SourcesSHA256`) or that do
not fit it (a voxel without its `old` label). The edits table and its sidecar
are listed in `Sources`; each answer with rows is a move of kind `manual`.
The input segmentation is never changed, and the derivative stays written
whole by the command, so an edit costs a `dseg` run.

## Sidecar keys: wm-dseg/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-dseg/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | The input segmentation, then with `--edits` the edits table and its sidecar (absolute paths) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `FilledLabels` | array of integers | yes | The labels of the WM mask |
| `WMLabel` | integer | yes | The label given to cavity and handle fills |
| `BettiBefore` | array of integers | yes | (b0, b1, b2) of the mask of the input's filled labels |
| `BettiAfter` | array of integers | yes | (b0, b1, b2) of the corrected mask |
| `CavityVoxels` | integer | yes | Voxels set to `WMLabel` by the cavity fill |
| `HandleFillVoxels` | integer | yes | Voxels set to `WMLabel` by handle fills |
| `HandleCutVoxels` | integer | yes | Voxels cut from the mask (cuts and stranded voxels) |
| `CutLabels` | object | yes | New label of the cut voxels (as a string): number of voxels |
| `SmallPieces` | integer | yes | Pieces removed by step 4 (small pieces) |
| `SmallPieceVoxels` | integer | yes | Voxels of those pieces: their value is 5 in the fixes image |
| `ManualVoxels` | integer | yes | Voxels changed by the manual edits (0 without `--edits`) |
| `Moves` | array of objects | yes | Per manual answer with rows, then per handle move, then per small piece: `Kind` (`manual`, `cut`, `fill`, `stranded` or `small_piece`), `Voxels` (count), `Centre` (0-based voxel of the grid), and for `manual` the `Answer` (its `Id`) |

## Sidecar keys: wm-dseg-fixes/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-dseg-fixes/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | As in `wm-dseg/1` |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | As in `wm-dseg/1` |
