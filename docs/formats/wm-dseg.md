# wm-dseg: the segmentation with a topologically correct WM mask

SpecVersion `wm-dseg/1` (the corrected segmentation) and `wm-dseg-fixes/1`
(the diagnostic image of the changes). Written by
`python -m baillarger.white dseg ASEG --out DERIV`
(`baillarger.white.write_wm_topo`); the keys are enforced by
`baillarger.white.DSEG_SPEC` and `FIXES_SPEC`, and `tests/test_bids.py` checks
that the tables below list the same keys, types and requirements.

## Files

In the BIDS derivatives root `DERIV`, for an input with entities `sub` and
`ses`:

| File (in `sub-<sub>/ses-<ses>/anat/`) | Content |
|---|---|
| `sub-<sub>_ses-<ses>_desc-wmtopo_dseg.nii.gz` | The corrected segmentation: the input's grid, affine, header and data type |
| `sub-<sub>_ses-<ses>_desc-wmtopo_dseg.tsv` | `index` and `name` (FreeSurfer LUT) of every label in it |
| `sub-<sub>_ses-<ses>_desc-wmtopo_dseg.json` | Sidecar, `wm-dseg/1` |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.nii.gz` | uint8, the changed voxels: 1 handle cut, 2 handle fill, 3 cavity fill; 0 elsewhere |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.tsv` | `index` and `name` of those values |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.json` | Sidecar, `wm-dseg-fixes/1` |

`DERIV/dataset_description.json` (`DatasetType` `derivative`) is written if
it is not there. BIDS's `desc` entity and `dseg` suffix are standard; the
`desc` values are ours.

## The correction

The hemisphere is the one whose WM label is in the input (41: R, 2: L; both
or neither is refused). The WM mask is the voxels with a filled label:

| Hemisphere | Filled labels (FreeSurfer LUT) |
|---|---|
| R | 41 WM, 43 lateral ventricle, 49 thalamus, 50 caudate, 51 putamen, 52 pallidum, 192 corpus callosum |
| L | 2 WM, 4 lateral ventricle, 10 thalamus, 11 caudate, 12 putamen, 13 pallidum, 192 corpus callosum |

1. Cavities of the mask (background components, Kuhn adjacency, that do not
   reach the border of the mask's bounding box plus 2 voxels) are filled.
2. Handles are removed (`baillarger.handles.remove_handles`): each is cut or
   filled, whichever changes fewer voxels.
3. Cavity and handle fills get the WM label. Each cut voxel gets the label
   of the nearest voxel (Euclidean, mm) that is outside the corrected mask
   and not cut.

Guarantee: the corrected segmentation's voxels with a filled label are the
corrected mask exactly, and its Betti numbers (Kuhn adjacency) are
(b0, 0, 0). Voxels outside the mask's bounding box are unchanged.

## Sidecar keys: wm-dseg/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-dseg/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | The input segmentation (absolute path) |
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
| `Moves` | array of objects | yes | Per handle move: `Kind` (`cut`, `fill` or `stranded`), `Voxels` (count), `Centre` (0-based voxel of the grid) |

## Sidecar keys: wm-dseg-fixes/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-dseg-fixes/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | The input segmentation (absolute path) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | As in `wm-dseg/1` |
