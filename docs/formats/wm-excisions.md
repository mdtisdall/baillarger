# wm-excisions: the aseg with its excisions set to outside

SpecVersion `wm-excisions/1`. Written by
`python -m baillarger.excisions ASEG FLASH --echo1 E1 --echo3 E3 --hemi H --out DERIV`
(`baillarger.excisions.write_excisions`). The keys are enforced by
`baillarger.excisions.EXCISIONS_SPEC`. `tests/test_bids.py` checks that the
table below lists the same keys, types and requirements.

An excision is a block of tissue that was cut out of the specimen before the
scan. The FLASH is black there. An aseg that imputes tissue (for example
purple-mri's) gives WM or cortex labels to these voxels. This command finds
these voxels and sets them to 0 (outside). The result is the input label of
`python -m baillarger.white_dseg` (`docs/formats/wm-dseg.md`).

This is a bootstrap tool. Use it only on an aseg that imputes tissue: the
Fomblin outside the specimen is as dark as an excision, so on other labels
the detector has no meaning. The input aseg is not changed.

## Files

In the BIDS derivatives root `DERIV`, for an input with entities `sub` and
`ses`:

| File (in `sub-<sub>/ses-<ses>/anat/`) | Content |
|---|---|
| `sub-<sub>_ses-<ses>_desc-excised_dseg.nii.gz` | The aseg with each voxel of the excision map set to 0: the aseg's grid, affine, header and data type |
| `sub-<sub>_ses-<ses>_desc-excised_dseg.tsv` | `index` and `name` (FreeSurfer LUT) of every label in it |
| `sub-<sub>_ses-<ses>_desc-excised_dseg.json` | Sidecar, `wm-excisions/1` |
| `sub-<sub>_ses-<ses>_desc-excisions_mask.nii.gz` | The excision map: uint8, 1 at each voxel that the command set to 0, 0 elsewhere. The aseg's grid and affine |
| `sub-<sub>_ses-<ses>_desc-excisions_mask.json` | Sidecar, `wm-excisions/1`, with the same values as the dseg's sidecar except `Description` |
| `sub-<sub>_ses-<ses>_desc-bubbles_mask.nii.gz` | The bubble mask: uint8, 1 at each voxel of an air bubble and its artefact at echo 3 (step 7), 0 elsewhere. The aseg's grid and affine. Later stages do not trust the data there |
| `sub-<sub>_ses-<ses>_desc-bubbles_mask.json` | Sidecar, `wm-excisions/1`, with the same values as the dseg's sidecar except `Description` |

`DERIV/dataset_description.json` (`DatasetType` `derivative`) is written if
it is not there. BIDS's `desc` entity and its `dseg` and `mask` suffixes are
standard. The `desc` values are ours.

## The detector

FLASH is echo 2 of the FLASH on the aseg's grid. E1 and E3 are echoes 1 and
3, on the same grid: they tell air bubbles from excisions (step 7). The
command refuses an echo with a different grid shape.

The tissue is the voxels with the WM label or the cortex label of the
hemisphere `H`:

| Hemisphere | WM label | Cortex label |
|---|---|---|
| R | 41 | 42 |
| L | 2 | 3 |

1. The reference level is the median of the FLASH over the cortex label
   (`CortexMedian`).
2. Smooth the FLASH with a Gaussian of `Sigma` voxels (float32).
3. The candidates are the tissue voxels where the smoothed FLASH is less
   than `Frac` × `CortexMedian`.
4. Keep the connected components of the candidates (face adjacency) that
   have `MinVoxels` voxels or more.
5. Grow the kept voxels by round(`MarginMM` / smallest voxel side) steps of
   binary dilation (face adjacency). Keep only the tissue voxels.

6. Drop each connected component (face adjacency) of the result of step 5
   that is smaller than `FragmentMM3` mm³. These are slivers that the
   dilation leaves on the rim of an excision. Their voxels keep their labels.
7. Air bubbles. For each connected component (face adjacency) of the result
   of step 6, take its bounding box grown by `BubbleBox` voxels. In echo e
   (1, 2, 3), the dark region V_e is the connected components (face
   adjacency) of (echo e smoothed by `Sigma` voxels) < `Frac` × (the median
   of echo e over the cortex label) that overlap the component, in the box.
   The component is an air bubble if no V_e reaches the side of the box (the
   dark region is closed) and V_3 ≥ `BubbleRatio` × V_1. A gas pocket
   darkens a region that grows with the echo time. A cut does not grow.
   Drop each bubble from the result of step 6: its voxels keep their labels,
   because the anatomy there is tissue with a gas pocket. The bubble mask
   is V_3 of each bubble, grown by ceil(r_3 − r_1) steps of binary dilation
   (face adjacency), where r_e is the radius of a ball of V_e voxels: the
   growth of the artefact from echo 1 to echo 3.
8. Set each voxel of the result of step 7 to 0.
9. Find the connected components (face adjacency) of the tissue that step 8
   leaves. Set to 0 each component that is smaller than `CrumbMM3` mm³ and
   touches the result of step 7: a voxel of the component is in it, or is
   next to it (26-adjacency). These are crumbs that an excision cut off. A
   small piece that no excision touches is not changed.

The excision map is the voxels that steps 8 and 9 set to 0. The command
changes no other voxel. Labels other than the WM and cortex labels (for
example the deep structures) are never set to 0.

The defaults: `FragmentMM3` 0.1, `CrumbMM3` 1.0, `BubbleRatio` 2.0,
`BubbleBox` 20. On the first hemisphere (INDD121850, R, 0.16 mm), the
fragments were 1–6 voxels (at most 0.025 mm³) and the smallest piece of the
detector was 785 voxels (3.2 mm³). The largest crumb of the WM mask was
0.53 mm³. Three closed pieces grew from echo 1 to echo 3 by 2.9, 3.1 and
4.9 (air bubbles). Every open piece grew by 1.8 or less.

## Sidecar keys: wm-excisions/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-excisions/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | The aseg, then echoes 2, 1 and 3 of the FLASH (absolute paths) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `TissueLabels` | array of integers | yes | The WM label and the cortex label that the detector can set to 0 |
| `Detector` | object | yes | The parameters: `Frac` (number), `Sigma` (number, voxels), `MinVoxels` (integer), `MarginMM` (number), `DilationSteps` (integer), `FragmentMM3` (number), `CrumbMM3` (number), `BubbleRatio` (number), `BubbleBox` (integer, voxels) |
| `CortexMedian` | number | yes | The median of echo 2 over the cortex label (step 1) |
| `ExcisedVoxels` | integer | yes | Voxels set to 0 (steps 8 and 9) |
| `ExcisedByLabel` | object | yes | Voxels set to 0, by old label (as a string): number of voxels |
| `CortexVoxels` | integer | yes | Voxels with the cortex label in the aseg |
| `Components` | integer | yes | Connected components of the excision map (face adjacency) |
| `Fragments` | object | yes | The pieces that step 6 dropped: `Pieces` (integer), `Voxels` (integer) |
| `Crumbs` | object | yes | The tissue pieces that step 9 set to 0: `Pieces` (integer), `Voxels` (integer) |
| `Bubbles` | object | yes | The air bubbles of step 7: `Pieces` (integer), `Voxels` (integer: the voxels dropped from the excision map), `MaskVoxels` (integer: the voxels of the bubble mask) |
