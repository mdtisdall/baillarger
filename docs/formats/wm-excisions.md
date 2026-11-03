# wm-excisions: the aseg with its excisions set to outside

SpecVersion `wm-excisions/1`. Written by
`python -m baillarger.excisions ASEG FLASH --hemi H --out DERIV`
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

`DERIV/dataset_description.json` (`DatasetType` `derivative`) is written if
it is not there. BIDS's `desc` entity and its `dseg` and `mask` suffixes are
standard. The `desc` values are ours.

## The detector

FLASH is one echo of the FLASH on the aseg's grid. Use echo 2. The command
refuses a FLASH with a different grid shape.

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

The excision map is the result of step 5. The command sets each voxel of it
to 0. It changes no other voxel. Labels other than the WM and cortex labels
(for example the deep structures) are never set to 0.

## Sidecar keys: wm-excisions/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-excisions/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | The aseg, then the FLASH (absolute paths) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `TissueLabels` | array of integers | yes | The WM label and the cortex label that the detector can set to 0 |
| `Detector` | object | yes | The parameters: `Frac` (number), `Sigma` (number, voxels), `MinVoxels` (integer), `MarginMM` (number), `DilationSteps` (integer) |
| `CortexMedian` | number | yes | The median of the FLASH over the cortex label (step 1) |
| `ExcisedVoxels` | integer | yes | Voxels set to 0 |
| `ExcisedByLabel` | object | yes | Voxels set to 0, by old label (as a string): number of voxels |
| `CortexVoxels` | integer | yes | Voxels with the cortex label in the aseg |
| `Components` | integer | yes | Connected components of the excision map (face adjacency) |
