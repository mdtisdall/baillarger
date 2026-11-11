# wm-excisions: the aseg with its excisions set to outside

SpecVersion `wm-excisions/1`. Written by
`python -m baillarger.excisions ASEG FLASH --echo1 E1 --echo3 E3 --hemi H --out DERIV`
(`baillarger.excisions.write_excisions`). The keys are enforced by
`baillarger.excisions.EXCISIONS_SPEC`. `tests/test_bids.py` checks that the
table below lists the same keys, types and requirements.

An excision is a block of tissue that was cut out of the specimen before the
scan. The outside medium fills it, and the FLASH is black there in each echo.
An aseg that imputes tissue (for example purple-mri's) gives tissue labels to
these voxels. This command finds these voxels and sets them to 0 (outside).
The result is the input label of `python -m baillarger.white_dseg`
(`docs/formats/wm-dseg.md`).

This is a bootstrap tool. Use it only on an aseg that imputes tissue: the
medium outside the specimen is the detector's reference for an excision. The
input aseg is not changed.

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
3, on the same grid. The command refuses an echo with a different grid shape.

The tissue labels (`TissueLabels`) are the labels of the aseg above 0, except
the ventricles and CSF (FreeSurfer 4, 5, 14, 15, 24, 43, 44, 72). The
detector checks each of them: an aseg that imputes tissue can fill an
excision with any label (on the first hemisphere: WM, cortex, caudate,
putamen and the corpus callosum).

Tissue with a short T2* (iron in the pallidum and the thalamus) is as dark as
the medium in echoes 2 and 3, but not in echo 1. The detector therefore
compares the three echoes of each voxel with a model of the medium and with
models of the tissue. It does not smooth the echoes: smoothing moves the edge
of an excision into the tissue.

1. At each voxel, x = (log E1, log FLASH, log E3). A value below 1 counts as
   1.
2. The medium (`MediumClass`): a Gaussian of x (mean and covariance) over
   the voxels with label 0 that are round(`MediumGapMM` / smallest voxel
   side) or more steps of binary dilation (face adjacency) from every voxel
   with a label above 0, and where each echo is above 0.
3. The tissue class of each tissue label (`TissueClasses`): EM on the
   label's voxels for a mixture of the medium (held fixed) and one Gaussian.
   The Gaussian is the label's tissue class.
4. The shares of each tissue label (`LabelWeights`): EM on the label's
   voxels for a mixture of the medium and every label's tissue class (all
   held fixed). A label does not tell which tissue a voxel is: iron-rich
   tissue in the thalamus label is like the pallidum's class.
5. A voxel of a tissue label is a candidate if (its label's medium share) ×
   (the medium's density at x) is larger than the sum, over the tissue
   classes, of (its label's share) × (the class's density at x). Keep the
   connected components of the candidates (face adjacency) that have
   `MinVoxels` voxels or more.

Each fit uses at most `Sample` voxels of its set, drawn with the seed `Seed`,
so the command gives the same result each time. `Ridge` is added to the
diagonal of each covariance. EM stops when no share changes by more than
10⁻⁷, or after 500 steps.

6. Drop each connected component (face adjacency) of the result of step 5
   that is smaller than `FragmentMM3` mm³. Their voxels keep their labels.
7. Air bubbles. For each connected component (face adjacency) of the result
   of step 6, take its bounding box grown by `BubbleBox` voxels. In echo e
   (1, 2, 3), the dark region V_e is the connected components (face
   adjacency) of (echo e smoothed by `Sigma` voxels) < `Frac` × (the median
   of echo e over the cortex label) that overlap the component, in the box.
   The component is an air bubble if no V_e reaches the side of the box (the
   dark region is closed), V_3 is not empty, and V_3 ≥ `BubbleRatio` × V_1.
   A gas pocket darkens a region that grows with the echo time. A cut does
   not grow. A thin piece that the smoothing lifts above the level has no
   V_3, and is no bubble. Drop each bubble from the result of step 6: its
   voxels keep their labels, because the anatomy there is tissue with a gas
   pocket. The bubble mask is V_3 of each bubble, grown by ceil(r_3 − r_1)
   steps of binary dilation (face adjacency), where r_e is the radius of a
   ball of V_e voxels: the growth of the artefact from echo 1 to echo 3.
8. Set each voxel of the result of step 7 to 0.
9. Find the connected components (face adjacency) of the tissue labels that
   step 8 leaves. Set to 0 each component that is smaller than `CrumbMM3`
   mm³ and touches the result of step 7: a voxel of the component is in it,
   or is next to it (26-adjacency). These are crumbs that an excision cut
   off. A small piece that no excision touches is not changed.

The excision map is the voxels that steps 8 and 9 set to 0. The command
changes no other voxel. The ventricles and CSF are never set to 0.

The defaults: `MinVoxels` 250, `MediumGapMM` 1.0, `Sample` 1,000,000, `Seed`
0, `Ridge` 10⁻⁶, `FragmentMM3` 0.1, `CrumbMM3` 1.0, `Frac` 0.25, `Sigma` 2.0,
`BubbleRatio` 2.0, `BubbleBox` 20.

## Sidecar keys: wm-excisions/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-excisions/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | The aseg, then echoes 2, 1 and 3 of the FLASH (absolute paths) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `TissueLabels` | array of integers | yes | The tissue labels that the detector checks and can set to 0 |
| `Detector` | object | yes | The parameters: `MinVoxels` (integer), `MediumGapMM` (number), `Sample` (integer), `Seed` (integer), `Ridge` (number), `FragmentMM3` (number), `CrumbMM3` (number), `Frac` (number), `Sigma` (number, voxels), `BubbleRatio` (number), `BubbleBox` (integer, voxels) |
| `MediumClass` | object | yes | The medium (step 2): `Voxels` (integer: the voxels of its set), `Mean` (array of 3 numbers), `Covariance` (3 × 3 array), on the log of echoes 1, 2, 3 |
| `TissueClasses` | object | yes | By tissue label (as a string): `Mean` and `Covariance` of its tissue class (step 3) |
| `LabelWeights` | object | yes | By tissue label (as a string): its shares (step 4), keyed `medium` and then by tissue label |
| `CortexMedian` | number | yes | The median of echo 2 over the cortex label (the hemisphere's) |
| `ExcisedVoxels` | integer | yes | Voxels set to 0 (steps 8 and 9) |
| `ExcisedByLabel` | object | yes | Voxels set to 0, by old label (as a string): number of voxels |
| `CortexVoxels` | integer | yes | Voxels with the cortex label in the aseg |
| `Components` | integer | yes | Connected components of the excision map (face adjacency) |
| `Fragments` | object | yes | The pieces that step 6 dropped: `Pieces` (integer), `Voxels` (integer) |
| `Crumbs` | object | yes | The tissue pieces that step 9 set to 0: `Pieces` (integer), `Voxels` (integer) |
| `Bubbles` | object | yes | The air bubbles of step 7: `Pieces` (integer), `Voxels` (integer: the voxels dropped from the excision map), `MaskVoxels` (integer: the voxels of the bubble mask) |
