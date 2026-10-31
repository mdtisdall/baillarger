# wm-dseg: the segmentation with a topologically correct WM mask

SpecVersion `wm-dseg/1` (the corrected segmentation) and `wm-dseg-fixes/1`
(the diagnostic image of the changes). Written by
`python -m baillarger.white dseg ASEG [--edits EDITS] [--fill-cost W] [--thicken-cuts R] [--thicken-fills R] [--trim-threads R] --out DERIV`
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
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.nii.gz` | uint8, the changed voxels: 1 handle cut, 2 handle fill, 3 cavity fill, 4 manual edit, 5 cut widening, 6 fill thickening, 7 thread trim; 0 elsewhere. A manually edited voxel that the correction then changes shows the correction's value |
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

The hippocampus (53 / 17) is not filled: it is archicortex, whose laminar
axis does not run out from the WM below it.

1. Cavities of the mask (background components, Kuhn adjacency, that do not
   reach the border of the mask's bounding box plus 2 voxels) are filled.
2. Handles are removed (`baillarger.handles.remove_handles`): each is cut or
   filled, whichever changes fewer voxels.
3. Cavity and handle fills get the WM label. Each cut voxel gets the label
   of the nearest voxel (Euclidean, mm) that is outside the corrected mask
   and not cut.

Guarantee: the corrected segmentation's voxels with a filled label are the
corrected mask exactly, and its Betti numbers (Kuhn adjacency) are
(b0, 0, 0). Voxels outside the mask's bounding box are unchanged. Each piece
of the mask is made handle- and cavity-free by construction, with the other
pieces as background; a fill of one piece can touch another and make a
handle or a cavity of the union, so the command checks the Betti numbers and
writes nothing if b1 or b2 is not 0.

## Variations of the fixes

For testing their effect, four options vary the correction; by default none
is used and the correction is as above. Each is recorded in
`TopologyOptions`, and each keeps the Betti numbers: they change voxels one
simple point at a time, or a whole piece whose removal keeps the homology.

| Option | Key | Default | What it does |
|---|---|---|---|
| `--fill-cost W` | `FillCost` | 1 | A handle fill's voxels count W times when the handle removal chooses between a cut and a fill: W > 1 favours cuts (the surface moves inward), W < 1 fills |
| `--thicken-cuts R` | `ThickenCuts` | 0 | After the handle removal, mask voxels within R voxels (Euclidean) of a cut are removed, nearest first, where simple: the cut's gap widened (inward) |
| `--thicken-fills R` | `ThickenFills` | 0 | Background voxels within R voxels of a fill are added, nearest first, where simple: the fill's plug thickened (outward, into what was outside the mask) |
| `--trim-threads R` | `TrimThreads` | 0 | Then the mask's voxels outside its opening by a ball of radius R voxels (features thinner than about 2R + 1 voxels), taken in connected pieces, are removed when a piece hangs from the rest by one attachment; a thin piece joining two thick parts is kept whole, and a dangling thread keeps a stub at most R voxels long where the opening reaches past the surface |

Widened, trimmed voxels are cuts and thickened voxels fills for the labels:
they take the nearest outside label, and the WM label.

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
| `ManualVoxels` | integer | yes | Voxels changed by the manual edits (0 without `--edits`) |
| `TopologyOptions` | object | yes | The variations used: `FillCost`, `ThickenCuts`, `ThickenFills`, `TrimThreads` (numbers; defaults 1, 0, 0, 0) |
| `CutWideningVoxels` | integer | yes | Voxels cut from the mask by `ThickenCuts` |
| `FillThickeningVoxels` | integer | yes | Voxels added to the mask by `ThickenFills` |
| `ThreadTrimVoxels` | integer | yes | Voxels cut from the mask by `TrimThreads` |
| `Moves` | array of objects | yes | Per manual answer with rows, then per handle move: `Kind` (`manual`, `cut`, `fill` or `stranded`), `Voxels` (count), `Centre` (0-based voxel of the grid), and for `manual` the `Answer` (its `Id`) |

## Sidecar keys: wm-dseg-fixes/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-dseg-fixes/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | As in `wm-dseg/1` |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | As in `wm-dseg/1` |
