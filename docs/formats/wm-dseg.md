# wm-dseg: the segmentation with a topologically correct WM mask

SpecVersion `wm-dseg/1` (the corrected segmentation) and `wm-dseg-fixes/1`
(the diagnostic image of the changes). Written by
`python -m baillarger.white dseg ASEG [--edits EDITS] [--no-gm-fills] [--thickness-mm R] [--gap-mm G] --out DERIV`
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
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.nii.gz` | uint8, the changed voxels: 1 handle cut, 2 handle fill, 3 cavity fill, 4 manual edit, 5 opening (removed: thin, or next to a narrow gap), 6 exception (kept, though thin or next to a narrow gap); 0 elsewhere. A manually edited voxel that the correction then changes shows the correction's value |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.tsv` | `index` and `name` of those values |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.json` | Sidecar, `wm-dseg-fixes/1` |
| `sub-<sub>_ses-<ses>_desc-wmtopo_exceptions.tsv` | The exceptions (below): `index`, `kind`, `voxels`, and the centre `i`, `j`, `k` (0-based voxel); only the header without the invariants |

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

## Invariants

Beyond the topology, three invariants can be asked for (off by default):

- `--no-gm-fills`: no handle fill takes a cortex voxel (label 42 right, 3
  left; hippocampus, amygdala and the rest are not cortex). A handle whose
  fill would is cut. Where WM meets cortex, the mask then only moves
  inward; elsewhere it may move either way. A cavity of the mask that holds
  cortex is a serious error that a person resolves: `dseg` refuses, listing
  each such cavity's size and centre, to be fixed by a manual edit
  (`--edits`) and a rerun.
- `--thickness-mm R`: every voxel of the mask lies in a ball of radius R
  (mm) inside the mask.
- `--gap-mm G`: no mask voxel is within G of a voxel outside the mask that
  no ball of radius G outside the mask holds (a gap narrower than about 2G
  between WM surfaces).

The last two are met by removing voxels (`baillarger.handles.open_thin`): the
voxels that break them, in connected pieces, each removed whole when that
keeps the topology, in passes until one removes nothing. A piece that
cannot be removed whole (a thin bridge between two thick parts) is kept as
it is, not thinned, and listed as an **exception** in the exceptions table.
Removed voxels take labels as cuts do. R has a floor of about 1.5 voxels:
below it the voxel staircase itself breaks the invariant.

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
| `Invariants` | object | yes | `NoGMFills` (boolean), `ThicknessMM`, `GapMM` (numbers, 0: off) |
| `OpeningVoxels` | integer | yes | Voxels removed for the thickness and the gap |
| `OpeningPasses` | integer | yes | Passes the opening took (0 when off) |
| `Exceptions` | integer | yes | Pieces kept though they break the thickness or the gap |
| `ExceptionsFile` | string | yes | The name of the exceptions table, in the same directory |
| `Moves` | array of objects | yes | Per manual answer with rows, then per handle move: `Kind` (`manual`, `cut`, `fill` or `stranded`), `Voxels` (count), `Centre` (0-based voxel of the grid), and for `manual` the `Answer` (its `Id`) |

## Sidecar keys: wm-dseg-fixes/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-dseg-fixes/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | As in `wm-dseg/1` |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | As in `wm-dseg/1` |
