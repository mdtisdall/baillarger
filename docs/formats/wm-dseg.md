# wm-dseg: the segmentation with a topologically correct WM mask

SpecVersion `wm-dseg/1` (the corrected segmentation) and `wm-dseg-fixes/1`
(the diagnostic image of the changes). Written by
`python -m baillarger.white dseg ASEG [--edits EDITS] [--scale-mm S] [--widen-gaps] [--well-composed] --out DERIV`
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
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.nii.gz` | uint8, the changed voxels: 1 handle cut, 2 handle fill, 3 cavity fill, 4 manual edit, 5 thickened, 6 exception (a target voxel the thickening could not add, or a voxel of a diagonal-only contact left), 7 added and 8 removed for diagonal-only contacts, 9 removed to widen a gap; 0 elsewhere. A manually edited voxel that the correction then changes shows the correction's value |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.tsv` | `index` and `name` of those values |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.json` | Sidecar, `wm-dseg-fixes/1` |
| `sub-<sub>_ses-<ses>_desc-wmtopo_exceptions.tsv` | The exceptions (below): `index`, `kind`, `voxels`, and the centre `i`, `j`, `k` (0-based voxel); only the header without `--scale-mm`, `--widen-gaps` or `--well-composed` |

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
   reach the border of the mask's bounding box plus a margin) are filled. A
   cavity that holds cortex (label 42 right, 3 left) is an error in the
   input that a person resolves: `dseg` refuses, listing each such cavity's
   size and centre, to be fixed by a manual edit (`--edits`) and a rerun.
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

## Thickening

With `--scale-mm S` (s, in mm; off by default), the corrected mask is then
thickened where it is thinner than about 2s, so that its normals are defined
at scale s (`baillarger.handles.thicken`). A mask voxel is thin when no ball
of radius s inside the mask holds it. The target is the mask with a ball of
radius s round each thin voxel, closed by radius s near those balls (a
fillet where they meet the mask). The mask grows into the target by simple
points only, so its Betti numbers stay, in passes until one adds nothing.
Added voxels get `WMLabel`. A target voxel that cannot be added (adding it
would join two parts of the mask or close a handle: a gap narrower than the
target) is left out; each connected piece of those is an **exception**
(`kind` `thicken`) in the exceptions table. s has a floor of about 1.5
voxels: below it the voxel staircase itself is thin. Growth into cortex is
allowed: the white surface is where columns start, not the GM/WM boundary.

## Widened gaps

With `--widen-gaps` and `--scale-mm S` (off by default), the mask's gaps
between two surfaces are then widened to about 2s
(`baillarger.handles.widen_gaps`). A gap voxel is a background voxel that
no ball of radius s in the background holds and whose addition to the mask
is not a simple point: it touches two surfaces at once. So gaps one voxel
wide are found (after thickening, the target voxels it could not add), and
slots open at one end too, whose walls face each other as a gap's do; wider
gaps and pits are not. The mask voxels within s of a gap voxel are removed
by simple points, in passes; removed voxels take labels as cuts do. Each
piece of those that could not be removed is an exception (`kind` `gap`).

## Well-composed

With `--well-composed` (off by default), the mask's diagonal-only contacts
are then removed, as FreeSurfer's `mri_pretess` does
(`baillarger.handles.well_compose`). A contact is a 2 x 2 square of voxels
in a checkerboard, or a 2 x 2 x 2 cube whose two voxels of one phase touch
only at a corner: either way the surface has a pinch of zero width there,
whichever phase the Kuhn adjacency joins. Only voxels of a contact change,
by simple points (the topology stays), each at most once. Additions and
removals that leave no contact at the voxel come first; when none is left,
additions that leave no more contacts at the voxel than before (a contact
through a cube's Kuhn diagonal takes two additions in a row). Added voxels
get `WMLabel`, removed ones take labels as cuts do. Each piece of the
contacts left is an exception (`kind` `pinch`).

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
| `ScaleMM` | number | yes | s of `--scale-mm` (0: no thickening) |
| `ThickenedVoxels` | integer | yes | Voxels set to `WMLabel` by the thickening |
| `ThickenPasses` | integer | yes | Passes the thickening took (0 when off) |
| `WidenGaps` | boolean | yes | Whether `--widen-gaps` was given |
| `GapVoxels` | integer | yes | Background voxels in gaps between two surfaces |
| `GapWidenedVoxels` | integer | yes | Voxels removed from the mask to widen them |
| `WellComposed` | boolean | yes | Whether `--well-composed` was given |
| `WellComposedAdded` | integer | yes | Voxels set to `WMLabel` to remove diagonal-only contacts |
| `WellComposedRemoved` | integer | yes | Voxels removed from the mask to remove them |
| `Exceptions` | integer | yes | Pieces left: of the target the thickening could not add (`thicken`), of the target the widening could not remove (`gap`), and of the contacts left (`pinch`) |
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
