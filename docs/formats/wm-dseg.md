# wm-dseg: the segmentation with a topologically correct WM mask

SpecVersion `wm-dseg/1` (the corrected segmentation) and `wm-dseg-fixes/1`
(the changes: an image, a fix table and a fix diff). Written by
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
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.nii.gz` | uint8, the changed voxels: 1 handle cut, 2 handle fill, 3 cavity fill, 4 manual edit, 5 small piece; 0 elsewhere. Each voxel shows the kind of the fix that changed it last (see The fixes). A voxel that a manual edit changes and no fix then changes shows 4 |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.tsv` | `index` and `name` of those values |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.json` | Sidecar, `wm-dseg-fixes/1` |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_fixes.tsv` | The fix table: one row per fix |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_fixes.json` | Sidecar, `wm-dseg-fixes/1` (the same content) |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_diff.npy` | The fix diff: one record per changed voxel |
| `sub-<sub>_ses-<ses>_desc-wmtopofixes_diff.json` | Sidecar, `wm-dseg-fixes/1` (the same content) |

In `sub-<sub>/ses-<ses>/wmtopofixes-history/` (next to `anat/`):

| File | Content |
|---|---|
| `<DiffSHA256>.npy`, `<DiffSHA256>.tsv`, `<DiffSHA256>.json` | A copy of the diff, the table and the sidecar of every run, by the diff's SHA-256 |

`DERIV/dataset_description.json` (`DatasetType` `derivative`) is written if
it is not there. BIDS's `desc` entity and `dseg` suffix are standard; the
`desc` values and the `fixes` and `diff` suffixes are ours. The history
directory is not BIDS: it keeps old runs, which BIDS has no place for. It is
not in `anat/`, so that readers of `anat/` see only the current run.

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

## The fixes

The change from the labels that `dseg` reads (the input with the manual
edits made: the "old" labels) to the labels that it writes (the "new"
labels) is split into fixes. Each fix has an integer id, from 1, in this
order:

1. each manual answer with rows, in the order of the answers;
2. each handle move, in the order that `remove_handles` makes them (kind
   `handle_cut` for a cut or stranded voxels, `handle_fill` for a fill);
3. each cavity (a connected piece of the cavity fill, Kuhn adjacency), in
   the order of `scipy.ndimage.label`;
4. each small piece, in the order of `scipy.ndimage.label`.

The same input gives the same ids. The steps run in another order: the
manual edits, the cavity fill, the handle moves, the small pieces. A voxel
belongs to the fix that changed it last in that order. A voxel whose new
label is its old label is in no fix (for example a cavity voxel that a
handle cut then gives back its old label).

### The fix table

`_desc-wmtopofixes_fixes.tsv`: tab-separated, a header row, one row per
fix, in id order. `n/a` marks a value that the fix does not have.

| Column | Type | Meaning |
|---|---|---|
| `id` | integer | The fix's id |
| `kind` | string | `manual_edit`, `handle_cut`, `handle_fill`, `cavity_fill` or `small_piece` (the names of the image's values) |
| `voxels` | integer | The voxels that the fix changed last. For a fix other than `manual_edit`, its rows in the diff. For `manual_edit`, the answer's voxels that no fix then changes |
| `i`, `j`, `k` | integer | The voxel of the fix that is nearest its centroid (0-based, on the grid; ties: the lowest linear index); `n/a` with no voxel |
| `x`, `y`, `z` | number | The centroid of the voxel centres, in mm (the input's affine) |
| `x_min` ... `z_max` | number | The bounding box of the voxel centres, in mm, per axis of mm |
| `wall` | integer | For a handle move: the voxels of its wall that the other front claimed (the move's voxels); `n/a` for the other kinds |
| `overrides` | integer | The `Id` of the answer whose verified voxel the fix changes; `n/a` if none |

The sum of `voxels` over the fixes of each kind is the count of that kind's
value in the image.

### The fix diff

`_desc-wmtopofixes_diff.npy`: a NumPy `.npy` file (version 1, no pickle) of
one-dimensional structured records, 16 bytes each:

| Field | Type | Meaning |
|---|---|---|
| `index` | `<i8` | The voxel's linear index on the grid (C order) |
| `fix` | `<i4` | The id of the fix that changed it last |
| `old` | `<i2` | Its old label (the input with the manual edits made) |
| `new` | `<i2` | Its label in `desc-wmtopo` (not `old`) |

The records are in Morton order of the 64³ block grid
(`baillarger.blocks.morton` of the voxel's block), and in C order of the
linear index within a block, so a reader of one block reads one run of
records. Why `.npy`: NumPy and other tools read it with no code of ours, it
can be memory-mapped, its header holds the record type, and 16 bytes a
voxel is the plan's limit (12 to 16). A `.tsv.gz` would need parsing and
could not be read by block.

`baillarger.white_dseg.replay_dseg(aseg, diff, edits)` rebuilds
`desc-wmtopo` exactly from the input, the edits and the diff, with no
correction. `baillarger.fix_diff.revert_fix(dseg, diff, id)` gives
`desc-wmtopo` with one fix undone, for inspection: only that fix's voxels
change, and the result can have a handle or a cavity.

### Every run's diff is kept

Each run copies its diff, its table and its sidecar to
`sub-<sub>/ses-<ses>/wmtopofixes-history/` (`white_dseg.history_dir`), named by the SHA-256 of the diff file.
`DiffInputs` in the sidecar gives the SHA-256 of the input labels
(`Labels`), of the edits (`Edits`, if any) and of P (`Probability`, from
phase 8), and the code (`Code`: the baillarger version and commit). Before
it writes, `dseg` reads the kept sidecars. If one has the same `DiffInputs`,
with no uncommitted change in either run (`Provenance` `Dirty` false), and
another `DiffSHA256`, `dseg` stops with an error and writes nothing: the
correction did not reproduce its diff.

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
| `FixTable` | string | yes | The file name of the fix table |
| `FixDiff` | string | yes | The file name of the fix diff |
| `DiffSHA256` | string | yes | The SHA-256 of the fix diff file |
| `DiffVoxels` | integer | yes | The records of the fix diff |
| `DiffInputs` | object | yes | `Labels`, `Edits` (if any), `Probability` (if any): SHA-256 of each input; `Code`: the baillarger version and commit |
| `Fixes` | object | yes | The number of fixes, by kind |
