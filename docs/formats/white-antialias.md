# white-antialias: the antialiased field of the white surface

SpecVersion `white-antialias/1`. Written by
`python -m baillarger.white_surf DSEG --out DERIV` (`baillarger.white_surf.write_white_surface`,
through `write_field`) next to the white surface (`white-surf.md`); read by
`baillarger.white_surf.read_field`. The sidecar's keys are enforced by
`baillarger.white_surf.FIELD_SPEC`, and `tests/test_bids.py` checks that the table below
lists the same keys, types and requirements.

## Purpose

The white surface is marching tetrahedra of this field at 0. A preview of a label edit
solves the field again only near the edit (`baillarger.antialias.local_solve`, with the
radius of `docs/preview-radius.md`), starting from this field, and re-marches only the
mesh blocks that the solve changes (`baillarger.white_surf.remarch`). Thus the preview
does not solve the whole field again.

## Files

In the BIDS derivatives root `DERIV`, for a segmentation with entities `sub` and `ses`,
hemisphere `<h>` (`R` or `L`):

| File (in `sub-<sub>/ses-<ses>/anat/`) | Content |
|---|---|
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-antialias_white.nii` | The field on its box: NIfTI-1, uncompressed, float32, scale slope 1 and intercept 0 |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-antialias_white.json` | Sidecar, `white-antialias/1` |

The `.nii` is uncompressed, so that a reader memory-maps it and reads a box of it
through the chunked reader of `baillarger.blocks` (`open_volume`); it reads only the
voxels of that box. At whole-brain scale the chunked store (NIfTI-Zarr) takes the place
of the `.nii`, through the same reader.

## The field

The field φ is the antialiased field of the WM mask (the voxels of the segmentation with
one of `FilledLabels`; `baillarger.antialias.thin_plate`): φ ≥ `Eps` at each voxel of the
mask, φ ≤ −`Eps` at each other voxel, the minimum of the thin-plate energy in the band
of `Band` voxels of signed distance round the boundary, and the signed distance outside
the band (`white-surf.md`, "The surface"). Thus the sign of φ gives the mask.

The file holds φ only on a box of the segmentation's grid: the box of the mask plus
`Band` + 4 voxels on each side, clipped to the grid. The box starts at grid voxel
`BoxOrigin` and has `BoxShape` voxels. Outside the box, φ is `Fill`, −(`Band` + 3): the
clipped signed distance of a voxel far outside the mask. Voxel (i, j, k) of the file is
voxel `BoxOrigin` + (i, j, k) of the grid. The NIfTI affine is the grid's affine
(`VoxelToRAS`) moved to the box: its translation is the scanner position of
`BoxOrigin`.

`read_field` gives φ on the whole grid (`GridShape`), as a `baillarger.blocks.Source`
(`EmbeddedSource`). It refuses a file that is not a float32 array of `BoxShape`. With
`verify` (the default) it also refuses a file whose SHA-256 is not `SHA256`; this check
reads the whole file once. The march of this Source at 0, in blocks of the surface's
`BlockCells`, gives the surface's faces and edge keys.

## Sidecar keys: white-antialias/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `white-antialias/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | The corrected segmentation (absolute path) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `FilledLabels` | array of integers | yes | The labels of the WM mask |
| `VoxelToRAS` | array of numbers | yes | The segmentation's 4 × 4 affine, row-major: grid voxel to scanner RAS mm |
| `GridShape` | array of integers | yes | The segmentation's grid (voxels) |
| `BoxOrigin` | array of integers | yes | The first grid voxel of the box |
| `BoxShape` | array of integers | yes | The box (voxels); the shape of the `.nii` |
| `Fill` | number | yes | φ outside the box: −(`Band` + 3) |
| `SHA256` | string | yes | SHA-256 of the `.nii` |
| `Antialias` | object | yes | How the field was solved, as in `white-surf/1`: `Method` (`thin-plate`), `Band`, `Eps`, `Tolerance`, `Iterations`, `Restarts`, `LastChange`, `Converged` |
