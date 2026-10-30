# white-surf: the white surface of a hemisphere

SpecVersion `white-surf/1`. Written by
`python -m baillarger.white mesh DSEG --out DERIV [--block-cells B] [--device D]`
(`baillarger.white.write_white_surface`) from a corrected segmentation
(`wm-dseg/1`); read by `baillarger.white.WhiteSurface.load`. The sidecar's
keys are enforced by `baillarger.white.SURF_SPEC`, and `tests/test_bids.py`
checks that the table below lists the same keys, types and requirements.

## Files

In the BIDS derivatives root `DERIV`, for a segmentation with entities `sub`
and `ses`, hemisphere `<h>` (`R` or `L`):

| File (in `sub-<sub>/ses-<ses>/anat/`) | Content |
|---|---|
| `sub-<sub>_ses-<ses>_hemi-<h>_white.surf.gii` | GIfTI XML: the vertices and the triangles, as external binary data |
| `sub-<sub>_ses-<ses>_hemi-<h>_white.dat` | Their data: the vertices, then the triangles |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-normals_white.func.gii` | GIfTI XML: the normals, as external binary data |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-normals_white.dat` | Their data |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-edgekeys_white.func.gii` | GIfTI XML: each vertex's grid-edge key, as external binary data |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-edgekeys_white.dat` | Their data |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-blocks_white.tsv` | The block index |
| `sub-<sub>_ses-<ses>_hemi-<h>_white.json` | Sidecar, `white-surf/1` |

`DERIV/dataset_description.json` (`DatasetType` `derivative`) is written if
it is not there. BIDS has no settled rule for surface derivatives; the names
follow its entity order, and the `.dat` and `desc-blocks` files are ours.

## The surface

The surface of the corrected segmentation's WM mask (its voxels with one of
`FilledLabels`); the command refuses a mask whose Betti numbers are not
(b0, 0, 0), so each piece is a closed genus-0 surface.

It is marching tetrahedra, on the Kuhn tetrahedralization of the grid, of the
mask's antialiased field φ at 0 (`baillarger.antialias`): φ ≥ `Eps` at every
voxel of the mask and φ ≤ −`Eps` at every other voxel, and otherwise as
smooth as it can be, the minimum of the thin-plate energy (the sum of its
squared second derivatives) in a band of `Band` voxels of signed distance
round the boundary (Lempitsky, CVPR 2010; Whitaker, 2000). So:

- the triangles, their orientation and the edge keys are the binary mask's
  (they follow from the voxels' signs alone), and so is the topology;
- each vertex lies on its own grid edge, between an inside and an outside
  voxel centre, where the true boundary also crosses: the staircase of the
  binary mask is smoothed without moving the surface past the data;
- the mesh has no self-intersections (one piecewise-linear level set; sheets
  keep at least about 2 `Eps` apart, far above float32 rounding);
- the field is the unique solution of a convex problem, so devices agree to
  the solver's tolerance (`Tolerance`: the mean change per iteration at the
  voxels next to the boundary; vertices within about 1e-3 voxel of the fully
  converged field).

Triangles are oriented so their normals point out of the WM. The normals are
the area-weighted vertex normals smoothed along the mesh (10 steps,
`baillarger.mesh_smoothing.smooth_normals`), out of the WM.

The surface has no per-vertex cortex label: every vertex is a column root.
Columns run out from the WM until they reach background (label 0) or a
maximum length; what they pass through (cortex, hippocampus, fluid
mislabelled as cortex) is left to later stages.

## The GIfTI files

Each data array has `Encoding="ExternalFileBinary"`, `Endian="LittleEndian"`,
`ArrayIndexingOrder="RowMajorOrder"`, and `ExternalFileName` the `.dat`'s
name alone (relative to the `.gii`), with its `ExternalFileOffset` in bytes.

| File | Data array | Type | Shape | Intent |
|---|---|---|---|---|
| `_white.surf.gii` | 1: vertices, scanner RAS mm | float32 | n × 3 | `NIFTI_INTENT_POINTSET` |
| `_white.surf.gii` | 2: triangles, indices into the vertices | int32 | m × 3 | `NIFTI_INTENT_TRIANGLE` |
| `_desc-normals_white.func.gii` | 1: unit normals, scanner RAS | float32 | n × 3 | `NIFTI_INTENT_VECTOR` |
| `_desc-edgekeys_white.func.gii` | 1: edge keys, each an int64 stored as its (low, high) int32 words (GIfTI has no 64-bit integers): read the array as little-endian int64 | int32 | n × 2 | `NIFTI_INTENT_NONE` |

The point set's `CoordinateSystemTransformMatrix` is the identity from
`NIFTI_XFORM_SCANNER_ANAT` to `NIFTI_XFORM_SCANNER_ANAT`; its metadata are
`AnatomicalStructurePrimary` (`CortexRight` or `CortexLeft`),
`AnatomicalStructureSecondary` (`GrayWhite`) and `GeometricType`
(`Anatomical`). The normals are vectors of the scanner space as computed in
voxel coordinates: for a non-isotropic grid, map them with the affine's
inverse transpose if a covector is needed.

Readers: nibabel (memory-mapped), Connectome Workbench, gifticlib and so
FreeSurfer and AFNI/SUMA, the MATLAB `gifti` toolbox and R's `gifti` read
external data; NiiVue does not. gifticlib opens the relative
`ExternalFileName` from the working directory, so run gifticlib-based tools
in the files' directory.

## Order and the block index

The grid's cells are divided into blocks of `BlockCells` cells a side,
starting at voxel (0, 0, 0). A block's index (`block_i`, `block_j`,
`block_k`) is its first cell / `BlockCells`. Blocks are in Morton (Z)
order: by the code that interleaves the bits of their indices, at each
level i's bit, then j's, then k's (bit b of i at 3b + 2, of j at 3b + 1,
of k at 3b). Neighbouring blocks are near each other in the files, and any
aligned cube of 2^m blocks a side is a run of consecutive blocks, so its
faces and owned vertices are each one contiguous range. The faces are
stored block by block: each face comes from one cell, so from one block. A
vertex lies on a grid edge; it is owned by the first block (in that order)
whose faces use it, and the vertices are stored by owner, then by their
grid edge. A block's faces use its own vertices and some of earlier
neighbouring blocks.

A vertex's edge key names the grid edge it lies on. Every Kuhn edge runs
from a grid point p to p + s, with s one of the 7 nonzero vectors of
{0, 1}^3; with lin(p) the linear index of p (C order, on a grid of
`GridShape`), the key is lin(p) × 7 + d, d = 4 s_i + 2 s_j + s_k − 1
(`baillarger.mesh.edge_key`; `edge_points` decodes it). The largest key is
about 7 × the number of grid points, so int64 holds it for any grid, and
keys sort by lower grid point, then upper. Vertex numbers depend on the
block order, so making the mesh again after a local change renumbers
vertices in later blocks; the key does not change where the surface did
not, so it is the
identity for anything stored by vertex across remeshes (column roots,
cached per-vertex values, edits).

The block index has a header row and one row per block that the surface
crosses (other blocks have no faces):

| Column | Type | Meaning |
|---|---|---|
| `block_i`, `block_j`, `block_k` | integer | The block's index |
| `vertex_start`, `vertex_stop` | integer | The vertices it owns: [start, stop) |
| `face_start`, `face_stop` | integer | Its faces: [start, stop) |
| `min_i`, `min_j`, `min_k`, `max_i`, `max_j`, `max_k` | number | The bounding box of the vertices its faces use, in voxels of the grid, as a reader gets them from the file (mm through the inverse of `VoxelToRAS` in float64, rounded to float32); 3 decimals, minima rounded down and maxima up, so the box contains them |

To read one block: its faces are rows `face_start` to `face_stop` of the
triangle array; the vertices they name are rows of the vertex array, and
the same rows of the normals and edge keys. The ranges of all blocks are
contiguous and cover the arrays. Readers, the viewer among them, read
blocks through `WhiteSurface.load` (`block=(i, j, k)`), which re-indexes
the faces and gathers the boundary vertices that neighbouring blocks own,
so that the file layout stays inside it.

## Sidecar keys: white-surf/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `white-surf/1` |
| `Description` | string | yes | What the files are |
| `Sources` | array of strings | yes | The corrected segmentation (absolute path) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `FilledLabels` | array of integers | yes | The labels of the WM mask |
| `BlockCells` | integer | yes | Block size, cells a side |
| `VoxelToRAS` | array of numbers | yes | The segmentation's 4 × 4 affine, row-major: voxel to scanner RAS mm |
| `Vertices` | integer | yes | n |
| `Faces` | integer | yes | m |
| `Blocks` | integer | yes | Rows of the block index |
| `Pieces` | array of objects | yes | Per connected piece, largest first: `V`, `F`, `boundary_loops`, `chi`, `genus`, `area_mm2` |
| `NormalsFile` | string | yes | Name of the normals' `.func.gii` |
| `BlocksFile` | string | yes | Name of the block index |
| `EdgeKeysFile` | string | yes | Name of the edge keys' `.func.gii` |
| `GridShape` | array of integers | yes | The segmentation's grid (voxels), which defines the edge keys |
| `DataSHA256` | object | yes | SHA-256 of each `.dat`, by name |
| `Antialias` | object | yes | How the field was solved: `Method` (`thin-plate`), `Band`, `Eps`, `Tolerance`, `Iterations`, `Restarts`, `LastChange`, `Converged` |
| `Device` | string | yes | Where the mesh steps ran (`baillarger.backend`): `numpy` (float64-capable reference, here in float32), `cpu`, `cuda` or `mps` (PyTorch, float32) |
