# white-surf: the white surface of a hemisphere

SpecVersion `white-surf/1`. Written by
`python -m baillarger.white mesh DSEG --out DERIV [--sigma S] [--block-cells B]`
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
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-blocks_white.tsv` | The block index |
| `sub-<sub>_ses-<ses>_hemi-<h>_white.json` | Sidecar, `white-surf/1` |

`DERIV/dataset_description.json` (`DatasetType` `derivative`) is written if
it is not there. BIDS has no settled rule for surface derivatives; the names
follow its entity order, and the `.dat` and `desc-blocks` files are ours.

## The surface

The surface of the corrected segmentation's WM mask (its voxels with one of
`FilledLabels`): marching tetrahedra at 0.5 of the binary mask on the Kuhn
tetrahedralization of the grid, so the mesh has the mask's topology; the
command refuses a mask whose Betti numbers are not (b0, 0, 0), so each piece
is a closed genus-0 surface. Then tangential relaxation and Gaussian
smoothing of `Sigma` voxels without shrinkage (`baillarger.mesh_smoothing`),
which move the vertices and keep the faces. The mesh before smoothing has no
self-intersections (it is the level set of one piecewise-linear function);
smoothing can push close sheets (thin WM blades, tight folds) through each
other, so `untangle` then blends the vertices round any intersecting faces
back toward their positions before smoothing (halving their blend, to 0 at
1/8, widening the region if needed) until no faces intersect: the written
mesh has no self-intersections (`baillarger.mesh_intersect`, exact).
Triangles are oriented so their normals point out of the WM. The normals are the smoothed unit vertex
normals, out of the WM.

## The GIfTI files

Each data array has `Encoding="ExternalFileBinary"`, `Endian="LittleEndian"`,
`ArrayIndexingOrder="RowMajorOrder"`, and `ExternalFileName` the `.dat`'s
name alone (relative to the `.gii`), with its `ExternalFileOffset` in bytes.

| File | Data array | Type | Shape | Intent |
|---|---|---|---|---|
| `_white.surf.gii` | 1: vertices, scanner RAS mm | float32 | n × 3 | `NIFTI_INTENT_POINTSET` |
| `_white.surf.gii` | 2: triangles, indices into the vertices | int32 | m × 3 | `NIFTI_INTENT_TRIANGLE` |
| `_desc-normals_white.func.gii` | 1: unit normals, scanner RAS | float32 | n × 3 | `NIFTI_INTENT_VECTOR` |

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
starting at voxel (0, 0, 0). Blocks are in C order of their indices
(`block_i`, `block_j`, `block_k`) = first cell / `BlockCells`. The faces are
stored block by block: each face comes from one cell, so from one block. A
vertex lies on a grid edge; it is owned by the first block (in that order)
whose faces use it, and the vertices are stored by owner, then by their
grid edge. A block's faces use its own vertices and some of earlier
neighbouring blocks.

The block index has a header row and one row per block that the surface
crosses (other blocks have no faces):

| Column | Type | Meaning |
|---|---|---|
| `block_i`, `block_j`, `block_k` | integer | The block's index |
| `vertex_start`, `vertex_stop` | integer | The vertices it owns: [start, stop) |
| `face_start`, `face_stop` | integer | Its faces: [start, stop) |
| `min_i`, `min_j`, `min_k`, `max_i`, `max_j`, `max_k` | number | The bounding box of the vertices its faces use, in voxels of the grid, as a reader gets them from the file (mm through the inverse of `VoxelToRAS` in float64, rounded to float32); 3 decimals, minima rounded down and maxima up, so the box contains them |

To read one block: its faces are rows `face_start` to `face_stop` of the
triangle array (bytes from the triangle array's offset plus
12 × `face_start`); the vertices they name are rows of the vertex array
(bytes 12 × index from its offset). The ranges of all blocks are contiguous
and cover the arrays.

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
| `Sigma` | number | yes | The smoothing, voxels |
| `BlockCells` | integer | yes | Block size, cells a side |
| `VoxelToRAS` | array of numbers | yes | The segmentation's 4 × 4 affine, row-major: voxel to scanner RAS mm |
| `Vertices` | integer | yes | n |
| `Faces` | integer | yes | m |
| `Blocks` | integer | yes | Rows of the block index |
| `Pieces` | array of objects | yes | Per connected piece, largest first: `V`, `F`, `boundary_loops`, `chi`, `genus`, `area_mm2` |
| `NormalsFile` | string | yes | Name of the normals' `.func.gii` |
| `BlocksFile` | string | yes | Name of the block index |
| `DataSHA256` | object | yes | SHA-256 of each `.dat`, by name |
| `SelfIntersections` | object | yes | Summary of `untangle`: `FacesBefore` (intersecting faces after smoothing), `VerticesMovedBack` (vertices left short of their smoothed position), `SmallestBlend` (0: some vertices are at their position before smoothing). The places are printed by the command, not stored |
