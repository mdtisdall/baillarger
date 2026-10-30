# white-overview: a white surface decimated for viewing

SpecVersion `white-overview/1`. Written by
`python -m baillarger.white mesh DSEG --out DERIV [--overview-eps MM] [--workers N]`,
right after the white surface (`white-surf/1`), or for a surface already written by
`python -m baillarger.white overview SURF [--eps MM] [--workers N]`
(`baillarger.white.write_white_overview`); read by
`baillarger.white.WhiteOverview.load`. The sidecar's keys are enforced by
`baillarger.white.OVERVIEW_SPEC`, and `tests/test_bids.py` checks that the
table below lists the same keys, types and requirements.

The overview is for display and navigation only (the viewer's surface panes,
its inflation, picking). Nothing measured is computed on it: every pick maps
to a true vertex, and the true surface is read from its own files.

## Files

Next to the white surface, in `DERIV/sub-<sub>/ses-<ses>/anat/`, hemisphere
`<h>`:

| File | Content |
|---|---|
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-overview_white.surf.gii` | GIfTI: the vertices and the triangles, embedded |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-overviewids_white.func.gii` | GIfTI: the true vertex of each overview vertex, embedded |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-overviewcarry_white.func.gii` | GIfTI XML: the true vertices that each face carries, as external binary data |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-overviewcarry_white.dat` | Their data |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-overview_white.json` | Sidecar, `white-overview/1` |

The overview and ids files are ordinary GIfTI (`Encoding` `GZipBase64Binary`,
no external data), so any surface reader opens them (nibabel, Connectome
Workbench, FreeSurfer, NiiVue). They are small: the overview's size is set by
the surface's shape and `EpsMM`, not by the voxel size (about 100 k vertices
for a hemisphere at 0.1 mm). The carry file has one entry per true vertex
(about 50 MB for a hemisphere), so its data is external and raw, like the
white surface's (`white-surf.md`, *The GIfTI files*), and a reader takes only
the faces it needs; its arrays are not per-vertex values, so surface viewers
do not show it.

| File | Data array | Type | Shape | Intent |
|---|---|---|---|---|
| `_desc-overview_white.surf.gii` | 1: vertices, scanner RAS mm | float32 | n × 3 | `NIFTI_INTENT_POINTSET` |
| `_desc-overview_white.surf.gii` | 2: triangles, indices into the vertices | int32 | m × 3 | `NIFTI_INTENT_TRIANGLE` |
| `_desc-overviewids_white.func.gii` | 1: the true vertex (an index into the white surface's vertices) | int32 | n | `NIFTI_INTENT_NONE` |
| `_desc-overviewcarry_white.func.gii` | 1: pointers: face k's carried vertices are entries `ptr[k]` to `ptr[k + 1]` of array 2 | int32 | (m + 1) × 1 | `NIFTI_INTENT_NONE` |
| `_desc-overviewcarry_white.func.gii` | 2: the carried true vertices (indices into the white surface's vertices), face by face | int32 | c × 1 | `NIFTI_INTENT_NONE` |

Each file's metadata has `AnatomicalStructurePrimary` (`CortexRight` or
`CortexLeft`).

## The overview

The white surface decimated by `baillarger.decimate` (half-edge collapses,
cheapest first by quadric error, in parallel by blocks of `BlockMM`, in two
passes on grids shifted by half a block). Its guarantees:

- the topology is the white surface's: its pieces, each closed and genus 0,
  and the orientation of the faces (out of the WM);
- every vertex of the white surface is within `EpsMM` of the overview (world
  mm); `MaxDistanceMM` is the largest distance measured, from each removed
  vertex to the overview face that stands for it;
- no two faces intersect: faces that would cross where two sheets of the white
  surface come within about 2 `EpsMM` of each other are found and kept at
  the white surface's resolution there (`Repairs`);
- every face faces the way of the surface it stands for: its normal is at
  less than 90 degrees to the sum of the white surface's normals (each
  vertex's faces' cross products, summed) at its corners and at the true
  vertices it stands for;
- the overview's vertices are vertices of the white surface, unmoved: vertex
  k is true vertex `ids[k]`, and its coordinates are that vertex's float32
  coordinates in the white surface's file, bit for bit. The vertices are in
  increasing order of their true vertex.

## Order

- Vertices: in increasing order of their true vertex.
- Faces: by their three true vertices, smallest first, then the middle one,
  then the largest (each face keeps its own corner order, so its orientation).
  True vertices are numbered in the white surface's Morton block order, so
  faces that are near each other are near each other in the files, and so
  are their carried vertices.
- Carried vertices: face by face in that order; within a face, in increasing
  order.

## Carried vertices

Each true vertex that is not an overview vertex is carried by exactly one
overview face: the face that stands for it in the decimation (each vertex
removed by a collapse goes to the nearest of the faces that the collapse
makes, which lie round it on the surface, so a vertex is not carried by a face
across a sheet the way a nearest-face search in space could do). Overview
vertices are carried by none; `desc-overviewids` gives them. `Carried` is
`TrueVertices` − `Vertices`.

A carried vertex's place on its face is not stored: it is the closest point of
the face to the vertex, computed on reading from the white surface's vertices
as its file holds them (float32 scanner mm, as the decimation used them), with
its barycentric coordinates and distance (`WhiteOverview.carry`;
`baillarger.decimate.closest_points`). The largest of those distances is
`MaxDistanceMM`, exactly.

To read a patch: its faces' entries of the pointers (array 1, small, read
whole) give the ranges of array 2 to read (byte `offset + 4 × ptr[k]` of the
`.dat`, `ExternalFileOffset` in the XML).

## Identity

The ids are indices into the white surface it was made from, which
`SourcesSHA256` records (the `.surf.gii` and its `.dat`). The overview is
written with the surface, so a remeshed surface comes with its own overview.
A grid-edge key, which survives a remesh, is one lookup away: the white
surface's `desc-edgekeys` at `ids`.

## Sidecar keys: white-overview/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `white-overview/1` |
| `Description` | string | yes | What the files are |
| `Sources` | array of strings | yes | The white surface's `.surf.gii` and `.dat` (absolute paths) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `VoxelToRAS` | array of numbers | yes | The grid's voxel-to-scanner-mm affine, 4 × 4, row-major (the white surface's) |
| `EpsMM` | number | yes | The distance bound, mm |
| `BlockMM` | number | yes | The decimation's block side, mm |
| `MaxDistanceMM` | number | yes | The largest distance measured, mm (at most `EpsMM`) |
| `Vertices` | integer | yes | Vertices of the overview |
| `Faces` | integer | yes | Faces of the overview |
| `TrueVertices` | integer | yes | Vertices of the white surface |
| `TrueFaces` | integer | yes | Faces of the white surface |
| `Pieces` | array of objects | yes | Per connected piece, largest first: `V`, `F`, `boundary_loops`, `chi`, `genus`, `area_mm2` |
| `IdsFile` | string | yes | The name of the `desc-overviewids` file |
| `Repairs` | array of objects | yes | Per repair round: `round`, `intersecting_faces`, `pairs`, `pinned` (true vertices kept so far); the last has no intersecting faces |
| `CarryFile` | string | yes | The name of the `desc-overviewcarry` file |
| `Carried` | integer | yes | True vertices carried by faces (`TrueVertices` − `Vertices`) |
| `DataSHA256` | object | yes | SHA-256 of the carry file's `.dat`, by name |
| `Passes` | array of objects | yes | Per decimation pass: `blocks`, `faces_before`, `faces_after`, `collapses`, `locked`, `seconds` |
