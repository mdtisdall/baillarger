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
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-overview_white.json` | Sidecar, `white-overview/1` |

Both GIfTI files are ordinary ones (`Encoding` `GZipBase64Binary`, no external
data), so any surface reader opens them (nibabel, Connectome Workbench,
FreeSurfer, NiiVue). They are small: the overview's size is set by the surface's
shape and `EpsMM`, not by the voxel size (about 200 k vertices for a hemisphere
at 0.1 mm).

| File | Data array | Type | Shape | Intent |
|---|---|---|---|---|
| `_desc-overview_white.surf.gii` | 1: vertices, scanner RAS mm | float32 | n × 3 | `NIFTI_INTENT_POINTSET` |
| `_desc-overview_white.surf.gii` | 2: triangles, indices into the vertices | int32 | m × 3 | `NIFTI_INTENT_TRIANGLE` |
| `_desc-overviewids_white.func.gii` | 1: the true vertex (an index into the white surface's vertices) | int32 | n | `NIFTI_INTENT_NONE` |

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
- every face faces the way of the surface it stands for: its normal is at
  less than 90 degrees to the sum of the white surface's normals (each
  vertex's faces' cross products, summed) at its corners and at the true
  vertices it stands for;
- the overview's vertices are vertices of the white surface, unmoved: vertex
  k is true vertex `ids[k]`, and its coordinates are that vertex's float32
  coordinates in the white surface's file, bit for bit. The vertices are in
  increasing order of their true vertex.

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
| `Passes` | array of objects | yes | Per decimation pass: `blocks`, `faces_before`, `faces_after`, `collapses`, `locked`, `seconds` |
