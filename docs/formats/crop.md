# crop: a box cut from the whole-hemisphere derivatives

SpecVersion `crop/1`. Written by `baillarger.crop.write_crop` (and
`python -m baillarger.crop`); read by `baillarger.crop.load_crop`. The
sidecar's keys are enforced by `baillarger.crop.CROP_SPEC`, and
`tests/test_bids.py` checks that the table below lists the same keys, types
and requirements.

A crop is a small test case or prototype site: one box of voxels of the
segmentation's grid, with the files on that grid cut to the box. Tests and
prototypes use this module. They do not write their own crop code.

## The box

The box is [`BoxLo`, `BoxHi`) in voxels of the source grid (lo inclusive, hi
exclusive). Three functions give a box:

| Function | `Request.Method` | The box |
|---|---|---|
| `voxel_box(lo, hi, shape)` | `voxel-box` | As given |
| `box_at_mm(centre_mm, size_mm, affine, shape)` | `centre-mm` | The voxel nearest the centre (scanner mm), and ceil(`size_mm` / 2 / voxel size) voxels on each side of it, per axis. The count is odd: 10 mm at 0.16 mm gives 65 voxels |
| `box_at_vertex(surf, vertex, size_mm)` | `vertex` | `box_at_mm` at the white-surface vertex |

A box that is not inside the grid is an error (`ValueError`). The function
does not clip the box.

The crop's affine is the source affine moved to the box: crop voxel
(0, 0, 0) is source voxel `BoxLo`.

## Files

In the BIDS derivatives root `ROOT`, for a segmentation with entities `sub`
and `ses`, all in `ROOT/sub-<sub>/ses-<ses>/anat/`. Each output has the name
of its source.

| Output | Role in `Outputs` | Content |
|---|---|---|
| The segmentation, `.nii.gz` | `dseg` | The box of the segmentation, with the data type and header of the source |
| Its `.json` and `.tsv` | `dseg.json`, `dseg.tsv` | Copies of the source's sidecar (`wm-dseg/1`) and label table. The sidecar describes the source segmentation, not the crop |
| Each echo, `.nii.gz` | `echo1`, `echo2`, … (in the order given) | The box of the echo; its raw BIDS JSON sidecar copied (`echo1.json`, …), so `EchoTime` is kept |
| Each other volume, `.nii.gz` | `volume1`, `volume2`, … | The box of the volume; its JSON sidecar copied, if there is one |
| The white surface: the `white-surf/1` set | `white`, `white.dat`, `white.json`, `normals`, `normals.dat`, `normlen`, `normlen.dat`, `keys`, `keys.dat`, `blocks` | The white surface of the crop (below) |
| `…_hemi-<h>_desc-antialias_white.nii`, `.json` | `field`, `field.json` | The box of the antialiased field, `white-antialias/1`, with `BoxOrigin` (0, 0, 0) and `BoxShape` the crop's grid |
| The overview: the `white-overview/1` files | `overview.overview`, `overview.ids`, `overview.carry`, `overview.carry_dat`, `overview.sidecar` | An overview made again from the crop's white surface (`write_white_overview`), not cut from the source overview |
| The label edits, `.tsv` and `.json` | `edits`, `edits.json` | The entries of the diff inside the box, on the crop's grid (`label-edits/1`: `Shape` and `Affine` of the crop). The labels of the crop do not change |
| `sub-<sub>_ses-<ses>_desc-crop_box.json` | (none) | Sidecar, `crop/1` |

`ROOT/dataset_description.json` (`DatasetType` `derivative`) is written if
it is not there. The `desc-crop` and `box` names are ours.

The white surface needs the segmentation of the crop: the white-surf/1 and
white-antialias/1 sidecars give the crop's segmentation as their `Sources`,
so `WhiteSurface.check` accepts the crop's dseg.

## The white surface of a crop

The crop keeps the faces of the grid cells inside the box. Each face of the
surface comes from one cell. A face's cell is the smallest lower end of the
grid edges of its vertices, and cell + 1 is their largest upper end. So a
face is kept if and only if the grid edges of its three vertices are in the
box, from `BoxLo` to `BoxHi` − 1. These are the faces that marching tetrahedra
of the cropped field gives. Faces that cross the outer half voxel of the box
are not kept.

- The vertex coordinates, smoothed normals and normal lengths are those of the
  whole surface, bit for bit. The normals are not smoothed again near the box
  faces.
- The edge keys are on the crop's grid (`GridShape` the box's shape).
- The block index and the order follow `white-surf/1` on the crop's grid:
  blocks of `BlockCells` from crop voxel (0, 0, 0), in Morton order; the
  faces block by block; the vertices by owner block, then edge key.
- `Pieces` counts the pieces of the crop. They have boundary loops at the
  box faces.
- `NormalSmoothingSteps`, `Antialias` and `Device` are the whole surface's.

A crop reader that needs the vertex numbers of the whole surface gets them
from `crop_white` (`WhiteCrop.source_ids`). The files do not keep them: the
edge keys of the whole surface are the edge keys of the crop moved by
`BoxLo`.

## Sidecar keys: crop/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `crop/1` |
| `Description` | string | yes | What the files are |
| `Sources` | array of strings | yes | The source files: the segmentation, the volumes, the white surface, the label edits (absolute paths) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `BoxLo` | array of integers | yes | The first voxel of the box, in the source grid |
| `BoxHi` | array of integers | yes | One past the last voxel of the box |
| `SourceGridShape` | array of integers | yes | The source grid (voxels) |
| `SourceVoxelToRAS` | array of numbers | yes | The source grid's 4 × 4 affine, row-major |
| `GridShape` | array of integers | yes | The crop's grid: `BoxHi` − `BoxLo` |
| `VoxelToRAS` | array of numbers | yes | The crop's 4 × 4 affine, row-major |
| `Request` | object | yes | How the box was named: `Method` (`voxel-box`, `centre-mm` or `vertex`) and its values (`Lo`, `Hi`; `CentreMM`, `SizeMM`; `Vertex`) |
| `Outputs` | object | yes | The role of each output (Files, above), by name |
| `OutputSources` | object | yes | The source of each output (absolute path), by name |
| `OutputsSHA256` | object | yes | SHA-256 of each output, by name |
