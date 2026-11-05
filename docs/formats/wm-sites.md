# wm-sites: where the white surface folds tightest

SpecVersion `wm-sites/1`. Written by `python -m baillarger.wm_sites SURF [--out DIR]`
(`baillarger.wm_sites.write_sites`) from a white surface (`white-surf/1`), its
corrected segmentation and its fixes (`wm-dseg/1`); read by
`baillarger.wm_sites.read_sites`. The sidecar's keys are enforced by
`baillarger.wm_sites.SITES_SPEC`, and `tests/test_bids.py` checks that the table
below lists the same keys, types and requirements.

The sites are the review queue for manual edits of the WM mask (`wm-edits/1`):
a reviewer looks at each in turn and records an answer. Nothing else reads
them.

## Files

Next to the white surface, in `DERIV/sub-<sub>/ses-<ses>/anat/`, or in `DIR`
if given, hemisphere `<h>`:

| File | Content |
|---|---|
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-wmsites_sites.tsv` | One row per site, in queue order |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-wmsites_sites.json` | Sidecar, `wm-sites/1` |

The `sites` suffix is ours (BIDS has none for this), in BIDS's entity order.

## Sites

A face of the white surface is misfacing when its normal points away from (or
square to) the sum of its three corners' normals, each corner's normal being
the sum of its faces' cross products. The surface is one level set of a
field, so no face is flipped against the field; a misfacing face marks a place
where the surface folds back within a few faces: a pinch, the tip of a thin
strand, the stub of a handle cut. Faces whose centroids are within
`GroupVoxels` of each other (single linkage) are one site; its centre is the
voxel nearest the mean of its faces' centroids.

## The table

Tab-separated, with a header row, one row per site. The file has ranked sites
only: every row has `kind` `misfacing`.

| Column | Type | Meaning |
|---|---|---|
| `rank` | integer | Place in the queue, from 1. 0 means no rank (a random site) |
| `kind` | string | `misfacing` (a ranked site, in the table) or `random` (see Random sites) |
| `i`, `j`, `k` | integer | The site's centre voxel (0-based) on the segmentation's grid |
| `faces` | integer | Its misfacing faces (0 for a random site) |
| `tissue` | string | The labels outside the WM mask within `TissueVoxels` (a cube): `grey matter` (cortex: 3 or 42), `fluid` (0, 24, or a ventricle or choroid plexus label: 4, 5, 31, 43, 44, 63; `label-set/1` puts these outside the WM mask, and ex vivo they hold outside medium), `mixed` (both) or `other` |
| `cut`, `fill`, `cavity` | integer | Voxels of each kind of topology fix (`desc-wmtopofixes` codes 1, 2, 3) within `FixVoxels` (a cube) |
| `wm_face_pieces`, `wm_any_pieces` | integer | Pieces of the WM mask within `ContactVoxels` (a cube), with face adjacency and with any adjacency: more with face adjacency means pieces that touch only at an edge or a corner |
| `gap_face_pieces`, `gap_any_pieces` | integer | The same for the voxels outside the mask |
| `wm_thickness` | number | The largest distance (voxels) from a WM voxel to the nearest voxel outside the mask, within `CoreVoxels` (a cube); 1 for a strand one voxel wide |
| `gap_width` | number | The same outside the mask; 1 for a gap one voxel wide |

The queue order: by `tissue` (grey matter, mixed, fluid, other), then more
faces first, then more fixes, then by the centre voxel.

## Random sites

Random sites are not ranked and are not in the file. They are a sample of the
surface, to compare with the ranked sites. A review session records `n` and
the `seed`. The same call then gives the same sites.

The draw: `baillarger.wm_sites.read_random_sites(surf, n, seed)` loads the
surface, the corrected segmentation and the fixes as `write_sites` does, then
calls `baillarger.wm_sites.random_sites(v_vox, n, seed, seg, filled, fixes)`.
That function takes the vertex numbers

    np.random.default_rng(seed).choice(len(v_vox), size=n, replace=False)

where `v_vox` holds the vertices of the white surface in voxels of the
segmentation's grid. Each vertex has the same weight. The `n` vertices are
different. `n` more than the number of vertices is an error. `n` = 0 gives no
sites.

The site of a vertex is its position rounded to the nearest voxel (`np.rint`),
then clipped into the grid. The row has the same columns as a ranked row. The
measures are those of a ranked site, taken at that voxel. `kind` is `random`,
`rank` is 0 and `faces` is 0. The rows are in the order of the draw.

## Sidecar keys: wm-sites/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-sites/1` |
| `Description` | string | yes | What the files are |
| `Sources` | array of strings | yes | The white surface's `.surf.gii` and `.dat`, the corrected segmentation and its fixes (absolute paths) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path (the segmentation's is the `DecidedAgainst` of answers decided on these sites) |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `FilledLabels` | array of integers | yes | The labels of the WM mask |
| `MisfacingFaces` | integer | yes | Misfacing faces of the surface |
| `Sites` | integer | yes | Rows of the table |
| `GroupVoxels` | number | yes | Faces whose centroids are this close are one site |
| `FixVoxels` | integer | yes | Half-side of the cube in which fixes are counted |
| `TissueVoxels` | integer | yes | Half-side of the cube that names the tissue |
| `ContactVoxels` | integer | yes | Half-side of the cube in which pieces are counted |
| `CoreVoxels` | integer | yes | Half-side of the cube in which thickness and gap are read |
