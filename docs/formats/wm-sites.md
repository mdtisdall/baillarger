# wm-sites: where the white surface folds tightest

SpecVersion `wm-sites/1`. Written by `python -m baillarger.wm_sites SURF [--out DIR]`
(`baillarger.wm_sites.write_sites`) from a white surface (`white-surf/1`), its
corrected segmentation and its fixes (`wm-dseg/1`); read by
`baillarger.wm_sites.read_sites`. The sidecar's keys are enforced by
`baillarger.wm_sites.SITES_SPEC`, and `tests/test_bids.py` checks that the table
below lists the same keys, types and requirements.

The sites are the review queue of the WM mask, one file per hemisphere: a
reviewer looks at each in turn and records an answer (`wm-edits/1`). The
review pane reads them with `baillarger.wm_sites.read_sites`. Nothing else
reads them.

## Site kinds

| Kind | From | In the file |
|---|---|---|
| `override` | A fix that changes edited voxels (the fix table's `overrides` is not `n/a`, `wm-dseg-fixes/1`) | yes |
| `misfacing` | Misfacing faces of the surface (below) | yes |
| `crease` | Crease edges of the surface (Creases) that no misfacing site takes | yes |
| `handle-fix` | A handle fix (`handle_cut` or `handle_fill`) with a wall of `MinWallVoxels` or more, that is not an override | yes |
| `random` | A random vertex of the surface, drawn for a session (Random sites) | no |

To accept the fix of an override site, the user writes the fix's labels
into the label edits (`label-edits/1`). Then the edited voxels have the
fix's labels, the next `dseg` run does not change them, and the fix is not
an override there. In the viewer's review pane, the choice `accept-fix` of an
override site does this write (`wm-edits/1`). The answer log (`wm-edits/1`)
is not read.

A site with creases (a `crease` site, or a `misfacing` site that took
creases) proposes a choice (`proposal`, `wm-choices`): `remove-strand` or
`fill-gap`. To accept it, the reviewer picks that choice at the site: the
review pane previews it, as any choice, and writes its labels into the label
edits. The reviewer can also pick another choice, paint, or answer `leave`
(the crease stays).

Without a fix table next to the corrected segmentation (an output older
than `wm-dseg-fixes/1` with its table), the file has misfacing sites only.

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

## Creases

A crease is an edge of the white surface whose two faces are folded back on
each other: the cosine of the angle between their normals (in mm) is at
most `CreaseCosine` (−0.9, 154.16°). It is the shared test of
`baillarger.mesh_checks` (`edge_folds`). The surface stays embedded at a
crease, but its normals are not well defined there.

The tip of a crease is a voxel: of the two vertices opposite the crease
edge, take the one nearer to an end of its own grid edge (its edge key,
`white-surf/1`); the tip is that end. On the first hemisphere (INDD121850,
R; 162 creases), both opposite vertices lay within 0.35 voxel of a voxel
centre, at one voxel or at two neighbouring voxels on the same side of the
mask: the end of a WM strand one voxel thick (about three quarters) or of a
gap one voxel wide (the rest).

Creases whose edge midpoints are within `GroupVoxels` of each other (single
linkage) are one group. A group with a midpoint within `GroupVoxels` of the
centroid of a misfacing face joins that face's site. Every other group is a
`crease` site, centred on the tip of its crease with the largest angle.

The proposal of a site with creases comes from the tip of its crease with
the largest angle: `remove-strand` if the tip is in the WM mask, `fill-gap`
if not. The choice acts at the site's centre voxel; for a `misfacing` site
that is not the tip, and the preview shows whether it reaches the tip. On
the first hemisphere, 48 of the 57 proposals changed the tip voxel when run
at the site's centre.

## The table

Tab-separated, with a header row, one row per site, in queue order: the
override sites (by fix id), then the misfacing sites (in their order,
below), then the crease sites (in their order, below), then the handle-fix
sites (larger walls first; ties: by fix id). An empty value is `n/a`.

| Column | Type | Meaning |
|---|---|---|
| `rank` | integer | Place in the queue, from 1. 0 means no rank (a random site) |
| `kind` | string | `override`, `misfacing`, `crease`, `handle-fix` (in the file) or `random` (see Random sites) |
| `i`, `j`, `k` | integer | The site's centre voxel (0-based) on the segmentation's grid |
| `faces` | integer | Its misfacing faces (0 for the other kinds) |
| `tissue` | string | The labels outside the WM mask within `TissueVoxels` (a cube): `grey matter` (cortex: 3 or 42), `fluid` (outside, 0; or a label of `baillarger.labels.TO_OUTSIDE`: 24, 4, 5, 31, 43, 44, 63, which `wm-dseg/1` sets to 0 and only a dseg written before that rule holds; ex vivo they hold outside medium), `mixed` (both) or `other` |
| `cut`, `fill`, `cavity` | integer | Voxels of each kind of topology fix (`desc-wmtopofixes` codes 1, 2, 3) within `FixVoxels` (a cube) |
| `wm_face_pieces`, `wm_any_pieces` | integer | Pieces of the WM mask within `ContactVoxels` (a cube), with face adjacency and with any adjacency: more with face adjacency means pieces that touch only at an edge or a corner |
| `gap_face_pieces`, `gap_any_pieces` | integer | The same for the voxels outside the mask |
| `wm_thickness` | number | The largest distance (voxels) from a WM voxel to the nearest voxel outside the mask, within `CoreVoxels` (a cube); 1 for a strand one voxel wide |
| `gap_width` | number | The same outside the mask; 1 for a gap one voxel wide |
| `fix` | integer | The fix's id in the fix table, for `override` and `handle-fix`; 0 for the other kinds |
| `wall` | integer | The fix's wall (the fix table's `wall`); 0 if none |
| `creases` | integer | The crease edges of the site; 0 for `override`, `handle-fix` and `random` |
| `crease_angle` | number or `n/a` | The largest angle (degrees, 1 decimal) between the normals of a crease's two faces; `n/a` with no crease |
| `proposal` | string or `n/a` | The proposed choice: `remove-strand` or `fill-gap` (Creases); `n/a` with no crease |

For a fix site, the centre voxel is the fix table's `i`, `j`, `k`. For a
crease site, it is the tip; its measures are taken there.

The order of the misfacing sites: by `tissue` (grey matter, mixed, fluid,
other), then more faces first, then more fixes, then by the centre voxel.
The order of the crease sites: by `tissue`, then more creases first, then
the larger `crease_angle` first, then by the centre voxel.

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
`rank` is 0, `faces` and `creases` are 0, and `crease_angle` and `proposal`
are `n/a`. The rows are in the order of the draw.

## Sidecar keys: wm-sites/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `wm-sites/1` |
| `Description` | string | yes | What the files are |
| `Sources` | array of strings | yes | The white surface's `.surf.gii` and `.dat`, the corrected segmentation and its fixes image, then (if there) the fix table (absolute paths) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path (the segmentation's is the `DecidedAgainst` of answers decided on these sites) |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `FilledLabels` | array of integers | yes | The labels of the WM mask |
| `MisfacingFaces` | integer | yes | Misfacing faces of the surface |
| `CreaseEdges` | integer | yes | Crease edges of the surface |
| `CreaseCosine` | number | yes | The crease test's cut-off: the cosine of the angle between the two normals at or below which an edge is a crease (`baillarger.mesh_checks.FOLD_COS`) |
| `Sites` | integer | yes | Rows of the table |
| `GroupVoxels` | number | yes | Faces whose centroids are this close are one site; also the grouping of creases, and how near a misfacing face a crease group joins its site |
| `FixVoxels` | integer | yes | Half-side of the cube in which fixes are counted |
| `TissueVoxels` | integer | yes | Half-side of the cube that names the tissue |
| `ContactVoxels` | integer | yes | Half-side of the cube in which pieces are counted |
| `CoreVoxels` | integer | yes | Half-side of the cube in which thickness and gap are read |
| `MinWallVoxels` | integer | yes | The smallest wall of a `handle-fix` site |
| `SitesByKind` | object | yes | Rows of the table, by kind |
