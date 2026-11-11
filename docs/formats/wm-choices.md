# wm-choices: the site answers of the WM review

The contract of `baillarger.wm_choices`: code with a written contract, as a
sampler has (`docs/version-2-plan.md` §1). The review pane (stream 4) calls
these functions for its previews. It writes the choice that the reviewer
picks to the answer log (`wm-edits/1`, `docs/formats/wm-edits.md`). This
document has no sidecar: the functions write no file.

## The choices

| Name | Function | What it changes |
|---|---|---|
| `remove-strand` | `remove_strand` | The thin WM piece at the site, to the labels round it |
| `thicken` | `thicken` | The cortex near the WM round the site, to WM |
| `fill-gap` | `fill_gap` | The thin gap between WM walls at the site, to WM |
| `widen-gap` | `widen_gap` | The WM walls of that gap, to the labels round them |
| `leave` | `leave` | Nothing. The answer records that the site was examined |

`manual-paint` is not a function here. It comes from the ITK-SNAP command
(`python -m baillarger.wm_paint`).

`CHOICES` maps each name to its function. `run(name, labels, site, hemi,
radius=, reach=)` calls the function and puts the name in the result.

## Inputs

Each function has the same signature:

    f(labels, site, hemi, *, radius=1, reach=4) -> Change

| Input | Type | Meaning |
|---|---|---|
| `labels` | 3D integer array | The current labels (`desc-wmtopo`) in a box round the site. The function does not change it |
| `site` | 3 integers | The site's voxel, in the box's coordinates |
| `hemi` | `R` or `L` | The hemisphere: it gives the WM mask (`FILLED_LABELS`), the WM label and the cortex label (`label-set/1`) |
| `radius` | integer, voxels | The width of the choice (below). Default 1. The review pane offers 1 and 2 for `thicken` |
| `reach` | integer, voxels | The radius of the ball round the site in which the choice acts. Default 4 |

All lengths are in voxels of the grid, not in mm. The grid is not
isotropic (0.16028 × 0.16 × 0.16028 mm), so a ball of voxels is not a ball
of mm. At this voxel size the difference is below 0.2 %.

"WM" means the WM mask: the voxels with a label in `FILLED_LABELS[hemi]`.
`B_r` is the set of offsets with a Euclidean norm of at most r (for r = 1:
the six face neighbours and the centre). The reach ball is the set of
voxels at a Euclidean distance of at most `reach` from the site. A piece is
a connected piece with the Kuhn adjacency (`baillarger.handles.kuhn_structure`,
plan §2). "The piece at the site" is the piece of the candidates that holds
the candidate nearest the site (ties: the lowest C-order linear index).

## The rules

- **`remove-strand`.** The candidates are WM ∧ ¬opening(WM, `B_r`), in the
  reach ball. The voxels changed are the piece at the site. Each gets the
  most common label among its face neighbours that are not WM and not
  changed (ties: the lowest label). A voxel with no such neighbour gets its
  label from the voxels already given one, by the same rule, layer by
  layer. A voxel that no layer reaches gets 0. With no candidate, nothing
  changes.

  The opening keeps the one voxel of a strand that touches a wall: the
  wall voxel under it is not eroded, so the dilation restores the strand's
  first voxel. A strand of one voxel width thus leaves a stub of one voxel
  on the wall.
- **`thicken`.** The voxels changed are the cortex voxels
  (`CORTEX_LABEL[hemi]`) in the reach ball whose Euclidean distance to the
  nearest WM voxel is at most `radius`. Each gets `WM_LABEL[hemi]`. A WM
  blade 2 voxels thick grows to 4 with radius 1, and to 6 with radius 2.
- **`fill-gap`.** The candidates are closing(WM, `B_r`) ∧ ¬WM, in the reach
  ball. The voxels changed are the piece at the site. Each gets
  `WM_LABEL[hemi]`. With radius 1, this fills a gap one voxel wide.
- **`widen-gap`.** The gap is the piece of `fill-gap`. The voxels changed
  are the WM voxels in the reach ball that share a face with it. Each gets
  the most common non-WM label among its face neighbours (ties: the lowest
  label). A gap one voxel wide between two walls opens to three voxels: one
  voxel goes from each wall.
- **`leave`.** Nothing changes.

## Outputs

A frozen `Change`:

| Field | Type | Meaning |
|---|---|---|
| `voxels` | (N, 3) int64 | The voxels changed, in the box's coordinates, sorted by C-order linear index, each once |
| `labels` | (N,) int64 | The new label of each. It is never the current label |
| `params` | object | `radius` and `reach`, to record in the answer (`Params`) |
| `choice` | string | The name of the choice (set by `run`) |

`apply(labels, change)` gives a copy of the labels with the change made.
The answer log takes the voxels in grid coordinates: add the box's offset.

## Locality

`support(radius, reach)` = reach + 2 · radius + 1 is the half-width (voxels)
of the cube round the site that a choice reads: the opening and the closing
read 2 · radius beyond the reach ball, and the face neighbours one voxel
more. On a box that holds this cube, a choice gives the same voxels and
labels as on the whole volume. `box_round(shape, site, half)` gives the
slices of that cube, cut by the volume's edges.

## The preview

    preview(labels, site, hemi, answer, *, voxel_mm, radius=1, reach=4, halo_mm=1.0) -> Preview

`answer` is the name of a choice (a key of `CHOICES`), or a `Change` (for
example a manual paint, in the coordinates of `labels`). The preview makes
the answer, then runs the correction (`baillarger.white_dseg.wm_topo`) on a
box round the site, with the answer's voxels as label edits (edited
voxels), as `dseg` does with the label edits (`label-edits/1`). It returns a frozen `Preview`:

| Field | Meaning |
|---|---|
| `answer` | The `Change` of the answer, in the coordinates of `labels` |
| `correction` | A `Change`: the voxels (coordinates of `labels`) that the correction then changes, and their new labels |
| `overrides` | Whether a fix of the correction changes a voxel of the answer |
| `box` | The slices of the box in `labels` |
| `cut_by_box` | Whether a face of the box that is not an edge of `labels` has WM on it, after the answer |

The box is the support cube, grown to hold the voxels of a `Change`, plus a
halo of `halo_mm` on each side (default 1.0 mm, rounded up to whole voxels
per axis), cut by the edges of `labels`. Reason: on the hemisphere, most
handle moves are 1 to 2 voxels (plan §3c, step 2), and an answer can make
or remove a handle only through the voxels that it changes. A halo of 1 mm
(about 6 voxels) holds a loop of WM round such a handle that is up to about
2 mm across.

What the preview cannot show: the correction sees only the box. Outside
the box, the mask is background. A handle whose loop leaves the box is not
a handle in the box, so the preview does not show its fix. WM that the box
face cuts can also look like a separate piece. `cut_by_box` is true in that
case, and the pane says that the preview is approximate there. The real
result comes from the `dseg` run on the hemisphere with the answer in the
label edits.

The correction changes the edited voxels last (`wm-dseg/1`, *Manual
edits*). An answer of one of the choices above changes
voxels only near the site, so the correction can nearly always fix a handle
that the answer makes through other voxels: the preview then shows the
correction and no override. An override comes when every fix goes through
edited voxels, for example label edits that hold the whole loop and its
hole.

## Tests

`tests/test_wm_choices.py`, on phantoms: a strand one voxel thick is removed
and nothing else changes (except the stub, above); a one-voxel gap is
filled; a widened gap opens to three voxels; `thicken` grows a two-voxel
blade to four (radius 1) and to six (radius 2); `leave` returns no voxels;
each choice on a box of `support` gives the same voxels as on the whole
phantom; the input does not change; the preview of `fill-gap` that closes a ring
shows a correction outside the answer and no override; the preview of a
manual paint that holds a whole ring with its rod (an answer that leaves
a handle) shows the override, a cut of one rod voxel; a box that cuts the
ring gives `cut_by_box` and no correction.
