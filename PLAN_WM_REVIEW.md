# Plan: a review queue for the white surface's tight folds

Planning file for reviewing, in the viewer, the places where the white (WM)
surface folds tightest, and recording each decision as a manual edit for
`dseg`. When the work is done, the lasting content goes into the module
docstrings and the format specs (`docs/formats/`), and this file is deleted.

## What is decided

- About 60 sites per hemisphere, the ones touching grey matter, go to the user
  as one queue. The queue comes after the automatic result, which must already
  be correct (no handles, no cavities); the review only refines it.
- Each answer is a recorded, replayable diff against the aseg: the manual
  edits of `wm-edits/1` (`baillarger.wm_edits`, `docs/formats/wm-edits.md`,
  PR #78). The aseg is never changed, and no edited copy of it is kept.
  `dseg --edits` applies the edits, then its automatic correction, so no
  answer can leave a handle or a cavity.
- It is built into the viewer (`python -m baillarger.viewer --white`), not
  ITK-SNAP, Freeview, Slicer or napari. ITK-SNAP is only an escape hatch for
  a site that none of the choices fits (step 4).

## The sites

A site is a place where the true white surface folds back within a few
faces: its faces whose normal points away from the summed normals of their
own corners (each vertex's faces' cross products, summed), grouped when within
3 voxels of each other. The surface there is still a valid level set; these
are where it is least well defined, and where columns rooted on it have no
clear direction.

On the example hemisphere (`dseg` and `mesh` from `main` at `b16db63`): 152
such faces at 91 sites.

- 58 sites have a topology fix (`desc-wmtopofixes`: handle cut, handle fill,
  cavity fill) within 4 voxels, against 0.66 % of random surface points: the
  handle fix cuts or fills at a bridge's thinnest place, usually a wall one
  voxel thick, and leaves stubs one voxel apart or a one-voxel plug.
- The other 33 are on thin WM, threads one to two voxels wide and their tips.
- By the labels outside the WM within 3 voxels: 60 sites touch grey matter,
  22 fluid, 9 both.

The queue: grey matter first, then mixed, then fluid (fluid optional); within
each, by the number of faces.

## Choices

Each choice is a function of the aseg in a box round the site that returns
the voxels it changes (`i j k old new`), so that a choice is exact and
testable, and the same rows are written as the answer. All act within a ball
of radius `R` (voxels, default 4) round the site, and only on the labels named.

| Choice | Changes |
|---|---|
| `remove-strand` | WM voxels of the thin part at the site (WM at most 1 voxel from the outside, by the distance transform, face-connected to the site, not in the largest face-connected piece left when they are removed) set to the commonest non-WM aseg label among their face neighbours |
| `thicken` | non-WM voxels labelled cortex within 1 (or 2) voxels of the WM, set to the hemisphere's WM label |
| `fill-gap` | non-WM voxels at most 1 voxel from the WM on two sides (a gap one voxel wide), set to the WM label |
| `widen-gap` | WM voxels face-adjacent to such a gap, set to the commonest non-WM label among their face neighbours |
| `leave` | nothing (an answer with no rows: the site counts as answered) |
| `manual-paint` | from ITK-SNAP (step 4) |

The rules above are the starting point; their tests fix them exactly.

## Preview

Before a choice is saved, the viewer shows what it would do: the aseg in a box
round the site (the site's ball plus a margin of 16 voxels) with the choice's
rows applied, then the same steps as `dseg` and `mesh`, called, not copied:
`white.wm_topo` on the box, `antialias.thin_plate` on its mask, marching
tetrahedra. The preview's surface and its tight-fold faces are drawn over the
planes, with the current surface for comparison.

A preview is approximate, and says so on screen. The automatic correction is
global (a handle can reach outside the box), and the field is solved with the
box's edge as its edge. The real result comes from rerunning `dseg --edits`
and `mesh`; each answer records the SHA-256 of the `dseg` it was decided
against (`DecidedAgainst`), so after a rerun the answers whose context changed
can be listed.

## Steps (one PR each)

1. **The site finder** (`baillarger.wm_sites`): the sites from a white
   surface, its `dseg` and its fixes, with per site: the centre voxel, the
   faces, the fixes within 4 voxels by kind, the tissue round it (grey matter,
   fluid, mixed), whether the mask or the gap touches only at edges or corners,
   the local WM thickness and gap. Written as a derivative
   (`desc-wmsites_sites.tsv` and sidecar) under a spec of its own
   (`docs/formats/wm-sites.md`), from
   `python -m baillarger.white sites SURF`.
2. **The volume pane in `--white` mode**: echo 2 (or 1 or 3) read in a box
   round the crosshair (`blocks.open_volume`), the true surface's section from
   its blocks (`WhiteSurface.load(block=...)`), and an overlay of the WM mask
   and the fixes. Needed for the review, and useful on its own.
3. **The review pane**: the queue, one site at a time, its planes and the
   local surface in 3D; the choices, each previewed, then saved at once with
   `Edits.with_answer` and `write_edits`; an answer reverted with `without`;
   the next site. Modelled on the pial-label review (`viewer/labels.py`):
   a ranked list, one answer per item, keys to move and decide, every change
   saved at once.
4. **The ITK-SNAP escape hatch**: export a site's box (the aseg's labels and
   the echoes, with the box's affine) as NIfTI; read the edited labels back,
   refuse changes outside the box, and save their difference from the aseg as
   one answer, `manual-paint`.

Step 1 does not need #78; steps 3 and 4 do, and wait for it (or are stacked on
it).

## Tests

- The site finder: on phantoms, a thin strand, a one-voxel gap and a
  one-voxel plug each give exactly one site at the right voxel, with the right
  tissue and contact; a smooth ball gives none.
- Each choice: on phantoms, exactly the voxels its rule names change, from the
  aseg's labels, and nothing outside its ball.
- The preview: on a phantom small enough that the box is the whole grid, it
  equals `dseg --edits` followed by `mesh`, exactly.
- The answers: saved and read back (`wm-edits/1`), reverted, and the site
  shown as answered; `DecidedAgainst` is the `dseg`'s SHA-256.
- The escape hatch: a box exported, changed and read back gives the same rows
  as the change made; a change outside the box is refused.
- The viewer: the server's new routes in `--white` mode, as the existing
  server tests do.

## After the review

`dseg --edits`, then `mesh` (with its overview); the sites are found again on
the new surface, and the answers whose `DecidedAgainst` is not the new
`dseg`'s are listed for a second look.

## Open

- Thicker handle fixes in `dseg` (simple-point steps, so a cut or fill is not
  one voxel thick): may remove many of the 38 grey-matter sites near fixes and
  shrink the queue. Undecided; the queue does not depend on it, but it would
  be reviewed after it.
- Fluid sites in the queue, or grey matter and mixed only.
- The choices' parameters (`R`, thicken by 1 or 2 voxels).
