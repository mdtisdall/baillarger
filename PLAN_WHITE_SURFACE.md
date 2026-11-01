# Plan: an initial white surface for cortical normals

Planning file for the `feature/wm-invariants` branch. When the work is done,
the lasting content goes into `docs/formats/wm-dseg.md`,
`docs/formats/white-surf.md` and the module docstrings, and this file is
deleted.

## The goal

The white surface is where columns start, and they run along its normals. Its
job is to have, at each point, the **cortical normal** there, as well as the
WM's geometry can tell it. It is not meant to be *the* GM/WM interface:
baillarger makes no pial surface, and the GM/WM transition is found along
each column like any other layer. So the surface may sit somewhat inside or
outside the WM label, and moving it there is allowed where that makes its
normals better defined.

This plan is for a good **initial** surface only. The surface may be
corrected later, from what ray casting finds; that is a later plan.

## Principles

- **Honest to the data as acquired.** Every step runs at the scan's own
  resolution (160 µm now, 50 µm whole brains later). No step is made
  tractable by downsampling or decimating. Steps are local, so that they run
  in blocks and on a GPU.
- **FreeSurfer is the model.** We follow its design and deviate only with
  good cause and data from the hemisphere. Each deviation is recorded below
  with its reason. FreeSurfer's settings are translated by kind:
  anatomical ones (mm) carry over; sampling ones (voxels, vertices,
  iterations) are re-derived for our resolution.
- **No assumption about the aseg's source.** The aseg is an input with
  unknown errors, in FreeSurfer's label convention (cortex 42 right, 3 left;
  WM and deep structures in FilledLabels). Evidence about what is real comes
  from the FLASH echoes, as statistics measured in each scan, not from one
  tool's labels.

## FreeSurfer's steps and ours

| FreeSurfer | What it does | Ours | Deviation and reason |
|---|---|---|---|
| `mri_segment` | WM from intensity; thickens thin strands (default on, under 4 mm, 20 largest) | The aseg is given; **thicken** (below) | The segmentation comes from outside; thickening is kept |
| `mri_edit_wm_with_aseg` | Adds the deep structures | FilledLabels | None |
| Manual WM edits | Edits of `wm.mgz` or `filled.mgz` | `dseg --edits` (wm-edits/1) | Edits are diffs against the aseg, which is never changed |
| `mri_pretess` | Adds voxels where WM touches only along an edge or at a corner | **Well-composed WM** (below) | None intended |
| `mri_fill` | Fills, picks one connected WM | Cavity fill; a cavity holding cortex is refused for a person to resolve | A cortex cavity is an error in the input, not something to fill |
| `mris_fix_topology` | Surface-based: maps to a sphere, retessellates each defect, picks cut or fill by per-defect intensity and curvature statistics | Volume-based fronts (`remove_handles`); cut or fill **by intensity** (below) | The sphere mapping is global and holds the whole surface in memory, so it cannot run in blocks at 50 µm. Volume fronts have precedent (Kriegeskorte and Goebel 2001; Bazin and Pham 2007). The intensity-based choice is kept |
| `mris_smooth`, `mris_place_surface` (white) | Moves the surface to the intensity edge, with smoothness and no self-intersection | **Regularise for normals** (below): no search for an edge | Position is not our goal; smoothness without self-intersection is |
| Pial surface | | None | Columns find the outer layers along their rays |
| Inflation, sphere, registration | | None | Not needed for columns |

Contrast: FreeSurfer's ex vivo route does not assume a contrast. It
estimates white and grey statistics in each brain, from the WM label
(`mris_exvivo_surfaces`, per defect in `mris_fix_topology`), and normalises a
learned bias field estimated at 1 mm (`mri_exvivo_norm`). We do the same:
statistics from the echoes inside and outside the WM near each place, in
floating point (not FreeSurfer's 8-bit, WM-at-110 convention). A bias field,
where needed, may be estimated coarsely, since it varies slowly.

## Properties of the surface

M is the corrected mask, S the surface meshed from it.

1. **Topology (hard).** M's Betti numbers (Kuhn adjacency) are (b0, 0, 0),
   so each piece of S is a sphere. Every change to the mask is a simple point
   or a set that keeps the homology (`handles._set_simple`), and `wm_topo`'s
   end check stays.
2. **No self-intersection (hard).** S is one level set of one field on the
   Kuhn tetrahedra, so it cannot cross itself. No step moves mesh vertices
   in a way that could break this.
3. **Normals defined at scale s.** A WM blade thinner than about 2s, or a gap
   between two WM surfaces narrower than about 2s, has no stable normal at
   scale s (its two sides cancel). Every piece of WM and every gap between
   two surfaces is made at least that wide, by moving the surface in either
   direction. s is in mm. Thickening a thin blade keeps the directions of its
   sides, so it changes where their normals are computed, not what they are.
4. **Stay near the data.** Among the changes that meet 3, the smaller is
   taken, and the surface stays within a bound of the input mask's boundary
   (about s). This keeps the surface from drifting to wherever its normals
   would be smoothest.

1 and 2 always hold. 3 holds wherever it can without breaking 1. Each place
where it cannot is an **exception**, listed with its kind and position. The
exceptions are the review queue (#80), answered by manual edits.

Movement into cortex is allowed: under this goal a root inside the cortex is
no more a problem than one inside the WM.

## The steps

**In `dseg` (volume):**

1. **Manual edits**, then **cavities** (as now): fill each cavity; refuse
   and list each one that holds cortex.
2. **Handles, cut or fill by evidence.** `remove_handles` finds each handle
   and its two options, as now. Instead of the smaller option, each option
   is scored by how well the voxels it changes fit WM or non-WM, from echo
   statistics measured around that handle (FreeSurfer's per-defect
   statistics). The voxel count stays as a tie-break and a prior. The first
   version keeps the size rule, and the intensity score is added and
   compared on the hemisphere.
3. **Thicken.** Where WM is thinner than 2s (thin strands of the aseg, and
   the one-voxel stubs and plugs that cuts and fills leave), add WM by
   simple points until it is 2s across, in any direction. This replaces
   #83's trimmed threads and thickened fills. Whether a thin strand is real
   or noise is decided from the echoes: noise is removed (a strand's tip is a
   simple point, so it retracts fully), real WM is thickened.
4. **Widen gaps between two surfaces.** A background voxel whose filling
   would join two parts of M that are separate nearby (filling it is not
   simple) is in a gap between surfaces. Where such a gap is narrower than
   2s, it is widened by removing WM by simple points on both sides. A pit in
   one surface (filling it is simple) is not a gap, and is left to step 6.
5. **Well-composed WM** (`mri_pretess`): add voxels where WM touches only
   along an edge or at a corner, by simple points, and the same for
   background touching only so.

**In `mesh` (surface):**

6. **Regularise for normals.** Start with today's antialiasing (thin-plate
   field, every voxel keeps its sign) on the mask from steps 1–5, and
   measure. If tight folds remain at the voxel scale, relax it: let the
   field change sign at simple points only, within the bound of property 4,
   minimising its smoothness energy (a topology-preserving level set, as in
   CRUISE; Han et al. 2004). This is the step that may move the surface off
   the voxel staircase.
7. **Normals**: the area-weighted vertex normals smoothed along the mesh, as
   now; or the gradient of the field at scale s, if that is better on the
   measures below.

## Measures

The surface is judged by its normals; its position is a diagnostic.

- **Neighbour-normal angles**: median, 90th and 99th percentiles.
- **Misfacing faces** (normals against their corners' normals), as sites,
  by side (GM, fluid, mixed) and near a fix or not.
- **Normal cancellation**: the length of the smoothed normal before it is
  normalised (now as low as 0.56 in thin blades, against 0.9998 typically).
- **Independent orientation**, as a diagnostic: agreement with orientation
  evidence that does not come from the WM shape, such as radial fibres in
  M1.
- **Changes**: voxels added and removed by step and by input label; the
  largest distance moved from the input boundary; surface area.
- **Exceptions** by kind.
- **Reference sites**: the first 20–30 sites of the review queue, answered
  by a person, as the yardstick for the automatic choices.

## What the hemisphere has shown so far

Tight-fold sites (misfacing faces within 3 voxels of each other) on the
example hemisphere, from `scratch/topo-variants/` and `scratch/wm-invariants/`:

| Variant | Sites | GM, near a fix | GM, no fix nearby | Area (mm²) |
|---|---|---|---|---|
| none | 90 | 36 | 23 | 72,089 |
| fills thickened (1 voxel) | 76 | 24 | 24 | 72,076 |
| cuts widened (1 voxel) | 85 | 35 | 23 | 72,076 |
| threads trimmed | 61 | 35 | 5 | 71,808 |
| cuts preferred, widened; threads trimmed | 42 | 21 | 5 | 71,890 |
| no fills into cortex | 131 | 66 | 23 | 72,295 |
| no fills into cortex; opening, r = g = 0.24 mm | 91 | 44 | 12 | 72,892 |
| same, 0.32 mm | 101 | 45 | 16 | 73,385 |
| same, 0.40 mm | 131 | 58 | 29 | 74,578 |

- Forcing cuts where a fill fits makes more folds (90 to 131).
- Thickening the fills helps (90 to 76), the direction FreeSurfer takes.
- The GM folds with no fix nearby are thin strands: trimming them takes 23
  to 5, by removing about 41,000 WM voxels. Step 3 tests whether thickening
  them does as well.
- The voxel opening (removing WM outside its opening by r, and next to gaps
  narrower than 2g) is worse at every radius, grows the area, and cascades
  over passes (14 to 31). On a crop, every "narrow gap" voxel at 0.24 mm was
  a pit in one surface, not a gap between two. It is dropped; step 4's gap
  test and step 6 replace it.

## Experiments, in order

Each runs on the hemisphere under `scripts/memguard`, one at a time, and is
reported with the measures above.

1. Steps 1–3 with the size rule: thicken strands and fixes instead of
   trimming. Compare with "threads trimmed" and "fills thickened".
2. Add step 5 (well-composed WM).
3. Add step 4 (gaps between two surfaces).
4. s over a range, from the floor of about 1.5 voxels (0.24 mm) upward.
5. The intensity score for cut or fill (step 2), against the size rule and
   the reference sites.
6. Step 6 relaxed, if the measures after 1–5 still show voxel-scale folds.

## Tests

- Exact shapes: a thin blade is thickened to 2s with its sides' normals
  kept; a thread of noise retracts; a one-voxel cut is widened to 2s; a pit
  is left alone; an edge-only contact is made well-composed; a handle whose
  bridge looks like non-WM is cut and one whose tunnel looks like WM is
  filled (synthetic intensities); a cavity holding cortex is refused.
- Properties on random objects: 1 exactly; 3 everywhere but the exceptions;
  each exception is a place where 3 fails; 4's bound holds.
- The default output stays the reference output until a step becomes the
  default (`tests/test_handles.py` digests).

## The code on this branch

Commit 14ecc4d implemented the earlier design (no growth into cortex, the
voxel opening). Under this plan:

- kept: the refusal of cavities holding cortex, the exceptions table
  (`desc-wmtopo_exceptions.tsv`), and the `nofill` argument of
  `remove_handles` (useful for manual "never fill here" edits);
- removed: `open_thin`, `--no-gm-fills`, `--thickness-mm` and `--gap-mm`,
  and the `Invariants` sidecar key, replaced by the steps above with s.

PR #83's options (fill cost, widened cuts, thickened fills, trimmed threads)
are superseded by steps 2–4; #83 is closed when this lands.

## Open

- **s**, from experiment 4. The floor is about 1.5 voxels: below it the
  voxel staircase itself is thinner than 2s (a cube's edges at 1 voxel).
- **The bound in property 4**: s, or a separate distance.
- **Fills across a sulcal fundus.** A fill there puts a WM plug in the
  cortex whose sides face along the layers, not across them. Step 2's
  intensity score should usually cut such a handle; whether that needs a
  rule of its own is decided from experiment 5 and the reference sites.
- **Which echoes**, and whether a bias field is needed for the statistics.
