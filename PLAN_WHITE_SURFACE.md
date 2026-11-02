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

Step numbers in this table and the next refer to "The steps", further down.

| FreeSurfer | What it does | Ours | Deviation and reason |
|---|---|---|---|
| `mri_segment` | WM from intensity; thickens thin strands (default on, under 4 mm, 20 largest) | The aseg is given; **thicken** (step 3) | The segmentation comes from outside; thickening is kept |
| `mri_edit_wm_with_aseg` | Adds the deep structures | FilledLabels | None |
| Manual WM edits | Edits of `wm.mgz` or `filled.mgz` | `dseg --edits` (wm-edits/1), step 1 | Edits are diffs against the aseg, which is never changed |
| `mri_pretess` | Adds voxels where WM touches only along an edge or at a corner | **Well-composed WM** (step 5) | None intended |
| `mri_fill` | Fills, picks one connected WM | Cavity fill (step 1); a cavity holding cortex is refused for a person to resolve | A cortex cavity is an error in the input, not something to fill |
| `mris_fix_topology` | Surface-based: maps to a sphere, retessellates each defect, picks cut or fill by per-defect intensity and curvature statistics | Volume-based fronts (`remove_handles`); cut or fill **by intensity** (step 2) | The sphere mapping is global and holds the whole surface in memory, so it cannot run in blocks at 50 µm. Volume fronts have precedent (Kriegeskorte and Goebel 2001; Bazin and Pham 2007). The intensity-based choice is kept |
| `mris_smooth`, `mris_place_surface` (white) | Moves the surface to the intensity edge, with smoothness and no self-intersection | **Regularise for normals** (step 6): no search for an edge | Position is not our goal; smoothness without self-intersection is |
| Pial surface | | None | Columns find the outer layers along their rays |
| Inflation, sphere, registration | | None | Not needed for columns |

Contrast: FreeSurfer's ex vivo route does not assume a contrast. It
estimates white and grey statistics in each brain, from the WM label
(`mris_exvivo_surfaces`, per defect in `mris_fix_topology`), and normalises a
learned bias field estimated at 1 mm (`mri_exvivo_norm`). We do the same:
statistics from the echoes inside and outside the WM near each place, in
floating point (not FreeSurfer's 8-bit, WM-at-110 convention). A bias field,
where needed, may be estimated coarsely, since it varies slowly.

## Other tools

Each tool's answer to the decisions above, and what we take from it.
"Unverified" marks what comes from memory, not from a source or the code.

| Tool | Topology | Cut or fill | Thin WM; smoothing | Self-intersection | Resolution | Licence | We take |
|---|---|---|---|---|---|---|---|
| FreeSurfer, ex vivo route | Corrected on the surface, through a sphere mapping | Per-defect intensity and curvature statistics | Thickens strands (`mri_segment`); `mri_pretess` adds voxels at diagonal contacts | Checked while placing the surface | Built for 1 mm; `-hires` keeps native voxels | Research only (FreeSurfer licence) | The model: steps 2, 3, 5 |
| CRUISE (Nighres) | A correction step, then a topology-preserving level set (Han et al. 2004) | From the correction step | Curvature force on the level set | None possible: one level set | 0.4 mm at 7T | Apache-2.0 | Step 6's relaxed field; well-composed topology (its default `wcs`), step 5 |
| CIVET (CLASP) | WM mask cleaned, `mincdefrag`, then `sphere_mesh` (genus 0); then a deformation from it | Unverified | Removes loosely connected voxels and fills tightly connected ones before meshing; then coarsens to 120,000 and resamples to 81,920 triangles | Explicit prevention and a self-proximity cost (MacDonald et al. 2000; Kim et al. 2005) | About 1 mm; its mesh is coarsened | Not stated on GitHub | Morphological clean-up before meshing (steps 3, 5); not the coarsening |
| CAT12 | Spherical harmonics patch each defect (Yotter et al. 2011) | Intensity decides | Central surface from projection-based thickness; refinement afterwards | A separate collision-reduction step | About 1 mm | GPL-2.0+ | Cut or fill by intensity, again (step 2) |
| BrainSuite | Graph-based (Shattuck and Leahy 2001) | The smaller change | | | About 1 mm | GPL-2.0 (source) | A baseline only |
| BrainVoyager (Kriegeskorte and Goebel 2001) | Region growing from deep inside, by distance, stopped where it touches itself | Per defect, at the thinnest place | | | About 1 mm | Commercial | Confirms our fronts (step 2) |
| BrainVISA (Mangin et al. 1995) | Homotopic growth of a sphere-topology interface toward the segmentation (`VipHomotopic`) | Implicit in the growth | | | About 1 mm | Unverified | Gaps defined by the skeleton between two banks (step 4) |
| Template flows: TopoFit (FreeSurfer's `mris_estimate_wm`), CorticalFlow, CortexODE | By construction: a template mesh moved by a learned diffeomorphic flow | None needed | Learned | Avoided in theory (CortexODE); conditions for manifold meshes (CorticalFlow) | Trained on 1 mm FreeSurfer surfaces | Varies | The idea of topology by construction, not the trained models |
| BigBrain (Lewis et al. 2014; Wagstyl et al. 2018, 2020) | Unverified | Unverified | | | 20 µm data; released surfaces have 163,842 vertices per hemisphere | Open data | Little for this step: the surfaces are far coarser than the data |

Two patterns stand out:

- **Topology by construction** (CIVET, BrainVISA, template flows): start
  from a sphere and move it, rather than correct defects afterwards.
  `remove_handles`' fronts are already a form of this, a homotopic growth
  from deep inside the WM, so it is not a separate path; but a deformation
  that keeps genus 0 throughout (step 6 relaxed) is the same idea applied to
  the surface.
- **Coarse meshes**: every tool that makes surfaces on large or fine data
  (purple-mri, CIVET, BigBrain) coarsens to 80,000–330,000 triangles. That is
  the step our principle rules out (next section).

## Why the other tools coarsen, and why we do not

Their reasons, and whether each applies to us:

1. **Mesh density matched to the data.** In vivo at 1 mm, vertices much
   closer than 1 mm add no information: their extra positions come from
   interpolation and the staircase. A mesh at about the voxel spacing is
   right. *For us:* the same rule gives a mesh at 160 µm (later 50 µm), not
   a coarse one.
2. **Mesh density matched to the analysis.** Thickness and curvature maps are
   smoothed with kernels around 10 mm before vertex-wise statistics, and
   parcellations vary over centimetres; more vertices add cost and multiple
   comparisons, not information. *For us:* laminar questions are about fine
   detail, so this does not apply.
3. **Correspondence for registration and group analysis.** The surface is
   resampled onto a template sphere so that every subject has the same
   vertices. CIVET's code says its 81,920-triangle sphere "is the one used
   for surface registration"; purple-mri decimates to 327,680 faces, the
   size of FreeSurfer's ico7, for `mris_register` and the DKT atlas;
   BigBrain's surfaces have ico7's 163,842 vertices. This is resampling to
   a template, and nothing in it requires the template to be coarse. *For
   us:* there is no atlas registration, so it does not apply.
4. **Tooling.** On fine data, what still forces coarsening is the software:
   CPU pipelines with whole volumes and surfaces in memory, about 0.5–1 KB
   per vertex in FreeSurfer, global steps (sphere mapping, whole-surface
   self-intersection checks, which scale poorly with mesh size), and
   downstream tools that expect ico7. This is not a fundamental limit:
   out-of-core and streaming meshes, spatial hierarchies for collision
   tests and GPU methods exist, and our block layout is a form of them.
   *For us:* the reason this project builds its own data structures.

The one sound idea behind coarsening survives: at the voxel scale, geometry
is mostly staircase and segmentation noise, so normals and curvature need a
scale larger than a voxel. Coarsening imposes that crudely, smoothing
positions and directions together and discarding the sampling density. This
plan separates the two: the mesh stays at the data's resolution, and the
normals are defined at an explicit scale s.

So coarsening the mesh has no justification for this project, and is not
reconsidered.

## Properties of the surface

M is the corrected mask, S the surface meshed from it.

**Property 1. Topology (hard).** M's Betti numbers (Kuhn adjacency) are (b0,
0, 0), so each piece of S is a sphere. Every change to the mask is a simple
point or a set that keeps the homology (`handles._set_simple`), and
`wm_topo`'s end check stays.

**Property 2. No self-intersection (hard).** S is one level set of one field
on the Kuhn tetrahedra, so it cannot cross itself. No step moves mesh
vertices in a way that could break this.

**Property 3. Normals defined at scale s.** A WM blade thinner than about
2s, or a gap between two WM surfaces narrower than about 2s, has no stable
normal at scale s (its two sides cancel). Every piece of WM and every gap
between two surfaces is made at least that wide, by moving the surface in
either direction. s is in mm. Thickening a thin blade keeps the directions
of its sides, so it changes where their normals are computed, not what they
are.

**Property 4. Stay near the data.** Among the changes that meet 3, the
smaller is taken, and the surface stays within a bound of the input mask's
boundary (about s). This keeps the surface from drifting to wherever its
normals would be smoothest.

Properties 1 and 2 always hold. Property 3 holds wherever it can without
breaking 1. Each place where it cannot is an **exception**, listed with its
kind and position. The exceptions are the review queue (#80), answered by
manual edits.

Movement into cortex is allowed: under this goal a root inside the cortex is
no more a problem than one inside the WM.

## The steps

Steps 1–5 run in `dseg`, on the volume; steps 6 and 7 run in `mesh`, on the
surface.

### Step 1. Manual edits, then cavities

As now: apply the manual edits, then fill each cavity; refuse and list each
one that holds cortex.

### Step 2. Handles, cut or fill by evidence

`remove_handles` finds each handle and its two options, as now. Instead of
the smaller option, each option is scored by how well the voxels it changes
fit WM or non-WM, from echo statistics measured around that handle
(FreeSurfer's per-defect statistics). The voxel count stays as a tie-break
and a prior. The first version keeps the size rule, and the intensity score
is added and compared on the hemisphere.

### Step 3. Thicken

Where WM is thinner than 2s (thin strands of the aseg, and the one-voxel
stubs and plugs that cuts and fills leave), add WM by simple points until it
is 2s across, in any direction. This replaces #83's trimmed threads and
thickened fills. Whether a thin strand is real or noise is decided from the
echoes: noise is removed (a strand's tip is a simple point, so it retracts
fully), real WM is thickened.

### Step 4. Widen gaps between two surfaces

A thin background voxel (no ball of radius s in the background holds it)
whose filling is not simple touches two surfaces at once: it is in a gap
between them. The WM within s of it is removed by simple points, on both
sides. This finds gaps one voxel wide, which is what step 3 leaves where it
could not close a gap, and slots open at one end, whose walls face each
other as a gap's do. Wider gaps, and pits in one surface (filling them is
simple), are not found; pits are left to step 6.

### Step 5. Well-composed WM

As `mri_pretess` does: add voxels where WM touches only along an edge or at
a corner, by simple points, and the same for background touching only so.

### Step 6. Regularise for normals

Start with today's antialiasing (thin-plate field, every voxel keeps its
sign) on the mask from steps 1–5, and measure. If tight folds remain at the
voxel scale, relax it: let the field change sign at simple points only,
within the bound of property 4, minimising its smoothness energy (a
topology-preserving level set, as in CRUISE; Han et al. 2004). This is the
step that may move the surface off the voxel staircase. A sign change must
also keep property 3: it may not make a blade voxel (a thin voxel whose
removal is not simple), a gap voxel, or a thin voxel in a thin piece of 10
voxels or more, within s of it. Without that test the energy shrinks widened gaps and thin blades
from every side (experiment 6).

### Step 7. Normals

The area-weighted vertex normals smoothed along the mesh, as now; or the
gradient of the field at scale s, if that is better on the measures below.

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

Steps 3 and 5 of this plan (`scratch/white-surface/`; "near a fix" counts
fixes values 1–3 only, so thickened and well-composed voxels are not fixes
here):

| Variant | Sites | GM, near a fix | GM, no fix nearby | Fluid | Area (mm²) | Voxels changed | Exceptions |
|---|---|---|---|---|---|---|---|
| none (main) | 90 | 36 | 23 | 22 | 72,089 | | |
| step 3, s = 0.32 mm | 84 | 16 | 32 | 32 | 76,367 | 767,487 added | 103 |
| step 5 | 50 | 21 | 13 | 11 | 72,073 | 936 added, 194 removed | 25 |
| steps 3 and 5, s = 0.32 mm | 54 | 10 | 13 | 29 | 76,371 | 767,292 + 6,806 added, 180 removed | 169 |
| steps 3 and 5, s = 0.30 mm | 43 | 11 | 8 | 23 | 74,797 | 331,524 + 427 added, 189 removed | 141 |
| steps 3, 4 and 5, s = 0.30 mm | 5 | 1 | 1 | 1 | 74,871 | 328,017 + 341 added, 7,034 removed | 92 |
| steps 3, 4 and 5, s = 0.26 mm | 12 | 5 | 3 | 2 | 74,578 | 298,724 + 1,032 added, 5,359 removed | 137 |
| steps 3, 4 and 5, s = 0.36 mm | 7 | 0 | 4 | 0 | 79,055 | 1,695,547 + 1,956 added, 13,106 removed | 124 |
| steps 3, 4 and 5, s = 0.42 mm | 4 | 1 | 1 | 2 | 79,143 | 2,024,339 + 355 added, 14,200 removed | 87 |
| steps 3, 4 and 5, s = 0.52 mm | 5 | 0 | 1 | 3 | 80,539 | 5,147,823 + 2,045 added, 15,815 removed | 84 |

Neighbour-normal angles of the output normals, over the edges (degrees):

| Variant | 50th | 90th | 99th | 99.9th | 99.99th | Edges over 60° | Over 90° |
|---|---|---|---|---|---|---|---|
| none (main) | 1.34 | 3.88 | 11.42 | 20.6 | 34.7 | 297 | 23 |
| step 5 | 1.33 | 3.88 | 11.38 | 20.3 | 33.2 | 243 | 19 |
| steps 3 and 5, 0.30 mm | 1.46 | 5.37 | 12.49 | 19.0 | 35.7 | 587 | 38 |
| steps 3, 4 and 5, 0.30 mm | 1.47 | 5.41 | 12.57 | 18.3 | 25.6 | 4 | 0 |
| steps 3, 4 and 5, 0.26 mm | 1.58 | 5.40 | 12.94 | 19.8 | 27.4 | 9 | 3 |
| steps 3, 4 and 5, 0.36 mm | 2.58 | 6.96 | 11.92 | 16.7 | 22.1 | 0 | 0 |
| steps 3, 4 and 5, 0.42 mm | 2.45 | 6.45 | 10.73 | 15.1 | 20.7 | 1 | 0 |
| steps 3, 4 and 5, 0.52 mm | 2.81 | 6.09 | 9.76 | 13.4 | 19.0 | 0 | 0 |

- **Diagonal-only contacts cause most folds.** At baseline, 74 of the 90
  sites have a checkerboard square or a corner-only cube of the mask within
  4 voxels, though the whole mask has only about 1,150 of them. Step 5
  alone takes the sites from 90 to 50, by changing 1,130 voxels, and leaves
  the area as it was; 10 of its 50 sites have a contact nearby.
- **Thickening removes the GM folds it is meant for** (GM sites 34 with
  step 5 alone, 19 with steps 3 and 5 at 0.30 mm), but adds folds where the
  WM meets unlabelled background (fluid, the medial cut, next to the
  ventricle and thalamus), many next to blocked target voxels: gaps that
  step 4 is for.
- **s must not be a whole number of voxels along an axis.** Axis 1 is
  0.16 mm and the others 0.16028 mm, so s = 0.32 mm is exactly 2 voxels along
  axis 1, and every ball has a lone voxel at each end along it. Their
  contacts multiplied the checkerboards in the planes of axes 0 and 2 (about
  390 to 7,000) and more than doubled the voxels added. At 0.30 mm the
  thickening adds 332,000 voxels and the sites drop from 54 to 43.
- **The remaining step-5 sites** are mostly thin WM inside cortex (largest
  depth within 4 voxels at most 2.4 voxels: about 15 sites) and handle
  fills of tens of voxels.
- **Steps 3, 4 and 5 together** take the sites from 90 to 5 (6 misfacing
  faces) and remove the tail of the neighbour-normal angles (edges over 60°
  from 297 to 4, none over 90°). Widening removes 7,034 voxels round 2,095
  gap voxels. The cost is the bulk: the 90th percentile rises from 3.9° to
  5.4°, the voxel-scale roughness of the ball unions that step 6 is for,
  and the area grows 3.9%.
- **s (experiment 4).** Sites (sites with mixed GM and fluid faces are in
  the total only) and the tail of the angles change little above 0.30 mm,
  but the bulk does: thickening adds about 300,000 voxels at 0.26 and
  0.30 mm, 1.7 million at 0.36 mm and 5.1 million at 0.52 mm, the area grows
  from 3.9% to 10–12% over main, and the median angle goes from 1.5° to
  2.5–2.8°. The jump between 0.30 and 0.36 mm (about 4 voxels of width,
  0.6–0.7 mm) is likely where 2s passes the width of much of the WM in
  gyral blades, so thickening there moves the boundary into cortex instead
  of filling noise. 0.26 mm is worse than 0.30 mm on every count (9 edges
  over 60°, 3 over 90°). On the final mask, gap voxels left are 57, 10, 204,
  21 and 23 from 0.26 to 0.52 mm, none in a piece of 10 voxels or more;
  thin voxels (against each run's own s, so not comparable) are 4,273,
  4,742, 16,497, 14,118 and 30,608. 0.30 mm is the working value.
- **Exceptions belong at the end.** The 81 thickening exceptions are
  counted before widening and well-composing. On the final mask, 10 gap
  voxels are left (9 pieces), but 4,742 voxels are thin, in 1,381 pieces, of
  which 79 have 10 voxels or more: widening and step 5 thin some WM again,
  and most pieces are single staircase voxels. The review queue needs the
  failures of property 3 on the final mask, with a size below which a piece
  is not a site.

Step 6 on crops (experiment 6; `scratch/step6-proto/REPORT.md`). Ten crops
of the steps 3–5 result at s = 0.30 mm, each a 64³ core measured inside a
16-voxel margin: the 5 windows with the most thickened voxels, a control
with no fix, and 4 random windows with at least the median WM boundary. The
relaxed solve is today's thin-plate solve warm-started from today's field;
a voxel within the bound of the input boundary may change sign only at a
simple point, one parity class at a time. Misfacing here is a face's normal
against the mean of its corners', not the site count above.

| Crops | Main | Today, steps 3–5 | Relaxed, bound 0.30 mm | Relaxed, thickened and widened voxels locked | Relaxed, no new blade or gap voxel |
|---|---|---|---|---|---|
| 4 random, 90th pct | 3.35° | 4.00° | 3.21° | 3.55° | 3.21° |
| 5 thickened, 90th pct | 8.86° | 10.08° | 6.18° | 8.75° | 6.36° |
| control, 90th pct | 2.06° | 2.06° | 1.70° | 1.70° | 1.70° |
| 5 thickened, edges over 30° | 2,554 | 332 | 318 | 342 | 198 |
| 5 thickened, edges over 60° | 119 | 1 | 59 | | 0 |
| 5 thickened, gap voxels | 573 | 2 | 11 | | 0 |
| 5 thickened, misfacing faces | 259 | 11 | 2 | 12 | 9 |

- **Relaxing works on the bulk.** The 90th percentile falls below main's on
  every kind of crop, the untouched control included, and the area of the
  thickened crops falls 6–23%. Betti numbers never change. Bounds of 0.30
  and 0.45 mm agree within 0.4°; 0.16 mm is worse (misfacing 11 to 39), so
  the bound is s.
- **But it undoes steps 3 and 4 in places.** The thin-plate energy shrinks
  small background regions (widened gaps, tunnels) and thin blades from
  every side; topology stops them closing, so they become slivers. In one
  crop it re-creates 11 gap voxels and the edges over 60° go from 1 to 59
  (maximum 113°), 231 of its 270 edges over 30° within 3 voxels of them.
- **Locking the voxels steps 3 and 4 changed is not enough**: it keeps
  property 3 but only a third to half of the gain, and gaps still narrow
  just outside the locked voxels. Step 6 needs property 3 as a constraint
  of its own.
- **A local test keeps property 3 and nearly all the gain.** A sign change
  is refused if it makes a new blade voxel or gap voxel within s of it,
  tested in a window round the voxel. A blade voxel is a thin voxel whose
  removal is not simple (the gap voxels of the background), as a gap voxel
  is a thin background voxel whose addition is not simple. Counting every
  thin voxel instead refuses most flips, because a one-voxel bump on a flat
  face is thin, so a new layer can never start (one crop: 14,000 refused,
  2,100 made, 90th percentile 6.80° against 5.77°). With the blade test the
  random crops and the control are exactly as unconstrained (nothing or
  about 40 flips refused), and the thickened crops keep 95% of the gain with
  no edge over 60° (the crop above: maximum 49°, 0 gap voxels). Misfacing
  faces are 9 against 2 unconstrained. The test costs 15–220 s per 96³ crop
  on the CPU, one window at a time.
- **The blade test misses tubes.** Seven of the 9 misfacing faces are on one
  strand that step 3 had thickened (6 of its 67 voxels thin in the input):
  relaxing thinned it to a tube 2 voxels across (49 thin), and removing a
  voxel from a tube's side is simple, so no blade voxel appears. The site
  was already misfacing before relaxing (6 faces), next to a thickening
  exception and a handle cut. Unconstrained, the strand retracts whole and
  the crop gains 18 diagonal-contact voxels, which undoes step 5. Splitting
  the test by direction (a removal may make no new thin voxel, an addition
  no new blade voxel) is worse on all three crops tried: removing a
  staircase corner makes a thin voxel, so most flips are refused (90th
  percentile 7.40°, 7.55° and 6.62°, against 6.78°, 6.67° and 5.77°), and
  the strand's site keeps 9 misfacing faces.
- **A size floor on thin pieces sees tubes.** Adding to the blade and gap
  test "no new thin voxel in a thin piece of 10 voxels or more" (in the
  window) keeps the strand as wide as it was (34 thin voxels round the
  site, 56 before relaxing). Its 6 misfacing faces are the 6 of before
  relaxing, next to the exception, with angles near 34°: a review site, not
  a step 6 failure. Pooled over the thickened crops: 90th percentile 6.59°
  (blade test 6.36°), edges over 30° 136 (198), none over 60°, thin voxels
  1,146 (1,368; input 723), misfacing 10 (9); the random crops and the
  control as before (3.22°, 1.70°). A floor of 5 refuses too much (90th
  percentile 0.2–0.6° worse than 10 on four crops). Thin pieces under the
  floor still grow, so relaxing keeps about 400 more thin voxels than its
  input; whether they matter depends on the size floor of the exceptions.

## Experiments, in order

Each runs on the hemisphere under `scripts/memguard`, one at a time, and is
reported with the measures above.

**Experiment 1.** Steps 1–3 with the size rule: thicken strands and fixes
instead of trimming. Compare with "threads trimmed" and "fills thickened".

**Experiment 2.** Add step 5 (well-composed WM).

**Experiment 3.** Add step 4 (gaps between two surfaces).

**Experiment 4.** s over a range, from the floor of about 1.5 voxels (0.24
mm) upward. Done: 0.26, 0.30, 0.36, 0.42 and 0.52 mm, with steps 3, 4 and 5
(results above); 0.30 mm is the working value.

**Experiment 5.** The intensity score for cut or fill (step 2), against the
size rule and the reference sites.

**Experiment 6.** Step 6 relaxed, on crops of the steps 3–5 result at
s = 0.30 mm (results above), then with the local blade and gap test, then
with a size floor of 10 on thin pieces. Next: the hemisphere.

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
voxel opening). It stays for reference on the branch
`archive/wm-invariants-opening`, which is never merged. Under this plan:

- kept: the refusal of cavities holding cortex, the exceptions table
  (`desc-wmtopo_exceptions.tsv`), and the `nofill` argument of
  `remove_handles` (useful for manual "never fill here" edits);
- removed: `open_thin`, `--no-gm-fills`, `--thickness-mm` and `--gap-mm`,
  and the `Invariants` sidecar key, replaced by the steps above with s.

PR #83's options (fill cost, widened cuts, thickened fills, trimmed threads)
are superseded by steps 2–4; #83 is closed when this lands.

## Open

- **s**: 0.30 mm from experiment 4, on one hemisphere. The floor is about
  1.5 voxels: below it the voxel staircase itself is thinner than 2s (a
  cube's edges at 1 voxel). s must also stay off whole numbers of voxels
  along any axis (lone tips).
- **Fills across a sulcal fundus.** A fill there puts a WM plug in the
  cortex whose sides face along the layers, not across them. Step 2's
  intensity score should usually cut such a handle; whether that needs a
  rule of its own is decided from experiment 5 and the reference sites.
- **Which echoes**, and whether a bias field is needed for the statistics.
