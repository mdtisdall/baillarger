# Flow-field columns on phantoms

The first stage of `docs/plans/column-solver.md`. It builds the field form
of the column solver (arm C there) and tests it on phantoms only, before
any chain arm and before any real data. Read first:

- `docs/plans/column-solver.md`: *The model*, *Stopping and contacts*,
  *The field form (arm C)*, *Adaptive refinement and coarsening (arm C)*
  and *Inference*. This plan uses those definitions and does not repeat
  them;
- `prototypes/scripts/fold_phantoms.py` and
  `prototypes/scripts/bent_phantom.py` (the phantoms and their truth);
- `reports/Bent columns trial.md`, *The fit on the phantoms*.

This is a prototype plan. Nothing in the ray sampler or `src/` changes.
The rules in `docs/plans/round-2.md` apply.

**Status:** draft for review. No code is written.

## Goal

Find the theoretical gaps in the field form, mainly in how it represents
kissing surfaces: a contact where the stop surface is pressed against
itself, a hairpin where it folds back, and a gap where it opens. The
phantoms have known geometry, so a gap shows as a clear failure, not as
noise in real data. The result decides whether the field form goes on to
real data, and what in it must change first.

## Questions

Each question names the phantoms that answer it (*Phantoms*).

1. **Fused contact.** In the fused part of the connected kiss phantom,
   does the field stop each bank's columns on the contact, with no flow
   line across it, with and without the dark band at the contact?
2. **Hairpin.** At the bottom of a fundus, as the slot narrows from
   0.48 mm to below one voxel, do the columns end on the hairpin with no
   focal point inside the cortex? Where do the octree's cells go finer
   than the image, and are they marked prior-shaped?
3. **The end of a gap.** Where the connected kiss phantom's slot changes
   from open to fused, the gap ends. With a smooth change the stop surface
   folds back in a hairpin across the slot. With a sharp change it turns
   through corners, and columns can graze the gap's end face. What does
   the field do in each case?
4. **The cut on an uncertain surface.** The cut rule needs a definite
   surface to count the regions of cortex in each cell, but the stop
   surface is a probability. Is a cut on the most probable surface enough,
   or must the region be part of each node's state (D-F3)?
5. **τ at a contact.** Both banks reach the top level at the contact, so
   the contact is one level of τ. Does the solved τ agree there, when the
   banks differ in thickness (the connected kiss and the curved fundus:
   2.5 and 1.2 mm)? Does each bank get its own speed?
6. **The anchor across a fold.** In the connected kiss phantom, as in a
   real brain, the banks' WM is connected under the fundus, so one anchor
   reaches both banks. The path between the two sides of the contact runs
   round the fundus, through the whole fold. Does τ agree across the
   contact, and how many sweeps does the alignment take, with and without
   the coarse level?
7. **Grid.** On every phantom: the spacing scan from 0.6 mm down to image
   resolution, then the adaptive octree. Does the octree match the finest
   uniform grid's columns with fewer nodes? Does the evidence test refuse
   refinement on noise and on a flat slab?
8. **Uncertainty.** With noise, are the per-column spreads from sampled
   fields calibrated: z = (true − mean)/σ with an SD near 1 for the stops
   and the τ levels?

## Truth under the stop rule

The phantoms' own truth must be restated for the rule that columns stop
at the stop surface (`column-solver.md`, *Stopping and contacts*). The
user decided (2026-11-11): cortex stops at a contact and does not overlap.
So at a fused contact each bank's column stops on the contact:

- **Connected kiss.** The depth function of the curved fundus with a slot
  half-width that changes along the fundus line: φ = 1 − (r − g(z))/t(θ).
  The true columns are its orthogonal trajectories, traced as in
  `bent_phantom.py`, and they stop on the slot's faces, which are the stop
  surface. Where g is below a voxel, the two faces are a fused contact, and
  each bank's column stops on it. The true τ levels are the level sets of
  φ, up to one monotone map shared by both banks.
- **Concentric kiss (optional, below).** `fold_phantoms.Fold.column` lets a
  fused column run through the contact to the far bank's WM. Under the
  stop rule, each bank's true column runs radially from its own WM to the
  contact cylinder Rc and stops there. The true stop surface is Rc in the
  fused sector and the two faces of the gap (Rc ± g/2) in the open
  sector.
- **fundus and curved.** The true columns are the orthogonal trajectories
  of the depth function, as now. They end on the slot's edge (r = g),
  which is the stop surface. As g goes to 0, they end on the slot's
  half-plane and converge on its edge line. That line is part of the stop
  surface, so columns that meet there do not count as a focal point.
- **The truth is a definition.** `bent_phantom.py` says so for its depth
  function. The adapters below state it for each phantom.

The adapters are new code (phase F0), with exact tests.

## Phantoms

| Phantom | Source | Answers |
|---|---|---|
| Slab (one flat bank) | new: a `fold_phantoms` fundus far from the slot, or a flat case | exactness; the evidence test refuses refinement |
| Crowns, w = 0.32, 0.64, 1.28 mm | `fold_phantoms.standard` | divergence; columns cover the ribbon |
| **Connected kiss**: the curved fundus (tA 2.5, tB 1.2 mm) with the slot open (g = 0.32 mm) along part of the fundus line and fused (g below one voxel) along the rest, joined by a smooth change and, in a variant, a sharp one; each with and without the dark band at the contact | new, from `bent_phantom.curved` | 1, 3, 4, 5, 6 |
| Fundus, g = 0.48, 0.16 mm | `fold_phantoms.standard` | 2 |
| Fundus, g below one voxel (for example 0.04 mm) | new parameter value | 2, 4 |
| Curved fundus (tA 2.5, tB 1.2, g 0.16), and g below one voxel | `bent_phantom.curved` | 2, 5 |
| Concentric `kiss`, `kiss_flat` (optional) | `fold_phantoms.standard` | whether the shared stop surface alone aligns τ between banks with separate WM |
| τ phantom: layers with known offsets and speeds, a flat band, an areal border | new | the warp alone (`column-solver.md`, exact tests) |

Every phantom is made with the noise of `fold_phantoms.make` and also
without noise. The oblique rotation stays, so nothing is aligned with the
grid.

## Measures

Diagnostics and views, not pass/fail rules, for each phantom, grid and
noise level:

- **Stops:** the distance from each column's stop to the true stop
  surface.
- **Paths:** the distance from each column to its true column
  (`bent_phantom.path_errors`, or `fold_eval.errors` for straight truth).
- **Levels:** the distance from each solved τ level to the true level,
  after the one shared monotone map is fitted.
- **Speeds:** the ratio of the two banks' speeds, against the ratio of
  their thicknesses.
- **Topology:** flow lines that cross the stop surface; focal points
  inside the cortex; places where the field falls to zero or reverses away
  from the stop surface; overlapping cortex (must be none by form; a count
  above zero is a defect).
- **The cut:** the count of regions per cell against the truth, near the
  contact, the hairpin and the gap's end.
- **Grid:** nodes per arm of the spacing scan; where the octree refines
  and coarsens; prior-shaped cells.
- **Uncertainty:** the spread of the stops and levels from sampled fields,
  and its calibration.
- **Cost:** time and peak memory.

## Exact tests

- On the slab: the solved columns are straight and normal to the bank, the
  levels are planes, and the octree stays at its coarsest level.
- With noise alone (no structure), the evidence test refuses every
  refinement.
- Moving the anchor changes every τ by one constant and nothing else.
- On the connected kiss without noise: no flow line crosses the fused
  contact, each bank's stops lie on it, and no cortex overlaps.
- On a fundus with g below one voxel, without noise: the columns end on
  the slot, no focal point is inside the cortex, and the tip cells are
  finer than the image and marked prior-shaped.
- The truth adapters: on each phantom, the adapter's true stops lie on the
  stop surface to within the integration step, and its true columns do
  not cross.

## Phases

Each phase ends with views for the user.

- **F0. Truth and phantoms.** The truth adapters under the stop rule, the
  new phantoms (slab, fundus with sub-voxel g, the connected kiss and its
  sharp variant, τ phantom), and their exact tests.
- **F1. The field solver core.** A uniform grid first. (τ, direction)
  candidates per node, the terms of `column-solver.md` written locally,
  the stop field, the cut on the most probable stop surface, the mean
  field, the integrator and the checks. NumPy reference on the CPU. Slab
  and crowns.
- **F2. Kissing surfaces.** The connected kiss and its sharp variant:
  questions 1, 3, 4, 5 and 6. Optional: the concentric kiss, to see whether
  the contact alone aligns τ between banks with separate WM. Real brains do
  not need this, because their WM is connected, but it shows how strongly
  the contact ties the banks.
- **F3. Hairpins and the grid.** The fundus over its g values and the
  curved fundus: question 2, then the spacing scan and the adaptive
  octree on every phantom (question 7), and calibration with noise
  (question 8).
- **F4. Checkpoint.** A short report of the gaps found and what each
  needs. The user decides whether the field form goes on to the column
  solver's later stages as it is, changed, or with the fallback of chains
  at hairpins.

Torch on MPS comes only after F4, for the kernels that the later stages
need at scale.

## Decisions for the user

- **D-F3. The cut on an uncertain surface.** Cut on the most probable stop
  surface and solve again (the plan's first form), or make the region part
  of each node's state. Question 4 gives the evidence.

## Not in this plan

- The chain arms (A, B) and the comparison between representations: later
  stages of `column-solver.md`.
- Real data, the coarse-to-fine schedule over the column graph, and the
  compressibility measure.
- GPU kernels before F4.
