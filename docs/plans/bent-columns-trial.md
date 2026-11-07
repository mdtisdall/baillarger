# Bent columns: a local trial

Stream 2, phase 2 of `docs/version-2-plan.md` (§4d), the open question
"which paths are right in real cortex" (§4c, *Open*). Read the plan's §4b
(*What a correct column looks like*), §4c (*Open*: the deflected normal ray
and its alternative form) and the memory rules on principles, roles and
estimators. The rules in `docs/plans/round-2.md` apply.

This is a prototype. Nothing in the ray sampler changes. "Normal rays stay
straight" (user, #47) stays the rule; a result here goes to the user as
evidence, and a change of the rule is a separate decision with its own plan.

## Goal

Find out whether the columns we have can be bent into a better form by an
optimisation on the data: the contrast along each column, agreement with
its neighbours, and the smoothness of cortex. Not by a geometry chosen
before the data is seen (a field, a front, equal volume).

## Why: what the straight rays do at a fundus

At true vertex 12182770 (`scratch/v2/derivatives-main`, hemisphere R), the
section (radius 5 mm, axis world x) runs lengthwise along the fundus of a
sulcus. The banks differ in thickness: about 2.5 mm on one side and about
1.2 mm on the other, read from a cut across the fundus. The user reads this
as the fast thickening from S1 to M1 at the bottom of the central sulcus.

- The fundus columns' rays point up the middle of the sulcus. The tip of
  the sulcus (outside medium, label 0) is 2.2–3 mm above the WM.
- A ray on the midline meets the tip. A ray 0.1 mm to the side runs up one
  bank's cortex, or up the fused gap, and meets outside at 4–8 mm, or not
  within 6 mm.
- Along the section, the depth of the first outside is a sawtooth: it rises
  slowly, then drops in one step. At the two drops, neighbouring columns
  4924217 and 4924847 (normals 0.8° apart, roots 0.064 mm apart) meet
  outside at 3.00 and 4.26 mm, and 12183700 and 12184198 (2.8°, 0.141 mm)
  at 5.67 and 2.22 mm.
- The normals are right for the white surface: neighbours differ by
  0.5–3°, and the roots lie within 0.014 mm rms of a smooth curve. Straight
  rays perpendicular to the WM cannot fan from a narrow fundus to two banks
  of different thickness.

## The model

Each column c keeps its root r_c and its straight inward part. Its outward
part bends.

- **Path.** x_c(s) = r_c + s n_c + d_c(s) for s ≥ 0, with d_c(0) = 0 and
  d_c perpendicular to n_c: two shear components in the root's tangent
  frame, as cubic B-splines in s (knots 0.5 mm apart). The along-ray
  component is not in the path: the depth warp below carries it.
- **Depth warp.** Each column has a monotone warp of its arc length, as in
  the joint warp (stream 3, §5c step 2), with that warp's regularisation.
  Profiles are compared in warped depth, so a column of a different length
  is not penalised for it there.
- **Soft top.** The top of a column is where its path first enters
  outside. To keep it differentiable, it is read from a smooth outside map:
  label 0 smoothed by a Gaussian of 1 voxel (0.16 mm), a stand-in until the
  segmentation gives an outside probability (it gives P of filled WM only,
  and none during bootstrapping). A column with no outside within the cap
  (8 mm) is censored.
- **Length.** L_c is the arc length of the bent path from the root to its
  soft top.

The costs, summed:

1. **Agreement below the top.** Between neighbours on the mesh, the squared
   difference of the gain-corrected echoes (`gain.py`) at equal warped
   depth, weighted by both columns' masks: 1 inside the column, falling
   to 0 at its soft top. Nothing above a top counts.
2. **Reach the top quickly.** Σ_c L_c over the columns that are not
   censored. Among paths with equal agreement, it picks the shortest. A
   shortest path from a root to the outer surface meets the surface at a
   right angle, so this term fixes the shear that agreement alone leaves
   free (a column that slides along a layer keeps its profile).
3. **Agreement in length.** Between neighbours, in two forms, each fitted
   and reported: an L2 penalty on L_c − L_c' (forbids jumps, flattens a
   real gradient), and an L2 penalty on the mesh Laplacian of L (allows a
   steady gradient, forbids jumps). Censored columns take their length
   from this term only.
4. **Smooth shear.** Bending stiffness ∫|x_c''|² along each path, and the
   squared difference of neighbours' shear at equal s.
5. **No crossing, no crowding.** A barrier on each mesh face: the triangle
   of its three columns' points at equal warped depth keeps a positive area
   and its orientation. A mild L2 term on the log of that area against its
   neighbours', so columns spread over the pia and do not crowd onto the
   nearer bank or the tip.
6. **No return.** A barrier on a path that enters WM (filled labels) again
   after it has left it.
7. **Texture** (only after step 2 below finds it): the angle between the
   path's tangent and the image's direction along radial texture, weighted
   by coherence. The plan records that at 160 µm no source shows radial
   fibres giving column orientation (§4c, *Open*), so this term is added
   only where the box shows it.

## Steps

1. **The box.** All roots within 5 mm of true vertex 12182770 along the
   surface (15,123), the echoes and labels in a box round their 8 mm paths.
   Straight rays from the sampler as the start.
2. **Texture.** From `orient` on the box: the structure tensor's
   eigenvectors and coherence. A map, on both banks, of the coherence of
   the direction along the fibres (the leading eigenvector of a texture
   that is linear, not planar) and its angle to the rays. A view, not a
   threshold: it says whether term 7 has anything to use here.
3. **The fit.** torch, from the straight rays, with terms 1–6. CPU is
   enough (feasibility, not speed). Under `scripts/memguard`, with the cap
   at about 1.5× the measured peak.
4. **Weights.** A scan of each term's weight over a decade each way from a
   start where the terms are of equal size at the straight rays. Report how
   the result moves: a sensitivity is a finding, never a setting to tune
   away.
5. **Controls on phantoms.** The fold phantoms of
   `prototypes/scripts/fold_phantoms.py` (crown 0.32, 0.64, 1.28 mm; fundus
   0.16, 0.48 mm; kiss; kiss_flat) have straight true columns. On them the
   fit must leave the columns close to straight: the path error against the
   true columns (`fold_eval.errors`), against the straight rays' errors.
6. **A curved phantom.** A fundus whose banks differ in thickness (2.5 and
   1.2 mm): laminae as level sets of a stated depth function, true columns
   as their orthogonal trajectories, traced finely. The truth here is a
   choice of definition, and the report says so. Path error and outside
   depth, against the straight rays.
7. **Term 7,** only if step 2 found texture: the fit again with it, and the
   difference it makes.
8. **A second site,** if the marked M1S1 points
   (`prototypes/data/M1S1_points_full0.json`) can be placed on the current
   derivatives: the same fit there, and the profile split at the marked
   points against normal's 0.17 mm (§4c).

## Diagnostics

Views for the user, not pass/fail numbers. For the straight rays and each
fit, side by side:

- the cuts across and along the section at the two drops (as
  `jumpcut.py`), with the bent paths;
- the depth of the first outside per step along the section: the sawtooth,
  or its absence;
- the section's profiles, drawn as in the viewer's section pane;
- a map on the box of each column's total bend angle and of the bend
  against depth (bent at the root, or growing with depth: the
  misalignment diagnostic's two cases, §4c);
- the lengths L_c on the mesh, and their difference from the straight
  rays' outside depth;
- where the tops land: each bank's pial area reached per unit of root area
  (crowding), and the share of fundus columns that end at the tip, on
  each bank, or censored;
- the angle between each path and the data normal where coherence is high:
  the independent check (§4b, *Smooth does not prove correct*);
- own-bank returns, and columns that enter outside before their top through
  a hole, a tear (catalogue type F) or an excision edge (type C).

## Outputs

- `prototypes/scripts/bent_columns.py`: the box, the fit and the views. It
  takes its data and output directories as arguments.
- A report, `reports/Bent columns trial.md`, with figures in
  `reports/figures/bent-columns/` (PNG, each well under 1 MB): the
  question, the code, the results by step, and what they do and do not
  show.
- One PR for the script and the report.

## Decisions for review (made by the coordinator)

- The root and the inward part stay fixed. Only the outward part bends.
- The soft top comes from smoothed label 0 until an outside probability
  exists.
- Censored columns get no top term; their length comes from their
  neighbours.
- Terms 1–6 first. Term 7 only where step 2 shows texture.
- Both forms of the length term are fitted and shown. Neither is chosen in
  advance.

## Open

- Whether "reach the top quickly" lets columns bend through the fused gap
  above a fundus tip toward a far outside, where a contact has no fluid.
  Term 3 limits it; the cuts show it.
- How much the result depends on the stand-in soft top, once an outside
  probability exists.
