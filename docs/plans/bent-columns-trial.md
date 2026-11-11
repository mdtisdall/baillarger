# Bent columns: a local trial

Stream 2, phase 2 of `docs/version-2-plan.md` (§4d), the open question
"which paths are right in real cortex" (§4c, *Open*). Read the plan's §4b
(*What a correct column looks like*), §4c (*Open*: the deflected normal ray
and its alternative form) and the memory rules on principles, roles and
estimators. The rules in `docs/plans/round-2.md` apply.

This is a prototype. Nothing in the ray sampler changes. "Normal rays stay
straight" (user, #47) stays the rule. A result here goes to the user as
evidence. A change of the rule is a separate decision with its own plan.

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
  component is not in the path. The depth warp below carries it.
- **Depth warp.** Each column has a warp of its depth, the joint warp of
  stream 3 (§5c step 2: `flattening.fit`, which calls `flatten.fit_warp`),
  with that warp's regularisation. Profiles are compared in warped depth,
  so a column of a different length is not penalised for it there.
- **Soft top.** The top of a column is where its path first enters
  outside. To keep it differentiable, it is read from a smooth outside map:
  label 0 smoothed by a Gaussian of 1 voxel (0.16 mm). This is a stand-in
  until the segmentation gives an outside probability. It gives P of
  filled WM only, and none during bootstrapping. A column with no outside
  within the cap (8 mm) is censored.
- **Length.** L_c is the arc length of the bent path from the root to its
  soft top.

The costs, summed:

1. **Agreement below the top.** Between neighbours on the mesh, the squared
   difference of the flattening features (`flattening.features`: echoes 1
   and 2, trend removed) at equal warped depth, weighted by both columns'
   masks. A mask is 1 inside the column and falls to 0 at its soft top.
   Nothing above a top counts.
2. **Reach the top quickly.** Σ_c L_c over the columns that are not
   censored. Among paths with equal agreement, it picks the shortest. A
   shortest path from a root to the outer surface meets the surface at a
   right angle. So this term fixes the shear that agreement alone leaves
   free: a column that slides along a layer keeps its profile.
3. **Agreement in length.** Between neighbours, in two forms. Each form is
   fitted and reported. Form A is an L2 penalty on L_c − L_c'. It forbids
   jumps and flattens a real gradient. Form B is an L2 penalty on the mesh
   Laplacian of L. It allows a steady gradient and forbids jumps. Censored
   columns take their length from this term only.
4. **Smooth shear.** Bending stiffness ∫|x_c''|² along each path, and the
   squared difference of neighbours' shear at equal s.
5. **No crossing, no crowding.** A barrier on each mesh face: the triangle
   of its three columns' points at equal warped depth keeps a positive area
   and its orientation. The barrier is extended: below a relative area of
   δ = 0.1 it continues as the quadratic of −log at δ, so it is finite.
   The straight rays at the site have inverted faces (3,230 of 29,749
   inner faces at s = 0.5 mm, 15,032 at 3 mm), so the pure barrier is
   infinite at the start. δ is fixed before any fit. A mild L2 term on the log of that area against its
   neighbours' areas, so columns spread over the pia and do not crowd onto
   the nearer bank or the tip.
6. **No return.** A barrier on a path that enters WM (filled labels) again
   after it has left it.
7. **Laminar alignment.** Σ w_k m tᵀ A t, with A = c (I − v vᵀ): v is
   orient's leading eigenvector l1 (the data normal, across the laminae),
   c its coherence (l1 − l2)/(l1 + l2), t the path's unit tangent. It is
   c sin²θ, θ the angle between t and v. It has no sign. It gives the axis
   on which a column crosses the laminae. Costs 2, 3 and 6 give the
   direction along that axis. Where the tensor is not valid (outside the
   cortex, outliers), A = 0. Cost 7 stays in the fit from phase 5 on.
   Each fit also runs once without it, to show its effect.

## Facts from the current code

### The columns of a box

- There is no API for a box. `RaySampler` works by superblock, level and
  halo (`rays.py` 152, 210, 241). All its outputs are in scanner mm.
- The viewer's `Columns.compute` (`viewer/columns.py` 363) is the recipe
  for "all roots within r of a vertex along the surface":
  1. `superblock(v)`, then `near(v, r)`: Dijkstra on `local_graph`, with
     edge lengths in mm.
  2. `rays.roots(sb, 0, r)`, matched to the near vertices by
     `searchsorted`.
  3. `rays.sample(...)` with `e_in` and `e_out` zero except at the chosen
     columns, then `select_columns` (`viewer/columns.py` 146).
- The viewer refuses more than 4,000 columns. A script that calls
  `RaySampler` directly has no such cap.
- `RaySampler.faces(sb, level, halo_mm)` (`rays.py` 328) gives the faces
  between roots. `mesh_smoothing.edges(f)` (36) gives the edges.
- Interval classes (`rays.py` 57–61): `OUTSIDE = 0`, `WM = 1`,
  `CORTEX = 2`, `OTHER = 3`, `OFF_GRID = 4`. Caps: `CAP_IN = 2.0`,
  `CAP_OUT = 8.0`. Step: `STEP = 0.04` mm.

### Sampling the echoes

- `ColumnVolumeSampler.sample(rays)` (`column_volume.py` 136) samples the
  echoes trilinearly at `rays.position`. It is numpy.
- `streamlines.sample(img, pos)` (`streamlines.py` 748) is
  `map_coordinates` of order 1 in voxel coordinates.
- No code in `src` or `prototypes/scripts` samples a volume in torch with
  gradients. There is no use of `grid_sample`.

### The warp

- `flatten.fit_warp` (`flatten.py` 238) is numpy and scipy (sparse CG,
  Gauss-Newton). Per trace, it fits offsets r_v(τ) on a grid of step
  `dtau`, with a smoothness term `eps`, an anchor at one level, and a gauge
  of zero mean per connected piece. It does not enforce monotonicity in
  the fit.
- `flattening.py` adapts the samplers to it: `features` (220),
  `mesh_edge_pairs` (285), `pair_shifts` (357), `fit` (407) and `Warp`
  (395). Constants: `DTAU = 0.08`, `EPS = 0.2`, `TREND_MM = 0.64`,
  `FEATURE_ECHOES = (1, 2)`.
- `gain.py` fits a gain and offset against templates (`solve_gain`, 68).
  The trial does not need it: the flattening features remove the trend.

### Texture

- H1 (user): at 160 µm no radial fibres are expected, and the box shows
  none. The tangential (laminar) texture is used: cost 7.
- Dark outliers (`bent_texture.outliers`, two thresholds: k = 5,
  grow = 3): seeds are voxels of the cortex whose detrended log echo is
  below −5 robust SD (median absolute deviation × 1.4826) in at least one
  echo. The outliers are the connected pieces (26 neighbours, in the
  cortex) of the voxels below −3 SD in at least one echo that hold a seed,
  dilated by 1 voxel. They leave the cortex mask before `orient`.
  - Why two thresholds: a thin vessel can be dark in different voxels in
    each echo. The first rule (below −5 SD in every echo) masked none of
    the user's third vessel (grid voxel 524, 210, 474), and the vessel
    turned l1 by a median of 22–54° within 0.6 mm of it.
  - At the site (painted box): 43,107 voxels (8.1 % of the cortex), against
    16,652 (3.1 %) for the first rule. It covers 388 of 390, 65 of 65 and
    39 of 40 dark voxels of the three vessels that the user marked.
  - Its cost: it masks 73 % of the cortex within 0.32 mm of the outside
    (the partial-volume rim along the pia), 13 % at 0.32–0.64 mm, and a
    dark band at the WM edge in places. There, cost 7 has no field (A = 0).
  - A shape filter is not used. The Hessian of the lowest z of the three
    echoes (σ 0.7–3 voxels, the scale of the largest response) classes
    voxels as sheet, tube or blob. It does not separate the vessels from
    the rest: the second vessel (along the WM edge) is 63 % sheet; most
    sheet-shaped seeds away from the pia are dark outside medium at slot
    ends and in a small pocket that the labels call cortex, which the mask
    must keep.
- The l1 streamlines are a view, not a method. At the end of the slot l1
  turns 90° (from parallel to the fundus floor to parallel to the slot
  wall) at low coherence (median 0.12). A streamline that keeps the sign
  nearest its heading cannot choose there. At the site, 86% of the
  streamlines that return to WM turned away from the nearest outside.
- `orient_arrays` (`orient.py` 281) returns the noise-whitened gradient
  tensor (6 components, NaN outside the cortex) and the local level.
  `leading`/`leading_any` (193, 360) give the leading eigenvector and the
  coherence (l1 − l2)/(l1 + l2).
- No code measures linear texture. For radial fibres, the gradients point
  across the fibre in two directions, so l1 ≈ l2 ≫ l3, and the fibre
  direction is the eigenvector of l3. For laminae, l1 ≫ l2 ≈ l3.

### Phantoms

- `prototypes/scripts/fold_phantoms.py`: `Fold` (45) with kinds "crown",
  "fundus" and "kiss". `standard(t=2.4)` gives the named set. `make(fold,
  site, noise, seed, feats)` gives the image, labels and noise. The image
  is in voxel index space (0.16 mm). `fold.column(X)` gives the true
  straight column of each root.
- `bent_phantom.py` gives the curved phantom (phase 4).
- `features(site, path)` needs `fold_features.json`, which
  `fold_measure.py` writes and the repository does not hold. Tests build
  the features inline (`test_fold_phantoms.py` 21).
- `prototypes/scripts/fold_eval.py`: `errors(fold, truth, pos, ns, vox)`
  (49) gives the largest, mean, end and contact errors in mm.
- `column_tracers.py` gives the phantom's white mesh and roots
  (`CT.white_mesh`, `CT.pick_vertices`).

### Tools

- `torch>=2.15` is a dependency. `backend.device()` (`backend.py` 48)
  picks cuda, then mps, then cpu. Torch code uses float32 (MPS has no
  float64), no sparse tensors and no atomic adds. `mesh_torch.MeshOps`
  (145) uses padded neighbour tables.
- `scripts/memguard <limit_GB> <command>`: the command gets no stdin.
  Exit 137 means memguard stopped it.
- `scripts/check` runs ruff and mypy on `src` and `tests` only. They
  exclude `prototypes`. Pytest runs only the files in `testpaths` of
  `pyproject.toml`. A new prototype test must be added there.
  `affected-tests` follows imports, so no map entry is needed.

### Labels at the site

The segmentation comes before the columns: the trial judges column
methods only where the labels are right. At the site the user painted a
crop of the dseg (`wm_paint export`, ITK-SNAP; 410 changed voxels, fluid
and cortex at the end of the slot). `wm_paint` is removed now:
`bent_box --paint <diff>` takes label edits (a `label-edits/1` diff, as the
viewer's paint mode writes) and applies them to the box's labels. The echoes and the straight rays do not
change; the rays' label intervals stay those of the dseg. The costs read
the box's labels, so the fit at the site uses the painted labels.
`bent_texture` applies the same changes to its grown labels.

## Files

Each file has one owner during a phase. Two agents never edit the same
file at the same time. All files are in `prototypes/scripts/`.

| File | Contents | Test file |
|---|---|---|
| `bent_box.py` | The roots, straight rays, faces, edges, echoes and labels of a box | `test_bent_box.py` |
| `bent_sample.py` | Trilinear sampling in torch, with gradients | `test_bent_sample.py` |
| `bent_texture.py` | The eigenvalues of the tensor and the fibre measure | `test_bent_texture.py` |
| `bent_terms.py` | The paths, soft tops, lengths, costs 1–7 and the closest-outside view in torch | `test_bent_terms.py` |
| `bent_phantom.py` | The curved phantom and its true columns | `test_bent_phantom.py` |
| `bent_fit.py` | The alternating fit | `test_bent_fit.py` |
| `bent_views.py` | The diagnostic figures | none |
| `bent_columns.py` | The command line: box, fit, views | none |

## Phases

H marks a point where the user looks at results. Do not start the next
phase until the user answers.

### Phase 0: set up

1. Make the branch `feature/bent-columns-trial` with the `start-task`
   skill. Sync the dependencies one time.
2. Add the six test files to `testpaths` in `pyproject.toml` as each one
   is written.
3. Open a draft PR after phase 1. Push one commit per phase.

### Phase 1: the box (`bent_box.py`)

1. Write `box(white, dseg, echoes, vertex, radius_mm, halo_mm)`. Follow
   the recipe of `Columns.compute`. Call `RaySampler` directly. Use
   `e_in = CAP_IN` and `e_out = CAP_OUT` for every chosen column.
2. Return the roots, normals, straight samples, label intervals, the faces
   and edges between the chosen roots, and a voxel box of the echoes and
   labels that holds every path with a margin of 1 mm.
3. Measure the peak memory under `scripts/memguard`. Record it. Set the
   cap of later runs to about 1.5 times the peak.

Tests:

- The box's columns equal the sampler's columns for the same vertex and
  radius: roots, normals, samples and intervals, exactly.
- The box's echo values at the straight samples equal
  `ColumnVolumeSampler.sample` within float32 rounding.
- Every face and edge joins two chosen roots.

### Phase 2: texture (`bent_texture.py`)

1. Call `orient_arrays` on the box.
2. Compute the three eigenvalues and eigenvectors of the tensor at each
   voxel.
3. Compute the fibre measure (l2 − l3)/(l2 + l3) and its direction (the
   eigenvector of l3).
4. Make the views: on both banks, the fibre measure, and the angle between
   the fibre direction and the straight rays, by depth.

Tests:

- A synthetic image of parallel lines gives a fibre measure near 1 and the
  direction of the lines.
- A synthetic image of parallel planes gives a fibre measure near 0.
- The leading eigenvector equals `orient.leading` on the same tensor.

**H1** (answered). No radial fibres at this scale. Cost 7 uses the laminar
texture (see *Texture*).

### Phase 3: sampling and costs (`bent_sample.py`, `bent_terms.py`)

Phases 3 and 4 can run at the same time. In phase 3, one agent writes
`bent_sample.py`. Then two agents write `bent_terms.py` in two parts, one
after the other, or one agent writes all of it.

1. Write `trilinear(volume, points)` in torch with gradients, with
   `grid_sample` or with a gather of the eight corners. Use float32.
2. Write the path: the B-spline shear, the positions x_c(s), and the
   tangent.
3. Write the soft outside map and the soft top, the mask, and the length
   L_c.
4. Write costs 1–7, each as its own function. Write the laminar field of
   cost 7 (`align_field`) and the view `closest_outside`: the smallest
   distance from a path to an outside voxel centre, which separates "almost
   reaches the top" from "censored".
5. Write cost 1 against a fixed warp. The warp comes from
   `flattening.fit` and is a constant inside the torch step.

Tests:

- `trilinear` equals `streamlines.sample` at interior points within
  float32 rounding. Its gradient equals a finite difference.
- A zero shear gives the straight samples exactly.
- On a slab of cortex with a flat top at depth h, a straight path has
  L = h within one step, and the mask is 0.5 at depth h.
- Cost 1 is zero for two identical profiles and grows with a shift.
- Cost 3: form A is zero for equal lengths. Form B is zero for lengths
  that change linearly along a mesh row.
- Cost 4 is zero for straight paths.
- Cost 5 is finite for parallel paths and becomes infinite as a triangle's
  area goes to zero.
- Cost 6 is zero for a path that leaves the WM one time.
- Cost 7 is zero for paths along v, equals c sin²θ Σ m w_k at angle θ,
  and does not change when v becomes −v.
- Every cost has a gradient that equals a finite difference.

### Phase 4: the curved phantom (`bent_phantom.py`)

This phase can run with phase 3.

1. Make a fundus with banks of 2.5 mm and 1.2 mm, in voxels of 0.16 mm,
   with labels and an image made as `fold_phantoms.make` makes them.
2. Define the laminae as the level sets of a stated depth function.
3. Trace the true columns as the orthogonal trajectories of the depth
   function, with a step of 0.01 mm.
4. State in the docstring that the truth is a choice of definition.

Tests:

- With equal banks far from the fundus, the true columns are straight
  within 0.01 mm.
- Each true column meets the WM and the pia at a right angle within 1°.
- The image's laminae lie on the level sets of the depth function.

### Phase 5: the fit, on the phantoms (`bent_fit.py`)

1. Write the alternating fit:
   1. Start from the straight rays.
   2. Sample the profiles along the current paths. Fit the warp with
      `flattening.fit`.
   3. With the warp fixed, take N steps of gradient descent on the paths
      (Adam or L-BFGS in torch).
   4. Repeat 2 and 3 until the paths move less than 0.01 mm, or for a
      stated number of rounds.
2. Record each cost at each round.
3. Set the start weights so that terms 1–7 are of equal size at the
   straight rays: w = 1/|T(p0)|. A term below 1e-9 at the straight rays
   takes w = 1/|T(p_ref)|, with p_ref a constant bend of curvature
   0.05/mm in a random direction per column; if that is also below 1e-9,
   w = 1. The arm without cost 7 sets its weight to 0 and keeps the
   others. (The first rule, 1/max(|T0|, |T_ref − T0|), gave the return
   term a weight of 1.7e28 from float32 rounding; it was replaced after one
   run of crown_w0.32, and every phantom was run again.)
4. Run the fit on the fold phantoms of `fold_phantoms.standard()` and on
   the curved phantom. Use the phantom's white mesh and roots from
   `column_tracers`.
5. Compare the path errors (`fold_eval.errors`) of the fit and the
   straight rays.
6. Run each fit with cost 7 and without it.
7. Settings of the phantom runs: 4 rounds of at most 15 L-BFGS iterations
   (strong Wolfe), float32 on cpu. Columns: the white-mesh vertices in a
   slab of ±0.7 mm, at least 1 mm from a face; those within 0.3 mm (on the
   mesh) of the slab's edge stay straight (a halo), because the barrier
   pushes an open mesh edge outward. Errors on the `pick_vertices` roots
   within ±0.4 mm and at least 2 mm from a face. Orient on the phantom uses
   the phantom's own noise (C_rest only) and no reach mask.

Tests:

- On a slab with flat laminae, the fit stays straight within 0.01 mm.
- The total cost does not increase from one round to the next.
- Two runs with the same seed give the same paths.

**H2.** The user looks at the phantom results:

- Do the columns of the fold phantoms stay straight? If they bend, stop.
  Find the cause and report it. Do not change a weight to hide it.
- How close is the fit to the true columns on the curved phantom?

### Phase 6: the fit, at the site (`bent_columns.py`, `bent_views.py`)

1. Run the fit on the box at true vertex 12182770, radius 5 mm, under
   `scripts/memguard`.
2. Run it once with each form of cost 3, each with and without cost 7.
   Labels: the painted box (`box-paint`). Texture: the two-threshold mask
   (`texture-paint-hyst`). Settings: those of the phantom runs (4 rounds
   of at most 15 L-BFGS iterations, extend 0.1, seed 0, float32 on cpu).
3. Scan each weight over a decade each way from the start weights: each
   weighted term × 0.1 and × 10, one at a time, in arm B7 (form B with
   cost 7) only. B7 was chosen because form B follows a real thickness
   gradient. The other arms are not scanned.
4. Make the views of *Diagnostics*, for the straight rays and each fit,
   side by side.
5. Sparse depths (evidence for scaling, not a refit). At the straight rays
   and at each fit's final paths, evaluate each cost with the dense samples
   (step 0.04 mm to the cap) and with sparse samples: one sample per voxel
   (0.16 mm) with a random offset per evaluation, the same depths for the
   columns of an edge or face, to the soft top plus a margin. Repeat with
   10 seeds. Evaluate a second sparse form with dense samples in a window
   round the soft top and where the labels change. Report each cost's
   sparse mean and spread against its dense value, and the columns whose
   top or return differs. Cost 4 needs no samples: compare the closed form
   ∫|x″|² of the spline with the sampled sum.
6. Mark the columns that pass bad labels: voxels below −5 robust SD in
   every echo, in 26-connected pieces of at least 200 voxels. The rule
   also marks the partial-volume rim at the pial edge, so the views count
   separately the columns that pass a piece deeper than 0.32 mm from
   label 0.
7. Phase 6 ran with H2 open: the user was away and told the coordinator
   not to stop at a checkpoint. Its results are provisional until H2 is
   answered.

**H3.** The user looks at the views. The user decides:

- if the bent columns are better at this fundus, worse, or not different.
- which form of cost 3 to keep for phase 7.
- if phase 7 is needed.

### Phase 6b: probes (`bent_probes.py`)

The user's design: straight rays are probes that collect laminar evidence
(the top and the bottom too); the columns are a discretised
diffeomorphism refined in parallel with the laminar agreement. Finding the
best cheap probe is a goal of its own.

1. The top of every probe: the first crossing of 0 by the antialiased
   field of label 0 (`baillarger.antialias.thin_plate`).
2. Probes: straight along the box normals; straight along normals
   smoothed by 24 and 97 more steps; an l1 streamline (c >= 0.3, turn at
   most 45°, else the heading is kept); the fit with only agreement,
   alignment, length and bending (bend_ac); B7.
3. Scores on the inner columns: censored, far top (more than twice the
   distance to label 0), jumps above 1 mm, drop 2, angle to l1, profile
   difference with neighbours at equal arc length, returns, and folding
   by the local Jacobian over 0.4 mm (not by triangles: slivers flip).
4. Views: the tops along the section, and maps of top, profile
   difference and angle to l1.

Tests: the first crossing is linear between samples; the l1 heading keeps
its sign and falls back to the heading; straight probe points; the local
Jacobian is 1 for a translation and negative for a mirror.

### Phase 7: a second site (only after H3)

1. If the marked M1S1 points (`prototypes/data/M1S1_points_full0.json`)
   can be placed on the current derivatives: run the fit there. Measure
   the profile split at the marked points against normal's 0.17 mm (§4c).
2. The points can be placed: the dseg and the echoes have the same grid,
   and all 23 points lie on label 42. The split measurement needs new
   code. Phase 7 waits for H3.

### Phase 8: the report

1. Write `reports/Bent columns trial.md`: the question, the code, the
   results of each phase, and what they show and do not show.
2. Put the figures in `reports/figures/bent-columns/`. Each figure is a
   PNG well under 1 MB.
3. Run `nix develop --command scripts/check --changed-since origin/main`.
4. Mark the PR ready for review.

**H4.** The user reviews the PR. A merge needs the user's instruction.

## Parallel work

- Phases 1 and 4 have no dependency on each other. They can start at the
  same time.
- Phase 2 needs phase 1.
- Phase 3 needs phase 1 for its real-data tests only. Its synthetic tests
  can start with phase 1.
- Phase 5 needs phases 3 and 4.
- Phase 6 needs phases 1, 2 (only for H1), 3 and 5, and the answer to H2.
- Phase 7 needs the answer to H3.
- Give each agent one file and its test file. The coordinator reviews
  each diff and runs the tests before the next phase starts.
- Run heavy runs one at a time, each under `scripts/memguard`.

## Diagnostics

Views for the user, not pass/fail numbers. For the straight rays and each
fit, side by side:

- two cuts at each of the two drops, through the column before the drop:
  one across the section and one along it, with the bent paths.
- the depth of the first outside per step along the section: the sawtooth,
  or its absence.
- the section's profiles, drawn as in the viewer's section pane.
- a map on the box of each column's total bend angle, and of the bend
  against depth (bent at the root, or growing with depth: the
  misalignment diagnostic's two cases, §4c).
- the lengths L_c on the mesh, and their difference from the straight
  rays' outside depth.
- where the tops land: each bank's pial area reached per unit of root area
  (crowding), and the share of fundus columns that end at the tip, on
  each bank, or censored.
- the angle between each path and the data normal where coherence is high.
  With cost 7 in the fit it is not independent; the run without cost 7
  gives the independent check (§4b, *Smooth does not prove correct*).
- `closest_outside` of censored columns.
- Dense against sparse depths (phase 6, step 5): per cost, the sparse
  estimates against the dense value; the columns whose top or return
  differs, on the mesh.
- own-bank returns, and columns that enter outside before their top
  through a hole, a tear (catalogue type F) or an excision edge (type C).

## Decisions for review (made by the coordinator)

- The root and the inward part stay fixed. Only the outward part bends.
- The soft top comes from smoothed label 0 until an outside probability
  exists.
- Censored columns get no top term. Their length comes from their
  neighbours.
- Cost 7 is the laminar alignment, in every fit from phase 5 on, with a
  run without it beside each fit.
- The cost 5 barrier is extended with δ = 0.1.
- Dark outliers are removed before `orient` with two thresholds (k = 5,
  grow = 3), set before the site fits.
- Both forms of cost 3 are fitted and shown. Neither is chosen in
  advance.
- The fit alternates: torch moves the paths with the warp fixed, and
  `flattening.fit` refits the warp. The warp is not written again in
  torch.
- Cost 1 uses the flattening features, not `gain.py`.
- The code is a prototype in `prototypes/scripts/`, with tests in
  `testpaths`. Nothing moves to `src` in this trial.

## Open

- Whether "reach the top quickly" lets columns bend through the fused gap
  above a fundus tip toward a far outside, where a contact has no fluid.
  Term 3 limits it. The cuts show it.
- How much the result depends on the stand-in soft top, once an outside
  probability exists.
- Whether the alternating fit converges, or the warp and the paths pull
  against each other. The cost record of each round shows it.
