# Crossing paths: contacts read from the laminar profile: plan

**Goal.** Run the laminar analysis straight through fused cortex, on paths that leave one bank's white matter, cross the contact and end on the other bank's white matter, and find the contact afterwards from the image profile along each path. The contact is then an output of the laminar data, not something placed beforehand by the thickness-weighted competition of fronts.

**Why.** At present the contact comes from the fronts and the thickness field (main plan, *Thickness and where the contact lies*). Its accuracy rests on the field's beliefs at fused roots, and those are biased (A6, A7, A9; findings 13, 16, 18). If the profile can say where the two stacks meet:
- laminar depth near a contact stops depending on the field's extrapolation across the evidence-free gap;
- the pial editor needs far less work in fused cortex (its base is the competition's contact: *Pial editing and edit priority*, last two Results sections);
- contacts can be told from folds, label errors and box faces by what the tissue shows, which no geometric test has done (finding 20, finding 27).

**Status.** The first round (location) has run at both faces (*Results*): at the V1V2 banks the profile places the split within about half a voxel; at M1S1 it misses by 0.5–0.8 mm, always into the thin bank, while the field's beliefs alone land within 0.07–0.12 mm. Statements marked [inference] are not from a result.

## What is known

**From the main plan (finding numbers as there):**
- **The layer normal is reliable and coherent in fused cortex** (finding 1), and runs parallel to the contact (finding 20: |cos| between layer normal and sheet normal 0.98 at contacts). It has no sign.
- **Columns up ∇T of the all-WM field cannot cross a contact** (finding 4): T has a ridge there and columns slide along it. Per-front fields signed away from one wall do cross: `per_bank_test2.py` (finding 5) traced bank A's columns from A's WM to B's WM through the V1V2 contact at j = 92, and the contact showed as a level dark line about 2.0 mm from A's WM and 2.2 mm from B's, with each bank's stria at about 0.8–1.2 mm from its own WM (`prototypes/figures/per_bank_test2.png`). Per-front fields need walls sorted into sides, which failed at PCC piece 1 (finding 21) and is not principled (*Representation*).
- **The contact has no fluid gap; two layer I's meet** (finding 17, M1S1, `a5_profiles.py`). Profiles along the contact normal at the 23 points match two open-pia profiles back to back with the fluid removed. Trend-removed echo 2 shows two bright superficial peaks about 0.6 mm apart (0.85 mm at open pia), with a notch at the points. Per point the notch is weak (depth over flank noise: median 1.1).
- **Profiles centred on the all-WM midline do not show the doubled band** (finding 20: doubled-band index −0.03 at contacts, −0.01 elsewhere). The midline is 0.6 mm from the contact at M1S1 (finding 11), so the profile missed it. A profile has to find its own centre.
- **Root-to-root length L is observed** (main plan, *What is observed*): t₁ + t₂ = L on a sheet (A5 holds at M1S1).
- **Thickness alone does not detect contacts.** L is about two thicknesses, but M1 is genuinely thick, two thin banks can sum to an ordinary value, and label errors and box faces make long columns (finding 27: 26–59 % of the label boundaries in the patches are "overlap-like", both sides past their expected thickness).
- **At the V1V2 banks the normal does not point across the contact at some points** (finding 25): the straight line along the layer normal through every point ran 4–7 mm through cortex to fluid with no far WM within 8 mm; coherence at points 1, 3, 10 was 0.15–0.38.
- **The stria** is a level dark band at 0.8–1.0 mm in 2.1–2.4 mm open V1 (finding 2), 1.5–2 voxels wide, at the minimum usable band size.
- **Minimum band 2–3 voxels** (rules). Layer I is about a voxel; it cannot carry the split alone.

**Benchmarks at the marked faces** (median |offset| from the user's points, mm):

| Face | Equal arrival | One pass, field weights [25] | Level set, posterior [29] | Per-front posterior [13, 18] | Other |
|---|---|---|---|---|---|
| M1/S1 (23 points) | 0.68 | 0.26 | 0.18 | 0.33 | contoured base with field weights 0.23 to the carved midline; field + clicks interleaved 0.11 (*Pial editing*) |
| V1V2 banks (35) | 0.42 (10 missing) | 0.40 at ratio 3; 0.16 at ratio 1 | 0.40 | 0.11 | |

**Code to reuse** (`prototypes/scripts/`):
- `curv_lib.py`: `load` (box from the full volume, trend-removed echoes `det[1]`, `det[2]`, face distance), `tensor_normal` (normal and coherence; the tensor components come from `flatten_fused2.tensor`).
- `a5_profiles.py`: `sample` and `disc` (disc-averaged profiles along a line), the open-pia reference profiles aligned at the cortex/fluid boundary, the notch measures.
- `onepass_lib.py`: `setup`, `run`, `contacts`, `open_thickness` (visible tops with `along=True`), `score_points_sides` (finding 25's score).
- `click_evidence.py`: `orientation` (the one pass's metric direction), `Thickness` (the field with finding 26's guards, μ and σ per root patch), `end_labels`.
- `per_bank_test2.py`: the "observed contact" (darkest point between the two banks along a crossing column), a template-free estimator to keep beside the new ones.

## The idea

A path that follows the layer normal from a root on bank A crosses the layers VI → I, crosses the contact, and continues I → VI on bank B to B's white matter. Its length is L. Along it:

- **At a contact** the profile is bank A's stack followed by bank B's stack reversed: mirrored about the contact, with each half stretched to its own thickness. The mirror point is set by every band with contrast, appearing twice, not by layer I alone.
- **At a fold tighter than the ribbon** the front meets itself inside the stack; the layers continue round the bend. A path along the normal turns to run along a layer there, or the reversal is at a deep layer, not at the top [inference].
- **Where the far "WM" is not white matter** (a label error, a dark band labelled WM) or **where one thick stack runs to the far end** (M1), there is no mirror: one stack stretched over the path.

So the profile gives two things: **where L splits** (location), and **whether the path crosses two stacks at all** (detection).

The principled form:
- **Location: a posterior over the split.** The profile is a likelihood for the split point x (t_A = x, t_B = L − x); the thickness field's belief is the prior. Where bands are clear the profile dominates; where the profile is flat the fronts' estimate stands. This is the pairwise posterior of the main plan with one more likelihood term, so it nests the current answer.
- **Detection: a likelihood ratio** between "two mirrored stacks" and "one stack" (and, separately, "reversal inside the stack": a fold).
- **Templates per bank,** learned from that bank's own open cortex near the path.

## Method

### 1. Crossing paths

A path is a streamline of the unsigned layer normal, traced in 3D.
- **Direction without sign:** interpolate the structure tensor's components (not the eigenvector) at each point and take the leading eigenvector there, then flip it to agree with the previous step. The first step points away from the WM (EDT gradient). Because the layers run parallel to the contact, the sign carries through it [inference; to be checked on the paths].
- **Steps:** RK2, 0.25 voxel.
- **Stops,** each recorded: WM reached after at least 0.6 mm (a crossing path); fluid (an open column); missing tissue; within 2 mm of a box face (left out, as the rules say); a turn of more than about 60° within 0.3 mm (where the path would run along a layer: a fold or a junction); a length cap of 8 mm.
- **Low coherence** is recorded along the path (fraction of length with coherence below 0.3), not a stop.

Path families (an open decision, D1):
- **(a) Normal streamlines** as above. Pure data; fails where the normal does not point across (V1V2 points 1, 3, 10).
- **(b) Column pairs from the one pass:** at a contact pair of the equal-arrival pass, trace each side's column back to its root and join them. Uses the existing geometry; the join point is the pass's contact, but the profile does not depend on it if the two columns are nearly collinear [inference].
- **(c) Diagnostic only, at marked points:** the line along the normal of a plane fitted to the user's neighbouring points (`a5_profiles.py`). It uses the truth to choose the path, so it scores only "is the split visible when the path is right", not the method.

Seeds: in the first tests, every marked point, traced both ways (one path through each point). For whole sheets later, one path per root patch on the WM, kept if it ends on WM.

**Lateral averaging.** One voxel-wide path is noisy (finding 17: notch depth over noise 1.1 per point). Average over a disc of radius about 0.3 mm perpendicular to the path at each step (`a5_profiles.disc`), as finding 17 did. Larger discs blur a curved contact; the radius is a setting to view, not to tune.

### 2. Profiles along a path

- **Features:** echo 2 and echo 3 log magnitudes, trend removed (the display and tensor features), and raw log echoes for display. Echo 3 has more stria contrast but more grain; the two enter the likelihood jointly with their own noise.
- **Coordinate:** arc length s from root A, 0 to L.
- **Trend removal near the contact** uses masked normalized convolution over cortex, so at a closed contact the blur spans both banks, while at open pia it stops at the fluid. Templates from open tops and profiles at contacts therefore see different trend removal in the top ~0.3 mm [inference]. Raw log profiles, or trend removal from each bank's own side, are the check.

### 3. Templates per bank

- **Source:** visible tops of the same bank (one-pass column tops, `open_thickness(along=True)`), roots within a few mm along the white surface of the path's root (a5 used tops 2–6 mm from the points), at least 2 mm from a face. Each open column gives a profile from its root to its pia, of known thickness t.
- **Bank identity:** the root's position on the white surface; tops are taken by geodesic distance from the path's two end roots, so no sides are sorted (as `score_points_sides`).
- **The template** is the median profile over those columns as a function of depth. Its noise is the spread of each column's own samples about that median (resampling the columns first would average neighbouring samples and shrink it by about 20 %, found on the synthetic test), plus the median's own sampling variance. How depth is defined is an open decision (D3):
  - *linear:* profile at s/t (distance; the current fallback for role 1);
  - *top-anchored:* the superficial part keeps its absolute thickness, only the deep part stretches (macaque: layers I–IV about constant, V–VI carry the gyral thickening; Hilgetag & Barbas 2006);
  - *equal volume* once an estimator of it is validated (not yet: main plan, *The fallback*).
- **Guard against wandering tops** (finding 29: 7.5 % of V1V2 tops follow d sideways about twice the straight distance): keep tops whose path length over straight distance to the root is below about 1.3, the guard finding 29 proposed.

### 4. Models along one path

With L fixed by the path:

- **H_contact(x):** s < x is template A stretched to t_A = x; s > x is template B, reversed, stretched to t_B = L − x. Per side, a gain and offset on each feature (intensity varies across the box) fitted in closed form.
- **H_fold(x, d):** as H_contact, but the reversal is at depth d < t on both sides: the superficial part of the templates never appears [inference: the expected profile of a tight fold; to be checked on a synthetic fold and the PCC piece 3 bend].
- **H_one:** one template stretched over the whole path, no reversal (label error, thick stack, a far end that is not WM).

**Likelihood:** residuals against the template, Gaussian per sample with the template's per-depth spread (inflated for the disc average's correlation along the path), or Student-t if a few voxels dominate (D5). The log-likelihood ℓ(x) is computed on a grid of x in 0.02 mm steps over [0.3 L, 0.7 L] widened to the whole path where the prior is broad.

**Estimators of the split, recorded separately** (principle, roles, estimators):
- **E0, symmetry, no templates:** the best mirror point of the profile with itself, allowing one stretch ratio between the halves (correlation of p(x − a u) against p(x + b u)). Uses no open cortex, so it works where tops are scarce.
- **E1, profile likelihood:** argmax of ℓ(x) with a flat prior.
- **E2, posterior:** ℓ(x) plus the prior from the thickness field, log N(x; μ_A, σ_A) + log N(L − x; μ_B, σ_B). Mean, MAP and SD per path. With a flat profile this is the main plan's pairwise posterior (σ² share, COHIBA).
- **E-dark:** the darkest point near the middle (`per_bank_test2.py`'s observed contact), as the existing template-free baseline.
- **E1 −dark, an ablation:** E1 with the samples within 0.24 mm of E-dark's point left out, so the split comes from the bands alone, not from a residual dark line.
- **Prior only:** the fronts' answer on the same path, for comparison.

**Detection** per path: log [max_x L(H_contact)] − log L(H_one), and the same against H_fold. Penalised for the extra parameter (BIC-like), or, better, calibrated against the distribution of the same statistic on open columns and on paths through single stacks (test 4).

### 5. From paths to a contact surface (later)

The split points of neighbouring paths must form a smooth surface (no faults within a stack: the contact is between stacks, but each stack's top is continuous). A 2D smoothing of x over the sheet, with each path's posterior SD as its weight, is the obvious step; it is left until the per-path results are seen.

## Tests, in order

1. **Synthetic, exact answers** (pytest in `prototypes/scripts/`, beside `test_onepass_lib.py`):
   - two apposed slabs of unequal thickness (12 and 18 voxels) with a banded profile (a stria-like dark band, a bright superficial band) plus noise at the measured flank noise; the split must come out at 12 within the noise's expected error, and E1's SD must be calibrated over noise draws;
   - the same with no bands (flat profile): E1's likelihood is flat and E2 returns the prior exactly;
   - one slab running into a "WM" label at its far end: H_one wins;
   - a fold tighter than the ribbon: the tracer stops on the turn, or H_fold wins;
   - the tracer: sign carried through a contact; a known normal field reproduced.
2. **Open cortex: spliced columns, with the truth known** (on each test box). Two open columns whose roots are at least 8 mm apart are joined pia to pia (A's profile, then B's reversed), so the split is at t_A exactly. Each side's templates come from open columns within 6 mm of its root along the white surface, leaving out those within 1 mm. The prior is a stand-in, the median and SD of those columns' t (the field saw these tops). The error is the profile's accuracy where the truth is dense; it is optimistic, because two open pia meet with their own edge signal, which a fused contact may lack (Risks).
3. **V1V2 banks, 35 points** (`V1V2_banks_points.json`; box and conventions as `onepass_face.py`). Paths through each point (families a, b, and c for diagnosis). The stria should appear twice, mirrored. Score each estimator's split against the points (offset along the path) and against the fronts' contact on the same path.
4. **M1S1, 23 points** (`M1S1_test_points.json`, "gap_all", 0-based crop coordinates from full_lo [419, 229, 439]). Noisier, agranular M1 against S1; M1's radial fibres give laminar contrast. Same scores. The unequal banks (2.0 and 3.3 mm) are where the profile matters most, since the prior is biased on the thin front.
5. **Detection,** after location works: the statistic on (i) the marked contacts; (ii) the user's non-contact readings (finding 20: 1,282 voxels, PCC piece 3's bend, the lower part of PCC piece 1); (iii) finding 27's overlap-like boundaries that cut across a gyrus in V1V2 (label errors); (iv) open columns (paths that end in fluid, reflected at the pia, as a null). Views first; the user's readings decide what the classes are.

## What would count as evidence

Views first (the rules: diagnostics, not pass criteria):
- **Profile stacks** per face, one row per point, aligned at the point (as `a5_profiles.png`), with each estimator's split and the fronts' contact marked. A mirrored stria at V1V2 is visible or not by eye.
- **Planes** through the points (echo 2, trend removed) with the paths, the split points and the fronts' contact.
- **Per point:** ℓ(x), the prior and the posterior on one axis.

Numbers as context:
- median |offset| and IQR per estimator against the benchmark table;
- E1's z-score SD (offset / its SD), and test 2's error on open cortex, to say whether the profile's uncertainty is honest;
- where E1 and the prior are both confident and disagree, which is right by the user's reading;
- for detection, the statistic's distributions per class, with no threshold chosen.

What would change the plan:
- If test 2 shows the profile cannot place a visible pia within about 0.2 mm, the location term is weak everywhere and its main use is detection.
- If the split is visible with path family (c) but not (a) or (b), the profile works and the paths are the problem (expected at V1V2 points 1, 3, 10).
- If E1 beats the posterior at M1S1, the field's bias (A9) is costing more than its information gives, which bears on the main plan's contact model.

## Costs

- **Per face:** box load and tensor as `onepass_face.py` (minutes), one equal-arrival pass for family (b) and for tops (under 10 s), the thickness field for the prior (most of the time and memory: finding 27's peaks are the curvature and the field). Tracing tens to thousands of paths and the 1D fits are cheap.
- **Memory:** about 5 GB at M1S1's 141-voxel box (finding 27); runs under `scripts/memguard 8 ...`, one at a time, `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4`.
- **New code:** a tracer with tensor interpolation and sign continuity, the template builder, the 1D likelihoods and estimators, the synthetic tests; perhaps 400–600 lines in `prototypes/scripts/` (`crossing_lib.py`, `test_crossing_lib.py`, `crossing_face.py`). Nothing in `src/`.

## Risks

- **The normal does not cross the contact where coherence is low** (finding 25, V1V2). Family (a) fails there; (b) inherits whatever the pass's columns do.
- **Weak bands.** Layer I is one voxel; at M1S1 the per-point notch is weak and M1 is agranular. The likelihood may be flat, and E2 then returns the prior (by design, but then nothing is learned).
- **The superficial layers are thinner at the contact than at open pia** (finding 17: by about 0.1–0.15 mm per side). Templates from open tops misfit the top of each stack, biasing the split toward or away from a bank by up to that. Not to be absorbed by tuning; a known, measured effect to state beside the results.
- **Trend removal differs at open pia and at a closed contact** (Method 2).
- **Templates vary along a bank** at areal borders (facies changes: the stria ends at V1/V2; layer IV pinches out between S1 and M1). Local templates limit this; a path whose two ends lie in different areas from their tops gets the wrong template.
- **Bok's effect moves bands with curvature** (finding 10: the stria fraction has ρ +0.63 against curvature on open V1), so a template from flatter open tops is stretched wrongly at a curved fused root. The stretch model (D3) is where this enters.
- **Circularity in the first tests:** paths through the marked points use the truth to choose where to look. Family (c) uses it to choose direction too. Whole-sheet seeding from roots removes this later.
- **The prior is biased** (A9): E2 inherits the field's drift where the profile is weak.

## The column volume

**Goal.** Re-index each box's cortex as a volume of columns (traces): one per vertex of the white mesh, sampled along the column in raw millimetres from the root. Seismic flattening analyses such a cube (horizon tracking, facies, slices at fixed relative depth). Here it is first a diagnostic: bands followed laterally from open cortex into fused cortex, the contact estimates seen as surfaces, and where the columns themselves fail.

**What it is not.** The volume holds 3D samples re-indexed, not slices: every sample keeps its 3D position. Nothing is fitted in it yet.

### Axes

- **Lateral:** the vertices of the white mesh (finding 28's: contoured from the WM label, relaxed, smoothed along its edges by twicing at 2 voxels, 0.32 mm; smoothed outward normals). Neighbours come from the mesh's edges. Local 2D charts for displays and image-style filters come later.
- **Depth:** arc length from the root in 0.04 mm steps (one tracer step), up to 8 mm. Samples past a trace's end are masked.
- **Features:** echo 2 and echo 3, trend removed, sampled at the trace's centre line, with no disc average. Lateral averaging belongs in the volume, across neighbouring traces along the mesh, where it can be chosen and undone.

### Traces and their failures

- **Primary tracer:** the streamline of the unsigned layer normal (`crossing_lib.trace`), started at the vertex along its outward normal; a WM label is ignored for the first 0.3 mm.
- **Fallback, where the primary stops on a turn or the length cap:** ascent of the equal-arrival pass's arrival from the vertex. It stops at fluid, at WM, or where the pass's label belongs to a root more than 5 mm away along the white surface (the pass's contact).
- **Recorded per trace:** which tracer made it; each tracer's outcome (fluid, WM, turn, length, face, missing, other); length; the root and end positions; and the end root's distance from the start root along the white surface.
- **Trace classes:**
  - *open:* ends in fluid;
  - *crossing:* ends on WM whose root is ≥ 5 mm away along the white surface (the gap of *Results*: under 1.9 or over 14 mm);
  - *own bank:* ends on WM within 5 mm, back on its own bank;
  - *to the contact:* the fallback stopped at the pass's contact;
  - *failed:* face, missing, other, or both tracers failed.
- **Faces:** traces whose root is within 2 mm of a box face are kept but flagged, and left out of every summary (the rules).
- **The check first:** where do primary traces turn: depth along the trace, coherence there, and distance from the WM. A turn just above the WM points at the tracer's start (the normal is unreliable at WM edges); a turn mid-column at low coherence is what the fallback is for.

### Horizons (per trace, in mm from the root; NaN where not defined)

- **top:** the length of an open trace.
- **far WM:** the length of a crossing trace (its L).
- **contact estimates, on crossing traces:**
  - the one pass's label boundary along the trace, equal and field weights (sides by root distance, as in *Results*);
  - the field prior's pairwise posterior on L (μ, σ at the two end roots' patches);
  - the profile's split, E1 top-anchored, with templates from open traces in the volume within 6 mm of each end root. This one is costly, so it is computed on a random subsample of crossing traces (about 3,000 per box) and marked as sparse.
- **contact from the fallback:** the length of a trace stopped at the pass's contact.

### Views

- **Sections:** along the white surface through each marked face. On each of the three planes holding most points, and for each of the two banks, the mesh vertices near the plane are ordered by distance along the mesh from a vertex facing the points. The image is echo 2 (trend removed), with lateral position along the white surface (mm) and depth from the root (mm). Over it go every horizon, your points' depth where a trace passes within 0.3 mm of one, and a strip marking each trace's class and tracer. Bands followed from open cortex into the fused stretch are the thing to read.
- **Roots on the planes:** the traces' classes and tracers at their roots, over echo 2.
- **Numbers:** class and tracer counts; where primary traces turn; horizon depths at the marked faces against *Results*.

### Tests

- Synthetic (`test_crossing_lib.py`, extended): on apposed slabs and a slab under fluid, every vertex trace has the right class and length; the fallback reproduces a slab's thickness; padding and masks are right; the section ordering of a straight line of vertices is monotone.

### Costs and limits

- A 141³ box has about 57–70k WM seed voxels and several times as many mesh vertices. If the mesh has more than about 300k vertices, traces are taken at a subset of vertices spaced about 0.16 mm apart (one voxel) along the mesh, recorded, so that the depth samples stay under about 2 GB.
- Runs under memguard, 4 threads, one at a time.

### Local charts: `chart.py`

A local 3D chart of the volume around a point, written as NIfTI cubes for ITK-SNAP.
- **Run** (from the repository root, after `column_volume.py` has run for the site):
  `nix develop --command uv run python prototypes/scripts/chart.py prototypes/scripts prototypes/data <site> <i> <j> <k> <radius mm>`
  The point is in ITK-SNAP coordinates (1-based) of `prototypes/data/column_volume_<site>_echo2_detrended.nii.gz` (written on first use). For `V1V2_banks` that is the same grid as `V1V2_banks_echo2_detrended.nii.gz`, so your marked points can be used as they are. For M1S1 the volume's box is centred on the points, not the test crop. `--zero-based` takes 0-based box voxels. A chart takes a few seconds.
- **What it does:**
  1. Takes every trace (roots ≥ 2 mm from a face) with a sample within the radius.
  2. Groups their roots into banks: the pieces of the roots' footprint on the white mesh (vertices within 1 mm of a root); pieces under 20 traces are dropped.
  3. Charts each bank: every root at its geodesic distance along the mesh from the root of the trace nearest the point, in the direction of its projection on that root's tangent plane. This approximates the exponential map: distances from the centre are exact, and angles are distorted where the surface bends round within the radius (`crossing_lib.exp_chart`, tested on a plane and a cylinder).
  0. Reports the point's tissue (cortex, fluid, WM, missing tissue, other) and the distance from the point to the nearest trace, and to each bank's anchor trace. Over 0.3 mm it warns: the chart is then centred on the nearest trace, not on the point. Outside the cortex this is the usual case: in fluid 1 mm from the cortex the anchors were 0.8–0.9 mm away; in WM 1 mm from the cortex, 1.0–1.6 mm away, in four banks. With no trace within the radius, nothing is written.
  4. Fills a cube: u and v on a 0.16 mm grid, depth from the root on a 0.04 mm grid. Each (u, v) column holds the nearest trace within 0.24 mm in the chart, copied as is (not interpolated), or nothing.
- **Outputs**, in `prototypes/data/`, per bank: `chart_<site>_<i>_<j>_<k>_r<radius>_bank<n>_echo2.nii.gz` and `_echo3`; `_overlay` (one-voxel horizons at each trace's depth: top, far WM, the fallback's contact, one pass equal and field, field prior, profile split; and your points where a trace passes within 0.3 mm), labels in `chart_labels.txt`; `_class` (each column's trace class), labels in `chart_class_labels.txt`; and a JSON with the point's tissue, the distance to the nearest trace, and per bank the counts, the anchor trace's distance from the point and its ITK-SNAP column. `column_volume.py` now also saves the tissue labels; for volumes built before that, `chart.py` rebuilds them from the box. The cube's spacing is 0.16 × 0.16 × 0.04 mm and its origin is the chart's corner, not scanner space: u and v are in-plane in ITK-SNAP's axial view and depth is the third axis, so the other two views are sections.
- **Checked** at V1V2 point (70, 75, 83), radius 3 mm: 3,017 traces in three banks (2,074, 918 and 25 traces). Sections through the point show the contact band at about 2 mm with the stria above it and mirrored below. At M1S1 point 13, radius 3 mm: two banks (2,059 and 420 traces; 64 % and 41 % of the cube's columns filled).

### Open (for the user, later)

- Lateral smoothing in the volume: along the mesh, its scale.
- Charts: the exponential-map approximation distorts angles on tightly bent patches; a least-squares conformal flattening would not, if that matters at the radii used.
- Whether to add the band-aligned depth warp, the main plan's z_c(T). Built and run at both faces: *Flattening in the volume* (method) and its first run in *Results*.

### Flattening in the volume

**Goal.** Fit one depth warp per trace so that bands line up across neighbouring traces: the band-aligned depth warp, z_c(T), of the main plan, fitted in the column volume. Seismic flattening (Lomask 2006; Wu and Hale 2015, horizon volumes) moves samples only along the trace. Here that is not a restriction, because the trace already follows the column. It is an estimator for role 1 (depth within a column); the contact is not used.

**Unknown.** For each trace v, the depth z_v(τ) (mm from the root) of the horizon at relative depth τ, on a τ grid of 0.08 mm. This is the horizon-volume form: a layer that thins to zero is ∂z/∂τ = 0, not a jump in τ. Written z_v(τ) = τ + r_v(τ).

**Data term: shifts between neighbouring traces.**
- Neighbours: traces whose roots are within 0.3 mm of each other and whose white-mesh normals agree (dot product over 0.7), so the two sides of a thin WM blade are not joined. Roots within 2 mm of a box face are left out, as everywhere.
- For each pair (v, w) and each depth z on v, the shift s_vw(z) that best maps w onto v near z: f_w(z + s) ≈ f_v(z). It is estimated by iterated gradient matching (Lucas–Kanade along the trace), for echoes 2 and 3 together, with derivative scale 0.16 mm and a Gaussian window of 0.32 mm, the scales of the structure tensor.
- Weight: the window's gradient energy (normalized) times the local correlation after the shift (0 if negative). Samples past a trace's end have no weight.

**Fit.** Weighted least squares, all levels and traces at once:
- r_w(τ) − r_v(τ) = s_vw(τ + r_v(τ)) for every pair and level (the shift is read where v's horizon is: Gauss–Newton outer iterations, as Lomask);
- ε (r_v(τ + Δ) − r_v(τ)) = 0 along each trace (Lomask's vertical regularization; ε = 0.2), which also carries the warp through levels without data;
- r_v(0) = 0 (the root is τ = 0);
- the mean of r over traces is 0 at each level, which fixes τ as the mean depth of its horizon in the box.
Solved by conjugate gradients on the normal equations.

**What it gives:**
- each trace resampled at constant τ: the flattened volume, read as a Wheeler section;
- the residual per trace and level (the fit's misfit): where a continuous warp disagrees with the data. By the no-faults rule a large jump between neighbours points at an upstream error;
- folds: levels where ∂z/∂τ ≤ 0, reported, not enforced (the first version).

**Tests (exact answers, `test_flatten_volume.py`):**
- the shift estimator recovers a known constant shift and a shift that varies with depth;
- the fit recovers a known warp on a grid of synthetic traces: a thickness that varies sideways and a band that rises in the middle, with traces of different lengths.

**Before the next flattening: the first pial surface.** The cross-correlation must not see anything past each trace's first pial surface: the fluid, or the contact and the far bank (the user, 2026-10-27). That surface is found as one horizon, in raw mm, by *The first-pia horizon* below.

**Views and checks, on the V1V2 banks first:**
- sections through the marked planes, raw depth beside τ, horizons mapped to τ;
- the stria: its depth on each open trace in raw mm and in τ (the spread should shrink), and the stacked profile at constant τ against the stacked raw profile (bands should sharpen);
- the residual and the folds on the roots;
- then M1S1, where the pinch-out of layer IV is the test the horizon-volume form is for.

### The first-pia horizon

**Goal.** On every trace, the depth c_v (mm from the root, raw) of the first pial surface: the open top, or the contact. It is a horizon across the traces. Thickness can change quickly but not abruptly (the user), so neighbouring traces' c_v are tied by a bound on their difference, not a quadratic penalty. A quadratic penalty reverts to the mean across an evidence-free gap (finding 16); a bound costs nothing inside it.

**Unknowns.** Every trace with its root ≥ 2 mm from a face, on depth labels from 0.4 to 5.0 mm at 0.04 mm.

**Open traces: soft anchors** (the user's choice, 2026-10-27, after the first run). An open trace ends in fluid, but it may cross a contact on the way, and then its top is not the first pia (first run, below). So an open trace's top is a strong preference, not a fixed value:
- leaving it costs P, more than any trace's whole range of data cost (1 + the largest max − min of D over traces). The cut therefore keeps as many open tops as the bounds allow, and among those choices the darkest horizon;
- a top beyond 5.0 mm has no label and is always overridden;
- bounds apply between every pair of neighbours, open traces included.

**Trend and g, re-measured once.** Stage 1 uses the trend and g from all open tops. Its open traces that keep their top (within half a label) are *consistent*; the trend and g are measured again on them, and stage 2 is solved with those. Only once: consistent pairs satisfy the bound by construction, so iterating would shrink g each time without converging to anything meaningful.

**Data cost per trace.** The mean of echoes 2 and 3 (trend removed, each divided by its SD), smoothed along the trace at 0.08 mm: darker is cheaper. Both the fluid and the contact are dark at the first pia (finding 17's notch at M1S1, the dark band at the V1V2 banks). Zero past a trace's end. This is the first version; a mirror-symmetry term for crossing traces is a candidate.

**Curvature trend.** log t = m + a·H, fitted on the open tops, with H the white-mesh mean curvature at the root (`baillarger.frame.mesh_curvatures`, 0.8 mm, as the thickness field). μ_v = exp(m + a·H_v). The bound applies to thickness after the trend is removed, so a change that folding explains costs nothing extra.

**The bound.** For neighbouring traces v, w (the flattening's pairs: roots within 0.25 mm, normals agreeing) at distance d:
|(c_v − μ_v) − (c_w − μ_w)| ≤ g·d.
- *g is measured, not chosen:* the slope |Δ(top − μ)| / d over pairs of open traces whose roots are 0.8–1.2 mm apart (same side, normals agreeing), at its 99th percentile. At 1 mm the top's voxel noise adds little; the slopes at 0.2, 0.5 and 2 mm are reported beside it.
- The bound is hard between every pair of neighbours. Disagreeing open tops show up as overridden anchors (QC).

**Solver.** Minimum cost over all c_v subject to the bounds, exactly and globally, as a minimum closed set (Li et al. 2006, the optimal-surface construction of layered OCT segmentation; Picard 1976) solved as a minimum s–t cut (`scipy.sparse.csgraph.maximum_flow`). The construction is exact for hard bounds of the form c_w − c_v ∈ [lo, hi] on integer labels.

**Tests (`test_pia_lib.py`):** against brute force on small graphs with random costs and asymmetric bounds; anchor penalties.

**Scoring.** At the 35 V1V2 points, as the column volume's horizons are scored: traces of the classes crossing and to the contact that pass within 0.3 mm of a point; the median |c_v − the point's depth| over each point's traces, then the median and 90th percentile over points. Also reported over traces of every unknown class.

### The joint warp: stretch and cut together (the current approach)

**Why** (the user, 2026-10-28). Three things are being estimated at once: (1) laminar features that are consistent but may fade in or out; (2) the stretch that aligns them; (3) the cut, where the pial surface is, known on columns with a clear pia. The pia exists in (almost) every column: its appearance can change gradually, its depth only gradually. Fitting these separately failed at plane 54 (*Several surfaces in one cut*, below): the contact's dark line fades out while its position continues, and a darkness horizon then jumps to the far stria.

**Unknown.** One monotone warp per column, z_v(τ), τ ∈ [0, 1], in mm from the root: z_v(0) = 0 at the WM, **z_v(1) = c_v, the pia**. The stretch is the shape of z_v; the cut is its top level, not a separate variable. Features sit at fixed τ; only their strength may vary laterally (fade). The pia's own look (fluid edge, dark line, merged layer I's) is a feature at τ = 1 whose appearance is not required to match.

**Terms:**
- **Conservation, inside each bank.** Between neighbouring columns at the same τ, the shift equations of *Flattening in the volume*, z_w(τ) − z_v(τ) = s_vw(z_v(τ)), counted only where both samples lie inside their own column's (δ₀, 1 − δ₁): tapered to zero over about a voxel below z(1) (the pia's own appearance: fluid partial volume, the dark line) and down-weighted in the first few tenths of a mm above the WM (partial volume, the tracer's turn spike). Position is conserved, not strength: a fading band lowers its own weights (correlation).
- **Two-sided conservation at contacts** (the user's choice). Tissue past one bank's cut belongs to the other bank. On a trace of length L that crosses a contact, c_A + c_B = L: what one bank excludes the other must include. This blocks the trap of masking (pulling the cut in to drop hard samples), and it is what makes a contact a contact: two sequences meeting at a bounding surface, each conserved on its own side. First version: c_A(v) + c_B(w) = L_v, with w the far bank's column whose root is nearest the trace's far end (normals facing), weighted by how antiparallel the two traces are. Later: the far bank's samples on A's trace, read in reverse, as conservation equations with w's column.
- **Anchors at τ = 1:** open tops, robust (iteratively reweighted, Huber or Cauchy), so that tops that crossed a contact lose their pull, as the soft anchors did in the cut.
- **Lateral smoothness of z_v(τ) at every level** (per-layer thickness changes gradually), detrended by curvature, robust, weak: for stretches with no correlated feature.
- **Vertical prior: equal volume.** Where nothing correlates within a column, z_v(τ) follows equal volume: the data decide where there is contrast.

**Solve (the user's choice, 2026-10-28): levels by QPBO, then the flattening between them.**
- **Levels.** The warp is solved first at a few levels, τ = 1/3, 2/3 and 1 (fractions to start; levels at features later). Each level's depth z_v^ℓ is a label on a grid of 0.04 mm, within ±1.0 mm of a starting estimate (the current horizon, scaled by τ) so that the graph fits in memory; a solution on a window's edge is reported and the window moved.
- **Terms, all functions of a difference except the ties** (Ishikawa 2003: hard bounds and L1 hinges on the difference of two labels are exactly submodular):
  - *conservation at each level:* w·|z_w^ℓ − z_v^ℓ − ŝ|, with ŝ = s_vw at the current z_v^ℓ and w its weight times the taper (zero within about a voxel below z(1), lower just above the WM). Linearized at the current warp; the shifts are read again at the new warp (Gauss–Newton);
  - *lateral smoothness at each level:* λ·|z_w^ℓ − z_v^ℓ − τ_ℓ·Δμ| (Δμ the curvature trend's difference), weak;
  - *ordering in a column:* z^{ℓ+1} − z^ℓ ≥ 0.1 mm (hard);
  - *vertical prior:* w_v·|z^ℓ − z^L − (τ_ℓ − 1)·ẑ^L|, linearized at the current pia ẑ^L (distance for now; equal volume when its estimator is validated);
  - *anchors:* a unary cost at z^L on open columns, robust (a constant beyond a tolerance, the soft anchor of the cut);
  - *the two-sided tie* w_t·|c_A + c_B − L|: a sum, and the banks are joined around their fold, so no flip of labels makes the whole problem submodular.
- **QPBO** (Kolmogorov and Rother 2007, roof duality; `thinqpbo`, double precision) solves all of it in one maximum flow on the binary encoding y = [z ≥ k] (the submodular graph of the difference terms, plus each tie as a non-submodular pairwise term). It returns a **partial labelling that agrees with some global minimum** (weak persistency), and leaves the rest undecided; QPBO-I (`improve`, Rother et al. 2007) then completes it from the optimum without the ties, never raising the energy. The undecided columns are reported: they are where the data and the ties disagree, and the first edit-priority map.
  - `slgbuilder` (Jeppesen et al. 2020) was read and not used: its layered helpers assume objects on a regular N-D grid; our columns sit on an irregular mesh graph with per-pair targets and sum-type ties.
  - Dual decomposition by subgradient (built first, then dropped): its subproblems were exact, but it converged slowly (within about 0.015 of the LP bound after 3,000 iterations on small models, each iteration a full maximum flow at scale), and a Polyak step aimed at the primal cycled.
- **Coarse to fine, and pruning by persistency** (the user, 2026-10-28; for size and speed):
  - *Coarse pass:* labels at 0.16 mm in a window of ±1.0 mm about the starting estimate. *Fine pass:* labels at 0.04 mm in a window about the coarse solution: ±0.16 mm (one coarse step) where the coarse pass certified the variable, ±0.48 mm where it did not. A solution on a window's edge (other than the label range's own ends) is counted, its window re-centred, and that pass repeated (up to three times). Costs are per mm (each hinge weight times the step), so the passes optimise the same energy at two resolutions.
  - *Pruning:* after QPBO on the fine pass, the certified variables are fixed (their labels agree with a global minimum of that pass's energy), their terms with uncertain neighbours become unary costs, and QPBO-I runs on the uncertain variables only. The coarse pass's certification is a resolution heuristic (it sets the window width), not a guarantee about the fine optimum.
  - *Subsampled traces* for the first real test: one trace per 0.3 mm of white surface (greedy), neighbours within 0.5 mm with normals agreeing.
  - *Weights* (first version, stated, to be checked on the synthetic volume): conservation w = the pair's shift weight (0–1) per mm; lateral smoothness 0.05 per mm; vertical prior 0.1 per mm; open-top anchor: 0 within one label of the top, then rising linearly to a cap of 5 at 1.0 mm (a capture range: a flat cost beyond one label pulled nothing when the top lay outside a window, and in a test a lone column then slid to the WM); tie 1 per mm times how directly the two roots face each other.
- **Fill.** The continuous warp z_v(τ) is then fitted between the levels by the flattening's least squares, with the levels as strong anchors and samples past z(1) masked.

**Tests first (exact answers).** A synthetic volume of two banks meeting at a contact, with layer thicknesses that vary sideways, a stria in each bank (so the far stack, mirrored, appears past the contact), open tops on part of each bank, and a contact line that fades out along the contact. z(1) must follow the true contact through the fade; with the two-sided tie off, the shrink trap must show (the check that the tie is doing the work).

**Then:** V1V2 (plane 54, scored on every trace), then M1S1, where there is no stria and the stretch must come from M1's and S1's own bands.

## Results

### Code and synthetic tests

`crossing_lib.py` (tracer, disc profile, depth maps, templates, estimators) and `test_crossing_lib.py` (11 tests, all pass; run with `nix develop --command uv run --with matplotlib pytest -q prototypes/scripts/test_crossing_lib.py`). The driver is `crossing_face.py <figures> <scripts> <data> <site>`; a run on the V1V2 banks box takes about 4.5 minutes, peak 4.4 GB under memguard, 4 threads.
- **Tracer:** a streamline started on bank A through a contact between apposed slabs of 2 and 3 mm carries its sign through the contact and stops within one step past B's WM face. Stops at fluid, length and a turn work.
- **Split, exact model:** with templates and contacts generated from the model each stretch assumes, E1 finds the split (median error under 0.1 mm at a noise half the band amplitude) and is calibrated: the z-score SD over 60 draws is 1.01–1.09 once the template's noise is measured from the columns' own samples. Measured after resampling, it was 2.4 (the fix in Method 3).
- **Flat profile:** E2 returns the pairwise posterior to within 0.005 mm.

### V1V2 banks, 35 points (box 141³ from the banks crop; every point ≥ 7.0 mm from a face)

**Open columns:** 9,003 traced from WM seeds; 4,063 reached fluid, 2,073 stopped on a turn, 1,393 returned to WM, 1,077 met a face. 4,058 kept after the wander guard, t 1.36 / 2.04 / 2.60 mm (5th / 50th / 95th).

**Test 2, 400 spliced open columns** (error of the split, mm; the true split known):

| Estimator | Median | IQR | \|err\| median | 90th of \|err\| | z SD (robust) | Posterior SD, median |
|---|---|---|---|---|---|---|
| E0 symmetry | +0.004 | 0.13 | 0.068 | 0.93 | | |
| E-dark | −0.02 | 0.04 | 0.020 | 0.10 | | |
| E1 linear | +0.004 | 0.10 | 0.049 | 0.12 | 2.38 | 0.029 |
| E1 top | +0.002 | 0.08 | 0.041 | 0.11 | 2.73 | 0.021 |
| E1 linear −dark | +0.008 | 0.17 | 0.087 | 0.30 | 2.02 | 0.063 |
| E1 top −dark | +0.007 | 0.12 | 0.057 | 0.17 | 2.19 | 0.039 |
| E2 linear / top | +0.004 / +0.002 | 0.10 / 0.08 | 0.049 / 0.040 | 0.12 / 0.11 | 2.36 / 2.74 | |
| Prior stand-in alone | −0.005 | 0.27 | 0.133 | 0.37 | 0.72 | 0.265 |

- The splice joins two open pia, and E-dark at 0.02 mm says the pia's own edge marks the join. So test 2 is an upper bound on what the profile can do. The −dark rows remove that edge: the bands alone place the split at 0.06–0.09 mm, against 0.13 for the prior stand-in.
- **The profile's uncertainty is not honest:** the robust z SD is 2.0–2.7 (plain SD 11–39: heavy tails). The samples count as independent every 0.24 mm, and real residuals are correlated further and are not Gaussian. Not corrected (no tuning); the fix needs a measured residual correlation or a heavier-tailed likelihood (open, below).

**Paths through the points:**
- *a) Normal streamlines:* 9 of 35 cross WM to WM. Toward bank A every path reaches WM at 1.7–3.4 mm. Toward the far bank, 26 fail: 17 on a turn and 7 into fluid at 1.0–4.2 mm, all where coherence is 0.05–0.2 (`crossing_V1V2_banks_planes.png`, lower row). Without the turn stop they run to the 8 mm cap or into fluid, not to the far WM. This is finding 25's observation, now on curved paths: near the lower points the far bank's normal fans out round the fundus where the banks join (finding 29: 16 points have their two roots under 5 mm apart) [inference from the views].
- *b) Column pairs:* 24 of 35. Five have no second side within 1.5 mm along the normal. Six more (points 10, 14, 24, 25, 26, 33) descend back to the bank they started on: their two end roots are under 1 mm apart along the white surface, finding 4's sliding along the arrival's ridge; they include the four longest paths (6.2–7.4 mm). They are counted as failed: at both faces the end roots of a path are either under 1.9 mm or over 14 mm apart, so the cut (`SAME_MM`, 5 mm) sits in an empty gap.
- *c) Plane line (diagnostic):* 12 of 35. At most points the user's neighbouring points lie on one k slice, nearly collinear, so the fitted plane's normal is ill-defined there (as finding 17 noted).

Offset of each estimator's split from the user's point along the path (mm; |offset| median, with the 90th percentile):

| Estimator | a normal (9) | b column pair (24) | c plane line (12) |
|---|---|---|---|
| E0 symmetry | 0.04 (0.18) | 0.07 (0.30) | 0.04 (0.29) |
| E-dark | 0.04 (0.05) | **0.04** (0.91) | 0.04 (0.08) |
| E1 linear | **0.01** (0.06) | 0.09 (1.03) | 0.03 (0.13) |
| E1 top | 0.03 (0.05) | **0.07** (0.37) | **0.02** (0.10) |
| E1 linear −dark | 0.13 (0.85) | 0.21 (0.92) | 0.07 (0.72) |
| E1 top −dark | 0.07 (0.10) | 0.09 (0.21) | 0.04 (0.17) |
| E2 linear | 0.01 (0.06) | 0.08 (0.46) | 0.04 (0.13) |
| E2 top | 0.02 (0.05) | 0.09 (0.15) | 0.02 (0.11) |
| Field prior alone (pairwise posterior) | 0.15 (0.28) | 0.20 (0.50) | 0.18 (0.51) |
| One pass, equal (on the same path) | 0.10 (0.38) | 0.46 (1.25) | 0.10 (0.44) |
| One pass, field | 0.14 (0.36) | 0.32 (1.11) | 0.26 (0.46) |

Benchmarks at this face: one pass with field weights 0.40 (finding 25, its own score), level set 0.40 (finding 29), per-front posterior 0.11 (finding 18).

**Reading:**
- **Where a path crosses, the profile places the split within about half a voxel** at the 24 column pairs (E1 top 0.07 mm, E2 top 0.09), against 0.20–0.32 mm for the field's beliefs and the one pass on the same paths, and on a par with finding 18's per-front posterior (0.11). On the 9 clean normal paths it is within 0.01–0.03 mm.
- **Much of it at this face is a dark line.** Most points sit on a residual dark line (visible in `crossing_V1V2_banks_profiles.png`), and E-dark alone gives 0.04 mm on the column pairs, with a long tail where the darkest point is elsewhere (90th 0.91). Without the line, the bands still carry the split: top-anchored 0.09 mm on the column pairs and 0.04–0.07 on the clean paths; linear 0.21 and 0.07–0.13.
- **Top-anchored beats linear** in every family, and most in the tail (90th 0.37 against 1.03 on the column pairs). The linear likelihood has a second mode about 1 mm off at points 16, 19, 20 and 22 (`crossing_V1V2_banks_curves_b.png`), where stretching the superficial layers with the deep ones lines the stria up with the wrong band [inference].
- **The field prior pulls the wrong way where the profile is confident and wrong.** At points 19 and 20 the likelihoods sit at −0.5 to −1.0, the prior at +0.5, and the posterior follows the prior. Because the likelihood is about 2–3× overconfident (test 2), E2 weighs the profile too heavily in general; here it hardly differs from E1.
- **The paths are the limit, not the profile.** The normal streamline crosses at 9 of 35 points; the column pair at 24, and six more of its paths slid back to their own bank.

**Caveats:**
- In family b, the two descents are joined at the user's point, so the path has a slight kink there. Families a and c pass smoothly through the point and agree on their subsets.
- Every path goes through a marked point; whole-sheet seeding from roots is not done.
- Only one face; it has a dark line where M1S1 has a weak notch (finding 17).

**Views:** `prototypes/figures/crossing_V1V2_banks_profiles.png` (one row per point, aligned at the point, each estimator's split), `..._curves_a.png`, `..._curves_b.png`, `..._curves_c.png` (likelihood, prior and posterior per point), `..._planes.png` (paths on the three planes holding most points, over echo 2 and over coherence; failed paths dashed), `..._spliced.png` (test 2). Numbers: `prototypes/scripts/crossing_V1V2_banks.json`.

### M1S1, 23 points (box 141³ centred on the points; every point ≥ 7.2 mm from a face; code unchanged)

Run: about 5.5 minutes, peak 5.3 GB.

**Open columns:** 9,301 traced; 3,191 reached fluid, 2,605 met a face, 1,816 stopped on a turn, 1,380 returned to WM. 3,188 kept, t 1.92 / 2.48 / 3.48 mm.

**Test 2, 400 spliced open columns** (|error| median, 90th in brackets, mm): E-dark 0.06 (0.10); E0 0.05 (0.37); E1 linear 0.05 (0.10), top 0.04 (0.10); without the dark line 0.08 (0.21) and 0.06 (0.15); prior stand-in 0.11 (0.30). Robust z SD 2.7–3.1. So on open M1/S1 cortex the profile places a known pia-to-pia join as well as at V1V2, with the same overconfidence.

**Paths through the points:**
- *a) Normal streamlines:* 9 of 23 cross (11 turn on the far side at low coherence, 1 runs into fluid, 2 slide back to their own bank).
- *b) Column pairs:* 13 of 23 (9 slide back to their own bank, 1 has no second side).
- *c) Plane line:* 19 of 23. Here the user's points span several slices, so the plane is well defined.

On the crossing paths the thick bank's side is 2.9–3.3 mm and the thin bank's 1.7–2.1 mm at most points (L 4.9–5.1 mm; finding 13's column arc lengths are 3.34 and 2.02).

Offset from the user's point along the path (|offset| median, 90th in brackets, mm; signed median: + toward the thin bank):

| Estimator | a normal (9) | b column pair (13) | c plane line (19) |
|---|---|---|---|
| E0 symmetry | 0.48 (2.06) | 0.48 (1.80) | 0.42 (1.88) |
| E-dark | 1.16 (1.30), −1.16 | 1.16 (1.24), −1.16 | 1.24 (1.93), −1.12 |
| E1 linear | 0.77 (0.94), +0.77 | 0.77 (0.86), +0.77 | 0.85 (1.15), +0.77 |
| E1 top | 0.57 (0.67), +0.57 | 0.57 (0.64), +0.57 | 0.54 (0.73), +0.50 |
| E1 linear −dark | 0.36 (0.84) | 0.69 (0.78) | 0.75 (1.08) |
| E1 top −dark | 0.33 (0.70) | 0.56 (0.64) | 0.50 (0.71) |
| E2 linear | 0.14 (0.24) | 0.13 (0.47) | 0.22 (0.65) |
| E2 top | 0.12 (0.51) | 0.28 (0.54) | 0.28 (0.59) |
| Field prior alone (pairwise posterior) | **0.07** (0.36) | **0.07** (0.20) | **0.12** (0.61) |
| One pass, equal (on the same path) | 0.62 (0.77) | 0.70 (0.92) | 0.74 (0.96) |
| One pass, field | 0.30 (0.48) | 0.26 (0.61) | 0.30 (0.45) |

(The b column's signs are flipped in the JSON, where + is toward the path's B end, which on b paths is the thick bank.) Benchmarks at this face: one pass with field weights 0.26 (finding 25), level set 0.18 (finding 29), per-front posterior in arc length 0.33 (finding 13), contoured base with field weights 0.23, field with interleaved clicks 0.11 (*Pial editing*).

**Reading:**
- **The profile misses at M1S1, consistently and confidently.** On every path family E1 puts the split 0.5–0.8 mm into the thin bank (signed median +0.50 to +0.77), with posterior SDs of 0.06–0.2 mm (`crossing_M1S1_curves_c.png`: narrow likelihood peaks well off the point). The thin bank would then be about 1.1–1.4 mm thick. Top-anchored misses less than linear (0.54–0.57 against 0.77–0.85).
- **It is not a dark line.** Leaving out the darkest point barely changes the miss. The darkest point is itself about 1.1–1.2 mm into the thick bank at almost every point (a dark band inside M1 at that depth), not at the contact; finding 17's notch at the points is too weak to dominate.
- **Test 2 says the method works on this box's open cortex** (0.04–0.06 mm). So the miss is specific to the fused stretch: either the profile there differs from the open templates near it, or the templates are of the wrong cortex [inference, untested: (i) the thin bank's templates come from tops within 6 mm of its root, whose thickness is about 2.5–2.9 mm (the field's μ there) against 1.7–2.1 mm on the path, so the stretch maps the wrong bands onto each other; (ii) finding 17's thinner superficial layers at the contact; (iii) an areal border along the central sulcus, so the nearby open tops are a different area from the fused part].
- **The field's beliefs alone land within 0.07–0.12 mm here**, better than any earlier rule at this face. They over-predict both banks (μ 2.3–2.9 on the thin bank, 3.2–3.8 on the thick; their sum exceeds L by 0.6–1.5 mm), and the σ² split shares that misfit so that it nearly cancels. This is the V1V2 situation of finding 18 (A9: correlated errors saved the split), and it uses the crossing path's L, not the column arc lengths of finding 13 [inference: why the two L's give such different answers is not checked].
- **E2 is worse than the prior alone** because the overconfident likelihood pulls it toward the wrong split (0.12–0.28 against 0.07–0.12).
- **The paths fail often here too.** The normal streamline crosses at 9 of 23 points, the column pair at 13 (9 slide back to their own bank), and the plane line, which uses the truth to choose its direction, at 19.

**Views:** `prototypes/figures/crossing_M1S1_profiles.png`, `..._curves_a.png`, `..._curves_b.png`, `..._curves_c.png`, `..._planes.png`, `..._spliced.png`. Numbers: `prototypes/scripts/crossing_M1S1.json`.

### The column volume: first build at both faces

`column_volume.py <figures> <scripts> <data> <site>`, with `trace_array`, `ascend`, `sample_traces` and `order_along` added to `crossing_lib.py` (3 more tests, 14 in all, pass). V1V2 banks: 3 minutes, peak 4.3 GB; M1S1: 4.5 minutes, peak 5.6 GB. The volumes are `prototypes/scripts/column_volume_<site>.npz`: positions, features, classes, tracers and horizons per trace.

**Size.** The white mesh has 307k (V1V2) and 339k (M1S1) vertices, about 0.57 voxel apart; one vertex per voxel cell and normal side gives 99,677 and 109,869 traces. 36 % and 34 % have roots within 2 mm of a box face and are left out below, leaving 64,046 and 72,350.

**Classes** (roots ≥ 2 mm from a face; primary + fallback):

| | V1V2 banks | M1S1 |
|---|---|---|
| Open (fluid) | 28,746 + 5,517 | 23,880 + 4,114 |
| Crossing (far WM ≥ 5 mm away) | 4,628 | 9,860 |
| Own bank (WM within 5 mm) | 4,646 | 2,732 |
| To the pass's contact (fallback) | 3,478 | 5,367 |
| Failed | 9,989 + 7,042 | 19,441 + 6,956 |
| Open length, p5 / 50 / 95 (mm) | 1.38 / 2.10 / 4.66 | 1.98 / 2.58 / 3.70 |
| Crossing length L, p5 / 50 / 95 (mm) | 3.62 / 4.34 / 5.65 | 3.86 / 5.06 / 6.26 |

- *Primary failures:* 15.7k and 15.9k turns; at M1S1, 19.4k traces run into a face, although their roots are ≥ 2 mm from it (columns that lean toward a face). The fallback fails mostly on a ridge (4.2k, 6.0k), where its ascent slides along the arrival's crest (finding 4).
- **Where the primary turns** (`column_volume_<site>_roots.png`, right panel): two groups.
  - *At the start:* a spike at 0.3–0.35 mm, the first depth at which a turn is tested; 16 % (V1V2) and 11 % (M1S1) of turns are within 0.3 mm of the WM. Here the data's normal and the mesh's normal disagree just above the WM, as the main plan's open question (WM edges) expects.
  - *Mid-column:* most turns are at 1–2 mm, at coherence 0.06–0.25 (median 0.11–0.13). The roots on the planes show them as gold and red patches along the fused stretches and the fundi.

**Sections** (`column_volume_<site>_sections.png`: echo 2 along each trace, lateral position along the white surface, depth from the root, horizons over them; three planes, two banks each):
- *V1V2 banks.* The volume reads like a seismic section. The stria is a continuous level band at about 0.8–1.2 mm from open into fused cortex. On crossing traces the contact is a dark band at about 2 mm that continues the open top horizon sideways without a step, and the far bank's stria appears below it, mirrored. Your points lie on that dark band. The profile split, the one pass and the field prior run along it.
- *M1S1, thin bank (bank 1 on all three planes).* The open top horizon (about 2.0–2.2 mm) also continues into the fused stretch as a dark line at about the same depth (clearest on j = 59 and 65). The one pass and the field prior run 0.2–0.4 mm deeper than that continuation (toward the thick bank), and the profile split 0.2–0.7 mm shallower [inference from the views: the continuation of the open top is a candidate contact estimate of its own, by the no-faults rule; not computed].
- *M1S1, thick bank.* Few crossing traces start there; most of its traces through the fused stretch fail or reach a face. M1's own banding is visible, more grainy than V1's.
- *Holes* (white columns) are failed traces. They cluster in the fused stretches, so the volume is thinnest exactly where it is wanted.

**Horizons at the marked points** (traces from the volume's own roots that pass within 0.3 mm of a point; the median over each point's traces, then the median and 90th percentile over points; mm):

| Horizon | V1V2 banks | M1S1 |
|---|---|---|
| Points reached by crossing traces | 12 of 35 | 12 of 23 |
| One pass, equal | 0.06 (0.29) | 0.64 (0.77) |
| One pass, field | 0.19 (0.42) | 0.26 (0.41) |
| Field prior (pairwise posterior on L) | 0.18 (0.42) | 0.13 (0.35) |
| Profile, top-anchored (subsample: 10 and 11 points) | **0.03** (0.05) | 0.41 (0.61) |
| Fallback's contact (19 and 12 points) | 0.92 (2.24) | 0.72 (0.84) |

- These traces start at the roots, not at the points, so this is the first score where nothing about a path is chosen from the points. It agrees with the path results: the profile works at V1V2 and misses at M1S1, where the field's beliefs do best.
- **The fallback's contact is wrong at both faces.** It stops where the pass's label belongs to a root more than 5 mm away along the white surface. At the V1V2 banks the two banks' roots near the fundus are closer than that, so the ascent runs on toward the far WM (the magenta horizon near 4 mm in the sections). The threshold problem of finding 25 again.

### Flattening in the volume: first run at both faces

**Code.** `flatten_lib.py` (shifts, fit, resampling) and `test_flatten_volume.py` (7 tests, all pass: `nix develop --command uv run pytest -q prototypes/scripts/test_flatten_volume.py`). Driver: `flatten_volume.py <figures> <work> <site> [--eps E] [--figs-only]`, after `column_volume.py`; it needs `uv run --with matplotlib`. Outputs: `prototypes/scripts/flatten_<site>.npz` and `.json`, and `prototypes/figures/flatten_<site>_{sections,profiles,roots}.png`.

**Synthetic.** On a grid of traces with a thickness that varies sideways by ±20 % and a band that rises in the middle, the fit recovers the horizons to a median of 0.001 mm (p95 0.007), against 0.19 mm unflattened. With noise at 0.2 of the band amplitude: a median of 0.013 mm. ε = 1 instead of 0.2 doubles the error without noise and leaves it unchanged with noise. The shift estimator recovers a constant shift of 0.10 mm within 0.005 mm, and a depth-varying one within 0.015 mm.

**Runs** (ε = 0.2; 4 outer iterations; each converged in about 1,600–2,000 CG iterations):

| | V1V2 banks | M1S1 |
|---|---|---|
| Traces, pairs | 64,046; 310,978 | 72,350; 338,794 |
| Time, peak memory | 5.3 min, 5.7 GB | 6.1 min, 6.8 GB |
| Pair shift \|s\|, p50 / p95 (weighted samples) | 0.047 / 0.225 mm | 0.056 / 0.266 mm |
| Weighted RMS misfit, first → last outer | 0.037 → 0.033 mm | 0.046 → 0.042 mm |
| Residual per trace and level, p50 / p90 / p99 | 0.022 / 0.108 / 0.300 mm | 0.043 / 0.142 / 0.349 mm |
| Folds (dz/dτ ≤ 0, within the trace) | 2.3 % of levels | 3.0 % |
| Darkest point in 0.5–1.5 mm (interior minima only), IQR: raw depth → τ | 0.24 → 0.18 mm (25,040 traces) | 0.40 → 0.29 mm (28,427 traces) |

The darkest-point measure first counted minima on the window's edge (a pile-up at 0.5 mm). It now keeps interior minima only; 7,064 and 9,360 traces had none. At M1S1 the darkest point in the window is not a known band; the number is for comparison only.

**What the views show** [readings by Claude; for the user's reading]:
- **V1V2, sections.** In τ the bands are level across open and fused cortex. The contact's dark band lies flat at τ ≈ 2.0 with your points on it, and the far bank's bands below it line up. The stria shows as a faint level band at τ ≈ 1.0–1.2 in places (plane 72 bank 2, plane 82 bank 1).
- **V1V2, stacked profiles.** The crossing traces' contact dip sharpens from a broad trough at 2.0–2.4 mm to a narrow one at τ = 2.0. The stria does not show in the stacked echo 2, raw or flattened. The box holds V2 as well as V1, and the plan reads the stria better in echo 3.
- **M1S1.** The thin bank's contact band is flat at τ ≈ 2.1 on plane 59, and the far bank's bands below it are more level. The stacked crossing profile has no contact dip in either form: the contact has no fluid gap there (finding 17). At plane 77 your points sit about 2.2–2.3 in τ, with the field prior 0.1–0.3 below them, as in raw depth.
- **Residual on the roots.** It is high at crowns and fundi at both faces, low along straight banks.
- **A fault the flattening keeps.** V1V2 plane 54, bank 1, about −4 mm along the surface: the contact band steps from τ 1.7 to 2.0 where the traces change from fallback to crossing. The step is in the raw section too. By the no-faults rule it is a trace error upstream, and the fit shows it rather than smoothing it.

**Limits seen.**
- **The strongest edges drive the fit.** These are the fluid edge at open tops and the contact. The dark samples at a trace's end are partial volume with fluid, not a band. The laminar bands below about 1.7 mm were already nearly level in raw depth, as the main plan's finding says for columns along the layer normal, so flattening changes them less.
- **One τ for the box.** τ is the mean depth of its horizon over every trace in the box. At M1S1 the open tops of thin S1 and thick M1 land at different τ: the warp is continuous, so a thicker area takes more τ. Whether equal τ is the same layer across an areal border is untested.
- **Holes and blocks.** Failed traces and fallback traces with uniform samples (M1S1 plane 77) are as in the raw volume. Flattening adds nothing there.

**Open (for the user):**
- whether to leave the samples near a trace's fluid end out of the data term, so that laminar bands rather than the fluid edge set the warp;
- echo weights: both echoes are scaled to unit SD; echo 3 carries more stria contrast;
- per-bank or per-area gauges in place of one τ for the box;
- the τ coordinate as the chart's depth axis (`chart.py`), and a re-trace along the flattened layer normal.

### The joint warp: the level solver

**Code.** `warp_cut.py`: `cut` (the exact minimum of the difference terms, by scipy's maximum flow; the reference) and `qpbo` (all terms, ties included, by `thinqpbo`). Tests: `test_warp_cut.py` (41: the cut against brute force on 40 random models, and a hinge whose target lies outside both ranges) and `test_warp_qpbo.py` (88; needs `uv run --with thinqpbo --with ortools`).

**Validation, with exact answers:**
- *Brute force, 60 random models with ties:* QPBO's labelling is never below the optimum; its lower bound is always valid; its certified labels always lie on an optimum (re-solved with them fixed).
- *20 models without ties:* every variable certified, the optimum found.
- *CP-SAT (OR-Tools, exact), 8 folded two-bank models* (two chains of 12 columns joined at a fold, 14 labels, hard and L1 lateral terms, sum ties between facing columns): QPBO reaches the optimum in 7 of 8; the certified fraction is 100 % on 5, 62 %, 25 % and 0 % on the others; on the one it misses (0 % certified) its labelling is 0.32 above the optimum, and the bound (−2.64) is below it (−2.46). Persistency held in every case (CP-SAT with the certified labels fixed reaches the optimum).

**Vectorized** (the graph and the binary terms; the same 129 tests pass).

**Benchmark** (random model: 144 columns on a grid, 3 levels × 51 labels, 4 lateral neighbours with two hinges each per level, hard ordering and a vertical hinge per column, 72 ties; 21,600 binary nodes, 308,425 pairwise terms; one core):
- Building and adding the terms: 0.1 s. **The QPBO solve: 15.5 s.** Without ties (submodular) still 8.6–9.4 s; without hard bounds 2.1 s. The capacity used for hard terms (1×, 10³×, 10⁶× the finite total) changes nothing. Long chains of infinite arcs (the label encoding plus the ordering bounds) are a known slow case for augmenting-path maximum flow.
- **Certification:** 100 % without ties, 100 % with ties but no hard bounds, **0 % with both**, on this random model (noise unaries 0.1, ties weight 1).
- **QPBO-I:** energy 1022.9 with lower bound 1014.1 (gap 0.9 %). CP-SAT for 60 s: 1 worker 2140.0 (bound 949.7); 8 workers 1029.2 (bound 949.9). QPBO-I is better on both counts here.
- At V1V2 scale (25–64k columns) the solve would be one to several hours on one core if it scaled linearly, and maximum flow often scales worse; the random model is not representative of real data.

### The joint warp: first real run (V1V2 banks, subsampled)

**Code.** `joint_levels.py` (the level model: build, a pass with re-centring, coarse to fine) with `test_joint_levels.py` (4 tests: a window re-centring to an anchor outside it, a shift carried by conservation, a tie placing an unanchored pia, re-centring never raising the energy); `joint_site.py <figures> <work> <site> [--spacing] [--radius]` (the driver); `warp_cut` gained `reduce` (fixing variables exactly; tested against the full energy), `qpbo_pruned`, and pairwise tables (submodular part in the cut; all of it in QPBO; tested by brute force). 313 tests pass.

**Changes made on the way, each from a diagnosed cause:**
- *The open-top anchor needs a capture range* (above, Weights).
- *No term may depend on a window's centre.* With conservation and the vertical prior linearized at the centre, re-centring changed the energy and the fine pass cycled between two states (73 of 75 columns at a window edge in each of the last six tries; mostly failed traces at τ = 1/3). Both are now exact tables over the two labels: w·|z_w − g(z_v)| and w·|z^ℓ − τ·z^L|.
- *Conservation tables are made submodular:* one weight per pair and level, read at the starting estimate (as the taper), and the neighbour map g(z) = z + s_vw(z) projected onto non-decreasing maps (a warp between neighbours cannot fold; folds are estimation errors). With a weight that varied with z_v and the raw map, certification fell to 27 % (real) and 8–17 % (a random test).
- *QPBO-I starts from the previous pass's solution* (mapped into the new window), so re-centring never raises the energy.
- Bugs: `energy` of a model with no variables; integer accumulators in the QPBO terms; reading an npz array inside a loop (200 s).

**Run** (one trace per 0.3 mm: 8,018 traces; 16,783 pairs; 547 ties; 4,249 open anchors; start: the single-surface horizon): 116 s, peak 3.7 GB.
- Coarse pass (0.16 mm): 6 tries to no edge hits; certified 96.5 %; energy 2306.3 → 2199.8 (bound 2199.7).
- Fine pass (0.04 mm): 7 tries to no edge hits; **certified 99.8 %** (51 variables left to QPBO-I); energy 1876.4 → 1872.3, bound 1872.3: a proven optimum of this energy to 0.001.
- **At the points** (every subsampled trace within 0.3 mm; 29 points, 87 traces): joint pia median 0.12 mm, p90 0.36; points over 0.3 mm: 4 (0.30), 5 (0.36), 6 (0.32), 19 (0.36), 20 (0.36), 25 (0.86), 28 (0.34). The single-surface horizon on the same traces: median 0.06, p90 0.46; over 0.3 mm: 10 (0.40), 19 (0.68), 20 (0.88), 22 (0.88), 27 (0.36).
- **Sections** (`joint_V1V2_banks_sections.png`) [readings by Claude; for the user's reading]: the joint pia holds at about 2.0 mm on plane 54 bank 1 from −2 to 0 mm, where the single-surface horizon dropped to the far stria (the user's error), and rises to the dark band at about 1.8 mm on plane 54 bank 2 at 5–7.5 mm, where the single surface sank to 3.6 mm. It lifts above the contact where no anchor holds it: plane 72 bank 2 at −3 to −2 mm (about 1.4 mm, against a dark contact at about 2.2), plane 82 bank 2 at −3 to −1 mm (about 1.3 mm). Along the marked contact it sits about 0.1 mm shallow of the points.
- So the solver is exact enough that the remaining errors belong to the model: the first-version weights, the vertical prior (distance), the anchor's capture range. The synthetic two-bank volume, with known answers, is to show which term pulls the pia shallow.

### The joint warp at full density: V1V2 banks and M1S1

**Code.** The level model now uses the merged package `baillarger.surfcut` (PR #30). `joint_levels.fill_warp`: the continuous warp z_v(τ), τ = j/24, between the solved levels (held strongly), from the neighbour-shift equations counted only where both samples lie more than 0.16 mm below their own pia (nothing past the pia enters), with a weak pull to the straight fill between levels. Test: on synthetic traces with known warps and a mirrored far stack past each pia, each fitted τ level set is one true horizon across columns (median 0.002 mm, p95 0.015 mm). Between levels τ is only a parametrization: a reparametrization shared by every column changes no equation, so alignment, not z_v(τ) itself, is what the data fix. With the taper below the pia switched off the errors are the same; the far stack reaches the fill only through the shift estimator's window. `joint_site.py --spacing 0.16` (neighbour radius 1.7 × spacing) runs a site at full density, fills the warp, and draws `joint_<site>_s0.16_{sections,tau_sections,profiles}.png`.

| | V1V2 banks | M1S1 |
|---|---|---|
| Traces, pairs, ties, open anchors | 23,124; 41,998; 1,597; — | 26,984; 49,506; 3,239; 10,622 |
| Time, peak memory | 11 min, 8.4 GB | 9.3 min, 10.5 GB |
| Fine pass: tries to no edge hits; certified; energy vs bound | 9; 96.6 %; 4837.41 vs 4837.38 | 11; 99.8 %; 1858.45 vs 1858.45 |
| **Joint pia at the points:** points; median / p90 | 31; 0.12 / 0.36 mm | 21; **0.24 / 0.40 mm** |
| Single-surface horizon, same traces | 0.08 / 0.56 mm | 0.80 / 1.04 mm |
| Darkest interior band: IQR raw depth → τ (in mm at the median pia) | 0.32 → 0.26 mm | 0.64 → 0.37 mm |

- **V1V2:** fewer large misses than the single surface (points 19, 20, 22 from 0.76–0.92 to 0.32–0.50 mm), a slightly worse median. In τ the open and crossing stacks coincide (on the subsample), which they do not in raw depth.
- **M1S1:** the single-surface horizon (the start) is 0.80 mm too deep at the points (signed +0.78; the darkness cost finds no contact line there); the field prior is at 0.13 mm (12 points). The joint pia recovers most of it: on plane 59 bank 1 it follows the contact band up to about 1.6 mm where the single surface sank to 2.8, and on plane 65 bank 1 it sits at 2.1–2.2 mm against 2.6–2.9. But **19 of 21 points are still too deep (median +0.20 mm)**; on plane 77 the points are at about 2.1 mm and the pia at about 2.4.
- **The remaining bias is not the start** (tested): the start enters the energy only through the conservation weights and tapers, read at the starting pia c0. Re-solving with the joint result as c0 (`--rounds 3`) reached a fixed point at once (rounds 1 and 2: the pia moved a median of 0.000 mm across traces), and at the points the median went from 0.24 to 0.20 mm (p90 0.40 → 0.44). So the +0.2 mm at M1S1 belongs to the model as it stands.

**Bands in the τ frame** (`tau_bands.py <figures> <work> <site> <tag>`): both echoes at constant τ; each trace's profile stacked with its neighbours' (roots within 1 mm, weights by distance and by the positive correlation of the two raw profiles, so that a neighbour with another pattern counts little); bands are the extrema of the site's stacked median profile (open and crossing traces) with prominence ≥ 0.15 and width at half prominence ≥ 1.5 voxels at the median pia, τ 0.05–0.92 (beyond is the pia's own appearance); per trace, each band's extremum within ±0.06 of its τ. The width follows the band rule (2–3 voxels, the stria at 1.5–2 at the limit and usable): a first threshold of 2.5 voxels dropped the stria, whose width is 1.67 voxels (V1V2, subsampled).

| | Band | τ (mm at the median pia) | Found on | Spread: τ, in mm / raw depth |
|---|---|---|---|---|
| V1V2 (23,124 traces, pia 2.0 mm) | bright | 0.18 (0.35) | 47 % | 0.09 / 0.20 mm |
| | dark (the stria) | 0.37 (0.74) | 41 % | 0.13 / 0.24 mm |
| | bright | 0.81 (1.61) | 62 % | 0.12 / 0.46 mm |
| M1S1 (26,984 traces, pia 2.4 mm) | bright | 0.14 (0.34) | 62 % | 0.11 / 0.23 mm |
| | dark | 0.33 (0.80) | 49 % | 0.13 / 0.32 mm |
| | bright | 0.82 (1.98) | 65 % | 0.13 / 0.71 mm |

- Open and crossing traces put each band at nearly the same τ (medians within 0.005–0.016; the V1V2 stria 0.36 against 0.40).
- The M1S1 dark band is not a stria (M1 and S1 have none); what it is, is open.
- Views: `bands_<tag>_profile.png` (single and stacked medians by class, the bands), `bands_<tag>_maps.png` (each band's τ and strength on the roots of the marked planes).

**Levels at the bands** (run at both sites, full density): `joint_site.py --levels <the bands' τ> --warp <the last joint result> --bands <its bands>`. The levels are the bands' τ plus the pia (snapped to 1/96); each band's stacked profile is a unary cost on its level (dark rewards dark, bright rewards bright; weight 0.5, stated; test: one column, the level moves from the vertical prior's 0.8 mm to the band's 1.0 mm); the start and the depth-to-τ map come from the last warp; the fill uses J = 96. **The fill between levels stays with distance**: no equal-volume estimator is validated yet (main plan, *The fallback*).

### Levels at the bands: results

Levels at V1V2 τ 0.177 / 0.375 / 0.802 and at M1S1 τ 0.146 / 0.333 / 0.823 (snapped to 1/96), plus the pia; start, c0 and depth-to-τ from the previous joint warp (V1V2: one round; M1S1: the `--rounds 3` fixed point); each band's stacked profile as evidence on its level.

| | V1V2 banks (31 points) | | M1S1 (21 points) | |
|---|---|---|---|---|
| | before | levels at bands | before | levels at bands |
| Pia at the points, median / p90 | **0.12** / 0.36 mm | 0.20 / 0.34 mm | 0.20 / 0.44 mm | **0.08 / 0.22 mm** |
| Signed median (pia minus point) | — | — | +0.18 mm (18 of 21 deep) | **+0.06 mm** (15 deep, 0 shallow) |
| The dark band's spread: τ IQR (mm at the median pia) | 0.122 (0.26 mm) | **0.072 (0.16 mm)** | 0.145 (0.37 mm) | **0.084 (0.22 mm)** |
| Certified (fine pass) | 96.6 % | 100 % | 99.8 % | 100 % |

- **M1S1: better on both counts.** In the sections (`joint_M1S1_s0.16_bands_sections.png`) the pia runs through the points on plane 77 (both banks, about 2.1 mm, where the fixed levels had about 2.4) and holds at about 1.9–2.0 mm in the fused part of plane 65 bank 1 (the single-surface start sank to 2.5–2.9). Two local failures, both on failed traces: a pia spike to about 0.4 mm at plane 59 bank 2, −6.8 mm, and a spike in the τ = 0.82 level near +2.5 mm on the same section.
- **V1V2: the bands tighten, the pia gets worse at the points.** The stria's spread in τ falls by 40 %, but the pia's median at the points rises from 0.12 to 0.20 mm. Not diagnosed. Candidates: the band evidence (weight 0.5, a first-version value) pulls the warp and the pia with it harder than the anchors and ties hold it; or the band costs were read through warps that dip (fixed since, below).
- **Neither fine pass converged:** both stopped at the cap of 20 re-centrings with 2 variables at a window edge. Runs now report this (`converged` in the JSON).
- **Run time:** 32 min (V1V2) and 46 min (M1S1), of which 26 and 38 min were spent re-reading the starting warp from its npz once per trace (fixed: the array is read once). With the fix: about 5–8 min per site.

**The V1V2 pia shift: diagnosis** (reruns with the review fixes, re-centring cap 40; all converged; 4–14 min each beside other sessions' jobs):

| V1V2, 31 points | Pia: median / p90 | Signed median (too deep / too shallow) | Stria spread in τ (mm) |
|---|---|---|---|
| Fixed levels τ = 1/3, 2/3 | 0.12 / 0.36 mm | −0.04 mm (11 / 14) | 0.122 (0.26) |
| Levels at the bands, band weight 0 | 0.14 / 0.40 mm | +0.12 mm (27 / 2) | 0.119 (0.26) |
| … band weight 0.25 | 0.20 / 0.34 mm | +0.20 mm (25 / 2) | 0.067 (0.14) |
| … band weight 0.5 | 0.20 / 0.34 mm | — | 0.072 (0.15) |
| … band weight 0.5, chain prior (`--vert chain`) | **0.08 / 0.32 mm** | — | **0.074 (0.16)** |

- Moving the levels to the bands' τ already moves the pia deep with no band evidence; the evidence adds to it, the same at half the weight. At M1S1 the same step moved the pia shallow, toward the points (+0.18 → +0.06 mm).
- **The cause is the vertical prior (tested).** The prior |z_ℓ − τ·z_pia| ties each band level to the pia with τ one constant for the box, so wherever a band sits at another fraction of the thickness (V1V2 stria: τ 0.40 on crossing traces, 0.36 on open ones), it moves the pia. The chain prior (`Params(vert="chain")`) keeps the prior between inner levels, |z_ℓ − (τ_ℓ/τ_ℓ₊₁)·z_ℓ₊₁|, and drops it from the last inner level to the pia. With it the pia returns and the stria stays tight:

| Chain prior, band weight 0.5 (all converged, 100 % certified) | Pia: median / p90 | Stria spread in τ (mm) | Run |
|---|---|---|---|
| V1V2 banks, 31 points | **0.08 / 0.32 mm** (pia-coupled 0.20 / 0.34; fixed levels 0.12 / 0.36) | 0.074 (0.16) | 443 s, peak 10.0 GB |
| M1S1, 21 points | 0.08 / 0.30 mm (pia-coupled **0.08 / 0.22**) | 0.088 (0.23; pia-coupled 0.084) | 377 s, peak 11.3 GB |

  - V1V2: the best pia at the points of any run so far, with the stria as tight as with the pia-coupled prior. Five points remain over 0.3 mm (0.32–0.52 mm), against seven for the single-surface horizon on the same traces.
  - M1S1: the median holds and the tail grows (p90 0.22 → 0.30 mm; points 18 and 23 at 0.32 and 0.36 mm). Not diagnosed: at M1S1 the pia-coupled prior had moved the pia toward the points, so the coupling may have been helping there.
  - Signed medians were not scored for these runs. Figures: `joint_<site>_s0.16_bands_vchain_{sections,tau_sections,profiles}.png`.

**The reported lower bounds were imprecise** (fixed in PR #32, `fix/surfcut-hard-cost`): in runs with hard bounds, QPBO's bound was 0.75–49 off the energy, sometimes above it. The labels and the certification were right: QPBO's binary encoding plus its offset equals `energy()` to 3e-12 on the real model. The cause was the hard-term cost, 1000 times the total of finite costs; QPBO's double-precision bound cancels those costs. At 1 + that total (the definition stated earlier) the error falls about 1000 times (real subsampled model: +0.0019 and −0.0007 against −0.75 and +0.86). `reduce`'s own big cost also left the tables out (latent; fixed in the same PR). The "bound" and "gap" figures above from runs with hard bounds overstate the gap; the energies and labels stand.

**Code review after these runs** (all in `joint_levels.py` / `joint_site.py`; 9 tests pass):
- *The band costs read depth → τ through each trace's warp with `np.interp`, which assumes the warp increases.* The fill is not forced to be monotone, so where a warp dipped the cost was undefined. Each warp is now made monotone first (running maximum). The runs above predate this.
- *Unconverged passes were silent:* now a warning and `converged` in the history and the JSON.
- *The monotone neighbour map was recomputed in every build* (each level, each re-centring): now once.
- *The band costs were a per-trace loop in every build:* now one batched interpolation (`interp_rows`, tested against `np.interp`, and the batched costs against the loop, both to 1e-12).
- Approximations kept and noted: a tie pairs a crossing trace with the facing bank's nearest subsampled column (c_A + c_B = L along slightly different paths); in band stacking, traces that fail before the pia are zero-filled past their end when profiles are correlated.

### Several surfaces in one cut: near stria, contact, far stria

**Why** (the user, 2026-10-28). Past a contact the far bank is the near bank's sequence mirrored, so its stria is a dark band like the contact: where the contact is faint, a single darkness horizon jumps to it (plane 54). And the stretch of a column and its clip depend on each other and on the neighbours. So the bands that set the stretch and the contact that sets the clip are solved together, as ordered surfaces in one minimum cut (Li et al. 2006's multi-surface form of the same construction; exact).

**Surfaces, per trace** (V1V2; each on its own depth grid):
1. *Near stria:* 0.4–2.0 mm at 0.08 mm.
2. *Contact (first pia):* 0.8–4.6 mm at 0.04 mm. Open tops are soft anchors as before; a top beyond 4.6 mm is always overridden.
3. *Far stria:* 1.2–6.0 mm at 0.08 mm. It has data only where a trace runs past its first pia; elsewhere it rides along at no cost.

**Data cost:** darkness (as *The first-pia horizon*) on every surface. The open-top preference is larger than the sum of the three surfaces' data ranges, so it still acts first.

**Spacings and trends from open cortex** (open traces whose top the single-surface cut kept, top under 3.5 mm; the near stria is each trace's most prominent interior dark band, prominence ≥ 0.3, between 0.4 mm and 0.4 mm short of the top). On the V1V2 box: 18,579 of 28,033 such traces have one; its depth is 0.80 mm (p1–p99 0.40–1.60), the top lies 1.30 mm beyond it (p1–p99 0.58–2.10), the fraction stria/top is 0.38 (p1–p99 0.19–0.65). The box holds V2 as well, where the darkest band is not a stria, so these ranges are wide.
- **Between surfaces, per trace:** contact − near stria within the trend in H of (top − stria) plus its residual's 1st–99th percentiles; far stria − contact within the 1st–99th percentiles of (top − stria) with no trend (the far bank's root is not known).
- **Between neighbours, per surface:** the contact keeps the bound of *The first-pia horizon* (trend and g from the open tops the single-surface cut kept). Both striae: the detrended stria depth's slope at 0.8–1.2 mm, 99th percentile; the far stria without the trend.
- Bounds are rounded outward to the grid.

**Scoring:** the contact on every trace, as now in `pia_horizon.py`.

**Before running it: darkness ties, and the contact fades like the stria** (2026-10-28; solver built, driver not).
- `pia_lib.min_surfaces` (several surfaces, each on its own grid; exact) passes brute force on 20 random two-surface problems, matches `min_surface` on one surface, and keeps a faint contact between two striae when the contact has its own evidence (`test_pia_lib.py`, 54 tests).
- **With darkness on every surface, ordering does not fix plane 54.** At the failing traces the samples at the user's depth are bright (+0.4). Contact there with the far stria on the 2.7 mm band costs about +0.4 − 1; the contact on the band with the far stria beyond it (no data, no cost) costs about −1. The wrong reading is cheaper.
- **Neither does a local pia template.** Profiles about the pia (smoothed darkness, ±0.8 mm, no stretch): open tops are bright 0.3–0.6 mm below and dark beyond; working contacts are bright on both sides (+0.9 at ±0.4 mm) with a dark line (−2.9 to −4.4); at the failing traces the user's depth is bright on both sides with no line. From about −1.2 mm on, the 2.7 mm band has the full contact shape (correlation 0.98–0.99 with the contact template) and the user's depth has none (−0.1 to −0.4). At points 19, 20 and 22 (10–14 open traces each) the correlation at the point's depth is −0.4 to −0.36; at point 27 it is 0.93, yet the cut is 0.48 mm off there.
- **Reading (the user):** the contact's dark line fades out over about half a millimetre while its position continues, as the stria fades from V1 to V2, a facies change in the contact's own appearance (probably residual fluid, gone where the banks are pressed fully together). In seismic terms, a dimming reflector: it is carried at a constant position relative to the reflectors that do not fade, not moved to the nearest strong event.
- **The near stria does not fade there:** 0.76–0.92 mm from −3.4 to +0.5 mm along plane 54 bank 1. Contact − stria is 1.16–1.36 mm on the left; about 1.1 mm on the right at the user's depth; about 1.85 mm at the cut's 2.7 mm.
- **On open cortex** (open tops the single-surface cut kept, with a stria pick), the detrended slope at 0.8–1.2 mm, p50 / p95 / p99: superficial thickness (top − stria) 0.09 / 0.36 / 0.66 mm per mm; stria depth 0.09 / 0.38 / 0.64; total top 0.10 / 0.36 / 0.53. The jump to 2.7 mm needs about 0.9–1.4 mm/mm in the superficial thickness.
- **The change proposed then** (superseded by *The joint warp*): (1) the contact's data cost clipped at 0, so a faded contact costs nothing; (2) two cuts in sequence: the near stria first; then the contact with its neighbour bound on (contact − stria), c_w − c_v within s_w − s_v ± g_u·d, which is a difference constraint once the stria is fixed (exact, as the curvature detrending). At M1S1, with no stria, the reference would be a near-bank horizon from the flattening.

### Correcting the joint pia: ambiguity, edit priority, local re-solve

**Code.**
- `baillarger.surfcut.qpbo` now also returns each variable's **ambiguity interval** `interval_lo`, `interval_hi` (PR #31). These are the labels its labelled binary nodes y = [x ≥ k] do not rule out. Some global minimum lies in every interval at once (weak persistency); a certified variable has zero width. Tests: brute force (80 models with ties and arbitrary tables) and CP-SAT (the folded models) confirm that the minimum over the box of intervals is the global minimum; a frustrated odd cycle leaves every window open.
- `edit_data.py` rebuilds a saved result's model (shifts, weights, ties checked against the saved ones), cached in `edit_data_<tag>.npz`. V1V2's fine model gives the saved energy exactly (4837.4135). For M1S1 r3 the last round read its taper at the previous round's pia, which is not saved; the final pia stands in, so this is the model of a fourth round (energy 1859.37 against 1856.53).
- `edit_priority.py`, tested by `test_edit_priority.py`:
  - `margins`: the energy rise of each variable moved alone, from the terms that touch it, vectorized and exact against `energy()` differences;
  - `ranks`: the fraction of traces with a defined doubt strictly lower, so "no doubt" ranks 0;
  - the map: one QPBO on the fine model centred on the saved levels gives certification and intervals; the flat margin is the smallest rise of the pia moved by ±0.1 or ±0.2 mm, on a 0.02 mm grid with the other levels fixed; the model doubts are listed below. Each doubt becomes a rank; the combined score is their mean, unweighted.
  - Views: `edit_priority_<tag>_sections.png` (echo 2, the joint pia, its interval where not certified, the single-surface horizon, the points; under each section, the combined score and every doubt's rank).
- `edit_resolve.py`, tested by `test_edit_resolve.py`:
  - an edit is a strong unary, 50 per mm of distance from the point's depth, on the pia of each trace passing within 0.3 mm of a marked point, at that trace's sample nearest the point;
  - the neighbourhood is every trace whose root lies within R of an edited trace's root, by geodesic distance on the white mesh;
  - the sub-model holds the neighbourhood and the ring of traces one term away, with the ring fixed by `reduce`, then `qpbo_pruned` from the current levels, with windows ±0.48 mm, re-centred if hit at an edge;
  - every edit made so far stays in the model;
  - tests: with R covering everything, the result equals a full solve; with a fixed ring, it equals the brute-force minimum, with or without an edit; edits persist; the neighbourhood is geodesic.

**The priority map** (both sites, 30 s, 6–7 GB):

| | V1V2 banks | M1S1 r3 |
|---|---|---|
| Pia certified; uncertified intervals p50 / p90 | 96.6 %; 0.48 / 0.48 mm (784 of the uncertified clipped by their window: lower bounds) | 99.9 %; 0.36 / 0.84 mm |
| Scored points with an uncertified trace | none | none |
| Flat margin p10 / p50 | 0.009 / 0.31 | 0.008 / 0.06 |
| Spearman with the point's error: combined; strongest single doubts | **0.63**; joint − single-surface 0.60, open top overridden 0.40 | **−0.11**; flat margin 0.27, field prior −0.52 |
| Points over 0.3 mm | 5, 6, 16, 19, 20, 22, 28 | 3, 4, 5, 14, 15, 16, 17, 18, 21 |
| Points in the top decile of the combined score | 19, 20, 22 (0.32–0.50 mm), 27, 31, 32 (0.08–0.16 mm) | 8 (0.20 mm) |

- **V1V2:**
  - **Flagged:** the map flags the three worst contact misses (19, 20, 22). The joint pia overrode those open tops by 2.5–2.9 mm and left the single-surface horizon by 0.26–0.60 mm.
  - **Flagged but right:** 27, 31 and 32 are flagged for the same reason (overrides of 3.0–3.8 mm), yet the pia is right there. An override is a doubt to look at, not an error.
  - **Missed:** 5, 6 and 28 (0.32–0.40 mm), which raise only the single-surface disagreement (0.28–0.32 mm). Also 16: its one trace has an override of 2.58 mm, but its combined score of 0.28 falls under the decile's 0.32.
- **M1S1:** certified and wrong. None of the 9 points over 0.3 mm is flagged.
  - Their traces are mostly where the field prior and profile split are not defined (both exist only on crossing traces, the split only on a subsample). Where those are defined, they agree with the joint pia: the field prior's rank correlation with the error is negative.
  - The +0.2 mm bias does not show in any doubt the pipeline writes. It needs a check of the model itself, not more doubts.
- **Margins:** on the 0.02 mm grid, 88 (V1V2) and 131 (M1S1) certified traces have a slightly negative margin (median −0.002, worst −0.034, all at ±0.1 mm). On the model's own 0.04 mm grid, single moves of ±1–3 labels lower the energy at no trace on either site. The finer grid rounds the smoothness targets and the anchors' capture offset differently, so these negatives come from the margin view's grid, not from the solve.

**Edit reach** (`edit_resolve.py <work> <tag>`). Three random orders of the scored points; per order, the first 12 are seeded one edit at a time from the saved result, and the rest (19 at V1V2, 9 at M1S1) are held out for every k. The cells give each order's held-out median error (mm) / held-out points over 0.3 mm.

| | k | R = 1 mm | R = 2 mm | R = 4 mm |
|---|---|---|---|---|
| V1V2 (saved: 0.14, 0.12, 0.14 / 4, 4, 5) | 3 | same / 4, 4, 5 | same / 2, 4, 3 | same / 2, 3, 3 |
| | 12 | same / 4, 3, 5 | same / 2, 3, 3 | same / 2, 3, 3 |
| M1S1 (saved: 0.20, 0.12, 0.16 / 4, 3, 4; signed +0.16, +0.12, +0.16) | 3 | same / 4, 3, 4 | same / 4, 3, 3 | same / 4, 3, 4 |
| | 12 | same / 4, 3, 4 | 0.20, 0.12, 0.12 / 3, 2, 3 | 0.20, 0.12, 0.08 / 3, 1, 2 (signed +0.12, +0.12, +0.08) |
| Time per edit, median (max); free traces | | 0.2 s (0.8); ~95 | 1.0 s (4); ~300 | 5–8 s (30); ~1,000 |

- **Seeded points:** 0.00 mm at every R.
- **Reach:** an edit fixes its own traces. At R ≥ 2 mm, up to 2 of a draw's 4–5 held-out misses at V1V2, and up to 2 of 3–4 at M1S1, go under 0.3 mm. Held-out medians do not move at V1V2, nor at M1S1 except one order at R = 4 mm with 12 edits.
- **The M1S1 bias:** it stays (+0.08 to +0.12 mm on the held-out points after 12 edits). Edits are unaries that the smoothness term passes on only weakly, and the deep pia belongs to the model's other terms.
- **What this asks of an editor:** go to every doubtful patch (a worklist), not a few seeds per site. A view to do this is being built separately: a conformal flat map of the white surface coloured by these doubts, with two orthogonal sections at its crosshair.

### The first-pia horizon: results at the V1V2 banks

**Code.** `pia_lib.py` (the minimum-closed-set solver, bounds) and `test_pia_lib.py` (28 tests, all pass: brute force on 25 random small graphs with asymmetric bounds and infeasible cases, a chain held by its anchor, the rounding and the penalty). Driver: `pia_horizon.py <figures> <work> <site> [--g-pct P]`, after `column_volume.py`, with `uv run --with matplotlib`. Shared section views are in `volume_views.py` (also used by `flatten_volume.py`). Outputs: `prototypes/scripts/pia_<site>.npz` (stage 2's c, stage 1's, μ, H, g) and `.json`, `prototypes/figures/pia_<site>_{sections,slopes}.png`. 64,046 traces × 116 labels, 310,978 pairs: 2.3 min for both stages, peak 16.8 GB.

**Why open tops are soft anchors.** A first run held open tops fixed and measured g on all open pairs. g came out at 3.44 mm/mm (p99 of the detrended slope at 0.8–1.2 mm; p50 0.14, p95 0.64). The tail is one kind of pair:

| Open pairs 0.8–1.2 mm apart | Pairs | Slope p50 / p95 / p99 (mm per mm) |
|---|---|---|
| Neither top over 3.5 mm | 1,699,542 | 0.13 / 0.52 / 0.83 |
| Exactly one top over 3.5 mm | 34,461 | 3.48 / 5.07 / 5.71 |
| Both over 3.5 mm | 54,605 | 0.38 / 1.48 / 2.07 |

- 99.7 % of the pairs above the 99th percentile have exactly one top over 3.5 mm: neighbours whose tops differ by about one cortical thickness, an error by the rule. 2,255 open traces (6.6 %) have tops over 3.5 mm (p99 6.42 mm, in cortex about 2.1–2.5 mm thick). Straight primary traces have the same tail; it is not wandering or the fallback.
- [inference] These traces cross a contact (the first pia) and run on to fluid beyond the far bank. The open class says only where a trace ends.
- With fixed anchors and g = 3.44 the horizon reached 26 points at a median of 0.07 mm but p90 1.04 mm, and was spiky in the sections.

**The two stages** (P = 9.56; 1,525 open tops lie beyond 5.0 mm and are always overridden):

| | Stage 1 | Stage 2 |
|---|---|---|
| Trend from | 34,263 open tops | 32,228 open tops kept by stage 1 |
| Trend log t = m + a·H | 0.741 + 0.383 H | 0.700 + 0.238 H |
| Detrended slope at 0.8–1.2 mm, p50 / p95 / p99 | 0.14 / 0.64 / 3.44 | 0.13 / 0.49 / 0.77 |
| … at 1.6–2.4 mm, p99 | 2.00 | 0.55 |
| g | 3.44 | 0.77 |
| Open tops overridden | 2,035 (75 % of tops over 3.5 mm; 1.1 % of the rest) | 6,992 (98.7 % of tops over 3.5 mm; 14.9 % of the rest) |
| Traces at the lowest label (0.4 mm) | 6.7 % | 4.5 % |
| At the points, crossing and to-the-contact traces only: points; median / p90 of per-point \|offset\| | 26; **0.04 / 0.73 mm** | 26; **0.04 / 0.14 mm** |
| **At the points, also open traces whose top was overridden** | | 33; **0.06 / 0.47 mm** |

- **The first score left out the error.** It counted crossing and to-the-contact traces only. Long open traces whose top was overridden have crossed a contact, and their c is an estimate too. Counted with them (stage 2), the horizon reaches 33 points at a median of 0.06 mm, p90 0.47 mm, and six points are 0.4–0.9 mm off: 15, 16, 19, 20, 22 and 27, all on plane 54. `pia_horizon.py` now scores those traces too (the JSON is from the run before this change; the numbers here are recomputed from its saved horizon).
- For comparison, the column volume's horizons at their 12 points: one pass equal 0.06 (p90 0.29), one pass field 0.19 (0.42), field prior 0.18 (0.42); the profile split 0.025 (0.05) at 10 points.
- **The overrides of ordinary tops are small.** Of stage 2's 3,974 overrides of tops up to 3.5 mm, 74 % move the top by 0.2 mm or less (median 0.10 mm, about two labels), and 77 % move it shallower. That is the size of the fluid edge's voxel raggedness; 10 % move by more than 0.66 mm. Their roots are a little more convex (median H +0.06 against −0.04 /mm) and their tops a little longer (2.26 against 2.02 mm).
- **Sections** (stage 2, `pia_V1V2_banks_sections.png`) [readings by Claude; for the user's reading]: the horizon is smooth and runs along the contact with your points on it on every marked plane, and along the open top elsewhere. Two places look wrong or unknown:
  - plane 54 bank 2, 5–7.5 mm along the surface: the horizon drops to about 3.6 mm through crossing and own-bank traces, while a dark band and the profile split are at about 1.9 mm. The open tops just left of it descend to about 3 mm; the bound carries them on;
  - **plane 54 bank 1, about −2 to 0 mm: wrong** (the user). Your points continue at about 1.9–2.0 mm; the horizon steps down to 2.6–2.8 mm at about −1.7 mm and stays there to about 0.6 mm. Diagnosis:
    - the traces there are open, with tops at 5.9–6.7 mm: they cross the contact and run through the far bank to fluid, so their tops are overridden and only the data cost and the bound place c;
    - at −1.99 mm the contact is still dark at the points' depth (cost −1.67 at 1.96 mm) and the horizon is on it;
    - from −1.65 mm on, the samples at 1.9–2.1 mm are neutral to bright (cost −0.41 to +0.52), and both echoes have a dark band at 2.6–2.85 mm (cost −0.7 to −1.2). The darkness cost prefers that band, and g = 0.77 lets the horizon reach it in about 0.6 mm;
    - so the solver does what it was asked; the data cost's assumption, that the first pia is dark, fails here.
    - **The band at 2.6–2.85 mm is the opposite bank's stria** (the user's reading): where the horizon cannot find the contact, it jumps to the far stria. Seen past a contact, the far bank is the near bank's sequence mirrored, so its stria is a dark band of the same width as the contact band, a decoy that darkness alone cannot tell apart (in seismic terms, picking the wrong event in the wavetrain).

**Open (for the user):**
- the plane 54 error: a data cost that does not rest on darkness alone, for example a one-sided template fit (does 0..c look like the near bank's own open profile stretched to c?), mirror symmetry about c, or the doubled layer-I signature of finding 17; or marked points as constraints; a smaller g would also carry 2.0 mm across, but that would be tuning to the symptom;
- the data cost near the WM: 4.5 % of traces still sit at 0.4 mm, mostly failed and own-bank ones;
- semi-automatic use: marked points as constraints, an edit-priority map from overridden tops, small cost margins and disagreement between horizons; first, how far one edit reaches (seed with some points, score on the rest);
- then: trim each trace at the first pia (tapered) and re-run the flattening; M1S1.

### Column traces at folds: phantoms, diagnostics, candidate tracers

**Question.** How accurate are the column traces at crowns, fundi and contacts, and does another tracer do better? The current primary tracer (`crossing_lib.trace_array`, the streamline of the summed structure tensor's leading eigenvector, started along the white mesh's normal) and its fallback (`crossing_lib.ascend`) against five candidates, on phantoms with known columns and on both marked sites without ground truth. All 3D.

**Code**: the package's `baillarger.streamlines` (the fields and integrators: structure tensor, fitted field, the trace with or without the blended start, the ascent) and `baillarger.column_traces` (the tissue, the white mesh and roots, the one pass, the fallback, the classes; tracers `current`, `fitted`, `fitted_blend`), with their tests (`tests/test_streamlines.py`, `tests/test_column_traces.py`); in `prototypes/scripts/`: `fold_measure.py` (the sites' features), `fold_phantoms.py` (phantoms), `column_tracers.py` (the candidates that were not chosen; the package's tracers, fallback and classes behind the same calls), `fold_eval.py` and `fold_summary.py` (phantom scores and views), `column_diag.py` (real-data diagnostics), `column_volume.py` (the builder, `--tracer`), `test_fold_phantoms.py` (22 tests, pass: the phantoms' geometry and truth, straight columns normal to the laminae, the noise match, the mesh curvature at a crown tip (within 15 %), the error measures (zero on the true column, the shift off it), the Laplace stencil (annihilates linear functions for any constant anisotropy; linear across a slab; logarithmic between cylinders), the fitted field on a slab (T is the distance), the blended start (its direction at every step as prescribed), the per-bank and multiscale tensors).

**Phantoms.** Laminae are offset surfaces of a core set, so the true column through any point is the straight segment from its foot, and its ends are known in closed form: **crown** (a capsule of WM, radius w = 0.32, 0.64, 1.28 mm; root mean curvature 1/w at the tip, 1/2w on the shaft), **fundus** (a fluid slot of half-width g = 0.16, 0.48 mm; white surface −1/2(g + t) round the fundus), **kiss** (concentric banks of 2.0 and 2.6 mm apposed at a contact cylinder of radius 3.6 mm with no fluid, a 120° open sector with a 0.32 mm gap; columns cross the contact radially), and **kiss_flat** (the same, with the fused banks' dark top held flat: no notch at the contact, the measured weak-notch case). Cortex 2.4 mm, voxels 0.16 mm, rotated off the grid. Image: per voxel, the mean over 27 sub-samples of the site's median open profile (echo 2 and 3, trend removed, by relative depth), and the site's WM and fluid levels outside (partial volume). Noise: Gaussian on the log echoes, matched so that the high-pass statistic of the trend-removed echo in the cortex equals the site's (V1V2 0.090 / 0.140, M1S1 0.071 / 0.112 for echo 2 / 3; SDs 0.093 and 0.074 for echo 2). Features, tensor, mesh, roots, stops, fallback and classes exactly as `column_volume.py`. The phantoms' coherence (median 0.79–0.94; kiss_flat 0.67–0.75) is above the sites' (0.77, 0.73): real texture and column-to-column variation are not modelled, so phantom errors are lower bounds.

**Tracers.**
- *current:* as above.
- *laplace* (candidate i): streamlines of the anisotropic Laplace potential, φ = 0 on WM and 1 on fluid, div(D ∇φ) = 0 with D = I − c n nᵀ (conduction along the data's layers 1, across them 1 − c).
- *fitted* (i-b, the main plan's fitted field, `curv_lib.fit_all_wm`'s system): T = 0 on all WM, ∇T fitted in least squares to the oriented data normal F = c n sgn + 0.1 g. It needs no top: in fused cortex T has a ridge between the banks, which the unsigned trace passes.
- *multiscale* (ii): the tensor at 2 and 4 voxels' integration, each normalized, blended by the fine scale's coherence.
- *blend* (iii): the direction w d_data + (1 − w) n_mesh, w = min(1, c s / 0.3 mm): the mesh normal at the root, the data's normal after 0.3 mm / c.
- *fitted_blend:* fitted's field with blend's start.
- *perbank* (iv): the tensor accumulated only over voxels whose direction away from the nearest WM agrees (six-direction partition of unity), so the two banks of a contact never mix.

**Phantoms, measured noise** (V1V2 features; M1S1's within a few hundredths, `fold_eval_<site>_n<0|1>.json`; `fold_summary.png`). Per phantom: the share of primary traces ending in the right class (open or crossing), then the largest distance from the true column with the fallback, median / p90, mm. Roots ≥ 2 mm from a face; about 5 % of every phantom's traces reach a face, so 0.95 is the ceiling.

| | crown 0.32 | crown 0.64 | crown 1.28 | fundus 0.16 | fundus 0.48 | kiss | kiss_flat |
|---|---|---|---|---|---|---|---|
| current | 0.93; 0.38 / **1.02** | 0.95; 0.16 / 0.39 | 0.95; 0.10 / 0.21 | **0.60**; 0.11 / 0.24 | 0.95; 0.07 / 0.12 | 0.90; 0.09 / 0.17 | 0.84; 0.14 / 0.35 |
| laplace | 0.95; 0.16 / 0.54 | 0.96; 0.10 / 0.26 | 0.95; 0.07 / 0.17 | 0.95; 0.08 / 0.15 | 0.96; 0.07 / 0.12 | **0.46**; 0.15 / 0.39 | **0.32**; 0.59 / 3.06 |
| fitted | 0.84; 0.16 / 0.47 | 0.83; 0.12 / 0.27 | 0.84; 0.08 / 0.19 | 0.86; 0.06 / 0.15 | 0.92; 0.05 / 0.10 | 0.86; 0.07 / 0.15 | 0.86; 0.10 / 0.20 |
| **fitted_blend** | 0.88; 0.13 / 0.41 | 0.86; 0.09 / 0.23 | 0.86; 0.06 / 0.17 | 0.88; **0.05 / 0.14** | 0.94; 0.04 / 0.09 | 0.87; 0.06 / 0.14 | 0.86; **0.09 / 0.20** |
| multiscale | 0.94; 0.21 / 0.53 | 0.95; 0.12 / 0.26 | 0.95; 0.07 / 0.16 | 0.52; 0.12 / 0.59 | 0.95; 0.06 / 0.11 | 0.90; 0.07 / 0.14 | 0.89; 0.09 / 0.23 |
| blend | 0.94; **0.14 / 0.37** | 0.95; 0.10 / 0.21 | 0.95; 0.07 / 0.14 | 0.60; 0.09 / 0.23 | 0.96; 0.05 / 0.10 | 0.90; 0.07 / 0.14 | 0.85; 0.13 / 0.33 |
| perbank | 0.93; 0.31 / 0.83 | 0.95; 0.15 / 0.36 | 0.95; 0.10 / 0.21 | 0.74; 0.09 / 0.22 | 0.95; 0.07 / 0.12 | 0.90; 0.08 / 0.17 | 0.78; 0.14 / 0.35 |

Contact error (where the trace crosses the contact, from the true crossing point; median / p90 mm): kiss: current 0.07 / 0.13, fitted_blend 0.05 / 0.09; kiss_flat: current 0.14 / 0.27, fitted_blend 0.07 / 0.14, multiscale 0.08 / 0.15; laplace 2.0 / 2.3.

- **Tight crowns: noise and curvature together.** Noise-free, the current tracer is within 0.10 / 0.23 mm at the 0.32 mm crown; at the measured noise, 0.38 / 1.02 mm. The ends move as much as the paths. The blended start removes most of it (0.14 / 0.37); the fields come next.
- **Tight fundi: the tensor tracers fail by geometry, not noise.** At the 0.16 mm slot, 40 % of primary traces turn, noise-free as well (the layers wrap round the slot within the tensor's integration scale). The fallback catches them but ends 1.2–2.6 mm off at p90. Laplace (95 %) and the fitted fields (86–88 %) trace through, within 0.05–0.08 mm. Multiscale is worst there (52 %).
- **Contacts: Laplace fails by construction.** Its only top is fluid, so in fused cortex its streamlines run sideways along the contact to the open sector. The tensor tracers and the fitted fields cross. Without a notch the fitted fields halve the current tracer's contact error.
- **The fitted fields' cost: 7–12 % fewer primary successes at crowns,** noise-free as well. They turn on the shaft just above the WM, where T's boundary is the voxel WM [inference: its staircase]; the blended start recovers a third of these.
- **Per bank:** no gain anywhere. In its test it halves the mixing of the far bank within 3 voxels of a contact, but the contact's own edge (at the derivative scale) remains, and it is worse at tight crowns.
- **Wide folds** (crown 1.28, fundus 0.48): every tracer within 0.04–0.10 mm median.

**Real data** (`column_diag_<site>.json`, `column_diag_<site>_{maps,curves}.png`; roots ≥ 2 mm from a face: 64,046 and 72,350; 4–5 minutes and 5.1–5.5 GB per site for all seven). Classes as `column_volume.py` (open: primary + fallback); points: the marked points reached within 0.3 mm by crossing traces (of 35 and 23).

| V1V2 banks | Open | Crossing | Own bank | To contact | Failed | Primary turns | Points |
|---|---|---|---|---|---|---|---|
| current | 28,746 + 5,517 | 4,628 | 4,646 | 3,478 | 17,031 | 15,692 | 12 |
| laplace | 49,411 + 234 | 348 | 142 | 665 | 13,246 | 1,317 | 2 |
| fitted | 30,754 + 4,678 | 5,388 | 1,679 | 4,140 | 17,407 | 15,907 | 16 |
| **fitted_blend** | 31,131 + 4,522 | **5,428** | **879** | 4,194 | 17,892 | 16,083 | **16** |
| multiscale | 29,374 + 4,185 | 5,019 | 6,135 | 2,686 | 16,647 | 11,929 | 12 |
| blend | 30,446 + 4,503 | 5,005 | 3,758 | 3,103 | 17,231 | 13,693 | 12 |
| perbank | 28,804 + 5,642 | 4,602 | 4,352 | 3,573 | 17,073 | 16,122 | 9 |

| M1S1 | Open | Crossing | Own bank | To contact | Failed | Primary turns | Points |
|---|---|---|---|---|---|---|---|
| current | 23,880 + 4,114 | 9,860 | 2,732 | 5,367 | 26,397 | 15,909 | 12 |
| laplace | 54,449 + 74 | 253 | 4 | 487 | 17,083 | 789 | 2 |
| fitted | 25,400 + 3,651 | 12,250 | 1,108 | 4,881 | 25,060 | 14,782 | **23** |
| **fitted_blend** | 25,669 + 3,624 | **12,228** | **373** | 5,035 | 25,421 | 15,265 | **23** |
| multiscale | 24,533 + 2,869 | 11,989 | 4,074 | 3,353 | 25,532 | 10,132 | 13 |
| blend | 24,805 + 3,556 | 10,484 | 1,607 | 5,113 | 26,785 | 14,771 | 16 |
| perbank | 24,021 + 4,023 | 9,703 | 2,544 | 5,648 | 26,411 | 16,423 | 12 |

- **The fitted fields cross where the current tracer slides back to its own bank.** Own-bank traces fall from 4,646 to 879 (V1V2) and 2,732 to 373 (M1S1); crossing traces rise by 17 and 24 %; the marked contacts reached by crossing traces go from 12 to 16 of 35 and from 12 to **all 23**. Of the current tracer's own-bank traces at V1V2, 2,870 turn under the fitted field instead and go to the fallback.
- **Laplace makes fused cortex open** (V1V2: 49k open, 348 crossing), as on the kiss phantoms.
- **Turns stay where they were.** Every tracer but Laplace turns 10–16k times, on the same traces (median coherence along them 0.55–0.61), at the fused stretches and fundi. Multiscale turns least (−24 %, −36 %) but sends more traces back to their own bank.
- **(c) Agreement with the data's normal** at coherence 0.8–1 (median angle between the tangent and the voxel's layer normal): tensor tracers 0.9–1.1°; fitted 6.2–6.9°; Laplace 5.5–8.6°. The fields are smoother than the voxelwise normal. On the phantoms that departure is not error (the fitted paths are as close or closer to the true columns); on the sites it is not tested.
- **(a) Round trip does not discriminate.** For every pure field the return retraces the path (median 0.00, p90 0.04 mm: integration only); blend's start is not reversible (0.04–0.05 / 0.11–0.15 mm), fitted_blend's barely (0.01 / 0.05–0.06).
- **(b) Order.** 18–31 % of traces have a neighbour whose offset, perpendicular to the traces at equal arc length, turns by more than 90° (7–10 % of 284k and 308k pairs; first at a median 1.6–1.9 mm), Laplace 12–13 %. They concentrate at fundi: at M1S1, 55–77 % of roots with mean curvature below −0.15 /mm against 12–30 % elsewhere (`_curves.png`). The fitted fields are gradients and cannot cross, yet score 27–31 %, so the measure also counts what is not a crossing [inference: neighbours converging into a fundus are compared at equal arc length, not equal depth, and fallback ascents are included]; it locates trouble, it does not count crossings. Measured at equal τ it would.
- **(d) Against curvature.** Failures are 80–97 % at roots below −0.3 /mm (fundi) for every tracer except Laplace (58–82 %), 15–45 % elsewhere, and rise again at the sharpest crowns (V1V2, > 0.6 /mm: 43–51 %, 1k roots). The maps on the marked planes (`_maps.png`: root curvature; per tracer the round trip, swaps and failures) put the failures and swaps along the fused stretches and fundi.

**Recommendation: fitted_blend for the column volume** (`column_volume.py --tracer fitted_blend`, the default). On the phantoms it is the most accurate at tight fundi and at contacts, notch or none, and within 0.01–0.04 mm of the best elsewhere. Its cost is 7–9 % more primary failures at crowns, which the fallback takes. On both sites it gives the most crossing traces, the fewest own-bank traces, and the most marked contacts reached. It does not reduce the turns in fused stretches and fundi, and it departs 6–7° from the data's normal where that is coherent. Not recommended: Laplace (fused cortex becomes open), per bank (no gain), multiscale alone (fails at tight fundi), blend alone (best only at tight crowns).

**v2 column volumes** (`column_volume_<site>_v2.npz` and `.json`, `column_volume_<site>_v2_{sections,roots}.png`; the same format plus `tracer_name`; the downstream steps read them by site name `<site>_v2`, not run yet). `column_volume.py <figures> <work> --name <site>_v2 --box … --points … --tracer fitted_blend`; `--tracer current` gives the volumes built before (`column_volume_<site>.npz`). V1V2: 2.7 min, peak 5.0 GB; M1S1: 4.1 min, 6.6 GB. Classes as in the diagnostics table above, trace for trace. The package's builder reproduces the earlier volumes: with `--tracer current`, `column_volume_<site>.npz` bit for bit at both sites (every key: mesh, roots, traces, outcomes, classes, features, horizons); with `fitted_blend`, every outcome, class and sample count of the v2 volumes, positions within 8e-6 voxels (V1V2) and 4e-4 voxels (M1S1). That residue is the multigrid: the v2 builds used pyamg's default setup, which estimates a spectral radius from a random start vector, so its field differs from run to run within the solver's tolerance; the package's setup is deterministic (local Jacobi weights; 17 iterations against 14 on M1S1, the fields within 4e-8 mm), and two package builds are bit-identical. `--wm-changes <wm_handles.npz>` builds on the handle-free WM label (`baillarger.white`) in place of the aseg's: one label for the mesh, the fitted field, the stops, the one pass and the classes (aseg WM voxels it cuts become cortex; voxels it fills, WM); the npz records it as `wm_label`. On V1V2 (198 voxels cut in the box), with fitted_blend: traces 99,677 → 99,953, crossing 5,428 → 5,436, own bank 879 → 903, to contact 4,194 → 4,179, open unchanged (35,653).
- **The start spike is gone:** primary turns within 0.3 mm of the WM fall from 15.6 % to 2.4 % (V1V2) and from 11.1 % to 0.5 % (M1S1) of turns.
- **Horizons at the marked points** (crossing traces within 0.3 mm; per-point median |horizon − point depth|, then median / p90 over points, mm; the point sets differ, so read the medians with the counts):

| | V1V2 (v1 → v2) | M1S1 (v1 → v2) |
|---|---|---|
| Points reached | 12 → 16 | 12 → 23 |
| One pass, equal | 0.06 / 0.29 → 0.10 / 0.36 | 0.64 / 0.77 → 0.66 / 0.84 |
| Field prior | 0.18 / 0.42 → 0.27 / 0.63 | 0.13 / 0.35 → 0.14 / 0.26 |
| Profile, top-anchored (subsample) | 0.03 / 0.05 (10) → 0.06 / 0.14 (16) | 0.41 / 0.61 (11) → 0.41 / 0.72 (23) |

  At M1S1 the estimators keep their accuracy on twice as many points; at V1V2 the four new points are harder ones (their medians rise). The pia from the joint warp on v2 is the test that matters.

**Next.**
- The fitted field's boundary on the white mesh, not the voxel WM (the crowns' turns), and a curvature prior: the root's fanning is fixed by the white surface's curvature (div of the unit column field = k1 + k2 at the root), parallel-surface or equivolume (Bok) beyond it, weighted against the data by coherence.
- Feed the joint warp's pia back as the field's top (Laplace between the white mesh and the estimated pia, contacts included) and its τ levels as laminae; iterate, checked on held-out points and the swap measure.
- Phantoms with equivolume laminae (curved true columns) and with a perturbed pia, to test both.
- The order measure at equal τ instead of equal arc length.

### A pia editor on an inflated surface

A prototype for correcting the joint warp's pia (the first pial level) trace by trace. The sections are always cut from the true 3D mesh. The inflated surface and the flat map only say where to click and which way is which.

**How to use it.** From the repository root, one site at a time (about 1 GB):

    nix develop --command scripts/memguard 8 uv run --with thinqpbo python prototypes/scripts/pia_editor.py 8782 prototypes/scripts M1S1_s0.16_r3

Then open http://127.0.0.1:8782/ (the server binds 127.0.0.1 only). The other tag is `V1V2_banks_s0.16`.
- *Navigate.* Drag rotates the 3D view, shift-drag pans, the wheel zooms. The *Surface* menu switches between the true surface and three inflation levels.
- *Crosshair.* A click sets the crosshair at the nearest subsampled trace. Gray vertices have no subsampled trace within 0.5 mm: box-face strips and white surface that does not face cortex, 31% of the V1V2 mesh.
- *Sections.* The two sections run along the screen's horizontal and vertical at the clicked point. Each is drawn on the 3D view and on the flat overview (cyan and yellow), so you can see where it really runs.
- *Edits.* A click on a section at a depth adds a pending edit. *Re-solve* (R = 1, 2 or 4 mm) makes the pending edits one step. *Undo step* puts the last step's edits back to pending. *OK mark* records "looked at, no change".
- *Worklist.* *Next* (key `n`) moves to the next worklist patch with no edit or OK mark.
- *Files.* *Save* writes `pia_edits_<tag>.json` and `pia_edits_<tag>_levels.npz` in `prototypes/scripts`. *Load* replays the saved list. *Replay check* rebuilds the pia from the list and reports the largest difference from the current pia.

**What the sections show.** Echo 2 (trend removed) along the traces, depth down from the white surface. Over it:
- the pia (current, and as saved);
- the two inner levels;
- the QPBO interval where the pia is not certified (from `edit_priority_<tag>.npz`; until that file exists, a red mark at the top of each uncertified trace);
- open tops (`H_top`) and the single-surface horizon (`pia_<site>.npz` `c`);
- your points within 0.3 mm (★ with their numbers), and the edits;
- ▾ above each trace that has a tie.

At a contact, the facing trace (through the crosshair trace's tie, or the nearest tie within 1 mm on either section) gets its own two sections through the same planes. Their roots are drawn in orange.

**Colour layers.** The combined priority, and each rank and doubt of `edit_priority_<tag>.npz`. Until that file is written, a placeholder is used: the mean rank of "not certified (saved)" and "|joint − single-surface|". The server reads the file again when it changes. Further layers: pia change since the saved result, joint − single-surface, open tops, and the flat map's area factor.

**Worklist.** The top decile of `combined`, joined into patches by the joint warp's neighbour pairs, ordered by size × mean score. At the time of writing, with the placeholder combined layer: 219 patches at V1V2 and 211 at M1S1.

**How it works.**
- *Inflation* (`inflate.py`). Implicit umbrella-Laplacian steps with the boundary loops (the box-face cuts) pinned. Without the pin, an open sheet collapses onto its boundary (measured on a test sheet). With the pin, the surface relaxes towards the membrane spanned by its cut loops. Area is not kept; levels are λ = 100, 400 and 1600.
- *Section directions.* At the crosshair, the screen's right and up are carried from the inflated triangles to the same triangles of the true mesh. The u-direction is the carried right, projected onto the tangent plane; v = n × u, so the two sections are orthogonal on the true surface. The page reports the angle between v and the carried up.
- *Sections* (`pia_sections.py`). The plane through the root that holds the direction and the normal cuts the true mesh in a polyline, which is followed both ways from the root. Distances are arc lengths along it. The traces shown are those rooted within 0.12 mm of the plane and of the curve. An earlier version ordered a band of vertices along its longest geodesic, as `volume_views.section_traces` does. Where the band forked (measured at M1S1) it mixed two branches. A plane cannot fork on a manifold mesh, so the polyline does not have this fault.
- *Re-solve.* Each step is `edit_resolve.local_solve(D, z, neighbourhood(G, roots, the step's traces, R), h = 0.04, edits = every edit so far)`. The steps replay in order from `joint_<tag>.npz`.
- *Determinism.* QPBO-I (`thinqpbo.improve()`, called by `baillarger.surfcut.qpbo_pruned`) visits variables in an order drawn from the C library's `rand()`. With identical inputs, the same step changed one trace by 0.12 mm depending on that state (seed 3 against seeds 1, 2, 7 and 12345). `surfcut` now reseeds `rand()` itself (`qpbo_pruned(seed=0)`, PR #33). With an older `surfcut` installed, the editor reseeds before every step. Either way the pia depends only on the list. Tested under two other `rand()` states, and the test fails without the reseed.
- *Flat overview* (`flat_map.py`). A least-squares conformal map of each piece of the white mesh. Pieces of genus > 0 are first cut along their shortest non-separating loops (greedy homotopy basis, cones in the holes).

**Edit records.** Each edit stores:
- the trace (index into `sub`) and the depth;
- its sample's box voxel, ITK-SNAP voxel (1-based, box) and full-grid voxel;
- world mm (the echo 2 affine);
- the time.

Each step stores R and its solve: free traces, seconds, number of pia changes, largest change.

**Tests** (exact answers, or a stated tolerance where only the limit is exact). 20 pass:
- `test_flat_map.py` (9):
  - a plane maps isometrically up to a similarity (1e-9);
  - a polygonal cylinder strip unrolls exactly;
  - a hemisphere cap keeps angles: the area-weighted mean of s1/s2 − 1 is under 0.025 at 4000 sphere points and halves when the spacing halves;
  - a torus's shortest handle is its meridian, exactly;
  - a genus-1 surface with a hole is cut to genus 0, then to a disk;
  - a long tube needs the disk cut (free-boundary LSCM folds on it);
  - packing and scale.
- `test_inflate.py` (5): a uniform plane grid does not move; the boundary stays pinned; convergence to the discrete harmonic membrane; the tangent transfer is the face map; the sulcal depth.
- `test_pia_sections.py` (5): a plane gives the row at Euclidean distances; a cylinder gives the ring at arc lengths; a U fold gives the U at arc lengths; the directions; the points.
- `test_pia_editor.py` (1): replay equals the live result under other `rand()` states.

**Results.**

| | M1S1 | V1V2 banks |
|---|---|---|
| white-mesh pieces | 3 | 5 |
| largest piece | genus 0, 3 boundary loops | genus 2, 3 boundary loops |
| handles cut | none | 2 |
| flat map, angle distortion s1/s2 − 1 (median / 95th pct) | 0.004 / 0.014 | 0.007 / 0.046 |
| flat map, log2 area factor (5th to 95th pct) | −2.5 to 1.1 | −5.8 to 1.7 |
| flat map, folded faces | 0 | 181 (0.08 mm²) |
| inflation λ 400, log2 area factor (5th to 95th pct) | −1.5 to 0.1 | −2.7 to 0.1 |
| inflation, run time | 17 s | 16 s |
| one edit: free traces, solve time | 213, 0.41 s | 192, 0.40 s |
| one edit: pia changes (largest) | 3 (0.32 mm) | 13 (0.24 mm) |
| replay difference | 0 mm | 0 mm |

**V1V2's two handles** (worth a look in ITK-SNAP):
- **Handle 1**, ITK-SNAP voxel (49, 134, 43): a loop 0.09 mm round, where two sheets of the mesh touch at a pinhole. A segmentation or meshing artefact.
- **Handle 2**, ITK-SNAP voxel (123, 119, 14), full grid (320, 176, 78), 2.1 mm from the box face: a loop 5.8 mm round, extent 1.3 × 1.9 × 0.8 mm, with 607 traces rooted within 1 mm. Possibly a white-matter bridge. On the flat map its two cut holes shrink to almost nothing (1% of the area at 2^-13), which is why the 3D view navigates and the map is only an overview.

**Views.**
- `prototypes/figures/flat_map_<site>_distortion.png`: the map coloured by log2 area factor and by angle distortion, with the handle cuts.
- `flat_map_<site>_hist_handles.png`: histograms, and the mesh round each handle loop in 3D.
- `inflate_<site>.png`: the area factor per inflation level.

**Open.**
- At strong inflation the carried vertical can be far from v. It was 56° at the V1V2 handle and 10–31° elsewhere in the checks. The sections stay orthogonal on the true surface, but their directions on screen are then only approximate.
- The colour layers wait for `edit_priority_<tag>.npz`.
- Whether the gray regions (no subsampled trace) hide cortex that should have traces.

### A handle-free white surface

The white mesh is made from the aseg's WM label (41), and every piece of it must have genus 0 for all later processing (flat map, inflation, editor). The label is corrected once, on the full grid, by the package: `baillarger.white.wm_changes` (on `baillarger.handles.remove_handles`) returns the changed voxels, `WMChanges.save` stores them, and `baillarger.white.white_matter_box(lo, hi, changes)` gives the corrected label of any box, and `baillarger.white.white_mesh` its mesh (marching tetrahedra on the binary label, then `baillarger.mesh_smoothing.relax` and `twice`). No question is asked per handle. The reports are in the package too: `python -m baillarger.white changes --out DIR` and `python -m baillarger.white box --changes DIR/wm_changes.npz --out DIR --name NAME --lo X Y Z --hi X Y Z`. The prototype drivers `wm_handles.py` and `wm_handles_boxes.py` call them, and add only the slice figure.

**How it works.**
- *Topology.* Marching tetrahedra contours on the Kuhn tetrahedralization, so the mesh's topology is that of the voxels with the 14-neighbour Kuhn adjacency, for object and background alike. The Betti numbers (b0, b1, b2) are computed on that complex. When b1 = 0, every surface piece cut out by any box has genus 0.
- *Competing fronts.* Each component is treated on its own, with its cavities filled until the end:
  - an object front grows from the deepest voxel, and a background front from the border;
  - both grow by simple points only, deepest first;
  - the fronts meet themselves at walls across each handle's bridge and across its tunnel;
  - the smaller wall, in voxels, is given to the other front, which is a cut or a fill.
- *Blocked walls.* An object wall and a background wall can block each other, so that no single voxel is simple for either front. Then sets are claimed whole, smallest first: a wall, two touching walls, or a cluster of walls. The test is that the front's homology stays the same (Mayer–Vietoris on the set and its front neighbours).
- *Left over.* Object voxels that still cannot be claimed are removed as 'stranded'. Each of them, put back alone, would make a handle.

**Tests** (exact answers). `tests/test_smoothing.py` (7): planes kept exactly, the boundary fixed, twicing keeps a ball's radius where Gaussian smoothing shrinks it, no mode amplified, normals, pieces. `tests/test_white.py` (9), on a synthetic aseg with a drilled block and a ring closed by a rod: the plug and the cut voxel in full-grid coordinates, boxes that equal crops of the corrected label (genus-0 mesh), save/load, the white mesh, the per-piece topology, the summaries and the command line. `tests/test_handles.py` (13):
- the tetrahedra and the link match the mesher, and χ(mesh) = 2·χ(voxels);
- the Betti numbers of known solids;
- the simple-point table against an independent link check, for all 2^14 cases;
- a drilled 2 × 2 tunnel is filled (4 voxels);
- a one-voxel bridge is cut (1 voxel);
- a thick ring is cut, not filled;
- a pinhole between two sheets is cut at the contact;
- cavities and islands are kept;
- random blobs end with b1 = 0 and genus-0 meshes, and any box crop has genus 0;
- on finer blobs, which made blocked walls in 40 of the first 75 seeds before the set claims, every stranded voxel would make a handle if put back.

**Hemisphere** (`prototypes/scripts/wm_handles.py`, 4 min, 19 GB peak):
- WM label: 31.6 M voxels; (b0, b1, b2) goes from (1, 243, 31) to (1, 0, 31).
- 101 cuts remove 3,334 voxels and 118 fills add 1,998. None are stranded.
- Most moves are tiny. The median cut is 2 voxels and the median fill is 1; 49 cuts and 69 fills are one voxel.
- The largest move is a cut of 2,047 voxels (8.4 mm³), around ITK-SNAP voxel (230, 92, 651) of the full grid. It spreads over 5 × 18 × 20 mm, where a blocked cluster was claimed whole. It is outside both boxes. **Worth a look in ITK-SNAP** (overlay `wm_handles_overlay.nii.gz`: 1 cut, 2 fill).

**The boxes** (`prototypes/scripts/wm_handles_boxes.py`; mesh as `column_volume.white_mesh`):

| | M1S1 | V1V2 banks |
|---|---|---|
| voxels removed / added in the box | 0 / 0 | 198 / 0 |
| pieces before → after | 3 → 3 | 5 → 6 |
| largest piece, genus before → after | 0 → 0 | 2 → 0 |
| every piece genus 0 after | yes | yes |
| white surface area, mm² | 2128.6 → 2128.6 | 1863.8 → 1866.4 |

V1V2's three cuts, in ITK-SNAP voxels (1-based, box):
- **(51, 136, 45), 1 voxel.** This is handle 1, the 0.09 mm pinhole at (49, 134, 43).
- **(124, 120, 15), 71 voxels** (0.29 mm³, a cut across the bridge). This is handle 2, the 5.8 mm loop at (123, 119, 14), so the cut takes the possible white-matter bridge apart.
- **(116, 139, 38), 126 voxels in the box** (144 in all), 0.32 mm from the box face. The handle closes outside the box. In the box, this cut separates a 38.7 mm² piece from the largest piece, which is the sixth piece. The cut runs along a thin strand of white matter rather than straight across it. The wall where the fronts meet need not be the smallest cut.

**Views.** In `scratch/topofix` (not kept):
- `wm_handles_<site>.png`: slices through the changes in each box;
- `wm_handles_<site>_overlay.nii.gz` and `wm_handles_<site>_wm_fixed.nii.gz` on the box grid.

**Not done yet.** `column_volume_<site>.npz` and everything built from it still use the uncorrected label. A rebuild needs the user's go-ahead.

**Open.**
- The 2,047-voxel cut.
- Whether a cut should be the smallest cross-section rather than the meeting wall, as in V1V2's third cut.

### Across both faces

- **The principle survives at V1V2 and fails as estimated at M1S1** (principle, roles, estimators): the same estimators with the same settings place a V1 contact within about half a voxel and miss an M1/S1 contact by 3–5 voxels, while placing open-cortex joins at both within a third of a voxel.
- **The paths are a problem at both faces.** Streamlines of the unsigned normal cross at only 9 of 35 and 9 of 23 points. Column pairs cross more often, but a third or more slide back to their own bank.
- **The likelihood is 2–3× overconfident everywhere** (spliced columns, both faces), which makes E2 follow the profile even where it is wrong.

### Next (for the user's decision)

- **Diagnose the M1S1 miss before any change:** compare, at the 19 plane-line paths, each side's profile with the template it was fitted to (the templates are not saved now), and with templates from the thin bank's own tops nearest the fused zone (finding 14: about 2.1–2.2 mm) instead of everything within 6 mm.
- **Make the likelihood's uncertainty honest:** measure the residual correlation along open columns, or use a Student-t likelihood. Either way, check it on the spliced columns before using E2 again.
- **The paths:** whether the turn stop should hand over to the column pair where the normal fails, and how to keep column pairs from sliding back to their own bank.
- **The column volume:**
  - the tracer's start: the spike of turns at 0.3 mm says the first few tenths of a millimetre should follow the mesh normal, blending into the data's normal with coherence, rather than the data's normal from the WM face;
  - the fallback's contact needs a rule without a separation threshold: the side switch along the trace's own labels (the score's two-ends rule) instead of "a root more than 5 mm away";
  - a contact estimate from the volume itself: the open top horizon continued sideways into the fused stretch (horizon tracking across neighbouring traces), to compare with the others at the points;
  - lateral smoothing along the mesh and local charts, still open.

## Decisions

Decided by the user (2026-10-27):
- **D1. Path family:** both (a) normal streamlines and (b) one-pass column pairs, compared per point; (c) as a diagnostic.
- **D3. Template stretch:** both linear in distance and top-anchored, as separate estimators.
- **D6. The prior:** reported three ways: profile only (flat prior, E1), profile with the thickness field (σ² posterior, E2), and the field alone.
- **D7, D8. First round:** location only. Synthetic tests and the open-cortex calibration (test 2), then the 35 V1V2 points, then the 23 M1S1 points. Detection after the user has seen location.

Still open (defaults used until the user says otherwise):
- **D2. Features:** trend-removed echo 2 and 3 jointly in the likelihood; raw log echoes in the views only.
- **D4. Template locality:** open columns whose roots lie within 6 mm of each end root along the white surface; wandering-top guard at path length over straight distance 1.3.
- **D5. Noise model:** Gaussian with the template's per-depth spread; Student-t if a few samples dominate the residuals.
