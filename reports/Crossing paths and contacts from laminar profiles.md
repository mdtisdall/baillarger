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

The tracing is in the package (`baillarger.column_traces`, `baillarger.streamlines`); `column_volume.py --columns` picks the method.

- **Primary tracer:** the streamline of the fitted field T (`fitted_blend`), started at the vertex along its outward normal and blended into the field over 0.3 mm. A WM label is ignored for the first 0.3 mm.
- **`--columns crest` (the default):**
  - *The field.* T is extended one voxel into the fluid and 0.5 mm into the WM, so its gradient is T's own up to the pia.
  - *No turn rule and no fallback.* Columns are traced through.
  - *Crest hits are recorded, not acted on.* Where a column meets another bank's field it crosses T's crest, and `crest_at` records its first sample there. The crest is where T rises from two sides into a kink (`crest_voxels`: a noise- and curvature-bounded kink test, *Column traces without the one pass*). It is the midline between the two banks' WM, not the anatomical contact, which the reading of the columns places.
  - *Reversals.* A column that reflects off a crest it cannot cross and goes back into its own WM is cut at T's largest value along it.
- **`--columns one-pass` (the method before, for comparison):** T on the cortex alone, the turn rule (60° within 0.3 mm), and where the primary turned or reached the length cap, the fallback: ascent of the equal-arrival pass's arrival, stopped at fluid, at WM, or where the pass's label belongs to a root 5 mm or more away along the white surface.
- **Recorded per trace:**
  - which tracer made it (`one-pass` only);
  - the outcome (fluid, WM, contact, turn, length, face, missing, other, ridge);
  - the length and the root and end positions;
  - the end's distance from the root along the white surface;
  - `crest_at`.
- **Trace classes:**
  - *open:* ends in fluid;
  - *crossing:* ends on WM whose root is 5 mm or more away along the white surface, measured on geodesic patches of the white mesh (`baillarger.mesh_graph`);
  - *own bank:* ends on WM within 5 mm;
  - *to the contact:* a reversal cut at the crest (`crest`), or the fallback stopped at the pass's contact (`one-pass`);
  - *failed:* face, missing, other, or the length cap (`one-pass`: also a ridge).
- **Faces:** traces whose root is within 2 mm of a box face are kept but flagged, and left out of every summary. Most failures are traces from deeper roots that run into this margin (*Column traces without the one pass*).

### Horizons (per trace, in mm from the root; NaN where not defined)

- **top:** the length of an open trace.
- **far WM:** the length of a crossing trace (its L).
- **contact:** the length of a to-the-contact trace (key `fallback contact`).
- **the profile's split, on crossing traces:** E1 top-anchored, with templates from open traces in the volume within 6 mm of each end root. This one is costly, so it is computed on a random subsample of crossing traces (about 3,000 per box) and marked as sparse.

The one pass's horizons (its label boundary with equal and field weights, and the field prior's pairwise posterior) were debugging information and are no longer written.

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

**The priority map** (both sites, 30 s, 6–7 GB): computed with seven doubts, including the field prior (|joint − field prior|), which has since been removed with the column volume's one-pass horizons; the combined score is now the mean of six ranks.

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

### Column traces without the one pass

**Code.**
- In the package:
  - `baillarger.column_traces`: `--columns crest` (the default) and `one-pass`; `crest_at`;
  - `baillarger.streamlines`: the extended fitted field (`fitted_field(fluid=, wm_band_mm=)`), `crest_voxels`, `step_noise`, `n_eff`, `fitted_free`, `descend`;
  - `baillarger.mesh_graph`: the white mesh's graph, geodesic patches, and the vertex where a path crossed the mesh.
- Diagnostics in `prototypes/scripts/`:
  - `turn_diag.py` and `turn_cases.py`: why the primaries turned;
  - `turn_experiment.py`: the extended field, with and without the turn rule;
  - `bend_sweep.py`: the bend threshold;
  - `crest_noise.py`: the noise model and the crest test on the phantoms.
- Inputs: all site results are on the handle-free WM label (`wm_handles.npz`) with `fitted_blend`, roots 2 mm or more from a box face.
- Tests (exact answers):
  - mesh graph: distances on a plane within the 8-neighbour bound; a blade's sides apart, and the crossing vertex on the entered side;
  - extended field: the signed distance on a slab, with |grad T| = 1 at the pia;
  - crest test: none in a curved single-source field (T = R − r), with or without a noise margin; exactly the kink plane between two flat banks;
  - `n_eff`: the window's variance reduction, Monte Carlo within 10 %;
  - crest method: columns cross parallel fused banks and meet the crest at the midline; in a narrow fused V they meet it on the bisector; on a slab they meet none;
  - crest cut: a returning column is cut at T's maximum.

**Patches on the white mesh.**
- The bank rule's distances along the white surface now come from geodesic patches on the white mesh, not on the voxel seed graph.
- The graph includes the edges across neighbouring triangles. Without them, Dijkstra overestimates by up to 41 % in some directions, because the triangles' own edges zigzag on a grid-aligned surface. With them, the bound on a plane is 8.2 %.
- Against the straight line between the same patch centres at M1S1 (p10, p50 of geodesic ÷ chord):

| Straight line | Seed graph | Mesh graph |
|---|---|---|
| 2–4 mm | 1.06, 1.11 | 1.01, 1.06 |
| 4–6 mm | 1.07, 1.12 | 1.02, 1.07 |
| 6–8 mm | 1.07, 1.11 | 1.02, 1.05 |

- Moving the patches alone changed 944 (V1V2) and 1,471 (M1S1) classes: mostly traces near the 5 mm rule, and the fallback's contact stops.

**Why the primaries turned** (`one-pass`'s tracer, on the handle-free label):
- **Where:** about 2 mm out, not at the root: 0.5–2.4 % of turns within 0.3 mm of the WM. The blended start had removed the start spikes.
- **Rate by root curvature at M1S1:** 59–65 % of fundus roots (H < −0.15 /mm) turn, against 8–16 % elsewhere. At V1V2 the rate is 19–35 % at every curvature.
- **|grad T| at the turns:** 0.35 near the fluid, 0.43 in the interior, 0.51–0.58 at crests. Typically it is 0.96 (0.76–1.08) in single-bank cortex, and 0.97–0.98 along traces that end normally.
- **Near the fluid:** an artefact. T was filled flat outside the cortex, so within a voxel of the pia its gradient shrank and tilted.
- **Elsewhere:** at T's crests, where two or three banks' fields meet.
- **The paths themselves were right:**
  - their bodies bend 0.27° per 0.04 mm step;
  - over the 0.3 mm before the turn they change heading by 15–20° (median);
  - a one-step wiggle where the field is weak tripped the 60° rule;
  - 14–16 % of roots then lost a good primary to the fallback.

**The extended field and the turn rule** (`turn_experiment.py`; primaries only):

| | Crowns and fundi, right class | Failed | Kissing pairs, right class |
|---|---|---|---|
| T as before, turn rule | 84–88 % | 12–16 % | 86–87 % |
| T extended, turn rule | 95–96 % | 4–5 % | 90–91 % |

- The phantoms' turns were all the pia artefact: with T extended, the turn rule changes nothing on them.
- **At the sites:** extending T turns about 1,700–1,800 turned roots per site into open columns. Elsewhere paths move a median 0.02 mm (p99 0.12 mm).
- **Without the turn rule:** of the remaining turned roots, 8,500 per site become "own bank", and they are bounces. They are half as long between their ends as along their path, half end on a wall facing like their root's, and only 1–3 % cross straight to an opposite wall. They reflect off a crest met at an angle.

**A bend threshold cannot decide** (`bend_sweep.py`; columns ended at a bend sharper than the threshold within 0.3 mm):

| Bend threshold | Classes changed from 60° | V1V2 crossing columns |
|---|---|---|
| 45° | 4.7–4.8 % | 4,665 |
| 30° | 11–12.5 % | 3,128 |
| 10° | 31–34 % | 365 |

Crossings kink where they meet a contact at an angle, by about the angle between the banks' layers. So an angle slices a continuum of real crossings. On the phantoms, whose contacts are met head on, the results are stable from 30° to 60°.

**The crest test** (`crest_noise.py`, on the fold phantoms at V1V2's features, noisy against noise-free):
- **The rule:** a voxel is a crest if T rises significantly into it, does not rise significantly beyond it, and its slope changes across it by more than L²κ plus the noise margin. κ is the concave curvature of T's level sets, from the white mesh's k2 carried to depth as |k2| / (1 − |k2| d). A smooth tangential maximum changes slope by at most L²κ; a kink changes it by a finite amount. Only values T was solved for count: not pinned at the WM's edge, not filled.
- **The noise model** (`step_noise`, no calibration):
  - the noise power is the cortex's median smallest eigenvalue of the structure tensor;
  - the angular error is σ²(S + σ²) / (S² N), with N = `n_eff` exact for the filters;
  - one component of the curl-free part keeps a ninth of the variance.
- **The noise power from λ₃** is within 1–7 % of the known value on the fundi and kissing pairs, and 25–45 % high on the crowns, where curvature leaks into λ₃.
- **Actual ÷ predicted step noise:** 1.0–1.2, and 1.2–1.6 in the lowest quartile.

| Phantom | Crest voxels (noisy) | Noisy crests within 1.5 voxels of the noise-free ones |
|---|---|---|
| Crowns | 0.01–0.03 % | 80–100 % |
| Fundi (no contact) | 0.5–1.1 % | 98 % |
| Kissing pairs | 5.3–6.2 % | 99.6–99.7 % |

**The crest is not the contact.** In the kissing pair (banks 2.0 and 2.6 mm thick, contact at r = 3.6 mm), the crest lies at r = 3.9 mm (p10–p90 3.8–4.0). That is the midline between the two WM surfaces, 1.6 and 6.2 mm. It lies there in the fused sector, and also in the open sector, where a fluid gap separates the banks: T follows the WM-distance direction in the fluid, so it does not see the gap. So the crest is recorded on each column (`crest_at`) and not used to end it: the boundary there has to come from the reading of the columns.

**At the sites:** see *One crossing rule: traverse and front*, which compares the methods on the current field.

### T's sign through tissue, and the cortex label at its crests

**Code.**
- `baillarger.streamlines`:
  - `tissue_wm_direction`;
  - `fitted_field`: leaves out edges with fluid at both ends;
  - `crest_kinks`: the crest test, which also returns the lattice direction across each kink (used by `crest_planes.py`; the column methods use `crest_voxels`).
- `prototypes/scripts/`:
  - `methods_phantoms.py`: the methods on the fold phantoms, through the package;
  - `crest_planes.py`: crest voxels on a site's marked planes.
- Tests (exact answers): the WM direction through tissue does not flip beside an open slot, and flips where banks are fused; columns stay open beside an off-centre fluid slot.

**T's sign through tissue** (all crest-based methods). T's target direction was signed by the straight-line distance to the nearest WM. Beside an open sulcus, that flips inside a bank's cortex, so T had a crest short of the fluid. It is now signed by the WM distance through tissue (fast marching, with fluid, other labels and missing tissue as obstacles):
- smoothed over tissue only;
- obstacle voxels take their nearest tissue voxel's direction;
- the fit leaves out edges with fluid at both ends, so two banks' fluid layers across a narrow sulcus do not couple.

On an off-centre 2-voxel slot, T now rises in each bank up to the fluid, with no crest in the cortex.

**The cortex label at T's crests.**
- On V1V2's marked planes (`crest_planes.py`; crops in ITK-SNAP), crest voxels lie along the marked "contact", which the user identified in 3D as a thin fluid channel; the crest follows it, departing into cortex at points. They also lie around a dark structure in thick cortex, which the user identified as a small piece of superficial vessel outside the cortex.
- Both are labelled cortex by the aseg, so T treats them as cortex and the two banks' fields meet across them.
- Next to the pia, a rim of fluid-like voxels is labelled cortex too (the user, on the column views). Along 8 V1V2 front columns its majority-fluid tail is 0.16–0.36 mm (median about 0.24 mm), relabelled in 7 of them by the pial-side label refinement (PR #49, `scratch/pial-labels/rim_columns.json`); a partial-volume ramp of another 0.15–0.3 mm stays cortex.
- So crest voxels without a fused contact mark places where the label is wrong, which makes them a flag in their own right.

### One crossing rule: traverse and front

**Question.** How should a column go on past T's crest, where T stops rising and another bank's field begins, and does moving the columns as one coupled surface (the way FreeSurfer grows the pial surface from the white) change them? Columns stay complete rays: they stop only at the fluid, the far bank's WM, missing or other tissue, a box face or the length cap, and never where two banks' fronts meet; the reading of the columns places the cut.

**The rule** (`baillarger.column_front`, shared by `traverse` and `front`):
1. Climb ∇T (the extended fitted field, as `crest`; started along the mesh normal and blended into the field over 0.3 mm, as fitted_blend).
2. Within the stencil of T's crest voxels (`crest_voxels`, dilated by one voxel along each axis for the central differences, reached by any trilinear corner), keep heading: T has no gradient at a kink. The sampled one there averages the two banks' and, where the crest rises along its length (a fused V, whose crest climbs toward the opening), points along the crest, so a climb that follows it slides along the crest and never reverses.
3. On leaving that stencil, if T falls ahead (the unit gradient points against the last step), descend −∇T for good, into the far bank. Across a kink between two banks, −∇T beyond is the continuation of +∇T before.

The rule is frozen (the user's decision, 2026-10-28).

**The two methods.**
- `traverse`: the rule on independent columns.
- `front`: the rule with the columns linked. The whole white mesh is advanced as one surface, 0.04 mm per step; the moving vertices' directions are smoothed along the mesh by a Gaussian of 2 voxels (the white mesh's own twicing scale, below which the mesh carries no shape), twiced, over the moving vertices only, then normalized. Each root's trajectory is its column. Per sample it records the root vertex's dual area (`area`, mm²) and the volume its column has swept per unit root area (`vol`, mm: the flux of the vertex's dual area vector over the steps, on the mesh halfway through each step), so `vol / vol[end]` is an equivolume (Bok) relative depth between the root and any cut. `--columns normal` (PR #47) records `vol` with the same meaning, from the root's curvatures.

**Code.**
- `baillarger.column_front` (`grow`, `past_crest`, `dual_areas`); `baillarger.column_traces`: `--columns traverse` and `front` (`area`, `vol`, `turn_at`: the sample where a column began to descend).
- `prototypes/scripts/column_volume.py` saves `area`, `vol`, `turn_at`; `prototypes/scripts/front_eval.py`: the four methods on the phantoms and the sites.
- Outputs: `scratch/front-columns/` (README).

**Tests (exact answers).**
- A ball and a cylinder of WM: front columns stay radial (median angle < 5°, max < 10°), and the front's area follows (1 + κs)² and (1 + κs): in total within 1 %, per vertex at the median within 3 %; the swept volume is the integral of that area within 1 %.
- A slab: straight columns, area constant, swept volume per root area equal to the length.
- Parallel fused slabs: traverse and front cross straight into the far slab, and begin to descend within the crest's stencil beyond the midline.
- Slabs at 30°: at least 90 % of the left bank's columns end in the right bank's WM, by either method.
- A narrow fused V: crest's columns slide along the crest to the fluid (median 15 voxels outward); traverse's and front's cross into the far wall near the height they started (median under 2 voxels).
- An off-centre fluid slot: columns stay open and end at the slot. T is extended one voxel into the fluid from each bank, so one voxel short of a narrow slot the sampled gradient differences the two extensions and points back; `turn_at` then records a descent on the last step, into the fluid.

**What the coupling does.**
- **Not sliding.** In the narrow V, front with the coupling and without rule 2 slides exactly as crest does; with rule 2, the coupling changes nothing there. Rule 2 is what stops the slide.
- **Bok's measure.** On the ball at 1.6 mm, each vertex's area is within 0.84–1.22 of (1 + κs)² (p5–p95) with the coupling at 2 voxels, 0.40–1.98 without it. A coupling of 4 voxels tilts the columns (median 8.6° off radial, against 3° at 2 voxels).
- **Tight crowns.** At a crown of radius 0.32 mm (2 voxels) the coupling averages over the whole tip and bends the columns (below).

**On the phantoms** (roots 2 mm or more from a face, at V1V2's features and measured noise; M1S1's within 0.02; `front_eval_phantoms_<site>.json`). Right class (open or crossing), then the path error, median / p90, mm:

| | crest | traverse | front | one-pass |
|---|---|---|---|---|
| Crown 0.32 | 0.94; 0.11 / 0.33 | 0.94; 0.11 / 0.33 | 0.95; 0.18 / **0.77** | 0.94; 0.13 / 0.41 |
| Crown 0.64 | 0.95; 0.08 / 0.17 | 0.95; 0.08 / 0.17 | 0.96; 0.10 / 0.25 | 0.95; 0.09 / 0.23 |
| Crown 1.28 | 0.95; 0.05 / 0.11 | 0.95; 0.05 / 0.11 | 0.95; 0.04 / 0.11 | 0.94; 0.06 / 0.17 |
| Fundus 0.16 | 0.95; 0.04 / 0.08 | 0.95; 0.04 / 0.08 | 0.95; 0.03 / 0.08 | 0.94; 0.05 / 0.14 |
| Fundus 0.48 | 0.96; 0.03 / 0.07 | 0.96; 0.03 / 0.07 | 0.96; 0.03 / 0.06 | 0.95; 0.04 / 0.09 |
| Kissing pair | 0.90; 0.05 / 0.11 | 0.91; 0.05 / 0.09 | 0.91; 0.04 / 0.07 | 0.90; 0.06 / 0.14 |
| Kissing pair, no dark band | 0.90; 0.08 / 0.18 | 0.90; 0.07 / 0.18 | 0.90; 0.06 / 0.12 | 0.90; 0.09 / 0.20 |

Contact error at the kissing pairs (median / p90, mm): crest 0.045 / 0.085 and 0.072 / 0.143; traverse 0.036 / 0.071 and 0.076 / 0.147; front 0.027 / 0.054 and 0.054 / 0.107.

- **The rule does not under-cross at fused contacts.** On both kissing pairs traverse and front reach the right class as often as crest, and cross where the banks meet head on or obliquely. About 9 % of every method's roots reach a box face.
- **Front is the most accurate wherever the folds are 0.64 mm or wider,** at the contacts most of all (contact error p90 0.054 against 0.071–0.085 mm). At the 0.32 mm crown its paths are the worst (p90 0.77 mm): the coupling's 2 voxels are the crown's radius.
- **Bok's measure on the phantoms** (front's swept volume against the true columns' equivolume volume, summed over the roots that reach s): within 0.2 % on the fundi, the kissing pairs and the 1.28 mm crown, at 0.4, 0.8 and 1.2 mm; within 1 % at the 0.64 mm crown; 6–15 % high at the 0.32 mm crown. Per column, the area's median is within 0.2 % of the truth except at the two tight crowns (2–13 % high); its p10–p90 widens with s, to 0.89–1.20 at 1.2 mm on the fundi.

**At the sites** (handle-free WM, fitted_blend, roots 2 mm or more from a box face; `front_eval_sites.json`; one-pass does not use the crest test and is unchanged):

| | V1V2: crest | V1V2: traverse | V1V2: front | V1V2: one-pass | M1S1: crest | M1S1: traverse | M1S1: front | M1S1: one-pass |
|---|---|---|---|---|---|---|---|---|
| Open | 34,668 | 34,532 | 33,675 | 35,752 | 27,755 | 29,313 | 27,487 | 29,586 |
| Crossing | 6,001 | 9,175 | 9,411 | 5,369 | 14,418 | 16,363 | 16,146 | 12,122 |
| Own bank (heading on / back) | 2,394 (2,394 / 0) | 4,511 (2,969 / 1,542) | 6,202 (3,146 / 3,056) | 970 | 2,353 (2,353 / 0) | 5,361 (3,297 / 2,064) | 7,453 (3,782 / 3,671) | 479 |
| To the contact | 7,325 | — | — | 4,139 | 6,796 | — | — | 4,100 |
| Failed (box face / other / missing / length / ridge) | 13,727 (9,447 / 2,253 / 662 / 1,365 / 0) | 15,897 (11,220 / 2,253 / 624 / 1,800 / 0) | 14,827 (9,897 / 2,253 / 655 / 2,022 / 0) | 17,885 (9,513 / 2,253 / 777 / 0 / 5,342) | 21,028 (20,757 / 0 / 0 / 271 / 0) | 21,313 (21,004 / 0 / 0 / 309 / 0) | 21,264 (20,900 / 0 / 0 / 364 / 0) | 26,063 (19,039 / 0 / 0 / 0 / 7,024) |
| Marked points crossed (of 35; 23) | 17 | 33 | **35** | 16 | 23 | 23 | 23 | 23 |
| Profile split at the points (median, p90 mm; points) | 0.065, 0.415 (17) | 0.070, 0.645 (29) | 0.063, 0.290 (34) | 0.067, 0.165 (16) | 0.236, 0.827 (21) | 0.244, 0.628 (21) | 0.358, 0.613 (22) | 0.384, 0.584 (23) |

"Own bank" is the 5 mm rule along the white surface; "heading on" columns enter the WM still heading out from their root (crossings to a near wall, such as a fundus's other side), "back" ones return.

- **Every column is a complete ray.** Traverse and front have no "to the contact" class: the columns crest cuts at T's maximum go on, mostly to a near wall (own bank, heading on) or across (crossing).
- **Front reaches every marked point at V1V2** (35 of 35, crest 17, traverse 33), and the profile split there is at least as good as crest's (median 0.063 mm, p90 0.29 against 0.415). At M1S1 all methods reach all 23; front's split median is worse than crest's and traverse's (0.36 against 0.24 mm) and like one-pass's; the estimator's +0.2 mm bias at M1S1 is unchanged, and the crossing columns differ, so the split reads different profiles.
- **Linking against independent columns** (front against traverse, same rule): paths within 0.065 / 0.57 / 3.1 mm (p50 / p90 / p99) at V1V2 and 0.060 / 0.33 / 1.8 mm at M1S1 where both end alike; 6–8 % of the classes change, mostly open or crossing to own bank (V1V2: 898 open and 656 crossing become own bank, 262 and 272 the other way). Front has more columns that turn back into their own WM (3,056 against 1,542; 3,671 against 2,064). Where neighbours on the front descend, a vertex is carried along with them.
- **The crossing rule against traverse's former lattice step** (one lattice step along the kink test's direction from the first crest voxel, then descent): that sent 32,320 (V1V2) and 30,568 (M1S1) columns into WM within 5 mm of their root. Of the columns it took from crest's open class (17,688 and 12,960), the rule leaves 88–91 % open. Along those columns T rises evenly, with slope about 1 from root to fluid, and the crest voxels they pass lie beside them, their kink directions mostly across the column's plane: crest voxels cover 10.5 % of V1V2's cortex (113,093 of 1.08 M), in broad patches. The step crossed them sideways and descended into the column's own WM; the rule, seeing T still rising ahead, goes on to the pia. On the phantoms the step also missed at the 0.16 mm fundus (0.55 right) and the kissing pairs (0.82, 0.78).
- **Most failures are still the box face** (65–75 % at V1V2, 98–99 % at M1S1). The length cap takes 0.4–3 %: columns that never meet fluid or WM within 8 mm.
- **Front's area at the sites:** at the open columns' tops it is 0.22–2.9 × the root's (p10–p90; median 0.76 V1V2, 0.82 M1S1), and the swept volume per root area is 0.51–1.6 × the column's length (median 0.80, 0.82): on average the columns converge outward at these faces, as expected where most roots face fundi and fused stretches.

**Not done.**
- Rebuild on the refined pial labels (PR #49) and measure again; crests where no banks meet should fall.
- Whether the crest test marks too much: 10.5 % of V1V2's cortex, mostly with kink directions out of the plane of the columns; a resliced view across those patches is the next look.
- The field-free equivolume front (speed ∝ A0/A, its own session) and `--columns normal` (PR #47), compared at the marked points.
- A full top boundary for T at the pia. The sign through tissue makes T respect labelled fluid; a fluid label that misses a channel still hides it.
- Padded boxes, so that columns can run past the crop (most failures). A 4 mm margin is about 2.5 × the volume, roughly 11 GB.
- Collisions between banks on the moving front, detected exactly (`mesh_intersect`), recorded, not acted on.
- Step 2's use of `crest_at` and `turn_at`, for example an edit-priority layer of crests without a fused contact, as likely label errors.

### Pial-side label refinement

A GM refinement step at the front of the pipeline, like the WM one: `baillarger.pial_labels` finds voxels that the aseg labels cortex but the image shows are fluid, writes a change file (`LabelChanges`, the counterpart of `baillarger.white.WMChanges`), and `column_traces.load_tissue(..., pial_changes=...)` applies it after the handle-free WM. Everything downstream (T, the crest, the columns, the viewer) reads the refined labels. The detector is trusted by default: its best guess, every cell it does not reject, is what the step writes and what `load_tissue` applies. Corrections go into the change file (statuses and brush edits), and the cells most worth a look are ranked first in a review queue, reviewed in the viewer (below).

**What the image shows.** At V1V2 the fluid is fomblin at the noise floor (45 in every echo, against 514 for cortex in echo 2). The aseg's cortex runs one to three voxels into it around most sulci, and fills the thin channels of sulci it fused, such as the user's marked "contact" at (270, 133, 139). The vessel the user identified at (324, 146, 139) is a one-voxel-wide dotted dark line, partial-volumed.

**The method** (all closed-form; module docstring):
1. **Darkness.** Echoes 1, 2 and 3, log, minus their trend over the cortex (as in `column_traces.tissue`, re-estimated without the voxels the first pass found dark). A voxel is dark when two things hold:
   - it is significantly darker than its local cortex: the noise-weighted contrast is below −k, k the normal quantile for α = 10⁻³ per voxel;
   - it is more fluid than cortex: its fluid fraction, between the local cortex level and the box's labelled-fluid level, is above ½.

   Echo 1 has the most signal and the largest contrast of tissue against the fomblin; echoes 2 and 3 add the T2* contrast of blood in vessels. The fitted field T of step 4 is made, as in the column traces, from echoes 2 and 3.
2. **Noise.** Per echo, from the structure tensor: the median smallest eigenvalue over the cortex (away from its edges), divided by the derivative filter's gain and by the median smallest eigenvalue of a Wishart matrix with `n_eff` degrees of freedom (the sample tensor's smallest eigenvalue is biased low: 0.68 at n_eff = 39).
   - On white noise it is within a few per cent, high (tests).
   - The real noise is spatially correlated: neighbour correlation 0.5–0.67, even in deep WM. The estimate is then the white noise with the same gradient power.
   - At V1V2 it gives 0.124 / 0.208 / 0.300 (echo 1 / 2 / 3), against a cortex MAD of 0.105 / 0.188 / 0.272, which includes the layers. At M1S1: 0.075 / 0.140 / 0.221 against 0.075 / 0.137 / 0.207.
3. **Components.** 26-connected components of the dark voxels, at least n_min voxels. n_min is the smallest size expected less than once in noise: N p (e·25·p)^(n−1) < 1, the bound on connected sets in a graph of degree 26. That gives 3 at V1V2 and 1 at M1S1. Each component is split into review cells, geodesic Voronoi cells 0.96 mm apart (the root patches' spacing), because the rim and a fused channel are often one component.
4. **Position** (per component). A component is "outer" when it touches fluid or other labels, or when T does not pass through it within one bank:
   - step along ∇T out of the component on both sides;
   - T passes through a voxel when both ends are cortex and T rises significantly into it and beyond it (k times `step_noise` per voxel step, as in `crest_kinks`);
   - more than half passing through means "inside a bank": rejected.

   **The stria trap.** The stria of Gennari is a dark band inside one bank: T passes through it, so it is rejected even when it is as dark as a 70 % fluid mixture. A fused channel lies where the banks' fields meet, and a superficial vessel at the top of the cortex, so T does not pass through either. **Limitation:** a hidden channel off T's crest that reaches no fluid looks like a band inside a bank, and is left to the brush.
5. **Status, per cell.**
   - Cells of "inside" components are rejected.
   - Cells of components that touch fluid are accepted at once when their relabel keeps the tissue's Betti numbers (Kuhn adjacency, as in `baillarger.handles`) in a box round the cell, given the cells accepted so far. This runs in passes, so the rim opens from the fluid inward.
   - The rest are proposed, the review queue: hidden channels, and cells that would open a cavity, a tunnel or a split (as a channel between two banks does). Proposed cells are part of the best guess; the status only says which cells to look at first.
6. **Ranking.** By the columns of a column volume that pass through the cell, then the crest voxels within one voxel of it, then size. Descriptors per cell: shape (Westin: tube, sheet or blob), fluid fraction, contrast, and local Betti numbers before and after.

**The step on its own:** `python -m baillarger.pial_labels propose --out DIR --name NAME --lo I J K --hi I J K --wm-changes FILE [--columns column_volume_<name>.npz]`, as the WM step runs with `python -m baillarger.white changes`. It writes, on the box grid:
- `pial_labels_<NAME>.npz`, the change file;
- `_labels.nii.gz`, the refined rim labels, i.e. the WM changes and the best guess applied (0 other, 1 outside, 2 WM, 3 cortex);
- a `.json` summary with the cells in rank order;
- an ITK-SNAP overlay of the statuses.

A status can be changed and brush edits added with `LabelChanges.with_status` and `with_brush`. `load_tissue(..., pial_changes=FILE)` and `column_volume.py --label-changes FILE` apply the best guess (every cell not rejected) and the brush edits; with `reviewed_only` / `--reviewed-only`, only the accepted cells and brush edits.

**Review in the viewer, before any column is traced:** `python -m baillarger.viewer --labels pial_labels_<site>.npz`, with no column volume (labels only: the volume and labels panes). With a column volume (`python -m baillarger.viewer <column volume> --cache DIR --labels ...`, panes "volume + labels") the traces and surfaces are there too.
- The volume pane shows echo 1, 2 or 3 (a drop-down): the echoes the detector reads.
- The labels pane lists the cells ranked, with status, kind, columns, crest hits, fluid fraction and topology. A row moves the crosshair to the cell, in every window.
- Keys: j / k for next / previous; a, r, u for accept, reject, back to proposed.
- The volume pane shows the overlay (proposed red, accepted green, rejected yellow, brush blue / magenta). A brush (to fluid or back to cortex, radius 0–3 voxels) paints a voxel ball at a clicked point.
- Every change is saved at once to the change file; the first change keeps `.orig.npz`.
- Checked in the browser at both sites: accept, reject, brush, the saved file and the overlay.

**Tests** (exact answers, synthetic boxes):
- the noise from white noise and from a layered cube;
- the gain of central differences;
- the Wishart factor's limit;
- the component-size bound;
- the fluid fraction of a known mixture;
- shapes, cells, the Betti change of a slot, a cavity and a tunnel, and traces counted per cell;
- the change file's round trip, application order, WM protection and overlay;
- `tissue` with changes, and the review backend (`tests/test_viewer_labels.py`).

**Proposals:**
- **The stria trap on fused banks.** A hidden fluid channel at the WM midline is proposed (queue: it reaches no fluid). Two dark bands, one per bank, 90 % of whose voxels are dark, are not proposed.
- **The stria trap on a slab with a vessel.** A mid-depth dark band is not proposed. A tube on the pial surface is accepted, and its shape is a tube.
- **The channel removes the crest.** With the channel relabelled, T's crest at the midline falls below a tenth of what it was.

**At the sites** (handle-free WM; echoes 1, 2 and 3 in `scratch/pial-labels-echo1/`; echoes 2 and 3 alone, the first version, in `scratch/pial-labels/`; each with a README):

| | V1V2, echoes 1–3 | V1V2, echoes 2–3 | M1S1, echoes 1–3 | M1S1, echoes 2–3 |
|---|---|---|---|---|
| Cortex voxels | 1,081,775 | | 1,599,739 | |
| Dark voxels | 44,140 (4.1 %) | 51,034 (4.7 %) | 26,606 (1.7 %) | 27,581 (1.7 %) |
| Components (n_min) | 634 (1) | 405 (3) | 864 (1) | 1,671 (1) |
| Cells accepted (voxels) | 2,202 (34,832) | 1,992 (37,172) | 2,100 (24,850) | 2,141 (23,562) |
| Cells in the queue (voxels) | 378 (8,522) | 494 (11,076) | 106 (1,258) | 578 (2,422) |
| Cells rejected, inside a bank (voxels) | 157 (786) | 191 (1,624) | 140 (498) | 438 (1,597) |
| Tissue Betti numbers of the box (components, tunnels, cavities), aseg | (1, 9, 6) | | (1, 11, 5) | |
| … with the accepted cells | (1, 9, 6) | (1, 9, 6) | (1, 11, 5) | (1, 11, 5) |
| … with the best guess (accepted and queue: the default) | (2, 207, 190) | (6, 314, 324) | (1, 41, 79) | (1, 63, 560) |

With echo 1, the combined contrast's noise per voxel falls: echo 1's noise is about 0.6 of echo 2's. n_min falls to 1 at V1V2. The queue shrinks by a quarter at V1V2 and by four fifths at M1S1, and the best guess opens far fewer cavities: most of the two-echo queue's pockets were noise that echo 1 no longer finds dark.

**The user's three places at V1V2** (echoes 1–3; echoes 2–3 in brackets):
- The marked "contact" (270, 133, 139): a queue cell one voxel away, an 11-voxel tube with fluid fraction 0.69 (an accepted 15-voxel cell). It is in the best guess either way.
- The dark pocket (289, 118, 139): the top-ranked cell, now accepted, 35 voxels (the top of the queue).
- The vessel (324, 146, 139): only fragments within 5 voxels, a 5-voxel queue cell (7). A vessel one voxel wide rarely reaches a majority-fluid voxel at this noise, so it is left to the brush.

The best guess changes the tissue's topology a lot. The queue is, by definition, the cells whose relabel opens a cavity, a tunnel or a split: dark pockets with no mouth to the fluid in the box (hidden channels, or noise), and channels that cut the tissue at the box faces. The user chose to trust it by default and correct it in the review (2026-10-28).

**The former traverse re-measured** (its lattice step along the kink direction, before the shared rule of *One crossing rule: traverse and front*; the best guess of the two-echo version, unreviewed: accepted and queue; not re-measured with echo 1, so as to measure columns only on reviewed labels; roots 2 mm or more from a face):

| | V1V2 crest | V1V2 crest, relabelled | V1V2 traverse | V1V2 traverse, relabelled | M1S1 crest | M1S1 crest, relabelled | M1S1 traverse | M1S1 traverse, relabelled |
|---|---|---|---|---|---|---|---|---|
| Open | 34,668 | 49,286 | 15,336 | 26,712 | 27,755 | 36,865 | 14,746 | 22,429 |
| Crossing | 6,001 | 717 | 4,967 | 641 | 14,418 | 9,611 | 9,916 | 7,385 |
| Own bank | 2,394 | 1,060 | 32,320 | 27,670 | 2,353 | 2,178 | 30,568 | 26,790 |
| To the contact | 7,325 | 3,218 | 194 | 10 | 6,796 | 4,547 | 8 | 9 |
| Failed | 13,727 | 9,834 | 11,298 | 9,082 | 21,028 | 19,149 | 17,112 | 15,737 |
| Open columns meeting a crest | 62 % | 56 % | 14 % | 19 % | 56 % | 49 % | 18 % | 16 % |

- **Most of what crossed was fluid.** At V1V2, relabelling turns most crossing columns into open ones (crest: 6,001 → 717 crossing, 34,668 → 49,286 open). Most of the fused stretch, the user's marked contacts included, is a fluid channel the aseg filled. At M1S1 a third of the crossings go.
- **The former traverse improves but is still wrong** for more than half of the open columns: own bank −14 % at V1V2 and −12 % at M1S1, open +74 % and +52 %.
- **The crest test fires inside the cortex.** The crest still covers 9.9 % of V1V2's cortex and 5.2 % of M1S1's (10.4 % and 5.9 % before; `crest_after_<site>.png`). Of the own-bank columns' crest meetings, only 12 % (V1V2) and 9 % (M1S1) lie within one voxel of fluid. So the remaining failure is the crest test's hits inside the cortex, mostly in thick cortex, not the labels. Traverse stays an option, not the default.
- **The profile split at the points** is made on a random subsample of crossing columns. With relabelling, far fewer columns cross at the marked points (V1V2: 10–14 of 35 points), so its numbers are not comparable.

**Not done:**
- **The user's corrections** at both sites. The table applies the best guess; corrections can take queue cells out, overturn an accepted or rejected cell, or add brush edits.
- **An automatic topology fix for the queue,** like `remove_handles` for the WM: open a mouth for a pocket, or leave it as cortex.
- **A detector for thin vessels and channels:** a line / sheet-matched statistic (Hessian at one voxel) would raise the SNR of one-voxel-wide structures above the per-voxel test.
- **The whole hemisphere at once,** as the WM refinement does; now per box.
- **The crest test's hits** in the cortex, which the former traverse crossed sideways (*One crossing rule: traverse and front*).
### Normal rays: columns without a field

**Code.**
- `baillarger.column_traces`: `--columns normal`; `equivolume`; `ColumnTraces.vol`.
- `baillarger.streamlines`: `ray` (straight paths in mm steps), `first_fall`, `tissue_wm_distance`.
- Tests (exact answers):
  - rays from a ball are radial, and `vol` matches s + s²/R + s³/(3R²) (median relative error < 0.1);
  - rays cross parallel fused banks and meet at the midline;
  - rays meet on the bisector of a narrow fused V;
  - rays see no meeting beside an open slot;
  - `ray` is a straight line in 0.04 mm steps; `first_fall` finds where a field stops rising.

**The method.** Each column leaves its root along the smoothed white-surface normal and goes straight at a constant speed. It stops only at fluid or WM labels (or missing or other tissue, the box face, the 8 mm length cap). It does not use T or the image.
- `crest_at`: the first sample where the WM distance through tissue stops rising along the ray. There the ray has passed the midline between its own bank and another bank, so it records the geometric meeting, not T's crest.
- `vol`: the equivolume coordinate per sample, the swept volume per unit root area, with area (1 + k1 s)(1 + k2 s) from the root's principal curvatures. vol / vol at the end is an equivolume relative depth. The speed stays constant; a speed set by curvature is the equivolume-front session's option.

**At the sites** (same field, labels and roots as the former traverse's table; `scratch/normal-columns/`):

| | V1V2: crest | V1V2: traverse | V1V2: normal | M1S1: crest | M1S1: traverse | M1S1: normal |
|---|---|---|---|---|---|---|
| Open | 34,668 | 15,336 | 38,625 | 27,755 | 14,746 | 31,863 |
| Crossing | 6,001 | 4,967 | 4,946 | 14,418 | 9,916 | 13,658 |
| Own bank | 2,394 | 32,320 | 386 | 2,353 | 30,568 | 26 |
| To the contact | 7,325 | 194 | 0 | 6,796 | 8 | 0 |
| Failed | 13,727 | 11,298 | 20,158 | 21,028 | 17,112 | 26,803 |
| …box face / length | 9,447 / 1,365 | 6,607 / 17 | 14,507 / 2,673 | 20,757 / 271 | 16,765 / 123 | 23,039 / 3,764 |
| Open columns meeting another bank | 62 % | 14 % | 12 % | 56 % | 18 % | 6 % |
| Profile split at the marked points (median, p90 mm; points) | 0.065, 0.415 (17) | 0.066, 0.165 (25) | 0.075, 0.183 (24) | 0.236, 0.827 (21) | 0.41, 0.63 (21) | 0.169, 0.597 (20) |

**On the phantoms** (right class, at V1V2's features; M1S1's within 0.02):

| | Crowns | Fundus, 0.16 mm slot | Fundus, 0.48 mm slot | Kissing pair | Kissing pair, no dark band |
|---|---|---|---|---|---|
| crest | 0.94–0.95 | 0.95 | 0.96 | 0.90 | 0.90 |
| traverse (former lattice step) | 0.94 | 0.55 | 0.79 | 0.82 | 0.78 |
| normal | 0.95 | 0.94 | 0.96 | 0.90 | 0.90 |
| one-pass | 0.94–0.95 | 0.93–0.94 | 0.95 | 0.90 | 0.90 |

The paths of normal are straight, so they are further from the phantoms' curved columns: path error p90 0.18–0.41 mm at the crowns, against 0.11–0.33 for crest; 0.25 at the kissing pairs, against 0.11–0.18.

**What this says.**
- Straight rays keep the right class as well as the field-based methods do, on every phantom, and they do not misroute through the fundus focus as traverse does: there is no field to focus.
- At the sites they almost never return to their own bank (386 and 26 columns), and they cross fused banks about as often as crest does. Their open columns rarely meet another bank (12 % and 6 %), so most of crest's and traverse's crests on open columns come from T, not from the geometry, which agrees with the label errors found under traverse.
- The profile split at the marked points is as good as the best method at V1V2 and the best at M1S1 (median 0.17 mm, against 0.24–0.41).
- The cost: more failures. Straight rays leave the box more often (about 5,000 and 2,300 more at the faces) and reach the 8 mm cap more often (2,673 and 3,764). Apart from the face, the cap is nearly all of normal's failures. The others are V1V2's 'other' (2,253, the same for every method) and 'missing' (725) labels.
- **The capped rays.** They start along the WM distance's gradient (cos 0.99, as crossing rays do), and 99.7 % or more pass another bank's front, at 2.4–2.7 mm. After that they run along the far bank's sheet, not into it: the median |cos| between the ray and the far WM distance's gradient is 0.3–0.4, against about 0.9 for crossing rays. They come within about 0.9 mm of WM and 0.5–0.7 mm of fluid, but reach neither in 8 mm. A straight ray crosses into a far bank that is not parallel to its own and grazes it. By the user's decision, normal rays stay straight; the capped rays are kept as rays that end in cortex (outcome "length").

**Not done:**
- The crossing rays' cut: like traverse, normal's crossings depend on a cortex label without fluid channels and vessels; it is measured again after the pial-side label refinement.

### Equivolume columns against the normal columns

**Code.**
- `baillarger.column_traces`:
  - `--columns equivolume`: normal's straight rays and stops, with `vol` measured on the rays (`offset_equivolume`);
  - `--columns front-equivolume`: the white mesh grown with no field at equivolume speed (`column_front.grow_equivolume`); `ColumnTraces.tau`;
  - `meet_at`: where a column meets another bank's columns, for both.
- `baillarger.column_front`: `grow_equivolume`, `front_normals`, `face_sums`; `SPEED_CAP`, `TAU_CAP`.
- `prototypes/scripts/equivolume_compare.py`: the three methods at the marked points.
- Tests (exact answers), for both methods:
  - ball and cylinder of WM: columns radial, and the coordinate is (r³ − r0³)/(3r0²) and (r² − r0²)/(2r0), the median column within 2 % at 0.4–1.6 mm;
  - slab: straight, the coordinate is the arc length;
  - the inside of a WM tube (the concave model of a fundus; area falling as r/r0): (r0² − r²)/(2r0), within 2 %;
  - the front's speed cap at a focus (a tube down to 3 voxels): the cap holds, the median column's flux lags tau, and at least 99 % of the columns reach the fluid.
  - r0 in these tests is the label surface's mean radius. The mesh's sub-voxel bumps (sd about 0.17 voxel in a root's radius) are carried out as an offset, so a column's own r0 is not its reference.
  - A fold with closed-form equivolume layers other than the tube is not cheap: at a V's corner the layers are singular (all the corner's volume is on one line).

**The two methods.** Both use the white mesh's own geometry and no image, and both measure a column's cross-section A(s) from its neighbours rather than modelling it from the root's curvatures as normal does.
- **equivolume (straight rays).** A column is normal's ray. Its cross-section at s is its root vertex's dual area on the white mesh with every vertex moved s along its own normal, projected on the root's normal and clipped at 0 past a focus. vol = ∫ A/A0 ds. The layers are the levels of vol across neighbouring rays, and the rays are not kept normal to them.
- **front-equivolume (bent columns).** The whole mesh grows as one front. Each step, a vertex moves along the front's own normal (area-weighted, smoothed as `smooth_normals` smooths the root mesh, at the same reach in mm however much the front has shrunk) at speed A0/A. Every column then sweeps equal volume per unit root area per step: the fronts are the levels of tau, the equivolume layers, and the columns bend to stay normal to them.
  - Midpoint (RK2) steps; a forward Euler step biased tau by about 2 % on the cylinder.
  - At a focus (a fundus seen from the WM) A → 0. The speed is capped at 4 times the root's, one voxel per 0.04 mm of tau, and each step advances tau so that no vertex moves more than 0.04 mm. Where the cap binds, the swept volume per root area (`vol`, the measured flux) falls below tau.
  - The front tangles where it shrinks: each concave bump of the mesh has its own focus, and vertices cross there. Where the front has an edge (a stopped neighbour) or a fold (a face whose normal points against its vertices' headings), a vertex has no normal and goes straight at its last speed. Without these two rules, 5 % of the tube's columns turned back into their own WM about 3 voxels from the axis.
  - No tangential relaxation: a column is a vertex's trajectory and its dual area is its cross-section, so moving vertices along the front would trade volume between columns and bend the paths off the normal.
  - Samples stay 0.04 mm of arc apart, as for every method; `tau` (npz) records each sample's coordinate, and `vol` and `area` are as front's.
  - At a crown the column slows (A grows). The front's time is capped at 16 mm, twice the 8 mm arc cap (`TAU_CAP`): a column on a cylinder-like crown of radius 0.32 mm (the tightest the mesh keeps) then reaches 2.9 mm of arc, 3.5 mm at 0.5 mm; on a point-like crown 1.4–2.7 mm. A column still running then ends as "length".
- **What "equivolume" means here** (both methods, and normal's `vol`): equal swept volume per unit root area. It is Bok's layer only if every column holds the same volume per unit root area from the WM to the pia; a relative depth between two cuts is then (vol(s) − vol(s0)) / (vol(s1) − vol(s0)).
- **Cost.** equivolume: about 2 minutes per site, as normal. front-equivolume: 16 minutes at V1V2 and 18 at M1S1 (peak 3.0 GB). The front does whole-mesh work every step until its slowest column stops: 400 or more steps of tau, two evaluations each, about 30 smoothing passes per evaluation.

**At the sites** (handle-free WM, roots 2 mm or more from a box face; `scratch/equivolume-front/`, normal's builds from `scratch/normal-columns/`; the same white mesh and roots):

| | V1V2: normal | V1V2: equivolume | V1V2: front-equivolume | M1S1: normal | M1S1: equivolume | M1S1: front-equivolume |
|---|---|---|---|---|---|---|
| Open | 38,625 | 38,625 | 34,425 | 31,863 | 31,863 | 29,566 |
| Crossing | 4,946 | 4,946 | 3,380 | 13,658 | 13,658 | 7,012 |
| Own bank | 386 | 386 | 219 | 26 | 26 | 229 |
| Failed | 20,158 | 20,158 | 26,091 | 26,803 | 26,803 | 35,543 |
| …box face / length | 14,507 / 2,673 | 14,507 / 2,673 | 15,198 / 7,697 | 23,039 / 3,764 | 23,039 / 3,764 | 24,240 / 11,303 |
| Open columns meeting another bank | 12 % | 20 % | 9 % | 6 % | 23 % | 6 % |
| Profile split at the marked points (median, p90 mm; points) | 0.075, 0.183 (24) | 0.075, 0.183 (24) | 0.061, 0.158 (16) | 0.169, 0.597 (20) | 0.169, 0.597 (20) | 0.553, 0.731 (20) |

The meeting is measured differently: normal by the WM distance through tissue (where it stops rising), equivolume and front-equivolume by the first column to reach a voxel at a smaller coordinate (`meet_at`).

**At the marked points** (every point is passed within 0.3 mm by columns of all three methods; per point, the median over its columns; then p50, p90 over the points):

| | V1V2 (35 points) | M1S1 (23 points) |
|---|---|---|
| Coordinate at the point (mm): normal / equivolume / front-equivolume | 2.28 / 2.34 / 2.25 | 1.94 / 2.06 / 1.88 |
| \|coordinate − normal's\|: equivolume | 0.23, 0.73 | 0.16, 1.51 |
| \|coordinate − normal's\|: front-equivolume | 0.10, 1.26 | 0.06, 0.20 |
| Relative depth on open columns: normal / equivolume / front-equivolume | 0.26 / 0.29 / 0.20 | (no open columns of normal's at the points) |
| \|relative depth − normal's\|: equivolume; front-equivolume | 0.05, 0.33; 0.10, 0.46 | — |
| Path difference from normal, largest over the common arc: front-equivolume | 1.17, 1.61 mm | 1.21, 1.81 mm |
| Path difference from normal at the point: front-equivolume | 0.31, 0.64 mm | 0.25, 0.55 mm |

**What this says.**
- **The bending is real, and large.** By the point, a front-equivolume column is a quarter to a third of a mm (2 voxels) off its straight ray, and over its length 1.2 mm at the median. The layers do tilt across neighbouring rays enough that keeping the columns normal to them moves the paths by more than a voxel.
- **It costs the classes.** The front's columns reach the time cap far more often (7,697 and 11,303 against 2,673 and 3,764), so open and crossing columns turn into failures: it crosses fused banks about half as often at M1S1 (7,012 against 13,658). Its profile split is slightly better at V1V2, on fewer points (16 against 24), and much worse at M1S1 (0.55 against 0.17 mm).
- **The coordinate at the points is close for the front, not for the measured rays.** On the same paths as normal, equivolume's measured coordinate differs from normal's modelled one by 0.16–0.23 mm at the median point but by 0.7–1.5 mm at p90. The front's coordinate is within 0.06–0.10 mm of normal's at the median point. The cause of equivolume's tail is not diagnosed (normal smooths the curvatures over 0.5 mm; the measured area sees the mesh at its own scale).
- **So:** the straight rays (normal or equivolume) keep the classes and are 8–9 times cheaper; the front changes the paths by more than a voxel and loses columns at crowns and fundi. Which paths are right is not settled here: the phantoms' true columns are the test for that.

**Not done:**
- front-equivolume on the fold phantoms (`methods_phantoms.py`), for path errors against the true columns.
- Equivolume's tail against normal's coordinate: where the measured and modelled areas part, and at what scale.
- A faster front: work only on the part of the mesh that still moves (most columns stop within a few mm, but every step costs the whole mesh), and set the smoothing passes per region.

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
- *Streamlines.* The panel *Streamlines at the crosshair* (below the sections) shows, in true geometry and in mm, the traces at the crosshair's root:
  - *from*: every trace rooted within r of the root along the white mesh (r = 0.5, 1 or 2 mm);
  - *to*: every other trace whose last sample is within r of the root in space, mostly crossing traces from the facing bank;
  - the true white mesh and the GM boundary (the cortex label's contour, translucent, its opacity on a slider) in a box of ± 3, 5 or 8 mm.

  Traces are coloured by class. Fallback traces are darker and *to* traces paler. The pia, current and as saved, is a dot on each subsampled trace. The panel draws the column volume's traces as saved, so it shows whichever tracer made `column_volume_<site>.npz`, and it names that tracer and the WM label (`tracer_name`, `wm_label`; files made before these were saved say so). Drag rotates, shift-drag pans, the wheel zooms. Its camera starts at the main view's orientation.
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
- *Streamlines* (`streamline_view.py`). *From* uses the geodesic distance along the mesh, not the distance in space. Across a sulcus the facing bank's roots can be closer in space than the root's own neighbours, so they are not counted as *from*. Their traces show as *to* where they end at the root. The GM boundary is contoured from a crop padded by the smoothing's reach (σ = 1 voxel, for display only), then clipped to the box and left open where the box cuts it.
- *Flat overview* (`flat_map.py`). A least-squares conformal map of each piece of the white mesh. Pieces of genus > 0 are first cut along their shortest non-separating loops (greedy homotopy basis, cones in the holes).

**Edit records.** Each edit stores:
- the trace (index into `sub`) and the depth;
- its sample's box voxel, ITK-SNAP voxel (1-based, box) and full-grid voxel;
- world mm (the echo 2 affine);
- the time.

Each step stores R and its solve: free traces, seconds, number of pia changes, largest change.

**Tests** (exact answers, or a stated tolerance where only the limit is exact). The flat map, inflation, section and streamline code now lives in `baillarger.viewer` (*A viewer in the package*, below), and its tests moved with it to `tests/`:
- `tests/test_viewer_flatmap.py` (9):
  - a plane maps isometrically up to a similarity (1e-9);
  - a polygonal cylinder strip unrolls exactly;
  - a hemisphere cap keeps angles: the area-weighted mean of s1/s2 − 1 is under 0.025 at 4000 sphere points and halves when the spacing halves;
  - a torus's shortest handle is its meridian, exactly;
  - a genus-1 surface with a hole is cut to genus 0, then to a disk;
  - a long tube needs the disk cut (free-boundary LSCM folds on it);
  - packing and scale.
- `tests/test_viewer_inflation.py` (5): a uniform plane grid does not move; the boundary stays pinned; convergence to the discrete harmonic membrane; the tangent transfer is the face map; the sulcal depth.
- `tests/test_viewer_sections.py` (5): a plane gives the row at Euclidean distances; a cylinder gives the ring at arc lengths; a U fold gives the U at arc lengths; the directions; the points.
- `prototypes/scripts/test_pia_editor.py` (1): replay equals the live result under other `rand()` states.
- `tests/test_viewer_streamlines.py` (6, with the TCK round trip):
  - on a plane, *from* is the geodesic disk and *to* the traces ending within r;
  - on a U fold, the facing sheet's traces are *to*, never *from*;
  - the mesh crop keeps exactly the faces in the box;
  - a ball's GM boundary is a closed sphere at its radius, and the box clips it open;
  - the pia point lies on the trace at its arc length.

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
| streamline panel at one point, r = 1 mm, box ± 5 mm | 110–133 from, 0–57 to; 0.2–0.4 s | — |

**V1V2's two handles** (worth a look in ITK-SNAP):
- **Handle 1**, ITK-SNAP voxel (49, 134, 43): a loop 0.09 mm round, where two sheets of the mesh touch at a pinhole. A segmentation or meshing artefact.
- **Handle 2**, ITK-SNAP voxel (123, 119, 14), full grid (320, 176, 78), 2.1 mm from the box face: a loop 5.8 mm round, extent 1.3 × 1.9 × 0.8 mm, with 607 traces rooted within 1 mm. Possibly a white-matter bridge. On the flat map its two cut holes shrink to almost nothing (1% of the area at 2^-13), which is why the 3D view navigates and the map is only an overview.

**Views.**
- `prototypes/figures/flat_map_<site>_distortion.png`: the map coloured by log2 area factor and by angle distortion, with the handle cuts.
- `flat_map_<site>_hist_handles.png`: histograms, and the mesh round each handle loop in 3D.
- `inflate_<site>.png`: the area factor per inflation level.

**Open.**
- At strong inflation the carried vertical can be far from v. It was 56° at the V1V2 handle and 10–31° elsewhere in the checks. The sections stay orthogonal on the true surface, but their directions on screen are then only approximate.
- Whether the gray regions (no subsampled trace) hide cortex that should have traces.

### A viewer in the package

`baillarger.viewer` shows a site's column data in the browser, for looking only: nothing in it edits or re-solves the pia. Which correction steps are useful, and how to do them, waits until the data can be seen well. The prototype editor (`prototypes/scripts/pia_editor.py`, above) still runs and imports the same code from the package.

**How to use it.** From the repository root, for example:

    nix develop --command scripts/memguard 8 uv run python -m baillarger.viewer prototypes/scripts/column_volume_M1S1.npz --cache scratch/viewer --joint prototypes/scripts/joint_M1S1_s0.16_r3.npz --pia prototypes/scripts/pia_M1S1.npz --priority prototypes/scripts/edit_priority_M1S1_s0.16_r3.npz --port 8791

Then open http://127.0.0.1:8791/ (127.0.0.1 only). Only the column volume is required:
- without `--joint`, every trace is shown, with no levels;
- without `--priority`, there are no priority layers;
- `--echo` gives echo 2 of the full grid (default: the data directory).

The flat map and the inflation are computed on the first start (41 s at M1S1) and cached in `--cache`, keyed by a hash of the mesh. A rebuilt column volume gets new ones.

**Two points.** The viewer keeps two points:
- the *crosshair*: a world point, which sets the three slices of the volume pane;
- the *WM selection*: a vertex of the true white surface, with the shown trace there. It sets the sections, the streamlines and the info line. That trace is the one rooted nearest along the mesh within 0.5 mm, else the root nearest in space within 1 mm, else none.

In each pane a click moves that pane's own point, and a shift-click moves the other one:

| pane | click | shift-click |
|---|---|---|
| volume (slices) | the crosshair | the WM selection, to the white-surface vertex nearest the clicked point |
| surface (true or inflated), map | the WM selection | the crosshair, to that point of the true white surface |

The WM selection is drawn as a sphere in the volume pane and as a cyan dot on the surface.

**Panes and windows.** Each window shows the panes named in its address: `?panes=volume,surface,map,sections` (the default), or any subset. *New window* opens another one. Every window gets every move of either point, and every layer change, from the server as it happens (Server-Sent Events from `/events`), so the panes can be spread over several windows, browsers or screens. A window opened later starts at the current points.
- *volume* (NiiVue 0.69, world mm):
  - echo 2 of the box in three orthogonal views and 3D, with the white surface and the GM boundary (translucent) where they cut each slice;
  - the traces from and to the crosshair, coloured by class (*to* paler), with the pia on them as small spheres, and your points;
  - optionally every 4th shown trace.

- *surface*: the white surface, true or inflated, coloured by the layer. A WM selection made here gives sections along the screen's axes.
- *map*: the flat overview. A WM selection made here gives sections along the map's axes.
- *sections*: the depth sections along the traces, as in the editor, without the edits.

The volume pane names the tracer and the WM label that made the column volume (`tracer_name`, `wm_label`). Files made before these were saved say "not saved".

**Why both NiiVue and our own WebGL.** NiiVue does the volume views, meshes, tracks and spheres with little code. It reads TCK tracks and GIfTI meshes that the server writes on the fly. Clicking a mesh in its 3D tile did not move the crosshair in 0.69 (left or right click), so the surface pane stays our own WebGL view. That view already picks a vertex and gives the screen axes that the sections need.

**Tests** (exact answers):
- `tests/test_viewer_streamlines.py` (6) and the three moved test files above.
- `tests/test_viewer_data.py` (6), on a synthetic site (a plane of roots, straight traces, a cortex slab, a full grid whose values are their own index):
  - the site reads its tracer and WM label;
  - the flat map and the inflation are cached by mesh, not recomputed for the same mesh, and recomputed for another;
  - world mm is the box's crop of the echo affine, and `locate` finds a root within 1 mm and nothing beyond;
  - a WM selection snaps to the nearest white-surface vertex and takes its trace; a vertex with no trace near it along the mesh takes the nearest root within 1 mm, or none;
  - the volume is exactly the box of the full grid;
  - sections and streamlines without a joint result; a point's depth along its trace;
  - two pages share state through the server: a POST from one page arrives on another's event stream, tracks come back as TCK, and a path outside `static/` is refused.

**Checked in the browser (M1S1).**
- All four panes loaded with no console errors (volume pane 3.5 s).
- A click on the inflated surface moved the sections, the flat map's crosshair and the NiiVue slices to trace 19920, with 130 traces from it.
- A second window (`?panes=volume`) opened at the same voxel. A click on the first window's flat map moved the second window's crosshair to the same voxel (86, 104, 66).
- The traces at the crosshair reload with each move (172 from; then 155 from, 2 to).
- The two points:
  - a click on the surface set the WM selection and left the crosshair alone;
  - a shift-click there moved the crosshair to the same vertex (242090);
  - a click on a slice moved only the crosshair;
  - a shift-click on a slice moved only the WM selection (to vertex 91408).

  A second window (`?panes=surface,map`) opened at the current WM selection and followed a shift-click made in the first.

**Open.**
- Placing new points from the volume pane: NiiVue has drawing tools, not yet wired to a points file.
- The camera is not shared between windows. Each 3D view keeps its own.

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
