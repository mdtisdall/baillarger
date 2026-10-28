# Seismic flattening for cortical layers: plan

**Goal.** Estimate one laminar depth field T in ex vivo cortex (7T FLASH, 160 µm, three echoes), with layer boundaries as its level sets, from the image's own laminar orientation. Geometry is only a fallback where the data are weak. In fused cortex, fronts grow from the white matter and the contact is where they collide.

**Why.** No geometric frame can put the layers on level sets: one equivolume surface lies in different layers in different areas (Wagstyl 2018). Seismic flattening fits a depth field to layer orientation. Cortex is the easier case, because it has no faults.

**Where it stands.** Tested qualitatively on stretches of the V1V2 and M1S1 patches:
- **The layer orientation** is reproducible almost everywhere, and bands stay level along columns that follow it.
- **Fronts from the white matter, fitted as a field rather than traced raw,** give a clean contact in fused cortex.
- **The kissing M1S1 stretch is unequal at the contact** (finding 12): measured as column length, about 2.0 mm on one side against 3.3 mm on the other. A straight-line lower bound gives 3.2 mm for the far side. Equal arrival misses it, while the posterior from a probabilistic thickness field, with one field per front and walls from roots, lands within about a voxel (finding 11). Finding 8's "near-equal" reading came from T, which under-reads long columns, and from a geometric wall that flipped at the curl.
- **T is not distance.** The least-squares fit under-reads long columns (finding 12). Distance must be the column's arc length, carried in the upwind pass, not T.
- **One principle:** where there is contrast, the data decide; where there isn't, depth within a column follows equal volume. "The metric" is three roles with separate choices (depth within a column, thickness carried sideways, arrival at a contact); see *The fallback* for the status of every estimator in every role.
- **One pass from all WM now places contacts without walls, sides or sheet classes** (findings 23–26). `baillarger.fronts` carries per-root start times and speeds (crystal growth). At the marked faces the label boundary is within 0.08–0.26 mm of the points at three faces, including PCC piece 1 where the walls failed, and about 0.3 mm off at the V1V2 banks unless the metric is made isotropic there (0.16 mm). Two things in it are not principled yet: the cost ratio across the layer normal, and the separation that makes a label boundary a contact.
- **The white surface should be a mesh smoothed along its edges, not the voxels** (finding 28, prototype; also as the start of a level set on a 2× grid, where the label's own distance gives 6–11 % wrong signs and a jumping foot map). Contoured from the label, then smoothed on the mesh at about 0.32 mm, it keeps V1's 2-voxel blades and both their sides (volume smoothing at the library's σ = 1 already loses 16 % of that thin WM), and as the march's start it removes the voxel start's 0.05–0.15 mm arrival bias and most of the noise in the carried-area integrand just above the WM. The march itself is then the larger error only on tight convex surface. Using it needs per-seed start values and a carried root position in `march` (not made).
- **A level set in normalized time places contacts within a voxel where the budgets are right** (finding 29; prototype, 1× grid). The fronts advance from the smoothed white mesh along the layer normal, and the contact is where both sides have used the same fraction of their budgets.
  - Length form, posterior budgets: PCC pieces 0.05–0.06 mm, M1/S1 0.18 mm, V1V2 banks 0.40 mm.
  - The V1V2 miss is in the evidence: tops whose paths wander sideways inflate the field.
  - The volume form fails on a biased carried area.
  - The stria is not at constant τ in either form.
- **Two things are not yet working.** Predicting thickness at fused roots from visible tops over-predicts the thin front at the kissing stretch (finding 13; see *The contact as an inference problem*). And no estimator of equal volume is validated yet: the closed form from white-surface curvature fails, and carried area is mixed (finding 10). Banks are no longer labelled geometrically: ownership comes from roots (finding 11).

**Next:**
0. The one-pass competition (Next steps, 0): four decisions are open for the user (the cost ratio, the contact threshold, the rule's tail, root identity). On the complete patches the package is fast and reaches all cortex, but box faces and label errors dominate its contact set (finding 27). *The fronts as PDE problems* recasts it as geometry (one unweighted eikonal), transport (arc length, root position and beliefs along the columns) and a free boundary for the contact, and places each decision in one of those pieces. The rule as run shares the misfit by σ, not the posterior's σ² (*Thickness and where the contact lies*). The white mesh is evaluated as boundary data (finding 28). A level set in normalized time (finding 29) is the alternative representation: in the length form within a voxel at both PCC pieces and 0.18 mm at M1/S1; before a package it needs a guard against wandering tops in the evidence.
1. The contact model is paused. *The contact as an inference problem* states what is observed and latent, the assumptions with their evidence, and where the fused-against-open selection enters. Next come the diagnostics listed there, starting with bank position once the user has read it; the model changes only after that.
2. Test carried area for depth within a column with its near-WM part following the layers, starting from the smoothed white mesh (finding 28).

Companion reports:
- *Cortical laminae segmentation lessons:* the literature behind the plan and the constraint inventory.
- *Fused sulci and thickness priors:* how tools and other fields handle fused contacts and thickness.
- *Pial editing and edit priority:* a second way to get the top: a nearly right pial surface, fixed by hand in a local editor, with maps that point to where edits are needed. The base models it starts from are the user's decision (its §1).

Their notes are in `research_notes/`. The review of collisions and contacts is summarized in this plan (*Collisions and contacts*), with notes in `research_notes/Collisions and contacts/`. The literature on thickness against curvature is summarized in this plan (Method, *Carrying thickness across curvature*), with notes in `research_notes/Thickness across curvature/`.

## Rules for the work

- **Always 3D.** Tensors, fields, fits and columns are computed in 3D. A slice only chooses seeds or shows results.
- **Diagnostic views, not pass criteria.** The tests are exploratory: careful views, read by eye. Numbers are supporting context. The user's reading of a view counts as a finding.
- **Display echo 2** (trend removed) by default. Echo 3 has more stria contrast but is too grainy to read.
- **No faults within a stack of layers.** A jump in a layer's depth between neighbouring columns means an error. Areal borders are facies changes: a layer's thickness kinks, a layer pinches out, or the contrast pattern changes at continuous depth. A fused contact is not a fault: it is the boundary between two stacks.
- **Minimum band about 2–3 voxels.** Thinner layers ride along as level sets. The stria, about 250–300 µm (1.5–2 voxels), is at that limit and was still usable through its edges.
- **Stay away from volume edges.** WM touching a crop face looks like a WM boundary, and fields and columns are truncated there.
  - Test regions stay at least about 6 mm from any box face.
  - Boxes are loaded from the full volume around the region, not the predefined patch crops.
  - Box faces are never treated as WM boundary.
  - Columns that come within about 2 mm of a face are flagged and left out.
- **Never label banks by geometry.** Each voxel belongs to the front that reaches it; a column's identity is its root on the white matter. Bank labels from geometry break where faces curl.
- **Keep principle, roles and estimators apart.** A result is recorded against one estimator in one role, in the status table under *The fallback*, with its finding. A choice changes only when the result named in its "would change if" column happens.
- **Ground truth from the user.** Points marked in ITK-SNAP (WM walls, joining lines, gaps, contact points) are the reference. ITK-SNAP numbers voxels from 1; this plan gives 0-based crop coordinates.

## Method

### Data term: the layer orientation

- **Features:** the per-echo log magnitudes of echoes 2 and 3, with the smooth WM-to-pia trend removed (minus a 4-voxel blur). R2* proxies from two echoes are too noisy to use.
- **Orientation:** the summed structure tensor of the two echoes. Derivative scale 1 voxel, integration scale 2 voxels, all smoothing as masked normalized convolution over the aseg cortex (missing tissue removed), so WM and fluid edges don't leak in. The layer normal is the leading eigenvector. Integration can be isotropic: gradient outer products don't cancel across alternating band edges.
- **Confidence:** tensor coherence. Agreement between echo 2 and echo 3 is so high (median 2–3°) that noise isn't what limits the weights.
- **Radial fibres** (M1) need no special case: at 160 µm their structure is planar like any band.
- **The orientation is fitted globally.** It's a property of the tissue, so a least-squares fit is right for it (Lomask-style: T's differences across voxel edges fitted to a target field).

### Representation: fronts, roots and collisions

- **Fronts from all white matter.** T is fitted with T = 0 on every WM-boundary voxel, steered by the data normal (weight: coherence; prior: the direction away from the nearest WM, weight 0.1). It gives the collision structure and every voxel's root. It does **not** place a contact (finding 11): the data normal has no sign, so it is signed by the Euclidean WM-distance gradient, and T's ridge is pinned to the geometric midline between WM surfaces.
- **Arrival at a contact comes from one field per front.** The walls are the roots on each side of a sheet, gathered into pieces and sorted into two sides by which pieces face each other, then grown 3 mm along the white surface. Each wall gets its own data-steered field, signed away from that wall, so nothing flips at a midline. Equal arrival is T₁ = T₂.
- **Roots, not bank labels.** One upwind pass in increasing T gives every cortical voxel its root: the WM-boundary voxel its column started from. A column's identity is its root. Nothing is labelled A or B.
- **Every collision is the front meeting itself** (the user's point). The WM is connected, so "two fronts meeting" and "a front meeting itself" differ only in how far apart the two meeting roots are along the white surface, and that is a continuum. The two banks of a kissing sulcus are one front folded at the fundus: their sheet runs from the sulcal mouth to the fundus, where its roots converge and it ends in a rim, like the sheet at a tight bend, only longer. So classifying sheets by a distance threshold, and grouping roots into global sides by connectivity, are not principled. They failed exactly as this predicts (findings 19–21). What separates a contact from a fold is physical, in the laminar structure at the meeting:
  - *a contact* is two ribbons' outer surfaces in apposition: the meeting sits at the top of both stacks (layer I against layer I, finding 17), the layers on both sides run parallel to it, and each side reaches it after about its full thickness;
  - *a fold tighter than the ribbon's thickness* focuses the front, which then meets itself inside the stack: at less than the thickness, with the layers continuing round the bend and across the meeting.

  Each voxel's root (its column's identity) stays well-defined. What lacks a principle is grouping roots into global sides. A sheet always has two sides locally. Contact detection is paused until this is rethought, with a review of existing approaches (Next).
- **Collisions** are ridges of T, where neighbouring roots are far apart along the white surface. By the generic structure of arrival fields (Blum's medial axis; Kimia's shock theory):
  - *Sheets* (two fronts) are the kissing faces.
  - *Curves* (three fronts) are junctions.
  - *Points* (four fronts) are isolated.
  - *Sheet edges* (two roots merging) are a front meeting itself at a fold, not a contact.
- **Columns** are integral curves of ∇T from their root, valid up to their top: the pia where the cortex is open, the collision where it's fused. The contact has zero thickness and can lie between voxel centres. Each stack keeps its own T and normalization; T need not be continuous across a contact.
- **Topology** comes from T increasing along every column.
- **Later:** depth as a monotone warp z_c(T) along each column (dz/dT ≥ 0), so a layer that pinches out is a zero-thickness interval. "Pinched out" and "present but invisible" are kept as separate outputs.

### Thickness and where the contact lies

**Thickness is a probabilistic field on the white surface.** Each root has a thickness belief (mean μ, uncertainty σ), estimated by a Gaussian field over the WM-boundary surface:
- **Evidence of graded strength:** a visible top is strong; a faint dark line and a landmark-predicted top are moderate. Columns with no feature contribute nothing.
- **Propagation:** a second-order Matérn-type prior with a level term (uncertainty stays bounded far from evidence) and a length of about 2 mm.
- **Later:** coupling weighted by laminar-profile similarity, so thickness can kink at areal borders. Region-specific priors are future work.

**The contact rule (general form): weighted competition of fronts.** A voxel belongs to the front that wins min over i of (T_i(x) − μ_i) / σ_i, where i runs over the fronts that reach it and μ_i, σ_i are the thickness belief at front i's root.
- It is well-posed for any number of fronts; ties form the sheets, curves and points above.
- It reduces to known rules as special cases: equal arrival (equal μ and σ) and normalized arrival (σ proportional to μ).
- **It is not the pairwise posterior.** On a sheet with x₁ + x₂ = L, equal (x − μ)/σ on the two sides gives x₁ = μ₁ + σ₁/(σ₁ + σ₂) · (L − μ₁ − μ₂): the misfit is shared in proportion to σ (equal z-scores). The posterior (COHIBA's inverse-variance split, below) shares it in proportion to σ², and comes from min (x − μ)/σ², that is start −μ/σ² and speed σ². The one-pass runs so far (findings 23–26) used σ, so their field-weight results are not directly comparable with findings 11–19, which used the posterior. Which condition is right is a choice of principle (*The fronts as PDE problems*).
- It needs a consistent multi-label front solver: each front moves at its own speed with its own start offset, so it is not a shortest-path search.

**On sheets: the pairwise posterior.** Where exactly two fronts meet, the two columns meeting there have total root-to-root length L along the pair. The contact sits at x₁ = μ₁ + σ₁²/(σ₁² + σ₂²) · (L − μ₁ − μ₂) from root 1, with uncertainty σ₁σ₂/√(σ₁² + σ₂²) (COHIBA's combination). At junctions and sheet edges the pairwise rule is not well-posed and is not applied.

Evidence that can refine or check the contact:
1. **A residual dark line or thin gap,** where visible: a direct observation, entered once.
2. **Each stack's top predicted from its own landmarks** (a band's fraction of the stack's thickness, calibrated on its open part). This works where the landmark is clear (the V1 stria) and failed with a crude landmark in noisy M1S1 data.
3. **A doubled layer I,** if layer I has contrast (untested).

### The fallback: equal volume

**The principle (a choice, not a finding).** Where there is contrast, the data decide. Where there isn't, depth within a column follows equal volume (Bok). The principle is revisited only if an estimator that has itself been validated fails it against visible bands. A failed estimator is not evidence against the principle.

**Three roles.** "The metric" covers three separate questions. Each has its own evidence and its own current choice, and a result about one does not decide the others.
1. **Depth within a column:** where a band sits between the root and the top. Bok's principle is about this role. It needs only the shape of a column's cross-section A(s), normalized at the top: depth = V(s)/V(top), with V(s) = ∫ A/A₀ ds.
2. **Thickness carried sideways:** comparing and propagating a column's thickness to neighbours whose top isn't visible.
3. **Arrival at a contact:** the coordinate the fronts compete in. A front's expected top must be in the same units as its arrival, so this role follows role 2.

**Estimators of a column's cross-section A(s):**

| Estimator | How | Assumes | Needs |
|---|---|---|---|
| Distance | A constant | no geometry | nothing |
| Closed form | Steiner: A/A₀ = (1 + κ₁s)(1 + κ₂s) from the root's curvature | columns along white-surface normals; a parallel shell | curvature at the root |
| Carried area | grown outward with the front: dA/A = div(∇T/\|∇T\|) ds along the column | the column field is right all the way, including just above the WM | a reliable T near the WM; a derivative of a fitted field |
| Two-surface | A(s) interpolated between the area at the root and at the top (Waehnert 2014) | a smooth, monotone change of area | the area at the top: the pia where open; the neighbouring columns' tops where fused |

**Status, by role** (finding numbers in brackets):

| Role | Distance | Closed form | Carried area | Two-surface |
|---|---|---|---|---|
| 1. Depth | Baseline. Tight on open V1 (IQR 0.059), but the stria fraction moves with curvature (ρ +0.63; Bok's effect) and the banks differ (0.47, 0.38) [10] | **Failed** [10]: reverses the trend on open V1 (ρ −0.69) and flips the bank order. The parallel-shell assumption is wrong: the pia is flatter than the white surface | **Mixed** [6, 10]: brings the banks together (0.44, 0.47), but over-corrects on open V1 (ρ −0.67); too noisy when transported upwind [8]. Suspected cause, untested: T just above the WM is pinned to the voxel WM shape, not to the layers | Untested. Applies directly to open columns; fused columns need the top's area from neighbours |
| 2. Thickness | Rises with curvature (slope +1.2 mm per /mm in V1V2, +2.7 in M1S1) [10]: needs a curvature trend in the prior mean. Carried thickness biased by +0.2–0.25 mm where fused [8, 11]. Must be column arc length: T under-reads long columns [12] | Volume per unit of white-surface area is **not conserved** across curvature: the steepest against it of all (ρ 0.95) [10] | As closed form: also steeper than distance (ρ 0.71–0.81) [10] | Untested as a thickness |
| 3. Arrival | Per-front fields with walls from roots [11]: posterior within about a voxel of the points (\|offset\| 0.12 mm, worst 0.30); a single all-WM field is pinned to the geometric midline. Earlier [5, 7, 8] with geometric walls. One pass from all WM with field weights [25]: \|offset\| 0.26 (M1/S1), 0.08 and 0.16 (PCC pieces 1, 2), and at the V1V2 banks 0.40 with the data-steered metric (cost ratio 3) against 0.16 with plain distance (ratio 1) | Not tried | Failed by noise [8] | Not tried |

**Current choices:**

| Role | Choice | Evidence | Would change if |
|---|---|---|---|
| 1. Depth | Equal volume (the principle). Estimator: carried area once its near-WM part follows the layers; distance until then | [6, 10] | Carried area still over-corrects on open V1 with the near-WM fix → test two-surface on open columns. If a validated estimator fails against bands, revisit the principle |
| 2. Thickness | Distance, with a curvature trend in the thickness field's prior mean: uncertain coefficient, per region, probably on log thickness (see *Carrying thickness across curvature*) | [10]; literature | A thickness quantity is found that is flat against curvature on open columns, or the trend fails held-out calibration |
| 3. Arrival | Distance, in one pass from all WM with per-root weights (crystal growth); cost ratio across the layer normal open (3 so far) | [5, 7, 8, 11, 25] | Role 2 changes, or the per-front thicknesses at the points prove wrong, or the cost ratio is decided (finding 25) |

**Carrying thickness across curvature (role 2): what is known.** The notes are in `research_notes/Thickness across curvature/`.

*From the literature:*
- **Crowns are thicker than fundi, everywhere it has been measured.** Crown-to-fundus ratios are about 1.2–1.3:
  - human MRI: Fischl & Dale 2000, 2.7 against 2.2 mm; Demirci & Holland 2022, N = 501, convex 2.88 against concave 2.47 mm, rising steadily across shape-index bins;
  - macaque histology: Hilgetag & Barbas 2006, 2.23 against 1.75 mm.

  Correlations are ρ ≈ 0.4–0.6, like ours (+0.5). Mechanical models of folding make thicker ridges a generic effect (Holland et al. 2018). So the trend is real, not an MRI artefact. No study has measured it ex vivo at high resolution.
- **No published slope in mm per (1/mm), and none against white-surface curvature.** Studies use midthickness or pial curvature, so our slopes cannot be compared directly.
- **The slope varies from place to place and grows with local thickness** (Demirci, Coalson, Holland, Van Essen & Glasser 2026). Within a patch of about 6 mm, thickness is close to linear in curvature (the squared terms were near zero). Across cortex, the coefficients change, and they are larger where cortex is thick. This suggests a multiplicative form, t ≈ t₀(x)(1 + aH) or a trend in log t, rather than one additive slope.
- **Areal steps contaminate the slope.** At the central sulcus, area 4 on the crown is thick and area 3b in the fundus is thin (BigBrain: about 4.5 against 1.8 mm). Local slopes against sulcal depth change sign in V1 and somatomotor cortex (Vandekar et al. 2016). Our M1S1 slope (+2.7) therefore mixes an areal step with folding.
- **What is conserved is the layer fraction within a column (Bok), and only with the real pial geometry.**
  - Wagstyl 2018 (BigBrain) built equivolume depth from the real white and layer I/II surfaces. With it, layer IV depth is almost flat against curvature; in raw depth it is strongly curvature-dependent. One brain, mid-surface curvature smoothed 3 mm.
  - Waehnert 2014 needs the area at both surfaces.
  - Neither total thickness nor neurons per column is conserved. In macaque, gyral thickening is almost all in layers V–VI (1140 against 640 µm), while layers I–IV keep about the same absolute thickness (Hilgetag & Barbas 2006).
- **Normalizations.**
  - *Volume per white-surface area* is the least conserved. It rises at crowns because thickness and the fan-out both grow there, which is consistent with Bok and not evidence against it.
  - *Volume per mid-surface area* is in effect thickness: ≈ s + Ks³/12 for a parallel shell [inference].
  - *Volume per pial area* is ≈ s(1 − Hs), the only single-surface normalization that partly cancels gyral thickening [inference; untested].
  - Thickness × area gives volume with a local error of up to about 20% (Winkler 2018).
- **No segmentation tool makes expected thickness depend on curvature.** KellyKapowski's prior image is spatially varying but has no curvature term. The nearest precedents are local regressions that remove curvature for group statistics (Demirci 2026; Vandekar 2016), not predictors of unseen thickness. A trend in a Gaussian field's prior mean (universal kriging) with curvature as the covariate has no neuroimaging precedent we found.

*From our open columns (distance; finding 10 and `split_depth.py` on the open-V1 strip, 490 columns, root H between about −0.25 and +0.3 /mm):*
- Total thickness: median 2.56 mm, nearly flat (slope +0.27 mm per /mm, ρ +0.09).
- WM to stria: 1.08 mm, rising (+0.45, ρ +0.57).
- Stria to pia: 1.48 mm, falling (−0.18, ρ −0.34).

So on this strip curvature moves depth from the superficial part to the deep part, as Bok predicts, while total thickness barely changes. Over the whole V1V2 box (a wider curvature range, several areas) thickness rises (+1.2 mm per /mm). Slope divided by median thickness is 0.56 there, against 0.94 in M1S1; the M1S1 figure includes the areal step.

*What this means for role 2* [inference]:
- **The trend belongs in the prior mean as a function of curvature, with a coefficient that is uncertain and varies by area.** Bayesian universal kriging does this: a prior on the coefficient makes its variance grow where a fused column's curvature lies outside the range of the evidence. A per-region coefficient with a shared prior fits the future region-specific priors.
- **Fit the trend on log thickness (multiplicative)** rather than an additive slope, following Demirci 2026. This is to be checked on our data.
- **Fit M1 and S1 sides separately,** or with a region term, before reading the folding slope there.
- **Two-part carrying is an option where a landmark is visible:** carry the deep and superficial parts separately (each with its own curvature trend), or carry total thickness plus the Bok fraction. This is only for V1-like cortex.
- **Check that the curvature of the fused banks' roots lies inside the range seen on open columns.** Where it doesn't, the trend extrapolates, and its uncertainty must show that.

**White-surface curvature** (for the closed form and the thickness trend) comes from a weighted quadric fitted to the exposed faces of the WM label around each WM-boundary voxel. The weights are Gaussian with a scale of 0.48 mm, and only faces on the same side enter (outward normals agreeing), so the two sides of a thin WM blade never mix. The sign is positive where the white surface is convex (crowns, blade tips). The Hessian of the smoothed signed WM distance failed: it mixes the two sides of V1's thin blades. The face fit's ripples (about ±0.1–0.2 /mm at 0.32 mm, weak at 0.48 mm) are mostly the label's own terraces, which a sub-voxel mesh keeps too (finding 28). A mesh of the label smoothed along its edges (twicing at 0.32 mm) agrees with the face fit (rank correlation 0.85–0.98 on the three patches), has about half its jitter, keeps both sides of blades by construction, and reads tight convex curvature right where the face fit over-reads it (+0.12 /mm on a 1.28 mm ball). It is the candidate replacement, with the thickness field moved onto its vertices (finding 28, package changes).

**The closed form in detail.** A(s)/A₀ = 1 + 2Hs + Ks², so V(s) = s + H s² + K s³/3, with H = (κ₁ + κ₂)/2 the mean curvature and K = κ₁κ₂. It breaks down at focal points (1 + κs → 0) of concave white surface, which coincide with sheet edges of the collision structure. Those columns are flagged: 22% of inner roots in the V1V2 box, 6% in M1S1.

### Bands and regions (later)

- **Band boundaries** are knots of z_c(T) at detected features. Between knots, T follows the background metric, which predicts where the next feature should appear.
- **Region templates** switch through a closed Potts-type partition on the template term only, never on the geometry.
- **The fluid** is the layer above the top band.

## The contact as an inference problem

A statement of what the contact model estimates, from what, and under which assumptions. It is written before any further change to the thickness field or the rules (finding 13), so that each change can be traced to an assumption and its evidence.

### What is observed

- **The image:** per-echo log magnitudes (echoes 2 and 3), giving the layer normal and its coherence everywhere in cortex, and bands where present (the V1 stria).
- **The segmentation:** WM, cortex, fluid (aseg) and missing tissue. The box faces are not tissue boundaries.
- **Visible tops:** cortex next to fluid. Each one ends a column whose top is seen; its thickness is the column's arc length there.
- **Weak evidence inside fused cortex:** a residual dark line or gap in places; landmarks such as the stria, where they exist.
- **Path length across a contact:** for two columns meeting at a sheet, the root-to-root length L along the pair. It is observed (geometry plus the column field) even though the split of L between the two banks is not.
- **The user's marked points:** the 23 contact points, and others for earlier stretches. They are ground truth for scoring. They are also used to close the kissing gap (fluid within 1.5 mm of them becomes cortex), so they are not independent of the scored geometry (assumption A10).

### What is latent

- **The column field and the roots:** which WM-boundary voxel each cortical voxel's column starts from. Estimated from the data (the fitted T and the upwind pass).
- **Arc length S_X(x):** the distance along a front's columns from the root to x. Computed, given the columns.
- **Thickness t(r) at each root r:** the arc length from r to the column's top. It is observed at roots with a visible top, and latent at fused roots.
- **The contact:** where two fronts' columns end. On a sheet it satisfies t₁(r₁) + t₂(r₂) = L for each meeting pair. At junctions and sheet edges, more than two roots or one root's own fold are involved.
- **Area identity and laminar structure:** not modelled yet (region templates, later).

### Assumptions and their evidence

| | Assumption | Status | Evidence |
|---|---|---|---|
| A1 | The data's layer normal is reliable in cortex | Supported | [1, 2]; less so right at WM edges (open question) |
| A2 | Columns are integral curves of a fitted field, and a voxel's root identifies its column | Supported | [3, 4, 11]; the user checked the columns at the kissing stretch |
| A3 | A front is the WM whose columns reach one side of a sheet: walls from roots, sorted into sides | Supported at the kissing stretch | [11, 12]; one stretch only |
| A4 | Distance is arc length along the columns, not T | Supported | [12, 13]: arc length agrees with traced length to about 0.1 mm; T under-reads long columns |
| A5 | The contact has zero thickness: the two columns' lengths add up to L | Supported at the kissing stretch | No fluid gap; two layer I's meet at the user's points, their superficial bands about 0.25 mm closer together than two open pia [17] |
| A6 | Thickness is smooth along the white surface (a second-order Gaussian field with a length of about 2 mm) | **Sound as built, but it reverts over the gap before the contact: the cause of front 1's over-prediction** | Held-out visible tops are predicted, if over-cautiously (z SD about 0.75) [11]. The graph is connected, the field honours its evidence (residual IQR 0.027 in log), and its correlation length is right [15, 16]. Across the 1–2.5 mm of evidence-free surface before the contact roots, it reverts toward a broad mean (2.6–2.7 mm) that is not the locally thin cortex (2.1) [16] |
| A7 | **Thickness at fused roots is like thickness at open roots in the same place** (given position and curvature): whether a top is visible does not depend on thickness, beyond what the covariates explain | Mostly holds at the kissing stretch | The tops nearest the fused zone are within about 0.2–0.25 mm of the contact's arc length, on both fronts [14]. Finding 13's +0.26 to +0.92 mm is the field's error, not the evidence's [14]. Finding 8's +0.2–0.25 mm was measured in T with flipped walls. Only one stretch has been seen, and only within about 4 mm of the fused zone |
| A8 | White-surface curvature at the root is the covariate that carries folding effects | Partly | Thickness rises with curvature on open columns [10], and within each front at the kissing stretch (ρ +0.30, +0.35) [14]. The coefficient is small where evidence is near [11, 13]. Sulcal depth carries a clear trend on front 2 (ρ −0.65) and none on front 1 [14] (Vandekar 2016 also finds thickness falls with depth). The contact roots of front 2 lie beyond the curvature range of its evidence [14] |
| A9 | The two roots' beliefs are independent Gaussians | **Fails, in a way that matters** | Both fronts' beliefs are biased upward at fused roots by the same drift [16, 18]. When both banks are biased alike (V1V2), the split cancels it [18]; when one is (M1/S1), the contact moves [13] |
| A10 | Closing the gap near the user's points does not bias the score | Supported | No closure and a point-free closure give the same arc lengths and nearly the same score [16]. The gap at the points is essentially closed in the image already |
| A11 | The user's points mark the contact | By rule | The user's reading counts as ground truth |

### Where the fused-against-open selection enters

- **Visibility depends on position.** A top is visible where the cortex is open: on crowns and near the mouths of sulci. It is hidden where banks fuse: deep in sulci and on kissing faces. So the roots with evidence and the roots that need a prediction are different samples of the white surface.
- **If thickness also depends on position, interpolation is biased.** If thickness changes along a bank (with sulcal depth, distance from the sulcal mouth, curvature or area), a field fitted to open roots and smoothed into the fused zone is biased there. The bias grows with the thickness gradient along the bank and with the distance into the fused zone. Without the right covariate, this is a covariate shift: the fused roots lie outside the range the open roots cover.
- **If visibility depends on thickness itself, covariates are not enough.** A sulcus may close because its banks are thick, or stay open over thin banks. In statistical terms the data are then missing not at random: even the right position covariates would leave a bias, because the visible tops over-represent the banks that did not fill the gap.
- **The fused zone is not unobserved: it gives sums.** Every meeting pair on a sheet gives t₁ + t₂ = L. At present L is used only afterwards, to split the misfit between the two banks (the pairwise posterior), and never as evidence for the thickness field. In a joint model, where the field is fitted to the visible tops and to the sums L together, the sums inform the fused roots directly. If A5 holds, the sums are exactly the fused zone's evidence, and the selection problem shrinks to how the sum is split. That split is where the visible tops, the covariates and the uncertainty still matter.
- **The pairwise rule inherits the bias.** If every fused root on one front is over-predicted by the same amount (A7), the two beliefs at a contact are not independent of the error (A9), and the posterior moves the contact toward the biased front by up to that amount, whatever its SD says.

### What would test each open assumption (no model change)

- **A6:** check the white-surface graph's connectivity between the kissing stretch's roots and the nearby visible tops (connected pieces, path lengths against straight distances), and the field's fit at the nodes nearest the contact. Done [15, 16].
- **A7 and A8, the bank-position diagnostic** (done [14]): on each front, plot thickness at visible tops against position along the bank. Use sulcal depth, and geodesic distance on the white surface to the fused zone. Overlay the arc lengths at the user's contact points. A trend toward the fused zone would point to a missing position covariate (A7 with A8). Contact points falling off an otherwise flat trend would point to selection on thickness itself.
- **A9:** the correlation between the two fronts' errors (μ − S) across the 23 points.
- **A10:** re-score with a gap closure that does not use the points. Done [16].

The bank-position diagnostic runs only after the user has read this section.

## Collisions and contacts: the review and what it points to

Contact detection was paused after PCC pieces 1 and 3 (findings 19–21), when the user pointed out that with connected WM every collision is the front meeting itself (Representation). This section summarizes a review of existing tools, of shock and medial-axis theory, of offset surfaces and folded sheets in other fields, and of front solvers. Notes are in `research_notes/Collisions and contacts/`: `neuro_tools_contacts.md`, `shock_theory.md`, `offsets_and_folds.md` and `front_solvers.md`. Sources are tagged there as verified, unverified or inference; here, "inference" marks what no source states.

### What the review found

- **No neuroimaging tool groups sources into global sides, and none separates a fold from apposition by principle.**
  - *Mesh tools* (FreeSurfer's pial deformation, CLASP, TGDM) treat anything outside a vertex's mesh neighbourhood as the other side. They freeze or repel a vertex that would cross the surface.
  - *Volume tools* (ACE in CRUISE, DELFMAP, BrainVISA's sulcal skeletons) never name the sides: the shock or skeleton itself separates them.
  - *Folds are handled by tuned constants:* a mesh neighbour ring, a minimum of 1 mm from the WM, a jump of 4 voxels between sources (DELFMAP), or a gradient threshold of 0.8–0.85 (ACE).
  - *Label errors and hidden gaps* are handled only upstream, in the segmentation. Any trace of CSF becomes a barrier or slows the front. No tool checks whether a gap holds non-tissue.
  - *No tool reports contact error at fused sulci, and none uses layer I contrast.* BrainVISA's skeleton is the only one whose position the image intensity sets.
  - *A claim to test:* CLASP's authors say Laplace flow, rather than distance, avoids collisions caused by tight folds alone.
- **Shock theory: the meeting set is a continuum along each sheet, but each piece is born in one of two ways.**
  - *The local types* (Giblin and Kimia; Pizer, Damon) are sheets (two contact points), Y-junction curves (three), rims (the two contact points coincide), and points where several of these meet.
  - *A rim lies at a focal point of the initial surface:* the distance equals a principal radius of curvature, at a crest of that curvature. A sheet is bounded by rims or by the domain's edge.
  - *The object angle α* (cos 2α = n₁·n₂, from the two arriving fronts' directions) runs continuously along a sheet: from 0 at a rim to π/2 where the fronts meet head-on. No threshold on single points can be principled.
  - *As the front advances* [inference], each connected piece of the meeting set is born either at a rim (focusing: local) or at an interior minimum of arrival on the sheet (fronts antiparallel: apposition, the 3D "neck"). Classing each point still needs arrival compared with thickness.
  - *The two sides are defined locally* by the two arriving fronts' directions at each sheet point (Damon's two spokes; the sheet normal is along their difference). This becomes ill-posed at rims, as it must.
  - *The theory is for distance fields;* nothing was found for anisotropic (data-steered) arrival.
- **Offset surfaces and folded sheets elsewhere.**
  - *CAD* defines local self-intersection (offset distance beyond a concave radius of curvature) and global self-intersection (distant points at a local distance minimum), but trims both the same way.
  - *Cloth simulation* treats a sheet folded onto itself like two sheets in contact (proximity plus thickness). Untangling assigns sides locally, by flooding both sides of each intersection curve.
  - *Geological fold frames* give each limb of a fold its own sign, by its rotation angle.
- **Front solvers: the weighted competition is a solvable weighted-Voronoi problem.**
  - *The formulation:* each root is a source with start time −μ/σ and speed σ, and a voxel goes to the earliest arrival in that rescaled time.
  - *Two versions exist:* fronts that ignore each other (pure weighted Voronoi, where regions can be disconnected), or fronts that cannot enter territory already claimed ("crystal growth"). A single pass in arrival order computes the second [inference, with literature support]. The second is also the physical one: a column cannot pass through cortex owned by another.
  - *A suitable solver:* Mirebeau's adaptive-stencil fast marching (FM-VR1, in the HFM / `agd` library). It is single-pass with a proved convergence, keeps grid bias low up to metric anisotropy of about 10², and has a Python interface.
    - It already offers per-seed start times, source labels, the Euclidean length of each path, and a stop where regions meet.
    - It lacks a speed that depends on the source. That needs either a custom pass that updates each label from its own neighbours only (numba, reusing the stencils), or the sources' σ spread along their columns as a position-dependent cost.
  - *Why the earlier solvers failed:* plain fast marching on axis-aligned neighbours is not causal for anisotropic metrics and distorts both fronts and region boundaries (Nishida and Sugihara). That, with Dijkstra's lattice limit, fits the 29° bias seen earlier (finding 9).
  - *Arc length* can be carried in the same pass. Competing on arc length while ordering by metric time breaks the single-pass guarantee; the consistent fix is a metric with unit cost along the layer normal, so that metric time is arc length.

### What it points to [inference; for the user's decision]

1. **Contacts come from one pass, with no sides and no classes.** One anisotropic fast-marching pass from all WM boundary, metric from the data's layer normal and coherence, carries for every voxel its arrival (equal to arc length along the normal), its root, and the root's weight. The weighted competition (start −μ/σ, speed σ, crystal growth) settles ownership wherever fronts meet. The contact is the boundary between roots, located at sub-voxel accuracy from the two best arrivals. This replaces the least-squares T, the walls, the per-front fields and the sheet classes. At a fold, nearby roots carry nearly the same weights and the split falls about midway, so folds need not be told from contacts to place the boundary.
2. **Classifying is for reporting, not for placing.** At each boundary point, the object angle, the arrival on each side against that side's expected thickness, and how the piece was born (rim or interior) label it a contact, a fold or an overlap (both sides already past their thickness: a fused sulcus or a label error). These flags are outputs and checks, not inputs.
3. **The thickness beliefs still need the sums.** μ and σ come from the thickness field: visible tops plus, in a joint fit, the sums t₁ + t₂ = L at contacts. The solver needs μ and σ before it runs, so the two alternate: equal weights first, then contacts, then sums and field, then weighted again, until stable. Whether this converges is open.
4. **Label errors are handled on their own.** Hidden gaps (thin strips inside labelled cortex that decay slowly, like Fomblin) can be flagged or opened before the pass. The T = 0 boundary could come from the image's own WM contrast.
5. **Tests, in order:**
   - synthetic shapes with exact answers: a U-slab whose fundus radius is below and above the ribbon's thickness; apposed slabs of unequal thickness (phantom P4); an omega fold;
   - then the marked faces (M1/S1, V1V2 banks, PCC pieces 1 and 2), against findings 11–19;
   - first, whether `agd` installs and runs at 141³.

## The fronts as PDE problems

The solver rests on PDE theory, but the parts around it were built ad hoc, and the problems met in findings 23–27 sit in those parts. This section states the problem as three PDE pieces, with the theory each brings. Sources are tagged as in the review; notes are in `research_notes/Collisions and contacts/pde_framing.md`.

### What already uses the theory

- **Arrival** is the viscosity solution of an anisotropic eikonal equation, computed by Mirebeau's adaptive-stencil fast marching: monotone, consistent and causal, so it converges (Barles–Souganidis). Its first-order accuracy is finding 22.
- **The collision set** is that solution's singular set (shocks, the medial axis): sheets, Y-curves and rims (*Collisions and contacts*).
- **The thickness field** is an SPDE (Lindgren–Rue); the earlier least-squares T was an elliptic fit.

### The three pieces [inference; for the user's decision]

1. **Geometry: one eikonal equation, with no weights.** One front from all WM, under the data's metric. It gives arrival and the columns (its characteristics). Nothing in it depends on the thickness beliefs, so a column cannot bend because its neighbour is believed thicker (the within-bank leakage of crystal growth with patch labels, finding 24's omega).
   - *Boundary data:* the white surface is an interface embedded in the grid. The standard start gives the voxels in a narrow band their exact distance to the surface; a second-order start comes from a bicubic fit of a level-set function with Newton steps to the interface (Chopp 2001) [unverified: abstract only]. With a first-order scheme, an O(h²) start buys little unless the solver error is smaller; the mesh question is a question about the accuracy of this boundary data, measured against the solver's error.
   - *Regularity:* a non-smooth boundary (the voxel staircase) makes shocks at concave corners, so some sheet edges near the WM may be staircase artefacts [inference]. The metric must be continuous and positive definite for stable characteristics; where coherence is low the direction is noisy and the roots traced along it become sensitive (the V1V2 banks, finding 25). The theory points to regularising the metric (anisotropy falling with coherence) rather than choosing one cost ratio.
2. **Transport: quantities carried along the characteristics.** Arc length, root position and the root's beliefs (μ, σ) each satisfy a transport equation along ∇T. Fast marching has a standard method for this, extension velocities (Adalsteinsson and Sethian 1999) [verified: citation and abstract]; carrying the closest point on the source is Tsai's closest-point formulation (2002) [unverified: preprint abstract]. `march`'s arc length is an instance. Carrying the root position gives each voxel a root below the voxel scale with no labels, and the beliefs are then read at that root.
3. **The contact: a free boundary.** In fused cortex, find the surface Γ where the two sides' carried quantities satisfy the contact condition, for example (S_A − μ_A)/σ_A = (S_B − μ_B)/σ_B (equal z-scores) or the posterior's σ² form. Γ is the zero set of the difference of the two sides' weighted arrivals.
   - *What it needs:* both sides' arrival and carried quantities near Γ, on both sides of it. A single-valued solver keeps one arrival per voxel; ours keeps a second best, usually a neighbouring patch (finding 27). Multiple arrivals are a solved problem for static Hamilton–Jacobi equations (Fomel and Sethian 2002, phase-space fast marching) [verified: citation]; a cheaper route here is to run the piece-2 transport on each side of the unweighted equal-arrival boundary and extend each side's weighted arrival a short way past it.
   - *What it gives:* a sensitivity, how far Γ moves per mm of error in μ on each side and how that depends on the angle between the two fronts (Damon's two spokes), so each contact can carry an uncertainty from the beliefs'.
   - *Prior work:* the multiplicatively weighted crystal-growth Voronoi diagram (Schaudt and Drysdale; Kobayashi and Sugihara, with fast marching) is what `march` with per-label speeds computes [verified: citation and abstract]. Competing-front segmentation (dual-front models; Chen, Mirebeau and Cohen) labels by first arrival [verified: abstracts]. Hamilton–Jacobi equations with coefficients discontinuous across an interface need their own notion of solution (Barles and Chasseigne's 2023 monograph) [verified: listing]; patch labels with different speeds make exactly such a coefficient. No source found states a meeting condition set by expected thickness: that part is ours.

### What it changes

- The geometry, the identity and the competition separate again, as before the one pass (roots from geometry, beliefs looked up at the root), but each piece now has theory and none needs side labels or patches.
- The open decisions of Next steps 0 become choices in specific pieces: the cost ratio is the metric's regularity (piece 1); root identity is the transport of root position (piece 2); the contact threshold and the rule's tail are the condition on Γ and where it applies (piece 3).
- The white mesh is piece 1's boundary data and piece 2's starting foot points. Finding 28 measured both (also on a 2× grid: a start from the label's own distance gives 6–11 % wrong signs and a jumping foot map there; the smoothed mesh none): a narrow-band start from a mesh smoothed along its edges (twicing at 0.32 mm) removes the voxel start's 0.05–0.15 mm bias and leaves the solver's own error (within 0.002 mm on gentle surface, first order); the band must reach as deep as the stencil (2.45 voxels at ratio 3), and the root position carried with arc length gives identity below the voxel scale.

## Findings so far

All computed in 3D and read qualitatively from diagnostic views.

1. **The orientation is reproducible and follows the cortex.** On three orthogonal slices through the V1V2 and M1S1 review points, about 91% of interior cortex has echo 2 and echo 3 normals within 10° (median 2–3°). Strokes along the estimated layer direction run parallel to the cortex, M1 included. In fused cortex the data orientation stays coherent where a boundary-based normal fails.
2. **Bands stay level along data-steered columns.** On an open V1 stretch (V1V2 slice k = 54, crop voxels (71, 89) to (84, 131)), the stria is a level dark band at 0.8–1.0 mm from the WM in 2.1–2.4 mm cortex.
3. **A fitted field, not raw tracing.**
   - **Raw tracing fails for structural reasons.** Columns traced step by step along the raw data normal stall, run on or skim WM edges. They have no scalar field behind them, no T = 0 condition at the WM, and no stopping place where the banks meet.
   - **A fitted T removes all three.** The least-squares fit in a 3D box has no stalls, and it bridges a fused gap raw tracing couldn't cross. Where coherence is above 0.5 it stays within about 7° of the data normal.
4. **Columns traced across a shared field's crest slide along it.** With T = 0 on every WM voxel, T has a crest at the contact. Columns traced up one side and down the other run along the crest: about half of one bank's paths returned to the other bank's WM. The shared field is still the right way to find collisions. What fails is tracing a single column across them; ownership comes from roots instead.
5. **Separate fronts give a clean contact on similar banks.**
   - **Where:** V1V2 plane j = 92 in ITK-SNAP numbering. Bank A's WM runs from crop voxel (71, 91, 18) to (67, 91, 47), bank B's from (38, 91, 50) to (42, 91, 5).
   - **The contact:** a level dark line about 2.0 mm from A's WM and 2.2 mm from B's. Equal arrival lands on it (median |offset| 0.04 mm).
   - **Caveat:** bank B's lower end (crop k = 5) is under 1 mm from the crop face. To be rechecked away from the edge.
6. **Evidence on the background metric, from divergence-based equivolume.** The stria's fraction of a bank's own thickness:

   | Setting | Distance | Equivolume (divergence-based) |
   |---|---|---|
   | Open V1 (calibration) | 0.40 | 0.48 |
   | Fused bank A (curved WM) | 0.53 | 0.48 |
   | Fused bank B (straight WM) | 0.39 | 0.44 |

   Equivolume holds the fraction nearly constant across curvature, Bok's signature, but it is much noisier within a bank (IQR up to 0.09 against 0.03), because of its divergence estimate. A bank's landmark fraction should be calibrated on its own open part.
7. **Mildly unequal banks, against a marked joining line** (M1S1 plane i = 85 in ITK-SNAP numbering, patch crop).
   - **The contact is oblique to the plane:** columns run 1.5–1.9 mm out of the plane.
   - **Thicknesses:** the joining points lie 2.2–3.1 mm from one bank's WM and 2.3–2.6 mm from the other's.
   - **Arrival rules:** equal arrival falls 0.16–0.36 mm short, toward the thicker bank. Normalized arrival falls 0.08–0.16 mm short.
   - **Landmark-predicted tops failed** with a crude darkest-point landmark (−1.24 to +0.96 mm).
8. **The kissing M1S1 stretch in a clean crop, against 23 marked contact points.**
   - **An earlier reading of this stretch** reported about 3.5 against 2.3 mm, with the midline 0.9 mm off. It ran in the patch crop, with one bank's end about 1.3 mm from the crop face, and took its truth from 4–6 columns, so its numbers are not trusted. Its direction (unequal) is confirmed in a clean box (finding 12).
   - **Setup.**
     - *Crop:* loaded from the full volume (full-grid corner 419, 229, 439; 91 × 136 × 141 voxels), with every point at least 7.2 mm from a face.
     - *Ground truth:* 23 user-marked contact points spanning about 5 mm of the kissing surface.
     - *Gap and walls:* the kissing gap closed near the points, and bank walls labelled geometrically.
     - *Evidence and field:* thickness evidence from every visible top in the crop; a Gaussian thickness field per bank.
   - **At the marked points, T read 2.22 mm (2.05–2.51) and 2.02 mm (1.70–2.16)** up to the contact. These are T values, not lengths, and bank A's geometric wall flips onto the other side at the curl. Measured as column length, the stretch is unequal (finding 12).
   - **Contact rules in distance** (median offset; IQR; median |offset|):

     | Rule | Median | IQR | Median \\|offset\\| |
     |---|---|---|---|
     | Equal arrival | −0.05 mm | 0.35 | 0.14 |
     | Normalized arrival, one thickness per bank | +0.13 mm | 0.39 | 0.20 |
     | Posterior from the thickness field | +0.14 mm | 0.17 | 0.15 (2 points without a crossing) |

     All are within about one voxel. Read by eye, the posterior follows the contact best (the user's reading), matching its smallest spread.
   - **Carried thickness is biased where the cortex is fused.** At the contact the field predicted about 0.2–0.25 mm more than measured on both banks, with too little uncertainty. The biases nearly cancel at the contact. Part of this may come from the wall-labelling failure below.
   - **Calibration on open cortex, distance:** one bank well calibrated (z-score SD 1.08, 92% inside ±1.96), the other over-cautious (SD 0.49).
   - **Geometric bank labels fail where the face curls.**
     - *The mechanism:* walls were labelled by which side of the local contact surface they face, with orientation chained point to point. Where the face curls tightly around the end of the open sulcus (points 4–6, ITK-SNAP 46, 73, 73; 46, 78, 77; 46, 84, 82), the chain flips, and the banks reverse partway along a slice (the user's observation).
     - *The effect at those points:* columns root on distant WM, the posterior has no crossing or is 2 mm off, and normalized arrival is pushed about 1 mm toward one bank. Equal arrival, which needs no lookup at the root, still passes through them.
     - *Not curvature:* the columns there are no more divergent than elsewhere.
   - **The equivolume transport (divergence-based) is unreliable.**
     - *The symptoms:* tops on one bank ranged from 3.2 to 12 mm, and its calibration SD was 5.3.
     - *The cause:* u was a second derivative of the data-steered T (the divergence of its gradient direction), integrated along columns and exponentiated, and smeared by the upwind averaging. All three earlier equivolume runs used this route.
     - *The fix:* the closed form from white-surface curvature (Method) computes the same quantity without differentiating T.
   - **The rules drawn across the whole crop are meaningless away from the contact.** Each field covers every gyrus, so its zero lines cut across unrelated cortex. Only collisions should be drawn and scored.
9. **(See also finding 10.) What failed, and why.**
   - **Geometric bank labels:** they flip where faces curl (finding 8). Roots replace them.
   - **A precomputed split between banks:** a global fit given the split in advance can't follow fronts that meet off the midline.
   - **A shortest-path search on the voxel grid as the front solver:** its arrival times are distorted along lattice directions (about 29° from the data normal, against 7° for the fit). With a history-dependent speed, the path search exploits it until T goes flat.
   - **Divergence-based equivolume:** too noisy (finding 8), and it over-corrects on open V1 (finding 10).
   - **Closed-form equivolume from white-surface curvature alone:** it over-corrects (finding 10).
   - **Curvature from the smoothed signed WM distance:** it mixes the two sides of thin WM blades.
   - **Crops too close to the volume edge:** they distorted the first reading of the kissing stretch (finding 8).
   - **T read as a length:** the least-squares T under-reads long columns, which made an unequal contact look equal (finding 12).
   - Each failure points to a constraint the representation must hold: ownership from roots, quantities carried along each column rather than optimized, a consistent front solver, curvature from a smooth surface rather than a noisy field, and margins from the volume edge.

10. **Closed-form equivolume from white-surface curvature over-corrects.**
    - **Setup.** Two boxes loaded from the full volume: V1V2 (full-grid corner 201, 70, 65; 143 × 144 × 140 voxels), covering the open V1 stretch, the fused k = 27 stretch and plane j = 91 with ≥ 7.2 mm margin; and M1S1 (the test crop grown by 15 voxels). One data-steered T per box with T = 0 on all WM boundary (box faces excluded). Columns go up ∇T from their root to fluid (open) or to the ridge (fused, extended 0.6 mm to find the dark contact). No column comes within 2 mm of a face. Roots: WM-boundary voxels within 1 mm of the user's in-plane paths, so the roots form 3D strips, not single lines.
    - **The curvature map is smooth and plausible.** WM blade tips and crowns are convex; WM at sulcal bottoms is concave. Bank A (the curling blade) is convex along its length (median H +0.24 /mm); bank B is nearly flat (−0.06 /mm). In the V1V2 box, 22% of inner roots have κ₂ < −0.4 /mm, so their columns reach a focal point within 2.5 mm (M1S1: 6%).
    - **Stria fractions** (open V1: of the pia; banks: of the dark contact; focal columns left out; ρ is Spearman's correlation with root H):

      | Setting | Distance | Equivolume, divergence | Equivolume, closed form |
      |---|---|---|---|
      | Open V1 (327 columns) | 0.435, IQR 0.059, ρ +0.63 | 0.449, IQR 0.073, ρ −0.67 | 0.424, IQR 0.100, ρ −0.69 |
      | Bank A, curved WM (286) | 0.469, IQR 0.101, ρ +0.26 | 0.439, IQR 0.082, ρ +0.08 | 0.361, IQR 0.096, ρ −0.19 |
      | Bank B, straight WM (381) | 0.382, IQR 0.080, ρ +0.38 | 0.465, IQR 0.069, ρ −0.02 | 0.436, IQR 0.086, ρ −0.15 |

      - In distance, the stria sits deeper in the column on convex roots (ρ +0.63 on open V1), Bok's effect.
      - Both equivolume routes reverse it on open V1. The closed form is the widest of the three there.
      - Between banks, distance gives 0.47 against 0.38 and the closed form 0.36 against 0.44: the order flips. The divergence route brings the banks closest (0.44, 0.47), with the smallest spread on the banks.
      - On open V1 the closed form also leaves out 163 of 490 open columns as focal.
    - **Scaling the root curvature by α** (0 is distance, 1 the closed form) shows the over-correction: open V1 is flat at α ≈ 0.25 (median 0.426, IQR 0.053, ρ −0.02), while the two banks agree at α ≈ 0.5–0.75. The pattern is the same at curvature scales 0.32, 0.48 and 0.80 mm, so it is not voxel staircase. No single factor fits both, so a damped white-surface curvature is not a fix.
    - **Open-column thickness against root H** (every fourth inner root; about 2,100 open columns per box):

      | Box | Distance | Equivolume, divergence | Equivolume, closed form |
      |---|---|---|---|
      | V1V2 | 2.16 mm, slope +1.2 mm per /mm, ρ +0.53 | 1.77, +3.2, ρ +0.71 | 1.89, +6.0, ρ +0.95 |
      | M1S1 | 2.88 mm, slope +2.7, ρ +0.51 | 2.39, +6.3, ρ +0.81 | 2.60, +13.0, ρ +0.95 |

      Crowns are thicker in every metric. Volume per white-surface area is the steepest, not the flattest.
    - **Reading.** The white surface alone does not fix the cross-section of a column: the cortex is not a parallel shell, and the area at the top matters (Waehnert's equivolume uses both surfaces). Where the top is the pia, its area is known; at a fused contact it would have to come from the neighbouring columns' tops.

11. **Collisions from all-WM fronts, and per-front arrival at the kissing stretch** (M1S1 test crop grown by 15 voxels; the gap closed within 1.5 mm of the 23 points; distance; all 3D).
    - **Collision structure.** One all-WM field and one upwind pass give a root to 98% of cortex. Voxels whose neighbours' roots are more than 1 mm apart along the white surface (same side of the WM) are collisions. They are classed by clustering the roots in each 3×3×3 neighbourhood at 3 mm: 44,131 sheet voxels, 2,401 junction and 63,595 sheet edge.
    - **By that classification, 19 of the 23 points lie on sheets and 4 on sheet edges** (points 7, 11, 12 and 17, which have many edge voxels nearby); no point is at a junction. The sheet edges are numerous across the box (concave white surface and fundi); the thresholds (1 and 3 mm) have not been tuned.
    - **The all-WM field cannot place the contact.** Its ridge, which is where ownership and equal arrival sit, is the Euclidean midline between the WM surfaces. That is where the data normal's sign flips. It lies about 0.6 mm from the points, on every point (`collide_map.png`, `collide_contact.png`). This was a stated limitation of the fit (fit_T).
    - **Per-front arrival.**
      - *Walls from roots, sorted into two sides:* the roots of the sheet voxels within 4 mm of the points, grown 1 mm along the white surface, fall into 3 pieces. Sorting them so that pieces facing each other across the sheets get different sides gives two sides with no conflict (16,003 voxel pairs). Each side, grown 3 mm, is a wall (15,567 and 7,990 voxels).
      - *Why all the pieces matter:* with only the most opposed pair of pieces, front 1 had no WM facing points 1–11. Its columns wandered 3–24 mm to a distant wall edge, its thickness was read at the wrong roots, and the contact curved more gently than the real curl (the user's reading of points 9–12 at the face's curl).
      - *Evidence ownership:* a visible top counts for front X only if its all-WM root lies on wall X. Otherwise tops of a third gyrus (about 4.6 mm) were counted.
      - *Fields:* one per wall, in the closed cortex within 9 mm of the points. Traced down from each point, front 1's columns are 1.6–2.3 mm long, matching its arrival.
      - *At the points:* front 1 reaches them at 2.01 mm (1.76–2.21) and front 2 at 2.81 mm (2.69–3.07). Traced, front 2's columns are 0.3–0.5 mm longer than its T: they curve, and the fitted T is not exactly path length.
      - *Rules:* equal arrival is 0.42 mm from the points (median |offset|, always toward front 2). The pairwise posterior has median offset −0.12 mm, IQR 0.09 and median |offset| 0.12 mm. Its worst points are 12 and 13 (0.28 and 0.30 mm), and none is missing. Its ±1 SD band is 0.07–0.24 mm.
      - *The thickness field still over-predicts front 1* at every point, by 0.07–0.54 mm (median about 0.25), within its SD (z ≤ +1.5). Front 2 is within −0.3 to +0.5 mm. This is the open-against-fused bias of finding 8.
      - *Views:* `collide2_contact.png`, `collide2_offsets.png`, `collide2_error.png`. In ITK-SNAP, `M1S1_contact_diag_overlay.nii.gz` adds each front's columns from the points, their roots, and the evidence tops by source.
    - **The stretch is unequal** (finding 12).
    - **Thickness field with a curvature trend** (distance; log t = m + (a₀ + w(x))H + z(x); H is white-surface mean curvature at a 0.8 mm scale; 6,466 evidence nodes):
      - The shared coefficient is small: a₀ = +0.07 ± 0.08 per (1/mm), against a prior of 0.5 ± 0.5. The local a(x) ranges from −0.09 to +0.21 (5th–95th percentile).
      - Where evidence is near, the 2 mm level field already follows curvature. So the trend barely changes held-out error (RMSE 0.118 against 0.120 in log thickness), with a slightly smaller bias at both ends of the curvature range.
      - Both versions are over-cautious: z SD 0.73 against 0.76; 98% inside ±1.96.
      - The trend's role is extrapolation, where a fused column is far from any visible top. That is not what held-out folds of visible tops test.
      - View: `collide_thickness.png`.

12. **The kissing stretch is unequal; finding 8's near-equality came from T and from a flipped geometric wall** (same box and closed gap as finding 11).
    - **Measures at each of the 23 points:**
      - T at the point;
      - the column traced down −∇T to its wall (length, and the WM voxel it reaches);
      - the straight distance to the nearest voxel of the wall, a lower bound that needs no field.

      They were taken for the two root walls (finding 11), and for finding 8's geometric walls rebuilt with its rule in the same box, each with its own field.
    - **Root walls, median and range over the points:**

      | | Front 1 | Front 2 |
      |---|---|---|
      | T | 2.01 mm | 2.81 mm |
      | Traced column length | 1.96 mm (1.64–2.28) | 3.29 mm (2.92–3.56) |
      | Straight distance (lower bound) | 1.96 mm | 3.19 mm (2.96–3.53) |

      Front 2's T is below even the straight-line bound, so the least-squares T under-reads long columns. For front 1, T, traced length and straight distance agree.
    - **The geometric walls.** Wall B is almost exactly root wall 1 (7,712 of its 8,389 voxels), with the same lengths. Wall A shares 2,854 voxels with root wall 2 but 1,767 with root wall 1: the flip at the curl. At points 2–7 and 12–13, some of A's columns end on front 1's side, and some wander (32 mm). Its T median is 2.21 against a straight-line bound of 3.02.
    - **Reading.** At the user's contact points, one bank's WM is about 2.0 mm away and the other's about 3.2–3.3 mm, by a measure that needs no field. Finding 8's 2.22 against 2.02 mm were T values on a contaminated wall. Which areas the two fronts belong to has not been checked. The user checked the walls and traced columns in ITK-SNAP and agrees with this reading.
    - **Views:** `unequal_walls.png` (both kinds of wall and their traced columns on the user's planes), `unequal_lengths.png`. In ITK-SNAP: `M1S1_unequal_overlay.nii.gz` with `M1S1_unequal_labels.txt` (the four walls, their columns, the points).

13. **In arc length, the posterior misses at the kissing stretch: the thickness field over-predicts the thin front.**
    - **Arc length:** carried in the upwind passes (all-WM and per front), and used for the thickness evidence and both rules. At the 23 points it matches the traced column length to within about 0.1 mm: front 1 is 2.02 mm (1.75–2.30), front 2 3.34 mm (3.10–3.70).
    - **Rules in arc length:**
      - *Equal arrival:* 0.67 mm off (median |offset|).
      - *Pairwise posterior:* median offset −0.33 mm, IQR 0.13, median |offset| 0.33 mm. In T it was 0.12 mm (finding 11), where T's under-reading of front 2 partly hid the error.
    - **The error is in front 1's prediction.** Front 2's μ is within about 0.1 mm of its arc length at most points, and within 0.4 mm at all of them. Front 1's μ is 0.26–0.92 mm too large (z up to +2.5). The curvature coefficient rose to a₀ = +0.17 ± 0.09, but H ≈ 0 at these roots, so the trend plays no part.
    - **Front 1's evidence.**
      - *Near the contact it is thin:* tops whose roots lie 2–6 mm from the contact's front-1 roots read 2.1–2.3 mm.
      - *Over the whole wall it is thicker:* 12,941 tops, median 2.56 mm.
      - *The field doesn't follow the nearby tops:* at point 16, the 50 evidence nodes within 3 mm have a median of 2.11 mm, but μ = 2.83.
    - **Variants of the level term change little** (`level_variants.py`). With level lengths of 4 or 6 mm, front 1 is still over-predicted by 0.53 and 0.46 mm, and the posterior's |offset| is 0.29–0.30 mm. A level offset for each front (the fronts as regions) comes out at about +0.02 for both. So neither a shared level nor too short a length explains it.
    - **Not yet understood**, and deliberately not patched further:
      - why the field departs from the nearby tops (its graph connectivity on the pooled white-surface nodes is unchecked);
      - whether the fused part of a bank is systematically thinner than its open part (a selection effect: fused roots are deeper in the sulcus);
      - whether the user's points, or the arc length near the contact, carry a bias of their own.

14. **Bank position at the kissing stretch: the visible tops nearest the fused zone already bracket the contact, so the field, not the evidence, is off** (A7, A8; `bank_position.py`).
    - **Setup.**
      - *Thickness:* arc length at each front's own visible tops, one value per root.
      - *Against:* geodesic distance along the white surface to the front's fused zone (the roots of its sheet voxels); sulcal depth below a hull (tissue closed with a 4 mm ball); and curvature H.
      - *Overlay:* the arc length at the user's 23 points, placed at their roots.
    - **Front 1 (thin):**
      - *Visible tops:* 4,532 roots, median 2.18 mm. Within 2 mm of the fused zone, 2.21; 2–4 mm away, 2.10.
      - *The contact points* (2.02 mm) sit at the lower edge of the tops at the same depth: about 0.1–0.2 mm thinner.
      - *Covariates:* no trend with depth (ρ −0.08); a mild one with curvature (ρ +0.35).
    - **Front 2 (thick):**
      - *Visible tops:* 1,718 roots, median 3.09 mm. They thin with sulcal depth, from 3.39 to 2.77 mm (ρ −0.65).
      - *The contact points* (3.34 mm, at depth 9–12 mm) sit on that trend.
      - *Curvature:* their roots are more convex (H 0.2–0.75 /mm) than most of the front's tops, so a curvature trend extrapolates there.
    - **Reading.**
      - *The evidence is not strongly selected (A7):* at this stretch, open and fused roots differ by about −0.2 mm (front 1) and +0.25 mm (front 2).
      - *The evidence near the fused zone predicts front 1 at about 2.2 mm, but the field said 2.5–2.8 mm (finding 13).* So most of the over-prediction comes from how the field is built (A6), not from what is observed.
      - *Sulcal depth matters on front 2* and is a candidate covariate (A8). Curvature at the contact roots lies outside the evidence's range on front 2.
    - **A limit of this view:** the walls reach only about 4 mm from the fused zone (the sheet roots grown 3 mm), so a gradient along the whole bank, from crown to fundus, cannot be seen here.
    - **View:** `bank_position.png`, with each wall unrolled and the fused zone, the tops and the points' roots marked.

15. **The field's graph and fit are sound; its second-order prior extrapolates a local rise of the evidence into the fused zone** (A6; `a6_check.py`; no model change).
    - **Connectivity.**
      - The node graph has 28,952 nodes in 4 pieces, with 27,086 in the main one, and a median degree of 8.
      - Every contact root node (both fronts, all 23 points) is in the main piece, with all the evidence nodes within 3 mm of it.
      - Graph distance to the nearest evidence is close to straight distance (0.3–3.6 mm). The exceptions are points 5–7: there the straight-nearest evidence node is 4–9 mm away along the graph, but another evidence node is about 2 mm away.
    - **Fit at evidence nodes:** residual (log thickness) median −0.001, IQR 0.027, 95th percentile of |residual| 0.09. The field honours its evidence, including at the nodes within 3 mm of the contact (front 1: evidence 2.1–2.5 mm, field 2.1–2.5).
    - **At front 1's contact root nodes the field reads 2.4–3.0 mm**, against arc lengths of 1.75–2.30 at the points.
      - *Its parts there:* the level term z is −0.10 to +0.11, the shared level m is 2.67 mm, and the curvature terms are within ±0.02.
      - *The nearest evidence along the graph rises toward the contact:* at points 16 and 19, tops 2–3 mm away read 2.10–2.11 mm, and tops 1.5–2 mm away 2.39–2.45 mm. All of them are on wall 1.
      - *The α = 2 prior extrapolates that gradient onward to 2.8 mm.* This is by design: second-order priors extrapolate gradients, which is why the plan chose them.
    - **Front 2's contact roots** read 2.9–3.9 mm, against nearby evidence of 2.9–3.5 mm, with the same extrapolation, smaller.
    - **Reading:** finding 13's over-prediction is not connectivity and not misfit. The edge rise came mostly from the gap closure, and it is not the main cause (finding 16).
    - **View:** `a6_check.png` (graph against straight distance; fit at evidence nodes; field against nearby evidence at the contact).

16. **The gap closure does not bias the score (A10); the field over-predicts because it reverts to a broad mean across the evidence-free gap before the contact roots (A6).**
    - **The gap at the points is essentially closed in the image already:** 1 fluid voxel within 0.5 mm of all 23 points. The point closure (fluid within 1.5 mm becomes cortex, 459 voxels) removed fluid near the contact, not in it.
    - **A fix in passing:** visible tops are now taken next to fluid that stays fluid after the closure. Before, cortex next to closed fluid still counted as a top.
    - **Three closures, the whole pipeline rerun** (`CLOSURE` switch in `collide.py`):

      | Closure | Fluid voxels closed | Arc length at the points (front 1 / 2) | Posterior: median, IQR, median \|offset\| | Front 1 μ − S | Front 2 μ − S |
      |---|---|---|---|---|---|
      | Points (within 1.5 mm) | 459 | 2.02 / 3.34 mm | −0.34, 0.15, 0.34 mm | +0.70 | −0.02 |
      | None | 0 | 2.02 / 3.34 | −0.35, 0.14, 0.35 | +0.68 | −0.03 |
      | Thin slivers (< about 3 voxels; no points used) | 1,616 | 2.02 / 3.34 | −0.41, 0.08, 0.41 | +0.86 | −0.00 |

      The score does not depend on the closure, so A10 holds.
    - **The edge rise of finding 15 came mostly from the point closure.** Front 1's tops 1–1.5 mm (graph) from its contact roots read 2.40 mm with the point closure, against 2.17 with none. With no closure they rise only slightly toward the contact: 2.09 mm at 3–4 mm, 2.26 at 0–1 mm. The over-prediction stays at +0.68 mm all the same.
    - **The prior's correlation length is what it should be** (`prior_length.py`). On the graph, correlation at nominal length 2 mm is 0.86 at 0.5–1 mm, 0.74 at 1–1.5 mm and 0.42 at 2–3 mm, against 0.88, 0.77 and 0.50 for the continuum Matérn. The graph's scaling is right: node spacing is 0.34 mm, with a Laplacian diagonal of 39 /mm² against 35 for a square grid.
    - **What the field does instead** (no closure; point 16): along the graph path from the last visible top (2.35 mm, 1.65 mm away) to the contact root, the mean rises steadily from 2.35 to 2.83 mm. Its level term goes from −0.09 to +0.05, slightly above the box's shared level of 2.67. Point 10 does the same (2.35 to 2.63 over 1.25 mm).
    - **Reading.** The contact roots lie 1–2.5 mm along the white surface beyond the last visible top. Across that evidence-free gap, a stationary Gaussian field reverts toward a broad mean of about 2.6–2.7 mm: the front and the box, not the locally thin cortex near the contact (about 2.1 mm). The second-order prior adds a small overshoot. This is extrapolation over a gap, not selection on thickness (A7 holds, finding 14), and not a construction error (A6's graph and fit are sound, finding 15). The sums t₁ + t₂ = L lie inside that gap: they are the evidence the field is missing there.

17. **The contact has no fluid gap: two superficial layers meet, thinner than at open pia (A5)** (`a5_profiles.py`; image only, no fitted field across the contact).
    - **Setup.**
      - *Profiles:* at each of the 23 points, from −2.5 to +2.5 mm along the contact's own normal (a plane fitted to the neighbouring points, signed toward front 2's wall), averaged over a 0.3 mm disc.
      - *Images:* echo 2 trend removed, log echo 2 and log echo 3.
      - *Reference:* 238 and 297 profiles across open pia on fronts 1 and 2, from visible tops 2–6 mm from any point, aligned at the cortex/fluid boundary and set back to back.
    - **The median contact profile is two pial profiles back to back with the fluid removed.**
      - *Raw log echo 2 and 3:* outside ±0.3 mm the contact median follows the back-to-back reference closely. At 0 there is only a slight notch: no fluid, where the reference drops by about 2.5 in log units.
      - *Trend-removed echo 2:* two bright superficial peaks at about −0.3 and +0.3 mm, with a notch exactly at the user's points. So the points mark where two layer I's meet.
    - **The superficial bands are closer together than at open pia.** The peaks are about 0.6 mm apart, against about 0.85 mm for the open-pia reference (−0.4 and +0.45 mm). The superficial layers are thinner where the banks are fused, by roughly 0.1–0.15 mm per side.
    - **Per point the notch is weak.** Its depth over the flank noise has a median of 1.1; only point 1 is clear (7.8). At 7 points (4–7, 12, 17 and 20), the plane fitted to the user's points is 56–90° from the data's layer normal: the neighbouring points are nearly collinear there, so those profiles may not cross the contact squarely.
    - **Reading:** A5 holds at the kissing stretch: the contact is a zero-thickness meeting of two layer I's, with no gap to add to L. The superficial thinning at the contact matches front 1 reaching the contact about 0.1–0.2 mm short of its nearby open thickness (finding 14).
    - **View:** `a5_profiles.png`.

18. **A second kissing face, the fused V1 banks: the per-front posterior lands within about a voxel; the field over-predicts both banks, and the bias cancels in the split** (`V1V2_banks`, 35 user points on seven k slices from 45 to 83; no gap closure; arc length; the same pipeline as findings 11–16, made site-independent).
    - **Geometry.** Walls from roots: the sheet's roots form 3 pieces, which sort into two sides with no conflict. At the points, front 1 reaches them at 2.18 mm (1.85–3.26) and front 2 at 2.31 mm (1.62–2.99): near-equal, matching the survey's open thicknesses (2.37 and 2.51 mm). The user's points lie within 0.36 mm of the all-WM sheet, consistent with near-equal banks.
    - **Rules at the 35 points:**
      - *Pairwise posterior:* median offset +0.06 mm, IQR 0.15, median |offset| 0.11 mm; worst 0.44 mm (point 23) and 0.34 (point 12). No point is missing.
      - *Equal arrival:* median −0.05, IQR 0.23, median |offset| 0.12 mm.
    - **The field over-predicts both banks at their contact roots.** Front 1's μ − S is +0.2 to +1.0 mm at most points (z up to +4.4), and front 2's +0.1 to +0.4 at many. This is the same drift across the gap before the contact as at M1/S1 (finding 16); the survey put front 1's contact roots about 0.9 mm beyond the nearest visible top. Because both beliefs are too large, the split of the sum still lands near the contact. At M1/S1 only the thin front was biased, and the posterior missed by 0.33 mm. This is A9: the two fronts' errors are correlated here, and that correlation, not the beliefs' accuracy, is what saved the split.
    - **Curvature trend:** small again, a₀ = +0.03 ± 0.08 per (1/mm) in the all-WM field and +0.12 ± 0.09 in the per-front one.
    - **Calibration on held-out tops:** over-cautious, with z SD 0.49.
    - **Views:** `V1V2_banks_collide2_contact.png`, `V1V2_banks_collide2_error.png` and the other `V1V2_banks_*.png` figures. In ITK-SNAP: `V1V2_banks_contact_overlay.nii.gz` with `V1V2_banks_contact_labels.txt` over `V1V2_banks_echo2_detrended.nii.gz`.

19. **PCC_SPL: one face works, one fails at the walls** (box of 141 voxels re-centred on the user's points; no gap closure; arc length).
    - **Piece 2** (`PCC_SPL_p2_run`, 37 points; open thickness 2.81 and 2.57 mm).
      - *The banks are close to equal* at the points: 2.79 mm (2.50–3.69) and 2.66 mm (2.07–3.17).
      - *Pairwise posterior:* median offset +0.06 mm, IQR 0.18, median |offset| 0.11 mm. Equal arrival: median |offset| 0.18 mm.
      - *Points 31–37 miss by 0.33–1.16 mm.* These are the lowest slice (k = 57) and the two points nearest the old crop's edge. There the two fronts' errors are large and of opposite signs (front 1 too thin by up to −0.94 mm, front 2 too thick by up to +0.83), unlike the rest of the face. Not yet looked into.
    - **Piece 1** (`PCC_SPL_p1_run`, 22 points): **the walls failed.** The roots on one side of the sheet near the points came to 158 voxels against 6,743 on the other. Front 2's field therefore starts from the wrong WM: it reaches the points at 5.5–7.2 mm, and no rule crosses near any point. Front 1 reaches them at 2.33 mm (2.02–2.74), with μ − S of −0.14 to +0.43 mm. The cause is the side-sorting, not the distance rule (finding 21).
    - **Views:** `PCC_SPL_p1_run_*.png`, `PCC_SPL_p2_run_*.png`.

20. **Three local tests do not separate contacts from a front meeting itself** (`sheet_tests.py`; every sheet voxel of the surveyed crops and the M1S1 stretch, at least 2 mm from the faces; the user's readings: contact 23,550 voxels, not a contact 1,282, unread 99,669).

    | Test | Contact | Not a contact |
    |---|---|---|
    | \|cos\| between the layer normal and the sheet normal: median (IQR) | 0.98 (0.05) | 0.97 (0.26) |
    | Distance to WM: median (10th percentile) | 2.50 mm (1.99) | 2.16 mm (1.68) |
    | Distance to fluid: median | 0.93 mm | 0.48 mm |
    | Doubled-band index (echo 2 trend removed, across the sheet): median | −0.03 | −0.01 |

    - **Orientation does not separate them.** At piece 3's tight bend the layers run nearly parallel to the sheet too (0.91), as at a contact.
    - **The band index fails even at known contacts.** The sheet is the all-WM midline, not the contact, so a profile centred on the sheet misses the notch (finding 17 centred it on the user's points).
    - **Distance to fluid is the only hint:** non-contacts lie closer to fluid. It is weak, and rests on only two non-contact regions (piece 3, and the lower part of PCC piece 1).
    - **Reading:** with the sheet taken from the all-WM field, local geometry and contrast do not tell two stacks from one stack bent back on itself. A test would need more labelled non-contacts, or would have to look at the contact found per front rather than the midline, or at whether the two sides' layer I's actually touch.
    - **Views:** `sheet_tests.png`; per crop, `<crop>_sheet_tests_overlay.nii.gz` with `sheet_tests_labels.txt` (sheet voxels by orientation).

21. **Why PCC piece 1's walls failed: both banks' roots joined into one piece before the sides were sorted** (checked at the 22 points; no model change).
    - **The two sides are separate fronts.** Just either side of each point (along the data's layer normal), the all-WM roots lie about 5 mm apart in a straight line and 11–18 mm along the white surface, at 20 of the 22 points. The collisions there are correctly classed as sheets (108–236 sheet voxels within 1 mm of each point). The distance rule did not merge them.
    - **Both sides' roots are on wall 1, at every point.**
      - *The mechanism:* at the sheet near the points there are 3,756 facing root pairs. In 3,753 of them (100%), both roots fall in the same piece: one grown piece of 6,743 voxels, against 84 and 74 for the others. The sheet's roots run continuously along the white surface from one bank, round the bottom of the sulcus, to the other. So growing them 1 mm joins the two banks into one piece.
      - *Why nothing flagged it:* the side-sorting only compares different pieces, so facings inside one piece are never seen, and it reported no conflicts.
      - *Why it worked elsewhere:* at M1/S1 and the V1V2 banks, the two sides' roots happened not to join.
    - **Reading:** sorting roots into sides by connected pieces is fragile. Sides should come from the facings themselves, root by root (two roots facing each other across a sheet are on opposite sides; neighbours on the white surface usually on the same side), with conflicts reported. This is a change to `collide2.py`'s wall step, not yet made.
    - **Piece 3, the false sheet: probably a thin Fomblin gap at a tight bend, labelled cortex.**
      - *The user's reading:* the sheet lies in the cortex at a tight bend with at most minimal kissing. It seems influenced by a mislabelled region caught in the sharp fold, connected to ITK-SNAP (82, 78, 71).
      - *That region:* labelled cortex (aseg 42 at all of its voxels), WM-dark in echo 3, 3.1 mm from the nearest labelled WM and 0.9 mm from the sheet. Connected below a WM/cortex threshold, it is a strip one voxel thick lying in one slice (k = 71), about 7 × 2 mm, touching no WM.
      - *Its echo decay says Fomblin, not WM.* The decay (log echo 1 − log echo 3, medians) is 0.68 in the region (0.47 at the user's point), against 0.95 for mid-cortex, 1.12 for deep cortex next to WM, 0.55 for cortex next to fluid, and 0.19 for Fomblin. WM's short T2* would make a WM piece decay faster than cortex; this decays slower, between cortex and the cortex–Fomblin partial-volume voxels.
      - *Reading:* a thin Fomblin-filled gap at the bend (the minimal kissing) is labelled cortex. The folded surfaces on either side are treated as continuous cortex, so their fronts meet across the hidden gap and make a sheet. This is a segmentation error of the opposite kind to a missed WM piece: fluid labelled as tissue.
      - *Not a thin WM spur either:* the WM's half-thickness at the sheet's 407 roots is about 0.8 mm, as elsewhere.
      - *No wide band of missed WM at the roots:* of the cortex-labelled voxels within 1 mm of the sheet's roots, 57% read WM-dark in echo 3, against 46% next to the inner white surface of the whole box. The difference is small; the dark strip at (82, 78, 71) is the clear anomaly.
      - *What it implies:* false sheets can come from errors in the aseg's labels, and the fronts start and stop where the aseg says. A sheet next to a thin, slowly decaying dark strip inside labelled cortex could be flagged as a hidden gap. Separately, the T = 0 boundary could come from the image's own WM contrast (the constraint inventory's contrast-defined deep anchor).
    - **Views (ITK-SNAP):**
      - `PCC_SPL_p1_run_walls_diag_overlay.nii.gz` with `PCC_SPL_p1_run_walls_diag_labels.txt`: walls 1 and 2, the one piece holding both sides, a path joining them, the sheet and the points.
      - `PCC_SPL_p3_roots_overlay.nii.gz` with `PCC_SPL_p3_roots_labels.txt`: the WM label, piece 3's sheet, its roots, the cortex whose columns start there, the dark strip connected to the user's point (82, 78, 71), and the point.
22. **`fronts` accuracy at cortical scales: arrival is good; arc length is good only where paths follow the metric's direction** (synthetic; ball sources of radius 3, 6 and 12 voxels; errors 12–20 voxels out, about 2–3.2 mm at 160 µm; the native grid and a 2× finer one; errors in native voxels).
    - **Converging is not the same as accurate.** The fixed transport scheme converges (the old one's error stayed flat, a bug). How large the error is on an affordable grid depends on how paths lie against the metric's direction.
    - **Worst case: a constant direction, so most paths cross it** (`fronts_length_misaligned.py`).
      - *Arrival:* 3–6% median at every anisotropy, about 0.05–0.13 mm. It halves on the 2× grid. At anisotropy 10 the 95th percentile is about 10%.
      - *Arc length:* 0.5–0.8 voxels at anisotropy 1, 1.3–2.4 at 3, and 4.3–5.6 (0.7–0.9 mm) at 10. The error barely shrinks with a flatter source. At anisotropy 10 the 2× grid gives only 2.6–3.5 voxels, an order of about 0.7. Refining further is a question of memory, not a limit: a whole 141³ box at 2× needs about 21 GB, which is allowed.
    - **The cortex-like case: the direction radial from the ball, so paths follow it** (`fronts_length_aligned.py`; the exact arrival and length are both r − R).

      | Anisotropy | Arrival, native → 2× | Arc length, native → 2× |
      |---|---|---|
      | 1 | 0.7–0.9 → 0.3–0.5 | 0.7–0.9 → 0.3–0.5 |
      | 3 | 0.7–0.8 → 0.4 | 0.6–0.7 → 0.3 |
      | 10 | 1.4–1.5 → 0.7–0.8 | 1.2–1.3 → 0.6 |

      - Arc length is 4–8 times better than in the worst case. It is a clean first order: it halves on the 2× grid.
      - The errors are a near-constant overestimate. The median error equals the bias, and the 95th percentile is close to the median. Such an offset partly cancels where fronts or lengths are compared, but not in absolute thickness. Whether it grows with distance has not been checked.
      - Anisotropy 3 costs nothing against an isotropic metric. Anisotropy 10 about doubles the error.
    - **Reading.** The weighted competition (start times and speeds) acts on arrival, so it is sound at native resolution. The solver's own arc length looks usable for the thickness beliefs if the data's direction follows the columns. The anisotropy we can afford depends on how far the data's direction tilts off the columns. If the solver's length is not good enough there, length can be traced back along the arrival gradient as a post-processing step.
    - **Not yet tested:** a direction tilted off the true paths (fixed tilts of 5, 10 and 20° and smooth random tilts, at anisotropy 3 and 10). This is the case between the two above, and it is the one real data will be in.

23. **The one-pass solver with per-source weights (PR #26), and what labels must be** (`baillarger.fronts`; `onepass_lib.py`, `test_onepass_lib.py`; all 3D).
    - **The solver.** `march` takes `start_time` and `speed` per label. A label's update solves the scheme with right-hand side 1/speed², so its arrival is start + (metric distance)/speed; labels update only from their own accepted neighbours, so the competition is crystal growth. Arc length is unchanged by both weights (tested to 1e-10). Correctness tests: exact plane arrival t₀ + x/c; t₀ + (unweighted)/c from a point source; hyperboloid boundaries from start times; the slow source's Apollonius ball inside its region, with a tail only in its shadow; connected regions under random weights; a swallowed late source; weights with anisotropic voxels and a wall; input checks (a speed whose 1/speed² overflows is rejected: found by the self-review, with a regression test).
    - **Labels are root patches, not root voxels.** With one label per WM-boundary voxel, each label's update sees only its own one-voxel-wide tube, and the arrival error doubles or triples (point-like ball source: 4.7 → 10.5 % isotropic, 6.5 → 16 % at anisotropy 3). Patches are geodesic Voronoi cells of about 1 mm (6 voxels) on the seed graph (26-neighbours whose outward normals agree, so the two sides of a thin blade are not joined). An earlier block-based patching left one-seed slivers whose regions pinched out, so their neighbours touched and looked like a contact; the geodesic cells remove them.
    - **Crystal growth against the pure weighted Voronoi partition** (brute force: one march per patch, the weighted minimum afterwards). For start times alone (equal speeds) the regions are provably star-shaped, so the two must agree: they do, except within 1 voxel of a boundary, where each label's one-sided update makes its arrival late by a median of 0.46 voxel (95th percentile 1.17). For unequal speeds they differ by design; the slow side's pure region lies inside its crystal region.
    - **A contact is a label boundary between roots far apart along the white surface.** Neighbouring patches' regions touch wherever one pinches out, so adjacency alone cannot be the test. The separation (geodesic on the seed graph) runs about 10 % above straight distance on a staircased plane. Its threshold is the weak point (finding 25).
    - **Thickness evidence must be column tops, signed by the front itself.** Cortex next to fluid counts as a top only where the front runs into the fluid (the voxel behind on the same label has an earlier arrival) and the column axis is within 45° of the step. Two failures led here: cortex that meets fluid sideways (the side faces of a filled patch) was counted as tops and inflated the beliefs under it (P4: 18.0 and 21.5 against 12.5 and 18.5); and a direction signed by the WM distance points the wrong way in a thin gap, where the far bank's tops are nearer the other WM (P8: the thick bank had no tops at all). Each has an exact test.

24. **Synthetic contacts: the pass places the boundary where its weights say, and folds are told apart by arrival over belief** (`synth_contacts.py`; ribbons 12–18 voxels, cost ratio 3 across the distance-gradient direction; voxels).

    | Shape | Known | Equal weights | Open tops (any fluid) | Column tops | Oracle |
    |---|---|---|---|---|---|
    | U-slab, fundus R = 6 < t = 16 | a fold: fronts meet on the bisector from arrival R to t (z 20–43.7) | meeting on the bisector, z 24–42; d/μ 0.89 on both sides | same | same | — |
    | U-slab, R = 22 > t | no meeting | 1 pair (separation 18.5, at the threshold) | none | none | — |
    | P4: banks 12 / 18, the gap filled, contact (dark line) at x = 25.5 | +2.45 for open thicknesses (the contact is not at the natural thicknesses) | +7.0 (the midline) | +4.9 (μ 18.0 / 21.5: side faces counted) | **+2.45** (μ 12.5 / 18.5) | 0.0 |
    | P8: the same banks touching at their natural thicknesses, x = 22.5 | 0 | +3.0 | +0.72 | **−0.19** | 0.0 |
    | Omega: chamber radius 34, neck filled, sides 12 / 18, x = cx − 3.5 | 0 | +3.0 | −0.21 | **−0.21** (worst 0.63) | 0.0 |

    - With weights equal to each side's true thickness, the boundary lands on the contact to sub-voxel accuracy, and the two arrivals at it add up to L. With the evidence's weights it lands within about 0.2 voxel where the contact is at the banks' natural thicknesses, and exactly where those thicknesses put it otherwise (P4's +2.45 is the rule's own prediction).
    - Crystal growth and the pure partition agree on the contact (0.2–4 % of voxels differ, all at boundaries).
    - At the fold (R < t), the meeting has d/μ ≈ 0.89 on both sides: both fronts are short of their thickness, as the theory predicts for a front meeting itself round a tight bend. At the contacts d/μ is about 1. This is the reporting class of *Collisions and contacts* (point 2), seen on a known case.
    - Where neighbouring patches have different weights, the faster one's region leaks sideways into the slower's (the omega's chamber floor, where μ steps from 12.5 to 18.5): crystal growth between neighbours, visible as d/μ > 1 beside the step.
    - Views: `onepass_synth_{uslab_R6,uslab_R22,P4,P8,omega}.png` (each weight setting; a zoom on the contact).

25. **The one pass at the four marked faces: it matches or beats the walls at three, and misses at the V1V2 banks** (`onepass_face.py`; one pass from all WM in a box of 141 voxels, no gap closure, cost ratio 3 across the data's layer normal; thickness field as findings 11–19, with evidence from column tops; mm).
    - **Scoring.** At each point the two sides are found locally: the labels at the two ends of the column line through the point (2.5 mm each way), every patch going to the end nearer along the white surface. The crossing is the label boundary between the two groups within 1 mm of the point and 0.4 mm of the line. No separation threshold enters, and the score with the 3 mm threshold agrees with it on every face. Offsets are + toward the side the field predicts thinner. The nearest crossing in any direction is reported beside it.

      | Face | Equal: \|offset\| median | Field weights: median, IQR, \|offset\| median, worst, missing | Nearest crossing (field), median / 90th | Before |
      |---|---|---|---|---|
      | M1/S1 (23 points) | 0.68 (all toward the thick side) | −0.26, 0.23, 0.26, 0.66, 0 | 0.23 / 0.44 | equal 0.67; posterior −0.33, IQR 0.13 [13] |
      | V1V2 banks (35) | 0.42, 10 missing | +0.03, 0.79, 0.40, 0.90, 4 | 0.31 / 0.65 | posterior +0.06, IQR 0.15, 0.11; equal 0.12 [18] |
      | PCC_SPL piece 1 (22) | 0.13 | +0.06, 0.09, **0.08**, 0.25, 0 | 0.12 / 0.18 | walls failed: no rule crossed near any point [19, 21] |
      | PCC_SPL piece 2 (37) | 0.15 | +0.05, 0.29, 0.16, 0.72, 1 | 0.16 / 0.36 | posterior +0.06, IQR 0.18, 0.11; points 31–37 off by 0.33–1.16 [19] |

      With σ = 0.1 μ in place of the field's σ: 0.31, 0.33, 0.08 and 0.12.
    - **M1/S1 reproduces finding 13 without walls or per-front fields.** Equal arrival misses by 0.68 mm, toward the thick side. Field weights halve it, and the error breakdown is finding 13's: the thin front is over-predicted (μ − S +0.5 to +0.8 at points 2–15) while the thick one is near zero. The box is re-centred on the points: the test crop grown by 15 voxels (findings 11–16) leaves one point 4.8 mm from a face, against the 6 mm rule.
    - **PCC piece 1 now works.** With no sides to sort, finding 21's failure cannot occur. The boundary passes through the points within 0.08 mm (worst 0.25). The views show the boundary scattered into a cloud, not a sheet, inside the dark band along the contact (j = 73), where coherence is low.
    - **PCC piece 2** is as before, with the same failure at points 31–37 (−0.35 to −0.67): the two fronts' errors are large and of opposite signs (thin side +0.2 to +1.0, thick side −0.6 to −1.3).
    - **The V1V2 banks miss, even with equal weights.** Along the data's layer normal through every point, there is no WM on the far side within 8 mm: the line runs 4–7 mm through cortex to fluid. Coherence at points 1, 3 and 10 is 0.15–0.38. So the normal there does not point across the contact, the far bank's front reaches the points late (5–6 mm at points 1–3), and the boundary sits about 0.3 mm off (nearest crossing). Finding 18's per-front fields, which follow the data only through a fit, were within 0.11. **The cost ratio across the normal decides it here** (\|offset\| median with field weights; σ = 0.1 μ in brackets):

      | Ratio | V1V2 banks | M1/S1 |
      |---|---|---|
      | 3 (the default) | 0.40 (0.33), 4 missing | 0.26 (0.31) |
      | 1.5 | 0.23 (0.24), none missing | 0.24 (0.25) |
      | 1 (isotropic: distance) | **0.16 (0.10)**, none missing | 0.30 (0.28) |

      Where the normal is unreliable at the contact, steering by it moves the boundary; plain distance does better there and no worse at M1/S1. The ratio has no evidence behind it yet (an open decision).
    - **The separation threshold cannot be principled.** At the V1V2 banks the banks join round a fundus near the points, so roots facing each other across the contact are under 3 mm apart and the 3 mm set drops the contact there. At 1 mm the set grows tenfold (394,000 pairs against 35,000) and fills with boundaries between neighbouring patches, which also bias any threshold-based score toward zero (M1/S1's equal arrival read 0.16 at 1 mm, against the 0.67 that finding 13 measured without a field). The local two-sides score above avoids the threshold for scoring; the contact set itself still needs one for display.
    - **Views:** `onepass_<face>_offsets.png` (offsets per point for each weight setting; μ − S per side with ±1 SD) and `onepass_<face>_planes.png` (the three planes holding most points, echo 2 trend removed). In ITK-SNAP: `onepass_<face>_overlay.nii.gz` with `onepass_labels.txt`, over `onepass_<face>_echo2_detrended.nii.gz`, for M1S1, V1V2_banks, PCC_SPL_p1, PCC_SPL_p2. `_sep1mm` variants: the 1 mm threshold.

26. **The weights need care: uncertain beliefs flood under crystal growth.**
    - **No evidence, no belief.** At the V1V2 banks the thickness field gave one patch μ = 5.9 m with σ = 1.4 × 10⁷ mm. Its node lies in a piece of the field's graph with no evidence, where the log-thickness SD is large and the lognormal mean exp(m + sd²/2) explodes. With σ = 0.1 μ that patch took 1,066,541 of 1,081,577 cortex voxels. Patches in pieces without evidence (9, 100, 77 and 7 at the four faces) now get the equal-arrival weight (the median μ and σ), counted.
    - **Evidence near box faces.** Ten patches with tops of 5.6 mm, centred about 3.6 mm from a face, owned the largest regions. The rule that columns near a face are left out now applies to the roots too (evidence only from roots ≥ 2 mm from a face).
    - **The rule's own tail** [inference]: min (d − μ)/σ means that past every root's expected thickness, the most uncertain root wins, and crystal growth lets it keep going. With sane beliefs this did not matter at the four faces, but a belief's σ also sets how far its root's territory can run. Whether that is wanted is open.

27. **`baillarger.fronts` in its intended use on the complete patches: fast and complete, but the contact set needs work outside the solver** (`fronts_eval.py`; one pass from all WM in each whole patch of `baillarger.data.PATCHES`, cost ratio 3, root patches of about 1 mm; equal weights, then field weights with the guards of finding 26; evaluated only, nothing changed).

    | | PCC_SPL | M1S1 | V1V2 |
    |---|---|---|---|
    | Patch (voxels); cortex | 125³; 0.93 M | 151³; 1.54 M | 140 × 150 × 245; 1.48 M |
    | Seeds; root patches | 57,659; 2,018 | 90,061; 3,102 | 81,560; 2,782 |
    | One pass (s); whole run (s); peak (GB) | 3.6–5.9; 59; 2.9 | 6.6–9.1; 112; 5.4 | 6.3–8.6; 202; 8.3 |
    | Cortex reached | 100 % | 100 % | 100 % |
    | Column-top thickness, p5 / 50 / 95 (mm) | 1.99 / 2.50 / 3.29 | 2.06 / 2.67 / 3.44 | 1.64 / 2.30 / 3.26 |
    | Arc length over arrival, p5 / 50 | 0.69 / 0.92 | 0.75 / 0.92 | 0.63 / 0.90 |
    | Field: raw μ maximum; patches without belief | 1.3 km; 27 | 5.9 mm; 79 | 4.0 m; 24 |
    | Contact pairs, equal / field | 11,679 / 12,718 | 37,772 / 42,621 | 36,670 / 42,983 |
    | Within 2 mm of a box face | 63 % | 44 % | 28 % |
    | Equal weights: contact-, fold-, overlap-like | 63 / 3 / 33 % | 69 / 4 / 26 % | 37 / 3 / 59 % |
    | Field weights: contact-, fold-, overlap-like, mixed | 62 / 2 / 27 / 8 % | 64 / 1 / 5 / 29 % | 42 / 7 / 27 / 24 % |
    | Opposing label known on both sides (sub-voxel) | 45 % | 40 % | 45 % |

    (Classes by d/μ on the two sides: contact-like both 0.8–1.25, fold-like both < 0.8, overlap-like both > 1.25. The peaks and most of the time are the prototype's curvature and thickness field, not the solver.)
    - **What works.** Each pass takes under 10 s at up to 5 M voxels, within a few GB. Every cortical voxel is reached. In the views, the contact-like boundaries lie in the fused and kissing sulci.
    - **The box faces dominate the contact set:** a quarter to two thirds of it lies within 2 mm of a face. The package knows only the mask, so a front that would have come from WM beyond the face is missing and its neighbours meet in its place. The plan's rule (columns that come within 2 mm of a face are left out) has to be applied by the caller, or the domain given a free boundary at the faces.
    - **Overlap-like boundaries are common** (26–59 % with equal weights), where both fronts are already past their thickness. In V1V2 several cut straight across a gyrus. Two WM pieces of one gyrus, disconnected in the label, meet there as separate bodies (infinite separation). Others run into the faces. Field weights move many to "mixed". They are flags for label errors and faces, as *Collisions and contacts* point 2 intended, but they are not yet separated from real contacts.
    - **The second-best arrival is often a neighbour.** `march` keeps the best arrival of any other label, which is usually a neighbouring patch, so at only 40–45 % of contact pairs are both sides' arrivals known. The rest are located by extrapolating one step. Keeping the best arrival of a far label, or of a few labels, would locate all of them.
    - **Arrival is not arc length where paths cross the normal:** at ratio 3 the arc length is up to 25–37 % below the metric arrival (5th percentiles). Thickness beliefs are in arc length and the competition in arrival, so the two units part where columns tilt (finding 22's open case).
    - **The package accepts any positive, finite speed.** A belief with σ = 10⁷ mm (from the field, finding 26) passes the input checks and floods the box. That is the caller's to guard, but worth a warning in the docstring.
    - **Labels are the caller's choice and matter.** One label per root voxel triples the arrival error (finding 23), so the intended use needs root patches outside the package, or a root position carried through the pass (open decision 4 in Next step 0).
    - **Views:** `fronts_eval_<patch>_planes.png` (three planes through each patch centre, the contact set by class over echo 2 trend removed) and `fronts_eval_<patch>_hist.png` (separation and L at the boundary). In ITK-SNAP: `fronts_eval_<patch>_overlay.nii.gz` with `fronts_eval_labels.txt` over `fronts_eval_<patch>_echo2_detrended.nii.gz`.

28. **The white surface as boundary data for the front: smooth on the mesh, not in the volume; a smooth start removes the voxel start's bias, its wrong signs on a refined grid, and most of the near-WM noise** (`mesh_smoothing.py`, `test_mesh_smoothing.py` (19 tests), `mesh_phantoms.py`, `mesh_real.py`, `mesh_real_band.py`; prototype only, nothing in `src/` changed; mm, 0.16 mm voxels; all 3D).
    - **What was compared.** Start surfaces from the WM label:
      - *raw voxel seeds:* the WM-boundary voxels at arrival 0, as `onepass_lib.setup` makes them, with one label or with its 1 mm root patches;
      - *binary contour:* marching tetrahedra on the label itself (every vertex at a grid-edge midpoint);
      - *`white_mesh` at σ = 0.5, 1, 1.5, 2 voxels:* the label smoothed in the volume, then contoured;
      - *mesh smoothing:* the binary contour, relaxed tangentially (vertices even out within the tangent plane; the triangles are already about half a voxel across, so this stands in for remeshing), then Gaussian smoothing along the mesh at σ = 0.08, 0.16, 0.32, 0.48 mm with its shrinkage removed by twicing (x = 2Gv − GGv; "twice" below). Also from the label smoothed by 0.5 voxel, and Taubin λ|μ with its standard pass band;
      - *voxel faces:* the label's own exposed faces (the "Lego" surface), as boundary data;
      - *implicit (Chopp 2001), no mesh:* closest points on the 0.5 level of the label smoothed by σ = 0.5 or 1 voxel, tricubic interpolation, Newton steps to the interface. Two fixes were needed, each with a failing test first: the smoothed label is a sigmoid on which Newton overshoots from more than about σ away (55 % of a 2-voxel band stalled at σ 0.5), so the level-set function is σ·probit(smoothed label), exactly the signed distance for a blurred plane, with the same zero set; and the tangential step is halved, because beyond the zero set's local radius of curvature the full step cycles (22 % stalled).

      Why this grid: the volume σ spans from nothing removed to past the point where a 2-voxel blade's smoothed peak drops below 0.5 (σ = w/1.35); the mesh scales match it in mm and reach the radius of curvature of a blade tip. **Taubin λ|μ was replaced by twicing:** Taubin's gain is above 1 inside its pass band, and moving the pass band up to set a short cutoff made it blow up (gain 1.05 per pair, ×4.8 over 30 pairs, at a 2-voxel cutoff); twicing's response lies in [0, 1] (tested) and its shrinkage is second order.
    - **The phantoms** have a known smooth surface and exact distance, foot point, normal and curvature at any point (tested against brute force): a ball of radius 1.28 mm (H 0.78 /mm); a wavy body (cos x cos y, H up to ±0.2 /mm, rotated off the grid); a plane 4° off a grid axis (terraces 2.3 mm long); thin blades 1.5–4 voxels thick (a wavy sheet, cortex on both sides); fins 1.5–4 voxels thick rising 1.9 mm from a base to a rounded tip.
    - **The surface itself** (vertex error = signed distance to the true surface; curvature by `mesh_curvatures` at 0.48 mm; the quadric row is `curv_lib.curvature` at its voxel centres):

      | Start surface | Wavy: \|d\| median, p95 | Ball: bias, \|d\| p95 | Normal error, wavy | H error median / p95, wavy; ball (1/mm) | Blades 1.5 / 2 / 3 voxels: sides kept | Fin 2 voxels: tip retraction |
      |---|---|---|---|---|---|---|
      | Raw voxels (face quadric) | 0.076, 0.145 | −0.065, 0.130 | 11.5° | 0.013 / 0.048; 0.147 / 0.203 | 1 / 1 / 1 (voxels) | — |
      | Binary contour | 0.035, 0.086 | +0.002, 0.088 | 7.7° | 0.037 / 0.070; 0.083 / 0.396 | 1 / 1 / 1 | 0.00 |
      | `white_mesh` σ 0.5 | 0.030, 0.060 | −0.001, 0.057 | 6.5° | 0.026 / 0.054; 0.041 / 0.264 | 1 / 1 / 1 | 0.02 |
      | `white_mesh` σ 1 | 0.011, 0.033 | −0.019, 0.037 | 2.6° | 0.016 / 0.050; 0.017 / 0.041 | **0.61** / 1 / 1 | 0.16 |
      | `white_mesh` σ 1.5 | 0.008, 0.025 | −0.046, 0.058 | 1.4° | 0.016 / 0.047; 0.029 / 0.043 | **0 / 0.09** / 1 | **1.55** |
      | `white_mesh` σ 2 | 0.010, 0.025 | −0.084, 0.092 | 1.1° | 0.016 / 0.047; 0.054 / 0.064 | **0 / 0** / 1 (sides −0.11) | **2.08** (gone) |
      | twice 0.16 mm | 0.020, 0.048 | +0.001, 0.036 | 4.8° | 0.016 / 0.049; 0.008 / 0.031 | 1 / 1 / 1 | 0.02 |
      | **twice 0.32 mm** | **0.007, 0.027** | −0.003, 0.028 | 1.5° | 0.015 / 0.046; 0.006 / 0.016 | 1 / 1 / 1 | 0.08 |
      | twice 0.48 mm | 0.005, 0.018 | −0.017, 0.058 | 0.9° | 0.014 / 0.042; 0.014 / 0.039 | 1 / 1 / 1 | 0.14 |

      - *Volume smoothing* shrinks convex surfaces by about σ²κ (the ball: −0.02, −0.05, −0.08 mm at σ 1, 1.5, 2) and **erases thin WM**: blades of 1.5 voxels lose 40 % at σ 1, blades of 2 voxels vanish at σ 1.5, a 3-voxel blade's sides move 0.11 mm inward at σ 2, and fin tips retract (0.16 mm at σ 1 for 2 voxels; 1.5 mm, the whole fin, at σ 1.5). Being level sets, its contours never self-intersect.
      - *Mesh smoothing* keeps every blade and both sides at every scale, with no self-intersections on the phantoms, no position bias on gentle surfaces, and a tip retraction of 0.02 / 0.08 / 0.14 mm (0.16 / 0.32 / 0.48 mm). Its error on the ball at 0.48 mm (|d| p95 0.06) shows the limit: the scale must stay well below the radius of curvature.
      - *The face quadric* sits at voxel centres (−0.5 voxel), and **over-reads tight convex curvature**: +0.12 /mm on the 1.28 mm ball at 0.48 mm (the mesh at 0.32 mm: 0.006). That is the crowns and blade tips, where the thickness trend's covariate is largest.
      - *The staircase left in curvature is in the label, not the method.* The plane's terraces (2.3 mm) leave 0.15 mm position error and a 4° normal error with every surface: no smoothing below the terrace length removes them. On the wavy body the 0.32 mm curvature error is the same ±0.1–0.2 /mm swirl for the binary contour, `white_mesh` σ 1 and twice 0.32 mm (`mesh_phantom_curvature.png`). The plan's "ripples of ±0.1–0.2 /mm at 0.32 mm" (*White-surface curvature*) are this aliasing.
    - **Seeding the march from a surface** (prototype `march_band`, a copy of `march` with per-seed start values, frozen seeds, and the root position carried like arc length without its source term; tested equal to `march` from point seeds, and exact (to 10⁻⁹) in arrival, arc length and root from an oblique plane at ratios 1 and 3). The seeds are the cortex voxels within 2.5 voxels of the surface plus every cortex voxel next to the WM label, at their metric distance from the tangent plane of the closest mesh point; the root is the metric foot point.
      - **The band must reach as deep as the stencil:** at ratio 3 Selling's offsets reach up to 2.45 voxels along the normal; with a 1-voxel band a plane's arrival was off by up to 1.4 voxels. Raw seeding has the same truncation (the stencil runs into the WM), which is part of its near-WM error.
      - **The inside/outside sign needs the pseudonormal** of the closest feature (edge or vertex; Bærentzen and Aanæs). With the face normal, staircase points on a 2× grid sat in a face's plane and got distance 0 (an exact test now covers it).
      - Limits: one label (no competition); the band uses the closest triangle's tangent plane; the metric along the *true* normal (the data normal's reliability at the WM edge is not tested here).
    - **At the top** (cortex 2.4 mm, top voxels within 0.16 mm of it; median signed error, IQR in brackets; root error tangential, median / p95; div = the divergence of the unit arrival gradient 0.24–0.48 mm above the WM, the carried-area integrand, IQR of its error in 1/mm). Wavy body:

      | Start | Arrival, ratio 1 | Arrival, ratio 3 | Arc length, ratio 3 | Root, ratio 1 | Root, ratio 3 | div IQR, ratio 1 / 3 |
      |---|---|---|---|---|---|---|
      | True surface (solver only) | +0.002 (0.010) | +0.001 (0.005) | +0.001 | 0.031 / 0.080 | 0.006 / 0.017 | 0.01 / 0.00 |
      | Raw voxels, one label | +0.053 (0.020) | +0.109 (0.038) | +0.070 | 0.086 / 0.258 | 0.022 / 0.054 | 0.99 / 2.41 |
      | Raw voxels, 1 mm patches | +0.089 (0.061) | +0.179 (0.114) | +0.078 | **0.362 / 0.798** | 0.116 / 0.309 | 1.31 / 2.78 |
      | Binary contour | −0.052 (0.020) | −0.029 (0.019) | −0.033 | 0.075 / 0.249 | 0.014 / 0.039 | 1.82 / 2.82 |
      | `white_mesh` σ 1 | −0.002 (0.021) | +0.002 (0.016) | +0.001 | 0.058 / 0.186 | 0.011 / 0.027 | 0.59 / 0.64 |
      | `white_mesh` σ 2 | +0.005 (0.026) | +0.005 (0.022) | +0.005 | 0.062 / 0.118 | 0.010 / 0.022 | 0.11 / 0.11 |
      | twice 0.16 mm | −0.013 (0.020) | −0.002 (0.017) | −0.004 | 0.060 / 0.227 | 0.013 / 0.035 | 1.09 / 1.44 |
      | **twice 0.32 mm** | **+0.000 (0.017)** | **+0.001 (0.013)** | +0.001 | 0.052 / 0.171 | 0.009 / 0.024 | **0.27 / 0.28** |
      | twice 0.48 mm | +0.002 (0.017) | +0.002 (0.012) | +0.002 | 0.047 / 0.116 | 0.008 / 0.020 | 0.09 / 0.10 |

      The ball (convex, R 1.28 mm), arrival at ratio 1 / ratio 3, 1× → 2× grid: true surface +0.049 → +0.024 / +0.015 → +0.007; raw +0.113 / +0.147; binary contour −0.001 / −0.010; `white_mesh` σ 1 +0.069 → +0.043 / +0.037 → +0.029; σ 2 +0.138 → +0.110 / +0.105 → +0.093; twice 0.32 mm +0.054 → +0.028 / +0.021 → +0.013. The 2-voxel blade matches the wavy body (twice 0.32 mm: +0.000 (0.015) and −0.000 (0.012); raw +0.053 / +0.108).
      - **Arrival and arc length: the voxel start costs a bias, a staircased start the opposite bias.** Raw seeds start half a voxel inside the WM: +0.05 to +0.15 mm at the top, more at ratio 3 (fronts leave the staircase across the normal at cost 3; arc length +0.07 against arrival +0.11). The binary contour reads 0.03–0.06 mm short: the distance to a rough surface is the distance to its nearest bumps. Volume smoothing adds its shrinkage on convex surfaces (σ 2: +0.10–0.14 mm on the ball). Mesh smoothing at 0.32–0.48 mm is unbiased within 0.002 mm on gentle surfaces and leaves a spread of 0.012–0.017 mm. **Solver against surface:** the true-surface start gives the solver's own error: within 0.002 mm (IQR 0.005–0.010) on the gentle shapes, +0.015 to +0.05 mm on the tight ball, and it halves on the 2× grid (first order). So on gentle white surface the start surface dominates at native resolution (raw: 0.05–0.15 mm against 0.002); on tight convex surface the two are comparable. A share of finding 22's near-constant overestimate (0.7–0.9 voxel from voxel ball sources) is the voxel start, not the solver [inference: on the R 8-voxel ball here, 0.71 voxel from raw seeds against 0.31 from the true surface].
      - **Root position: the metric decides more than the surface.** Along the true normal (ratio 3) every smooth start places the root within 0.01 mm (p95 0.02–0.03); raw seeds 0.02 / 0.05. Isotropic, the surface's normal error carries up the column: p95 0.26 mm raw, 0.12–0.17 smoothed, 0.08 for the solver alone; on the 4°-terraced plane every surface puts the root about 0.16 mm off (2.4 mm × tan 4°). **Root patches are the largest root error:** 0.36 / 0.80 mm isotropic, 0.12 / 0.31 at ratio 3, against 0.09 / 0.26 and 0.02 / 0.05 with one label: the per-label update confines each front to its own patch's tube (finding 23's arrival penalty, here in root position). The carried root position gives identity below the voxel scale without patches (open decision 4).
      - **Area near the WM is where the start matters most.** The carried-area integrand just above the WM has an error spread (IQR) of 1–2.4 /mm from raw seeds and 1.8–2.8 /mm from the binary contour, against true values of −0.03 to +1.2 /mm. It falls to 0.6 (`white_mesh` σ 1), 0.27 (twice 0.32 mm) and 0.1 (twice 0.48 mm, `white_mesh` σ 2). At ratio 3 the staircase noise runs up the columns as streaks through the whole thickness (`mesh_phantom_march.png`: IQR 1.5–2.0 /mm still at 0.5–1 mm from the WM with raw seeds). Its median is near zero, so this is noise, not the bias that finding 10 saw; it supports the suspected cause of carried area's mixed result only as far as noise can [inference].
    - **The start as boundary data** (the narrow band |d| ≤ 2 voxels around the true surface, on the 1× grid and on a 2× grid (80 µm) from the same 1× label; the signed distance to each start surface and its closest point, the foot, against the truth; mm). Wavy body; blades at 2×; triangles on the wavy body:

      | Start | Signed distance error, IQR (p95), 1× / 2× | Foot error median / tangential p95 | Wrong sign at 2× (wavy; blade 2 voxels) | Foot-map jump per step at 2×: p95; share > 2 steps | Blades 1.5 / 2 voxels at 2×: foot on the far side; inside read as outside | Triangles: min angle p5, aspect p95, dual-area CV |
      |---|---|---|---|---|---|---|
      | Voxel faces | 0.084 (0.076) / 0.075 (0.085) | 0.069 / 0.098 | 0.078; 0.110 | 3.00; 0.105 | 0; 0 / 0; 0 | — |
      | Binary contour | 0.074 (0.072) / 0.068 (0.078) | 0.063 / 0.107 | 0.064; 0.096 | 2.83; 0.093 | 0; 0 / 0; 0 | 35°, 1.5, 0.16 |
      | `white_mesh` σ 0.5 | 0.053 (0.055) / 0.052 (0.056) | 0.050 / 0.118 | 0.000; 0.000 | 2.59; 0.073 | 0; 0.157 / 0; 0 | 29°, 1.7, 0.19 |
      | `white_mesh` σ 1 | 0.022 (0.034) / 0.022 (0.033) | 0.020 / 0.040 | 0.000; 0.002 | 1.20; 0.000 | 0.024; **0.468** / 0; 0.007 | **12°, 3.2**, 0.30 |
      | `white_mesh` σ 1.5 | 0.016 (0.025) / 0.016 (0.025) | 0.012 / 0.021 | 0.000; **0.275** | 1.08; 0.000 | gone / **0.276; 0.937** | **7°, 5.1**, 0.32 |
      | twice 0.16 mm | 0.037 (0.048) / 0.037 (0.047) | 0.039 / 0.070 | 0.003; 0.002 | 1.47; 0.024 | 0; 0.004 / 0; 0.003 | 31°, 1.7, 0.20 |
      | **twice 0.32 mm** | **0.013 (0.028) / 0.013 (0.027)** | **0.010 / 0.024** | **0; 0** | **1.05; 0** | **0; 0 / 0; 0** | 32°, 1.6, 0.19 |
      | twice 0.48 mm | 0.010 (0.018) / 0.010 (0.018) | 0.006 / 0.010 | 0; 0 | 1.03; 0 | 0; 0 / 0; 0 | 32°, 1.6, 0.19 |
      | Implicit σ 0.5 | 0.047 (0.055) / 0.047 (0.054) | 0.048 / 0.102 | 0.004; 0.002 | 2.38; 0.065 | 0; 0 / 0; 0 | — |
      | Implicit σ 1 | 0.021 (0.034) / 0.022 (0.034) | 0.019 / 0.037 | 0; 0 | 1.12; 0 | 0.001; **0.278** / 0; 0 | — |

      - **A refined grid does not remove the voxel-plus-interpolation error: that error is in the boundary data.** Every staircased start (voxel faces, binary contour, `white_mesh` σ 0.5, implicit σ 0.5) has a signed-distance spread of 0.05–0.08 mm on both grids, and on the 2× grid makes the foot map jump by more than 2 steps at 7–11 % of neighbouring pairs (p95 2.4–3.0 steps against about 1 for a smooth surface). The label's own faces and the binary contour also give 6–11 % of the 2× band the wrong sign (points between the true surface and the label's half-voxel steps). Starting a level set on a 2× grid from the label's own signed distance carries all of that. A start smoothed along the mesh at 0.32 mm has no wrong signs and a foot-map jump p95 of 1.05 on both grids, and 0.013 mm spread.
      - **The implicit route buys nothing over the mesh here.** Its zero set is the smoothed label's 0.5 level, so it inherits volume smoothing's errors: staircase at σ 0.5, lost thin blades at σ 1 (28 % of a 1.5-voxel blade's interior read as outside), shrinkage of convex surface (ball: +0.12 voxel at σ 1). Its second-order interpolation does not matter while the data are quantized to half a voxel. A level-set function from the image itself was not tried.
      - **Thin blades at 2×:** with twicing at 0.16–0.48 mm no foot lands on the far side of a 1.5–4-voxel blade and no interior point reads as outside; volume smoothing at σ 1.5 puts 28 % of the feet beside a 2-voxel blade on its far side. At the 2-voxel fin's tip a few feet cross (0.5 % at 0.32 mm). On V1V2's thin WM (cortex within 2 voxels of it, 1× and 2×) twicing at 0.32 mm reads 1.4 % / 4.9 % of the points as inside, all within 0.16 voxel of the surface and at points with twice as much WM around them as the rest (concave corners the smoothing slightly fills, not a side swap); `white_mesh` σ 1 and σ 1.5 put the foot more than 2 voxels away for 11–14 % and 31–32 % of them (the blade gone); the label's own faces and the binary contour are two-sided by construction.
      - **Triangles** (for a finite-element thickness field, or a moving mesh): marching tetrahedra on a volume-smoothed label make slivers (σ 1: minimum angle 5th percentile 12°, aspect 95th percentile 3.2; σ 2: 6°, 6.4; on the ball 3° and 11). The binary contour, relaxed and smoothed along the mesh, keeps 31–38° and 1.4–2.1 at every scale, with a dual-area coefficient of variation of 0.18–0.35. The relaxation is the remeshing that matters at this density; edge splits and collapses were not needed.
    - **Surface error against solver error at the top, stated explicitly** (true-surface start = solver alone; arrival, ratio 3, 1× → 2×): ball +0.015 → +0.007 mm (solver) against +0.021 → +0.013 mm from twice 0.32 mm, so the surface adds a constant +0.006 mm that the finer grid does not touch; wavy spread 0.005 → 0.002 (solver) against 0.013 → 0.012. At 1× the two are comparable on tight convex surface and the surface dominates on gentle surface; **at 2× the start surface is the larger error everywhere** (2–6×), and with raw voxel seeds it is larger by an order of magnitude (+0.05 to +0.15 mm). An O(h²) start is worth having on a 2× grid only if the start itself is sub-voxel accurate, which the label is not (its half-voxel steps); twicing at 0.32 mm is the cheapest start that reaches the solver's level.
    - **The root map near the WM, and false collisions** (6-neighbour pairs within 1 mm of the WM; a "false collision" voxel has a neighbour whose carried root is more than 1 mm away, Euclidean, a lower bound on the distance along the white surface):
      - *Phantoms (no true collisions):* no start made a single false-collision voxel near the WM; 1 mm root patches made 48–143 in the whole cortex. The root map's jump per voxel step near the WM, p99.9 (voxels): true surface 1.0–1.3; twice 0.32 mm 1.0–1.6 (2× grid 1.2–1.9); raw seeds 1.4–2.1; binary contour 2.4–2.9 (2× grid 4.5–4.9); root patches 2.4–4.4. The staircase makes the root map jump by 2–5 voxels (0.3–0.8 mm) between neighbours, below the 1 mm rule.
      - *Real patches* (whole boxes; isotropic and ratio 3 along the data's normal; raw seeds / `white_mesh` σ 1 / twice 0.32 mm): near the WM 2 / 0 / 0 and 68 / 131 / 131 voxels in M1S1, 199 / 484 / 215 and 360 / 793 / 413 in V1V2, 2 / 0 / 0 and 56 / 54 / 57 in PCC_SPL (at most 0.2 % of the cortex within 1 mm of the WM); over all cortex 27,000 (M1S1), 33,000–39,000 (V1V2) and 6,700–7,100 (PCC_SPL), within 4 % of each other across starts. Jump p95 near the WM 1.40–1.45 voxels per step from raw seeds against 1.22–1.34 from the meshes. **By this rule the staircase makes almost no collisions:** the M1S1 collision set is the same with every start. The plan's 63,595 sheet-edge voxels in M1S1 come from another rule (roots merging, finding 11) and were not recounted.
    - **The real patches** (whole boxes of `baillarger.data.PATCHES`; at ≥ 2 mm from the faces). Thin WM = WM outside the opening of the label by a ball of radius 2 voxels, in pieces of ≥ 50 voxels; covered = a mesh vertex within 1 voxel of an exposed face, with a normal on the same side.

      | | V1V2 | M1S1 | PCC_SPL |
      |---|---|---|---|
      | Thin WM: voxels, pieces, half-width median / p90 | 7,108, 14, 1.0 / 1.4 voxels | none | 58, 1 |
      | Thin WM lost (`white_mesh` σ 0.5 / 1 / 1.5 / 2) | 0 / 16 / 43 / 75 % of voxels; pieces ≥ 95 % covered 14 / 10 / 1 / 1 of 14 | — | 0 / 48 / 93 / 100 % |
      | Mesh smoothing: thin faces covered (0.16 / 0.32 / 0.48 mm) | 0.998 / 0.992 / 0.941; pieces ≥ 95 %: 14 / 14 / 5 | — | 1 / 1 / 0.99 |
      | Self-intersecting face pairs | 0 for all except twice 0.48 mm (36, all near thin WM) and σ 0.5 + twice 0.32 mm (26, all near thin WM) | 0 | 0 |
      | Curvature against the face quadric (Spearman, 0.32 / 0.48 mm); median \|ΔH\| at 0.48 | 0.81–0.89 / 0.85–0.88; 0.03–0.06 /mm | 0.88–0.95 / 0.91–0.98; 0.02–0.04 | 0.87–0.94 / 0.89–0.98; 0.02–0.04 |
      | Curvature jitter at 0.32 / 0.48 mm (median \|H − local mean\|, 1/mm): quadric; binary contour; twice 0.32 mm; `white_mesh` σ 1 | 0.029 / 0.010; 0.014 / 0.004; 0.012 / 0.004; 0.017 / 0.006 | 0.026 / 0.007; 0.013 / 0.003; 0.011 / 0.003; 0.015 / 0.005 | 0.025 / 0.007; 0.012 / 0.003; 0.010 / 0.003; 0.015 / 0.005 |
      | Image's WM edge (steepest step of log echo 2, trend removed) from the surface, p25 / 50 / 75 | −0.11 / 0.00 / +0.06 mm | −0.03 / +0.03 / +0.10 | −0.02 / +0.05 / +0.10 |

      - In V1V2, `white_mesh` at its default σ 1 already thins or loses parts of 4 of the 14 thin pieces; at σ ≥ 1.5 most are gone (`mesh_V1V2_sections.png`: the blade cores' lines shorten and vanish). Mesh smoothing at 0.32 mm keeps all 14 with no crossings; at 0.48 mm the two sides of 2-voxel blades start to cross.
      - The meshes' curvature agrees with the quadric (rank 0.85–0.98) and has about half its jitter. Pearson's r in V1V2 was unstable (0.00–0.69) because of a few extreme values (|H| > 3 /mm at 0.1–0.2 % of points, on one-voxel WM fragments); ranks are not.
      - The image's own WM edge lies within about 0.05 mm of every surface (median), with the same spread (IQR 0.13–0.18 mm) whichever surface: the label limits it, not the mesh (`mesh_<patch>_edge.png`).
    - **Reading.**
      - *Mesh, then smooth on the mesh.* Contour the label itself, relax, smooth along the mesh with twicing at about 0.32 mm. It keeps thin blades and both their sides, has no position bias on gentle surface and no crossings on the three patches, retracts a 2-voxel tip by 0.08 mm, and cuts the near-WM noise 4–9-fold against raw seeds. 0.48 mm halves the near-WM noise again but crosses the sides of V1's thinnest blades and retracts tips by 0.14 mm; 0.16 mm leaves most of the staircase. Volume smoothing (`white_mesh`) is the wrong tool where blades are 2 voxels thick: at σ 1 it already loses part of them.
      - *What a smoother start buys at the top:* arrival and arc length lose a 0.05–0.15 mm bias (raw) and become as good as the solver on gentle surface; root position gains little where the metric follows the layers, and some where it is isotropic; area near the WM gains most.
      - *What it does not buy:* curvature below the label's own terrace length (about 1–2 mm on near-grid-aligned surface), and the label's own position against the image (about 0.05 mm, spread 0.15 mm).
      - *For a level set on a 2× grid* (the route now preferred): initialize its signed distance and foot points (the root map R) from the mesh smoothed at 0.32 mm, not from the label's own distance; that keeps the sign and the foot two-sided at thin blades and the foot map continuous, and leaves the 2× grid's gain to the solver.
    - **Package changes this would need** (described, not made):
      - For any front (fast marching or a level set): a narrow-band start from a mesh, giving signed distance, foot point (root) and arc length in a band at least as deep as the scheme's stencil, with the pseudonormal sign; on a 2× grid, from the 1× label's mesh.
      - `baillarger.fronts.march`: per-seed start values (arrival and arc length per seed voxel, the seeds frozen), and the root position carried with arc length (three more transported numbers per voxel; the prototype's cost is the same order). `fronts` stays self-contained: the caller passes arrays.
      - `baillarger.frame` (or `mesh`): the narrow-band start above (closest point with pseudonormal sign, metric distance from the tangent plane, band ≥ the stencil's reach); a mesh smoother (tangential relaxation and twicing); and `white_mesh` faces oriented outward (they now point into the WM while its normals point out, which inverts any signed distance taken from the faces).
      - The thickness field on mesh vertices instead of pooled WM-boundary voxels: the mesh's edges as the graph (the two sides of a blade are never joined, by construction), `mesh_curvatures` for the trend's covariate, and each voxel's belief read at its carried root. With carried roots the competition's labels could be root patches on the mesh, or none (open decision 4); untested here.
    - **Views:** `mesh_phantom_blades.png`, `mesh_phantom_fins.png` (sections of every start surface against the true one), `mesh_phantom_curvature.png` (curvature error over the wavy body), `mesh_phantom_march.png` (arrival error and the carried-area integrand in a plane, raw / true / twice 0.16 mm, ratio 3); `mesh_<patch>_sections.png` (every mesh in the plane with most thin WM, over echo 2 trend removed) and `mesh_<patch>_edge.png` (profiles and the edge's offset) for V1V2, M1S1, PCC_SPL. In ITK-SNAP: `mesh_<patch>_thin_overlay.nii.gz` with `mesh_thin_labels.txt` (thin WM by the `white_mesh` σ at which it is lost) over `fronts_eval_<patch>_echo2_detrended.nii.gz`; meshes `mesh_<patch>_white_mesh_s1.vtk` and `mesh_<patch>_twice_0p32mm.vtk` (mean curvature at 0.48 mm). Numbers: `mesh_phantoms.json`, `mesh_phantoms_tables.md`, `mesh_real_<patch>.json`, `mesh_real_band_<patch>.json` in `prototypes/data`.

29. **Level-set filling in normalized time: within a voxel on phantoms and at both PCC pieces, 0.18 mm at M1/S1, off at the V1V2 banks because of the evidence; the volume form fails on a biased carried area** (`levelset_fill.py` and its tests (41, plus 1 strict xfail), `levelset_synth.py`, `levelset_face.py`, `levelset_v1.py`; prototype only; 1× grid, all 3D; full results in *Level-set filling test plan*, Results).
    - **The method.** The fronts start from finding 28's mesh in a band at exact signed distance and advance along the data's layer normal, at rate μ(R) (length form) or v(R)/a (volume form). The seam lies where the two sides arrive at equal τ. The sides are separated by sign agreement of the directions.
      - The march has two phases. Phase 1, ordered by arc length, fixes the geometry. Phase 2, ordered by τ, holds the competition. With one τ-ordered phase the budgets changed the visible tops; now 98–99 % of evidence vertices are identical across budget settings.
      - The low-coherence blend goes toward ∇s, not ∇τ.
    - **Phantoms (voxels):**
      - slab exact; crown and annulus level sets within 0.21;
      - P4 and P8 exact with the oracle budgets, and +2.40 / −0.20 with open 12/18, as the rule predicts;
      - the mesh field end to end on P4 gives +2.42.
      - **Sign agreement pins a seam to a sharp flip of the data normal, whatever the budgets:** 0.05 / 0.27 / 1.15 voxels from τ_A = τ_B at 20° / 45° / 90°.
      - At 90° with no coherence at the bisector, the two fronts merge and no seam forms (a known limitation).
    - **Faces, length form, median |offset|:**

      | Face | Equal | Prior | Posterior |
      |---|---|---|---|
      | M1/S1 | 0.63 | 0.39 | 0.18 mm (39 % within 0.16 mm) |
      | V1V2 banks | 0.35 | 0.24 | 0.40 |
      | PCC piece 1 | 0.17 | 0.09 | 0.06 (95 %) |
      | PCC piece 2 | 0.15 | 0.05 | 0.05 (78 %) |

      This beats the one pass (finding 25) at three faces, and the walls at M1/S1 and PCC piece 2.
    - **V1V2 fails in the budgets, not the competition.**
      - 7.5 % of its tops follow d sideways about twice the straight distance to fluid (s ≈ 5 mm). They cluster, and the field (Gaussian noise) builds patches of 4.5–5.7 mm budgets from them. Points with a root in a patch go from within 0.06 mm with the prior to 0.8–0.9 mm with the posterior.
      - Separately, 16 of its 35 points have their two roots under 5 mm apart along the mesh, so both budgets come from one neighbourhood. Those points miss by 0.49 mm against 0.20.
    - **Volume form: failing.**
      - The carried area a from the divergence falls to 0.73 at 2–3 mm, because the data normal's noise accumulates. a from ∇R degenerates.
      - Its budget field has a curvature coefficient of 1.76 (thickness 0.12) and a held-out z SD of 2.89 (thickness 1.15; the old voxel-graph field 0.90).
      - At M1/S1 it ties the median (0.19 mm) with three times the spread.
      - Whether v removes the curvature trend cannot be judged until a is unbiased.
    - **Open V1:** the stria is not at constant τ. The length form gives τ 0.387, IQR 0.051, ρ +0.61 against curvature, as distance does. The volume form gives 0.402, IQR 0.094, ρ −0.79, over-correcting like finding 10's equivolume.
    - **For a package:** the length form, the band start, the two-phase march, seams as data, and the field on mesh vertices, with a guard against wandering tops (a ratio of path length to straight distance, or a robust likelihood). Close roots are an open problem.
    - **Deferred:** α = 1, the τ = 1 stop, convergence, topology at a face, sulcal depth, covariate ranges.

## The findings against theory and prior work

- **FreeSurfer's collision is the equal-thickness case.** FreeSurfer grows the pial surface from the white surface under image forces and a thickness limit, and fused banks stop where they collide. Without visible fluid that is equal arrival, which is right only when the banks are equally thick. All tested banks were near-equal at the contact, so this is untested on a strong asymmetry.
- **DELFMAP and CRUISE weight the rate by local evidence, which is complementary.** Both place the seam where weighted-distance fronts meet, slowing fronts in dark (CSF-like) voxels; DELFMAP finds collisions as ridges of its field (|∇d| < 0.8), as we now do. Thickness sets the expected meeting; darkness, where present, pulls it onto the observed line.
- **Shock theory says how many fronts meet.** In a generic arrival field, two fronts meet on sheets, three along curves, and four at points, and sheets can end where a front meets itself (Blum; Kimia; Giblin). A pairwise contact rule is well-posed on sheets only; the weighted competition of fronts is the general form.
- **Normalized arrival is a normalized depth coordinate with an estimated outer surface.** Laplace and equivolume frames (Jones 2000; Waehnert 2014; LayNii) run from 0 at the white surface to 1 at a known pia. Our contact is where both fronts reach their own 1 at once, with each 1 set by an expected thickness rather than a known surface.
- **Carried thickness is a thickness prior, as in mature layered segmentation.** OCT methods learn per-column thickness bounds (Garvin 2009) or impose soft priors between neighbours (Song 2013, error 6.30 → 5.14 µm). The constraint inventory listed thickness continuity as a missing constraint.
- **No tool places a fused contact from expected thickness.** FreeSurfer, DELFMAP, CRUISE/ACE, CAT12's PBT and ANTs DiReCT all split fused banks symmetrically, with no uncertainty. Their thickness priors are upper limits. DiReCT's spatially varying limit (KellyKapowski's thickness-prior image) is the closest relative and a possible baseline. PBT's full text was not read; its abstract claims it handles sulcal asymmetries.
- **Geostatistics already solves the split.** COHIBA (Norwegian Computing Center; Abrahamsen 2005) builds the layers between two reflectors from both ends and combines them by inverse-variance weights. With Gaussian thickness beliefs and the gap fixed, the contact's expected position is μ_A + σ_A²/(σ_A² + σ_B²) · (gap − μ_A − μ_B). The sharing rule depends on how variance scales with thickness:

  | Variance law | Share of the misfit to the thicker bank (M1 vs S1) |
  |---|---|
  | Constant | 0.50 (the midline rule's split) |
  | Proportional to thickness | 0.60 (exactly normalized arrival) |
  | Proportional to thickness² | 0.70 |

  Carrying thickness sideways with uncertainty is Bayesian kriging (Abrahamsen 1993). Inequality-constrained kriging gives proper posteriors for thickness ≥ 0 and for a contact inside the gap.
- **Probabilistic fields on surfaces.** The model is a Matérn/SPDE Gaussian field (Lindgren–Rue), with a level term, mesh-aware weights and second-order smoothness. Layered segmentation uses soft thickness priors (Song 2013) and, rarely, full posteriors (Rathke 2014); none calibrates its intervals.
- **Equivolume and the Steiner formula.** Along white-surface normals, the divergence of the column direction is exactly κ₁/(1 + κ₁s) + κ₂/(1 + κ₂s), and integrating it gives the closed form. The two routes agree only if the columns' cross-section follows parallel surfaces. They don't: both over-correct on open V1 (finding 10), as expected when the pial surface is flatter than the white surface. Waehnert's equivolume (2014) uses the areas at both surfaces for this reason.
- **Seismic analogue.** RGT time is ordinal and each stack can carry its own time scale. Two stacks meeting at a contact resemble two depositional sequences meeting at a bounding surface; neither sequence's time continues into the other.

## Open questions

- **Why does carried area over-correct on open V1?** Suspected: T near the WM follows the voxel WM shape, not the layers. On phantoms a voxel start makes the carried-area integrand just above the WM noisy (error IQR 1–2.4 /mm against true values up to 1.2 /mm), and the noise runs up the columns at ratio 3; a start smoothed along the mesh at 0.32 mm cuts it to 0.27 /mm (finding 28). Its median stays near zero, so the staircase explains noise, not a bias; whether it explains the over-correction is the re-score of Next step 2 from a smoothed start. If not that, the two-surface rule is next.
- **Curvature staircase.** Mostly answered (finding 28): the ripples are the label's own terraces, and a sub-voxel mesh keeps them (the same ±0.1–0.2 /mm pattern at 0.32 mm from every surface; a 4°-tilted plane keeps a 4° normal error). Smoothing along the mesh halves the jitter against the face quadric and fixes its over-reading of tight convex curvature. Open: whether the thickness trend needs curvature below about 0.5 mm at all; if not, 0.48 mm curvature from the 0.32 mm mesh is enough.
- **Open against fused thickness.** Carried thickness overpredicted the kissing part by about 0.2–0.25 mm. Is that geometric (absorbed by equivolume), caused by mixed bank labels, or real (needing a trend or region term)?
- **The unequal kissing stretch** (finding 12): about 2.0 against 3.3 mm. Which areas are the two sides? The posterior placed it within about a voxel; is that because the thickness field's evidence is near on both sides?
- **Label errors make false sheets.** At piece 3, a thin Fomblin gap labelled cortex probably joined two folded surfaces (finding 21). Can hidden gaps be found from the image (a thin, slowly decaying dark strip inside labelled cortex) and opened before the fronts are grown? And should the T = 0 boundary come from the image's own WM contrast (the constraint inventory's contrast-defined deep anchor)?
- **Which sheets are contacts between two stacks?** A sheet also forms where a front meets itself round a tight bend, with its roots more than 3 mm apart along the white surface (`PCC_SPL_p3`, by the user's reading). Three local tests did not separate them (finding 20). PCC piece 1's failure is a different problem: its sides were joined when the roots were grouped into pieces (finding 21). Candidate tests, to be checked against the user's readings of the marked faces:
  - the angle between the data's layer normal and the sheet (perpendicular at a contact, in-plane at a false sheet);
  - whether the sheet starts at the pial surface (false) or meets fluid only at its edges (contact);
  - the profile across it (the doubled superficial bands of finding 17).
- **Is the pairwise posterior enough?** Superseded: the weighted competition in one pass is built and tested (findings 23–25); it is well-posed at junctions and edges too.
- **What makes a label boundary a contact?** Separation along the white surface needs a threshold, and none works everywhere (finding 25). The fold signature, both fronts short of their thickness at the meeting (finding 24), is a candidate class for reporting.
- **How much should the data steer arrival?** At the V1V2 banks the data-steered metric moves the contact by about 0.3 mm against plain distance; at M1/S1 it makes no difference (finding 25). The PDE framing points to regularising the metric where coherence is low rather than choosing one ratio.
- **Which condition holds on the contact?** Equal z-scores (σ, as run) or the posterior (σ²). The choice should follow a stated principle; at M1/S1 it moves the share to the thicker bank from 0.60 to 0.70 when σ is proportional to μ.
- **Is crystal growth's within-bank leakage large on real data?** Untested: compare each scored point's root traced back in the unweighted and the weighted pass, and the μ at each.
- **Is there literature on a meeting condition set by expected thickness?** None found in a first search (*The fronts as PDE problems*); the closest are crystal-growth Voronoi diagrams and competing-front segmentation.
- **WM edges.** The data normal is unreliable right at a WM edge, where columns start. The label's surface itself lies within about 0.05 mm of the image's own WM edge in echo 2 (median; spread 0.13–0.18 mm, the same for every mesh, finding 28), so the start's position is limited by the label, not the mesh. The surface start was tested with the metric along the true normal only; with the data normal at the WM edge it is untested.
- **Landmarks outside V1.** A crude darkest-point landmark failed in M1S1.
- **Band identity beyond the stria,** the V1/V2 border (the stria should end while T stays continuous), pinch-outs (layer IV between S1 and M1), and regions.
- **Scaling** from stretches to whole patches, and to PCC_SPL.

## Next steps

### 1. Per-front arrival at every sheet

- **Done** (findings 11–13): the collision structure from all-WM fronts; walls from roots, sorted into two sides; per-front fields at the kissing stretch; thickness field with a curvature trend; the pairwise posterior and scoring at the 23 points (the user read it as following the contact); held-out calibration with and without the trend; the check that the stretch is unequal, by column length; arc length in place of T, and the re-score in it.
- **Next:**
  1. **The diagnostics in *The contact as an inference problem*:** bank position, A6, A10 and A5 are done (findings 14–17). Only A9 (error correlation) remains. The picture now:
     - the geometry, the evidence and the contact's zero thickness all hold;
     - the field reverts across the evidence-free gap before the contact roots, where the sums t₁ + t₂ = L are the missing evidence.

     The model is still unchanged. The next model step to discuss is a joint fit of the field to the visible tops and to the sums L.
  2. **Extend per-front arrival to every sheet in the box** (one pair of walls per sheet piece), so contacts are found everywhere fronts meet.
  3. **Tune the collision thresholds** (1 mm same front, 3 mm far), and look at where sheet edges sit.
  4. **Test the trend where it matters:** hold out whole fused-like patches far from evidence, not scattered visible tops.

### 0. One-pass weighted competition (the new representation)

- **Decided (user):** contacts come from one anisotropic fast-marching pass from all WM. Each voxel carries its arrival, its root and the root's weight; the competition is "crystal growth" (a front cannot enter territory already claimed). There are no sides, walls or sheet classes (*Collisions and contacts: the review and what it points to*).
- **The solver is a self-contained module, `baillarger.fronts`.** It imports nothing else from `baillarger`, so it can be broken out as a package later. Built in two PRs, both merged.
  - *PR #25:*
    - `march` gives the arrival from labelled sources under a Riemannian metric, with each voxel's label, Euclidean arc length, the best arrival of another label, and the acceptance order. Its stencils are adaptive, from Selling's decomposition (Mirebeau's scheme), with labels updated separately and a line-of-sight check for walls.
    - Also `riemann_from_direction` and `selling`.
    - Tests: exact plane sources, point sources under isotropic, constant anisotropic and manufactured metrics, convergence, bisector labels, connected regions, arc length (exact from oblique planes, converging from curved sources), walls including diagonal ones, symmetry and input checks.
    - Measured limits (in `march`'s docstring): first order; 4–8% arrival error from a point source; direction error about 1° at anisotropy 3–10 and 5–6° near isotropy; arc length exact from flat sources but first order with a large error on curved fronts under strong anisotropy (about 35% from a 6-voxel ball at anisotropy 10, at 41³). That error is the worst case, with paths crossing the metric's direction. Where paths follow it, as columns do in cortex, arc length is within about 1.3 voxels at anisotropy 10 and halves on a 2× grid (finding 22).
    - Cost: 141³ in about 9 s and a 2.6 GB peak.
    - *PR #27:* `benchmarks/fronts_accuracy.py` prints these numbers against exact answers (Selling; point-source arrival and direction; a manufactured metric; arc length from plane, point and ball sources under refinement; the 141³ run). Rerun it after a solver change and compare.
  - *PR #26:* per-source `start_time` and `speed`, indexed by label (defaults 0 and 1). A label's update uses right-hand side 1/speed², so arrival = start + (metric distance)/speed; the competition is crystal growth; arc length is unchanged by both weights. Tests and the self-review's fix: finding 23.
- **Done since** (findings 23–26):
  - the one-pass prototype `onepass_lib.py`, with exact-answer tests (`test_onepass_lib.py`, 28 tests): root patches as labels, contacts by separation along the white surface, column tops as evidence, the pure weighted Voronoi partition as a reference, and a local two-sides score at marked points;
  - the synthetic contacts (U-slabs, P4, P8, omega): the boundary lands where the weights put it, and a fold shows as both fronts short of their thickness (finding 24);
  - the four marked faces: M1/S1 reproduces finding 13 without walls; PCC piece 1, where the walls failed, is within 0.08 mm; PCC piece 2 is as before; the V1V2 banks miss by about 0.3 mm at cost ratio 3 and are within 0.16 mm at ratio 1 (finding 25);
  - weights from the field need two guards: no belief where the field's graph piece has no evidence, and no evidence from roots near a box face (finding 26);
  - the start surface (finding 28): a mesh of the label smoothed along its edges (twicing, about 0.32 mm), seeded as a narrow band at least as deep as the stencil, removes the voxel start's bias; the package changes it needs are listed there (per-seed start values and the carried root in `march`; the band, the smoother and outward faces in `frame`; the thickness field on mesh vertices).
- **Open decisions (for the user):**
  1. *The cost ratio across the layer normal* (role 3, arrival). Ratio 3 misleads where coherence at the contact is low (V1V2 banks); ratio 1 is plain distance. Candidates: a ratio that falls with coherence, or distance for arrival with the data used only for depth within a column.
  2. *The contact set's separation threshold.* No value is principled (finding 25). Scoring no longer needs one. The set itself might be classed by d/μ on the two sides instead (finding 24's fold signature), or by how each piece is born (rim or interior).
  3. *The rule's tail:* past every root's expected thickness the most uncertain root wins (finding 26). Bound it, or accept it?
  4. *Root identity for vertex correspondence:* the patches give identity at about 1 mm only. Carrying each voxel's root position through the pass, as arc length is carried (sum ρ_j a_j (X − X_j) = 0), gives a continuous map from the top to the white surface without per-voxel labels. Prototyped in `mesh_smoothing.march_band` (finding 28): exact from a plane, within 0.01 mm (p95 0.02–0.03) of the true foot at the top when the metric follows the layers, against 0.12 / 0.31 mm with 1 mm root patches. A solver change (a PR 3), not started in the package.
- **The package on complete patches** (finding 27): each pass under 10 s and all cortex reached; the contact set is dominated by box faces and by overlap-like boundaries (label errors), and the second-best arrival is usually a neighbouring patch. Not fixed; candidate changes: a free boundary or face flag, the best arrival of a far label, a warning on extreme speeds, and the root position carried through the pass.
- **Next:** the user's decisions above; then the face handling and the second-best arrival in the package (one PR each); then the joint fit of the thickness field to the visible tops and the sums L.
- **Memory:** heavy runs go through `scripts/memguard <GB> <command>` (tracked, PR #28; the same as `prototypes/scripts/memguard.sh`) one at a time. The cap guards against runaway code, not a compute budget: about 1.5× the run's measured or expected peak (a 2× grid of a 141³ box, about 21 GB, gets about 32 GB). The watchdog runs its command in the background, so scripts must be files, not heredocs.

### 1a. More kissing faces outside the M1/S1 stretch (for marking)

Before any change to the contact model, the method is to be tested on other kissing faces in this brain (and later in other brains), so that nothing rests on one sulcus. Crops loaded from the full volume, with the all-WM collision structure and walls from roots, are ready in `prototypes/data` for the user to mark contact points (`site_survey.py`; template `sites_points_template.json`):

| Site (files `<site>_*`) | Full-grid corner, size | Largest sheet piece (ITK-SNAP, crop numbering) | Open thickness, side 1 / 2 | Contact roots beyond the nearest top, side 1 / 2 |
|---|---|---|---|---|
| `V1V2_banks` (the fused V1 banks of finding 5) | (198, 58, 65), 141³ | 8,197 voxels at (78, 67, 73), 5 × 12 × 17 mm | 2.37 / 2.51 mm | 0.9 / 0.4 mm |
| `V1V2_collapsed` (the review points' collapsed sulcus) | (212, 30, 105), 141³ | the same face as `V1V2_banks` | as above | 1.3 / 0.5 mm |
| `V1V2_thin` (the thin space with a bridge) | (225, 16, 197), 141³ | 9,688 voxels at (46, 100, 86), but only 126 facing pairs: mostly a thin space | 2.46 / 2.55 mm | 0.4 / 0 mm |
| `PCC_SPL` | (427, 93, 335), 175³ | piece 1: 4,568 voxels at (67, 62, 85); piece 2: 906 at (119, 117, 63); piece 3: 723 at (95, 111, 126); piece 4: 539 at (101, 60, 54) | 2.54 / 2.69; 2.81 / 2.31; 2.19 / 3.02; 2.84 / 2.64 mm | 0.6 / 0.4; 0.3 / 0.3; 0.2 / 0.4; 0 / 0.2 mm |

- **The gap before the contact is short at these sites:** 0–0.6 mm, against 1–2.5 mm at the M1/S1 stretch. The exception is the V1V2 banks, at about 1 mm on one side. If finding 16's reading is right, the field's drift should matter less here. That is a test of the reading, not only of the model.
- **Marked so far:**
  - `V1V2_banks`: 35 points (finding 18).
  - `PCC_SPL` piece 1: 22 points. The user saw no contact on the lower part of its sheet (j about 39–70, k about 70–90).
  - `PCC_SPL_p2`: 37 points, in a crop re-centred on the face (`PCC_SPL_p2_points.json`).
- **A false sheet:** `PCC_SPL_p3` (piece 3, re-centred) is not a contact. By the user's reading, the cortex bends tightly there, with at most minimal kissing, and the sheet cuts inwards from the surface through the middle of the cortex. This is a front meeting itself round a tight bend: a sheet edge in the taxonomy. But its two sides' roots are more than 3 mm apart along the white surface, so the distance rule classed it as a sheet. Distance along the white surface cannot separate one stack bent back on itself from two stacks, so sheets need another test before they are trusted unmarked (open question).
- **Then:** replicate the current pipeline at each marked face (per-front arc length, the pairwise rule and the error breakdown), and judge the joint fit on all of them. To run a face: `site_setup.py <data> <work> <run> <points file> [--piece N] [--centre]`, then `collide.py`, `collide2.py`, `post_error.py` and `collide2_itksnap.py` with `SITE=<run> POINTS=<run>_points_box.json GROW=0 CLOSURE=none`.

### 2. Carried area for depth within a column

- **The test:** the role-1 re-score of finding 10 (open V1; banks A and B), with the divergence of the column field taken from the layers just above the WM rather than from T's pinned level sets (for example, from the data normal where coherence is high, or with T's near-WM part released from the voxel WM shape). Start the fronts from the white mesh smoothed along its edges (0.32 mm), not the voxels: on phantoms the voxel start alone puts 1–2.4 /mm of noise into the integrand just above the WM (finding 28).
- **What to look at:** the stria fraction against root curvature and its spread, against distance and the current carried area.

## Later tests

- **The weighted competition of fronts** with a multi-label front solver, if junctions matter.
- **A genuinely unequal contact** (M1 against S1), well away from the volume edge.
- **Contrast-weighted coupling** in the thickness field, and the KellyKapowski baseline.
- **Synthetic, as checks on the fit:**
  - reversion to the background metric where data are absent;
  - a pinch-out against a band that is present but invisible;
  - contact placement with known unequal banks and known thickness gradients.
- **On real data:**
  - a strongly curved stretch, crown or fundus;
  - the V1/V2 border;
  - then whole patches.

## Future work: region-specific thickness priors

Not yet available; recorded so the thickness field is built to accept them later. The idea comes from facies-controlled modelling in geostatistics (from general knowledge, not yet verified against sources such as Pyrcz & Deutsch or Armstrong et al. on pluri-Gaussian simulation).
- **Per-region priors.** Each area gets its own expected thickness, uncertainty and correlation length, so where evidence is absent the field falls back toward that area's typical thickness rather than a global one.
- **Soft regions, not hard cuts.** Each column gets probabilities over areas, not a hard label. Hard "cookie-cutter" cuts make thickness jump at borders and pass boundary errors straight into the map, against the no-faults rule. Truncated or pluri-Gaussian models give gradual borders and respect the natural order of areas (M1 → 3a → 3b).
- **Mixtures near borders.** A column uncertain between two areas has a mixed, possibly bimodal, thickness belief. The per-column contact posterior can carry it.
- **Thickness and pattern need not change together.** Laminar contrast should weaken thickness coupling, not cut it, so the data can show a border with no thickness change.

## Code, data and figures

All prototype code and its inputs and outputs are in `prototypes/` at the top of the repository (untracked; not part of the package):
- `prototypes/scripts/` holds the scripts.
- `prototypes/data/` holds the crops for ITK-SNAP, their overlays and label files, and the user's points.
- `prototypes/figures/` holds every figure referred to below.

The source images come from `data/` through `baillarger.data` (see DATA.md).

**Running a script.** The scripts import each other and read their inputs from one working directory, passed as an argument. So copy scripts and data into one directory, and run from the repository root:

```
mkdir -p /tmp/proto && cp prototypes/scripts/* prototypes/data/* /tmp/proto/
nix develop --command uv run --with matplotlib python /tmp/proto/<script>.py <arguments>
```

Most scripts take the output directory and the working directory, for example `fit_T.py fused /tmp/proto /tmp/proto --equivolume`. `prob_thickness.py` takes the working directory only, plus `--distance`. Stretch coordinates are hard-coded near the top of each script. Figures without `_e2` in their name come from earlier versions of the same scripts, which displayed echo 3.

**Points the user marked** (ITK-SNAP numbering is the 0-based value plus one on each axis):

| Data | Where it is |
|---|---|
| Open V1 stretch (V1V2 crop, slice k = 54) | `flatten_checkpoint.py` (P1, P2) |
| Fused stretch, flat wall (V1V2 crop, slice k = 27) | `flatten_fused2.py` (P1, P2) |
| Separate-front stretch (V1V2 crop, plane j = 92) | `per_bank_test.py` (A_PTS, B_PTS) |
| Mildly unequal stretch and joining line (M1S1 patch crop, plane i = 85) | `per_bank_m1s1.py` (BANKS, CONTACTS) |
| Kissing stretch: bank ends, and 23 contact points | `data/M1S1_test_points.json` (`banks`; `gap_all` is the 23 points in test-crop coordinates, with point 17 corrected) |

**Which code produced which result:**

| Finding | Script (arguments) | Inputs | Figures |
|---|---|---|---|
| 28. White-surface meshing and a surface start | `mesh_smoothing.py` (phantoms with exact truth; contours, tangential relaxation, twicing and Taubin smoothing; closest point with pseudonormals; `march_band`, the march from per-seed start values carrying the root position); tests `test_mesh_smoothing.py` (`nix develop --command uv run pytest prototypes/scripts/test_mesh_smoothing.py`); `mesh_phantoms.py <figures> <json> [surface|band|fin|march]` (environment `STARTS` runs only the named starts); `mesh_real.py <figures> <data> <json> <patch>`; `mesh_real_band.py <json> <patch>` (collisions from each start; thin-blade sides at 1× and 2×); `mesh_tables.py <json>` (the tables) | `baillarger.data.PATCHES`; phantoms in `mesh_smoothing.py` | `mesh_phantom_{blades,fins,curvature,march}.png`, `mesh_<patch>_{sections,edge}.png`; ITK-SNAP `mesh_<patch>_thin_overlay.nii.gz` with `mesh_thin_labels.txt`; meshes `mesh_<patch>_{white_mesh_s1,twice_0p32mm}.vtk` |
| 27. The package on complete patches | `fronts_eval.py <figures> <work> <data> <patch>` (one pass from all WM in V1V2, M1S1 or PCC_SPL, equal then field weights; cost and memory per stage; reach; column-top thickness; contacts classed by d/μ) | the patches of `baillarger.data.PATCHES` | `fronts_eval_<patch>_planes.png`, `fronts_eval_<patch>_hist.png`; ITK-SNAP `fronts_eval_<patch>_overlay.nii.gz` with `fronts_eval_labels.txt` over `fronts_eval_<patch>_echo2_detrended.nii.gz` |
| 25, 26. One pass at the marked faces | `onepass_face.py <figures> <work> <data> <site>` (site M1S1, V1V2_banks, PCC_SPL_p1, PCC_SPL_p2; environment `RATIO` (cost across the normal, default 3), `SEP_MM` (contact separation, default 2.88), `TAG` (output suffix)); writes `onepass_<site>.json` | the points files; the full volume | `onepass_<site>_offsets.png`, `onepass_<site>_planes.png` (with `_sep1mm`, `_ratio1.5`, `_ratio1.0` variants); ITK-SNAP `onepass_<site>_overlay.nii.gz` with `onepass_labels.txt` over `onepass_<site>_echo2_detrended.nii.gz` |
| 24. Synthetic contacts | `synth_contacts.py <figures> <work>` (U-slabs, P4, P8, omega; four weight settings; the pure partition as a check; writes `synth_contacts.json`) | `baillarger.phantoms` | `onepass_synth_{uslab_R6,uslab_R22,P4,P8,omega}.png` |
| 23. The one-pass library | `onepass_lib.py` (setup and root patches, the pass, contacts, separation, column tops, the pure partition, point scores); tests `test_onepass_lib.py` (`nix develop --command uv run pytest prototypes/scripts/test_onepass_lib.py`) | — | — |
| 1. Orientation reproducible | `orientation_map.py`; `texture_check.py` (planar against linear structure, numbers only) | V1V2 and M1S1 patches | `orientation_V1V2.png`, `orientation_M1S1.png` |
| 2. Bands level along data-steered columns | `flatten_checkpoint.py` | V1V2 open V1 stretch | `flatten_V1.png` |
| 3. Fitted field, not raw tracing | `flatten_fused.py`, `flatten_fused2.py`, `diagnose_fused.py` (raw tracing); `fit_T.py openV1` and `fit_T.py fused`, with `--equivolume` and `--show-echo` (the fit) | V1V2 stretches | `flatten_fused.png`, `flatten_fused2.png`, `diagnose_fused.png`, `fit_{openV1,fused}_{us,ev}[_e2]_{slices,flat}.png`, `compare_openV1_norm.png` |
| 4. Columns slide across a shared field's crest | `cheap_test.py`; `bankB_check.py` (numbers only); `two_bank_view.py`, which reads `state_*.npz` written by `fit_T.py` and `prop_T.py`; `prop_T.py` (grid propagation) | V1V2 fused stretch | `cheap_test.png`, `two_bank_view.png`, `fit_fused_prop_*.png` |
| 5. Separate fronts, similar banks | `per_bank_test.py`, `per_bank_test2.py` | V1V2 plane j = 92 | `per_bank_test.png`, `per_bank_test2.png` |
| 6. Stria fractions by metric | `cheap_test.py` (open-V1 calibration: 0.40 and 0.48); `per_bank_test2.py` (bank fractions) | V1V2 | as for findings 4 and 5 |
| 7. Mildly unequal banks, joining line | `per_bank_m1s1.py` | M1S1 patch crop, plane i = 85 | `per_bank_m1s1.png`, `m1s1_plane_diag.png` |
| 8. Kissing stretch in the clean crop | `make_test_crop.py` (writes the crop, overlay and points file); `prob_thickness.py [--distance]` | `data/M1S1_test_points.json`; the full volume | `prob_{dist,ev}_{slices,field,calib}.png`, `prob_dist_faces45.png` |
| 8, superseded first reading | `near_touch_m1s1.py` | M1S1 patch crop (edge-affected) | `near_touch_m1s1.png` |
| 10. White-surface curvature | `curv_lib.py` (shared: boxes, curvature, T from all WM, columns); `curv_map.py` (writes `curv_V1V2.npz`, `curv_M1S1.npz`); `test_curv.py` (synthetic sphere, hole, cylinder, blade); `stair_check.py` (staircase check) | boxes from the full volume around V1V2 and the M1S1 test crop | `curv_map_{V1V2,M1S1}.png`, `curv_map_{V1V2,M1S1}_3d.png` (the 3D display mesh is staircased; `curv_stair_check_M1S1.png` is smooth) |
| 10. Deep and superficial parts against curvature | `split_depth.py` (reads `ev_closed_results.pkl`; numbers only) | open-V1 columns | — |
| 16. A10: the gap closure; prior length | `collide.py`, `collide2.py`, `unequal_check.py` with the `CLOSURE` switch (`points` by default, `none`, `thin`; visible tops next to fluid that stays fluid); `a10_summary.py` (compares the three runs); `prior_length.py` (the prior's correlation against distance on the graph) | as finding 11, one run folder per closure | `a10_closure.png` |
| 19. PCC_SPL pieces 1 and 2 | `site_setup.py` (now: any points file, `--piece`, `--centre` for a box re-centred on the points), then the site pipeline | `PCC_SPL_points.json` (piece 1), `PCC_SPL_p2_points.json` | `PCC_SPL_p1_run_*.png`, `PCC_SPL_p2_run_*.png`; ITK-SNAP `PCC_SPL_p1_run_contact_overlay.nii.gz`, `PCC_SPL_p2_run_contact_overlay.nii.gz` (over `<run>_echo2_detrended.nii.gz`) |
| 22. `fronts` accuracy at cortical scales | `fronts_length_misaligned.py` (constant direction); `fronts_length_aligned.py` (direction radial from the ball); both run under `memguard.sh 12` | synthetic ball sources | numbers only |
| 21. Piece 1's walls; piece 3's roots | `finding21_checks.py` with `sides` (each side's roots at the points; facing pairs per piece), `roots` (piece 3's roots, WM half-thickness, WM-dark cortex at the roots) and `strip` (the dark strip at a point and its echo decay); arguments: subcommand, work folder, data folder; reproduces the numbers above | `site_PCC_SPL_p1_run` and `sites` work folders | ITK-SNAP `PCC_SPL_p1_run_walls_diag_overlay.nii.gz`, `PCC_SPL_p3_roots_overlay.nii.gz` with their label files |
| 20. Sheet tests | `sheet_tests.py` (orientation, distance to WM and to fluid, doubled-band index at every sheet voxel; the user's readings in the script) | the surveyed crops; the M1S1 stretch | `sheet_tests.png`; ITK-SNAP `<crop>_sheet_tests_overlay.nii.gz`, `sheet_tests_labels.txt` |
| 18. V1V2 banks | `site_setup.py` (points and curvature for a site); `collide.py`, `collide2.py` and `collide2_itksnap.py` with the `SITE`, `POINTS` and `GROW` settings | `V1V2_banks_sites.json`, `V1V2_banks_points.json` | `V1V2_banks_*.png`; ITK-SNAP `V1V2_banks_contact_overlay.nii.gz`, `V1V2_banks_contact_labels.txt` |
| Sites for marking (Next step 1a) | `site_survey.py` (all-WM fronts, roots, collision classes, sheet pieces with walls from roots, and a rough reading per piece; writes the crops below) | boxes from the full volume | — |
| 17. A5: profiles across the contact | `a5_profiles.py` (profiles along the contact's own normal at the 23 points; open-pia reference per front) | as finding 11 (unclosed image) | `a5_profiles.png` |
| 15. A6: graph and fit | `a6_check.py` (refits the field as in collide2.py and inspects it; `Field.fit` now keeps its posterior mean for inspection) | as finding 14 | `a6_check.png` |
| 14. Bank position | `bank_position.py` (thickness at each front's tops against distance to its fused zone, sulcal depth and curvature; writes `bank_position.npy`) | `collide2_state.npz`, `collide_state.npz`, `curv_M1S1.npz` | `bank_position.png` |
| 13. Arc length; level variants | `collide2.py` (arc length S carried in the upwind passes; evidence and rules in S; the T version's outputs kept as `collide2_state_T.npz`, `collide2_rows_T.npy`); `level_variants.py` (level length and per-front level, scored with the pairwise formula; numbers only) | as finding 11 | `collide2_contact.png`, `collide2_offsets.png`, `collide2_error.png` (now in arc length) |
| 12. Unequal stretch | `unequal_check.py` (finding 8's geometric walls rebuilt; per-bank fields; T, traced length and straight distance at each point) | as finding 11, with `collide2_state.npz` | `unequal_walls.png`, `unequal_lengths.png`; ITK-SNAP `M1S1_unequal_overlay.nii.gz`, `M1S1_unequal_labels.txt` |
| 11. Collisions and per-front arrival | `collide.py` (all-WM field, roots, collision classes, thickness field with trend, calibration; writes `collide_state.npz`, caches `T_M1S1_closed.npz`); `collide2.py` (walls from roots, per-front fields, rules as fields, offsets); `post_error.py` (each front's μ − T at the points, from `collide2_rows.npy`); `collide2_diag.py` (each front's column from each point traced to its root, the root's belief and the evidence near it; writes the diagnostic overlay) | `curv_M1S1.npz`; the M1S1 test crop grown by 15 voxels | `collide_map.png`, `collide_contact.png` (all-WM, pinned to the midline), `collide_thickness.png`, `collide_offsets.png`, `collide2_contact.png`, `collide2_offsets.png`, `collide2_error.png` |
| 10. Three-metric re-score | `ev_closed.py` (reads `curv_*.npz`; caches `T_*.npz`; writes `ev_closed_results.pkl`); `alpha_sweep.py` (damped curvature and other scales, numbers only) | same boxes | `ev_openV1.png`, `ev_openV1_columns.png`, `ev_banks.png`, `ev_banks_columns.png`, `ev_thickness.png` |

**Crops for marking points in ITK-SNAP:**

| Crop | Made by | Files |
|---|---|---|
| V1V2 | `make_itksnap_crop.py` | `V1V2_echo3.nii.gz`, `V1V2_boundaries.nii.gz`, `V1V2_labels.txt`; with review-point markers `V1V2_boundaries_fused.nii.gz`, `V1V2_labels_fused.txt` |
| M1S1 patch | `make_itksnap_crop_m1s1.py` | `M1S1_echo2.nii.gz`, `M1S1_boundaries.nii.gz`, `M1S1_labels.txt`; with bank markers `M1S1_banks.nii.gz`, `M1S1_banks_labels.txt` |
| M1S1 test crop | `make_test_crop.py` | `M1S1_test_echo2.nii.gz`, `M1S1_test_overlay.nii.gz`, `M1S1_test_labels.txt` |
| Kissing faces for marking (Next step 1a) | `site_survey.py` | `<site>_echo2_detrended.nii.gz`, `<site>_sites_overlay.nii.gz` (WM boundary, sheet, junction, sheet edge, walls by side, piece centres, review points), `<site>_sites_labels.txt`, `<site>_sites.json` (corner and pieces), for `V1V2_banks`, `V1V2_collapsed`, `V1V2_thin`, `PCC_SPL`; `sites_points_template.json` |
| M1S1 test crop, per-front contact (finding 11) | `collide2_itksnap.py` (reads `collide2_state.npz`, `collide_state.npz`); `collide2_diag.py` | `M1S1_test_echo2_detrended.nii.gz`, `M1S1_contact_overlay.nii.gz` (WM boundary; posterior and equal-arrival contacts within 2 mm of a sheet; points 7–13 and the others as dots), `M1S1_contact_labels.txt`; `M1S1_contact_diag_overlay.nii.gz` and `M1S1_contact_diag_labels.txt` add each front's columns from the points, their roots, and the evidence tops by source |

Open a crop in ITK-SNAP with:

```
/Applications/ITK-SNAP.app/Contents/bin/itksnap -g prototypes/data/M1S1_test_echo2.nii.gz -s prototypes/data/M1S1_test_overlay.nii.gz -l prototypes/data/M1S1_test_labels.txt
```

and the per-front contact with:

```
/Applications/ITK-SNAP.app/Contents/bin/itksnap -g prototypes/data/M1S1_test_echo2_detrended.nii.gz -s prototypes/data/M1S1_contact_overlay.nii.gz -l prototypes/data/M1S1_contact_labels.txt
```
