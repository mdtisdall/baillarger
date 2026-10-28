# Level-set filling: test plan

**Purpose.** This plan tests one theory of how cortex fills outward from the white surface. It is self-contained: a session that has not seen the design discussion should be able to do the work from this document and the files it points to. Results go in *Results* below, written as current state.

**The question.** At 1× grid (160 µm), does level-set filling place contacts within one voxel (0.16 mm) of the correct answers? The correct answers are exact contacts on phantoms and the user's marked points on the four faces. Refinement (finer grids, a local mesh step at the end) is out of scope until this question is answered.

## 1. The theory

- **Filling, not distance.** Cortex is filled outward from the white surface. Each patch of white surface supplies a volume of cortex, its **budget**. All of the white surface advances together in **normalized time τ**: τ = 0 at the white surface, and τ = 1 where a patch has used its whole budget.
- **Direction from the data.** Material moves along the data's layer normal n: the leading eigenvector of the structure tensor, with coherence c. That direction is signed by the motion: each point keeps the sign of the direction it arrived with.
- **Rate from the budget, not the path.** The rate along a path never changes the path. Paths follow n only, so they do not bend toward faster neighbours. This is advection along a fixed direction field, not an eikonal equation with variable speed, which would refract paths (finding 24's omega leakage).
- **The fronts are the level sets of τ.** Laminae follow equal volume: depth within the cortex is τ, the fraction of the budget used.
- **Where the cortex is open,** the front stops at fluid. That stop measures the patch's volume and thickness: the evidence for budgets.
- **Where the cortex is fused,** two parts of the front meet. The **contact is the seam where they arrive at equal τ**: both sides have used the same fraction of their budgets. Nothing is fixed afterwards. The rule falls out of everything advancing in common τ.
- **This is normalized arrival** in the main plan's terms (the misfit is shared in proportion to the budgets). Uncertainty is not in the first version. A later version can add start offsets in τ to give the posterior.

### Length or volume

There are two forms of the rate. Build both; they differ in one line.

- **Length:** ds/dτ = μ(R), where μ is the expected thickness at root R.
- **Volume:** ds/dτ = v(R) / a(x). Here v is the expected volume per unit of white-surface area (mm), and a is the cross-section ratio: area of the bundle of paths at x over its area at the white surface. In a flat region a = 1 and the two forms agree.

Volume is the target, and length is the reference. On a gyral crown they differ, because the bundle widens outward. Exact case: an annulus from radius 10 to 13 has quarter-volume boundaries at radii 10.83, 11.60 and 12.31. Quarter-length boundaries are at 10.75, 11.50 and 12.25.

## 2. Representation (all 3D, on the image grid)

**Fixed inputs:**
- **Labels:** WM, cortex, fluid, missing tissue. The box faces are flagged; they are not tissue boundaries.
- **The data's normal n and coherence c,** computed as in the main plan (*Method*, data term). Phantoms use their known normal.
- **The white surface,** for starting values and roots.
  - On phantoms: the analytic surface.
  - On real data: finding 28's recommended surface, built with the prototypes in `mesh_smoothing.py`.
    - **Build:** contour the WM label itself (binary marching tetrahedra), relax the vertices tangentially, then smooth along the mesh with twicing (2G − GG, Gaussian) at about 0.32 mm.
    - **Not** `white_mesh`'s volume smoothing: at σ = 1 it already loses 16% of V1's thin-WM voxels, and it shrinks convex surfaces.
    - **Orientation:** `white_mesh` orients its faces into the WM. Any face-based signed distance must account for that, or use finding 28's oriented mesh.
    - **Curvature** for the trend comes from this mesh. The label's terraces still leave curvature ripples of ±0.1–0.2 per mm at 0.32 mm (finding 28).

**Carried per voxel, once claimed by the front:**
- **τ:** normalized time.
- **R:** root, a point on the white surface (3 coordinates, or a mesh triangle and barycentric coordinates).
- **s:** arc length along the path.
- **a:** cross-section ratio. Two ways to compute it, and compare them:
  1. from the gradient of R: the area-scaling of R's differential on the plane perpendicular to the direction;
  2. from the divergence of the signed direction, using d(ln a)/ds = ∇·d.
- **w = ∫ a ds:** volume per unit of white area so far.
- **d:** the signed direction the voxel was reached along.
- **The root's budget** (μ or v), carried along the path with R, so the march never looks a budget up from the mesh. Test that a carried budget equals the budget looked up at the carried R, away from seams and rims. Near rims and seams, report how much they differ.

**On the white surface:** each root's budget (μ or v), from the thickness field (§4).

## 3. The algorithm

The front only moves outward. So the level-set evolution can be computed in its static form, as an ordered upwind march in τ: the same relation fast marching has to level-set evolution.

1. **Start.** Each WM voxel next to cortex is claimed with τ = 0 and R = its foot point on the white surface. Each also gets s = w = its distance to the white surface, a = 1, and d = n signed to point away from the WM. Better, and needed on real data: start a narrow band of voxels at their exact distance to the surface (analytic on phantoms, the mesh on real data). Each band voxel also gets its foot point as R. Finding 28's rules for this band:
- **It must reach as deep as the update looks back.** For Selling's stencils at ratio 3 that is 2.45 voxels; a 1-voxel band put a plane's arrival off by up to 1.4 voxels. Here the trace-back step is h, so the band must be at least h plus the interpolation stencil deep. Test the depth on the flat slab.
- **The sign of the distance must come from the pseudonormal** of the closest feature (edge or vertex). The face normal gives distance 0 at staircase edges.
- Finding 28's prototype `march_band` already seeds from such a band. Reuse its band construction.
2. **Candidates.** For a cortex voxel x next to claimed voxels, compute a candidate τ for each side it can be reached from.
   1. Sign the data normal at x to agree with the direction of the claimed neighbour it comes from, so that d_x · d_y > 0.
   2. Trace back one step h from x along −d_x to a point p (semi-Lagrangian).
   3. Interpolate τ, R, s, w and a at p. Use only claimed voxels whose own direction agrees with d_x (d_y · d_x > 0). This is a check on sign agreement, not on distance, and it is what keeps the two sides of a seam from mixing.
   4. Set τ(x) = τ(p) + h / rate(p), and s and w likewise.

   Where coherence is low, blend d toward the normalized gradient of arc length s from the claimed neighbours, weighted by c. Record the blend weight.
   - Not the gradient of τ: τ depends on the budgets, so blending toward it made paths bend with the budgets wherever c < 1. On real data that broke §1's rule that the rate never changes the path; the visible tops then differed between budget settings.
   - s does not depend on the budgets, so paths stay fixed.
3. **Accept in τ order** (a heap). An accepted voxel keeps the candidate with the smallest τ, together with that candidate's R, s, w, a and d.
4. **Stops.** Fluid, missing tissue and the box are never claimed.
   - A claimed voxel next to fluid is a **visible top** only where its path runs into the fluid: its direction is within 45° of the step into fluid, as in `onepass_lib.open_thickness`.
   - At a visible top, record its root R, its s at the exit point (between voxels, not at the voxel centre), and its w. That is the evidence.
   - Tops near box faces or missing tissue are not evidence.
5. **The seam (contact).** Neighbouring claimed voxels whose signed directions oppose each other (d_x · d_y < 0) lie on a seam.
   - Its position between voxels: along the line joining the pair, extrapolate each side's τ and take the point where they are equal.
   - At each seam point, report τ on each side: τ ≈ 1 means the budgets fit the gap. Also report the angle between the two arriving directions.
   - A seam whose two directions converge rather than oppose (a fold tighter than the thickness) is reported as a fold, not a contact (§5, the U-slab).
6. **Topology** (optional, measured):
   - A simple-point test on accepted voxels, with (26, 6) connectivity, prevents changes of topology. Examples are a handle from a label error, or an enclosed cavity.
   - **It does not create contacts.** When two banks meet in a fused sulcus, the slot between them is filled from its closed end (the fundus) upward. Each step there is a simple point, so the topology test allows the banks to join.
   - **The separation of the two sides comes from step 2's sign agreement,** not from topology. Run with and without the topology test, and report where it blocks anything.

**Evidence, then the budgeted pass.** The evidence does not depend on the budgets. A path's direction comes from the data, and s and w accumulate along it whatever the rate; the rate only orders the march. So away from seams, any budgets give the same visible tops. Check this: the tops' s and w must agree between runs with different budgets, except within a voxel or two of seams.
1. **Evidence run:** a run with the prior budgets (below) gives the visible-top evidence.
2. **Fit the field** (§4), with the prior as its mean.
3. **Budgeted pass:** the fitted budgets.

Three settings of the budgets, reported side by side:
- **equal budgets** (the median): the equal-arrival reference, comparable with finding 13's equal arrival;
- **prior only** (the field's prior with no evidence, §4): how far the prior alone places the contacts;
- **posterior** (the field fitted to the evidence): the main result.

The difference between prior-only and posterior shows how much the evidence moves the contacts. The joint fit, which adds fused-zone volumes as evidence, is out of scope.

**Coverage and the budget misfit.** The march is one pass and ends when no candidates remain, so covering all the cortex is a completeness check, not convergence. A front that has used its budget keeps going (τ > 1), so every cortex voxel that is connected to the WM gets claimed.
- *Report coverage:* the fraction of cortex claimed, and the unclaimed connected pieces by cause. Causes are: no connection to the WM (a label error or an island); blocked by the topology test; cut off by a box face or missing tissue; or left behind by a failure of sign agreement. Each is a diagnostic. None of them shows the contacts are right.
- *Report the misfit:* τ on each side at the seam, and the volume claimed beyond τ = 1. A version that stops each patch at τ = 1 would turn coverage into a check on the budgets. Cortex left unfilled means the budgets are too small; a front still pushing at the seam means they are too large. Report that version as a diagnostic beside the main one, which continues past τ = 1. That misfit is the signal the later joint fit would use.
- *What is a convergence question:* whether the contacts stop moving when evidence, field and pass are repeated: taking the tops again from the budgeted pass, refitting the field, and running again. Report how far they move.

**Ordering** is causal, because τ increases along every path when the rate is positive. Guard against rates that are zero or that blow up: cap a from below, and report how often the cap applies. Small a happens at focal points of concave white surface (fundi), where paths converge.

## 4. Budgets

- **On phantoms:**
  - the oracle budgets, which are each side's true thickness or volume per area;
  - the budgets from the visible tops through the field.
- **On real data:**
  - **The model:** the thickness field as in `prototypes/scripts/collide.py` (class `Field`): a second-order Gaussian field of log thickness, with a level term and a length of about 2 mm.
    - **Fit it on the white mesh's vertices, with a proper mesh Laplacian:** cotangent weights with a mass matrix, set so the length stays 2 mm. `collide.graph_L` is scaled for nodes about 0.32 mm apart. The mesh's vertices are about 0.11 mm apart, so reusing it would shrink the length to about 0.7 mm.
    - **Check that the field is unchanged.** At one face, rerun the held-out calibration of findings 11 and 15: 5 spatial folds of evidence nodes, the held-out z SD (about 0.75 before), and the residual IQR in log thickness. Compare with the old field on the voxel graph. If they differ a lot, report it before using the mesh field at the faces.
  - **The quantities:** fit log μ for the length form and log v for the volume form.
  - **The prior:** the field's prior mean is a level plus a curvature trend, as in `collide.py`. Areas are not labelled. The earlier pipeline found M1/S1's unequal banks without labels: each bank's fused roots took their thickness from that bank's own visible tops. This works because the two banks' roots are 11–18 mm apart along the white surface, although about 5 mm apart in space, and the field's length is about 2 mm. Keep it that way. The root map R carries that separation into the level set, so a seam's two budgets come from two different neighbourhoods of the white surface.
    - **The known bias is in how the field extrapolates, not in the evidence** (main plan, findings 13–16). The nearest tops on the thin front read about 2.1–2.2 mm, but over the 1–2.5 mm of white surface without evidence before the contact roots, the field rose to 2.4–3.0 mm. The second-order prior extrapolates the local gradient, and the field reverts toward the box-wide level of 2.67 mm.
    - Report the face results with the field as built (second order, single level) and with a first-order prior (α = 1), which extrapolates no gradient and reverts more locally. Treat the variant as a diagnostic of A6, not as a tuned fix.
    - The seam misfit (§3, τ on each side) is the evidence the field lacks there: the volume or length the fused zone actually holds. Using it is the joint fit, which is out of scope. Report it.
  - **Curvature and other covariates.** Curvature enters the field's mean as a trend, as in `collide.py`: log t = m + (a₀ + w(x))·H + z(x), where H is the white mesh's mean curvature at each vertex at a 0.8 mm scale. Use the surface finding 28 recommends once it exists; until then use `mesh_curvatures` at its default. The main plan's findings on this:
    - the shared coefficient is small where evidence is near (a₀ +0.07 to +0.17 per 1/mm), and the trend barely changes held-out error (finding 11);
    - its role is extrapolation into the gap before a contact;
    - at the M1/S1 contact roots H ≈ 0, so it plays no part there (finding 13);
    - on the thick front, sulcal depth carries a clear trend (ρ −0.65, finding 14), and the contact roots there lie outside the curvature range of the evidence.

    For the level set:
    1. **Fit the same model to both budgets:** thickness μ (length form) and volume per white area v (volume form). Report a₀ with its SD, the range of w, and held-out error (5 spatial folds), with and without the trend.
    2. **Test whether the correlation is geometric.** On a crown, the bundle of paths widens outward, so v exceeds t. In a fundus it narrows, so v falls below t. If much of the known link between thickness and curvature comes from this geometry, v's curvature coefficient should be smaller than t's and its held-out error no worse. Report that comparison. It decides whether the volume form also simplifies the prior.
    3. **Sulcal depth** (below a hull: tissue closed with a 4 mm ball, as in finding 14) as a second covariate, as a diagnostic.
    4. **Report, at each face, the covariate values at the contact roots against the range at the evidence roots.** A contact root outside the evidence's range is an extrapolation and should be flagged (A7, A8).
  - **Converting to mm:** use the median, exp(m), not the lognormal mean. The mean blew up where the SD was large (main plan, finding 26).
  - **Pieces of the graph with no evidence** get the median budget, counted and reported.
  - **Roots within 2 mm of a box face** give no evidence.

## 5. Tests, in order

**Scope of the first report.**
- **Must have:**
  - the phantoms. Their budgets come from the median of each side's visible tops within 2 mm, plus the mesh field once on P4 as an end-to-end check.
  - at the faces: equal, prior-only and posterior budgets, in both the length and the volume form;
  - coverage, and the seam misfit (τ on each side);
  - the check that the tops don't depend on the budgets;
  - the distance between each point's two roots along the white mesh;
  - the calibration of the mesh field against the old field at M1/S1;
  - the curvature coefficient for v against t;
  - open V1.
- **Second:** α = 1; the τ = 1 stop; one convergence iteration; topology on and off at one face.
- **Later:** sulcal depth, and the tables of covariate ranges.

Say in the report what was deferred.

**Face scoring** follows `score_points_sides`, adapted to seams:
- the line through each point along the data normal, 2.5 mm each way;
- the two ends' roots define the two fronts. Each seam voxel goes to the end whose root is nearer along the white mesh (Dijkstra on the mesh's edges).
- A crossing is a seam pair split between the ends, within 1 mm of the point and 0.4 mm of the line. The nearest seam crossing in any direction is reported beside it.
- The sign (+ toward the thinner side) is fixed per point from the posterior length budgets and used for every run.

**Open V1** uses finding 10's own stria definition and roots exactly, so its ρ is comparable.

Each piece gets exact-answer tests in `prototypes/scripts/test_levelset_fill.py` before it is used on data. Diagnose failures, write a failing test, then fix. Never tune constants to hide a symptom. The one-voxel target is the user's headline question. Everything else is reported as diagnostics, with views.

1. **Flat slab, one side open.** A WM plane and cortex of thickness 12 voxels, topped by fluid.
   - τ, s and w must be exact up to discretisation; R must be the foot point.
   - The tops must give thickness 12.
2. **Crown** (a spherical shell, 3D) and the annulus (a cylinder).
   - The level sets of τ under the volume form must match the exact equal-volume radii, and the length form the equal-length radii.
   - a from ∇R must match a from the divergence and the exact value.
3. **Apposed slabs, P4 and P8** (`synth_contacts.py`, `baillarger.phantoms`).
   - The seam must lie at the exact contact with the oracle budgets, and where the rule predicts with the budgets from the evidence. In the earlier one pass, P4 gave +2.45 voxels and P8 gave 0.
   - Compare with the one pass of findings 23–24.
4. **Wedge:** two banks meeting at an angle (20°, 45°, 90°), with unequal thicknesses.
   - Exact reference: run each bank alone over the whole domain, then solve τ_A = τ_B.
   - **Three versions of the known normal:**
     - **A:** each bank's own wall normal, up to the exact seam;
     - **B:** the nearer wall's normal, flipping at the bisector;
     - **C:** B, with coherence falling to 0 in a band about 2 voxels wide around the bisector, so the blend to the front's own normal decides there.
   - Report each seam's distance from τ_A = τ_B at each angle. Sign agreement pins a seam to a sharp flip of the normal, whatever the budgets. State that as a finding.
   - At the faces, report coherence and the blend weight at each point, so it is clear which regime each point is in: decided by visible layers, or by the budgets.
5. **U-slab** with fundus radius 6 and 22, and thickness 16 (`synth_contacts.u_slab`).
   - Where the radius is below the thickness, paths converge at the fundus. Report the fold signature (τ < 1 on both sides) and whether the seam is classed as a fold.
   - Report how often the a cap applies.
6. **Omega:** a junction and a step in budget.
   - Paths must not bend toward the faster side. Check it: the root of each voxel must match its root under the oracle direction field.
7. **Thin blade:** WM 2 voxels thick. The two sides' fronts must not mix: R must stay on its own side.
8. **The four marked faces:** M1/S1, V1V2 banks, PCC_SPL pieces 1 and 2.
   - Load and score as in `prototypes/scripts/onepass_face.py`: the same boxes, points and local two-sides score (`onepass_lib.score_points_sides`), with points 6 mm or more from box faces.
   - Run equal, prior-only and posterior budgets (§3), each in the length and the volume form.
   - Report:
     - median |offset| and the share of points within 0.16 mm;
     - worst point and missing points;
     - each side's error, μ − S or v − w, at each point;
     - each point's two roots' distance along the white mesh. It should be large, as in the old pipeline (11–18 mm at M1/S1). A short distance means the two budgets came from the same neighbourhood.
   - Compare with the earlier results:

     | Face | Walls with the posterior (findings 13, 18, 19) | One pass, field weights (finding 25) |
     |---|---|---|
     | M1/S1 | 0.33 mm (equal arrival 0.67) | 0.26 mm |
     | V1V2 banks | 0.11 mm | 0.40 mm at cost ratio 3; 0.16 mm at ratio 1 |
     | PCC piece 1 | walls failed | 0.08 mm |
     | PCC piece 2 | 0.11 mm | 0.16 mm |
9. **Open V1** (in the V1V2 box). Does the stria sit at constant τ under the volume form? Earlier measures: the stria fraction against curvature, ρ +0.63 (main plan, finding 10). Compare with the length form.

## 6. Rules

- **Read first:**
  - `CLAUDE.md`;
  - the main plan, `reports/Seismic flattening for cortical layers.md`: the top summary; *Method*; *The contact as an inference problem*; *The fronts as PDE problems*; findings 10–13 and 22–27;
  - the memory notes listed in `MEMORY.md`.
- **Prototype code only,** in `prototypes/scripts/`: for example `levelset_fill.py` (the library), `test_levelset_fill.py`, `levelset_synth.py` and `levelset_face.py`.
  - Do not edit `src/`, commit, branch or open PRs. The prototypes and reports are untracked by design.
  - If the method should become a package, write that up as a proposal.
- **Heavy runs:** `scripts/memguard <GB> <command>`, one at a time.
  - The cap guards against runaway code. Set it to about 1.5× the run's measured or expected peak.
  - Measure a small case first.
  - memguard runs its command in the background, so pass scripts as files.
- **Always 3D.** Slices are for display only.
- **Speed:** numba for the march; reuse `baillarger.fronts` conventions where they fit.
- **Views:**
  - figures in `prototypes/figures/levelset_*.png`, showing echo 2 with the trend removed;
  - ITK-SNAP overlays in `prototypes/data/levelset_*_overlay.nii.gz`, with a label file. Include τ, the seam, side ownership, and your points.
- **Other sessions edit the main plan.** Edit around their text. Use the next free finding number at the time of writing (finding 28 is taken by the mesh evaluation).

## 7. When done

1. **Record the results in *Results* below,** as current state: tables, figures and what failed.
2. **Add a short finding to the main plan.** It should give the headline answer to the one-voxel question and point to this document. Also update the main plan's top summary and Next steps 0 if the results change them.
3. **Message the session that started you with the conclusions.** It appears in your linked sessions; use SendMessage with its `local_…` id. Answer:
   - Is the level set within one voxel at each phantom and each face, in each form?
   - Which form does better, length or volume?
   - Does the stria sit at constant τ?
   - Where does it fail, and why: the budgets, or the propagation? Use the per-side errors.
   - What would a package need?

## Sources

All are cited from the design discussion. Notes are in `research_notes/Collisions and contacts/pde_framing.md` and `filling_literature.md`.
- **Topology-preserving level sets:** Han, Xu and Prince 2003. Topology correction with fast marching: Bazin and Pham 2005. Both from memory.
- **CRUISE and ACE:** Han et al. 2004. ACE puts the contact at the midline.
- **DiReCT:** Das et al. 2009. Its source is in ANTs, `Utilities/itkDiReCTImageFilter.hxx`. Its thickness prior only caps a side's thickness; it does not split fused banks.
- **Fast marching and extension velocities:** Sethian; Adalsteinsson and Sethian 1999.
- **Equal volume:** Bok; Waehnert et al. 2014; Leprince et al. (BrainVISA isovolume).

## Results

**Headline.**
- **Phantoms: within one voxel in both forms,** with two exceptions: the wedge at 90° with the nearer wall's normal (version B, median 1.15 voxels), and the 90° wedge whose normal has no coherence at the bisector (version C, no seam at all).
- **Faces, length form with the posterior budgets:**
  - PCC piece 1: 0.06 mm (95 % of points within 0.16 mm);
  - PCC piece 2: 0.05 mm (78 %);
  - M1/S1: 0.18 mm (39 %);
  - V1V2 banks: 0.40 mm (26 %), worse than its prior-only 0.24 mm.

  So it is within a voxel at the median at both PCC pieces, just outside at M1/S1, and off at the V1V2 banks.
- **Length or volume:** the volume form ties on M1/S1's median (0.19 mm) with three times the spread. Its carried area is biased and its budget field is badly calibrated, so it is reported as failing (below).
- **The stria does not sit at constant τ** in either form at open V1.
- **Where it fails:**
  - At V1V2 the failure is in the budgets: evidence from wandering paths, and two roots in one neighbourhood.
  - In the propagation, the failure is at a 90° flip of the normal without coherence.
  - Where the fronts behave, the competition places the seam where the budgets say.

Library `prototypes/scripts/levelset_fill.py`; tests `test_levelset_fill.py` (41 pass, 1 strict xfail: the 90° wedge, version C); scripts `levelset_phantoms.py`, `levelset_mesh.py`, `levelset_synth.py`, `levelset_face.py`, `levelset_v1.py`, `levelset_diag_area.py`, `levelset_diag_v1v2_{budget,patches,tops}.py`. Results JSON and logs in `prototypes/data/levelset_*.json` and `levelset_log_*.txt`. All 3D at 1× grid. Phantom numbers are in voxels, face numbers in mm.

### What the algorithm became

Three changes to §3, each made after a failing test:
- **Two phases.** Phase 1 accepts in order of s and chooses each voxel's side by s, so the geometry (sides, R, s, w, a, d) does not depend on the budgets. Phase 2 accepts in τ order.
  - A voxel keeps its phase-1 side and values when that side has an accepted neighbour. Otherwise it takes the other side's semi-Lagrangian candidate.
  - Voxels whose side changes between the phases are counted as `moved`.
  - **Why:** with a single τ-ordered march, the order of acceptance decided which neighbours a voxel's blend and trace-back could use. At M1/S1 the visible tops then differed between budget settings, at 100 % of vertices with the τ blend and still widely with the s blend.
  - The test `low_coherence_paths_do_not_depend_on_budgets` (noise 0.15 and 0.4 on the normal) now holds except at moved voxels.
- **The low-coherence blend goes toward the least-squares gradient of s** over that side's accepted neighbours, not of τ. This is the user-approved change, with its own exact test.
- **The trace-back goes to the face of the upwind cell** (h = 1/max|d_i|), not a fixed unit step. The unit step always had x itself as a corner, and s read low on curved fronts (`test_crown_level_sets`).
  - The divergence for a is computed independently of the march order: each neighbour's normal is sign-aligned with d, blended like d, and used only within 45°. Otherwise it falls back to a one-sided difference.
  - Seams are 6-neighbour pairs with d·d ≤ 0 (perpendicular directions count).

**Start band (§3.1).** It holds WM within 2 voxels of the surface and next to cortex, plus cortex with φ < 1 voxel, at exact signed distance (pseudonormal sign), with R the foot point and d the surface normal.
- **Departure from §3:** ln a = φ·D in the band, with D the surface's mean-curvature divergence. a is not 1 there.
- The band depth is tested on a tilted plane (`band_depth`).
- On real data the surface is finding 28's: binary contour, relaxed, twicing at 2 voxels (0.32 mm), smoothed normals.

### Phantoms (voxels; `levelset_synth.json`, `levelset_synth_panels.png`)

| Phantom | Length form | Volume form |
|---|---|---|
| Flat slab, thickness 12 | τ, s, w exact (10⁻¹⁵); tops 12.000 | same |
| Cylinder (annulus), level-set radius error | median 0.03, max 0.09 | median 0.04, max 0.11 |
| Sphere (crown), level-set radius error | median 0.08, max 0.17 | median 0.09, max 0.21 |
| a, p95 error | from the divergence 0.6 % (cylinder) and 1.5 % (sphere); from ∇R 2.6 % and 4.3 % | |
| P4, oracle budgets | seam exact (τ = 1 both sides) | exact |
| P4, open 12/18 (oracle and evidence budgets) | +2.40 = the rule's prediction (finding 24's one pass +2.45) | +2.40 |
| P8, oracle | exact | median 0, max 0.16 (lip rows) |
| P8, open 12/18 | −0.20 = prediction (one pass −0.19) | −0.20, max 0.37 (lip rows) |
| P4, mesh field end to end (finding 28's surface) | budgets 12.0 / 18.0; seam +2.42 | — |

- **The evidence budgets** are the median of each side's tops within 2 mm (12.5 voxels). They read 12.000 and 18.000, so the evidence and oracle runs agree.
- **The P8 lip rows:** the seam moves up to 0.37 voxel in the volume form where a ≠ 1 at the lips. The test documents those rows; in the length form they match the flat rows.

**Wedge** (two banks at 20°, 45°, 90°, unequal thickness; distance of the seam from the exact τ_A = τ_B line, median / max):

| Normal | 20° | 45° | 90° |
|---|---|---|---|
| A: each bank's own normal | 0 / 0 (volume max 0.06) | 0 / 0 (volume 0.13) | 0 / 0 |
| B: the nearer wall's, flipping at the bisector | 0.05 / 0.07 | 0.27 / 0.41 | **1.15 / 1.95** |
| C: B with coherence 0 in a 2-voxel band | 0.025 / 0.05 | 0.14 / 0.27 | **no seam** |

- **Pinning finding:** sign agreement pins a seam to a sharp flip of the normal, whatever the budgets. With version B the seam sits at the flip, not at τ_A = τ_B, and the error grows with the angle: 1.15 voxels at 90°.
  - At the faces, where the data normal flips sharply between visible layers, the visible layers decide the contact, not the budgets.
- **Limitation (strict xfail):** version C at 90°. With no coherence at the bisector, the s-gradient blend rotates the directions through the band, so the two fronts merge and never oppose. No seam forms. At 20° and 45° the blend halves B's error.

**Other shapes:**
- **U-slab, fundus radius 6, thickness 16:**
  - the seam lies on the bisector (|x − c| ≤ 0.004 length, 0.31 volume), z 21–43 (expected 20–43.7);
  - τ is 0.40–0.98 on both sides, the fold signature;
  - 56 of 184 pairs at the top of the fold are classed contact, 128 fold;
  - a ≥ 0.14, never capped.
- **U-slab R 22:** no seam.
- **Omega:**
  - the neck's seam is exact in the length form, max 0.57 in the volume form;
  - floor roots match the oracle direction field (p99 0.09), with the step budget identical to the equal one, so paths do not bend toward the faster side;
  - 1 % of all voxels have roots 30 voxels from the oracle's: an assignment across the neck, not examined further.
- **Blade, WM 2 voxels thick:** no voxel takes a root on the wrong side; max root error 0.36.

### The four faces (mm; `levelset_<site>.json`, `levelset_log_<site>.txt`)

Median |offset| (share of all points within 0.16 mm; worst; missing):

| Face | Equal | Prior only | Posterior | Walls (13, 18, 19) | One pass (25) |
|---|---|---|---|---|---|
| M1/S1, length (23) | 0.63 (0 %; 0.85; 1) | 0.39 | **0.18** (39 %; 0.90; 0) | 0.33 | 0.26 |
| M1/S1, volume | 0.76 (0 %; 0.95; 8) | 0.74 | 0.19 (43 %; 0.58; 3) | — | — |
| V1V2 banks, length (35) | 0.35 (20 %; 0.80; 6) | **0.24** (29 %; 0.75; 4) | 0.40 (26 %; 0.92; 6) | 0.11 | 0.40 (0.16 at ratio 1) |
| PCC piece 1, length (22) | 0.17 (41 %; 0.45; 0) | 0.09 (64 %) | **0.06** (95 %; 0.17; 0) | failed | 0.08 |
| PCC piece 2, length (37) | 0.15 (54 %; 0.66; 2) | 0.05 (86 %) | **0.05** (78 %; 0.55; 2) | 0.11 | 0.16 |

- **The volume form was run at M1/S1 only.** The other faces are in the length form, on the parent's instruction once the volume form had failed (below).
- **M1/S1's τ-blend run, labelled** (before the s blend and the two phases): posterior length 0.17 mm (39 %, worst 0.71, 1 missing). The posterior volume run found no crossings, with 231,541 voxels at the a floor. Its tops agreed between budget settings at 0 % of vertices.

**Checks at every face** (posterior, length unless noted):
- **Tops against budgets:** the posterior run's tops equal the evidence run's at 98.3 % (V1V2), 99.0 % (M1/S1), 99.2 % and 99.4 % (PCC) of evidence vertices. The p99 |Δlog s| is 0–0.004. §1's "the rate never changes the path" holds.
- **Carried budget against a lookup at the carried R:** median |Δlog| 0.1–0.2 %, p99 1.3–6.2 %.
- **Coverage:** 98.3 % (M1/S1, V1V2), 99.3 % (PCC 2), 99.7 % (PCC 1). Unclaimed pieces by cause are in the JSON's `coverage`.
- **Volume past τ = 1:** 6.3 % (M1/S1), 10.1 % (V1V2), 2.9 % and 2.5 % (PCC).
- **Seam classes:** contact / fold / overlap / mixed, inner pairs. At M1/S1, length: 15,996 / 8,778 / 1,216 / 1,065. At M1/S1, volume: 9,996 / 63,899 / 759 / 6,669; the volume form classes most pairs as fold because its τ runs low.
- **a floor (0.05):**
  - length form: never more than 9 voxels;
  - volume form at M1/S1: 2,716 voxels (equal) and 4,236 (posterior), all beyond 3 mm, in the overshoot past contacts.
- **Moved voxels:** 45,344 (M1/S1, length), 185,047 (volume), 30,781 (V1V2), 7,772 and 11,900 (PCC).

**Per point** (`rows` in each JSON):
- **Two roots' distance along the white mesh:**
  - M1/S1: median 18.3 mm, none under 5 mm;
  - PCC 1: 14.1 mm, none under 5 mm;
  - PCC 2: 20.7 mm, 2 under 5 mm;
  - **V1V2: 16 of 35 under 5 mm** (p10 0.6 mm). There the two ends' roots lie in one neighbourhood, so both budgets come from it.
  - At V1V2, |offset| is 0.49 mm at those points against 0.20 at the rest.
- **Coherence and blend at the point:**
  - median c 0.77 (M1/S1), 0.66 (V1V2), 0.87 (PCC 1), 0.47 (PCC 2);
  - points with blend > 0.5: 6, 8, 3 and 20;
  - |offset| at those points: 0.12, 0.47, 0.06, 0.06. Low coherence alone does not hurt; at PCC 2 most points are in the blended regime and still within 0.06 mm.
- **Per-side error, b − S (the budget minus the arc length to the seam), posterior length:**
  - M1/S1: thin +0.48, thick +0.31 mm (IQR 0.24 each). Both budgets exceed the seam's s and τ < 1 on both sides, as in finding 13's over-prediction; the ratio, not the level, places the seam.
  - PCC 1: −0.01 / 0.00. The budgets fit the gap.
  - PCC 2: +0.14 / +0.30.
  - V1V2: +0.39 / **+0.81** (IQR 0.7).
  - Spearman ρ between the two sides' errors: +0.21, −0.01, −0.43, +0.51.
  - M1/S1, volume (v − W): thin +0.32, **thick +1.33**.

**Why the V1V2 posterior is worse than its prior** (`levelset_diag_v1v2_*.py`, `levelset_log_diag_v1v2.txt`):
- The posterior length budget is above 4.5 mm at 8,840 vertices, in four patches of 684–6,176 vertices. The largest is about 6 × 8 mm.
- The prior there is 2.6 mm, and the evidence median within 2 mm along the mesh of the affected points' roots is about 2.4 mm.
- The patches are not mesh artefacts: their μ is smooth (|μ − 1-ring mean| ≤ 0.03) and their vertex areas are ordinary.
- **They come from long tops.**
  - 7.5 % of V1V2's used tops have s > 4 mm, at a straight-line distance to fluid of 2.5 mm (ratio 1.96, against 1.06 for all tops).
  - Their paths follow d sideways before they meet fluid within 45°.
  - They cluster, and the field follows them: its noise is Gaussian with σ_ε = 0.08, so nothing down-weights them.
- Points 20, 22 and 27 have thick-side budgets of 4.5–5.2 mm against S of 1.8–2.0 mm. Each goes from within 0.06 mm with the prior to 0.79–0.92 mm with the posterior.
- **This is a failure of the evidence, not of the competition.** A top should be rejected when its path is much longer than the straight distance to fluid, or the field needs a robust likelihood. Neither was tried; it needs a failing test first.
- The second V1V2 failure is the 16 points with roots under 5 mm apart: the sides of the seam are not separated along the surface there.

**Views:**
- `prototypes/figures/levelset_<site>_planes.png`: echo 2 with the trend removed; τ at 0.25/0.5/0.75 in blue and 1 in yellow; seam pairs; your points; three planes × equal / prior / posterior.
- `levelset_<site>_offsets.png`.
- ITK-SNAP: `prototypes/data/levelset_<site>_overlay.nii.gz` (seam, side ownership, your points; labels in `levelset_labels.txt`), `levelset_<site>_tau.nii.gz`, `levelset_<site>_echo2_detrended.nii.gz`.
- At M1/S1 the posterior seam follows the points along the bank. At V1V2 the misses lie toward the end of the sulcus, where the cortex is thick and the seams scatter.

### Budgets: the mesh field (M1/S1, `levelset_M1S1_calibration.json`)

- **The field:** cotangent FEM on finding 28's mesh (339k vertices at M1/S1), second order, ℓ = 2 mm. Posterior mean by conjugate gradients preconditioned with the prior covariance, checked against a direct solve on a small mesh. A direct LU of the posterior passed 13.6 GB at M1/S1.
- **Held-out calibration,** 5 spatial folds, 20 perturb-and-MAP samples per fold:

| Field | Evidence units | z SD | within ±1.96 | RMSE (log) | residual IQR |
|---|---|---|---|---|---|
| Mesh field, length | 15,743 vertices | 1.15 | 91 % | 0.146 | 0.029 |
| Old voxel-graph field, same tops | 5,018 pooled nodes | 0.90 | 95 % | 0.123 | 0.025 |
| Mesh field, volume | 15,743 vertices | **2.89** | **59 %** | 0.381 | 0.098 |

- The mesh field's length budgets are about 15 % overconfident. The old field is slightly underconfident; finding 11's 0.75 was on different evidence.
- The two are not on the same units: one evidence value per vertex against per pooled node, and the vertices are 3× denser. They fit equally well.
- The SDs were not used by the competition, which takes exp(mean).
- The fits per face, length:

| Face | a₀ | level m | residual IQR | vertices without belief |
|---|---|---|---|---|
| M1/S1 | +0.12 | 2.79 mm | 0.029 | 964 |
| V1V2 | −0.05 | 2.42 mm | 0.039 | 30,650 |
| PCC 1 | +0.18 | 2.37 mm | 0.027 | 8,988 |
| PCC 2 | +0.28 | 2.48 mm | 0.029 | 574 |

### The volume form: failing, with the diagnosis

- **The carried area is biased.**
  - At M1/S1, a from the divergence has median 0.99 at 0–0.5 mm above the WM, falling to 0.73 at 2–3 mm.
  - The data divergence D has IQR 0.42–0.54 per mm, against finding 28's geometric 0.27.
  - a from ∇R degenerates beyond 1.5 mm. On phantoms both are within 1.5 % and 4.3 % (above), so the bias comes from the data normal's noise accumulating in ∫∇·d, not from the scheme.
- **The volume budgets inherit it.**
  - v's curvature coefficient is a₀ = 1.76 at M1/S1 (0.91–2.05 at the other faces), against 0.12 for thickness. Its residual IQR is 3.4× the length form's, its b p99 is 31.7 mm, and its calibration is z SD 2.89.
  - §4's geometric question (is v's curvature coefficient smaller than t's?) cannot be answered with this a. v's large coefficient is not decisive either way.
- **The a floor:** in the single-phase march it applied to 7.5 % of claimed cortex, all beyond 3 mm. Now it applies to 2,716–4,236 voxels, all beyond 3 mm.
- **At the seam,** the volume form's thick-side error is +1.33 mm. Its median |offset| ties the length form's, but its IQR is 0.36 against 0.13, and 3 points are missing.
- Fixing a (smoothing the divergence along d, or a from a regularised ∇R) was not tried in this report.

### Open V1 (`levelset_openV1.json`, `levelset_openV1.png`)

- **Setting:** finding 10's roots (WM boundary within 1 mm of the in-plane path, slice k = 54) and stria (darkest echo 2, smoothed, 0.5–1.6 mm from the WM). Open columns only, ≥ 2 mm from faces.
- **Columns:** traced along the volume run's d from each root's foot on finding 28's mesh. 491 columns.

| Depth of the stria | Median | IQR | ρ against root curvature Hm |
|---|---|---|---|
| Arc length over its own top, s/s_top | 0.373 | 0.053 | +0.62 |
| Volume over its own top, w/w_top | 0.427 | 0.089 | −0.63 |
| τ, length form (posterior) | 0.387 | 0.051 | +0.61 |
| τ, volume form (posterior) | 0.402 | 0.094 | **−0.79** |
| Finding 10: distance; divergence equivolume | 0.435; 0.449 | 0.059; 0.073 | +0.63; −0.67 |

- **The stria does not sit at constant τ.** The length form reproduces distance's positive trend. The volume form over-corrects more strongly than finding 10's equivolume, as a biased a toward small values would.
- The fractions are lower than finding 10's because s here starts at the mesh foot, not the voxel boundary.

### Deferred

- α = 1 (first-order prior);
- the τ = 1 stop;
- one convergence iteration;
- topology on and off at a face (the simple-point test is built and tested, not run on faces);
- sulcal depth as a covariate;
- the tables of covariate ranges at contact roots;
- the volume form at V1V2 and PCC;
- a fix for the carried area;
- the rejection of wandering tops.

### What a package would need

1. **The start band from a mesh:**
   - signed distance with pseudonormal sign, foot point and normal, ≥ 2 voxels deep;
   - finding 28's smoother;
   - ln a = φD in the band.
2. **A two-phase march** (numba, about 50–100 s per 141³ box per run):
   - phase 1 by s fixes sides, R, s, w, a and d;
   - phase 2 by τ competes;
   - the semi-Lagrangian step to the upwind cell's face, interpolation restricted to agreeing directions, and the low-coherence blend toward ∇s;
   - the root's budget carried with R.
3. **Seams as data:** pairs with d·d ≤ 0, their crossing position, τ, s, w and budget on each side, and the class.
4. **The thickness field on mesh vertices:**
   - cotangent FEM;
   - conjugate gradients with the prior as preconditioner;
   - pieces without evidence set to the median;
   - roots near faces excluded;
   - **a guard on the evidence:** reject tops whose path is much longer than the straight distance to fluid, or use a robust likelihood. V1V2 shows this is needed.
5. **Not yet:** the volume form, which needs an unbiased a first. The length form is the one to package.
6. **Open:** what to do where a point's two roots are close along the surface. Neither the competition nor the field can separate the sides there.
