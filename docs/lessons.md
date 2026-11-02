# Lessons for the restart

What the work so far has established, for a clean start on four streams:

1. WM segmentation and white surface generation
2. Columns traced from that surface
3. Laminar profiles along the columns, with possible feedback to the white surface
4. A viewer for manual interaction and exploration

This document states current knowledge only. How the project got here is in git, the PRs and the reports. Each item points to its evidence:

- `#N` is https://github.com/mdtisdall/baillarger/pull/N.
- SF is `reports/Seismic flattening for cortical layers.md`, and Fn is its finding n.
- CP is `reports/Crossing paths and contacts from laminar profiles.md`.
- LS is `reports/Level-set filling test plan.md`.
- PE is `reports/Pial editing and edit priority.md`.
- CL is `reports/Cortical laminae segmentation lessons.md`.
- FS is `reports/Fused sulci and thickness priors.md`.
- PL is `PLAN_LAYER_REFINE.md`.

**The data behind every result:**

- One right hemisphere, imaged with 7 T multi-echo FLASH in Fomblin: magnitude only, TE 9.37 / 20.71 / 32.05 ms, TR 80, flip 25°.
- The voxel is 0.16028 × 0.16 × 0.16028 mm, which is not isotropic.
- The aseg comes from purple-mri at native resolution.
- Most stream 2 and 3 results come from 141³ boxes (V1V2, M1S1, PCC_SPL), checked against the user's marked points: M1/S1 23, V1V2 35, PCC pieces 22 and 37.
- Only stream 1 has run on the whole hemisphere.

## 1. Goals, constraints and working rules

### Goals

One top-level goal per stream:

1. **WM segmentation and the white surface:** a closed white surface without handles for each hemisphere, at the acquired resolution, whose normals are the cortical normals.
2. **Columns:** a complete ray from every white-surface vertex through the cortex, to fluid or into the far bank's WM, recording what it meets on the way.
3. **Laminar profiles:** where each layer lies along every column, read from the image's own contrast and aligned across columns and across contacts between banks, with what it learns fed back to stream 1 as an input.
4. **The viewer:** a way to see any result at full resolution anywhere in a whole brain, and to answer each stage's review queue.

Each stream publishes its outputs in a specified format and reads only the others' published outputs. Each reaches a first version good enough for the others to build on, then improves along its own course; a change in one stream should not force work in another.

### Constraints

These bind every stream from its first version. A first version may be rough, but its algorithms and data structures must be ones that scale.

- **Native resolution.** Work at the acquired resolution: 160 µm now, 50 µm whole brains later. Never decimate or downsample to make a step tractable. The decimated overview is for display only.
- **Whole brain.** The target is about 2·10¹⁰ voxels and 3.5·10⁸ surface vertices. Stages work on blocks with a halo, with no whole-volume float arrays. Volumes are read through one reader, chunked when large: NIfTI-Zarr, 64³ chunks sharded in Morton order.
- **GPU where it pays at the large scale.** PyTorch: CUDA on big machines, MPS on the laptop, with numpy as the reference. Each device is tested against numpy on the laptop. These stay on the CPU, parallel over independent pieces: sequential steps (handle removal), I/O and checks.
- **Results identical across devices up to float32 rounding.** No discrete jump that rounding can flip.
- **Always 3D.** Slices only for seeding or display.
- **Standard formats where they fit, extended where scale needs it.** Volumes are NIfTI and surfaces GIfTI, with BIDS-derivative names and JSON sidecars that carry provenance. Where whole-brain scale needs more than a standard gives, extend it so that a standard reader still opens the file, and specify the extension:
  - `white-surf/1` is GIfTI with external binary data (nibabel reads it, memory-mapped), plus our own block index and edge keys.
  - Chunked volumes are to be NIfTI-Zarr, still a draft standard, exported to `.nii.gz` for other tools.
  - Where BIDS has no rule, names are BIDS-like and the spec records the choice.

  Each output has one written spec, enforced by one writer and one reader.
- **No GPL dependencies,** with no exceptions. Where only a GPL library fits, use a permissive one or write the piece ourselves. State the licence when proposing a library.
- **FreeSurfer is the model.** Its topology correction, pretess and strand thickening are the defaults to follow. Deviate only with cause and data. We do not run FreeSurfer, even as a benchmark.
- **A person decides only where the data cannot.** Places that need judgment go into one review queue per stage. Each answer is a replayable edit: the automatic result is correct without it, and the review refines it.
- **Assume no particular aseg source.** The aseg is an input with its own errors, read in the FreeSurfer label convention. Take evidence from the echoes, not from one tool's quirks.

### Working rules

- **Test, don't patch.** Diagnose first, then write a test with an exact answer that fails, then fix. Never move a threshold to hide a symptom.
- **Diagnostics over criteria.** While a method is exploratory, design each checkpoint around views the user can read, not pass/fail numbers. The default image view is echo 2 with the depth trend removed.
- **Keep principle, role and estimator apart.** Record each result against one estimator in one role. A failed estimator is not a failed principle.
- **Feasibility, not speed,** for prototypes and interactive tools. Correctness checks still apply.
- **Heavy runs go under `scripts/memguard`,** one at a time, with the cap at about 1.5× the run's peak.

## 2. Facts that cut across streams

**The image** (CL; CP)

- Laminar contrast is mostly R2\*, and that is mostly iron that follows myelin. Dark is not always myelin: cell-dense layers can be dark too.
- What each echo gives:
  - Echo 3 has the most stria contrast but is the grainiest.
  - Echo 1 has the most SNR and the best tissue/Fomblin contrast, with noise about 0.6 of echo 2's.
  - R2\* proxies from two echoes are too noisy to use.
- **The noise is spatially correlated:** neighbouring voxels correlate at 0.5–0.67, even in deep WM. What works is a white-noise model with the same gradient power: the median smallest structure-tensor eigenvalue, corrected by the derivative filter's gain and the Wishart min-eigenvalue bias (0.68 at n_eff = 39). It needs no calibration (actual over predicted noise is 1.0–1.6).
- Laminar structure is planar at 160 µm everywhere, M1 included (median λ2/λ1 ≤ 0.20). Radial and tangential fibres give the same tensor signature.

**The aseg we have, from purple-mri, errs in repeatable ways** (CP; SF F21; #46, #49)

These describe this one aseg, not asegs in general. It is the example data we have today, so these are the errors the tools must cope with now. Later versions of purple-mri may fix some of them, and another source will err in its own ways. Treat them as cases to test against, not as rules to build in.

- Its cortex runs 1–3 voxels into the Fomblin around most sulci.
- It fills thin fluid channels of collapsed sulci with cortex, and labels superficial vessels and thin Fomblin gaps at tight bends as cortex. These make false contacts. At V1V2, relabelling such cells took the columns crossing fused banks from 6,001 to 717.
- Missing tissue (excised blocks that the aseg imputes) is 14.6 % of V1V2's cortex label. It shows as FLASH below 0.25 × the cortex median.

**Boxes distort results**

- 65–99 % of column failures and a quarter to two thirds of contacts lie at box faces (SF F26, F27).
- Any box-scale measurement needs regions at least 6 mm from every face, and leaves out roots within 2 mm of a face.
- The restart works on the hemisphere. Box numbers in this document are evidence about methods, not about the hemisphere.

**Identity, topology and precision** (code; #61)

- **One tetrahedralization: the Kuhn split.** Voxel topology uses Kuhn adjacency (14 neighbours, the same for object and background), so the voxels' topology equals the mesh's. Do not use the (26,6) or (6,26) codes: their answers can disagree with the mesh.
- **Vertex identity is the grid-edge key:** lin(p) × 7 + (4 s_i + 2 s_j + s_k − 1), in int64. It survives remeshing, so anything stored per vertex keys on it.
- **The block layout:** the grid starts at voxel 0, with 64-voxel blocks in Morton order (i at bit 3b+2, j at 3b+1, k at 3b). A vertex is owned by the first block that uses it.
- **Precision:** mesh steps run in float32, with float64 only for the affine, area sums and intersection tests. Torch kernels use no float64, no CSR and no atomics, so each run gives the same result. An unavailable device is an error.
- **Manual and feedback inputs** are diffs against a SHA-pinned input, applied before the automatic step and never baked into the source (`wm-edits/1`, #78).

**Licences**

- **`thinqpbo` is GPL-3** (checked: its dist-info LICENSE), so it has to go. `surfcut.qpbo` wraps it for the joint warp. `src/` imports it only behind `surfcut.qpbo`; the prototypes call it.
- PyMaxflow and CHOLMOD are GPL and avoided. scipy's `maximum_flow` gives the same optimal cuts on int32 capacities. Keep the total below 2³⁰, because int64 silently returns a flow of 0 (#14).
- nnU-Net is Apache-2.0 (#86).
- Licences believed but not checked: Madagascar, CAT12/SPM and cut-pursuit are GPL; Nighres, LayNii, ANTs and HFM/agd are not.

**Tooling and libraries**

- Numba is a hard dependency: mesh intersection takes 0.2 s against 13 s in numpy, and a simple-point test 0.3 µs against 59 µs (#8).
- CVXPY's HiGHS breaks OR-Tools CP-SAT in the same process, so the CVXPY reference runs in its own uv project (#61).
- CI needs uv's download cache once torch is required (#63).

## 3. Stream 1: WM segmentation and the white surface

### Established

- **The pipeline that works on the hemisphere** (#61, #67, #79):
  1. Fill a WM mask: aseg 41 plus 43, 49–52, 192, with cavities filled; the hippocampus is not filled.
  2. Remove handles.
  3. Antialias to a thin-plate field φ (φ ≥ ε inside, φ ≤ −ε outside).
  4. Contour φ by blocked marching tetrahedra.

  Results:
  - Betti (1, 111, 182) → (1, 0, 0).
  - One closed genus-0 piece: 12.7 M vertices, 25.3 M faces, no self-intersecting faces.
  - CPU and MPS give identical meshes, bit for bit.
  - Peaks: `dseg` about 22 GB, `mesh` 20–24 GB.
- **Handle removal** (`handles.remove_handles`) (#38, #48, #67):
  - Object and background fronts grow by simple points. Each handle's wall is cut or filled, whichever changes fewer voxels; ties go to the cut, then to the lowest voxel index.
  - Walls that block each other are claimed as sets (Mayer–Vietoris), or voxels are stranded.
  - Fills of one piece can touch another piece, so check the union's Betti numbers and refuse to write unless b1 = b2 = 0.
  - Most moves are 1–2 voxels.
  - The meeting wall need not be the smallest cut: one cut of 2,047 voxels ran along a thin strand.
  - Batching walls per pass changes the result, so don't.
- **Antialiasing beats smoothing** (#61; SF F28).
  - Each vertex stays on its own grid edge, so the surface has no self-intersections by construction and keeps the edge keys. A tilted plane's normals come out within 0.5° (the raw staircase is 31° off).
  - Smoothing the label and then contouring is wrong where WM blades are 2 voxels thick. At σ = 1 voxel it loses 16 % of V1V2's thin WM, it shrinks convex surfaces by about σ²κ, and it makes slivers.
  - Taubin λ|μ smoothing is unstable at short cutoffs. Tukey twicing never amplifies.
  - The label's own terraces limit curvature and normals below about 1–2 mm on surfaces near a grid axis, whatever the method.
- **Tight folds (misfacing faces).** The hemisphere surface has 152 at 91 sites, and diagonal-only contacts cause most of them: 74 of 90 sites have a checkerboard or corner-only contact within 4 voxels (#81, #85).
  - Thicken to 2s, widen gaps between surfaces, then make the mask well-composed, at s = 0.30 mm. This takes 90 sites to 5 and edges over 60° from 297 to 4, at a cost of +3.9 % area and a 90th-percentile neighbour angle of 5.4° against 3.9° (#85).
  - s must not be a whole number of voxels along any axis: at 0.32 mm (exactly 2 voxels on one axis) the checkerboards rise from about 390 to 7,000. 0.26 mm is worse on every count; above 0.30 mm the bulk and area grow.
  - Synthetic antialiased phantoms make no misfacing faces, so fold tests need hand-built folds plus a hemisphere run.
- **The deep structures belong in the mask** (ventricle, thalamus, ...). The white mesh must include imputed WM, so the edge of a cut does not become a false white surface (#66; PL).
- **Curvature** (SF F28): the 0.32 mm mesh with curvature at 0.48 mm agrees with quadrics fitted to the voxel faces (rank correlation 0.85–0.98), with about half their jitter.
  - Never take curvature from the Hessian of a smoothed signed distance: it mixes the two sides of thin blades.
  - Valence-weighted smoothing biases Gaussian curvature. Use smoothing that keeps mass.
- **Overview decimation** (`decimate`; #70–#77): 25.3 M → 210 k faces at ε = 0.1 mm in 186 s, keeping topology, the ε bound, unmoved true vertices, orientation and no self-intersections.
  - The fold test must compare against the input normals each face stands for, not the face's last state; otherwise 1,027 faces turned over.
  - Pairs of sheets closer than 2ε at necks cross. Pinning the crossing faces' input vertices and re-decimating clears them.
  - Map anything back to true vertices with the decimation's own carried-vertex map. A nearest-face search lands on the wrong side of thin sheets.

### Doesn't work (stops a retry)

- Forcing cuts, with no fills into cortex: 90 → 131 sites (#85).
- Voxel opening at 0.24–0.40 mm: worse at every radius, grows the area, and cascades over passes (#85).
- Relaxing the level set without a local "no new blade or gap voxel within s" test: the thin-plate energy shrinks widened gaps and blades into slivers. With the test it keeps 95 % of the gain on crops, but misses tubes (#85).
- A level set started on a 2× grid from the label's own signed distance: 6–11 % wrong signs. A finer grid cannot fix an error that is in the boundary data (SF F28).
- Inside/outside from the face normal: distance 0 at staircase edges. Use the pseudonormal of the closest feature (SF F28).

### Standing decisions

- The aseg is never rewritten. Corrections are derivatives (`desc-wmtopo_dseg`), and manual edits are replayable diffs (`wm-edits/1`) applied before the automatic correction.
- The generator fixes topology automatically, with no question per handle. Places that need judgment go to one review queue, asked in one sitting. Each answer becomes a `wm-edits` row.
- **The white surface exists for its normals, not its position.** Columns start on it and run along its normals. There is no pial surface and no "true GM/WM boundary": the GM/WM transition is aligned afterwards like any other layer.
  - Judge methods by normal quality: neighbour-angle spread, misfacing faces, cancellation in thin blades, and agreement with orientation from the image.
  - The surface may move into GM or WM to make normals well defined, as long as it keeps the topology and does not self-intersect.
- WM may grow into cortex: thin gyral WM is thickened, not removed, as FreeSurfer's mri_segment does.
- The meshing uses no cortex label. Every white-mesh vertex is a column root, and what columns pass through is for later stages to judge.

### Open

- Fixes chosen by intensity (FreeSurfer's cut or fill by intensity; #85 step 2); the tube case; the size threshold for exceptions; whether fills through cortex are as acceptable as thickening.
- Whether to use our own WM segmentation of the FLASH (nnU-Net, with review answers as labels; #86). Its interface (labels required, probseg optional) keeps the surface steps independent of it.
- Whether T = 0 should come from the image's own WM contrast rather than the label. BigBrain puts the L6/WM boundary superficial to the tissue-class white surface (CL).
- Finding hidden fluid gaps from the image before columns are traced: a thin, slowly decaying dark strip inside labelled cortex (SF F21).

## 4. Stream 2: columns

### Established

- **Columns are complete rays.** They run from the white surface, inward as well as outward, through GM to fluid, or through a contact into the far bank's WM. They stop only at fluid, far WM, missing tissue, the domain edge or a length cap.
  - Meetings are recorded on the column (`crest_at`, `meet_at`, `turn_at`) and never acted on. The cut is read from the profile in stream 3.
- **The crossing rule is frozen** (user, #52; `column_front.past_crest`): climb ∇T; inside T's crest stencil, keep the heading; on leaving it, if T falls ahead, descend −∇T for good. Methods that differ in one respect share it.
  - Column rules stay simple. A field artefact is recorded in a test, never patched with a new clause.
- **The data normal** (SF F1): structure tensor of trend-removed log echoes 2 and 3, at derivative scale 1 voxel and integration scale 2 voxels, by masked normalized convolution over cortex.
  - For about 91 % of interior cortex the echoes agree within 10°.
  - Coherence falls to 0.05–0.38 at contacts and fundi, which is where most failures happen.
  - The normal has no sign. Sign it by the WM distance through tissue, with fluid as an obstacle. A straight-line distance flips the sign beside open sulci (CP; #46).
- **The fitted field T** (Lomask-style least squares of ∇T against the normal, T = 0 on WM) has no stalls and bridges fused gaps.
  - T is not a distance: it under-reads long columns.
  - It must be extended 1 voxel into fluid and 0.5 mm into WM, or it makes spurious turns at the pia.
  - pyamg needs local Jacobi weights to be deterministic (#40, #45).
- **T's crest is the midline between the two WM surfaces, not the contact.** On the kissing phantom it sits 0.3 mm from the contact, even across a fluid gap (CP). The crest test (T rises into a voxel but not beyond, with a slope change above L²κ plus noise) needs no calibration.
- **Column methods**, measured at the marked points (CP; #47, #52, #58):
  - **normal** (straight rays along the smoothed white normal, no field, no image): keeps the right class on every phantom and almost never returns to its own bank. Its profile split is the best at M1/S1 (median 0.17 mm) and 0.075 mm at V1V2. Its costs are more stops at caps and edges, and 0.18–0.41 mm path error against curved true columns at crowns. The user's decision: normal rays stay straight.
  - **front** (the white mesh grown as one surface, directions coupled along the mesh at 2 voxels): the most accurate wherever folds are 0.64 mm or wider (contact p90 0.054 mm), and it reaches all 35 V1V2 points. Its swept volume matches equal volume within 0.2 %. It is worst at 0.32 mm crowns (p90 0.77 mm), because the coupling scale equals the crown radius.
  - **fitted_blend start** (mesh normal at the root, blended into the field over 0.3 mm): the most accurate field tracer at tight fundi and contacts, and the default start for the field methods. It removes the turn spike at 0.3 mm from the WM (turns fall from 11–16 % to 0.5–2.4 %).
  - **front-equivolume** (speed A₀/A along the smoothed normals): bends 0.25–0.31 mm off the straight ray, ends many columns at the caps, splits poorly at M1/S1 (0.55 mm), and runs 8–9× slower. Which paths are right is untested.
- **Fast marching** (`baillarger.fronts`; Mirebeau's Selling stencils) is causal for anisotropic metrics and never jumps a one-voxel wall.
  - Arrival is first order, with 4–8 % error from a point source.
  - Arc length is good only where paths follow the metric's direction.
  - 141³ takes about 10 s at 2.7 GB.
  - Per-source start times and speeds give crystal growth: connected regions.
  - The root's position must be carried through the pass, as arc length is; it then lands within 0.01 mm. One label per root voxel triples the arrival error, and root patches misplace the root by 0.36–0.80 mm (SF F23).
- **Distances on the mesh** need the edges across neighbouring triangles: 8.2 % bound on a plane, against 41 % along edges alone (#10, #45).
- **Theory** (SF; shock theory): because the WM is connected, every collision is a front meeting itself. A fold and a contact differ only in how far apart the meeting roots lie along the surface, which is a continuum, so no pointwise quantity separates them.

### Doesn't work (stops a retry)

- Tracing step by step along the raw tensor normal: no field, no stop, so paths stall and run on (SF F3).
- Laplace streamlines: they make fused cortex look open (49 k open against 348 crossing at V1V2) and need the outer boundary first.
- Any threshold on bend angle, turn, or root separation along the surface: each cuts a continuum. Between 10° and 60°, 5–34 % of classes change. A 5 mm "same bank" fallback is wrong at both faces.
- Dijkstra or 6-neighbour fast marching for anisotropic metrics: about 29° lattice bias.
- A lattice step across T's crest: about 31 k returns to the own bank per site.
- Crest methods on the raw aseg: they cross the aseg's fluid channels and vessels. They need the pial-label refinement first (#46).
- Sorting front sides by connected pieces, and geometric walls between banks: fragile where banks join round a fundus. The one-pass march needs no sides.
- A single all-WM field cannot place an unequal contact: its ridge is the geometric midline (F11).

### Standing decisions

- Columns root on `white-surf/1` vertices, keyed by edge key, and run inward and outward. No step uses the fixes image to decide where to search, or re-finds a boundary as the depth origin.
- Banks are told apart by distance along the white mesh, using the across-triangle graph (`SAME_MM` 5 mm), so the two sides of a thin blade stay apart.
- Bok's measure for a vertex is the exact flux of its dual cell. Equivolume depth is (V(s) − V(s₀)) / (V(s₁) − V(s₀)).

### Open

- Which paths are right in real cortex: straight rays, field traces or the front. The planned check is phantoms with curved, equivolume true columns.
- What the crests inside thick cortex are: they cover 5–10 % of cortex even after relabelling.
- The cost ratio across the layer normal for arrival (3, 1, or falling with coherence).
- `fronts` accuracy when the metric is tilted 5–20° off the true paths, which is the case on real data.
- Thin-blade normals keep as little as 0.56 of their length after averaging (`docs/scaling.md`). The column stage has to handle them.

## 5. Stream 3: laminar profiles, contacts and alignment

### Established

**Depth within a column (role 1)**

- The principle: where there is contrast, the data decide; where there is none, depth follows equal volume. Geometry alone cannot put layers on level sets: one equal-volume surface lies in different layers in different areas (Wagstyl 2018).
- The current estimator is distance, until an equal-volume estimator is validated against visible bands. The results so far:
  - **The closed form from white-surface curvature** (A/A₀ = (1 + κ₁s)(1 + κ₂s)): see the conflict in §9. With curvature at 0.32–0.80 mm it over-corrects on open V1 (ρ −0.69), because the pia is flatter than a parallel shell (SF F10). With curvature smoothed at 1 mm it gives no stria trend in the band model (PL; #23).
  - **Carried area / div u:** over-corrects about 2× and is noisy upwind (SF F6, F10; #23).
  - **The volume form of the level set:** its carried area is biased (LS).
  - **The front's swept volume:** exact on phantoms, not yet tested against bands (CP).
  - **Two-surface forms with the pial curvature:** too noisy (PL).
- On open V1, curvature moves depth between the deep and superficial parts while total thickness barely changes (Bok's effect).

**Thickness carried sideways (role 2)**

- Crowns are thicker than fundi everywhere measured: V1V2 slope +1.2 mm per (1/mm), M1S1 +2.7. Fit the slope per area, because M1S1 mixes an areal step with folding (SF F10).
- Volume per unit white-surface area is the least conserved quantity against curvature (ρ 0.95). Equal-volume thickness does not absorb the curvature trend.
- A stationary Matérn/SPDE field (on log thickness, about 2 mm long) fits tightly at the evidence. But across the 1–2.5 mm without evidence before a contact it reverts to a broad mean and over-predicts the thin bank by 0.26–0.92 mm. The missing evidence is the sums t₁ + t₂ = L at contacts (F13–F16).
- Convert to mm with the median, exp(m), not the lognormal mean (LS).

**Contacts**

- **M1/S1 is genuinely unequal:** 2.0 mm against 3.2–3.3 mm. At its contact two layer I's meet with no fluid gap, and the superficial layers are 0.1–0.15 mm thinner per side than at open pia (F12, F13, F17).
- V1V2's banks are near equal (2.18 and 2.31 mm) and sit on a residual dark line, which explains much of every estimator's success there.
- The two fronts' errors are correlated. Where both banks are over-predicted alike, a σ² split cancels the bias.
- How a contact split shares the misfit: the Gaussian posterior shares it in proportion to σ², and equal z-scores share it in proportion to σ. Past runs used both, so they are not directly comparable.

**The column volume and flattening**

- **The column volume:** one trace per white-mesh vertex, sampled every 0.04 mm, each sample keeping its 3D position. It reads like a seismic section: the stria runs as a continuous band from open into fused cortex, and the far bank's stria appears mirrored past a contact (CP).
- **Flattening by shifts between neighbours** (`flatten`; Lucas–Kanade shifts, the Lomask / Wu–Hale form, CG):
  - Exact on synthetic warps.
  - On the boxes: 5–6 min, 6–7 GB, weighted RMS misfit 0.033–0.042 mm.
  - It keeps real upstream faults visible rather than smoothing them away.
- **Dynamic warping adds nothing between neighbours.** It agrees with the gradient shifts on 98.4 % of samples, and widening the bound makes the fit worse. Use it for long-range matching: templates, across gaps, between areas (#57, #60).
- **The joint warp** (one monotone z(τ) per column, z(1) = the pia, solved by QPBO at a few levels then filled):
  - Its terms: conservation of shifts inside each bank, the two-sided tie c_A + c_B = L at contacts, robust open-top anchors, weak lateral smoothness, a vertical prior and hard ordering.
  - It certifies 96.6–100 % of variables, so the remaining errors belong to the model.
  - The vertical prior must be the chain form, |z_ℓ − (τ_ℓ/τ_ℓ₊₁) z_ℓ₊₁|.
  - Levels placed at detected bands tighten the stria's spread by about 40 %.
  - It reaches 0.08 mm at both M1/S1 and V1V2.
- **QPBO engineering** (#30–#33):
  - Hard costs must be 1 + the finite total, not 1000×.
  - QPBO-I must reseed rand() for determinism.
  - No term may depend on a window's centre (that made the passes cycle).
- **Bands in τ:** stack each trace with neighbours weighted by profile correlation, and keep extrema at least 1.5 voxels wide. A 2.5-voxel threshold drops the stria (1.67 voxels).
  - V1V2: bright at τ 0.18, the stria at 0.37, bright at 0.81.
  - M1S1 has a dark band at τ 0.33 that is not a stria.
- **The profile likelihood of a split** (template A stretched to x, B reversed to L − x):
  - On spliced columns with a known answer it is within 0.02–0.09 mm.
  - It is 2–3× overconfident, because the residuals are correlated.
  - The top-anchored stretch beats the linear one.
  - At M1/S1 it misses by 0.5–0.8 mm into the thin bank. This is not diagnosed.

**Bands, regions and edges** (PL; #12–#24)

- Bands per region: Bellman DP, with K chosen by 2-fold cross-validation. Do not refit the held-out gain, and take the least K within max(1 SE, 5 % of the held-out gain). This finds the true K in 160 of 160 synthetic cases.
- The voxels within 1 voxel of the WM or the pia are partial volume. They made half the data term and drew 30 % of band boundaries; leave them out.
- Ward merging stops at the column graph's connected pieces.
- Cumulative bottom tests react late, so test each bin on its own.

**From the literature** (CL; FS)

- Nothing published fits a depth field to laminar contrast, or places a fused contact from expected thickness: every tool splits fused banks symmetrically.
- Design requirements:
  - Keep "present but invisible" apart from "pinched out".
  - Model partial volume at both edges.
  - Use sub-voxel boundaries.
  - Avoid top-down sequential chains.
  - Put a soft prior on changes of band thickness.
- **Build order into the representation; let contrast decide placement.**
  - Constraints that hold everywhere belong in the representation: ordering, no crossings, zero thickness allowed, topology.
  - Constraints that hold only on average belong in the energy: smoothness, equivolume, templates.
  - Every surveyed method that broke this rule needed patches.

### Doesn't work (stops a retry)

- A single darkness horizon for the pia: where the dark line fades it jumps to the far bank's stria, a mirrored decoy (CP).
- Profile templates from open tops far from the fused stretch: they miss at M1/S1.
- Phase-unwrapping relative geologic time (it slips cycles at facies changes). Multi-object level sets (they cannot let a band vanish). k-means with no spatial prior. Per-column classifiers glued together afterwards.

### Standing decisions

- **No faults in cortex.** A depth jump between neighbouring columns is an upstream error. Areal borders are facies changes: a thickness kink, a pinch-out, or a contrast change at continuous depth. A fused contact is a boundary between two stacks, not a fault.
- Never label banks by geometry. A voxel's bank is its column's root.
- Search only bands of at least 2–3 voxels; thinner layers ride along on the levels. The stria is at this limit and is the reference case.
- The first pial surface is a level of the joint warp (τ = 1), not a separate cut.
- Ties at contacts are two-sided (c_A + c_B = L).
- Report a split three ways (profile alone, profile with field, field alone), and with both stretches.
- Region-specific thickness priors are future work, as soft regions.
- The B0-orientation covariate is deferred: a possible later check only if facies changes track folding.

### Open

- The joint fit of the thickness field to visible tops and to the contact sums L, and whether it converges.
- σ or σ² on the contact. It should follow a stated principle.
- The M1/S1 profile miss. The candidates: template locality, superficial thinning at the contact, or an areal border.
- An honest likelihood: measure the residual correlation, or use Student-t.
- Band identity beyond the stria; whether equal τ is the same layer across an areal border; what M1S1's τ 0.33 band is.
- How the stream 3 → 1 feedback is expressed, as a spec'd input to stream 1 in the way `wm-edits` is.

## 6. Stream 4: the viewer

### Established

- **The architecture works:** one server bound to 127.0.0.1, holding the data and doing the computation. Shared state goes over POST /state and Server-Sent Events, so any number of windows and panes stay in step. There are two linked points: a crosshair in world mm, and a selected true vertex (#43, #44).
  - This keeps a server on a big machine with a tunnelled browser possible.
  - No response may scale with the whole mesh or volume.
- **Display through the overview:** every pick maps to a true vertex, nothing is measured on the overview, and caches are keyed by the source SHA. Hemisphere start takes 24 s the first time and 0.8 s cached (#70–#72).
- **The brick pyramid volume viewer** (32³ bricks, 2×2×2 means, a WebGL2 page-table atlas): the hemisphere echo builds in 6.8 s at 1.4 GB, and every sampled pixel matches the data. Its `BrickSource` is the interface for a NIfTI-Zarr store (#84).
- **NiiVue 0.69:**
  - Its 2D mesh display is unusable at 160 µm. Draw exact plane sections on a 2D canvas instead (16–22 ms for 873 k triangles).
  - A click on a mesh does not move the crosshair, so the surface pane is our own WebGL view.
  - Load it from the pinned `+esm` URL.
  - `readTCK` drops the first line: start the data with a NaN row.
  - `loadMeshes`/`removeMesh` race: use `addMeshFromUrl`.
  - Catch keys in the capture phase.
- **Geometry tools:**
  - Sections from a plane cut of the true mesh cannot fork.
  - Closed pieces inflate by mapping back to their own second moments; area restoration gives a needle.
  - Open pieces need their boundary pinned.
  - Flat maps work per geodesic patch round a selection (LSCM, R ≤ 5 mm in 0.1–0.44 s). A whole-hemisphere flat map is out of scope.
  - `splu` with MMD and no pivoting; pyamg does not converge on LSCM.
- **Editing lessons** (PE; CP):
  - Edits are stored as data (world mm, roots) that replay exactly in any order. A vertex drag must be saved as that vertex.
  - An edit reaches about 1 mm, so the base matters most, and an editor needs a worklist over every doubtful site.
  - A surface built as heights over crossing columns tangles. A contoured base cannot cross itself.
  - Send binary arrays, not VTK.

### Standing decisions

- The viewer reads only published outputs.
- Looking comes first; which correction tools to build waits until the data can be seen well.
- Review queues offer a few concrete options per site with previews, not a free-form editor. Answers are saved as spec'd edits.

### Open

- Overlays, mesh outlines and the 3D tile on the brick viewer.
- Placing points from the volume pane.
- A camera shared between windows.

## 7. Code worth carrying over

These are verdicts for a restart. Most tests are exact: phantoms, brute force, reference solvers, device equality.

| Verdict | Modules |
|---|---|
| Port as is | `backend`, `bids` (strict sidecars, spec docs tested against code), `gifti`, `timing`, `_jit`, `mesh` (Kuhn marching, edge keys, Morton blocks: the most reusable piece), `mesh_intersect`, `phantoms`, `handles` (hemisphere scale), `wm_edits`, `decimate`, `viewer.flatmap`, `viewer.inflation`, `viewer.sections`, `scripts/check`, `scripts/affected-tests`, `scripts/memguard` |
| Port with changes | `blocks` (add the chunked source; never read large inputs whole), `mesh_torch` (block-local tables), `mesh_smoothing` (keep normals, edges, pieces), `mesh_graph` (local form; fix the bug below), `white` (split into dseg, surface and overview; drop the box `white_mesh`), `antialias` (band only, multigrid), `streamlines` (integrators and tensor tools in blocks), `column_front` (per mesh block; `past_crest` byte for byte), `flatten` (blocked solve), `viewer.server`, `viewer.overview`, `viewer.white_view`, `viewer.surface` |
| Box scale only | `fronts`: correct, but a sequential heap on whole arrays. Don't build a whole-brain step on it. |
| Rewrite, keep the idea | `column_traces` (root on `white-surf/1`, publish a spec'd output; keep the methods, classes and constants), `pial_labels` (keep the noise estimators; emit spec'd edits as a stream 1 input), `streamlines.fitted_field` (blocked or multigrid), `bands` and `gain` (if stream 3 needs them), `surfcut.qpbo` (needs a non-GPL QPBO), the viewer's `data`, `filters`, `labels`, `streamlines` and page (they read prototype `.npz` files) |
| Drop | `topology` and `wellcomposed` ((26,6) codes), `growth` and `joint` (old seam model), `frame` (move `missing_tissue` to stream 1), the patch pipeline: `columns`, `regions`, `surfaces`, `bottom`, `tv`, `refine`, `output` |

Problems in the current code that a restart should not inherit:

- **A bug:** `mesh_graph.edge_graph` builds a COO matrix and `tocsr()` sums duplicates. An across-triangle edge that is also a mesh edge therefore gets twice its length (2.8 % of entries on a marching-tetrahedra ball). `column_traces` and `pial_labels` use it. #69 has the fix.
- **Four white-mesh constructions.** The restart has one: `white-surf/1`.
- **Streams coupled through code:**
  - `pial_labels` and `column_traces` import each other's outputs.
  - The viewer reads prototype files.
  - `pyproject.toml` collects 14 prototype test files, and the prototypes import `src` modules.

## 8. Interfaces between streams

The specs in `docs/formats/` stay at version 1 until a release. Each is enforced by a `bids.SidecarSpec`, and its document is tested against the code.

| Spec | Defines |
|---|---|
| `wm-dseg/1`, `wm-dseg-fixes/1` | The corrected segmentation and the fixes image (1 cut, 2 fill, 3 cavity, 4 manual); Betti before and after; each move |
| `wm-edits/1` | Voxel-row diffs against a SHA-pinned aseg, with each answer's site and choice |
| `white-surf/1` | GIfTI with external data: float32 vertices in scanner mm, int32 triangles, float32 normals, int64 edge keys, a block index in Morton order |
| `white-overview/1` | The decimated display mesh, its true-vertex ids and the true vertices each face carries; five guarantees |

There is no spec yet for:

- column traces (stream 2's output, which stream 3 and the viewer need);
- pial-label changes;
- the missing-tissue mask;
- any stream 3 output, including feedback to stream 1;
- the review-queue sites (`wm-sites/1` is on #81).

## 9. Scaling gaps

From `docs/scaling.md`, in the order needed before a whole-brain run:

1. The midline cut.
2. `dseg` in blocks. Handle removal as "find globally, fix in local boxes" needs a design first: one piece is about 43 GB, and whole-array `dseg` at 50 µm is about 1 TB.
3. NIfTI-Zarr, together with item 2.
4. A band-only multigrid antialias solver: FISTA on the whole volume costs about h⁻⁵. On the hemisphere it takes 1,651 iterations, 16 min on MPS.

Decimation also needs int32 ids and a block-wise first pass.

Gaps beyond that document:

- Streams 2 and 3 exist only at box scale, with whole-box arrays: `column_traces`, `fitted_field` (pyamg), `fronts`, `fit_warp`, `mesh_graph`.
- Normals are computed on the whole mesh.
- `NiftiSource` reads a `.nii.gz` whole.
- Many scales are hard-wired to 0.16 mm voxels: the 0.04 mm sampling, the 1.5-voxel band width, the tapers, the 0.32 mm twicing and the 2-voxel coupling. Restate each as mm or as voxels on purpose.
- Box costs per 141³ on one CPU core: column volume 3–5 min and 6 GB, the single-surface cut 18 GB, joint warp about 10 min and 10 GB, flattening about 6 min.

## 10. Unresolved conflicts in the evidence

These need a decision or a test, not a choice by default.

1. **The equal-volume closed form from white-surface curvature.**
   - PL (#23) adopted it with curvature smoothed at 1 mm: no stria trend.
   - SF F10 lists the same form as failed with curvature at 0.32–0.80 mm.
   - The tests differ in curvature scale, columns and measure. The dependence on scale is unresolved.
   - Normal rays (#47) use it at 0.5 mm, unvalidated.
2. **The best M1/S1 estimator depends on the paths and the score.**
   - The field prior alone on crossing paths gives 0.07–0.12 mm, against 0.26–0.33 mm in other scores.
   - The profile split misses by 0.5–0.8 mm on tensor paths but gets 0.17 mm on normal rays.
   - The crossing path's L and the column arc lengths were never reconciled.
   - Compare estimators only on one shared set of paths with one score.
3. **Radial fibres.** At 160 µm the texture is planar, so no source shows radial fibres giving column orientation. Their fade-out depth can still be a planar laminar edge.
