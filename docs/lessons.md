# Lessons for the restart

## The approach

Most tools define a white surface and a pial surface, then treat the cortex as everything between them. We do not.

- **There is no true GM/WM boundary to find,** so we build only an approximate white surface. It may sit somewhat inside the GM or the WM.
- **The white surface exists for its normals.** We ask that its normals align with the cortical normal, so that columns traced from it run along the cortex's normal. That is how white-surface methods are judged, not by where the surface sits.
- **Everything along a column is read from the image.** The contrast along the columns identifies the laminae, the GM/WM transition and the pial edge of the cortex. Each is aligned across columns after the fact, like any other layer. There is no separately built pial surface; the pial edge is a level of that alignment.

**Why other tools take the two-surface frame, and why we can drop it.**

- **In vivo tools have no laminar contrast.** Surface tools (FreeSurfer, CAT12, CIVET) work at about 1 mm, where the two tissue-class boundaries are the only signal. In vivo laminar tools (LayNii; Nighres' volumetric layering, Waehnert 2014) work at about 0.5–0.8 mm, with no laminar contrast to read, so they impute layers from geometry between two boundaries. For all of them the frame is a consequence of the data.
- **Ex vivo MRI pipelines inherit that frame.** Edlow 2019 and purple-mri segment GM and WM and build FreeSurfer-style surfaces. Edlow's 100 µm data shows laminar contrast, but, as far as we know, the pipeline does not use it to define the cortex.
- **Our data has laminar contrast,** so the image can decide what other tools must impute.

**Precedents.**

- FreeSurfer's `mris_make_surfaces` already finds the pial edge from intensity along each vertex normal, but it then builds a surface and treats the cortex as the region between surfaces. The mechanism has precedent; the strategy does not.
- BigBrain (Wagstyl 2018, 2020; histology at 20 µm) has strong laminar contrast and still builds both surfaces first, reading layers between them. Its network nonetheless placed the layer 6/WM boundary away from the tissue-class white surface, which supports the premise that there is no single true GM/WM boundary.
- Not checked: whether any ex vivo or histology work reads the pial edge from profiles without a prior surface. The earlier survey did not cover every ex vivo pipeline.

## About this document

What the work so far has established, for a clean start.

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
- Most stream 2 and 3 results come from **boxes**: 141³-voxel crops (about 22.6 mm on a side) cut from the hemisphere volume round a test site (V1V2, M1S1, PCC_SPL). A **box face** is one of a box's six bounding planes, where the data is cut off. The results were checked against the user's marked points: M1/S1 23, V1V2 35, PCC pieces 22 and 37.
- Only stream 1 has run on the whole hemisphere.

## 1. Goals, constraints and working rules

### Goals

The work is split into four streams, each a tool:

1. WM segmentation and the white surface (§3)
2. Columns (§4)
3. Laminar profiles (§5)
4. The viewer (§6)

Each stream's section gives its goal, its high-level design, its implementation design (for each design step, its inputs and outputs, then its details and lessons) and its implementation plan.

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
- **The WM label comes from our own segmentation,** which gives labels in the FreeSurfer convention and P(filled WM) (§3). The surface steps require both; there is no labels-only path. The label is still an input with its own errors: take evidence from the echoes, not from the quirks of the labels.

### Working rules

- **Test, don't patch.** Diagnose first, then write a test with an exact answer that fails, then fix. Never move a threshold to hide a symptom.
- **Diagnostics over criteria.** While a method is exploratory, design each checkpoint around views the user can read, not pass/fail numbers. The default image view is echo 2 with the depth trend removed.
- **Keep principle, role and estimator apart.** Record each result against one estimator in one role. A failed estimator is not a failed principle.
- **Feasibility, not speed,** for prototypes and interactive tools. Correctness checks still apply.
- **Judge efficiency by analysis, not by wall time on the laptop.** For each algorithm, state:
  - its computational complexity in the sizes that grow (voxels, vertices, columns, candidate pairs);
  - its memory traffic: bytes moved per element, the working set per block, and whether it is bound by bandwidth or by compute.

  Evaluate both at whole-brain 50 µm on the target hardware. Laptop timings are supporting context only. A step that is quick on a 160 µm hemisphere can still be the wrong design if its cost grows badly, and a slow one can be right.
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

**Domain edges: what the boxes taught**

The restart does not work on boxes. Two things from them still matter:

- **Read the box numbers with their edge bias in mind.** Most stream 2 and 3 evidence comes from boxes, and 65–99 % of column failures and a quarter to two thirds of contacts lay at box faces (SF F26, F27). The numbers here are evidence about methods; expect them to change on the hemisphere.
- **Blocked processing has the same edge effect at every block boundary.** Near any cut in the data, columns run out and fronts from beyond the cut never arrive, so their neighbours meet in their place. On the boxes it took regions at least 6 mm from every face, and leaving out roots within 2 mm of a face, to keep results clean. So a block's halo must reach as far as anything that crosses it: a column (up to the 8 mm length cap), a front, a smoothing kernel. Otherwise the result depends on where the block boundaries fall.

**Identity, topology and precision** (code; #61)

- **One tetrahedralization: the Kuhn split.** Voxel topology uses Kuhn adjacency (14 neighbours, the same for object and background), so the voxels' topology equals the mesh's. Do not use the (26,6) or (6,26) codes: their answers can disagree with the mesh.
- **Vertex identity is the grid-edge key:** lin(p) × 7 + (4 s_i + 2 s_j + s_k − 1), in int64. It survives remeshing, so anything stored per vertex keys on it.
- **The block layout:** the grid starts at voxel 0, with 64-voxel blocks in Morton order (i at bit 3b+2, j at 3b+1, k at 3b). A vertex is owned by the first block that uses it.
- **Precision:** mesh steps run in float32, with float64 only for the affine, area sums and intersection tests. Torch kernels use no float64, no CSR and no atomics, so each run gives the same result. An unavailable device is an error.
- **Manual and feedback inputs** are replayable inputs, applied before the automatic step and never baked into the source (`wm-edits/1`, #78).

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

### 3a. Goal

A closed surface without handles near the grey/white-matter boundary of each hemisphere (the "white surface"), at the acquired resolution, whose normals are the cortical normals.

### 3b. High-level design

1. **Segment the WM.** Our own nnU-Net segmentation of the FLASH gives labels in the FreeSurfer convention, a probability of filled WM and its spread across folds. It is trained in generations through human review, starting from the purple-mri aseg as seed labels.
2. **Correct the topology (`dseg`).** Apply the edits, fill the WM mask, then remove handles and cavities and repair tight folds, automatically.
3. **Make the surface (`mesh`).** Antialias the corrected mask and contour it by blocked marching tetrahedra into `white-surf/1`.
4. **Make an overview for display (`overview`).** A decimated mesh that maps every pick back to true vertices.
5. **Review.** Gather the places that need judgment into one queue per hemisphere. Each answer becomes an edit for step 2 and a training label for step 1.

**The flow runs one way. Feedback enters only as inputs.**

- The flow: input label + edits → `dseg` (automatic correction, with a fixes image) → white surface.
  - No later step writes back into the input label or into `dseg`. The surface stage never changes the label it read.
  - The automatic fixes (cuts, fills, thickening) stay in the derivative and its fixes image. They never become part of the input.
- The loop closes through inputs. A person's answer, or later stream 3's feedback, becomes an edit: the verified label of each voxel it covers, valid for any input. Stream 1 then reruns from the start.
  - Edits are kept apart from the automatic changes, so they apply again when the correction changes.
  - `DecidedAgainst` records the SHA of the `dseg` the person looked at, so a rerun can list the answers whose context changed (`wm-edits/1`, #78).
- Steps 2–4 must not depend on which generation produced the label.

**Standing decisions**

- **The white surface exists for its normals, not its position.** Columns start on it and run along its normals. There is no pial surface and no "true GM/WM boundary": the GM/WM transition is aligned afterwards like any other layer.
  - Judge methods by normal quality: neighbour-angle spread, misfacing faces, cancellation in thin blades, and agreement with orientation from the image.
  - The surface may move into GM or WM to make normals well defined, as long as it keeps the topology and does not self-intersect.
- The input label is never rewritten. Corrections are derivatives, and manual edits are replayable inputs applied before the automatic correction.
- The generator fixes topology automatically, with no question per handle. Review refines a result that is already correct.
- WM may grow into cortex: thin gyral WM is thickened, not removed, as FreeSurfer's mri_segment does.
- The meshing uses no cortex label. Every white-mesh vertex is a column root, and what columns pass through is for later stages to judge.

### 3c. Implementation design

Measured numbers in this section come from the one hemisphere we have, segmented by purple-mri, whose aseg bootstraps our model. Read them as evidence about methods: our own labels may show other defects. That aseg's own errors are in §2.

#### Step 1: our own WM segmentation

*Inputs*

| Input | From | Use |
|---|---|---|
| The three FLASH echoes | the scan | the network's channels |
| Seed labels | the purple-mri aseg (bootstrapping only; 3d, phase 2) | weak training labels |
| Verified answers (`wm-edits/1`) | step 5 | training labels |
| Test regions | labelled from scratch (3d, phase 0) | evaluation only, never training |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| `_dseg`: labels in the FreeSurfer convention, missing tissue included | none yet | step 2 (the filled mask) |
| `_label-WMfilled_probseg`: P(filled WM) | none yet | step 2 (where handle walls fall); step 3 (the antialiasing data term); step 5 (wall margins) |
| `_label-WMfilled_desc-foldsd_probseg`: the spread of P across folds | none yet | step 5 (ranking the queue) |
| The model generation, in each sidecar | none yet | tracing a label to its model |

*Design and lessons*

*Aim.* The white surface needs per-voxel evidence from the FLASH itself: a probability of filled WM, improved wherever a person has answered. The network is the means to that, not a goal of its own.

*What it is*

- nnU-Net v2 (Apache-2.0), `3d_fullres` with a residual-encoder preset (M or L), trained on the three FLASH echoes as channels.
- It is trained in generations, through continuous human-in-the-loop review. Each generation segments the hemispheres, the review queue corrects its output (step 5), and the verified answers join the training labels for the next generation. No number of hemispheres or generations is fixed in advance. This needs:
  - held-out test regions that never enter training, so each generation can be compared with the last;
  - the model generation recorded in each output's sidecar, so a label can be traced to the model that made it;
  - a choice, at each generation, of whether to rerun earlier hemispheres with it. A rerun changes everything downstream of the label.
- **Bootstrapping.** The first generation trains on the purple-mri aseg as weak seed labels, and verified labels gradually take their place (3d, phase 2). Until the first generation exists, the surface steps are built and tested on that aseg.

*Probabilities*

- Steps 2 and 3 require labels and P; there is no labels-only path. Step 5 uses the fold spread when it exists; during bootstrapping it ranks without it.
- During bootstrapping, P is a stand-in from purple-mri (3d, phase 1). That lets the surface steps run before the first generation exists, so the surface work does not wait for the network, and the network does not wait for the surface work.

The coupling that is real, decided once:

1. **The label set.** The filled mask is `FILLED_LABELS` (WM, lateral ventricle, caudate, putamen, pallidum, thalamus, hippocampus, CC). The network must output at least these classes, separately from cortex and outside, or the surface changes.
2. **Missing tissue.** Blocks cut out before the scan are labelled as missing tissue, a class of its own (proposed), or ignored in training. Step 2 gives the rule for the surface.
3. **Calibration.** P must be calibrated near 0.5, because step 2 places walls by it. Calibration is a diagnostic of every generation.
4. **The defect count.** The number of handles and cavities before correction sets the length of the review queue (step 5). It is a diagnostic of every generation, not a target to optimise inside the network.

*Decisions taken*

- The network gives evidence. Topology is guaranteed after it, by our own code (`handles`, `antialias`), never by the network.
- Steps 2 and 3 require P as well as labels.
- Only verified labels train later generations.
- Test regions are labelled from scratch and never trained on.
- **Labelling happens in our own viewer** (stream 4): the test regions, the site answers and the scribbles. It reads the blocked formats we rely on (NIfTI-Zarr volumes, `white-surf/1` with its block index, the overview), so a person can label anywhere in a whole brain at full resolution, against the surface. ITK-SNAP and 3D Slicer load whole volumes into memory and read neither format, so at 50 µm they would only ever see exported crops. The viewer also writes answers directly as spec'd edits (`wm-edits/1`). ITK-SNAP stays only as an escape hatch for free painting on an exported crop.
- **nnU-Net is an external tool, not a package dependency.** Only the export and import modules know about it: the export writes nnU-Net's raw dataset format, nnU-Net is called on it (on the cluster for training), and the import reads its predictions back into spec'd outputs.
- Training runs on a compute cluster with CUDA GPUs; how to reach it is to be settled when training starts. Inference and data preparation can run on the laptop.

*Literature* (from memory; check before citing)

- nnU-Net v2 and residual encoders: Isensee et al., *Nature Methods* 2021; "nnU-Net revisited", MICCAI 2024.
- Sparse labels with the ignore label: Gotkowski et al., "Embarrassingly simple scribble supervision for 3D medical segmentation", 2024.
- Interactive refinement: Wang et al., DeepIGeoS (TPAMI 2018) and BIFSeg (TMI 2018); MONAI Label (Diaz-Pinto et al., *Medical Image Analysis* 2024); nnInteractive (arXiv 2503.08373).
- Topology correction from probabilities: Han, Xu and Prince (topology-preserving level sets); Bazin and Pham (fast-marching topology correction, TOADS); Ségonne et al. (FreeSurfer's defect correction).
- Distance outputs: Xue et al., AAAI 2020; Navarro et al., MLMI 2019; Ma et al., MIDL 2020; for cortex, SegRecon (Gopinath et al. 2021), DeepCSR (Cruz et al., CVPR 2021), recon-all-clinical (Gopinath et al. 2024).
- Uncertainty: Kendall and Gal, NeurIPS 2017; β-NLL, Seitzer et al., ICLR 2022.
- Topology losses: Hu et al., NeurIPS 2019; clDice (Shit et al., CVPR 2021); Betti matching (Stucki et al., ICML 2023).
- purple-mri, the bootstrap labels: Khandelwal et al., *Imaging Neuroscience* 2024; the purple-mri repository (topology notes and surface scripts).

#### Step 2: topology correction (`dseg`)

*Inputs*

| Input | From | Use |
|---|---|---|
| `_dseg` and P | step 1 | the filled mask; handle removal ordered by P |
| `_desc-wmedits_edits.tsv` and sidecar (`wm-edits/1`) | step 5 | edits applied before the correction |
| The FLASH | the scan | fixes chosen by intensity, only if adopted (Open) |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| `_desc-wmtopo_dseg`: the corrected labels; filled mask with Betti (b0, 0, 0) | `wm-dseg/1` | step 3; step 5; stream 2 (masks and stops) |
| `_desc-wmtopofixes_dseg`: 1 handle cut, 2 handle fill, 3 cavity fill, 4 manual edit | `wm-dseg-fixes/1` | step 5; the viewer |
| Fills inside missing tissue as a value of their own; each fix's margin in P (planned) | none yet | step 5 |

*Design and lessons*

- **Missing tissue** (PL). Blocks of tissue were cut out of the specimen before the scan, and the FLASH is black there. In V1V2 the missing tissue is 14.6 % of the cortex label.
  - It can be found from the image: FLASH below 0.25 × the cortex median after smoothing by 2 voxels, in blocks of at least 250 voxels, grown by 0.3 mm. This detector supplies the class for the seed labels, since the purple-mri aseg labels the blocks as if the tissue were there.
  - It is an obstacle for columns, not a source.
- **The filled mask** (#61, #66, #67). Aseg 41 plus 43, 49–52, 53 and 192 (right hemisphere; 2, 4, 10–13, 17 and 192 left), with cavities filled. The deep structures (ventricle, thalamus, ...) and the hippocampus belong in the mask. The code's `FILLED_LABELS` still leaves the hippocampus out and must add it (3d). The surface results in this section were measured without it. Missing tissue counts as outside (rule below). A voxel cut from the mask takes the label of the nearest (mm) voxel outside it.
- **The rule for missing tissue** (proposed). The cut face is treated as a domain edge: close the surface there, invent nothing, and mark exactly which vertices lie on it.
  1. Missing tissue is a published stream 1 label.
  2. For the filled mask, missing tissue counts as outside. The mask stops at the cut face, and the surface runs along it.
  3. Handle removal and the cavity fill then run as usual, and fixes may fill into missing tissue, by the usual rule (whichever changes fewer voxels). A plug where there is no data alters nothing real, whereas forbidding it would force cuts through real WM. Such fills are recorded in the fixes image as fills inside missing tissue. Missing tissue enclosed by WM cannot be a real excision, since excisions open to the outside, so filling it is right in any case.
  4. Each vertex lies on one grid edge (its edge key), joining a mask voxel to a non-mask voxel. The vertex is on the cut face exactly when that non-mask voxel is missing tissue. This flag needs no threshold and is published per vertex, keyed by edge key. Plugs are flagged by the same test.
  5. Stream 2 roots no columns on flagged vertices. Columns from nearby vertices that run into missing tissue stop there, as at any obstacle.
- **The guarantee still holds.** The mesh of any voxel mask is closed, and `dseg` refuses to write unless the Betti numbers are (b0, 0, 0). An excision that cuts a gyrus clean off makes another piece, which `wm-dseg/1` already allows: one or more closed, genus-0 pieces. What changes is that part of the surface is not grey/white, and it is flagged exactly.
- **Handle removal** (`handles.remove_handles`; #38, #48, #67):
  - Topology uses Kuhn adjacency, so it equals the mesh's (§2).
  - Object and background fronts grow by simple points. Each handle's wall is cut or filled, whichever changes fewer voxels; ties go to the cut, then to the lowest voxel index.
  - Walls that block each other are claimed as sets (Mayer–Vietoris), or voxels are stranded.
  - Fills of one piece can touch another piece, so check the union's Betti numbers and refuse to write unless b1 = b2 = 0.
  - On the hemisphere: Betti (1, 111, 182) → (1, 0, 0) in 3.4 min at about 22 GB. Most moves are 1–2 voxels.
  - The meeting wall need not be the smallest cut: one cut of 2,047 voxels ran along a thin strand.
  - Batching walls per pass changes the result, so don't.
- **Handle removal by probability** (planned; plan phase 1). The fronts claim voxels by P, not by distance to the other phase as the current code does: the object front claims in order of decreasing P and the background front in order of increasing P, so each wall falls where P is closest to 0.5. That is right only if P is calibrated near 0.5. Each fix records its margin, the P along its wall, and the review queue ranks by it. P matters to the surface mainly here: it decides which side of a fused bank a cut falls.
- **Tight folds (misfacing faces)** (#81, #85). On the hemisphere the surface has 152 at 91 sites. Diagonal-only contacts in the label cause most of them: 74 of 90 sites have a checkerboard or corner-only contact within 4 voxels.
  - Thicken to 2s, widen gaps between surfaces, then make the mask well-composed, at s = 0.30 mm. This takes 90 sites to 5 and edges over 60° from 297 to 4, at a cost of +3.9 % area and a 90th-percentile neighbour angle of 5.4° against 3.9°.
  - s must not be a whole number of voxels along any axis: at 0.32 mm (exactly 2 voxels on one axis) the checkerboards rise from about 390 to 7,000. 0.26 mm is worse on every count; above 0.30 mm the bulk and area grow.
  - Synthetic antialiased phantoms make no misfacing faces, so fold tests need hand-built folds plus a hemisphere run.
  - Doesn't work: forcing cuts, with no fills into cortex (90 → 131 sites); voxel opening at 0.24–0.40 mm (worse at every radius, grows the area, cascades over passes).
  - Relaxing the level set needs a local "no new blade or gap voxel within s" test, or the thin-plate energy shrinks widened gaps and blades into slivers. With the test it keeps 95 % of the gain on crops, but misses tubes.

#### Step 3: the surface (`mesh`)

*Inputs*

| Input | From | Use |
|---|---|---|
| `_desc-wmtopo_dseg` | step 2 | the filled mask; missing tissue for the cut-face flag |
| P | step 1 | the antialiasing data term |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| `_hemi-<h>_white.surf.gii` (+ `.dat`): vertices and triangles | `white-surf/1` | step 4; stream 2 (roots); stream 3 (column graph); the viewer |
| `_desc-normals_white.func.gii`: unit normals | `white-surf/1` | stream 2 (ray directions) |
| `_desc-edgekeys_white.func.gii`: each vertex's grid-edge key | `white-surf/1` | every per-vertex output downstream |
| `_desc-blocks_white.tsv`: the block index | `white-surf/1` | blocked readers downstream |
| The cut-face flag per vertex, keyed by edge key (planned) | none yet | stream 2 (no roots there) |

*Design and lessons*

- **Antialiasing beats smoothing** (#61; SF F28).
  - A thin-plate field φ with φ ≥ ε on the mask and φ ≤ −ε off it, in a band of the signed distance, solved by FISTA with restarts in PyTorch.
  - Each vertex stays on its own grid edge, so the surface has no self-intersections by construction and keeps the edge keys.
  - A tilted plane's normals come out within 0.5° (the raw staircase is 31° off).
  - On the hemisphere: one closed genus-0 piece, 12.7 M vertices, 25.3 M faces, no self-intersecting faces; CPU and MPS identical bit for bit; 1,651 iterations, 16 min on MPS, peak 20–24 GB.
- **Antialiasing with probability** (planned; plan phase 1). Besides thin-plate smoothness, a data term pulls φ near the boundary towards a smooth function of P − 0.5, weighted by a per-voxel confidence (1/σ², once σ exists). The sign constraints stay, so the topology and the freedom from self-intersection stay guaranteed. This smooths the staircase where P carries sub-voxel information; position finer than a voxel is a by-product, not a goal.
- **Smoothing is the wrong tool here:**
  - Smoothing the label and then contouring loses 2-voxel WM blades: at σ = 1 voxel it loses 16 % of V1V2's thin WM, shrinks convex surfaces by about σ²κ, and makes slivers.
  - Taubin λ|μ smoothing is unstable at short cutoffs. Tukey twicing never amplifies.
- **Limits of any binary label.** Its terraces limit curvature and normals below about 1–2 mm on surfaces near a grid axis, whatever the method. Thin-blade normals keep as little as 0.56 of their length after averaging (`docs/scaling.md`); the column stage has to handle them.
- **Normals** point out of the WM: area-weighted, then 10 smoothing steps.
- **Curvature** (SF F28): the 0.32 mm mesh with curvature at 0.48 mm agrees with quadrics fitted to the voxel faces (rank correlation 0.85–0.98), with about half their jitter.
  - Never take curvature from the Hessian of a smoothed signed distance: it mixes the two sides of thin blades.
  - Valence-weighted smoothing biases Gaussian curvature. Use smoothing that keeps mass.
- **Doesn't work** (both from SF F28, which tested the surface as the start of a front or level set for stream 2; the smooth surface there was a mesh smoothed by twicing at 0.32 mm, which the antialiased surface now replaces):
  - **Refining the grid instead of smoothing the boundary.** The idea was to get sub-voxel accuracy by running a level set on a grid twice as fine (80 µm), started from the signed distance to the label's own voxel faces (or to the binary contour). It fails because the label's half-voxel steps are in the boundary data, and a finer grid reproduces them faithfully:
    - 6–11 % of the points within 2 voxels of the surface got the wrong inside/outside sign: they lie between the true surface and the label's steps.
    - The closest-point (foot) map jumped by more than 2 grid steps between neighbouring points at 7–11 % of pairs, so neighbouring points were assigned roots far apart.
    - The signed-distance error kept a spread of 0.05–0.08 mm on both grids: refining did not shrink it.

    A smooth surface, by contrast, gave no wrong signs, a foot-map jump p95 of 1.05 steps, and a spread of 0.013 mm. A finer grid only reduces the solver's own error, so it is worth having only once the start surface is accurate below a voxel.
  - **Taking inside/outside from the closest face's normal.** A signed distance to a mesh needs the sign of each point: inside or outside. Reading it from the normal of the face that holds the closest point fails when the closest point lies on an edge or a vertex, which is common on a staircase-like surface: on the 2× grid, points sat exactly in a face's plane, the dot product with its normal was 0, and they got distance 0 with no sign. Use the angle-weighted pseudonormal of the closest feature (face, edge or vertex; Bærentzen and Aanæs 2005), which gives the right sign on any closed mesh. An exact test covers this case.

#### Step 4: the overview (`overview`)

*Inputs*

| Input | From | Use |
|---|---|---|
| `white-surf/1` | step 3 | the mesh to decimate |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| `_desc-overview_white.surf.gii`: the decimated mesh | `white-overview/1` | the viewer |
| `_desc-overviewids_white.func.gii`: the true vertex of each overview vertex | `white-overview/1` | the viewer (picks) |
| `_desc-overviewcarry_white.func.gii`: the true vertices each face carries | `white-overview/1` | the viewer (mapping back) |

*Design and lessons*

- `decimate` (#70–#77): 25.3 M → 210 k faces at ε = 0.1 mm in 186 s, keeping topology, the ε bound, unmoved true vertices, orientation and no self-intersections. The Numba block worker is bit for bit the Python reference.
- The fold test must compare against the input normals each face stands for, not the face's last state; otherwise 1,027 faces turned over.
- Pairs of sheets closer than 2ε at necks cross. Pinning the crossing faces' input vertices and re-decimating clears them.
- Map anything back to true vertices with the decimation's own carried-vertex map. A nearest-face search lands on the wrong side of thin sheets.
- For display only: nothing is measured on it.

#### Step 5: review

*Inputs*

| Input | From | Use |
|---|---|---|
| Corrected labels and fixes image | step 2 | finding sites |
| P and fold spread | step 1 | finding and ranking sites |
| The FLASH, the surface, the overview | the scan; steps 3 and 4 | display in the viewer |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| The sites in the queue | `wm-sites/1` (#81) | the viewer's review pane |
| The answers: `_desc-wmedits_edits.tsv` and sidecar | `wm-edits/1` | step 2 (edits); step 1 (verified training labels) |

*Design and lessons*

The design follows `wm-edits/1` (#78), the review plan in #80 and the site finder in #81.

- **The queue.** The pipeline finds the sites that need judgment and gathers them into one queue per hemisphere, after the automatic result:
  - tight folds (`wm-sites/1`) and large handle fixes;
  - the topology walls with the smallest margin in P;
  - the places with the largest fold spread, where the model is least sure.
- **Each answer serves twice:**
  - as an edit that corrects this hemisphere now;
  - as a verified label for training the next generation.
- **Answers stand on their own.** Each is the verified label of each voxel it covers, valid for any input, not a diff against one segmentation. That way answers survive from one generation to the next. Each records its site (if any), choice, time and the SHA of the `dseg` the person looked at. An answer with no voxels (`leave`) is recorded too, so the site counts as answered.
- **Two kinds of answer:**
  - **Site answers:** at a site from the queue, the viewer offers a few concrete choices with previews: `remove-strand`, `thicken`, `fill-gap`, `widen-gap`, `leave`, `manual-paint`. Each choice is a function of the current label that returns the voxels it changes. ITK-SNAP, on an exported crop, is the escape hatch for free painting.
  - **Scribbles:** a few strokes marking voxels as WM or not WM, wherever the reviewer sees an error, including errors no site finder flags. Everything unmarked stays unlabelled, so nnU-Net trains on the strokes with its ignore label. Scribbles cost little per answer.
- Only verified labels train the model. Automatic fixes and the model's own predictions never do.
- The held-out test regions are labelled from scratch, apart from the queue.

#### Open

- Fixes chosen by intensity (FreeSurfer's cut or fill by intensity; #85 step 2); the tube case; the size threshold for exceptions; whether fills through cortex are as acceptable as thickening.
- Missing tissue: how far the cut face bends the normals of vertices near it but not on it, and whether stream 2 should use distance along the mesh from flagged vertices; whether the viewer needs the cut face as a surface label of its own; how much the detector's errors (dark vessels or fluid taken for missing tissue) matter now that the mask follows it.
- Whether T = 0 should come from the image's own WM contrast rather than the label. BigBrain puts the L6/WM boundary superficial to the tissue-class white surface (CL).
- Finding hidden fluid gaps from the image before columns are traced: a thin, slowly decaying dark strip inside labelled cortex (SF F21).
- **How `dseg` applies standalone answers** to a new generation's labels: whether a verified voxel always overrides the input, and how to show the reviewer where a new generation disagrees with an earlier answer.
- **Whether a scribble also corrects the current hemisphere,** or only trains the next generation. A few strokes give no full-voxel edit, so the current correction needs something to spread them: a rerun after fine-tuning, or an interactive model. Candidates (from memory; check before choosing):
  - **nnInteractive** (DKFZ, 2025): a promptable 3D model from the nnU-Net team, taking points, scribbles, boxes and lassos, with no retraining. Code Apache-2.0; the weights are believed to be non-commercial and share-alike (CC BY-NC-SA), which must be read before use.
  - **MONAI Label with DeepEdit** (Apache-2.0): a U-Net with click inputs, whose server can train in the background as labels are saved. Fully permissive, but a looser fit with nnU-Net.
  - **VISTA3D** (NVIDIA, via MONAI): automatic and interactive segmentation in one model; weights believed non-commercial.
  - 3D SAM variants (SAM-Med3D, SegVol): general-purpose and weak on fine structure; not a first choice.

  Since labelling happens in our viewer, an interactive model would run on the viewer's server, not inside Slicer or ITK-SNAP. Whichever spreads a scribble, its output is the model's guess, not a verified label. Only the strokes, and what the reviewer explicitly accepts, train the next generation.
- Whether an automatic fix that a reviewer accepted (`leave`) counts as verified, and so trains the model.
- How stream 3's feedback (for example, the GM/WM transition found along columns) becomes a stream 1 input. Does it edit the label, or adjust the surface stage?
- **The bootstrap's stand-in P.** If purple-mri cannot save its probabilities, the surface steps need another stand-in P before the first generation exists.
- Open decisions for our own segmentation:
  - Missing tissue: a class of its own (proposed, with the rule in step 2) or ignored in training.
  - How stream 2 treats roots on the part of the white surface that wraps the hippocampus, which is not neocortex: a medial-wall label, or roots kept and judged later.
  - **How fast a generation follows a review.** nnU-Net can start from the previous weights (`-pretrained_weights`) with a short custom schedule, sampling patches mostly round the new verified labels and replaying the older labels so the model does not forget them. On a CUDA GPU that is hours, not days, so a generation could follow each review session. To choose: the schedule length, and how much replay is needed (measured on the test regions).
  - **Whether a different network would do better.** Probably not by architecture: in "nnU-Net revisited" (MICCAI 2024) careful benchmarking erased most claimed gains of transformer and Mamba models (Swin UNETR, nnFormer, U-Mamba); MedNeXt was on a par. The bottleneck is labels. Worth testing once there are several hemispheres: self-supervised pretraining of the residual-encoder nnU-Net on all our unlabelled FLASH volumes (DKFZ has tooling for this; check its state and licence).

### 3d. Implementation plan

Three tracks that can run in parallel: the surface steps, our own segmentation, and the review loop. Until the first generation exists, the surface steps run on the purple-mri aseg, the bootstrap labels.

**Surface steps (2–4)**

- Add the hippocampus (53 right, 17 left) to `FILLED_LABELS`, with a test that it is filled.
- Port as is: `handles` (hemisphere scale), `wm_edits`, `decimate`, and the core it needs (§7).
- Port with changes: `white`, split into `dseg`, surface and overview modules plus readers, without the box `white_mesh`; `antialias` band only.
- Finish the fold repair of #85: intensity-chosen fixes (step 2 there), the relaxation on the hemisphere with the blade/gap test, the tube case, the exception size threshold.
- Make missing tissue a published label, and the cut-face flag a published per-vertex output keyed by edge key (neither has a spec yet). Then apply the missing-tissue rule in `dseg`, with phantom tests: an excised block that leaves a tunnel (fixed by a plug, flagged), one that cuts a piece off (b0 rises), and enclosed missing tissue (filled).
- Rewrite `pial_labels` as a stream 1 input: keep its noise estimators and detector, and have it emit spec'd edits that relabel fluid channels and vessels that the label marks as cortex, which crest methods in stream 2 need (§4c, step 2).
- Scale to whole brains, in this order (`docs/scaling.md`):
  1. The midline cut (the corpus callosum joins the hemispheres).
  2. `dseg` in blocks: blocked connected components with a merge, Euler sums per block, the cavity fill, and handle removal as "find globally, fix in local boxes". That last part needs a design and a prototype first: one piece is about 43 GB, and whole-array `dseg` at 50 µm is about 1 TB.
  3. NIfTI-Zarr volumes, together with item 2.
  4. A band-only multigrid antialias solver with a fused kernel: FISTA on the whole volume costs about h⁻⁵; the target is about h⁻² per cycle.
  5. Decimation with int32 ids and a block-wise first pass.
- GPU, by cost: antialias first, then the normal-smoothing tables, then mesh and volume connected components. Handle removal, cut relabelling, decimation and I/O stay on the CPU.

**Our own segmentation (step 1).** Each phase is one or more PRs. Phase 1 does not depend on phases 2–4.

- **Phase 0: data, label set, test regions.**
  - List the FLASH hemispheres that exist and are expected, with protocol, resolution and missing blocks. The answer decides how much phase 6 is needed.
  - Fix the label set (FreeSurferColorLUT): outside, WM, cortex, the `FILLED_LABELS` deep structures, hippocampus, missing tissue, and other labels only if a later stage needs them.
  - Test regions: a few boxes per hemisphere (start with V1V2, PCC_SPL, M1S1, and the V1V2 seam review points). They are labelled from scratch, not by correcting the seed labels, so the seed does not anchor them, and they are never used in training. Format: a BIDS `_dseg` with a versioned sidecar spec that records the box and the labeller.
- **Phase 1: the surface steps read probabilities.** Independent of the network; exact-answer tests on phantoms first.
  1. `handles.remove_handles` takes a required priority volume, P: the object front claims in order of decreasing P and the background front in order of increasing P, so the walls fall where P is closest to 0.5. Distance ordering goes. Phantom test: a handle whose bridge has a known low-P place away from its thinnest place; the cut must fall there.
  2. `antialias` takes a required data term: near the boundary, φ is pulled towards a smooth function of P − 0.5, weighted by a per-voxel confidence. The sign constraints stay, so the topology and the freedom from self-intersection stay guaranteed. Phantom test: a plane at a known sub-voxel offset, with P a blurred step across it; the vertices must recover the offset.
  3. Each fix in `desc-wmtopofixes` records its margin, the P along its wall. The review queue ranks by it.
  4. Bootstrapping: a stand-in P on the real hemisphere until our own exists: purple-mri rerun with its probabilities saved (nnU-Net `--save_probabilities`), summed over `FILLED_LABELS`. Ask the purple-mri authors whether their container can do this.
  5. Diagnostics, P ordering against distance ordering on the hemisphere: where the walls move, the normal metrics, and views of the walls that move most.
- **Phase 2: training data.**
  - Export to nnU-Net's raw dataset format: the three echoes as channels `_0000`–`_0002`, with labels that use an ignore label.
  - Bootstrapping: purple-mri's labels as weak seed labels. Its labels in the interiors; the missing blocks from the step 2 detector; ignore in a band of about 2 voxels each side of every purple-mri boundary, and where a simple FLASH check disagrees with purple-mri (the check to be chosen on the test regions).
  - The existing `wm-edits` answers as verified labels, never ignored.
  - Verified labels are kept apart from seed labels, with provenance, so a generation can be retrained with or without the seed.
  - The export is a package module (`baillarger.wmseg`) with its own format spec. nnU-Net stays an external tool, called on the exported dataset.
- **Phase 3: the first generation.**
  - Five folds, split by hemisphere when there are several, by box otherwise. A split by box makes the folds' accuracy an upper bound.
  - Inference writes the `_dseg`, the WM-filled `_probseg` and the fold spread, with sidecars.
  - Diagnostics against the purple-mri seed labels as a baseline, on the test regions and the hemisphere: boundary distance and Dice on the test regions; calibration of P near 0.5; handles and cavities before correction; the phase 1 walls and their margins; views of where ours and purple-mri's differ most.
- **Phase 4: the review loop.** As in step 5. Each generation is retrained from the previous one's weights with the verified labels added. Its diagnostics are on the test regions only: those of phase 3, plus the length of the queue.
- **Phase 5: uncertainty.**
  - Two estimates that need no new network output:
    - the fold spread, which is epistemic: what more labels can fix;
    - σ from a calibrated P and a distance, which is aleatoric: what the image leaves open. If the boundary distance d at a voxel is Gaussian, d ~ N(μ, σ²), then P(inside) = Φ(−μ/σ).
  - Uses:
    - Rank the queue. A high-spread site is a good training label; a high-σ site is a judgment and is recorded as one.
    - Weight the antialias data term by 1/σ².
    - Give each white-surface vertex a position uncertainty.
  - Diagnostic: does σ predict the boundary errors on the test regions?
- **Phase 6: only if the diagnostics ask for it.**
  - A signed-distance output (capped at about 5 voxels, L1, trained only where every voxel within the cap is labelled), or a Laplace head with β-NLL, if the σ derived from P is poorly calibrated or saturated where it matters.
  - Synthetic training images from label maps (SynthSeg, PostSynth), as augmentation, if the problem is generalising to other hemispheres or protocols.
  - A topology loss (clDice, Betti matching), if the queue is too long. It can only reduce defects; the guarantee stays in phase 1's decoding.

**Review loop (step 5)** (the steps of the #80 plan, with #81 and #82 as its first PRs)

1. The site finder, written as a derivative with its own spec (`wm-sites/1`, #81).
2. The volume pane in `--white` mode (#82).
3. The review pane, with site answers and scribbles.
4. The ITK-SNAP escape hatch: export a crop, import the painted voxels as an answer.
5. `wm-edits/1` edited so that answers stand on their own.

## 4. Stream 2: columns

### 4a. Goal

A complete ray from every white-surface vertex through the cortex, to fluid or into the far bank's WM, recording what it meets on the way.

### 4b. High-level design

1. **Roots.** One root per `white-surf/1` vertex, keyed by edge key, with rays both inward and outward.
2. **Orientation evidence.** The data normal from the echoes' structure tensor, and a field T fitted to it, for the methods that follow the image.
3. **Trace complete rays.** A small set of simple methods that differ in one respect each and share the rest, including one frozen crossing rule.
4. **Record what each ray meets.** Crests, meetings with other fronts, turns, and the class of each column. Nothing is cut here: the cut is read from the profile in stream 3.
5. **Publish the column traces** in a spec'd format that stream 3 and the viewer read.

**Standing decisions**

- **Columns are complete rays.** They stop only at fluid, far WM, missing tissue, the domain edge or a length cap (8 mm), never where two fronts meet.
- **The crossing rule is frozen** (user, #52; `column_front.past_crest`): climb ∇T; inside T's crest stencil, keep the heading; on leaving it, if T falls ahead, descend −∇T for good. Methods that differ in one respect share it.
- **Column rules stay simple.** A field artefact is recorded in a test, never patched with a new clause.
- Normal rays stay straight (user, #47).
- No step uses the fixes image to decide where to search, or re-finds a boundary as the depth origin.
- Banks are told apart by distance along the white mesh, using the across-triangle graph (`SAME_MM` 5 mm), so the two sides of a thin blade stay apart.

### 4c. Implementation design

#### Step 1: roots

*Inputs*

| Input | From | Use |
|---|---|---|
| `white-surf/1`: vertices, normals, edge keys, block index | stream 1, step 3 | one root per vertex |
| The cut-face flag (planned) | stream 1, step 3 | roots left out |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| Roots keyed by edge key, with position and normal | internal | steps 3–4 |
| The across-triangle mesh graph | internal | distances along the surface (steps 3–4) |

*Design and lessons*

- One root per white-mesh vertex (about one per voxel), except vertices flagged as cut face (§3c, step 2). Rays go both inward and outward from the surface.
- **Distances on the mesh** need the edges across neighbouring triangles: 8.2 % bound on a plane, against 41 % along edges alone (#10, #45).
- **Theory** (SF; shock theory): because the WM is connected, every collision is a front meeting itself. A fold and a contact differ only in how far apart the meeting roots lie along the surface, which is a continuum, so no pointwise quantity separates them.

#### Step 2: orientation evidence

*Inputs*

| Input | From | Use |
|---|---|---|
| FLASH echoes 2 and 3 (trend-removed log) | the scan | the structure tensor |
| `_desc-wmtopo_dseg` | stream 1, step 2 | cortex mask; WM for T = 0 and the sign; fluid and missing tissue as obstacles |
| Relabelled fluid channels and vessels (planned, from `pial_labels`) | stream 1 (an edit) | crests off the label's errors |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| The data normal, its coherence and sign | internal, no spec yet | step 3; stream 3 (planned checks) |
| The field T and its crests | internal, no spec yet | steps 3 and 4 |

*Design and lessons*

- **The data normal** (SF F1): structure tensor of trend-removed log echoes 2 and 3, at derivative scale 1 voxel and integration scale 2 voxels, by masked normalized convolution over cortex.
  - For about 91 % of interior cortex the echoes agree within 10°.
  - Coherence falls to 0.05–0.38 at contacts and fundi, which is where most failures happen.
  - The normal has no sign. Sign it by the WM distance through tissue, with fluid as an obstacle. A straight-line distance flips the sign beside open sulci (CP; #46).
- **The fitted field T** (Lomask-style least squares of ∇T against the normal, T = 0 on WM) has no stalls and bridges fused gaps.
  - T is not a distance: it under-reads long columns.
  - It must be extended 1 voxel into fluid and 0.5 mm into WM, or it makes spurious turns at the pia.
  - pyamg needs local Jacobi weights to be deterministic (#40, #45).
- **T's crest is the midline between the two WM surfaces, not the contact.** On the kissing phantom it sits 0.3 mm from the contact, even across a fluid gap (CP). The crest test (T rises into a voxel but not beyond, with a slope change above L²κ plus noise) needs no calibration.
- **Crests on the aseg's errors.** Many crests sit on the aseg's fluid channels and vessels, so crest methods need those relabelled first. That relabelling (`pial_labels`: closed-form darkness at α = 10⁻³, fluid fraction above ½, a noise-bound component size, the stria trap) belongs to stream 1 as a spec'd input (#46, #49).
- **Doesn't work:**
  - Tracing step by step along the raw tensor normal: no field, no stop, so paths stall and run on (SF F3).
  - Laplace streamlines: they make fused cortex look open (49 k open against 348 crossing at V1V2) and need the outer boundary first.
  - A single all-WM field cannot place an unequal contact: its ridge is the geometric midline (F11).

#### Step 3: tracing methods

*Inputs*

| Input | From | Use |
|---|---|---|
| Roots | step 1 | ray starts |
| The data normal and T | step 2 | directions, for methods that follow the image |
| `_desc-wmtopo_dseg` | stream 1, step 2 | stops: fluid, far WM, missing tissue |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| Per root and method, one ray: samples with 3D positions and arc length, and why it stopped | internal | steps 4 and 5 |

*Design and lessons*

Measured at the marked points (CP; #47, #52, #58):

- **normal** (straight rays along the smoothed white normal, no field, no image): keeps the right class on every phantom and almost never returns to its own bank. Its profile split is the best at M1/S1 (median 0.17 mm) and 0.075 mm at V1V2. Its costs are more stops at caps and edges, and 0.18–0.41 mm path error against curved true columns at crowns.
- **front** (the white mesh grown as one surface, directions coupled along the mesh at 2 voxels): the most accurate wherever folds are 0.64 mm or wider (contact p90 0.054 mm), and it reaches all 35 V1V2 points. Its swept volume matches equal volume within 0.2 %. It is worst at 0.32 mm crowns (p90 0.77 mm), because the coupling scale equals the crown radius.
- **crest and traverse,** with the **fitted_blend start** (mesh normal at the root, blended into the field over 0.3 mm): the most accurate field tracers at tight fundi and contacts. The blended start removes the turn spike at 0.3 mm from the WM (turns fall from 11–16 % to 0.5–2.4 %).
- **front-equivolume** (speed A₀/A along the smoothed normals): bends 0.25–0.31 mm off the straight ray, ends many columns at the caps, splits poorly at M1/S1 (0.55 mm), and runs 8–9× slower. Which paths are right is untested.
- **one-pass** (fronts from all WM in one fast-marching pass):
  - `baillarger.fronts` (Mirebeau's Selling stencils) is causal for anisotropic metrics and never jumps a one-voxel wall. Arrival is first order (4–8 % error from a point source); arc length is good only where paths follow the metric's direction; 141³ takes about 10 s at 2.7 GB.
  - Per-source start times and speeds give crystal growth: connected regions.
  - The root's position must be carried through the pass, as arc length is; it then lands within 0.01 mm. One label per root voxel triples the arrival error, and root patches misplace the root by 0.36–0.80 mm (SF F23).
  - It is a sequential heap on whole arrays: box scale only.
- **Bok's measure** for a vertex is the exact flux of its dual cell. Equivolume depth is (V(s) − V(s₀)) / (V(s₁) − V(s₀)).
- **Doesn't work:**
  - Dijkstra or 6-neighbour fast marching for anisotropic metrics: about 29° lattice bias.
  - A lattice step across T's crest: about 31 k returns to the own bank per site.
  - Sorting front sides by connected pieces, and geometric walls between banks: fragile where banks join round a fundus. The one-pass march needs no sides.

#### Step 4: recording what each ray meets

*Inputs*

| Input | From | Use |
|---|---|---|
| The rays | step 3 | what each column meets |
| T's crests | step 2 | `crest_at` |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| Per column: `crest_at`, `meet_at`, `turn_at`, the root it meets, its class | internal | step 5 |

*Design and lessons*

- Meetings are recorded on the column (`crest_at`, `meet_at`, `turn_at`) and never acted on.
- Classes: open, crossing, own-bank return, to-contact, failed.
- **Doesn't work:** any threshold on bend angle, turn, or root separation along the surface. Each cuts a continuum: between 10° and 60°, 5–34 % of classes change. A 5 mm "same bank" fallback is wrong at both faces.

#### Step 5: the published traces

*Inputs*

| Input | From | Use |
|---|---|---|
| Rays and what they meet | steps 3 and 4 | the traces |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| Column traces keyed by edge key and method: each sample's 3D position, the meetings, the class; readable by block | none yet | stream 3; the viewer |

*Design and lessons*

- Today the traces are a prototype `.npz` with no spec, built on a box white mesh with no edge keys.
- They must key on edge keys, carry each sample's 3D position and the recorded meetings, and be readable by block.

#### Open

- Which paths are right in real cortex: straight rays, field traces or the front. The planned check is phantoms with curved, equivolume true columns.
- What the crests inside thick cortex are: they cover 5–10 % of cortex even after relabelling.
- The cost ratio across the layer normal for arrival (3, 1, or falling with coherence).
- `fronts` accuracy when the metric is tilted 5–20° off the true paths, which is the case on real data.
- **Radial fibres.** At 160 µm the texture is planar, so no source shows radial fibres giving column orientation. Their fade-out depth can still be a planar laminar edge.

### 4d. Implementation plan

- Write the spec for column traces first: it is the interface stream 3 and the viewer wait on.
- Rewrite `column_traces` to root on `white-surf/1` vertices by edge key and trace by mesh blocks, keeping the methods, classes, constants and the frozen rule.
- Port with changes: the `streamlines` integrators and tensor tools (torch, blocks); `column_front` per mesh block, with `past_crest` byte for byte; `mesh_graph` in a local form, with the duplicate-edge bug fixed (§7).
- Rewrite `fitted_field` as a blocked or multigrid solve, if T stays.
- Size each block's halo to the 8 mm length cap (§2, domain edges).
- Leave `fronts` at box scale; do not build a whole-brain step on it.
- Run the open experiments above on the hemisphere, away from box faces.

## 5. Stream 3: laminar profiles, contacts and alignment

### 5a. Goal

Where each layer lies along every column, read from the image's own contrast and aligned across columns and across contacts between banks, with what it learns fed back to stream 1 as an input.

### 5b. High-level design

1. **The column volume.** Sample the echoes along every column into traces, each sample keeping its 3D position.
2. **Flatten.** Estimate shifts between neighbouring traces and fit a warp, so that layers become level.
3. **The joint warp.** One monotone depth map per column, tied across contacts and anchored at open tops. The pial edge is its top level (τ = 1), and the GM/WM transition one of its levels.
4. **Bands and layer identity.** Find the bands in the flattened frame, and tell which layer each is, including where an area's pattern changes.
5. **Thickness and contacts.** Carry thickness sideways from where it is visible, and place each contact between two banks.
6. **Feed back** to stream 1 as a spec'd input.

**Standing decisions**

- **Where there is contrast, the data decide; where there is none, depth follows equal volume.** "The metric" is three roles: depth within a column, thickness carried sideways, and arrival at a contact. Each has its own estimator status.
- **No faults in cortex.** A depth jump between neighbouring columns is an upstream error. Areal borders are facies changes: a thickness kink, a pinch-out, or a contrast change at continuous depth. A fused contact is a boundary between two stacks, not a fault.
- Never label banks by geometry. A voxel's bank is its column's root.
- Search only bands of at least 2–3 voxels; thinner layers ride along on the levels. The stria is at this limit and is the reference case.
- **Build order into the representation; let contrast decide placement** (CL). Constraints that hold everywhere (ordering, no crossings, zero thickness allowed, topology) belong in the representation; those that hold only on average (smoothness, equal volume, templates) belong in the energy. Every surveyed method that broke this rule needed patches.
- The B0-orientation covariate is deferred: a possible later check only if facies changes track folding.

### 5c. Implementation design

#### Step 1: the column volume

*Inputs*

| Input | From | Use |
|---|---|---|
| Column traces | stream 2, step 5 | where to sample |
| The FLASH echoes | the scan | the samples |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| One trace per column, sampled every 0.04 mm, each sample with its 3D position; partial-volume samples marked | internal | steps 2–4 |

*Design and lessons*

- One trace per white-mesh vertex, sampled every 0.04 mm, each sample keeping its 3D position. It reads like a seismic section: the stria runs as a continuous band from open into fused cortex, and the far bank's stria appears mirrored past a contact (CP).
- The voxels within 1 voxel of the WM or the pia are partial volume. They made half the data term and drew 30 % of band boundaries; leave them out (PL; #24).

#### Step 2: flattening

*Inputs*

| Input | From | Use |
|---|---|---|
| The column volume | step 1 | the traces to align |
| Neighbouring columns | the mesh graph on `white-surf/1` | which traces to compare |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| Shifts between neighbouring traces, and the fitted warp | internal | steps 3 and 4 |

*Design and lessons*

- **Shifts between neighbours** (`flatten`; Lucas–Kanade shifts, the Lomask / Wu–Hale form, CG):
  - Exact on synthetic warps.
  - On the boxes: 5–6 min, 6–7 GB, weighted RMS misfit 0.033–0.042 mm.
  - It keeps real upstream faults visible rather than smoothing them away.
- **Dynamic warping adds nothing between neighbours.** It agrees with the gradient shifts on 98.4 % of samples, and widening the bound makes the fit worse. Use it for long-range matching: templates, across gaps, between areas (#57, #60).
- **Doesn't work:** phase-unwrapping relative geologic time (it slips cycles at facies changes).

#### Step 3: the joint warp

*Inputs*

| Input | From | Use |
|---|---|---|
| Flattened traces and shifts | step 2 | conservation of shifts |
| Contacts and open tops | stream 2 (meetings and stops) | ties and anchors |
| Detected bands | step 4 | levels, when placed at them |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| Per column, a monotone depth map z(τ); the pial edge at τ = 1, the GM/WM transition one of its levels | none yet | steps 4–6; the viewer |

*Design and lessons*

- One monotone z(τ) per column, z(1) = the pia, solved by QPBO at a few levels and then filled between them.
- Its terms: conservation of shifts inside each bank, the two-sided tie c_A + c_B = L at contacts (the user's choice), robust open-top anchors, weak lateral smoothness, a vertical prior and hard ordering.
- It certifies 96.6–100 % of variables, so the remaining errors belong to the model.
- The vertical prior must be the chain form, |z_ℓ − (τ_ℓ/τ_ℓ₊₁) z_ℓ₊₁|.
- Levels placed at detected bands tighten the stria's spread by about 40 %. It reaches 0.08 mm at both M1/S1 and V1V2.
- The first pial surface is a level of the joint warp (τ = 1), not a separate cut. The fill between levels uses distance until an equal-volume estimator is validated.
- **QPBO engineering** (#30–#33):
  - Hard costs must be 1 + the finite total, not 1000×.
  - QPBO-I must reseed rand() for determinism.
  - No term may depend on a window's centre (that made the passes cycle).
  - scipy's `maximum_flow` takes int32 capacities only; keep the total below 2³⁰.
  - The QPBO in use (`thinqpbo`) is GPL-3 and has to be replaced.
- **Doesn't work:** a single darkness horizon for the pia. Where the dark line fades it jumps to the far bank's stria, a mirrored decoy (CP).

#### Step 4: bands and layer identity

*Inputs*

| Input | From | Use |
|---|---|---|
| Traces in the flattened frame | steps 2 and 3 | band search |
| The column graph | stream 1's `white-surf/1` | stacking neighbours; regions |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| Bands per region: position in τ, bright or dark, width | none yet | step 3 (levels); step 5; the viewer |
| Which layer each band is, and where an area's pattern changes | none yet | step 5; the viewer |

*Design and lessons*

- **Bands in τ:** stack each trace with neighbours weighted by profile correlation, and keep extrema at least 1.5 voxels wide. A 2.5-voxel threshold drops the stria (1.67 voxels).
  - V1V2: bright at τ 0.18, the stria at 0.37, bright at 0.81.
  - M1S1 has a dark band at τ 0.33 that is not a stria.
- **Bands per region** (PL; #12–#24): Bellman DP, with K chosen by 2-fold cross-validation. Do not refit the held-out gain, and take the least K within max(1 SE, 5 % of the held-out gain). This finds the true K in 160 of 160 synthetic cases.
- Ward merging stops at the column graph's connected pieces. Cumulative bottom tests react late, so test each bin on its own.
- **From the literature** (CL; FS): nothing published fits a depth field to laminar contrast. Areal types are astriate (M1), unistriate (V2) and bistriate (V1); counting peaks mislabels unistriate areas. Keep "present but invisible" apart from "pinched out"; model partial volume at both edges; use sub-voxel boundaries; avoid top-down sequential chains; put a soft prior on changes of band thickness.
- **Doesn't work:** multi-object level sets (they cannot let a band vanish); k-means with no spatial prior; per-column classifiers glued together afterwards.

#### Step 5: thickness and contacts

*Inputs*

| Input | From | Use |
|---|---|---|
| The joint warp and column tops | step 3 | thickness evidence |
| Contacts and their lengths L | stream 2 | contact splits |
| White-surface curvature | from stream 1's `white-surf/1` | the depth estimators |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| A thickness field on the white-surface vertices | none yet | contact splits; the viewer |
| Each contact's split, reported three ways | none yet | step 6; the viewer |
| Depth within each column, by the chosen estimator | none yet | the viewer |

*Design and lessons*

**Depth within a column (role 1)**

- Geometry alone cannot put layers on level sets: one equal-volume surface lies in different layers in different areas (Wagstyl 2018).
- The current estimator is distance, until an equal-volume estimator is validated against visible bands. The results so far:
  - **The closed form from white-surface curvature** (A/A₀ = (1 + κ₁s)(1 + κ₂s)): the evidence conflicts (open, below).
  - **Carried area / div u:** over-corrects about 2× and is noisy upwind (SF F6, F10; #23).
  - **The volume form of the level set:** its carried area is biased (LS).
  - **The front's swept volume:** exact on phantoms, not yet tested against bands (CP).
  - **Two-surface forms with the pial curvature:** too noisy (PL).
- On open V1, curvature moves depth between the deep and superficial parts while total thickness barely changes (Bok's effect).

**Thickness carried sideways (role 2)**

- Crowns are thicker than fundi everywhere measured: V1V2 slope +1.2 mm per (1/mm), M1S1 +2.7. Fit the slope per area, because M1S1 mixes an areal step with folding (SF F10).
- Volume per unit white-surface area is the least conserved quantity against curvature (ρ 0.95). Equal-volume thickness does not absorb the curvature trend.
- A stationary Matérn/SPDE field (on log thickness, about 2 mm long, on the white-mesh vertices) fits tightly at the evidence. But across the 1–2.5 mm without evidence before a contact it reverts to a broad mean and over-predicts the thin bank by 0.26–0.92 mm. The missing evidence is the sums t₁ + t₂ = L at contacts (F13–F16).
- Thickness evidence is the front's own column tops, from roots at least 2 mm from any domain edge, with a guard against wandering tops. Convert to mm with the median, exp(m), not the lognormal mean (LS).
- Region-specific thickness priors are future work, as soft regions.

**Contacts**

- **M1/S1 is genuinely unequal:** 2.0 mm against 3.2–3.3 mm. At its contact two layer I's meet with no fluid gap, and the superficial layers are 0.1–0.15 mm thinner per side than at open pia (F12, F13, F17).
- V1V2's banks are near equal (2.18 and 2.31 mm) and sit on a residual dark line, which explains much of every estimator's success there.
- Equal arrival is right only for near-equal banks: at M1/S1 it misses by 0.63–0.68 mm.
- The two fronts' errors are correlated. Where both banks are over-predicted alike, a σ² split cancels the bias.
- How a contact split shares the misfit: the Gaussian posterior shares it in proportion to σ², and equal z-scores share it in proportion to σ. Past runs used both, so they are not directly comparable.
- **The profile likelihood of a split** (template A stretched to x, B reversed to L − x):
  - On spliced columns with a known answer it is within 0.02–0.09 mm.
  - It is 2–3× overconfident, because the residuals are correlated.
  - The top-anchored stretch beats the linear one.
  - At M1/S1 it misses by 0.5–0.8 mm into the thin bank. This is not diagnosed. Templates from open tops far from the fused stretch are one suspect.
- Report a split three ways (profile alone, profile with field, field alone), and with both stretches.

#### Step 6: feedback to stream 1

*Inputs*

| Input | From | Use |
|---|---|---|
| The GM/WM transition and other levels | steps 3–5 | the feedback |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| An input to stream 1; content and format open | none yet | stream 1, steps 1–2 |

*Design and lessons*

- Nothing is built. The feedback must arrive as a spec'd input to stream 1, as `wm-edits` does, never as a code dependency.

#### Open

- **Conflict: the equal-volume closed form from white-surface curvature.** PL (#23) adopted it with curvature smoothed at 1 mm (no stria trend); SF F10 lists it as failed with curvature at 0.32–0.80 mm (over-corrects on open V1, ρ −0.69, because the pia is flatter than a parallel shell). The tests differ in curvature scale, columns and measure, so the dependence on scale is unresolved. Normal rays (#47) use it at 0.5 mm, unvalidated.
- **Conflict: the best M1/S1 estimator depends on the paths and the score.** The field prior alone on crossing paths gives 0.07–0.12 mm, against 0.26–0.33 mm in other scores. The profile split misses by 0.5–0.8 mm on tensor paths but gets 0.17 mm on normal rays. The crossing path's L and the column arc lengths were never reconciled. Compare estimators only on one shared set of paths with one score.
- The joint fit of the thickness field to visible tops and to the contact sums L, and whether it converges.
- σ or σ² on the contact. It should follow a stated principle.
- The M1/S1 profile miss. The candidates: template locality, superficial thinning at the contact, or an areal border.
- An honest likelihood: measure the residual correlation, or use Student-t.
- Band identity beyond the stria; whether equal τ is the same layer across an areal border; what M1S1's τ 0.33 band is.
- What the feedback to stream 1 says, and in what format.

### 5d. Implementation plan

- Wait for the column-trace spec (stream 2) and build the column volume on it, by blocks.
- Port `flatten` with a blocked or local solve, and a torch version if it stays heavy.
- Port `surfcut.model` and `surfcut.cut`; replace `surfcut.qpbo` with a QPBO that is not GPL (our own, or a permissive library).
- Rewrite `bands` (DP with the cross-validated K) and `gain` only if they are needed. Drop the old patch pipeline (`columns`, `regions`, `surfaces`, `bottom`, `tv`, `refine`, `output`): it needs a pial exit and writes no BIDS output.
- Salvage the layer-maps diagnostics from #56 (isopachs, pinch-outs, facies; `layer_maps_lib.py` and its tests) once columns exist on the hemisphere.
- Write specs for every output, including the feedback to stream 1.
- Resolve the two conflicts above on one shared set of hemisphere columns, away from domain edges.

## 6. Stream 4: the viewer

### 6a. Goal

A way to see any result at full resolution anywhere in a whole brain, and to answer each stage's review queue.

### 6b. High-level design

1. **A server next to the data,** holding the data and doing the computation, with shared state across any number of browser windows.
2. **Panes** on that state: volume, surface (true or inflated), sections, flat-map patch.
3. **Review panes** for each stage's queue, saving answers as spec'd edits.
4. **Only published outputs** are read, never a stream's internals.

**Standing decisions**

- The server binds 127.0.0.1. Keep a server on a big machine with a tunnelled browser possible: no response may scale with the whole mesh or volume.
- Looking comes first; which correction tools to build waits until the data can be seen well.
- **The viewer is the labelling tool** for every stream, because it reads the blocked formats that whole-brain data needs and general tools do not (§3c, step 1).
- Review queues offer a few concrete options per site with previews, not a free-form editor, plus scribbles for stream 1.

### 6c. Implementation design

#### Step 1: the server and shared state

*Inputs*

| Input | From | Use |
|---|---|---|
| Paths of the published outputs to show | the user | what the server loads |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| Shared state (crosshair in world mm, selected true vertex) over POST /state and Server-Sent Events | internal | every pane and window |
| Binary arrays | internal | the panes |

*Design and lessons*

- One server; shared state over POST /state and Server-Sent Events, so any number of windows and panes stay in step (#43, #44).
- Two linked points: a crosshair in world mm, and a selected true vertex.
- Send binary arrays, not VTK. Caches are keyed by the source SHA.

#### Step 2: panes

*Inputs*

| Input | From | Use |
|---|---|---|
| Volumes: the FLASH, labels, P | the scan; stream 1 | the volume pane |
| `white-surf/1`, `white-overview/1` | stream 1 | surface, sections, inflation, flat map |
| Column traces; stream 3's outputs | streams 2 and 3, as their specs appear | their panes |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| Views; picks written to the shared state | internal | the user; other panes |

*Design and lessons*

- **Volume:** the brick pyramid (32³ bricks, 2×2×2 means, a WebGL2 page-table atlas). The hemisphere echo builds in 6.8 s at 1.4 GB, and every sampled pixel matches the data. Its `BrickSource` is the interface for a NIfTI-Zarr store (#84).
- **Surface:** through the overview. Every pick maps to a true vertex and nothing is measured on the overview. Hemisphere start takes 24 s the first time and 0.8 s cached (#70–#72). Our own WebGL view, because a click on a NiiVue mesh does not move the crosshair.
- **Sections:** exact plane cuts of the true mesh, which cannot fork, drawn on a 2D canvas (16–22 ms for 873 k triangles). NiiVue 0.69's 2D mesh display is unusable at 160 µm.
- **Inflation:** closed pieces inflate by mapping back to their own second moments (area restoration gives a needle); open pieces need their boundary pinned.
- **Flat map:** per geodesic patch round a selection (LSCM, R ≤ 5 mm in 0.1–0.44 s). A whole-hemisphere flat map is out of scope. `splu` with MMD and no pivoting; pyamg does not converge on LSCM; CHOLMOD is GPL.
- **NiiVue 0.69 pitfalls:** load it from the pinned `+esm` URL; `readTCK` drops the first line (start the data with a NaN row); `loadMeshes`/`removeMesh` race (use `addMeshFromUrl`); catch keys in the capture phase.

#### Step 3: review panes

*Inputs*

| Input | From | Use |
|---|---|---|
| Each stage's queue and the data round each site (stream 1: `wm-sites/1`, labels, P, surface) | the stage | site answers and scribbles |

*Outputs*

| Output | Spec | Used by |
|---|---|---|
| Answers as spec'd edits (stream 1: `wm-edits/1`) | the stage's edit spec | the stage that queued them |

*Design and lessons*

- **Lessons from the pial editor** (PE; CP):
  - Edits are stored as data (world mm, roots) that replay exactly in any order. A vertex drag must be saved as that vertex.
  - An edit reaches about 1 mm, so the base matters most, and an editor needs a worklist over every doubtful site.
  - A surface built as heights over crossing columns tangles. A contoured base cannot cross itself.
- The stream 1 review pane is described in §3c, step 5.

#### Open

- Overlays, mesh outlines and the 3D tile on the brick viewer.
- Placing points from the volume pane.
- A camera shared between windows.

### 6d. Implementation plan

- Port with changes: `viewer.server`, `viewer.overview`, `viewer.white_view`, `viewer.surface` (the parts that already read published outputs).
- Port as is: `viewer.flatmap`, `viewer.inflation`, `viewer.sections`.
- Rewrite against published specs: the viewer's `data`, `filters`, `labels`, `streamlines` and page, which read prototype `.npz` files today. Keep the filter-job pattern and the TCK writer.
- Bring in the brick volume viewer (#84) and the `--white` volume pane (#82); expect a small conflict between them.
- Bring in the geodesic-patch flat map (#69), with its `edge_graph` fix split out.
- Build the stream 1 review pane, with site answers and scribbles.
- Add panes for stream 2 and 3 outputs as their specs appear.

## 7. Shared code

Each stream's plan lists its own modules. These are shared, or belong to no stream. Most tests are exact: phantoms, brute force, reference solvers, device equality.

| Verdict | Modules |
|---|---|
| Port as is | `backend`, `bids` (strict sidecars, spec docs tested against code), `gifti`, `timing`, `_jit`, `mesh` (Kuhn marching, edge keys, Morton blocks: the most reusable piece), `mesh_intersect`, `phantoms`, `scripts/check`, `scripts/affected-tests`, `scripts/memguard` |
| Port with changes | `blocks` (add the chunked source; never read large inputs whole), `mesh_torch` (block-local tables), `mesh_smoothing` (keep normals, edges, pieces), `mesh_graph` (local form; fix the bug below) |
| Drop | `topology` and `wellcomposed` ((26,6) codes), `growth` and `joint` (old seam model), `frame` (move `missing_tissue` to stream 1) |

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

## 9. Scaling gaps across streams

Stream 1's ordered scaling steps are in its plan (§3d). Across the streams:

- Streams 2 and 3 exist only at box scale, with whole-box arrays: `column_traces`, `fitted_field` (pyamg), `fronts`, `fit_warp`, `mesh_graph`.
- Normals are computed on the whole mesh.
- `NiftiSource` reads a `.nii.gz` whole, and `BlockReader` keeps volumes up to 4 GB resident on the device.
- Many scales are hard-wired to 0.16 mm voxels: the 0.04 mm sampling, the 1.5-voxel band width, the tapers, the 0.32 mm twicing and the 2-voxel coupling. Restate each as mm or as voxels on purpose.
- Box costs per 141³ on one CPU core: column volume 3–5 min and 6 GB, the single-surface cut 18 GB, joint warp about 10 min and 10 GB, flattening about 6 min.
