# How neuroimaging tools find the contact between apposed cortical banks, and how they identify the two sides

Scope: what object represents the contact, how the two sides are named, how tools treat tight folds, label errors and thin CSF slivers, reported accuracy at buried sulci, use of image contrast at the contact, and whether any tool separates local (curvature-focal) self-intersection from global (apposition) self-intersection.

Tags: **[verified]** = checked this session against a primary source (paper full text, official doc page, or the tool's source code). **[unverified]** = background knowledge or a secondary summary, not checked against a primary source this session. **[inference]** = my reasoning from verified material.

Companion file with overlapping material (thickness caps, DiReCT, DL tools, DELFMAP validation numbers): `research_notes/Fused sulci and thickness priors/neuro_tools.md`. These notes do not repeat that material; they cover the contact object and side identification.

Primary sources read this session:
- FreeSurfer source, `dev` branch: [utils/mrishash.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrishash.cpp), [utils/mrisurf_timeStep.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrisurf_timeStep.cpp), [utils/mrisurf_deform.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrisurf_deform.cpp), [utils/mrisurf_sseTerms.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrisurf_sseTerms.cpp), [mris_make_surfaces/mris_place_surface.cpp](https://github.com/freesurfer/freesurfer/blob/dev/mris_make_surfaces/mris_place_surface.cpp).
- Kim et al. 2005 (CLASP) full text, author reprint: [Kim2005.pdf](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf). Citation: Kim JS et al., NeuroImage 27:210–221, doi 10.1016/j.neuroimage.2005.03.036.
- Han, Xu, Prince 2003 (TGDM) full text: [IEEE TPAMI 25(6):755–768](https://iacl.ece.jhu.edu/~chenyang/research/pami_han_xu_prince.pdf).
- Osechinskiy & Kruggel 2012 (DELFMAP): [PMC3296314](https://pmc.ncbi.nlm.nih.gov/articles/PMC3296314/).
- Huo et al. 2016 (MaCRUISE, with the ACE equations): [PMC4927397](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927397/).
- BrainVISA "Cortical Fold Arg" process doc: [brainvisa.info](https://brainvisa.info/axon/en/processes/AnaComputeCorticalFoldArg.html).

Not retrieved: the Dahnke 2013 PBT full text (the Manchester mirror timed out again; ResearchGate returned 403), the Han et al. 2004 CRUISE paper, Mangin et al. 1995 (J Math Imaging Vis), MacDonald et al. 2000 (ASP), and Kriegeskorte & Goebel 2001.

---

## Q1. What object represents the contact?

### Takeaway
There are four kinds of contact object. (1) Mesh self-intersection, prevented as a hard geometric veto (FreeSurfer) or penalized as a proximity energy (ASP/CLASP). (2) A CSF skeleton made from the CSF segmentation alone (CLASP). (3) Shocks or skeletons of a distance or arrival-time field grown from WM (ACE, DELFMAP, PBT; MaACE adds atlas evidence). (4) A topology barrier: the front may not touch itself (TGDM, Kriegeskorte & Goebel, Morphologist's homotopic skeleton). Only DELFMAP explicitly uses a collision of streamlines, i.e. a jump in which WM point each voxel comes from. That is the closest analogue to the project's "roots".

### Cited Findings

**FreeSurfer (mris_make_surfaces / mris_place_surface)**
- [verified] The contact is not represented explicitly. It is wherever a vertex move would make a triangle intersect another triangle of the same mesh. `isVectorFilled(vno, dx,dy,dz)` asks: "If vertex vtxno is moved by (dx,dy,dz), will the adjoining faces then intersect with some other faces in the surface?" Faces in the hash table are sampled into voxel buckets, and an exact triangle–triangle test is run against faces in those buckets — [mrishash.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrishash.cpp).
- [verified] During deformation, `mrisLimitGradientDistance` zeroes the vertex's whole step when that test fires (`v->odx = v->ody = v->odz = 0.0; v->cropped++`). Contact is therefore a frozen vertex, not a placed surface — [mrisurf_timeStep.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrisurf_timeStep.cpp). The test is skipped when the `IPFLAG_NO_SELF_INT_TEST` flag is set (same file).
- [verified] After placement, `MRISremoveIntersections` marks intersecting faces and runs "soap bubble" smoothing (100 iterations) on the marked vertices, with unmarked vertices fixed. It loops up to ~100 times, keeps the configuration with the fewest intersections, and may end with intersections left ("terminating search with %d intersecting"). A comment notes that expanding the neighbourhood "was causing the surface to do wacky things" and was disabled — [mrisurf_deform.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrisurf_deform.cpp). `mris_place_surface` calls it before and after placement — [mris_place_surface.cpp](https://github.com/freesurfer/freesurfer/blob/dev/mris_make_surfaces/mris_place_surface.cpp).
- [verified] There is also a soft vertex–vertex repulsion energy, `RepulsiveEnergy`: a 1/(d+0.25)^6 force between vertices sharing a hash bucket, skipping self and mesh neighbours (`vtotal`). In `mris_place_surface` its weight `l_repulse` is 5.0 for the white surface and 0.0 by default for the pial surface. For the pial surface, `l_surf_repulse = 5.0` with a mandatory `--repulse-surf` (the white surface): "ERROR: must supply a repulse surface when placing pial surface" — [mrisurf_sseTerms.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrisurf_sseTerms.cpp), [mris_place_surface.cpp](https://github.com/freesurfer/freesurfer/blob/dev/mris_make_surfaces/mris_place_surface.cpp).
- [inference] So in FreeSurfer the pial contact in a fused sulcus is wherever the two pial sheets, moving out from the white surface, first come close enough that a further step would cross. Its position is set by the intensity-driven targets and step sizes of the two banks, not by a medial construction.
- [verified] A developer comment on the mailing list: mris_expand is slow "precisely because it prohibits self-intersection" — [FreeSurfer list](https://mail.nmr.mgh.harvard.edu/archives/list/freesurfer@nmr.mgh.harvard.edu/thread/OWBZAP4YSKZEDC5GPSI2KBEX5X526NMO/) (search snippet only).

**ASP (MacDonald et al. 2000) / CLASP (Kim et al. 2005) / CIVET**
- [verified] CLASP pial surface: a mesh is expanded from the WM surface along the gradient of a Laplace field. The field has its minimum on WM voxels and its maximum on "the skeletonized CSF and background". The contact object is therefore a **CSF skeleton**: "the partial volume CSF calculated is binarized, setting any voxel containing CSF to 1 and all others to 0, and then skeletonized … using a 2-subfield connectivity-preserving medial surface skeletonization algorithm (Ma and Wan, 2001; Ma et al., 2002)" — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).
- [verified] Kim 2005 also keeps ASP's mesh self-proximity term: Σ over pairs of non-adjacent polygons of (d_min(P_i,P_j) − d_ij)² when d_min < d_ij. "Pairs of adjacent polygons are not included … as the distance … is a constant zero" — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf). The numeric value of the threshold d_ij is not stated in the extracted text.
- [verified] Per Kim 2005, ASP is a "topology-preserving deformation model with proximities" that forces "the cortical thickness to lie within a defined anatomically-plausible range" through a vertex–vertex proximity term linking points on the two surfaces. CLASP replaced that term with an intensity term that is "only activated to shrink the vertices if vertices are found within CSF" — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).

**CRUISE / ACE (Han et al. 2004) and MaACE (Huo et al. 2016); Nighres `cruise_cortex_extraction`**
- [verified] ACE computes a weighted arrival time T from the GM/WM 0.5-isosurface by fast marching, F‖∇T‖ = 1 with F = 1 − 0.9·μ_CSF. The contact is the **shock set** S = {x | F‖∇_c T‖ ≤ Q and T(x) > 1}, with Q = 0.85, then thinned to one voxel. On S the GM membership is scaled by F‖∇T‖, which carves a thin separation — [Huo 2016, PMC4927397](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927397/).
- [verified] MaACE uses F = 1 − 0.9·max(μ_CSF, μ_boundary). μ_boundary = 1 where a voxel's 26-neighbourhood holds ≥ 2 different multi-atlas cortical labels and no WM label. With no CSF and no atlas boundary, F = 1, which is pure Euclidean distance (the midpoint case) — [PMC4927397](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927397/).

**DELFMAP (Osechinskiy & Kruggel 2012)**
- [verified] Contact = **collision of correspondence**. A vector ψ assigns each non-WM voxel its source point on the WM boundary, carried along streamlines of an electrostatic potential: (∇φ/‖∇φ‖)·∇ψ_i = 0. A voxel is a skeleton candidate if the maximum Euclidean distance between its ψ and its 6-neighbours' ψ exceeds D_min = 4 voxels. A second, shock-based test flags voxels where path length d > 1 mm and ‖∇d‖ < T = 0.8. The paper calls the correspondence test the robust one and the shock points sparse discretisation artefacts — [PMC3296314](https://pmc.ncbi.nlm.nih.gov/articles/PMC3296314/).

**CAT12 PBT (Dahnke et al. 2013)**
- [verified, abstract snippet only] PBT estimates a WM distance and "projects the local maxima" onto GM voxels. It claims to handle "sulcal blurring, and sulcal asymmetries without explicit sulcus reconstruction via skeleton or thinning methods" — [Manchester mirror](https://personalpages.manchester.ac.uk/staff/fumie.costen/pastwork/DATA/readingmaterial/corticalthickness4.pdf), [Scinapse record](https://scinapse.io/papers/1999522048).
- [unverified] Contact = local maxima of the WM distance inside GM, i.e. a ridge of the WM-distance map, used implicitly with no explicit skeleton.

**BrainVISA / Morphologist (Mangin et al. 1995; Rivière et al. 2002)**
- [verified] The sulcal skeleton comes from skeletonizing the complement of the "white object" (grey + CSF inside the closed hemisphere). The erosion is homotopic and non-uniform: "the grey level is an altitude, white matter is the mountain's crests". The mean curvature of the MR isosurfaces is the "crevasse detector" that pulls the skeleton into the depths of folds. Skeleton points form where erosion fronts meet — [BrainVISA Cortical Fold Arg doc](https://brainvisa.info/axon/en/processes/AnaComputeCorticalFoldArg.html).
- [verified, secondary] Folds are the skeleton of a "negative cast" of the brain, made by topology-preserving erosions — [arXiv 2211.16213](https://arxiv.org/pdf/2211.16213); [ResearchGate: Sulcus identification and labeling](https://www.researchgate.net/publication/282567484_Sulcus_Identification_and_Labeling).

**TGDM (Han, Xu, Prince 2003)**
- [verified] No explicit contact object. Topology is preserved by letting the level-set function change sign only at simple points ("topological numbers are computed locally"). Two parts of one front therefore cannot merge, and fronts that approach each other stay digitally separate (the two middle fingers in the hand example, two bone cells "at the weak gap") — [Han 2003](https://iacl.ece.jhu.edu/~chenyang/research/pami_han_xu_prince.pdf).

**Kriegeskorte & Goebel 2001**
- [verified via Kim 2005] "Their method uses a self-touching sensitive region growing process prioritized by distance-to-surface voxels considered for inclusion" — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf). Citation: Kriegeskorte N, Goebel R. NeuroImage 2001;14:329–346 [unverified pages].

**highres-cortex (Leprince et al.; neurospin)**
- [verified] The input is a 3-label classification: 0 = exterior/CSF, 100 = cortical GM, 200 = subcortical WM. The README says nothing about fused sulci — [GitHub neurospin/highres-cortex](https://github.com/neurospin/highres-cortex).
- [inference] highres-cortex expects the CSF separation to be present in the labels already. It has no contact detector, so a fused bank pair labelled as one GM blob would be treated as one thick ribbon.

### Inferences
- [inference] Every WM-seeded volumetric method (ACE, DELFMAP, PBT, Morphologist erosion seen from the WM side) locates the contact as a **shock or ridge** of a front grown from WM. That is the same object as the project's "collision". They differ only in the metric (Euclidean, CSF-weighted, electrostatic, intensity-altitude) and in the test that declares a shock a contact (gradient magnitude, correspondence jump, minimum distance from WM).
- [inference] Mesh methods (FreeSurfer, ASP, CLASP's self-proximity) never construct a contact. They forbid crossing, and the contact is a by-product of where motion stops.

### Gaps
- Exact PBT ridge definition and its handling of "asymmetries" (full text not retrieved this session or the previous one).
- Mangin 1995's exact erosion priority (intensity vs. curvature ordering) is known only through the BrainVISA doc page.
- ASP's numeric self-proximity threshold and the size of the neighbour-exclusion ring were not found.

---

## Q2. How are the two sides or banks identified: globally or locally?

### Takeaway
No surveyed tool labels "sides" as global groups of roots. Mesh tools rely on **mesh adjacency**: anything that is not a 1-ring (or `vtotal`-ring) neighbour is "another side". This is purely topological and local in index space, with no notion of geodesic distance along the WM. Volumetric tools never name the sides. The shock or skeleton is the separator, and each GM voxel belongs implicitly to whichever WM source its front came from. DELFMAP is the only tool that compares sources explicitly, using the 3D Euclidean distance between WM source points of neighbouring voxels, not the along-surface distance.

### Cited Findings
- [verified] FreeSurfer intersection test: excluded faces are the moved face plus all faces sharing any vertex with it (`MHTexpandToTouchingFaces`). Every other face in nearby hash buckets is a candidate "opposite side" — [mrishash.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrishash.cpp).
- [verified] FreeSurfer repulsion: excluded vertices are self and `vt->v[0..vtotal)`, the vertex's neighbourhood at the current neighbourhood size. All others within the hash bucket repel — [mrisurf_sseTerms.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrisurf_sseTerms.cpp).
- [verified] CLASP: only adjacent polygons are excluded from the self-proximity term — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).
- [verified] DELFMAP: the side identity is ψ, the WM source coordinate. A contact is a jump of > 4 voxels in ψ between 6-neighbours, measured as Euclidean distance between source coordinates, "not along the surface" (fetch summary of the paper's formula) — [PMC3296314](https://pmc.ncbi.nlm.nih.gov/articles/PMC3296314/).
- [verified] TGDM: sides are implicit in digital topology. Simple-point checks use "the chosen digital connectivity pair" (e.g. 26/6), so whether two near voxels "touch" depends on the connectivity convention — [Han 2003](https://iacl.ece.jhu.edu/~chenyang/research/pami_han_xu_prince.pdf).
- [verified] Han's approach was criticized by Kim et al.: it "estimates the position of sulci by assuming that they are equidistant, on either side, to the WM surface … creating the representative topology based on the WM surface misplaces asymmetrically folded sulci" — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).
- [verified] Morphologist: the sides are not labelled at skeleton time. Folds are later split into elementary surfaces, and buried gyri (plis de passage) are detected by "depth and surface curvature criteria". The doc says this splitting step is "still far to be fully satisfying" — [BrainVISA doc](https://brainvisa.info/axon/en/processes/AnaComputeCorticalFoldArg.html).

### Inferences
- [inference] Mesh-adjacency exclusion is the mesh analogue of the project's continuum problem. With a 1-ring exclusion, the two flanks of a tight fundus a few vertices apart already count as "other side", so focal convergence and true apposition are treated identically. FreeSurfer avoids false contacts in this case only because the hard test fires on actual crossing, not on proximity.
- [inference] DELFMAP's Euclidean source-jump criterion fails the same way the project's distance thresholds failed: two banks whose WM sources are < 4 voxels apart in 3D (a narrow fundus) are not flagged, and a strongly curved single WM surface could produce a jump. Using the 3D source jump rather than the geodesic one makes it worse at a fundus, where the sources are close in 3D but far along the surface only for deep sulci.
- [inference] An alternative that no tool uses: define the sides locally at each contact voxel from the directions of the two arriving fronts (the antiparallel component of the arrival gradients, or ⟨n_A, n_B⟩ < 0), not from root identity. TGDM, ACE (‖∇T‖ < Q) and DELFMAP's shock test all approximate this, because ‖∇T‖ drops below 1 exactly where the arrival directions disagree. But they threshold its magnitude rather than reading its geometry.

### Gaps
- No source found where a tool computes the geodesic distance along the WM surface between colliding sources. This may be truly absent, or it may be in uncited cortical-thickness papers.

---

## Q3. Handling of (a) tight folds or focal convergence, (b) label errors, (c) thin partial-volume CSF slivers

### Takeaway
(a) Tight folds are handled by a **minimum distance from WM** before a shock may count (ACE T > 1 mm, DELFMAP d > 1 mm), by **Laplace streamlines** instead of normals (CLASP, DELFMAP), or by **freezing and smoothing** (FreeSurfer). None reasons explicitly about focal convergence. (b) Label errors are handled upstream (PVE classification, subcortical classes, T2/FLAIR, multi-atlas), not at the contact. (c) Slivers are the main use of image evidence: any CSF fraction becomes a barrier (CLASP), slows the front (ACE), or sets the erosion altitude (Morphologist).

### Cited Findings
**(a) Tight folds / focal convergence**
- [verified] Kim 2005, Fig. 6 caption: "While the distance maps may lead neighboring vertices to collide as shown in region A of (b) and (c), the Laplacian map smoothly expands vertices and reduces the collision." The text adds that Laplace streamlines "are unique as well as non-intersecting", like "cortical striations", and that distance maps from WM vs. from the CSF skeleton "can produce different values" — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).
- [verified] CLASP scales vertex speed by the Laplace gradient so that "every vertex on the surface moves to the outer boundary of GM at a uniform rate, proportional to the local thickness". Otherwise, vertices that arrive early "would be moved by other model constraint terms such as the stretch and the self-intersect terms" — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).
- [verified] ACE excludes shocks within T ≤ 1 (1 mm) of the GM/WM surface — [PMC4927397](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927397/). Osechinskiy & Kruggel criticize this "minimum-distance cutoff" and report spurious branches and discontinuities in ACE skeletons — [PMC3296314](https://pmc.ncbi.nlm.nih.gov/articles/PMC3296314/).
- [verified] DELFMAP's shock test also requires d > d_min = 1 mm. A 6 mm maximum path length (d_max) limits advection — [PMC3296314](https://pmc.ncbi.nlm.nih.gov/articles/PMC3296314/). The paper does not explain how same-gyrus convergence is told apart from opposing banks (fetch summary).
- [verified] FreeSurfer: a vertex whose move would intersect is frozen. Leftover intersections are soap-bubble smoothed. A separate `MRISremoveHighAngleHinges` smooths high-angle hinges (the docstring suggests thresh = 100°) — [mrisurf_deform.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrisurf_deform.cpp).
- [verified] TGDM: the result "can clearly depend on the initialization in a dramatic way" (filling a tunnel vs. cutting a handle), and "we have no formulation of an optimality criterion that would choose one of these solutions over the other" — [Han 2003](https://iacl.ece.jhu.edu/~chenyang/research/pami_han_xu_prince.pdf).
- [verified] ASP's thickness-range constraint "introduces a bias … in populations which may have much thicker or much thinner cortices", and it was "especially prominent in pediatric brain images with tightly folded gyri" — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).

**(b) Label errors**
- [verified] ASP did not handle insula–putamen confusion. CLASP added subcortical classes and a gradient term on the WM surface — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).
- [verified] DELFMAP vs. FreeSurfer: discrepancies are attributed to FreeSurfer over-segmenting into meningeal space and DELFMAP over-segmenting into cerebellar GM. There is no label-error study — [PMC3296314](https://pmc.ncbi.nlm.nih.gov/articles/PMC3296314/).
- [verified] FreeSurfer T2/FLAIR refinement (`-T2_min`/`-T2_max`, "threshold for min/max T2 values allowed to be interior to the cortical ribbon") is the only explicit guard against non-cortex inside the ribbon among the tools here — [FreeSurfer wiki](https://surfer.nmr.mgh.harvard.edu/fswiki/mris_make_surfaces).
- [verified] MaACE: where CSF is not seen, the multi-atlas label boundary forces the sulcus "regardless of CSF evidence" — [PMC4927397](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927397/).
- [inference] No tool tests whether a medium inside a claimed GM region is non-tissue (for example, Fomblin, which gives no signal). A "fluid labelled as cortex" error makes a false contact in the volumetric tools, because the WM front crosses the gap. In CLASP it would remove the CSF skeleton there.

**(c) Thin CSF slivers / partial volume**
- [verified] CLASP binarizes any voxel "containing CSF" from the PVE classification, then skeletonizes it. This is maximally sensitive to slivers. Kim: "The worst case partial volume effect is having no CSF within identified sulci, and thus, no apparent sulci" — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).
- [verified] ACE: fractional μ_CSF slows the front (F down to 0.1), which pulls the shock toward residual CSF. "ACE might not be able to detect tight sulci when the presence of CSF is not well captured" — [PMC4927397](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927397/).
- [verified] DELFMAP: permittivity follows GM probability, so CSF-like voxels have low permittivity and act as barriers to flux — [PMC3296314](https://pmc.ncbi.nlm.nih.gov/articles/PMC3296314/).
- [verified, abstract] PBT claims to handle partial volume and "sulcal blurring" without a skeleton — [Manchester mirror](https://personalpages.manchester.ac.uk/staff/fumie.costen/pastwork/DATA/readingmaterial/corticalthickness4.pdf).

### Inferences
- [inference] The minimum-distance cutoff (1 mm in ACE and DELFMAP) is the only explicit "local vs. global" heuristic in the volumetric tools. It is a distance threshold of the kind that already failed in this project. At 160 µm with ~2.5–4 mm cortex, 1 mm would be ~6 voxels. It would suppress fundus shocks but also real contacts of thin banks.
- [inference] Laplace or electrostatic streamlines (CLASP, DELFMAP) remove focal self-intersection of the *offset surfaces* in tight folds, because streamlines cannot cross. But they need the outer boundary (the CSF skeleton) to be known first, which is circular for the contact problem. DELFMAP avoids the circularity by using a WM-to-infinity potential, so its streamlines also converge into sulci, where it detects the jump.

### Gaps
- No tool-specific handling of non-tissue embedding media (Fomblin, agar) was found. That is ex vivo only, and the in vivo tools would not address it.

---

## Q4. Quantitative accuracy at buried sulci; use of image contrast at the contact

### Takeaway
No surveyed tool reports contact-position error stratified by buried or fused sulci against ground truth. Reported numbers are global (surface RMS, phantom distances, test–retest). Image contrast at the contact enters only as CSF evidence (PVE fraction, fuzzy membership, raw grey level and mean curvature in Morphologist). None uses layer I (a bright or dark band of the molecular layer) or the pia–pia interface as positive evidence.

### Cited Findings
- [verified] CLASP: with a synthetic thick cortex, the mean simulated-vs-extracted surface difference was 4.6 ± 1.4 mm for ASP and 0.5 ± 0.4 mm for CLASP. GM volume matched 13% more accurately. Repeatability over 16 scans: 30% lower RMS for the GM surface and 8% lower for the WM surface. All at 1 mm isotropic, with no sulcus-specific metric — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).
- [verified] DELFMAP: no depth-stratified numbers. Pial inter-method distance with FreeSurfer was 0.40/0.42 mm (L/R), with < 10% of vertices > 1 mm and < 1% > 2 mm. Simulated unresolved or bridged sulci were assessed only qualitatively ("finds a geometrically plausible solution") — [PMC3296314](https://pmc.ncbi.nlm.nih.gov/articles/PMC3296314/).
- [verified] MaCRUISE: no sulcus-specific or ACE-vs-MaACE numbers. Sulcal behaviour is shown only qualitatively (Fig. 5) — [PMC4927397](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927397/).
- [verified] TGDM on cortex: a qualitative comparison only (the parametric model self-intersects, TGDM does not; the standard GDM had 40 handles). The topology constraint added < 7% run time — [Han 2003](https://iacl.ece.jhu.edu/~chenyang/research/pami_han_xu_prince.pdf).
- [verified] Morphologist is the only tool where raw image grey level shapes the contact: grey level as altitude plus mean-curvature "crevasse detector". It admits that "the crevasse detector is imperfect" and that homotopic erosion "can fill holes in the crevasse surfaces" (fetch summary) — [BrainVISA doc](https://brainvisa.info/axon/en/processes/AnaComputeCorticalFoldArg.html).
- [verified] Kim 2005, Fig. 2 caption: "Note how sulci are generally localized in the darker areas" of T1 and negative curvature images — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).
- [verified] The only tool whose motivation explicitly mentions "sulcal asymmetries" is PBT (abstract) — [Manchester mirror](https://personalpages.manchester.ac.uk/staff/fumie.costen/pastwork/DATA/readingmaterial/corticalthickness4.pdf).

### Inferences
- [inference] For this project, Morphologist's design is the nearest precedent for "contact = intensity valley found by a topology-constrained front meeting". A dark residual CSF or pia line at 160 µm would play the role of its crevasse. The reported weakness is that a homotopic erosion can bridge gaps in the valley, which is the same mechanism as a Fomblin or sliver false positive.
- [inference] A layer I criterion (the contact flanked symmetrically by the two banks' layer I, and the two banks' depth profiles mirrored across it) would be new relative to these tools. None has laminar contrast at 1 mm.

### Gaps
- No quantitative failure analysis at buried sulci was found in any of the primary sources read. The DELFMAP ex vivo specimens (0.25–0.35 mm) were not evaluated at contacts.
- The ex vivo 7T deep-learning work from Penn (Khandelwal et al.) and Ravikumar et al. 2023 (Laplace loss for unresolved sulci) were not read this session. See the companion notes.

---

## Q5. Does any tool explicitly separate local (curvature-focal) from global (apposition) self-intersection?

### Takeaway
No. Every tool treats all self-approach the same way and separates "near" from "far" with either mesh adjacency (FreeSurfer: 1-ring faces, `vtotal` vertices; CLASP: adjacent polygons) or a fixed metric threshold (ACE and DELFMAP: 1 mm from WM; DELFMAP: 4-voxel source jump in 3D). Topology-preserving methods (TGDM, Kriegeskorte & Goebel, Morphologist) forbid both kinds equally. Focal convergence is avoided mainly by flow construction (Laplace or electrostatic streamlines do not cross) and by smoothing, not by a classification.

### Cited Findings
- [verified] FreeSurfer: only faces sharing a vertex are exempt from the crossing test, and only `vtotal` neighbours are exempt from repulsion. No curvature or geodesic criterion — [mrishash.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrishash.cpp), [mrisurf_sseTerms.cpp](https://github.com/freesurfer/freesurfer/blob/dev/utils/mrisurf_sseTerms.cpp).
- [verified] CLASP: only adjacent polygons are exempt — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).
- [verified] ACE: a gradient-magnitude shock test plus T > 1 — [PMC4927397](https://pmc.ncbi.nlm.nih.gov/articles/PMC4927397/).
- [verified] DELFMAP: a correspondence-jump D_min = 4 voxels (Euclidean between sources) plus d > 1 mm and ‖∇d‖ < 0.8 — [PMC3296314](https://pmc.ncbi.nlm.nih.gov/articles/PMC3296314/).
- [verified] TGDM: local simple-point tests forbid any merger, whatever its cause — [Han 2003](https://iacl.ece.jhu.edu/~chenyang/research/pami_han_xu_prince.pdf).
- [verified] Kim 2005 notes that distance-map flows make "neighboring vertices … collide" at bulges, and that Laplace flow "reduces the collision". This acknowledges focal convergence but solves it by changing the flow, not by classifying it — [Kim 2005](http://www.bic.mni.mcgill.ca/users/jason/reprints/Kim2005.pdf).

### Inferences
- [inference] The project's insight (every collision is the WM front meeting itself; "two banks" vs. "fold meeting itself" is a continuum in along-WM distance) is consistent with these tools. None of them has a principled separator, and each has a tuned constant (1 mm, 4 voxels, 0.85, 0.8, neighbour ring) that plays the role of the project's failed threshold.
- [inference] Candidate physical criteria that none of the tools uses, suggested by their mechanisms:
  1. **Arrival-direction opposition**: at a contact the two arriving fronts are near-antiparallel (‖∇T‖ → 0 for a symmetric meeting). At focal convergence inside one ribbon the arrival directions span a fan with no single opposing pair. This is the geometric content behind ACE's and DELFMAP's ‖∇‖ < threshold tests.
  2. **Both-sided completeness**: a true contact has a full cortical depth on both sides (each side's column reaches its own layer I). A focal shock in a fundus lies at depth < thickness along one family of columns.
  3. **Image evidence at the contact**: a dark line (CSF, pia), as in ACE, CLASP and Morphologist, extended to layer I contrast and to a check that the medium between the banks is non-tissue (Fomblin = no signal), not cortex.
  4. **Laplace or electrostatic flow instead of a distance front**, to remove offset-surface self-intersection in tight folds (CLASP, DELFMAP), so that the remaining collisions are more likely to be real contacts.

### Gaps
- PBT may contain an implicit separator (projection along WM-distance neighbourhoods); the full text is still needed.
- Mangin 1995 and Rivière 2002 may define buried-gyrus vs. fold criteria in more detail than the doc page. Not read.
- I found no paper that frames self-intersection in cortical reconstruction as a local vs. global classification problem. Absence was not confirmed by a systematic search.
