# Front solvers with source identity and weighted competition

Scope: algorithms that grow anisotropic fronts from many sources (WM-boundary roots), carry each voxel's source label and per-source quantities, and resolve a weighted competition locally with low grid bias. Labels: [verified] = read in the cited source (full text or abstract); [unverified] = from a search summary or secondary description, not read in the primary text; [inference] = my reasoning, no source.

## Anisotropic eikonal solvers with low grid bias (OUM, Mirebeau FM-LBR / FM-VR1 / HFM-agd, fast sweeping, FIM)

### Takeaway
For a 3D Riemannian metric on a ~140^3 Cartesian grid, Mirebeau's FM-VR1 in the HFM / `agd` library is the most direct fit: single pass, 12-point adaptive stencils in 3D, convergence proved, recommended anisotropy up to about 10:1, Python interface, and built-in seed values, Voronoi labels and Euclidean path length. OUM and fast sweeping are valid alternatives, but OUM's cost grows with anisotropy and sweeping is iterative. FIM is a parallel iterative variant with no special advantage for anisotropy.

### Cited Findings
- FM-LBR (Fast Marching using Lattice Basis Reduction) solves the Riemannian eikonal equation for an arbitrary continuous metric on 2D/3D Cartesian grids. At each grid point it computes a reduced basis of the grid with respect to the local metric tensor. Its complexity is logarithmic in the maximum anisotropy ratio, and consistency is proved [verified, abstract] — [Mirebeau, arXiv 1201.1546 / SINUM 2014](https://arxiv.org/abs/1201.1546)
- Approximate complexity of FM-LBR is O(N ln N + N ln κ), with κ the maximum anisotropy ratio [unverified, search summary] — [arXiv 1201.1546](https://arxiv.org/pdf/1201.1546)
- The error analysis of anisotropic FM for arbitrary continuous metrics on general domains was not completed, so the follow-up paper restricts it to constant metrics [unverified, search summary] — [Mirebeau, "On the accuracy of anisotropic fast marching", arXiv 1205.5300](https://ar5iv.labs.arxiv.org/html/1205.5300)
- FM-VR1 (Voronoi's first reduction) solves Riemannian eikonal equations "in a single pass over the domain" with adaptive upwind finite differences. It handles eigenvalue ratios of 10^2 and more, and convergence is established with rates [verified, abstract] — [Mirebeau, SINUM 2019, doi 10.1137/17M1127466](https://epubs.siam.org/doi/10.1137/17M1127466); preprint [HAL hal-01507334](https://hal.archives-ouvertes.fr/hal-01507334)
- HFM decomposes the dual metric as D = Σ α_i e_i⊗e_i, with α_i ≥ 0, integer offsets e_i and d(d+1)/2 terms (6 in 3D). The decomposition is computed with Selling's algorithm (d ∈ {2,3} only). The offsets satisfy ‖e_i‖ ≤ Cond(D)^α with α = 1 in 3D, where Cond(D) := sqrt(‖D‖‖D⁻¹‖) [verified] — [Mirebeau & Portegies, IPOL 2019, "Hamiltonian Fast Marching"](https://doi.org/10.5201/ipol.2019.227)
- 3D Riemannian schemes in HFM use I = 12 neighbours. Overall complexity is O(I N ln N), and the per-point update solves a quadratic whose larger root satisfies λ* ≥ U(q_i) for all used neighbours (causality) [verified] — [IPOL 2019](https://doi.org/10.5201/ipol.2019.227)
- FM-VR1 is faster than FM-LBR, "especially in dimension 3". It is more accurate with second-order enhancement and slightly less accurate without it [verified] — [IPOL 2019](https://doi.org/10.5201/ipol.2019.227)
- The agd notebook recommends metric tensor condition numbers of "10^2 at most" and shows usable but out-of-theory results at 100^2 and 50^2 with discontinuous metrics (3D tube example). The notebook's condition-number convention may differ from the IPOL Cond(D) (square root) [verified text; convention ambiguity is my note] — [agd Riemannian notebook](https://nbviewer.org/github/Mirebeau/AdaptiveGridDiscretizations_showcase/blob/master/Notebooks_FMM/Riemannian.ipynb)
- Second-order mode (`order: 2`): "second order convergence rates can indeed be achieved in certain cases, but only in the average sense, and with boundary data less singular than the single seeds" [verified] — [agd Riemannian notebook](https://nbviewer.org/github/Mirebeau/AdaptiveGridDiscretizations_showcase/blob/master/Notebooks_FMM/Riemannian.ipynb)
- `agd` has 'Riemann2' / 'Riemann3' models, a C++ CPU solver and a CUDA GPU eikonal solver. Advanced stopping criteria and automatic differentiation are not implemented on the GPU. Generic Riemannian metrics in dimension ≥ 4 are not handled [verified] — [agd Riemannian notebook](https://nbviewer.org/github/Mirebeau/AdaptiveGridDiscretizations_showcase/blob/master/Notebooks_FMM/Riemannian.ipynb); [HFM GitHub](https://github.com/Mirebeau/HamiltonFastMarching)
- A standard (axis-stencil) fast marching method is not causal for an anisotropic, flow-type Hamiltonian. It can accept a point with a larger arrival time before one with a smaller arrival time, which produces visible distortions of fronts and Voronoi bisectors (boat-sail metric, uniform flow) [verified] — [Nishida & Sugihara, METR 2003-45](https://www.keisu.t.u-tokyo.ac.jp/data/2003/METR03-45.pdf)
- Even for isotropic speed, FMM has a large error along diagonal directions. The multi-stencil FMM uses 6 stencils covering the 26 neighbours in 3D [unverified, abstract-level] — [Hassouna & Farag, CVPR 2006](https://mlanthology.org/cvpr/2006/hassouna2006cvpr-accurate)
- OUM (Sethian & Vladimirsky) is single pass, O(M log M), and uses characteristic-direction information. The 2003 SINUM paper gives OUMs for general anisotropic static HJ equations and proves convergence of one scheme to the viscosity solution [unverified, search summary of abstracts] — [Sethian & Vladimirsky, SINUM 41(1) 2003](https://pi.math.cornell.edu/~vlad/papers/oums_sinum_03.pdf); [PNAS 2001](https://pmc.ncbi.nlm.nih.gov/articles/PMC58684)
- OUM cost is reported as O(η M log M), with η an anisotropy coefficient [unverified, secondary] — [search result pointing to Alton & Mitchell, JSC 2011](https://www.cs.ubc.ca/~mitchell/Papers/myJSC11.pdf)
- At high anisotropy, OUM beats adaptive Gauss–Seidel only above about 725² nodes [unverified, search summary] — [Bornemann & Rasch, arXiv math/0403517](https://arxiv.org/pdf/math/0403517)
- Fast sweeping (Tsai, Cheng, Osher, Zhao) has Godunov fluxes for strictly convex homogeneous Hamiltonians such as sqrt(ap²+bq²−2cpq) with c² < ab (that is, 2D Riemannian), uses Gauss–Seidel sweeps and needs no sorting [unverified, search summary] — [UCLA CAM 01-27](https://ww3.math.ucla.edu/camreport/cam01-27.pdf)
- Lax–Friedrichs sweeping (Kao, Osher, Qian, JCP 196 (2004) 367–391) handles arbitrary static HJ in any dimension, but unlike Godunov-based sweeping it needs computational boundary conditions [unverified, search summary] — [Kao, Osher, Qian 2004](https://users.math.msu.edu/users/qian/papers/KaoOsherQianJCP2004.pdf)
- The Fast Iterative Method (Jeong & Whitaker, SISC 30(5):2512–2534, 2008) keeps an active list updated until convergence, with no ordered data structure, and targets parallel hardware. It was extended to triangulated domains and GPUs in SISC 33(5) 2011 [unverified, record summaries] — [UUCS-07-010](https://www-old.cs.utah.edu/docs/techreports/2007/pdf/UUCS-07-010.pdf); [Fu et al. 2011](https://scholarworks.unist.ac.kr/handle/201301/2929)
- An experimental survey compares FMM, FSM, FIM and variants [not read in detail] — [Gómez et al., arXiv 1506.03771](https://arxiv.org/pdf/1506.03771)
- Point-source singularities limit finite-difference eikonal solvers to "polluted" first-order convergence. Factoring the traveltime (multiplicative or additive factor) in anisotropic fast sweeping restores clean first order [unverified, abstract] — [Luo & Qian, JSC 52 (2012) 360–382](https://users.math.msu.edu/users/qian/papers/LuoQianJSC2012.pdf)

### Inferences
- [inference] The ~29° arrival bias seen with Dijkstra comes from the fixed graph stencil, which can only follow lattice directions (a metrication error). Adaptive-stencil FM (VR1) chooses offsets aligned with each local tensor, so it is the principled fix. Plain 6-neighbour FMM with an anisotropic metric is not causal (boat-sail evidence) and should be avoided.
- [inference] At 160 µm and ~140^3 voxels (~2.7 M points), HFM-CPU cost of O(12 N ln N) is seconds-scale. Grid size is not a constraint, but stencil width is: offsets grow with Cond(D), so a strong anisotropy (for example 10:1) gives offsets of several voxels. In thin cortex (a few mm ≈ 15–30 voxels) this interacts with walls and domain boundaries; IPOL notes adaptive stencils are "often 5 to 10 pixels" wide in strongly anisotropic cases.
- [inference] The sources here are a surface (the WM boundary), not isolated points, so the point-source singularity problem (factoring) matters little. Seeds are a band of voxels with given values.

### Gaps
- No quantitative angular-bias numbers for FM-VR1 vs Dijkstra in 3D were found. The SINUM paper's tables were not read.
- The exact definition and scaling of the OUM anisotropy factor in 3D was not verified from the primary text.
- The pip/conda package name and current maintenance status of `agd` were not verified (the notebook mentions a "conda package").
- Whether HFM exposes source factoring for Riemann3 was not checked.

## Source tracking / labels in fast marching (geodesic Voronoi)

### Takeaway
Propagating the label of the upwind data that produced the accepted value is standard: Kimmel–Sethian, Peyré's numerical tours, Nishida–Sugihara Algorithm 2, and HFM `seedFlags` → `voronoiFlags`. It costs no extra asymptotic time. With wide anisotropic stencils, label boundaries blur, and the literature gives no formal consistency proof for labels.

### Cited Findings
- Kimmel & Sethian use FMM on triangulated surfaces to compute geodesic Voronoi diagrams and offsets, including weighted domains, in O(M log M) [unverified, search summary] — [Kimmel & Sethian 1999](https://www.cs.technion.ac.il/wp-content/ron-kimmel/papers/voronoi_malo99.pdf)
- Peyré's numerical tours compute a Voronoi segmentation during FM propagation, storing region labels in a partition array Q [unverified, search summary] — [Numerical tours, fastmarching_1_2d](https://nbviewer.jupyter.org/github/gpeyre/numerical-tours/blob/master/matlab/fastmarching_1_2d.ipynb)
- Nishida & Sugihara, Algorithm 2: each grid point stores a harbour number S_ij. When a recomputed T is smaller, S_ij is set to "the harbor number of the neighbor grid points whose values are used" in the update. Complexity stays O(N log N), independent of the number of sources [verified] — [METR 2003-45](https://www.keisu.t.u-tokyo.ac.jp/data/2003/METR03-45.pdf)
- HFM: assigning `seedFlags` (labels) triggers computation of the Voronoi diagram (output `voronoiFlags`). Identical labels merge regions. `voronoiStoppingCriterion: 'RegionsMeeting'` stops when regions meet and returns the minimal geodesic through the meeting point in two halves [verified] — [IPOL 2019](https://doi.org/10.5201/ipol.2019.227); [agd Riemannian notebook §4](https://nbviewer.org/github/Mirebeau/AdaptiveGridDiscretizations_showcase/blob/master/Notebooks_FMM/Riemannian.ipynb)
- Mirebeau: Voronoi diagrams are "more challenging with anisotropic metrics", because "wide adaptive stencils ... risks blurring the lines between the regions". Results are "qualitatively convincing" because FM-VR1 stencils stay rather compact [verified] — [agd Riemannian notebook](https://nbviewer.org/github/Mirebeau/AdaptiveGridDiscretizations_showcase/blob/master/Notebooks_FMM/Riemannian.ipynb)
- Pink library: FM label propagation gives a partition "similar to a Voronoi diagram", which is the Euclidean Voronoi diagram for unit speed [unverified, search summary] — [Pink fmm.c](https://perso.esiee.fr/~coupriem/Pink/doc/html/fmm_8c.html)

### Inferences
- [inference] Pitfall: Nishida–Sugihara's rule is ambiguous when the neighbours used in one update have different labels. In anisotropic VR1 a 12-point stencil near a boundary will often mix labels. A consistent rule: for each candidate label ℓ, solve the local update using only accepted neighbours carrying ℓ (treat others as +∞, like Far/Trial points), then take the minimum over ℓ and assign that ℓ. Each per-label update still satisfies λ* ≥ U(q_i), so accept-in-increasing-order causality is kept. The cost is one extra quadratic per distinct label in the stencil (usually 1–2).
- [inference] HFM's built-in `voronoiFlags` path probably does not do per-label updates (the scheme mixes all accepted neighbours). This is fine for equal-speed Voronoi, where mixing at the interface gives an O(h)-consistent value, but not for label-dependent speed (next section).
- [inference] With one label per WM-boundary voxel (thousands of labels), labels act as "root IDs". Per-label stencil updates stay cheap because only a few distinct labels meet at any voxel.

### Gaps
- No paper found that proves convergence of discrete Voronoi labels (the label boundary set) for anisotropic FM.
- Kimmel–Sethian's label-assignment rule was not read in the primary text.

## Weighted competition: additive (start-time) and multiplicative (speed) weights, label-dependent speed, one-pass solvability

### Takeaway
Two different objects must not be confused. (a) The pure additively/multiplicatively weighted geodesic Voronoi diagram, argmin_r (d_r(x) − μ_r)/σ_r with d_r the unobstructed distance: its regions can be disconnected, and a fast source can wrap around a slow one. (b) The crystal-growth version, where each front advances at its own speed but cannot enter territory already claimed. A single Dijkstra-like FM pass ordered by rescaled time τ, with speed taken from the label, computes (b), not (a). Kobayashi–Sugihara did exactly this for the multiplicatively weighted crystal-growth Voronoi diagram. Additive weights (start times −μ/σ) are trivial: seed values. HFM supports these (`seedValues`) but not label-dependent speed.

### Cited Findings
- HFM accepts per-seed boundary values (`seedValues`), for example u(p) = min{‖p−p1‖, σ2 + ‖p−p2‖}, which is an additively weighted distance [verified] — [IPOL 2019, Fig. 7](https://doi.org/10.5201/ipol.2019.227)
- The multiplicatively weighted crystal-growth Voronoi diagram partitions the plane among crystals with different growth speeds. Kobayashi & Sugihara derive the boundary analytically for a simple case and approximate the general case with a fast marching method (FGCS 2002) [unverified, abstract via search] — [Kobayashi & Sugihara, RIMS Kôkyûroku 1185](https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1185-12.pdf)
- The crystal-growth diagram differs from the ordinary multiplicatively weighted Voronoi diagram because a crystal cannot enter an area that is already occupied [unverified, search summary of the same paper] — [Kobayashi & Sugihara](https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1185-12.pdf)
- The multiplicatively weighted crystal-growth Voronoi diagram was introduced by Schaudt & Drysdale (1991); a Dartmouth thesis treats crystals with different constant growth rates [unverified, record] — [Multiplicatively Weighted Crystal Growth Voronoi Diagrams](https://digitalcommons.dartmouth.edu/dissertations/64)
- Competing first-passage growth of two species with their own passage-time distributions on Z^d (each site owned by the first arrival) is well approximated at large scale by a deterministic crystal-growth model in R^d, with critical phenomena in a cone geometry [verified, abstract] — [Blair-Stahn, arXiv 1212.6254](https://arxiv.org/abs/1212.6254)
- Nishida & Sugihara: FM-based Voronoi with an anisotropic (flow) metric; Voronoi complexity independent of the number of sources [verified] — [METR 2003-45](https://www.keisu.t.u-tokyo.ac.jp/data/2003/METR03-45.pdf)
- HFM supports time-dependent cost c(p,u) through an explicit scheme with an O(h²) extra error [verified] — [IPOL 2019](https://doi.org/10.5201/ipol.2019.227)

### Inferences
- [inference] Well-posedness of (b): define τ(x) = min over r and over paths γ from r to x inside r's own region of [−μ_r/σ_r + L_F(γ)/σ_r]. Because speed is fixed per label and the seed offsets are just initial values, this is a Dijkstra-ordered problem. Accept points in increasing τ, and update each Trial point per label (previous section) with the cost scaled by 1/σ_ℓ. The causality property (update ≥ used neighbour values) holds per label, so one pass suffices. Negative start times are fine: they only shift the queue keys.
- [inference] Equal μ and σ reduce (b) to the ordinary geodesic Voronoi diagram (the midline), as desired. For (b), regions are connected to their seed by construction. For (a), they need not be. A root with large σ claims its whole unobstructed reach, including space "behind" slower neighbours.
- [inference] Which one matches "S_r(x) = arc length along r's own column"? A column is a single characteristic, not a region. The physically natural model is (b) with a strongly anisotropic metric (cheap along the layer normal, expensive across it). The anisotropy ratio then sets how much a fast root's territory can spread sideways into slow neighbours' columns. With ~1 label per root voxel, low anisotropy lets high-σ roots take over laterally. This is a modelling pitfall, not a numerical one.
- [inference] Non-monotone weights: the per-label speed must be positive (σ_r > 0). Arbitrary μ_r is fine. If σ or μ were allowed to depend on the arrival time or on the path history ("history-dependent speed", as tried with Dijkstra), Dijkstra ordering is no longer guaranteed to be correct. HFM handles only explicit c(p,t) dependence.
- [inference] Using HFM with no code changes: label-dependent speed cannot be expressed. Two approximations: (i) extend σ (and μ) from roots into the cortex along the column field, and use cost(x) = 1/σ(x) plus seedValues = −μ/σ. This is a position-dependent proxy, exact only if σ is constant along each column and columns do not cross. (ii) Run a custom FM (Python+numba, or modify HFM C++) with per-label updates. For ~2.7 M voxels a numba heap FM with 12-point VR1 stencils from `agd`'s Selling decomposition is feasible.
- [inference] Option (a) is O(#roots) separate solves if done naively. It can be localised if the metric confines each root to a thin column (narrow-band per-root solves), but it loses the "no overlap" guarantee.

### Gaps
- Kobayashi–Sugihara's FM modification was not read in the primary text: how they assign speed at mixed-label stencils, and their accuracy.
- No paper was found on anisotropic/Riemannian multiplicatively weighted crystal growth with per-label update rules or convergence proofs.
- "Competing fast marching" for DTI with multiple competing seeds was not found as such (see the brain-imaging section).

## Carrying arc length S and per-source quantities along characteristics

### Takeaway
Auxiliary quantities can be transported with the front in the same pass. HFM already computes the Euclidean length of the minimal geodesic simultaneously with u. Extension velocities (∇T·∇F = 0) and Yezzi–Prince's upwind length PDEs are the classical Eulerian versions. Per-source constants (μ_r, σ_r, r) are carried as the label.

### Cited Findings
- HFM computes the Euclidean length l(p) of the minimal path from each point simultaneously with u(p), usable as a stopping criterion [verified] — [IPOL 2019, Fig. 9 and §3](https://doi.org/10.5201/ipol.2019.227)
- HFM exports the upwind geodesic flow (`exportGeodesicFlow`), V = D∇u in the Riemannian case. Its norm is not exactly 1 because of discretisation error [verified] — [IPOL 2019](https://doi.org/10.5201/ipol.2019.227); [agd Riemannian notebook](https://nbviewer.org/github/Mirebeau/AdaptiveGridDiscretizations_showcase/blob/master/Notebooks_FMM/Riemannian.ipynb)
- Yezzi & Prince (IEEE TMI 22(10):1332–1339, 2003): thickness is the length of correspondence trajectories that follow a smooth vector field between two boundaries. They solve a pair of PDEs guided by the field, whose sum gives thickness, with upwind finite differences [verified, abstract/record] — [Georgia Tech repository](https://repository.gatech.edu/entities/publication/b0159a94-1b8a-457e-ba62-d728d13cd34a)
- The Eulerian approach has limited thickness accuracy in thin structures because it is solved on a discrete grid [unverified, quoted by search from Acosta et al. 2009] — [PMC3068613](https://pmc.ncbi.nlm.nih.gov/articles/PMC3068613)
- Extension velocities are covered in Sethian's 1999 book (Ch. 11) and implemented in ITK's ExtensionVelocitiesImageFilter [verified, docs] — [ITK doc](https://docs.itk.org/projects/doxygen/en/v2.2.0/classitk_1_1ExtensionVelocitiesImageFilter.html). The formula ∇T·∇F = 0 (Adalsteinsson & Sethian, JCP 1999) is [unverified]; the paper itself was not retrieved.

### Inferences
- [inference] S along the characteristic satisfies ⟨∇S, V/|V|⟩ = 1 (unit Euclidean rate along the geodesic flow). In FM it can be computed at acceptance time from the same upwind stencil weights that determine u: interpolate S at the stencil "foot" and add the Euclidean step. This is what HFM's l(p) does in principle.
- [inference] Key pitfall: if the competition key is (S − μ)/σ with S the Euclidean length, but the front is ordered by metric time T, the two keys differ wherever the geodesic leaves the preferred direction. The single-pass guarantee holds only for the key that is ordered. Options: (1) make the metric's unit-speed direction equal arc length (cost 1 along the layer normal, higher across), so T ≈ S on well-aligned columns, and compete on T. (2) Order by T but label by S, which loses the guarantee near interfaces. (1) is the consistent choice.
- [inference] Per-source quantities (μ_r, σ_r, root id, root position) are carried exactly by storing the label. Quantities that vary along the path (S, accumulated misfit) need the upwind interpolation above. With mixed-label stencils, interpolate only over the winning label's neighbours.

### Gaps
- The exact Yezzi–Prince PDE form (∇L0·T = 1, etc.) and the Adalsteinsson–Sethian equations were not read in the primary texts.
- No published analysis was found of the accuracy of HFM's simultaneous Euclidean-length output.

## Locating the meeting surface at sub-voxel accuracy

### Takeaway
No dedicated source was found. Standard FM stores only the winning arrival, so a sub-voxel interface needs the losing label's arrival near the interface too. HFM's 'RegionsMeeting' gives only the first meeting point and geodesic.

### Cited Findings
- HFM returns the meeting point of two Voronoi regions together with the two half-geodesics (`geodesics_voronoiDiagram`) when stopping at 'RegionsMeeting' [verified] — [agd Riemannian notebook](https://nbviewer.org/github/Mirebeau/AdaptiveGridDiscretizations_showcase/blob/master/Notebooks_FMM/Riemannian.ipynb)
- Bisector distortions in FM-based Voronoi diagrams come from non-causal stencils, as in the boat-sail case [verified] — [Nishida & Sugihara](https://www.keisu.t.u-tokyo.ac.jp/data/2003/METR03-45.pdf)

### Inferences
- [inference] Method: in the per-label update, keep the best and second-best (label, τ) at each Trial point (the "two-arrival" FM). The second-best is valid only where its own upwind stencil is all from that label, so it is reliable within one or two voxels of the interface. The interface is the zero level of Δτ = τ_b − τ_a. Locate it by linear interpolation of Δτ between neighbouring voxels with opposite winners, or by marching cubes on Δτ in a narrow band. Accuracy is then O(h) set by τ's first-order error, and better than voxel quantisation.
- [inference] Alternative: after the pass, run short local solves for the two competing labels in a narrow band around the label boundary (both fronts extended a few voxels past the interface), then take the Δτ = 0 isosurface.
- [inference] This is the same "shock = equal arrival" locus discussed in shock_theory.md. The Δτ isosurface is that shock set for crystal-growth semantics.

### Gaps
- No paper found on sub-voxel Voronoi/shock interface extraction from fast marching with multiple labels. The search was limited; the medial-axis/shock literature (Siddiqi et al.) was not checked here.

## Applications in brain imaging

### Takeaway
Eulerian thickness PDEs (Yezzi–Prince) and FM tractography (Parker et al.; Jbabdi et al. with anisotropic FM) are the closest relatives. No application of label-competing anisotropic FM to cortical columns was found.

### Cited Findings
- Yezzi & Prince 2003 Eulerian thickness (see above) [verified] — [Georgia Tech repository](https://repository.gatech.edu/entities/publication/b0159a94-1b8a-457e-ba62-d728d13cd34a); CVPR 2001 precursor [verified, record] — [mlanthology](https://mlanthology.org/cvpr/2001/yezzi2001cvpr-pde/)
- Fast marching tractography (Parker et al. 2002): a front evolves with speed proportional to the colinearity between the front normal and the principal diffusion eigenvector (direction-dependent speed), validated against macaque connectivity (NeuroImage 15(4), 2002) [unverified, review summary] — [ZHAW record](https://digitalcollection.zhaw.ch/handle/11475/5024)
- Staempfli et al. (NeuroImage 2006) adapted the FMT speed function to local tensor shape to resolve crossings and allow branching [unverified, summary] — [same search, Yale/ISMRM results](https://cds.ismrm.org/protected/09MProceedings/PDFfiles/01443.pdf)
- Jbabdi et al. 2008, "Accurate anisotropic fast marching for diffusion-based geodesic tractography", J. Biomed. Imaging 2008:2–12 [verified as a citation in IPOL refs] — [doi 10.1155/2008/320195](https://doi.org/10.1155/2008/320195)

### Inferences
- [inference] Parker's FMT direction-dependent speed is a non-Riemannian (non-convex in general) speed law solved with a plain FM. It shares the grid-bias and causality issues of the project's Dijkstra attempt. A Riemannian metric from the coherence-weighted normal field, solved with FM-VR1, is the cleaner formulation.

### Gaps
- No published use of competing-label anisotropic FM for cortical columns or laminar depth was found within the call budget. DTI "competing seeds" connectivity papers (for example multi-seed geodesic connectivity) were not located.
