# Multi-surface layered segmentation: lessons for thin, ordered layers with varying count

Scope: optimal surface graph search (Li/Wu/Chen/Sonka, LOGISMOS, Song priors), OCT retinal layers (Garvin, Chiu, Lang/AURA, He et al. deep nets, Carass MGDM), coupled surfaces / multi-object level sets (Zeng, MacDonald, Bogovic MGDM). The project context: one laminar depth field T for ex vivo cortex MRI at 160 µm. Bands are at least 2–3 voxels thick, thinner layers ride along, and bands pinch out laterally. The current code places band boundaries per column graph in an optimal-surfaces style.

Verification status: every citation below with a DOI/PMC link was checked against an index or full-text record in this session (title, authors, venue, year). Items marked **[UNVERIFIED]** were not checked.

---

## Q1. Optimal surface graph search (Li 2006; Song 2013; LOGISMOS 2010; Garvin 2009): what it represents, its constraints and its limits

### Takeaway
The Li et al. 2006 formulation finds k terrain-like (height-field) surfaces, one voxel per column, at the global optimum of a sum of per-voxel costs, by minimum s-t cut. It has hard smoothness bounds Δx, Δy and hard min/max separations δl, δu between surface pairs. δl = 0 is in the notation, so coincident surfaces (zero thickness) are representable. Constraints can vary by location: Garvin 2009 learned them per column, with the minimum truncated at 0. Smoothness is hard and not discontinuity-preserving, so a real lateral depth jump is penalized. That fits a model of "no depth jumps". Song 2013 replaced hard constraints with convex soft priors on shape and on inter-surface distance, which improved OCT error from 6.30 to 5.14 µm.

### Cited Findings
- **Citation:** K. Li, X. Wu, D. Z. Chen, M. Sonka, "Optimal Surface Segmentation in Volumetric Images—A Graph-Theoretic Approach," IEEE TPAMI 28(1):119–134, Jan 2006, DOI 10.1109/TPAMI.2006.19, PMC2646122 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122); [CMU RI record](https://ri.cmu.edu/publications/optimal-surface-segmentation-in-volumetric-images-a-graph-theoretic-approach)
- **Li 2006, representation:** each surface intersects exactly one voxel of each z-column (the "multicolumn" model). Terrain-like surfaces are handled directly, and cylindrical surfaces after unfolding — [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)
- **Li 2006, smoothness:** if (x,y,z) and (x+1,y,z′) are both on a surface, then |z−z′| ≤ Δx (and likewise Δy). Smaller Δ makes the surface stiffer — [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)
- **Li 2006, separation:** for a non-crossing pair, surface N2 lies below N1 with a minimum distance of δl voxels and a maximum of δu voxels, where δl, δu ≥ 0. The paper does not discuss δl = 0 explicitly. A crossing variant instead bounds how far one surface can go above or below the other — [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)
- **Li 2006, graph construction:**
  - Node weight = c(z) − c(z−1).
  - Intracolumn arcs point downward.
  - Intercolumn arcs go to z−Δ in each 4-neighbour column.
  - Intersurface arcs go from V1(z) to V2(z−δu) and from V2(z) to V1(z+δl).
  - "Deficient" (infeasible) nodes are removed.
  - Runtime is one min-cut on a kn-node graph and is independent of Δ and δ — [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)
- **Li 2006, cost terms:**
  - Edge cost: a combination of first and second derivatives, with orientation and position penalties.
  - Region cost: a per-voxel approximation of Chan–Vese (minimum variance in and out) — [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)
- **Li 2006, limitations the authors state:**
  - Only terrain-like or cylindrical surfaces.
  - Smoothness is not discontinuity-preserving.
  - Contradictory constraints are not handled.
  - Automatically setting variable constraints is called a "key open problem". Section 8.1 says constraints can vary spatially if the arcs are rearranged — [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)
- **Li 2006, accuracy:**
  - Airway CT at 0.7×0.7×0.6 mm: unsigned error 0.10±0.11 mm (inner) and 0.12±0.12 mm (outer). This is well under one voxel.
  - Vascular MR at 0.3×0.3 mm pixels: about 0.90–0.93 pixels.
  - Diaphragm CT: 0.50±0.62 voxels — [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)
- **Citation:** M. K. Garvin, M. D. Abràmoff, X. Wu, S. R. Russell, T. L. Burns, M. Sonka, "Automated 3-D Intraretinal Layer Segmentation of Macular Spectral-Domain Optical Coherence Tomography Images," IEEE TMI 28(9):1436–1447, 2009, DOI 10.1109/TMI.2009.2016958, PMC2911837. The author list beyond Garvin was not individually checked in this session — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2911837)
- **Garvin 2009, learned spatially varying constraints:**
  - Smoothness bounds are set per neighbouring column pair, with separate maximum increase and maximum decrease. They are learned from training segmentations as mean ± 2.6σ (about 99% coverage).
  - Thickness bounds are set per column as mean thickness ± 2.6 SD, with the minimum "truncated so as not to go below 0".
  - So zero thickness is allowed where the training data supports it.
  - Training used 13 scans; testing used 28 images from 14 subjects — [PMC2911837](https://pmc.ncbi.nlm.nih.gov/articles/PMC2911837)
- **Garvin 2009, costs:**
  - The outer surfaces use signed Sobel edge costs plus local regional terms.
  - The interior surfaces combine signed edge costs with "true regional" in-region costs: Gaussian fuzzy membership in dark, medium and bright classes, cost = 1 − membership.
  - Region parameters are estimated per volume from regions defined by the learned thickness constraints.
  - The edge/region weight α was tuned per surface (0.80–0.95) — [PMC2911837](https://pmc.ncbi.nlm.nih.gov/articles/PMC2911837)
- **Garvin 2009, pipeline:**
  1. Downsample by 4.
  2. Find the outer surfaces.
  3. Fit a thin-plate spline to the bottom surface and flatten the volume.
  4. Apply anisotropic diffusion.
  5. Segment the outer surfaces jointly at full resolution, then the interior surfaces 2–5 jointly — [PMC2911837](https://pmc.ncbi.nlm.nih.gov/articles/PMC2911837)
- **Garvin 2009, accuracy:**
  - Volume is 200×200×1024 voxels over 6×6×2 mm, so voxels are about 30 µm lateral and about 1.95 µm axial (derived, nominal).
  - Overall unsigned error is 5.69±2.41 µm (about 2.9 axial voxels), against interobserver variability of 5.71±1.98 µm.
  - Per border: 2.85 µm (border 1) to 8.47 µm (border 7) — [PMC2911837](https://pmc.ncbi.nlm.nih.gov/articles/PMC2911837)
- **Citation:** Q. Song, J. Bai, M. K. Garvin, M. Sonka, J. M. Buatti, X. Wu, "Optimal Multiple Surface Segmentation With Shape and Context Priors," IEEE TMI 32(2):376–386, Feb 2013, DOI 10.1109/TMI.2012.2227120. Abràmoff is not an author — [PMC (pmid 23193309)](https://pmc.ncbi.nlm.nih.gov/articles/pmid/23193309); [Iowa record](https://iro.uiowa.edu/esploro/outputs/journalArticle/Optimal-Multiple-Surface-Segmentation-With-Shape/9984040396902771)
- **Song 2013, priors:** pairwise (arc-based) energy terms.
  - The shape prior penalizes local changes in shape relative to a model.
  - The context prior penalizes local changes in inter-surface distance relative to a model of expected distances.
  - The optimum is still global, by max-flow.
  - On OCT intraretinal layers, error went from 6.30±1.58 µm (conventional graph search) to 5.14±0.99 µm — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/pmid/23193309)
- **Citation:** Y. Yin, X. Zhang, R. Williams, X. Wu, D. D. Anderson, M. Sonka, "LOGISMOS—Layered Optimal Graph Image Segmentation of Multiple Objects and Surfaces: Cartilage Segmentation in the Knee Joint," IEEE TMI 29(12):2023–2037, Dec 2010, DOI 10.1109/TMI.2010.2058861, PMC3131162 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3131162)
- **LOGISMOS:** several interacting surfaces of several interacting objects are encoded in one n-D graph and solved to the global optimum. Knee bone and cartilage (6 surfaces): signed error from 0.04±0.12 to 0.16±0.22 mm, trained on 9 images — [PMC3131162](https://pmc.ncbi.nlm.nih.gov/articles/PMC3131162)

### Inferences
- **Fit to T:** a per-column, height-field graph search fits a depth field T with monotone isosurfaces exactly. Each column (along the T gradient or a profile direction) is a z-axis, and each band boundary is one surface.
- **Zero thickness:** δl = 0 with non-strict ordering lets two boundaries coincide, so a pinched-out band is representable. The cost function must then allow two boundaries on the same voxel. If each boundary's cost wants a real edge, a coincident pair is "paid twice" or else gets a spurious edge. A specific cost or prior is needed for the "absent band" state (see Q3).
- **Hard Δ smoothness:** this forbids depth jumps. That matches "areal borders are lateral changes in band pattern, never depth jumps". But hard Δ also makes a fast pinch-out (boundary B moving onto boundary A over a few columns) compete with smoothness. Either make Δ larger than the pinch-out slope, or make it per-column and learned (Garvin) or soft (Song).
- **Soft context priors:** Song's prior on inter-surface distance is the natural mechanism for "thickness is locally consistent but may change". A convex penalty on the change in thickness between neighbour columns lets thickness go to 0 slowly, not in one step. This suits a pinch-out that tapers.
- **Coarse-to-fine:** Garvin's order (outer, high-contrast surfaces first, then interior surfaces in a flattened volume) maps onto the user's setting. Find pia and WM first, define the depth field, then search for interior bands in "flattened" T-coordinates.

### Gaps
- Wu & Chen's earlier "optimal net surface" work (around 2002) and the closed-surface variants were not checked.
- The exact form of Song 2013's convex penalty functions (for example linear or quadratic in deviation) was not taken from the full text.
- None of these papers deals explicitly with a surface whose contrast disappears laterally while the surface is required to exist.

---

## Q2. OCT retinal layers: graph/DP methods, learned costs, deep nets with topology guarantees, and the fovea (layers that merge or vanish)

### Takeaway
OCT is the closest mature analogue: about 7–9 ordered, near-planar layers, axial pixels of 2–4 µm, errors of about 1 pixel, and layers at the fovea that thin to zero.

- **Engineered handling at the fovea:** detect the foveal region, then widen or relax search windows and constraints there (Chiu; Garvin's learned per-column bounds).
- **Better representation:** a normalized "flat space" depth coordinate (Carass MGDM). It removes the ≥1-voxel thickness floor and is conceptually the same as a laminar depth field T.
- **Deep nets:** they enforce order with ReLU on thickness, which allows zero thickness. They match graph methods at about 2.8 µm but lose 3D smoothness.

### Cited Findings
- **Citation:** S. J. Chiu, X. T. Li, P. Nicholas, C. A. Toth, J. A. Izatt, S. Farsiu, "Automatic segmentation of seven retinal layers in SDOCT images congruent with expert manual segmentation," Optics Express 18(18):19413–19428, 2010, DOI 10.1364/OE.18.019413, PMC3408910 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408910)
- **Chiu 2010, method:** a 2D B-scan shortest path (Dijkstra, a dynamic-programming method).
  - Each pixel is a node with 8-neighbour edges.
  - Weight w_ab = 2 − (g_a + g_b) + w_min, where g is the normalized vertical gradient and w_min = 1e-5.
  - There are separate weight maps for dark-to-light and light-to-dark boundaries, so edge polarity is used.
  - Endpoint trick: add a column on each side with minimal weight, so the path can start and end anywhere — [PMC3408910](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408910)
- **Chiu 2010, sequential layers:** boundaries are found one at a time, most prominent first. Each new search is limited to a region bounded by boundaries already found (for example, stop 20 px above IS–OS). So ordering comes from the search windows, not from a joint model — [PMC3408910](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408910)
- **Chiu 2010, fovea:**
  - Foveal columns are those where the mean thickness of NFL, GCL–IPL, INL and OPL falls below 5 px.
  - The centre is at the minimum total retinal thickness, ±20 px.
  - Near the fovea, the top of the search region for IPL–INL, INL–OPL and OPL–ONL is extended up to the vitreous–NFL boundary, "to handle layer merging" — [PMC3408910](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408910)
- **Chiu 2010, resolution and accuracy:**
  - Axial pixel 3.23 µm; lateral 6.7 or 13.4 µm; 10 normal adults; 100 B-scans compared.
  - Total-retina thickness difference: 0.94 px (about 3.0 µm) automatic vs. manual, against 2.22 px (about 7.2 µm) between two graders.
  - NFL: 0.88 px automatic vs. 1.73 px inter-grader — [PMC3408910](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408910)
- **Citation:** A. Lang, A. Carass, M. Hauser, E. S. Sotirchos, P. A. Calabresi, H. S. Ying, J. L. Prince, "Retinal layer segmentation of macular OCT images using boundary classification," Biomedical Optics Express 4(7):1133–1152, 2013, PMC3704094. The full author list was not individually checked in this session; DOI 10.1364/BOE.4.001133 **[UNVERIFIED]** — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3704094)
- **Lang 2013, method and accuracy:** a random forest classifies boundary pixels, giving one probability map per boundary. The maps feed a graph-based surface extraction. All nine boundaries are within 4.3 µm. There was no accuracy difference between healthy and MS subjects — [PMC3704094](https://pmc.ncbi.nlm.nih.gov/articles/PMC3704094)
- **Lang 2013, features (per Carass 2014):** 27 features per voxel:
  - position relative to the fovea
  - local intensities
  - oriented Gaussian directional derivatives
  - vertical gradients below the pixel — [PMC3986003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003)
- **Citation:** A. Carass, A. Lang, M. Hauser, P. A. Calabresi, H. S. Ying, J. L. Prince, "Multiple-object geometric deformable model for segmentation of macular OCT," Biomedical Optics Express 5(4):1062–1074, 2014, DOI 10.1364/BOE.5.001062, PMC3986003. An erratum (2015) corrects Tables 1–2 because of a file-format conversion error in the ground truth — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003); [erratum](https://pmc.ncbi.nlm.nih.gov/articles/PMC4399673)
- **Carass 2014, flat space:**
  - Each A-scan is resampled to proportional depth between Bruch's membrane (0) and the ILM (1): 89 samples, plus 10 padding samples above and below.
  - The mean proportional positions of the internal boundaries are learned from 37 scans with the foveas aligned.
  - So depth is normalized by local retinal thickness — [PMC3986003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003)
- **Carass 2014, the thin-layer argument:**
  - Voxel-grid methods often force each layer to be at least one voxel thick, even where the true layer is thinner or absent (fovea).
  - Graph methods can allow zero thickness, but they still do not represent true sub-voxel thickness.
  - The LCLS level-set ordering needs at least one voxel between neighbours, so a thin layer forces its neighbours to be thick.
  - Upsampling is rejected: it keeps the floor and adds Gibbs ringing.
  - Working in flat space and mapping back allows surfaces closer than one voxel. Boundaries are taken as sub-voxel isosurfaces of the level-set functions — [PMC3986003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003)
- **Carass 2014, forces:** a gradient vector flow (GVF) field from each RF boundary-probability map gives a capture range over the whole domain. Speed = GVF · ∇(signed distance), plus a curvature regularizer that suits flat space — [PMC3986003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003)
- **Carass 2014, limitations the authors state:**
  - About 2 h 9 min per volume.
  - The flat mapping may fail in strong pathology.
  - The same 37 subjects built the flat-space model and validated it.
  - Ground truth is from a single rater.
  - Cross-study comparisons are flawed because scanners and layer definitions differ.
  - No explicit analysis of a layer that vanishes entirely — [PMC3986003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003)
- **Citation:** Y. He, A. Carass, Y. Yun, C. Zhao, B. M. Jedynak, S. D. Solomon, S. Saidha, P. A. Calabresi, J. L. Prince, "Topology guaranteed segmentation of the human retina from OCT using convolutional neural networks," arXiv:1803.05120 (2018). The publication venue (MICCAI 2018?) and the full author list are **[UNVERIFIED]** — [arXiv](https://arxiv.org/pdf/1803.05120)
- **He 2018, method:** a cascade of two networks. The first labels pixels; the second regresses layer thicknesses. A final ReLU keeps thicknesses non-negative, so boundaries made by cumulative sums stay in order. Reported MAD is 2.82 µm vs. 2.83 µm for graph methods. The authors argue that graph methods need distance and smoothness parameters tuned by hand per region and per pathology — [arXiv](https://arxiv.org/pdf/1803.05120); [DeepAI summary](https://deepai.org/publication/topology-guaranteed-segmentation-of-the-human-retina-from-oct-using-convolutional-neural-networks)
- **Citation:** Y. He, A. Carass, Y. Liu, B. M. Jedynak, S. D. Solomon, S. Saidha, P. A. Calabresi, J. L. Prince, "Structured layer surface segmentation for retina OCT using fully convolutional regression networks," Medical Image Analysis 68:101856, 2021 (online Oct 2020), PMC7855873 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7855873)
- **He 2021, surface representation and order:**
  - Each surface is one row position per A-scan column. A per-column softmax over rows plus soft-argmax gives a sub-pixel position.
  - Order is enforced in the output layer, at training and at test, by a ReLU update of each surface relative to the one above it, starting from the ILM.
  - The order is non-strict, so equal positions (zero thickness) are allowed. This is an inference; the paper does not discuss it.
  - Because the update runs top-down, an error in an upper surface can propagate downward. The authors flag this as a limitation — [PMC7855873](https://pmc.ncbi.nlm.nih.gov/articles/PMC7855873)
- **He 2021, the fovea and graph methods:** the authors note that shortest-path methods can fail at the fovea, where distances between surfaces can be zero, and that they do not guarantee order there — [PMC7855873](https://pmc.ncbi.nlm.nih.gov/articles/PMC7855873)
- **He 2021, losses:** cross entropy on the discretized surface distribution, plus smooth-L1 on the soft-argmax positions. A separate pixel-wise branch (Dice + CE) handles lesions, which have no fixed topology — [PMC7855873](https://pmc.ncbi.nlm.nih.gov/articles/PMC7855873)
- **He 2021, data and accuracy:**
  - Spectralis OCT: axial 3.9 µm, lateral 5.8 µm, B-scan spacing 123.6 µm.
  - 14 healthy controls and 21 people with MS, 9 surfaces. MAD 2.83 µm vs. AURA (Lang 2013 retrained) 2.95 µm (p < 0.05). AURA is better on ILM and BM.
  - DME data: MAD 6.70 µm vs. 7.71–9.54 µm for other methods.
  - In edema, surface probability is lower but the soft-argmax surfaces stay smooth.
  - Lost relative to graph methods: smoothness between B-scans (3D) — [PMC7855873](https://pmc.ncbi.nlm.nih.gov/articles/PMC7855873)

### Inferences
- **Flat space is the depth field.** Carass's flat space is in effect an equivolume-free, proportional laminar depth coordinate for the retina. The user's T is the same idea. The key lesson: segment interior bands in T-space with a sub-voxel boundary representation, then map back. This removes the one-voxel thickness floor that native-grid methods impose. It matters at 160 µm when thin layers ride along.
- **Representation over special cases.** Fovea handling in Chiu (a detected region, then widened search windows) is an ad hoc pinch-out handler. Learned per-column bounds (Garvin), a non-negative thickness parameterization (He), or a depth coordinate (Carass) handle merging without special cases. For areal borders that occur at unknown places, a special-case detector does not generalize. Prefer the representation that allows zero thickness everywhere, with costs that decide.
- **Thickness parameterization.** "Boundaries = cumulative sum of non-negative thicknesses" (He 2018) is a clean monotone parameterization. It suits the user's per-column graph: the unknowns can be thicknesses with a prior at 0, not positions.
- **Top-down error propagation.** He 2021 reports a failure mode of sequential or top-down ordering: an error in one surface pushes all deeper surfaces. Joint optimization (Li/Garvin min-cut) avoids this.
- **Accuracy scale.** Mature OCT methods reach about 0.7–1.5 axial pixels MAD (2.8–5.7 µm at 2–4 µm pixels), about equal to inter-rater variability. At 160 µm, a realistic target for clear bands is about 1 voxel MAD. Thinner bands should be judged against inter-rater variability, not against zero.

### Gaps
- No OCT paper found here models a layer that loses contrast laterally while it still exists (as opposed to thinning). Pathology work (DME) handles this with a separate lesion branch, not with surface logic.
- The Chiu 2015 and Rathke 2017 DME methods (probabilistic graphical models with shape priors, relevant to missing layers in pathology) were not checked.
- Lang 2013's exact graph constraints were not taken from the full text.

---

## Q3. Coupled surfaces and multi-object level sets (Zeng 1999; MacDonald 2000; Bogovic MGDM 2013): how they keep order and whether they allow zero thickness

### Takeaway
- **Coupled-surface methods for cortex** (Zeng; MacDonald) keep two surfaces ordered with a thickness or proximity coupling term. They were designed for a cortex that is roughly constant in thickness, not for vanishing layers.
- **MGDM** keeps any number of objects without gaps or overlaps, with topology fixed by a template ("digital homeomorphism"). Only objects that are adjacent in the template may touch.
- On a voxel grid, level-set ordering usually implies at least one voxel between neighbours. Zero or sub-voxel thickness needs a transformed (flat) domain and isosurface extraction (Carass 2014).

### Cited Findings
- **Citation:** X. Zeng, L. H. Staib, R. T. Schultz, J. S. Duncan, "Segmentation and Measurement of the Cortex from 3-D MR Images Using Coupled-Surfaces Propagation," IEEE TMI 18(10):927–937, 1999 (index record checked; full text not read) — [dblp](https://dblp.dagstuhl.de/pid/36/5199.html); [MICCAI 1998 version](https://rd.springer.com/content/pdf/10.1007/BFb0056237.pdf)
- **Zeng 1999, method:**
  - Two embedded level-set surfaces evolve together, each driven by its own image information, with a coupling between them.
  - The method is motivated by the nearly constant thickness of cortex.
  - It gives area, curvature and thickness.
  - The exact coupling form — a speed that depends on the distance to the other surface, kept inside a [min, max] thickness range — is **[UNVERIFIED]** (from memory) — [MICCAI 1998 version](https://rd.springer.com/content/pdf/10.1007/BFb0056237.pdf)
- **Goldenberg et al. 2002 (TMI):** reformulated Zeng's coupled surfaces as a variational problem, noting that the original propagation equations were empirical — [Goldenberg et al. 2002 PDF](https://csaws.cs.technion.ac.il/~ehudr/publications/pdf/GoldenbergKRR02a.pdf)
- **Citation:** D. MacDonald, N. Kabani, D. Avis, A. C. Evans, "Automated 3-D Extraction of Inner and Outer Surfaces of Cerebral Cortex from MRI," NeuroImage 12:340–356, 2000, DOI 10.1006/nimg.1999.0534 — [PDF](https://pages.ucsd.edu/~msereno/569/readings/mcdonald.pdf)
- **MacDonald 2000:** the two new features are explicit prevention of self-intersecting surfaces and deformation of several surfaces with intersurface proximity constraints. CLASP is a later, separate paper (Kim et al., NeuroImage 27(1):210–221, 2005), a Laplacian-constrained variant — [PDF](https://pages.ucsd.edu/~msereno/569/readings/mcdonald.pdf); [summary of Kim 2005](https://s-space.snu.ac.kr/handle/10371/11567)
- **Citation:** J. A. Bogovic, J. L. Prince, P.-L. Bazin, "A multiple object geometric deformable model for image segmentation," Computer Vision and Image Understanding 117(2):145–157, 2013, DOI 10.1016/j.cviu.2012.10.006, PMC3539759 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3539759)
- **MGDM:**
  - Keeps object relationships and topology, and avoids overlaps and gaps.
  - The level-set decomposition builds in digital homeomorphism constraints.
  - Supports objects with several compartments.
  - A template with the desired topology initializes the evolution, and objects can touch only if they are adjacent in the template — [PMC3539759](https://pmc.ncbi.nlm.nih.gov/articles/PMC3539759); [IACL MGDM page](https://iacl.ece.jhu.edu/index.php?title=MGDM)
- **MGDM and order for ordered layers:**
  - For ordered layers, digital homeomorphism gives the same order constraint as LCLS.
  - In native space that implies a minimum thickness of about one voxel.
  - Flat space plus isosurface extraction removes it — [Carass 2014, PMC3986003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003)

### Inferences
- **Fixed topology versus pinch-out.** A topology-preserving multi-object method (MGDM) cannot let a band vanish over a region, because the band object would split or change topology. It also forbids non-adjacent bands from touching. For pinch-outs, this is the wrong constraint unless the band may degenerate to zero thickness in a continuous coordinate. Height-field and graph models that allow δl = 0 are more permissive. That favors the user's T-plus-column-graph design over a multi-object level set.
- **Two-surface coupling as a precedent.** Coupled-surface cortex methods are a precedent for the outer pair (pia and WM) with a thickness range. The interior bands are better handled in T-space.

### Gaps
- Zeng's coupling equation and its thickness bounds (in mm) were not taken from the TMI full text.
- The Bazin/Waehnert-type equivolume layering work (cortical depth fields) is the direct cortex analogue. It was outside this scope and was not checked here.

---

## Q4. Ordinal/monotone parameterizations and 2.5D (per-column DP plus joint across columns)

### Takeaway
Three recurring patterns:

1. **Height fields per column, solved jointly in 3D** (Li 2006 min-cut). Global optimum with hard inter-column smoothness.
2. **Per-column or per-B-scan 2D DP or shortest path** (Chiu). Fast, but order comes from sequential windows and there is no 3D consistency.
3. **Monotone parameterizations:**
   - non-negative thickness increments (He 2018)
   - ReLU-ordered positions (He 2021)
   - normalized depth coordinate, "flat space" (Carass 2014)

He 2021 notes explicitly that its per-B-scan network gives up the 3D smoothness that graph methods have.

### Cited Findings
- **Li 2006:** one voxel per column, many surfaces, solved jointly over all columns with min-cut. Runtime does not depend on Δ or δ — [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)
- **Chiu 2010:** a 2D shortest path per B-scan; boundaries segmented one after another within bounded search regions — [PMC3408910](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408910)
- **He 2018:** boundaries come from non-negative thickness outputs (ReLU), so order is guaranteed — [arXiv](https://arxiv.org/pdf/1803.05120)
- **He 2021:**
  - Positions per column come from soft-argmax, with order set by ReLU.
  - Graph methods' advantage is smoothness between B-scans (3D).
  - Graph methods cannot be trained end-to-end with a deep network — [PMC7855873](https://pmc.ncbi.nlm.nih.gov/articles/PMC7855873)
- **Carass 2014:** a proportional depth coordinate per A-scan, from the mean boundary positions learned in flat space — [PMC3986003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003)
- **Garvin 2009:** a flattening transform (thin-plate spline fit to a reference surface) before the interior surfaces are searched — [PMC2911837](https://pmc.ncbi.nlm.nih.gov/articles/PMC2911837)

### Inferences
- **The user's design is pattern 3 plus pattern 1.** T is a monotone depth warp ("flat space"). Band boundaries are height fields in T-coordinates. To get the lateral coupling that OCT 2D methods lack, solve the columns jointly: either a Li-style min-cut over all columns, or alternate between per-column DP and lateral smoothing.
- **Soft thickness continuity, not hard Δ.** Areal borders are changes in band pattern, not depth jumps. So the lateral coupling should be on thickness or contrast (Song's context prior), not only on position. Position smoothness is mostly inherited from T.

### Gaps
- No "monotone warp of a template" layer method outside OCT and cortex was found in this session. Examples would be layered geology or seismic horizon picking, which has a direct pinch-out literature and may be worth a separate search.

---

## Q5. Cost/feature design, and lessons for (a) thin layers, (b) vanishing or low-contrast layers, (c) costs, (d) validation

### Takeaway
- **Costs.** The field moved from hand-made polarity-aware edge costs (Chiu, Li) to edge-plus-region hybrids with weights tuned per surface (Garvin), then to learned boundary probabilities (Lang RF, then CNN). Learned costs combined with structured inference beat either part alone.
- **Validation.** Mature practice reports:
  - signed and unsigned boundary error in µm and in voxels, per surface
  - compared against inter-rater variability
  - with the train/test split and the rater count stated
  - with honest notes on mismatched cross-study comparisons

### Cited Findings
- **Polarity-aware edges:** Chiu uses separate dark-to-light and light-to-dark gradient maps. Li 2006 includes an orientation term in its edge cost — [PMC3408910](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408910); [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)
- **Region costs:**
  - Garvin's "true regional" costs (fuzzy dark/medium/bright membership) are weighted with edge costs, α = 0.80–0.95 per surface.
  - Region parameters are re-estimated per volume from the constraint-defined regions.
  - Li 2006 gives a Chan–Vese region cost — [PMC2911837](https://pmc.ncbi.nlm.nih.gov/articles/PMC2911837); [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)
- **Learned costs:**
  - A random forest with 27 features (including position relative to a landmark, the fovea) gives boundary probabilities, then graph extraction: within 4.3 µm.
  - The same probabilities drive the MGDM forces through GVF — [PMC3704094](https://pmc.ncbi.nlm.nih.gov/articles/PMC3704094); [PMC3986003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003)
- **Priors on inter-surface distance:** these improve accuracy over hard constraints alone (6.30 to 5.14 µm) — [Song 2013](https://pmc.ncbi.nlm.nih.gov/articles/pmid/23193309)
- **Validation practice:**
  - Garvin: error per border against the average of two ophthalmologists, with interobserver variability (5.71 µm) beside the algorithm (5.69 µm). Separate training (13) and test (28) sets — [PMC2911837](https://pmc.ncbi.nlm.nih.gov/articles/PMC2911837)
  - Chiu: thickness difference per layer in px and µm, against a second grader on a subset (29 B-scans) — [PMC3408910](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408910)
  - He 2021: baseline retrained on the same data, significance test (p < 0.05), MAD and RMSE per surface — [PMC7855873](https://pmc.ncbi.nlm.nih.gov/articles/PMC7855873)
  - Carass 2014: lists the weaknesses of its own validation (one rater, the same subjects for the model and the test, invalid cross-study comparisons), and an erratum corrected a ground-truth format-conversion error — [PMC3986003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003); [erratum PMC4399673](https://pmc.ncbi.nlm.nih.gov/articles/PMC4399673)
- **Sub-voxel accuracy is normal for clear boundaries:** Li 2006 airway inner and outer walls, about 0.10–0.12 mm at 0.6–0.7 mm voxels — [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)

### Inferences (lessons for the user's method)

**(a) Thin layers near resolution**
- Do not represent a thin layer as a set of voxels. Use boundary positions in a continuous depth coordinate (T) with sub-voxel output (soft-argmax, or parabolic interpolation of the column cost). Carass's analysis shows that voxel-grid ordering forces thickness ≥ 1 voxel, and that this distorts the neighbouring layers too.
- Track only bands of 2–3 or more voxels as independent surfaces. Thinner layers should "ride along" as fixed fractional positions between tracked boundaries, the way flat space places the inner retinal boundaries at learned proportional depths. That matches the user's plan.

**(b) Layers that vanish or lose contrast**
- Allow δl = 0 (Li, Garvin) or non-negative thickness (He), so a pinch-out is representable everywhere without a special-case detector.
- Add an explicit cost for the "absent band" state, and a soft prior on lateral thickness change (Song-style). Two coincident boundaries should not have to find two edges.
- Treat contrast loss separately from thinning. OCT handles missing or abnormal contrast with a separate pixel-wise branch (He 2021 lesions) and observes that soft-argmax surfaces "stay smooth" through low-probability regions, that is, the prior carries them.
- For the user: where contrast vanishes, keep the boundary at its prior depth in T (it rides along) and output a per-column confidence or contrast map. Do not force thickness to 0, so that "lost contrast" and "pinched out" stay distinct outputs.
- Beware sequential or top-down chains (Chiu windows; He's ReLU from the ILM). An error in one boundary spreads. Prefer joint solves.

**(c) Costs**
- Use signed, polarity-specific edge terms per boundary; band order fixes the expected contrast sign, so use it.
- Add region terms with per-volume re-estimated class means (Garvin). This is robust to drift in intensity across the slab.
- Combine echoes as channels, ideally through a learned per-voxel boundary probability (RF or CNN). Position relative to a landmark (here, depth T) is a feature, as Lang used distance to the fovea.
- Per-boundary weights between edge and region terms differ (Garvin's α differed by surface).

**(d) Validation**
- Report signed and unsigned boundary error per surface in µm and in voxels.
- Get at least two raters on a subset and report the inter-rater variability beside the algorithm.
- Keep the data that builds the priors (thickness and depth models) separate from the test data. Carass flags its own failure here.
- Retrain baselines on the same data.
- Check the ground-truth file conversions. The Carass erratum is a warning.

### Gaps
- None of the papers found reports validation that targets pinch-out locations specifically (for example, error as a function of local layer thickness, or detection accuracy for where a layer ends). This would need a new metric. One option is to score each column as band present or absent, against expert labels.
- No multi-echo or multi-contrast MRI layered-surface paper was checked in this session. The multi-channel cost points above are inferences from single-channel OCT practice.

### Constraint ledger per method: up front (in the representation or the energy) vs. fix-up (pre- or post-processing)

Purpose (scope refinement from the user): avoid post-hoc patching. Each table lists:
- **Up front (U):** the constraint is in the representation or the energy.
- **Pre-step (P):** a preprocessing or staging step.
- **Post-step (F):** a fix-up after the optimization.
- **n/a:** the method does not have the constraint.

For each P or F row, the table gives the failure the step repairs and the up-front constraint that would make the step unnecessary. Facts come from the sources cited above. Text in *italics* is my inference (the "repairs" and "would prevent" columns are usually inference unless a source is given).

**Li, Wu, Chen, Sonka 2006 (TPAMI) — [PMC2646122](https://pmc.ncbi.nlm.nih.gov/articles/PMC2646122)**

| Constraint | How | U/P/F | Failure repaired / up-front alternative |
|---|---|---|---|
| Ordering of surfaces | intersurface arcs (non-crossing variant) | U | — |
| Topology | height field: one voxel per column | U (by representation) | Surfaces that fold back are impossible. Cylindrical surfaces need an unfolding pre-step (P). |
| Thickness min/max | δl, δu, both ≥ 0, so δl = 0 (coincident surfaces) is in the notation | U | Spatially varying values are possible (Sec. 8.1), but how to set them automatically is called "open". |
| Smoothness | hard Δx, Δy | U | Not discontinuity-preserving. That is acceptable when depth jumps are forbidden. |
| Shape prior | none beyond Δ/δ | n/a | Song 2013 adds it later, in the energy. |
| Infeasible nodes | "deficient" nodes removed | P (graph pruning, not a fix to the result) | — |
| Feasibility of constraints | assumed | n/a | Contradictory constraints give no solution. *Up-front alternative: soft constraints (Song).* |

**Garvin et al. 2009 (TMI) — [PMC2911837](https://pmc.ncbi.nlm.nih.gov/articles/PMC2911837)**

| Constraint | How | U/P/F | Failure repaired / up-front alternative |
|---|---|---|---|
| Ordering, topology | Li-type graph | U | — |
| Thickness min/max, incl. 0 | per column, learned: mean ± 2.6 SD, minimum truncated at 0 | U (learned prior in the graph) | — |
| Smoothness | per column pair, separate max increase and max decrease, learned (± 2.6σ) | U | — |
| Flattening | thin-plate spline fit to surface 7, columns shifted | P | *Repairs: whole-retina curvature and tilt that would need large Δ or wide columns. Up-front alternative: a depth coordinate, i.e. columns along a fitted depth field (the T field).* |
| Two-stage order (surfaces 1, 6, 7 first, then 2–5) | sequential joint solves | P | *Repairs: region costs need region means, which come from the outer surfaces and the learned thickness bounds. Up-front alternative: estimate region parameters jointly (EM-style) inside the energy.* |
| Speckle | anisotropic diffusion | P | *Repairs: noisy edge costs. Up-front alternative: learned or region costs that are robust to noise.* |
| Edge/region weight α | tuned per surface | P (hyper-parameter) | — |

**Song et al. 2013 (TMI) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/pmid/23193309)**

| Constraint | How | U/P/F | Notes |
|---|---|---|---|
| Ordering | graph | U | — |
| Shape prior | pairwise penalty on local shape change relative to a model | U (in the energy, convex, globally optimal) | Replaces hard Δ with a soft prior. |
| Context (thickness) prior | pairwise penalty on local change in surface distance relative to a model | U | This is the up-front form of "thickness changes slowly". Error went from 6.30 to 5.14 µm. |

**LOGISMOS, Yin et al. 2010 (TMI) — [PMC3131162](https://pmc.ncbi.nlm.nih.gov/articles/PMC3131162)**

| Constraint | How | U/P/F | Notes |
|---|---|---|---|
| Surfaces within one object, and interactions between objects | encoded in one n-D graph | U | All are solved together to the global optimum. The pre-steps (initial object meshes and column directions) were not checked in this session. |

**Chiu et al. 2010 (Opt. Express) — [PMC3408910](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408910)**

| Constraint | How | U/P/F | Failure repaired / up-front alternative |
|---|---|---|---|
| One boundary per column | shortest path across the B-scan | U (one surface at a time) | — |
| Edge polarity | separate dark-to-light and light-to-dark weights | U (in the cost) | — |
| Ordering between boundaries | each new search is limited to a window bounded by boundaries already found (e.g. 20 px above IS–OS) | F (sequential windows) | *Repairs: a path jumping to the wrong, stronger edge. Up-front alternative: a joint multi-surface graph with ordering arcs (Li).* |
| Thickness ≥ 0 / merging at the fovea | fovea detected (mean inner-layer thickness < 5 px), then search windows extended up to the vitreous–NFL boundary | F (special-case detector) | Repairs: the windows exclude merged layers at the fovea. *Up-front alternative: δl = 0 everywhere, or non-negative thickness, so no detector is needed.* |
| Free endpoints | added low-weight columns, removed afterward | P/F | Repairs: the start and end of the path would otherwise be fixed. |
| Smoothness between B-scans | none (2D) | n/a | *Up-front alternative: a 3D graph.* |

**Lang et al. 2013, RF + graph (BOE) — [PMC3704094](https://pmc.ncbi.nlm.nih.gov/articles/PMC3704094)**

| Constraint | How | U/P/F | Notes |
|---|---|---|---|
| Learned boundary evidence | random forest, 27 features, incl. position relative to the fovea | U (in the cost) | — |
| Ordering and smoothness | applied in the graph stage that follows the classifier | U in the graph stage, but F relative to the classifier | The RF output does not guarantee order, so the graph repairs it. The exact graph constraints are **[UNVERIFIED]**. |

**Carass et al. 2014, MGDM in flat space (BOE) — [PMC3986003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3986003)**

| Constraint | How | U/P/F | Failure repaired / up-front alternative |
|---|---|---|---|
| Ordering, no gaps or overlaps, topology | MGDM with digital homeomorphism | U | — |
| Sub-voxel thickness, no 1-voxel floor | flat space: proportional depth between BrM and ILM, then isosurface extraction | P (domain transform) + U (sub-voxel level sets) | Repairs: ordering on the voxel grid forces each layer to be ≥ 1 voxel thick. This is the closest analogue to the T field. |
| Prerequisite boundaries | ILM, BrM and the fovea estimated first | P | *Errors in these steps propagate. Up-front alternative: estimate the depth field and the bands jointly.* |
| Capture range | GVF of the RF boundary maps | P (force design) | Repairs: a narrow capture range of the boundary probabilities. |
| Smoothness | curvature term | U (in the evolution energy) | — |
| A layer vanishing entirely | not analyzed | n/a | *Topology preservation probably conflicts with a layer that disappears over a region.* |

**Bogovic, Prince, Bazin 2013, MGDM (CVIU) — [PMC3539759](https://pmc.ncbi.nlm.nih.gov/articles/PMC3539759); [IACL](https://iacl.ece.jhu.edu/index.php?title=MGDM)**

| Constraint | How | U/P/F | Notes |
|---|---|---|---|
| No overlaps or gaps; topology | level-set decomposition with digital homeomorphism | U | — |
| Allowed adjacencies | from a topology template used to initialize | P (the template sets them) | Only objects adjacent in the template may touch. *This makes a pinch-out impossible: band A cannot touch band C when B vanishes between them.* |

**He et al. 2018, topology-guaranteed CNN — [arXiv](https://arxiv.org/pdf/1803.05120)**

| Constraint | How | U/P/F | Failure repaired / up-front alternative |
|---|---|---|---|
| Ordering, thickness ≥ 0 (zero allowed) | second network regresses thicknesses, final ReLU, cumulative boundaries | U (in the output layer of the 2nd network) | The 2nd network is a structured repair of the 1st, which is a pixel-wise labeling with no order guarantee. *Up-front alternative: regress thicknesses from the start (done in He 2021).* |

**He et al. 2021, structured layer surface regression (MedIA) — [PMC7855873](https://pmc.ncbi.nlm.nih.gov/articles/PMC7855873)**

| Constraint | How | U/P/F | Failure repaired / up-front alternative |
|---|---|---|---|
| One position per column, sub-pixel | soft-argmax over the row distribution in each column | U | — |
| Ordering, non-strict (zero thickness allowed, by inference) | ReLU update relative to the surface above, starting from the ILM; in the network, at training and at test | U | Failure: errors propagate top-down. *Up-front alternative: a joint or symmetric ordering projection.* |
| Lesions without topology | separate pixel-wise branch | U (separate model) | — |
| Smoothness between B-scans | none | n/a | Stated as the advantage graph methods keep. |
| Smoothness within a B-scan | none explicit (not checked in this session) | — | The surfaces stay smooth through edema (Fig. 9). |

**Zeng et al. 1999 (TMI); MacDonald et al. 2000 (NeuroImage) — [MICCAI 1998 version](https://rd.springer.com/content/pdf/10.1007/BFb0056237.pdf); [MacDonald PDF](https://pages.ucsd.edu/~msereno/569/readings/mcdonald.pdf)**

| Constraint | How | U/P/F | Notes |
|---|---|---|---|
| Zeng: coupling of two surfaces, thickness | coupling in the joint level-set propagation, motivated by nearly constant thickness | U | The exact range form is **[UNVERIFIED]**. The equations are empirical, per Goldenberg 2002. |
| MacDonald: no self-intersection; proximity between surfaces | explicit terms in the mesh deformation | U | — |

#### Summary pattern (inference)
- **Usual fix-ups:**
  - ordering by sequential search windows (Chiu)
  - special-case detectors at the fovea (Chiu)
  - flattening or domain transforms (Garvin, Carass)
  - estimating outer boundaries first (Garvin, Carass)
  - a structured stage that repairs an unstructured classifier (Lang, He 2018)
- **The same jobs, done up front:**
  - ordering arcs or ReLU ordering in the model
  - δl = 0 or non-negative thickness, so merging is allowed everywhere
  - learned per-column thickness bounds (Garvin) or soft context priors (Song)
  - a depth coordinate as the domain of the representation
- **For the user's design:** making T the column domain, with boundaries as non-negative thickness increments, a soft lateral prior on thickness, and a joint solve, would remove all five fix-up types.
- **What remains:** low-contrast columns. In every method, a prior carries the boundary through those columns, and none gives an explicit up-front model of "present but invisible".
