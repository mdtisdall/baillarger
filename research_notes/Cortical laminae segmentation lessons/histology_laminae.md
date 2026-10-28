# Histology-based segmentation and surfacing of cortical laminae (BigBrain and related): lessons for 160 µm ex vivo MRI

Scope note: the coordinator asked, per method, for the constraints built into the representation or energy up front and the constraints added later as fix-ups. For each fix-up, this file records the failure it repairs and the up-front constraint that would have prevented it. Those tables are in each section. All "Inferences" are the researcher's own reasoning, not claims from the sources.

Verification status of citations (title, authors, year, venue, DOI):
- VERIFIED (DOI checked against publisher page, Europe PMC or PubMed):
  - Wagstyl 2020 PLOS Biol, 10.1371/journal.pbio.3000678
  - Wagstyl 2018 Cereb Cortex, 10.1093/cercor/bhy074
  - Paquola 2019 PLOS Biol, 10.1371/journal.pbio.3000284
  - Amunts 2013 Science, 10.1126/science.1235381
  - Schleicher 1999 NeuroImage, 10.1006/nimg.1998.0385
  - Annese 2004 NeuroImage, 10.1016/j.neuroimage.2003.08.024
  - Nieuwenhuys 2013 Brain Struct Funct, 10.1007/s00429-012-0460-z. Metadata and abstract only; the paper is paywalled.
  - Nieuwenhuys & Broere 2017 Brain Struct Funct, 10.1007/s00429-016-1228-7
  - Aggarwal 2015 NeuroImage, 10.1016/j.neuroimage.2014.10.053
  - Wang 2024 Neuroinformatics, 10.1007/s12021-024-09688-0
  - Kutten 2024 MethodsX, 10.1016/j.mex.2024.102674
  - Waehnert 2014 NeuroImage 93:210, PMID 23603284. PubMed citation match only; the text was not read.
- UNVERIFIED: these were not fetched in this round and are listed only as leads.
  - Paquola et al. 2021 eLife "BigBrainWarp" (believed to be 10.7554/eLife.70119)
  - Schiffer et al. 2021 contrastive learning of cytoarchitecture in BigBrain/Jülich sections (believed to be an ISBI/MICCAI-type venue)
  - Spitzer et al. 2018 MICCAI, siamese networks for cytoarchitectonic areas
  - Lewis et al. BigBrain tissue classification and surface extraction (an OHBM abstract, not a journal paper, as far as I know)
  - "Kiwonga": I found no source that matches this name.
  - Štajduhar et al. 2023: not found in this round.
  - Braitenberg 1962 J Comp Neurol 118:141–156 "A note on myeloarchitectonics": the citation was seen only in a reference-list snippet.

---

## 1. Wagstyl et al. 2020 (PLOS Biology): 1D CNN on BigBrain profiles, six layer surfaces

### Takeaway
The 3D atlas is a per-profile (per-column) 1D classifier with no built-in ordering, topology or thickness prior. Every inter-surface coherence property arrives afterwards:
- argmax transitions
- interpolation over failed vertices
- 0.5 mm surface smoothing and 20 iterations of mesh smoothing
- column-vector smoothing to stop profiles crossing

Accuracy is 83 ± 2% per point. Layer IV is the least confident class, and accuracy falls from 85% at 20 µm to 76% at 100 µm and 60% at 1000 µm.

### Cited Findings
**Citation:** Wagstyl K, Larocque S, Cucurull G, Lepage C, Cohen JP, Bludau S, Palomero-Gallagher N, Lewis LB, Funck T, Spitzer H, Dickscheid T, Fletcher PC, Romero A, Zilles K, Amunts K, Bengio Y, Evans AC. "BigBrain 3D atlas of cortical layers: Cortical and laminar thickness gradients diverge in sensory and motor cortices." *PLoS Biol* 18(4):e3000678, 2020. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)

**Training labels**
- Manual six-layer segmentations of 51 cortical regions on 13 BigBrain sections, rescanned at 5 µm in-plane. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)
- The regions were chosen for:
  - varied cytoarchitecture
  - rostrocaudal spread
  - gyri and sulci
  - tangentially (not obliquely) sectioned areas
- V1 and motor cortex were deliberately included.
- Expert anatomists (SB, NPG, KZ) verified the labels.
- The 2D labels were converted to more than 500,000 1D training profiles. Labels were extended 0.25–0.75 mm above pial and below L6, so that background and WM are classes.

**Class frequency in the training profiles**

| Class | Share |
|---|---|
| CSF/background | 14.6% |
| L1 | 7.5% |
| L2 | 5.6% |
| L3 | 20.8% |
| L4 | 5.5% |
| L5 | 14.8% |
| L6 | 17.8% |
| WM | 13.4% |

The loss is median-frequency-weighted cross-entropy, which up-weights the thin L2 and L4. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)

**Profiles (the column geometry)**
- Pial and white surfaces came from a 200 µm tissue classification, with vertex-matched meshes. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)
- A midsurface was upsampled from 163,842 to 655,362 vertices. It was placed by curvature weighting: closer to the pial surface in sulci and closer to the white surface in gyri.
- The pial→white vectors were smoothed across the midsurface "to avoid crossing profiles, which can result in mesh self-intersections". The Methods give 3 mm FWHM. The S1 Fig caption says around 2 mm FWHM "drastically decreases" self-intersections. The paper does not reconcile the two values. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)
- Each profile has 200 equidistant samples and extends 0.5 mm beyond the surfaces.
- Intensity was regressed on the posterior–anterior coordinate to remove a staining gradient.
- There are two input channels: raw intensity, and intensity after anisotropic diffusion (max effective FWHM 0.163 mm).
- Curvature is not a network input.
- The volume was processed as 125 overlapping blocks.

**Network**
- A 1D fully convolutional segmentation net: stacked BN–ReLU–conv blocks, 6 layers, filter size 49 (out of 200 samples, so about a quarter of the profile). — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)
- Output: a 1×1 conv to 8 classes.
- Training: LR 0.0005, weight decay 0.001, chosen by 50 random hyperparameter runs, with early stopping after 50 epochs without improvement.

**Accuracy**
- 10-fold CV gives 83% ± 2% per-point accuracy before post-processing. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)
- Resolution study: 85% at 20 µm, 76% at 100 µm, 60% at 1000 µm. The 40/200/400 µm values are only in S3 Fig, which was not inspected.
- Boundary depth deviation (S3 Fig): pial/L1 and L1/L2 are the smallest, then L2/3. L3/4 and L6/WM are larger.
- Confidence is defined as the difference between the top two class probabilities. Layer IV is "least confident overall"; suprapial and WM confidence is high (S2 Fig). — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)

**Surfaces from class transitions**
- "Transitions between predicted layers were located for each profile", and the transition coordinates became mesh vertices. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)
- Where "the network failed (less than 1%)", vertex locations were interpolated from neighbours. The paper does not define failure; for example, it does not say whether a missing or out-of-order class counts.
- Surface indices were smoothed at 0.5 mm FWHM, then 20 iterations of shrinkage-free mesh smoothing were applied. The stated reason is high-frequency artefacts from section misalignment.
- No layer-order constraint is described, and no inter-surface self-intersection check is described.

**V1, agranular cortex and the WM boundary**
- L4 is thick, with sublayers, in V1 and thin in V2, with a sharp change at the border. The authors suggest the network "may have internally learned" area-specific rules; they do not test this. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)
- The 8 classes have no V1 sublayers (IVa/b/c). The paper does not say whether the network outputs a thin L4 in agranular motor cortex.
- The network's L6/WM boundary sits near the VIa/VIb border, which is the maximum intensity gradient. This is superficial to the original BigBrain white surface (S4 Fig).

**Exclusions and limitations**
- Masked out: the medial wall, temporal-lobe saw cuts, and allocortex (for laminar comparisons). — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)
- Stated limitations:
  - single specimen (65-year-old male)
  - tears and shears
  - L4 and L5/L6 boundaries are hard to place
  - L2 agrees poorly with von Economo (r = 0.11/0.06)
  - the insula is hard at low resolution

**Validation against von Economo**
- Total thickness: r = 0.86. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)
- Per layer (left): L1 0.50, L2 0.11, L3 0.74, L4 0.80, L5 0.66, L6 0.66.
- Total thickness ranges from 1.8 mm (calcarine) to 4.5 mm (precentral).
- Shrinkage: 1.931 volumetric, 1.245 linear (anisotropic 1.15/1.22/1.43).

**Up-front vs fix-up (Wagstyl 2020)**

| Constraint | Up front or fix-up | Failure it repairs | Up-front alternative that would prevent it |
|---|---|---|---|
| Columns do not cross | Fix-up: 2–3 mm smoothing of the pial→white vectors | Profile crossing produces layer-mesh self-intersections | Define depth as a scalar field T solved over the whole ribbon (Laplace/equivolume), so that columns are its streamlines, which cannot cross |
| Curvature-aware column centre | Partly up front: curvature-weighted midsurface | Equidistant sampling biases sulcal and gyral depths | Equivolume depth (Bok), built into T |
| Layer order (CSF<L1<…<L6<WM) | Not enforced; argmax per point | Out-of-order or missing transitions, presumably part of the <1% "failures" | Monotone depth parameterisation: label = f(T) with ordered thresholds, or a CRF/HMM with only forward transitions |
| One transition per boundary per column | Not enforced; transitions read off argmax | Spurious multiple transitions (speckle) | Same as above: a single threshold crossing on a monotone T |
| Lateral smoothness of layer depth | Fix-up: 0.5 mm FWHM smoothing of indices plus 20 mesh-smoothing iterations | Section-misalignment jitter and per-column noise | A smoothness term on T (gradient/curvature regularizer on the isosurfaces) inside the fit |
| Gap filling | Fix-up: interpolation from neighbours for <1% of vertices | Network failure on a column | A regularized field fit, which fills such gaps by itself |
| Non-intersection between layer surfaces | Not checked | Surfaces could cross after independent smoothing | Isosurfaces of one scalar field cannot intersect |
| Thickness limits / per-area template | None (implicit in the learned filters) | V1 and agranular cortex handled only implicitly | An allowed band-pattern library per area type, with a lateral switching variable |

### Inferences
- Wagstyl 2020 is, in effect, "classify each column independently, then glue". Every global property the user wants (order, no crossing, no intersection, lateral smoothness) was added as smoothing afterwards. A single depth field T delivers all of them by construction. This is the strongest argument from this literature for the user's design.
- The 49-sample filter on a 200-sample profile means each prediction sees about 25% of cortical depth. The CNN works by learning band-pattern context, not local edges. An MRI analogue at 160 µm has about 12–25 voxels across 2–4 mm, so it can carry about 3–6 independent band features per column. Only bands of 2 or more voxels (≥320 µm) are reliable, which matches the user's premise.
- The accuracy-vs-resolution curve (85% → 76% from 20 → 100 µm) is for cell-body (Nissl/Merker) contrast, where boundaries are texture edges. Myelin bands are smoother and broader, so MRI myelin bands may degrade less with resolution. This is not tested in the source.
- The WM boundary shift to VIa/VIb shows that "the white surface" depends on contrast. The outer limit of T should come from the band data, not be fixed to a separately segmented WM surface.

### Gaps
- No per-class accuracy, and no quantitative error near sulcal fundi or gyral crowns. S3 Fig values at 40/200/400 µm were not inspected.
- No explicit statement of what "failure" means, and no statement of how agranular cortex is labelled.
- The discrepancy between 2 mm and 3 mm FWHM is unresolved in the paper.

---

## 2. Wagstyl et al. 2018 (Cerebral Cortex): iterative peak-snapping + smoothing for the L1/L2 and L4 surfaces

### Takeaway
This is a hand-built "snap to a feature, smooth, re-snap" loop on 100-point profiles from 40 µm data, run in equivolumetric depth. It got L4 to a 72 µm unsigned error (L1/2: 63 µm), but only where L4 exists: agranular and dysgranular samples were excluded. In equivolumetric depth, L4 depth hardly varies with curvature (β = −1.07), whereas in raw depth curvature dominates (β = −15.5). This is the empirical basis for "depth fields should be equivolume".

### Cited Findings
**Citation:** Wagstyl K, Lepage C, Bludau S, Zilles K, Fletcher PC, Amunts K, Evans AC. "Mapping Cortical Laminar Structure in the 3D BigBrain." *Cereb Cortex* 28(7):2551–2562, 2018. — [OUP](https://academic.oup.com/cercor/article/28/7/2551/4990976)

**Profiles**
- 100 equidistant points between linked pial and white vertices, sampled on a 40 µm volume after anisotropic heat-equation smoothing. — [OUP](https://academic.oup.com/cercor/article/28/7/2551/4990976)
- The smoothing maximum is 0.163 mm FWHM, chosen "so most profiles had 3–5 peaks". In other words, the smoothing scale was tuned to the number of features expected per column.

**L1/L2 surface**
- Initialised from the 200 µm tissue classification. If nothing suitable was found, it was placed 200 µm below the pial surface. — [OUP](https://academic.oup.com/cercor/article/28/7/2551/4990976)
- Loop:
  1. Snap each vertex to the nearest maximum intensity gradient (a first-derivative peak).
  2. Smooth the matched values over the surface with 3 nearest-neighbour iterations, then re-snap.
  3. Where there is no match, or the match is deeper than 200 µm, use the neighbours' average depth.

**L4 surface**
- Initialised at the first major intensity peak after an upward inflection; the paper says this inflection marks L2/L3 more reliably than an L2 peak does. — [OUP](https://academic.oup.com/cercor/article/28/7/2551/4990976)
- Loop:
  1. Taubin shrinkage-free mesh smoothing.
  2. Smoothing of equivolumetric depth with a 10 mm FWHM Gaussian.
  3. Snap each vertex to the nearest intensity peak.
- The loop repeats until fewer than 100 of 163,842 vertices change.
- The search window is not stated. No explicit missing-peak rule for L4 was found in the text.

**Validation**
- Manual comparison on 7 of 7,404 sections at 5 µm, 6 regions each: 42 samples. — [OUP](https://academic.oup.com/cercor/article/28/7/2551/4990976)
- Removed: dysgranular or agranular samples where L4 was not visible (the text gives 2 + 4), and 1 sample for tearing, leaving 37.

| Surface | Mean signed error | SD | Range | Unsigned error | Test vs zero |
|---|---|---|---|---|---|
| L1/2 | −13 µm | 40 | −120 to 70 µm | 63 µm | p = 0.08 |
| L4 | −11 µm | 68 | −161 to +140 µm | 72 µm | p = 0.31 |

- Depth correlation with manual labels: mean r = 0.72 (0.80 for section 2807).
- The source's arithmetic is inconsistent: 42 − 5 − 1 = 36, not 37, and the agranular/dysgranular counts are 2 + 4 = 6. Check the original before quoting.

**Curvature decomposition**
- Mean curvature on the midsurface, smoothed 3 mm. Excluded: vertices with thickness under 0.5 mm, the medial wall, and the hippocampus. — [OUP](https://academic.oup.com/cercor/article/28/7/2551/4990976)
- Raw L4 depth vs curvature:
  - linear β = −15.5, F(1, 302273) = 4.164×10⁵
  - quadratic and cubic fits are also significant
  - BIC favours cubic: 1,881,609 vs 1,891,660 (quadratic) vs 1,893,302 (linear)
- Equivolumetric L4 depth: β = −1.07, F = 2706, "almost no change with curvature".
- Area residuals are described qualitatively, for example a deeper L4 in V1 and in parts of medial prefrontal cortex. No R² or formal variance split was found in the extracted text.

**Limitations**
- The authors state that interareal differences prevent extending the snapping method to all six layers, and suggest more flexible methods such as deep learning. This motivated the 2020 work. — [OUP](https://academic.oup.com/cercor/article/28/7/2551/4990976)
- Oblique sectioning: about 69% of vertices are more than 18° oblique, giving roughly 5% thickness overestimation in 2D measures.

**Up-front vs fix-up (Wagstyl 2018)**

| Constraint | Up front or fix-up | Failure it repairs | Up-front alternative |
|---|---|---|---|
| Equivolume depth | Up front (smoothing is done in equivolumetric depth) | Curvature-driven depth shifts | Same; keep it |
| Lateral smoothness | Built into the loop, but as alternation (10 mm FWHM depth smoothing + Taubin), not as one energy | Noisy per-column peak picks | A joint energy: data term (distance to feature) + smoothness term on the surface or T. The alternation is a crude block-coordinate descent of that energy |
| Feature definition | Up front: peak (L4) or max gradient (L1/2) | — | For MRI, a band centre is a peak and a band edge is a gradient extremum. Choose per band |
| Missing feature | Fix-up (L1/2): neighbour-average depth, 200 µm cap. Fix-up (L4): exclude agranular areas | Snapping to the wrong peak, or no peak at all, in agranular cortex | A model in which band amplitude can go to zero laterally while depth stays continuous. The user's "areal borders are lateral changes in band pattern, never depth jumps" is exactly this |
| Number of features per column | Implicit: smoothing tuned to give 3–5 peaks | Over- or under-segmentation | Choose the smoothing or regularization scale from the minimum band thickness (2–3 voxels) |
| Order between L1/2 and L4 | Implicit (separate searches from separate initialisations) | Could cross | One T field |

### Inferences
- The 10 mm FWHM depth smoothing is very large compared with areal sizes. It works only because equivolume depth of L4 is nearly constant across curvature. It would blur real areal depth changes (for example at the V1 border), which the 72 µm average may hide. For MRI, put the smoothness on T in equivolume coordinates, but make it weak enough (or edge-aware, or anisotropic) to let band depth change over about 1–3 mm across borders.
- The finding that the L2/3 inflection is more reliable than the L2 peak transfers directly: thin bands are better anchored by edges than by their own peak.
- The quadratic/cubic curvature residual in raw depth is what equivolume removes. For the user, the residual area effect after equivolume is the signal, and the curvature effect is a nuisance that T's construction must absorb.

### Gaps
- Search window, iteration counts and convergence time were not reported in the extracted text.
- No R² for curvature vs area.
- No error stratified by sulcus or gyrus.

---

## 3. Paquola et al. 2019 (PLOS Biology): microstructure profile covariance (MPC), BigBrain to MRI

### Takeaway
This is not a segmentation. Equivolumetric intensity profiles are compared between parcels by partial correlation, which removes the mean profile. The approach transferred to MRI (T1w/T2w at 0.7 mm) only after dropping the superficial and deepest surfaces. It is robust to histology resolution from 100 to 400 µm (r > 0.81).

### Cited Findings
**Citation:** Paquola C, Vos De Wael R, Wagstyl K, Bethlehem RAI, Hong S-J, Seidlitz J, Bullmore ET, Evans AC, Misic B, Margulies DS, Smallwood J, Bernhardt BC. "Microstructural and functional gradients are increasingly dissociated in transmodal cortices." *PLoS Biol* 17(5):e3000284, 2019. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000284)

**BigBrain profiles**
- BigBrain at 100 µm; 200, 300 and 400 µm were also tested. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000284)
- 10–100 equivolumetric surfaces were tested, and 18 were chosen for matrix stability.
- The top 10% (approximating L1) and the deepest surface were removed.
- Intensity was corrected for the y-coordinate (r = −0.68 with intensity).
- Parcels: 1,012 nodes. Vertices more than 3 scaled MADs from the parcel median were excluded.
- Partial correlations control for the mean whole-cortex profile.

**MRI**
- HCP T1w/T2w (0.7 mm), 14 surfaces. The robustness section uses 23 surfaces, so the paper is internally inconsistent here. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000284)
- Agreement with the histology gradient G1: r = 0.63 at group level, 0.41–0.62 for individuals.
- qT1: r = 0.81. MT: r = 0.79.
- No profile alignment or warping beyond equivolume and the y-correction.

**Up-front vs fix-up (Paquola 2019)**

| Constraint | Up front or fix-up | Failure it repairs | Up-front alternative |
|---|---|---|---|
| Equivolume sampling | Up front | Curvature bias | — |
| Drop superficial/deep surfaces | Fix-up | Partial volume with CSF and WM | Model partial-volume boundary layers explicitly in the profile model |
| Outlier removal (3 MAD) | Fix-up | Tears, misregistration | Robust data term |
| Mean-profile removal | Analysis choice | Shared profile shape dominates the correlations | — |

### Inferences
- Even with 1.5 cm² parcels and mean-profile removal, MRI–histology profile agreement is moderate (r of about 0.5–0.6). Equivolume depth alone does not align bands well enough for column-wise comparisons, so residual band-depth variation between areas remains. This is consistent with Wagstyl 2018's residual area effects.

### Gaps
- No quantitative evaluation of laminar position error in MRI.

---

## 4. Schleicher et al. observer-independent border detection (GLI profiles, feature vectors, Mahalanobis distance)

### Takeaway
Areal borders are defined as positions along the cortex where the shape of blocks of neighbouring depth-normalised profiles changes significantly. This is the opposite assumption from per-column laminar segmentation, and it matches the user's "areal borders are lateral changes in band pattern". Profiles are compared through shape features, not aligned point by point.

### Cited Findings
**Citation:** Schleicher A, Amunts K, Geyer S, Morosan P, Zilles K. "Observer-independent method for microstructural parcellation of cerebral cortex: A quantitative approach to cytoarchitectonics." *NeuroImage* 9(1):165–177, 1999. — [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:9918738&format=json&resultType=core) (DOI 10.1006/nimg.1998.0385)

**Method**
- Laminar profiles run from the pial surface to the WM boundary. — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:9918738&format=json&resultType=core)
- Profile sources: GLI images for Nissl; densitometry for myelin, IHC and receptor autoradiography.
- Neighbouring profile shapes are compared by distances between feature vectors.
- Mahalanobis distance beat Euclidean distance in sensitivity and spatial resolution.
- On human neocortex, the method confirmed visually defined borders and found new ones.

**Features**
- Ten shape descriptors based on central moments are extracted per profile, and the Mahalanobis distance function defines borders. — [PMC figure legend (J Comp Neurol paper)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3429851/figure/F3)
- A 1998 Acta Stereologica precursor reports the Mahalanobis-over-Euclidean result on occipital cortex. — [Acta Stereologica](https://popups.uliege.be/0351-580x/index.php?id=2499)

**Unverified (from prior knowledge; full text not read)**
- The ten features are mean GLI, x-coordinate of the centre of gravity, SD, skewness and kurtosis of the profile, and the same five for the profile's first derivative.
- Profiles are resampled to 100 points of relative depth.
- A sliding pair of adjacent blocks of b profiles (b roughly 8–24) is compared, with Hotelling's T² and permutation testing.
- A border is accepted when it recurs across multiple block sizes and adjacent sections.

**Up-front vs fix-up (Schleicher)**

| Constraint | Up front or fix-up | Failure it repairs | Up-front alternative |
|---|---|---|---|
| Depth normalisation (0–100%) | Up front | Thickness variation | Equivolume T |
| Moment-based shape features (no point alignment) | Up front | Small depth shifts between columns | — (alignment-invariant by design) |
| Block averaging over b profiles | Up front | Single-profile noise | — |
| Border significance and reproducibility (multiple b, adjacent sections) | Post hoc acceptance test | False borders from artefacts or oblique cuts | A lateral switching prior with a cost per border (piecewise-constant area label) |

### Inferences
- Moment features are tolerant of band-depth jitter but blur band identity. For a band-tracking T, the analogous "areal border detector" is a lateral change in the amplitude or presence of bands along T isosurfaces, not a change in T.
- 2D section-based borders are sensitive to oblique cuts, which motivated 3D profiles in BigBrain. 3D MRI avoids this by construction.

### Gaps
- Exact parameters were not verified from the full text.

---

## 5. Annese et al. 2004 (NeuroImage): myelo-architectonic profile classification with PDE columns, wavelets and templates

### Takeaway
This is the closest histology precedent for "myelin-band profiles along a field-defined column":
- Profiles follow a PDE solution that models the intermediate laminae, rather than straight normals.
- Profiles are classified by wavelet-selected components.
- Profiles in each class are normalised and averaged into per-area myeloarchitectonic templates.

### Cited Findings
**Citation:** Annese J, Pitiot A, Dinov ID, Toga AW. "A myelo-architectonic method for the structural classification of cortical areas." *NeuroImage* 21(1):15–26, 2004. — [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:14741638&format=json&resultType=core) (DOI 10.1016/j.neuroimage.2003.08.024)

- An automated method for myelin-stained sections. It measures the distribution of myelin density across cortical thickness using "a model of cortical curvature and intrinsic geometry". — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:14741638&format=json&resultType=core)
- Profile lines come from solving a PDE that models the intermediate cortical layers. My understanding is that this is a Laplace-type equation, but the PDE type was not verified from the full text.
- Classification uses significant wavelet components.
- Profiles in each class are normalised and averaged into area-specific myeloarchitectonic templates.

| Constraint | Up front or fix-up |
|---|---|
| Non-crossing, curvature-following columns | Up front (PDE) |
| Per-area templates | Built after classification, as class means |
| Profile alignment before averaging | "Normalised"; details unverified |

### Inferences
- PDE-defined columns give the non-crossing property for free, unlike Wagstyl 2020's vector smoothing. This supports deriving columns from a scalar field. The user's T is a stronger version, because T's isosurfaces are fitted to the bands, not only to the two boundaries.

### Gaps
- The full text was not read. Details of the normalisation, number of classes and validation are unknown.

---

## 6. Myeloarchitectonic band nomenclature (Vogt–Vogt school) and which bands are robust

### Takeaway
In myelin stains, the robust tangential bands are:
- the outer band of Baillarger (layer 4)
- the inner band of Baillarger (layer 5b)
- in V1, the stria of Gennari (IVb, about 250–300 µm thick)

Areas are typed by whether these bands are visible:
- astriate: no bands, for example area 4 (M1), where dense radial fibres dominate
- unistriate: one band visible, for example area 18 with only the inner band
- bistriate: both bands visible, for example areas 17 and 9

The Vogt–Vogt school mapped about 180–185 myeloarchitectonic areas. Hopf could distinguish about 50 areas from the Baillarger stripes alone.

### Cited Findings
**Nieuwenhuys 2013**
- Nieuwenhuys R. "The myeloarchitectonic studies on the human cerebral cortex of the Vogt–Vogt school, and their significance for the interpretation of functional neuroimaging data." *Brain Struct Funct* 218:303–352, 2013. DOI 10.1007/s00429-012-0460-z. Paywalled; the abstract only was read. — [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1007/s00429-012-0460-z&format=json&resultType=core)
- It estimates about 185 myeloarchitectonic areas and argues they should complement Brodmann's map for functional imaging. — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1007/s00429-012-0460-z&format=json&resultType=core)

**Nieuwenhuys et al. 2015 map and Nieuwenhuys & Broere 2017**
- The 2015 map has 180 areas: 64 frontal, 30 parietal, 6 insular, 17 occipital and 63 temporal, on Colin27/MNI305. — [Nieuwenhuys & Broere 2017, PMC5225164](https://doi.org/10.1007/s00429-016-1228-7)
- Hopf's overall myelin density decreases stepwise with distance from the primary areas. Heavily myelinated clusters lie beyond the primaries: orbitofrontal, intraparietal, posterolateral temporal and basal temporal. — [Nieuwenhuys & Broere 2017](https://doi.org/10.1007/s00429-016-1228-7)
- Hopf also analysed "the relation of the bands of Baillarger to each other and to neighboring layers", the band of Kaes-Bechterew, and the calibre of the radial bundles. These are the data a per-area band-pattern library would need. — [Nieuwenhuys & Broere 2017](https://doi.org/10.1007/s00429-016-1228-7)
- Hopf "could distinguish 50 areas on fresh cortical cross-sections based solely on the behavior of the light horizontal stripes (Baillarger stripes)". This is from a German-language source, read only as a search snippet. — [thehumanbrain.info Hopf PDF](https://www.thehumanbrain.info/thehumanbrain.info/brain/db_literature/HopfAdolf3.pdf)

**Aggarwal et al. 2015 (ex vivo MRI with histology)**
- Aggarwal M, Nauen DW, Troncoso JC, Mori S. "Probing region-specific microstructure of human cortical areas using high angular and spatial resolution diffusion MRI." *NeuroImage* 105:198–207, 2015. — [PMC4262592](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262592/)
- Acquisition: 11.7 T, 92 µm isotropic, 30 directions, b = 2000 s/mm², about 28 h.
- Silver-stain band appearance by area:

| Area | Type | Bands seen |
|---|---|---|
| 4 (M1) | Astriate | No conspicuous bands; a dense radial-fibre mid-zone (high FA) |
| 18 (V2) | Unistriate | Only the inner band (ibB) |
| 17 (V1) | Bistriate | Stria of Gennari (IVb) plus a less well-defined layer-V inner band |
| 9 | Bistriate | Faint inner band and diffuse outer band |

- In area 9, diffusion MRI shows the inner band as an FA minimum.
- Stria of Gennari about 250 µm thick; a tangential track-density band about 300 µm thick.
- The extracted text labelled the stria both IVb and IVa; IVb is the classical assignment.
- Area 3b low-FA band at 69.2 ± 4.1% depth.
- The T2w/b0 minimum did not exactly coincide with the FA minimum in area 17, which shows that different contrasts place the "same" band at slightly different depths.
- No laminar position error is reported. — [PMC4262592](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262592/)

**Background**
- The outer band of Baillarger is in layer IV; the inner band is in layer V; the outer band is enlarged in V1 as the line of Gennari. — [Wikipedia](https://en.wikipedia.org/wiki/Band_of_Baillarger)

### Inferences
- The band count per column changes laterally between 0 (astriate), 1 (unistriate) and 2 (bistriate). In V1 there are extra components (Gennari plus the inner band, with IVa/IVc structure), and Kaes-Bechterew sits in layer 3. A depth field T that follows the "band at ≥2–3 voxels" must allow:
  - bands whose amplitude goes to zero (astriate M1)
  - an outer band that thickens greatly (V1)
  - all of this without T jumping
- This is compatible with the user's no-fault assumption only if the band model is "the band at depth T = t_k has area-varying amplitude and width". It is not compatible with "the k-th visible peak is band k". Peak counting (Wagstyl 2018 style) would mislabel unistriate areas: is the single visible band the inner or the outer one?
- In an astriate area such as M1, T is not identifiable from tangential bands. Its isosurfaces must be carried in from neighbouring areas by the smoothness prior, or constrained by the boundaries alone (equivolume). Radial-fibre contrast (high FA or anisotropy) is a possible alternative signal there.
- At 160 µm, Gennari (about 250–300 µm) is only about 1.5–2 voxels. That is at or below the user's 2–3 voxel threshold, and other areas' Baillarger bands are generally thinner or fainter (no numbers found). Treat the outer and inner bands as "ride-along" features in many areas, not always as fitting targets.

### Gaps
- No source found, in this round, that defines unitostriate, propeastriate or conjunctostriate. The Nieuwenhuys 2013 full text, which likely defines them, was not accessible. Treat these terms as unverified.
- No quantitative band thickness or contrast across areas (outside V1) was found.
- Braitenberg 1962 and Hopf's photometric work were not retrieved.
- No source found on which band is most robust across all areas. Based on the Aggarwal table, the inner band is visible in more area types (18, 17, 9) than the outer band. This is an inference from four areas.

---

## 7. Other laminar segmentation work (BigBrain-derived and non-human)

### Takeaway
More recent 2D deep-learning methods report figures similar to Wagstyl 2020: 95th-percentile Hausdorff distance of about 92 µm in marmoset, and a Jaccard index of 85% vs 83% on BigBrain. Mixture-model methods that put equivolumetric depth into the feature space are the clearest example of building a depth prior into the classifier itself.

### Cited Findings
**Wang et al. 2024 (marmoset, 2D deep learning)**
- Wang J, Gong R, Heidari S, Rogers M, Tani T, Abe H, Ichinohe N, Woodward A, Delmas PJ. "A Deep Learning-based Pipeline for Segmenting the Cerebral Cortex Laminar Structure in Histology Images." *Neuroinformatics* 22(4):745–761, 2024. DOI 10.1007/s12021-024-09688-0. — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:38660047&format=json&resultType=core)
- Data: marmoset Nissl and myelin sections. Two stages: automated cortical labels, then a layer-segmentation network.
- Mean 95HD = 92.15 µm, vs 94.17 µm for Wagstyl 2020 adapted to 2D.
- BigBrain Jaccard index: 85.318% vs 83.000%.
- Cortical-label Euclidean error about 1274.75 ± 156.40 µm. The authors called this acceptable relative to half the cortical thickness (about 1800 µm).
- The abstract does not describe layer-order or topology constraints. — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:38660047&format=json&resultType=core)

**Kutten et al. 2024 (cat, 3D reconstructed histology)**
- Kutten KS, Trieu J, Dawson J, Hou L, Sollmann L, Kral A, Hubka P, Ratnanather JT. "Multidimensional Alternating Kernel Method for cortical layer segmentation in 3D reconstructed histology." *MethodsX* 12:102674, 2024. DOI 10.1016/j.mex.2024.102674. — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:39417954&format=json&resultType=core)
- Data: cat auditory cortex, four brains.
- Method: a multivariate Gaussian mixture whose feature space is staining intensity together with an equivolumetric depth map.
- No accuracy numbers in the abstract.

**Amunts et al. 2013 (BigBrain)**
- Amunts K, Lepage C, Borgeat L, Mohlberg H, Dickscheid T, Rousseau MÉ, Bludau S, Bazin PL, Lewis LB, Oros-Peusquens AM, Shah NJ, Lippert T, Zilles K, Evans AC. "BigBrain: an ultrahigh-resolution 3D human brain model." *Science* 340:1472–1475, 2013. DOI 10.1126/science.1235381. — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:23788795&format=json&resultType=core)
- About 20 µm resolution, reconstructed from 7,404 sections.

**Up-front vs fix-up (Kutten 2024)**

| Constraint | Up front or fix-up |
|---|---|
| Depth prior via an equivolume depth feature | Up front (in the likelihood) |
| Order | Implied softly by the depth feature, not enforced |

### Inferences
- Putting depth into the classifier's features (Kutten) is a soft ordering prior. The user's approach inverts it: depth (T) is the unknown, and band identity follows from T. This removes the need for an ordering constraint.

### Gaps
- Paquola 2021 BigBrainWarp, Schiffer 2021, Spitzer 2018, Lewis BigBrain surfaces, "Kiwonga" and Štajduhar 2023 were not retrieved or verified.
- No mouse or macaque tool (e.g. "LaminaR", "CytoNet") was verified.

---

## 8. Profile registration or warping across columns

### Takeaway
None of the BigBrain methods read here aligns or warps profiles point to point:
- Paquola uses equivolume depth plus mean-profile removal.
- Schleicher uses alignment-invariant moment features.
- Wagstyl 2020 lets a wide-receptive-field CNN absorb shifts.

Annese 2004 normalises profiles within a class before averaging into templates, but how was not verified. No dynamic-time-warping style profile alignment was found in this round.

### Cited Findings
- Equivolume depth removes most curvature-driven L4 depth variation (β from −15.5 to −1.07). — [Wagstyl 2018](https://academic.oup.com/cercor/article/28/7/2551/4990976)
- Paquola 2019 uses no profile alignment beyond equivolume and the y-coordinate correction. — [PLOS Biol](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000284)
- Annese 2004 classifies profiles, then "normalizes and averages" them per class into templates. — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:14741638&format=json&resultType=core)
- Aggarwal 2015 normalises to percent depth only. — [PMC4262592](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262592/)

### Inferences
- The user's T field is a profile-registration method. Fitting T so that bands lie on isosurfaces is equivalent to warping every column's depth axis onto a common laminar coordinate, with lateral smoothness tying neighbouring warps together.
- Two consequences follow:
  - The warp should be monotone, so that ordering is preserved. That is the 1D-DTW monotonicity constraint, which comes for free if T increases from pial to white.
  - The warp's flexibility, i.e. how far it may deviate from equivolume, needs a prior. Wagstyl 2018's 72 µm L4 error after equivolume plus smoothing, and its area-specific depth residuals, suggest the deviation from equivolume is of order 0.1–0.3 mm. That is about 1–2 voxels at 160 µm. This is an estimate, not a reported number.

### Gaps
- No histology paper found that explicitly does DTW or elastic alignment of laminar profiles. A targeted search, for example on "elastic registration laminar profiles" or "Bok equivolume histology validation", may find some.

---

## 9. Minimum resolvable band thickness and robust features

### Takeaway
For cytoarchitecture:
- 6-class per-point accuracy drops to 76% at 100 µm and 60% at 1 mm.
- L4 and the L5/L6 boundary are the weak points.
- L1/L2 and the pial and WM edges are the strongest.

For myelin, the Gennari stria is about 250–300 µm. Other Baillarger bands are thinner or fainter, or absent (astriate). At 160 µm, therefore, only V1's stria and perhaps the inner band in strongly bistriate areas meet a 2–3 voxel threshold.

### Cited Findings
- Accuracy: 85% at 20 µm, 76% at 100 µm, 60% at 1000 µm. L4 is the least confident; L3/4 and L6/WM depth deviations are the largest. — [Wagstyl 2020](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000678)
- Errors at 40 µm with smoothing: L1/2 unsigned 63 µm, L4 unsigned 72 µm (SD 68 µm). — [Wagstyl 2018](https://academic.oup.com/cercor/article/28/7/2551/4990976)
- Stria of Gennari about 250 µm; a tangential track-density band about 300 µm at 92 µm MRI. — [Aggarwal 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262592/)
- Profile covariance is robust from 100 to 400 µm (r > 0.81). The gross profile shape survives coarse resolution even when the layers are not segmentable. — [Paquola 2019](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000284)
- Paquola drops the top 10% of depth and the deepest surface for partial-volume reasons. — [Paquola 2019](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000284)

### Inferences
- A 72 µm L4 error at 40 µm sampling is about 2 samples, and about 0.45 voxel at 160 µm. Sub-voxel band-depth placement at 160 µm needs lateral pooling, i.e. smoothness over many columns, as Wagstyl does with 10 mm smoothing. That supports a strong lateral regularizer on T, but with area-border awareness.
- The most robust "bands" for T are the two boundaries (pial/L1 and the deep L6/WM gradient) plus the dominant mid-depth myelin band where present. The user should expect T in the middle of astriate cortex to be constrained mostly by the boundaries and equivolume.

### Gaps
- No direct histology measurement found of outer and inner Baillarger band thickness outside V1.
- No study found that degraded myelin-stained profiles to a given resolution and measured band detectability.
