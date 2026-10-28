# Conserved quantities across cortical curvature

Tags: [verified] = read in a fetched source this session; [unverified] = from search snippet, secondary summary, or prior knowledge, not checked against the primary text; [inference] = my own derivation or reasoning.

Notation for the inferences: s = cortical thickness along the column; A_w, A_p, A_mid = area of a column segment at white, pial, mid surface; H = mean curvature, K = Gaussian curvature of the white surface, sign so that H > 0 at a gyral crown (white surface convex toward pia).

## Bok / Waehnert equivolume: what is conserved and what is not

### Takeaway
Bok's principle conserves the *volume fraction of each layer within a column segment* (a relative quantity); it says nothing that makes total thickness or total segment volume per white-surface area constant. The Waehnert form needs both the white and the pial area of each segment, and assumes area changes linearly with depth; driving it from white-surface curvature alone (parallel-surface pial area) is outside its assumptions.

### Cited Findings
- [verified] Waehnert et al. 2014 (NeuroImage 93 pt 2: 210-220, doi 10.1016/j.neuroimage.2013.03.078): abstract states the model, from Bok 1929, assumes cortical segments preserve their volume while layer thickness changes to compensate folding; this is used to build a cortical coordinate system — [PubMed 23603284](https://pubmed.ncbi.nlm.nih.gov/23603284/)
- [verified] BrainVoyager's implementation of the same model: segment volume is "approximated using the equation of a volume frustum"; intermediate grids are placed so that "volume fractions relative to the overall volume of a segment remains constant"; area at each depth is from nearest neighbours, "larger in sections close to white matter" (at a gyrus); equivolume and equidistant agree in straight cortex — [BrainVoyager docs](https://download.brainvoyager.com/bv/doc/UsersGuide/HighResDataAnalysis/EquiVolumeDepthModelling.html)
- [unverified, formula from memory; not found in fetched text] Waehnert Eq.: rho = (1/(A_out - A_in)) * ( -A_in + sqrt( alpha*A_out^2 + (1 - alpha)*A_in^2 ) ), alpha = volume fraction, rho = distance fraction, A_in/A_out = segment areas at the two bounding surfaces. Derived under area linear in depth. My derivation below reproduces it.
- [verified, abstract only] Younes, Kutten & Ratnanather 2019 (arXiv 1911.07999) build "normal and equivolumetric coordinate systems" by diffeomorphic methods; search snippet says they write cross-section as linear in depth and solve a quadratic V(rho) = alpha V(1) — [arXiv](https://arxiv.org/abs/1911.07999) (the linear-section detail is from the search summary [unverified])
- [verified] Hilgetag & Barbas 2006 characterise Bok's model as "isometric" (volume-preserving): folding changes relative laminar thickness and neuron shape, but not absolute thickness or absolute neuron number; they say their data contradict that — [PLoS Comput Biol 2:e22](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0020022)

### Inferences
- [inference] With A(rho) = A_in + rho (A_out - A_in), V(rho) proportional to A_in rho + (A_out - A_in) rho^2 / 2. Setting V(rho)/V(1) = alpha gives the quadratic whose root is exactly the formula above. So the only geometric inputs are the two end areas; the curvature enters only through A_out/A_in.
- [inference] Bok's invariant is dimensionless (layer volume / segment volume). It does not by itself fix how big the segment is. Using it to carry a stria position sideways needs a second, separate assumption about the total (thickness or segment volume), and that is the quantity in question.
- [inference] For a true parallel (Steiner) shell, A(t) = A_w (1 + 2 H t + K t^2), which is quadratic in t, not linear; Waehnert's linear-area frustum and the Steiner volume differ at order K s^3. More important: if the real pial surface is flatter than the parallel surface (A_p/A_w closer to 1 than 1 + 2Hs + Ks^2 at crowns), white-curvature Steiner areas overstate A_out/A_in and the equivolume depth moves too far — consistent with the over-correction the project saw. Fix: use measured A_p/A_w (or both surfaces' curvatures), as Wagstyl did.

### Gaps
- Waehnert 2014 full text (exact equation number, whether alpha is measured from pial or white, how segment areas are computed on the level-set surfaces) was not fetched; NeuroImage paywall. Bok 1929 original not consulted.

## Neurons per column / per unit area across curvature

### Takeaway
Across *areas* (non-V1), neurons under 1 mm^2 are roughly constant (Rockel 1980; Carlo & Stevens 2013 reconfirm), but within one area across *folding*, Hilgetag & Barbas find gyral columns hold more neurons per mm^2 (deep layers +52-64%) and are thicker, at unchanged neuron density. So neither "constant neurons per column" nor Bok's "constant absolute thickness" holds at gyri vs sulci in that macaque prefrontal data; what was constant there was *volumetric neuron density per layer*.

### Cited Findings
- [verified] Carlo & Stevens 2013 (PNAS 110:1488): 94.0 ± 10.7 thousand neurons/mm^2 (Rockel et al. 1980 same areas/species: 108.9 ± 2.9 thousand); neuron count vs thickness slope -2.7e3 per mm, not significant (P = 0.63); glia per mm^2 rises with thickness (slope 21.1e3, P = 0.0001); "None of the counts ... exhibited a dependence of neuron number on column length." Primate V1 (~2x) excluded — [PMC3557031](https://pmc.ncbi.nlm.nih.gov/articles/PMC3557031)
- [verified] Carlo & Stevens counted in columns "perpendicular to the pial surface, extending down to the white matter"; sites chosen for estimated perpendicularity; rodent prefrontal excluded for "small radius of curvature". No gyral/sulcal stratification reported; the reference surface for "per mm^2" is not defined beyond the column strip — [PMC3557031](https://pmc.ncbi.nlm.nih.gov/articles/PMC3557031)
- [verified] Mean across species/areas is given as 99,780 in text and 97,780 in the Fig. 3 caption (internal inconsistency) — [PMC3557031](https://pmc.ncbi.nlm.nih.gov/articles/PMC3557031)
- [unverified, search summary] Herculano-Houzel et al. 2008 PNAS report ~3-fold variation in neurons under 1 mm^2 across primate species, contradicting uniformity — via search results for Carlo & Stevens
- [verified] Hilgetag & Barbas 2006, macaque prefrontal: total thickness sulcal 1,747 ± 112 µm, intermediate 1,791 ± 90 µm, gyral 2,227 ± 132 µm (rank rho = 0.56, p < 0.01). Deep layers (V+VI) absolute: 640 ± 70, 788 ± 66, 1,140 ± 95 µm; gyral excess "due to an absolute expansion of the deep layers." Upper:deep ratio 1.8, 1.3, 1.0 (sulcal, intermediate, gyral) — [PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0020022)
- [verified] Same paper: neuronal density in upper and deep layers not significantly different among gyral/intermediate/sulcal (ANOVA p > 0.05). Deep-layer neurons per 1 mm^2: gyral 57,396 ± 7,973; intermediate 42,508 ± 4,823; sulcal 32,205 ± 5,257 (rho = 0.73, p < 0.001); +52% gyral vs sulcal in area 13, +64% in area 8; upper layers have more neurons in sulcal areas; whole column more neurons per area in gyral (t test p < 0.03). Column neuron numbers "are not constant as is still widely assumed" — [PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0020022)
- [verified, abstract] Hilgetag & Barbas 2005 ("Developmental mechanics of the primate cerebral cortex"): folding forces significantly affect relative and absolute layer thickness in gyri and sulci; may produce more neurons in gyral regions — via [Springer Medicine](https://www.springermedicine.com/developmental-mechanics-of-the-primate-cerebral-cortex/21095220) (journal listed as Anat Embryol 210:411-7 elsewhere; sources disagree)

### Inferences
- [inference] Hilgetag & Barbas numbers: total thickness gyral/sulcal = 2227/1747 = 1.27; deep layers 1140/640 = 1.78; upper (total - deep) = 1087 vs 1107 µm, i.e. upper-layer absolute thickness is about the same. In this data set the *upper (supragranular + IV + I) absolute thickness* is close to invariant between gyri and sulci, and the curvature-dependent part is almost all in V+VI. This would make "distance from pia to layer IV" more conserved than any fraction — a candidate worth testing, though from one species/region with a coarse 3-class curvature label.
- [inference] Constant density + more neurons per mm^2 at gyri means "neurons per column" is not conserved there; any per-area count depends on which surface the mm^2 is on, and that paper's surface is not stated in what I read. If their mm^2 is at the pial side and the gyral column fans out toward the pia, per-white-area counts would differ again.
- [inference] Carlo & Stevens' constancy is across areas with different thickness, sampled at locally flat, perpendicular sites; it is not evidence about curvature.

### Gaps
- Rockel, Hiorns & Powell 1980 original, Young et al. / Collins et al. (Collins 2010 PNAS isotropic fractionator gives neurons per area across macaque cortex) not fetched.
- Which surface Hilgetag & Barbas used for "per 1 mm^2" and how they sampled column orientation: not in the portion read (last ~26k chars of the page not read).
- No human histology found in this session on neurons per column vs local curvature.

## White vs mid vs pial area; volume = thickness × area

### Takeaway
Volume ≈ thickness × area is biased in a geometry-dependent direction for any single surface; mid-surface reduces the bias but does not remove it (Winkler et al. 2018). For a smooth shell, volume per mid-surface area equals thickness to second order, so it is not a separate invariant from thickness.

### Cited Findings
- [verified] Winkler et al. 2018 (Cereb Cortex, "Joint analysis of cortical area and thickness as a replacement for the analysis of the volume of the cerebral cortex"): the area × thickness "multiplicative method ... incurs severe bias, the direction of which varies according to the local geometry"; tissue is "left unmeasured in the gyri, or measured repeatedly in sulci"; using the mid-surface reduces but does not solve it — [PMC5972607](https://pmc.ncbi.nlm.nih.gov/articles/PMC5972607)
- [verified] Same: analytic volume splits each white/pial face pair (oblique truncated triangular prism) into three tetrahedra (Aw,Bw,Cw,Ap), (Ap,Bp,Cp,Bw), (Ap,Cp,Bw,Cw), each |u·(v×w)|/6; default in FreeSurfer 6.0.0. Local differences vs multiplicative up to ~20%; vertexwise correlation ~0.80 in insula; whole-brain multiplicative 516,533.0 ± 44,562.1 mm^3 vs analytic 509,235.1 ± 44,080.0 mm^3 (multiplicative higher, p < 0.0001) — [PMC5972607](https://pmc.ncbi.nlm.nih.gov/articles/PMC5972607)
- [unverified, search snippet] The NeuroImage area paper ("Measuring and comparing brain cortical surface area and other areal quantities") is Winkler et al. **2012**, NeuroImage 61, not 2018 — [puma record](https://puma.ub.uni-stuttgart.de/bibtex/15d50386ba56e745e4634db0ecc4531a7?lang=de)
- [unverified, mailing-list snippet] FreeSurfer's default area is white-surface area; curvature is computed on white — [FreeSurfer list](https://mail.nmr.mgh.harvard.edu/archives/list/freesurfer@nmr.mgh.harvard.edu/message/HMNQ2C4SXKIRRM6QUNRSMUSRDGUFWRVT/)

### Inferences
- [inference] Steiner shell: V/A_w = s + H s^2 + K s^3/3; A_mid/A_w = 1 + H s + K s^2/4; so V/A_mid = s (1 + Hs + Ks^2/3)/(1 + Hs + Ks^2/4) = s + K s^3/12 + O(s^4). Volume per mid area is thickness up to a small Gaussian term. Choosing "per mid-area" therefore gives the same answer as "thickness", and it can be flat only if thickness is flat — which histology (above) and MRI (below) say it is not.
- [inference] V/A_p = s (1 + Hs + Ks^2/3)/(1 + 2Hs + Ks^2) ≈ s (1 - Hs + ...). Per-pial-area volume falls with H while thickness rises with H, so it is the one single-surface normalization that can partly cancel the gyral thickening. With Hilgetag & Barbas' 27% gyral thickening, flatness would need H s ≈ 0.2-0.25 at crowns — plausible order for white-surface crowns, but this is untested arithmetic, not a finding.
- [inference] Rising V/A_w with curvature (the project's observation) is the expected product of two effects with the same sign at crowns: s grows (gyri thicker) and the (1 + Hs) fan-out factor grows. It is not evidence against Bok.

### Gaps
- Van Essen 2005 surface-area paper not consulted. No source found that plots V/A_mid or V/A_p against curvature directly.

## Folding mechanics: predictions for thickness and layers

### Takeaway
Mechanical models (tension-based and differential-growth/buckling) both predict thicker gyri than sulci; buckling models produce it generically (symmetry breaking), without regional patterning. None of them conserve thickness; Hilgetag & Barbas attribute the laminar asymmetry to folding forces acting on deep vs upper layers.

### Cited Findings
- [verified, abstract] Holland, Budday, Goriely & Kuhl 2018, PRL 121:228002: in growing bilayers "thickness becomes larger at ridges than at valleys"; symmetry breaking is "a generic property of such systems"; analysis of hundreds of human brains shows region-dependent thickness variation; "gyri are universally thicker than sulci" — [CRIS FAU](https://cris.fau.de/publications/218673561); >500 scans per a press summary [unverified] — [Notre Dame](https://commandlab.nd.edu/news/notre-dame-led-team-unlocks-key-information-on-the-development-of-the-brain/)
- [unverified, search summary] Holland et al. 2020 ("Folding drives cortical thickness variations"): as growth progresses gyri thicken and sulci thin even without regional information — [CRIS FAU](https://cris.fau.de/publications/245466886)
- [verified, abstract] Budday, Steinmann & Kuhl 2014 (JMPS 72:75-92): thickness, stiffness and growth control folding onset/wavelength (Föppl–von Kármán); no gyral/sulcal thickness result in abstract — via search results
- [verified] Hilgetag & Barbas 2006: axonal tension in corticocortical connections proposed as a significant factor in cortical shape; folding forces influence laminar morphology and possibly late migration; they suggest late migration, active growth, or selective cell death to explain why gyral columns have more neurons — [PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0020022)
- [unverified, prior knowledge] Van Essen 1997 (Nature 385:313) tension-based morphogenesis; Tallinen et al. 2014 PNAS / 2016 Nature Physics differential tangential growth of a gel model reproducing gyral patterns; Bayly et al. 2013 likewise. Not fetched; their stated thickness predictions not checked.

### Inferences
- [inference] Buckling models treat the cortex as a growing plate whose mass is redistributed by the deformation — tissue moves toward the ridge — so they imply neither constant thickness nor constant volume per white-area; they are qualitatively consistent with more tissue (and, if neurons move with tissue, more neurons) under gyral crowns, matching Hilgetag & Barbas.
- [inference] None of these models, as summarised, gives a closed-form invariant to carry across curvature; they argue that the gyral/sulcal thickness ratio is a function of local geometry, which could be calibrated empirically from the visible-top columns rather than assumed.

### Gaps
- No mechanical paper read here models separate layer thickness (upper vs deep) under folding.

## Wagstyl 2018 BigBrain: layer IV depth vs curvature

### Takeaway
On BigBrain, raw layer IV depth falls steeply with curvature (beta = -15.5) while equivolumetric depth barely changes (beta = -1.07). Curvature was mean curvature on a *mid-surface*; equivolumetric depth used the actual white and layer I/II surfaces (highres-cortex), not a parallel-shell model from white curvature.

### Cited Findings
- [verified] Wagstyl et al. 2018, Cereb Cortex 28:2551-2562, doi 10.1093/cercor/bhy074: mean curvature computed at each vertex on a mid-surface between white and layer I/II surfaces; curvature and layer IV depth smoothed 3 mm FWHM; negative curvature = sulcal fundi, positive = gyral crowns, units mm^-1 — [PMC5998962](https://pmc.ncbi.nlm.nih.gov/articles/PMC5998962/)
- [verified] Equivolumetric depth computed volumetrically at 200 µm between white and layer I/II surfaces with highres-cortex (github.com/neurospin/highres-cortex), smoothed 10 mm FWHM on the surface; no explicit alpha/rho formula given in the paper — [PMC5998962](https://pmc.ncbi.nlm.nih.gov/articles/PMC5998962/)
- [verified] Raw layer IV depth vs curvature: linear beta = -15.5, F(1,302273) = 4.164e5, P < 0.0001; cubic preferred by BIC (linear 1893302, quadratic 1891660, cubic 1881609). Equivolumetric: beta = -1.07, F(1,302273) = 2706, P < 0.0001 ("almost no change"). No R^2 reported — [PMC5998962](https://pmc.ncbi.nlm.nih.gov/articles/PMC5998962/)
- [verified] Layers I-III thinner on gyral crowns and thicker in fundi; V-VI the reverse (inferred from layer IV position, not measured separately). Total thickness vs curvature not analysed. Single 65-year-old male brain. Residual inter-areal variation beyond folding — [PMC5998962](https://pmc.ncbi.nlm.nih.gov/articles/PMC5998962/)

### Inferences
- [inference] Wagstyl's test is that the *fraction* (Bok) is flat against curvature; it does not test that thickness or any per-area volume is flat. It is compatible with gyral thickening.
- [inference] Its success depends on using the real pial (layer I/II) geometry. The project's over-correction from white curvature alone is what one expects if the real A_p/A_w is closer to 1 than the Steiner value — i.e. the pial is flatter than a parallel surface. A test: compare measured A_p/A_w (or pial curvature) against 1 + 2Hs + Ks^2 in the visible-top columns.
- [inference] Beta units: depth in mm (raw) vs fraction (equivolume)? Paper text as read does not let me confirm whether -15.5 and -1.07 are on comparable scales; check before quoting them as a ratio.

### Gaps
- Whether the raw-depth regression is in absolute mm or fractional equidistant depth was not confirmed.

## Is thickness × area-ratio or volume per mid-area flat against curvature?

### Takeaway
I found no study that directly plots volume per mid-surface area, or thickness corrected by an area ratio, against curvature. Available evidence points the other way for absolute quantities: thickness (and hence V/A_mid) rises at gyri; layer *fractions* (Bok) are what histology supports as approximately curvature-invariant, and, in one macaque data set, upper-layer *absolute* thickness.

### Cited Findings
- [verified] Gyral > sulcal thickness: Hilgetag & Barbas 2006 (27% in macaque PFC) — [PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0020022); Holland et al. 2018 human MRI ("universally") — [CRIS FAU](https://cris.fau.de/publications/218673561)
- [verified] Product estimates of volume are geometry-biased for any one surface; mid-surface partly helps — [PMC5972607](https://pmc.ncbi.nlm.nih.gov/articles/PMC5972607)
- [verified] Equivolume fraction of layer IV nearly flat vs mid-surface curvature on BigBrain — [PMC5998962](https://pmc.ncbi.nlm.nih.gov/articles/PMC5998962/)

### Inferences
- [inference] Ranking of candidates to carry sideways, by evidence: (1) layer volume fraction within a segment using measured A_w and A_p (Bok; Wagstyl) — best supported, but needs the pial side; (2) absolute depth of the stria/IV from pia, or upper-layer absolute thickness — suggested by Hilgetag & Barbas numbers, untested in human; (3) thickness / V per mid-area — not conserved (gyri thicker); (4) V per white area — least conserved (thickness and fan-out both rise at crowns); (5) neurons per column — not conserved across folding in macaque PFC.
- [inference] For fused columns without a visible top, a practical route is to fit thickness (or A_p/A_w) as a function of local white-surface curvature from the visible-top columns, then apply Bok with that predicted pial geometry, instead of assuming any per-area quantity is constant.

### Gaps
- No source found testing V/A_mid or V/A_p flatness; no human histology linking gyral/sulcal absolute layer thickness to continuous curvature beyond Wagstyl's IV fraction. Fischl & Dale 2000 and Van Essen 2005 not consulted.
