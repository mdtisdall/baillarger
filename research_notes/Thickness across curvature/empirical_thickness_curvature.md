# Empirical relationship between human cortical thickness and cortical curvature

Notation in these notes: H = mean curvature (1/mm), K = Gaussian curvature (1/mm²), SI = shape index (no unit). Sign conventions differ between studies and are given for each one. FreeSurfer's `?h.curv` is positive in sulci (concave seen from outside). The project's convention is the reverse: H > 0 at a convex crown.

Tags: [verified] = read in the primary full text this session (through PMC or the journal page). [unverified] = from memory or a search snippet only.

Comparison target (project, ex vivo 7T, 160 µm): thickness rises with white-surface H. Slope about +1.2 mm per 1/mm in V1/V2 (median T 2.16 mm) and about +2.7 mm per 1/mm in M1/S1 (median T 2.88 mm). Spearman about +0.5.

## Q1. Crown vs fundus thickness: differences, ratios, effect sizes (histology and MRI)

### Takeaway
All sources agree on the direction: crowns are thicker than fundi. In MRI the crown/fundus (convex/concave) ratio is about 1.2–1.3 (about 0.4–0.6 mm), with Cohen's d about 0.5–0.85 depending on how the curvature classes are defined. Rank correlations of thickness with a gyral/sulcal index are about 0.36–0.56. No study I found reports a slope in mm per (1/mm) of curvature, so the project's slopes cannot be compared directly with a published number. Only class contrasts and correlations are available.

### Cited Findings
- **Fischl & Dale 2000 (PNAS), FreeSurfer, 30 subjects:** gyral regions 2.7 ± 0.3 mm vs sulcal regions 2.2 ± 0.3 mm. That is a difference of about 0.5 mm and a ratio of about 1.23 (ratio computed by me). The paper states "the crowns of gyri are thicker than the fundi of sulci". Thickness = "the average of this distance measured from each surface to the other" (white→pial and pial→white, averaged). The conclusions section instead says "shortest distance". [verified] — [Fischl & Dale 2000, PMC27146](https://pmc.ncbi.nlm.nih.gov/articles/PMC27146)
- **Fischl & Dale 2000:** postmortem comparison values were "taken at gyral crowns". Regional MRI vs postmortem: lateral 2.9 ± 0.3 vs 3.5 mm, medial 2.4 ± 0.3 vs 2.7 mm, inferior 2.7 ± 0.3 vs 3.0 mm. The overall average was 2.5 mm in both. [verified] — [PMC27146](https://pmc.ncbi.nlm.nih.gov/articles/PMC27146)
- **Wagstyl et al. 2015 (NeuroImage):** the introduction says gyri are "on average 20% thicker than sulci", citing Van Essen & Maunsell 1980 and Fischl & Dale 2000. Unsmoothed FreeSurfer gyri were significantly thicker than sulci (two-sample t, p < 0.001). No effect size is given. [verified] — [Wagstyl 2015, PMC4401442](https://pmc.ncbi.nlm.nih.gov/articles/PMC4401442)
- **Demirci & Holland 2022 (Hum Brain Mapp), ABIDE-I, N = 501 adults at 3T, FreeSurfer white/pial surfaces decimated to about 40k vertices per hemisphere:**
  - Sign of H: negative = convex.
  - Thickness by sign of H: convex 2.88 mm vs concave 2.47 mm (p ≪ 1e-10), Cohen's d = 0.53. This is a difference of 0.41 mm and a ratio of 1.17 (computed by me).
  - Shape classes from the signs of the principal curvatures: convex 2.96, saddle 2.70, concave 2.33 mm, with d values reported as 0.85, 0.53, 0.36 (pairs not clearly labelled in the extract). Convex vs concave is 0.63 mm, ratio 1.27 (computed by me).
  - Thickness in 9 shape-index bins from cup to cap: 2.34, 2.37, 2.39, 2.62, 2.74, 2.79, 2.86, 3.02, 3.10 mm. The rise is monotonic and about 0.76 mm across the full SI range.
  - Sulcal depth: exterior 2.95 vs interior 2.48 mm, d = 0.60.
  - Insula: gyral subregions 3.4 vs sulcal 2.3 mm.
  - Mean thickness 2.71 ± 0.8 mm.
  - Thickness limits 0.5–5 mm excluded about 2% of vertices.
  - Caveat: the paper's convex/concave labelling for K is internally inconsistent (noted by the extractor).
  - [verified] — [Demirci & Holland 2022, PMC8933257](https://pmc.ncbi.nlm.nih.gov/articles/PMC8933257)
- **Demirci, Hoffman & Holland 2023 (NeuroImage), across primates:** "thickness ratio" = mean convex thickness / mean concave thickness. It ranges from about 1.05 (Senegal galago) to about 1.3 (humans). In every species, convex > saddle > concave. The ratio does not change systematically with age in humans. Curvature was computed on the **pial** surface. [verified] — [Demirci et al. 2023, PMC10443624](https://pmc.ncbi.nlm.nih.gov/articles/PMC10443624)
- **Hilgetag & Barbas 2006 (PLoS Comput Biol), macaque prefrontal histology, 21 areas:**
  - Total thickness: sulcal 1,747 ± 112 µm, intermediate 1,791 ± 90 µm, gyral 2,227 ± 132 µm (ρ = 0.56, p < 0.01). Gyral/sulcal ratio about 1.27 (computed by me).
  - MRI in 5 monkeys: thickness vs gyral/sulcal elevation ρ = 0.36 (p = 1e-6).
  - [verified] — [Hilgetag & Barbas 2006](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0020022)
- **Wagstyl et al. 2020 (BigBrain 20 µm histology, PLoS Biol):** "folding-related differences are clearly visible on the BigBrain, with sulci being thinner than their neighboring gyri". No numbers are given. [verified] — [Wagstyl 2020, PMC7159250](https://pmc.ncbi.nlm.nih.gov/articles/PMC7159250)
- **Bok 1929 (histology, 2D sections):** crowns thicker than fundi, as cited by Demirci & Holland 2022. [verified as citation only; original not read] — [PMC8933257](https://pmc.ncbi.nlm.nih.gov/articles/PMC8933257)
- **von Economo & Koskinas (1925):** crown thicker than fundus throughout, often stated as a crown/fundus difference of roughly a factor of 1.5–2 in their descriptions. [unverified: from memory, not read; no precise number confirmed]

### Inferences
- Ratios from MRI (about 1.2–1.3) and from macaque histology (about 1.27) are similar. This suggests the effect is mostly not an MRI artefact, but it does not test that directly (see Q4).
- Rough check of the project's slope against the literature (my estimate, not a published number): a typical white-surface H at a crown is about +0.3 to +0.5 /mm and at a fundus about −0.3 to −0.5 /mm, a span of about 0.6–1.0 /mm. The project's slopes then give crown–fundus differences of about 0.7–1.2 mm (V1/V2) and 1.6–2.7 mm (M1/S1). The MRI class contrasts (0.4–0.6 mm) are averages over whole curvature classes, not extremes, so they are expected to be smaller. The V1/V2 slope looks plausible. The M1/S1 slope is at the high end, possibly inflated by the precentral (thick, crown) / area 3b (thin, fundus) confound (see Q3).
- The Spearman of about +0.5 in the project is close to the Hilgetag & Barbas histology ρ = 0.56 and above the MRI ρ = 0.36.

### Gaps
- I found no published regression slope of thickness on curvature in mm per (1/mm), for any surface. Demirci & Holland report local Pearson r in Supplementary Figures 4–5, which I did not read.
- The original von Economo and Bok crown/fundus numbers were not read. BigBrain crown/fundus numbers are not reported in Wagstyl 2018 or 2020.

## Q2. Which curvature (white / mid / pial; mean / Gaussian / shape index / sulcal depth) and which scale predicts thickness best?

### Takeaway
No study I read compares white, mid and pial curvature directly. Among the measures tested on the same data (Demirci & Holland 2022), the classes from the signs of the principal curvatures (convex / saddle / concave, d up to 0.85) and the shape index separate thickness more strongly than the sign of H (d = 0.53) or sulcal depth (d = 0.60). In BigBrain, layer IV depth against mid-surface H (3 mm FWHM) is non-linear; a cubic fit is best by BIC.

### Cited Findings
- **Demirci & Holland 2022:**
  - Smoothing: H, K and SI were computed on the decimated mesh after Laplacian (50 iterations) and Taubin (100 iterations) smoothing, then 2 iterations of distance-weighted neighbour averaging.
  - Which of the FreeSurfer surfaces was used is not stated in the part read.
  - Units: H in 1/mm, K in 1/mm². Mean K was 0.12 /mm² at inward-bulging points and 0.08 /mm² at outward-bulging points.
  - Effect sizes: sign of H d = 0.53; principal-curvature-sign classes d = 0.85 (largest pair); sulcal depth d = 0.60. Sulcal depth was measured from an alpha-shape hull surface offset by 7 mm.
  - SI bins give a monotonic thickness rise from 2.34 to 3.10 mm.
  - Saddle points are about 55% of the surface area.
  - [verified] — [PMC8933257](https://pmc.ncbi.nlm.nih.gov/articles/PMC8933257)
- **Demirci et al. 2023:** curvature measures were computed on the **pial** surface, with 2 iterations of local averaging. They used SI because it is scale-invariant and so allows comparison across species. [verified] — [PMC10443624](https://pmc.ncbi.nlm.nih.gov/articles/PMC10443624)
- **Wagstyl et al. 2018 (Cereb Cortex, BigBrain):**
  - Curvature: "Mean cortical curvature was calculated at each vertex on a mid-surface". Curvature and depth data were both "smoothed with 3 mm isotropic FWHM Gaussian kernel". Sign: positive = gyral crowns.
  - Layer IV relative depth vs H: linear β = −15.5 (percentage depth per 1/mm). Quadratic and cubic fits are better; BIC: linear 1,893,302, quadratic 1,891,660, cubic 1,881,609.
  - [verified] — [Wagstyl 2018, PMC5998962](https://pmc.ncbi.nlm.nih.gov/articles/PMC5998962)
- **Hilgetag & Barbas 2006:** used a categorical gyral/intermediate/sulcal label per area in histology, and "elevation" (a sulcal-depth-like measure) in MRI. [verified] — [Hilgetag & Barbas 2006](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0020022)
- **Wagstyl 2015:** removed folding effects by averaging within 100 random equal-area parcels, repeated 10 times. The authors say this is "almost equivalent to surface-based Gaussian kernel smoothing". No curvature regression is described in the main text. [verified] — [PMC4401442](https://pmc.ncbi.nlm.nih.gov/articles/PMC4401442)
- **HCP pipelines:** produce a curvature-regressed thickness map ("corrThickness"). [unverified: from memory; I did not find the regression details in Van Essen 2012 (PMC3432236) in the part I read, and Glasser et al. 2013 was not read]

### Inferences
- The best MRI separator (principal-curvature-sign classes or SI) combines both principal curvatures. SI is invariant to scale, so it may transfer across areas with different fold sizes (V1 vs M1) better than raw H.
- The white surface is the sharper surface at crowns, and the pial surface is the sharper surface at fundi. A white-surface H predictor, as the project uses, will resolve crowns well and fundi poorly. This is a geometric argument, not a measured comparison.
- Smoothing scales used in the literature: about 3 mm FWHM (Wagstyl 2018) and a few neighbour-averaging iterations on a 40k-vertex mesh (Demirci). No study I found tunes the scale for best prediction.

### Gaps
- No head-to-head study of white vs mid vs pial curvature as a predictor of thickness, and no study of the best smoothing scale.

## Q3. Area dependence: V1, V2, M1 (precentral), S1 (area 3b, fundus of the central sulcus)

### Takeaway
Areal differences in thickness are of the same size as, or larger than, the folding effect, and in the central region they are confounded with folding. Area 3b (posterior bank and fundus) is about 1.7–1.8 mm, and area 4 (anterior bank and crown) is about 2.7 mm in MRI and up to more than 4.5 mm in BigBrain. A crown/fundus slope pooled across M1/S1 therefore includes an areal step. V1 is thin (about 1.5–2.9 mm), and its fold-related variation is within that range. I found no published thickness–curvature slope reported separately for V1, V2, M1 or S1.

### Cited Findings
- **Fischl & Dale 2000, central sulcus:**
  - Anterior bank (area 4/3a): 2.69 mm (manual MRI); postmortem 2.7 mm.
  - Posterior bank (area 3b/1): 1.81 mm (manual MRI); postmortem 1.7 mm.
  - Introduction: area 3 is "less than 2 mm", and area 4 frequently exceeds 4 mm.
  - [verified] — [PMC27146](https://pmc.ncbi.nlm.nih.gov/articles/PMC27146)
- **Wagstyl 2020, BigBrain (shrinkage-corrected):**
  - "Thickness values range from 1.8 mm in the calcarine sulcus to 4.5 mm in the precentral gyrus".
  - Calcarine 95% range 1.67–2.86 mm.
  - von Economo area OC: 1.8–2.3 mm. Amunts 2007 V1: 1.47 ± 0.24 (left) and 1.57 ± 0.41 mm (right).
  - Precentral gyrus over 4.5 mm; von Economo area FA 3.5–4.5 mm.
  - Layer III is thick in precentral areas and especially thin in V1.
  - Regional correlation of BigBrain thickness with von Economo r = 0.86; with HCP MRI r = 0.62 (left) and 0.75 (right).
  - Shrinkage factors: 1.931 (3D), 1.245 (1D).
  - [verified] — [PMC7159250](https://pmc.ncbi.nlm.nih.gov/articles/PMC7159250)
- **Wagstyl 2018:** layer IV in V1 (calcarine) is relatively deep, "close to the white surface". Equivolumetric layer IV depth still varies by region after the curvature correction. [verified] — [PMC5998962](https://pmc.ncbi.nlm.nih.gov/articles/PMC5998962)
- **Fischl & Dale 2000:** primary sensory areas have "reduced contrast" (because of high myelin), which affects surface placement there. [verified] — [PMC27146](https://pmc.ncbi.nlm.nih.gov/articles/PMC27146)
- **Demirci & Holland 2022:** cites Zhang et al. 2017: convex regions were thicker than concave regions on adjacent banks of the precentral gyrus. [verified as citation; Zhang 2017 not read] — [PMC8933257](https://pmc.ncbi.nlm.nih.gov/articles/PMC8933257)
- **Meyer et al. 1996:** used the precentral/postcentral thickness asymmetry to identify the central sulcus on MRI. [unverified: snippet only] — [AJNR](https://www.ajnr.org/node/9112)

### Inferences
- **M1/S1:** the project's +2.7 mm per 1/mm probably contains a between-area component. The precentral crown is area 4 (thick), and the central-sulcus fundus and posterior bank are area 3b (thin, about 1.8 mm). An area-4/3b step of about 0.9 mm (MRI) to more than 2 mm (BigBrain) sits on top of any within-area folding effect. To carry thickness across a fused central sulcus, it is better to fit the M1 and S1 sides separately, or to add an area term.
- **V1/V2:** the smaller slope (+1.2 mm per 1/mm) is consistent with V1 being thin everywhere. Gyral and sulcal V1 thicknesses both fall within the 1.5–2.9 mm range. The V1/V2 border also changes thickness and is often on the calcarine banks, so the same confound applies there.

### Gaps
- No per-area slope or correlation for V1, V2, M1 or S1 was found in the sources read.

## Q4. How much of the MRI curvature–thickness relation is methodological bias?

### Takeaway
Histology shows the same direction and similar size of effect: macaque gyral/sulcal ratio about 1.27; Bok; BigBrain. So the effect is at least partly real. No study I found measures the fraction that comes from bias (partial volume in fundi, pial-surface placement, thickness definition) against high-resolution ex vivo ground truth.

### Cited Findings
- **Fischl & Dale 2000:**
  - The pial surface is hard to locate "because of limitations on the resolution of MRI".
  - Measuring thickness on 2D slices "will always result in overestimates" unless the surface is orthogonal to the slice plane.
  - Smoothness constraints that minimise curvature give surfaces that are "inaccurate" at highly curved fundi and crowns.
  - Precision depends on contrast-to-noise.
  - Postmortem values (taken at crowns) were up to about 0.5 mm higher than MRI regional means.
  - [verified] — [PMC27146](https://pmc.ncbi.nlm.nih.gov/articles/PMC27146)
- **Demirci & Holland 2022:** partial volume is not discussed in the part read. Robustness checks were thickness limits and a single site (Yale). Histological support for FreeSurfer thickness is cited (Cardinale 2014; Rosas 2002). [verified] — [PMC8933257](https://pmc.ncbi.nlm.nih.gov/articles/PMC8933257)
- **Wagstyl 2015:** FreeSurfer vs histology in auditory cortex, 2.84/2.88 mm vs 2.9 mm (SD 0.31 mm). This check concerns myelin-related underestimation, not folding. [verified] — [PMC4401442](https://pmc.ncbi.nlm.nih.gov/articles/PMC4401442)
- **FreeSurfer vs histology on resected tissue:** Cardinale et al. 2014 (epilepsy surgery) found agreement, but did not test crown vs fundus. [unverified: snippet only] — [Cardinale 2014 PDF](https://re.public.polimi.it/bitstream/11311/791518/4/Validation%20of%20FreeSurfer-Estimated%20Brain%20Cortical%20Thickness_11311-791518_Ferrigno.pdf)
- **Partial volume can fuse opposing banks:** this can produce "bridged" sulci and unresolved fundi. [unverified: review snippet] — [search result](https://ora.ox.ac.uk/objects/uuid:0d2d474d-cb66-4aaa-8303-3d75e6834d61)
- **Kleinnijenhuis et al. (7T / ex vivo diffusion):** diffusion orientation varies with gyral position (radial near crowns, tangential in deep layers of the fundus). This concerns microstructure, not thickness. [unverified: abstract only] — [Kleinnijenhuis OHBM 2014](https://users.fmrib.ox.ac.uk/~michielk/MendeleyLibrary/abstracts/Kleinnijenhuis_OHBM-2014_abstract.pdf)

### Inferences
- The likely biases go in both directions:
  - Fundal partial volume and fused banks tend to make sulcal cortex look thicker, which would reduce the measured crown–fundus difference.
  - Missed thin pial CSF at crowns, or myelin-related shifts of the white surface, can change crown values.
  - Because of this, the MRI ratio (about 1.2–1.3) cannot be called inflated or deflated without ground truth.
- The project's 160 µm ex vivo data are themselves among the higher-resolution evidence for this relation. Its open-column slopes would be a new kind of measurement, not a replication of a published value.

### Gaps
- No ex vivo high-resolution MRI study that reports thickness against curvature with histology validation was found.
- Partial-volume fraction of the effect: not quantified anywhere I found.

## Q5. Laminar thickness against curvature (which layers thicken at crowns vs fundi)

### Takeaway
At crowns the deep layers (V–VI) are thicker; in fundi the upper layers (I–III) take a larger share. In macaque prefrontal cortex, gyral cortex is thicker mainly because V+VI expand (1,140 vs 640 µm), and neuron density per volume does not differ. In BigBrain, layer IV moves deeper in sulci (β = −15.5 % depth per 1/mm of mid-surface H), and the equivolumetric (Bok) model removes almost all of that dependence (β = −1.07).

### Cited Findings
- **Hilgetag & Barbas 2006, macaque prefrontal:**
  - Ratio of upper to deep layer thickness: sulcal 1.8 ± 0.1, intermediate 1.3 ± 0.1, gyral 1.0 ± 0.1.
  - V+VI absolute thickness: sulcal 640 ± 70, intermediate 788 ± 66, gyral 1,140 ± 95 µm (p < 0.01). The extra gyral thickness comes from the deep layers.
  - Deep-layer neurons per mm² of surface: gyral 57,396 ± 7,973; intermediate 42,508 ± 4,823; sulcal 32,205 ± 5,257 (ρ = 0.73). By area: +52% in area 13, +64% in area 8.
  - Neuron density per mm³: no difference between gyral and sulcal cortex (p > 0.05).
  - The authors say the absolute-thickness and neuron-count data contradict Bok's isometric (constant-volume) hypothesis, while the relative layer thicknesses agree with it.
  - [verified] — [Hilgetag & Barbas 2006](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0020022)
- **Wagstyl 2018, BigBrain:**
  - Layer IV relative depth vs mid-surface H (3 mm FWHM): β = −15.5, with a non-linear (cubic) fit best. Layer IV is shallower (more superficial) in gyri and deeper in sulci.
  - Equivolumetric depth of layer IV vs H: β = −1.07, "almost no change".
  - Automated vs manual layer IV relative depth r = 0.72.
  - [verified] — [PMC5998962](https://pmc.ncbi.nlm.nih.gov/articles/PMC5998962)
- **Wagstyl 2020:** layer thickness gradients (III, V, VI) are reported along processing hierarchies, but not against curvature. Mean laminar fractions in the training profiles: I 7.5%, II 5.6%, III 20.8%, IV 5.5%, V 14.8%, VI 17.8% (with CSF and white matter in the rest). [verified] — [PMC7159250](https://pmc.ncbi.nlm.nih.gov/articles/PMC7159250)
- **Bok 1929:** in fundi the upper layers are thick and the deep layers thin; at crowns the reverse, so that each layer segment keeps its volume. [unverified in original; the summary agrees with the descriptions in Hilgetag & Barbas and Wagstyl 2018]

### Inferences
- Total thickness and laminar position behave differently. Equivolume explains where layer IV sits, but not why total thickness changes with curvature (Hilgetag & Barbas: more neurons in deep layers at crowns). A depth model should use equivolume for the relative depth inside the column and treat total thickness as a separate field that depends on curvature and area.

### Gaps
- No human laminar-thickness-vs-curvature table with mm values per layer was found (BigBrain papers give depth only, or no curvature analysis).
- Human V1/M1-specific laminar curvature effects were not found.
