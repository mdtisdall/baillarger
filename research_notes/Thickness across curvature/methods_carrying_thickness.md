# Methods that model, correct or carry cortical thickness across curvature

Scope: curvature covariates, curvature-corrected thickness, thickness priors and spatial models that include folding. Context: an ex vivo 7T project needs a Gaussian (Matérn/SPDE) thickness field on the white surface. Its open-column data give slopes of ~+1.2 mm per 1/mm (V1/V2) and ~+2.7 mm per 1/mm (M1/S1) vs white-surface mean curvature. Equivolume per unit white area is not conserved.

Tags: [verified] means I read it in the primary source (full text or abstract). [unverified] means it comes from a search snippet, a secondary source or memory. [inference] means my own reasoning.

## Q1. Curvature-corrected or curvature-regressed thickness in MRI: model form, variance explained, residual structure

### Takeaway
There are two families. (a) The older approach takes one global linear regression of thickness on mean curvature per subject: CT = β·H + c, then subtract β·H. HCP / Glasser-Van Essen use it. (b) The newest and strongest approach is Demirci, Coalson, Holland, Van Essen & Glasser (2026). It uses a *local* (6 mm FWHM Gaussian-weighted) multiple regression at each vertex on five curvature measures plus their squares. It shows that the global linear form underfits. It also shows that the folding effect is regionally variable and scales with regional mean thickness: thick regions need more compensation, and thin visual cortex needs less. That matches the project's finding of a ~2× larger slope in M1/S1 than in V1/V2. No source I found reports a global R² for thickness on curvature.

### Cited Findings
**Demirci, Coalson, Holland, Van Essen, Glasser 2026, "Compensating cortical thickness for cortical folding-related variation", Imaging Neuroscience 4, doi 10.1162/imag.a.1106 (PMC12836397)** — [PMC full text via PubMed](https://pmc.ncbi.nlm.nih.gov/articles/PMC12836397/), [DOI](https://doi.org/10.1162/IMAG.a.1106)
- [verified] Prior global method: FreeSurfer mean curvature H, model CT = β·H + c, compensated CT = CT − β·H. "Global regression of mean curvature from cortical thickness maps leaves substantial folding-related variation unaccounted for."
- [verified] Their stated reasons why global linear underfits: (1) the effect of folding "depends on what the underlying thickness would have been in the absence of folding … which varies regionally", so one coefficient underfits. (2) Mean curvature is an incomplete description of folding. (3) The effects are nonlinear.
- [verified] Model: at each vertex, a Gaussian-weighted multiple regression of CT on five demeaned measures (k_max, k_min, Gaussian curvature K, shape index, curvedness) and their squares. Mean curvature is left out because it is collinear with k_max and k_min. Compensated CT = CT − Σ(linear + quadratic terms).
- [verified] They found the second-order coefficients "close to zero, indicating that the relationship is predominantly linear". The nonlinearity in practice comes from *spatially varying* linear coefficients, not curvature saturation.
- [verified] Surface/scale: curvatures are computed on the **midthickness** surface, resampled to a regular 164k mesh. The surface is smoothed at 2.14 mm FWHM and the curvature maps at 2.52 mm FWHM. The local-regression patch is 6 mm FWHM. The search range for the patch was 3–10 mm, and the minimum of 3 mm makes sure that each patch holds at least one inward and one outward fold. The parameters were chosen by golden-section search to maximize the variance in each individual compensated map that the group-average map explains. The group average after global regression serves as a "silver standard" (folding averages out across subjects after MSMAll).
- [verified] Thickness is FreeSurfer thickness from HCP pipelines (0.7 mm HCP-YA, 0.8 mm HCA).
- [verified] Difference map: "sulci get thicker with compensation and gyri get thinner … by similar amounts". As a second-order effect, changes are "more pronounced in cortical regions that are thick on average (e.g., insular … anterior cingulate …)" and "more modest in … thin on average (e.g., visual cortex, superior parietal cortex)".
- [verified] Mean within-area SD falls from 0.38 to 0.22 mm (HCP-YA) and from 0.40 to 0.25 mm (HCA). The age effect size rises by 41% (abstract).
- [verified] After local compensation the thickness vs (shape index × sulc) table becomes uniform. After global linear compensation, "the relationship … is still discernible".
- [verified] Residual structure: "Much of the remaining high-spatial frequency variation … is related to sub-mm errors in surface placement". Inter-subject CoV stays high in insula, visual and somatosensory cortex (thin, myelinated cortex, so surfaces are placed poorly).
- [verified] They state that true areal differences are preserved, e.g. "thicker motor and thinner somatosensory cortices". The central-sulcus banks are an example of an architectural difference at similar (flat) geometry.
- [verified] They chose midthickness over a mid-equivolume surface because equivolume coordinates depend on white- and pial-surface curvature and so could propagate surface-placement errors into the curvature covariates.

**Demirci & Holland 2022, "Cortical thickness systematically varies with curvature and depth in healthy human brains", Hum Brain Mapp 43(6):2064–2084, doi 10.1002/hbm.25776** — [PMC8933257](https://pmc.ncbi.nlm.nih.gov/articles/PMC8933257)
- [verified] N = 501 (ABIDE-I controls). Thickness is FreeSurfer symmetric closest-point: the mean of white→pial and pial→white nearest distances. Vertices outside 0.5–5 mm were excluded (~2%). Mean thickness is 2.71 ± 0.8 mm.
- [verified] The model is binned group means with t-tests and Cohen's d. There is **no regression, slope or R²**.
- [verified] Results: by mean-curvature sign, convex 2.88 vs concave 2.47 mm (d = 0.53). By Gaussian curvature, convex 2.96 / saddle 2.70 / concave 2.33 mm. Across shape-index bins from cup to cap, thickness rises monotonically from 2.34 to 3.10 mm. By sulcal depth, exterior 2.95 vs interior 2.48 mm (d = 0.60). In the insula, gyral subregions are ~3.4 mm and sulcal subregions ~2.3 mm.
- [verified] Mesh pre-processing: decimated to ~40k vertices per hemisphere, then 50 Laplacian + 100 Taubin smoothing iterations and two rounds of neighbour averaging. Sulcal depth is the distance to an alpha-shape hull (α = 20) shrunk by 7 mm. The text I read does not state which surface the curvatures come from.
- [verified] Saturation at crowns or fundi is not discussed. The shape-index bins are monotonic.

**Vandekar et al. 2016, "Subject-level measurement of local cortical coupling", NeuroImage 133:88–97** — [PMC4889557](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889557)
- [verified] Local Gaussian-weighted linear regression of CT on FreeSurfer sulcal depth at each vertex. Kernels are 5/10/15 mm FWHM, distances are Euclidean on the inflated surface, and data are FreeSurfer 5.3 on fsaverage5. N = 932 youths aged 8–22. The slope is the "coupling", and a local R² is also given.
- [verified] Coupling is mostly negative (thinner in deeper cortex) and spatially heterogeneous. It is positive in some clusters in V1, somatomotor and inferior temporal cortex at some kernels. The authors attribute this to areal junctions, e.g. "precentral cortex is substantially thicker than postcentral cortex at a similar SD". They say it is "not driven by an inversion of the normal relationship". Local R² is low in visual cortex and the central sulcus, and they flag this as a "violation in the linearity assumption".
- [verified] The sign within a sulcus can change with kernel size (positive at 5 mm, negative at 15 mm). No numeric coupling range is reported. Sulcal depth was in arbitrary units before FS6.
- [unverified] Vandekar et al. 2015, J Neurosci (N = 932): development-related thinning is greatest in sulcal depths, with early thickening on gyri (search snippet only).

**Wagstyl et al.** — [Wagstyl 2015 NeuroImage, PMC4401442](https://pmc.ncbi.nlm.nih.gov/articles/PMC4401442); [Wagstyl 2016 Transl Psychiatry, PMC4872401](https://pmc.ncbi.nlm.nih.gov/articles/PMC4872401/)
- [verified] Wagstyl, Ronan, Goodyer, Fletcher 2015 ("Cortical thickness gradients in structural hierarchies") states that thickness was "adjusted to account for the local effects of folding" but gives **no formula** in the text I read. The described procedure averages thickness over 10 random 100-parcel parcellations. Thickness is the white→pial shortest distance.
- [verified] Wagstyl et al. 2016 ("Multiple markers of cortical morphology reveal evidence of supragranular thinning in schizophrenia") has **no curvature-corrected thickness**. It splits gyri and sulci by the sign of mean curvature and uses a gyral–sulcal thickness difference (GSD, 25 mm radius) as a marker. Its rationale is that folding stretches and thins outer layers and compresses and thickens inner ones. It cites von Economo: supragranular layers are 70% of fundus thickness vs 49% of crown thickness.
- So the brief's guess that Wagstyl 2016 is a curvature-regression paper is **not confirmed**. It treats the gyral–sulcal difference as signal, not as nuisance.

### Inferences
- [inference] The Demirci 2026 result that compensation scales with regional mean thickness fits a *multiplicative* curvature effect, e.g. T ≈ T0(x)·(1 + a·H), rather than an additive slope common to all regions. The project's slopes are ~1.2 in V1/V2 (thin) and ~2.7 in M1/S1 (thick). If one fits T = T0 + β·H with β ∝ T0, a single dimensionless a = β/T0 might explain both. This is worth testing: compute β/mean(T) for each region on the open columns.
- [inference] The precedent supports a prior mean with a curvature term plus a residual Matérn field, and either (i) a curvature coefficient that varies by region or space, or (ii) a multiplicative / log-thickness form. Global linear on its own is shown to underfit (Demirci 2026).
- [inference] The Vandekar areal-junction warning applies directly to M1/S1. The central sulcus combines a curvature effect with a large areal step at similar geometry. A coefficient fitted across that border will mix the two. Fit within areas, or include an areal (facies) offset.
- [inference] Demirci 2026 removes folding effects for group statistics. The project needs the opposite: it wants to *predict* thickness from curvature. Their local-regression coefficients are spatially varying slopes estimated from data that include sulcal fundi. In fused banks the project lacks those data, so its slope must be extrapolated from open columns.

### Gaps
- No source I read gives a global R² or variance explained for thickness on mean curvature within a subject. Demirci 2026 optimizes a different R² (individual vs group map) and reports values only in figures I could not read.
- No source gives slopes in mm per (1/mm) on the white surface that could be compared with the project's +1.2 / +2.7. Demirci 2026 uses midthickness curvature, and FreeSurfer curvature sign conventions differ between sources.
- I could not read the Demirci 2026 coefficient maps (figure only), so the regional size of β is unknown.
- I did not identify the citation of the original temporal-lobe linear regression (the earlier work Demirci 2026 cites). [unverified] I believe the global regression applied to the whole cortex is from Glasser & Van Essen 2011 (J Neurosci, myelin maps) and Glasser et al. 2016 (HCP-MMP1.0). The names are elided in the text I could retrieve.

## Q2. Thickness priors with geometric covariates in segmentation tools; Bayesian surface models

### Takeaway
Segmentation tools take thickness priors as a constant or as a user-supplied image. I found no tool whose expected thickness depends on curvature. Bayesian SPDE/GP surface models exist for cortical data (mainly fMRI GLMs: Mejia et al.). I found none that models thickness with a curvature covariate in the mean.

### Cited Findings
- [verified] ANTs KellyKapowski (DiReCT; Das et al. 2009 NeuroImage) has `-t/--thickness-prior-estimate` (a scalar upper constraint, default 10 mm) and `-a/--thickness-prior-image` (an "image with spatially varying prior thickness values"). Nothing in the docs ties it to curvature. Older (2.1.0) man pages word this option differently. — [Debian man page](https://dyn.manpages.debian.org/unstable/ants/KellyKapowski.1.en.html), [ANTsPy source](https://antspy.readthedocs.io/en/latest/_modules/ants/segmentation/kelly_kapowski.html)
- [verified] The ANTsPy `kelly_kapowski` wrapper has no named prior-thickness argument. It would have to be passed through kwargs. — [ANTsPy docs](https://antspy.readthedocs.io/en/latest/api/ants.segmentation.kelly_kapowski.html)
- [unverified] The surface SPDE prior (Lindgren et al. 2011 Matérn–SPDE) is used on cortical meshes for a spatial Bayesian GLM of task fMRI (Mejia et al. 2020 JASA, arXiv 1706.00959; Spencer et al. 2022 NeuroImage, arXiv 2106.06669). Its parameters separately control range and variance. These are fMRI coefficient fields, not thickness. — [arXiv 1706.00959](https://arxiv.org/pdf/1706.00959), [arXiv 2106.06669](https://arxiv.org/pdf/2106.06669)
- [unverified] Whiteman, Johnson & Kang (arXiv 2306.03663): group-level cortical-surface image-on-scalar regression with GP priors, covariate effects that vary across the surface, and a Vecchia approximation. — [arXiv 2306.03663](https://arxiv.org/abs/2306.03663)
- [unverified] Thickness-specific Bayesian spatial models model spatial correlation of *group effects* (e.g. ADNI latent spatial model; bivariate thickness/volume model, arXiv 1901.00068), not curvature in the mean. — [ADNI PDF](https://adni.loni.usc.edu/adni-publications/Dai-A%20Bayesian%20latent%20spatial%20model%20for%20mappin.pdf), [arXiv 1901.00068](https://arxiv.org/abs/1901.00068v4)

### Inferences
- [inference] Precedent for the project's design is at the level of components: SPDE fields on cortical meshes (Mejia), spatially varying coefficients (Whiteman; Demirci's local regression). Combining them into "Matérn residual + curvature trend, predict at unobserved vertices" has no thickness precedent that I found. The project would be the first or near-first such design, so it must validate the design itself, e.g. hold out open columns at high |H|.
- [inference] KellyKapowski's prior image is a hook into which a curvature-predicted thickness map could be fed. That is a use, not an existing method.

### Gaps
- CRUISE/ACE (Han et al. 2004) and CLASP/CIVET: I did not check whether any of them uses curvature-dependent expected thickness. [unverified, memory] ACE uses a distance-based skeleton to separate banks in tight sulci, not a thickness prior.
- BrainSpace (Vos de Wael 2020) provides gradients and spin/Moran null models, not thickness priors. Not checked further.
- I found no surface SPDE/INLA paper with thickness as the response.

## Q3. Universal kriging / trend + residual on a surface with a curvature covariate

### Takeaway
I found no neuroimaging use of universal kriging or a "trend + residual" model with curvature as the drift. The nearest neighbours are local GWR-style regressions (Vandekar 2016 coupling; Demirci 2026 compensation), i.e. spatially varying coefficients estimated by kernel weighting rather than kriged.

### Cited Findings
- [verified] Vandekar 2016 and Demirci 2026 both use vertex-centred Gaussian-kernel weighted least squares (geographically weighted regression in form), with FWHM 5–15 mm and 6 mm respectively. — [PMC4889557](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889557), [PMC12836397](https://pmc.ncbi.nlm.nih.gov/articles/PMC12836397/)
- [unverified] COHIBA-style geostatistics (surface depth conversion): a trend with Gaussian priors on the coefficients, plus a residual Gaussian field, solved jointly by universal kriging / Bayesian kriging. Not checked in this session.

### Inferences
- [inference] Bayesian universal kriging with a Gaussian prior on β (mean ~ the open-column slope, SD reflecting the 1.2 vs 2.7 regional spread) carries the uncertainty in β into the fused-bank predictions. A plug-in β would understate the variance exactly where H is far from the fitted range (deep fundi), since the variance contribution of β is H²·Var(β).
- [inference] If β varies by region, model it as either (a) one β per areal facies with a hierarchical prior, or (b) a second, smoother Matérn field β(x). Option (b) is the SPDE analogue of Demirci's local regression. Identifiability needs curvature variation within each range length. Demirci needed ≥ 3 mm FWHM patches holding one inward and one outward fold.

### Gaps
- No precedent quantifies how well kriged or local-regression thickness predicts held-out sulcal thickness.

## Q4. Curvature scale, surface, and linearity across the range

### Takeaway
The surfaces in use are midthickness (Demirci 2026) or unspecified/FreeSurfer (Demirci & Holland 2022). Scales are ~2–3 mm FWHM of curvature smoothing, with the regression pooled over a 6 mm FWHM patch. Within a local patch the relation is close to linear (quadratic terms ≈ 0). Across the whole cortex, binned thickness rises monotonically from cup to cap with no reported saturation. Nobody I found used *white-surface* curvature as the covariate.

### Cited Findings
- [verified] Demirci 2026: midthickness, 2.14 mm FWHM surface smoothing, 2.52 mm FWHM curvature smoothing, 6 mm FWHM regression patch. Quadratic terms are near zero. — [PMC12836397](https://pmc.ncbi.nlm.nih.gov/articles/PMC12836397/)
- [verified] Demirci & Holland 2022: monotonic increase over 9 shape-index bins (2.34 → 3.10 mm). Saturation is not discussed. Heavy mesh smoothing (Laplacian + Taubin, decimation to 40k). — [PMC8933257](https://pmc.ncbi.nlm.nih.gov/articles/PMC8933257)
- [verified] Vandekar 2016: the sign of the local thickness–depth slope can flip between 5 and 15 mm kernels, so the inferred relation depends on scale. — [PMC4889557](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889557)

### Inferences
- [inference] White-surface vs midthickness curvature: at a gyral crown the white surface is sharper (smaller radius) than the midthickness surface. In a fundus the white surface is broader, while the pial surface is sharp, or fused in the project's case. A slope per unit H_white therefore differs from a slope per unit H_mid, and the two are not linearly interchangeable where |H|·T is not small. The project's white-surface slopes cannot be compared numerically with HCP coefficients.
- [inference] On white-surface curvature, saturation or blow-up near fundi is plausible geometrically. In a fused sulcus the white surface in the fundus is broad, so H_white there is small, while the bank thickness is set by the neighbour. The covariate may therefore lose predictive power precisely in the target region. Check the open-column scatter for curvature leverage. The fused banks may lie inside the open-column H range (interpolation) or outside it (extrapolation).

### Gaps
- No source compares the thickness–curvature relation computed on white vs pial vs midthickness curvature.
- No source tests saturation at extreme convexity explicitly.

## Q5. Thickness definitions and their effect on the curvature relation

### Takeaway
Most curvature–thickness studies use FreeSurfer symmetric closest-point thickness. That definition, like any closest-point definition, interacts with curvature: in curved regions the nearest point on the opposite surface is not the matching column end. Laplace / streamline / data-steered column lengths differ from it most at crowns and fundi. I could not retrieve a quantitative definition comparison in this session (Lerch & Evans 2005; Jones 2000).

### Cited Findings
- [verified] Demirci & Holland 2022 use the FreeSurfer average of white→pial and pial→white nearest distances. — [PMC8933257](https://pmc.ncbi.nlm.nih.gov/articles/PMC8933257)
- [verified] Wagstyl 2015/2016 use the white→pial shortest distance (one-sided closest point). — [PMC4401442](https://pmc.ncbi.nlm.nih.gov/articles/PMC4401442), [PMC4872401](https://pmc.ncbi.nlm.nih.gov/articles/PMC4872401/)
- [unverified] Lerch's 2005 thesis / Lerch & Evans 2005 NeuroImage compared six thickness metrics in a population simulation (50 subjects, artificial thinning in half). t-link had the highest precision and 93% sensitivity. — [BIC announcement](https://mailman.bic.mni.mcgill.ca/pipermail/bic-announce/2005-April/000073.html)
- [unverified] Equidistant layering misplaces layers at crowns and fundi, and equivolume (Bok; Waehnert et al. 2014 NeuroImage, doi 10.1016/j.neuroimage.2013.03.078) differs most there. — [BrainVoyager guide](https://download.brainvoyager.com/bv/doc/UsersGuide/HighResDataAnalysis/EquiVolumeDepthModelling.html), [nighres docs](https://nighres.readthedocs.io/en/latest/laminar/volumetric_layering.html)
- [verified] Demirci 2026 discusses whether cortex obeys Bok's equi-volume principle as an open question. It says deviations would imply real gyral/sulcal differences in circuitry. — [PMC12836397](https://pmc.ncbi.nlm.nih.gov/articles/PMC12836397/)
- [unverified, memory] Jones, Buchbinder & Aharon 2000 (Hum Brain Mapp) introduced Laplace thickness (streamline length of the potential field between the inner and outer boundary). It is unique and symmetric, and avoids the closest-point ambiguity in curved regions.

### Inferences
- [inference] The project's thickness (length along data-steered columns) is closest to Laplace/streamline thickness. Published slopes from closest-point FreeSurfer thickness may overstate or understate the crown/fundus contrast relative to column length. Use published work for the *sign* and the *regional scaling*, not for the magnitude.
- [inference] Equivolume per unit white-surface area not being conserved fits Demirci 2026 treating Bok's principle as unsettled, and Wagstyl 2016's laminar-redistribution argument (supragranular share differs between crown and fundus). Total thickness can follow curvature while the relation is not a pure area-ratio rule.

### Gaps
- No quantitative comparison of thickness definitions as a function of curvature was retrieved (Lerch & Evans 2005 full text; Jones 2000; Aganj, Das 2009 DiReCT vs FreeSurfer).
- No study reports how the thickness–curvature slope changes with the thickness definition.
