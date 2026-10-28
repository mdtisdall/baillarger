# Seismic flattening and relative geologic time (RGT) volumes: formulations, and how they handle faults, unconformities and layer terminations

Scope note: the coordinator updated the scope partway through. Cortex has no true faults, so fault methods get only short coverage. More depth goes on (1) lateral facies change at constant RGT, (2) pinch-outs and thinning to zero thickness, (3) fault likelihood or fit misfit used as a QC check, and (4) methods where the top surface is unknown.

Verification legend:
- **[V-full]**: checked against the primary full text.
- **[V-meta]**: citation metadata (title, authors, venue, year, volume, pages, DOI) checked on Crossref; content not read in full text.
- **[V-abs]**: abstract read.
- **[MEM]**: from memory and not checked in this session. Treat as unverified.

Verified citation list, from Crossref metadata ([Crossref API](https://api.crossref.org/works)):
- Lomask, Guitton, Fomel, Claerbout, Valenciano (2006). Flattening without picking. *Geophysics* 71(4), P13–P20. https://doi.org/10.1190/1.2210848 [V-meta]. SEP report version read in full: Lomask, Guitton, Fomel, Claerbout (2005). "Update on flattening without picking." SEP-120, 137–159. https://sepwww.stanford.edu/data/media/public/docs/sep120/jesse1.pdf [V-full]
- Lomask (2003). Flattening 3-D seismic cubes without picking. SEG Expanded Abstracts, 1402–1405. https://doi.org/10.1190/1.1817552 [V-meta]
- Stark (2003). Unwrapping instantaneous phase to generate a relative geologic time volume. SEG Expanded Abstracts, 1707–1710. https://doi.org/10.1190/1.1844072 [V-meta]
- Stark (2004). Relative geologic time (age) volumes—Relating every seismic sample to a geologically reasonable horizon. *The Leading Edge* 23(9), 928–932. https://doi.org/10.1190/1.1803505 [V-meta]
- Stark (2006). Visualization of spectral decomposition implied bed thickness as a function of relative geologic time by utilizing age and seismic-Wheeler volumes. SEG Expanded Abstracts, 993–997. https://doi.org/10.1190/1.2370423 [V-meta]
- Fomel (2002). Applications of plane-wave destruction filters. *Geophysics* 67(6), 1946–1960. https://doi.org/10.1190/1.1527095 [V-meta]
- Fomel (2010). Predictive painting of 3D seismic volumes. *Geophysics* 75(4), A25–A30. https://doi.org/10.1190/1.3453847 [V-meta]
- Fomel & Liu (2010). Seislet transform and seislet frame. *Geophysics* 75(3), V25–V38. https://doi.org/10.1190/1.3380591 [V-meta]
- Wu & Zhong (2012). Generating a relative geologic time volume by 3D graph-cut phase unwrapping method with horizon and unconformity constraints. *Geophysics* 77(4), O21–O34. https://doi.org/10.1190/geo2011-0351.1 [V-meta]
- Hale (2013). Methods to compute fault images, extract fault surfaces, and estimate fault throws from 3D seismic images. *Geophysics* 78(2), O33–O43. https://doi.org/10.1190/geo2012-0331.1 [V-meta]
- Luo & Hale (2013). Unfaulting and unfolding 3D seismic images. *Geophysics* 78(4), O45–O56. https://doi.org/10.1190/geo2012-0350.1 [V-meta]
- Wu & Hale (2015a). Horizon volumes with interpreted constraints. *Geophysics* 80(2), IM21–IM33. https://doi.org/10.1190/geo2014-0212.1 [V-meta]
- Wu & Hale (2015b). 3D seismic image processing for unconformities. *Geophysics* 80(2), IM35–IM44. https://doi.org/10.1190/geo2014-0323.1 [V-meta]
- Wu, Luo, Hale (2016). Moving faults while unfaulting 3D seismic images. *Geophysics* 81(2), IM25–IM33. https://doi.org/10.1190/geo2015-0381.1 [V-meta]
- Wu & Hale (2016). 3D seismic image processing for faults. *Geophysics* 81(2), IM1–IM11. https://doi.org/10.1190/geo2015-0380.1 [V-meta]
- Wu & Hale (2016). Automatically interpreting all faults, unconformities, and horizons from 3D seismic images. *Interpretation* 4(2), T227–T237. https://doi.org/10.1190/int-2015-0160.1 [V-meta]
- Wu & Fomel (2018). Least-squares horizons with local slopes and multigrid correlations. *Geophysics* 83(4), IM29–IM40. https://doi.org/10.1190/geo2017-0830.1 [V-meta, V-abs]
- Geng, Wu, Shi, Fomel (2020). Deep learning for relative geologic time and seismic horizons. *Geophysics* 85(4), WA87–WA100. https://doi.org/10.1190/geo2019-0252.1 [V-meta, V-abs]
- Bi, Wu, Geng, Li (2021). Deep relative geologic time: a deep learning method for simultaneously interpreting 3-D seismic horizons and faults. *JGR Solid Earth* 126(9). https://doi.org/10.1029/2021JB021882 [V-meta]
- Qayyum, de Groot, Hemstra (2012). Using 3D Wheeler diagrams in seismic interpretation – the HorizonCube method. *First Break* 30(3). https://doi.org/10.3997/1365-2397.30.3.56681 [V-meta]
- Yan & Wu (2021). Seismic horizon extraction with dynamic programming. *Geophysics* 86(2), IM51–IM62. https://doi.org/10.1190/geo2020-0039.1 [V-meta]
- Dou, Wu, Gao, Bi (2026). Learning stratigraphically consistent relative geologic time from 3D seismic data via sinusoidal mapping. arXiv:2605.01273. https://arxiv.org/abs/2605.01273 [V-abs]
- Wu, X. (2016). *3D Seismic Image Processing for Interpretation*. PhD thesis, Colorado School of Mines. https://repository.mines.edu/entities/publication/445cb01e-6283-4cc2-b38c-211302629209. The record was found, but the PDF returned HTTP 403.
- "Parks 2010" on flattening: **not found**. Crossref returned nothing that matched. The citation may be wrong. Do not cite it.

---

## Q1. Lomask et al. 2006, "Flattening without picking": formulation, faults, limitations

### Takeaway
Flattening is a 2D Poisson/least-squares integration of local dips into a vertical time-shift field τ, with the "dip evaluated at the unknown τ" nonlinearity handled by Gauss–Newton and an FFT Laplacian inverse. Faults are handled only by binary zero-weights on the residual. The weights work only when the fault tip-line is inside the volume, so dips can be integrated around the fault. Unconformities and pinch-outs set continuity (ε) against flatness. The method moves samples only vertically and keeps one τ per (x, y, t).

### Cited findings (all from the SEP-120 full text [V-full], https://sepwww.stanford.edu/data/media/public/docs/sep120/jesse1.pdf; journal version https://doi.org/10.1190/1.2210848)
- Inputs: two dip components p_x and p_y at every sample, from plane-wave destruction (Claerbout 1992; Fomel 2002, "we primarily use the latter").
- Unknown: a time- or depth-shift field τ(x, y, t) "such that its gradient approximates the dip p(x, y, τ)". The traces are shifted vertically to match a reference trace held constant.
- Objective (eq. 1): J(τ) = ∫∫ [(p_x(τ,x,y) − ∂τ/∂x)² + (p_y(τ,x,y) − ∂τ/∂y)²] dx dy.
- Euler–Lagrange (eq. 2): ∂²τ/∂x² + ∂²τ/∂y² = ∂p_x/∂x + ∂p_y/∂y + ½ ∂(p_x² + p_y²)/∂τ. The last term is dropped, which gives the Poisson equation ∇ᵀ∇τ = ∇ᵀp, the normal equations of the regression ∇τ ≈ p (eqs. 3–5).
- Nonlinearity: p is evaluated at the unknown τ. Gauss–Newton iteration (eqs. 6–8): r = ∇τ_k − p(τ_k); Δτ = (∇ᵀ∇)⁻¹∇ᵀr; τ_{k+1} = τ_k + Δτ (the sign is as printed).
- Solver: the Laplacian inverse is applied in the Fourier domain, Δτ ≈ FFT2D⁻¹[FFT2D[∇ᵀr] / (−Z_x⁻¹ − Z_y⁻¹ + 4 − Z_x − Z_y)], one forward/inverse FFT per iteration. The paper notes this is "much like" Ghiglia & Romero (1994) phase unwrapping. Each time slice is solved independently.
- Vertical regularization (eq. 10): ∇_ε τ = [∂τ/∂x, ∂τ/∂y, ε ∂τ/∂z]ᵀ ≈ [p_x, p_y, 0]ᵀ. ε controls vertical smoothness of the shifts. It needs a full 3D solve with 3D FFTs. The stated purpose is "to insure a monotonic and continuous result". Smoothing the input dips in time is suggested first.
- Faults (eqs. 11–12): weighted regression W∇τ ≈ Wp, residual r = W[∇τ_k − p]. The FFT preconditioner stays unweighted (it is "not approximating the inverse as well"), so more iterations are needed, up to about 100 with weights and depth-varying structure. In the synthetic, W was binary: 0 at the fault "and a few samples on each side", 1 elsewhere. In field data, W was a "crude binary weight cube" made by manual picking. The authors say an automatic fault detector "can also, in principle" make it.
- Fault limitation: success needs the fault to be "limited laterally". The tip-line must be inside the cube, "enabling dips to be summed around the fault". Faults with significant displacement "contaminate the dip estimation".
- Unconformity / pinch-out: on the Elf North Sea angular unconformity, ε = 1.0 gave a result that was not completely flat, and ε = 0 would be flatter "but would lose its continuity". "The trade-off between continuity and flatness emerges in cases of pinch-outs and unconformities." For picking horizons, ε = 0 is better: "horizons that lead to the angular unconformity are tracked". The horizons "track up to and along the unconformity".
- Other limitations (Discussion): shifts act only vertically, so only vertical faults and beds that are not overturned can be properly restored. True stratigraphic thickness is lost and replaced by the vertical thickness at the reference trace.
- Amplitude variation degrades the dip estimates, but its effect is "less significant because this method can flatten the entire data cube at once, globally, in a least-squares sense".
- Horizons are obtained by "unflattening" flat surfaces with the τ field (the inverse map).

### Inferences
- The flattening map is a horizon-volume representation: for each output (x, y, t₀) it stores the shift to the input depth. RGT is the inverse function, so the method is single-valued in z and requires monotonicity in t. ε is the only control that prevents fold-over.
- To port to cortex: replace "vertical" with a column or streamline direction, or use a curvilinear coordinate frame. Without this, the "vertical shift only" restriction is equivalent to assuming the depth axis is normal to the layers everywhere.

### Gaps
- I did not read the published 2006 Geophysics text. It may differ from SEP-120, for example in the Valenciano coauthor contribution. Lomask & Guitton 2006 SEP (geological constraints from a few picked reflectors) was seen in search but not read.

---

## Q2. Stark: RGT ("age") volumes, phase unwrapping, Wheeler volumes, unconformities

### Takeaway
Stark defined the RGT volume as a scalar field on the seismic grid whose isosurfaces are horizons. The values are only ordinal until calibrated. It is produced by interpolating between picked horizons or by 3D unwrapping of instantaneous phase. A "seismic-Wheeler volume" is the data resampled into (x, y, RGT). I could not confirm, from a source read in this session, how unconformities are represented in Stark's own papers.

### Cited findings
- Definition: an RGT volume has the same dimensions as the seismic volume, but each sample "contains an estimate of relative geologic time". You know whether sample A is older or younger than B, but not by how much "until a calibration step is performed". A surface of constant value is a seismic horizon — [AAPG 2004 abstract, Stark](https://www.searchanddiscovery.com/abstracts/html/2004/annual/abstracts/Stark.htm)
- Generation: (a) interpret horizons, assign each a constant approximating its age, and interpolate between them; (b) unwrap the instantaneous phase (Stark 2003 SEG) — [same](https://www.searchanddiscovery.com/abstracts/html/2004/annual/abstracts/Stark.htm); [Stark 2003, doi:10.1190/1.1844072](https://doi.org/10.1190/1.1844072) [V-meta]
- Constant-RGT amplitude slices are compared to Posamentier's "proportional slices" and Zeng's "stratal slices" — [AAPG 2004 abstract](https://www.searchanddiscovery.com/abstracts/html/2004/annual/abstracts/Stark.htm)
- Age volumes together with seismic-Wheeler volumes were used to show spectral-decomposition-implied bed thickness as a function of RGT — [Stark 2006, doi:10.1190/1.2370423](https://doi.org/10.1190/1.2370423) [V-meta]
- Lomask et al. describe Stark (2004) as a full-volume approach to the same goal by "unwrapping instantaneous phase" — [SEP-120](https://sepwww.stanford.edu/data/media/public/docs/sep120/jesse1.pdf)
- Wu & Zhong (2012) extended phase-unwrapping RGT with a 3D graph-cut unwrapper "with horizon and unconformity constraints" — [doi:10.1190/geo2011-0351.1](https://doi.org/10.1190/geo2011-0351.1) [V-meta, title only]
- A 2006 Oil IT Journal item reports that Stark Research holds a US patent covering the Wheeler volume — [oilit.com](https://oilit.com/HTML_Articles/2006_4_8.php). Several Stark-related patents also exist, e.g. US 6853922 "System for information extraction from geologic time volumes" — [Google Patents](https://patents.google.com/patent/US6853922)

### Inferences
- Phase unwrapping ties T to the waveform phase, so one cycle of phase is one "layer". This is the most fragile approach under lateral facies change (polarity flips, tuning). If the stria changes character laterally, a phase-based T would slip cycles.
- [MEM] My recollection is that in Stark's age volumes, an unconformity gives a jump (a discontinuity) in age across the surface, and a Wheeler volume shows the hiatus as a blank (no-data) region. Not verified here.

### Gaps
- The Leading Edge 2004 full text was not read. The exact unwrapping algorithm and the treatment of unconformities and faults in Stark's own work are unverified.

---

## Q3. Fomel: plane-wave destruction (PWD), predictive painting, seislets

### Takeaway
PWD gives local slopes. Predictive painting spreads a value (for example the time coordinate of a reference trace) along those slopes by recursive trace-to-trace prediction, which yields an RGT volume directly. It is fast and accurate in continuous structure. It "has difficulty spreading information across faults with significant displacement", and later work adds fault-slip estimates.

### Cited findings
- Predictive painting: slopes estimated (inline and crossline in 3D) by plane-wave destruction, then a seed trace's information is spread through the volume. The stated uses are automatic horizon picking and flattening — [CSEG Lunchbox talk, Fomel 2009](https://old.cseg.ca/technical/view/predictive-painting-of-3d-seismic-volumes)
- Known limitation: predictive painting "recursively spreads information from a seed trace to its neighbors by following the local dips". "However, it has difficulty spreading information across faults with significant displacement". A later fix uses fault-slip estimates from local similarity scanning — [TCCS Newsletter Spring 2017](https://tccs.beg.utexas.edu/files/tccs/2056/TCCS%20Newsletter%20Spring_2017.pdf)
- Papers: Fomel 2002 PWD, [doi:10.1190/1.1527095](https://doi.org/10.1190/1.1527095); Fomel 2010 painting, [doi:10.1190/1.3453847](https://doi.org/10.1190/1.3453847); seislet, [doi:10.1190/1.3380591](https://doi.org/10.1190/1.3380591) [V-meta]

### Formulation [MEM, unverified, check against Fomel 2002/2010 before porting]
- PWD: for a local plane wave with slope σ, the 2D filter is C(σ) ≈ 1 − Z_x · P(σ), where P(σ) is an all-pass fractional time-shift (Thiran / maximally flat FIR approximation of e^{iωσ}). σ(x, t) is estimated by minimizing ‖C(σ)d‖² with shaping or smoothness regularization, using Gauss–Newton on the linearized filter.
- Painting: P_{k−1→k} predicts trace k from trace k−1 with the local slopes. Recursive application from a reference trace r gives s_k = P_{k−1→k}…P_{r→r+1} s_r. To get RGT, the trace "painted" is the time axis itself, τ(t) = t at the reference trace. The result is a volume whose isosurfaces follow the slopes. A least-squares form solves D τ ≈ 0 with τ fixed on the reference trace (D = the PWD operator), which spreads the reference in a regularized way. In 3D the painting runs inline and crossline. Results from several reference traces can be combined to reduce error accumulation.
- Seislet: a wavelet-like transform whose predict/update lifting steps use PWD predictions along the local slope. Coefficients are sparse for locally planar events. It is a compression and denoising tool, not an RGT solver.

### Inferences
- Painting is a hyperbolic (transport/"marching") formulation: information flows along the slope field from a seed. Lomask's is elliptic (global Poisson). The analogy for cortex: painting is like streamline integration from a seed, with errors that accumulate along the path. The global least-squares form averages errors.

### Gaps
- The Fomel 2010 full text was not read (the Madagascar paper URLs tried returned the wiki main page or 404). The exact equations above come from memory.

---

## Q4. Wu & Hale family: horizon volumes, constraints, unconformities, unfaulting; Wu & Fomel 2018

### Takeaway
Wu & Hale make the horizon volume (or its inverse, the RGT volume) the unknown. It is fitted to structure-tensor slopes in weighted least squares with interpreter control points as hard constraints, solved by preconditioned CG. Faults are removed first by unfaulting with interpolated throw vectors. Unconformities are detected (unconformity likelihood) and used as constraints, so that horizons converge onto them. Wu & Fomel 2018 add multi-grid DTW correlations so that horizons can jump discontinuities and stay phase-consistent.

### Cited findings
- Wu & Hale 2015a computes "a volume of horizons that honor interpreted constraints", rather than tracking horizons one at a time. Control points are incorporated "with simple constraint preconditioners in the conjugate gradient method" — [Wu 2016 thesis abstract, Mines repository, via search snippet](https://repository.mines.edu/entities/publication/445cb01e-6283-4cc2-b38c-211302629209); [doi:10.1190/geo2014-0212.1](https://doi.org/10.1190/geo2014-0212.1)
- Unfaulting: "unfault a seismic image using a field of vector shifts obtained by interpolating estimated throw vectors between faults". Increased continuity "facilitates unfolding of 3D seismic images so that reflectors are horizontal". "The composite process of unfaulting and unfolding is equivalent to the construction of an entire 3D volume of chronostratigraphic horizons" — [Wu 2016 thesis abstract](https://repository.mines.edu/entities/publication/445cb01e-6283-4cc2-b38c-211302629209); also [Luo & Hale 2013](https://doi.org/10.1190/geo2012-0350.1), [Wu, Luo & Hale 2016](https://doi.org/10.1190/geo2015-0381.1)
- The thesis gives a method for "a unconformity likelihood image that highlights both the termination areas and the corresponding parallel unconformities and correlative conformities". It also provides an unfaulted image "with continuous reflectors across faults and unconformities as constraints for an automatic flattening method" — [Wu 2016 thesis, via search snippets](https://repository.mines.edu/entities/publication/445cb01e-6283-4cc2-b38c-211302629209); paper: [Wu & Hale 2015b](https://doi.org/10.1190/geo2014-0323.1)
- A related EAGE abstract describes horizons as isosurfaces of a volume, with faults and unconformities treated as internal boundaries — [EarthDoc 10.3997/2214-4609.201412612, via search snippet](https://www.earthdoc.org/content/papers/10.3997/2214-4609.201412612). Only the snippet was read.
- Wu & Fomel 2018 (abstract): slope-based methods "often fail to correctly track horizons across discontinuities such as faults and noise because the local slopes can only correctly follow laterally continuous reflections". Amplitude and phase are not used. Fix: estimate local slopes with structure tensors, compute "multigrid slopes" by DTW between traces on laterally coarse grids, and fit the horizon's slopes in least squares to both. Fine and coarse slopes agree where there are no lateral discontinuities. Across discontinuities "the coarse-grid slopes will help to find the corresponding reflections", and the method also yields "phase-consistent horizons" — [doi:10.1190/geo2017-0830.1, Semantic Scholar abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1190/geo2017-0830.1?fields=title,year,abstract)
- Yan & Wu 2021 use dynamic programming for horizon extraction — [doi:10.1190/geo2020-0039.1](https://doi.org/10.1190/geo2020-0039.1) [V-meta]

### Formulation details [MEM, unverified; check against doi:10.1190/geo2014-0212.1]
- Unknown: shift r(x, y, τ) so that the horizon with RGT τ sits at z = τ + r(x, y, τ) (horizon volume). The RGT volume is the inverse map z → τ.
- Data equations (weighted): w(x, y, z)[∂r/∂x − p_x(x, y, τ + r)] ≈ 0 and w[∂r/∂y − p_y(...)] ≈ 0. Slopes p_x = −u_x/u_z and p_y = −u_y/u_z come from the eigenvector u (normal) of the smoothed gradient structure tensor. Weight w = a planarity/linearity measure from the eigenvalues, e.g. (λ_u − λ_v)/λ_u.
- Regularization: ε ∂r/∂τ ≈ 0 (vertical smoothness), the same role as Lomask's ε.
- Nonlinearity (p evaluated at τ + r) handled by outer fixed-point iterations. The inner solve is CG.
- Control points (x_i, y_i, z_i) on one horizon: equality constraints r(x_i, y_i, τ_c) = z_i − τ_c. They are enforced by a preconditioner that makes the constrained shifts identical, so CG only explores feasible solutions, plus smoothing preconditioners.
- Unconformities: picked or extracted unconformity surfaces are added as constraints. Horizons above and below converge to the unconformity, so in the horizon volume a range of τ values maps to the same z (∂z/∂τ = 0). In the RGT volume this shows as a jump in τ across the unconformity surface.
- Hale 2013 fault likelihood: semblance s is computed in thin windows scanned over fault strike and dip. Fault likelihood f = 1 − s⁸ (the power sharpens the contrast). Maxima over orientations give fault likelihood, strike and dip images. These are thinned and linked into quad-mesh fault surfaces. Throws are estimated by dynamic warping of samples on opposite sides of each fault surface. Unfaulting then applies an interpolated vector shift field.

### Inferences
- The horizon-volume form (unknown z(τ)) represents zero thickness directly: dz/dτ = 0 over a τ interval is a pinch-out or hiatus, with no singularity. The RGT form (unknown τ(z)) needs a discontinuity there. For cortex, solving for depth-as-a-function-of-T (a "horizon volume") is the more natural way to allow a layer to thin to zero. The RGT form T(x) with a smooth regularizer will smear a hiatus.

### Gaps
- The full texts of Wu & Hale 2015a/b, Hale 2013 and Luo & Hale 2013 were not obtained (the Mines thesis PDF returned 403, and Hale's papers site failed with a certificate error). The equations above are memory-level only.

---

## Q5. Wheeler domain and HorizonCube: lateral facies change, hiatuses, pinch-outs

### Takeaway
The Wheeler domain resamples the data at constant RGT. Each horizon becomes a flat time line, and lateral facies change shows as along-row variation in amplitude or character. A hiatus or pinch-out shows as a gap (no data) or as converged horizons. HorizonCube tracks a dense set of horizons on a dip field, with an explicit "truncated" mode that lets horizons stop when they converge. This mode is the most direct seismic analogue of laminar terminations.

### Cited findings
- HorizonCube: a dense stack of correlated 3D stratigraphic surfaces. The primary input is a dip field. Horizons can be tracked as truncated (they stop when they get too close together) or continuous (they stay together when they converge). The truncated mode helps identify onlaps, downlaps and toplaps — [dGB/geoexpro and dGB material, via search results](https://dgbes.com/blog/wheeler-diagram-unraveling-the-depositional-system-from-2d-seismic-data/); paper [Qayyum, de Groot, Hemstra 2012](https://doi.org/10.3997/1365-2397.30.3.56681) [V-meta]
- Data-driven vs model-driven: horizons are auto-tracked on the dip volume. In complex settings, a model-driven proportional interpolation between framework horizons (modes: proportional, parallel-to-upper, parallel-to-lower) is preferred — [dGB blog](https://dgbes.com/blog/wheeler-diagram-unraveling-the-depositional-system-from-2d-seismic-data/)
- In the Wheeler domain the data is flattened along chronostratigraphic horizons "that honor truncations and depositional hiatuses". Every horizon is flat with regular spacing, which reveals hiatuses and condensed sections — [dGB blog](https://dgbes.com/blog/wheeler-diagram-unraveling-the-depositional-system-from-2d-seismic-data/)
- Caveat: an independent Norwegian MSc thesis found the 3D HorizonCube-derived WheelerCube had "low resolution and inaccurate outcome". It found tracking poor and slow in 3D, but good in 2D — [UiS Brage](https://uis.brage.unit.no/uis-xmlui/handle/11250/183565)
- Most HorizonCube sources are vendor material — (same search set; see the caveat).

### Lateral facies change at constant RGT: what the methods do
- Slope-based methods (Lomask, painting, Wu & Hale) use only the local orientation of the reflections, not amplitude. A horizon whose amplitude or character changes laterally keeps a continuous T as long as the orientation estimate stays coherent. Lomask: amplitude variation affects the dip quality, but the global least-squares fit minimizes "the effect of questionable dip information" — [SEP-120](https://sepwww.stanford.edu/data/media/public/docs/sep120/jesse1.pdf)
- Phase- or correlation-based terms (Stark unwrapping; the Wu & Fomel 2018 DTW multigrid term, which is designed to give "phase-consistent horizons") assume the waveform is similar laterally — [Wu & Fomel 2018 abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1190/geo2017-0830.1?fields=title,year,abstract). Inference: under real facies change these terms can pull T onto the wrong cycle, so for cortex they should be down-weighted, or used only where the local similarity is high.
- In the Wheeler domain, facies change is the signal of interest: a stratal slice at constant T is read laterally (Zeng / Posamentier stratal slices) — [Stark AAPG abstract](https://www.searchanddiscovery.com/abstracts/html/2004/annual/abstracts/Stark.htm). For cortex, a stria that fades at the V1/V2 border with continuous T is a facies change. It is not a fault.

### Inferences
- How "missing time" can be represented, by formulation:
  1. **Jump in T** across the unconformity or termination surface (RGT form; Stark-style age volumes [MEM]; Wu & Hale constrained RGT [MEM]).
  2. **Zero thickness** in the horizon volume, dz/dτ = 0 (Wu & Hale horizon volumes; HorizonCube "continuous" mode, where horizons converge and stay together).
  3. **Blank region** in the Wheeler domain (hiatus shown as no data) [MEM, standard Wheeler diagram convention].
  4. **Truncated horizons**: individual isosurfaces end at the convergence point (HorizonCube truncated mode, verified above).
- Smooth least-squares RGT (Lomask with ε > 0) gives none of these cleanly. It smooths the termination into a region of compressed T gradients. This is the "continuity vs flatness" trade-off that Lomask names.

### Gaps
- No primary source read here gives the HorizonCube tracking equations or the exact "too close" threshold rule.

---

## Q6. Pinch-outs and thinning to zero: how the RGT field behaves

### Takeaway
At a pinch-out, the horizons converge, so |∇T| becomes very large (the RGT form) or ∂z/∂T → 0 (the horizon-volume form). Least-squares slope fits with vertical smoothing (ε) resist this and create artifacts. The documented fixes are ε tuning (Lomask), explicit unconformity constraints (Wu & Hale; Wu & Zhong), or allowing truncation (HorizonCube).

### Cited findings
- Lomask: the trade-off between continuity and flatness "emerges in cases of pinch-outs and unconformities". A smaller ε gives a flatter result with artifacts. A larger ε keeps continuity but does not flatten fully — [SEP-120](https://sepwww.stanford.edu/data/media/public/docs/sep120/jesse1.pdf)
- HorizonCube truncated vs continuous convergence modes — [dGB](https://dgbes.com/blog/wheeler-diagram-unraveling-the-depositional-system-from-2d-seismic-data/)
- Unconformity likelihood highlights "termination areas and the corresponding parallel unconformities and correlative conformities" — [Wu thesis abstract](https://repository.mines.edu/entities/publication/445cb01e-6283-4cc2-b38c-211302629209)
- The Dou et al. 2026 abstract calls RGT "a topologically constrained continuous field" where errors spread globally. It says pixel-wise regression "struggle[s] to capture thin horizons" and yields "unstable ordering". Its tests include "large unconformities" and clinoforms — [arXiv:2605.01273](https://arxiv.org/abs/2605.01273)

### Inferences
- A cortical analogue: if a sublayer vanishes laterally (a layer termination at an areal border), the method must allow T levels to converge to zero thickness. A practical choice: solve for z(x, T) (horizon volume) with a non-negativity constraint ∂z/∂T ≥ 0, which keeps monotonic ordering and allows equality (zero thickness). Do not put a smoothness penalty on ∂T/∂n.

### Gaps
- I found no source that gives a closed-form treatment of the singular |∇T| at pinch-outs in the slope-fitting methods.

---

## Q7. Fault likelihood or misfit as a QC/diagnostic

### Takeaway
No source read here proposes residual-based QC of a continuous RGT fit as a method in itself. The building blocks are present: the weighted residual r = W(∇τ − p) (Lomask), the structure-tensor planarity weights, and Hale's semblance-based fault likelihood f = 1 − s⁸ [MEM]. Any of them, computed on the original image or on the flattened image, can show where a continuous T is inconsistent with the data.

### Cited findings
- Lomask's weighting removes equations "affected by the bad dips estimated at the faults". The weight can "in principle" be made from an automatic fault detector — [SEP-120](https://sepwww.stanford.edu/data/media/public/docs/sep120/jesse1.pdf)
- After flattening, the cube "is very flat except in the area of the fault itself", and "within the salt, the horizons gradually get off their respective reflector events as the estimated dip becomes very inaccurate" — [SEP-120](https://sepwww.stanford.edu/data/media/public/docs/sep120/jesse1.pdf). Inference: the flatness of the flattened image is a direct QC.
- Hale 2013 fault images, surfaces and throws — [doi:10.1190/geo2012-0331.1](https://doi.org/10.1190/geo2012-0331.1) [V-meta]

### Inferences
- QC recipes for cortex:
  1. Map |W(∇T − p)| (or the angle between ∇T and the structure-tensor normal) after the fit. High values that are coherent along a surface show a missed termination or a segmentation error.
  2. Run a fault-likelihood scan on the flattened (T-domain) image. A real "fault" would mean a laminar depth jump, which the user says is an error, for example a misregistration or a tissue tear.
  3. Plot the thickness ∂z/∂T. Near-zero bands show pinch-outs. Negative values show fold-over, which means a regularization failure.

### Gaps
- No published RGT-misfit QC method was found in the time available.

---

## Q8. Unknown top surface; eikonal / geodesic links

### Takeaway
None of the flattening/RGT methods read here requires a known top surface. RGT is relative and defined up to calibration, and flattening references a single trace or datum. Unconformities (erosional tops) are found from the data (unconformity likelihood) rather than given. No method read here is eikonal (|∇T| = f). They are all gradient-direction fits (∇T ∥ normal, or ∇_xy τ = p). Painting is a transport/marching analogue.

### Cited findings
- RGT values give order only until calibrated — [Stark AAPG abstract](https://www.searchanddiscovery.com/abstracts/html/2004/annual/abstracts/Stark.htm)
- Lomask: shifts are applied relative to a reference trace held constant; nothing needs a top boundary — [SEP-120](https://sepwww.stanford.edu/data/media/public/docs/sep120/jesse1.pdf)
- Unconformity surfaces are extracted from an unconformity-likelihood image — [Wu thesis abstract](https://repository.mines.edu/entities/publication/445cb01e-6283-4cc2-b38c-211302629209)

### Inferences
- For cortex with an unknown pial surface: solve the slope fit with only the WM boundary as a Dirichlet condition (T = 0) and leave the top free (natural boundary). Then normalize T, or find the pial surface jointly as the outermost "unconformity", which is an erosional top analogue. The seismic methods normalize T only after the fit, by calibration.
- The Lomask Poisson structure (∇ᵀ∇τ = ∇ᵀp) is related to Laplace-equation cortical thickness methods (Jones et al. 2000) but is not the same. Lomask fits the gradient to a data direction, where the Laplace methods impose harmonicity between two known boundaries.
- [MEM] Implicit structural modeling (Mallet's GeoChron / Paradigm; Caumon et al.) represents stratigraphy as a scalar field with discontinuity "jump" functions at faults and unconformities. It could be a useful related literature. Not verified here; one patent title seen in search: "Geologic stratigraphy via implicit and jump functions" (US 11175434) — [USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11175434)

### Gaps
- I found no seismic RGT method in the form |∇T| = f, and none that uses geodesic distance or fast marching. Searches in the time available did not cover this specifically. Treat it as "not found", not "does not exist".

---

## Q9. Deep-learning RGT (brief) and typical failure modes and fixes

### Takeaway
CNN regression of RGT (Geng et al. 2020; Bi et al. 2021) is trained on synthetic seismic/RGT pairs and handles crossing faults and complex folds better than the slope-based methods. Newer work (2026) targets ordering and thin-layer failures with a sinusoidal-space loss. The failure modes of the classic methods are faults with large throw, error build-up along integration paths, cycle skips, fold-over or non-monotonic T, and the smoothing of terminations.

### Cited findings
- Geng et al. 2020: an encoder–decoder–refinement CNN trained on 2080 synthetic pairs and tested on 960. It generalizes to field data and does better than conventional methods "especially in dealing with complex structures such as crossing faults and complicatedly folded horizons". Conventional methods use "only local image features", which makes discontinuities (faults, unconformities) hard — [abstract, doi:10.1190/geo2019-0252.1](https://doi.org/10.1190/geo2019-0252.1)
- Bi et al. 2021 interpret 3D horizons and faults together — [doi:10.1029/2021JB021882](https://doi.org/10.1029/2021JB021882) [V-meta]
- Dou et al. 2026: sinusoidal mapping encodes the "periodic stratigraphic semantics" of RGT, with optional sparse-horizon priors — [arXiv:2605.01273](https://arxiv.org/abs/2605.01273)
- Failure modes and fixes, with sources:
  - Large-throw faults break dip integration → weights plus a tip-line inside the volume (Lomask); unfaulting first (Luo & Hale; Wu et al.); multigrid DTW correlations (Wu & Fomel 2018); fault-slip-guided painting (TCCS 2017).
  - Fold-over / non-monotonic T → smooth the dips in time, or ε ∂τ/∂z regularization (Lomask).
  - Noisy or inaccurate dips (salt) → horizons drift (Lomask). Planarity weights (Wu & Hale [MEM]).
  - Cycle skips / non-phase-consistent horizons → DTW correlation term (Wu & Fomel 2018).
  - Pinch-outs and unconformities → the continuity vs flatness trade-off (Lomask); unconformity constraints (Wu & Hale 2015b; Wu & Zhong 2012); truncated tracking (HorizonCube).
  - Human-known horizons → hard control-point constraints by CG preconditioning (Wu & Hale 2015a).

### Inferences
- DL RGT nets learn the training-set priors (synthetic folds and faults). Without new synthetics, a port to cortex would need laminar-specific training data, such as simulated cortical profiles with areal borders.

### Gaps
- I did not review the DL RGT literature after 2021 beyond the 2026 abstract.
