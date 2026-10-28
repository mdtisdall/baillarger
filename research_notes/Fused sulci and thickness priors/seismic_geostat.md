# Seismic / geostatistical equivalents: lateral thickness propagation with uncertainty, and splitting a gap between two stacks of layers

Scope note: a researcher's notes. Citation status uses three tags: **[verified]** (title, authors, venue and year checked against a primary or publisher or repository record found in this session), **[partly verified]** (some fields checked, others not), **[unverified]** (from memory, not checked in this session). The richest primary source found is the COHIBA user manual (Norwegian Computing Center, NR), v7.0. I downloaded it and read the relevant sections in full: https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf. It implements Abrahamsen (1993), Abrahamsen & Benth (2001) and Abrahamsen (2005) in one framework. The section numbers and equation numbers below refer to that manual.

**Headline:** a direct, published, production-software analogue of the user's "split the gap by each stack's expected thickness" exists. It is the **"multiple models"** mechanism in NR's COHIBA (Abrahamsen 2005, "Combining methods for subsurface prediction"). In COHIBA, internal surfaces between two well-resolved seismic reflectors (an "envelope") are obtained in two ways: by stacking isochores down from the top reflector, and by stacking them up from the base reflector. COHIBA then combines the two by inverse-variance (BLUE / kriging) weights. A second COHIBA option, `<scale-isochores-to-seismic-envelopes>`, rescales every isochore by the ratio (envelope thickness / sum of isochores). That is exactly a proportional split by expected thickness.

---

## 1. Isochore/isopach mapping and depth conversion: kriging, cokriging, Bayesian kriging, and how uncertainty grows away from wells

### Takeaway
In the oil industry, the standard model for surfaces and the thicknesses between them is this: every surface is a sum of interval thicknesses. Each interval thickness = (linear trend in known maps, with Bayesian-prior coefficients) + (a zero-mean Gaussian random field). Wells give near-exact local data. Seismic gives laterally dense but inexact trends. Prediction is Bayesian kriging/cokriging, and uncertainty is the kriging SD. The SD is ~0 at wells and rises toward the prior (trend + residual) SD at distances beyond the variogram range. This maps closely onto "carry thickness laterally from where the stack's top is visible, with uncertainty."

### Cited Findings
- **Abrahamsen (1993)** [verified]: "Bayesian kriging for seismic depth conversion of a multi-layer reservoir", in Soares (ed.), *Geostatistics Tróia '92*, Kluwer, pp. 385–398, DOI 10.1007/978-94-011-1739-5_31 — [NR PDF](https://projects.nr.no/sites/default/files/files/Bayesian-Multilayer-Abrahamsen-1993.pdf); bibliographic details from [COHIBA manual references](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf). One secondary listing gives 1992 / pp. 358–398, a known discrepancy ([search summary of Springer listing](https://link.springer.com/doi/10.1007/978-3-540-33236-7_3)).
  - Model (from the PDF text): depth to subsurface L is the sum of interval thicknesses, $z_L(x)=\sum_{l=1}^{L}\Delta z_l(x)$, with $\Delta z_l = v_l\,\Delta t_l$. Interval velocity $V_l(x)=\sum_p A_l^p g_l^p(x)+R^v_l(x)$, where the $A$ are multi-Gaussian prior coefficients and $R$ is a zero-mean GRF. Travel-time error is $T_l=t_l+R^t_l$. After linearisation, $Z_L(x)=\sum_l(\tilde V_l+R^v_l)\Delta t_l + R^z_L(x)$ — [Abrahamsen 1993 PDF](https://projects.nr.no/sites/default/files/files/Bayesian-Multilayer-Abrahamsen-1993.pdf).
  - The paper's stated consequence: uncertainty in all interval velocities above surface L adds to the depth uncertainty of L. Uncertainty accumulates down a stack — [Abrahamsen 1993 PDF](https://projects.nr.no/sites/default/files/files/Bayesian-Multilayer-Abrahamsen-1993.pdf).
  - The abstract states two limits. With exact prior knowledge, the method equals cokriging with 2L covariables. With no prior knowledge, it equals universal kriging combined with cokriging. The motivation is "very limited number of wells", so prior information is necessary and the method stays stable for any number of wells "including zero" — [Abrahamsen 1993 PDF](https://projects.nr.no/sites/default/files/files/Bayesian-Multilayer-Abrahamsen-1993.pdf).
- **COHIBA general model (manual §20.2)**: $Z_l(x)=z_{ref}(x)+\sum_{i\in I_l}\Delta Z_i(x)$, with $\Delta Z_i=\Delta m_i+\Delta\epsilon_i$ and trend $\Delta m_i(x)=\sum_p\beta_{i,p}f_{i,p}(x)$. The trend maps are "constants, isochore maps, or any map". The residuals are independent across intervals: $\mathrm{Cov}\{\Delta\epsilon_i(x),\Delta\epsilon_j(y)\}=\delta_{ij}\sigma_i(x)\sigma_j(y)\rho(\|x-y\|)$. So two surfaces that share intervals are correlated through the residuals they share — [COHIBA manual v7.0](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- **Bayesian trend estimation (manual eq. 20.20–20.21)**: prior $\beta\sim N(\beta_0,\Sigma_0)$; $\hat\beta=\beta_0+\Sigma_0F'(K+F\Sigma_0F')^{-1}(Z-F\beta_0)$, $\hat\Sigma=\Sigma_0-\Sigma_0F'(K+F\Sigma_0F')^{-1}F\Sigma_0$. This is "robust for any N, including N = 0". It returns GLS / universal kriging as $\Sigma_0\to\infty$ and simple kriging as $\Sigma_0\to0$ — [COHIBA manual v7.0](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- The manual lists the causes of collinearity / ill-posed trend estimation as "too few well points, clustering of wells, unobserved surfaces due to pinch outs or too many coefficients". The Bayesian prior is the remedy — [COHIBA manual v7.0 §20.3.2](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- Isochore trend input: $\Delta z(x,y)=a\cdot m(x,y)$ with a user isochore map m. "Pinch outs are possible so m(x,y) can be zero in large areas." Interval-thickness uncertainty is given as a constant or a map, plus a variogram per interval — [COHIBA manual v7.0 §1.4](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- Conditional simulation: residuals are GRFs simulated by FFT (Abrahamsen, Kvernelv & Barker 2018, ECMOR XVI, DOI 10.3997/2214-4609.201802134). The SD of many realisations "will coincide with the prediction error" — [COHIBA manual v7.0](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf); [NR COHIBA page](https://nr.no/en/COHIBA).
- An NR review note (Abrahamsen 1996, "Geostatistics for seismic depth conversion") says universal kriging beats plain cokriging for depth conversion, and Bayesian cokriging is better still for including the correlations between surfaces and interval velocities — [NR PDF](https://www3.nr.no/sites/default/files/files/Geostatistics-DC-Abrahamsen-1996.pdf) (based on the search summary; I did not read the PDF in full).
- Textbook reference used by COHIBA: Chilès & Delfiner, *Geostatistics: Modeling Spatial Uncertainty* (Wiley, 1999; 2nd ed. 2012) — cited in the [COHIBA manual](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf) [partly verified: edition details from memory].

### Inferences
- How uncertainty grows away from data. This is the standard kriging variance (general geostatistics, not quoted from one source): $\sigma^2_K(x_0)=C(0)-c_0^\top K^{-1}c_0+(\text{trend-uncertainty term})$. It is ~0 at a datum (or the pick-error variance, if there is a nugget). It rises with distance, at a rate set by the variogram range, to the prior residual variance plus the posterior trend variance. For the cortex: "where the stack's top is visible" = wells. The variogram range = the lateral propagation length. The kriging SD = the uncertainty of the carried thickness. Bayesian kriging gives a principled fallback (the prior mean thickness) where there are no visible-top observations.
- A "propagation energy" maps onto the variogram / GRF precision. A Matérn GRF is the solution of an SPDE whose precision is a polynomial in the Laplacian. So a smoothness (membrane / thin-plate) energy on the thickness field is equivalent to a GRF prior, and the posterior is the kriging solution. This is standard (Lindgren et al. 2011) [unverified in this session].

### Gaps
- I did not read the 1996 NR note, Abrahamsen (2005), or Chilès & Delfiner directly. Their content here comes from the COHIBA manual and search summaries.
- I did not look at collocated cokriging of thickness with a seismic attribute (Xu et al. 1992 style) in this session.

---

## 2. Direct analogue: two stacks / an envelope, and splitting the gap (COHIBA "multiple models", Abrahamsen 2005)

### Takeaway
COHIBA solves exactly the "ambiguous multi-layer model" problem: should a surface between two reflectors be built from above or from below? It builds both, then combines them with weights that minimise the prediction variance, which are essentially inverse-variance weights. A separate option rescales isochores proportionally to fit the envelope. The first corresponds to a Gaussian posterior for the contact. The second corresponds to the user's proportional "expected-thickness" split.

### Cited Findings
- Reference [verified via the COHIBA manual's reference list]: Abrahamsen, P. (2005). "Combining methods for subsurface prediction." In Leuangthong & Deutsch (eds.), *Geostatistics Banff 2004*, vol. 2, pp. 601–610, Springer. DOI 10.1007/978-1-4020-3610-1_61 — [COHIBA manual v7.0 references](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- Problem statement in the manual (§1.2): "It is sometimes unclear when it is best to add layers from above or from below. This is solved by specifying multiple models and let COHIBA automatically combine the alternatives so that the precision is optimized." The typical case is a layered reservoir outlined by two accurate seismic reflectors, and COHIBA fits the layers "inside the envelope" — [COHIBA manual v7.0](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- §11: "Multiple models is a convenient way of squeezing surfaces between envelopes of well defined surfaces such as seismic reflectors. Intervals with large uncertainty are used for absorbing the inconsistencies between the models." Fig. 11.1 compares the "straight forward" approach (internal zones hung from the top reflector) with the combined approach, which gives "a compromise between zones of constant thickness and the shape given by the two seismic reflectors". Note that the reflectors themselves are also moved by the constant-thickness assumption — [COHIBA manual v7.0 §11](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- Worked example, Brent Group, North Sea (§11.1, Tables 11.1–11.3). There are five formations between TopBrent and BaseBrent, each with an isochore SD of 30% of its thickness. Stacking isochores from TopBrent gives a BaseBrent SD of ≈29 m. Depth-converting the whole Brent interval from seismic gives ≈15 m. The model weights are $29^2/(29^2+15^2)=0.79$ for the base path and 0.21 for the top path. Weights shift smoothly with depth inside the envelope (TopNess is 90.5% top-built, TopBroom is 78.6% base-built). The combined SD is always less than either path's SD: "COHIBA chooses the model weights so that the uncertainty is minimized." — [COHIBA manual v7.0 §11.1](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- The weights are sensitive to the stated uncertainties (Table 11.3). If the BaseBrent TWT/velocity SDs double, the base weight drops from 78.8% to 48.4%. The weights "can be negative and even larger than 1 in some rare cases. This is caused by correlations between different models." — [COHIBA manual v7.0 §11.1.1](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- Proportional option `<scale-isochores-to-seismic-envelopes>` (§17.2.14.2): reflectors and isochores need not agree, and oversized isochore packages can "force faults to open up". With this option, COHIBA scales all isochores in a package by $f(x,y)=\dfrac{\text{seismic envelope thickness}(x,y)}{\text{isochore package thickness}(x,y)}$ (eq. 17.2) — [COHIBA manual v7.0](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- `<weight-isochore-package-above>` (§11.1.4) rescales the isochore SDs so that a user-chosen model weight is approximately reached. This is an explicit knob between "trust the stack thickness" and "trust the bounding surfaces" — [COHIBA manual v7.0](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- A warning the manual says is common: "The average thickness between depth trend surfaces ... is negative ... Probable cause: Conflict between multiple models." Inconsistent stacks can cross — [COHIBA manual v7.0 §11](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).

### Inferences (equations for the cortex case)
- Cortex mapping. Two known bounding surfaces, the white-matter surface of bank A and that of bank B, are the "reflectors". Gap $D(x)$ between them. Unknown contact $c$. Thicknesses $h_A, h_B$ with $h_A+h_B=D$. Gaussian thickness priors $h_k\sim N(\mu_k(x),\sigma_k^2(x))$, where $\mu_k,\sigma_k$ come from kriging each bank's thickness laterally from where its pial side is visible. Conditioning on the sum gives the COHIBA two-path combination:
  $$E[h_A\mid D]=\mu_A+\frac{\sigma_A^2}{\sigma_A^2+\sigma_B^2}\,(D-\mu_A-\mu_B),\qquad \mathrm{Var}[h_A\mid D]=\frac{\sigma_A^2\sigma_B^2}{\sigma_A^2+\sigma_B^2}.$$
  The misfit $D-\mu_A-\mu_B$ goes mostly to the less certain stack ("intervals with large uncertainty absorb the inconsistencies"). The contact posterior SD is below both prior SDs.
- Equal split (midline) is the special case $\mu_A=\mu_B,\ \sigma_A=\sigma_B$. This explains why it fails for unequal stacks.
- The user's proportional rule $h_A=D\,\mu_A/(\mu_A+\mu_B)$ is COHIBA's envelope-scaling factor f. In the Gaussian picture, it is what you get when the errors are multiplicative and fully correlated between the two stacks (one common scale factor), or approximately when $\log h_k$ are Gaussian with equal log-SDs. With additive errors and a constant CV ($\sigma_k=\kappa\mu_k$, as in the Brent example's 30%), the posterior is $\mu_A+\frac{\mu_A^2}{\mu_A^2+\mu_B^2}(D-\Sigma\mu)$. This allocates the misfit in proportion to $\mu^2$, not $\mu$. Which rule fits the cortex is an empirical question.
- Inequality constraints ($0\le h_A\le D$, contact inside the gap) give a truncated Gaussian posterior. This is the setting of §3.

### Gaps
- I did not read the Abrahamsen (2005) paper itself. Its contents here come from the COHIBA manual. I found no published analysis of when proportional scaling should be preferred over variance-weighted combination.
- I found no geoscience case where the two stacks face each other (opposite polarity) and are not in vertical superposition. COHIBA's sign handling ("adding or subtracting intervals" when building up from a deeper surface) covers the algebra, though.

---

## 3. Inequality constraints, pinch-outs, erosion/onlap (how two units meeting is handled)

### Takeaway
Geomodellers handle units meeting, thinning to zero, or truncating with three tools: hard ordering of surfaces, truncation rules (erosive / onlapped), and kriging with inequality constraints, solved by Data Augmentation MCMC (Abrahamsen & Benth 2001). Thickness priors enter as isochore trend maps, and those maps may be zero.

### Cited Findings
- **Abrahamsen & Benth (2001)** [verified]: "Kriging with inequality constraints", *Mathematical Geology* 33(6):719–744, DOI 10.1023/A:1011078716252 — [NR PDF](https://projects.nr.no/sites/default/files/files/MathGeol-33-Abrahamsen-Bent-2001.pdf); [NR record](https://nr.no/en/publication/434653). The paper covers a GRF with an unknown linear trend. It derives the distribution of the trend coefficients given exact data and inequalities, and the conditional distribution of the field at any location (prediction, error and conditional simulation), using a Data Augmentation Algorithm. It is efficient for few locations and expensive on large grids — [search summary of the paper/abstract](https://repository.geologyscience.ru/handle/123456789/26691).
- COHIBA uses it for zone logs ("surfaces are above or below the well path at any particular location"): it runs the Data Augmentation Algorithm on the constraint points — [COHIBA manual v7.0 §23](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- Pinch-outs: when a zone has zero thickness, one well transition observes two surfaces at the same point. COHIBA flags these points (P in the PinchOut column) — [COHIBA manual v7.0 §8.2.2, §8.3.2](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- Erosive / onlapped surfaces: "erosive surfaces truncate surfaces below whereas onlapped surfaces truncate surfaces above". A surface can be both. Upper erosive surfaces erode lower ones, and an erosive surface erodes an onlapped one. The truncation hierarchy is set by the strict surface ordering and is applied in post-processing — [COHIBA manual v7.0 §8.3.3, §20.1](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf).
- Implicit-modelling equivalent: Calcagno, Chilès, Courrioux & Guillen (2008), "Geological modelling from field data and geological knowledge, Part I", *Physics of the Earth and Planetary Interiors*. It couples potential-field interpolation with geological rules (stratigraphic pile; erode/onlap relations between series) [partly verified: title, authors and venue from search; volume 171, pp. 147–157 and DOI 10.1016/j.pepi.2008.06.013 are unverified] — [search result reference](https://link.springer.com/doi/10.1007/bf02775087).

### Inferences
- For the cortex: a "thickness ≥ 0" constraint and "contact lies between the two WM surfaces" are inequality data. Treat them by truncated-Gaussian posteriors (Abrahamsen & Benth) rather than clipping. Clipping biases the mean and understates the variance.
- An erosive/onlap rule is a deterministic truncation applied after prediction. The fused-bank contact is better modelled as two conformable stacks with a shared boundary, as in §2, than as an erosion.

### Gaps
- I did not look at how commercial packages (Petrel, RMS) implement isochore-based "conformance" during gridding.

---

## 4. Stratigraphic grids: proportional vs top/base-conformable layering; GeoChron / UVT

### Takeaway
Reservoir grids split each zone into layers in one of two ways. **Proportional** layering uses fixed fractions of the local zone thickness: pseudo-time $t=(z-z_{base})/(z_{top}-z_{base})\in[0,1]$, which is exactly a normalised arrival / equivolume-free relative depth. **Follow-top / follow-base** layering uses a fixed cell thickness parallel to one bounding surface, which truncates against the other. Mallet's GeoChron generalises this to a full (u,v,t) "Wheeler-space" parameterisation, with t = geological time.

### Cited Findings
- Proportional layering: each layer thickness is proportional to the distance between the zone's bounding surfaces — [US 8271244 "Hybrid stratigraphic layering using pseudo-Wheeler space"](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8271244). A pseudo-time is defined from 0 at the base to 1 at the top — [US 12596203 "Layering for geomodeling"](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12596203).
- Follow-top / follow-base layers run parallel to the top or the base — [EP2227706A1](https://patents.google.com/patent/EP2227706A1/en). These styles are driven by a target cell thickness, not a cell count — [US 12596203](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12596203). In the Petrel zone dialog, a number of layers means proportional, and a cell thickness means build from top or base — [Petrel course slides (Scribd; secondary)](https://es.scribd.com/doc/256649934/Petrel-Introduction-Course).
- RESQML (the industry data standard) has an explicit "IJK Grid with Proportional Layering" representation — [Energistics RESQML docs](https://docs.energistics.org/RESQML/RESQML_TOPICS/RESQML-000-297-0-C-sv2010.html).
- A known drawback: proportional layers follow the reference horizons near those horizons but diverge from true reflectors over thick intervals — [Figshare dataset, Paleoscan vs proportional layering](https://commons.datacite.org/doi.org/10.6084/m9.figshare.14219654).
- **Mallet (2004)** [verified]: "Space–time mathematical framework for sedimentary geology", *Mathematical Geology* 36(1):1–32, DOI 10.1023/B:MATG.0000016228.75495.7c. It builds on Wheeler's time-stratigraphy and defines a space where all horizons are horizontal and faults are removed. In the (u,v,t) curvilinear coordinates, t is the deposition time and (u,v) are paleo-geographic coordinates — [HAL record](https://hal.univ-lorraine.fr/hal-04056043v1); [repository copy](https://repository.geologyscience.ru/items/de1a3366-f2d3-4a62-a099-c4aa57123dff/full).
- **Moyen, Mallet, Frank, Leflon & Royer (2004)** [verified]: "3D-parameterization of the 3D geological space – the GeoChron model", ECMOR IX, Cannes. The parameterisation is computed on a fault-respecting tetrahedral mesh, independent of any stratigraphic grid. DOI 10.3997/2214-4609-pdb.9.A004 — [HAL](https://hal.univ-lorraine.fr/hal-04056065); [EarthDoc](https://www.earthdoc.org/content/papers/10.3997/2214-4609-pdb.9.A004). Book: Mallet, *Elements of Mathematical Sedimentary Geology: the GeoChron Model*, EAGE 2014 — [EarthDoc](https://www.earthdoc.org/content/books/9789462820081) [partly verified].

### Inferences
- The user's normalised arrival (and its equivolume variant) is the cortical counterpart of the proportional pseudo-time t. The two-stack contact problem matters because proportional layering assumes the zone's top and base are both known. In fused cortex, one bounding surface (the contact) is unknown. COHIBA-style isochore conditioning (§2) supplies it, and proportional layering then applies within each bank.
- Follow-top layering is the analogue of hanging layers at fixed absolute depths from the pial surface (Baillarger bands at constant mm). Proportional layering is the analogue of normalised depth.

### Gaps
- I did not find Petrel/RMS official documentation (it is behind a paywall). Patent and course material are secondary sources.

---

## 5. Horizon interpolation and tracking with uncertainty; RGT/flattening

### Takeaway
Seismic flattening / RGT methods (Lomask; Wu & Hale) are mostly deterministic least-squares methods. They take soft constraints from interpreted horizons, but they do not output a posterior. Horizon uncertainty in practice is quantified by ensembles of interpretations, interpreter "uncertainty envelopes", or by geostatistical surface models (COHIBA) that treat the interpreted travel-time map as data with a GRF error.

### Cited Findings
- Wu & Hale, "Horizon volumes with interpreted constraints", *Geophysics* 80(2):IM21–IM33, 2015 [partly verified: title, venue, volume and pages via an arXiv reference list; DOI 10.1190/geo2014-0212.1 unverified] — [arXiv 2605.01273 reference list](https://arxiv.org/pdf/2605.01273).
- Lomask et al., "Flattening without picking", *Geophysics* 71(4):P13–P20, 2006 [unverified journal details]. The SEP reports show the method: dips are turned into time shifts by (non-linear) least squares, and faults are handled with IRLS weighting — [SEP report](https://sepwww.stanford.edu/data/media/public/docs/sep120/jesse1/paper_html/index.html).
- Travel-time maps as uncertain data: COHIBA takes a per-reflector interpretation uncertainty (constant, e.g. 4 ms, or a map) plus a variogram for its spatial correlation — [COHIBA manual v7.0 §1.4](https://projects.nr.no/sites/default/files/files/cohiba_manual_7_0.pdf). Abrahamsen (1993) models seismic error as $T_l=t_l+R^t_l$ — [PDF](https://projects.nr.no/sites/default/files/files/Bayesian-Multilayer-Abrahamsen-1993.pdf).
- **Schaaf & Bond (2019)** [verified]: "Quantification of uncertainty in 3-D seismic interpretation: implications for deterministic and stochastic geomodeling and machine learning", *Solid Earth* 10:1049–1061, DOI 10.5194/se-10-1049-2019. It uses 78 student interpretations of Gullfaks (Base Cretaceous Unconformity, Top Ness, faults), relates interpretation variability to seismic image quality, and proposes quality attributes as a proxy for uncertainty — [Copernicus](https://se.copernicus.org/articles/10/1049/2019/).
- An AAPG 2014 abstract describes per-horizon interpreter "uncertainty envelopes" — [Search and Discovery](https://www.searchanddiscovery.com/abstracts/html/2014/90189ace/abstracts/1840716.html) (based on the search summary only).

### Inferences
- The user's step, from a deterministic flattening to a posterior over the contact, goes past standard RGT practice. The geostatistical surface-modelling line (Abrahamsen/COHIBA) is the ready-made probabilistic layer to borrow, not the RGT line.

### Gaps
- I did not search specifically for Bayesian horizon picking or probabilistic RGT (for example MCMC horizon tracking, or Bayesian-neural-network RGT). That literature is unchecked.

---

## 6. Implicit (potential-field) modelling and probabilistic / Bayesian geomodels

### Takeaway
Implicit modelling represents the stratigraphy as iso-values of one scalar field, interpolated (by co-kriging / RBF / GP) from interface points and orientation (gradient) data. That is a direct analogue of an RGT / normalised-depth field. Bayesian versions (GemPy + PyMC) put priors on the inputs and sample the model posterior. Thickness enters only indirectly, through the spacing of potential iso-values and the interface data.

### Cited Findings
- **Lajaunie, Courrioux & Manuel (1997)** [partly verified: title, authors, venue and DOI; year from search; vol/issue unconfirmed]: "Foliation fields and 3D cartography in geology: principles of a method based on potential interpolation", *Mathematical Geology*, DOI 10.1007/BF02775087. It interpolates a scalar field whose gradient is orthogonal to the measured planes. Interfaces are its isovalues, and the orientation data need not lie on the interface — [Springer](https://link.springer.com/doi/10.1007/bf02775087).
- **de la Varga, Schaaf & Wellmann (2019)** [verified]: "GemPy 1.0: open-source stochastic geological modeling and inversion", *Geosci. Model Dev.* 12:1–32, DOI 10.5194/gmd-12-1-2019. It is an implicit potential-field method in Python/Theano. Automatic differentiation links it to probabilistic machine-learning / Bayesian inference frameworks — [GMD](https://gmd.copernicus.org/articles/12/1/2019/).
- **Wellmann & Caumon (2018)** [verified]: "3-D structural geological models: concepts, methods, and uncertainties", *Advances in Geophysics* 59:1–121, DOI 10.1016/bs.agph.2018.09.001 — [HAL](https://hal.univ-lorraine.fr/hal-01921494).
- Hillier, Schetselaar, de Kemp & Perron (2014), "Three-dimensional modelling of geological surfaces using generalized interpolation with radial basis functions", *Mathematical Geosciences* [partly verified: title and year from a reference list; author list and DOI unverified]. The GSC companion report is Open File 7814, DOI 10.4095/301665 — [GSC OF 7814](https://download-telecharger.services.geo.ca/pub/nrcan_rncan/publications/STPublications_PublicationsST/301/301665/of_7814.pdf). It is implemented in LoopStructural — [Grose et al. 2021, GMD 14:3915](https://gmd.copernicus.org/articles/14/3915/2021/).
- Gonçalves, Kumaira & Guadagnin (2017), "A machine learning approach to the potential-field method for implicit modeling of geological structures", *Computers & Geosciences* 103, DOI 10.1016/j.cageo.2017.03.015 [partly verified via a reference list]. It frames potential-field co-kriging as Gaussian-process regression [GP framing from memory, unverified].

### Inferences
- The potential field is in effect a GP-interpolated RGT. Interface points = "these voxels are at the same relative time". Orientations = gradient data. A cortex analogue: interpolate normalised depth as a GP with gradient (streamline) constraints, then read off the contact as an iso-value with posterior uncertainty. Unlike COHIBA, standard implicit modelling has no explicit per-unit thickness prior. Thickness is implicit in the scalar-field spacing, so COHIBA's explicit-isochore formulation fits the user's "expected thickness per stack" idea more directly.

### Gaps
- I did not verify any paper that adds explicit thickness or isochore likelihoods to a Bayesian implicit model. Candidates: Wellmann group "geological knowledge as likelihood" work; Grose et al. LoopStructural fold/thickness constraints. Both unchecked.
- Caumon's specific papers on stratigraphic thickness constraints were not looked up.
