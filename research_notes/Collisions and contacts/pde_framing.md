# The fronts as PDE problems: sources

A first literature check, made after findings 23–27, for the plan's section *The fronts as PDE problems*. These are web searches only, with no full texts read. Tags: **verified** means the citation and abstract were seen; **unverified** means only a preprint, a listing or my recollection was seen; **inference** means no source states it.

## Pieces and sources

| Piece | Topic | Source | Tag | What it gives us |
|---|---|---|---|---|
| 1 Geometry | Second-order start from an embedded interface | Chopp, "Some improvements of the fast marching method", SIAM J. Sci. Comput. 23(1), 2001 ([SIAM](https://epubs.siam.org/doi/10.1137/S106482750037617X)) | unverified (abstract only; the author is attributed from the title) | A bicubic fit of the level-set function on each cell, then Newton steps to the interface. Gives second-order initial values, and a third-order fast marching scheme. |
| 1 Geometry | Augmented fast marching for reinitialisation | Salac, [arXiv 1111.6903](https://arxiv.org/pdf/1111.6903) | verified (abstract) | Distance and its derivatives up to second order. The cells that hold the interface go straight into the accepted set. |
| 1 Geometry | Higher-order fast marching | Merino-Caviedes et al., IEEE TIP 2018 ([listing](https://lmi.bwh.harvard.edu/lmi/publications/second-order-multi-stencil-fast-marching-method-non-constant-local-cost-model)) | unverified | Second-order multi-stencil fast marching. |
| 2 Transport | Extension velocities | Adalsteinsson and Sethian, J. Comput. Phys. 148, 2–22, 1999 ([Berkeley page](https://math.berkeley.edu/%7Esethian/2006/History/chart.extensions.coupling.html)) | verified (citation and abstract) | Quantities known only on the front are carried along the characteristics in the same fast marching pass. This is the standard method. |
| 2 Transport | Closest-point formulation of distance | Tsai, "Rapid and accurate computation of the distance function using grids" ([UCLA CAM 00-36](https://ww3.math.ucla.edu/camreport/cam00-36.pdf)) | unverified (preprint abstract; journal details not confirmed) | Carrying the closest point gives a distance without building a Voronoi diagram. It extends to piecewise-linear interfaces, which fits a mesh's foot points. |
| 2 Transport | Biharmonic extension of fields | [arXiv 1704.08897](https://arxiv.org/pdf/1704.08897) | verified (title) | An alternative, smoother extension. |
| 3 Contact | Multiplicatively weighted crystal-growth Voronoi diagram | Schaudt and Drysdale ([Dartmouth](https://digitalcommons.dartmouth.edu/dissertations/64)); Kobayashi and Sugihara, with fast marching ([METR03-45](https://www.keisu.t.u-tokyo.ac.jp/data/2003/METR03-45.pdf)) | verified (abstracts) | `march` with per-label speeds computes exactly this. Fast regions wrap round slow ones, and the pure weighted diagram has disconnected cells. |
| 3 Contact | Multiple arrivals for static Hamilton–Jacobi equations | Fomel and Sethian, PNAS 99(11), 2002 ([eScholarship](https://escholarship.org/content/qt6k99t2pq/qt6k99t2pq.pdf)) | verified (citation) | All arrivals from all sources in one phase-space pass. One route to having both sides' arrivals near Γ. |
| 3 Contact | Hamilton–Jacobi equations with discontinuous coefficients | Barles and Chasseigne, *On modern approaches of Hamilton–Jacobi equations and control problems with discontinuities*, Birkhäuser 2023 ([listing](https://hoepli.it/libro/on-modern-approaches-of-hamilton-jacobi-equations-and-control-problems-with-discontinuities/9783031493737.html)); [arXiv math/0303288](https://arxiv.org/pdf/math/0303288) | verified (listing and title) | Classical comparison fails, so a notion of solution suited to the interface is needed. Patch labels with different speeds make such a coefficient. |
| 3 Contact | Competing-front segmentation | "Efficient segmentation based on Eikonal and diffusion equations" ([ResearchGate](https://www.researchgate.net/publication/220661378_Efficient_segmentation_based_on_Eikonal_and_diffusion_equations)); dual-front models, Chen, Mirebeau, Cohen et al. ([arXiv 2006.07839](https://arxiv.org/pdf/2006.07839v3), [arXiv 1912.10122](https://arxiv.org/pdf/1912.10122)) | verified (abstracts) | Labels by first arrival, with each front's own PDE. No meeting condition is set by expected thickness. |
| 3 Contact | Multi-label fast marching | Risser et al., regularised multi-label fast marching ([ResearchGate](https://www.researchgate.net/publication/325515900_Regularized_multi-label_fast_marching_and_application_to_whole-body_image_segmentation)) | verified (title) | Segmentation by labelled fronts. |

## Gaps

- **No source found states a meeting condition set by an expected thickness on each side.** The free-boundary condition on Γ is our own. The searches were broad: "competing fronts", "two-phase eikonal free boundary", "multi-label fast marching".
- **Candidate follow-ups**, not yet read:
  - for interface conditions, Barles and Chasseigne's chapters on codimension-1 discontinuities;
  - Kobayashi and Sugihara's full text, for their crystal-growth boundary equation and the scheme's accuracy;
  - Chopp 2001's full text, for how the initial values are built.

## Points that are inference

- **Staircase shocks.** A staircased white surface makes false shocks (sheet edges and rims) at concave corners near the WM.
- **Root position as a transport equation.** Root position can be carried as a transport equation, Σ ρ_j a_j (X − X_j) = 0, giving identity below the voxel scale without labels.
- **The cheap two-sided route for piece 3.** Run the transport on each side of the unweighted equal-arrival boundary, and extend each side's weighted arrival a short way past it.
- **Contact sensitivity.** The sensitivity of Γ to μ depends on the angle between the two fronts.
