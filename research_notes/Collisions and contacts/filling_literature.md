# Filling the cortex from the white surface: prior work

A first literature check (web searches, abstracts and secondary descriptions only; no full texts read). Tags as in `pde_framing.md`. It asks whether any method fills the cortex outward from the white surface, and whether any uses a volume or thickness budget to place an outer surface that is not seen, as in fused sulci.

| Method | What it does | Relation to the moving-mesh filling | Tag |
|---|---|---|---|
| DiReCT (Das, Avants, Grossman and Gee, NeuroImage 45(3), 2009; [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2836782); ANTs KellyKapowski) | Starts from a mantle of thickness ε on the WM/GM interface. Propagates it outward along the surface normal by greedy diffeomorphic registration, until it reaches the GM/CSF boundary or a thickness prior's maximum. The diffeomorphism stops opposing banks in buried sulci from crossing. | The closest relative: filling on the grid, with fused banks kept apart. Its thickness prior is an upper limit and does not decide where fused banks meet (symmetric in practice). There is no volume budget and no data steering. | verified (abstract and [CortexMorph](https://arxiv.org/pdf/2307.11567)'s description) |
| Coupled surfaces propagation (Zeng, Staib, Schultz and Duncan, IEEE TMI 18(10), 1999; [MICCAI 1998](https://rd.springer.com/content/pdf/10.1007/BFb0056237.pdf)) | Two level sets, inner and outer, each driven by the image. Their distance is held within a set range (about 3 mm). | Implicit fronts with a thickness constraint. The constant-thickness band is a crude budget. Known weakness: two GM/CSF boundaries closer than a voxel. | verified (abstract and secondary) |
| FreeSurfer pial deformation (Dale, Fischl and Sereno 1999) | Deforms the white mesh outward to the pia under image forces and a thickness limit, and blocks self-intersection. | The moving mesh with collision handling. Fused banks stop where they collide, which is equal arrival. | from the earlier review (`neuro_tools_contacts.md`) |
| Osechinskiy and Kruggel, "Cortical surface reconstruction from high-resolution MR brain images", 2012 ([Wiley](https://onlinelibrary.wiley.com/doi/10.1155/2012/870196)) | Finds the unresolved boundaries between banks in narrow sulci with a field in an inhomogeneous dielectric. Advects the pia along that field's gradient under a topology-preserving level set, constrained by sulcal skeletons and optionally a maximum distance. | A pressure-like field steering the outer surface, for buried sulci. The contact is set by the field and the skeleton, not by budgets. | verified (abstract) |
| Waehnert et al. 2014; Leprince et al. (BrainVISA [isovolume](https://brainvisa.info/axon/en/processes/isovolume.html)); Younes, Kutten and Ratnanather, "Normal and equivolumetric coordinate systems for cortical areas" ([arXiv 1911.07999](https://arxiv.org/pdf/1911.07999); MethodsX 2024) | Equal-volume layers between known inner and outer surfaces: along streamlines (Waehnert), by a combined Laplace and equal-volume level-set model (Leprince), or by diffeomorphic registration of the two surfaces reparametrised to equal volume (Younes). | The equal-volume speed V/A, but with both surfaces given. None uses volume to place an outer surface that is not seen. | verified (abstracts) |
| Chung et al., surface deformation as phase-boundary evolution ([pdf](https://pages.stat.wisc.edu/~mchung/papers/surfacetech.pdf)) | Models change of a cortical mesh over time, with phase-boundary evolution under internal growth. | A growth view, for longitudinal change, not for filling. | verified (abstract) |
| Robust explicit surfaces: Bridson, Fedkiw and Anderson 2002 (cloth collisions); Brochu and Bridson 2009 (El Topo) | Continuous collision detection, contact, and topology changes with remeshing on explicit triangle meshes. | The machinery for the tangles of a moving mesh. | from memory, unverified |

## Gap

No method found places a fused contact by filling each side to a volume or thickness budget estimated from evidence elsewhere on the bank.
- DiReCT, FreeSurfer and coupled surfaces use thickness only as a limit or a constant band.
- The equal-volume methods need the outer surface first.

The moving-mesh filling would combine three things: FreeSurfer's moving white mesh with collision handling; the equal-volume speed of Waehnert and Leprince, run forward; and budgets from the thickness evidence. Only the budgets appear to be new.

## Next to read

- DiReCT's full text: how it keeps banks apart, and what its thickness prior does in buried sulci.
- Osechinskiy and Kruggel's dielectric field: a possible pressure model.
- El Topo, for the remeshing and collision operations.
- Mechanical models of cortical growth and folding (for example Tallinen and colleagues): not searched. They model physical growth, not inference, but may bear on what tension means.
