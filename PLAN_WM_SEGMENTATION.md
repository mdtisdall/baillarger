# Plan: our own WM segmentation of the FLASH, with a review loop

Planning file for a WM segmentation that baillarger trains itself, on the
FLASH echoes, with the user's answers as training labels. When the method is
settled, the lasting content goes into the module docstrings, the format specs
and the README, and this file is deleted.

## Why

The aseg comes from purple-mri. Purple's network was trained on 7 T T2w at
0.3 mm; its accuracy on the 160 µm FLASH is not established, and it gives only
labels (the argmax), not the probabilities behind them. Purple's own topology
steps do not help us: its WM topology fix is FreeSurfer's
`mris_fix_topology` on its own mesh, never written back to the labels, and
its buried-sulcus removal is a second network on the labels alone, with no
topology guarantee. On the current aseg, `white dseg` still makes 48 handle
cuts, 29 handle fills and 140 cavity fills.

The aim is not a better network for its own sake. It is the evidence the
white surface needs: a probability of filled WM per voxel, from the FLASH
itself, made better where the user has answered.

## How segmentation and the white surface depend on each other

The white surface depends on the segmentation through one interface, not
through the method that made it. The segmentation gives:

| Output | Required | BIDS name | Use in the white stage |
|---|---|---|---|
| Labels, FreeSurferColorLUT | yes | `_dseg` | the filled mask, as now |
| P(filled WM), float | no | `_label-WMfilled_probseg` | order of handle removal; data term of the antialiasing |
| Fold spread of P | no | `_label-WMfilled_desc-foldsd_probseg` | uncertainty for the review queue |

With labels only, the white stage works as now (handle removal ordered by
distance to the other phase; antialiasing by thin-plate smoothness alone), so
any segmentation tool still fits ([[aseg-source-agnostic]]). With P, the same
algorithm gets better decisions. So the surface work does not wait for the
network, and the network does not wait for the surface work.

The coupling that is real, and that this plan decides once:

1. **The label set.** The filled mask is `FILLED_LABELS` (WM, lateral
   ventricle, caudate, putamen, pallidum, thalamus, CC; not the
   hippocampus). The network must output at least these classes separately
   from cortex, hippocampus and outside, or the surface changes.
2. **Missing tissue.** Purple invents labels in the blocks cut out before
   the scan. Our network labels them as missing (or they are ignored in
   training). The white stage then needs a rule for a filled mask that ends
   at missing tissue: open, see **Decisions**.
3. **Calibration.** Handle removal by P puts each wall where P is lowest
   along the bridge or tunnel. That is only right if P is calibrated near
   0.5. Calibration is a diagnostic of every network round.
4. **Defect count.** The number of handles and cavities before correction
   sets the length of the review queue. It is a diagnostic of every round,
   not a target to optimise inside the network.

What the surface is judged by does not change: its normals
([[white-surface-for-normals]]). P matters to the surface mainly where it
moves a wall (which side of a fused bank a cut falls) and where it smooths
the staircase, both of which change normals. Position finer than a voxel is a
by-product, not a goal.

## Phases

Each phase is one or more PRs. Phase 1 does not depend on phases 2–4 and can
run first or alongside.

### Phase 0: data, label set, test regions

- List the FLASH hemispheres that exist and are expected, with protocol,
  resolution and missing blocks. The answer decides how much phase 5 is
  needed.
- Fix the label set (FreeSurferColorLUT): outside, WM, cortex, the
  `FILLED_LABELS` deep structures, hippocampus, missing tissue, and the rest
  of purple's labels only if a later stage needs them.
- Test regions: a few boxes per hemisphere (start with V1V2, PCC_SPL, M1S1,
  and the V1V2 seam review points), labelled by the user from scratch, not
  by correcting purple, so purple does not anchor them. Never used in
  training. Format: a BIDS `_dseg` with a versioned sidecar spec
  (`docs/formats/`), with the box and the labeller in the sidecar.
- Labelling tool for the test regions: decision below.

### Phase 1: the white stage reads probabilities

Independent of the network; exact-answer tests on phantoms first
([[test-not-patch]]).

1. `handles.remove_handles` takes an optional priority volume. With P, the
   object front claims in order of decreasing P and the background front in
   order of increasing P; the walls then fall where P is closest to 0.5. The
   distance ordering stays the default. Phantom test: a handle whose bridge
   has a known low-P place away from its thinnest place; the cut must fall
   there.
2. `antialias` takes an optional data term: near the boundary, phi is pulled
   towards a smooth function of P − 0.5, weighted by a per-voxel confidence;
   the sign constraints stay, so topology and freedom from self-intersection
   stay guaranteed. Phantom test: a plane at a known sub-voxel offset with a
   P that is a blurred step across it; the vertices must recover the offset.
3. Each fix in `desc-wmtopofixes` records its margin: the P along its wall.
   The review queue ranks by it.
4. Stand-in P on the real hemisphere, until phase 3 gives ours: purple rerun
   with its probabilities saved (nnU-Net `--save_npz` / `--save_probabilities`),
   summed over `FILLED_LABELS`. Ask the purple authors whether their
   container can do this.
5. Diagnostics, P ordering against distance ordering on the hemisphere: where
   the walls move, the normal metrics of [[white-surface-for-normals]], and
   views of the walls that move most.

### Phase 2: training data

- Export to nnU-Net's raw dataset format: the three echoes as channels
  `_0000`–`_0002`, labels with an ignore label.
- Seed labels from purple, as weak labels:
  - purple's labels in the interiors;
  - ignore in a band of about 2 voxels on each side of every purple boundary,
    in the missing blocks, and where a simple FLASH check disagrees with
    purple (the check to be chosen on the test regions, as a diagnostic);
  - the existing `wm-edits` answers as verified labels, never ignored.
- Verified labels are kept apart from the seed labels, with provenance, so a
  round can be retrained with or without the seed.
- The export is a module of the package (`baillarger.wmseg`) with its own
  format spec; nnU-Net itself stays an external tool, called on the exported
  dataset (decision below).

### Phase 3: round 0

- nnU-Net v2, `3d_fullres` with a residual-encoder preset (M or L), its own
  plans and augmentation, five folds split by hemisphere when there are
  several, by box otherwise (and then treat the folds' accuracy as an upper
  bound).
- Inference writes the `_dseg`, the WM-filled `_probseg` and the fold spread,
  with sidecars ([[bids-where-possible]]).
- Diagnostics against purple, on the test regions and on the hemisphere:
  boundary distance and Dice on the test regions; calibration of P near 0.5;
  handles and cavities before correction; the walls of phase 1 and their
  margins; views of the places where ours and purple's differ most.

### Phase 4: the review loop

1. Sites: the topology walls with the smallest margin, and the places with
   the largest fold spread, gathered into one queue per hemisphere
   ([[review-queue]]), each with a few concrete options and a preview.
2. Answers: label edits with provenance. Today `wm-edits` are diffs against
   one input segmentation; with a segmentation that changes every round, an
   answer must stand on its own: the verified label of each voxel, valid for
   any input. `wm-edits/1` is edited in place to say so
   ([[spec-versions-until-release]]).
3. Retrain from the previous round's weights with the verified labels added.
4. The rule against drift: only labels the user has verified become
   training labels. The decoded, topology-corrected labels and the network's
   own predictions never do.
5. Diagnostics per round, on the test regions only: the same as phase 3,
   plus the length of the queue.

### Phase 5: uncertainty

- Two estimates that need no new network output: the fold spread
  (epistemic: what more labels can fix), and σ from a calibrated P and a
  distance (aleatoric: what the image leaves open). If the boundary distance
  d at a voxel is Gaussian, d ~ N(μ, σ²), then P(inside) = Φ(−μ/σ).
- Uses: rank the queue (a high-spread site is a good training label; a
  high-σ site is a judgment and is recorded as one); weight the antialias
  data term by 1/σ²; a position uncertainty per white-surface vertex.
- Diagnostic: does σ predict the boundary errors on the test regions?

### Phase 6: only if the diagnostics ask for it

- A signed-distance output (capped at about 5 voxels, L1, trained only where
  every voxel within the cap is labelled), or a Laplace head with β-NLL, if
  the σ derived from P is poorly calibrated or saturated where it matters.
- Synthetic training images from label maps (SynthSeg, PostSynth), as
  augmentation, if generalising to other hemispheres or protocols is the
  problem.
- A topology loss (clDice, Betti matching), if the queue is too long. It can
  only reduce defects; the guarantee stays in phase 1's decoding.

## Decisions

Open, for the user:

1. How many FLASH hemispheres exist now, and how many are expected.
2. Where to train (nnU-Net needs CUDA for training; inference runs on the
   laptop).
3. The labelling tool for the test regions and the answers: our viewer, or
   ITK-SNAP / 3D Slicer with nnInteractive (Apache-2.0 code; check the
   licence of its weights, which may be non-commercial).
4. Missing tissue: a class of its own, or ignored; and what the filled mask
   does where it meets missing tissue.
5. Whether the hippocampus stays outside the filled mask when our labels
   replace purple's.
6. nnU-Net (Apache-2.0) as a dependency of the package, or an external tool
   that only the export and import modules know about. Proposed: external.

Taken:

- The network gives evidence; topology is guaranteed after it, by our own
  code (`handles`, `antialias`), never by the network.
- The white stage keeps working from labels alone.
- Only verified labels train later rounds.
- Test regions are labelled from scratch and never trained on.

## Literature

From memory; check before citing.

- nnU-Net v2 and residual encoders: Isensee et al., *Nature Methods* 2021;
  "nnU-Net revisited", MICCAI 2024.
- Sparse labels with the ignore label: Gotkowski et al., "Embarrassingly
  simple scribble supervision for 3D medical segmentation", 2024.
- Interactive refinement: Wang et al., DeepIGeoS (TPAMI 2018) and BIFSeg (TMI
  2018); MONAI Label (Diaz-Pinto et al., *Medical Image Analysis* 2024);
  nnInteractive (arXiv 2503.08373).
- Topology correction from probabilities: Han, Xu and Prince (topology-
  preserving level sets); Bazin and Pham (fast-marching topology correction,
  TOADS); Ségonne et al. (FreeSurfer's defect correction).
- Distance outputs: Xue et al., AAAI 2020; Navarro et al., MLMI 2019; Ma et
  al., MIDL 2020; for cortex, SegRecon (Gopinath et al. 2021), DeepCSR (Cruz
  et al., CVPR 2021), recon-all-clinical (Gopinath et al. 2024).
- Uncertainty: Kendall and Gal, NeurIPS 2017; β-NLL, Seitzer et al., ICLR
  2022.
- Topology losses: Hu et al., NeurIPS 2019; clDice (Shit et al., CVPR 2021);
  Betti matching (Stucki et al., ICML 2023).
- Purple: Khandelwal et al., *Imaging Neuroscience* 2024; the purple-mri
  repository (topology notes and surface scripts).
