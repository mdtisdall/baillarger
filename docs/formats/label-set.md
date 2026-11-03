# label-set: the classes of the segmentation

SpecVersion `label-set/1`. The code is `baillarger.labels`. The segmentation
(`_dseg`) of a hemisphere uses these classes. Labellers and the network
follow this document. The values are the values of FreeSurferColorLUT.

This document has no sidecar. A change to a class or to a rule changes the
version.

A decision marked **Proposed** is a proposal. The user decides it in the
review of the PR that adds this document. Until then the code uses the
proposal.

## The classes

A segmentation gives each voxel one class. The right hemisphere uses the
Right values, the left hemisphere the Left values.

| Class | Right | Left | Filled mask | Status |
|---|---|---|---|---|
| Outside | 0 | 0 | no | decided |
| White matter (WM) | 41 | 2 | yes | decided |
| Cortex | 42 | 3 | no | decided |
| Lateral ventricle | 43 | 4 | yes | decided |
| Thalamus | 49 | 10 | yes | decided |
| Caudate | 50 | 11 | yes | decided |
| Putamen | 51 | 12 | yes | decided |
| Pallidum | 52 | 13 | yes | decided |
| Corpus callosum | 192 | 192 | yes | decided |
| Accumbens | 58 | 26 | yes | Proposed |
| Ventral DC | 60 | 28 | yes | Proposed |
| Vessel | 62 | 30 | yes | Proposed |
| Choroid plexus | 63 | 31 | yes | Proposed |
| Inferior lateral ventricle | 44 | 5 | no | Proposed |
| Amygdala | 54 | 18 | no | Proposed |
| Hippocampus | 53 | 17 | open | open |

The network gives at least outside, WM, cortex, the filled classes and the
hippocampus. The hippocampus is a class of its own, separate from cortex and
outside. A label that is not in this table is not in the set. A later step
adds a class only when it needs it.

Both hemispheres share the value 192. The hemisphere is the one whose WM
value is in the segmentation (41: right, 2: left).

## The filled mask

The filled mask is the voxels with a label in this table. It is the WM mask
of `docs/formats/wm-dseg.md`.

| Hemisphere | Filled labels |
|---|---|
| R | `41, 43, 49, 50, 51, 52, 58, 60, 62, 63, 192` |
| L | `2, 4, 10, 11, 12, 13, 26, 28, 30, 31, 192` |

The hippocampus (53, 17) is not in the mask. Phase 1.8 decides whether it
joins the mask. This document does not decide it.

### Decisions for the labels that FreeSurfer fills differently

FreeSurfer makes its filled WM with `mri_edit_wm_with_aseg` and `mri_fill`.
The function `edit_segmentation` in
<https://github.com/freesurfer/freesurfer/blob/dev/mri_edit_wm_with_aseg/mri_edit_wm_with_aseg.cpp>
(branch `dev`; the lines and the commit were not recorded) sets these labels
to WM: caudate, putamen, pallidum, thalamus, ventral DC, accumbens, vessel.
It sets the lateral ventricle to WM if the ventricle is within one voxel of
cerebral WM, or in all other cases with the default option `-fillven 1`.
`recon-all` runs the program on the aseg (`recon-all`, `mri_edit_wm_with_aseg`
call, the section "Edit with ASeg"). `mri_fill` reads the aseg with
`-segmentation` and replaces the corpus callosum label with WM.

The aseg of the purple-mri hemisphere
(`sub-INDD121850_ses-7THemix20250516_aseg.nii.gz`, right) holds only the
labels 0, 41, 42, 43, 49, 50, 51, 52, 53 and 192. So every decision below
has 0 voxels in this aseg and changes no output now. The decisions apply to
later labels, from the network or from a person.

| Label | Decision | Reason |
|---|---|---|
| Accumbens (58, 26) | In. Proposed | FreeSurfer sets it to WM. It is deep grey matter inside the basal forebrain. |
| Ventral DC (60, 28) | In. Proposed | FreeSurfer sets it to WM. It touches the brainstem, so the review can look at it. |
| Vessel (62, 30) | In. Proposed | FreeSurfer sets it to WM. A vessel that is left out makes a cavity in the mask. |
| Choroid plexus (63, 31) | In. Proposed | It lies in the lateral ventricle, which is in. FreeSurfer fills it only if no inferior lateral ventricle is within one voxel. A class set cannot hold that rule. This is a deviation from FreeSurfer. |
| Inferior lateral ventricle (44, 5) | Out. Proposed | FreeSurfer does not fill it. It is the temporal horn beside the hippocampus, and the hippocampus is not in the mask. |
| Amygdala (54, 18) | Out. Proposed | FreeSurfer does not fill it. |

FreeSurfer also sets the WM hypointensities (77) to WM when they are not
within one voxel of cortex. The set does not have this label now, because the
aseg has none.

## Excisions are outside

The label set has no class for missing tissue. A voxel where tissue was cut
away has the class outside. The correction of the filled mask treats a fill
into an excision as a fill into outside (`docs/version-2-plan.md`, §3c,
step 2).

## The majority rule for partial volume

A voxel takes the class that fills most of it. A person who paints labels
follows this rule.

1. Estimate how much of the voxel each class fills.
2. Give the voxel the class with the largest part.

Two cases follow from the rule:

- A fluid channel that is narrower than half a voxel is not fluid. The voxel
  is cortex.
- Two cortical banks that touch pia to pia, with no fluid between them, are
  cortex. The voxel is cortex.

A fluid channel that fills more than half of a voxel is fluid (outside).
