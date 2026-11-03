# label-set: the classes of the segmentation

SpecVersion `label-set/1`. The code is `baillarger.labels`. The segmentation
(`_dseg`) of a hemisphere uses these classes. Labellers and the network
follow this document. The values are the values of FreeSurferColorLUT.

This document has no sidecar. A change to a class or to a rule changes the
version.

Each decision in this document is decided by the user.

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
| Accumbens | 58 | 26 | yes | decided |
| Ventral DC | 60 | 28 | yes | decided |
| Vessel | 62 | 30 | yes | decided |
| Choroid plexus | 63 | 31 | follows the lateral ventricle | decided |
| Inferior lateral ventricle | 44 | 5 | no | decided |
| Amygdala | 54 | 18 | no | decided |
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
`recon-all` runs `mri_edit_wm_with_aseg` on the aseg, then `mri_fill`
with `-segmentation` (FreeSurfer 7.4.1, `bin/recon-all`, section "Edit with
ASeg"). All links below are to commit
`83d686e2f10c8fc1c7b70553e30fa5201a20c2b7` of the `dev` branch of
`freesurfer/freesurfer` (read on 2026-11-03). `E` is
[`mri_edit_wm_with_aseg/mri_edit_wm_with_aseg.cpp`](https://github.com/freesurfer/freesurfer/blob/83d686e2f10c8fc1c7b70553e30fa5201a20c2b7/mri_edit_wm_with_aseg/mri_edit_wm_with_aseg.cpp)
and `F` is
[`mri_fill/mri_fill.cpp`](https://github.com/freesurfer/freesurfer/blob/83d686e2f10c8fc1c7b70553e30fa5201a20c2b7/mri_fill/mri_fill.cpp).
The function `edit_segmentation` starts at E line 485. In it, a voxel of
the WM image that is below `WM_MIN_VAL` takes the value `AUTO_FILL` (it
becomes WM) in these cases:

| Aseg label | What FreeSurfer does | Source |
|---|---|---|
| Accumbens, caudate, putamen, pallidum, thalamus, ventral DC, vessel | Fills the voxel | E L888-912 |
| Lateral ventricle | Fills the voxel if cerebral WM is within one voxel. Else fills it if `-fillven` is not 0 (default 1) | E L594-626, L83 |
| Choroid plexus | Fills the voxel only if no inferior lateral ventricle is within one voxel | E L576-593 |
| Inferior lateral ventricle | Does not fill the voxel. It fills the cortex voxels beside it and some neighbours | E L627-771 |
| Hippocampus | Does not fill the voxel in this pass. It fills neighbours under distance rules | E L773 onward |
| WM hypointensities (77) | Not in the label set. Decided | The network labels them as WM, which is filled. Add 77 only if a later analysis needs it as its own class. |
| Third ventricle | No case in the first pass | E L485-912 |
| Amygdala | No case in the first pass. Later passes of `edit_segmentation` (E L1102 onward) name it with the inferior lateral ventricle and the hippocampus, and set voxels near it. I did not trace them | E L1102-1130 |
| Unknown (0), and the lateral ventricle again | A later pass fills outside voxels that touch both ventricle and WM, and ventricle within two voxels of WM | E L939-990 |

`mri_fill` with `-segmentation` reads the aseg, replaces the corpus callosum
label `CC_Central` with WM (F L2277-2290; the option is read at F L3569-3573),
and calls `fill_with_aseg` (F L3041, L6463-6685). In `fill_with_aseg`, the
aseg only gives the hemisphere: cortex, WM, the deep structures, the
hippocampus, the amygdala, the inferior lateral ventricle and the others
all count for their hemisphere. The voxels that are filled are the voxels
that the WM image (the output of `mri_edit_wm_with_aseg`) has as WM. Each
hemisphere keeps its largest connected component (18-adjacency) and has its
holes removed. So the table above decides the filled mask. `edit_segmentation`
has no case for the corpus callosum. So FreeSurfer's own evidence for the label
192 is `CC_Central` only. We keep 192 in the mask, as in `wm-dseg/1`.

The aseg of the purple-mri hemisphere
(`sub-INDD121850_ses-7THemix20250516_aseg.nii.gz`, right) holds only the
labels 0, 41, 42, 43, 49, 50, 51, 52, 53 and 192. It holds the right lateral
ventricle (43): about 3.18 million voxels, 13.1 cm³. So the ventricle
decision changes the output on this aseg. The other decisions below have 0
voxels in this aseg and change no output now. They apply to later labels,
from the network or from a person.

| Label | Decision | Reason |
|---|---|---|
| Lateral ventricle (43, 4) | In. Decided by the user on 2026-11-03 | FreeSurfer fills it (`-fillven`). Phase 1.0 measured the white surface with the ventricle set to outside. The ventricle, open through the medial cut, then adds 5.9 % vertices and 4,743 mm² of wall that is not grey/white. It also adds 33 topology fixes and nearly 3 times the overview repairs. Filled, the surface has no inner wall. |
| Accumbens (58, 26) | In. Decided | FreeSurfer fills it. It is deep grey matter inside the basal forebrain. |
| Ventral DC (60, 28) | In. Decided | FreeSurfer fills it. It touches the brainstem. At the whole-brain stage, the cut that separates the hemispheres and the brainstem must separate the ventral DC from the midbrain. |
| Vessel (62, 30) | In. Decided | FreeSurfer fills it. A vessel that is left out makes a cavity in the mask. |
| Choroid plexus (63, 31) | Follows the lateral ventricle. Decided | It has the same class as the lateral ventricle, so it is filled for the same reason (the row above). If the user makes the ventricles outside, the choroid plexus is outside with them. FreeSurfer uses a neighbourhood rule, which a class set cannot hold. |
| Inferior lateral ventricle (44, 5) | Out. Decided | FreeSurfer does not fill the voxel. It is the temporal horn beside the hippocampus, and the hippocampus is not in the mask. |
| Amygdala (54, 18) | Out. Decided | FreeSurfer has no case for it. |
| WM hypointensities (77) | Not in the label set. Decided | The network labels them as WM, which is filled. Add 77 only if a later analysis needs it as its own class. |

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
