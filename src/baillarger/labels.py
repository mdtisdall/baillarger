"""The label set of the segmentation (spec: docs/formats/label-set.md).

Values follow FreeSurferColorLUT. A label has one value for each hemisphere
("R", "L"), except the corpus callosum, which has one value for both.
"""

VERSION = "label-set/1"

#: the one label of outside: the medium round the specimen, excisions and,
#: ex vivo, the ventricles (docs/formats/label-set.md, "The labels that
#: become outside")
OUTSIDE = 0
CC = 192

#: FreeSurferColorLUT labels that inherited labels (the purple-mri aseg) can
#: hold, and that are outside in an isolated hemisphere: CSF (24), the lateral
#: ventricle (4, 43), the inferior lateral ventricle (5, 44) and the choroid
#: plexus (31, 63). They are not classes of the set: the dseg step
#: (baillarger.white_dseg) sets them to OUTSIDE.
TO_OUTSIDE = (4, 5, 24, 31, 43, 44, 63)

#: the class of the white matter, of the cortex and of the hippocampus
WM_LABEL = {"R": 41, "L": 2}
CORTEX_LABEL = {"R": 42, "L": 3}
HIPPOCAMPUS_LABEL = {"R": 53, "L": 17}

#: the labels filled into the WM mask, by hemisphere: WM, thalamus, caudate,
#: putamen, pallidum, corpus callosum, accumbens, ventral DC and vessel. The
#: amygdala is not filled. The hippocampus is open (phase 1.8).
FILLED_LABELS = {
    "R": (41, 49, 50, 51, 52, 58, 60, 62, 192),
    "L": (2, 10, 11, 12, 13, 26, 28, 30, 192),
}

#: FreeSurferColorLUT names of the labels of the set
LUT_NAMES = {
    0: "Unknown",
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    30: "Left-vessel",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
    62: "Right-vessel",
    192: "Corpus_Callosum",
}

#: pieces of less than CRUMB_MM3 (mm³) are removed: tissue pieces cut off by
#: an excision (baillarger.excisions, wm-excisions/1) and pieces of the WM
#: mask (baillarger.white_dseg, wm-dseg/1)
CRUMB_MM3 = 1.0
