"""The label set of the segmentation (spec: docs/formats/label-set.md).

Values follow FreeSurferColorLUT. A label has one value for each hemisphere
("R", "L"), except the corpus callosum, which has one value for both.
"""

VERSION = "label-set/1"

OUTSIDE = 0
CC = 192

#: the class of the white matter, of the cortex and of the hippocampus
WM_LABEL = {"R": 41, "L": 2}
CORTEX_LABEL = {"R": 42, "L": 3}
HIPPOCAMPUS_LABEL = {"R": 53, "L": 17}

#: the labels filled into the WM mask, by hemisphere: WM, lateral ventricle,
#: thalamus, caudate, putamen, pallidum, corpus callosum, and the Proposed
#: accumbens, ventral DC, vessel and choroid plexus. The hippocampus is open
#: (phase 1.8). The inferior lateral ventricle and the amygdala are outside.
FILLED_LABELS = {
    "R": (41, 43, 49, 50, 51, 52, 58, 60, 62, 63, 192),
    "L": (2, 4, 10, 11, 12, 13, 26, 28, 30, 31, 192),
}

#: the filled labels that are Proposed, not decided
PROPOSED_FILLED = {"R": (58, 60, 62, 63), "L": (26, 28, 30, 31)}

#: FreeSurferColorLUT names of the labels of the set
LUT_NAMES = {
    0: "Unknown",
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    30: "Left-vessel",
    31: "Left-choroid-plexus",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
    62: "Right-vessel",
    63: "Right-choroid-plexus",
    192: "Corpus_Callosum",
}
