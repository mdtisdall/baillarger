"""A test crop around the almost-touching M1S1 stretch, at least 45 voxels (7.2 mm) from every
face, saved for ITK-SNAP with the same overlay, the bank end points and the user's gap points."""
import json, sys
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from baillarger.data import load_box

out = sys.argv[1]
M1S1_LO = np.array((380, 175, 421))
banks = {"A": [(84, 99, 97), (84, 122, 113)], "B": [(84, 113, 63), (84, 142, 86)]}
gap = [(84, 108, 78), (84, 114, 82), (84, 120, 85), (84, 126, 90), (84, 131, 94), (84, 137, 99), (84, 144, 103)]
allp = np.array(sum(banks.values(), []) + gap) + M1S1_LO          # full-grid
M = 45
lo = np.maximum(allp.min(0) - M, 0)
hi = np.minimum(allp.max(0) + M + 1, (700, 384, 1148))
p = load_box(tuple(int(v) for v in lo), tuple(int(v) for v in hi), name="M1S1_test")
rim = p.rim
wm = rim == 2
# WM boundary that ignores the box faces: erode with the outside counted as WM
shell = wm & ~ndi.binary_erosion(wm, border_value=1)
seg = np.zeros(rim.shape, np.uint8)
seg[(rim == 3) & ndi.binary_dilation(rim == 1)] = 2
seg[shell] = 1
ii, jj, kk = np.indices(seg.shape)
conv = lambda q: tuple(int(v) for v in np.array(q) + M1S1_LO - lo)
for lab, pts in ((4, banks["A"]), (5, banks["B"]), (6, gap)):
    for q in pts:
        c = conv(q)
        seg[(ii - c[0])**2 + (jj - c[1])**2 + (kk - c[2])**2 <= 1] = lab
nib.save(nib.Nifti1Image(p.echoes[..., 1].astype(np.float32), p.affine), f"{out}/M1S1_test_echo2.nii.gz")
nib.save(nib.Nifti1Image(seg, p.affine), f"{out}/M1S1_test_overlay.nii.gz")
open(f"{out}/M1S1_test_labels.txt", "w").write(
    '    0     0    0    0        0  0  0    "Clear Label"\n'
    '    1   255  220    0        1  1  1    "WM boundary"\n'
    '    2     0  200  255      0.6  1  1    "Cortex next to fluid"\n'
    '    4    40  120  255        1  1  1    "Bank A ends"\n'
    '    5   255  130   20        1  1  1    "Bank B ends"\n'
    '    6   255    0  255        1  1  1    "Gap points"\n')
meta = {"full_lo": [int(v) for v in lo], "full_hi": [int(v) for v in hi], "shape": list(rim.shape),
        "banks": {k: [conv(q) for q in v] for k, v in banks.items()}, "gap": [conv(q) for q in gap]}
json.dump(meta, open(f"{out}/M1S1_test_points.json", "w"), indent=1)
print("box full-grid lo", lo, "hi", hi, "shape", rim.shape)
for k, v in meta["banks"].items():
    print("bank", k, "test-crop 0-based", v, "ITK-SNAP", [tuple(c + 1 for c in q) for q in v])
print("gap test-crop 0-based", meta["gap"])
pts = np.array(sum(meta["banks"].values(), []) + meta["gap"])
print("minimum distance of any point to a box face: %d voxels (%.1f mm)" % (min(pts.min(), (np.array(rim.shape) - 1 - pts).min()), 0.16 * min(pts.min(), (np.array(rim.shape) - 1 - pts).min())))
