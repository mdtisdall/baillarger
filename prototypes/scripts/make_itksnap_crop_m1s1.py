import sys
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from baillarger.data import load_patch

out = sys.argv[1]
p = load_patch("M1S1")
wm = p.rim == 2
fl = p.rim == 1
seg = np.zeros(p.rim.shape, np.uint8)
seg[(p.rim == 3) & ndi.binary_dilation(fl)] = 2
seg[wm & ~ndi.binary_erosion(wm)] = 1
c = (75, 75, 75)  # Dan's boundary point in M1S1 crop coordinates
ii, jj, kk = np.indices(seg.shape)
seg[(ii - c[0]) ** 2 + (jj - c[1]) ** 2 + (kk - c[2]) ** 2 <= 2] = 3
nib.save(
    nib.Nifti1Image(p.echoes[..., 1].astype(np.float32), p.affine),
    f"{out}/M1S1_echo2.nii.gz",
)
nib.save(nib.Nifti1Image(seg, p.affine), f"{out}/M1S1_boundaries.nii.gz")
open(f"{out}/M1S1_labels.txt", "w").write(
    '    0     0    0    0        0  0  0    "Clear Label"\n'
    '    1   255  220    0        1  1  1    "WM boundary"\n'
    '    2     0  200  255      0.6  1  1    "Cortex next to fluid"\n'
    '    3   255    0  255        1  1  1    "M1/S1 boundary point"\n'
)
print("ok", p.rim.shape, "lo", p.lo)
