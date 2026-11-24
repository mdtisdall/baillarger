import sys
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from baillarger.data import load_patch

out = sys.argv[1]
p = load_patch("V1V2")
wm = p.rim == 2
fl = p.rim == 1
seg = np.zeros(p.rim.shape, np.uint8)
seg[(p.rim == 3) & ndi.binary_dilation(fl)] = 2  # pial edge of the cortex label
seg[wm & ~ndi.binary_erosion(wm)] = 1  # WM boundary shell
nib.save(
    nib.Nifti1Image(p.echoes[..., 2].astype(np.float32), p.affine),
    f"{out}/V1V2_echo3.nii.gz",
)
nib.save(nib.Nifti1Image(seg, p.affine), f"{out}/V1V2_boundaries.nii.gz")
open(f"{out}/V1V2_labels.txt", "w").write(
    '    0     0    0    0        0  0  0    "Clear Label"\n'
    '    1   255  220    0        1  1  1    "WM boundary"\n'
    '    2     0  200  255      0.6  1  1    "Cortex next to fluid"\n'
)
print("ok", p.rim.shape)
