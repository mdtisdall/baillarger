"""Cross-echo correlation of the excess over thermal noise, in deep WM and in far background.

Texture is the tissue, so it is shared by the echoes: their residuals correlate.
Noise from the acquisition (thermal, or quantization of each echo's k-space)
is independent per echo: their residuals do not. The residual is each echo
minus its local mean (normalised Gaussian convolution, s voxels, inside the
mask), as in noise_excess.py; the neighbour difference along each axis keeps
only the voxel scale.

usage: noise_cross_echo.py <data_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

data, out = Path(sys.argv[1]), Path(sys.argv[2])
sub = "sub-INDD121850_ses-7THemix20250516"
echo_f = lambda e: data / f"{sub}_acq-channelCOMBx160um_dir-positive_run-02_echo-{e}_part-mag_FLASH.nii.gz"
aseg = np.asarray(nib.load(data / f"{sub}_aseg.nii.gz").dataobj).astype(np.int16)
tis = aseg != 0
deep = ndimage.binary_erosion(aseg == 41, iterations=6)  # >= 6 voxels from any non-WM
far = ~ndimage.binary_dilation(tis, iterations=20)  # >= 20 voxels from any tissue
del aseg, tis


def bbox(mask):
    idx = np.argwhere(mask)
    return tuple(slice(a, b) for a, b in zip(idx.min(0), idx.max(0) + 1))


def residual(img, mask, s=4.0):
    m = mask.astype(np.float32)
    num = ndimage.gaussian_filter(img * m, s)
    den = ndimage.gaussian_filter(m, s)
    return np.where(mask, img - num / np.maximum(den, 1e-3), 0.0), mask & (den > 0.5)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean()))


res = {}
for name, mask in (("deep_wm", deep), ("far_background", far)):
    sl = bbox(mask)
    m0 = mask[sl]
    R, M = {}, None
    for e in (1, 2, 3):
        img = np.asarray(nib.load(echo_f(e)).dataobj[sl]).astype(np.float32)
        R[e], m = residual(img, m0)
        M = m if M is None else M & m
        del img
    r = {"voxels": int(M.sum()), "residual": {}, "neighbour_difference": {}}
    for a_, b_ in ((1, 2), (1, 3), (2, 3)):
        r["residual"][f"{a_}{b_}"] = round(corr(R[a_][M], R[b_][M]), 4)
        for ax in range(3):
            ma = np.moveaxis(M, ax, 0)
            both = ma[1:] & ma[:-1]
            da = np.diff(np.moveaxis(R[a_], ax, 0), axis=0)[both]
            db = np.diff(np.moveaxis(R[b_], ax, 0), axis=0)[both]
            r["neighbour_difference"][f"{a_}{b_}_ax{ax}"] = round(corr(da, db), 4)
    res[name] = r
    print(name, json.dumps(r), flush=True)
    del R, M

(out / "noise_cross_echo.json").write_text(json.dumps(res, indent=1))
