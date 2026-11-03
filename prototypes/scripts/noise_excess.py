"""Excess over thermal noise by axis, in deep WM and in background 'shadows' of tissue.

usage: noise_excess.py <data_dir> <out_dir>
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
wm = aseg == 41
# deep WM: >= 6 voxels from any non-WM
deep = ndimage.binary_erosion(wm, iterations=6)
# background blocks far from tissue (>= ~20 voxels), with tissue-shadow fraction per axis
far = ~ndimage.binary_dilation(tis[::4, ::4, ::4], iterations=5)
proj = [tis.any(axis=a) for a in range(3)]
bg_blocks, shadow_frac = [], [{}, {}, {}]
for i in range(0, tis.shape[0] - 48, 40):
    for j in range(0, tis.shape[1] - 48, 40):
        for k in range(0, tis.shape[2] - 48, 40):
            if not far[i // 4:(i + 48) // 4 + 1, j // 4:(j + 48) // 4 + 1, k // 4:(k + 48) // 4 + 1].all():
                continue
            bg_blocks.append((i, j, k))
            sl = [slice(i, i + 48), slice(j, j + 48), slice(k, k + 48)]
            for a in range(3):
                shadow_frac[a][(i, j, k)] = float(proj[a][tuple(x for n, x in enumerate(sl) if n != a)].mean())
print("bg blocks", len(bg_blocks), flush=True)
far_vox = np.repeat(np.repeat(np.repeat(far, 4, 0), 4, 1), 4, 2)[: tis.shape[0], : tis.shape[1], : tis.shape[2]]
del aseg

def detrend(img, mask, s=4.0):
    # remove a smooth local mean inside the mask (normalised convolution)
    m = mask.astype(np.float32)
    num = ndimage.gaussian_filter(img * m, s)
    den = ndimage.gaussian_filter(m, s)
    return np.where(mask, img - num / np.maximum(den, 1e-3), 0.0), mask & (den > 0.5)

res = {}
rng = np.random.default_rng(0)
for e in (1, 2, 3):
    img = np.asarray(nib.load(echo_f(e)).dataobj).astype(np.float32)
    r = {}
    # thermal sigma from far background (chi, ~2 channels): std / 0.67
    bgv = img[far_vox][rng.choice(int(far_vox.sum()), 2_000_000, replace=False)]
    r["bg_mean"], r["bg_std"] = float(bgv.mean()), float(bgv.std())
    sigma = r["bg_std"] / 0.67
    r["sigma_est"] = sigma
    # deep WM, work in a bounding box to save memory
    idx = np.argwhere(deep)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    d, m = detrend(img[sl], deep[sl])
    r["wm_mean"] = float(img[sl][m].mean())
    r["wm_std_over_sigma"] = float(d[m].std() / sigma)
    r["wm_diffvar_over_2sigma2"] = {}
    for a in range(3):
        mm = np.moveaxis(m, a, 0); dd = np.moveaxis(d, a, 0)
        both = mm[1:] & mm[:-1]
        r["wm_diffvar_over_2sigma2"][a] = float(((dd[1:] - dd[:-1])[both] ** 2).mean() / (2 * sigma**2))
    # background lag-1 correlation per axis, 48^3 blocks far from tissue, split by tissue shadow
    r["bg_lag1"] = {}
    acc = {}
    B = 48
    for (i, j, k) in bg_blocks:
        b = img[i:i + B, j:j + B, k:k + B].astype(np.float64)
        b = b - ndimage.gaussian_filter(b, 8.0)
        for a in range(3):
            cls = "in_shadow" if shadow_frac[a][(i, j, k)] > 0.9 else ("no_shadow" if shadow_frac[a][(i, j, k)] < 0.1 else None)
            if cls is None:
                continue
            bb = np.moveaxis(b, a, 0)[8:-8, 8:-8, 8:-8]
            acc.setdefault(f"ax{a}_{cls}", []).append(((bb[1:] * bb[:-1]).mean() / (bb**2).mean(), bb.std() / sigma))
    for key, v in sorted(acc.items()):
        v = np.array(v)
        r["bg_lag1"][key] = {"lag1": round(float(np.median(v[:, 0])), 4), "std_over_sigma": round(float(np.median(v[:, 1])), 3), "n_blocks": len(v)}
    res[e] = r
    print(e, json.dumps(r), flush=True)
    del img
(out / "noise_excess.json").write_text(json.dumps(res, indent=1))
