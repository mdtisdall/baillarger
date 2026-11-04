"""Where the background's residuals correlate across echoes: a map in 16^3 blocks.

Each echo's residual is the echo minus its local mean (normalised Gaussian
convolution, s = 4 voxels) inside the background (outside the tissue grown by
2 voxels), as in noise_cross_echo.py. Per block: the correlation of each echo
pair's residuals, the background voxels, the mean of each echo, the distance
to the tissue, and the tissue's shadow along each axis (the fraction of the
block's lines along that axis that meet tissue). Ghosts of the tissue lie in
its shadow along the phase-encode axes; container walls, bubbles and Fomblin
structure need not.

usage: noise_cross_echo_map.py <data_dir> <out_dir>
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
B = 16
aseg_img = nib.load(data / f"{sub}_aseg.nii.gz")
tis = np.asarray(aseg_img.dataobj) != 0
sh = tuple(s // B * B for s in tis.shape)
tis = tis[: sh[0], : sh[1], : sh[2]]
bg = ~ndimage.binary_dilation(tis, iterations=2)
dist = ndimage.distance_transform_edt(~tis).astype(np.float32)  # voxels
gs = tuple(s // B for s in sh)


def blocks(x):
    return x.reshape(gs[0], B, gs[1], B, gs[2], B)


def bsum(x):
    return blocks(x).sum(axis=(1, 3, 5))


nbg = bsum(bg.astype(np.float32))
dmin = blocks(np.where(bg, dist, np.inf)).min(axis=(1, 3, 5))
# shadow: per axis, the fraction of the block's lines along that axis that meet tissue anywhere
shadow = []
for a in range(3):
    u, v = (x for x in range(3) if x != a)
    p = tis.any(axis=a)  # 2D projection along axis a
    q = p.reshape(gs[u], B, gs[v], B).mean(axis=(1, 3))
    shadow.append(np.broadcast_to(np.expand_dims(q, a), gs).astype(np.float32))
del dist

R, means = {}, {}
m = bg.astype(np.float32)
den = ndimage.gaussian_filter(m, 4.0)
for e in (1, 2, 3):
    img = np.asarray(nib.load(echo_f(e)).dataobj)[: sh[0], : sh[1], : sh[2]].astype(np.float32)
    means[e] = bsum(np.where(bg, img, 0)) / np.maximum(nbg, 1)
    R[e] = np.where(bg, img - ndimage.gaussian_filter(img * m, 4.0) / np.maximum(den, 1e-3), 0).astype(np.float32)
    del img
del den, m
ss = {e: bsum(R[e] ** 2) for e in R}
corr = {}
for a_, b_ in ((1, 2), (1, 3), (2, 3)):
    corr[f"{a_}{b_}"] = bsum(R[a_] * R[b_]) / np.sqrt(np.maximum(ss[a_] * ss[b_], 1e-12))
del R

ok = nbg >= 0.9 * B**3
np.savez_compressed(
    out / "noise_cross_echo_map.npz",
    nbg=nbg, ok=ok, dmin=dmin, shadow=np.stack(shadow), mean1=means[1], mean2=means[2], mean3=means[3],
    **{f"corr{k}": v for k, v in corr.items()}, tissue_frac=bsum(tis.astype(np.float32)) / B**3,
)
c = corr["12"]
summ = {"blocks_used": int(ok.sum()), "by_distance_vox": {}, "by_shadow": {}, "by_mean1_over_median": {}}
for lo, hi in ((2, 10), (10, 20), (20, 40), (40, 80), (80, 1e9)):
    s = ok & (dmin >= lo) & (dmin < hi)
    summ["by_distance_vox"][f"{lo}-{hi:g}"] = {"blocks": int(s.sum()), "median_corr12": round(float(np.median(c[s])), 4) if s.any() else None,
                                              "p90_corr12": round(float(np.percentile(c[s], 90)), 4) if s.any() else None}
for a in range(3):
    for name, s in (("in", shadow[a] > 0.9), ("out", shadow[a] < 0.1)):
        s = ok & s & (dmin >= 20)
        summ["by_shadow"][f"ax{a}_{name}"] = {"blocks": int(s.sum()), "median_corr12": round(float(np.median(c[s])), 4) if s.any() else None}
rel = means[1] / np.median(means[1][ok])
for lo, hi in ((0, 1.2), (1.2, 1.5), (1.5, 3), (3, 1e9)):
    s = ok & (rel >= lo) & (rel < hi) & (dmin >= 20)
    summ["by_mean1_over_median"][f"{lo}-{hi:g}"] = {"blocks": int(s.sum()), "median_corr12": round(float(np.median(c[s])), 4) if s.any() else None}
(out / "noise_cross_echo_map.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
