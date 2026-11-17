"""Crop a cube around a site: three echoes, labels, white-surface piece; save npz and a slice figure.

usage: tube_crop.py ECHO_DIR DSEG WHITE_GII X Y Z HALF_MM OUT_DIR
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

echo_dir, dseg_p, white_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
c_mm = np.array([float(a) for a in sys.argv[4:7]])
half = float(sys.argv[7])
out = Path(sys.argv[8]); out.mkdir(parents=True, exist_ok=True)

d = nib.load(dseg_p)
A = d.affine; Ai = np.linalg.inv(A)
c_vox = (Ai @ np.r_[c_mm, 1])[:3]
zooms = np.array(d.header.get_zooms()[:3])
hv = np.ceil(half / zooms).astype(int)
lo = np.maximum(np.round(c_vox).astype(int) - hv, 0)
hi = np.minimum(np.round(c_vox).astype(int) + hv + 1, d.shape)
sl = tuple(slice(a, b) for a, b in zip(lo, hi))
lab = np.asarray(d.dataobj[sl])
echoes = []
for k in (1, 2, 3):
    p = sorted(echo_dir.glob(f"*echo-{k}_*.nii"))[0]
    echoes.append(np.asarray(nib.load(p).dataobj[sl]).astype(np.float32))
echoes = np.stack(echoes)
A_crop = A.copy(); A_crop[:3, 3] = (A @ np.r_[lo, 1])[:3]

g = nib.load(white_p)
v = g.darrays[0].data.astype(np.float64); f = g.darrays[1].data
box_lo = (A @ np.r_[lo - 0.5, 1])[:3]; box_hi = (A @ np.r_[hi - 0.5, 1])[:3]
blo, bhi = np.minimum(box_lo, box_hi), np.maximum(box_lo, box_hi)
inside = np.all((v >= blo) & (v <= bhi), axis=1)
fk = f[inside[f].all(axis=1)]
keep = np.unique(fk); remap = -np.ones(len(v), np.int64); remap[keep] = np.arange(len(keep))
np.savez_compressed(out / "site.npz", echoes=echoes, labels=lab, affine=A_crop, lo=lo, hi=hi,
                    site_mm=c_mm, white_v=v[keep], white_f=remap[fk], white_ids=keep)
print("crop", lab.shape, "voxels", lo, hi, "white vertices", len(keep), "faces", len(fk))
print("labels present", {int(u): int((lab == u).sum()) for u in np.unique(lab)})

# figure: three slices through the site, echo 2 with label outlines and the white surface's vertices near the slice
cv = np.round(c_vox).astype(int) - lo
vv = (np.linalg.inv(A_crop) @ np.c_[v[keep], np.ones(len(keep))].T).T[:, :3]
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
for n, a in enumerate(ax):
    idx = [slice(None)] * 3; idx[n] = cv[n]
    img = echoes[1][tuple(idx)]; L = lab[tuple(idx)]
    a.imshow(img.T, cmap="gray", origin="lower", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
    a.contour((L == 42).T.astype(float), [0.5], colors="orange", linewidths=0.7)
    a.contour((L == 41).T.astype(float), [0.5], colors="cyan", linewidths=0.7)
    near = np.abs(vv[:, n] - cv[n]) < 0.5
    o = [k for k in range(3) if k != n]
    a.plot(vv[near, o[0]], vv[near, o[1]], ",", color="lime")
    a.plot(cv[o[0]], cv[o[1]], "r+", ms=14, mew=1.5)
    a.set_title(f"{'xyz'[n]} = {cv[n] + lo[n]} (voxel)  orange: cortex label  cyan: WM label  green: white surface")
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"site at {c_mm.round(2)} mm, echo 2, {2 * half:.0f} mm box")
fig.tight_layout(); fig.savefig(out / "site_slices.png", dpi=110)
