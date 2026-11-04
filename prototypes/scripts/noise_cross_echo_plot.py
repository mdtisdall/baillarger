"""Views of noise_cross_echo_map.py's block map: the echo 1-2 correlation of the
background residuals on three planes through the tissue's centre (tissue
outlined), the same projected (max over each axis), and against the distance
to the tissue and the echo 1 mean.

usage: noise_cross_echo_plot.py <map_dir> <fig_dir>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src, fig = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(src / "noise_cross_echo_map.npz")
c = np.where(d["ok"], d["corr12"], np.nan)
t = d["tissue_frac"]
ctr = np.round(np.argwhere(t > 0.5).mean(0)).astype(int)
kw = dict(cmap="magma", vmin=0, vmax=0.6, interpolation="nearest", origin="lower")

f, ax = plt.subplots(2, 3, figsize=(15, 9))
for a in range(3):
    sl = [slice(None)] * 3
    sl[a] = ctr[a]
    im = ax[0, a].imshow(c[tuple(sl)].T, **kw)
    ax[0, a].contour(t[tuple(sl)].T, [0.5], colors="c", linewidths=0.8)
    ax[0, a].set_title(f"plane through the centre, axis {a} = {ctr[a]} (blocks)")
    ax[1, a].imshow(np.nanmax(c, axis=a).T, **kw)
    ax[1, a].contour(t.max(axis=a).T, [0.5], colors="c", linewidths=0.8)
    ax[1, a].set_title(f"max along axis {a}")
f.colorbar(im, ax=ax, shrink=0.6, label="echo 1-2 correlation of background residuals (16³ blocks)")
f.savefig(fig / "cross_echo_map.png", dpi=80)

ok = d["ok"]
rel = d["mean1"] / np.median(d["mean1"][ok])
f, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(d["dmin"][ok] * 0.16, d["corr12"][ok], s=2, alpha=0.3)
ax[0].set_xlabel("distance from the block to the tissue (mm)")
ax[0].set_ylabel("echo 1-2 correlation")
ax[1].scatter(rel[ok], d["corr12"][ok], s=2, alpha=0.3)
ax[1].set_xscale("log")
ax[1].set_xlabel("echo 1 block mean / median background mean")
for a_ in ax:
    a_.axhline(0, color="k", lw=0.5)
f.tight_layout()
f.savefig(fig / "cross_echo_scatter.png", dpi=90)
