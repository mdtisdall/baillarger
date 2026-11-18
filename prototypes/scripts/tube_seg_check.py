"""Close-up check of the segmentation and the white surface at a site crop:
parallel slices along each axis through the site, echo 2 and echo 3, with
the label edges (cortex 42 orange, WM 41 cyan, outside medium 0 blue) and
the white surface's intersection with each slice plane (lime lines).

usage: tube_seg_check.py SITE_NPZ OUT_PNG [n_slices] [step_vox]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402


def plane_cut(v, f, ax, c):
    """Segments of the mesh (voxel coordinates) cut by the plane v[:, ax] = c."""
    d = v[f, ax] - c  # (F, 3)
    s = np.sign(d)
    cut = (s.min(1) < 0) & (s.max(1) > 0)
    segs = []
    for tri, dd in zip(f[cut], d[cut]):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (dd[i] < 0) != (dd[j] < 0):
                t = dd[i] / (dd[i] - dd[j])
                pts.append(v[tri[i]] + t * (v[tri[j]] - v[tri[i]]))
        if len(pts) == 2:
            segs.append(pts)
    o = [k for k in range(3) if k != ax]
    return np.array(segs)[:, :, o] if segs else np.zeros((0, 2, 2))


def main() -> None:
    z = np.load(sys.argv[1])
    out = Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    lab, E = z["labels"], z["echoes"]
    A = z["affine"]
    v = (np.linalg.inv(A) @ np.c_[z["white_v"], np.ones(len(z["white_v"]))].T).T[:, :3]
    f = z["white_f"]
    cv = np.round((np.linalg.inv(A) @ np.r_[z["site_mm"], 1])[:3]).astype(int)
    offs = (np.arange(n) - n // 2) * step
    fig, axs = plt.subplots(6, n, figsize=(4.2 * n, 25))
    for ax in range(3):
        for k, o in enumerate(offs):
            c = int(np.clip(cv[ax] + o, 0, lab.shape[ax] - 1))
            idx = [slice(None)] * 3
            idx[ax] = c
            L = lab[tuple(idx)].T
            segs = plane_cut(v, f, ax, c)
            for row, e in ((2 * ax, 1), (2 * ax + 1, 2)):
                a = axs[row, k]
                img = E[e][tuple(idx)].T
                a.imshow(img, cmap="gray", origin="lower", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
                a.contour(L == 42, [0.5], colors="orange", linewidths=0.6)
                a.contour(L == 41, [0.5], colors="cyan", linewidths=0.6, linestyles=":")
                a.contour(L == 0, [0.5], colors="deepskyblue", linewidths=0.6)
                a.add_collection(LineCollection(segs, colors="lime", linewidths=0.9))
                if o == 0:
                    oth = [i for i in range(3) if i != ax]
                    a.plot(cv[oth[0]], cv[oth[1]], "r+", ms=10)
                a.set_title(f"{'xyz'[ax]} = {c} ({o * 0.16:+.2f} mm), echo {e + 1}", fontsize=9)
                a.set_xticks([])
                a.set_yticks([])
    fig.suptitle("site crop: label edges (orange cortex, cyan dotted WM, blue outside medium), white surface cut (lime)")
    fig.tight_layout()
    fig.savefig(out, dpi=80)


if __name__ == "__main__":
    main()
