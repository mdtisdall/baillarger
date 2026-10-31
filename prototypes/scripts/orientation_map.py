"""Qualitative T1: does the image carry a reproducible laminar orientation?

For a patch and a point, three orthogonal slices through the point. Columns:
  1. echo 3 (log magnitude)
  2. echo 3 detrended (minus a masked 4-voxel blur): the bands, without the
     smooth WM-to-pia trend
  3. agreement of the layer normal between echo 2 and echo 3, raw tensor
  4. the same on the detrended images, with short layer-direction strokes
Agreement is the angle between the two normals: green < 10 deg, yellow
10-20, red > 20 (random axes give a median of about 60). All smoothing is
masked normalized convolution over the cortex, so WM and fluid edges do not
leak in.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage as ndi

from baillarger.data import load_patch
from baillarger.frame import missing_tissue

SD, SI, SHP, MARGIN = 1.0, 2.0, 4.0, 2


def mconv(x, m, s):
    return ndi.gaussian_filter(x * m, s) / np.maximum(ndi.gaussian_filter(m, s), 1e-6)


def normals(img, m):
    sm = mconv(img, m, SD)
    g = np.stack(np.gradient(sm), -1)
    T = np.empty(img.shape + (3, 3))
    for i in range(3):
        for j in range(i, 3):
            T[..., i, j] = T[..., j, i] = mconv(g[..., i] * g[..., j], m, SI)
    return T


def first_eig(T, idx):
    w, V = np.linalg.eigh(T[idx])
    return V[:, :, 2]


def run(name, pt, out):
    p = load_patch(name)
    rim, vox = p.rim, p.voxel_mm
    ctx = (rim == 3) & ~missing_tissue(rim, p.flash, vox)
    m = ctx.astype(np.float64)
    inner = ndi.binary_erosion(ctx, iterations=MARGIN)
    logs = {
        e: np.log(np.maximum(p.echoes[..., e].astype(np.float64), 1.0)) for e in (1, 2)
    }
    det = {e: (L - mconv(L, m, SHP)) * m for e, L in logs.items()}
    d_wm = ndi.distance_transform_edt(rim != 2, sampling=vox)
    d_fl = ndi.distance_transform_edt(rim != 1, sampling=vox)
    gphi = np.stack(np.gradient(ndi.gaussian_filter(d_wm - d_fl, 2.0)), -1)

    # slices: a box of +-W voxels around pt in each plane
    W = 45
    lo = [max(0, c - W) for c in pt]
    hi = [min(s, c + W) for c, s in zip(pt, rim.shape)]
    sub = tuple(slice(a, b) for a, b in zip(lo, hi))
    planes = []
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = pt[ax] - lo[ax]
        planes.append((ax, tuple(sl)))

    # tensors only on the sub-box (plus padding) for speed
    pad = 12
    blo = [max(0, a - pad) for a in lo]
    bhi = [min(s, b + pad) for b, s in zip(hi, rim.shape)]
    box = tuple(slice(a, b) for a, b in zip(blo, bhi))
    inner_b = inner[box]
    off = tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, blo))
    res = {}
    for key, src in (("det", det),):
        idx = np.nonzero(inner_b)
        n2 = first_eig(normals(src[1][box], m[box]), idx)
        n3 = first_eig(normals(src[2][box], m[box]), idx)
        ang = np.full(inner_b.shape, np.nan)
        ang[idx] = np.degrees(np.arccos(np.clip(np.abs((n2 * n3).sum(1)), 0, 1)))
        nn = np.full(inner_b.shape + (3,), np.nan)
        nn[idx] = n3
        gn = gphi[box][idx]
        gn /= np.linalg.norm(gn, axis=1, keepdims=True) + 1e-12
        dev = np.full(inner_b.shape, np.nan)
        dev[idx] = np.degrees(np.arccos(np.clip(np.abs((gn * n3).sum(1)), 0, 1)))
        res[key] = (ang[off], nn[off], dev[off])
        print(
            f"   deviation from boundary-geometry normal: median {np.median(dev[idx]):.1f} deg, "
            f">20 deg: {np.mean(dev[idx] > 20):.0%}"
        )
        a = ang[idx]
        print(
            f"{name} {key}: interior voxels in box {a.size:,}; angle <10: "
            f"{np.mean(a < 10):.0%}, <20: {np.mean(a < 20):.0%}, median {np.median(a):.1f} deg"
        )

    img3 = logs[2][sub]
    det3 = det[2][sub]
    ctx_s = ctx[sub]
    cmap = ListedColormap(["#1a9850", "#fee08b", "#d73027"])
    norm = BoundaryNorm([0, 10, 20, 90], 3)
    names = ["i", "j", "k"]
    fig, axs = plt.subplots(3, 4, figsize=(18, 13.5))
    for r, (ax, sl) in enumerate(planes):

        def show(a, x, **kw):
            a.imshow(np.asarray(x[sl]).T, origin="lower", interpolation="nearest", **kw)
            a.set_xticks([])
            a.set_yticks([])

        im = img3[sl]
        v = np.percentile(im[ctx_s[sl]], [1, 99]) if ctx_s[sl].any() else (None, None)
        show(axs[r, 0], img3, cmap="gray", vmin=v[0], vmax=v[1])
        d = det3[sl]
        dv = np.percentile(np.abs(d[ctx_s[sl]]), 98) if ctx_s[sl].any() else 1
        show(axs[r, 1], np.where(ctx_s, det3, np.nan), cmap="gray", vmin=-dv, vmax=dv)
        ang, nn, dev = res["det"]
        show(axs[r, 2], img3, cmap="gray", vmin=v[0], vmax=v[1], alpha=0.5)
        show(axs[r, 2], ang, cmap=cmap, norm=norm, alpha=0.85)
        show(axs[r, 3], img3, cmap="gray", vmin=v[0], vmax=v[1], alpha=0.5)
        hm = axs[r, 3].imshow(
            dev[sl].T,
            origin="lower",
            interpolation="nearest",
            cmap="magma_r",
            vmin=0,
            vmax=40,
            alpha=0.9,
        )
        if r == 0:
            fig.colorbar(
                hm, ax=axs[:, 3], shrink=0.4, label="degrees from boundary geometry"
            )
        inplane = [d_ for d_ in range(3) if d_ != ax]
        segs, cols = [], []
        a2, n2d = ang[sl], nn[sl]
        for x in range(0, a2.shape[0], 3):
            for y in range(0, a2.shape[1], 3):
                if np.isnan(a2[x, y]):
                    continue
                nx, ny = n2d[x, y, inplane[0]], n2d[x, y, inplane[1]]
                if np.hypot(nx, ny) < 0.7:
                    continue
                tx, ty = -ny, nx
                L = 1.4 / np.hypot(tx, ty)
                segs.append([(x - tx * L, y - ty * L), (x + tx * L, y + ty * L)])
                cols.append(cmap(norm(a2[x, y])))
        xl, yl = axs[r, 2].get_xlim(), axs[r, 2].get_ylim()
        axs[r, 2].add_collection(LineCollection(segs, colors="k", linewidths=2.2))
        axs[r, 2].add_collection(LineCollection(segs, colors=cols, linewidths=1.2))
        axs[r, 2].set_xlim(xl)
        axs[r, 2].set_ylim(yl)
        axs[r, 0].set_ylabel(f"slice {names[ax]} = {pt[ax]}", fontsize=12)
        for c in range(4):
            xl, yl = axs[r, c].get_xlim(), axs[r, c].get_ylim()
            axs[r, c].plot(
                *(np.array([pt[d_] - lo[d_] for d_ in inplane])), "c+", ms=12, mew=2
            )
            axs[r, c].set_xlim(xl)
            axs[r, c].set_ylim(yl)
    titles = [
        "echo 3 (log)",
        "echo 3, trend removed",
        "echo 2 vs 3 agreement + layer strokes",
        "data normal vs boundary geometry",
    ]
    for c, t in enumerate(titles):
        axs[0, c].set_title(t, fontsize=13)
    fig.suptitle(
        f"{name}: layer-normal agreement between echo 2 and echo 3 "
        f"(green < 10°, yellow 10–20°, red > 20°; cyan + = review point; "
        f"{W * 2 * 0.16:.0f} mm field)",
        fontsize=14,
    )

    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    outdir = sys.argv[1]
    run("V1V2", (66, 62, 62), f"{outdir}/orientation_V1V2.png")
    run("M1S1", (75, 75, 75), f"{outdir}/orientation_M1S1.png")
