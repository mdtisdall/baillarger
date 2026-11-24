"""Where do the failing data-normal streamlines go on the k = 27 fused stretch,
and what does the data orientation look like there?"""

import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy import ndimage as ndi

import flatten_fused2 as F
from baillarger.data import load_patch
from baillarger.frame import missing_tissue


def trace_all(seed, field, rim, miss, sign0):
    x = np.asarray(seed, float)
    d_prev = sign0 / np.linalg.norm(sign0)
    x = x + d_prev * F.ENTER
    pts, s = [x.copy()], F.ENTER
    while s < F.LMAX:
        n1 = field(x)
        if s < 4.0:
            n1 = n1 if np.dot(n1, d_prev) >= 0 else -n1
            n1 = (n1 + d_prev) / np.linalg.norm(n1 + d_prev)
        if np.dot(n1, d_prev) < 0:
            n1 = -n1
        n2 = field(x + 0.5 * F.STEP * n1)
        if np.dot(n2, n1) < 0:
            n2 = -n2
        x = x + F.STEP * n2
        s += F.STEP
        d_prev = n2
        v = tuple(np.clip(np.round(x).astype(int), 0, np.array(rim.shape) - 1))
        pts.append(x.copy())
        if miss[v]:
            return np.array(pts), "missing"
        lab = rim[v]
        if lab == 1:
            return np.array(pts), "exit"
        if lab == 2 and s > 12.0:
            return np.array(pts), "facing WM"
        if lab != 3:
            return np.array(pts), "other"
    return np.array(pts), "length"


def run(out):
    p = load_patch("V1V2")
    rim, vox = p.rim, p.voxel_mm
    miss = missing_tissue(rim, p.flash, vox)
    ctx = (rim == 3) & ~miss
    m = ctx.astype(float)
    k = F.P1[2]
    lo = np.array(
        [min(F.P1[0], F.P2[0]) - 30, min(F.P1[1], F.P2[1]) - 30, k - 30]
    ).clip(0)
    hi = np.minimum(
        np.array([max(F.P1[0], F.P2[0]) + 31, max(F.P1[1], F.P2[1]) + 31, k + 31]),
        rim.shape,
    )
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    rb, mb, missb = rim[box], m[box], miss[box]
    e = {i: np.log(np.maximum(p.echoes[box + (i,)].astype(float), 1.0)) for i in (1, 2)}
    det = {i: (L - F.mconv(L, mb, F.SHP)) * mb for i, L in e.items()}
    Tc = F.tensor(det[1], mb) + F.tensor(det[2], mb)
    d_wm = ndi.distance_transform_edt(rb != 2, sampling=vox)
    gwm = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0)), 0)
    wm = rb == 2
    shell = wm & ~ndi.binary_erosion(wm)
    kk = k - lo[2]
    a = (F.P1[0] - lo[0], F.P1[1] - lo[1])
    b = (F.P2[0] - lo[0], F.P2[1] - lo[1])
    path = F.boundary_path(shell[:, :, kk], a, b)
    arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))]) * vox[0]

    # per-voxel eigen-analysis on the slice
    M = np.stack(
        [
            np.stack([Tc[0], Tc[1], Tc[2]], -1),
            np.stack([Tc[1], Tc[3], Tc[4]], -1),
            np.stack([Tc[2], Tc[4], Tc[5]], -1),
        ],
        -2,
    )[:, :, kk]
    w, V = np.linalg.eigh(M)
    n = V[..., :, 2]
    coh = (w[..., 2] - w[..., 1]) / np.maximum(w[..., 2] + w[..., 1], 1e-12)
    ctx_s = ctx[box][:, :, kk]
    coh = np.where(ctx_s, coh, np.nan)
    outp = np.where(
        ctx_s, np.degrees(np.arcsin(np.clip(np.abs(n[..., 2]), 0, 1))), np.nan
    )

    field = lambda x: F.normal_at(Tc, x)
    cols = {
        "exit": "#1f78b4",
        "facing WM": "#33a02c",
        "length": "#e31a1c",
        "other": "#ff7f00",
        "missing": "#6a3d9a",
    }
    traces = []
    for i, j in path:
        sd = (i, j, kk)
        g0 = np.array([F.interp(c, sd) for c in gwm])
        pts, st = trace_all(sd, field, rb, missb, g0)
        traces.append((pts, st))

    sl = det[2][:, :, kk]
    v = np.nanpercentile(np.abs(sl[ctx_s]), 98)
    xl = (path[:, 0].min() - 22, path[:, 0].max() + 28)
    yl = (path[:, 1].min() - 18, path[:, 1].max() + 18)
    fig, axs = plt.subplots(1, 4, figsize=(24, 7.5))
    for ax in axs:
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    # 1: streamlines by outcome, in-plane projection
    ax = axs[0]
    ax.imshow(
        np.where(ctx_s, sl, np.nan).T, origin="lower", cmap="gray", vmin=-v, vmax=v
    )
    ax.contour(shell[:, :, kk].T, levels=[0.5], colors="y", linewidths=0.8)
    for pts, st in traces:
        ax.plot(pts[:, 0], pts[:, 1], "-", c=cols[st], lw=1.0)
        ax.plot(pts[-1, 0], pts[-1, 1], "o", c=cols[st], ms=3)
    for st, c in cols.items():
        nst = sum(1 for _, s_ in traces if s_ == st)
        if nst:
            ax.plot([], [], "-", c=c, label=f"{st} ({nst})")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("streamlines by outcome (in-plane projection)")
    # 2: out-of-slice travel of each streamline
    ax = axs[1]
    for c_, (pts, st) in enumerate(traces):
        s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))] * vox[0]
    ax.set_visible(False)
    ax2 = fig.add_axes(axs[1].get_position())
    for c_, (pts, st) in enumerate(traces):
        s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))] * vox[0]
        ax2.plot(s_, (pts[:, 2] - kk) * vox[2], "-", c=cols[st], lw=0.9)
    ax2.set_xlabel("arc length along the streamline (mm)")
    ax2.set_ylabel("distance out of the slice (mm)")
    ax2.set_title("how far each streamline leaves the slice")
    ax2.axhline(0, c="k", lw=0.5)
    # 3: layer strokes coloured by coherence
    ax = axs[2]
    ax.imshow(
        np.where(ctx_s, sl, np.nan).T,
        origin="lower",
        cmap="gray",
        vmin=-v,
        vmax=v,
        alpha=0.6,
    )
    segs, cc = [], []
    for x in range(int(xl[0]), int(xl[1]), 2):
        for y in range(int(yl[0]), int(yl[1]), 2):
            if (
                x < 0
                or y < 0
                or x >= ctx_s.shape[0]
                or y >= ctx_s.shape[1]
                or not ctx_s[x, y]
            ):
                continue
            nx, ny = n[x, y, 0], n[x, y, 1]
            tx, ty = -ny, nx
            h = np.hypot(tx, ty)
            if h < 0.2:
                continue
            L = 0.9 / h * min(h, 1)
            segs.append([(x - tx * L, y - ty * L), (x + tx * L, y + ty * L)])
            cc.append(coh[x, y])
    lc = LineCollection(segs, array=np.array(cc), cmap="viridis", linewidths=1.6)
    lc.set_clim(0, 0.8)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, shrink=0.7, label="coherence (λ1−λ2)/(λ1+λ2)")
    ax.contour(shell[:, :, kk].T, levels=[0.5], colors="y", linewidths=0.8)
    ax.plot(path[:, 0], path[:, 1], "y-", lw=2)
    ax.set_title("layer direction (in-plane), coloured by coherence")
    # 4: how far the normal tilts out of the slice
    ax = axs[3]
    im = ax.imshow(outp.T, origin="lower", cmap="magma_r", vmin=0, vmax=90)
    fig.colorbar(
        im, ax=ax, shrink=0.7, label="layer normal's tilt out of the slice (deg)"
    )
    ax.contour(shell[:, :, kk].T, levels=[0.5], colors="y", linewidths=0.8)
    ax.plot(path[:, 0], path[:, 1], "c-", lw=2)
    ax.set_title("normal out of plane: 0 = in slice, 90 = along k")
    # mark the gap
    gap = [c_ for c_, (_, st) in enumerate(traces) if st != "facing WM"]
    fig.suptitle(
        "V1V2 slice k = 27 fused stretch: failing streamlines and the data orientation "
        f"(seeds along the yellow path, {arc[-1]:.1f} mm)",
        fontsize=14,
    )
    fig.savefig(out, dpi=105, bbox_inches="tight")
    stats = {}
    for pts, st in traces:
        stats.setdefault(st, []).append(np.abs(pts[:, 2] - kk).max() * vox[2])
    for st, vv in stats.items():
        print(
            f"{st}: {len(vv)} streamlines, max out-of-slice travel median {np.median(vv):.2f} mm"
        )
    print("wrote", out)


if __name__ == "__main__":
    run(sys.argv[1])
