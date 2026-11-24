"""Checkpoint: do bands stay level when profiles follow the data's layer normal?

V1V2, one open V1 stretch in slice k chosen by the user. Seeds: the WM
boundary pixels of that slice on the shortest boundary path between the two
points. From each seed, a 3D streamline along the layer normal (the
eigenvector of the summed structure tensor of trend-removed echo 2 and echo 3,
trilinearly interpolated as a tensor), signed by continuity, until it enters
fluid (exit) or fails (WM, missing tissue, other labels, 5 mm). Profiles of
trend-removed echo 3 and an R2* proxy ln(S2/S3)/dTE are stacked by relative
depth s/L and by arc length s. The same is done with a crude boundary-geometry
normal, grad(d_wm - d_fluid), for comparison.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from baillarger.data import load_patch
from baillarger.frame import missing_tissue

P1, P2 = (32, 78, 27), (49, 114, 27)
SD, SI, SHP = 1.0, 2.0, 4.0
STEP, ENTER, LMAX = 0.25, 1.5, 44.0  # voxels (0.04, 0.24, 5.1 mm)


def mconv(x, m, s):
    return ndi.gaussian_filter(x * m, s) / np.maximum(ndi.gaussian_filter(m, s), 1e-6)


def tensor(img, m):
    g = np.stack(np.gradient(mconv(img, m, SD)), -1)
    comps = []
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        comps.append(mconv(g[..., i] * g[..., j], m, SI))
    return np.stack(comps, 0)


def interp(vol, x):
    return ndi.map_coordinates(vol, np.asarray(x, float).reshape(3, 1), order=1, mode="nearest")[0]


def normal_at(Tc, x):
    t = [interp(c, x) for c in Tc]
    M = np.array([[t[0], t[1], t[2]], [t[1], t[3], t[4]], [t[2], t[4], t[5]]])
    w, V = np.linalg.eigh(M)
    return V[:, 2]


def boundary_path(shell2d, a, b):
    ii, jj = np.nonzero(shell2d)
    idx = -np.ones(shell2d.shape, int)
    idx[ii, jj] = np.arange(ii.size)
    rows, cols, w = [], [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == dj == 0:
                continue
            ni, nj = ii + di, jj + dj
            ok = (ni >= 0) & (nj >= 0) & (ni < shell2d.shape[0]) & (nj < shell2d.shape[1])
            ok[ok] = idx[ni[ok], nj[ok]] >= 0
            rows += list(np.nonzero(ok)[0]); cols += list(idx[ni[ok], nj[ok]])
            w += [np.hypot(di, dj)] * int(ok.sum())
    G = coo_matrix((w, (rows, cols)), shape=(ii.size, ii.size)).tocsr()
    s, t = idx[a], idx[b]
    _, pred = dijkstra(G, indices=s, return_predecessors=True)
    path = [t]
    while path[-1] != s:
        path.append(pred[path[-1]])
    path = path[::-1]
    return np.stack([ii[path], jj[path]], 1)


def trace(seed, field, rim, miss, sign0):
    x = np.asarray(seed, float)
    d_prev = sign0 / np.linalg.norm(sign0)
    x = x + d_prev * ENTER  # leave the WM along the outward direction first
    pts, s = [x.copy()], ENTER
    while s < LMAX:
        n1 = field(x)
        if s < 4.0:  # near the WM corner: lean on the outward direction
            n1 = n1 if np.dot(n1, d_prev) >= 0 else -n1
            n1 = (n1 + d_prev) / np.linalg.norm(n1 + d_prev)
        if np.dot(n1, d_prev) < 0:
            n1 = -n1
        xm = x + 0.5 * STEP * n1
        n2 = field(xm)
        if np.dot(n2, n1) < 0:
            n2 = -n2
        x = x + STEP * n2
        s += STEP
        d_prev = n2
        v = tuple(np.clip(np.round(x).astype(int), 0, np.array(rim.shape) - 1))
        if miss[v]:
            return None, "missing"
        lab = rim[v]
        pts.append(x.copy())
        if lab == 1:
            return np.array(pts), "exit"
        if lab == 2 and s > 12.0:
            return np.array(pts), "facing WM"
        if lab != 3:
            return None, "wm" if lab == 2 else "other"
    return None, "length"


def run(out):
    p = load_patch("V1V2")
    rim, vox = p.rim, p.voxel_mm
    miss = missing_tissue(rim, p.flash, vox)
    ctx = (rim == 3) & ~miss
    m = ctx.astype(float)
    k = P1[2]
    # box around the stretch
    lo = np.array([min(P1[0], P2[0]) - 30, min(P1[1], P2[1]) - 30, k - 30]).clip(0)
    hi = np.minimum(np.array([max(P1[0], P2[0]) + 31, max(P1[1], P2[1]) + 31, k + 31]), rim.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    rb, mb, missb = rim[box], m[box], miss[box]
    e = {i: np.log(np.maximum(p.echoes[box + (i,)].astype(float), 1.0)) for i in (1, 2)}
    det = {i: (L - mconv(L, mb, SHP)) * mb for i, L in e.items()}
    Tc = tensor(det[1], mb) + tensor(det[2], mb)
    d_wm = ndi.distance_transform_edt(rb != 2, sampling=vox)
    d_fl = ndi.distance_transform_edt(rb != 1, sampling=vox)
    gphi = np.stack(np.gradient(ndi.gaussian_filter(d_wm - d_fl, 2.0)), 0)
    gwm = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0)), 0)
    dte = (p.echo_times_ms[2] - p.echo_times_ms[1]) / 1000.0
    r2 = mconv((e[1] - e[2]) / dte, mb, 0.5) * mb  # R2* proxy, s^-1

    wm = rb == 2
    shell = wm & ~ndi.binary_erosion(wm)
    a = (P1[0] - lo[0], P1[1] - lo[1]); b = (P2[0] - lo[0], P2[1] - lo[1])
    kk = k - lo[2]
    path = boundary_path(shell[:, :, kk], a, b)
    seeds = [(i, j, kk) for i, j in path]
    arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))]) * vox[0]

    fields = {
        "data normal": lambda x: normal_at(Tc, x),
        "distance from WM (D-like)": lambda x: (lambda g: g / (np.linalg.norm(g) + 1e-12))(
            np.array([interp(c, x) for c in gwm])),
    }
    nbin = 60
    results = {}
    for name, f in fields.items():
        img_r = np.full((nbin, len(seeds)), np.nan)
        img_rr = np.full((nbin, len(seeds)), np.nan)
        img_s = np.full((int(LMAX), len(seeds)), np.nan)  # arc length, 1 voxel bins
        lines, status, marks = [], {}, []
        for c, sd in enumerate(seeds):
            g0 = np.array([interp(cc, sd) for cc in gwm])
            pts, st = trace(sd, f, rb, missb, g0)
            status[st] = status.get(st, 0) + 1
            if pts is None:
                continue
            lines.append(pts)
            endtype = st
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            L = seg[-1]
            vals = ndi.map_coordinates(det[2], pts.T, order=1)
            rv = ndi.map_coordinates(r2, pts.T, order=1)
            q = (np.arange(nbin) + 0.5) / nbin * L
            img_r[:, c] = np.interp(q, seg, vals)
            img_rr[:, c] = np.interp(q, seg, rv)
            qs = np.arange(img_s.shape[0]) + 0.5
            ok = qs < L
            img_s[ok, c] = np.interp(qs[ok], seg, vals)
            if st == "facing WM":
                sm = ndi.gaussian_filter1d(vals, 4)
                lo_, hi_ = int(0.2 * len(sm)), int(0.8 * len(sm))
                cpos = seg[lo_ + int(np.argmin(sm[lo_:hi_]))]
                marks.append((arc[c], cpos * vox[0], 0.5 * L * vox[0]))
        ends = [l[-1] for l in lines]
        results[name] = (img_r, img_rr, img_s, lines, marks)
        if marks:
            mk = np.array(marks)
            print(f"   {name}: {len(mk)} columns reach the facing WM; contact at "
                  f"{np.median(mk[:,1]):.2f} mm, midline at {np.median(mk[:,2]):.2f} mm; "
                  f"median contact - midline {np.median(mk[:,1]-mk[:,2]):+.2f} mm")
        print(f"{name}: {len(seeds)} seeds, {status}")

    sl = det[2][:, :, kk]
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.5),
                            gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
    for r, (name, (img_r, img_rr, img_s, lines, marks)) in enumerate(results.items()):
        ax = axs[r, 0]
        ax.imshow(np.where(ctx[box][:, :, kk], sl, np.nan).T, origin="lower", cmap="gray",
                  interpolation="nearest")
        ax.contour(shell[:, :, kk].T, levels=[0.5], colors="y", linewidths=0.6)
        for pts in lines:
            ax.plot(pts[:, 0], pts[:, 1], "c-", lw=0.6)
        ax.plot(path[:, 0], path[:, 1], "y-", lw=1.5)
        ax.set_title(f"{name}: streamlines (in-plane projection)")
        ax.set_xlim(path[:, 0].min() - 25, path[:, 0].max() + 25)
        ax.set_ylim(path[:, 1].min() - 25, path[:, 1].max() + 25)
        ext = [0, arc[-1], 0, 1]
        v = np.nanpercentile(np.abs(img_r), 98)
        axs[r, 1].imshow(img_r, origin="lower", aspect="auto", cmap="gray", vmin=-v, vmax=v, extent=ext)
        axs[r, 1].set_title(f"{name}: echo 3, trend removed, by relative depth")
        rv = np.nanpercentile(img_rr, [2, 98])
        axs[r, 2].imshow(img_rr, origin="lower", aspect="auto", cmap="magma", vmin=rv[0], vmax=rv[1], extent=ext)
        axs[r, 2].set_title(f"{name}: R2* proxy (s⁻¹), by relative depth")
        axs[r, 3].imshow(img_s, origin="lower", aspect="auto", cmap="gray", vmin=-v, vmax=v,
                         extent=[0, arc[-1], 0, img_s.shape[0] * vox[0]])
        if marks:
            mk = np.array(marks)
            axs[r, 3].plot(mk[:, 0], mk[:, 1], "o", mfc="none", mec="c", ms=6, label="darkest point (contact?)")
            axs[r, 3].plot(mk[:, 0], mk[:, 2], "x", c="orange", ms=6, label="midline between WM surfaces")
            axs[r, 3].legend(loc="upper left", fontsize=8)
        axs[r, 3].set_title(f"{name}: echo 3, trend removed, by depth (mm)")
        for c in (1, 2):
            axs[r, c].set_ylabel("relative depth (WM 0 → pia 1)")
        axs[r, 3].set_ylabel("depth from WM (mm)")
        for c in (1, 2, 3):
            axs[r, c].set_xlabel("position along the WM boundary (mm)")
    fig.suptitle("V1V2, slice k = 27, fused stretch between (32, 78) and (49, 114): "
                 "streamlines run from one bank's WM to fluid or across the contact to the facing WM", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    run(sys.argv[1])
