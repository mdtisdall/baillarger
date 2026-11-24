"""Per-bank fronts across a fused contact (V1V2, plane j = 91, the user's four points).

Each bank's WM wall is labelled in 3D (WM-boundary voxels within 3 mm of that bank's
in-plane boundary path and facing the same way). T_A is fitted with T = 0 on A's wall
only, steered by the data normal signed away from A everywhere (no ridge); T_B likewise.
A's columns follow grad T_A from A's wall until they reach B's wall (or fluid); B's
follow grad T_B. Along A's columns: echo 2 (trend removed), each bank's stria, the
observed contact (darkest point between them), equal arrival (T_A = T_B), and each
bank's top predicted from its stria by distance (fraction 0.40) or equivolume (0.48),
both calibrated on open V1. Everything is computed in 3D; the plane is for display.
"""

import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import flatten_fused2 as F
from baillarger.data import load_patch
from baillarger.frame import missing_tissue
from fit_T import fill, solve_field

S = sys.argv[1]
A_PTS = [(71, 91, 18), (67, 91, 47)]
B_PTS = [(38, 91, 50), (42, 91, 5)]
J = 91
F_US, F_EV = 0.403, 0.480
LAM, STEP = 0.1, 0.25


def main():
    p = load_patch("V1V2")
    rim, vox = p.rim, p.voxel_mm
    vv = np.array(vox)
    miss = missing_tissue(rim, p.flash, vox)
    allp = np.array(A_PTS + B_PTS)
    lo = np.array([allp[:, 0].min() - 22, J - 30, allp[:, 2].min() - 22]).clip(0)
    hi = np.minimum(
        np.array([allp[:, 0].max() + 23, J + 31, allp[:, 2].max() + 23]), rim.shape
    )
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    rb, missb = rim[box], miss[box]
    ctx = (rb == 3) & ~missb
    mb = ctx.astype(float)
    wm = rb == 2
    shell = wm & ~ndi.binary_erosion(wm)
    for q in A_PTS + B_PTS:
        print(q, "on WM boundary:", bool(shell[tuple(np.array(q) - lo)]))
    e = {i: np.log(np.maximum(p.echoes[box + (i,)].astype(float), 1.0)) for i in (1, 2)}
    det = {i: (L - F.mconv(L, mb, F.SHP)) * mb for i, L in e.items()}
    Tc = F.tensor(det[1], mb) + F.tensor(det[2], mb)
    Mt = np.stack(
        [
            np.stack([Tc[0], Tc[1], Tc[2]], -1),
            np.stack([Tc[1], Tc[3], Tc[4]], -1),
            np.stack([Tc[2], Tc[4], Tc[5]], -1),
        ],
        -2,
    )
    w_, V = np.linalg.eigh(Mt)
    nh = V[..., :, 2]
    coh = np.clip(
        (w_[..., 2] - w_[..., 1]) / np.maximum(w_[..., 2] + w_[..., 1], 1e-12), 0, 1
    )
    disp = det[1]
    jj = J - lo[1]
    d_out = ndi.distance_transform_edt(~wm, sampling=vox)
    gout = np.stack(np.gradient(ndi.gaussian_filter(d_out, 1.0), *vox), -1)
    gout /= np.linalg.norm(gout, axis=-1, keepdims=True) + 1e-12

    def bank(pts):
        a = (pts[0][0] - lo[0], pts[0][2] - lo[2])
        b = (pts[1][0] - lo[0], pts[1][2] - lo[2])
        path2 = F.boundary_path(shell[:, jj, :], a, b)  # (i, k) pairs in the plane
        path = np.stack([path2[:, 0], np.full(len(path2), jj), path2[:, 1]], 1)
        mean_n = gout[tuple(path.T)].mean(0)
        mean_n /= np.linalg.norm(mean_n)
        seedmask = np.zeros(rb.shape, bool)
        seedmask[tuple(path.T)] = True
        dpath = ndi.distance_transform_edt(~seedmask, sampling=vox)
        wall = shell & (dpath < 3.0) & ((gout * mean_n).sum(-1) > 0.3)
        return path, wall, mean_n

    pathA, wallA, nA = bank(A_PTS)
    pathB, wallB, nB = bank(B_PTS)
    both = wallA & wallB
    wallA &= ~both
    wallB &= ~both
    print(
        f"walls: A {wallA.sum()} voxels, B {wallB.sum()}, overlap removed {both.sum()}; "
        f"A and B normals dot {np.dot(nA, nB):+.2f} (facing each other if negative)"
    )

    fields = {}
    for nm, wall in (("A", wallA), ("B", wallB)):
        dom = ctx | wall
        dW = ndi.distance_transform_edt(~wall, sampling=vox)
        g = np.stack(np.gradient(ndi.gaussian_filter(dW, 1.0), *vox), -1)
        gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
        s = np.sign((nh * gh).sum(-1))
        s[s == 0] = 1
        dirv = coh[..., None] * nh * s[..., None] + LAM * gh
        dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
        T, info = solve_field(dom, wall, [dirv[..., a] for a in range(3)], vox)
        Tf = fill(T)
        gT = np.stack(np.gradient(Tf, *vox), -1)
        n1 = gT / (np.linalg.norm(gT, axis=-1, keepdims=True) + 1e-12)
        n1s = np.stack([ndi.gaussian_filter(n1[..., a], 1.0) for a in range(3)], -1)
        n1n = n1s / (np.linalg.norm(n1s, axis=-1, keepdims=True) + 1e-12)
        div = sum(np.gradient(n1n[..., a], vox[a], axis=a) for a in range(3))
        div = np.clip(np.where(ctx & (Tf > 0.3), div, 0.0), -2, 2)
        fields[nm] = dict(T=T, Tf=Tf, g=[gT[..., a] for a in range(3)], div=div)
        print(f"T_{nm}: cg info {info}, median {np.nanmedian(T[ctx]):.2f}")

    def trace(seed, fld, target):
        x = np.array(seed, float)
        g0 = np.array([F.interp(c, x) for c in fld["g"]])
        x = x + g0 / (np.linalg.norm(g0) + 1e-12)
        pts = [np.array(seed, float), x.copy()]
        for _ in range(240):
            v = np.array([F.interp(c, x) for c in fld["g"]])
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                return np.array(pts), "stall"
            x = x + STEP * v / nv
            pts.append(x.copy())
            q = tuple(np.clip(np.round(x).astype(int), 0, np.array(rb.shape) - 1))
            if target[q] or (wm[q] and len(pts) > 12):
                return np.array(pts), "other WM" if not target[q] else "facing wall"
            if rb[q] == 1:
                return np.array(pts), "fluid"
        return np.array(pts), "length"

    recs, linesA, linesB = [], [], []
    for sd in pathA:
        pts, st = trace(sd, fields["A"], wallB)
        linesA.append((pts, st))
        if st != "facing wall":
            recs.append(None)
            continue
        s = (
            np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0) * vv, axis=1))]
            + 0.08
        )
        L = s[-1] + 0.08
        ev = ndi.gaussian_filter1d(ndi.map_coordinates(disp, pts.T, order=1), 1.5)
        TA = ndi.map_coordinates(fields["A"]["Tf"], pts.T, order=1)
        TB = ndi.map_coordinates(fields["B"]["Tf"], pts.T, order=1)
        sb = L - s
        mA = (s >= 0.5) & (s <= 1.6)
        mB = (sb >= 0.5) & (sb <= 1.6)
        if not mA.any() or not mB.any():
            recs.append(None)
            continue
        kA = np.flatnonzero(mA)[np.argmin(ev[mA])]
        kB = np.flatnonzero(mB)[np.argmin(ev[mB])]
        mid = (s > s[kA] + 0.4) & (s < s[kB] - 0.4)
        kc = np.flatnonzero(mid)[np.argmin(ev[mid])] if mid.any() else None
        d = TA - TB
        cross = np.flatnonzero(np.diff(np.sign(d)) != 0)
        eq = s[cross[0]] if len(cross) else np.nan
        ds = np.diff(s, prepend=s[0])
        uA = np.cumsum(ndi.map_coordinates(fields["A"]["div"], pts.T, order=1) * ds)
        vA = np.cumsum(np.exp(uA) * ds)
        dB = ndi.map_coordinates(fields["B"]["div"], pts.T, order=1)[::-1]
        dsb = np.diff(sb[::-1], prepend=sb[-1])
        uB = np.cumsum(dB * dsb)
        vB = np.cumsum(np.exp(uB) * dsb)
        kBr = len(s) - 1 - kB
        topA_ev = np.interp(vA[kA] / F_EV, vA, s) if vA[kA] / F_EV <= vA[-1] else np.nan
        topB_ev = (
            L - np.interp(vB[kBr] / F_EV, vB, sb[::-1])
            if vB[kBr] / F_EV <= vB[-1]
            else np.nan
        )
        recs.append(
            dict(
                s=s,
                e=ev,
                L=L,
                sA=s[kA],
                sB=s[kB],
                contact=s[kc] if kc is not None else np.nan,
                eq=eq,
                topA_us=s[kA] / F_US,
                topB_us=L - (L - s[kB]) / F_US,
                topA_ev=topA_ev,
                topB_ev=topB_ev,
            )
        )
    for sd in pathB:
        linesB.append(trace(sd, fields["B"], wallA))
    import collections

    print("A columns:", collections.Counter(st for _, st in linesA))
    print("B columns:", collections.Counter(st for _, st in linesB))
    ok = [r for r in recs if r is not None]
    print(f"{len(ok)} of {len(recs)} A columns reach B's wall with both strias found")
    if ok:
        K = {
            k: np.array([r[k] for r in ok])
            for k in (
                "L",
                "contact",
                "eq",
                "topA_us",
                "topB_us",
                "topA_ev",
                "topB_ev",
                "sA",
                "sB",
            )
        }
        print(
            f"  WM-to-WM median {np.median(K['L']):.2f} mm; contact at {np.nanmedian(K['contact']):.2f} mm from A"
        )
        for nm in ("eq", "topA_us", "topB_us", "topA_ev", "topB_ev"):
            print(
                f"  {nm} - contact: median {np.nanmedian(K[nm] - K['contact']):+.2f} mm, "
                f"IQR {np.subtract(*np.nanpercentile(K[nm] - K['contact'], [75, 25])):.2f}"
            )

    # ---------- figure
    arcA = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pathA, axis=0) * vv, axis=1))]
    fig = plt.figure(figsize=(26, 15))
    gs = fig.add_gridspec(2, 4)
    sl = (slice(None), jj, slice(None))
    v = np.nanpercentile(np.abs(disp[ctx]), 98)
    xr = (
        min(pathA[:, 0].min(), pathB[:, 0].min()) - 12,
        max(pathA[:, 0].max(), pathB[:, 0].max()) + 12,
    )
    zr = (
        min(pathA[:, 2].min(), pathB[:, 2].min()) - 12,
        max(pathA[:, 2].max(), pathB[:, 2].max()) + 12,
    )

    def base(ax, title):
        ax.imshow(
            np.where(ctx, disp, np.nan)[sl].T,
            origin="lower",
            cmap="gray",
            vmin=-v,
            vmax=v,
        )
        ax.contour(shell[sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.6)
        ax.set_xlim(*xr)
        ax.set_ylim(*zr)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_xlabel("i")
        ax.set_ylabel("k")

    ax = fig.add_subplot(gs[0, 0])
    base(
        ax,
        f"walls (3D, this plane): A blue, B orange\nplane j = {J + 1} in ITK-SNAP numbering",
    )
    ax.imshow(
        np.where(wallA[sl], 1.0, np.nan).T,
        origin="lower",
        cmap="Blues",
        vmin=0,
        vmax=1.2,
    )
    ax.imshow(
        np.where(wallB[sl], 1.0, np.nan).T,
        origin="lower",
        cmap="Oranges",
        vmin=0,
        vmax=1.2,
    )
    ax.plot(pathA[:, 0], pathA[:, 2], "-", c="#1f5fbf", lw=2)
    ax.plot(pathB[:, 0], pathB[:, 2], "-", c="#d9731f", lw=2)
    for c_, (nm, lines, col) in enumerate(
        (("A", linesA, "#4a90d9"), ("B", linesB, "#e8913a"))
    ):
        ax = fig.add_subplot(gs[0, 1 + c_])
        base(
            ax,
            f"bank {nm}'s columns along grad T_{nm} (3D, projected): "
            + ", ".join(
                f"{k} {n}"
                for k, n in collections.Counter(st for _, st in lines).items()
            ),
        )
        for pts, st in lines:
            ax.plot(
                pts[:, 0], pts[:, 2], "-", c=col if st == "facing wall" else "r", lw=0.7
            )
    ax = fig.add_subplot(gs[0, 3])
    base(ax, "T_A (blue) and T_B (orange) every 0.5;\ngreen = equal arrival T_A = T_B")
    TA, TB = fields["A"]["T"], fields["B"]["T"]
    ax.contour(
        np.where(ctx, TA, np.nan)[sl].T,
        levels=np.arange(0.5, 8, 0.5),
        colors="#4a90d9",
        linewidths=0.6,
    )
    ax.contour(
        np.where(ctx, TB, np.nan)[sl].T,
        levels=np.arange(0.5, 8, 0.5),
        colors="#e8913a",
        linewidths=0.6,
    )
    ax.contour(
        np.where(ctx, TA - TB, np.nan)[sl].T, levels=[0], colors="lime", linewidths=1.5
    )
    nb = 80
    img = np.full((nb, len(recs)), np.nan)
    for c, r in enumerate(recs):
        if r is None:
            continue
        q = np.arange(nb) * 0.08 + 0.04
        okq = q <= r["s"][-1]
        img[okq, c] = np.interp(q[okq], r["s"], r["e"])
    idx = [c for c, r in enumerate(recs) if r is not None]
    for c_, (nm, a_, b_) in enumerate(
        (("distance", "topA_us", "topB_us"), ("equivolume", "topA_ev", "topB_ev"))
    ):
        ax = fig.add_subplot(gs[1, 2 * c_ : 2 * c_ + 2])
        ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            cmap="gray",
            vmin=-v,
            vmax=v,
            extent=[0, arcA[-1], 0, nb * 0.08],
        )
        if idx:
            xs = arcA[idx]
            ax.plot(
                xs,
                K["contact"],
                "o",
                mfc="none",
                mec="c",
                ms=7,
                label="observed contact (darkest point)",
            )
            ax.plot(
                xs,
                K["eq"],
                "_",
                c="lime",
                ms=12,
                mew=2,
                label="equal arrival T_A = T_B",
            )
            ax.plot(
                xs,
                K[a_],
                "v",
                c="#4a90d9",
                ms=6,
                label="A's top predicted from its stria",
            )
            ax.plot(
                xs,
                K[b_],
                "^",
                c="#e8913a",
                ms=6,
                label="B's top predicted from its stria",
            )
            ax.plot(xs, K["sA"], ".", c="#4a90d9", ms=5, label="A's stria")
            ax.plot(xs, K["sB"], ".", c="#e8913a", ms=5, label="B's stria")
            ax.plot(xs, K["L"], "-", c="w", lw=0.8, label="B's wall")
        ax.set_xlabel("position along bank A's WM path (mm)")
        ax.set_ylabel("distance from A's WM along A's column (mm)")
        ax.set_title(
            f"{nm}: echo 2 (trend removed) along bank A's columns, with contact rules"
        )
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Per-bank fronts across the fused contact, V1V2 plane j = 92 (ITK-SNAP numbering); all computed in 3D",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(f"{S}/per_bank_test.png", dpi=90)
    print("wrote", f"{S}/per_bank_test.png")


if __name__ == "__main__":
    main()
