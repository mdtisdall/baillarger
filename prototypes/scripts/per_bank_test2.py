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
    hi = np.minimum(np.array([allp[:, 0].max() + 23, J + 31, allp[:, 2].max() + 23]), rim.shape)
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
    Mt = np.stack([np.stack([Tc[0], Tc[1], Tc[2]], -1), np.stack([Tc[1], Tc[3], Tc[4]], -1),
                   np.stack([Tc[2], Tc[4], Tc[5]], -1)], -2)
    w_, V = np.linalg.eigh(Mt)
    nh = V[..., :, 2]
    coh = np.clip((w_[..., 2] - w_[..., 1]) / np.maximum(w_[..., 2] + w_[..., 1], 1e-12), 0, 1)
    disp = det[1]
    jj = J - lo[1]
    d_out = ndi.distance_transform_edt(~wm, sampling=vox)
    gout = np.stack(np.gradient(ndi.gaussian_filter(d_out, 1.0), *vox), -1)
    gout /= np.linalg.norm(gout, axis=-1, keepdims=True) + 1e-12

    def bank(pts):
        a = (pts[0][0] - lo[0], pts[0][2] - lo[2]); b = (pts[1][0] - lo[0], pts[1][2] - lo[2])
        path2 = F.boundary_path(shell[:, jj, :], a, b)  # (i, k) pairs in the plane
        path = np.stack([path2[:, 0], np.full(len(path2), jj), path2[:, 1]], 1)
        mean_n = gout[tuple(path.T)].mean(0); mean_n /= np.linalg.norm(mean_n)
        seedmask = np.zeros(rb.shape, bool); seedmask[tuple(path.T)] = True
        dpath = ndi.distance_transform_edt(~seedmask, sampling=vox)
        wall = shell & (dpath < 3.0) & ((gout * mean_n).sum(-1) > 0.3)
        return path, wall, mean_n

    pathA, wallA, nA = bank(A_PTS)
    pathB, wallB, nB = bank(B_PTS)
    both = wallA & wallB
    wallA &= ~both; wallB &= ~both
    print(f"walls: A {wallA.sum()} voxels, B {wallB.sum()}, overlap removed {both.sum()}; "
          f"A and B normals dot {np.dot(nA, nB):+.2f} (facing each other if negative)")

    fields = {}
    for nm, wall in (("A", wallA), ("B", wallB)):
        dom = ctx | wall
        dW = ndi.distance_transform_edt(~wall, sampling=vox)
        g = np.stack(np.gradient(ndi.gaussian_filter(dW, 1.0), *vox), -1)
        gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
        s = np.sign((nh * gh).sum(-1)); s[s == 0] = 1
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

    MARGIN = 0.8

    def trace(seed, X, Y):
        """Own WM -> along grad T_X; stop MARGIN mm past equal arrival (T_X >= T_Y)."""
        fX, fY = fields[X], fields[Y]
        x = np.array(seed, float)
        g0 = np.array([F.interp(c, x) for c in fX["g"]]); x = x + g0 / (np.linalg.norm(g0) + 1e-12)
        pts = [np.array(seed, float), x.copy()]
        s, s_eq = 0.16, None
        for _ in range(240):
            v = np.array([F.interp(c, x) for c in fX["g"]])
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                return np.array(pts), "stall"
            x = x + STEP * v / nv
            s += STEP * 0.16
            pts.append(x.copy())
            q = tuple(np.clip(np.round(x).astype(int), 0, np.array(rb.shape) - 1))
            if s_eq is None and F.interp(fX["Tf"], x) >= F.interp(fY["Tf"], x):
                s_eq = s
            if s_eq is not None and s >= s_eq + MARGIN:
                return np.array(pts), "past contact"
            if rb[q] == 1:
                return np.array(pts), "fluid"
            if wm[q] and s > 1.0:
                return np.array(pts), "WM"
        return np.array(pts), "length"

    def measure(X, Y, path):
        out, lines = [], []
        for sd in path:
            pts, st = trace(sd, X, Y)
            lines.append((pts, st))
            if st != "past contact":
                out.append(None); continue
            sx = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0) * vv, axis=1))] + 0.08
            ev = ndi.gaussian_filter1d(ndi.map_coordinates(disp, pts.T, order=1), 1.5)
            TX = ndi.map_coordinates(fields[X]["Tf"], pts.T, order=1)
            TY = ndi.map_coordinates(fields[Y]["Tf"], pts.T, order=1)
            ieq = np.flatnonzero(TX >= TY)
            if not len(ieq):
                out.append(None); continue
            eq = sx[ieq[0]]
            m = (sx >= 0.5) & (sx <= 1.6)
            k_ = np.flatnonzero(m)[np.argmin(ev[m])]
            mc = sx >= sx[k_] + 0.4
            kc = np.flatnonzero(mc)[np.argmin(ev[mc])] if mc.any() else None
            ds = np.diff(sx, prepend=sx[0])
            u = np.cumsum(ndi.map_coordinates(fields[X]["div"], pts.T, order=1) * ds)
            vX = np.cumsum(np.exp(u) * ds)
            tgt = vX[k_] / F_EV
            top_ev = np.interp(tgt, vX, sx) if tgt <= vX[-1] else np.nan
            fr_us = sx[k_] / sx[kc] if kc is not None else np.nan
            fr_ev = vX[k_] / vX[kc] if kc is not None else np.nan
            edge = kc is None or sx[kc] > sx[-1] - 0.15
            out.append(dict(fr_us=fr_us, fr_ev=fr_ev, edge=edge, s=sx, e=ev, stria=sx[k_], contact=sx[kc] if kc is not None else np.nan,
                            cpt=pts[kc] if kc is not None else None, eq=eq, top_us=sx[k_] / F_US,
                            top_ev=top_ev, end=sx[-1]))
        return out, lines

    import collections
    res = {}
    for X, Y, path in (("A", "B", pathA), ("B", "A", pathB)):
        out, lines = measure(X, Y, path)
        res[X] = (out, lines, path)
        ok = [r for r in out if r is not None]
        print(f"bank {X}: columns {collections.Counter(st for _, st in lines)}; {len(ok)} measured")
        if ok:
            K = {k: np.array([r[k] for r in ok]) for k in ("stria", "contact", "eq", "top_us", "top_ev")}
            print(f"   stria at {np.median(K['stria']):.2f} mm, observed contact at {np.nanmedian(K['contact']):.2f} mm "
                  f"from bank {X}'s WM (IQR {np.subtract(*np.nanpercentile(K['contact'], [75, 25])):.2f})")
            for nm, lab in (("eq", "equal arrival"), ("top_us", "top from stria, distance"), ("top_ev", "top from stria, equivolume")):
                d = K[nm] - K["contact"]
                print(f"   {lab:28s} - contact: median {np.nanmedian(d):+.2f} mm, IQR {np.subtract(*np.nanpercentile(d, [75, 25])):.2f}, "
                      f"|offset| median {np.nanmedian(np.abs(d)):.2f}")
            good = [r for r in ok if not r["edge"]]
            fu = np.array([r["fr_us"] for r in good]); fe = np.array([r["fr_ev"] for r in good])
            print(f"   stria / contact fraction, {len(good)} columns with a contact inside the window: "
                  f"distance {np.median(fu):.3f} (IQR {np.subtract(*np.percentile(fu, [75, 25])):.3f}), "
                  f"equivolume {np.median(fe):.3f} (IQR {np.subtract(*np.percentile(fe, [75, 25])):.3f})")
            res[X] = (out, lines, path, K)

    # ---------- figure
    fig = plt.figure(figsize=(26, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    sl = (slice(None), jj, slice(None))
    v = np.nanpercentile(np.abs(disp[ctx]), 98)
    xr = (min(pathA[:, 0].min(), pathB[:, 0].min()) - 12, max(pathA[:, 0].max(), pathB[:, 0].max()) + 12)
    zr = (min(pathA[:, 2].min(), pathB[:, 2].min()) - 12, max(pathA[:, 2].max(), pathB[:, 2].max()) + 12)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.where(ctx, disp, np.nan)[sl].T, origin="lower", cmap="gray", vmin=-v, vmax=v)
    ax.contour(shell[sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.6)
    TA, TB = fields["A"]["T"], fields["B"]["T"]
    ax.contour(np.where(ctx, TA - TB, np.nan)[sl].T, levels=[0], colors="lime", linewidths=1.2)
    for X, col in (("A", "#4a90d9"), ("B", "#e8913a")):
        out, lines = res[X][0], res[X][1]
        for (pts, st) in lines:
            ax.plot(pts[:, 0], pts[:, 2], "-", c=col if st == "past contact" else "r", lw=0.7)
        cp = np.array([r["cpt"] for r in out if r is not None and r["cpt"] is not None])
        if len(cp):
            ax.plot(cp[:, 0], cp[:, 2], "o", mfc="none", mec="c", ms=4)
    ax.set_xlim(*xr); ax.set_ylim(*zr); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("each bank's columns from its own WM, stopped 0.8 mm past equal arrival\n"
                 "(A blue, B orange; red = ended elsewhere; green = equal arrival; cyan = observed contact)")
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    rows = []
    for X in ("A", "B"):
        if len(res[X]) < 4:
            continue
        K = res[X][3]
        for nm, lab in (("eq", "equal arrival"), ("top_us", "stria → top, distance"), ("top_ev", "stria → top, equivolume")):
            d = K[nm] - K["contact"]
            rows.append([f"bank {X}", lab, f"{np.nanmedian(d):+.2f}", f"{np.subtract(*np.nanpercentile(d, [75, 25])):.2f}",
                         f"{np.nanmedian(np.abs(d)):.2f}"])
    if rows:
        tb = ax.table(cellText=rows, colLabels=["bank", "rule", "median offset (mm)", "IQR (mm)", "median |offset| (mm)"],
                      loc="center", cellLoc="center")
        tb.scale(1, 2.2); tb.set_fontsize(11)
    ax.set_title("each rule's offset from the observed contact, measured on each bank's own columns")
    for c_, (X, metric) in enumerate((("A", None), ("B", None))):
        if len(res[X]) < 4:
            continue
        out, lines, path, K = res[X]
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0) * vv, axis=1))]
        nb = 50
        img = np.full((nb, len(out)), np.nan)
        for c, r in enumerate(out):
            if r is None:
                continue
            q = np.arange(nb) * 0.08 + 0.04
            okq = q <= r["s"][-1]
            img[okq, c] = np.interp(q[okq], r["s"], r["e"])
        idx = [c for c, r in enumerate(out) if r is not None]
        xs = arc[idx]
        for row, (nm, lab) in enumerate((("top_us", "distance"), ("top_ev", "equivolume"))):
            ax = fig.add_subplot(gs[row, 1 + c_])
            ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=-v, vmax=v, extent=[0, arc[-1], 0, nb * 0.08])
            ax.plot(xs, K["contact"], "o", mfc="none", mec="c", ms=7, label="observed contact")
            ax.plot(xs, K["eq"], "_", c="lime", ms=12, mew=2, label="equal arrival")
            ax.plot(xs, K[nm], "v", c="#4a90d9" if X == "A" else "#e8913a", ms=6, label=f"top from stria ({lab})")
            ax.plot(xs, K["stria"], ".", c="w", ms=5, label="stria")
            ax.set_xlabel(f"position along bank {X}'s WM path (mm)")
            ax.set_ylabel(f"distance from bank {X}'s WM along its own column (mm)")
            ax.set_title(f"bank {X}'s own columns, echo 2 (trend removed); stria → top by {lab}")
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Per-bank fronts, each bank measured on its own columns, V1V2 plane j = 92 (ITK-SNAP numbering); "
                 "all computed in 3D", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{S}/per_bank_test2.png", dpi=90)
    print("wrote", f"{S}/per_bank_test2.png")


if __name__ == "__main__":
    main()
