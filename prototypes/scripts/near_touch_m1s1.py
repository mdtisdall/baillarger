"""Almost-touching banks (M1S1, plane i = 84 crop, 85 in ITK-SNAP): ground truth from the gap.

Bank A (1): crop (84, 99, 97) to (84, 122, 113); bank B (2): (84, 114, 63) to (84, 143, 86),
snapped to the WM boundary in the plane. All computed in 3D.

1. Truth: each bank's T fitted with the gap as fluid; its columns run to fluid, giving each
   column's true thickness (own pia across the gap).
2. Closed gap: the fluid between the two walls (within 4 mm of both) is relabelled as cortex,
   both fields are refitted, and each bank's columns run on past equal arrival. Contact rules
   are compared with each column's true top:
     equal arrival T_X = T_Y
     calibrated arrival T_X / t_X = T_Y / t_Y, t = the bank's median true thickness over its
       columns more than 1 mm away along the bank (thickness carried from neighbours)
     landmark -> top (distance and equivolume), fraction from the same neighbours.
"""

import collections
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
PLANE = 84
BANKS = {"A": [(84, 99, 97), (84, 122, 113)], "B": [(84, 114, 63), (84, 143, 86)]}
LAM, STEP, MARGIN = 0.1, 0.25, 1.6


def main():
    p = load_patch("M1S1")
    rim, vox = p.rim, p.voxel_mm
    vv = np.array(vox)
    miss = missing_tissue(rim, p.flash, vox)
    allp = np.array(BANKS["A"] + BANKS["B"])
    lo = np.array([PLANE - 30, allp[:, 1].min() - 22, allp[:, 2].min() - 22]).clip(0)
    hi = np.minimum(
        np.array([PLANE + 31, allp[:, 1].max() + 23, allp[:, 2].max() + 23]), rim.shape
    )
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    rb0, missb = rim[box].copy(), miss[box]
    wm = rb0 == 2
    shell = wm & ~ndi.binary_erosion(wm)
    ip = PLANE - lo[0]
    # snap the user's points to the WM boundary within the plane
    jj_, kk_ = np.nonzero(shell[ip])
    snapped = {}
    for X, pts in BANKS.items():
        sp = []
        for q in pts:
            d = np.hypot(jj_ - (q[1] - lo[1]), kk_ - (q[2] - lo[2]))
            t = d.argmin()
            sp.append((ip, jj_[t], kk_[t]))
            print(
                f"bank {X} point {q} (crop) -> WM boundary {(PLANE, jj_[t] + lo[1], kk_[t] + lo[2])}, {d[t]:.1f} voxels away"
            )
        snapped[X] = sp
    ctx0 = (rb0 == 3) & ~missb
    mb0 = ctx0.astype(float)
    e = {i: np.log(np.maximum(p.echoes[box + (i,)].astype(float), 1.0)) for i in (1, 2)}
    det = {i: (L - F.mconv(L, mb0, F.SHP)) * mb0 for i, L in e.items()}
    disp = det[1]
    Tc = F.tensor(det[1], mb0) + F.tensor(det[2], mb0)
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
    d_out = ndi.distance_transform_edt(~wm, sampling=vox)
    gout = np.stack(np.gradient(ndi.gaussian_filter(d_out, 1.0), *vox), -1)
    gout /= np.linalg.norm(gout, axis=-1, keepdims=True) + 1e-12

    paths, walls = {}, {}
    for X in ("A", "B"):
        a, b = snapped[X]
        p2 = F.boundary_path(shell[ip], (a[1], a[2]), (b[1], b[2]))
        path = np.stack([np.full(len(p2), ip), p2[:, 0], p2[:, 1]], 1)
        mn = gout[tuple(path.T)].mean(0)
        mn /= np.linalg.norm(mn)
        sm = np.zeros(rb0.shape, bool)
        sm[tuple(path.T)] = True
        dpath = ndi.distance_transform_edt(~sm, sampling=vox)
        walls[X] = shell & (dpath < 3.0) & ((gout * mn).sum(-1) > 0.3)
        paths[X] = path
    both = walls["A"] & walls["B"]
    for X in walls:
        walls[X] &= ~both
    dA = ndi.distance_transform_edt(~walls["A"], sampling=vox)
    dB = ndi.distance_transform_edt(~walls["B"], sampling=vox)
    gap = (rb0 == 1) & (dA < 4.0) & (dB < 4.0)
    print(
        f"walls A {walls['A'].sum()}, B {walls['B'].sum()} voxels; gap voxels {gap.sum()}"
    )

    def fit(ctx):
        out = {}
        for X in ("A", "B"):
            wall = walls[X]
            dom = ctx | wall
            dW = dA if X == "A" else dB
            g = np.stack(np.gradient(ndi.gaussian_filter(dW, 1.0), *vox), -1)
            gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
            sg = np.sign((nh * gh).sum(-1))
            sg[sg == 0] = 1
            cw = np.where(ctx0, coh, 0.0)  # no data weight inside the (closed) gap
            dirv = cw[..., None] * nh * sg[..., None] + LAM * gh
            dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
            T, info = solve_field(dom, wall, [dirv[..., a] for a in range(3)], vox)
            Tf = fill(T)
            gT = np.stack(np.gradient(Tf, *vox), -1)
            n1 = gT / (np.linalg.norm(gT, axis=-1, keepdims=True) + 1e-12)
            n1s = np.stack([ndi.gaussian_filter(n1[..., a], 1.0) for a in range(3)], -1)
            n1n = n1s / (np.linalg.norm(n1s, axis=-1, keepdims=True) + 1e-12)
            div = sum(np.gradient(n1n[..., a], vox[a], axis=a) for a in range(3))
            div = np.clip(np.where(ctx & (Tf > 0.3), div, 0.0), -2, 2)
            out[X] = dict(T=T, Tf=Tf, g=[gT[..., a] for a in range(3)], div=div)
        return out

    def trace(seed, fX, fY, rb, stop_fluid):
        x = np.array(seed, float)
        g0 = np.array([F.interp(c, x) for c in fX["g"]])
        x = x + g0 / (np.linalg.norm(g0) + 1e-12)
        pts = [np.array(seed, float), x.copy()]
        s, s_eq = 0.16, None
        for _ in range(320):
            v = np.array([F.interp(c, x) for c in fX["g"]])
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                return np.array(pts), "stall"
            x = x + STEP * v / nv
            s += STEP * 0.16
            pts.append(x.copy())
            q = tuple(np.clip(np.round(x).astype(int), 0, np.array(rb.shape) - 1))
            if rb[q] == 1:
                return np.array(pts), "fluid"
            if fY is not None:
                if s_eq is None and F.interp(fX["Tf"], x) >= F.interp(fY["Tf"], x):
                    s_eq = s
                if s_eq is not None and s >= s_eq + MARGIN:
                    return np.array(pts), "past"
            if wm[q] and s > 1.0:
                return np.array(pts), "WM"
        return np.array(pts), "length"

    def along(pts, fX):
        sx = (
            np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0) * vv, axis=1))]
            + 0.08
        )
        ev = ndi.gaussian_filter1d(ndi.map_coordinates(disp, pts.T, order=1), 1.5)
        ds = np.diff(sx, prepend=sx[0])
        u = np.cumsum(ndi.map_coordinates(fX["div"], pts.T, order=1) * ds)
        return sx, ev, np.cumsum(np.exp(u) * ds)

    # ---- 1. truth with the gap as fluid
    f0 = fit(ctx0)
    truth = {}
    for X in ("A", "B"):
        rows = []
        for sd in paths[X]:
            pts, st = trace(sd, f0[X], None, rb0, True)
            if st != "fluid":
                rows.append(None)
                continue
            sx, ev, v = along(pts, f0[X])
            q = tuple(
                np.clip(np.round(pts[-1]).astype(int), 0, np.array(rb0.shape) - 1)
            )
            rows.append(
                dict(
                    t=sx[-1], v=v[-1], sx=sx, ev=ev, vv=v, in_gap=bool(gap[q]), pts=pts
                )
            )
        truth[X] = rows
        ok = [r for r in rows if r is not None]
        print(
            f"truth bank {X}: {len(ok)} of {len(rows)} columns reach fluid; {sum(r['in_gap'] for r in ok)} into the gap; "
            f"thickness median {np.median([r['t'] for r in ok]):.2f} mm (range {min(r['t'] for r in ok):.2f}-{max(r['t'] for r in ok):.2f})"
        )

    arc = {
        X: np.r_[0, np.cumsum(np.linalg.norm(np.diff(paths[X], axis=0) * vv, axis=1))]
        for X in paths
    }

    def neighbours(X, c, key):
        vals = [
            r[key]
            for c2, r in enumerate(truth[X])
            if r is not None and r["in_gap"] and abs(arc[X][c2] - arc[X][c]) > 1.0
        ]
        return np.median(vals) if vals else np.nan

    def frac(X, c):
        fu, fe = [], []
        for c2, r in enumerate(truth[X]):
            if r is None or not r["in_gap"] or abs(arc[X][c2] - arc[X][c]) <= 1.0:
                continue
            m = (r["sx"] >= 0.2 * r["t"]) & (r["sx"] <= 0.7 * r["t"])
            k_ = np.flatnonzero(m)[np.argmin(r["ev"][m])]
            fu.append(r["sx"][k_] / r["t"])
            fe.append(r["vv"][k_] / r["v"])
        return (np.median(fu), np.median(fe)) if fu else (np.nan, np.nan)

    # ---- 2. close the gap
    ctx1 = ctx0 | gap
    rb1 = rb0.copy()
    rb1[gap] = 3
    f1 = fit(ctx1)
    results = {}
    for X, Y in (("A", "B"), ("B", "A")):
        tY = np.nanmedian([r["t"] for r in truth[Y] if r is not None and r["in_gap"]])
        out = []
        for c, sd in enumerate(paths[X]):
            tr = truth[X][c]
            if tr is None or not tr["in_gap"]:
                out.append(None)
                continue
            pts, st = trace(sd, f1[X], f1[Y], rb1, False)
            if st != "past":
                out.append(None)
                continue
            sx, ev, v = along(pts, f1[X])
            TX = ndi.map_coordinates(f1[X]["Tf"], pts.T, order=1)
            TY = ndi.map_coordinates(f1[Y]["Tf"], pts.T, order=1)
            tX = neighbours(X, c, "t")
            ieq = np.flatnonzero(TX >= TY)
            ical = np.flatnonzero(TX / tX >= TY / tY)
            fu, fe = frac(X, c)
            m = (sx >= 0.2 * tX) & (sx <= 0.7 * tX)
            k_ = np.flatnonzero(m)[np.argmin(ev[m])] if m.any() else None
            tgt = v[k_] / fe if k_ is not None else np.nan
            out.append(
                dict(
                    true=tr["t"],
                    eq=sx[ieq[0]] if len(ieq) else np.nan,
                    cal=sx[ical[0]] if len(ical) else np.nan,
                    lm_us=sx[k_] / fu if k_ is not None else np.nan,
                    lm_ev=np.interp(tgt, v, sx)
                    if k_ is not None and tgt <= v[-1]
                    else np.nan,
                    nb=tX,
                    sx=sx,
                    ev=ev,
                    pts=pts,
                )
            )
        results[X] = out
        ok = [r for r in out if r is not None]
        print(
            f"closed gap, bank {X}: {len(ok)} columns compared with their true top (true median {np.median([r['true'] for r in ok]):.2f} mm)"
        )
        for k_, lab in (
            ("eq", "equal arrival"),
            ("cal", "thickness-calibrated arrival"),
            ("lm_us", "landmark -> top, distance"),
            ("lm_ev", "landmark -> top, equivolume"),
            ("nb", "neighbours' thickness alone"),
        ):
            d = np.array([r[k_] - r["true"] for r in ok])
            print(
                f"   {lab:30s} - true top: median {np.nanmedian(d):+.2f} mm, IQR {np.subtract(*np.nanpercentile(d, [75, 25])):.2f}, "
                f"|offset| {np.nanmedian(np.abs(d)):.2f}"
            )

    # ---- figure
    fig = plt.figure(figsize=(26, 13))
    gsp = fig.add_gridspec(2, 3)
    sl = (ip, slice(None), slice(None))
    raw = e[1]
    rv = np.nanpercentile(raw[ctx0], [1, 99.5])
    allpath = np.vstack([paths["A"], paths["B"]])
    xr = (allpath[:, 1].min() - 14, allpath[:, 1].max() + 14)
    yr = (allpath[:, 2].min() - 14, allpath[:, 2].max() + 14)
    for c_, (title, ff, rbx, key) in enumerate(
        (
            ("truth: gap as fluid, each bank's columns to its own pia", f0, rb0, "t"),
            (
                "gap closed: equal arrival (lime), calibrated arrival (magenta)",
                f1,
                rb1,
                "c",
            ),
        )
    ):
        ax = fig.add_subplot(gsp[0, c_])
        ax.imshow(raw[sl].T, origin="lower", cmap="gray", vmin=rv[0], vmax=rv[1])
        ax.contour(wm[sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.7)
        ax.contour(
            (rb0 == 1)[sl].T.astype(float), levels=[0.5], colors="c", linewidths=0.7
        )
        ax.imshow(
            np.where(gap[sl], 1.0, np.nan).T,
            origin="lower",
            cmap="cool",
            alpha=0.5,
            vmin=0,
            vmax=1,
        )
        for X, col in (("A", "#4a90d9"), ("B", "#e8913a")):
            ax.plot(paths[X][:, 1], paths[X][:, 2], "-", c=col, lw=2.2)
            rows = truth[X] if c_ == 0 else results[X]
            for r in rows:
                if r is not None:
                    ax.plot(r["pts"][:, 1], r["pts"][:, 2], "-", c=col, lw=0.6)
        if c_ == 1:
            tA = np.nanmedian(
                [r["t"] for r in truth["A"] if r is not None and r["in_gap"]]
            )
            tB = np.nanmedian(
                [r["t"] for r in truth["B"] if r is not None and r["in_gap"]]
            )
            ax.contour(
                np.where(ctx1, f1["A"]["T"] - f1["B"]["T"], np.nan)[sl].T,
                levels=[0],
                colors="lime",
                linewidths=1.3,
            )
            ax.contour(
                np.where(ctx1, f1["A"]["T"] / tA - f1["B"]["T"] / tB, np.nan)[sl].T,
                levels=[0],
                colors="m",
                linewidths=1.3,
            )
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            title
            + "\nraw echo 2; yellow WM, cyan fluid, pink = the gap closed in the test",
            fontsize=10,
        )
    ax = fig.add_subplot(gsp[0, 2])
    ax.axis("off")
    rows_t = []
    for X in ("A", "B"):
        ok = [r for r in results[X] if r is not None]
        for k_, lab in (
            ("eq", "equal arrival"),
            ("cal", "calibrated arrival"),
            ("lm_us", "landmark, distance"),
            ("lm_ev", "landmark, equivolume"),
            ("nb", "neighbours' thickness"),
        ):
            d = np.array([r[k_] - r["true"] for r in ok])
            rows_t.append(
                [
                    f"bank {X}",
                    lab,
                    f"{np.nanmedian(d):+.2f}",
                    f"{np.subtract(*np.nanpercentile(d, [75, 25])):.2f}",
                    f"{np.nanmedian(np.abs(d)):.2f}",
                ]
            )
    tb = ax.table(
        cellText=rows_t,
        colLabels=[
            "bank",
            "rule",
            "median offset (mm)",
            "IQR (mm)",
            "median |offset| (mm)",
        ],
        loc="center",
        cellLoc="center",
    )
    tb.scale(1, 2.0)
    tb.set_fontsize(10)
    ax.set_title("each rule against each column's true top (gap closed)")
    v = np.nanpercentile(np.abs(disp[ctx0]), 98)
    for c_, X in enumerate(("A", "B")):
        ax = fig.add_subplot(gsp[1, c_])
        out = results[X]
        nb = 70
        img = np.full((nb, len(out)), np.nan)
        for c, r in enumerate(out):
            if r is None:
                continue
            q = np.arange(nb) * 0.08 + 0.04
            okq = q <= r["sx"][-1]
            img[okq, c] = np.interp(q[okq], r["sx"], r["ev"])
        ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            cmap="gray",
            vmin=-v,
            vmax=v,
            extent=[0, arc[X][-1], 0, nb * 0.08],
        )
        idx = [c for c, r in enumerate(out) if r is not None]
        xs = arc[X][idx]
        g = lambda k: np.array([out[c][k] for c in idx])
        ax.plot(
            xs,
            g("true"),
            "s",
            mfc="none",
            mec="y",
            ms=7,
            label="true top (own pia across the gap)",
        )
        ax.plot(xs, g("eq"), "_", c="lime", ms=12, mew=2, label="equal arrival")
        ax.plot(
            xs, g("cal"), "_", c="m", ms=12, mew=2, label="thickness-calibrated arrival"
        )
        ax.plot(
            xs, g("lm_ev"), "v", c="#4a90d9", ms=5, label="landmark → top (equivolume)"
        )
        ax.plot(
            xs, g("lm_us"), "^", c="#e8913a", ms=5, label="landmark → top (distance)"
        )
        ax.set_xlabel(f"position along bank {X}'s WM path (mm)")
        ax.set_ylabel(f"distance from bank {X}'s WM (mm)")
        ax.set_title(
            f"bank {X}, gap closed: echo 2 (trend removed) along its own columns"
        )
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Almost-touching banks, M1S1 plane i = 85 (ITK-SNAP numbering): truth from the visible gap, then the gap "
        "closed; all computed in 3D",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(f"{S}/near_touch_m1s1.png", dpi=90)
    print("wrote", f"{S}/near_touch_m1s1.png")


if __name__ == "__main__":
    main()
