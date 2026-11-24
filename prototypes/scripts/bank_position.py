"""Bank-position diagnostic (assumptions A7, A8 of 'The contact as an inference problem').

For each front at the kissing stretch: thickness at its visible tops (arc length S along its own columns,
collide2.py) against the position of the top's root on the white surface:
  - geodesic distance along the white surface (same-side graph) to the front's fused zone: the roots,
    on that front's wall, of the all-WM sheet voxels (collide.py);
  - sulcal depth: distance of the root below a hull (tissue closed with a 4 mm ball; box faces are
    left open, so depth is not read within 6 mm of a face);
  - white-surface mean curvature H at the root (0.8 mm scale), for comparison.
Overlay: each front's arc length at the user's 23 points, placed at the position of the point's root.
A trend toward the fused zone that the points continue points to a missing position covariate; points
below a flat trend point to selection on thickness itself. Views only; no model change.

Usage: bank_position.py <outdir> <workdir>. Needs collide2_state.npz, collide_state.npz, curv_M1S1.npz.
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr

import curv_lib as C
from collide import GROW, shell_graph

OUT, S = sys.argv[1], sys.argv[2]
HULL_MM, FACE_MM = 4.0, 6.0


def main():
    m = json.load(open(f"{S}/M1S1_test_points.json"))
    lo = np.array(m["full_lo"]) - GROW
    hi = np.array(m["full_hi"]) + GROW
    B = C.load(lo, hi, "M1S1")
    vox = B["vox"]
    st = np.load(f"{S}/collide2_state.npz")
    cs = np.load(f"{S}/collide_state.npz")
    cz = np.load(f"{S}/curv_M1S1.npz")
    pts, nrm = cz["pts"], cz["n"]
    H = np.clip(0.5 * (cz["k1_0.8"] + cz["k2_0.8"]), -0.8, 0.8)
    P = np.round(np.array(m["gap_all"], float)).astype(int) + GROW
    G = shell_graph(pts, nrm, vox)
    # sulcal depth below a hull: tissue (WM, cortex, missing) closed with a 4 mm ball
    tissue = (B["rim"] == 2) | (B["rim"] == 3) | B["miss"]
    dil = ndi.distance_transform_edt(~tissue, sampling=vox) <= HULL_MM
    closed = ndi.distance_transform_edt(dil, sampling=vox) > HULL_MM
    depth = ndi.distance_transform_edt(closed, sampling=vox)
    depth_r = depth[tuple(pts.T)]
    face_r = B["dface"][tuple(pts.T)]
    ctype = cs["ctype"]
    sheet = np.argwhere(ctype == 1)
    ev = st["ev_src"]
    res = {}
    for X in ("1", "2"):
        wall = st[f"wall{X}"]
        rootX = st[f"root{X}"]
        r_sheet = rootX[tuple(sheet.T)]
        fz = np.unique(r_sheet[(r_sheet >= 0)])
        fz = fz[wall[fz]]
        dgeo = dijkstra(G, indices=fz, min_only=True, limit=15.0)
        tq = np.argwhere(ev == int(X))
        r = rootX[tuple(tq.T)]
        t = st[f"S{X}"][tuple(tq.T)].astype(float)
        ok = (r >= 0) & np.isfinite(t)
        r, t = r[ok], t[ok]
        # one value per root: median over its tops
        ur, inv = np.unique(r, return_inverse=True)
        tm = np.array([np.median(t[inv == i]) for i in range(len(ur))])
        rp = rootX[tuple(P.T)]
        sp = np.array(
            [
                rr["T1" if X == "1" else "T2"]
                for rr in np.load(f"{S}/collide2_rows.npy", allow_pickle=True)
            ]
        )
        res[X] = dict(
            r=ur,
            t=tm,
            dgeo=dgeo[ur],
            depth=depth_r[ur],
            face=face_r[ur],
            H=H[ur],
            fz=fz,
            pr=rp,
            ps=sp,
            p_dgeo=dgeo[np.maximum(rp, 0)],
            p_depth=depth_r[np.maximum(rp, 0)],
            p_H=H[np.maximum(rp, 0)],
            wall=wall,
        )
        R = res[X]
        okd = R["face"] >= FACE_MM
        print(
            f"front {X}: {len(ur):,} roots with visible tops (median {np.median(tm):.2f} mm); fused-zone roots {len(fz):,}; "
            f"contact points: arc length median {np.median(sp):.2f}, distance to fused zone median {np.median(R['p_dgeo']):.2f} mm, "
            f"depth median {np.median(R['p_depth']):.2f} mm"
        )
        for lab, x in (
            ("distance to fused zone", R["dgeo"]),
            ("sulcal depth", np.where(okd, R["depth"], np.nan)),
            ("curvature H", R["H"]),
        ):
            f = np.isfinite(x)
            rho = spearmanr(x[f], tm[f]).statistic
            qs = np.nanpercentile(x[f], [0, 20, 40, 60, 80, 100])
            meds = [
                np.median(tm[f][(x[f] >= a) & (x[f] <= b)])
                for a, b in zip(qs[:-1], qs[1:])
            ]
            print(
                f"   thickness against {lab}: rho {rho:+.2f}; medians by quintile "
                f"{', '.join(f'{a:.1f}-{b:.1f}: {mm:.2f}' for a, b, mm in zip(qs[:-1], qs[1:], meds))}"
            )
        near = R["dgeo"] < 2.0
        print(
            f"   tops whose root is within 2 mm of the fused zone: {near.sum()} roots, median {np.median(tm[near]) if near.any() else np.nan:.2f}; "
            f"2-4 mm: {np.median(tm[(R['dgeo'] >= 2) & (R['dgeo'] < 4)]):.2f}; 4-8 mm: {np.median(tm[(R['dgeo'] >= 4) & (R['dgeo'] < 8)]):.2f}; "
            f"beyond 8 mm: {np.median(tm[R['dgeo'] >= 8]) if np.any(R['dgeo'] >= 8) else np.nan:.2f}"
        )
    np.save(
        f"{S}/bank_position.npy",
        {X: {k: v for k, v in R.items() if k != "wall"} for X, R in res.items()},
        allow_pickle=True,
    )

    # ---------- views
    fig, axs = plt.subplots(2, 4, figsize=(30, 13))
    for r_, X in enumerate(("1", "2")):
        R = res[X]
        col = "#2b7bd9" if X == "1" else "#e8913a"
        okd = R["face"] >= FACE_MM
        for c_, (lab, x, px, xl) in enumerate(
            (
                (
                    "geodesic distance along the white surface to the fused zone (mm)",
                    R["dgeo"],
                    R["p_dgeo"],
                    (0, 12),
                ),
                (
                    "sulcal depth below a 4 mm hull (mm; roots >= 6 mm from the faces)",
                    np.where(okd, R["depth"], np.nan),
                    R["p_depth"],
                    None,
                ),
                (
                    "white-surface mean curvature H at the root (1/mm)",
                    R["H"],
                    R["p_H"],
                    (-0.8, 0.8),
                ),
            )
        ):
            a = axs[r_, c_]
            f = np.isfinite(x)
            a.scatter(
                x[f],
                R["t"][f],
                s=5,
                alpha=0.3,
                c=col,
                label="visible tops (one per root)",
            )
            qs = np.nanpercentile(x[f], np.linspace(0, 100, 11))
            cen = [
                np.median(x[f][(x[f] >= a_) & (x[f] <= b_)])
                for a_, b_ in zip(qs[:-1], qs[1:])
            ]
            med = [
                np.median(R["t"][f][(x[f] >= a_) & (x[f] <= b_)])
                for a_, b_ in zip(qs[:-1], qs[1:])
            ]
            p25 = [
                np.percentile(R["t"][f][(x[f] >= a_) & (x[f] <= b_)], 25)
                for a_, b_ in zip(qs[:-1], qs[1:])
            ]
            p75 = [
                np.percentile(R["t"][f][(x[f] >= a_) & (x[f] <= b_)], 75)
                for a_, b_ in zip(qs[:-1], qs[1:])
            ]
            a.fill_between(cen, p25, p75, color="k", alpha=0.12)
            a.plot(cen, med, "k-o", ms=4, label="median and IQR in deciles")
            a.scatter(
                px,
                R["ps"],
                s=60,
                marker="X",
                c="r",
                edgecolors="w",
                linewidths=0.8,
                zorder=5,
                label="arc length at your contact points",
            )
            if xl:
                a.set_xlim(*xl)
            a.set_ylim(0, 5)
            a.set_xlabel(lab)
            a.set_ylabel(f"front {X} thickness (arc length, mm)")
            a.set_title(
                f"front {X}: thickness against {lab.split(' (')[0]} "
                f"(rho {spearmanr(x[f], R['t'][f]).statistic:+.2f})",
                fontsize=11,
            )
            a.legend(fontsize=8, loc="upper right")
        # unrolled wall: PCA of the wall's roots, coloured by thickness at visible tops
        a = axs[r_, 3]
        wl = np.flatnonzero(R["wall"])
        Pw = pts[wl] * vox
        cen = Pw.mean(0)
        U = np.linalg.svd(Pw - cen, full_matrices=False)[2]
        xy = (Pw - cen) @ U[:2].T
        a.scatter(xy[:, 0], xy[:, 1], s=2, c="#dddddd")
        fzm = np.isin(wl, R["fz"])
        a.scatter(
            xy[fzm, 0], xy[fzm, 1], s=3, c="#ff2bd6", label="fused zone (sheet roots)"
        )
        exy = (pts[R["r"]] * vox - cen) @ U[:2].T
        sc = a.scatter(
            exy[:, 0],
            exy[:, 1],
            s=6,
            c=R["t"],
            cmap="viridis",
            vmin=1.5,
            vmax=3.5,
            label="visible tops (thickness)",
        )
        pxy = (pts[np.maximum(R["pr"], 0)] * vox - cen) @ U[:2].T
        a.scatter(
            pxy[:, 0],
            pxy[:, 1],
            s=60,
            marker="X",
            c=R["ps"],
            cmap="viridis",
            vmin=1.5,
            vmax=3.5,
            edgecolors="r",
            linewidths=1.2,
            label="your points' roots (arc length)",
        )
        fig.colorbar(sc, ax=a, shrink=0.8, label="thickness (mm)")
        a.set_aspect("equal")
        a.set_xlabel("mm along the wall (principal axis 1)")
        a.set_ylabel("mm (axis 2)")
        a.set_title(
            f"front {X}'s wall, unrolled (PCA): grey wall, magenta fused zone",
            fontsize=11,
        )
        a.legend(fontsize=8, loc="lower right", markerscale=2)
    fig.suptitle(
        "Bank position (A7, A8): each front's thickness at its visible tops against where the top's root lies; "
        "red X: arc length at your contact points (all 3D)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/bank_position.png", dpi=72)
    plt.close(fig)
    print("wrote bank_position.png")


if __name__ == "__main__":
    main()
