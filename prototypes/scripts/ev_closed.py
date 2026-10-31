"""Closed-form equivolume against distance and the divergence-based equivolume.

All in 3D, in the boxes of curv_map.py (loaded from the full volume; box faces are not WM
boundary). One unit-speed data-steered T with T = 0 on all WM boundary per box. Columns go up
grad T from their root (a WM-boundary voxel) to fluid ('open') or to where T stops increasing
('ridge', a collision); a ridge column is extended straight on for 0.6 mm to find the dark
contact just beyond the midline. Columns that come within 2 mm of a box face are left out.

Depth along a column in three metrics (curv_lib.column):
  distance        s
  equivolume, divergence-based   v = int exp(u) ds, u = int div(grad T / |grad T|) ds
                                  (frozen beyond the ridge)
  equivolume, closed form         V = s + Hm s^2 + K s^3 / 3, Hm = (k1 + k2) / 2, K = k1 k2 at
                                  the root (scale 0.48 mm); flagged as focal where
                                  (1 + k1 s)(1 + k2 s) < 0.1 before the column's top
Stria: darkest point of echo 2 (trend removed) 0.5-1.6 mm from the WM; contact (ridge columns):
darkest point from 0.4 mm past the stria to the end of the extension.

Root sets: open V1 (WM boundary within 1 mm of the in-plane path, crop k = 54), bank A and bank B
(within 1 mm of their paths in crop plane j = 91), and every 4th inner root (>= 6 mm from the
faces) of each box for thickness against curvature.

Usage: ev_closed.py <outdir> <workdir>.  Needs curv_<box>.npz from curv_map.py.
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

import curv_lib as C
import flatten_fused2 as F

OUT, S = sys.argv[1], sys.argv[2]
V1V2 = np.array((214, 37, 105))
SIG = "0.48"
EXT_MM = 0.6
METRICS = (
    ("s", "distance"),
    ("v_div", "equivolume, divergence"),
    ("V", "equivolume, closed form"),
)
MCOL = {"s": "#4a4a4a", "v_div": "#d9822b", "V": "#2b7bd9"}


def path_roots(B, a_full, b_full, plane_axis):
    """WM-boundary voxels within 1 mm of the in-plane boundary path from a to b."""
    a = np.array(a_full) - B["lo"]
    b = np.array(b_full) - B["lo"]
    other = [x for x in range(3) if x != plane_axis]
    sl = [slice(None)] * 3
    sl[plane_axis] = a[plane_axis]
    p2 = F.boundary_path(B["shell"][tuple(sl)], tuple(a[other]), tuple(b[other]))
    path = np.zeros((len(p2), 3), int)
    path[:, other] = p2
    path[:, plane_axis] = a[plane_axis]
    m = np.zeros(B["rim"].shape, bool)
    m[tuple(path.T)] = True
    d = ndi.distance_transform_edt(~m, sampling=B["vox"])
    return np.argwhere(B["shell"] & (d <= 1.0)), path


def run_columns(B, Fd, K, roots, fused=False):
    vox = B["vox"]
    out = []
    for r in roots:
        i = K["idx"][tuple(r)]
        k1, k2 = K["k1"][i], K["k2"][i]
        pts, st = C.trace(B, Fd, r)
        rec = dict(root=r, k1=k1, k2=k2, Hm=0.5 * (k1 + k2), K=k1 * k2, st=st, pts=pts)
        if st not in ("open", "ridge") or len(pts) < 8:
            out.append(rec)
            continue
        ir = len(pts) - 1
        if st == "ridge":
            d = pts[-1] - pts[-2]
            d /= np.linalg.norm(d * vox)
            n_ext = int(np.ceil(EXT_MM / (C.STEP * 0.16)))
            ext = pts[-1] + np.outer(np.arange(1, n_ext + 1) * C.STEP * 0.16, d)
            ext = np.clip(ext, 0, np.array(B["rim"].shape) - 1)
            pts = np.vstack([pts, ext])
        col = C.column(B, Fd, pts, k1, k2, freeze_after=ir)
        rec.update(col)
        rec["ir"] = ir
        rec["pts"] = pts
        s, e = col["s"], col["e"]
        top = ir
        m = (s >= 0.5) & (s <= 1.6) & (np.arange(len(s)) <= top)
        if not m.any():
            out.append(rec)
            continue
        ks = np.flatnonzero(m)[np.argmin(e[m])]
        rec["ks"] = ks
        if st == "ridge":
            mc = s >= s[ks] + 0.4
            kc = np.flatnonzero(mc)[np.argmin(e[mc])] if mc.any() else None
            rec["kc"] = kc
            rec["edge"] = kc is None or s[kc] > s[-1] - 0.15
            top = kc if kc is not None else ir
        rec["kt"] = top
        a = (1 + k1 * s[: top + 1]) * (1 + k2 * s[: top + 1])
        rec["focal"] = bool(a.min() < 0.1)
        for mk, _ in METRICS:
            rec[f"top_{mk}"] = col[mk][top]
            rec[f"fr_{mk}"] = col[mk][ks] / col[mk][top]
        out.append(rec)
    return out


def usable(recs, kind):
    ok = [r for r in recs if r["st"] == kind and "kt" in r and r["dface"] >= 2.0]
    if kind == "ridge":
        ok = [r for r in ok if not r["edge"]]
    return ok


def summary(rs, label):
    lines = []
    for mk, mn in METRICS:
        f = np.array([r[f"fr_{mk}"] for r in rs if not r["focal"]])
        h = np.array([r["Hm"] for r in rs if not r["focal"]])
        rho = spearmanr(h, f).statistic if len(f) > 5 else np.nan
        q = np.percentile(f, [25, 50, 75])
        lines.append((label, mn, len(f), q[1], q[2] - q[0], rho))
        print(
            f"  {label:10s} {mn:26s} n {len(f):4d}  median {q[1]:.3f}  IQR {q[2] - q[0]:.3f}  "
            f"Spearman(fraction, Hm) {rho:+.2f}"
        )
    nf = sum(r["focal"] for r in rs)
    print(f"  {label}: {nf} focal columns left out of the fractions")
    return lines


def profile_img(rs, mk, top_key="kt", nb=50):
    img = np.full((nb, len(rs)), np.nan)
    q = (np.arange(nb) + 0.5) / nb * 1.25
    for c, r in enumerate(rs):
        if r["focal"] and mk == "V":
            continue  # V is not monotone past a focal point
        x = r[mk] / r[mk][r[top_key]]
        okq = q <= x[-1]
        img[okq, c] = np.interp(q[okq], x, r["e"])
    return img


def fraction_figure(groups, title, fname, vmax):
    """groups: list of (label, recs). Per group: echo 2 by normalized depth in each metric
    (columns sorted by root mean curvature), then fraction against Hm per metric."""
    ng = len(groups)
    fig, axs = plt.subplots(ng + 1, 3, figsize=(24, 5.2 * (ng + 1)))
    for g, (lab, rs) in enumerate(groups):
        rs = sorted(rs, key=lambda r: r["Hm"])
        hm = np.array([r["Hm"] for r in rs])
        foc = np.array([r["focal"] for r in rs])
        for c, (mk, mn) in enumerate(METRICS):
            a = axs[g, c]
            img = profile_img(rs, mk)
            a.imshow(
                img,
                origin="lower",
                aspect="auto",
                cmap="gray",
                vmin=-vmax,
                vmax=vmax,
                extent=[0, len(rs), 0, 1.25],
                interpolation="nearest",
            )
            fr = np.array([r[f"fr_{mk}"] for r in rs])
            fr = np.where(foc & (mk == "V"), np.nan, fr)
            a.plot(
                np.arange(len(rs)) + 0.5,
                fr,
                ".",
                c="#ffd23f",
                ms=3,
                label="stria (darkest 0.5-1.6 mm)",
            )
            a.axhline(1.0, c="c", lw=0.8, ls="--")
            a.axhline(np.median(fr[~foc]), c="#ffd23f", lw=0.8, ls=":")
            a.set_ylim(0, 1.25)
            if foc.any():
                a.plot(
                    np.flatnonzero(foc) + 0.5,
                    np.full(foc.sum(), 0.02),
                    "|",
                    c="r",
                    ms=8,
                    label="focal (flagged; blank in the closed form)",
                )
            ticks = np.linspace(0, len(rs) - 1, 6).astype(int)
            a.set_xticks(ticks + 0.5)
            a.set_xticklabels([f"{hm[t]:+.2f}" for t in ticks])
            a.set_xlabel("columns sorted by root mean curvature Hm (1/mm)")
            a.set_ylabel(
                "depth / depth at the top (1 = "
                + ("pia" if "open" in lab or "V1 open" in lab else "dark contact")
                + ")"
            )
            a.set_title(f"{lab}: echo 2 by {mn}", fontsize=12)
            if c == 0:
                a.legend(loc="upper left", fontsize=8)
    # fraction against curvature, all groups, per metric
    for c, (mk, mn) in enumerate(METRICS):
        a = axs[ng, c]
        for g, (lab, rs) in enumerate(groups):
            rs2 = [r for r in rs if not r["focal"]]
            h = np.array([r["Hm"] for r in rs2])
            f = np.array([r[f"fr_{mk}"] for r in rs2])
            mkr = ["o", "s", "^"][g % 3]
            a.scatter(
                h,
                f,
                s=7,
                alpha=0.35,
                marker=mkr,
                label=f"{lab} (median {np.median(f):.3f}, IQR "
                f"{np.subtract(*np.percentile(f, [75, 25])):.3f}, rho {spearmanr(h, f).statistic:+.2f})",
            )
            qs = np.quantile(h, np.linspace(0, 1, 7))
            cen = [np.median(h[(h >= qs[b]) & (h <= qs[b + 1])]) for b in range(6)]
            med = [np.median(f[(h >= qs[b]) & (h <= qs[b + 1])]) for b in range(6)]
            a.plot(cen, med, "-", marker=mkr, c="k", lw=1.5, ms=6)
        a.set_ylim(0.15, 0.85)
        a.set_xlabel(
            "root mean curvature Hm (1/mm): + convex (crown), - concave (fundus)"
        )
        a.set_ylabel("stria fraction of the column's top")
        a.set_title(
            f"{mn}: stria fraction against curvature (black: medians in sextiles)",
            fontsize=12,
        )
        a.legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=80)
    plt.close(fig)
    print("wrote", fname)


def columns_view(B, groups, plane_axis, plane_full, fname, title, vmax):
    """Columns (3D, projected) on one plane, coloured by root mean curvature."""
    c0 = plane_full - B["lo"][plane_axis]
    other = [x for x in range(3) if x != plane_axis]
    sl = [slice(None)] * 3
    sl[plane_axis] = c0
    sl = tuple(sl)
    fig, a = plt.subplots(figsize=(11, 10))
    a.imshow(
        np.where(B["ctx"][sl], B["disp"][sl], np.nan).T,
        origin="lower",
        cmap="gray",
        vmin=-vmax,
        vmax=vmax,
    )
    a.contour(B["shell"][sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.6)
    allp = []
    for lab, rs in groups:
        for r in rs:
            if abs(r["root"][plane_axis] - c0) > 1:
                continue
            p = r["pts"]
            col = plt.cm.RdBu_r(np.clip((r["Hm"] + 1) / 2, 0, 1))
            a.plot(p[:, other[0]], p[:, other[1]], "-", c=col, lw=0.8)
            if "ks" in r:
                a.plot(*p[r["ks"], other], ".", c="#ffd23f", ms=3)
            if r.get("kc") is not None:
                a.plot(*p[r["kc"], other], "o", mfc="none", mec="c", ms=4)
            allp.append(p[:, other])
    if allp:
        allp = np.vstack(allp)
        a.set_xlim(allp[:, 0].min() - 15, allp[:, 0].max() + 15)
        a.set_ylim(allp[:, 1].min() - 15, allp[:, 1].max() + 15)
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(
        title
        + "\ncolumns rooted within 1 voxel of the plane, coloured by root mean curvature (red convex, "
        "blue concave);\nyellow dot stria, cyan circle dark contact",
        fontsize=11,
    )
    fig.savefig(f"{OUT}/{fname}", dpi=80, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def setup(name, lo, hi):
    B = C.load(lo, hi, name)
    cz = np.load(f"{S}/curv_{name}.npz")
    assert np.array_equal(cz["lo"], B["lo"])
    idx = -np.ones(B["rim"].shape, np.int64)
    idx[tuple(cz["pts"].T)] = np.arange(len(cz["pts"]))
    K = dict(idx=idx, k1=cz[f"k1_{SIG}"], k2=cz[f"k2_{SIG}"], pts=cz["pts"])
    try:
        z = np.load(f"{S}/T_{name}.npz")
        Fd = dict(T=z["T"], Tf=z["Tf"], g=z["g"], div=z["div"])
        print(f"{name}: T from cache")
    except FileNotFoundError:
        Fd = C.fit_all_wm(B)
        print(f"{name}: T fitted, cg info {Fd['info']}")
        np.savez_compressed(
            f"{S}/T_{name}.npz", T=Fd["T"], Tf=Fd["Tf"], g=Fd["g"], div=Fd["div"]
        )
    return B, Fd, K


def main():
    rng = np.random.default_rng(0)
    m = json.load(open(f"{S}/M1S1_test_points.json"))
    res = {}
    # ---------------- V1V2
    B, Fd, K = setup("V1V2", (201, 70, 65), (344, 214, 205))
    vmax = np.nanpercentile(np.abs(B["disp"][B["ctx"]]), 98)
    rO, _ = path_roots(
        B, (71 + 214, 89 + 37, 54 + 105), (84 + 214, 131 + 37, 54 + 105), 2
    )
    rA, _ = path_roots(
        B, np.array((71, 91, 18)) + V1V2, np.array((67, 91, 47)) + V1V2, 1
    )
    rB, _ = path_roots(
        B, np.array((38, 91, 50)) + V1V2, np.array((42, 91, 5)) + V1V2, 1
    )
    print(f"roots: open V1 {len(rO)}, bank A {len(rA)}, bank B {len(rB)}")
    cO, cA, cB = (run_columns(B, Fd, K, r) for r in (rO, rA, rB))
    import collections

    for lab, cc in (("open V1", cO), ("bank A", cA), ("bank B", cB)):
        print(
            f"  {lab}: outcomes {dict(collections.Counter(r['st'] for r in cc))}, "
            f"near a face {sum(1 for r in cc if r.get('dface', 99) < 2.0)}"
        )
    gO = usable(cO, "open")
    gA = usable(cA, "ridge")
    gB = usable(cB, "ridge")
    print("stria fractions (focal columns left out):")
    lines = summary(gO, "open V1") + summary(gA, "bank A") + summary(gB, "bank B")
    for lab, g in (("bank A", gA), ("bank B", gB)):
        print(
            f"  {lab}: root Hm median {np.median([r['Hm'] for r in g]):+.2f}, k1 {np.median([r['k1'] for r in g]):+.2f}, "
            f"k2 {np.median([r['k2'] for r in g]):+.2f}; contact at s {np.median([r['s'][r['kc']] for r in g]):.2f} mm"
        )
    fraction_figure(
        [("V1 open", gO)],
        "Open V1 (V1V2 crop k = 54), columns rooted within 1 mm of the stretch, all in 3D: "
        "stria fraction in three metrics",
        "ev_openV1.png",
        vmax,
    )
    fraction_figure(
        [("bank A", gA), ("bank B", gB)],
        "Fused V1 banks (V1V2 crop plane j = 91, box from the full volume), "
        "columns from all-WM T to the dark contact, all in 3D: stria fraction of the contact in three metrics",
        "ev_banks.png",
        vmax,
    )
    columns_view(
        B,
        [("open", gO)],
        2,
        54 + 105,
        "ev_openV1_columns.png",
        "Open V1, V1V2 crop k = 54",
        vmax,
    )
    columns_view(
        B,
        [("A", gA), ("B", gB)],
        1,
        91 + 37,
        "ev_banks_columns.png",
        "Fused banks, V1V2 crop j = 91",
        vmax,
    )
    res["lines"] = lines

    # ---------------- thickness against curvature, every 4th inner root
    thick = {}
    for name, lo, hi in (
        ("V1V2", None, None),
        (
            "M1S1",
            tuple(np.array(m["full_lo"]) - 15),
            tuple(np.array(m["full_hi"]) + 15),
        ),
    ):
        if name != "V1V2":
            B, Fd, K = setup(name, lo, hi)
        inner = K["pts"][B["dface"][tuple(K["pts"].T)] >= 6.0]
        sel = inner[rng.permutation(len(inner))[: len(inner) // 4]]
        cc = run_columns(B, Fd, K, sel)
        g = usable(cc, "open")
        print(
            f"{name}: {len(sel)} inner roots, {len(g)} open columns usable, "
            f"{sum(r['focal'] for r in g)} focal"
        )
        thick[name] = g
    fig, axs = plt.subplots(2, 3, figsize=(22, 12))
    for r_, (name, g) in enumerate(thick.items()):
        nf = [r for r in g if not r["focal"]]
        fo = [r for r in g if r["focal"]]
        h = np.array([r["Hm"] for r in nf])
        for c, (mk, mn) in enumerate(METRICS):
            a = axs[r_, c]
            t = np.array([r[f"top_{mk}"] for r in nf])
            a.scatter(h, t, s=4, alpha=0.25, c=MCOL[mk])
            if fo:
                a.scatter(
                    [r["Hm"] for r in fo],
                    [r[f"top_{mk}"] for r in fo],
                    s=6,
                    c="r",
                    marker="x",
                    alpha=0.6,
                    label=f"focal ({len(fo)})",
                )
            qs = np.quantile(h, np.linspace(0, 1, 11))
            cen = [np.median(h[(h >= qs[b]) & (h <= qs[b + 1])]) for b in range(10)]
            med = [np.median(t[(h >= qs[b]) & (h <= qs[b + 1])]) for b in range(10)]
            p25 = [
                np.percentile(t[(h >= qs[b]) & (h <= qs[b + 1])], 25) for b in range(10)
            ]
            p75 = [
                np.percentile(t[(h >= qs[b]) & (h <= qs[b + 1])], 75) for b in range(10)
            ]
            a.fill_between(cen, p25, p75, color="k", alpha=0.12)
            a.plot(cen, med, "k-o", ms=4)
            core = (h > np.percentile(h, 5)) & (h < np.percentile(h, 95))
            sl_ = np.polyfit(h[core], t[core], 1)
            rho = spearmanr(h, t).statistic
            a.set_title(
                f"{name}, {mn}: median {np.median(t):.2f} mm, slope {sl_[0]:+.2f} mm per 1/mm "
                f"(central 90% of Hm), rho {rho:+.2f}",
                fontsize=11,
            )
            a.set_xlim(np.percentile(h, 1) - 0.05, np.percentile(h, 99) + 0.05)
            a.set_ylim(0, np.percentile(t, 99.5) * 1.1)
            a.set_xlabel("root mean curvature Hm (1/mm): + crown, - fundus")
            a.set_ylabel(f"open-column thickness, {mn} (mm)")
            if fo:
                a.legend(loc="upper left", fontsize=8)
            print(
                f"  thickness {name} {mn:26s}: median {np.median(t):.2f}, slope {sl_[0]:+.2f} mm per 1/mm, rho {rho:+.2f}"
            )
    fig.suptitle(
        "Open-column thickness against root mean curvature (every 4th WM-boundary voxel >= 6 mm from the faces; "
        "columns traced in 3D; black: medians and IQR in deciles of Hm)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/ev_thickness.png", dpi=80)
    print("wrote ev_thickness.png")
    import pickle

    pickle.dump(
        dict(
            open=gO,
            A=gA,
            B=gB,
            thick={
                k: [
                    {kk: vv for kk, vv in r.items() if kk not in ("pts", "e")}
                    for r in v
                ]
                for k, v in thick.items()
            },
        ),
        open(f"{S}/ev_closed_results.pkl", "wb"),
    )


if __name__ == "__main__":
    main()
