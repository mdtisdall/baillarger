"""Per-front arrival at the kissing stretch, with walls taken from roots (M1S1). Follows collide.py.

Why: the all-WM T of collide.py signs the data normal by the Euclidean WM distance, so its ridge
(equal arrival and ownership) is pinned to the geometric midline between WM surfaces. Here:
1. Walls from roots: the roots on the two sides of the sheet pairs within 4 mm of the user's points
   (collide.py), grown 1 mm along the white surface and split into connected pieces; the pieces are
   sorted into two sides so that pieces facing each other across the sheets differ (conflicts are
   counted); each side, grown by 3 mm, is a front's wall. Nothing is labelled by
   geometry: a wall is the WM whose columns reach the sheet.
2. One data-steered field per front, T = 0 on its own wall only, signed away from that wall (nothing
   flips at a midline), in the closed cortex within 9 mm of the points; roots per front by the upwind pass.
3. Distance is arc length along each front's own columns (S), carried in the upwind pass with the roots;
   T itself under-reads long columns (finding 12). Thickness (distance): as collide.py, but evidence is
   arc length at visible tops: each front's own S on the two walls (tops whose all-WM root lies on its
   wall), the all-WM field's S elsewhere; log t = m + (a0 + w(x)) H + z(x).
4. Rules as fields, in arc length: equal arrival S_1 = S_2; pairwise posterior
   (S_1 - mu_1)/s_1^2 = (S_2 - mu_2)/s_2^2 (the COHIBA split written per voxel), with mu, s read at each
   voxel's own root in each front. The signed distance from the posterior contact is
   g = ((S_1 - mu_1) s_2^2 - (S_2 - mu_2) s_1^2) / (s_1^2 + s_2^2) (mm); the band is |g| < s_1 s_2 / sqrt(s_1^2 + s_2^2).
5. Offsets at the 23 points along grad T_1, + toward the side predicted thinner.

Usage: collide2.py <outdir> <workdir>. Needs collide_state.npz, T_M1S1_closed.npz (collide.py).
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import connected_components, dijkstra

import curv_lib as C
import flatten_fused2 as F
from collide import (
    close_gap,
    CLOSE_MM,
    FACE_MM,
    GROW,
    H_CLIP,
    H_SCALE,
    POINTS,
    SIG_EPS,
    SITE,
    Field,
    graph_L,
    lognormal,
    shell_graph,
    view_planes,
)
from fit_T import fill, solve_field
from prob_thickness import upwind_fields

OUT, S = sys.argv[1], sys.argv[2]
NEAR_MM, GROW_WALL_MM, REGION_MM = 4.0, 3.0, 9.0


def main():
    m = json.load(open(f"{S}/{POINTS}"))
    lo = np.array(m["full_lo"]) - GROW
    hi = np.array(m["full_hi"]) + GROW
    B = C.load(lo, hi, SITE)
    vox = B["vox"]
    shp = np.array(B["rim"].shape)
    P = np.array(m["gap_all"], float) + GROW
    cz = np.load(f"{S}/curv_{SITE}.npz")
    pts, nrm = cz["pts"], cz["n"]
    ppos = pts * vox
    Hroot = np.clip(0.5 * (cz[f"k1_{H_SCALE}"] + cz[f"k2_{H_SCALE}"]), -H_CLIP, H_CLIP)
    st = np.load(f"{S}/collide_state.npz")
    root_all, ctype, node_of = st["root"], st["ctype"], st["node_of"]
    rim0 = B["rim"]
    rim = rim0.copy()
    rim[tuple(close_gap(rim0, P, vox, B["dface"]).T)] = 3
    ctx0 = B["ctx"]
    ctx = (rim == 3) & ~B["miss"]
    T_all = np.load(f"{S}/T_{SITE}_closed.npz")["T"]
    # arc length along the all-WM columns, carried in the upwind pass (T itself under-reads long columns)
    dom_all = np.isfinite(T_all)
    Tf_all = fill(T_all)
    g_all = np.stack(np.gradient(Tf_all, *vox), -1)
    n1_all = g_all / (np.linalg.norm(g_all, axis=-1, keepdims=True) + 1e-12)
    rid_all = -np.ones(rim.shape, np.int64)
    rid_all[tuple(pts.T)] = np.arange(len(pts))
    order = np.argwhere(dom_all)[np.argsort(Tf_all[dom_all])].astype(np.int64)
    _, S_all, _ = upwind_fields(
        order,
        dom_all,
        dom_all & B["wm"],
        rid_all,
        Tf_all,
        n1_all,
        np.zeros(rim.shape),
        vox.astype(float),
    )
    S_all = np.where(dom_all, S_all, np.nan)

    # ---- 1. walls from the roots at sheets near the points
    G = shell_graph(pts, nrm, vox)
    sheet = np.argwhere(ctype == 1)
    dP = np.min(np.linalg.norm((sheet[:, None, :] - P[None]) * vox, axis=2), 1)
    sheet = sheet[dP < NEAR_MM]
    pairs = []
    for d in np.eye(3, dtype=int):
        for sgn in (1, -1):
            q = np.clip(sheet + sgn * d, 0, shp - 1)
            r1, r2 = root_all[tuple(sheet.T)], root_all[tuple(q.T)]
            ok = (
                (r1 >= 0)
                & (r2 >= 0)
                & (
                    np.linalg.norm(
                        ppos[np.maximum(r1, 0)] - ppos[np.maximum(r2, 0)], axis=1
                    )
                    > 2.0
                )
            )
            pairs.append(np.c_[r1[ok], r2[ok]])
    pairs = np.vstack(pairs)
    R = np.unique(pairs)
    # pieces: the sheet roots grown 1 mm along the white surface (closing small gaps), then split
    D1 = dijkstra(G, indices=R, limit=1.0, min_only=True)
    M = np.flatnonzero(np.isfinite(D1))
    nc, labM = connected_components(G[M][:, M], directed=False)
    comp = -np.ones(len(pts), int)
    comp[M] = labM
    size = np.bincount(labM, minlength=nc)
    opp = {}
    for a, b in pairs:
        ca, cb = comp[a], comp[b]
        if ca != cb:
            key = (min(ca, cb), max(ca, cb))
            opp[key] = opp.get(key, 0) + 1
    # two sides: colour the pieces so that pieces facing each other across the sheets differ,
    # weighting each facing by its number of voxel pairs (greedy, largest facings first)
    side = -np.ones(nc, int)
    start = int(np.argmax(size))
    side[start] = 0
    conflicts = []
    changed = True
    while changed:
        changed = False
        for (ca, cb), n in sorted(opp.items(), key=lambda kv: -kv[1]):
            if side[ca] >= 0 and side[cb] < 0:
                side[cb] = 1 - side[ca]
                changed = True
            elif side[cb] >= 0 and side[ca] < 0:
                side[ca] = 1 - side[cb]
                changed = True
    for (ca, cb), n in opp.items():
        if side[ca] >= 0 and side[ca] == side[cb]:
            conflicts.append((ca, cb, n))
    print(
        f"sheet voxels within {NEAR_MM} mm of the points: {len(sheet):,}; roots grown 1 mm form {nc} pieces; "
        f"sides: {np.sum(size[side == 0])} and {np.sum(size[side == 1])} voxels in {np.sum(side == 0)} and "
        f"{np.sum(side == 1)} pieces; {np.sum(side < 0)} pieces face nothing; same-side facings (conflicts) "
        f"{len(conflicts)} ({sum(c[2] for c in conflicts)} voxel pairs, against {sum(opp.values())})"
    )
    walls = {}
    for X, sd_ in (("1", 0), ("2", 1)):
        seeds = M[side[labM] == sd_]
        D = dijkstra(G, indices=seeds, limit=GROW_WALL_MM, min_only=True)
        walls[X] = np.isfinite(D)
    both = walls["1"] & walls["2"]
    for X in walls:
        walls[X] &= ~both
        print(
            f"   wall {X}: {walls[X].sum():,} WM-boundary voxels (overlap removed {both.sum()})"
        )

    # ---- 2. per-front fields
    dpt = ndi.distance_transform_edt(~_pts_mask(P, rim.shape), sampling=vox)
    region = dpt <= REGION_MM
    nh, coh = C.tensor_normal(B)
    Fr = {}
    for X in ("1", "2"):
        wmask = np.zeros(rim.shape, bool)
        wmask[tuple(pts[walls[X]].T)] = True
        dom = (ctx & region) | wmask
        dW = ndi.distance_transform_edt(~wmask, sampling=vox)
        g = np.stack(np.gradient(ndi.gaussian_filter(dW, 1.0), *vox), -1)
        gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
        sg = np.sign((nh * gh).sum(-1))
        sg[sg == 0] = 1
        dirv = coh[..., None] * nh * sg[..., None] + C.LAM * gh
        dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
        T, info = solve_field(dom, wmask, [dirv[..., a] for a in range(3)], tuple(vox))
        Tf = fill(T)
        gT = np.stack(np.gradient(Tf, *vox), -1)
        n1 = gT / (np.linalg.norm(gT, axis=-1, keepdims=True) + 1e-12)
        rootid = -np.ones(rim.shape, np.int64)
        rootid[tuple(pts[walls[X]].T)] = np.flatnonzero(walls[X])
        order = np.argwhere(dom)[np.argsort(Tf[dom])].astype(np.int64)
        _, Sx, root = upwind_fields(
            order, dom, wmask, rootid, Tf, n1, np.zeros(rim.shape), vox.astype(float)
        )
        Sx = np.where(dom, Sx, np.nan)
        Fr[X] = dict(
            T=np.where(dom, T, np.nan),
            Tf=Tf,
            S=Sx,
            Sf=fill(Sx),
            g=[gT[..., a] for a in range(3)],
            root=np.where(dom, root, -1),
            dom=dom,
        )
        print(f"front {X}: cg {info}; domain {dom.sum():,} voxels")

    # ---- 3. thickness evidence and field
    nN = node_of.max() + 1
    counts = np.bincount(node_of, minlength=nN)
    npos = (
        np.stack([np.bincount(node_of, ppos[:, a], nN) for a in range(3)], 1)
        / counts[:, None]
    )
    nH = np.bincount(node_of, Hroot, nN) / counts
    Gc = G.tocoo()
    ea, eb = node_of[Gc.row], node_of[Gc.col]
    ed = np.unique(np.sort(np.c_[ea, eb][ea != eb], 1), axis=0)
    L = graph_L(npos, ed)
    tops = ctx0 & ndi.binary_dilation(rim == 1) & (B["dface"] >= FACE_MM)
    onwall = np.zeros(len(pts), bool)
    onwall[walls["1"] | walls["2"]] = True
    vals = {}  # node -> list of top thicknesses
    # elsewhere: the all-WM field, as in collide.py (roots not on the two walls)
    tq = np.argwhere(tops & (root_all >= 0) & np.isfinite(S_all))
    rr = root_all[tuple(tq.T)]
    tv = S_all[tuple(tq.T)]
    keep = ~onwall[rr] & (tv > 0.8) & (tv < 7.0)
    for n_, v in zip(node_of[rr[keep]], tv[keep]):
        vals.setdefault(n_, []).append(v)
    # on the walls: each front's own T at its own tops
    for X, Y in (("1", "2"), ("2", "1")):
        A_, B_ = Fr[X], Fr[Y]
        onX = np.zeros(len(pts) + 1, bool)
        onX[:-1] = walls[X]  # index -1 (no root) maps to False
        own = tops & onX[root_all] & (A_["root"] >= 0) & np.isfinite(A_["S"])
        tq = np.argwhere(own)
        rr = A_["root"][tuple(tq.T)]
        tv = A_["S"][tuple(tq.T)]
        keep = (tv > 0.8) & (tv < 7.0)
        for n_, v in zip(node_of[rr[keep]], tv[keep]):
            vals.setdefault(n_, []).append(v)
        print(f"   front {X}: {keep.sum():,} own top voxels")
    y = np.full(nN, np.nan)
    for n_, v in vals.items():
        y[n_] = np.log(np.median(v))
    FLD = Field(L, nH)
    mu, sd, coef = FLD.fit(y, trend=True)
    tmean, tsd = lognormal(mu, sd)
    print(
        f"field: {np.isfinite(y).sum():,} evidence nodes; a0 = {coef['a0'][0]:+.2f} +- {coef['a0'][1]:.2f}"
    )

    # ---- 4. rules
    def at_root(X, arr):
        r = Fr[X]["root"]
        return np.where(r >= 0, arr[node_of[np.maximum(r, 0)]], np.nan)

    mu1, s1 = fill(at_root("1", tmean)), fill(at_root("1", tsd))
    mu2, s2 = fill(at_root("2", tmean)), fill(at_root("2", tsd))
    T1, T2 = Fr["1"]["Sf"], Fr["2"]["Sf"]  # arc length along each front's columns
    both_dom = Fr["1"]["dom"] & Fr["2"]["dom"] & ctx
    eq = np.where(both_dom, T1 - T2, np.nan)
    gpost = np.where(
        both_dom, ((T1 - mu1) * s2**2 - (T2 - mu2) * s1**2) / (s1**2 + s2**2), np.nan
    )
    band = s1 * s2 / np.sqrt(s1**2 + s2**2)
    gA = [ndi.gaussian_filter(c, 0.5) for c in Fr["1"]["g"]]
    rows = []
    for k_, q in enumerate(P):
        n_ = np.array([F.interp(c, q) for c in gA])
        n_ /= np.linalg.norm(n_) + 1e-12
        ss = np.arange(-2.5, 2.5001, 0.02)
        line = q[None] + ss[:, None] * n_[None] / vox
        sgn = (
            1.0 if F.interp(mu1, q) >= F.interp(mu2, q) else -1.0
        )  # + toward the thinner side
        res = {}
        for nm, fld in (("eq", eq), ("post", gpost)):
            f = ndi.map_coordinates(np.nan_to_num(fld, nan=0.0), line.T, order=1)
            okl = (
                ndi.map_coordinates(np.isfinite(fld).astype(float), line.T, order=0) > 0
            )
            cr = np.flatnonzero((np.diff(np.sign(f)) != 0) & okl[:-1] & okl[1:])
            if len(cr):
                c0 = cr[np.argmin(np.abs(ss[cr]))]
                res[nm] = sgn * (
                    ss[c0] - f[c0] * (ss[c0 + 1] - ss[c0]) / (f[c0 + 1] - f[c0] + 1e-12)
                )
            else:
                res[nm] = np.nan
        rows.append(
            dict(
                k=k_,
                eq=res["eq"],
                post=res["post"],
                band=F.interp(band, q),
                T1=F.interp(T1, q),
                T2=F.interp(T2, q),
                Tfit1=F.interp(Fr["1"]["Tf"], q),
                Tfit2=F.interp(Fr["2"]["Tf"], q),
                mu1=F.interp(mu1, q),
                mu2=F.interp(mu2, q),
                s1=F.interp(s1, q),
                s2=F.interp(s2, q),
                kind=int(ctype[tuple(np.round(q).astype(int))]),
            )
        )
    print(
        "per point: arc length S_1, S_2 up to the point (T_1, T_2 in brackets); mu +- s at each root; offsets (+ toward thinner)"
    )
    for r in rows:
        print(
            f"   {r['k'] + 1:2d}: S {r['T1']:.2f}/{r['T2']:.2f} (T {r['Tfit1']:.2f}/{r['Tfit2']:.2f})  mu {r['mu1']:.2f}+-{r['s1']:.2f} / {r['mu2']:.2f}+-{r['s2']:.2f}  "
            f"eq {r['eq']:+.2f}  post {r['post']:+.2f} (band {r['band']:.2f})"
        )
    for nm in ("eq", "post"):
        v = np.array([r[nm] for r in rows])
        print(
            f"   {'equal arrival' if nm == 'eq' else 'posterior':14s} median {np.nanmedian(v):+.2f}, IQR "
            f"{np.subtract(*np.nanpercentile(v, [75, 25])):.2f}, |offset| median {np.nanmedian(np.abs(v)):.2f}, missing {np.isnan(v).sum()}"
        )
    t1 = np.array([r["T1"] for r in rows])
    t2 = np.array([r["T2"] for r in rows])
    print(
        f"   at the points: front 1 thickness up to them median {np.median(t1):.2f} [{t1.min():.2f}-{t1.max():.2f}], "
        f"front 2 {np.median(t2):.2f} [{t2.min():.2f}-{t2.max():.2f}]"
    )
    np.save(f"{S}/collide2_rows.npy", rows, allow_pickle=True)
    ev_src = np.zeros(
        rim.shape, np.int8
    )  # 1: top used as evidence via front 1, 2: via front 2, 3: via the all-WM field
    tq = np.argwhere(tops & (root_all >= 0) & np.isfinite(S_all))
    rr = root_all[tuple(tq.T)]
    tv = S_all[tuple(tq.T)]
    k3 = ~onwall[rr] & (tv > 0.8) & (tv < 7.0)
    ev_src[tuple(tq[k3].T)] = 3
    for X, Y, code in (("1", "2", 1), ("2", "1", 2)):
        A_, B_ = Fr[X], Fr[Y]
        onX = np.zeros(len(pts) + 1, bool)
        onX[:-1] = walls[X]
        own = tops & onX[root_all] & (A_["root"] >= 0) & np.isfinite(A_["S"])
        tq = np.argwhere(own)
        tv = A_["S"][tuple(tq.T)]
        ev_src[tuple(tq[(tv > 0.8) & (tv < 7.0)].T)] = code
    np.savez_compressed(
        f"{S}/collide2_state.npz",
        eq=eq.astype(np.float32),
        gpost=gpost.astype(np.float32),
        band=band.astype(np.float32),
        both_dom=both_dom,
        T1=Fr["1"]["T"].astype(np.float32),
        T2=Fr["2"]["T"].astype(np.float32),
        S1=Fr["1"]["S"].astype(np.float32),
        S2=Fr["2"]["S"].astype(np.float32),
        S_all=S_all.astype(np.float32),
        root1=Fr["1"]["root"].astype(np.int32),
        root2=Fr["2"]["root"].astype(np.int32),
        wall1=walls["1"],
        wall2=walls["2"],
        y=y,
        tmean=tmean,
        tsd=tsd,
        a=coef["a"],
        nH=nH,
        ev_src=ev_src,
    )

    # ---- views: contact rules on the user's planes
    disp = B["disp"]
    vmax = np.nanpercentile(np.abs(disp[ctx0]), 98)
    planes = view_planes(P)
    w1 = np.zeros(rim.shape, bool)
    w1[tuple(pts[walls["1"]].T)] = True
    w2 = np.zeros(rim.shape, bool)
    w2[tuple(pts[walls["2"]].T)] = True
    fig, axs = plt.subplots(2, 4, figsize=(28, 14.5))
    for a, (ax_i, c0, lab_) in zip(axs.ravel(), planes):
        sl = [slice(None)] * 3
        sl[ax_i] = c0
        sl = tuple(sl)
        oth = [x for x in range(3) if x != ax_i]
        a.imshow(
            np.where(ctx[sl], disp[sl], np.nan).T,
            origin="lower",
            cmap="gray",
            vmin=-vmax,
            vmax=vmax,
        )
        a.imshow(
            np.where(both_dom[sl] & (np.abs(gpost[sl]) < band[sl]), 1.0, np.nan).T,
            origin="lower",
            cmap="Greens",
            alpha=0.35,
            vmin=0,
            vmax=1.5,
        )
        a.imshow(
            np.where(w1[sl], 1.0, np.nan).T,
            origin="lower",
            cmap="Blues",
            vmin=0,
            vmax=1.1,
        )
        a.imshow(
            np.where(w2[sl], 1.0, np.nan).T,
            origin="lower",
            cmap="Oranges",
            vmin=0,
            vmax=1.1,
        )
        a.contour(B["wm"][sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.5)
        a.contour(
            (rim0 == 1)[sl].T.astype(float), levels=[0.5], colors="c", linewidths=0.5
        )
        vv = np.argwhere(ctype[sl] == 1)
        a.scatter(
            vv[:, 0], vv[:, 1], s=2, c="#ff2bd6", marker="s", linewidths=0, alpha=0.5
        )
        a.contour(eq[sl].T, levels=[0], colors="lime", linewidths=1.4)
        a.contour(gpost[sl].T, levels=[0], colors="#30d5ff", linewidths=1.4)
        selp = np.abs(P[:, ax_i] - c0) <= 1.5
        a.plot(P[selp, oth[0]], P[selp, oth[1]], "x", c="#ffd23f", ms=9, mew=2)
        win = P[selp][:, oth].mean(0) if selp.any() else P[:, oth].mean(0)
        a.set_xlim(win[0] - 35, win[0] + 35)
        a.set_ylim(win[1] - 35, win[1] + 35)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"{lab_} (ITK-SNAP numbering)", fontsize=12)
    fig.suptitle(
        f"Per-front arrival at the kissing stretch ({SITE}, distance; all 3D). Walls from roots: front 1 blue, front 2 orange. "
        "Lime: equal arrival (T1 = T2); cyan: pairwise posterior; green band: +-1 SD;\nfaint magenta: the all-WM "
        "field's sheet (pinned to the geometric midline); yellow x: your points",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/collide2_contact.png", dpi=75)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(12, 6))
    ks = np.arange(1, len(rows) + 1)
    a.plot(
        ks,
        [r["eq"] for r in rows],
        "_",
        ms=14,
        mew=2,
        c="lime",
        label="equal arrival (T1 = T2)",
    )
    a.errorbar(
        ks,
        [r["post"] for r in rows],
        yerr=[r["band"] for r in rows],
        fmt="o",
        c="#30a0d0",
        label="pairwise posterior +- SD",
    )
    a.axhline(0, c="k", lw=0.8)
    a.set_xlabel("your contact point")
    a.set_ylabel(
        "rule's contact - your point (mm, + toward the side predicted thinner)"
    )
    a.set_title("Per-front arrival: offsets at your 23 points")
    a.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/collide2_offsets.png", dpi=80)
    plt.close(fig)
    print("wrote collide2_contact.png, collide2_offsets.png")


def _pts_mask(P, shape):
    m = np.zeros(shape, bool)
    m[tuple(np.round(P).astype(int).T)] = True
    return m


if __name__ == "__main__":
    main()
