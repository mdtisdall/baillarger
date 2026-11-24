"""Collisions from fronts grown from all white matter, and the pairwise posterior contact (M1S1).

All computed in 3D, in the M1S1 test crop grown by 15 voxels (the box of curv_map.py), loaded from
the full volume; box faces are never WM boundary. No bank labels anywhere.

1. The kissing gap is closed by the rule in CLOSURE (environment variable; close_gap): 'points' (default),
   fluid within 1.5 mm of the user's 23 contact points; 'none'; or 'thin', slivers thinner than about 3 voxels.
2. Fronts: one data-steered unit-speed T with T = 0 on all WM boundary (curv_lib.fit_all_wm; the data
   tensor uses the unclosed cortex). Roots: one upwind pass in increasing T (prob_thickness.upwind_fields):
   every cortex voxel takes the root of its best upwind neighbour on the same front.
3. Collisions: a voxel whose 6-neighbours have roots more than D_SAME along the white surface (geodesic
   on the WM-boundary graph, same side of the WM only) from its own. Classified by the roots in its
   3x3x3 neighbourhood, clustered at D_FAR: two clusters D_FAR apart = sheet (two fronts: a contact);
   three or more = junction; roots differing but all within D_FAR = sheet edge (a front meeting itself).
4. Thickness (distance): visible tops (cortex next to fluid that stays fluid after the closure, >= 2 mm
   from the faces) give their
   root's node log(T at the top). Field on the WM-boundary graph (shell voxels pooled in 2x2x2 blocks,
   split by side), log t = m + (a0 + w(x)) H(x) + z(x) + noise:
     z  Matern-like GMRF, length 2 mm, SD 0.2 (log): the level, carried sideways
     w  GMRF, length 6 mm, SD 0.3 per (1/mm): the curvature coefficient, varying in space (the
        continuous form of a per-region coefficient with a shared prior; no area labels yet)
     a0 ~ N(0.5, 0.5^2) per (1/mm), m ~ N(log 2.5, 1): shared coefficient and level
     H  white-surface mean curvature at the root, scale 0.80 mm, clipped to +-0.8 /mm
   Exact posterior mean; variances from 100 perturb-and-MAP samples. Predictive thickness: lognormal.
5. Contact on sheets: for each pair of 6-neighbours across a sheet, with roots r1, r2, the path length
   L = T(p) + T(q) + |pq|; equal arrival at the sub-voxel point of pq where both arrival times agree; contact at x1 = mu1 + s1^2/(s1^2 + s2^2) (L - mu1 - mu2) from root 1, SD
   s1 s2 / sqrt(s1^2 + s2^2). Junctions and edges are flagged, not given a contact.
6. Scoring at the 23 points: pairs whose equal-arrival point is within 1 mm; offset along the local
   column direction across the sheet (n1(p) - n1(q), normalized),
   + toward the side predicted thinner. Calibration: 5 spatial folds of evidence nodes, with and
   without the curvature trend.

Usage: collide.py <outdir> <workdir>. Needs curv_M1S1.npz (curv_map.py).
"""

import json
import os
import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu
from skimage.measure import marching_cubes

import curv_lib as C
import flatten_fused2 as F
from prob_thickness import upwind_fields

OUT, S = sys.argv[1], sys.argv[2]
GROW = int(os.environ.get("GROW", 15))  # margin added around the points file's box (voxels)
SITE = os.environ.get("SITE", "M1S1")  # names the curvature and cache files: curv_<SITE>.npz, T_<SITE>_closed.npz
POINTS = os.environ.get("POINTS", "M1S1_test_points.json")  # full_lo, full_hi and gap_all (0-based, that box)
CLOSE_MM, FACE_MM = 1.5, 2.0
D_SAME, D_FAR = 1.0, 3.0
SIG_Z, ELL_Z, SIG_W, ELL_W = 0.2, 2.0, 0.3, 6.0
A0_MU, A0_SD, M_MU, M_SD, SIG_EPS = 0.5, 0.5, np.log(2.5), 1.0, 0.08
H_SCALE, H_CLIP = "0.8", 0.8
NSAMP = 100
TYPES = {1: ("sheet", "#ff2bd6"), 2: ("junction", "#ff3b30"), 3: ("sheet edge", "#30d5ff")}


def view_planes(P):
    """Eight planes for the views: the user's checked slices at M1S1; elsewhere the four k slices with the
    most points, and the i and j planes through the points' median and through their quartiles."""
    if SITE == "M1S1":
        planes = [(0, 45 + GROW, "i = 46"), (0, 27 + GROW, "i = 28"), (0, 19 + GROW, "i = 20"), (2, 89 + GROW, "k = 90")]
        for k_ in (3, 4):
            qk = np.round(P[k_]).astype(int)
            planes += [(1, qk[1], f"j = {qk[1] - GROW + 1} (point {k_ + 1})"), (2, qk[2], f"k = {qk[2] - GROW + 1} (point {k_ + 1})")]
        return planes
    Pr = np.round(P).astype(int)
    ks, cnt = np.unique(Pr[:, 2], return_counts=True)
    top = sorted(ks[np.argsort(cnt)[::-1][:4]])
    planes = [(2, int(k), f"k = {int(k) - GROW + 1}") for k in top]
    for ax in (0, 1):
        for q in (25, 75):
            c = int(np.percentile(Pr[:, ax], q))
            planes.append((ax, c, f"{'ij'[ax]} = {c - GROW + 1}"))
    return planes[:8]


# ------------------------------------------------------------------ the kissing gap
CLOSURE = os.environ.get("CLOSURE", "points")


def close_gap(rim0, P, vox, dface):
    """Fluid voxels that become cortex. 'points': within CLOSE_MM of the user's points (uses the points);
    'none': nothing; 'thin': fluid slivers that an opening with the 6-neighbour ball removes (thinner than
    about 3 voxels), at least 2 mm from the box faces (does not use the points)."""
    fl = rim0 == 1
    if CLOSURE == "points":
        fi = np.argwhere(fl)
        dfl = np.min(np.linalg.norm((fi[:, None, :] - P[None]) * vox, axis=2), 1)
        return fi[dfl < CLOSE_MM]
    if CLOSURE == "none":
        return np.zeros((0, 3), int)
    if CLOSURE == "thin":
        thin = fl & ~ndi.binary_opening(fl, structure=ndi.generate_binary_structure(3, 1)) & (dface >= FACE_MM)
        return np.argwhere(thin)
    raise ValueError(CLOSURE)


# ------------------------------------------------------------------ geometry
def shell_graph(pts, n, vox):
    """26-neighbour graph on WM-boundary voxels, edges only between the same side (normals agree)."""
    idx = -np.ones(tuple(pts.max(0) + 2), np.int64); idx[tuple(pts.T)] = np.arange(len(pts))
    rows, cols, w = [], [], []
    for d in np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]):
        q = pts + d
        ok = np.all(q >= 0, 1) & np.all(q < np.array(idx.shape), 1)
        j = np.full(len(pts), -1); j[ok] = idx[tuple(q[ok].T)]
        sel = (j >= 0) & ((n * n[np.maximum(j, 0)]).sum(1) > 0.0)
        i0, j0 = np.flatnonzero(sel), j[sel]
        dd = np.linalg.norm(d * vox)
        rows += [i0, j0]; cols += [j0, i0]; w += [np.full(len(i0), dd)] * 2
    return sparse.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(len(pts),) * 2)


def geodesics(G, pairs, limit):
    """Geodesic distance (inf beyond limit) for each (r1, r2) pair."""
    out = np.full(len(pairs), np.inf)
    src = np.unique(pairs[:, 0])
    pos = {s: i for i, s in enumerate(src)}
    order = np.argsort(pairs[:, 0], kind="stable")
    ps = pairs[order]
    bounds = np.searchsorted(ps[:, 0], src)
    for c0 in range(0, len(src), 150):
        ch = src[c0:c0 + 150]
        D = dijkstra(G, indices=ch, limit=limit)
        for li, s in enumerate(ch):
            b0 = bounds[c0 + li]; b1 = bounds[c0 + li + 1] if c0 + li + 1 < len(src) else len(ps)
            out[order[b0:b1]] = D[li, ps[b0:b1, 1]]
    return out


# ------------------------------------------------------------------ thickness field
def graph_L(pos, adj_pairs):
    i, j = adj_pairs[:, 0], adj_pairs[:, 1]
    d2 = np.maximum(((pos[i] - pos[j]) ** 2).sum(1), 0.16 ** 2)
    n = len(pos)
    W = sparse.csr_matrix((np.r_[1 / d2, 1 / d2], (np.r_[i, j], np.r_[j, i])), shape=(n, n))
    deg = np.asarray(W.sum(1)).ravel()
    return ((sparse.diags(deg) - W) / max(np.median(deg) * 0.32 ** 2 / 4, 1e-9)).tocsc()


def prior_op(L, ell, sig):
    n = L.shape[0]
    Aop = ((1 / ell) ** 2 * sparse.identity(n) + L).tocsc()
    lu = splu(Aop)
    probe = np.random.default_rng(0).choice(n, size=min(40, n), replace=False)
    E = np.zeros((n, len(probe))); E[probe, np.arange(len(probe))] = 1
    v = (lu.solve(E) ** 2).sum(0).mean()  # (A^-1 A^-T)_ii = |A^-T e_i|^2 (A symmetric)
    tau = v / sig ** 2
    return dict(A=Aop, lu=lu, tau=tau, Q=(tau * (Aop.T @ Aop)).tocsc())


class Field:
    def __init__(self, L, H):
        self.n = L.shape[0]
        self.H = H
        self.pz = prior_op(L, ELL_Z, SIG_Z)
        self.pw = prior_op(L, ELL_W, SIG_W)

    def fit(self, y, trend=True, nsamp=NSAMP, seed=1):
        """y: log thickness per node (nan = no evidence). Returns predictive mean and SD of log t per
        node, and the posterior of (m, a0)."""
        n, H = self.n, self.H
        o = np.flatnonzero(np.isfinite(y))
        nu = 2 * n + 2 if trend else n + 1
        im = nu - 2 if trend else n
        ia = nu - 1
        blocks = [self.pz["Q"]] + ([self.pw["Q"]] if trend else [])
        P = sparse.block_diag(blocks + [sparse.diags([1 / M_SD ** 2] + ([1 / A0_SD ** 2] if trend else []))]).tocsc()
        pm = np.zeros(nu); pm[im] = M_MU
        if trend:
            pm[ia] = A0_MU
        rows = np.r_[np.arange(len(o)), np.arange(len(o))]
        colz = np.r_[o, np.full(len(o), im)]
        vals = np.ones(2 * len(o))
        if trend:
            rows = np.r_[rows, np.arange(len(o)), np.arange(len(o))]
            colz = np.r_[colz, n + o, np.full(len(o), ia)]
            vals = np.r_[vals, H[o], H[o]]
        A = sparse.csr_matrix((vals, (rows, colz)), shape=(len(o), nu))
        Q = (P + A.T @ A / SIG_EPS ** 2).tocsc()
        lu = splu(Q)
        xhat = lu.solve(P @ pm + A.T @ y[o] / SIG_EPS ** 2)
        self.last = dict(xhat=xhat, n=n, im=im, ia=ia, trend=trend)  # for inspection only

        def pred(x):
            out = x[:n] + x[im]
            if trend:
                out = out + H * (x[n:2 * n] + x[ia])
            return out

        rng = np.random.default_rng(seed)
        S_ = []
        for _ in range(nsamp):
            x0 = pm.copy()
            x0[:n] += self.pz["lu"].solve(rng.standard_normal(n)) / np.sqrt(self.pz["tau"])
            if trend:
                x0[n:2 * n] += self.pw["lu"].solve(rng.standard_normal(n)) / np.sqrt(self.pw["tau"])
                x0[ia] += A0_SD * rng.standard_normal()
            x0[im] += M_SD * rng.standard_normal()
            yp = y[o] + SIG_EPS * rng.standard_normal(len(o))
            xs = x0 + lu.solve(A.T @ (yp - A @ x0) / SIG_EPS ** 2)
            S_.append(np.r_[pred(xs), xs[im], xs[ia] if trend else 0.0, xs[n:2 * n] + xs[ia] if trend else np.zeros(n)])
        S_ = np.array(S_)
        mu = pred(xhat)
        sd = S_[:, :n].std(0)
        coef = dict(m=(xhat[im], S_[:, n].std()), a0=((xhat[ia], S_[:, n + 1].std()) if trend else (0.0, 0.0)),
                    a=(xhat[n:2 * n] + xhat[ia]) if trend else np.zeros(n), a_sd=S_[:, n + 2:].std(0))
        return mu, sd, coef


def lognormal(mu, sd):
    m = np.exp(mu + sd ** 2 / 2)
    return m, m * np.sqrt(np.expm1(sd ** 2))


# ------------------------------------------------------------------ main
def main():
    m = json.load(open(f"{S}/{POINTS}"))
    lo = np.array(m["full_lo"]) - GROW; hi = np.array(m["full_hi"]) + GROW
    B = C.load(lo, hi, SITE)
    vox = B["vox"]
    P = np.array(m["gap_all"], float) + GROW
    cz = np.load(f"{S}/curv_{SITE}.npz")
    assert np.array_equal(cz["lo"], B["lo"])
    pts, nrm = cz["pts"], cz["n"]
    Hroot = np.clip(0.5 * (cz[f"k1_{H_SCALE}"] + cz[f"k2_{H_SCALE}"]), -H_CLIP, H_CLIP)

    # ---- 1. close the gap
    rim0 = B["rim"]
    rim = rim0.copy()
    closed = close_gap(rim0, P, vox, B["dface"])
    rim[tuple(closed.T)] = 3
    ctx0 = B["ctx"]
    ctx = (rim == 3) & ~B["miss"]
    print(f"box {rim.shape}; gap closure '{CLOSURE}': {len(closed)} fluid voxels become cortex")

    # ---- 2. fronts and roots
    try:
        z = np.load(f"{S}/T_{SITE}_closed.npz")
        Fd = dict(T=z["T"], Tf=z["Tf"], g=z["g"])
        print("T from cache")
    except FileNotFoundError:
        Fd = C.fit_all_wm(B, dom_ctx=ctx)
        np.savez_compressed(f"{S}/T_{SITE}_closed.npz", T=Fd["T"], Tf=Fd["Tf"], g=Fd["g"])
        print(f"T fitted, cg info {Fd['info']}")
    T, Tf = Fd["T"], Fd["Tf"]
    dom = np.isfinite(T)
    fixed = dom & B["wm"]
    rootid = -np.ones(rim.shape, np.int64); rootid[tuple(pts.T)] = np.arange(len(pts))
    assert (rootid[fixed] >= 0).all()
    n1 = np.moveaxis(Fd["g"], 0, -1); n1 = n1 / (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-12)
    order = np.argwhere(dom)[np.argsort(Tf[dom])].astype(np.int64)
    _, _, root = upwind_fields(order, dom, fixed, rootid, Tf, n1, np.zeros(rim.shape), vox.astype(float))
    root = np.where(ctx, root, -1)
    print(f"roots reached {np.mean(root[ctx] >= 0):.1%} of cortex")

    # ---- 3. collisions
    G = shell_graph(pts, nrm, vox)
    ppos = pts * vox
    cv = np.argwhere(ctx & (root >= 0))
    shp = np.array(rim.shape)
    nb6 = [d for d in np.eye(3, dtype=int)]
    pair_list = []
    for d in nb6:
        q = cv + d
        ok = np.all(q < shp, 1)
        a, b = cv[ok], q[ok]
        rb_ = root[tuple(b.T)]
        keep = rb_ >= 0
        a, b, ra, rb_ = a[keep], b[keep], root[tuple(a[keep].T)], rb_[keep]
        far = np.linalg.norm(ppos[ra] - ppos[rb_], axis=1) > 0.5
        pair_list.append(np.c_[a[far], b[far], ra[far], rb_[far]])
    PL = np.vstack(pair_list)
    rp = np.sort(PL[:, 6:8], 1)
    g = geodesics(G, rp, D_FAR + 0.5)
    PL = PL[g > D_SAME]; gP = g[g > D_SAME]
    print(f"voxel pairs with roots > {D_SAME} mm apart along the white surface: {len(PL):,}")
    coll = np.zeros(rim.shape, bool)
    coll[tuple(PL[:, 0:3].T)] = True; coll[tuple(PL[:, 3:6].T)] = True
    cvox = np.argwhere(coll)
    # classify by clustering the roots of each collision voxel's 3x3x3 neighbourhood
    offs = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
    nbr = cvox[:, None, :] + offs[None]
    nbr = np.clip(nbr, 0, shp - 1)
    R = root[tuple(nbr.reshape(-1, 3).T)].reshape(len(cvox), 27)
    cand = set()
    Rl = []
    for row in R:
        u = np.unique(row[row >= 0])
        Rl.append(u)
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                if np.linalg.norm(ppos[u[i]] - ppos[u[j]]) > 0.5:
                    cand.add((u[i], u[j]))
    cand = np.array(sorted(cand))
    gc = geodesics(G, cand, D_FAR + 0.5)
    gd = {tuple(c): v for c, v in zip(cand, gc)}
    gget = lambda a, b: 0.0 if np.linalg.norm(ppos[a] - ppos[b]) <= 0.5 else gd.get((min(a, b), max(a, b)), np.inf)
    ctype = np.zeros(rim.shape, np.int8)
    for v, u in zip(cvox, Rl):
        reps = []
        for r in u:
            if not any(gget(r, s) <= D_FAR for s in reps):
                reps.append(r)
        ctype[tuple(v)] = 1 if len(reps) == 2 else (2 if len(reps) >= 3 else 3)
    for t, (nm, _) in TYPES.items():
        print(f"   {nm}: {np.sum(ctype == t):,} voxels")

    # ---- 4. thickness field
    # nodes: shell voxels pooled in 2x2x2 blocks, split by side (dominant normal direction)
    side = np.argmax(np.abs(nrm), 1) * 2 + (nrm[np.arange(len(nrm)), np.argmax(np.abs(nrm), 1)] > 0)
    key = np.c_[pts // 2, side]
    _, node_of, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    node_of = node_of.ravel()
    nN = node_of.max() + 1
    npos = np.stack([np.bincount(node_of, ppos[:, a], nN) for a in range(3)], 1) / counts[:, None]
    nnrm = np.stack([np.bincount(node_of, nrm[:, a], nN) for a in range(3)], 1)
    nnrm /= np.linalg.norm(nnrm, axis=1, keepdims=True) + 1e-12
    nH = np.bincount(node_of, Hroot, nN) / counts
    Gc = G.tocoo()
    ea, eb = node_of[Gc.row], node_of[Gc.col]
    ed = np.unique(np.sort(np.c_[ea, eb][ea != eb], 1), axis=0)
    L = graph_L(npos, ed)
    ncomp, comp = sparse.csgraph.connected_components(L != 0, directed=False)
    print(f"field: {nN:,} nodes (from {len(pts):,} WM-boundary voxels), {len(ed):,} edges, {ncomp} pieces")
    # evidence: visible tops
    tops = ctx0 & ndi.binary_dilation(rim == 1) & (B["dface"] >= FACE_MM) & (root >= 0) & np.isfinite(T)
    tq = np.argwhere(tops)
    tv = T[tuple(tq.T)]
    tr = root[tuple(tq.T)]
    ok = (tv > 0.8) & (tv < 7.0)
    tq, tv, tr = tq[ok], tv[ok], tr[ok]
    tn = node_of[tr]
    y = np.full(nN, np.nan)
    srt = np.argsort(tn)
    ends = np.r_[np.flatnonzero(np.diff(tn[srt])) + 1, len(srt)]
    starts = np.r_[0, ends[:-1]]
    for s0, s1 in zip(starts, ends):
        y[tn[srt[s0]]] = np.log(np.median(tv[srt[s0:s1]]))
    ev = np.flatnonzero(np.isfinite(y))
    print(f"evidence: {len(tv):,} top voxels -> {len(ev):,} nodes; thickness median {np.exp(np.median(y[ev])):.2f} mm, "
          f"H at evidence nodes p5/50/95 {np.percentile(nH[ev], [5, 50, 95]).round(2)}")
    FLD = Field(L, nH)
    mu, sd, coef = FLD.fit(y, trend=True)
    mu0, sd0, coef0 = FLD.fit(y, trend=False)
    print(f"trend: a0 = {coef['a0'][0]:+.2f} +- {coef['a0'][1]:.2f} per (1/mm) in log thickness; "
          f"local a(x) p5/50/95 {np.percentile(coef['a'], [5, 50, 95]).round(2)}; m = {np.exp(coef['m'][0]):.2f} mm")
    tmean, tsd = lognormal(mu, sd)
    tmean0, tsd0 = lognormal(mu0, sd0)

    # ---- calibration, with and without the trend
    c = npos[ev]
    ax_ = np.linalg.svd(c - c.mean(0))[2][0]
    folds = np.array_split(ev[np.argsort((c - c.mean(0)) @ ax_)], 5)
    cal = {}
    for trend in (True, False):
        zs, dd, hh, err = [], [], [], []
        for f in folds:
            yy = y.copy(); yy[f] = np.nan
            m_, s_, _ = FLD.fit(yy, trend=trend, nsamp=60)
            rest = np.setdiff1d(ev, f)
            zs.append((y[f] - m_[f]) / np.sqrt(s_[f] ** 2 + SIG_EPS ** 2))
            err.append(y[f] - m_[f])
            from scipy.spatial import cKDTree
            dd.append(cKDTree(npos[rest]).query(npos[f])[0])
            hh.append(nH[f])
        cal[trend] = tuple(np.concatenate(x) for x in (zs, dd, hh, err))
        z_, d_, h_, e_ = cal[trend]
        print(f"calibration ({'with' if trend else 'without'} trend): {len(z_)} held-out nodes; z sd {z_.std():.2f}, "
              f"|z| < 1.96 {np.mean(np.abs(z_) < 1.96):.0%}; RMSE (log) {np.sqrt(np.mean(e_ ** 2)):.3f}; "
              f"mean error by H tercile {[round(float(np.mean(e_[(h_ >= a) & (h_ <= b)])), 3) for a, b in zip(np.percentile(h_, [0, 33, 67]), np.percentile(h_, [33, 67, 100]))]}")

    # ---- 5. contact on sheets
    sh = (ctype[tuple(PL[:, 0:3].T)] == 1) & (ctype[tuple(PL[:, 3:6].T)] == 1) & (gP > D_FAR)
    SP = PL[sh]
    p, q = SP[:, 0:3].astype(float), SP[:, 3:6].astype(float)
    r1, r2 = SP[:, 6], SP[:, 7]
    n1_, n2_ = node_of[r1], node_of[r2]
    m1, s1, m2, s2 = tmean[n1_], tsd[n1_], tmean[n2_], tsd[n2_]
    dpq = np.linalg.norm((q - p) * vox, axis=1)
    T1, T2 = T[tuple(SP[:, 0:3].T)], T[tuple(SP[:, 3:6].T)]
    Lp = T1 + T2 + dpq
    x1 = np.clip(m1 + s1 ** 2 / (s1 ** 2 + s2 ** 2) * (Lp - m1 - m2), 0, Lp)
    sdc = s1 * s2 / np.sqrt(s1 ** 2 + s2 ** 2)
    # equal arrival at sub-voxel position on pq: T(p) + t = T(q) + (|pq| - t)
    tt = np.clip((dpq + T2 - T1) / 2, 0, dpq)
    mid = p + (tt / dpq)[:, None] * (q - p)
    # the column direction across the sheet, from side 1 into side 2 (not the grid axis of pq)
    g1 = n1[tuple(SP[:, 0:3].T)]; g2 = n1[tuple(SP[:, 3:6].T)]
    u = g1 - g2; u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    ridge1 = T1 + tt  # equal arrival, measured from root 1
    delta = x1 - ridge1  # mm along the columns, + into side 2
    cpt = mid + (delta[:, None] * u) / vox
    print(f"sheet pairs: {len(SP):,}; posterior shift from the ridge median |{np.median(np.abs(delta)):.2f}| mm, "
          f"contact SD median {np.median(sdc):.2f} mm; misfit L - mu1 - mu2 median {np.median(Lp - m1 - m2):+.2f} mm")

    # ---- 6. scoring at the 23 points
    rows = []
    for k_, Pk in enumerate(P):
        dm = np.linalg.norm((mid - Pk) * vox, axis=1)
        near = np.flatnonzero(dm < 1.0)
        cvn = cvox[np.linalg.norm((cvox - Pk) * vox, axis=1) < 1.0]
        tcount = {t: int(np.sum(ctype[tuple(cvn.T)] == t)) for t in TYPES} if len(cvn) else {t: 0 for t in TYPES}
        kind = max(tcount, key=tcount.get) if sum(tcount.values()) else 0
        if len(near) == 0:
            rows.append(dict(k=k_, kind=kind, tc=tcount, eq=np.nan, post=np.nan, sd=np.nan, n=0)); continue
        thick1 = m1[near] >= m2[near]  # orient + toward the side predicted thinner
        sgn = np.where(thick1, 1.0, -1.0)
        oe = ((mid[near] - Pk) * vox * u[near]).sum(1) * sgn
        op = ((cpt[near] - Pk) * vox * u[near]).sum(1) * sgn
        rows.append(dict(k=k_, kind=kind, tc=tcount, eq=np.median(oe), post=np.median(op), sd=np.median(sdc[near]), n=len(near),
                         mu=(np.median(np.maximum(m1[near], m2[near])), np.median(np.minimum(m1[near], m2[near]))),
                         L=np.median(Lp[near])))
    print("per point: type (voxel counts sheet/junction/edge within 1 mm), pairs, equal arrival, posterior (mm, + toward thinner)")
    for r in rows:
        print(f"   {r['k'] + 1:2d} ITK {tuple((P[r['k']] - GROW + 1).astype(int))}: {TYPES.get(r['kind'], ('none',))[0]:10s} "
              f"{r['tc'][1]:4d}/{r['tc'][2]:3d}/{r['tc'][3]:3d}  n {r['n']:3d}  eq {r['eq']:+.2f}  post {r['post']:+.2f}"
              + (f"  sd {r['sd']:.2f}  mu {r['mu'][0]:.2f}/{r['mu'][1]:.2f}  L {r['L']:.2f}" if r['n'] else ""))
    for lab, kk in (("equal arrival", "eq"), ("posterior", "post")):
        v = np.array([r[kk] for r in rows])
        print(f"   {lab:14s} median {np.nanmedian(v):+.2f}, IQR {np.subtract(*np.nanpercentile(v, [75, 25])):.2f}, "
              f"|offset| median {np.nanmedian(np.abs(v)):.2f}, missing {np.isnan(v).sum()}")
    np.savez_compressed(f"{S}/collide_state.npz", root=root, ctype=ctype, tmean=tmean, tsd=tsd, node_of=node_of,
                        cpt=cpt, mid=mid, sdc=sdc, u=u, delta=delta)

    # ------------------------------------------------------------------ views
    disp = B["disp"]
    vmax = np.nanpercentile(np.abs(disp[ctx0]), 98)
    rpos = ppos[np.maximum(root, 0)]
    lo3 = P.min(0) * vox - 6; hi3 = P.max(0) * vox + 6

    def own_rgb(sl):
        rp_ = rpos[sl]
        rgb = np.clip((rp_ - lo3) / (hi3 - lo3), 0, 1)
        a_ = np.where(ctx[sl] & (root[sl] >= 0), 0.5, 0.0)
        return np.concatenate([rgb, a_[..., None]], -1)

    def descend(x):
        pts_ = [x.copy()]
        for _ in range(400):
            v = -np.array([F.interp(cc, x) for cc in Fd["g"]]); nv = np.linalg.norm(v)
            if nv < 1e-9:
                break
            x = x + 0.25 * v / nv; pts_.append(x.copy())
            if B["wm"][tuple(np.clip(np.round(x).astype(int), 0, shp - 1))]:
                break
        return np.array(pts_)

    def panel(a, ax_i, c0, window, show_post=False):
        sl = [slice(None)] * 3; sl[ax_i] = c0; sl = tuple(sl)
        oth = [x for x in range(3) if x != ax_i]
        a.imshow(np.where(ctx[sl], disp[sl], np.nan).T, origin="lower", cmap="gray", vmin=-vmax, vmax=vmax)
        if not show_post:
            a.imshow(np.transpose(own_rgb(sl), (1, 0, 2)), origin="lower")
        a.contour(B["wm"][sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.6)
        a.contour((rim0 == 1)[sl].T.astype(float), levels=[0.5], colors="c", linewidths=0.5)
        for t, (nm, col) in TYPES.items():
            vv = np.argwhere(ctype[sl] == t)
            if len(vv) and (not show_post or t == 1):
                a.scatter(vv[:, 0], vv[:, 1], s=4, c=col, marker="s", linewidths=0, label=nm)
        if not show_post:
            sel = np.flatnonzero(np.abs(SP[:, ax_i] - c0) == 0)
            if len(sel):
                for s_ in sel[np.linspace(0, len(sel) - 1, min(8, len(sel))).astype(int)]:
                    for st in (SP[s_, 0:3], SP[s_, 3:6]):
                        tr_ = descend(st.astype(float))
                        a.plot(tr_[:, oth[0]], tr_[:, oth[1]], "-", c="w", lw=0.9)
                        rr = pts[root[tuple(st)]]
                        a.plot(rr[oth[0]], rr[oth[1]], "o", c="w", ms=3)
        else:
            sel = np.flatnonzero(np.abs(np.round(mid[:, ax_i]) - c0) <= 0.5)
            if len(sel):
                cp = cpt[sel]; uu = u[sel] / vox; uu /= np.linalg.norm(uu, axis=1, keepdims=True)
                for li, (s_, cc_) in enumerate(zip(sel, cp)):
                    e0 = cc_ - sdc[s_] / 0.16 * uu[li]; e1 = cc_ + sdc[s_] / 0.16 * uu[li]
                    a.plot([e0[oth[0]], e1[oth[0]]], [e0[oth[1]], e1[oth[1]]], "-", c="lime", lw=0.5, alpha=0.5)
                a.scatter(cp[:, oth[0]], cp[:, oth[1]], s=5, c="lime", label="posterior contact (+- 1 SD)")
        selp = np.abs(P[:, ax_i] - c0) <= 1.5
        a.plot(P[selp, oth[0]], P[selp, oth[1]], "x", c="#ffd23f", ms=9, mew=2, label="your points")
        if window is not None:
            a.set_xlim(window[0] - 35, window[0] + 35); a.set_ylim(window[1] - 35, window[1] + 35)
        a.set_xticks([]); a.set_yticks([])

    planes = view_planes(P)
    for fname, post in (("collide_map.png", False), ("collide_contact.png", True)):
        fig, axs = plt.subplots(2, 4, figsize=(28, 14.5))
        for a, (ax_i, c0, lab) in zip(axs.ravel(), planes):
            oth = [x for x in range(3) if x != ax_i]
            selp = np.abs(P[:, ax_i] - c0) <= 1.5
            win = P[selp][:, oth].mean(0) if selp.any() else P[:, oth].mean(0)
            panel(a, ax_i, c0, win, show_post=post)
            a.set_title(f"{lab} (ITK-SNAP numbering)", fontsize=12)
        axs[0, 0].legend(loc="lower left", fontsize=8, markerscale=2)
        if not post:
            fig.suptitle(f"Collisions from fronts grown from all WM ({SITE}, gap closure '{CLOSURE}'; all 3D). Cortex tinted by "
                         "each voxel's root position (colour = where on the white surface its column starts);\nmagenta sheet "
                         "(two fronts), red junction, cyan sheet edge (a front meeting itself); white: columns from sheet "
                         "voxels traced down to their roots; yellow x: your points", fontsize=13)
        else:
            fig.suptitle("Pairwise posterior contact on sheets (distance; thickness field with curvature trend): green dots "
                         "with +-1 SD along the pair; magenta: sheet voxels (equal arrival lies between them); yellow x: "
                         "your points", fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{fname}", dpi=75)
        plt.close(fig)
        print("wrote", fname)

    # thickness on the white surface (3D, within 7 mm of the points)
    verts, faces, _, _ = marching_cubes(ndi.gaussian_filter(B["wm"].astype(float), 2.0), 0.5, step_size=1)
    vi = np.clip(np.round(verts).astype(int), 0, shp - 1)
    idx = -np.ones(rim.shape, np.int64); idx[tuple(pts.T)] = np.arange(len(pts))
    _, ind = ndi.distance_transform_edt(idx < 0, return_indices=True)
    vnode = node_of[idx[tuple(ind[:, vi[:, 0], vi[:, 1], vi[:, 2]])]]
    Pv = verts * vox
    cen = P.mean(0) * vox
    keep = (np.abs(Pv[faces].mean(1) - cen) < 7.0).all(1) & (B["dface"][tuple(vi.T)] >= 2.0)[faces].all(1)
    f_ = faces[keep]; tp = Pv[f_]
    fn = np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    evn = np.zeros(nN, bool); evn[ev] = True
    panels3 = [("predicted thickness mean (mm), with trend", tmean, "viridis", np.percentile(tmean[ev], [2, 98])),
               ("predicted thickness SD (mm), with trend", tsd, "magma", (0, np.percentile(tsd, 98))),
               ("local curvature coefficient a(x) (log t per 1/mm)", coef["a"], "PuOr", (-1.2, 1.2)),
               ("white-surface mean curvature H (1/mm, scale 0.8 mm)", nH, "RdBu_r", (-0.8, 0.8)),
               ("mean without trend minus with trend (mm)", tmean0 - tmean, "RdBu_r", (-0.5, 0.5)),
               ("evidence (visible tops, nodes in white)", evn.astype(float), "Greys_r", (0, 1))]
    fig = plt.figure(figsize=(30, 18))
    for c_, (lab, val, cm, (vmin, vmax_)) in enumerate(panels3):
        a = fig.add_subplot(2, 3, c_ + 1, projection="3d")
        el, az = 30, -60
        light = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
        shade = 0.45 + 0.55 * np.abs(fn @ light)
        fv = val[vnode][f_].mean(1)
        col = plt.get_cmap(cm)(np.clip((fv - vmin) / (vmax_ - vmin + 1e-12), 0, 1)); col[:, :3] *= shade[:, None]
        a.add_collection3d(Poly3DCollection(tp, facecolors=col, linewidths=0))
        qq = Pv[np.unique(f_)]
        a.set_xlim(qq[:, 0].min(), qq[:, 0].max()); a.set_ylim(qq[:, 1].min(), qq[:, 1].max()); a.set_zlim(qq[:, 2].min(), qq[:, 2].max())
        a.set_box_aspect(np.ptp(qq, 0)); a.view_init(el, az)
        a.scatter(*(P * vox).T, c="r", s=10, depthshade=False)
        fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin, vmax_)), ax=a, shrink=0.6)
        a.set_title(lab, fontsize=13)
    fig.suptitle(f"Thickness field on the white surface within 7 mm of your points (red); log t = m + (a0 + w(x)) H + z(x); "
                 f"a0 = {coef['a0'][0]:+.2f} +- {coef['a0'][1]:.2f}", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{OUT}/collide_thickness.png", dpi=70); plt.close(fig)
    print("wrote collide_thickness.png")

    # offsets and calibration
    fig, axs = plt.subplots(1, 4, figsize=(30, 7))
    a = axs[0]
    for kind in (1, 2, 3, 0):
        rr = [r for r in rows if r["kind"] == kind]
        if not rr:
            continue
        nm, col = TYPES.get(kind, ("none", "k"))
        a.scatter([r["k"] + 1 for r in rr], [r["eq"] for r in rr], marker="_", s=120, c=col, label=f"equal arrival ({nm})")
        a.errorbar([r["k"] + 1 for r in rr], [r["post"] for r in rr], yerr=[r["sd"] for r in rr], fmt="o", c=col,
                   ms=6, label=f"posterior +- SD ({nm})")
    a.axhline(0, c="k", lw=0.8)
    a.set_xlabel("your contact point"); a.set_ylabel("rule's contact - your point (mm, + toward the side predicted thinner)")
    a.set_title("offsets at your 23 points, by the collision type nearby"); a.legend(fontsize=8)
    for c_, (trend, lab) in enumerate(((True, "with curvature trend"), (False, "without trend"))):
        z_, d_, h_, e_ = cal[trend]
        axs[1].hist(z_, bins=np.linspace(-5, 5, 41), alpha=0.5, label=f"{lab}: sd {z_.std():.2f}, RMSE {np.sqrt(np.mean(e_ ** 2)):.3f}")
        axs[2].scatter(d_, np.abs(z_), s=6, alpha=0.5, label=lab)
        axs[3].scatter(h_, e_, s=6, alpha=0.4, label=lab)
    xx = np.linspace(-5, 5, 200)
    axs[1].plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi) * len(cal[True][0]) * 0.25, "k--", label="N(0,1)")
    axs[1].set_title("held-out z-scores (5 spatial folds of evidence)"); axs[1].legend(fontsize=9)
    axs[2].axhline(1.96, c="k", ls="--"); axs[2].set_xlabel("distance to nearest kept evidence (mm)"); axs[2].set_ylabel("|z|")
    axs[2].set_title("calibration against distance from evidence"); axs[2].legend(fontsize=9)
    axs[3].axhline(0, c="k", lw=0.8); axs[3].set_xlabel("white-surface mean curvature H at the node (1/mm)")
    axs[3].set_ylabel("held-out error, log thickness"); axs[3].set_title("held-out error against curvature"); axs[3].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/collide_offsets.png", dpi=75); plt.close(fig)
    print("wrote collide_offsets.png")


if __name__ == "__main__":
    main()
