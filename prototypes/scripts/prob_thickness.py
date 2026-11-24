"""Probabilistic thickness field and contact posterior (M1S1 test crop, user's 23 contact points).

All computed in 3D in the test crop (>= 7 mm from every face; box faces are never WM boundary).
1. Walls: WM-boundary voxels within 7 mm of the contact points, split into bank A / bank B by
   the side of the local contact surface (PCA normal of nearby points, orientation propagated
   from bank A), facing the contact.
2. The kissing gap is closed: aseg fluid within 1.5 mm of a contact point becomes cortex.
3. Per bank: T fitted from its own wall (data-steered); in increasing T, an upwind pass carries
   u = ln(A/A0), the equivolume coordinate V and the column's root (wall voxel) outward.
4. Evidence: cortex voxels next to (unclosed) fluid, owned by the bank (T_X <= T_Y), >= 2 mm from
   the box faces, give their root a measured thickness (V or T at the top).
5. Thickness field on each wall graph: Gaussian, prior precision tau (k^2 I + L)^T (k^2 I + L)
   (Matern-like, second order, with a level term), prior mean = the bank's median evidence,
   prior SD SIG0, observations with SD SIG_OBS. Exact means; variances from solves.
6. Contact rules as fields: equal arrival (V_A = V_B), normalized arrival with one thickness
   per bank (V_A/t_A = V_B/t_B), posterior (P_A = P_B, P_X = Phi((mu_X(root) - V_X)/sd_X(root))).
   Offsets: along the line through each contact point in bank A's column direction.
7. Calibration: 5 spatial folds of evidence nodes per bank held out in turn.
Metric: equivolume (V) by default; --distance uses T instead.
"""

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import ndtr

ELL, SIG0, SIG_OBS, CLOSE_MM, WALL_MM, FACE_MM = 2.0, 0.6, 0.15, 1.5, 7.0, 2.0


@numba.njit
def upwind_fields(order, dom, fixed, rootid, T1, n1, div, vox):
    nx, ny, nz = dom.shape
    u = np.full(dom.shape, np.nan)
    V = np.full(dom.shape, np.nan)
    root = np.full(dom.shape, -1, np.int64)
    for t in range(order.shape[0]):
        i, j, k = order[t, 0], order[t, 1], order[t, 2]
        if fixed[i, j, k]:
            u[i, j, k] = 0.0
            V[i, j, k] = 0.0
            root[i, j, k] = rootid[i, j, k]
            continue
        sw = 0.0
        su = 0.0
        sv = 0.0
        bw = 0.0
        best = -1
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    a, b, c = i - di, j - dj, k - dk
                    if a < 0 or b < 0 or c < 0 or a >= nx or b >= ny or c >= nz:
                        continue
                    if not dom[a, b, c] or np.isnan(u[a, b, c]):
                        continue
                    ex, ey, ez = di * vox[0], dj * vox[1], dk * vox[2]
                    d = np.sqrt(ex * ex + ey * ey + ez * ez)
                    al = (n1[i, j, k, 0] * ex + n1[i, j, k, 1] * ey + n1[i, j, k, 2] * ez) / d
                    if al <= 0.3:
                        continue
                    same = (n1[i, j, k, 0] * n1[a, b, c, 0] + n1[i, j, k, 1] * n1[a, b, c, 1]
                            + n1[i, j, k, 2] * n1[a, b, c, 2])
                    if same < 0.5 or T1[a, b, c] >= T1[i, j, k]:
                        continue
                    w = al * al / d
                    ds = d * al
                    ux = u[a, b, c] + div[i, j, k] * ds
                    if ux > 1.2:
                        ux = 1.2
                    elif ux < -1.2:
                        ux = -1.2
                    su += w * ux
                    sv += w * (V[a, b, c] + np.exp(0.5 * (u[a, b, c] + ux)) * ds)
                    sw += w
                    if w > bw:
                        bw = w
                        best = root[a, b, c]
        if sw > 0:
            u[i, j, k] = su / sw
            V[i, j, k] = sv / sw
            root[i, j, k] = best
    return u, V, root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scratch")
    ap.add_argument("--distance", action="store_true")
    args = ap.parse_args()
    S = args.scratch
    sys.path.insert(0, S)
    import flatten_fused2 as F
    from baillarger.data import load_box
    from baillarger.frame import missing_tissue
    from fit_T import fill, solve_field
    metric = "distance" if args.distance else "equivolume"

    m = json.load(open(f"{S}/M1S1_test_points.json"))
    p = load_box(tuple(m["full_lo"]), tuple(m["full_hi"]))
    rim, vox = p.rim.copy(), p.voxel_mm
    vv = np.array(vox)
    miss = missing_tissue(rim, p.flash, vox)
    wm = rim == 2
    shell = wm & ~ndi.binary_erosion(wm, border_value=1)
    idx = np.indices(rim.shape)
    dface = np.min(np.stack([idx[a] for a in range(3)] + [rim.shape[a] - 1 - idx[a] for a in range(3)]), 0) * 0.16
    P = np.array(m["gap_all"], float)
    A_end = np.array(m["banks"]["A"], float)

    # ---- contact normals (PCA of nearby points), orientation propagated from bank A
    nrm = np.zeros_like(P)
    for n_, q in enumerate(P):
        d = np.linalg.norm((P - q) * vv, axis=1)
        nb = P[np.argsort(d)[:max(4, int((d < 2.5).sum()))]]
        w_, V_ = np.linalg.eigh(np.cov(((nb - nb.mean(0)) * vv).T))
        nrm[n_] = V_[:, 0]
    start = int(np.argmin(np.linalg.norm((P - A_end.mean(0)) * vv, axis=1)))
    if np.dot(nrm[start], (A_end.mean(0) - P[start]) * vv) < 0:
        nrm[start] *= -1
    done = {start}
    while len(done) < len(P):
        best = None
        for a in done:
            for b in range(len(P)):
                if b in done:
                    continue
                dd = np.linalg.norm((P[a] - P[b]) * vv)
                if best is None or dd < best[0]:
                    best = (dd, a, b)
        _, a, b = best
        if np.dot(nrm[a], nrm[b]) < 0:
            nrm[b] *= -1
        done.add(b)

    # ---- walls
    sh = np.argwhere(shell)
    dpt = np.linalg.norm((sh[:, None, :] - P[None]) * vv, axis=2)
    near = dpt.min(1)
    nn = dpt.argmin(1)
    side = ((sh - P[nn]) * vv * nrm[nn]).sum(1)
    d_out = ndi.distance_transform_edt(~wm, sampling=vox)
    gout = np.stack(np.gradient(ndi.gaussian_filter(d_out, 1.0), *vox), -1)
    gout /= np.linalg.norm(gout, axis=-1, keepdims=True) + 1e-12
    to_p = (P[nn] - sh) * vv
    to_p /= np.linalg.norm(to_p, axis=1, keepdims=True) + 1e-12
    facing = (gout[tuple(sh.T)] * to_p).sum(1) > 0.3
    keep = (near < WALL_MM) & facing & (np.abs(side) < 5.0)
    walls = {}
    for X, sgn in (("A", 1), ("B", -1)):
        sel = keep & (sgn * side > 0.5)
        wmask = np.zeros(rim.shape, bool)
        wmask[tuple(sh[sel].T)] = True
        walls[X] = wmask
    print(f"walls: A {walls['A'].sum()}, B {walls['B'].sum()} voxels")

    # ---- close the kissing gap
    fl = rim == 1
    dP = np.full(rim.shape, np.inf)
    fl_idx = np.argwhere(fl)
    if len(fl_idx):
        dfl = np.linalg.norm((fl_idx[:, None, :] - P[None]) * vv, axis=2).min(1)
        closed = fl_idx[dfl < CLOSE_MM]
        rim[tuple(closed.T)] = 3
        print(f"closed {len(closed)} fluid voxels within {CLOSE_MM} mm of the contact points")
    ctx0 = (p.rim == 3) & ~miss
    ctx = (rim == 3) & ~miss
    mb0 = ctx0.astype(float)

    e = {i: np.log(np.maximum(p.echoes[..., i].astype(float), 1.0)) for i in (1, 2)}
    det = {i: (L - F.mconv(L, mb0, F.SHP)) * mb0 for i, L in e.items()}
    Tc = F.tensor(det[1], mb0) + F.tensor(det[2], mb0)
    Mt = np.stack([np.stack([Tc[0], Tc[1], Tc[2]], -1), np.stack([Tc[1], Tc[3], Tc[4]], -1),
                   np.stack([Tc[2], Tc[4], Tc[5]], -1)], -2)
    w_, V_ = np.linalg.eigh(Mt)
    nh = V_[..., :, 2]
    coh = np.where(ctx0, np.clip((w_[..., 2] - w_[..., 1]) / np.maximum(w_[..., 2] + w_[..., 1], 1e-12), 0, 1), 0.0)

    # ---- per-bank fields
    B = {}
    for X in ("A", "B"):
        wall = walls[X]
        dom = ctx | wall
        dW = ndi.distance_transform_edt(~wall, sampling=vox)
        g = np.stack(np.gradient(ndi.gaussian_filter(dW, 1.0), *vox), -1)
        gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
        sg = np.sign((nh * gh).sum(-1)); sg[sg == 0] = 1
        dirv = coh[..., None] * nh * sg[..., None] + 0.1 * gh
        dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
        T, info = solve_field(dom, wall, [dirv[..., a] for a in range(3)], vox)
        Tf = fill(T)
        gT = np.stack(np.gradient(Tf, *vox), -1)
        n1 = gT / (np.linalg.norm(gT, axis=-1, keepdims=True) + 1e-12)
        n1s = np.stack([ndi.gaussian_filter(n1[..., a], 1.0) for a in range(3)], -1)
        agree = (n1 * n1s).sum(-1) / (np.linalg.norm(n1s, axis=-1) + 1e-12)
        n1n = n1s / (np.linalg.norm(n1s, axis=-1, keepdims=True) + 1e-12)
        div = sum(np.gradient(n1n[..., a], vox[a], axis=a) for a in range(3))
        div = np.clip(np.where(ndi.binary_erosion(dom) & (agree > 0.8) & (Tf > 0.3), div, 0.0), -2, 2)
        rootid = np.full(rim.shape, -1, np.int64)
        wnodes = np.argwhere(wall)
        rootid[tuple(wnodes.T)] = np.arange(len(wnodes))
        pts_ = np.argwhere(dom)
        order = pts_[np.argsort(Tf[dom])].astype(np.int64)
        u, Vf, root = upwind_fields(order, dom, wall, rootid, Tf, n1, div, vv)
        B[X] = dict(div=div, T=np.where(dom, T, np.nan), Tf=Tf, V=np.where(dom, Vf, np.nan), root=np.where(dom, root, -1),
                    nodes=wnodes, n1=n1, dom=dom)
        print(f"T_{X}: cg {info}; nodes {len(wnodes)}; V reached {np.isfinite(Vf[ctx]).mean():.0%} of cortex")

    # ---- evidence
    fluid_adj = ctx & ndi.binary_dilation(rim == 1) & (dface >= FACE_MM)
    for X, Y in (("A", "B"), ("B", "A")):
        own = fluid_adj & (B[X]["Tf"] <= B[Y]["Tf"]) & (B[X]["root"] >= 0)
        q = np.argwhere(own)
        r = B[X]["root"][tuple(q.T)]
        val = (B[X]["T"] if args.distance else B[X]["V"])[tuple(q.T)]
        ok = np.isfinite(val) & (val > 0.8) & (val < 7.0)  # physically plausible tops only
        print(f"   bank {X}: {np.isfinite(val).sum()} top voxels, {ok.sum()} within 0.8-7 mm; "
              f"all-top percentiles 5/50/95: {np.round(np.nanpercentile(val, [5, 50, 95]), 2)}")
        r, val = r[ok], val[ok]
        nN = len(B[X]["nodes"])
        obs = np.full(nN, np.nan)
        for rr in np.unique(r):
            obs[rr] = np.median(val[r == rr])
        B[X]["obs"] = obs
        print(f"bank {X}: {np.isfinite(obs).sum()} of {nN} wall nodes have a visible top; "
              f"thickness ({metric}) median {np.nanmedian(obs):.2f}, IQR {np.subtract(*np.nanpercentile(obs, [75, 25])):.2f}")

    # ---- Gaussian field on each wall graph
    def graph_L(nodes):
        n = len(nodes)
        lut = {tuple(x): i for i, x in enumerate(nodes)}
        rows, cols, ws = [], [], []
        for i, x in enumerate(nodes):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        j = lut.get((x[0] + dx, x[1] + dy, x[2] + dz))
                        if j is not None and j > i:
                            d2 = (dx * 0.16) ** 2 + (dy * 0.16) ** 2 + (dz * 0.16) ** 2
                            rows += [i, j]; cols += [j, i]; ws += [1 / d2, 1 / d2]
        Wg = sparse.csr_matrix((ws, (rows, cols)), shape=(n, n))
        deg = np.asarray(Wg.sum(1)).ravel()
        # normalise so L approximates the Laplace-Beltrami operator in mm^-2
        Lg = (sparse.diags(deg) - Wg) / max(np.median(deg) * 0.16 ** 2 / 4, 1e-9)
        return Lg.tocsc()

    def gmrf(X, mask_obs=None):
        nodes, obs = B[X]["nodes"], B[X]["obs"].copy()
        if mask_obs is not None:
            obs[mask_obs] = np.nan
        n = len(nodes)
        Lg = B[X].setdefault("L", graph_L(nodes))
        kap = 1.0 / ELL
        Aop = (kap ** 2 * sparse.identity(n) + Lg).tocsc()
        Q0 = (Aop.T @ Aop).tocsc()
        if "tau" not in B[X]:
            lu0 = splu(Q0)
            probe = np.random.default_rng(0).choice(n, size=min(40, n), replace=False)
            E = np.zeros((n, len(probe))); E[probe, np.arange(len(probe))] = 1
            var0 = (lu0.solve(E)[probe, np.arange(len(probe))]).mean()
            B[X]["tau"] = var0 / SIG0 ** 2
        Q0 = Q0 * B[X]["tau"]
        mu0 = np.nanmedian(B[X]["obs"])
        o = np.isfinite(obs)
        W = sparse.diags(np.where(o, 1 / SIG_OBS ** 2, 0.0))
        Q = (Q0 + W).tocsc()
        lu = splu(Q)
        rhs = W @ np.where(o, obs - mu0, 0.0)
        mu = mu0 + lu.solve(rhs)
        return mu, lu

    def variances(lu, idxs, n):
        out = np.empty(len(idxs))
        for s in range(0, len(idxs), 200):
            blk = idxs[s:s + 200]
            E = np.zeros((n, len(blk))); E[blk, np.arange(len(blk))] = 1
            out[s:s + 200] = lu.solve(E)[blk, np.arange(len(blk))]
        return out

    for X in ("A", "B"):
        mu, lu = gmrf(X)
        n = len(B[X]["nodes"])
        B[X]["mu"] = mu
        B[X]["sd"] = np.sqrt(variances(lu, np.arange(n), n))
        print(f"bank {X}: field mean {np.median(mu):.2f} (range {mu.min():.2f}-{mu.max():.2f}), "
              f"sd median {np.median(B[X]['sd']):.2f} (range {B[X]['sd'].min():.2f}-{B[X]['sd'].max():.2f})")

    # ---- calibration: 5 spatial folds of evidence nodes
    calib = {}
    for X in ("A", "B"):
        nodes, obs = B[X]["nodes"], B[X]["obs"]
        ev = np.flatnonzero(np.isfinite(obs))
        if len(ev) < 10:
            continue
        c = (nodes[ev] * vv)
        ax_ = np.linalg.svd(c - c.mean(0))[2][0]
        proj = (c - c.mean(0)) @ ax_
        folds = np.array_split(ev[np.argsort(proj)], 5)
        zs, dists = [], []
        n = len(nodes)
        for f in folds:
            mask = np.zeros(n, bool); mask[f] = True
            mu, lu = gmrf(X, mask_obs=mask)
            sd = np.sqrt(variances(lu, f, n))
            rest = np.setdiff1d(ev, f)
            dmin = np.linalg.norm((nodes[f][:, None] - nodes[rest][None]) * vv, axis=2).min(1)
            zs.append((obs[f] - mu[f]) / np.sqrt(sd ** 2 + SIG_OBS ** 2))
            dists.append(dmin)
        z, dd = np.concatenate(zs), np.concatenate(dists)
        calib[X] = (z, dd)
        print(f"calibration bank {X}: {len(z)} held-out nodes; z mean {z.mean():+.2f}, sd {z.std():.2f}; "
              f"|z|<1.96: {np.mean(np.abs(z) < 1.96):.0%}; distance to nearest kept evidence median {np.median(dd):.2f} mm")

    # ---- contact rules as fields
    key = "T" if args.distance else "V"
    VA, VB = fill(B["A"][key]), fill(B["B"][key])
    tA, tB = np.nanmedian(B["A"]["obs"]), np.nanmedian(B["B"]["obs"])

    def postP(X, Vx):
        r = B[X]["root"]
        mu = np.where(r >= 0, B[X]["mu"][np.maximum(r, 0)], np.nan)
        sd = np.where(r >= 0, B[X]["sd"][np.maximum(r, 0)], np.nan)
        return ndtr((fill(mu) - Vx) / np.maximum(fill(sd), 1e-3))

    PA, PB = postP("A", VA), postP("B", VB)
    rules = {"equal arrival": VA - VB, "normalized (one t per bank)": VA / tA - VB / tB,
             "posterior": PB - PA}
    gA = [ndi.gaussian_filter(np.gradient(B["A"]["Tf"], *vox)[a], 0.5) for a in range(3)]
    offs = {k: [] for k in rules}
    band = []
    for q in P:
        n_ = np.array([F.interp(c, q) for c in gA]); n_ /= np.linalg.norm(n_) + 1e-12
        ss = np.arange(-2.5, 2.5001, 0.04)
        line = q[None] + ss[:, None] * n_[None] / vv
        for k, fld in rules.items():
            f = ndi.map_coordinates(fld, line.T, order=1)
            cr = np.flatnonzero(np.diff(np.sign(f)) != 0)
            if len(cr):
                c0 = cr[np.argmin(np.abs(ss[cr]))]
                offs[k].append(ss[c0] - f[c0] * (ss[c0 + 1] - ss[c0]) / (f[c0 + 1] - f[c0] + 1e-12))
            else:
                offs[k].append(np.nan)
        pa = ndi.map_coordinates(PA, line.T, order=1); pb = ndi.map_coordinates(PB, line.T, order=1)
        both = (pa > 0.16) & (pb > 0.16)
        band.append(float(np.ptp(ss[both])) if both.any() else 0.0)
    tp = {X: np.array([F.interp(B[X]["Tf"], q) for q in P]) for X in ("A", "B")}
    rp = {X: np.array([B[X]["root"][tuple(np.round(q).astype(int))] for q in P]) for X in ("A", "B")}
    for X in ("A", "B"):
        mu_p = np.array([B[X]["mu"][r] if r >= 0 else np.nan for r in rp[X]])
        sd_p = np.array([B[X]["sd"][r] if r >= 0 else np.nan for r in rp[X]])
        print(f"   at your points, bank {X}: T_{X} (its thickness up to the contact) median {np.median(tp[X]):.2f} "
              f"[{np.min(tp[X]):.2f}-{np.max(tp[X]):.2f}]; field mean at the column foot {np.nanmedian(mu_p):.2f}, sd {np.nanmedian(sd_p):.2f}")
    print(f"contact rules against the 23 marked points ({metric}; offset > 0 = toward bank B):")
    for k, v in offs.items():
        v = np.array(v)
        print(f"   {k:28s} median {np.nanmedian(v):+.2f} mm, IQR {np.subtract(*np.nanpercentile(v, [75, 25])):.2f}, "
              f"|offset| median {np.nanmedian(np.abs(v)):.2f}, missing {np.isnan(v).sum()}")
    for n_, v in enumerate(offs["posterior"]):
        if np.isnan(v):
            q = P[n_].astype(int)
            print(f"   no posterior crossing at your point {n_ + 1}: test crop 0-based {tuple(q)}, ITK-SNAP {tuple(q + 1)}; "
                  f"equal arrival {offs['equal arrival'][n_]:+.2f}, normalized {offs['normalized (one t per bank)'][n_]:+.2f}; "
                  f"P_A {F.interp(PA, P[n_]):.2f}, P_B {F.interp(PB, P[n_]):.2f}, T_A {tp['A'][n_]:.2f}, T_B {tp['B'][n_]:.2f}")
    for n_ in range(len(P)):
        for X in ("A",):
            # mean divergence of bank A's columns along the straight path from its foot to the point
            q = P[n_]
            g_ = np.array([F.interp(c, q) for c in gA]); g_ /= np.linalg.norm(g_) + 1e-12
            ss = np.arange(0.2, tp[X][n_], 0.05)
            line = q[None] - ss[::-1, None] * g_[None] / vv
            dv = ndi.map_coordinates(B[X]["div"], line.T, order=1)
            if n_ in (3, 4) or n_ % 5 == 0:
                print(f"   point {n_ + 1}: bank A column divergence mean {dv.mean():+.2f} /mm "
                      f"({'diverging: crown-like' if dv.mean() > 0 else 'converging: fundus-like'}); "
                      f"posterior offset {offs['posterior'][n_]:+.2f}")
    print(f"   posterior uncertainty band (both banks > 16%): median width {np.median(band):.2f} mm")

    # ---------- figures
    tag = "dist" if args.distance else "ev"
    raw = e[1]
    rv = np.nanpercentile(raw[ctx0], [1, 99.5])
    # 1. slices: walls, points, contact rules
    planes = [(0, int(round(np.median(P[P[:, 0] > 40, 0])))), (0, 27), (0, 19), (2, 89)]
    fig, axs = plt.subplots(2, len(planes), figsize=(7 * len(planes), 13))
    for c_, (ax_i, pl) in enumerate(planes):
        sl = [slice(None)] * 3; sl[ax_i] = pl; sl = tuple(sl)
        oth = [a for a in range(3) if a != ax_i]
        selp = np.abs(P[:, ax_i] - pl) <= 1.5
        for r_ in range(2):
            a = axs[r_, c_]
            a.imshow(raw[sl].T, origin="lower", cmap="gray", vmin=rv[0], vmax=rv[1])
            a.contour(wm[sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.6)
            a.contour((rim == 1)[sl].T.astype(float), levels=[0.5], colors="c", linewidths=0.6)
            a.plot(P[selp, oth[0]], P[selp, oth[1]], "x", c="m", ms=9, mew=2)
            if r_ == 0:
                a.imshow(np.where(walls["A"][sl], 1.0, np.nan).T, origin="lower", cmap="Blues", vmin=0, vmax=1.1)
                a.imshow(np.where(walls["B"][sl], 1.0, np.nan).T, origin="lower", cmap="Oranges", vmin=0, vmax=1.1)
                a.set_title(f"{'ijk'[ax_i]} = {pl + 1} (ITK-SNAP): walls A blue, B orange; magenta = your points")
            else:
                band_m = np.where(ctx[sl] & (PA[sl] > 0.16) & (PB[sl] > 0.16), 1.0, np.nan)
                a.imshow(band_m.T, origin="lower", cmap="Greens", alpha=0.35, vmin=0, vmax=1.5)
                for (k, fld), col in zip(rules.items(), ("lime", "m", "c")):
                    a.contour(np.where(ctx, fld, np.nan)[sl].T, levels=[0], colors=col, linewidths=1.3)
                a.set_title("lime = equal arrival, magenta = normalized (one t per bank),\n"
                            "cyan = posterior; green band = both banks > 16%")
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"M1S1 test crop, gap closed, {metric}: walls and contact rules against your 23 points (all computed in 3D)", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{S}/prob_{tag}_slices.png", dpi=80); plt.close(fig)

    # 1b. the estimated face near points 4 and 5, masked to the space between the two walls
    dWA = ndi.distance_transform_edt(~walls["A"], sampling=vox)
    dWB = ndi.distance_transform_edt(~walls["B"], sampling=vox)
    between = ctx & (dWA + dWB <= 6.0)
    fig, axs = plt.subplots(2, 3, figsize=(21, 14))
    for r_, n_ in enumerate((3, 4)):
        q = P[n_].astype(int)
        for c_, ax_i in enumerate((0, 1, 2)):
            sl = [slice(None)] * 3; sl[ax_i] = q[ax_i]; sl = tuple(sl)
            oth = [a for a in range(3) if a != ax_i]
            a = axs[r_, c_]
            a.imshow(raw[sl].T, origin="lower", cmap="gray", vmin=rv[0], vmax=rv[1])
            a.contour(wm[sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.6)
            a.imshow(np.where(walls["A"][sl], 1.0, np.nan).T, origin="lower", cmap="Blues", vmin=0, vmax=1.1)
            a.imshow(np.where(walls["B"][sl], 1.0, np.nan).T, origin="lower", cmap="Oranges", vmin=0, vmax=1.1)
            a.imshow(np.where(~between[sl], 1.0, np.nan).T, origin="lower", cmap="Greys", alpha=0.35, vmin=0, vmax=1)
            for (k, fld), col in zip(rules.items(), ("lime", "m", "c")):
                a.contour(np.where(between, fld, np.nan)[sl].T, levels=[0], colors=col, linewidths=1.6)
            selp = np.abs(P[:, ax_i] - q[ax_i]) <= 1.5
            a.plot(P[selp, oth[0]], P[selp, oth[1]], "x", c="w", ms=9, mew=2)
            a.plot(q[oth[0]], q[oth[1]], "o", mfc="none", mec="r", ms=16, mew=2.5)
            a.set_xlim(q[oth[0]] - 30, q[oth[0]] + 30); a.set_ylim(q[oth[1]] - 30, q[oth[1]] + 30)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"point {n_ + 1} (red circle), slice {'ijk'[ax_i]} = {q[ax_i] + 1} (ITK-SNAP)\n"
                        "lime equal arrival, magenta normalized, cyan posterior; white x = your points")
    fig.suptitle(f"Estimated contact face near points 4 and 5 ({metric}), drawn only between the two walls (grey = outside); "
                 "walls A blue, B orange; all computed in 3D", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{S}/prob_{tag}_faces45.png", dpi=80); plt.close(fig)

    # 2. thickness field along each wall (unrolled by PCA of the wall)
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    for c_, X in enumerate(("A", "B")):
        nodes = B[X]["nodes"] * vv
        cen = nodes.mean(0)
        U = np.linalg.svd(nodes - cen)[2]
        xy = (nodes - cen) @ U[:2].T
        ev = np.isfinite(B[X]["obs"])
        pp = (P * vv - cen) @ U[:2].T
        for r_, (vals, lab, cm) in enumerate(((B[X]["mu"], f"mean thickness ({metric})", "viridis"),
                                              (B[X]["sd"], "uncertainty (SD)", "magma"))):
            a = axs[r_, c_]
            sc = a.scatter(xy[:, 0], xy[:, 1], c=vals, s=6, cmap=cm)
            fig.colorbar(sc, ax=a, shrink=0.8, label=lab)
            a.scatter(xy[ev, 0], xy[ev, 1], s=10, facecolors="none", edgecolors="w", linewidths=0.4)
            a.plot(pp[:, 0], pp[:, 1], "x", c="r", ms=7, mew=1.5)
            a.set_aspect("equal"); a.set_title(f"bank {X}: {lab}; white rings = visible-top evidence, red x = contact points (projected)")
            a.set_xlabel("mm along the wall (principal axis 1)"); a.set_ylabel("mm (axis 2)")
    fig.tight_layout(); fig.savefig(f"{S}/prob_{tag}_field.png", dpi=80); plt.close(fig)

    # 3. calibration + offsets
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    for X, col in (("A", "#4a90d9"), ("B", "#e8913a")):
        if X in calib:
            z, dd = calib[X]
            axs[0].hist(z, bins=np.linspace(-5, 5, 41), alpha=0.5, color=col, label=f"bank {X} (sd {z.std():.2f})")
            axs[1].scatter(dd, np.abs(z), s=8, color=col, label=f"bank {X}")
    xx = np.linspace(-5, 5, 200)
    axs[0].plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi) * 40 * 0.25, "k--", label="N(0,1), scaled")
    axs[0].set_title("held-out z-scores (5 spatial folds)"); axs[0].legend()
    axs[1].axhline(1.96, c="k", ls="--"); axs[1].set_xlabel("distance to nearest kept evidence (mm)")
    axs[1].set_ylabel("|z|"); axs[1].set_title("calibration against distance from evidence"); axs[1].legend()
    for n_, (k, v) in enumerate(offs.items()):
        axs[2].scatter(np.arange(len(v)) + 0.15 * n_, v, s=25, label=k)
    axs[2].axhline(0, c="k", lw=0.8)
    axs[2].set_xlabel("contact point (yours, in order)"); axs[2].set_ylabel("rule's contact - your point (mm, + toward B)")
    axs[2].set_title("contact rules at each of your points"); axs[2].legend()
    fig.tight_layout(); fig.savefig(f"{S}/prob_{tag}_calib.png", dpi=80); plt.close(fig)
    print("wrote", f"{S}/prob_{tag}_*.png")


if __name__ == "__main__":
    main()
