"""Does the thickness field's shared level pull the thin front up at the contact? Variants of the
level term, scored at the 23 points with the pairwise posterior in arc length.

Uses collide2.py's saved evidence (arc length at visible tops), walls and roots; refits only the field.
Variants: the level's length (2, 4, 6 mm), with and without a level offset per front (one extra unknown
per wall, prior N(0, 0.3^2) in log thickness: the fronts as regions, taken from roots).
Score per point: x1 = mu1 + s1^2/(s1^2 + s2^2) (L - mu1 - mu2), L = S1 + S2 at the point; offset = x1 - S1
(+ = the contact lies further from front 1's WM than the point). Also each front's mu - S at the points.

Usage: level_variants.py <workdir> <workdir> (the shared scripts read the second). Needs collide2_state.npz, collide2_rows.npy, collide_state.npz, curv_M1S1.npz.
"""

import sys

sys.path.insert(0, sys.argv[1])
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

import collide as CO
from collide import Field, graph_L, lognormal, shell_graph

S = sys.argv[1]
VOX = np.array([0.16028, 0.16, 0.16028])
SIG_FRONT = 0.3


class FieldFront(Field):
    """Field plus one level offset per front: log t += b_X on wall X's nodes."""

    def __init__(self, L, H, groups):
        super().__init__(L, H)
        self.groups = groups  # list of node index arrays

    def fit(self, y, trend=True, nsamp=60, seed=1):
        n, H = self.n, self.H
        ng = len(self.groups)
        o = np.flatnonzero(np.isfinite(y))
        nu = 2 * n + 2 + ng
        im, ia = 2 * n, 2 * n + 1
        ib = 2 * n + 2 + np.arange(ng)
        gm = np.zeros((n, ng))
        for k, g in enumerate(self.groups):
            gm[g, k] = 1.0
        P = sparse.block_diag([self.pz["Q"], self.pw["Q"],
                               sparse.diags([1 / CO.M_SD ** 2, 1 / CO.A0_SD ** 2] + [1 / SIG_FRONT ** 2] * ng)]).tocsc()
        pm = np.zeros(nu); pm[im] = CO.M_MU; pm[ia] = CO.A0_MU
        rows = [np.arange(len(o))] * 4; cols = [o, np.full(len(o), im), n + o, np.full(len(o), ia)]
        vals = [np.ones(len(o)), np.ones(len(o)), H[o], H[o]]
        for k in range(ng):
            sel = gm[o, k] > 0
            rows.append(np.flatnonzero(sel)); cols.append(np.full(sel.sum(), ib[k])); vals.append(np.ones(sel.sum()))
        A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(o), nu))
        Q = (P + A.T @ A / CO.SIG_EPS ** 2).tocsc()
        lu = splu(Q)
        xhat = lu.solve(P @ pm + A.T @ y[o] / CO.SIG_EPS ** 2)
        pred = lambda x: x[:n] + x[im] + H * (x[n:2 * n] + x[ia]) + gm @ x[ib]
        rng = np.random.default_rng(seed)
        Sm = []
        for _ in range(nsamp):
            x0 = pm.copy()
            x0[:n] += self.pz["lu"].solve(rng.standard_normal(n)) / np.sqrt(self.pz["tau"])
            x0[n:2 * n] += self.pw["lu"].solve(rng.standard_normal(n)) / np.sqrt(self.pw["tau"])
            x0[im] += CO.M_SD * rng.standard_normal(); x0[ia] += CO.A0_SD * rng.standard_normal()
            x0[ib] += SIG_FRONT * rng.standard_normal(ng)
            yp = y[o] + CO.SIG_EPS * rng.standard_normal(len(o))
            Sm.append(pred(x0 + lu.solve(A.T @ (yp - A @ x0) / CO.SIG_EPS ** 2)))
        return pred(xhat), np.array(Sm).std(0), xhat[ib]


def main():
    st = np.load(f"{S}/collide2_state.npz"); cs = np.load(f"{S}/collide_state.npz"); cz = np.load(f"{S}/curv_M1S1.npz")
    rows = np.load(f"{S}/collide2_rows.npy", allow_pickle=True)
    pts, nrm = cz["pts"], cz["n"]; node_of = cs["node_of"]
    y, nH = st["y"], st["nH"]
    nN = node_of.max() + 1
    cnt = np.bincount(node_of, minlength=nN)
    npos = np.stack([np.bincount(node_of, pts[:, k] * VOX[k], nN) for k in range(3)], 1) / cnt[:, None]
    G = shell_graph(pts, nrm, VOX)
    Gc = G.tocoo()
    ea, eb = node_of[Gc.row], node_of[Gc.col]
    ed = np.unique(np.sort(np.c_[ea, eb][ea != eb], 1), axis=0)
    L = graph_L(npos, ed)
    groups = [np.unique(node_of[st["wall1"]]), np.unique(node_of[st["wall2"]])]
    # each point's root node in each front
    import json
    m = json.load(open(f"{S}/M1S1_test_points.json"))
    P = np.round(np.array(m["gap_all"], float)).astype(int) + CO.GROW
    rn = {X: node_of[np.maximum(st[f"root{X}"][tuple(P.T)], 0)] for X in ("1", "2")}
    S1 = np.array([r["T1"] for r in rows]); S2 = np.array([r["T2"] for r in rows])  # arc length (collide2.py)
    print("variant: offset median / IQR / median |offset| (mm, + = further from front 1's WM); front 1 mu - S median; "
          "front 2 mu - S median; per-front level offsets")
    for ell in (2.0, 4.0, 6.0):
        CO.ELL_Z = ell  # Field reads the level's length from collide when it is built
        for per_front in (False, True):
            CO_field = (FieldFront(L, nH, groups) if per_front else Field(L, nH))
            if per_front:
                mu, sd, b = CO_field.fit(y)
            else:
                mu, sd, _ = CO_field.fit(y, nsamp=60); b = None
            tm, ts = lognormal(mu, sd)
            m1, s1, m2, s2 = tm[rn["1"]], ts[rn["1"]], tm[rn["2"]], ts[rn["2"]]
            Lp = S1 + S2
            x1 = m1 + s1 ** 2 / (s1 ** 2 + s2 ** 2) * (Lp - m1 - m2)
            off = x1 - S1
            print(f"  level length {ell:.0f} mm, {'per-front level' if per_front else 'shared level   '}: "
                  f"{np.median(off):+.2f} / {np.subtract(*np.percentile(off, [75, 25])):.2f} / {np.median(np.abs(off)):.2f}; "
                  f"front 1 {np.median(m1 - S1):+.2f} (SD {np.median(s1):.2f}), front 2 {np.median(m2 - S2):+.2f} (SD {np.median(s2):.2f})"
                  + (f"; offsets b1 {b[0]:+.2f}, b2 {b[1]:+.2f}" if b is not None else ""))


if __name__ == "__main__":
    main()
