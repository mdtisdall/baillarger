"""A6, continued: the thickness field's actual prior correlation against distance on its white-surface
graph, for a nominal length (Matern-type, alpha = 2, kappa = 1/length). Covariance columns are solved
exactly at random nodes of the main piece and binned by straight distance between node centres. The
continuum reference for alpha = 2 on a surface is rho(d) = kappa d K1(kappa d).

Usage: prior_length.py <workdir> <workdir>. Needs collide_state.npz, curv_M1S1.npz.
"""

import sys

sys.path.insert(0, sys.argv[2])
import numpy as np
from scipy.special import k1

from collide import graph_L, prior_op, shell_graph

S = sys.argv[1]
VOX = np.array([0.16028, 0.16, 0.16028])
cs = np.load(f"{S}/collide_state.npz")
cz = np.load(f"{S}/curv_M1S1.npz")
pts, nrm = cz["pts"], cz["n"]
node_of = cs["node_of"]
nN = node_of.max() + 1
cnt = np.bincount(node_of, minlength=nN)
npos = (
    np.stack([np.bincount(node_of, pts[:, k] * VOX[k], nN) for k in range(3)], 1)
    / cnt[:, None]
)
G = shell_graph(pts, nrm, VOX).tocoo()
ea, eb = node_of[G.row], node_of[G.col]
ed = np.unique(np.sort(np.c_[ea, eb][ea != eb], 1), axis=0)
L = graph_L(npos, ed)
nn = np.linalg.norm(npos[ed[:, 0]] - npos[ed[:, 1]], axis=1)
print(
    f"node spacing along edges: median {np.median(nn):.2f} mm (p10 {np.percentile(nn, 10):.2f}, p90 {np.percentile(nn, 90):.2f}); "
    f"L diagonal median {np.median(L.diagonal()):.0f} per mm^2 (a square grid of the median spacing would give {4 / np.median(nn) ** 2:.0f})"
)
rng = np.random.default_rng(0)
probe = rng.choice(nN, 30, replace=False)
bins = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])
for ell in (2.0, 6.0):
    po = prior_op(L, ell, 0.2)
    rows = []
    for i in probe:
        e = np.zeros(nN)
        e[i] = 1.0
        c = po["lu"].solve(po["lu"].solve(e))  # (A^2)^-1 e, up to tau
        d = np.linalg.norm(npos - npos[i], axis=1)
        rows.append((d, c / c[i]))
    d = np.concatenate([r[0] for r in rows])
    rho = np.concatenate([r[1] for r in rows])
    kap = 1 / ell
    print(f"nominal length {ell} mm:")
    for a, b in zip(bins[:-1], bins[1:]):
        m = (d >= a) & (d < b)
        x = (a + b) / 2
        print(
            f"   {a}-{b} mm: correlation median {np.median(rho[m]):.2f} (continuum {kap * x * k1(kap * x):.2f})"
        )
