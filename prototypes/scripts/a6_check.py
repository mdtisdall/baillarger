"""A6 diagnostic: is the thickness field's graph connected where the contact needs it, and does the field
honour the evidence near the contact? No model change: the field is refitted exactly as in collide2.py
(same evidence, same graph, same prior) and inspected.

1. The node graph (WM-boundary voxels pooled in 2x2x2 blocks, split by side; edges between same-side
   neighbours): its connected pieces; the piece of each contact root node (front 1 and 2 at the user's
   23 points) and of the evidence nodes within 3 mm of it; graph distance (shortest path through node
   centres) against straight distance from each contact root node to the nearest evidence node.
2. The fit: residual y - mu (log thickness) at evidence nodes, overall and near the contact; the parts
   of mu at the contact root nodes (level z, shared level m, curvature terms); the graph neighbourhood
   size and degree there.

Usage: a6_check.py <outdir> <workdir>. Needs collide2_state.npz, collide_state.npz, curv_M1S1.npz.
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from collide import GROW, Field, graph_L, shell_graph

OUT, S = sys.argv[1], sys.argv[2]
VOX = np.array([0.16028, 0.16, 0.16028])


def main():
    m = json.load(open(f"{S}/M1S1_test_points.json"))
    st = np.load(f"{S}/collide2_state.npz")
    cs = np.load(f"{S}/collide_state.npz")
    cz = np.load(f"{S}/curv_M1S1.npz")
    pts, nrm = cz["pts"], cz["n"]
    node_of = cs["node_of"]
    y, nH = st["y"], st["nH"]
    nN = node_of.max() + 1
    cnt = np.bincount(node_of, minlength=nN)
    npos = (
        np.stack([np.bincount(node_of, pts[:, k] * VOX[k], nN) for k in range(3)], 1)
        / cnt[:, None]
    )
    G = shell_graph(pts, nrm, VOX)
    Gc = G.tocoo()
    ea, eb = node_of[Gc.row], node_of[Gc.col]
    ed = np.unique(np.sort(np.c_[ea, eb][ea != eb], 1), axis=0)
    L = graph_L(npos, ed)
    w = np.linalg.norm(npos[ed[:, 0]] - npos[ed[:, 1]], axis=1)
    NG = sparse.csr_matrix(
        (np.r_[w, w], (np.r_[ed[:, 0], ed[:, 1]], np.r_[ed[:, 1], ed[:, 0]])),
        shape=(nN, nN),
    )
    deg = np.asarray((NG > 0).sum(1)).ravel()
    ncomp, comp = connected_components(NG, directed=False)
    csize = np.bincount(comp)
    print(
        f"node graph: {nN:,} nodes, {len(ed):,} edges, {ncomp} pieces (sizes {sorted(csize)[::-1][:6]}), "
        f"degree median {np.median(deg):.0f} (p5 {np.percentile(deg, 5):.0f}, p95 {np.percentile(deg, 95):.0f})"
    )
    ev = np.flatnonzero(np.isfinite(y))
    # the field, refitted as in collide2.py, then taken apart
    FLD = Field(L, nH)
    mu, sd, coef = FLD.fit(y, trend=True, nsamp=40)
    x = FLD.last["xhat"]
    n = FLD.last["n"]
    z, wv, mlev, a0 = x[:n], x[n : 2 * n], x[FLD.last["im"]], x[FLD.last["ia"]]
    res = y[ev] - mu[ev]
    print(
        f"fit at all {len(ev):,} evidence nodes: residual (log) median {np.median(res):+.3f}, IQR "
        f"{np.subtract(*np.percentile(res, [75, 25])):.3f}, |res| p95 {np.percentile(np.abs(res), 95):.3f}; shared level "
        f"m = {np.exp(mlev):.2f} mm, a0 = {a0:+.2f}"
    )
    P = np.round(np.array(m["gap_all"], float)).astype(int) + GROW
    out = []
    for X in ("1", "2"):
        rn = node_of[np.maximum(st[f"root{X}"][tuple(P.T)], 0)]
        Dg = dijkstra(NG, indices=np.unique(rn), limit=20.0)
        rowof = {u: i for i, u in enumerate(np.unique(rn))}
        print(
            f"front {X}: point | root node piece (size) | evidence nodes within 3 mm: n, same piece | nearest evidence: "
            f"straight / graph distance (mm) | mu at the root node, its parts (z, m, curvature) | nearby evidence median, "
            f"mu there"
        )
        for k, n0 in enumerate(rn):
            de = np.linalg.norm(npos[ev] - npos[n0], axis=1)
            near = ev[de < 3.0]
            same = np.sum(comp[near] == comp[n0])
            dg = Dg[rowof[n0], ev]
            i_s, i_g = np.argmin(de), np.argmin(dg)
            curv = nH[n0] * (wv[n0] + a0)
            o = dict(
                X=X,
                k=k,
                piece=comp[n0],
                psize=csize[comp[n0]],
                nnear=len(near),
                same=same,
                dstr=de[i_s],
                dgr=dg[i_g],
                dgr_of_nearest=dg[i_s],
                mu=np.exp(mu[n0]),
                z=z[n0],
                m=mlev,
                curv=curv,
                ev_near=np.exp(np.median(y[near])) if len(near) else np.nan,
                mu_near=np.exp(np.median(mu[near])) if len(near) else np.nan,
                deg=deg[n0],
            )
            out.append(o)
            print(
                f"   {k + 1:2d} | {o['piece']} ({o['psize']:,}) | {o['nnear']:3d}, {o['same']:3d} | {o['dstr']:.2f} / "
                f"{o['dgr']:.2f} (graph distance to the straight-nearest one {o['dgr_of_nearest']:.2f}) | {o['mu']:.2f}: "
                f"z {o['z']:+.2f}, m {np.exp(o['m']):.2f}, curv {o['curv']:+.3f} | {o['ev_near']:.2f}, {o['mu_near']:.2f}; degree {o['deg']}"
            )
    np.save(f"{S}/a6_rows.npy", out, allow_pickle=True)

    # ---------- view
    fig, axs = plt.subplots(1, 3, figsize=(27, 8))
    a = axs[0]
    for X, col in (("1", "#2b7bd9"), ("2", "#e8913a")):
        rr = [o for o in out if o["X"] == X]
        a.scatter(
            [o["dstr"] for o in rr],
            [o["dgr"] for o in rr],
            c=col,
            s=40,
            label=f"front {X}: nearest evidence node",
        )
        a.scatter(
            [o["dstr"] for o in rr],
            [o["dgr_of_nearest"] for o in rr],
            c=col,
            s=40,
            marker="x",
            label=f"front {X}: graph distance to the straight-nearest evidence node",
        )
    lim = max(
        4,
        max(o["dgr_of_nearest"] for o in out if np.isfinite(o["dgr_of_nearest"]))
        * 1.05,
    )
    a.plot([0, lim], [0, lim], "k--", lw=0.8)
    a.set_xlabel("straight distance from the contact root node to evidence (mm)")
    a.set_ylabel("graph distance (mm)")
    a.set_title("connectivity: graph against straight distance, per contact point")
    a.legend(fontsize=8)
    a = axs[1]
    de_all = np.min(
        np.stack(
            [
                np.linalg.norm(
                    npos[ev]
                    - npos[node_of[np.maximum(st[f"root{X}"][tuple(P.T)], 0)]][
                        :, None, :
                    ],
                    axis=2,
                ).min(0)
                for X in ("1", "2")
            ]
        ),
        0,
    )
    sc = a.scatter(
        np.exp(y[ev]),
        np.exp(mu[ev]),
        c=np.minimum(de_all, 10),
        s=4,
        cmap="viridis_r",
        alpha=0.6,
    )
    a.plot([1, 5], [1, 5], "k--", lw=0.8)
    fig.colorbar(
        sc, ax=a, label="distance to the nearest contact root node (mm, capped at 10)"
    )
    a.set_xlim(1, 5)
    a.set_ylim(1, 5)
    a.set_xlabel("evidence at the node (mm)")
    a.set_ylabel("field mean at the node (mm)")
    a.set_title("fit at evidence nodes: does the field honour its evidence?")
    a = axs[2]
    ks = np.arange(1, 24)
    for X, col in (("1", "#2b7bd9"), ("2", "#e8913a")):
        rr = [o for o in out if o["X"] == X]
        a.plot(
            ks,
            [o["mu"] for o in rr],
            "o-",
            c=col,
            label=f"front {X}: field mean at the contact root node",
        )
        a.plot(
            ks,
            [o["ev_near"] for o in rr],
            "s:",
            c=col,
            label=f"front {X}: evidence within 3 mm (median)",
        )
        a.plot(
            ks,
            [o["mu_near"] for o in rr],
            "^--",
            c=col,
            alpha=0.6,
            label=f"front {X}: field mean at those evidence nodes",
        )
    a.set_xticks(ks)
    a.set_xlabel("your contact point")
    a.set_ylabel("thickness (mm)")
    a.set_title("at the contact: field mean against the nearby evidence")
    a.legend(fontsize=8)
    fig.suptitle(
        "A6: the thickness field's graph and fit at the kissing stretch (no model change)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/a6_check.png", dpi=75)
    plt.close(fig)
    print("wrote a6_check.png")


if __name__ == "__main__":
    main()
