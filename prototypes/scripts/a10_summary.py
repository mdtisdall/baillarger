"""A10: does the gap closure built from the user's points bias the score? Compares runs of collide.py and
collide2.py under three closures ('points', 'none', 'thin'; one folder each), all with visible tops taken
next to fluid that stays fluid.

Per closure: the posterior and equal-arrival offsets at the 23 points; each front's mu - S at the points;
and front 1's visible-top evidence by graph distance from its contact root nodes (the edge rise of finding 15).

Usage: a10_summary.py <outdir> <folder_points> <folder_none> <folder_thin>
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from collide import GROW, shell_graph

OUT = sys.argv[1]
RUNS = dict(zip(("points", "none", "thin"), sys.argv[2:5]))
VOX = np.array([0.16028, 0.16, 0.16028])
BINS = ((0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4))


def main():
    summ = {}
    for name, D in RUNS.items():
        st = np.load(f"{D}/collide2_state.npz"); cs = np.load(f"{D}/collide_state.npz"); cz = np.load(f"{D}/curv_M1S1.npz")
        rows = np.load(f"{D}/collide2_rows.npy", allow_pickle=True)
        m = json.load(open(f"{D}/M1S1_test_points.json"))
        P = np.round(np.array(m["gap_all"], float)).astype(int) + GROW
        pts, nrm = cz["pts"], cz["n"]; node_of = cs["node_of"]; y = st["y"]
        nN = node_of.max() + 1; cnt = np.bincount(node_of, minlength=nN)
        npos = np.stack([np.bincount(node_of, pts[:, k] * VOX[k], nN) for k in range(3)], 1) / cnt[:, None]
        G = shell_graph(pts, nrm, VOX).tocoo()
        ea, eb = node_of[G.row], node_of[G.col]
        ed = np.unique(np.sort(np.c_[ea, eb][ea != eb], 1), axis=0)
        w = np.linalg.norm(npos[ed[:, 0]] - npos[ed[:, 1]], axis=1)
        NG = sparse.csr_matrix((np.r_[w, w], (np.r_[ed[:, 0], ed[:, 1]], np.r_[ed[:, 1], ed[:, 0]])), shape=(nN, nN))
        out = dict(post=np.array([r["post"] for r in rows]), eq=np.array([r["eq"] for r in rows]),
                   e1=np.array([r["mu1"] - r["T1"] for r in rows]), e2=np.array([r["mu2"] - r["T2"] for r in rows]),
                   S1=np.array([r["T1"] for r in rows]), S2=np.array([r["T2"] for r in rows]))
        ev = np.isfinite(y)
        rise = {}
        for X in ("1", "2"):
            rn = np.unique(node_of[np.maximum(st[f"root{X}"][tuple(P.T)], 0)])
            d = dijkstra(NG, indices=rn, limit=4.0, min_only=True)
            rise[X] = [(np.sum(ev & (d >= a) & (d < b)), np.exp(np.median(y[ev & (d >= a) & (d < b)])) if np.any(ev & (d >= a) & (d < b)) else np.nan)
                       for a, b in BINS]
        out["rise"] = rise
        summ[name] = out
        print(f"closure '{name}': posterior median {np.nanmedian(out['post']):+.2f}, IQR {np.subtract(*np.nanpercentile(out['post'], [75, 25])):.2f}, "
              f"|offset| median {np.nanmedian(np.abs(out['post'])):.2f}; equal arrival |offset| {np.nanmedian(np.abs(out['eq'])):.2f}; "
              f"S at the points {np.median(out['S1']):.2f} / {np.median(out['S2']):.2f}; mu - S median front 1 {np.median(out['e1']):+.2f}, "
              f"front 2 {np.median(out['e2']):+.2f}")
        for X in ("1", "2"):
            print(f"   front {X} evidence by graph distance from its contact roots: " +
                  ", ".join(f"{a}-{b} mm n {n_} {v:.2f}" for (a, b), (n_, v) in zip(BINS, rise[X])))
    fig, axs = plt.subplots(1, 3, figsize=(26, 7))
    ks = np.arange(1, 24)
    cols = {"points": "#7f7f7f", "none": "#2b7bd9", "thin": "#e8913a"}
    for name, o in summ.items():
        axs[0].plot(ks, o["post"], "o-", c=cols[name], label=f"posterior, closure '{name}'")
        axs[0].plot(ks, o["eq"], "_", ms=12, mew=2, c=cols[name], label=f"equal arrival, '{name}'")
        axs[1].plot(ks, o["e1"], "o-", c=cols[name], label=f"front 1, '{name}'")
        axs[1].plot(ks, o["e2"], "s:", c=cols[name], label=f"front 2, '{name}'")
        for X, mk in (("1", "o-"), ("2", "s:")):
            xs = [np.mean(b) for b in BINS]
            axs[2].plot(xs, [v for _, v in o["rise"][X]], mk, c=cols[name], label=f"front {X}, '{name}'")
    axs[0].axhline(0, c="k", lw=0.8); axs[0].set_xticks(ks); axs[0].set_xlabel("your contact point")
    axs[0].set_ylabel("offset (mm, + toward the side predicted thinner)"); axs[0].set_title("offsets at your points"); axs[0].legend(fontsize=7)
    axs[1].axhline(0, c="k", lw=0.8); axs[1].set_xticks(ks); axs[1].set_xlabel("your contact point")
    axs[1].set_ylabel("predicted thickness - arc length at the point (mm)"); axs[1].set_title("each front's error"); axs[1].legend(fontsize=7)
    axs[2].set_xlabel("graph distance from the front's contact root nodes (mm)"); axs[2].set_ylabel("visible-top evidence, median (mm)")
    axs[2].set_title("evidence approaching the fused zone (the edge rise)"); axs[2].legend(fontsize=7)
    fig.suptitle("A10: the gap closure. 'points': fluid within 1.5 mm of your points; 'none'; 'thin': slivers under about 3 voxels. "
                 "Visible tops next to fluid that stays fluid, in all three.", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/a10_closure.png", dpi=75); plt.close(fig)
    print("wrote a10_closure.png")


if __name__ == "__main__":
    main()
