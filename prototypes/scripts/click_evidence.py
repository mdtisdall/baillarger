"""Contact clicks as thickness evidence for the one-pass competition, tested on the 23 marked
M1S1 contact points (M1S1_test_points.json, "gap_all"). All 3D.

The idea (from the session "Contour mesh to laminar analysis workflow"): a click on a contact
gives, for the root on each bank, how far that bank's front travels to the contact. Added to the
thickness field's evidence (the visible column tops), the field's belief at those roots moves,
and the weighted pass moves the whole contact, not only a 2 mm neighbourhood as an editor edit.

Setup as onepass_face.py for M1S1 (finding 25): a box of 141 voxels re-centred on the points
(every point >= 4.8 mm from a face), the data's layer normal with cost ratio 3 across it, root
patches of 6 voxels, the thickness field of collide.py from the column tops of an unweighted
pass (roots >= 2 mm from a face), the no-belief guard of finding 26.

A click's evidence: the two fronts at the point are the labels at the two ends of the column
line through it (score_points_sides' sides); each side's arrival at the point, S, is read from
the equal-arrival pass (the crossing's arrival on that side, moved along the line to the point).
log S goes to the field nodes of that root patch's seeds (replacing any top there). Arrival, not
arc length: the competition compares arrival with the belief (finding 27: arc length over
arrival is about 0.92).

Scored with score_points_sides (finding 25's score: the crossing of the two sides' labels near
the line through the point; offset + toward the side the model predicts thinner) only at points
the model did not see:
- tops only: no clicks (finding 25's field weights);
- interleaved: clicks at the even-numbered points score the odd ones, and the reverse;
- blocks: the points in three spatial groups (k-means); each group scored by the other two;
- in sample: every click, every point scored (how closely the model can follow its evidence).
Equal arrival is scored beside them.

Writes <work>/click_evidence_M1S1.json and <figures>/click_evidence_M1S1.png.

Usage: click_evidence.py <figures dir> <work dir> <data dir>
"""

import json
import sys
import time

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2
from scipy.sparse.csgraph import connected_components, dijkstra

import curv_lib as C
from collide import H_CLIP, Field, graph_L, lognormal, shell_graph
from onepass_lib import contact_points, contacts, open_thickness, run, score_points_sides, setup

FIG, WORK, DATA = sys.argv[1:4]
FULL = np.array([700, 384, 1148])
HALF, RATIO, PATCH, FACE_MM, REACH_MM = 70, 3.0, 6, 2.0, 2.5
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def orientation(B):
    """The metric direction of onepass_face.py: the coherence-weighted data normal, signed by the WM
    distance gradient, plus 0.1 x that gradient."""
    vox = B["vox"]
    nh, coh = C.tensor_normal(B)
    d_wm = ndi.distance_transform_edt(~B["wm"], sampling=vox)
    g = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0), *vox), -1)
    gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    sgn = np.sign((nh * gh).sum(-1)); sgn[sgn == 0] = 1
    dirv = coh[..., None] * nh * sgn[..., None] + 0.1 * gh
    return dirv / (np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12)


class Thickness:
    """The thickness field of onepass_face.py: nodes on the WM boundary, evidence from column tops,
    weights per root patch with finding 26's no-belief guard. y_tops: log mm per node (nan: none)."""

    def __init__(self, B, s, h):
        from scipy.spatial import cKDTree

        wm = B["wm"]
        self.s, self.h = s, h
        K = C.curvature(B, 0.8)
        pts, nrm = K["pts"], K["n"]
        Hroot = np.clip(0.5 * (K["k1"] + K["k2"]), -H_CLIP, H_CLIP)
        ptsid = -np.ones(wm.shape, np.int64); ptsid[tuple(pts.T)] = np.arange(len(pts))
        side = np.argmax(np.abs(nrm), 1) * 2 + (nrm[np.arange(len(nrm)), np.argmax(np.abs(nrm), 1)] > 0)
        _, node_of, counts = np.unique(np.c_[pts // 2, side], axis=0, return_inverse=True, return_counts=True)
        node_of = node_of.ravel(); nN = node_of.max() + 1
        npos = np.stack([np.bincount(node_of, (pts * B["vox"])[:, a], nN) for a in range(3)], 1) / counts[:, None]
        nH = np.bincount(node_of, Hroot, nN) / counts
        Gc = shell_graph(pts, nrm, B["vox"]).tocoo()
        ea, eb = node_of[Gc.row], node_of[Gc.col]
        ed = np.unique(np.sort(np.c_[ea, eb][ea != eb], 1), axis=0)
        L = graph_L(npos, ed)
        self.fld = Field(L, nH)
        self.L, self.nH, self.npos = L, nH, npos
        _, self.piece = connected_components(L != 0, directed=False)
        seed_pts = ptsid.ravel()[s.sflat]
        miss = seed_pts < 0
        if miss.any():
            seed_pts[miss] = cKDTree(pts).query(np.stack(np.unravel_index(s.sflat[miss], wm.shape), -1))[1]
        self.seed_node = node_of[seed_pts]
        self.seed_patch = s.patch.ravel()[s.sflat]
        self.cnode = node_of[seed_pts[s.rep]]
        fluid_f = (B["rim"] == 1) & (B["dface"] >= FACE_MM)
        mu_tops, has = open_thickness(s, fluid_f, RATIO, along=True)
        root_ok = B["dface"].ravel()[s.sflat] >= FACE_MM
        ok = has[self.seed_patch] & root_ok & (mu_tops[self.seed_patch] * h > 0.8) & (mu_tops[self.seed_patch] * h < 7.0)
        self.y_tops = self._per_node(self.seed_node[ok], np.log(mu_tops[self.seed_patch[ok]] * h), nN)
        self.nN = nN

    @staticmethod
    def _per_node(nd, val, nN):
        y = np.full(nN, np.nan)
        o = np.argsort(nd); nd, val = nd[o], val[o]
        st = np.flatnonzero(np.r_[True, np.diff(nd) != 0])
        for i, a in enumerate(st):
            y[nd[a]] = np.median(val[a: st[i + 1] if i + 1 < st.size else nd.size])
        return y

    def with_clicks(self, clicks):
        """y_tops with each click's evidence, [(patch, S mm), ...], on the nodes of that patch's
        seeds (replacing tops there; several clicks on one node: their median)."""
        y = self.y_tops.copy()
        nd, val = [], []
        for patch, S in clicks:
            n_ = np.unique(self.seed_node[self.seed_patch == patch])
            nd.append(n_); val.append(np.full(n_.size, np.log(S)))
        if nd:
            yc = self._per_node(np.concatenate(nd), np.concatenate(val), self.nN)
            y[np.isfinite(yc)] = yc[np.isfinite(yc)]
        return y

    def weights(self, y):
        """Per root patch: mu, sigma (voxels), and how many patches got no belief."""
        mu_n, sd_n, _ = self.fld.fit(y, trend=True)
        tmean, tsd = lognormal(mu_n, sd_n)
        mu, sd = tmean[self.cnode] / self.h, tsd[self.cnode] / self.h
        ev_piece = np.bincount(self.piece[np.isfinite(y)], minlength=self.piece.max() + 1) > 0
        nobel = ~ev_piece[self.piece[self.cnode]]
        mu[nobel] = np.median(mu[~nobel]); sd[nobel] = np.median(sd[~nobel])
        return mu, sd, int(nobel.sum())


def end_labels(s, f, P, u_pts, h):
    """The two fronts at each point, as score_points_sides finds them: the labels of the deepest
    cortex on each side along the column line (REACH_MM either way); (-1, -1) where one is missing."""
    steps = np.arange(-REACH_MM, REACH_MM + 1e-9, 0.05) / h
    shape = np.array(s.cortex.shape)
    out = []
    for k, Pk in enumerate(P):
        u = u_pts[k] / (np.linalg.norm(u_pts[k]) + 1e-12)
        line = np.round(Pk + steps[:, None] * u).astype(int)
        inside = np.all((line >= 0) & (line < shape), 1)
        lab = np.full(steps.size, -1); lab[inside] = f.label[tuple(line[inside].T)]
        cx = np.zeros(steps.size, bool); cx[inside] = s.cortex[tuple(line[inside].T)]
        neg = np.flatnonzero(cx & (lab >= 0) & (steps < 0)); pos = np.flatnonzero(cx & (lab >= 0) & (steps > 0))
        out.append((lab[neg[0]], lab[pos[-1]]) if neg.size and pos.size else (-1, -1))
    return out


def score(s, f, mu, sd, P, u_pts, h, idx):
    c_all = contacts(s, f, mu, sd, sep_min=0.0)
    rows = score_points_sides(s, f, P[idx], u_pts[idx], c_all, contact_points(c_all, s.cortex.shape), mu, sd, h)
    for r, k in zip(rows, idx):
        r["k"] = int(k) + 1
    return rows


def main():
    m = json.load(open(f"{DATA}/M1S1_test_points.json"))
    full = np.array(m["gap_all"]) + np.array(m["full_lo"])
    c = np.round(full.mean(0)).astype(int)
    lo, hi = c - HALF, c + HALF + 1
    assert np.all(lo >= 0) and np.all(hi <= FULL)
    P = (full - lo).astype(float)
    n = len(P)
    B = C.load(lo, hi, "click_evidence_M1S1")
    h = float(np.mean(B["vox"]))
    say(f"box {tuple(hi - lo)} from {tuple(lo)}; {n} points, nearest "
        f"{np.min(np.minimum(P, (hi - lo) - 1 - P)) * h:.1f} mm from a face")
    dirv = orientation(B)
    s = setup(B["wm"], B["ctx"], patch=PATCH, direction=dirv)
    u_pts = dirv[tuple(np.round(P).astype(int).T)]
    del dirv
    say(f"seeds {s.seeds.sum():,}, patches {s.npatch:,}")
    T = Thickness(B, s, h)
    say(f"thickness field: {T.nN:,} nodes, {np.isfinite(T.y_tops).sum():,} with tops")

    # ---- the clicks' evidence, from the equal-arrival pass
    mu0, sd0, _ = T.weights(T.y_tops)
    mu_eq = np.full(s.npatch, float(np.median(mu0))); sd_eq = 0.1 * mu_eq
    f_eq = run(s, mu_eq, sd_eq, ratio=RATIO)
    rows_eq = score(s, f_eq, mu_eq, sd_eq, P, u_pts, h, np.arange(n))
    ends = end_labels(s, f_eq, P, u_pts, h)
    clicks = []  # per point: [(patch, S mm) for each side] or []
    for (lm, lp), r in zip(ends, rows_eq):
        # equal mu: score_points_sides takes lp as the "thin" end, so S_thin is lp's, S_thick lm's
        ok = lm >= 0 and np.isfinite(r["S_thin"]) and r["S_thin"] > 0.3 and r["S_thick"] > 0.3
        clicks.append([(int(lm), float(r["S_thick"])), (int(lp), float(r["S_thin"]))] if ok else [])
    say(f"clicks with evidence on both sides: {sum(bool(cl) for cl in clicks)} of {n}; "
        f"S median {np.median([S for cl in clicks for _, S in cl]):.2f} mm")

    # ---- the settings: which clicks are evidence, which points are scored
    folds = {"tops only": [([], np.arange(n))],
             "interleaved": [(np.arange(0, n, 2), np.arange(1, n, 2)), (np.arange(1, n, 2), np.arange(0, n, 2))],
             "in sample": [(np.arange(n), np.arange(n))]}
    _, grp = kmeans2(P * h, 3, seed=1, minit="++")
    folds["blocks"] = [(np.flatnonzero(grp != g), np.flatnonzero(grp == g)) for g in range(3)]
    results = {"equal arrival": dict(rows=rows_eq)}
    for name, fl in folds.items():
        rows, info = [], []
        for ev, held in fl:
            y = T.with_clicks([e for k in ev for e in clicks[k]])
            mu, sd, nobel = T.weights(y)
            f = run(s, mu, sd, ratio=RATIO)
            rows += score(s, f, mu, sd, P, u_pts, h, np.asarray(held))
            # how far the belief moved at the scored points' roots (mm), and the evidence there
            info.append(dict(clicks=int(len(ev)), scored=int(len(held)), no_belief=nobel,
                             mu_mm_at_scored_roots=[round(float(mu[p] * h), 2) for k in held for p, _ in clicks[k]],
                             mu0_mm_at_scored_roots=[round(float(mu0[p] * h), 2) for k in held for p, _ in clicks[k]],
                             S_mm_at_scored_points=[round(S, 2) for k in held for _, S in clicks[k]]))
            say(f"{name}: {len(ev)} clicks, {len(held)} scored")
        results[name] = dict(rows=sorted(rows, key=lambda r: r["k"]), folds=info)
    summary = {}
    for name, r in results.items():
        off = np.array([x["offset"] for x in r["rows"]]); near = np.array([x.get("nearest_mm", np.nan) for x in r["rows"]])
        summary[name] = dict(median=round(float(np.nanmedian(off)), 3),
                             iqr=round(float(np.subtract(*np.nanpercentile(off, [75, 25]))), 3),
                             median_abs=round(float(np.nanmedian(np.abs(off))), 3),
                             worst=round(float(np.nanmax(np.abs(off))), 3), missing=int(np.isnan(off).sum()),
                             nearest_median=round(float(np.nanmedian(near)), 3),
                             nearest_p90=round(float(np.nanpercentile(near, 90)), 3))
        print(f"   {name:14s} |offset| median {summary[name]['median_abs']:.2f} mm (median {summary[name]['median']:+.2f}, "
              f"IQR {summary[name]['iqr']:.2f}, worst {summary[name]['worst']:.2f}, missing {summary[name]['missing']}); "
              f"nearest crossing median {summary[name]['nearest_median']:.2f}, p90 {summary[name]['nearest_p90']:.2f}",
              flush=True)
    json.dump(dict(box_lo=lo.tolist(), groups=grp.tolist(), clicks=clicks, summary=summary, results=results),
              open(f"{WORK}/click_evidence_M1S1.json", "w"), indent=1, default=float)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    k = np.arange(1, n + 1)
    for j, (name, col) in enumerate(zip(results, ("#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"))):
        off = np.array([x["offset"] for x in results[name]["rows"]])
        ax.plot(k + (j - 2) * 0.15, np.abs(off), "o", color=col, ms=5,
                label=f"{name}: median {summary[name]['median_abs']:.2f} mm, missing {summary[name]['missing']}")
    ax.set_xticks(k); ax.set_xlabel("point"); ax.set_ylabel("|offset| of the contact (mm)")
    ax.set_title("M1S1, 23 marked contact points: clicks as thickness evidence (each point scored by a model "
                 "that did not see it, except 'in sample')", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/click_evidence_M1S1.png", dpi=110)
    say("done")


if __name__ == "__main__":
    main()
