"""Why click evidence reaches only about a millimetre (click_evidence.py), on the same box, field
and clicks. All 3D.

1. Distances in the field's own terms: for each scored point's two roots, the distance along the
   white surface (the seed graph) to the nearest root that carries click evidence, against the
   distance between the marked points; and the z field's prior correlation between the scored
   root's node and the most correlated click node.
2. What holds the belief back: at the scored roots of the block folds, |mu - S| (mm) with the
   field fitted to tops only, tops + clicks, and clicks only (no tops). If clicks alone predict
   the scored roots but tops + clicks do not, the tops pull the belief back; if clicks alone do
   not either, the prior's reach (or an error that is not smooth) limits it. Plus mu0 - S per
   point along the stretch.
3. The prior's scale chosen on the visible tops, not on the points: 5-fold cross-validation over
   spatial blocks of 4 mm (the held-out tops' nodes lie up to a few mm from training evidence,
   as a fused stretch does), for ELL_Z in {1, 2, 4, 8} mm and SIG_Z in {0.1, 0.2, 0.4}; scored by
   the mean log predictive density of the held-out log thickness (predictive variance sd^2 +
   SIG_EPS^2), with RMSE and the z-score SD beside it. The rule fixed before the run: the
   setting with the highest mean log predictive density.
4. The click test (tops only, interleaved, blocks) rerun with that setting, if it is not the
   current one (2 mm, 0.2).

Writes <work>/click_reach_M1S1.json.

Usage: click_reach.py <figures dir> <work dir> <data dir>
"""

import json
import sys
import time

sys.path.insert(0, sys.argv[2])
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.sparse.csgraph import dijkstra

import curv_lib as C
from click_evidence import (
    FULL,
    HALF,
    PATCH,
    RATIO,
    Thickness,
    end_labels,
    orientation,
    score,
)
from collide import SIG_EPS, lognormal, prior_op
from onepass_lib import run, setup

FIG, WORK, DATA = sys.argv[1:4]
T0 = time.time()
GRID = [(e, s_) for e in (1.0, 2.0, 4.0, 8.0) for s_ in (0.1, 0.2, 0.4)]
CURRENT = (2.0, 0.2)


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def set_prior(T, ell, sig):
    T.fld.pz = prior_op(T.L, ell, sig)


def means(T, y):
    """Predictive mean of thickness (mm) per root patch (no samples: the mean only)."""
    mu_n, sd_n, _ = T.fld.fit(y, trend=True, nsamp=2)
    return np.exp(mu_n)[T.cnode]  # the median; the lognormal mean needs sd


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
    dirv = orientation(B)
    s = setup(B["wm"], B["ctx"], patch=PATCH, direction=dirv)
    u_pts = dirv[tuple(np.round(P).astype(int).T)]
    del dirv
    T = Thickness(B, s, h)
    mu0, sd0, _ = T.weights(T.y_tops)
    mu_eq = np.full(s.npatch, float(np.median(mu0)))
    sd_eq = 0.1 * mu_eq
    f_eq = run(s, mu_eq, sd_eq, ratio=RATIO)
    rows_eq = score(s, f_eq, mu_eq, sd_eq, P, u_pts, h, np.arange(n))
    clicks = []
    for (lm, lp), r in zip(end_labels(s, f_eq, P, u_pts, h), rows_eq):
        clicks.append([(int(lm), float(r["S_thick"])), (int(lp), float(r["S_thin"]))])
    _, grp = kmeans2(
        P * h, 3, seed=1, minit="++"
    )  # the same groups as click_evidence.py
    folds = {
        "interleaved": [
            (np.arange(0, n, 2), np.arange(1, n, 2)),
            (np.arange(1, n, 2), np.arange(0, n, 2)),
        ],
        "blocks": [
            (np.flatnonzero(grp != g), np.flatnonzero(grp == g)) for g in range(3)
        ],
    }
    say("setup done (box, field, equal pass, clicks)")
    out: dict = {}

    # ---- 1. distances along the white surface, and the prior's correlation
    A = T.fld.pz["A"].tocsc()
    from scipy.sparse.linalg import splu

    luA = splu(A)
    cov = lambda cols: luA.solve(luA.solve(cols))  # noqa: E731  (A^-2: the z prior's covariance, up to tau)
    res1 = {}
    for name, fl in folds.items():
        rows = []
        for ev, held in fl:
            ev_patch = np.array([p for k in ev for p, _ in clicks[k]])
            dmin = dijkstra(s.graph, indices=s.rep[ev_patch], min_only=True)
            ev_nodes = np.unique(T.cnode[ev_patch])
            E = np.zeros((T.nN, ev_nodes.size))
            E[ev_nodes, np.arange(ev_nodes.size)] = 1
            Ce = cov(E)  # columns: cov(., click node)
            var_e = Ce[ev_nodes, np.arange(ev_nodes.size)]
            for k in held:
                dpts = float(np.min(np.linalg.norm(P[ev] - P[k], axis=1)) * h)
                for p, _ in clicks[k]:
                    i = T.cnode[p]
                    ei = np.zeros(T.nN)
                    ei[i] = 1
                    var_i = cov(ei)[i]
                    corr = Ce[i] / np.sqrt(var_i * var_e)
                    rows.append(
                        dict(
                            point=int(k) + 1,
                            root=int(p),
                            point_dist_mm=round(dpts, 2),
                            root_dist_mm=round(float(dmin[s.rep[p]] * h), 2),
                            max_prior_corr=round(float(corr.max()), 3),
                        )
                    )
        rd = np.array([r["root_dist_mm"] for r in rows])
        pdm = np.array([r["point_dist_mm"] for r in rows])
        cr = np.array([r["max_prior_corr"] for r in rows])
        res1[name] = dict(
            rows=rows,
            point_dist_median=round(float(np.median(pdm)), 2),
            root_dist_median=round(float(np.median(rd)), 2),
            root_dist_p90=round(float(np.percentile(rd, 90)), 2),
            prior_corr_median=round(float(np.median(cr)), 3),
        )
        say(
            f"1. {name}: points {res1[name]['point_dist_median']} mm apart (median), roots "
            f"{res1[name]['root_dist_median']} mm along the white surface (p90 {res1[name]['root_dist_p90']}); "
            f"prior correlation with the best click node, median {res1[name]['prior_corr_median']}"
        )
    out["distances"] = res1

    # ---- 2. what holds the belief back (means only)
    res2 = {}
    y_none = np.full(T.nN, np.nan)
    for name, fl in folds.items():
        acc = {"tops only": [], "tops + clicks": [], "clicks only": []}
        for ev, held in fl:
            ev_clicks = [e for k in ev for e in clicks[k]]
            ys = {
                "tops only": T.y_tops,
                "tops + clicks": T.with_clicks(ev_clicks),
                "clicks only": Thickness.with_clicks(_Blank(T, y_none), ev_clicks),
            }
            for key, y in ys.items():
                mu = means(T, y)
                acc[key] += [float(mu[p] - S) for k in held for p, S in clicks[k]]
        res2[name] = {
            k: dict(
                abs_median=round(float(np.median(np.abs(v))), 3),
                median=round(float(np.median(v)), 3),
            )
            for k, v in acc.items()
        }
        say(
            f"2. {name}: |mu - S| at the scored roots (mm, median): "
            + "; ".join(
                f"{k} {v['abs_median']} (signed {v['median']:+})"
                for k, v in res2[name].items()
            )
        )
    mu_all0 = means(T, T.y_tops)
    res2["tops_only_residual_by_point"] = [
        [round(float(mu_all0[p] - S), 2) for p, S in clicks[k]] for k in range(n)
    ]
    out["holding_back"] = res2

    # ---- 3. the prior's scale from the visible tops
    ev_nodes = np.flatnonzero(np.isfinite(T.y_tops))
    blk = np.floor(T.npos[ev_nodes] / 4.0).astype(int)
    _, bid = np.unique(blk, axis=0, return_inverse=True)
    fold_of = np.random.default_rng(3).integers(0, 5, bid.max() + 1)[bid.ravel()]
    res3 = []
    for ell, sig in GRID:
        set_prior(T, ell, sig)
        lpd, sq, zz, dist = [], [], [], []
        for k in range(5):
            y = T.y_tops.copy()
            te = ev_nodes[fold_of == k]
            y[te] = np.nan
            mu_n, sd_n, _ = T.fld.fit(y, trend=True, nsamp=30)
            v = sd_n[te] ** 2 + SIG_EPS**2
            r = T.y_tops[te] - mu_n[te]
            lpd += list(-0.5 * np.log(2 * np.pi * v) - 0.5 * r**2 / v)
            sq += list(r**2)
            zz += list(r / np.sqrt(v))
            tr = ev_nodes[fold_of != k]
            from scipy.spatial import cKDTree

            dist += list(cKDTree(T.npos[tr]).query(T.npos[te])[0])
        lpd, sq, zz, dist = map(np.asarray, (lpd, sq, zz, dist))
        far = dist >= 1.5
        row = dict(
            ell_mm=ell,
            sig=sig,
            lpd=round(float(lpd.mean()), 4),
            rmse_log=round(float(np.sqrt(sq.mean())), 4),
            z_sd=round(float(zz.std()), 3),
            lpd_far=round(float(lpd[far].mean()), 4),
            rmse_log_far=round(float(np.sqrt(sq[far].mean())), 4),
            far_share=round(float(far.mean()), 3),
        )
        res3.append(row)
        say(
            f"3. ELL_Z {ell} mm, SIG_Z {sig}: lpd {row['lpd']}, rmse(log) {row['rmse_log']}, z SD {row['z_sd']}; "
            f">= 1.5 mm from training ({row['far_share']:.0%}): lpd {row['lpd_far']}, rmse {row['rmse_log_far']}"
        )
    best = max(res3, key=lambda r: r["lpd"])
    out["prior_cv"] = dict(
        rows=res3, chosen=best, folds="5, over 4 mm spatial blocks of the tops' nodes"
    )
    say(
        f"3. chosen by the held-out log predictive density: ELL_Z {best['ell_mm']} mm, SIG_Z {best['sig']}"
    )

    # ---- 4. the click test with the chosen prior
    if (best["ell_mm"], best["sig"]) != CURRENT:
        set_prior(T, best["ell_mm"], best["sig"])
        res4 = {}
        for name, fl in {"tops only": [([], np.arange(n))], **folds}.items():
            rows = []
            for ev, held in fl:
                mu, sd, _ = T.weights(T.with_clicks([e for k in ev for e in clicks[k]]))
                f = run(s, mu, sd, ratio=RATIO)
                rows += score(s, f, mu, sd, P, u_pts, h, np.asarray(held))
            off = np.array([r["offset"] for r in rows])
            res4[name] = dict(
                median_abs=round(float(np.nanmedian(np.abs(off))), 3),
                median=round(float(np.nanmedian(off)), 3),
                worst=round(float(np.nanmax(np.abs(off))), 3),
                missing=int(np.isnan(off).sum()),
            )
            say(
                f"4. {name}: |offset| median {res4[name]['median_abs']} mm (signed {res4[name]['median']:+}), "
                f"worst {res4[name]['worst']}, missing {res4[name]['missing']}"
            )
        out["click_test_chosen_prior"] = res4
    json.dump(out, open(f"{WORK}/click_reach_M1S1.json", "w"), indent=1, default=float)
    say("done")


class _Blank:
    """A Thickness stand-in with no tops, for Thickness.with_clicks."""

    def __init__(self, T, y):
        self.y_tops, self.seed_node, self.seed_patch, self.nN = (
            y,
            T.seed_node,
            T.seed_patch,
            T.nN,
        )
        self._per_node = T._per_node


if __name__ == "__main__":
    main()
