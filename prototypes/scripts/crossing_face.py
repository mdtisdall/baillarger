"""Crossing paths at a marked face: the contact read from the laminar profile (plan: reports/Crossing
paths and contacts from laminar profiles.md). Location only; all 3D.

1. Box and points as onepass_face.py (V1V2_banks: the banks crop; M1S1: 141 voxels on the points).
2. Open columns: streamlines of the unsigned layer normal (crossing_lib.trace) from WM seeds (every
   root patch's centre seed, plus random seeds up to N_OPEN), kept where they end in fluid, >= 2 mm
   from a face, with path length over straight root-to-top distance below WANDER. Profile: trend-
   removed echo 2 and echo 3, disc-averaged (0.3 mm) over cortex only. Thickness t: the path length.
3. Test 2, spliced open columns (truth known): column A's profile followed by column B's reversed
   (roots >= 8 mm apart), split at t_A. Each side's templates from open columns whose roots lie
   within TPL_MM of its root along the white surface, leaving out those within HOLD_MM (and itself).
   Prior stand-in (no field: the field saw these tops): median and SD of the template columns' t.
4. Crossing paths through each marked point, three families:
   a) normal: the streamline of the unsigned normal both ways from the point until WM;
   b) column pair: descent of the equal-arrival pass's arrival from the point, and from the first
      point along the normal (either way, within 1.5 mm) whose label's root is > 3 mm along the white
      surface from the first descent's root, joined through the straight step between them;
   c) diagnostic: the straight line along the normal of a plane fitted to the neighbouring points
      (uses the truth to choose the direction; scores only whether the split is visible).
   Each side's templates: open columns with roots within TPL_MM of the path's end root on that side.
   Prior: the thickness field (click_evidence.Thickness, finding 26's guards) at the end roots' patches.
5. Estimators per path (crossing_lib): E0 symmetry, E-dark (echo 2), E1 flat prior and E2 with the
   prior, each for the linear and top-anchored stretch; prior alone (pairwise posterior); the one
   pass's label boundary on the same path, equal and field weights (sides by root distance along the
   white surface to the two end roots).
   Offset: estimate minus the point's position along the path (mm), + toward the path's B end.

Writes <work>/crossing_<site>.pkl and .json, and <figures>/crossing_<site>_*.png.
Usage: crossing_face.py <figures dir> <work dir> <data dir> <site> [--figs-only]
"""

import json
import pickle
import sys
import time

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import crossing_lib as X
import curv_lib as C
import flatten_fused2 as F
from click_evidence import Thickness, orientation
from onepass_lib import run, setup

FIG, WORK, DATA, SITE = sys.argv[1:5]
FIGS_ONLY = "--figs-only" in sys.argv
FULL = np.array([700, 384, 1148])
HALF, RATIO, PATCH, FACE_MM = 70, 3.0, 6, 2.0
N_OPEN, WANDER, TPL_MM, HOLD_MM, PAIR_MM, N_SPLICE = 8000, 1.3, 6.0, 1.0, 8.0, 400
RADIUS = 0.3
SAME_MM = 5.0  # end roots closer than this along the white surface: one bank (see figures())
DARK_MM = 0.24  # the ablation's gap: 1.5 voxels either side of the darkest point
KINDS = ("linear", "top")
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def box_and_points(site):
    if site == "M1S1":
        m = json.load(open(f"{DATA}/M1S1_test_points.json"))
        full = np.array(m["gap_all"]) + np.array(m["full_lo"])
        c = np.round(full.mean(0)).astype(int)
        lo, hi = c - HALF, c + HALF + 1
    else:
        pf = json.load(open(f"{DATA}/V1V2_banks_points.json"))
        box = json.load(open(f"{DATA}/V1V2_banks_sites.json"))
        # sorted and unique, as site_setup.py: the point numbers of finding 18
        full = np.unique(np.array(pf["contact_points"], int), axis=0) - 1 + np.array(box["full_lo"])
        lo, hi = np.array(box["full_lo"]), np.array(box["full_hi"])
    assert np.all(lo >= 0) and np.all(hi <= FULL)
    return lo, hi, (full - lo).astype(float)


def descend(u, labels, x0, vox, step_mm=X.STEP_MM, lmax=8.0):
    """Path down the gradient of arrival u from x0 until WM (labels == 2); (points, reached WM)."""
    g = np.stack(np.gradient(u), 0)
    shape = np.array(u.shape)
    x = np.asarray(x0, float).copy()
    pts = [x.copy()]
    L = 0.0
    while L < lmax:
        v = -np.array([ndi.map_coordinates(c, x.reshape(3, 1), order=1, mode="nearest")[0] for c in g])
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            return np.array(pts), False
        x = x + step_mm * v / nv / vox
        L += step_mm
        pts.append(x.copy())
        q = tuple(np.clip(np.round(x).astype(int), 0, shape - 1))
        if labels[q] == 2:
            return np.array(pts), True
        if labels[q] != 0:
            return np.array(pts), False
    return np.array(pts), False


def straight(labels, x0, d, vox, step_mm=X.STEP_MM, lmax=8.0):
    shape = np.array(labels.shape)
    pts = [np.asarray(x0, float)]
    for i in range(1, int(lmax / step_mm)):
        x = x0 + i * step_mm * d / vox
        pts.append(x)
        q = tuple(np.clip(np.round(x).astype(int), 0, shape - 1))
        if labels[q] == 2:
            return np.array(pts), True
        if labels[q] != 0 or np.any(x < 0) or np.any(x > shape - 1):
            return np.array(pts), False
    return np.array(pts), False


def estimate(y, L, tplA, tplB, prior):
    """All estimators on one profile y (2, m) of length L. tplA, tplB: {kind: Template}; prior:
    (muA, sdA, muB, sdB) mm or None. Positions in mm from the A end."""
    xs = X.xgrid(L, 0.1, 0.9)
    out = {}
    sym = X.symmetry(y, L, xs)
    out["E0 symmetry"] = dict(x=float(xs[np.nanargmax(sym)]) if np.isfinite(sym).any() else np.nan)
    out["E-dark"] = dict(x=X.darkest(y[0], L))
    curves = dict(xs=xs, sym=sym)
    lp = X.gauss_prior(xs, L, *prior) if prior is not None else None
    if prior is not None:
        px, psd = X.pairwise_posterior(L, *prior)
        out["prior"] = dict(x=float(px), sd=float(psd))
        curves["prior"] = lp
    for k in KINDS:
        if tplA.get(k) is None or tplB.get(k) is None:
            continue
        ll = X.loglik(y, tplA[k], tplB[k], L, xs)
        curves[f"ll {k}"] = ll
        r = X.summarize(xs, ll)
        out[f"E1 {k}"] = dict(x=r["mean"], sd=r["sd"], map=r["map"])
        if lp is not None:
            r = X.summarize(xs, ll + lp)
            out[f"E2 {k}"] = dict(x=r["mean"], sd=r["sd"], map=r["map"])
        # ablation: without the dark line, so the split comes from the bands alone (samples within
        # DARK_MM of E-dark's point left out; that point is read from the data, not the truth)
        xd = out["E-dark"]["x"]
        if np.isfinite(xd):
            yd = y.copy()
            yd[:, np.abs(np.arange(y.shape[1]) * X.STEP_MM - xd) <= DARK_MM] = np.nan
            r = X.summarize(xs, X.loglik(yd, tplA[k], tplB[k], L, xs))
            out[f"E1 {k} -dark"] = dict(x=r["mean"], sd=r["sd"], map=r["map"])
    return out, curves


def main():
    lo, hi, P = box_and_points(SITE)
    B = C.load(lo, hi, f"crossing_{SITE}")
    vox = B["vox"]
    h = float(np.mean(vox))
    wm, ctx, rim = B["wm"], B["ctx"], B["rim"]
    say(f"{SITE}: box {tuple(hi - lo)} from {tuple(lo)}; {len(P)} points; nearest "
        f"{np.min(np.minimum(P, (hi - lo) - 1 - P)) * h:.1f} mm from a face; voxel {vox.round(4)}")
    labels = np.full(rim.shape, 4, np.int16)
    labels[ctx] = 0
    labels[rim == 1] = 1
    labels[wm] = 2
    labels[B["miss"]] = 3
    feats = [B["det"][1], B["det"][2]]  # echo 2 and echo 3, trend removed
    mb = ctx.astype(float)
    Tc = F.tensor(feats[0], mb) + F.tensor(feats[1], mb)
    nh, coh = C.tensor_normal(B)

    # ---- one pass and the thickness field (prior; family b)
    dirv = orientation(B)
    s = setup(wm, ctx, patch=PATCH, direction=dirv)
    del dirv
    say(f"seeds {s.seeds.sum():,}, patches {s.npatch:,}")
    Thk = Thickness(B, s, h)
    mu_f, sd_f, nobel = Thk.weights(Thk.y_tops)
    say(f"thickness field: {np.isfinite(Thk.y_tops).sum():,} nodes with tops; {nobel} patches without belief")
    f_eq = run(s, ratio=RATIO)
    f_fd = run(s, mu_f, sd_f, ratio=RATIO)
    say("passes done")
    seed_pos = np.stack(np.unravel_index(s.sflat, wm.shape), -1).astype(float)
    seed_tree = cKDTree(seed_pos)
    seed_patch = s.patch.ravel()[s.sflat]

    def root_of(x):
        return int(seed_tree.query(x)[1])

    def geo_from(seed, limit_mm):
        return dijkstra(s.graph, indices=seed, limit=limit_mm / h) * h

    # ---- open columns
    rng = np.random.default_rng(0)
    d_wm = ndi.distance_transform_edt(~wm, sampling=vox)
    gw = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0), *vox), -1)
    gw /= np.linalg.norm(gw, axis=-1, keepdims=True) + 1e-12
    ok_seed = B["dface"].ravel()[s.sflat] >= FACE_MM
    pick = np.unique(np.r_[s.rep[ok_seed[s.rep]], rng.choice(np.flatnonzero(ok_seed), N_OPEN, replace=False)])
    x0 = seed_pos[pick] + 0.5 * gw[tuple(seed_pos[pick].astype(int).T)]
    paths, outc = X.trace(Tc, labels, B["dface"], x0, gw[tuple(seed_pos[pick].astype(int).T)], vox,
                          wm_after=0.3, lmax=6.0)
    cols = dict(root=[], t=[], prof=[], end=[])
    for i, (p, o) in enumerate(zip(paths, outc)):
        if o != X.FLUID or len(p) < 10:
            continue
        p = p[:-1]  # the last sample is in fluid
        pr, ss = X.resample(p, vox)
        straight_mm = np.linalg.norm((pr[-1] - seed_pos[pick[i]]) * vox)
        if ss[-1] < 0.8 or ss[-1] > 6.0 or ss[-1] / max(straight_mm, 1e-6) > WANDER:
            continue
        cols["root"].append(pick[i]); cols["t"].append(ss[-1]); cols["end"].append(pr[-1])
        cols["prof"].append(X.profile(feats, ctx, pr, vox, RADIUS))
    cols["root"] = np.array(cols["root"]); cols["t"] = np.array(cols["t"])
    oc = np.bincount(outc, minlength=8)
    say(f"open columns: {len(pick):,} traced ({', '.join(f'{X.OUTCOME[k]} {oc[k]}' for k in range(8) if oc[k])}); "
        f"kept {cols['t'].size:,}, t p5/50/95 {np.percentile(cols['t'], [5, 50, 95]).round(2)}")
    col_of_seed = {}
    for j, r in enumerate(cols["root"]):
        col_of_seed.setdefault(int(r), []).append(j)

    def templates(seed, exclude_mm=0.0, exclude=()):
        d = geo_from(seed, TPL_MM)
        dc = d[cols["root"]]
        sel = np.flatnonzero((dc <= TPL_MM) & (dc >= exclude_mm))
        sel = np.setdiff1d(sel, np.asarray(exclude, int))
        out = {k: X.build_template([cols["prof"][j] for j in sel], cols["t"][sel], k) for k in KINDS}
        return out, sel

    # ---- test 2: spliced open columns
    spl = []
    order = rng.permutation(cols["t"].size)
    for ia in order:
        if len(spl) >= N_SPLICE:
            break
        ra = int(cols["root"][ia])
        far = np.flatnonzero(np.linalg.norm((seed_pos[cols["root"]] - seed_pos[ra]) * vox, axis=1) >= PAIR_MM)
        if far.size == 0:
            continue
        ib = int(rng.choice(far))
        rb = int(cols["root"][ib])
        tA, tB = cols["t"][ia], cols["t"][ib]
        y = np.concatenate([cols["prof"][ia], cols["prof"][ib][:, ::-1]], 1)
        L = (y.shape[1] - 1) * X.STEP_MM
        tpA, selA = templates(ra, HOLD_MM, [ia])
        tpB, selB = templates(rb, HOLD_MM, [ib])
        if tpA["linear"] is None or tpB["linear"] is None:
            continue
        prior = (np.median(cols["t"][selA]), np.std(cols["t"][selA]) + 0.05,
                 np.median(cols["t"][selB]), np.std(cols["t"][selB]) + 0.05)
        est, _ = estimate(y, L, tpA, tpB, prior)
        spl.append(dict(tA=tA + X.STEP_MM / 2, tB=tB, L=L, nA=selA.size, nB=selB.size, est=est))
        if len(spl) % 50 == 0:
            say(f"   spliced {len(spl)}")
    say(f"spliced columns: {len(spl)}")

    # ---- crossing paths through the points
    # arrival 0 on all WM (the pass starts from the seeds; WM inside is outside its domain), so
    # descent near the white surface points into the WM, not away from an undefined interior
    u = np.where(wm, 0.0, np.where(np.isfinite(f_eq.arrival), f_eq.arrival,
                                   np.nanmax(f_eq.arrival[np.isfinite(f_eq.arrival)])))
    rows = []
    for k_, Pk in enumerate(P):
        q = tuple(np.round(Pk).astype(int))
        n0 = nh[q] / (np.linalg.norm(nh[q]) + 1e-12)
        fam = {}
        # a) normal streamline both ways
        pp, oo = X.trace(Tc, labels, B["dface"], np.array([Pk, Pk]), np.array([n0, -n0]), vox)
        fam["a normal"] = (pp[1][::-1], pp[0], bool(oo[0] == X.WM and oo[1] == X.WM),
                           f"{X.OUTCOME[oo[1]]}/{X.OUTCOME[oo[0]]}")
        # b) column pair from the equal-arrival pass
        d1, ok1 = descend(u, labels, Pk, vox)
        start2, ok2, d2 = None, False, None
        if ok1:
            r1 = root_of(d1[-1])
            g1 = geo_from(r1, 40.0)
            for dist in np.arange(0.08, 1.5, 0.04):
                for sgn in (1, -1):
                    x = Pk + sgn * dist * n0 / vox
                    qq = tuple(np.clip(np.round(x).astype(int), 0, np.array(wm.shape) - 1))
                    lab = f_eq.label[qq]
                    if labels[qq] == 0 and lab >= 0 and g1[s.rep[lab]] > 3.0:
                        start2 = x
                        break
                if start2 is not None:
                    break
            if start2 is not None:
                d2, ok2 = descend(u, labels, start2, vox)
        if ok1 and ok2:
            seg = np.linspace(Pk, start2, 5)[1:-1]
            fam["b column pair"] = (d1[::-1], np.r_[seg, d2], True, "WM/WM")
        else:
            fam["b column pair"] = (d1[::-1], d2 if d2 is not None else Pk[None], False,
                                    "no second side" if ok1 else "first descent failed")
        # c) straight along the normal of the plane fitted to neighbouring points
        dd = np.linalg.norm((P - Pk) * vox, axis=1)
        nb = P[np.argsort(dd)[:max(4, int((dd < 2.5).sum()))]]
        _, V = np.linalg.eigh(np.cov(((nb - nb.mean(0)) * vox).T))
        nc = V[:, 0] * (1 if np.dot(V[:, 0], n0) >= 0 else -1)
        sa, oa = straight(labels, Pk, nc, vox)
        sb, ob = straight(labels, Pk, -nc, vox)
        fam["c plane line"] = (sb[::-1], sa, bool(oa and ob), "")
        row = dict(k=k_ + 1, coh=float(coh[q]), angle_plane_normal=float(np.degrees(np.arccos(min(1, abs(np.dot(n0, nc)))))))
        for name, (half_a, half_b, ok, why) in fam.items():
            rec = dict(ok=ok, why=why, halves=(half_a, half_b))
            if ok:
                path = np.r_[half_a, half_b[1:]]
                pr, ss = X.resample(path, vox)
                sP = float(X.arclength(half_a, vox)[-1])
                L = float(ss[-1])
                y = X.profile(feats, ctx, pr, vox, RADIUS)
                ra, rb = root_of(path[0]), root_of(path[-1])
                tpA, selA = templates(ra)
                tpB, selB = templates(rb)
                pa, pb = seed_patch[ra], seed_patch[rb]
                prior = (mu_f[pa] * h, sd_f[pa] * h, mu_f[pb] * h, sd_f[pb] * h)
                est, curves = estimate(y, L, tpA, tpB, prior)
                # the one pass's label boundary on the same path: sides by root distance
                ga, gb = geo_from(ra, 60.0), geo_from(rb, 60.0)
                for pname, f in (("one pass, equal", f_eq), ("one pass, field", f_fd)):
                    qi = np.clip(np.round(pr).astype(int), 0, np.array(wm.shape) - 1)
                    lab = f.label[tuple(qi.T)]
                    side = np.where(lab >= 0, np.where(ga[s.rep[np.maximum(lab, 0)]] <= gb[s.rep[np.maximum(lab, 0)]], 0, 1), -1)
                    sw = np.flatnonzero((side[:-1] == 0) & (side[1:] == 1))
                    est[pname] = dict(x=float(ss[sw[np.argmin(np.abs(ss[sw] - sP))]] + X.STEP_MM / 2) if sw.size else np.nan)
                rec.update(L=L, sP=sP, prior=prior, nA=selA.size, nB=selB.size, est=est, curves=curves,
                           y=y, path=pr, roots_sep_mm=float(ga[rb]) if np.isfinite(ga[rb]) else np.inf,
                           low_coh=float(np.mean(ndi.map_coordinates(coh, pr.T, order=1) < 0.3)))
            rows.append(dict(row, family=name, **rec))
        say(f"point {k_ + 1}: " + "; ".join(f"{r['family']}: {'ok' if r['ok'] else r['why']}"
                                            for r in rows[-3:]))
    disp = B["disp"]
    pickle.dump(dict(site=SITE, P=P, lo=lo, vox=vox, spl=spl, rows=rows, disp=disp, ctx=ctx, coh=coh, wm=wm,
                     t_open=cols["t"]), open(f"{WORK}/crossing_{SITE}.pkl", "wb"))
    say("saved")


# ---------------------------------------------------------------- summaries and views

EST_ORDER = ["E0 symmetry", "E-dark", "E1 linear", "E1 top", "E1 linear -dark", "E1 top -dark", "E2 linear", "E2 top", "prior",
             "one pass, equal", "one pass, field"]
COL = {"E1 linear -dark": "#7fb3ea", "E1 top -dark": "#8fd6b8", "E0 symmetry": "#8a8984", "E-dark": "#000000", "E1 linear": "#2a78d6", "E1 top": "#1baf7a",
       "E2 linear": "#6a3dc4", "E2 top": "#0c7a52", "prior": "#e34948", "one pass, equal": "#eb6834",
       "one pass, field": "#c79a00"}


def stats(v):
    v = np.asarray(v, float)
    f = v[np.isfinite(v)]
    if f.size == 0:
        return dict(n=0)
    return dict(n=int(f.size), missing=int(v.size - f.size), median=round(float(np.median(f)), 3),
                iqr=round(float(np.subtract(*np.percentile(f, [75, 25]))), 3),
                median_abs=round(float(np.median(np.abs(f))), 3), p90_abs=round(float(np.percentile(np.abs(f), 90)), 3))


def figures():
    R = pickle.load(open(f"{WORK}/crossing_{SITE}.pkl", "rb"))
    # a path whose two ends lie on one bank is not a crossing: it slid along the arrival's ridge
    # back to its own root (finding 4). At both faces the end roots are either under 1.9 mm or over
    # 14 mm apart along the white surface, so any cut between is the same; SAME_MM sits in the gap.
    for r in R["rows"]:
        if r["ok"] and r["roots_sep_mm"] < SAME_MM:
            r["ok"], r["why"] = False, "both ends on one bank"
    out = dict(site=SITE)
    # test 2
    spl = R["spl"]
    t2 = {}
    for e in EST_ORDER:
        err = [r["est"][e]["x"] - r["tA"] for r in spl if e in r["est"]]
        if err:
            t2[e] = stats(err)
            if "sd" in spl[0]["est"].get(e, {}):
                z = [(r["est"][e]["x"] - r["tA"]) / r["est"][e]["sd"] for r in spl if e in r["est"] and r["est"][e]["sd"] > 0]
                t2[e]["z_sd"] = round(float(np.std(z)), 2)
                t2[e]["z_sd_robust"] = round(float(1.4826 * np.median(np.abs(np.array(z) - np.median(z)))), 2)
                t2[e]["sd_median"] = round(float(np.median([r["est"][e]["sd"] for r in spl if e in r["est"]])), 3)
    out["spliced"] = dict(n=len(spl), errors_mm=t2)
    # points
    fams = sorted({r["family"] for r in R["rows"]})
    out["points"] = {}
    for fm in fams:
        rr = [r for r in R["rows"] if r["family"] == fm]
        okr = [r for r in rr if r["ok"]]
        d = dict(paths_ok=len(okr), of=len(rr), failures=sorted({r["why"] for r in rr if not r["ok"]}))
        for e in EST_ORDER:
            d[e] = stats([r["est"][e]["x"] - r["sP"] if e in r["est"] else np.nan for r in okr])
            if e.startswith(("E1", "E2")):  # includes the ablations
                z = [(r["est"][e]["x"] - r["sP"]) / r["est"][e]["sd"] for r in okr if e in r["est"] and r["est"][e]["sd"] > 0]
                d[e]["z_sd_robust"] = round(float(1.4826 * np.median(np.abs(np.array(z) - np.median(z)))), 2) if z else None
                d[e]["sd_median"] = round(float(np.median([r["est"][e]["sd"] for r in okr if e in r["est"]])), 3) if okr else None
        out["points"][fm] = d
    json.dump(out, open(f"{WORK}/crossing_{SITE}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

    # ---- figure 1: spliced columns
    fig, ax = plt.subplots(1, 3, figsize=(22, 6.5))
    bins = np.linspace(-1.5, 1.5, 61)
    for e in ["E0 symmetry", "E-dark", "E1 linear", "E1 top", "E1 linear -dark", "E2 linear", "prior"]:
        err = np.array([r["est"][e]["x"] - r["tA"] for r in spl if e in r["est"]])
        ax[0].hist(np.clip(err, -1.5, 1.5), bins, histtype="step", lw=1.8, color=COL[e],
                   label=f"{e}: |err| median {np.nanmedian(np.abs(err)):.2f}")
    ax[0].set_xlabel("estimated split minus the true one (mm; clipped at ±1.5)"); ax[0].legend(fontsize=9)
    ax[0].set_title("spliced open columns: error of each estimator")
    for e in ["E1 linear", "E1 top", "E2 linear", "E2 top"]:
        z = np.array([(r["est"][e]["x"] - r["tA"]) / r["est"][e]["sd"] for r in spl if e in r["est"]])
        ax[1].hist(np.clip(z, -8, 8), np.linspace(-8, 8, 65), histtype="step", lw=1.8, color=COL[e], label=e)
    zz = np.linspace(-8, 8, 200)
    ax[1].plot(zz, len(spl) * 0.25 * np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k:", label="N(0, 1)")
    ax[1].set_xlabel("z = error / posterior SD (clipped at ±8)"); ax[1].legend(fontsize=9)
    ax[1].set_title("is the profile likelihood's uncertainty honest?")
    tA = np.array([r["tA"] for r in spl])
    for e in ["E1 linear", "prior"]:
        err = np.array([r["est"][e]["x"] - r["tA"] for r in spl])
        ax[2].plot(tA, err, ".", ms=4, color=COL[e], label=e, alpha=0.7)
    ax[2].axhline(0, c="k", lw=0.6); ax[2].set_ylim(-1.5, 1.5)
    ax[2].set_xlabel("true t_A (mm)"); ax[2].set_ylabel("error (mm)"); ax[2].legend(fontsize=9)
    ax[2].set_title("error against the true thickness")
    fig.suptitle(f"{SITE}, test 2: two open columns spliced pia to pia (roots ≥ {PAIR_MM:.0f} mm apart), "
                 f"templates held out within {HOLD_MM} mm; prior: the local tops' median and SD (a stand-in)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIG}/crossing_{SITE}_spliced.png", dpi=80); plt.close(fig)

    # ---- figure 2: profile stacks per family, aligned at the point
    sv = np.arange(-3.0, 3.0 + 1e-9, X.STEP_MM)
    fig, ax = plt.subplots(1, len(fams), figsize=(9 * len(fams), 12))
    for a, fm in zip(np.atleast_1d(ax), fams):
        rr = [r for r in R["rows"] if r["family"] == fm]
        img = np.full((len(rr), sv.size), np.nan)
        for i, r in enumerate(rr):
            if r["ok"]:
                ss = np.arange(r["y"].shape[1]) * X.STEP_MM - r["sP"]
                img[i] = np.interp(sv, ss, r["y"][0], left=np.nan, right=np.nan)
        v = np.nanpercentile(np.abs(img - np.nanmedian(img)), 97)
        a.imshow(img, aspect="auto", cmap="gray", extent=[sv[0], sv[-1], len(rr) + 0.5, 0.5],
                 vmin=np.nanmedian(img) - v, vmax=np.nanmedian(img) + v, interpolation="nearest")
        a.axvline(0, c="#ffd23f", lw=0.8)
        for e, mk in (("E1 linear", "o"), ("E1 top", "s"), ("E1 linear -dark", "v"), ("E2 linear", "D"), ("prior", "x"),
                      ("one pass, field", "+"), ("E0 symmetry", "^")):
            xs_ = [r["est"][e]["x"] - r["sP"] if r["ok"] and e in r["est"] else np.nan for r in rr]
            a.plot(xs_, np.arange(1, len(rr) + 1), mk, mfc="none", mec=COL[e], ms=7, mew=1.5, label=e)
        for i, r in enumerate(rr):
            if r["ok"]:
                a.plot([-r["sP"], r["L"] - r["sP"]], [i + 1, i + 1], "|", c="#e34948", ms=10)
            else:
                a.text(-2.9, i + 1, r["why"], color="#e34948", fontsize=7, va="center")
        a.set_xlim(sv[0], sv[-1])
        a.set_xlabel("position along the path (mm; 0 = your point; + toward the path's B end; red ticks: the WM ends)")
        a.set_ylabel("your point"); a.set_title(f"{fm}: echo 2 trend removed along the crossing path", fontsize=12)
        a.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{SITE}: crossing paths through your points, the split each estimator reads (all 3D; disc {RADIUS} mm)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{FIG}/crossing_{SITE}_profiles.png", dpi=72); plt.close(fig)

    # ---- figure 3: likelihood, prior and posterior per point (family a, else b)
    for fm in fams:
        rr = [r for r in R["rows"] if r["family"] == fm and r["ok"]]
        if not rr:
            continue
        nc_ = 6
        nr = int(np.ceil(len(rr) / nc_))
        fig, ax = plt.subplots(nr, nc_, figsize=(4 * nc_, 2.6 * nr), squeeze=False)
        for a, r in zip(ax.ravel(), rr):
            c = r["curves"]
            xs = c["xs"] - r["sP"]
            for key, colr, lab in (("ll linear", COL["E1 linear"], "profile, linear"), ("ll top", COL["E1 top"], "profile, top"),
                                   ("prior", COL["prior"], "field prior")):
                if key in c:
                    v = np.exp(c[key] - np.nanmax(c[key])); a.plot(xs, v, c=colr, lw=1.3, label=lab)
            if "ll linear" in c:
                v = c["ll linear"] + c["prior"]; a.plot(xs, np.exp(v - np.nanmax(v)), c=COL["E2 linear"], lw=1, ls="--", label="posterior, linear")
            sym = c["sym"]
            a.plot(xs, (sym - np.nanmin(sym)) / (np.nanmax(sym) - np.nanmin(sym) + 1e-12), c=COL["E0 symmetry"], lw=0.8, label="symmetry (scaled)")
            a.axvline(0, c="#ffd23f", lw=1.2)
            fx = r["est"]["one pass, field"]["x"]
            if np.isfinite(fx):
                a.axvline(fx - r["sP"], c=COL["one pass, field"], lw=1, ls=":")
            a.set_xlim(-1.5, 1.5); a.set_yticks([])
            a.set_title(f"#{r['k']}: L {r['L']:.2f}, coh {r['coh']:.2f}", fontsize=9)
        for a in ax.ravel()[len(rr):]:
            a.axis("off")
        ax[0, 0].legend(fontsize=6, loc="upper left")
        fig.suptitle(f"{SITE}, {fm}: profile likelihood (flat prior), field prior and posterior over the split, "
                     f"against your point (yellow, 0) and the one pass with field weights (dotted); mm", fontsize=12)
        fig.tight_layout(); fig.savefig(f"{FIG}/crossing_{SITE}_curves_{fm.split()[0]}.png", dpi=72); plt.close(fig)

    # ---- figure 4: planes holding most points, paths projected, splits marked
    P, disp, ctx = R["P"], R["disp"], R["ctx"]
    kz = np.round(P[:, 2]).astype(int)
    planes = [k for k, _ in sorted(((k, (kz == k).sum()) for k in np.unique(kz)), key=lambda t: -t[1])[:3]]
    fig, axs = plt.subplots(2, len(planes), figsize=(9 * len(planes), 18), squeeze=False)
    for bg, a, k in [(bg, a, k) for bg, ax in zip(("echo", "coh"), axs) for a, k in zip(ax, planes)]:
        if bg == "echo":
            sl = np.where(ctx[:, :, k], disp[:, :, k], np.nan)
            v = np.nanpercentile(np.abs(sl - np.nanmedian(sl)), 98)
            a.imshow(sl.T, cmap="gray", origin="lower", vmin=np.nanmedian(sl) - v, vmax=np.nanmedian(sl) + v)
        else:
            a.imshow(np.where(ctx[:, :, k], R["coh"][:, :, k], np.nan).T, cmap="viridis", origin="lower", vmin=0, vmax=1)
            a.contour(R["wm"][:, :, k].T.astype(float), [0.5], colors="w", linewidths=0.6)
        for r in R["rows"]:
            if abs(P[r["k"] - 1, 2] - k) > 0.5:
                continue
            c = {"a normal": "#2a78d6", "b column pair": "#eb6834", "c plane line": "#1baf7a"}[r["family"]]
            if not r["ok"]:
                if r["family"] != "c plane line":
                    for hh in r["halves"]:
                        a.plot(hh[:, 0], hh[:, 1], "--", c=c, lw=0.8)
                        a.plot(hh[-1, 0], hh[-1, 1], "x", c="#e34948", ms=5)
                continue
            a.plot(r["path"][:, 0], r["path"][:, 1], "-", c=c, lw=0.8, alpha=0.8)
            for e, mk in (("E1 linear", "o"), ("E2 linear", "D"), ("one pass, field", "+")):
                x = r["est"][e]["x"]
                if np.isfinite(x):
                    j = min(int(round(x / X.STEP_MM)), len(r["path"]) - 1)
                    a.plot(r["path"][j, 0], r["path"][j, 1], mk, mfc="none", mec=c, ms=6)
        m = np.abs(P[:, 2] - k) < 0.5
        a.plot(P[m, 0], P[m, 1], "*", c="#ffd23f", ms=9)
        for i in np.flatnonzero(m):
            a.annotate(str(i + 1), (P[i, 0], P[i, 1]), color="#ffd23f", fontsize=8, xytext=(4, -8), textcoords="offset points")
        a.set_title(f"k = {k} (0-based), {'echo 2 trend removed' if bg == 'echo' else 'coherence (WM outlined)'}: paths (blue a, "
                    f"orange b, green c; dashed: failed, x where it stopped); o E1 linear, ◇ E2 linear, + one pass field", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/crossing_{SITE}_planes.png", dpi=72); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    if not FIGS_ONLY:
        main()
    figures()
