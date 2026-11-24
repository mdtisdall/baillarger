"""Where to look first when correcting the joint warp's pia: an edit-priority map (plan:
reports/Crossing paths and contacts from laminar profiles.md, *Correction tools*).

- margins: per variable, the energy rise when its label alone moves by d, from the terms that
  touch it (vectorized over terms; exact against baillarger.surfcut.energy differences).
- main: for a saved joint result, per subsampled trace,
  (a) the solver's doubt: the pia not certified by QPBO and its ambiguity interval (the labels
      QPBO could not rule out; baillarger.surfcut.Result.interval_lo/hi), from one QPBO on the
      fine model centred on the saved levels;
  (b) a flat margin: the smallest energy rise of the pia moved alone by +-0.1 and +-0.2 mm, every
      other variable fixed (on a 0.02 mm grid, so that both moves are whole labels);
  (c) model doubts the pipeline already writes: the joint pia against the single-surface horizon
      (pia_<site>.npz c), the profile split (column_volume H_profile, top); open tops the pia
      left by more than the anchor's capture range (overridden); bands not found
      (bands_<tag>.npz taus nan).
  Each doubt as a percentile rank over the traces; the combined score is their mean (no weights).
  Scores at the marked points against the points' errors, and views on the marked planes.
Writes <work>/edit_priority_<tag>.npz and .json, <figures>/edit_priority_<tag>_sections.png.

Usage: edit_priority.py <figures> <work> <tag>   (tag as joint_<tag>.npz, e.g. V1V2_banks_s0.16)
Needs baillarger.surfcut with Result.interval_lo/hi (PR #31, merged).
"""

import json
import sys
import time

import numpy as np

import baillarger.surfcut as WC
from baillarger.surfcut.model import floats, hard_terms, hinge_terms, ints, tie_terms

MOVES_MM = (-0.2, -0.1, 0.1, 0.2)


def margins(M, k, deltas):
    """(len(deltas), n): energy(k with k_i + d) - energy(k) for each variable i moved alone; inf
    where k_i + d leaves its window or breaks a hard bound. k must satisfy the hard bounds."""
    n, K, o = M.n, ints(M.K), ints(M.o)
    k = ints(k)
    x = o + k
    start = np.r_[0, np.cumsum(K)]
    U = np.concatenate([floats(u) for u in M.unary])
    ha, hb, hlo, hhi = hard_terms(M)
    ga, gb, gt, gw = hinge_terms(M)
    ta, tb, tL, tw = tie_terms(M)
    diff = x[hb] - x[ha]
    diff_g = x[gb] - x[ga] - gt
    r0 = np.abs(diff_g)
    s = x[ta] + x[tb] - tL
    s0 = np.abs(s)
    tabs = []
    for qa, qb, T in M.table:
        qa, qb, T = ints(qa), ints(qb), floats(T)
        p = np.arange(qa.size)
        tabs.append((qa, qb, T, p, T[p, k[qa], k[qb]]))
    out = np.zeros((len(deltas), n))
    for r, d in enumerate(deltas):
        k2 = k + d
        ok = (k2 >= 0) & (k2 < K)
        kc = np.clip(k2, 0, K - 1)
        e = np.where(ok, U[start[:-1] + kc] - U[start[:-1] + k], np.inf)
        # hard bounds: lo <= x_b - x_a <= hi with one end moved
        bad_a = (diff - d < hlo) | (diff - d > hhi)
        bad_b = (diff + d < hlo) | (diff + d > hhi)
        bad = np.zeros(n, bool)
        bad[ha[bad_a]] = True
        bad[hb[bad_b]] = True
        e[bad] = np.inf
        # hinges w |x_b - x_a - t|
        e += np.bincount(ga, gw * (np.abs(diff_g - d) - r0), minlength=n)
        e += np.bincount(gb, gw * (np.abs(diff_g + d) - r0), minlength=n)
        # ties w |x_a + x_b - L|
        dt = tw * (np.abs(s + d) - s0)
        e += np.bincount(ta, dt, minlength=n) + np.bincount(tb, dt, minlength=n)
        # tables theta(k_a, k_b), nan padding outside the windows
        for qa, qb, T, p, t0 in tabs:
            ia, ib = np.clip(k[qa] + d, 0, T.shape[1] - 1), np.clip(k[qb] + d, 0, T.shape[2] - 1)
            # outside a window: the unary part above is already inf; keep the table finite there
            da = np.where(ok[qa], T[p, ia, k[qb]] - t0, 0.0)
            db = np.where(ok[qb], T[p, k[qa], ib] - t0, 0.0)
            e += np.bincount(qa, np.nan_to_num(da, nan=np.inf), minlength=n)
            e += np.bincount(qb, np.nan_to_num(db, nan=np.inf), minlength=n)
        out[r] = e
    return out


def ranks(x):
    """Rank in [0, 1): the fraction of all traces whose doubt is defined and strictly below x (nan:
    no doubt known, rank 0). Ties share the lower rank, so no doubt (0) ranks 0 however many
    traces share it."""
    x = np.asarray(x, float)
    f = np.isfinite(x)
    srt = np.sort(x[f])
    out = np.zeros(x.size)
    out[f] = np.searchsorted(srt, x[f], side="left") / x.size
    return out


def main(fig, work, tag):
    import matplotlib

    matplotlib.use("Agg")
    from scipy.stats import spearmanr

    import edit_data as ED
    import joint_levels as JL

    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    D, J, Z = ED.load(work, tag, log=say)
    site = ED.site_of(tag)
    Q = np.load(f"{work}/pia_{site}.npz")
    sub, z, cert_saved = J["sub"], J["z"], J["cert"]
    L, n = z.shape
    h = 0.04
    prm = JL.Params()
    # (a) QPBO on the fine model centred on the saved levels: certification and intervals
    M, _ = JL.build(D, h, z, np.where(cert_saved, 0.16, 0.48), prm)
    r = WC.qpbo(M, improve=False, log=say)
    pv = slice((L - 1) * n, L * n)
    cert = r.certified[pv]
    width = h * (r.interval_hi[pv] - r.interval_lo[pv])
    lo_mm = h * (np.asarray(M.o)[pv] + r.interval_lo[pv])
    hi_mm = h * (np.asarray(M.o)[pv] + r.interval_hi[pv])
    # an interval clipped by its window (at an edge that is not depth 0): its width is a lower bound
    at_edge = ~cert & (((r.interval_lo[pv] == 0) & (np.asarray(M.o)[pv] > 0))
                       | (r.interval_hi[pv] == np.asarray(M.K)[pv] - 1))
    say(f"pia certified {cert.mean():.1%} (saved {cert_saved[-1].mean():.1%}); "
        f"interval width of the uncertified: median {np.median(width[~cert]) if (~cert).any() else 0:.2f} mm")
    # (b) the flat margin, on a 0.02 mm grid: the pia +-0.2 mm, the other levels fixed
    h2 = 0.02
    half2 = np.zeros((L, n))
    half2[-1] = 0.2 + h2
    M2, _ = JL.build(D, h2, z, half2, prm)
    k2 = np.round(z.ravel() / h2).astype(int) - np.asarray(M2.o)
    dl = [int(round(m / h2)) for m in MOVES_MM]
    mg = margins(M2, k2, dl)[:, pv]
    flat = np.min(np.where(np.isfinite(mg), mg, np.nan), axis=0)
    say(f"margins: median flat margin {np.nanmedian(flat):.3f}; negative at {np.sum(flat < -1e-9):,} traces")
    # (c) model doubts
    c_single = Q["c"][sub]
    prof = Z["H_profile, top"][sub].astype(float)
    top = J["top"]
    over = np.where(np.isfinite(top), np.abs(z[-1] - top) - prm.capture, np.nan)
    Bz = np.load(f"{work}/bands_{tag}.npz")
    assert np.array_equal(Bz["sub"], sub)
    nob = np.sum(~np.isfinite(Bz["taus"]), axis=0).astype(float)
    doubts = {
        "interval width": np.where(cert, 0.0, width),
        "flat margin": -flat,
        "|joint - single-surface|": np.abs(z[-1] - c_single),
        "|joint - profile split|": np.abs(z[-1] - prof),
        "open top overridden": np.where(over > 0, over, np.where(np.isfinite(over), 0.0, np.nan)),
        "bands not found": nob,
    }
    R_ = {k: ranks(v) for k, v in doubts.items()}
    combined = np.mean(np.stack(list(R_.values())), axis=0)
    # at the points
    hits = ED.point_traces(Z, sub)
    err = ED.point_errors(z[-1], hits)
    has = np.array([i.size > 0 for i, _ in hits])
    per = {k: np.array([np.median(v[i]) if i.size else np.nan for i, _ in hits]) for k, v in R_.items()}
    per["combined"] = np.array([np.median(combined[i]) if i.size else np.nan for i, _ in hits])
    per_raw = {k: np.array([np.median(np.nan_to_num(v[i], nan=0.0)) if i.size else np.nan for i, _ in hits])
               for k, v in doubts.items()}
    q90 = float(np.percentile(combined, 90))
    rho = {k: round(float(spearmanr(v[has], err[has]).statistic), 3) for k, v in per.items()}
    bad = has & (err > 0.3)
    flagged = has & (per["combined"] >= q90)
    points = [dict(point=int(p + 1), traces=int(hits[p][0].size), error_mm=round(float(err[p]), 3),
                   combined=round(float(per["combined"][p]), 3),
                   **{k: round(float(per_raw[k][p]), 3) for k in doubts})
              for p in np.flatnonzero(has)]
    summary = dict(tag=tag, traces=int(n), pia_certified=round(float(cert.mean()), 4),
                   pia_certified_saved=round(float(cert_saved[-1].mean()), 4),
                   uncertified_interval_mm_p50_p90=[round(float(x), 3) for x in np.percentile(width[~cert], [50, 90])]
                   if (~cert).any() else [0, 0],
                   uncertified_interval_at_window_edge=int(at_edge.sum()),
                   flat_margin_p10_p50=[round(float(x), 3) for x in np.nanpercentile(flat, [10, 50])],
                   negative_margin_traces=int(np.sum(flat < -1e-9)),
                   combined_q90=round(q90, 3), spearman_with_point_error=rho,
                   points_over_0p3=[int(p + 1) for p in np.flatnonzero(bad)],
                   points_flagged_top_decile=[int(p + 1) for p in np.flatnonzero(flagged)],
                   over_0p3_and_flagged=[int(p + 1) for p in np.flatnonzero(bad & flagged)],
                   points=points, seconds=round(time.time() - t0, 1))
    np.savez_compressed(f"{work}/edit_priority_{tag}.npz", sub=sub, combined=combined, cert=cert, lo=lo_mm,
                        hi=hi_mm, flat=flat, margins=mg, moves_mm=np.array(MOVES_MM),
                        **{"doubt_" + k: v for k, v in doubts.items()}, **{"rank_" + k: v for k, v in R_.items()})
    json.dump(summary, open(f"{work}/edit_priority_{tag}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "points"}, indent=1))
    sections(fig, tag, Z, sub, z, lo_mm, hi_mm, cert, combined, R_, c_single)
    say("done")


def sections(fig, tag, Z, sub, z, lo_mm, hi_mm, cert, combined, rk, c_single):
    """Per marked plane and bank: echo 2 with the joint pia, its interval where not certified, the
    single-surface horizon and the points; under it, the combined score and each doubt's rank."""
    import matplotlib.pyplot as plt

    import crossing_lib as X
    from volume_views import mesh_graph, planes_of, section_traces

    tv, ns, fv, pos, P, vox = Z["tv"], Z["ns"], Z["fv"], Z["pos"], Z["P"], Z["vox"]
    ax, planes = planes_of(P)
    v = Z["v"]
    G = mesh_graph(v, Z["f"], vox)
    tv_set = np.zeros(len(v), bool)
    tv_set[tv[sub]] = True
    trace_of = -np.ones(len(v), int)
    trace_of[tv] = np.arange(len(tv))
    loc = -np.ones(len(tv), int)
    loc[sub] = np.arange(sub.size)
    rows = [(k, bi, tr, d) for k in planes
            for bi, (tr, d) in enumerate(section_traces(Z, G, ax, k, P[np.abs(P[:, ax] - k) <= 2], trace_of, tv_set, vox))]
    step = X.STEP_MM
    depth = np.arange(0, 5.0 + 1e-9, step)
    vmax = np.nanpercentile(np.abs(fv[0][sub]), 97)
    names = list(rk)
    fig_, axs = plt.subplots(2 * len(rows), 1, figsize=(20, 6.5 * len(rows)), squeeze=False,
                             gridspec_kw=dict(height_ratios=[5, 1.6] * len(rows)))
    for ri, (k, bi, tr, dist) in enumerate(rows):
        a, b = axs[2 * ri, 0], axs[2 * ri + 1, 0]
        li = loc[tr]
        ok = li >= 0
        lj = np.maximum(li, 0)
        img = np.full((depth.size, tr.size), np.nan)
        for j, t in enumerate(tr):
            m_ = min(ns[t], depth.size)
            img[:m_, j] = fv[0, t, :m_]
        ext = [dist[0], dist[-1], depth[-1], 0] if tr.size > 1 else None
        a.imshow(img, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=ext, interpolation="nearest")
        unc = ok & ~cert[lj]
        a.fill_between(dist, np.where(unc, lo_mm[lj], np.nan), np.where(unc, hi_mm[lj], np.nan), color="#e34948",
                       alpha=0.45, lw=0, label="pia interval (not certified)")
        a.plot(dist, np.where(ok, c_single[lj], np.nan), "--", c="#30d5ff", lw=1.0, label="single-surface horizon")
        a.plot(dist, np.where(ok, z[-1, lj], np.nan), "-", c="#ff2bd6", lw=1.4, label="joint pia")
        Pk = P[np.abs(P[:, ax] - k) <= 2]
        for j, t in enumerate(tr):
            p = pos[t, :ns[t]].astype(float) * vox
            d_ = np.linalg.norm(p[:, None] - (Pk * vox)[None], axis=2)
            if d_.min() <= 0.3:
                a.plot(dist[j], np.unravel_index(np.argmin(d_), d_.shape)[0] * step, "*", c="#ffd23f", ms=9)
        a.set_ylim(depth[-1], 0)
        a.set_ylabel("depth (mm)")
        a.set_title(f"plane {'ijk'[ax]} = {k}, bank {bi + 1}: echo 2; magenta joint pia, red its QPBO interval "
                    f"where not certified, cyan dashed single-surface horizon, ★ your points", fontsize=10)
        if ri == 0:
            a.legend(fontsize=8, loc="lower right", ncol=3)
        R_ = np.stack([np.where(ok, combined[lj], np.nan)] + [np.where(ok, rk[nm][lj], np.nan) for nm in names])
        b.imshow(R_, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest",
                 extent=[dist[0], dist[-1], R_.shape[0] - 0.5, -0.5] if tr.size > 1 else None)
        b.set_yticks(range(R_.shape[0]), ["combined"] + names, fontsize=7)
        b.set_xlabel("distance along the white surface (mm)")
    fig_.tight_layout()
    fig_.savefig(f"{fig}/edit_priority_{tag}_sections.png", dpi=55)
    plt.close(fig_)


if __name__ == "__main__":
    sys.path.insert(0, sys.argv[2])
    main(*sys.argv[1:4])
