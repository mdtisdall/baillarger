"""The joint warp's levels on a site's column volume, subsampled (plan: reports/Crossing paths and
contacts from laminar profiles.md, *The joint warp*). Reads <work>/column_volume_<site>.npz and
<work>/pia_<site>.npz (the single-surface horizon: start, curvature trend).

1. Traces: roots >= 2 mm from a face; one per `--spacing` mm of white surface (greedy, normals
   agreeing). Neighbours within `--radius` mm, normals agreeing (dot > 0.7).
2. Shifts between neighbours (flatten_lib.shifts; echoes 2 and 3, each / its SD) and weights.
3. Anchors: every open top. Ties: each crossing trace to the subsampled trace whose root is
   nearest its far end (within `--radius`, normals facing: dot < -0.3), L = its length, weight
   = -dot. Start: the single-surface horizon c, levels at tau c.
4. joint_levels.coarse_to_fine. Scores of the pia level at the points (every subsampled trace
   passing within 0.3 mm) beside the single-surface horizon on the same traces. Sections.
Writes <work>/joint_<site>.npz and .json, <figures>/joint_<site>_sections.png.

Usage: joint_site.py <figures> <work> <site> [--spacing MM] [--radius MM]
"""

import argparse
import json
import sys
import time

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import crossing_lib as X
import flatten_lib as FL
import joint_levels as JL
from volume_views import mesh_graph, planes_of, section_traces

ap = argparse.ArgumentParser()
ap.add_argument("fig")
ap.add_argument("work")
ap.add_argument("site")
ap.add_argument("--spacing", type=float, default=0.3)
ap.add_argument("--radius", type=float, default=None)  # default 1.7 x spacing
ap.add_argument("--rounds", type=int, default=1)  # > 1: re-solve from the last result (c0 and start)
ap.add_argument("--levels", default=None)  # inner levels' tau, comma-separated (snapped to 1/96); the pia is added
ap.add_argument("--warp", default=None)  # a joint_<tag>.npz: start (levels read from its warp), c0, and depth -> tau
ap.add_argument("--bands", default=None)  # a bands_<tag>: each level's evidence (the band nearest its tau)
ap.add_argument("--w-band", type=float, default=0.5)  # weight of the band evidence (Params.w_band)
ap.add_argument("--recentre", type=int, default=20)  # cap on window re-centrings per pass
ap.add_argument("--vert", default="pia", choices=["pia", "chain"])  # the vertical prior (Params.vert)
A = ap.parse_args()
A.radius = A.radius if A.radius is not None else 1.7 * A.spacing
FIG, WORK, SITE = A.fig, A.work, A.site
TAG = ((f"{SITE}" if abs(A.spacing - 0.3) < 1e-9 else f"{SITE}_s{A.spacing:g}") + (f"_r{A.rounds}" if A.rounds > 1 else "")
       + ("_bands" if A.levels else "") + (f"_wb{A.w_band:g}" if A.levels and A.w_band != 0.5 else "") + ("_vchain" if A.vert == "chain" else ""))
LEVELS = (tuple(round(float(x) * 96) / 96 for x in A.levels.split(",")) + (1.0,)) if A.levels else JL.TAU
STEP, NDOT = X.STEP_MM, 0.7
OPEN, CROSS = 0, 1
CCOL = {0: "#1baf7a", 1: "#2a78d6", 2: "#eb6834", 3: "#c79a00", 4: "#e34948"}
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def subsample(root, nr, cand, r):
    tree = cKDTree(root[cand])
    taken = np.zeros(cand.size, bool)
    blocked = np.zeros(cand.size, bool)
    for i in np.random.default_rng(0).permutation(cand.size):
        if blocked[i]:
            continue
        taken[i] = True
        nb = np.array(tree.query_ball_point(root[cand[i]], r), int)
        blocked[nb[(nr[cand[nb]] @ nr[cand[i]]) > NDOT]] = True
    return cand[taken]


def score(c, idx, Z):
    pos, ns, P, vox = Z["pos"], Z["ns"], Z["P"], Z["vox"]
    tree = cKDTree((P * vox).astype(float))
    off, pts = [], []
    for t in idx:
        p = pos[t, :ns[t]].astype(float) * vox
        d, k = tree.query(p)
        i = int(np.argmin(d))
        if d[i] <= 0.3 and np.isfinite(c[t]):
            off.append(c[t] - i * STEP)
            pts.append(int(k[i]) + 1)
    off, pts = np.array(off), np.array(pts)
    if not off.size:
        return dict(points=0)
    u = np.unique(pts)
    per = np.array([np.median(np.abs(off[pts == k])) for k in u])
    return dict(traces=int(off.size), points=int(u.size), median=round(float(np.median(per)), 3),
                p90=round(float(np.percentile(per, 90)), 3),
                over_0p3={int(k): round(float(m), 2) for k, m in zip(u, per) if m > 0.3})


def main():
    Z = np.load(f"{WORK}/column_volume_{SITE}.npz")
    Q = np.load(f"{WORK}/pia_{SITE}.npz")
    tv, ns, fv, cls, v, nrm, vox = Z["tv"], Z["ns"], Z["fv"], Z["cls"], Z["v"], Z["nrm"], Z["vox"]
    keep = Q["keep"]
    root, nr = v[tv] * vox, nrm[tv]
    sub = subsample(root, nr, np.flatnonzero(keep), A.spacing)
    n = sub.size
    pr = cKDTree(root[sub]).query_pairs(A.radius, output_type="ndarray")
    pr = pr[(nr[sub[pr[:, 0]]] * nr[sub[pr[:, 1]]]).sum(1) > NDOT]
    say(f"{n:,} traces (of {keep.sum():,}), {len(pr):,} pairs")
    F = fv[:, sub].astype(float)
    M_ = np.isfinite(F[0]) & np.isfinite(F[1])
    for k in range(2):
        F[k] /= np.nanstd(fv[k][keep])
    F = np.where(M_[None], F, 0.0)
    S = np.zeros((len(pr), F.shape[2]), np.float32)
    W = np.zeros_like(S)
    EN, NC, VA = np.zeros_like(S), np.zeros_like(S), np.zeros(S.shape, bool)
    for a0 in range(0, len(pr), 20000):
        p = pr[a0:a0 + 20000]
        s_, ncc, en, va = FL.shifts(F[:, p[:, 0]], F[:, p[:, 1]], M_[p[:, 0]], M_[p[:, 1]], STEP)
        S[a0:a0 + 20000], NC[a0:a0 + 20000], EN[a0:a0 + 20000], VA[a0:a0 + 20000] = s_, ncc, en, va
    W[:] = FL.pair_weights(NC, EN, VA, float(np.median(EN[VA])))
    say("shifts done")
    top = np.where(cls[sub] == OPEN, Z["H_top"][sub].astype(float), np.nan)
    # ties: crossing traces to the facing subsampled root nearest their far end
    tree = cKDTree(root[sub])
    pos, far_wm = Z["pos"], Z["H_far WM"]  # load once: each npz access reads the array again
    ties, tL, tw = [], [], []
    for i in np.flatnonzero(cls[sub] == CROSS):
        t = sub[i]
        d, j = tree.query(pos[t, ns[t] - 1].astype(float) * vox)
        if d <= A.radius:
            dot = float(nr[sub[j]] @ nr[t])
            if dot < -0.3 and j != i:
                ties.append((i, j)); tL.append(float(far_wm[t])); tw.append(-dot)
    ties = np.array(ties, int).reshape(-1, 2)
    say(f"{len(ties):,} ties, {np.isfinite(top).sum():,} open anchors")
    D = JL.Data(step=STEP, ns=ns[sub], top=top, mu=Q["mu"][sub], pairs=pr, S=S, W=W,
                ties=ties, tie_L=np.array(tL), tie_w=np.array(tw))
    D.tau = LEVELS
    c0 = np.clip(Q["c"][sub], 0.5, 5.0)
    start = np.stack([t * c0 for t in LEVELS])
    if A.warp:
        Wz = np.load(f"{WORK}/joint_{A.warp}.npz")
        assert np.array_equal(Wz["sub"], sub), "the warp's traces differ"
        wtg, wzf = Wz["tg"], Wz["zf"]  # load once: each npz access reads the whole array again
        c0 = wzf[:, -1].copy()
        start = np.stack([[np.interp(t, wtg, wzf[i]) for i in range(n)] for t in LEVELS])
        D.warp_tg, D.warp_z = wtg, wzf
    if A.bands:
        Bz = np.load(f"{WORK}/bands_{A.bands}.npz")
        Bj = json.load(open(f"{WORK}/bands_{A.bands}.json"))
        assert np.array_equal(Bz["sub"], sub), "the bands' traces differ"
        D.band_tf, cost, bS = Bz["tf"], [], Bz["S"].astype(float)
        for t in LEVELS[:-1]:
            b = min(Bj["bands"], key=lambda b_: abs(b_["tau"] - t))
            ok = abs(b["tau"] - t) < 0.02
            cost.append((bS if b["kind"] == "dark" else -bS) if ok else None)
            say(f"level {t:.3f}: band {b['kind']} {b['tau']:.3f}" if ok else f"level {t:.3f}: no band")
        D.band_cost = cost
    D.c0 = c0
    rounds = []
    for rd in range(A.rounds):
        z, cert, r, hist = JL.coarse_to_fine(D, start, prm=JL.Params(w_band=A.w_band, vert=A.vert), recentre=A.recentre, log=say)
        moved = float(np.median(np.abs(z[-1] - D.c0)))
        if len(Z["P"]):
            pia_r = np.full(len(tv), np.nan)
            pia_r[sub] = z[-1]
            sc = score(pia_r, sub, Z)
        else:
            sc = {}
        rounds.append(dict(round=rd, median_pia_change_from_c0=round(moved, 3), energy=float(r.energy),
                           lower_bound=float(r.lower_bound), score=sc))
        say(f"round {rd}: pia moved {moved:.3f} mm (median) from c0; at the points {sc}")
        D.c0 = z[-1].copy()  # the next round reads weights and tapers at this pia, and starts here
        start = z.copy()
    say("levels solved")
    tg, zf, finfo = JL.fill_warp(pr, S, W, STEP, z, tau=LEVELS, J=96 if A.levels else 24, log=say)
    say("warp filled")
    pia = np.full(len(tv), np.nan)
    pia[sub] = z[-1]
    c_single = np.full(len(tv), np.nan)
    c_single[sub] = Q["c"][sub]
    summary = dict(site=SITE, traces=int(n), pairs=int(len(pr)), ties=int(len(ties)),
                   open_anchors=int(np.isfinite(top).sum()), history=hist["coarse"] + hist["fine"],
                   fine_pruned_free=int(r.n_free),
                   certified_fine_by_level=[round(float(x), 3) for x in cert.mean(1)],
                   converged=dict(coarse=bool(hist["coarse"][-1].get("converged", True)),
                                  fine=bool(hist["fine"][-1].get("converged", True))),
                   fill=finfo, rounds=rounds, seconds=round(time.time() - T0, 1))
    if len(Z["P"]):
        summary["at the points: joint pia"] = score(pia, sub, Z)
        summary["at the points: single-surface horizon, same traces"] = score(c_single, sub, Z)
    fh = hist["fine"]
    summary["stria"] = stria_stats(fv, ns, cls, sub, tg, zf)
    np.savez_compressed(f"{WORK}/joint_{TAG}.npz", sub=sub, z=z, cert=cert, pairs=pr, ties=ties, tg=tg, zf=zf,
                        coarse=hist["coarse_depths"], hits=np.stack([h["hits"] for h in fh]),
                        low=np.stack([h["low"] for h in fh]), zs=np.stack([h["z"] for h in fh]),
                        top=top, c0=c0)
    for h_ in hist["coarse"] + hist["fine"]:
        for k_ in ("hits", "low", "centre", "z"):
            h_.pop(k_, None)
    json.dump(summary, open(f"{WORK}/joint_{TAG}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v_ for k, v_ in summary.items() if k != "history"}, indent=1, default=float))
    figures(Z, sub, z, cert, c_single)
    flat_views(Z, sub, tg, zf)


def figures(Z, sub, z, cert, c_single):
    tv, ns, fv, cls, pos, P, vox = Z["tv"], Z["ns"], Z["fv"], Z["cls"], Z["pos"], Z["P"], Z["vox"]
    if not len(P):
        return
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
    depth = np.arange(0, 6.0 + 1e-9, STEP)
    vmax = np.nanpercentile(np.abs(fv[0][sub]), 97)
    fig, axs = plt.subplots(len(rows), 1, figsize=(20, 6 * len(rows)), squeeze=False)
    for ri, (k, bi, tr, dist) in enumerate(rows):
        a = axs[ri, 0]
        Pk = P[np.abs(P[:, ax] - k) <= 2]
        img = np.full((depth.size, tr.size), np.nan)
        for j, t in enumerate(tr):
            m_ = min(ns[t], depth.size)
            img[:m_, j] = fv[0, t, :m_]
        ext = [dist[0], dist[-1], depth[-1], 0] if tr.size > 1 else None
        a.imshow(img, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=ext, interpolation="nearest")
        li = loc[tr]
        a.plot(dist, Z["H_top"][tr], "-", c="w", lw=0.8, label="open top")
        a.plot(dist, c_single[tr], "--", c="#30d5ff", lw=1.0, label="single-surface horizon")
        cols = ["#ffd23f", "#eb6834", "#1baf7a", "#30d5ff"][: z.shape[0] - 1] + ["#ff2bd6"]
        for l, col in zip(range(z.shape[0]), cols):
            a.plot(dist, np.where(li >= 0, z[l, np.maximum(li, 0)], np.nan), "-", c=col, lw=1.4 if l == z.shape[0] - 1 else 0.8,
                   label=f"joint level tau={LEVELS[l]:.2f}")
        unc = (li >= 0) & ~cert[-1, np.maximum(li, 0)]
        a.plot(dist[unc], np.full(unc.sum(), 0.15), "v", c="#e34948", ms=5, label="pia not certified")
        for j, t in enumerate(tr):
            p = pos[t, :ns[t]].astype(float) * vox
            d_ = np.linalg.norm(p[:, None] - (Pk * vox)[None], axis=2)
            if d_.min() <= 0.3:
                a.plot(dist[j], np.unravel_index(np.argmin(d_), d_.shape)[0] * STEP, "*", c="#ffd23f", ms=9)
        for j, t in enumerate(tr):
            a.plot(dist[j], depth[-1] - 0.08, "s", ms=4, c=CCOL[int(cls[t])])
        a.set_ylim(depth[-1], 0)
        a.set_ylabel("depth from the root (mm)")
        a.set_title(f"plane {'ijk'[ax]} = {k}, bank {bi + 1}: echo 2 on the subsampled traces; magenta: joint pia; "
                    f"cyan dashed: single-surface horizon; ★ your points", fontsize=10)
        if ri == 0:
            a.legend(fontsize=8, loc="lower right", ncol=4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/joint_{TAG}_sections.png", dpi=60)
    plt.close(fig)


def resample_tau(fv, ns, sub, tg, zf, nt=97):
    """Echo 2 of the subsampled traces at constant tau (nt levels in [0, 1]); nan past the trace."""
    tf = np.linspace(0, 1, nt)
    zz = np.stack([np.interp(tf, tg, zf[i]) for i in range(sub.size)])
    G = FL.resample(fv[:1, sub].astype(float), ns[sub], zz, STEP)[0]
    return tf, G


def stria_stats(fv, ns, cls, sub, tg, zf):
    """The darkest interior band of each open or crossing trace (smoothed echo 2), in raw depth
    and in tau; interior: depth 0.4 mm to 0.4 mm short of the pia, minimum not on the window edge."""
    from scipy import ndimage as ndi

    out = dict(depth=[], tau=[], pia=[])
    for i, t in enumerate(sub):
        if cls[t] not in (0, 1):
            continue
        e = fv[0, t, :ns[t]].astype(float)
        if not np.isfinite(e).all() or e.size < 20:
            continue
        e = ndi.gaussian_filter1d(e, 2)
        d = np.arange(e.size) * STEP
        pia = zf[i, -1]
        w = (d >= 0.4) & (d <= pia - 0.4)
        if w.sum() < 5:
            continue
        j = np.flatnonzero(w)
        k = j[np.argmin(e[j])]
        if k in (j[0], j[-1]):
            continue
        out["depth"].append(d[k]); out["tau"].append(float(np.interp(d[k], zf[i], tg))); out["pia"].append(pia)
    dd, tt, pp = (np.array(out[k]) for k in ("depth", "tau", "pia"))
    iqr = lambda x: float(np.subtract(*np.percentile(x, [75, 25])))  # noqa: E731
    return dict(traces=int(dd.size), depth_p25_50_75=np.percentile(dd, [25, 50, 75]).round(3).tolist(),
                tau_p25_50_75=np.percentile(tt, [25, 50, 75]).round(3).tolist(), iqr_depth_mm=round(iqr(dd), 3),
                iqr_tau=round(iqr(tt), 3), iqr_tau_in_mm_at_median_pia=round(iqr(tt) * float(np.median(pp)), 3))


def flat_views(Z, sub, tg, zf):
    """Sections in tau beside raw depth, and stacked profiles, from the filled warp."""
    tv, ns, fv, cls, P, vox = Z["tv"], Z["ns"], Z["fv"], Z["cls"], Z["P"], Z["vox"]
    tf, G = resample_tau(fv, ns, sub, tg, zf)
    vmax = np.nanpercentile(np.abs(fv[0][sub]), 97)
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    depth = np.arange(fv.shape[2]) * STEP
    for c, col, lab in ((0, CCOL[0], "open"), (1, CCOL[1], "crossing")):
        m = cls[sub] == c
        raw = fv[0, sub[m]].astype(float)
        for a, (x, Y, xl) in zip(axs, ((depth, raw, "raw depth (mm)"), (tf, G[m], "tau (0 WM, 1 pia)"))):
            cnt = np.sum(np.isfinite(Y), 0)
            q = np.full((3, Y.shape[1]), np.nan)
            ok = cnt >= 30
            q[:, ok] = np.nanpercentile(Y[:, ok], [25, 50, 75], axis=0)
            a.fill_between(x, q[0], q[2], color=col, alpha=0.2)
            a.plot(x, q[1], c=col, label=f"{lab} (median, IQR)")
            a.set_xlabel(xl)
    axs[0].set_xlim(0, 4)
    for a in axs:
        a.legend()
        a.set_ylabel("echo 2, trend removed")
    axs[1].set_title("stacked at constant tau (joint warp; nothing past the pia)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/joint_{TAG}_profiles.png", dpi=72)
    plt.close(fig)
    if not len(P):
        return
    ax, planes = planes_of(P)
    v = Z["v"]
    Gm = mesh_graph(v, Z["f"], vox)
    tv_set = np.zeros(len(v), bool)
    tv_set[tv[sub]] = True
    trace_of = -np.ones(len(v), int)
    trace_of[tv] = np.arange(len(tv))
    loc = -np.ones(len(tv), int)
    loc[sub] = np.arange(sub.size)
    rows = [(k, bi, tr, d) for k in planes
            for bi, (tr, d) in enumerate(section_traces(Z, Gm, ax, k, P[np.abs(P[:, ax] - k) <= 2], trace_of, tv_set, vox))]
    fig, axs = plt.subplots(len(rows), 1, figsize=(20, 4 * len(rows)), squeeze=False)
    for ri, (k, bi, tr, dist) in enumerate(rows):
        a = axs[ri, 0]
        li = loc[tr]
        img = np.full((tf.size, tr.size), np.nan)
        for j, l_ in enumerate(li):
            if l_ >= 0:
                img[:, j] = G[l_]
        a.imshow(img, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax,
                 extent=[dist[0], dist[-1], 1, 0] if tr.size > 1 else None, interpolation="nearest")
        for tt in LEVELS[:-1]:
            a.axhline(tt, c="#ffd23f", lw=0.6, ls=":")
        a.set_ylabel("tau")
        a.set_title(f"plane {'ijk'[ax]} = {k}, bank {bi + 1}: echo 2 at constant tau (0 WM, 1 pia); dotted: levels",
                    fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/joint_{TAG}_tau_sections.png", dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    main()
