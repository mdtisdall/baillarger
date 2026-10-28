"""The first-pia horizon of a site's column volume (plan: reports/Crossing paths and contacts from
laminar profiles.md, *The first-pia horizon*). Reads <work>/column_volume_<site>.npz.

1. Every trace whose root is >= 2 mm from a box face is an unknown, on depth labels 0.4-5.0 mm
   at 0.04 mm. Open traces: soft anchors (leaving the top costs more than any trace's data range).
2. Curvature trend log t = m + a H on the open tops (H: white-mesh mean curvature at the root,
   0.8 mm); mu = exp(m + a H).
3. g: the slope |d(top - mu)| / d over pairs of open roots 0.8-1.2 mm apart (normals agreeing),
   99th percentile; slopes at 0.2, 0.5 and 2 mm reported beside it.
4. Data cost: echoes 2 and 3 (trend removed, each / its SD), averaged and smoothed along the trace
   at 0.08 mm (darker is cheaper; 0 past the end). Hard bounds between every pair of neighbours.
5. pia_lib.min_surface (stage 1); the trend and g measured again on the open tops it kept;
   pia_lib.min_surface again (stage 2). Scores at the points as column_volume.py; views of stage 2.
Writes <work>/pia_<site>.npz and .json, <figures>/pia_<site>_{sections,slopes}.png.

Usage: pia_horizon.py <figures> <work> <site> [--g-pct P]
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
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import crossing_lib as X
import pia_lib as PL
from baillarger.frame import mesh_curvatures
from volume_views import mesh_graph, planes_of, section_traces

ap = argparse.ArgumentParser()
ap.add_argument("fig")
ap.add_argument("work")
ap.add_argument("site")
ap.add_argument("--g-pct", type=float, default=99.0)
A = ap.parse_args()
FIG, WORK, SITE = A.fig, A.work, A.site
STEP, D0, D1 = X.STEP_MM, 0.4, 5.0
PAIR_MM, NDOT, H_SCALE, SMOOTH_MM = 0.25, 0.7, 0.8, 0.08
G_BAND = (0.8, 1.2)
BANDS = [(0.15, 0.25), (0.4, 0.6), (0.8, 1.2), (1.6, 2.4)]
OPEN, CROSS, OWN, TOCONTACT, FAILED = range(5)
CNAME = {OPEN: "open", CROSS: "crossing", OWN: "own bank", TOCONTACT: "to the contact", FAILED: "failed"}
CCOL = {OPEN: "#1baf7a", CROSS: "#2a78d6", OWN: "#eb6834", TOCONTACT: "#c79a00", FAILED: "#e34948"}
HOR = ["top", "far WM", "profile, top"]
HCOL = {"top": "#ffffff", "far WM": "#30d5ff", "profile, top": "#2a78d6"}
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def slopes(root, nr, r, band):
    """|r_v - r_w| / d over pairs of roots lo < d <= hi with normals agreeing."""
    lo, hi = band
    p = cKDTree(root).query_pairs(hi, output_type="ndarray")
    d = np.linalg.norm(root[p[:, 0]] - root[p[:, 1]], axis=1)
    ok = (d > lo) & ((nr[p[:, 0]] * nr[p[:, 1]]).sum(1) > NDOT)
    return np.abs(r[p[ok, 0]] - r[p[ok, 1]]) / d[ok]


def score(c, ic, Z):
    """As column_volume.figures: traces in ic passing within 0.3 mm of a point; per point the median
    |c - the point's depth| over its traces; then the median and 90th percentile over points."""
    pos, ns, P, vox = Z["pos"], Z["ns"], Z["P"], Z["vox"]
    tree = cKDTree((P * vox).astype(float))
    off, pts = [], []
    for t in ic:
        p = pos[t, :ns[t]].astype(float) * vox
        d, k = tree.query(p)
        i = int(np.argmin(d))
        if d[i] <= 0.3 and np.isfinite(c[t]):
            off.append(c[t] - i * STEP)
            pts.append(int(k[i]) + 1)
    off, pts = np.array(off), np.array(pts)
    if not off.size:
        return dict(traces=0, points=0)
    per = np.array([np.median(np.abs(off[pts == k])) for k in np.unique(pts)])
    sgn = np.array([np.median(off[pts == k]) for k in np.unique(pts)])
    return dict(traces=int(off.size), points=int(per.size),
                abs_median_per_point_median=round(float(np.median(per)), 3),
                abs_median_per_point_p90=round(float(np.percentile(per, 90)), 3),
                signed_median_per_point_median=round(float(np.median(sgn)), 3),
                per_point={int(k): round(float(m), 3) for k, m in zip(np.unique(pts), per)})


def trend_and_g(H, top, root, nr, idx):
    """log t = m + a H on the tops of traces idx; g and the slopes by band on their detrended tops."""
    Xd = np.c_[np.ones(idx.size), H[idx]]
    (m, a), *_ = np.linalg.lstsq(Xd, np.log(top[idx]), rcond=None)
    mu = np.exp(m + a * H)
    r = top[idx] - mu[idx]
    sl = {f"{lo}-{hi} mm": slopes(root[idx], nr[idx], r, (lo, hi)) for lo, hi in BANDS}
    g = float(np.percentile(sl[f"{G_BAND[0]}-{G_BAND[1]} mm"], A.g_pct))
    info = dict(m=round(float(m), 4), a=round(float(a), 4), tops=int(idx.size), g=round(g, 4),
                top_p5_50_95=np.percentile(top[idx], [5, 50, 95]).round(3).tolist(),
                slopes_p50_95_99={k: dict(pairs=int(s.size), pcts=np.percentile(s, [50, 95, 99]).round(3).tolist())
                                  for k, s in sl.items()})
    return mu, g, sl, info


def run():
    Z = np.load(f"{WORK}/column_volume_{SITE}.npz")
    tv, ns, fv, cls, near_face = Z["tv"], Z["ns"], Z["fv"], Z["cls"], Z["near_face"]
    v, f, nrm, vox = Z["v"], Z["f"], Z["nrm"], Z["vox"]
    top = Z["H_top"].astype(float)
    keep = ~near_face & (ns > 1)
    k1, k2 = mesh_curvatures(v * vox, f, nrm, H_SCALE)
    H = 0.5 * (k1 + k2)[tv]
    root, nr = v[tv] * vox, nrm[tv]
    is_open = keep & (cls == OPEN) & np.isfinite(top)
    iu = np.flatnonzero(keep)
    loc = -np.ones(len(tv), int)
    loc[iu] = np.arange(iu.size)
    depths = D0 + STEP * np.arange(int(round((D1 - D0) / STEP)) + 1)
    K = depths.size
    summary = dict(site=SITE, traces=int(iu.size), open_traces=int(is_open.sum()), labels=K)

    # ---- data cost, every trace
    F = fv[:, iu].astype(float)
    Mk = np.isfinite(F).all(0)
    for k in range(2):
        F[k] /= np.nanstd(fv[k][keep][np.isfinite(fv[k][keep])])
    E = np.where(Mk, F.mean(0), 0.0)
    del F
    sig = SMOOTH_MM / STEP
    num = ndi.gaussian_filter1d(E, sig, axis=1, mode="constant")
    den = ndi.gaussian_filter1d(Mk.astype(float), sig, axis=1, mode="constant")
    Es = np.where(Mk & (den > 0.5), num / np.maximum(den, 1e-6), 0.0)
    j0 = int(round(D0 / STEP))
    D = Es[:, j0:j0 + K].copy()
    if D.shape[1] < K:
        D = np.pad(D, ((0, 0), (0, K - D.shape[1])))
    del E, Es, num, den
    # ---- soft anchors: leaving an open top costs more than any trace's whole range of data cost
    Pc = 1.0 + float(np.max(D.max(1) - D.min(1)))
    io = np.flatnonzero(is_open)
    at_top = np.abs(depths[None] - top[io, None]) <= STEP / 2 + 1e-9
    pref = np.zeros_like(D)
    pref[loc[io]] = np.where(at_top, 0.0, Pc)
    summary["open preference P"] = round(Pc, 3)
    summary["open tops beyond the label range (always overridden)"] = int(np.sum(~at_top.any(1)))

    # ---- the graph: every pair of neighbours
    pr = cKDTree(root[iu]).query_pairs(PAIR_MM, output_type="ndarray")
    pr = iu[pr]
    pr = pr[(nr[pr[:, 0]] * nr[pr[:, 1]]).sum(1) > NDOT]
    dist = np.linalg.norm(root[pr[:, 0]] - root[pr[:, 1]], axis=1)
    say(f"{iu.size:,} traces x {K} labels; {len(pr):,} pairs; P = {Pc:.2f}")

    stages, c_st = [], []
    idx = io
    for st in (1, 2):
        mu, g, sl, info = trend_and_g(H, top, root, nr, idx)
        lo, hi = PL.bound_labels(mu[pr[:, 1]] - mu[pr[:, 0]], g * dist, STEP)
        say(f"stage {st}: trend m {info['m']} a {info['a']:+}; g {g:.3f} mm/mm from {idx.size:,} open tops")
        cl = PL.min_surface(D + pref, loc[pr], lo, hi)
        c = np.full(len(tv), np.nan)
        c[iu] = depths[cl]
        kept = np.abs(c[io] - top[io]) <= STEP / 2 + 1e-9
        long_ = top[io] > 3.5
        info.update({
            "open tops kept": int(kept.sum()), "open tops overridden": int((~kept).sum()),
            "overridden, of tops over 3.5 mm (diagnostic split only)": round(float(np.mean(~kept[long_])), 3),
            "overridden, of tops up to 3.5 mm": round(float(np.mean(~kept[~long_])), 3),
            "overridden: c minus top, p5/50/95 (mm)": np.percentile((c[io] - top[io])[~kept], [5, 50, 95]).round(3).tolist()
            if (~kept).any() else None,
            "c by class, p5/50/95 (mm)": {CNAME[k]: np.percentile(c[keep & (cls == k)], [5, 50, 95]).round(3).tolist()
                                         for k in CNAME if np.any(keep & (cls == k))},
            "c at the lowest label (fraction)": round(float(np.mean(cl == 0)), 4)})
        say(f"stage {st}: min cut done; open tops overridden {int((~kept).sum()):,}")
        stages.append(info)
        c_st.append(c)
        idx = io[kept]  # consistent open tops: the trend and g are measured again on them
        last = (mu, g, sl)
    summary["stages"] = stages
    c = c_st[-1]
    mu, g, sl = last
    np.savez_compressed(f"{WORK}/pia_{SITE}.npz", c=c, c_stage1=c_st[0], mu=mu, H=H, keep=keep, is_open=is_open,
                        g=g, depths=depths)

    # ---- scores
    if len(Z["P"]):
        ic = np.flatnonzero(keep & np.isin(cls, [CROSS, TOCONTACT]))
        sc = {}
        for st, cc in zip((1, 2), c_st):
            # an open trace whose top was overridden crossed a contact: score it too
            moved = is_open & (np.abs(cc - top) > STEP / 2 + 1e-9)
            sc[f"first pia, stage {st} (crossing, to the contact)"] = score(cc, ic, Z)
            sc[f"first pia, stage {st} (crossing, to the contact, overridden open)"] = score(
                cc, np.flatnonzero(keep & (np.isin(cls, [CROSS, TOCONTACT]) | moved)), Z)
            sc[f"first pia, stage {st} (every trace)"] = score(cc, np.flatnonzero(keep), Z)
        for h in ["profile, top"]:
            sc[h + " (crossing, to the contact)"] = score(Z["H_" + h].astype(float), ic, Z)
        summary["at the points (|horizon - point depth|, mm)"] = sc
    json.dump(summary, open(f"{WORK}/pia_{SITE}.json", "w"), indent=1)
    printable = {k: v_ for k, v_ in summary.items() if k != "at the points (|horizon - point depth|, mm)"}
    print(json.dumps(printable, indent=1))
    if "at the points (|horizon - point depth|, mm)" in summary:
        for k, s in summary["at the points (|horizon - point depth|, mm)"].items():
            s = {kk: vv for kk, vv in s.items() if kk != "per_point"}
            print(f"{k}: {s}")
    figures(Z, c, sl, g)


def figures(Z, c, sl, g):
    tv, ns, fv, cls, pos, P, vox = Z["tv"], Z["ns"], Z["fv"], Z["cls"], Z["pos"], Z["P"], Z["vox"]
    keep = ~Z["near_face"] & (ns > 1)
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    for k, s in sl.items():
        axs.hist(s, np.linspace(0, 2, 81), histtype="step", density=True, label=f"{k} (p99 {np.percentile(s, 99):.2f})")
    axs.axvline(g, c="k", ls="--", label=f"g = {g:.3f}")
    axs.set_xlabel("|change in detrended open top| / distance (mm per mm)")
    axs.set_title(f"{SITE}: slopes of the open top by root distance")
    axs.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/pia_{SITE}_slopes.png", dpi=72)
    plt.close(fig)
    if not len(P):
        return
    ax, planes = planes_of(P)
    v = Z["v"]
    G = mesh_graph(v, Z["f"], vox)
    tv_set = np.zeros(len(v), bool)
    tv_set[tv] = True
    trace_of = -np.ones(len(v), int)
    trace_of[tv] = np.arange(len(tv))
    rows = [(k, bi, tr, d) for k in planes
            for bi, (tr, d) in enumerate(section_traces(Z, G, ax, k, P[np.abs(P[:, ax] - k) <= 2], trace_of, tv_set, vox))]
    depth = np.arange(0, 6.0 + 1e-9, STEP)
    vmax = np.nanpercentile(np.abs(fv[0][keep]), 97)
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
        for h in HOR:
            a.plot(dist, Z["H_" + h][tr], "." if h == "profile, top" else "-", c=HCOL[h], ms=4, lw=1, label=h)
        cc = np.where(keep[tr], c[tr], np.nan)
        a.plot(dist, cc, "-", c="#ff2bd6", lw=1.6, label="first pia (this tracker)")
        for j, t in enumerate(tr):
            p = pos[t, :ns[t]].astype(float) * vox
            d_ = np.linalg.norm(p[:, None] - (Pk * vox)[None], axis=2)
            if d_.min() <= 0.3:
                a.plot(dist[j], np.unravel_index(np.argmin(d_), d_.shape)[0] * STEP, "*", c="#ffd23f", ms=9)
        for j, t in enumerate(tr):
            a.plot(dist[j], depth[-1] - 0.08, "s", ms=4, c=CCOL[int(cls[t])])
        a.set_ylim(depth[-1], 0)
        a.set_ylabel("depth from the root (mm)")
        a.set_xlabel("position along the white surface (mm)")
        a.set_title(f"plane {'ijk'[ax]} = {k}, bank {bi + 1}: echo 2; magenta: first pia; ★ your points; "
                    f"strip: class (green open, blue crossing, orange own bank, gold to contact, red failed)", fontsize=10)
        if ri == 0:
            a.legend(fontsize=8, loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/pia_{SITE}_sections.png", dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    run()
