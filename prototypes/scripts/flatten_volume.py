"""Flattening in the column volume of a site (plan: reports/Crossing paths and contacts from laminar
profiles.md, *Flattening in the volume*). Reads <work>/column_volume_<site>.npz (column_volume.py).

1. Traces: every trace whose root is >= 2 mm from a box face (any class; a failed trace keeps its
   samples up to where it stopped). Features: echo 2 and echo 3 (trend removed), each divided by its
   SD over the traces' samples.
2. Pairs: roots within 0.25 mm whose white-mesh normals agree (dot > 0.7).
3. Shifts per pair (flatten_lib.shifts, or flatten_lib.dynamic_shifts with --shifts dyn) and
   weights; the fit (flatten_lib.fit_warp).
4. Saves <work>/flatten_<tag>.npz (tag: the site, plus _dyn for dynamic
   shifts and _s<smax> for a bound other than 0.3 mm) (horizons z, tau, residuals, folds) and .json; views in <figures>:
   sections raw beside flattened, stacked profiles and the darkest band's depth, the residual and
   the Wheeler slice on the roots.

Usage: flatten_volume.py <figures> <work> <site> [--eps E] [--shifts lk|dyn] [--smax MM] [--figs-only]
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
import flatten_lib as FL
from volume_views import mesh_graph, planes_of, section_traces

ap = argparse.ArgumentParser()
ap.add_argument("fig")
ap.add_argument("work")
ap.add_argument("site")
ap.add_argument("--eps", type=float, default=0.2)
ap.add_argument("--shifts", choices=["lk", "dyn"], default="lk")
ap.add_argument("--smax", type=float, default=0.3, help="largest shift between neighbours (mm)")
ap.add_argument("--figs-only", action="store_true")
A = ap.parse_args()
FIG, WORK, SITE = A.fig, A.work, A.site
TAG = SITE + ("_dyn" if A.shifts == "dyn" else "") + (f"_s{A.smax:g}" if A.smax != 0.3 else "")
STEP, DTAU, TAU_MAX, PAIR_MM, NDOT, CHUNK = X.STEP_MM, 0.08, 6.0, 0.25, 0.7, 20000
OPEN, CROSS, OWN, TOCONTACT, FAILED = range(5)
CCOL = {OPEN: "#1baf7a", CROSS: "#2a78d6", OWN: "#eb6834", TOCONTACT: "#c79a00", FAILED: "#e34948"}
HOR = ["top", "far WM", "profile, top"]
HCOL = {"top": "#ffffff", "far WM": "#30d5ff", "profile, top": "#2a78d6"}
DARK_WIN = (0.5, 1.5)  # mm: where the darkest band is looked for (the stria at V1)
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def fit():
    Z = np.load(f"{WORK}/column_volume_{SITE}.npz")
    tv, ns, fv, near_face, v, nrm, vox = Z["tv"], Z["ns"], Z["fv"], Z["near_face"], Z["v"], Z["nrm"], Z["vox"]
    keep = np.flatnonzero(~near_face & (ns > 1))
    N = keep.size
    F = fv[:, keep].astype(np.float64)
    M = np.isfinite(F[0]) & np.isfinite(F[1])
    for k in range(F.shape[0]):
        F[k] /= np.nanstd(F[k][M])
    F = np.where(M[None], F, 0.0)
    nsk = ns[keep]
    root = v[tv[keep]] * vox
    nr = nrm[tv[keep]]
    pairs = np.array(sorted(cKDTree(root).query_pairs(PAIR_MM)))
    pairs = pairs[(nr[pairs[:, 0]] * nr[pairs[:, 1]]).sum(1) > NDOT]
    say(f"{N:,} traces, {len(pairs):,} pairs ({2 * len(pairs) / N:.1f} per trace)")
    P_, n = len(pairs), F.shape[2]
    S = np.zeros((P_, n), np.float32)
    NC = np.zeros((P_, n), np.float32)
    EN = np.zeros((P_, n), np.float32)
    VA = np.zeros((P_, n), bool)
    for a in range(0, P_, CHUNK):
        p = pairs[a:a + CHUNK]
        est = FL.dynamic_shifts if A.shifts == "dyn" else FL.shifts
        s, ncc, en, va = est(F[:, p[:, 0]], F[:, p[:, 1]], M[p[:, 0]], M[p[:, 1]], STEP, smax=A.smax)
        S[a:a + CHUNK], NC[a:a + CHUNK], EN[a:a + CHUNK], VA[a:a + CHUNK] = s, ncc, en, va
    eref = float(np.median(EN[VA]))
    W = FL.pair_weights(NC, EN, VA, eref).astype(np.float32)
    say(f"shifts done; energy ref {eref:.4g}; shift |s| p50/p95 over weighted samples "
        f"{np.percentile(np.abs(S[W > 0.3]), [50, 95]).round(3).tolist()} mm")
    del NC, EN
    tau, r, info = FL.fit_warp(pairs, S, W, N, STEP, dtau=DTAU, tau_max=TAU_MAX, eps=A.eps, log=say)
    z = tau[None] + r
    wt, mis = FL.residuals(pairs, S, W, r, tau, STEP)
    # per trace and level: weighted rms of its pairs' misfits
    num = np.zeros((N, tau.size))
    den = np.zeros((N, tau.size))
    for c in (0, 1):
        np.add.at(num, pairs[:, c], wt * mis ** 2)
        np.add.at(den, pairs[:, c], wt)
    res = np.sqrt(num / np.maximum(den, 1e-9))
    res[den < 0.05] = np.nan
    fold = np.diff(z, axis=1) <= 0
    np.savez_compressed(f"{WORK}/flatten_{TAG}.npz", keep=keep, tau=tau, z=z.astype(np.float32),
                        res=res.astype(np.float32), wsum=den.astype(np.float32), fold=fold, pairs=pairs, eps=A.eps, S=S, W=W)
    json.dump(dict(site=SITE, shifts=A.shifts, smax=A.smax, traces=int(N), pairs=int(P_), eps=A.eps,
                   energy_ref=eref, outer=info),
              open(f"{WORK}/flatten_{TAG}_fit.json", "w"), indent=1)
    say("saved")


# ---------------------------------------------------------------- views


def figures():
    Z = np.load(f"{WORK}/column_volume_{SITE}.npz")
    Fz = np.load(f"{WORK}/flatten_{TAG}.npz")
    keep, tau, z, res, fold = Fz["keep"], Fz["tau"], Fz["z"].astype(float), Fz["res"], Fz["fold"]
    tv, ns, fv, cls, pos, P, vox = Z["tv"], Z["ns"], Z["fv"], Z["cls"], Z["pos"], Z["P"], Z["vox"]
    H = {h: Z["H_" + h] for h in HOR}
    N = keep.size
    loc = -np.ones(len(tv), int)
    loc[keep] = np.arange(N)
    # flattened features on a 0.04 mm tau grid
    tf = np.arange(0, TAU_MAX + 1e-9, STEP)
    zf = np.stack([np.interp(tf, tau, z[i]) for i in range(N)])
    E2 = fv[0, keep].astype(float)
    G2 = FL.resample(E2[None], ns[keep], zf, STEP)[0]
    vmax = np.nanpercentile(np.abs(E2), 97)
    summary = dict(site=SITE, shifts=A.shifts, smax=A.smax, traces=int(N), eps=float(Fz["eps"]))
    summary["fit"] = json.load(open(f"{WORK}/flatten_{TAG}_fit.json"))["outer"]
    summary["fold fraction (levels with dz/dtau <= 0, below each trace's end)"] = float(
        np.mean(fold[z[:, 1:] < (ns[keep, None] - 1) * STEP]))
    rr = res[np.isfinite(res)]
    summary["residual per trace and level, p50/p90/p99 (mm)"] = np.percentile(rr, [50, 90, 99]).round(4).tolist()

    # ---- stacked profiles and the darkest band, open and crossing traces
    oc = np.isin(cls[keep], [OPEN, CROSS])
    depth = np.arange(E2.shape[1]) * STEP
    sm = ndi.gaussian_filter1d(np.nan_to_num(E2), 2, axis=1)
    sm[~np.isfinite(E2)] = np.nan
    w_ = (depth >= DARK_WIN[0]) & (depth <= DARK_WIN[1])
    cand = np.where(w_[None] & np.isfinite(sm), sm, np.inf)
    im = np.argmin(cand, 1)
    w_idx = np.flatnonzero(w_)
    # an interior minimum only: a minimum on the window's edge is the slope of something outside it
    ok = (oc & np.isfinite(cand).any(1) & (ns[keep] * STEP > DARK_WIN[1] + 0.3)
          & (im > w_idx[0]) & (im < w_idx[-1]))
    dmin = np.where(ok, depth[im], np.nan)
    tmin = FL.tau_of(dmin, zf, tf)
    iqr = lambda x: float(np.subtract(*np.nanpercentile(x, [75, 25])))  # noqa: E731
    summary["darkest band in 0.5-1.5 mm (open and crossing traces)"] = dict(
        traces=int(ok.sum()),
        traces_with_minimum_on_window_edge=int(np.sum(oc & (ns[keep] * STEP > DARK_WIN[1] + 0.3)) - ok.sum()),
        depth_mm_p25_50_75=np.nanpercentile(dmin, [25, 50, 75]).round(3).tolist(),
        tau_p25_50_75=np.nanpercentile(tmin, [25, 50, 75]).round(3).tolist(),
        iqr_depth=round(iqr(dmin), 3), iqr_tau=round(iqr(tmin), 3))
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    for a, (Y, lab, x_) in zip(axs[:2], ((E2, "raw: depth from the root (mm)", depth), (G2, "flattened: tau (mm)", tf))):
        for c, col in ((OPEN, CCOL[OPEN]), (CROSS, CCOL[CROSS])):
            m = cls[keep] == c
            cnt = np.sum(np.isfinite(Y[m]), 0)
            q = np.full((3, Y.shape[1]), np.nan)
            okc = cnt >= 50
            q[:, okc] = np.nanpercentile(Y[m][:, okc], [25, 50, 75], axis=0)
            a.fill_between(x_, q[0], q[2], color=col, alpha=0.2)
            a.plot(x_, q[1], c=col, label=f"{'open' if c == OPEN else 'crossing'} (median, IQR)")
        a.set_xlabel(lab)
        a.set_ylabel("echo 2, trend removed")
        a.set_xlim(0, 6)
        a.legend()
        a.set_title(f"{SITE}: stacked echo 2, {lab.split(':')[0]}")
    a = axs[2]
    a.hist(dmin[ok], np.arange(*DARK_WIN, 0.04) , alpha=0.5, label=f"raw depth (IQR {iqr(dmin):.3f} mm)")
    a.hist(tmin[ok], np.arange(*DARK_WIN, 0.04), alpha=0.5, label=f"tau (IQR {iqr(tmin):.3f} mm)")
    a.set_xlabel("mm")
    a.set_title("darkest point in 0.5-1.5 mm on each open or crossing trace")
    a.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG}/flatten_{TAG}_profiles.png", dpi=72)
    plt.close(fig)

    # ---- sections, raw beside flattened
    if len(P):
        ax, planes = planes_of(P)
        v = Z["v"]
        G = mesh_graph(v, Z["f"], vox)
        tv_set = np.zeros(len(v), bool)
        tv_set[tv] = True
        trace_of = -np.ones(len(v), int)
        trace_of[tv] = np.arange(len(tv))
        rows = [(k, bi, tr, d) for k in planes
                for bi, (tr, d) in enumerate(section_traces(Z, G, ax, k, P[np.abs(P[:, ax] - k) <= 2], trace_of, tv_set, vox))]
        fig, axs = plt.subplots(len(rows), 2, figsize=(26, 5.5 * len(rows)), squeeze=False)
        for ri, (k, bi, tr, dist) in enumerate(rows):
            Pk = P[np.abs(P[:, ax] - k) <= 2]
            li = loc[tr]
            for ci in (0, 1):
                a = axs[ri, ci]
                img = np.full((tf.size, tr.size), np.nan)
                for j, (t, l_) in enumerate(zip(tr, li)):
                    if l_ < 0:
                        continue
                    img[:, j] = (E2[l_, :tf.size] if ci == 0 else G2[l_])
                ext = [dist[0], dist[-1], tf[-1], 0] if tr.size > 1 else None
                a.imshow(img, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=ext, interpolation="nearest")
                for h in HOR:
                    y = H[h][tr].astype(float)
                    if ci == 1:
                        y = np.array([FL.tau_of(np.array([yy]), zf[l_:l_ + 1], tf)[0] if l_ >= 0 else np.nan
                                      for yy, l_ in zip(y, li)])
                    a.plot(dist, y, "." if h == "profile, top" else "-", c=HCOL[h], ms=4, lw=1, label=h)
                for j, (t, l_) in enumerate(zip(tr, li)):
                    p = pos[t, :ns[t]].astype(float) * vox
                    d_ = np.linalg.norm(p[:, None] - (Pk * vox)[None], axis=2)
                    if d_.min() <= 0.3:
                        dp = np.unravel_index(np.argmin(d_), d_.shape)[0] * STEP
                        if ci == 1:
                            dp = FL.tau_of(np.array([dp]), zf[l_:l_ + 1], tf)[0] if l_ >= 0 else np.nan
                        a.plot(dist[j], dp, "*", c="#ffd23f", ms=8)
                for j, t in enumerate(tr):
                    a.plot(dist[j], tf[-1] - 0.08, "s", ms=4, c=CCOL[int(cls[t])])
                a.set_ylim(tf[-1], 0)
                a.set_ylabel("depth from the root (mm)" if ci == 0 else "tau (mm)")
                a.set_xlabel("position along the white surface (mm)")
                a.set_title(f"plane {'ijk'[ax]} = {k}, bank {bi + 1}: {'raw' if ci == 0 else 'flattened'} echo 2; "
                            f"★ your points; strip: class", fontsize=10)
                if ri == 0 and ci == 0:
                    a.legend(fontsize=7, loc="lower right", ncol=4)
        fig.tight_layout()
        fig.savefig(f"{FIG}/flatten_{TAG}_sections.png", dpi=60)
        plt.close(fig)
    else:
        ax, planes = 2, [Z["disp"].shape[2] // 4, Z["disp"].shape[2] // 2, 3 * Z["disp"].shape[2] // 4]

    # ---- on the roots: residual (rms over levels) and the Wheeler slice at the darkest band's tau
    t_dark = float(np.nanmedian(tmin))
    j_dark = int(round(t_dark / STEP))
    wheeler = G2[:, j_dark]
    raw_at = np.array([E2[i, int(round(t_dark / STEP))] for i in range(N)])
    rms_tr = np.sqrt(np.nanmean(res ** 2, 1))
    summary["Wheeler slice tau (mm)"] = round(t_dark, 3)
    v = Z["v"]
    disp, ctx = Z["disp"], Z["ctx"]
    fig, axs = plt.subplots(3, len(planes), figsize=(8 * len(planes), 24), squeeze=False)
    rest = [d for d in range(3) if d != ax]
    for pi_, k in enumerate(planes):
        m = np.abs(v[tv[keep]][:, ax] - k) < 0.75
        xy = v[tv[keep]][m][:, rest]
        sl = np.take(np.where(ctx, disp, np.nan), k, axis=ax)
        for ri, (val, lab, cmap, lim) in enumerate((
                (rms_tr[m], "residual rms over levels (mm)", "magma", (0, np.nanpercentile(rms_tr, 95))),
                (raw_at[m], f"raw echo 2 at depth {t_dark:.2f} mm", "gray", (-vmax, vmax)),
                (wheeler[m], f"flattened echo 2 at tau {t_dark:.2f} mm (Wheeler slice)", "gray", (-vmax, vmax)))):
            a = axs[ri, pi_]
            a.imshow(sl.T, cmap="gray", origin="lower", vmin=-vmax, vmax=vmax, alpha=0.35)
            sc = a.scatter(xy[:, 0], xy[:, 1], c=val, s=10, cmap=cmap, vmin=lim[0], vmax=lim[1], edgecolors="none")
            plt.colorbar(sc, ax=a, fraction=0.04)
            a.set_title(f"plane {'ijk'[ax]} = {k}: roots, {lab}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/flatten_{TAG}_roots.png", dpi=60)
    plt.close(fig)
    json.dump(summary, open(f"{WORK}/flatten_{TAG}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if not A.figs_only:
        fit()
    figures()
