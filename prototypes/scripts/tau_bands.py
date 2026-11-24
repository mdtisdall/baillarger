"""Bands in the joint warp's tau frame (plan: reports/Crossing paths and contacts from laminar
profiles.md, *The joint warp*, bands). Reads <work>/column_volume_<site>.npz and
<work>/joint_<tag>.npz (the filled warp of joint_site.py).

1. Echoes 2 and 3 (trend removed, each / its SD) of every subsampled trace at constant tau
   (0 WM, 1 pia), nt levels; nothing past the pia.
2. Lateral stacking: each trace's profile averaged with its neighbours' (roots within `--radius`
   mm, normals agreeing), weights exp(-d^2 / 2 (radius/2)^2) times the positive correlation of the
   two raw profiles (edge-aware: a neighbour with another pattern counts little).
3. Bands: extrema of the site's stacked median profile (dark minima and bright maxima) with a
   prominence of at least `--prominence` and a width at half prominence of at least 1.5 voxels at the
   median pia (the band rule: 2-3 voxels, the stria at 1.5-2 at the limit and usable).
4. Per trace and band: the extremum of its stacked profile within +-`--window` tau of the band
   (tau and prominence; nan where none). The spread of each band's tau over traces, by class;
   maps of tau and strength on the roots.
Writes <work>/bands_<tag>.npz and .json, <figures>/bands_<tag>_{profile,maps}.png.

Usage: tau_bands.py <figures> <work> <site> <tag> [--radius MM] [--prominence P] [--window W]
"""

import argparse
import json
import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks, peak_prominences
from scipy.spatial import cKDTree

import flatten_lib as FL
from volume_views import planes_of

ap = argparse.ArgumentParser()
ap.add_argument("fig")
ap.add_argument("work")
ap.add_argument("site")
ap.add_argument("tag")
ap.add_argument("--radius", type=float, default=1.0)
ap.add_argument("--prominence", type=float, default=0.15)
ap.add_argument("--window", type=float, default=0.06)
ap.add_argument("--nt", type=int, default=193)
A = ap.parse_args()
FIG, WORK, SITE, TAG = A.fig, A.work, A.site, A.tag
STEP, NDOT, VOXEL = 0.04, 0.7, 0.16
CLS = {0: "open", 1: "crossing", 2: "own bank", 3: "to contact", 4: "failed"}


def main():
    Z = np.load(f"{WORK}/column_volume_{SITE}.npz")
    J = np.load(f"{WORK}/joint_{TAG}.npz")
    sub, tg, zf = J["sub"], J["tg"], J["zf"]
    tv, ns, fv, cls, v, nrm, vox = Z["tv"], Z["ns"], Z["fv"], Z["cls"], Z["v"], Z["nrm"], Z["vox"]
    n = sub.size
    tf = np.linspace(0, 1, A.nt)
    zz = np.stack([np.interp(tf, tg, zf[i]) for i in range(n)])
    keep = ~Z["near_face"] & (ns > 1)
    F = np.stack([fv[k, sub].astype(float) / np.nanstd(fv[k][keep]) for k in range(2)])
    Gt = FL.resample(F, ns[sub], zz, STEP).mean(0)  # (n, nt), both echoes
    pia_med = float(np.median(zf[:, -1]))
    # 2. edge-aware lateral stacking
    root, nr = v[tv[sub]] * vox, nrm[tv[sub]]
    nb = cKDTree(root).query_ball_point(root, A.radius)
    G0 = np.nan_to_num(Gt)
    Gc = G0 - G0.mean(1, keepdims=True)
    norm = np.sqrt((Gc ** 2).sum(1)) + 1e-9
    S = np.zeros_like(G0)
    for i in range(n):
        j = np.array(nb[i])
        j = j[(nr[j] @ nr[i]) > NDOT]
        d = np.linalg.norm(root[j] - root[i], axis=1)
        corr = (Gc[j] @ Gc[i]) / (norm[j] * norm[i])
        w = np.exp(-0.5 * (d / (A.radius / 2)) ** 2) * np.maximum(corr, 0.0)
        w[j == i] = 1.0
        S[i] = (w[:, None] * G0[j]).sum(0) / w.sum()
    # 3. bands on the site's stacked median profile (open and crossing traces)
    oc = np.isin(cls[sub], [0, 1])
    prof = np.median(S[oc], 0)
    # the project's band rule: at least 2-3 voxels, with the stria (1.5-2 voxels) at the limit and
    # usable; so 1.5 voxels (width at half prominence) at the median pia, in samples
    width = 1.5 * VOXEL / pia_med / (tf[1] - tf[0])
    inner = (tf > 0.05) & (tf < 0.92)  # the pia's own appearance is not a band
    bands = []
    for sign, kind in ((-1, "dark"), (1, "bright")):
        pk, pr = find_peaks(sign * prof, prominence=A.prominence, width=width)
        for p_, prom in zip(pk, pr["prominences"]):
            if inner[p_]:
                bands.append(dict(kind=kind, tau=round(float(tf[p_]), 4), prominence=round(float(prom), 3),
                                  depth_mm_at_median_pia=round(float(tf[p_] * pia_med), 3)))
    bands.sort(key=lambda b: b["tau"])
    # 4. per trace and band
    w_ = int(round(A.window / (tf[1] - tf[0])))
    taus = np.full((len(bands), n), np.nan)
    strength = np.full((len(bands), n), np.nan)
    for b, band in enumerate(bands):
        c = int(round(band["tau"] / (tf[1] - tf[0])))
        sign = -1 if band["kind"] == "dark" else 1
        lo, hi = max(1, c - w_), min(A.nt - 2, c + w_)
        seg = sign * S[:, lo:hi + 1]
        k = np.argmax(seg, 1)
        interior = (k > 0) & (k < hi - lo)
        for i in np.flatnonzero(interior):
            p_ = lo + k[i]
            prom = peak_prominences(sign * S[i], [p_])[0][0]
            if prom > 0:
                taus[b, i] = tf[p_]
                strength[b, i] = prom
    iqr = lambda x: float(np.subtract(*np.nanpercentile(x, [75, 25]))) if np.isfinite(x).sum() > 10 else None  # noqa: E731
    for b, band in enumerate(bands):
        band["found_fraction"] = round(float(np.isfinite(taus[b]).mean()), 3)
        band["tau_iqr_by_class"] = {CLS[c]: (round(iqr(taus[b, cls[sub] == c]), 4) if iqr(taus[b, cls[sub] == c]) is not None else None)
                                    for c in CLS}
        band["tau_median_by_class"] = {CLS[c]: (round(float(np.nanmedian(taus[b, cls[sub] == c])), 4)
                                                if np.isfinite(taus[b, cls[sub] == c]).sum() > 10 else None) for c in CLS}
        # the same band's spread in raw depth (for comparison)
        dz = np.array([np.interp(t, tg, zf[i]) if np.isfinite(t) else np.nan for i, t in enumerate(taus[b])])
        band["depth_iqr_mm"] = round(iqr(dz), 3) if iqr(dz) is not None else None
        band["tau_iqr_in_mm_at_median_pia"] = round(iqr(taus[b]) * pia_med, 3) if iqr(taus[b]) is not None else None
    summary = dict(site=SITE, tag=TAG, traces=int(n), median_pia_mm=round(pia_med, 3), bands=bands,
                   radius=A.radius, prominence=A.prominence, window=A.window)
    np.savez_compressed(f"{WORK}/bands_{TAG}.npz", sub=sub, tf=tf, S=S.astype(np.float32), G=Gt.astype(np.float32),
                        taus=taus, strength=strength, prof=prof)
    json.dump(summary, open(f"{WORK}/bands_{TAG}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    views(Z, sub, tf, Gt, S, prof, bands, taus, strength, pia_med)


def views(Z, sub, tf, Gt, S, prof, bands, taus, strength, pia_med):
    cls = Z["cls"][sub]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for c, col in ((0, "#1baf7a"), (1, "#2a78d6")):
        m = cls == c
        ax.plot(tf, np.nanmedian(Gt[m], 0), c=col, lw=0.8, alpha=0.6, label=f"{CLS[c]}: single traces, median")
        ax.plot(tf, np.median(S[m], 0), c=col, lw=1.8, label=f"{CLS[c]}: stacked, median")
    ax.plot(tf, prof, c="k", lw=1, ls="--", label="site profile (stacked, open + crossing)")
    for b in bands:
        ax.axvline(b["tau"], c="#e34948" if b["kind"] == "dark" else "#ffb000", lw=0.8)
        ax.text(b["tau"], np.nanmax(prof), f"{b['kind'][0]} {b['tau']:.2f}", fontsize=7,
                rotation=90, va="top")
    sec = ax.secondary_xaxis("top", functions=(lambda t: t * pia_med, lambda d: d / pia_med))
    sec.set_xlabel("depth at the median pia (mm)")
    ax.set_xlabel("tau (0 WM, 1 pia)")
    ax.set_ylabel("echoes 2 and 3, trend removed, / SD")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/bands_{TAG}_profile.png", dpi=80)
    plt.close(fig)
    P, v, tv = Z["P"], Z["v"], Z["tv"]
    if not len(P) or not bands:
        return
    axn, planes = planes_of(P)
    rest = [d for d in range(3) if d != axn]
    disp, ctx = Z["disp"], Z["ctx"]
    vmax = np.nanpercentile(np.abs(disp[ctx]), 97)
    nb = len(bands)
    fig, axs = plt.subplots(2 * nb, len(planes), figsize=(6 * len(planes), 5.5 * 2 * nb), squeeze=False)
    for b, band in enumerate(bands):
        for pi_, k in enumerate(planes):
            m = np.abs(v[tv[sub]][:, axn] - k) < 0.75
            xy = v[tv[sub]][m][:, rest]
            sl = np.take(np.where(ctx, disp, np.nan), k, axis=axn)
            for r, (val, lab, cmap, lim) in enumerate((
                    (taus[b, m], f"tau of {band['kind']} band {band['tau']:.2f}", "viridis",
                     (band["tau"] - A.window, band["tau"] + A.window)),
                    (strength[b, m], "its strength (prominence)", "magma",
                     (0, np.nanpercentile(strength[b], 95) if np.isfinite(strength[b]).any() else 1)))):
                a = axs[2 * b + r, pi_]
                a.imshow(sl.T, cmap="gray", origin="lower", vmin=-vmax, vmax=vmax, alpha=0.35)
                sc = a.scatter(xy[:, 0], xy[:, 1], c=val, s=6, cmap=cmap, vmin=lim[0], vmax=lim[1], edgecolors="none")
                plt.colorbar(sc, ax=a, fraction=0.04)
                a.set_title(f"plane {'ijk'[axn]} = {k}: {lab}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/bands_{TAG}_maps.png", dpi=55)
    plt.close(fig)


if __name__ == "__main__":
    main()
