"""Where layers differ: isopachs, pinch-outs and facies from the joint warp, mapped on the roots
(plan and results: reports/Crossing paths and contacts from laminar profiles.md, *Where layers
differ*). Functions in layer_maps_lib.py, tests in test_layer_maps.py. Diagnostics, not criteria.

Reads <work>/column_volume_<site>.npz, <work>/pia_<site>.npz (H, the white-mesh mean curvature at
each root), <work>/joint_<tag>.npz (the filled warp z_v(tau) and its solved levels), optionally
<work>/bands_<bands>.json (each level's band kind) and <work>/flat_map_<site>.npz (the flat map).

1. Isopachs: thickness in mm of each interval between the solved levels (WM, the bands, the pia)
   and of fixed tau slabs (`--slabs`), per trace; the curvature trend of each interval
   (log t = m(x) + a H, a from neighbouring traces within `--trend-radius` mm) removed.
2. Pinch-out: each band interval's fraction of the whole thickness, smoothed on the roots
   (`--smooth` mm), and its lateral gradient.
3. Band amplitude: the stacked profile at each band's tau (dark bands sign-flipped: larger is
   stronger).
4. Facies: echoes 2 and 3 (trend removed, / SD) at constant tau 0.05-0.92, stacked on the roots
   (`--smooth` mm), Gaussian mixtures on 8 PCs, k 1-8 by ICL, transitions folded into end-members;
   end-members holding under 5 % of the traces are set aside (reported, mapped) and the mixture fitted
   again without them.
3b. Band amplitude gradient: where a band dims (a facies change of one band).
Writes <out>/layer_maps_<tag>.npz and .json, <fig>/layer_maps_<tag>_*.png.

Usage: layer_maps.py <figures> <work> <out> <site> <tag> [--bands TAG] [--slabs N] [--smooth MM] [--k K]
       (with --k, outputs are named layer_maps_<tag>_k<K>)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import flatten_lib as FL
import layer_maps_lib as LM
from volume_views import planes_of

ap = argparse.ArgumentParser()
ap.add_argument("fig")
ap.add_argument("work")
ap.add_argument("out")
ap.add_argument("site")
ap.add_argument("tag")
ap.add_argument("--bands", default=None)
ap.add_argument("--slabs", type=int, default=5)
ap.add_argument("--smooth", type=float, default=0.5)
ap.add_argument("--trend-radius", type=float, default=1.0)
ap.add_argument("--grad-radius", type=float, default=0.5)
ap.add_argument("--nt", type=int, default=45)
ap.add_argument("--k", type=int, default=None)  # facies: this k (ICL and BIC still reported over 1-8)
A = ap.parse_args()
STEP, NDOT = 0.04, 0.7
CLS = {0: "open", 1: "crossing", 2: "own bank", 3: "to contact", 4: "failed"}
T0 = time.time()
NAME = f"{A.tag}_k{A.k}" if A.k else A.tag


def say(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    Z = np.load(f"{A.work}/column_volume_{A.site}.npz")
    Q = np.load(f"{A.work}/pia_{A.site}.npz")
    J = np.load(f"{A.work}/joint_{A.tag}.npz")
    sub, tg, zf, zl = J["sub"], J["tg"], LM.monotone(J["zf"]), J["z"]
    tv, ns, cls, v, nrm, vox = Z["tv"], Z["ns"], Z["cls"][J["sub"]], Z["v"], Z["nrm"], Z["vox"]
    n = sub.size
    # the solved levels' tau: the warp's grid level that passes through them
    lev = [float(tg[np.argmin(np.abs(J["zf"] - zl[k][:, None]).sum(0))]) for k in range(len(zl))]
    kinds = ["?"] * (len(lev) - 1)
    if A.bands:
        Bj = json.load(open(f"{A.work}/bands_{A.bands}.json"))
        kinds = [min(Bj["bands"], key=lambda b: abs(b["tau"] - t))["kind"] for t in lev[:-1]]
    say(f"{n:,} traces; levels {[round(t, 3) for t in lev]} ({', '.join(kinds)})")
    root, nr = v[tv[sub]] * vox, nrm[tv[sub]].astype(float)
    H = Q["H"][sub].astype(float)
    pr_t = LM.neighbour_pairs(root, nr, A.trend_radius, NDOT)
    pr_s = LM.neighbour_pairs(root, nr, A.smooth, NDOT)
    pr_g = LM.neighbour_pairs(root, nr, A.grad_radius, NDOT)
    d_s = np.linalg.norm(root[pr_s[:, 0]] - root[pr_s[:, 1]], axis=1)
    w_s = np.exp(-0.5 * (d_s / (A.smooth / 2)) ** 2)
    say(f"pairs: trend {len(pr_t):,}, smoothing {len(pr_s):,}, gradient {len(pr_g):,}")

    # 1. isopachs
    sets = {"bands": [0.0] + lev, "slabs": list(np.linspace(0, 1, A.slabs + 1))}
    names = {"bands": [f"WM-{kinds[0][0]}{lev[0]:.2f}"]
             + [f"{kinds[i][0]}{lev[i]:.2f}-{kinds[i + 1][0]}{lev[i + 1]:.2f}" for i in range(len(lev) - 2)]
             + [f"{kinds[-1][0]}{lev[-2]:.2f}-pia"],
             "slabs": [f"tau {a:.1f}-{b:.1f}" for a, b in zip(sets["slabs"][:-1], sets["slabs"][1:])]}
    total = zl[-1]
    ok = np.isin(cls, [0, 1, 3]) & (ns[sub] * STEP >= total)  # traces that reach their pia
    out, summ = {}, dict(site=A.site, tag=A.tag, traces=int(n), traces_used=int(ok.sum()), levels=lev,
                         kinds=kinds, smooth_mm=A.smooth, trend_radius_mm=A.trend_radius)
    for s_, edges in sets.items():
        t = LM.isopachs(J["zf"], tg, edges)
        t[~ok] = np.nan
        a = [LM.curvature_trend(t[:, k], H, pr_t) for k in range(t.shape[1])]
        td = np.stack([LM.detrend(t[:, k], H, a[k][0]) for k in range(t.shape[1])], 1)
        out[f"thick_{s_}"], out[f"thick_{s_}_detrended"] = t, td
        summ[f"isopachs_{s_}"] = [
            dict(interval=nm, trend_a=round(ak, 3), trend_pairs=np_,
                 raw_p10_50_90=np.nanpercentile(t[:, k], [10, 50, 90]).round(3).tolist(),
                 detrended_p10_50_90=np.nanpercentile(td[:, k], [10, 50, 90]).round(3).tolist(),
                 under_0p04mm=round(float(np.nanmean(td[:, k] < 0.04)), 4))
            for k, (nm, (ak, np_)) in enumerate(zip(names[s_], a))]
    summ["total_trend_a"] = round(LM.curvature_trend(np.where(ok, total, np.nan), H, pr_t)[0], 3)
    say("isopachs done")

    # 2. pinch-out: fractions smoothed on the roots, then their lateral gradient
    tb = out["thick_bands"]
    r = np.where(ok[:, None], tb / total[:, None], np.nan)
    rs = smooth_nan(r, pr_s, w_s)
    gr = np.stack([np.linalg.norm(LM.lateral_gradient(rs[:, k], root, nr, pr_g), axis=1)
                   for k in range(r.shape[1])], 1)
    out["fraction"], out["fraction_smoothed"], out["fraction_grad"] = r, rs, gr
    summ["fractions"] = [dict(interval=nm, p10_50_90=np.nanpercentile(rs[:, k], [10, 50, 90]).round(3).tolist(),
                              grad_p50_90_99=np.nanpercentile(gr[:, k], [50, 90, 99]).round(3).tolist())
                         for k, nm in enumerate(names["bands"])]
    say("pinch-out done")

    # 3 and 4. profiles at constant tau, stacked
    keep = ~Z["near_face"] & (Z["ns"] > 1)
    fv = Z["fv"]
    F = np.stack([fv[k, sub].astype(float) / np.nanstd(fv[k][keep]) for k in range(2)])
    tf = np.linspace(0.05, 0.92, A.nt)
    zz = LM.depth_at(J["zf"], tg, tf)
    G = FL.resample(F, ns[sub], zz, STEP)  # (2, n, nt)
    X = np.concatenate([G[0], G[1]], 1)
    X[~ok] = np.nan
    Xs = smooth_nan(X, pr_s, w_s)
    Xs[~ok] = np.nan
    amp = np.stack([(-1 if kd == "dark" else 1) * np.array([np.interp(t, tf, row) for row in Xs[:, :A.nt]])
                    if np.isfinite(t) else np.full(n, np.nan) for t, kd in zip(lev[:-1], kinds)], 1)
    out["band_amplitude"] = amp
    out["band_amplitude_grad"] = np.stack([np.linalg.norm(LM.lateral_gradient(amp[:, k], root, nr, pr_g), axis=1)
                                           for k in range(amp.shape[1])], 1)
    say("profiles stacked")
    f, f0, aside = LM.facies_set_aside(Xs, ks=tuple(range(1, 9)))
    if A.k:  # the same set-aside traces, one k
        f = LM.facies(np.where(aside[:, None], np.nan, Xs), ks=(A.k,))
        f["bic"], f["icl"] = summ_k(Xs, aside)

    def fsum(f_):
        return dict(k=f_["k"], bic={k: round(b, 1) for k, b in f_["bic"].items()},
                    icl={k: round(b, 1) for k, b in f_["icl"].items()},
                    pca_explained=f_["explained"].round(3).tolist(), ends=f_["ends"],
                    transitions=[dict(component=c, between=[a, b], alpha=round(al, 3))
                                 for c, a, b, al in f_["transitions"]],
                    sizes=[int((f_["labels"] == c).sum()) for c in range(f_["k"])],
                    mixed_fraction=round(float(np.nanmean(LM.entropy(f_["end_memberships"][ok]) > 0.3)), 3)
                    if len(f_["ends"]) > 1 else 0.0)

    own0 = np.where(aside, np.nan_to_num(f0["end_memberships"]).argmax(1), -1)
    summ["facies_first_fit"] = fsum(f0)
    summ["facies_set_aside"] = [dict(end_member=int(e), traces=int((own0 == e).sum()),
                                     centre_itksnap=[int(c) + 1 for c in np.round(v[tv[sub[own0 == e]]].mean(0))],
                                     pia_p10_50_90=np.percentile(total[own0 == e], [10, 50, 90]).round(2).tolist())
                                for e in np.unique(own0[own0 >= 0])]
    summ["facies"] = fsum(f)
    out.update(memberships=f["memberships"], end_memberships=f["end_memberships"], labels=f["labels"],
               means=f["means"], set_aside=aside, set_aside_end_member=own0)
    say(f"facies: {aside.sum():,} traces set aside; k {f['k']} (ICL {summ['facies']['icl']}), ends {f['ends']}, "
        f"transitions {f['transitions']}")
    np.savez_compressed(f"{A.out}/layer_maps_{NAME}.npz", sub=sub, tf=tf, ok=ok, H=H, **out)
    json.dump(summ, open(f"{A.out}/layer_maps_{NAME}.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    views(Z, sub, ok, names, out, f, tf, lev, kinds)
    say("figures done")


def summ_k(Xs, aside):
    """BIC and ICL over k 1-8 of the kept traces (for the report when k is forced)."""
    g = LM.facies(np.where(aside[:, None], np.nan, Xs), ks=tuple(range(1, 9)))
    return g["bic"], g["icl"]


def smooth_nan(val, pairs, w):
    """smooth_on_roots that leaves nan entries out (weights renormalized); nan where none is left."""
    fin = np.isfinite(val)
    num = LM.smooth_on_roots(np.where(fin, val, 0.0), pairs, w)
    den = LM.smooth_on_roots(fin.astype(float), pairs, w)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def flat_uv(sub, tv):
    try:
        M = np.load(f"{A.work}/flat_map_{A.site}.npz")
    except FileNotFoundError:
        return None
    uv, piece = M["uv_v"], np.zeros(len(M["uv_v"]), int)
    piece[M["orig"]] = M["piece"]
    return uv[tv[sub]], piece[tv[sub]]


def views(Z, sub, ok, names, out, f, tf, lev, kinds):
    tv, P, v = Z["tv"], Z["P"], Z["v"]
    fl = flat_uv(sub, tv)
    pts_uv = None
    if fl is not None and len(P):
        from scipy.spatial import cKDTree

        _, j = cKDTree(v[tv[sub]]).query(P)
        pts_uv = fl[0][j]
    maps = []
    for k, nm in enumerate(names["bands"]):
        maps.append((out["thick_bands_detrended"][:, k], f"isopach {nm} (mm, curvature removed)", "viridis", None))
    for k, nm in enumerate(names["bands"]):
        maps.append((out["fraction_smoothed"][:, k], f"fraction {nm}", "cividis", None))
    for k, nm in enumerate(names["bands"]):
        maps.append((out["fraction_grad"][:, k], f"|grad fraction| {nm} (per mm)", "magma", None))
    for k, (t, kd) in enumerate(zip(lev[:-1], kinds)):
        maps.append((out["band_amplitude"][:, k], f"{kd} band tau {t:.2f}: amplitude (stacked)", "RdBu_r", "sym"))
    for k, (t, kd) in enumerate(zip(lev[:-1], kinds)):
        maps.append((out["band_amplitude_grad"][:, k], f"{kd} band tau {t:.2f}: |grad amplitude| (per mm)", "magma",
                     None))
    em = f["end_memberships"]
    for e in range(em.shape[1]):
        maps.append((em[:, e], f"facies end-member {e + 1}: membership", "plasma", (0, 1)))
    maps.append((np.where(out["set_aside"], out["set_aside_end_member"] + 1.0, 0.0),
                 "facies: set aside (end-member of the first fit; 0 kept)", "tab10", (0, 9)))
    if em.shape[1] > 1:
        maps.append((LM.entropy(em) / np.log(em.shape[1]), "membership entropy / log k (0 pure, 1 even mix)",
                     "Greys", (0, 1)))
    if fl is not None:
        uv, piece = fl
        big = np.bincount(piece[ok]).argmax()
        m = ok & (piece == big)
        draw(maps, uv[m], m, pts_uv, "flat", "flat map u (mm)", "v (mm)")
    if len(P):
        axn, planes = planes_of(P)
        rest = [d for d in range(3) if d != axn]
        for k in planes[:1]:
            m = ok & (np.abs(v[tv[sub]][:, axn] - k) < 6)
            draw(maps, v[tv[sub]][m][:, rest], m, P[np.abs(P[:, axn] - k) < 6][:, rest], f"plane{k}",
                 f"{'ijk'[rest[0]]} (voxel)", f"{'ijk'[rest[1]]} (voxel); roots within 6 voxels of {'ijk'[axn]} = {k}")
    # facies profiles and the criteria
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
    nt = tf.size
    for c in range(f["k"]):
        lab = f"component {c} ({(f['labels'] == c).sum():,})" + (" end-member" if c in f["ends"] else " transition")
        for e, ls in ((0, "-"), (1, ":")):
            axs[e].plot(tf, f["means"][c, e * nt:(e + 1) * nt], ls, label=lab)
    for e in range(2):
        for t in lev[:-1]:
            axs[e].axvline(t, c="k", lw=0.5)
        axs[e].set_xlabel("tau (0 WM, 1 pia)")
        axs[e].set_title(f"echo {e + 2}: facies means (stacked, trend removed, / SD)")
        axs[e].legend(fontsize=7)
    ks = sorted(f["bic"])
    axs[2].plot(ks, [f["bic"][k] for k in ks], "o-", label="BIC")
    axs[2].plot(ks, [f["icl"][k] for k in ks], "s-", label="ICL (k = its lowest unless --k)")
    axs[2].set_xlabel("k")
    axs[2].set_title(f"criteria over the kept traces; mapped: k = {f['k']}")
    axs[2].legend()
    fig.tight_layout()
    fig.savefig(f"{A.fig}/layer_maps_{NAME}_facies.png", dpi=72)
    plt.close(fig)


def draw(maps, xy, m, pts, name, xl, yl):
    nc = 4
    nrow = -(-len(maps) // nc)
    fig, axs = plt.subplots(nrow, nc, figsize=(6 * nc, 5 * nrow), squeeze=False)
    for a in axs.ravel()[len(maps):]:
        a.axis("off")
    for a, (val, title, cmap, lim) in zip(axs.ravel(), maps):
        x = val[m]
        if lim == "sym":
            q = np.nanpercentile(np.abs(x), 98)
            lim = (-q, q)
        elif lim is None:
            lim = tuple(np.nanpercentile(x, [2, 98]))
        o = np.argsort(np.nan_to_num(-np.abs(x - np.nanmedian(x)), nan=-np.inf))
        sc = a.scatter(xy[o, 0], xy[o, 1], c=x[o], s=2, cmap=cmap, vmin=lim[0], vmax=lim[1], edgecolors="none")
        if pts is not None and len(pts):
            a.scatter(pts[:, 0], pts[:, 1], marker="x", c="r", s=14, lw=0.8)
        plt.colorbar(sc, ax=a, fraction=0.04)
        a.set_aspect("equal")
        a.set_title(title, fontsize=9)
        a.set_xlabel(xl, fontsize=8)
        a.set_ylabel(yl, fontsize=8)
    fig.suptitle(f"{A.site} ({NAME}): {name}; red x: the marked points", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{A.fig}/layer_maps_{NAME}_{name}.png", dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    main()
