"""Why the primary column traces turn (plan: reports/Crossing paths and contacts from laminar
profiles.md, *Column traces at folds*). A diagnostic, not a rule: views and rates, no thresholds
that decide anything downstream.

Everything is built fresh from the package, on the label given (the handle-free WM with
--wm-changes): the tissue, the white mesh and its roots, the fitted field T, the fitted_blend
primary traces (baillarger.column_traces, baillarger.streamlines), and the mesh patches with
their distances along the white surface.

Per root: the curvature of the white mesh there (k1 >= k2 and H = (k1 + k2) / 2, per mm, smoothed
over 1 mm; > 0 convex, a crown), the data's coherence just outside the root, the primary's
outcome. Per turn (the trace turned more than 60 deg within 0.3 mm): the arc length; the distance
to the fluid, the WM, missing tissue and other labels; the coherence; whether T fell over the
last 0.3 mm (the trace went over a crest of T); and where the descent of T from one voxel past
the turn (along the direction before it) reaches the WM: another bank (>= 5 mm along the white
surface from the root) or the same one.

Descriptive categories of a turn, for the maps and rates (first that applies): near the fluid
(< 0.3 mm), near missing or other tissue (< 0.3 mm), over a crest to another bank, over a crest
within the bank, interior.

Writes <work>/turn_diag_<name>.json and .npz, <figures>/turn_diag_<name>_{rates,planes}.png.
Usage: turn_diag.py <figures dir> <work dir> --name NAME --box I J K [--size N]
                    [--wm-changes FILE] [--points FILE]
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from baillarger import column_traces as CT
from baillarger import mesh_graph as G
from baillarger import streamlines as S
from baillarger.frame import mesh_curvatures
from baillarger.white import WMChanges

HBINS = [-np.inf, -0.6, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.6, np.inf]
CBINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
NEAR_MM = 0.3
CATS = ["near fluid", "near missing/other", "crest, other bank", "crest, same bank", "interior"]
CCOL = ["#30d5ff", "#9b59b6", "#e34948", "#ff8c00", "#ffd23f"]
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def planes_of(P):
    """The axis along which the points take the fewest values, and the three planes on it with the
    most points within 2 voxels (as column_volume.py)."""
    ax = int(np.argmin([len(np.unique(np.round(P[:, a]))) for a in range(3)]))
    ks = np.unique(np.round(P[:, ax]).astype(int))
    cnt = sorted([(int(np.sum(np.abs(P[:, ax] - k) <= 2)), int(k)) for k in ks], reverse=True)
    return ax, sorted(k for _, k in cnt[:3])


def rate_table(H, turned, cat, keep):
    """Per H bin: roots, turn rate, and each category's share of the roots."""
    rows = []
    for lo, hi in zip(HBINS[:-1], HBINS[1:]):
        m = keep & (H >= lo) & (H < hi)
        n = int(m.sum())
        row = dict(H=f"[{lo}, {hi})", roots=n, turned=round(float(turned[m].mean()), 4) if n else None)
        for c, name in enumerate(CATS):
            row[name] = round(float((turned & (cat == c))[m].mean()), 4) if n else None
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fig")
    ap.add_argument("work")
    ap.add_argument("--name", required=True)
    ap.add_argument("--box", type=int, nargs=3, required=True)
    ap.add_argument("--size", type=int, default=141)
    ap.add_argument("--wm-changes")
    ap.add_argument("--points")
    A = ap.parse_args()
    lo = tuple(A.box)
    hi = tuple(a + A.size for a in lo)
    changes = WMChanges.load(Path(A.wm_changes)) if A.wm_changes else None
    t = CT.load_tissue(lo, hi, A.name, changes=changes)
    vox = t.vox
    base = CT.base_tensor(t)
    _, coh = CT.data_normal(base, t.ctx)
    v, f, nrm = CT.white_mesh(t.wm)
    tv = CT.root_vertices(v, nrm)
    P = CT.mesh_patches(v, f, vox)
    T, info = S.fitted_field(base, t.ctx, t.wm, vox)
    del base
    say(f"{A.name}: {len(tv):,} roots; WM {'handle-free' if changes else 'aseg'}; T {info}")
    k1, k2 = mesh_curvatures(v * vox, f, nrm, 1.0)
    H = 0.5 * (k1 + k2)
    x0 = v[tv]
    pos, ns, out = S.trace(S.gradient_tensor(T, vox), t.labels, t.dface, x0, nrm[tv], vox, CT.NMAX,
                           CT.BLEND_MM)
    q0 = np.clip(np.round(x0).astype(int), 0, np.array(t.wm.shape) - 1)
    keep = t.dface[tuple(q0.T)] >= CT.FACE_MM
    # coherence just outside the root (0.2 mm along the normal)
    coh_root = ndi.map_coordinates(coh, (x0 + nrm[tv] * 0.2 / vox).T, order=1)
    turned = out == S.TURN
    say("primary: " + ", ".join(f"{S.OUTCOME[k]} {np.sum(keep & (out == k)):,}" for k in range(10)
                                 if np.sum(keep & (out == k))))

    # ---- at each turn
    it = np.flatnonzero(turned)
    k = np.arange(it.size)
    end = pos[it, ns[it] - 1].astype(float)
    lag = CT._LAG
    i1 = np.maximum(ns[it] - 1 - lag, 0)
    i0 = np.maximum(ns[it] - 1 - 2 * lag, 0)
    d_before = pos[it, i1].astype(float) - pos[it, i0]
    d_before /= np.maximum(np.linalg.norm(d_before, axis=1, keepdims=True), 1e-12)

    def dist_to(mask):
        return ndi.map_coordinates(ndi.distance_transform_edt(~mask, sampling=vox), end.T, order=1)

    d_fluid = dist_to(t.labels == 1)
    d_wm = dist_to(t.labels == 2)
    d_miss = dist_to((t.labels == 3) | (t.labels == 4))
    c_turn = ndi.map_coordinates(coh, end.T, order=1)
    Ta = S.sample(T, pos[it])
    T_end = Ta[k, ns[it] - 1]
    T_back = Ta[k, i1]
    T_fell = T_end < T_back - 1e-6
    T_max = np.nanmax(Ta, 1)
    T_fell_any = T_end < T_max - 1e-6
    pd, nd, od = S.descend(T, t.labels, t.dface, end + d_before, vox, CT.NMAX)
    foot = pd[k, nd - 1].astype(float)
    last = foot - pd[k, np.maximum(nd - 2, 0)]
    fvert = G.crossing_vertex(v, nrm, foot, last, vox)
    sep = np.where(od == S.WM, P.D[P.of_vertex[tv[it]], P.of_vertex[fvert]], np.nan)
    other = sep >= CT.SAME_MM
    crest = T_fell_any & (od == S.WM)
    cat_t = np.full(it.size, 4)
    cat_t[crest & ~other] = 3
    cat_t[crest & other] = 2
    cat_t[d_miss < NEAR_MM] = 1
    cat_t[d_fluid < NEAR_MM] = 0
    cat = np.full(len(tv), -1)
    cat[it] = cat_t
    kt = keep[it]
    summary = dict(
        site=A.name, wm="handle-free" if changes else "aseg", roots=int(keep.sum()), T_solver=info,
        primary={S.OUTCOME[o]: int(np.sum(keep & (out == o))) for o in range(10) if np.sum(keep & (out == o))},
        turns=int(kt.sum()),
        turn_categories={c: int(np.sum(kt & (cat_t == i))) for i, c in enumerate(CATS)},
        at_turn_p10_50_90={
            "arc length (mm)": np.percentile(((ns[it] - 1) * S.STEP_MM)[kt], [10, 50, 90]).round(2).tolist(),
            "to fluid (mm)": np.percentile(d_fluid[kt], [10, 50, 90]).round(2).tolist(),
            "to WM (mm)": np.percentile(d_wm[kt], [10, 50, 90]).round(2).tolist(),
            "coherence": np.percentile(c_turn[kt], [10, 50, 90]).round(2).tolist(),
            "T fell over the last 0.3 mm (share)": round(float(T_fell[kt].mean()), 3),
            "below T's maximum along the trace (share)": round(float(T_fell_any[kt].mean()), 3),
        },
        by_root_curvature=rate_table(H[tv], turned, cat, keep),
        by_root_coherence=[dict(coh=f"[{a}, {b})", roots=int(np.sum(m := keep & (coh_root >= a) & (coh_root < b))),
                                turned=round(float(turned[m].mean()), 4) if m.any() else None)
                           for a, b in zip(CBINS[:-1], CBINS[1:])],
    )
    json.dump(summary, open(f"{A.work}/turn_diag_{A.name}.json", "w"), indent=1)
    print(json.dumps({k_: summary[k_] for k_ in ("primary", "turns", "turn_categories", "at_turn_p10_50_90")}, indent=1))
    # the primary paths, ragged: path i is path_xyz[path_start[i]:path_start[i + 1]] (box voxel
    # coordinates; add lo for the full grid), sample j at j * STEP_MM along it
    path_start = np.r_[0, np.cumsum(ns)].astype(np.int64)
    path_xyz = np.concatenate([pos[i, :ns[i]] for i in range(len(tv))]).astype(np.float32)
    category = np.full(len(tv), -1, np.int8)  # per root: the turn's category, -1 if it did not turn
    category[it] = cat_t
    np.savez_compressed(f"{A.work}/turn_diag_{A.name}.npz", lo=np.array(lo), vox=vox, step_mm=S.STEP_MM,
                        wm_label="handle-free" if changes else "aseg", categories=np.array(CATS),
                        outcomes=np.array(S.OUTCOME), v=v.astype(np.float32), f=f, nrm=nrm.astype(np.float32),
                        tv=tv, H=H[tv], k1=k1[tv], k2=k2[tv], keep=keep, out=out, ns=ns, coh_root=coh_root,
                        category=category, path_start=path_start, path_xyz=path_xyz,
                        it=it, end=end, cat=cat_t, d_fluid=d_fluid, d_wm=d_wm, d_miss=d_miss, c_turn=c_turn,
                        T_fell=T_fell, sep=sep)

    # ---- views
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    rows = summary["by_root_curvature"]
    xs = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for i, c in enumerate(CATS):
        h = np.array([r[c] or 0 for r in rows])
        axs[0].bar(xs, h, bottom=bottom, color=CCOL[i], label=c)
        bottom += h
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels([r["H"] for r in rows], rotation=30, fontsize=8)
    axs[0].set_xlabel("mean curvature H at the root (1/mm; < 0 fundus, > 0 crown)")
    axs[0].set_ylabel("share of roots whose primary turns")
    axs[0].legend(fontsize=8)
    for i, r in enumerate(rows):
        axs[0].text(i, bottom[i], f"n={r['roots']}", ha="center", va="bottom", fontsize=7)
    m = keep & turned
    axs[1].hist(d_fluid[keep[it]], bins=40, range=(0, 3), alpha=0.6, label="to fluid", color=CCOL[0])
    axs[1].hist(d_wm[keep[it]], bins=40, range=(0, 3), alpha=0.6, label="to WM", color="#1baf7a")
    axs[1].set_xlabel("distance at the turn (mm)")
    axs[1].legend(fontsize=8)
    axs[1].set_title(f"{m.sum():,} turns", fontsize=10)
    fig.suptitle(f"{A.name}: why the primary turns ({summary['wm']} WM)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{A.fig}/turn_diag_{A.name}_rates.png", dpi=80)
    plt.close(fig)

    if A.points:
        Pts = np.array(json.load(open(A.points))["points_full0"], float) - np.array(lo)
        Pts = Pts[np.all((Pts >= 0) & (Pts < A.size), 1)]
        ax, planes = planes_of(Pts)
        img = t.det[1]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
        vmax = np.nanpercentile(np.abs(img[t.ctx]), 97)
        for a, kpl in zip(axs, planes):
            sl = [slice(None)] * 3
            sl[ax] = kpl
            a.imshow(np.asarray(img[tuple(sl)]).T, origin="lower", cmap="gray", vmin=-vmax, vmax=vmax)
            a.contour(np.asarray(t.labels[tuple(sl)] == 2).T, levels=[0.5], colors="#1baf7a", linewidths=0.6)
            a.contour(np.asarray(t.labels[tuple(sl)] == 1).T, levels=[0.5], colors="#30d5ff", linewidths=0.6)
            other_ax = [i for i in range(3) if i != ax]
            near = np.abs(x0[:, ax] - kpl) <= 0.5
            a.scatter(x0[near & keep & ~turned][:, other_ax[0]], x0[near & keep & ~turned][:, other_ax[1]],
                      s=1, c="#555555")
            ne = (np.abs(end[:, ax] - kpl) <= 0.5) & keep[it]
            for i, c in enumerate(CATS):
                mm = ne & (cat_t == i)
                a.scatter(end[mm][:, other_ax[0]], end[mm][:, other_ax[1]], s=6, c=CCOL[i], label=c)
            pp = np.abs(Pts[:, ax] - kpl) <= 2
            a.scatter(Pts[pp][:, other_ax[0]], Pts[pp][:, other_ax[1]], marker="*", s=60, c="w", edgecolors="k")
            a.set_title(f"plane {'xyz'[ax]} = {kpl}: turns (dots by category), roots that did not turn (grey)",
                        fontsize=8)
        axs[0].legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(f"{A.fig}/turn_diag_{A.name}_planes.png", dpi=80)
        plt.close(fig)
    say("done")


if __name__ == "__main__":
    main()
