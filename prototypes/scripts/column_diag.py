"""Diagnostics of the column traces on real data, without ground truth (plan: reports/Crossing
paths and contacts from laminar profiles.md, *Column traces at folds*). All 3D.

Per site, for each tracer of column_tracers (each with column_volume.py's fallback and classes),
per trace, roots >= 2 mm from a face:
(a) round trip: from the primary trace's last sample in the cortex, traced back along the same
    field (the base tensor for blend; WM no longer ignored); the return's closest approach to
    the root (mm), where the return ends on WM;
(b) order: neighbouring roots (within 1.5 voxels, normals agreeing, cos > 0.5) are compared at
    equal arc length; a pair swaps where their offset, taken perpendicular to the traces, turns
    by more than 90 deg from its offset at the roots (the columns crossed); swaps per trace;
(c) agreement: on 4,000 random traces, the angle between each cortex sample's tangent and the
    data's layer normal at the nearest voxel, by the coherence there;
(d) everything against the root's mean curvature (white mesh, baillarger.frame.mesh_curvatures,
    1 mm) and the mean coherence along the trace; maps on the planes of the marked points.
Also, as column_volume.py, the marked points reached by crossing traces within 0.3 mm.

Usage: column_diag.py <figures dir> <work dir> SITE [--tracers ...]
Writes <work>/column_diag_<site>.json and .npz, <figures>/column_diag_<site>_{maps,curves}.png.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import column_tracers as CT
import crossing_lib as X
from baillarger.frame import mesh_curvatures

NMAX = 201
T0 = time.time()
HBINS = [-np.inf, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.6, np.inf]
CBINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def round_trip(Tc, labels, dface, vox, pos, ns, outc, tracer, x0):
    """(a) per trace (nan where not run): distance (mm) of the return's end from the root, and the
    return's outcome."""
    n = len(ns)
    rt = np.full(n, np.nan)
    rto = np.full(n, -1)
    go = np.flatnonzero(
        (tracer == 0) & ((outc == X.FLUID) | (outc == X.WM)) & (ns >= 4)
    )
    xs = pos[go, ns[go] - 2].astype(float)
    ds = pos[go, ns[go] - 3].astype(float) - xs
    pb, nb, ob = X.trace_array(Tc, labels, dface, xs, ds, vox, NMAX, wm_after=0.0)
    # the return's closest approach to the root (its end lies up to a step inside the WM)
    d = np.nanmin(
        np.linalg.norm((pb.astype(float) - x0[go, None]) * vox, axis=2), axis=1
    )
    rt[go] = np.where(ob == X.WM, d, np.nan)
    rto[go] = ob
    return rt, rto


def neighbour_pairs(x0, nrm, keep):
    tree = cKDTree(x0)
    pr = tree.query_pairs(1.5, output_type="ndarray")
    ok = (
        (np.sum(nrm[pr[:, 0]] * nrm[pr[:, 1]], 1) > 0.5)
        & keep[pr[:, 0]]
        & keep[pr[:, 1]]
    )
    return pr[ok]


def swaps(pos, ns, pairs, vox, chunk=20000):
    """(b) per pair: the first arc length (mm) at which the pair's perpendicular offset has turned
    by more than 90 deg from its offset at the roots (nan: never)."""
    first = np.full(len(pairs), np.nan)
    for a in range(0, len(pairs), chunk):
        pr = pairs[a : a + chunk]
        m = np.minimum(ns[pr[:, 0]], ns[pr[:, 1]])
        P1 = pos[pr[:, 0]].astype(float) * vox
        P2 = pos[pr[:, 1]].astype(float) * vox
        t = np.gradient(P1, axis=1) + np.gradient(P2, axis=1)
        t /= np.linalg.norm(t, axis=2, keepdims=True) + 1e-12
        r = P2 - P1
        rp = r - np.sum(r * t, 2, keepdims=True) * t
        r0 = rp[:, :1]  # the offset at the roots, perpendicular to the traces
        dot = np.sum(rp * r0, 2)
        valid = np.arange(pos.shape[1])[None] < (m[:, None] - 1)
        valid[:, 0] = False
        flip = valid & (dot < 0)
        has = flip.any(1)
        first[a : a + chunk][has] = np.argmax(flip[has], 1) * X.STEP_MM
    return first


def agreement(pos, ns, nh, coh, ctx, vox, sel):
    """(c) angle (deg) between the tangent and the data's normal, and coherence, at cortex samples."""
    ang, cc = [], []
    for t in sel:
        p = pos[t, : ns[t]].astype(float)
        if len(p) < 4:
            continue
        tg = np.gradient(p * vox, axis=0)
        tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12
        q = np.clip(np.round(p).astype(int), 0, np.array(ctx.shape) - 1)
        m = ctx[tuple(q.T)]
        n = nh[tuple(q[m].T)]
        ang.append(np.degrees(np.arccos(np.clip(np.abs(np.sum(tg[m] * n, 1)), 0, 1))))
        cc.append(coh[tuple(q[m].T)])
    return np.concatenate(ang), np.concatenate(cc)


def mean_along(vol, pos, ns, chunk=20000):
    out = np.full(len(ns), np.nan)
    for a in range(0, len(ns), chunk):
        p = pos[a : a + chunk]
        ok = np.isfinite(p[..., 0])
        v = np.full(ok.shape, np.nan)
        v[ok] = ndi.map_coordinates(vol, p[ok].T.astype(float), order=1, mode="nearest")
        with np.errstate(invalid="ignore"):
            out[a : a + chunk] = np.nanmean(v, 1)
    return out


def points_reached(P, pos, ns, cls, keep, vox):
    if not len(P):
        return None
    tree = cKDTree(P * vox)
    hit = set()
    for t in np.flatnonzero(keep & (cls == CT.CROSS)):
        d, k = tree.query(pos[t, : ns[t]].astype(float) * vox)
        i = int(np.argmin(d))
        if d[i] <= 0.3:
            hit.add(int(k[i]))
    return len(hit)


def binned(vals, x, bins):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(vals)
        out.append(
            [
                int(m.sum()),
                round(float(np.median(vals[m])), 3) if m.sum() else None,
                round(float(np.percentile(vals[m], 90)), 3) if m.sum() else None,
            ]
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fig")
    ap.add_argument("work")
    ap.add_argument("site")
    ap.add_argument("--tracers", nargs="*", default=CT.TRACERS)
    A = ap.parse_args()
    S = CT.load_site(A.site)
    B, labels, P, vox = S["B"], S["labels"], S["P"], S["vox"]
    ctx = B["ctx"]
    v, f, nrm = CT.white_mesh(B["wm"])
    tv = CT.pick_vertices(v, nrm)
    x0, d0 = v[tv], nrm[tv]
    keep = (
        B["dface"][
            tuple(np.clip(np.round(x0).astype(int), 0, np.array(labels.shape) - 1).T)
        ]
        >= 2.0
    )
    k1, k2 = mesh_curvatures(v * vox, f, nrm, 1.0)
    H = ((k1 + k2) / 2)[tv]
    say(f"{A.site}: {len(tv):,} traces, {keep.sum():,} roots >= 2 mm from a face")
    base = CT.base_tensor(B)
    nh, coh = CT.eig(base)
    coh = np.where(ctx, coh, 0.0).astype(np.float32)
    Pa = CT.Pass(B)
    say("pass done")
    pairs = neighbour_pairs(x0, d0, keep)
    rng = np.random.default_rng(0)
    sel = rng.choice(np.flatnonzero(keep), min(4000, keep.sum()), replace=False)
    summary = dict(
        site=A.site,
        traces=int(len(tv)),
        kept=int(keep.sum()),
        pairs=int(len(pairs)),
        H_bins=[str(b) for b in HBINS],
        coh_bins=CBINS,
    )
    arrays = dict(tv=tv, keep=keep, H=H, x0=x0, pairs=pairs)
    for tr in A.tracers:
        Tf, notes = CT.fields(B, labels, [tr], base=base)
        Tc = Tf[tr]
        pos1, ns1, out1 = CT.primary(tr, Tc, labels, B["dface"], x0, d0, vox, NMAX)
        pos, ns, outc, tracer, cls, _, _ = CT.with_fallback(
            Pa, B, labels, x0, pos1, ns1, out1, NMAX
        )
        del pos1
        say(f"{tr}: traced")
        rt, rto = round_trip(
            base if tr == "blend" else Tc,
            labels,
            B["dface"],
            vox,
            pos,
            ns,
            outc,
            tracer,
            x0,
        )
        sw = swaps(pos, ns, pairs, vox)
        nsw = np.bincount(pairs[np.isfinite(sw)].ravel(), minlength=len(tv))
        ang, cc = agreement(pos, ns, nh, coh, ctx, vox, sel)
        cmean = mean_along(coh, pos, ns)
        say(f"{tr}: diagnostics done")
        k = keep
        row = dict(notes=notes)
        row["classes"] = {
            CT.CLASS[c]: [
                int(np.sum(k & (cls == c) & (tracer == 0))),
                int(np.sum(k & (cls == c) & (tracer == 1))),
            ]
            for c in CT.CLASS
        }
        row["primary outcomes"] = {
            X.OUTCOME[o]: int(np.sum(k & (out1 == o)))
            for o in range(10)
            if np.sum(k & (out1 == o))
        }
        okrt = k & np.isfinite(rt)
        row["round trip (mm) median, p90, n; returns not ending on WM"] = [
            round(float(np.median(rt[okrt])), 3),
            round(float(np.percentile(rt[okrt], 90)), 3),
            int(okrt.sum()),
            int(np.sum(k & (rto >= 0) & (rto != X.WM))),
        ]
        row["pairs swapping, fraction; median depth of the first swap (mm)"] = [
            round(float(np.mean(np.isfinite(sw))), 4),
            round(float(np.nanmedian(sw)), 2) if np.isfinite(sw).any() else None,
        ]
        row["traces with any swap, fraction"] = round(float(np.mean(nsw[k] > 0)), 4)
        row["tangent vs data normal (deg): median per coherence bin"] = binned(
            ang, cc, CBINS
        )
        row["round trip by root curvature [n, median, p90]"] = binned(
            np.where(k, rt, np.nan), H, HBINS
        )
        row["swapping traces by root curvature [n, fraction]"] = [
            [
                int(np.sum(k & (H >= lo) & (H < hi))),
                round(float(np.mean(nsw[k & (H >= lo) & (H < hi)] > 0)), 4)
                if np.sum(k & (H >= lo) & (H < hi))
                else None,
            ]
            for lo, hi in zip(HBINS[:-1], HBINS[1:])
        ]
        row["failed by root curvature [n, fraction]"] = [
            [
                int(np.sum(k & (H >= lo) & (H < hi))),
                round(float(np.mean(cls[k & (H >= lo) & (H < hi)] == CT.FAILED)), 3)
                if np.sum(k & (H >= lo) & (H < hi))
                else None,
            ]
            for lo, hi in zip(HBINS[:-1], HBINS[1:])
        ]
        row["round trip by mean coherence [n, median, p90]"] = binned(
            np.where(k, rt, np.nan), cmean, CBINS
        )
        row["points reached by crossing traces"] = points_reached(
            P, pos, ns, cls, keep, vox
        )
        summary[tr] = row
        print(json.dumps({tr: row}, indent=None)[:3000], flush=True)
        for key, arr in (
            ("cls", cls),
            ("tracer", tracer),
            ("out1", out1),
            ("rt", rt),
            ("nsw", nsw),
            ("sw", sw),
            ("cmean", cmean),
            ("L", (ns - 1) * X.STEP_MM),
        ):
            arrays[f"{tr}:{key}"] = arr
        del pos, Tf, Tc
        gc.collect()
    json.dump(summary, open(f"{A.work}/column_diag_{A.site}.json", "w"), indent=1)
    np.savez_compressed(f"{A.work}/column_diag_{A.site}.npz", **arrays)
    figures(A, S, arrays, A.tracers)
    say("saved")


def planes_of(P):
    """The axis along which the points take the fewest values, and the three planes on it with the
    most points within 2 voxels."""
    ax = int(np.argmin([len(np.unique(np.round(P[:, a]))) for a in range(3)]))
    ks = np.unique(np.round(P[:, ax]).astype(int))
    cnt = [(k, int(np.sum(np.abs(P[:, ax] - k) <= 2))) for k in ks]
    chosen = []
    for k, _ in sorted(cnt, key=lambda t: -t[1]):
        if all(abs(k - c) > 4 for c in chosen):
            chosen.append(k)
        if len(chosen) == 3:
            break
    return ax, chosen


def figures(A, S, Z, tracers):
    P, B = S["P"], S["B"]
    x0, keep, H = Z["x0"], Z["keep"], Z["H"]
    if len(P):
        ax, planes = planes_of(P)
    else:
        ax, planes = 2, [35, 70, 105]
    rest = [d for d in range(3) if d != ax]
    disp = np.where(B["ctx"], B["disp"], np.nan)
    vmax = np.nanpercentile(np.abs(disp), 97)
    rows = [("H", "root mean curvature (1/mm)")] + [
        (tr, f"{tr}: round trip (mm); x: swapping; ^: failed") for tr in tracers
    ]
    fig, axs = plt.subplots(
        len(rows),
        len(planes),
        figsize=(6.5 * len(planes), 6.2 * len(rows)),
        squeeze=False,
    )
    for j, k in enumerate(planes):
        m = keep & (np.abs(x0[:, ax] - k) < 0.75)
        sl = np.take(disp, k, axis=ax)
        for i, (tr, title) in enumerate(rows):
            a = axs[i, j]
            a.imshow(sl.T, cmap="gray", origin="lower", vmin=-vmax, vmax=vmax)
            if tr == "H":
                sc = a.scatter(
                    x0[m, rest[0]],
                    x0[m, rest[1]],
                    c=H[m],
                    s=6,
                    cmap="coolwarm",
                    vmin=-0.6,
                    vmax=0.6,
                )
            else:
                rt = Z[f"{tr}:rt"]
                sc = a.scatter(
                    x0[m, rest[0]],
                    x0[m, rest[1]],
                    c=rt[m],
                    s=6,
                    cmap="magma_r",
                    vmin=0,
                    vmax=0.8,
                )
                ms = m & (Z[f"{tr}:nsw"] > 0)
                a.plot(
                    x0[ms, rest[0]], x0[ms, rest[1]], "x", c="#30d5ff", ms=4, mew=0.8
                )
                mf = m & (Z[f"{tr}:cls"] == CT.FAILED)
                a.plot(
                    x0[mf, rest[0]],
                    x0[mf, rest[1]],
                    "^",
                    mfc="none",
                    mec="#1baf7a",
                    ms=3,
                    mew=0.6,
                )
            mp = np.abs(P[:, ax] - k) <= 2 if len(P) else np.zeros(0, bool)
            if len(P):
                a.plot(P[mp, rest[0]], P[mp, rest[1]], "*", c="#ffd23f", ms=8)
            a.set_title(f"plane {'ijk'[ax]} = {k}: {title}", fontsize=9)
            plt.colorbar(sc, ax=a, fraction=0.04)
    fig.tight_layout()
    fig.savefig(f"{A.fig}/column_diag_{A.site}_maps.png", dpi=60)
    plt.close(fig)
    # curves against curvature
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    centres = np.arange(len(HBINS) - 1)
    labels_ = [f"{lo:+.2f}..{hi:+.2f}" for lo, hi in zip(HBINS[:-1], HBINS[1:])]
    for tr in tracers:
        rt, nsw, cls = Z[f"{tr}:rt"], Z[f"{tr}:nsw"], Z[f"{tr}:cls"]
        med, swf, ff = [], [], []
        for lo, hi in zip(HBINS[:-1], HBINS[1:]):
            m = keep & (H >= lo) & (H < hi)
            r = rt[m]
            med.append(np.nanmedian(r) if np.isfinite(r).any() else np.nan)
            swf.append(np.mean(nsw[m] > 0) if m.any() else np.nan)
            ff.append(np.mean(cls[m] == CT.FAILED) if m.any() else np.nan)
        for a, y in zip(axs, (med, swf, ff)):
            a.plot(centres, y, "o-", label=tr)
    for a, t in zip(
        axs,
        (
            "round trip, median (mm)",
            "traces with a swap (fraction)",
            "failed (fraction)",
        ),
    ):
        a.set_xticks(centres)
        a.set_xticklabels(labels_, rotation=45, fontsize=8)
        a.set_xlabel("root mean curvature (1/mm; - fundus, + crown)")
        a.set_title(f"{A.site}: {t}")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(f"{A.fig}/column_diag_{A.site}_curves.png", dpi=72)
    plt.close(fig)


if __name__ == "__main__":
    main()
