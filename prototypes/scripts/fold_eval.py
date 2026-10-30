"""Each tracer on the fold phantoms (fold_phantoms.py) against the true columns (plan: reports/
Crossing paths and contacts from laminar profiles.md, *Column traces at folds*). All 3D.

Per phantom (and site's features, and noise scale): the white mesh and roots as column_volume.py,
the equal-arrival pass for the fallback, then per tracer, per root at least 2 mm from a face:
- path error: the largest (and the mean) distance of the trace's samples from the true column's
  line, over the samples whose position along that line lies within the column (mm);
- end error: the distance from the trace's last sample to the true column's end (the top, or the
  far bank's WM where the column crosses a contact), mm;
- contact error (kiss, fused sector): the distance from the true contact point of the sample where
  the trace crosses the contact cylinder;
- outcome: the class reached (open, crossing, own bank, to contact, failed) against the true one;
- the root's true mean curvature, and the base tensor's mean coherence along the true column.
Primary traces and primary + fallback (column_volume.py's) are both reported.

Usage: fold_eval.py <figures dir> <work dir> [--site V1V2_banks|M1S1] [--noise 1.0] [--phantoms ...]
                    [--tracers ...] [--tag TAG]
Writes <work>/fold_eval_<site>_n<noise>[_TAG].json and .npz, <figures>/fold_eval_<site>_n<noise>_<phantom>.png.
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
from scipy import ndimage as ndi

import column_tracers as CT
import crossing_lib as X
import fold_phantoms as FP

NMAX = 201
T0 = time.time()
TCOL = {
    "current": "#e34948",
    "laplace": "#2a78d6",
    "fitted": "#30d5ff",
    "fitted_blend": "#ff2bd6",
    "multiscale": "#1baf7a",
    "blend": "#c79a00",
    "perbank": "#9b59b6",
}


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def errors(fold, truth, pos, ns, vox):
    """Per trace: max and mean lateral distance from the true line within the column, end error,
    contact error (nan where none), all mm."""
    n = pos.shape[0]
    emax, emean, eend, econ = (np.full(n, np.nan) for _ in range(4))
    c, u, r0, r1, rc = truth["c"], truth["u"], truth["r0"], truth["r1"], truth["rc"]
    lo, hi = np.minimum(r0, r1), np.maximum(r0, r1)
    for i in range(n):
        p = pos[i, : ns[i]].astype(float)
        Y = fold.to_local(p)
        v = Y - c[i]
        a = v @ u[i]
        perp = np.linalg.norm(v - a[:, None] * u[i], axis=1)
        m = (a >= lo[i]) & (a <= hi[i])
        if m.any():
            emax[i], emean[i] = perp[m].max(), perp[m].mean()
        end_true = c[i] + r1[i] * u[i]
        eend[i] = np.linalg.norm(Y[-1] - end_true)
        if np.isfinite(rc[i]):
            r = np.linalg.norm(Y - fold.foot(Y), axis=1)
            s = np.sign(r - rc[i])
            k = np.flatnonzero(s[1:] != s[:-1])
            if k.size:
                j = k[0]
                f = (rc[i] - r[j]) / (r[j + 1] - r[j] + 1e-12)
                q = Y[j] + f * (Y[j + 1] - Y[j])
                econ[i] = np.linalg.norm(q - (c[i] + rc[i] * u[i]))
    return emax, emean, eend, econ


def true_class(truth):
    """Open where the column ends at a top, crossing where it ends on the far bank's WM."""
    return np.where(np.isfinite(truth["rc"]), CT.CROSS, CT.OPEN)


def coherence_along(fold, truth, coh, step=0.04):
    n = len(truth["r0"])
    out = np.full(n, np.nan)
    for i in range(n):
        a = np.arange(
            min(truth["r0"][i], truth["r1"][i]),
            max(truth["r0"][i], truth["r1"][i]),
            step,
        )
        Y = truth["c"][i] + a[:, None] * truth["u"][i]
        out[i] = ndi.map_coordinates(
            coh, fold.to_vox(Y).T, order=1, mode="nearest"
        ).mean()
    return out


def run_phantom(name, fold, site, noise, tracers):
    B = FP.make(fold, site, noise=noise)
    labels = B["labels"]
    v, f, nrm = CT.white_mesh(B["wm"])
    tv = CT.pick_vertices(v, nrm)
    x0, d0 = v[tv], nrm[tv]
    keep = (
        B["dface"][
            tuple(np.clip(np.round(x0).astype(int), 0, np.array(labels.shape) - 1).T)
        ]
        >= 2.0
    )
    truth = fold.column(x0)
    base = CT.base_tensor(B)
    _, coh = CT.eig(base)
    coh = np.where(B["ctx"], coh, 0.0)
    ccol = coherence_along(fold, truth, coh)
    Pa = CT.Pass(B)
    Tf, notes = CT.fields(B, labels, tracers, base=base)
    res = dict(
        noise_sd=B["noise_sd"],
        roots=int(keep.sum()),
        notes=notes,
        coh_ctx_p10_50_90=np.percentile(coh[B["ctx"]], [10, 50, 90]).round(3).tolist(),
    )
    per = dict(
        hmean=truth["hmean"][keep], coh=ccol[keep], true_cls=true_class(truth)[keep]
    )
    paths = {}
    for tr in tracers:
        pos1, ns1, out1 = CT.primary(
            tr, Tf[tr], labels, B["dface"], x0, d0, B["vox"], NMAX
        )
        e1 = errors(fold, truth, pos1, ns1, B["vox"])
        pos, ns, outc, tracer, cls, _, _ = CT.with_fallback(
            Pa, B, labels, x0, pos1.copy(), ns1, out1, NMAX
        )
        cls1 = np.where(
            tracer == 0, cls, CT.FAILED
        )  # the primary's own class; failed where it handed over
        e = errors(fold, truth, pos, ns, B["vox"])
        for key, arr in (
            ("emax1", e1[0]),
            ("eend1", e1[2]),
            ("econ1", e1[3]),
            ("cls1", cls1),
            ("out1", out1),
            ("emax", e[0]),
            ("emean", e[1]),
            ("eend", e[2]),
            ("econ", e[3]),
            ("cls", cls),
            ("tracer", tracer),
        ):
            per[f"{tr}:{key}"] = arr[keep]
        paths[tr] = (pos, ns)
        say(
            f"  {name} {tr}: primary ok {np.mean(cls1[keep] == per['true_cls']):.2f}, "
            f"path err med {np.nanmedian(e[0][keep]):.3f} p90 {np.nanpercentile(e[0][keep], 90):.3f}, "
            f"end err med {np.nanmedian(e[2][keep]):.3f}"
        )
    view(name, fold, B, x0, keep, truth, paths, tracers)
    return res, per


def summarize(per, tracers):
    """Per tracer: medians and 90th percentiles over roots, all and by curvature group."""
    out = {}
    hm = per["hmean"]
    groups = {"all": np.ones(hm.size, bool)}
    for h in np.unique(np.round(hm, 3)):
        groups[f"H={h:+.3f}"] = np.round(hm, 3) == h
    tc = per["true_cls"]
    for tr in tracers:
        d = {}
        for g, m in groups.items():
            if m.sum() < 5:
                continue
            ok1 = per[f"{tr}:cls1"][m] == tc[m]
            ok = per[f"{tr}:cls"][m] == tc[m]
            row = dict(
                n=int(m.sum()),
                primary_right_class=round(float(ok1.mean()), 3),
                right_class=round(float(ok.mean()), 3),
                fallback=round(float(np.mean(per[f"{tr}:tracer"][m] == 1)), 3),
            )
            for key in ("emax1", "eend1", "emax", "eend", "econ"):
                a = per[f"{tr}:{key}"][m]
                if key.endswith("1"):
                    a = a[
                        per[f"{tr}:cls1"][m] == tc[m]
                    ]  # primary errors on its right-class traces
                a = a[np.isfinite(a)]
                if a.size:
                    row[key] = [
                        round(float(np.median(a)), 3),
                        round(float(np.percentile(a, 90)), 3),
                    ]
            d[g] = row
        out[tr] = d
    return out


def view(name, fold, B, x0, keep, truth, paths, tracers):
    """A plane through the fold: echo 2 (trend removed), the true columns and each tracer's paths
    for roots within 0.25 mm of the plane."""
    kind = fold.kind
    if kind == "crown":
        ax_u, ax_v, ax_n = 0, 2, 1
    else:
        ax_u, ax_v, ax_n = 0, 1, 2
    ext = np.array(fold.shape) * FP.H / 2
    g = np.arange(-ext.min(), ext.min(), FP.H / 2)
    U, Vv = np.meshgrid(g, g, indexing="ij")
    Y = np.zeros(U.shape + (3,))
    Y[..., ax_u], Y[..., ax_v] = U, Vv
    img = ndi.map_coordinates(
        B["det"][1],
        fold.to_vox(Y.reshape(-1, 3)).T,
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(U.shape)
    Yr = fold.to_local(x0)
    near = keep & (np.abs(Yr[:, ax_n]) < 0.25)
    sel = np.flatnonzero(near)[::3]
    nt = len(tracers)
    fig, axs = plt.subplots(1, nt, figsize=(5.2 * nt, 5.4), squeeze=False)
    vmax = np.nanpercentile(np.abs(img), 98)
    for a, tr in zip(axs[0], tracers):
        a.imshow(
            img.T,
            origin="lower",
            cmap="gray",
            vmin=-vmax,
            vmax=vmax,
            extent=[g[0], g[-1], g[0], g[-1]],
        )
        pos, ns = paths[tr]
        for i in sel:
            c, u, r0, r1 = truth["c"][i], truth["u"][i], truth["r0"][i], truth["r1"][i]
            seg = np.array([c + r0 * u, c + r1 * u])
            a.plot(seg[:, ax_u], seg[:, ax_v], "-", c="#ffd23f", lw=0.6, alpha=0.8)
            p = fold.to_local(pos[i, : ns[i]].astype(float))
            a.plot(p[:, ax_u], p[:, ax_v], "-", c=TCOL.get(tr, "c"), lw=0.8)
        a.set_title(f"{name}: {tr} (gold: true columns)", fontsize=9)
        a.set_xlim(g[0], g[-1])
        a.set_ylim(g[0], g[-1])
    fig.tight_layout()
    fig.savefig(f"{FIG}/fold_eval_{SITE}_n{NOISE:g}_{name}.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fig")
    ap.add_argument("work")
    ap.add_argument("--site", default="V1V2_banks")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--phantoms", nargs="*")
    ap.add_argument("--tracers", nargs="*", default=CT.TRACERS)
    ap.add_argument("--tag", default="")
    A = ap.parse_args()
    FIG, SITE, NOISE = A.fig, A.site, A.noise
    allp = FP.standard()
    names = A.phantoms or list(allp)
    summary, arrays = {}, {}
    for nm in names:
        say(f"{nm}: shape {allp[nm].shape}")
        res, per = run_phantom(nm, allp[nm], SITE, NOISE, A.tracers)
        res["summary"] = summarize(per, A.tracers)
        summary[nm] = res
        for k, v in per.items():
            arrays[f"{nm}|{k}"] = v
    tag = f"_{A.tag}" if A.tag else ""
    out = f"{A.work}/fold_eval_{SITE}_n{NOISE:g}{tag}"
    json.dump(summary, open(out + ".json", "w"), indent=1)
    np.savez_compressed(out + ".npz", **arrays)
    say("saved " + out)
