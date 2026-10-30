"""What the field and the geometry look like at the primary traces' turns (after turn_diag.py),
for the turns turn_diag could not explain: interior turns and turns near the fluid.

Reads <work>/turn_diag_<name>.npz (the paths and the turn categories) and rebuilds the fitted
field T on the same label. Per turn, over the last 0.6 mm before it (two turn windows):
- |grad T| (mm/mm; 1 where T is a distance) at the turn and its minimum before it;
- the angle between the trace's direction before the turn and after it, and between each and
  grad T there (the unsigned trace follows grad T up to sign);
- the distance from the turn to the edge of T's domain (the cortex and the WM next to it), in
  voxels: outside it T is filled with the nearest domain value, so grad T there is not T's;
- the data's coherence along the last 0.6 mm.

Writes <work>/turn_cases_<name>.json and <figures>/turn_cases_<name>_*.png (examples: local
slices of T and the data with the paths, for the categories asked).
Usage: turn_cases.py <figures dir> <work dir> --name NAME --box I J K [--size N] [--dseg FILE]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from baillarger import column_traces as CT
from baillarger import streamlines as S

WIN = 15  # samples: 0.6 mm


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def angle(a, b, signed=True):
    c = (unit(a) * unit(b)).sum(-1)
    return np.degrees(np.arccos(np.clip(c if signed else np.abs(c), -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fig")
    ap.add_argument("work")
    ap.add_argument("--name", required=True)
    ap.add_argument("--box", type=int, nargs=3, required=True)
    ap.add_argument("--size", type=int, default=141)
    ap.add_argument("--dseg", help="the corrected segmentation (python -m baillarger.white dseg)")
    ap.add_argument("--examples", type=int, default=6)
    A = ap.parse_args()
    Z = np.load(f"{A.work}/turn_diag_{A.name}.npz")
    lo = tuple(A.box)
    hi = tuple(a + A.size for a in lo)
    dseg = Path(A.dseg) if A.dseg else None
    t = CT.load_tissue(lo, hi, A.name, dseg=dseg)
    vox = t.vox
    base = CT.base_tensor(t)
    _, coh = CT.data_normal(base, t.ctx)
    T, _ = S.fitted_field(base, t.ctx, t.wm, vox)
    del base
    g = np.stack(S.gradient(T, vox), -1)
    gn = np.linalg.norm(g, axis=-1)
    dom = t.ctx | (t.wm & ndi.binary_dilation(t.ctx))
    d_edge = ndi.distance_transform_edt(dom)  # voxels from the domain's edge (inside it)
    starts, xyz = Z["path_start"], Z["path_xyz"]
    it, cat, keep = Z["it"], Z["cat"], Z["keep"]
    names = [str(c) for c in Z["categories"]]
    rows = []
    for j, r in enumerate(it):
        if not keep[r]:
            continue
        p = xyz[starts[r]:starts[r + 1]].astype(float)
        n = len(p)
        if n < 2 * 8 + 2:
            continue
        lag = 8  # 0.3 mm
        d_in = p[n - 1 - lag] - p[n - 1 - 2 * lag]
        d_out = p[n - 1] - p[n - 1 - lag]
        w = p[max(0, n - 1 - WIN):]
        gw = np.stack([ndi.map_coordinates(g[..., a], w.T, order=1) for a in range(3)], -1)
        gt = gw[-1]
        rows.append(dict(
            r=int(r), cat=int(cat[j]), s_mm=(n - 1) * S.STEP_MM,
            turn_deg=float(angle(d_in, d_out)),
            in_vs_grad=float(angle(d_in, gw[-1 - lag], signed=False)),
            out_vs_grad=float(angle(d_out, gt, signed=False)),
            grad_at_turn=float(np.linalg.norm(gt)),
            grad_min=float(np.linalg.norm(gw, axis=1).min()),
            edge_vox=float(ndi.map_coordinates(d_edge, p[-1][:, None], order=1)[0]),
            edge_min_vox=float(ndi.map_coordinates(d_edge, w.T, order=1).min()),
            coh_mean=float(ndi.map_coordinates(coh, w.T, order=1).mean()),
            H=float(Z["H"][r]),
        ))
    out = {}
    for c in range(len(names)):
        R = [x for x in rows if x["cat"] == c]
        if not R:
            continue
        out[names[c]] = dict(n=len(R), **{k: np.percentile([x[k] for x in R], [10, 50, 90]).round(3).tolist()
                                            for k in R[0] if k not in ("r", "cat")})
        R2 = [x for x in R if x["H"] > 0.3]
        if R2:
            out[names[c] + ", crowns (H > 0.3)"] = dict(
                n=len(R2), **{k: np.percentile([x[k] for x in R2], [10, 50, 90]).round(3).tolist()
                              for k in R2[0] if k not in ("r", "cat")})
    print(json.dumps(out, indent=1))
    json.dump(dict(per_category=out, turns=rows), open(f"{A.work}/turn_cases_{A.name}.json", "w"))

    # ---- examples: local slices through the turn, in the plane of the turn
    rng = np.random.default_rng(0)
    for c in (names.index("interior"), names.index("near fluid")):
        R = [x for x in rows if x["cat"] == c and (c != names.index("interior") or x["H"] > 0.3)]
        if not R:
            continue
        pick = rng.choice(len(R), min(A.examples, len(R)), replace=False)
        fig, axs = plt.subplots(2, len(pick), figsize=(4 * len(pick), 8), squeeze=False)
        for col, i in enumerate(pick):
            x = R[i]
            r = x["r"]
            p = xyz[starts[r]:starts[r + 1]].astype(float)
            ax_ = int(np.argmin(np.ptp(p, 0)))  # the plane the path lies most in
            k = int(round(p[-1, ax_]))
            o = [a for a in range(3) if a != ax_]
            c0 = np.round(p[-1, o]).astype(int)
            half = 18
            sl = [slice(None)] * 3
            sl[ax_] = k
            sl[o[0]] = slice(max(c0[0] - half, 0), c0[0] + half)
            sl[o[1]] = slice(max(c0[1] - half, 0), c0[1] + half)
            org = np.array([sl[o[0]].start, sl[o[1]].start])
            for row, (img, cmap, title) in enumerate(((T, "viridis", "T (mm)"), (t.det[1], "gray", "echo 2"))):
                a = axs[row, col]
                im = np.asarray(img[tuple(sl)]).T
                vmax = np.nanpercentile(np.abs(t.det[1][t.ctx]), 97) if row else None
                a.imshow(im, origin="lower", cmap=cmap, vmin=-vmax if row else None, vmax=vmax)
                lab = np.asarray(t.labels[tuple(sl)]).T
                a.contour(lab == 2, levels=[0.5], colors="#1baf7a", linewidths=0.8)
                a.contour(lab == 1, levels=[0.5], colors="#30d5ff", linewidths=0.8)
                a.contour(np.asarray(dom[tuple(sl)]).T, levels=[0.5], colors="#ff2bd6", linewidths=0.5,
                          linestyles="dotted")
                # neighbouring paths through this window (roots within 3 voxels of this one's)
                x0 = Z["v"][Z["tv"][r]]
                near = np.flatnonzero(np.linalg.norm(Z["v"][Z["tv"]] - x0, axis=1) < 4)
                for q in near[:60]:
                    pq = xyz[starts[q]:starts[q + 1]]
                    a.plot(pq[:, o[0]] - org[0], pq[:, o[1]] - org[1], "-", c="w", lw=0.4, alpha=0.6)
                a.plot(p[:, o[0]] - org[0], p[:, o[1]] - org[1], "-", c="#e34948", lw=1.5)
                a.plot(p[-1, o[0]] - org[0], p[-1, o[1]] - org[1], "x", c="#e34948", ms=8)
                a.set_title(f"{title}; root {r}, H {x['H']:.2f}\n|grad T| {x['grad_at_turn']:.2f}, "
                            f"edge {x['edge_vox']:.1f} vox, coh {x['coh_mean']:.2f}", fontsize=8)
                a.set_xticks([])
                a.set_yticks([])
        fig.suptitle(f"{A.name}: {names[c]} turns (red; white: neighbouring roots' paths; green WM, "
                     f"cyan fluid, magenta dotted: T's domain)", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{A.fig}/turn_cases_{A.name}_{names[c].replace(' ', '_')}.png", dpi=80)
        plt.close(fig)


if __name__ == "__main__":
    main()
