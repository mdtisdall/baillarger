"""Where streamlines.crest_kinks puts T's crest in a site, for the eye: the three marked planes
(the plane axis and the three planes holding most marked points, as column_volume.py), echo 2
(trend removed) with the WM (green) and fluid (cyan) outlined, crest voxels in red, marked points
as stars. Built as the crest and traverse methods build T (extended, sign through tissue).

Writes <figures>/crest_planes_<name>.png.
Usage: crest_planes.py <figures dir> --name NAME --box I J K --points FILE [--dseg FILE]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baillarger import column_traces as CT
from baillarger import streamlines as S


def planes_of(P):
    ax = int(np.argmin([len(np.unique(np.round(P[:, a]))) for a in range(3)]))
    ks = np.unique(np.round(P[:, ax]).astype(int))
    cnt = sorted(
        [(int(np.sum(np.abs(P[:, ax] - k) <= 2)), int(k)) for k in ks], reverse=True
    )
    return ax, sorted(k for _, k in cnt[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fig")
    ap.add_argument("--name", required=True)
    ap.add_argument("--box", type=int, nargs=3, required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument(
        "--dseg", help="the corrected segmentation (python -m baillarger.white dseg)"
    )
    A = ap.parse_args()
    lo = tuple(A.box)
    hi = tuple(a + 141 for a in lo)
    dseg = Path(A.dseg) if A.dseg else None
    t = CT.load_tissue(lo, hi, A.name, dseg=dseg)
    base = CT.base_tensor(t)
    T, _ = S.fitted_field(
        base, t.ctx, t.wm, t.vox, fluid=t.labels == 1, wm_band_mm=CT.WM_BAND_MM
    )
    noise, _ = S.step_noise(
        base, t.ctx, S.tissue_wm_direction(t.wm, t.ctx, t.vox), t.vox
    )
    del base
    v, f, nrm = CT.white_mesh(t.wm)
    kappa = CT.level_set_concavity(v, f, nrm, T, t.ctx, t.vox)
    free = S.fitted_free(
        t.ctx, t.wm, t.vox, fluid=t.labels == 1, wm_band_mm=CT.WM_BAND_MM
    )
    crest, _ = S.crest_kinks(T, t.ctx, noise, kappa, t.vox, free=free)
    P = np.array(json.load(open(A.points))["points_full0"], float) - np.array(lo)
    P = P[np.all((P >= 0) & (P < 141), 1)]
    ax, planes = planes_of(P)
    img = t.det[1]
    vmax = np.nanpercentile(np.abs(img[t.ctx]), 97)
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.5))
    for a, k in zip(axs, planes):
        sl = [slice(None)] * 3
        sl[ax] = k
        sl = tuple(sl)
        a.imshow(
            np.asarray(img[sl]).T, origin="lower", cmap="gray", vmin=-vmax, vmax=vmax
        )
        lab = np.asarray(t.labels[sl]).T
        a.contour(lab == 2, levels=[0.5], colors="#1baf7a", linewidths=0.7)
        a.contour(lab == 1, levels=[0.5], colors="#30d5ff", linewidths=0.7)
        cy, cx = np.nonzero(np.asarray(crest[sl]).T)
        a.plot(cx, cy, "s", c="#e34948", ms=1.6, mew=0)
        o = [i for i in range(3) if i != ax]
        pp = np.abs(P[:, ax] - k) <= 2
        a.plot(P[pp, o[0]], P[pp, o[1]], "*", c="#ffd23f", ms=9, mec="k", mew=0.4)
        share = crest[sl][t.ctx[sl]].mean()
        a.set_title(
            f"plane {'ijk'[ax]} = {k}: crest voxels (red) {share:.1%} of cortex",
            fontsize=10,
        )
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(
        f"{A.name}: T's crest (crest_kinks) over echo 2; WM green, fluid cyan, marked points yellow",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(f"{A.fig}/crest_planes_{A.name}.png", dpi=90)


if __name__ == "__main__":
    main()
