"""The real data's features for the fold phantoms (fold_phantoms.py): per site, the median open
profile of echo 2 and 3 (log, trend removed as curv_lib.load) against relative depth, the WM and
fluid levels on the same scale, the high-pass noise statistic of the trend-removed echoes in the
cortex, the open thickness, and the coherence. Light (one box load, about 1 GB).

Usage: fold_measure.py <work dir>   (writes <work>/fold_features.json)
"""

import json
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np
from scipy import ndimage as ndi

import curv_lib as C
import flatten_fused2 as F

WORK = sys.argv[1]
SITES = {"V1V2_banks": (198, 58, 65), "M1S1": (378, 224, 448)}
NBIN = 50


def hp_stat(det, ctx):
    """The high-pass statistic: 1.4826 MAD of det - mconv(det, ctx, 1 voxel) over the cortex at least
    one voxel inside its edge. fold_phantoms matches its noise to this."""
    m = ctx.astype(float)
    r = det - F.mconv(det, m, 1.0)
    inner = ndi.binary_erosion(ctx)
    v = r[inner]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def main():
    out = {}
    for site, lo in SITES.items():
        lo = np.array(lo)
        B = C.load(lo, lo + 141, site)
        Z = np.load(f"{WORK}/column_volume_{site}.npz")
        ctx, wm, rim = B["ctx"], B["wm"], B["rim"]
        mb = ctx.astype(float)
        res = dict()
        keep = (Z["cls"] == 0) & (Z["tracer"] == 0) & ~Z["near_face"] & (Z["ns"] > 20)
        idx = np.flatnonzero(keep)
        L = (Z["ns"][idx] - 1) * 0.04
        res["open_thickness_p10_50_90"] = np.percentile(L, [10, 50, 90]).round(3).tolist()
        pos = Z["pos"][idx]
        for e in (1, 2):
            lg = np.log(np.maximum(B["p"].echoes[..., e].astype(float), 1.0))
            trend = F.mconv(lg, mb, F.SHP)
            dall = lg - trend  # trend-removed everywhere (the trend is the cortex's)
            det = dall * mb
            # profile along open traces against relative depth (centre line, as the column volume)
            prof = np.full((idx.size, NBIN), np.nan)
            edges = np.linspace(0, 1, NBIN + 1)
            for a, t in enumerate(idx):
                p = pos[a, :Z["ns"][t]].astype(float)
                vals = ndi.map_coordinates(det, p.T, order=1, mode="nearest")
                rel = np.arange(p.shape[0]) / (p.shape[0] - 1)
                b = np.minimum((rel * NBIN).astype(int), NBIN - 1)
                s = np.bincount(b, vals, NBIN)
                c = np.bincount(b, minlength=NBIN)
                prof[a] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            med = np.nanmedian(prof, 0)
            d_ctx = ndi.distance_transform_edt(~ctx)
            wm_lvl = float(np.median(dall[wm & (d_ctx >= 2) & (d_ctx <= 3)]))
            fl_lvl = float(np.median(dall[(rim == 1) & (d_ctx >= 2) & (d_ctx <= 3)]))
            res[f"echo{e + 1}"] = dict(profile=med.round(5).tolist(), wm_level=round(wm_lvl, 5),
                                       fluid_level=round(fl_lvl, 5), hp_stat=round(hp_stat(det, ctx), 5),
                                       profile_range=round(float(np.nanmax(med) - np.nanmin(med)), 5))
        coh = Z["coh"][ctx]
        res["coh_p10_25_50_75_90"] = np.percentile(coh, [10, 25, 50, 75, 90]).round(3).tolist()
        out[site] = res
        print(site, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "profile"})
                                for k, v in res.items()}, indent=1), flush=True)
    out["bins"] = NBIN
    json.dump(out, open(f"{WORK}/fold_features.json", "w"), indent=1)


if __name__ == "__main__":
    main()
