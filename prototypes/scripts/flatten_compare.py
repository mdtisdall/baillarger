"""Two flattening fits of one site compared (plan: reports/Crossing paths and contacts from laminar
profiles.md, *Flattening in the volume*, dynamic warping). Reads <work>/flatten_<tag>.npz and
_fit.json and .json for each tag (flatten_volume.py; both on the same traces and pairs).

1. Shifts: on samples weighted (W > 0.3) in either fit, |s_b - s_a| percentiles and the fraction over
   one sample (0.04 mm); the fraction of b's shifts beyond 0.3 mm; mean weight in each.
2. Horizons: |z_b - z_a| per trace and level, percentiles, over levels inside the trace in both fits
   (trace lengths from <work>/column_volume_<site>.npz).
3. The summaries side by side: weighted rms misfit, residual percentiles, folds, the darkest band's IQR.

Usage: flatten_compare.py <work> <site> <tag a> <tag b> [<tag c> ...]  (each later tag against the first)
"""

import json
import sys

import numpy as np

STEP = 0.04


def load(work, tag):
    Z = np.load(f"{work}/flatten_{tag}.npz")
    return Z, json.load(open(f"{work}/flatten_{tag}.json"))


def main():
    work, site, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    Za, Ja = load(work, tags[0])
    ns = np.load(f"{work}/column_volume_{site}.npz")["ns"][Za["keep"]]
    end = (ns[:, None] - 1) * STEP
    out = {}
    for tb in tags[1:]:
        Zb, Jb = load(work, tb)
        assert np.array_equal(Za["pairs"], Zb["pairs"]) and np.array_equal(Za["keep"], Zb["keep"])
        Sa, Sb = Za["S"].astype(float), Zb["S"].astype(float)
        Wa, Wb = Za["W"].astype(float), Zb["W"].astype(float)
        m = (Wa > 0.3) | (Wb > 0.3)
        d = np.abs(Sb - Sa)[m]
        za, zb = Za["z"].astype(float), Zb["z"].astype(float)
        dz = np.abs(zb - za)
        below = (za < end) & (zb < end)
        cmp = {
            "shift |b - a| p50/p90/p99 (mm), weighted samples": np.percentile(d, [50, 90, 99]).round(4).tolist(),
            "shift: fraction of weighted samples differing by > 0.04 mm": round(float(np.mean(d > STEP)), 4),
            "shift: fraction of b's weighted samples beyond 0.3 mm": round(float(np.mean(np.abs(Sb[Wb > 0.3]) > 0.3)), 4),
            "mean weight a / b (all samples)": [round(float(Wa.mean()), 4), round(float(Wb.mean()), 4)],
            "samples with weight > 0.3, a / b": [int((Wa > 0.3).sum()), int((Wb > 0.3).sum())],
            "horizons |z_b - z_a| p50/p90/p99 (mm)": np.percentile(dz[below], [50, 90, 99]).round(4).tolist(),
            "horizons: traces whose largest |z_b - z_a| > 0.1 mm": int(np.sum(np.where(below, dz, 0).max(1) > 0.1)),
        }
        out[f"{tb} vs {tags[0]}"] = cmp
    keys = ["residual per trace and level, p50/p90/p99 (mm)",
            "fold fraction (levels with dz/dtau <= 0, below each trace's end)"]
    side = {}
    for t in tags:
        Z, J = load(work, t)
        fj = json.load(open(f"{work}/flatten_{t}_fit.json"))
        dark = J["darkest band in 0.5-1.5 mm (open and crossing traces)"]
        side[t] = {"weighted rms misfit, first -> last outer (mm)":
                   [round(fj["outer"][0]["weighted_rms_mm"], 4), round(fj["outer"][-1]["weighted_rms_mm"], 4)],
                   **{k: J[k] for k in keys},
                   "darkest band IQR, depth -> tau (mm)": [dark["iqr_depth"], dark["iqr_tau"]],
                   "darkest band traces": dark["traces"]}
    out["summaries"] = side
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{work}/flatten_compare_{tags[0]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
