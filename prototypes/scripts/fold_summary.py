"""Summary figure of the fold phantoms (fold_eval.py's JSON): per phantom and tracer, the share of
primary traces of the right class, and the path error (median and 90th percentile) with the
fallback, at the measured noise, for both sites' features.

Usage: fold_summary.py <figures dir> <work dir>   (writes <figures>/fold_summary.png)
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG, WORK = sys.argv[1:3]
COL = {"current": "#e34948", "laplace": "#2a78d6", "fitted": "#30d5ff", "fitted_blend": "#ff2bd6",
       "multiscale": "#1baf7a", "blend": "#c79a00", "perbank": "#9b59b6"}

fig, axs = plt.subplots(2, 2, figsize=(18, 9))
for j, site in enumerate(("V1V2_banks", "M1S1")):
    J = json.load(open(f"{WORK}/fold_eval_{site}_n1.json"))
    phs = list(J)
    trs = list(J[phs[0]]["summary"])
    w = 0.8 / len(trs)
    for k, tr in enumerate(trs):
        x = np.arange(len(phs)) + (k - len(trs) / 2 + 0.5) * w
        ok = [J[p]["summary"][tr]["all"]["primary_right_class"] for p in phs]
        med = [J[p]["summary"][tr]["all"].get("emax", [np.nan, np.nan])[0] for p in phs]
        p90 = [J[p]["summary"][tr]["all"].get("emax", [np.nan, np.nan])[1] for p in phs]
        axs[0, j].bar(x, ok, w, color=COL.get(tr), label=tr)
        axs[1, j].bar(x, p90, w, color=COL.get(tr), alpha=0.35)
        axs[1, j].bar(x, med, w, color=COL.get(tr), label=tr)
    for a in axs[:, j]:
        a.set_xticks(np.arange(len(phs)))
        a.set_xticklabels(phs, fontsize=8)
    axs[0, j].set_ylim(0, 1)
    axs[0, j].set_title(f"{site} features, measured noise: primary traces of the right class", fontsize=10)
    axs[1, j].set_ylim(0, 1.2)
    axs[1, j].set_title("largest distance from the true column (mm), with fallback: median (solid), p90 (light)",
                        fontsize=10)
axs[0, 0].legend(fontsize=8, ncol=4)
fig.tight_layout()
fig.savefig(f"{FIG}/fold_summary.png", dpi=72)
