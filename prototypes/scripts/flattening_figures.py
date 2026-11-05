"""Figures of phase 3.1 from the outputs of flattening_region.py.

    flattening_figures.py decay OUT_PNG REGION.json [REGION.json ...]
    flattening_figures.py local OUT_PNG REGION.json

decay: the largest |dr| over levels, per distance bin along the mesh edges,
for a perturbation of one pair's shift and of all the pairs of one column
(as a fraction of the perturbation), with the 1 % line.
local: the difference between the local solve and the region's solve, for
the owned cube, against the halo.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def decay(out: Path, files: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for f in files:
        d = json.loads(f.read_text())
        c = d["decay"]
        delta = 0.08
        x = np.array(c["bin_mm"]) + 0.125
        for name, ls in (("pair", "-"), ("column", "--")):
            ax.semilogy(
                x, np.array(c[name]) / delta, ls, label=f"{d['name']}: one {name}"
            )
    ax.axhline(0.01, color="k", lw=0.8)
    ax.text(0.2, 0.012, "1 % of the perturbation (0.08 mm = dtau)", fontsize=8)
    ax.set_xlim(0, 12)
    ax.set_xlabel("distance from the perturbed column along the mesh edges (mm)")
    ax.set_ylabel("max over levels |dr| / perturbation")
    ax.set_title("Decay of the warp's response to a perturbed shift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=100)


def local(out: Path, f: Path) -> None:
    d = json.loads(f.read_text())
    rows = d["local"]["rows"]
    h = [r["halo_mm"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, lab in ((0, "median"), (1, "99th percentile"), (2, "largest")):
        ax.semilogy(h, [r["abs_dr_p50_p99_max"][i] for r in rows], "o-", label=lab)
    ax.axhline(0.0008, color="k", lw=0.8)
    ax.text(h[0], 0.0009, "1 % of dtau", fontsize=8)
    ax.set_xlabel("halo (mm)")
    ax.set_ylabel("|r local - r region| over owned columns and levels (mm)")
    ax.set_title(f"{d['name']}: a {d['local']['cube_mm']} mm cube with a halo")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=100)


if __name__ == "__main__":
    kind, png = sys.argv[1], Path(sys.argv[2])
    png.parent.mkdir(parents=True, exist_ok=True)
    if kind == "decay":
        decay(png, [Path(a) for a in sys.argv[3:]])
    else:
        local(png, Path(sys.argv[3]))
