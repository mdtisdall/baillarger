"""What the colour sweeps leave behind: the spatial structure of each tube's
mean τ change per sweep (diagnostic 7).

usage: tube_sweep_diag.py RUN_DIR OUT_DIR
Reads RUN_DIR/sweeps_dtau.npz (per sweep, the mean Δτ of each tube's
unknowns) and RUN_DIR/result.npz (patch centres). For each sweep: the
normalised variogram γ(d) / var(Δτ) over the patch centres in distance bins
(about 1 everywhere: no spatial structure, the sweeps' local smoothing is
working; well below 1 at short range: the remaining change is smooth over
many tubes, a coarse-level problem), Moran's I over the patch adjacency, and
a map of Δτ for the last sweep of each step. Writes sweep_diag.json and
sweep_diag.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

BINS = np.array([0.0, 0.6, 1.0, 2.0, 4.0, 8.0])


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(run / "sweeps_dtau.npz")
    r = np.load(run / "result.npz")
    vox = None
    cen = r["vertex_vox"][r["centre"]]
    zaff = np.load(run / "result.npz")
    vox = np.array([0.16027875, 0.16, 0.16027875])  # crop voxel size (mm); scale only
    pmm = cen * vox
    tree = cKDTree(pmm)
    pairs = tree.query_pairs(BINS[-1], output_type="ndarray")
    dist = np.linalg.norm(pmm[pairs[:, 0]] - pmm[pairs[:, 1]], axis=1)
    b = np.digitize(dist, BINS) - 1
    adj = r["adj"]
    rows = []
    for k in range(len(d["step"])):
        x = d["dtau"][k].astype(float)
        ok = np.isfinite(x)
        if ok.sum() < 10:
            continue
        v = np.var(x[ok])
        m = ok[pairs[:, 0]] & ok[pairs[:, 1]]
        dd = (x[pairs[m, 0]] - x[pairs[m, 1]]) ** 2 / 2
        gam = [float(dd[b[m] == i].mean() / v) if np.any(b[m] == i) else float("nan") for i in range(len(BINS) - 1)]
        ma = ok[adj[:, 0]] & ok[adj[:, 1]]
        xc = x - np.nanmean(x)
        moran = float(np.sum(xc[adj[ma, 0]] * xc[adj[ma, 1]]) / ma.sum() / np.mean(xc[ok] ** 2))
        rows.append({"sweep": k, "step": int(d["step"][k]), "full": bool(d["full"][k]), "tubes": int(ok.sum()),
                     "sd_mm": float(np.sqrt(v)), "mean_mm": float(np.nanmean(x)), "gamma_norm": gam, "moran_I": moran})
    (out / "sweep_diag.json").write_text(json.dumps({"bins_mm": BINS.tolist(), "rows": rows}, indent=1))
    steps = sorted({r_["step"] for r_ in rows})
    fig, axs = plt.subplots(1, 3 + min(3, len(steps)), figsize=(5 * (3 + min(3, len(steps))), 4.6))
    cm = plt.get_cmap("viridis")
    mid = 0.5 * (BINS[1:] + BINS[:-1])
    for r_ in rows:
        axs[0].plot(mid, r_["gamma_norm"], "-", color="C3" if r_["full"] else cm(r_["step"] / max(steps)), alpha=0.6)
    axs[0].axhline(1, color="k", ls=":")
    axs[0].set_xlabel("patch-centre distance (mm)")
    axs[0].set_title("γ(d) / var(Δτ) per sweep (red: full depth)")
    axs[1].plot([r_["moran_I"] for r_ in rows], "o-", ms=3)
    axs[1].set_title("Moran's I of Δτ over patch adjacency, per sweep")
    axs[1].set_xlabel("sweep (all steps in order)")
    axs[2].semilogy([r_["sd_mm"] for r_ in rows], "o-", ms=3, label="SD")
    axs[2].semilogy(np.abs([r_["mean_mm"] for r_ in rows]), "s-", ms=3, label="|mean|")
    axs[2].legend()
    axs[2].set_title("Δτ per sweep over tubes (mm)")
    pick = [s for s in steps][-3:]
    for a, st in zip(axs[3:], pick):
        k = max(i for i, r_ in enumerate(rows) if r_["step"] == st)
        x = d["dtau"][rows[k]["sweep"]]
        ok = np.isfinite(x)
        lim = np.nanpercentile(np.abs(x), 98) + 1e-9
        sc = a.scatter(pmm[ok, 0], pmm[ok, 1], c=x[ok], s=2, cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(sc, ax=a, label="Δτ (mm)")
        a.set_title(f"step {st}, last sweep: Δτ at patch centres (x-y)")
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "sweep_diag.png", dpi=95)
    print(json.dumps(rows[-3:], indent=1))


if __name__ == "__main__":
    main()
