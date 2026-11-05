"""The hemisphere check of the WM-distance sampler (phase 1.1): the superblocks
of the test sites, their time and peak, a view of one axial slice each, and
the fraction of cortex with no path to the WM within the cap.

usage:
    wm_distance_check.py site DSEG OUT_DIR NAME I J K [--cap MM] [--level M]
    wm_distance_check.py pair DSEG OUT_DIR I J K AXIS [--cap MM] [--halo MM]
    wm_distance_check.py figure OUT_DIR FIG_DIR

site: samples the superblock (level M, 64-voxel blocks) that holds voxel
(I, J, K), writes OUT_DIR/NAME_capC.json and OUT_DIR/NAME_capC_slice.npz.
Run each site in its own process, so that the peak is the site's.
pair: samples the superblock of (I, J, K) and its neighbour along AXIS, with
a halo, and counts the voxels whose values differ (float64 and float32).
figure: one PNG of the three sites' slices (labels and distance).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from baillarger import wm_distance as W
from baillarger.timing import peak_rss_gb


def site(a: argparse.Namespace) -> None:
    s = W.WMDistanceSampler(Path(a.dseg))
    centre = np.array([a.i, a.j, a.k])
    sb = tuple(int(c) // (s.cells << a.level) for c in centre)
    out, read = s.boxes(sb, a.level, 0.0, a.cap)  # type: ignore[arg-type]
    t = time.perf_counter()
    r = s.sample(sb, a.level, 0.0, a.cap)  # type: ignore[arg-type]
    dt = time.perf_counter() - t
    lab = s.labels.read(r.box)
    ctx = lab == s.cortex
    d = r.distance
    n_read = int(np.prod([q.stop - q.start for q in read]))
    rec = {
        "name": a.name,
        "centre": centre.tolist(),
        "superblock": list(sb),
        "level": a.level,
        "cap": a.cap,
        "box": [[q.start, q.stop] for q in r.box],
        "read_box": [[q.start, q.stop] for q in read],
        "read_voxels": n_read,
        "seconds": dt,
        "peak_gb": peak_rss_gb(),
        "ns_per_read_voxel": dt / n_read * 1e9,
        "cortex_voxels": int(ctx.sum()),
        "cortex_inf": int(np.sum(ctx & (d == np.inf))),
        "cortex_inf_fraction": float(np.mean(d[ctx] == np.inf)) if ctx.any() else None,
        "nan_voxels": int(np.isnan(d).sum()),
        "wm_minus_inf": int(np.sum(d == -np.inf)),
    }
    print(json.dumps(rec, indent=1))
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    tag = f"{a.name}_cap{a.cap:g}"
    (o / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    k = int(centre[2]) - r.box[2].start
    np.savez_compressed(
        o / f"{tag}_slice.npz",
        labels=lab[:, :, k],
        distance=d[:, :, k],
        cortex=s.cortex,
        filled=s.filled,
        k=int(centre[2]),
        origin=np.array([q.start for q in r.box]),
    )


def pair(a: argparse.Namespace) -> None:
    s = W.WMDistanceSampler(Path(a.dseg))
    sb = [int(c) // (s.cells << 1) for c in (a.i, a.j, a.k)]
    nb = list(sb)
    nb[a.axis] += 1
    t = time.perf_counter()
    da, ba = s.sample64(tuple(sb), 1, a.halo, a.cap)  # type: ignore[arg-type]
    db, bb = s.sample64(tuple(nb), 1, a.halo, a.cap)  # type: ignore[arg-type]
    dt = time.perf_counter() - t
    lo = [max(p.start, q.start) for p, q in zip(ba, bb, strict=True)]
    hi = [min(p.stop, q.stop) for p, q in zip(ba, bb, strict=True)]
    x = da[tuple(slice(i - q.start, j - q.start) for i, j, q in zip(lo, hi, ba, strict=True))]
    y = db[tuple(slice(i - q.start, j - q.start) for i, j, q in zip(lo, hi, bb, strict=True))]
    same_nan = bool(np.array_equal(np.isnan(x), np.isnan(y)))
    f = np.isfinite(x) & np.isfinite(y)
    diff = np.abs(x[f] - y[f])
    x32, y32 = x[f].astype(np.float32), y[f].astype(np.float32)
    rec = {
        "superblocks": [sb, nb],
        "halo_mm": a.halo,
        "cap": a.cap,
        "overlap_voxels": int(x.size),
        "finite_both": int(f.sum()),
        "same_nan": same_nan,
        "inf_mismatch": int(np.sum(np.isinf(x) != np.isinf(y))),
        "float64_differ": int(np.sum(diff > 0)),
        "float64_max_diff_mm": float(diff.max(initial=0.0)),
        "float32_differ": int(np.sum(x32 != y32)),
        "seconds_two_calls": dt,
        "peak_gb": peak_rss_gb(),
    }
    print(json.dumps(rec, indent=1))
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / f"pair_{'_'.join(map(str, sb))}_axis{a.axis}_cap{a.cap:g}.json").write_text(
        json.dumps(rec, indent=1)
    )


def figure(a: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, TwoSlopeNorm

    o = Path(a.out)
    names = ("V1V2", "M1S1", "PCC_SPL")
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 7.2), dpi=90)
    lab_cmap = ListedColormap(["#f2f2f2", "#d9a441", "#4a7fb5", "#8c8c8c"])
    dist_cmap = matplotlib.colormaps["RdBu_r"].copy()
    dist_cmap.set_bad("#ffffff")
    im = None
    for c, n in enumerate(names):
        z = np.load(o / f"{n}_cap8_slice.npz")
        lab, d = z["labels"], z["distance"].astype(np.float64)
        cls = np.full(lab.shape, 3)
        cls[lab == 0] = 0
        cls[lab == int(z["cortex"])] = 1
        cls[np.isin(lab, z["filled"])] = 2
        ax[0, c].imshow(cls.T, origin="lower", cmap=lab_cmap, vmin=0, vmax=3, interpolation="nearest")
        ax[0, c].set_title(f"{n}: labels, axial k = {int(z['k'])}", fontsize=9)
        shown = np.where(np.isinf(d), np.nan, d)
        im = ax[1, c].imshow(
            shown.T, origin="lower", cmap=dist_cmap,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-8.0, vmax=8.0), interpolation="nearest",
        )
        inf = np.isinf(d)
        over = np.zeros((*d.shape, 4))
        over[inf & (d > 0)] = (0.15, 0.15, 0.15, 1.0)
        over[inf & (d < 0)] = (0.55, 0.0, 0.55, 1.0)
        ax[1, c].imshow(np.transpose(over, (1, 0, 2)), origin="lower", interpolation="nearest")
        ax[1, c].set_title(f"{n}: D (mm), cap 8 mm", fontsize=9)
        for r in (0, 1):
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(color="#f2f2f2", label="outside"),
            Patch(color="#d9a441", label="cortex"),
            Patch(color="#4a7fb5", label="filled WM"),
            Patch(color="#8c8c8c", label="other label"),
            Patch(color=(0.15, 0.15, 0.15), label="+inf (cortex, no path within cap)"),
            Patch(color=(0.55, 0.0, 0.55), label="-inf (WM deeper than cap)"),
            Patch(facecolor="#ffffff", edgecolor="#999999", label="NaN (obstacle)"),
        ],
        loc="lower center", ncol=4, fontsize=8, frameon=False,
    )
    assert im is not None
    cb = fig.colorbar(im, ax=ax[1, :], shrink=0.8, pad=0.01)
    cb.set_label("D, mm (negative in WM)", fontsize=8)
    fd = Path(a.fig)
    fd.mkdir(parents=True, exist_ok=True)
    fig.savefig(fd / "sites.png", bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("site")
    s.add_argument("dseg")
    s.add_argument("out")
    s.add_argument("name")
    for c in "ijk":
        s.add_argument(c, type=int)
    s.add_argument("--cap", type=float, default=W.CAP)
    s.add_argument("--level", type=int, default=1)
    q = sub.add_parser("pair")
    q.add_argument("dseg")
    q.add_argument("out")
    for c in "ijk":
        q.add_argument(c, type=int)
    q.add_argument("axis", type=int)
    q.add_argument("--cap", type=float, default=W.CAP)
    q.add_argument("--halo", type=float, default=2.0)
    f = sub.add_parser("figure")
    f.add_argument("out")
    f.add_argument("fig")
    a = p.parse_args()
    {"site": site, "pair": pair, "figure": figure}[a.cmd](a)


if __name__ == "__main__":
    main()
