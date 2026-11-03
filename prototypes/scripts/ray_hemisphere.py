"""The ray sampler on a hemisphere: the diagnostics of phase 2.0.

    python prototypes/scripts/ray_hemisphere.py SURF DSEG OUT [--pieces] [--level M]
        [--only N] [--overview OVERVIEW] [--cache-gb G]

Samples every superblock that the surface crosses (or the first N, in Morton
order) at the full caps and the default step, and writes to OUT:

- `classes.npy`: the derived class of each vertex (uint8, 0 where no call
  gave the vertex);
- `summary.json`: the counts of each derived class; the classes against the
  normal length |n̄| (bins); the fraction of roots whose ray goes from its WM
  straight into outside with no cortex between (cut faces); the time and the
  peak memory of each call and of the pass;
- `wm_to_outside.npy`: per vertex, whether its ray goes from WM into
  outside with no cortex between;
- with `--pieces`, after a pass, in its own process: the counts by
  connected piece of the surface, added to `summary.json`;
- with `--overview`, `classes_overview.png`: the derived classes on the
  overview mesh, from four sides.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

from baillarger import rays
from baillarger.blocks import morton

NAMES = {rays.OPEN: "open", rays.IN_WM: "in WM", rays.NEITHER: "neither"}
BINS = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0001)


def peak_gb() -> float:
    """The peak resident memory of this process so far, GB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 30) if sys.platform == "darwin" else r / (1 << 20)


def wm_to_outside(r: rays.Rays) -> npt.NDArray[np.bool_]:
    """Per column: the first outward interval that is not WM is outside (the
    ray leaves its WM into outside, with no cortex between)."""
    out = np.zeros(len(r.roots.vertex), bool)
    for c in range(len(out)):
        for q in range(r.interval_start[c], r.interval_start[c + 1]):
            if r.interval_s1[q] <= 0 or r.interval_class[q] == rays.WM:
                continue
            out[c] = r.interval_class[q] == rays.OUTSIDE
            break
    return out


def piece_labels(surf: Path, n: int) -> npt.NDArray[np.int64]:
    """The connected piece of each vertex (before the pass, so that its
    memory is freed before the sampler's)."""
    from nibabel.gifti.gifti import GiftiImage

    from baillarger.mesh_smoothing import pieces

    g = GiftiImage.from_filename(str(surf), mmap=True)
    f = np.asarray(g.darrays[1].data, np.int64)
    vlab, _ = pieces(f, n)
    return vlab


def pieces_table(
    vlab: npt.NDArray[np.int64], cls: npt.NDArray[np.uint8], cut: npt.NDArray[np.bool_]
) -> list[dict[str, int]]:
    """Per connected piece of the surface, largest first: its roots, their
    derived classes and the roots whose ray goes from WM into outside."""
    out = []
    for p in np.argsort(-np.bincount(vlab[vlab >= 0]), kind="stable"):
        sel = vlab == p
        row = {"piece": int(p), "roots": int(sel.sum())}
        row |= {NAMES[k]: int(np.sum(cls[sel] == k)) for k in NAMES}
        row["wm_to_outside"] = int(cut[sel].sum())
        out.append(row)
    return out


def add_pieces(surf: Path, out: Path) -> None:
    """The counts by piece, from a pass's saved arrays, in their own process
    (the piece labels of the whole mesh need several GB)."""
    cls = np.load(out / "classes.npy")
    cut = np.load(out / "wm_to_outside.npy")
    vlab = piece_labels(surf, len(cls))
    summary = json.loads((out / "summary.json").read_text())
    summary["by_piece"] = pieces_table(vlab, cls, cut)
    summary["pieces_peak_gb"] = round(peak_gb(), 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("surf", type=Path)
    p.add_argument("dseg", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--only", type=int, default=0)
    p.add_argument("--overview", type=Path)
    p.add_argument("--cache-gb", type=float, default=0.25)
    p.add_argument(
        "--pieces",
        action="store_true",
        help="after a pass: add the counts by piece of the surface to summary.json",
    )
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.pieces:
        add_pieces(a.surf, a.out)
        return
    t0 = time.perf_counter()
    sampler = rays.RaySampler(a.surf, a.dseg, int(a.cache_gb * (1 << 30)))
    t_open = time.perf_counter() - t0
    m = a.level
    sbs = np.unique(sampler.block_ijk >> m, axis=0)
    sbs = sbs[np.argsort(morton(sbs), kind="stable")]
    if a.only:
        sbs = sbs[: a.only]
    n = int(sampler.sidecar["Vertices"])
    cls = np.zeros(n, np.uint8)
    nl = np.full(n, np.nan, np.float32)
    cut = np.zeros(n, bool)
    calls = []
    for sb in sbs:
        t = time.perf_counter()
        r = sampler.sample((int(sb[0]), int(sb[1]), int(sb[2])), m)
        dt = time.perf_counter() - t
        v = r.roots.vertex
        cls[v] = r.derived_class
        nl[v] = r.roots.normal_length
        cut[v] = wm_to_outside(r)
        calls.append(
            dict(
                superblock=sb.tolist(),
                columns=len(v),
                samples=len(r.s),
                intervals=len(r.interval_s0),
                seconds=round(dt, 3),
                peak_gb=round(peak_gb(), 3),
            )
        )
        print(json.dumps(calls[-1]), flush=True)
        del r
    done = cls > 0
    counts = {NAMES[k]: int(np.sum(cls == k)) for k in NAMES}
    table = {}
    if np.isfinite(nl[done]).any():
        idx = np.digitize(nl, BINS) - 1
        for b in range(len(BINS) - 1):
            sel = done & (idx == b)
            table[f"[{BINS[b]}, {min(BINS[b + 1], 1.0)})"] = {
                NAMES[k]: int(np.sum(cls[sel] == k)) for k in NAMES
            } | {"columns": int(sel.sum())}
    summary = dict(
        level=m,
        superblocks=len(sbs),
        columns=int(done.sum()),
        vertices=n,
        counts=counts,
        by_normal_length=table,
        wm_to_outside=int(cut[done].sum()),
        wm_to_outside_fraction=float(cut[done].mean()) if done.any() else 0.0,
        cache_gb=a.cache_gb,
        chunk_reads=sampler.labels.reads,
        open_seconds=round(t_open, 3),
        pass_seconds=round(time.perf_counter() - t0, 3),
        peak_gb=round(peak_gb(), 3),
        calls=calls,
    )
    np.save(a.out / "classes.npy", cls)
    np.save(a.out / "wm_to_outside.npy", cut)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    if a.overview:
        draw(
            a.overview,
            [
                ("derived class", cls, ["#bbbbbb", "#2a7fd4", "#d4552a", "#e8c547"]),
                ("WM into outside", cut.astype(np.uint8), ["#bbbbbb", "#7a1fa2"]),
            ],
            a.out / "classes_overview.png",
        )


def draw(
    overview: Path,
    values: list[tuple[str, npt.NDArray[np.uint8], list[str]]],
    png: Path,
) -> None:
    """Per-vertex values on the overview mesh (each overview vertex shows its
    true vertex): one row per value, orthographic views from both sides of
    each axis of the scanner, faces drawn far to near."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import ListedColormap
    from nibabel.gifti.gifti import GiftiImage

    g = GiftiImage.from_filename(str(overview))
    x, f = g.agg_data(("pointset", "triangle"))
    x = np.asarray(x, np.float64)
    f = np.asarray(f, np.int64)
    name = overview.name.replace("desc-overview_", "desc-overviewids_")
    name = name.replace(".surf.gii", ".func.gii")
    ids = GiftiImage.from_filename(str(overview.with_name(name))).darrays[0].data
    ids = np.asarray(ids).reshape(-1)
    views = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    labels = ["+x", "-x", "+y", "-y", "+z", "-z"]
    fig, axes = plt.subplots(
        len(values), len(views), figsize=(4 * len(views), 4.5 * len(values))
    )
    axes = np.atleast_2d(axes)
    for row, (title, val, colours) in enumerate(values):
        c = val[ids]
        face = np.array([np.bincount(t, minlength=len(colours)).argmax() for t in c[f]])
        cmap = ListedColormap(colours)
        for col, (axis, sign) in enumerate(views):
            ax = axes[row, col]
            plane = [a for a in range(3) if a != axis]
            depth = sign * x[f, axis].mean(1)
            o = np.argsort(depth)  # far first
            pc = PolyCollection(x[f[o]][:, :, plane], cmap=cmap, linewidths=0)
            pc.set_array(face[o].astype(float))
            pc.set_clim(-0.5, len(colours) - 0.5)
            ax.add_collection(pc)
            ax.set_xlim(x[:, plane[0]].min(), x[:, plane[0]].max())
            ax.set_ylim(x[:, plane[1]].min(), x[:, plane[1]].max())
            ax.set_aspect("equal")
            if sign > 0:
                ax.invert_xaxis()
            ax.set_title(f"{title}, seen from {labels[col]}", fontsize=9)
            ax.set_axis_off()
    fig.savefig(png, dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
