"""Wave 2.1, Part A: the maps of the white surface's smoothed normals
(baillarger.normal_maps) on a surface from white_surf, and the outputs for the
review (docs/plans/wave-2.1-normals.md).

Writes, in OUT: normal_maps.func.gii (one named scalar map for each of
normal_maps.MAPS, as external binary data, so the viewer's --overlay reads it)
and run.json (the input set, the times, the peak memory). In FIG:

- hist_<map>.png: the histogram of each map, with its median and its 99th and
  99.9th percentiles, and the share beyond a few round values (for
  description, not pass or fail); stats.json holds the same numbers.
- overview_<map>.png: the map on the overview, lateral and medial. Each
  overview face shows the largest value of its corners and of the true
  vertices that it carries.
- ranked_<map>.tsv: the ranked list (normal_maps.ranked_table): 200 sites,
  1 mm apart, highest first (normlen: lowest first, where the normals
  cancel).
- density_nbrmax_normlen.png, density_renorm_normlen.png.

usage:
    normal_maps.py SURF OUT FIG
"""

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from baillarger import normal_maps as NM  # noqa: E402
from baillarger.bids import file_sha256, sidecar_path  # noqa: E402
from baillarger.gifti import DataArray, write_external  # noqa: E402
from baillarger.white_overview import (  # noqa: E402
    WhiteOverview,
    overview_paths,
    surface_points,
)
from baillarger.white_surf import WhiteSurface  # noqa: E402

#: the round values of the histograms: the share above each (normlen: below)
ROUND = dict(
    nbrmax=(10, 30, 60),
    nbrmaxpermm=(30, 100, 300),
    normlen=(0.5, 0.8, 0.95),
    renorm=(1, 5, 20),
    smoothing=(10, 30, 60),
    misfacingraw=(1, 2, 3),
    misfacingsmooth=(1, 2, 3),
)
SITES, SPACING = 200, 1.0
#: the end of each map that the ranked list starts from (default: high)
ORDER = dict(normlen="low")


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes


def face_max(ov: WhiteOverview, a: np.ndarray) -> np.ndarray:
    """For each overview face, the largest finite value of its corners and of
    the true vertices that it carries (NaN if none)."""
    nf = len(ov.f)
    out = np.full(nf, np.nan, np.float32)
    for k in range(3):
        out = np.fmax(out, a[ov.vertex_ids[ov.f[:, k]]])
    step = 100_000
    for f0 in range(0, nf, step):
        ids, face = ov.carried(np.arange(f0, min(f0 + step, nf)))
        np.fmax.at(out, face, a[ids])
    return out


def render(
    ax: Any,
    v: np.ndarray,
    f: np.ndarray,
    val: np.ndarray,
    view: np.ndarray,
    lo: float,
    hi: float,
) -> None:
    """The faces seen from +view (up: +z), painter's order, flat Lambert shade
    times viridis of (val - lo) / (hi - lo); grey where val is NaN."""
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, view)
    right /= np.linalg.norm(right)
    up2 = np.cross(view, right)
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    front = nrm @ view > 0
    o = np.argsort((tri.mean(1) @ view)[front])
    t = tri[front][o]
    x = val[front][o]
    col = plt.get_cmap("viridis")(np.clip((x - lo) / (hi - lo), 0, 1))
    col[~np.isfinite(x)] = (0.6, 0.6, 0.6, 1)
    col[:, :3] *= (0.4 + 0.6 * np.abs(nrm[front][o] @ view))[:, None]
    xy = np.stack([t @ right, t @ up2], -1)
    ax.add_collection(PolyCollection(xy, facecolors=col, edgecolors="none"))
    ax.set_xlim(xy[..., 0].min(), xy[..., 0].max())
    ax.set_ylim(xy[..., 1].min(), xy[..., 1].max())
    ax.set_aspect("equal")
    ax.axis("off")


def stats(name: str, x: np.ndarray) -> dict[str, Any]:
    fin = x[np.isfinite(x)]
    p50, p99, p999 = (float(q) for q in np.percentile(fin, [50, 99, 99.9]))
    share = {}
    for r in ROUND[name]:
        if name == "normlen":
            share[f"below {r}"] = float(np.mean(fin < r))
        elif name.startswith("mis"):
            share[f"at least {r}"] = float(np.mean(fin >= r))
        else:
            share[f"above {r}"] = float(np.mean(fin > r))
    return dict(
        unit=NM.UNITS[name],
        finite=int(len(fin)),
        nan=int(len(x) - len(fin)),
        min=float(fin.min()),
        median=p50,
        p99=p99,
        p999=p999,
        max=float(fin.max()),
        share=share,
    )


def histogram(
    fig: Path, name: str, x: np.ndarray, st: dict[str, Any], label: str
) -> None:
    fin = x[np.isfinite(x)]
    f, ax = plt.subplots(figsize=(6.4, 3.6), dpi=90)
    if name.startswith("mis"):
        k = np.arange(0, int(fin.max()) + 2) - 0.5
        ax.hist(fin, bins=k, color="#1f5f99")
    else:
        hi = max(st["max"], st["p999"])
        ax.hist(fin, bins=200, range=(min(0.0, st["min"]), hi), color="#1f5f99")
    ax.set_yscale("log")
    for q, s in (("median", "-"), ("p99", "--"), ("p999", ":")):
        ax.axvline(st[q], color="#c25e1f", ls=s, lw=1.2, label=f"{q} {st[q]:.3g}")
    unit = f" ({st['unit']})" if st["unit"] else ""
    ax.set_xlabel(name + unit)
    ax.set_ylabel("vertices")
    sh = ", ".join(f"{k}: {100 * v:.3g} %" for k, v in st["share"].items())
    ax.set_title(f"{name}, {label}\n{sh}", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    f.tight_layout()
    f.savefig(fig / f"hist_{name}.png")
    plt.close(f)


def overview_figure(
    fig: Path,
    name: str,
    ov: WhiteOverview,
    vmm: np.ndarray,
    a: np.ndarray,
    st: dict[str, Any],
    hemi: str,
    label: str,
) -> None:
    val = face_max(ov, a)
    if name.startswith("mis"):
        lo, hi = 0.0, 3.0
    elif name == "normlen":
        lo, hi = float(np.nanpercentile(a, 0.1)), 1.0
    else:
        lo, hi = 0.0, st["p999"]
    lat = np.array([1.0, 0, 0]) if hemi == "R" else np.array([-1.0, 0, 0])
    f, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=80)
    for k, (view, t) in enumerate(((lat, "lateral"), (-lat, "medial"))):
        render(ax[k], vmm, ov.f, val, view, lo, hi)
        ax[k].set_title(t, fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(lo, hi))
    unit = f" ({st['unit']})" if st["unit"] else ""
    f.colorbar(sm, ax=ax, shrink=0.6, label=name + unit)
    f.suptitle(
        f"{name}: the largest value under each overview face, {label}", fontsize=10
    )
    f.savefig(fig / f"overview_{name}.png")
    plt.close(f)


def density(fig: Path, mp: dict[str, np.ndarray], x: str, y: str, label: str) -> None:
    ok = np.isfinite(mp[x]) & np.isfinite(mp[y])
    f, ax = plt.subplots(figsize=(5.6, 4.4), dpi=90)
    hi = float(np.percentile(mp[x][ok], 99.9))
    h = ax.hist2d(
        mp[x][ok],
        mp[y][ok],
        bins=(200, 200),
        range=((0, hi), (0, 1)),
        norm=LogNorm(),
        cmap="viridis",
    )
    f.colorbar(h[3], ax=ax, label="vertices")
    ax.set_xlabel(f"{x} ({NM.UNITS[x]})")
    ax.set_ylabel(y)
    ax.set_title(f"{x} against {y}, {label}", fontsize=9)
    f.tight_layout()
    f.savefig(fig / f"density_{x}_{y}.png")
    plt.close(f)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0])
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append(
            "\t".join(
                f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("surf", type=Path, help="a ..._white.surf.gii (white_surf)")
    ap.add_argument("out", type=Path, help="the directory of the maps file")
    ap.add_argument("fig", type=Path, help="the directory of the figures")
    ap.add_argument("--no-figures", action="store_true", help="the maps file only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.fig.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    times: dict[str, float] = {}

    def lap(m: str) -> None:
        times[m] = round(time.time() - t0, 1)
        print(f"{times[m]:8.1f} s  {peak_gb():5.2f} GB  {m}", flush=True)

    W = WhiteSurface.load(a.surf)
    pts = surface_points(a.surf)
    side = W.sidecar
    sha = file_sha256(sidecar_path(a.surf))
    label = f"{a.surf.name.removesuffix('_white.surf.gii')} (sidecar {sha[:16]})"
    lap("load")
    mp = NM.normal_maps(
        W.v, pts, W.f, W.normals, W.normlen, int(side["NormalSmoothingSteps"]), lap
    )
    gii = a.out / "normal_maps.func.gii"
    write_external(
        gii,
        [
            DataArray(mp[k], "NIFTI_INTENT_NONE", dict(Name=k, Unit=NM.UNITS[k]))
            for k in NM.MAPS
        ],
        dict(
            Surface=a.surf.name,
            SurfaceSidecarSHA256=sha,
            Producer="prototypes/scripts/normal_maps.py (baillarger.normal_maps)",
        ),
    )
    lap("write the maps")
    run: dict[str, Any] = dict(
        surface=str(a.surf),
        sidecar_sha256=sha,
        vertices=len(W.v),
        faces=len(W.f),
        steps=int(side["NormalSmoothingSteps"]),
    )
    if not a.no_figures:
        st = {k: stats(k, mp[k]) for k in NM.MAPS}
        (a.fig / "stats.json").write_text(json.dumps(st, indent=1))
        for k in NM.MAPS:
            histogram(a.fig, k, mp[k], st[k], label)
        lap("histograms")
        ov = WhiteOverview.load(overview_paths(a.surf)["overview"])
        ov.check(W)
        vmm = ov.v @ ov.affine[:3, :3].T + ov.affine[:3, 3]
        for k in NM.MAPS:
            overview_figure(a.fig, k, ov, vmm, mp[k], st[k], side["Hemisphere"], label)
        lap("overview figures")
        keys = W.vertex_key
        for k in NM.MAPS:
            write_tsv(
                a.fig / f"ranked_{k}.tsv",
                NM.ranked_table(mp, k, pts, keys, SITES, SPACING, ORDER.get(k, "high")),
            )
            lap(f"ranked list {k}")
        density(a.fig, mp, "nbrmax", "normlen", label)
        density(a.fig, mp, "renorm", "normlen", label)
        lap("density plots")
    run.update(times=times, peak_gb=round(peak_gb(), 2))
    (a.out / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
