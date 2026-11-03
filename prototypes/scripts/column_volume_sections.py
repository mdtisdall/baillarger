"""Seismic sections of the column volume on a hemisphere (docs/formats/column-volume.md).

One superblock that holds a marked point, at a level, with a halo in mm: the ray sampler gives
the columns (extents from the label intervals), the column-volume sampler gives the echoes in
batches of columns. For each point, one section: the columns whose root lies in a slab of one
voxel round the point, ordered along the slab's long axis, echo 2 against s, with
(a) the echo, (b) the partial-volume marks and the label changes, (c) the class of the label.

Extents (from the label intervals, only to set the demonstration): outward to the first
OUTSIDE interval after the cortex plus 0.3 mm (to the end of the cortex plus 0.3 mm if no
OUTSIDE interval follows; 1 mm with no cortex), at most the cap; inward 0.5 mm.

Usage: column_volume_sections.py <figures dir> --surf SURF --dseg DSEG_GZ --echoes E1 E2 E3
           --echo-dir DIR --points FILE --point-index N [N ...] [--level 1] [--halo-mm 8]
Writes <figures dir>/column_volume_<name>.png and .json (counts, time, peak).
The echoes are the .nii.gz files; DIR holds their decompressed copies (blocks.open_volume).
"""

import argparse
import json
import resource
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from baillarger import rays as R
from baillarger.blocks import ChunkedSource, open_volume
from baillarger.column_volume import ColumnVolumeSampler, take_columns

T0 = time.time()


def say(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def extents(rays: R.Rays) -> tuple[np.ndarray, np.ndarray]:
    """e_in and e_out of each column from its label intervals (see the module text)."""
    n = len(rays.roots.vertex)
    e_in = np.full(n, 0.5)
    e_out = np.ones(n)
    for c in range(n):
        q0, q1 = rays.interval_start[c], rays.interval_start[c + 1]
        s0, s1, cl = (
            rays.interval_s0[q0:q1],
            rays.interval_s1[q0:q1],
            rays.interval_class[q0:q1],
        )
        cortex_end = None
        for a, b, k in zip(s0, s1, cl, strict=True):
            if b <= 0:
                continue
            if k == R.CORTEX:
                cortex_end = float(b)
            elif k == R.OUTSIDE and cortex_end is not None:
                e_out[c] = min(R.CAP_OUT, max(float(a), 0.0) + 0.3)
                break
        else:
            if cortex_end is not None:
                e_out[c] = min(R.CAP_OUT, cortex_end + 0.3)
    return e_in, e_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fig")
    ap.add_argument("--surf", type=Path, required=True)
    ap.add_argument("--dseg", type=Path, required=True)
    ap.add_argument("--echoes", type=Path, nargs=3, required=True)
    ap.add_argument("--echo-dir", type=Path, required=True)
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--point-index", type=int, nargs="+", required=True)
    ap.add_argument("--name", default="column_volume")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--halo-mm", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=20000)
    A = ap.parse_args()
    fig_dir = Path(A.fig)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(json.loads(A.points.read_text())["points_full0"], np.int64)
    pts = pts[A.point_index]

    rs = R.RaySampler(A.surf, A.dseg)
    echoes = []
    for f in A.echoes:
        src = open_volume(f, A.echo_dir / f.with_suffix("").name)
        echoes.append(ChunkedSource(src, 64, 1 << 30))
        aff = src.affine
    smp = ColumnVolumeSampler(echoes, rs.affine, [aff] * 3, rs.shape)  # type: ignore[arg-type]

    side = rs.cells << A.level
    sb = tuple(int(x) // side for x in pts[0])
    assert all(tuple(int(x) // side for x in p) == sb for p in pts), (
        "points in one superblock"
    )
    say(f"superblock {sb} at level {A.level}, halo {A.halo_mm} mm")
    roots = rs.roots(sb, A.level, A.halo_mm)  # type: ignore[arg-type]
    say(f"{len(roots.vertex)} columns, {int(roots.owned.sum())} owned")
    r0 = rs.sample(sb, A.level, e_in=0.0, e_out=0.0, halo_mm=A.halo_mm)  # type: ignore[arg-type]
    e_in, e_out = extents(r0)
    del r0
    t = time.time()
    rays = rs.sample(sb, A.level, e_in=e_in, e_out=e_out, halo_mm=A.halo_mm)  # type: ignore[arg-type]
    t_rays = time.time() - t
    n = len(rays.roots.vertex)
    say(f"rays: {len(rays.s)} samples in {t_rays:.1f}s")

    # the columns of the sections: a slab of one voxel round each point
    vox = (
        np.linalg.inv(rs.affine[:3, :3])
        @ (rays.roots.root.astype(np.float64) - rs.affine[:3, 3]).T
    ).T
    sections = []
    for p in pts:
        sel = np.flatnonzero(
            (np.abs(vox[:, 1] - p[1]) <= 0.5)
            & (np.abs(vox[:, 0] - p[0]) <= 48)
            & (np.abs(vox[:, 2] - p[2]) <= 48)
        )
        xy = vox[sel][:, [0, 2]]
        u = np.linalg.svd(xy - xy.mean(0), full_matrices=False)[2][0]
        sections.append(sel[np.argsort((xy - xy.mean(0)) @ u)])
    wanted = {int(c): None for sec in sections for c in sec}

    t = time.time()
    got: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    n_valid = n_partial = 0
    for lo in range(0, n, A.batch):
        hi = min(n, lo + A.batch)
        part = take_columns(rays, lo, hi)
        cv = smp.sample(part, None)
        n_valid += int(cv.valid.sum())
        n_partial += int(cv.partial.sum())
        for c in range(lo, hi):
            if c in wanted:
                a, b = part.sample_start[c - lo], part.sample_start[c - lo + 1]
                got[c] = (
                    part.s[a:b].copy(),
                    cv.echo[a:b, 1].copy(),
                    cv.partial[a:b].copy(),
                )
    t_cv = time.time() - t
    N = len(rays.s)
    say(
        f"column volume: {N} samples in {t_cv:.1f}s ({N / t_cv / 1e6:.2f} Msamples/s); "
        f"valid {n_valid / N:.4f}, partial {n_partial / N:.4f}"
    )

    cmap = ListedColormap(["#222222", "#e8e8e8", "#c0392b", "#2a78d6", "#f1c40f"])
    for name_i, (p, sec) in enumerate(zip(pts, sections, strict=True)):
        step = rays.step
        k0 = int(rays.k_in.max())
        K = k0 + int(rays.k_out.max()) + 1
        ec = np.full((len(sec), K), np.nan)
        pm = np.zeros((len(sec), K), bool)
        cl = np.full((len(sec), K), np.nan)
        chg = np.full((len(sec), K), False)
        for x, c in enumerate(sec):
            s, e2, pa = got[int(c)]
            ki = np.rint(s / step).astype(int) + k0
            ec[x, ki] = e2
            pm[x, ki] = pa
            q0, q1 = rays.interval_start[c], rays.interval_start[c + 1]
            s0, s1, cls = (
                rays.interval_s0[q0:q1],
                rays.interval_s1[q0:q1],
                rays.interval_class[q0:q1],
            )
            sk = (ki - k0) * step
            idx = np.clip(np.searchsorted(s1, sk, side="right"), 0, len(cls) - 1)
            cl[x, ki] = cls[idx]
            for b in s0[1:]:
                j = int(np.rint(b / step)) + k0
                if 0 <= j < K:
                    chg[x, j] = True
        s_axis = (np.arange(K) - k0) * step
        lo_v, hi_v = np.nanpercentile(ec, [1, 99])
        fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True, sharey=True)
        ext = (-0.5, len(sec) - 0.5, s_axis[-1] + step / 2, s_axis[0] - step / 2)
        for a_ in ax:
            a_.set_ylabel("s (mm; 0 = root, outward is down)")
        ax[0].imshow(
            ec.T,
            cmap="gray",
            vmin=lo_v,
            vmax=hi_v,
            extent=ext,
            aspect="auto",
            interpolation="nearest",
        )
        ax[0].set_title(
            f"{A.name}: echo 2, point {tuple(int(v) for v in p)}, {len(sec)} neighbouring columns"
        )
        ax[1].imshow(
            ec.T,
            cmap="gray",
            vmin=lo_v,
            vmax=hi_v,
            extent=ext,
            aspect="auto",
            interpolation="nearest",
        )
        ov = np.zeros(pm.T.shape + (4,))
        ov[pm.T] = [1, 0.2, 0.2, 0.55]
        ax[1].imshow(ov, extent=ext, aspect="auto", interpolation="nearest")
        yy, xx = np.nonzero(chg.T)
        ax[1].plot(xx, s_axis[yy], ".", color="#30d5ff", ms=2)
        ax[1].set_title("partial-volume samples (red) and the label changes (cyan)")
        ax[2].imshow(
            np.ma.masked_invalid(cl.T),
            cmap=cmap,
            vmin=-0.5,
            vmax=4.5,
            extent=ext,
            aspect="auto",
            interpolation="nearest",
        )
        ax[2].set_title(
            "class of the label: outside (dark), WM (light), cortex (red), other (blue), off grid (yellow)"
        )
        ax[2].set_xlabel("column, in order along the section")
        fig.tight_layout()
        png = fig_dir / f"{A.name}_{name_i}.png"
        fig.savefig(png, dpi=110)
        plt.close(fig)
        say(f"wrote {png}")

    stats = dict(
        name=A.name,
        superblock=list(sb),
        level=A.level,
        halo_mm=A.halo_mm,
        columns=n,
        owned=int(roots.owned.sum()),
        samples=int(N),
        seconds_rays=round(t_rays, 1),
        seconds_column_volume=round(t_cv, 1),
        valid_fraction=n_valid / N,
        partial_fraction=n_partial / N,
        mean_extent_out_mm=float(e_out.mean()),
        max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
        total_seconds=round(time.time() - T0, 1),
    )
    (fig_dir / f"{A.name}.json").write_text(json.dumps(stats, indent=1))
    say(json.dumps(stats))


if __name__ == "__main__":
    main()
