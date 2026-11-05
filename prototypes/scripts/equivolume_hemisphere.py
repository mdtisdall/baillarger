"""The equivolume cross-section on a hemisphere: the diagnostics of phase 2.1.

    python prototypes/scripts/equivolume_hemisphere.py SURF DSEG OUT
        [--method equivolume|normal] [--level M] [--only N] [--densest]
        [--cache-gb G]
    python prototypes/scripts/equivolume_hemisphere.py --views OUT OVERVIEW

A pass samples every superblock that the surface crosses (or the first N in
Morton order, or with --densest only the superblock with the most columns)
at the full caps and the default step. It writes to OUT:

- `coef.npy`, `err.npy`, `focus.npy`: per vertex, a0, a1, a2 of A(s), their
  error bounds, and r_lo, r_hi (NaN where no call gave the vertex);
- `normlen.npy`: per vertex, |n̄|;
- `summary.json`: per s, the fraction of samples past a focal point and the
  fraction with A <= 0; the columns with A(0) <= 0, by |n̄|; the fraction of
  each column's samples past a focal point, by |n̄|; the near-umbilic
  columns that the rounding bound does not cut (a2 > 0, disc < -B, the
  minimum s* inside the caps), with A(s*)/A(0); the time and the peak
  memory of each call and of the pass.

With --method normal, the pass gives only the times and the peak memory.

--views draws, from a pass's arrays: `ratios.png`, the histograms of
A(+2 mm)/A(0) and A(-1 mm)/A(0); `past_focus.png`, the fraction past a
focal point by s and by |n̄|; `ratio_overview.png`, A(+2 mm)/A(0) on the
overview mesh.
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

BINS = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0001)
F64 = npt.NDArray[np.float64]


def peak_gb() -> float:
    """The peak resident memory of this process so far, GB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 30) if sys.platform == "darwin" else r / (1 << 20)


def ratio(coef: F64, s: float) -> F64:
    """A(s)/A(0) per column."""
    out: F64 = 1 + coef[:, 1] / coef[:, 0] * s + coef[:, 2] / coef[:, 0] * s * s
    return out


def bin_table(nl: F64, sel: npt.NDArray[np.bool_]) -> dict[str, list[int]]:
    """Per bin of |n̄|: [columns in sel, columns]."""
    idx = np.digitize(nl, BINS) - 1
    out = {}
    for b in range(len(BINS) - 1):
        inb = idx == b
        out[f"[{BINS[b]}, {min(BINS[b + 1], 1.0)})"] = [
            int(np.sum(sel & inb)),
            int(np.sum(inb)),
        ]
    return out


def run(a: argparse.Namespace) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sampler = rays.RaySampler(a.surf, a.dseg, int(a.cache_gb * (1 << 30)))
    m = a.level
    sbs = np.unique(sampler.block_ijk >> m, axis=0)
    sbs = sbs[np.argsort(morton(sbs), kind="stable")]
    if a.densest:
        nv = np.diff(sampler.block_vertices, axis=1).ravel()
        key = morton(sampler.block_ijk >> m)
        tot = {int(k): 0 for k in key}
        for k, c in zip(key, nv, strict=True):
            tot[int(k)] += int(c)
        best = max(tot, key=lambda k: tot[k])
        sbs = sbs[morton(sbs) == best]
    if a.only:
        sbs = sbs[: a.only]
    n = int(sampler.sidecar["Vertices"])
    coef = np.full((n, 3), np.nan)
    err = np.full((n, 3), np.nan)
    focus = np.full((n, 2), np.nan)
    nl = np.full(n, np.nan)
    frac_past = np.full(n, np.nan)
    nk = int(round((rays.CAP_IN + rays.CAP_OUT) / rays.STEP)) + 1
    k_in = int(round(rays.CAP_IN / rays.STEP))
    past_by_s = np.zeros(nk, np.int64)
    nonpos_by_s = np.zeros(nk, np.int64)
    all_by_s = np.zeros(nk, np.int64)
    calls = []
    for sb in sbs:
        t = time.perf_counter()
        sbt = (int(sb[0]), int(sb[1]), int(sb[2]))
        r = sampler.sample(sbt, m, method=a.method)
        dt = time.perf_counter() - t
        row = dict(
            superblock=sb.tolist(),
            columns=len(r.roots.vertex),
            samples=len(r.s),
            seconds=round(dt, 3),
            peak_gb=round(peak_gb(), 3),
        )
        if a.method == "equivolume":
            v = r.roots.vertex
            _, e = sampler.area_coef(r.roots, sbt, m, 0.0)
            coef[v], err[v], focus[v] = r.area_coef, e, r.focus
            nl[v] = r.roots.normal_length
            # the full caps: every column has the samples k = -K_in .. K_out
            assert np.all(r.k_in == k_in) and np.all(r.k_out == nk - 1 - k_in)
            past = r.past_focus.reshape(len(v), nk)
            all_by_s += len(v)
            past_by_s += past.sum(0)
            nonpos_by_s += (r.area.reshape(len(v), nk) <= 0).sum(0)
            frac_past[v] = past.mean(1)
        calls.append(row)
        print(json.dumps(row), flush=True)
        del r
    summary: dict[str, object] = dict(
        method=a.method,
        level=m,
        superblocks=len(sbs),
        pass_seconds=round(time.perf_counter() - t0, 3),
        peak_gb=round(peak_gb(), 3),
        calls=calls,
    )
    if a.method == "equivolume":
        done = np.isfinite(nl)
        a0, a1, a2 = coef[:, 0], coef[:, 1], coef[:, 2]
        disc = a1 * a1 - 4 * a0 * a2
        bound = (
            2 * np.abs(a1) * err[:, 1]
            + err[:, 1] ** 2
            + 4
            * (np.abs(a0) * err[:, 2] + np.abs(a2) * err[:, 0] + err[:, 0] * err[:, 2])
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            star = -a1 / (2 * a2)
            a_star = -disc / (4 * a2 * a0)
        umb = (
            done
            & (a0 > 0)
            & (a2 > 0)
            & (disc < -bound)
            & (star > -rays.CAP_IN)
            & (star < rays.CAP_OUT)
        )
        q = [0.0, 0.01, 0.1, 0.5, 0.9, 1.0]
        summary |= dict(
            columns=int(done.sum()),
            s=(np.arange(nk) - k_in) * rays.STEP,
            past_focus_by_s=(past_by_s / np.maximum(all_by_s, 1)).round(6).tolist(),
            area_nonpositive_by_s=(nonpos_by_s / np.maximum(all_by_s, 1))
            .round(6)
            .tolist(),
            a0_nonpositive=int(np.sum(done & ~(a0 > 0))),
            a0_nonpositive_by_normal_length=bin_table(nl, done & ~(a0 > 0)),
            double_root_columns=int(
                np.sum(done & (a0 > 0) & (a2 > 0) & (np.abs(disc) <= bound))
            ),
            near_umbilic_not_cut=int(umb.sum()),
            near_umbilic_s_star_quantiles=dict(
                zip(
                    map(str, q),
                    np.quantile(star[umb], q).round(4).tolist(),
                    strict=True,
                )
            )
            if umb.any()
            else {},
            near_umbilic_a_star_over_a0_quantiles=dict(
                zip(
                    map(str, q),
                    np.quantile(a_star[umb], q).round(5).tolist(),
                    strict=True,
                )
            )
            if umb.any()
            else {},
            near_umbilic_by_normal_length=bin_table(nl, umb),
            focus_inside_caps=dict(
                inward=int(np.sum(done & (focus[:, 0] > -rays.CAP_IN))),
                outward=int(np.sum(done & (focus[:, 1] < rays.CAP_OUT))),
            ),
            past_focus_fraction_by_normal_length={
                k: float(np.nanmean(frac_past[(np.digitize(nl, BINS) - 1) == b]))
                for b, k in enumerate(bin_table(nl, done))
            },
        )
        summary["s"] = np.asarray(summary["s"]).round(4).tolist()
        np.save(a.out / "coef.npy", coef)
        np.save(a.out / "err.npy", err)
        np.save(a.out / "focus.npy", focus)
        np.save(a.out / "normlen.npy", nl.astype(np.float32))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


def views(out: Path, overview: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from nibabel.gifti.gifti import GiftiImage

    coef = np.load(out / "coef.npy")
    nl = np.load(out / "normlen.npy").astype(np.float64)
    summary = json.loads((out / "summary.json").read_text())
    ok = np.isfinite(coef[:, 0]) & (coef[:, 0] > 0)
    r2 = ratio(coef[ok], 2.0)
    r1 = ratio(coef[ok], -1.0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for axi, val, name in [
        (ax[0], r2, "A(+2 mm) / A(0)"),
        (ax[1], r1, "A(-1 mm) / A(0)"),
    ]:
        lo, hi = np.quantile(val, [0.005, 0.995])
        axi.hist(np.clip(val, lo, hi), bins=200, color="#2a7fd4")
        axi.set_yscale("log")
        axi.set_xlabel(f"{name} (clipped to its 0.5-99.5 % range)")
        axi.set_title(
            f"{name}: median {np.median(val):.2f}, "
            f"{100 * np.mean(val <= 0):.1f} % <= 0",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out / "ratios.png", dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    s = np.asarray(summary["s"])
    ax[0].plot(s, summary["past_focus_by_s"], label="past a focal point")
    ax[0].plot(s, summary["area_nonpositive_by_s"], label="A <= 0")
    ax[0].set_xlabel("s (mm)")
    ax[0].set_ylabel("fraction of samples")
    ax[0].legend(fontsize=8)
    t = summary["past_focus_fraction_by_normal_length"]
    ax[1].bar(range(len(t)), list(t.values()), color="#d4552a")
    ax[1].set_xticks(range(len(t)), list(t.keys()), fontsize=7)
    ax[1].set_xlabel("|n̄|")
    ax[1].set_ylabel("mean fraction of a column's samples past a focal point")
    fig.tight_layout()
    fig.savefig(out / "past_focus.png", dpi=90)
    plt.close(fig)

    g = GiftiImage.from_filename(str(overview))
    x, f = g.agg_data(("pointset", "triangle"))
    x = np.asarray(x, np.float64)
    f = np.asarray(f, np.int64)
    name = overview.name.replace("desc-overview_", "desc-overviewids_")
    name = name.replace(".surf.gii", ".func.gii")
    ids = np.asarray(
        GiftiImage.from_filename(str(overview.with_name(name))).darrays[0].data
    )
    ids = ids.reshape(-1)
    full = np.full(len(coef), np.nan)
    full[ok] = r2
    val = np.log2(np.clip(full[ids], 1 / 16, 16))
    face = np.nanmedian(val[f], axis=1)
    views_ = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    labels = ["+x", "-x", "+y", "-y", "+z", "-z"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (axis, sign) in enumerate(views_):
        axi = axes.ravel()[i]
        plane = [b for b in range(3) if b != axis]
        o = np.argsort(sign * x[f, axis].mean(1))
        verts = x[f[o]][:, :, plane]
        pc = PolyCollection(verts, cmap="RdBu_r", linewidths=0)  # type: ignore[arg-type]
        pc.set_array(face[o])
        pc.set_clim(-3, 3)
        axi.add_collection(pc)
        axi.set_xlim(x[:, plane[0]].min(), x[:, plane[0]].max())
        axi.set_ylim(x[:, plane[1]].min(), x[:, plane[1]].max())
        axi.set_aspect("equal")
        if sign > 0:
            axi.invert_xaxis()
        axi.set_title(f"seen from {labels[i]}", fontsize=9)
        axi.set_axis_off()
    fig.colorbar(pc, ax=axes, shrink=0.6, label="log2 A(+2 mm)/A(0), clipped to ±3")
    fig.savefig(out / "ratio_overview.png", dpi=70, bbox_inches="tight")
    plt.close(fig)
    _ = nl


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--views":
        views(Path(argv[1]), Path(argv[2]))
        return
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("surf", type=Path)
    p.add_argument("dseg", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--method", default="equivolume", choices=rays.METHODS)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--only", type=int, default=0)
    p.add_argument("--densest", action="store_true")
    p.add_argument("--cache-gb", type=float, default=0.25)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
