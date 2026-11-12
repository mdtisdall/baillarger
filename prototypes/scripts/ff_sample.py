"""Sampled fields and the calibration of the spreads (docs/plans/flow-field-phantoms.md,
*Questions* item 8, *Measures: Uncertainty*, design decision R10; PR 6, task 6.4).

The mean-field posterior is a product over the nodes. A draw takes each node's state
independently from its kept slots and the stopped state (the probabilities q over the K
slots and q_stop; they sum to 1). The columns are traced through each drawn field from
the
roots of the truth, and the spread of the stops and of the τ levels over the S draws is
compared with the truth.

Usage:
    python ff_sample.py RUN_DIR --truth TRUTH_DIR --features FEATURES_JSON
        [--n 32] [--seed 0] [--site SITE]

RUN_DIR holds run.json and field.npz of a finished run (uniform or octree grid). The
script
rebuilds the phantom (for the geometry only) and the grid, and writes RUN_DIR/
calibration.json, calibration.npz and calibration.png. It changes no other file of the
run.

NumPy or Torch (the plan's *NumPy or Torch*)
--------------------------------------------
The kernels of this task: (a) one draw of a field, (b) the integrator on the drawn
field,
S times. All counts below are estimates from the loop structure, not measurements.

(a) The draw. Per node: a cumulative sum over K + 1 probabilities (K = prune_max = 32 in
the defaults), one comparison and one arg-max over the same K + 1 values, and a gather
of
the chosen slot (τ, 3 direction components, q). That is about 4 (K + 1) elementary
operations, about 130 per node, and about 1 KB of memory traffic per node (reading q,
writing the cumulative sums and the mask, in blocks of BLOCK nodes so that the
temporaries
stay in cache and in a bounded size). At a phantom size of 10^4 to 1.4·10^6 nodes: about
10^6 to 2·10^8 operations and 10 MB to 1.5 GB of traffic per draw. At the target size of
column-solver.md, 4·10^6 to 4·10^7 nodes: about 5·10^8 to 5·10^9 operations and 4 to 40
GB
per draw. The loop is over blocks of nodes. The work is regular and data-parallel (no
scatter, no reduction across nodes), and the one random number per node is drawn for the
whole field at once.

(b) The integrator is the cost. Per column and RK4 step it needs about 600 operations
and
1.8 KB of gathers (`ff_integrate`, *NumPy or Torch*). A draw therefore costs one
integrator
run: about 10^9 to 10^10 operations at the phantom size, and S = 32 of them. The draw is
about 10^-3 of that at the phantom size. At the target size, with about 3.5·10^8
columns,
all S samples of all columns would be about 10^13 column-steps; so a calibration there
would use a subset of the columns (a choice for that PR, not made here).

The loop structure: for each sample s, one draw (blocks of nodes), then the integrator's
loop over steps vectorised over the columns. The S samples are independent, so they
could
form a leading batch dimension on a GPU; that costs S times the memory of the sampled
fields (about 64 B per node and sample here, because a drawn field has one slot).

The choice: NumPy. The draw is already whole-array NumPy, and Torch would not change its
complexity. The part that is worth a GPU, the integrator, calls `Grid.locate`, which is
a
NumPy function of the Grid protocol (the same reason as in ff_integrate's choice). A
Torch port belongs to the integrator, not to this module; this module then needs only
the
draw (cumsum, compare, arg-max, gather: no `index_add_`, no `scatter_add_`) to move with
it.

Design choices (provisional, listed in the PR body)
---------------------------------------------------
- **The drawn field** has one slot (K = 1): slot 0 holds the drawn candidate (τ,
  direction), q = 1, q_stop = 0, n_kept = 1; a node drawn as stopped has q_stop = 1, q =
  0,
  n_kept = 0 and the padding values (τ nan, direction 0). So q is one-hot, and q_stop =
  1
  is the other one-hot state. One slot keeps the memory of a drawn field at about 64 B
  per
  node instead of 5 K · 8 B. The integrator reads a field only through q, q_stop, τ and
  direction, so the number of slots does not matter to it.
- **Hanging nodes** (octree): the mean field gives each hanging node its own free
  state (coordinator decision CD-10, `ff_octree`), and the integrator reads that
  state. So a hanging node is drawn independently, as every other node.
- **The random numbers**: sample s uses `numpy.random.default_rng([seed, s])`, so a
  sample
  does not depend on the number of samples.
- **The stop measure** is the geometry's signed distance at the stop,
  `geometry.stop_sdf`
  (negative in tissue, positive in the stopped region), as `ff_run.measure_stops` does,
  and
  not the distance along the true column. The contract of `stop_sdf` is exact where it
  is
  negative and above −1 mm, and only its sign elsewhere (`ff_types.Geometry`).
- **The level measure** is `ff_run.measure_levels`: the signed mm from the point where
  g(solved τ) first reaches the level to the true level surface. One monotone map g
  (`ff_run.fit_tau_map`) is fitted once, on the columns of the posterior-mean field (the
  field as given, traced from the same roots), and the same g is used for all samples.
- **Per column** over the samples that give a value (for the stops: those that ended
  STOPPED; for a level: those where the level was found): mean m, SD σ (ddof = 1), and
  z = (0 − m) / σ, since the deviation of the truth is 0. A column (or a column and
  level)
  with fewer than MIN_SAMPLES = 8 values, or with σ = 0 (all values equal), has no z and
  is counted. A column with fewer than 8 values has no σ either. The columns' `stop_sd`
  and `level_sd` are σ in mm (nan where fewer than 8 values).
- If the posterior is calibrated, the SD of z over the columns is near 1 (slightly
  above,
  because m and σ come from 32 draws), the fraction with |z| < 1 is near 0.683, with
  |z| < 2 near 0.954, and the median of |z| near 0.674 (the values for N(0, 1)).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import ff_integrate as FI  # noqa: E402
import ff_octree as FOC  # noqa: E402
import ff_run as FR  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

Array = T.Array

MIN_SAMPLES = 8
"""The smallest number of values per column (per column and level) that gets a σ."""

BLOCK = 1 << 18
"""Nodes per block of a draw: the temporaries (K + 1 float64 per node, three times) stay
below about 200 MB."""

MAX_PNG_BYTES = 900_000


# ---------------------------------------------------------------- one draw


def sample_field(field: T.Field, rng: np.random.Generator) -> T.Field:
    """One draw of a field (the module docstring): each node takes one of its kept slots
    or the stopped state, with the probabilities q and q_stop, independently. The result
    has one slot (K = 1) and a point mass in every node. The same generator state gives
    the same field."""
    n, K = field.q.shape
    u = np.asarray(rng.random(n))
    tau = np.full((n, 1), np.nan)
    direction = np.zeros((n, 1, 3))
    q = np.zeros((n, 1))
    q_stop = np.zeros(n)
    n_kept = np.zeros(n, np.int64)
    for a in range(0, n, BLOCK):
        b = min(n, a + BLOCK)
        p = np.concatenate(
            [
                np.maximum(field.q[a:b], 0.0),
                np.maximum(field.q_stop[a:b], 0.0)[:, None],
            ],
            axis=1,
        )  # (m, K + 1); the last column is the stopped state
        cum = np.cumsum(p, axis=1)
        x = np.asarray(u[a:b] * cum[:, -1])
        hit = cum > x[:, None]
        state = np.argmax(hit, axis=1)
        # rounding at the top (or a node with no mass): the last state with probability
        # > 0, which is the stopped state if there is none
        last = K - np.argmax(p[:, ::-1] > 0, axis=1)
        state = np.where(hit.any(axis=1), state, last)
        moving = np.flatnonzero(state < K)
        rows, slot = a + moving, state[moving]
        tau[rows, 0] = field.tau[rows, slot]
        direction[rows, 0] = field.direction[rows, slot]
        q[rows, 0] = 1.0
        n_kept[rows] = 1
        q_stop[a:b][state >= K] = 1.0
    return T.Field(
        grid=field.grid, tau=tau, direction=direction, q=q, q_stop=q_stop, n_kept=n_kept
    )


# ---------------------------------------------------------------- the sampled columns


@dataclass
class Samples:
    """What the calibration needs from S sampled fields, for N columns and L levels.

    stop_sdf: (S, N) `stop_sdf` at the stop of each sample's column, nan unless it ended
    STOPPED. level_dev: (S, N, L) the deviation to the true level surface (mm), nan
    where
    the level was not found. end: (S, N) EndReason. stops: (S, N, 3) voxel space.
    tau_arrival: (S, N). The mean_* fields are the same for the posterior-mean field
    (the
    field as given), whose columns also fit the shared map g (`g_fitted`)."""

    n_samples: int
    seed: int
    levels_phi: Array  # (L,)
    levels_tau: Array  # (L,)
    stop_sdf: Array
    level_dev: Array
    end: npt_int8
    stops: Array
    tau_arrival: Array
    time_s: Array  # (S,) seconds for each sample (draw, integrate, measure)
    mean_columns: T.Columns
    mean_stop_sdf: Array  # (N,)
    mean_level_dev: Array  # (N, L)
    g_fitted: bool


npt_int8 = np.ndarray[Any, np.dtype[np.int8]]


def _stop_sdf(cols: T.Columns, geo: T.Geometry) -> Array:
    out = np.full(len(cols.end), np.nan)
    stopped = cols.end == int(T.EndReason.STOPPED)
    if stopped.any():
        out[stopped] = geo.stop_sdf(geo.to_local(cols.stops[stopped]))
    return out


def sampled_columns(
    field: T.Field,
    truth: T.Truth,
    settings: T.SolverSettings = T.DEFAULTS,
    n_samples: int = 32,
    seed: int = 0,
    *,
    geo: T.Geometry,
    tau_scale: float,
    levels_phi: Sequence[float] = FR.LEVEL_PHI,
) -> Samples:
    """Trace the columns from `truth.roots` through the posterior-mean field and through
    `n_samples` drawn fields. Only the values that the measures need are kept (the
    signed stop distances, the level deviations, the stops, the end reasons, τ at the
    stop); the paths of a sample are dropped when its measures are done, so the memory
    is one integrator run plus S small arrays. `geo` and `tau_scale` come from the
    phantom of the run (the geometry is read only for φ, ∇φ and `stop_sdf`)."""
    roots = truth.roots
    n = len(roots)
    phis = np.asarray(levels_phi, float)
    taus = (phis - 1.0) * tau_scale
    nl = len(taus)
    cols0, _ = FI.integrate(field, roots, roots, settings, levels=taus)
    g = FR.fit_tau_map(FR.PathData.of(cols0, geo, tau_scale))
    mean_dev = _level_dev(cols0, geo, tau_scale, g, phis)
    S = n_samples
    stop_sdf = np.full((S, n), np.nan)
    level_dev = np.full((S, n, nl), np.nan)
    end = np.zeros((S, n), np.int8)
    stops = np.zeros((S, n, 3))
    arrival = np.zeros((S, n))
    secs = np.zeros(S)
    for s in range(S):
        t0 = time.perf_counter()
        fs = sample_field(field, np.random.default_rng([seed, s]))
        cols, _ = FI.integrate(fs, roots, roots, settings, levels=taus)
        stop_sdf[s] = _stop_sdf(cols, geo)
        level_dev[s] = _level_dev(cols, geo, tau_scale, g, phis)
        end[s], stops[s], arrival[s] = cols.end, cols.stops, cols.tau_arrival
        del fs, cols
        secs[s] = time.perf_counter() - t0
    return Samples(
        n_samples=S,
        seed=seed,
        levels_phi=phis,
        levels_tau=taus,
        stop_sdf=stop_sdf,
        level_dev=level_dev,
        end=end,
        stops=stops,
        tau_arrival=arrival,
        time_s=secs,
        mean_columns=cols0,
        mean_stop_sdf=_stop_sdf(cols0, geo),
        mean_level_dev=mean_dev,
        g_fitted=g is not None,
    )


def _level_dev(
    cols: T.Columns, geo: T.Geometry, tau_scale: float, g: Any | None, phis: Array
) -> Array:
    """(N, L) the level deviations of `ff_run.measure_levels` for these columns and the
    given map g."""
    pd = FR.PathData.of(cols, geo, tau_scale)
    _, lev = FR.measure_levels(pd, g, geo, tau_scale, list(phis), len(cols.counts))
    return np.asarray(lev.dev)


# ---------------------------------------------------------------- the calibration


@dataclass(frozen=True)
class Moments:
    """Per column (and level) over the samples: n values, mean m, SD σ (ddof = 1) and
    z = (0 − m) / σ. σ is nan where n < min_n; z is nan where n < min_n or σ = 0."""

    n: Array
    m: Array
    sd: Array
    z: Array
    few: Any  # n < min_n
    zero_sd: Any  # n >= min_n and σ = 0


def moments(x: Array, min_n: int = MIN_SAMPLES) -> Moments:
    """The moments over axis 0 of x (S, ...), ignoring nan. σ is exactly 0 when all the
    values are equal (the mean of equal values can differ from them in the last bit)."""
    ok = np.isfinite(x)
    n = ok.sum(axis=0)
    xz = np.where(ok, x, 0.0)
    m = xz.sum(axis=0) / np.maximum(n, 1)
    ss = (np.where(ok, x - m, 0.0) ** 2).sum(axis=0)
    sd = np.sqrt(ss / np.maximum(n - 1, 1))
    lo = np.where(ok, x, np.inf).min(axis=0)
    hi = np.where(ok, x, -np.inf).max(axis=0)
    const = (n > 0) & (lo == hi)
    sd = np.where(const, 0.0, sd)
    m = np.where(const, lo, m)
    few = n < min_n
    zero = (~few) & (sd == 0.0)
    good = ~few & ~zero
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(good, (0.0 - m) / np.where(good, sd, 1.0), np.nan)
    return Moments(
        n=n.astype(np.int64),
        m=np.where(n > 0, m, np.nan),
        sd=np.where(few, np.nan, sd),
        z=z,
        few=few,
        zero_sd=zero,
    )


@dataclass(frozen=True)
class Calibration:
    stop: Moments  # (N,)
    level: Moments  # (N, L)


def calibrate(samples: Samples, min_n: int = MIN_SAMPLES) -> Calibration:
    return Calibration(
        stop=moments(samples.stop_sdf, min_n), level=moments(samples.level_dev, min_n)
    )


def with_spread(cols: T.Columns, samples: Samples, cal: Calibration) -> T.Columns:
    """`cols` with the Columns fields stop_sd (N,) and level_sd (N, L) (mm) and levels
    (L,) (the τ levels) set from the calibration."""
    return dataclasses.replace(
        cols, stop_sd=cal.stop.sd, level_sd=cal.level.sd, levels=samples.levels_tau
    )


def zstats(z: Array) -> dict[str, Any]:
    """n, mean, SD (ddof = 1), median |z|, and the fractions with |z| < 1 and < 2 of the
    finite values of z."""
    f = np.asarray(z, float).ravel()
    f = f[np.isfinite(f)]
    if f.size == 0:
        nan = float("nan")
        return dict(n=0, mean=nan, sd=nan, median_abs=nan, frac_lt1=nan, frac_lt2=nan)
    a = np.abs(f)
    return dict(
        n=int(f.size),
        mean=float(f.mean()),
        sd=float(f.std(ddof=1)) if f.size > 1 else float("nan"),
        median_abs=float(np.median(a)),
        frac_lt1=float((a < 1.0).mean()),
        frac_lt2=float((a < 2.0).mean()),
    )


def report(samples: Samples, cal: Calibration) -> dict[str, Any]:
    """The JSON-able measures of the calibration."""
    st, lv = cal.stop, cal.level
    ends = {
        r.name: int((samples.end == int(r)).sum()) for r in T.EndReason
    }  # over all samples and columns
    per_level = [
        dict(
            phi=float(samples.levels_phi[j]),
            tau=float(samples.levels_tau[j]),
            n_columns_used=int(np.isfinite(lv.z[:, j]).sum()),
            n_excluded_few=int(lv.few[:, j].sum()),
            n_excluded_zero_sd=int(lv.zero_sd[:, j].sum()),
            z=zstats(lv.z[:, j]),
            sd_mm=FR.stats(lv.sd[:, j]),
        )
        for j in range(lv.z.shape[1])
    ]
    return dict(
        n_samples=samples.n_samples,
        seed=samples.seed,
        min_samples=MIN_SAMPLES,
        n_columns=int(len(st.n)),
        end_reasons_all_samples=ends,
        stops=dict(
            measure="geometry.stop_sdf at the stop (mm), samples that ended STOPPED",
            n_columns_used=int(np.isfinite(st.z).sum()),
            n_excluded_few=int(st.few.sum()),
            n_excluded_zero_sd=int(st.zero_sd.sum()),
            z=zstats(st.z),
            sd_mm=FR.stats(st.sd),
            mean_field_signed_mm=FR.stats(samples.mean_stop_sdf),
        ),
        levels=dict(
            measure="ff_run.measure_levels deviation (mm), one shared map g",
            g_fitted=samples.g_fitted,
            n_pairs_used=int(np.isfinite(lv.z).sum()),
            n_excluded_few=int(lv.few.sum()),
            n_excluded_zero_sd=int(lv.zero_sd.sum()),
            z=zstats(lv.z),
            sd_mm=FR.stats(lv.sd),
            per_level=per_level,
        ),
        reference_n01=dict(sd=1.0, median_abs=0.6745, frac_lt1=0.6827, frac_lt2=0.9545),
    )


def arrays(samples: Samples, cal: Calibration) -> dict[str, Array]:
    """The per column arrays of calibration.npz."""
    return dict(
        stop_n=cal.stop.n,
        stop_m=cal.stop.m,
        stop_sd=cal.stop.sd,
        stop_z=cal.stop.z,
        level_n=cal.level.n,
        level_m=cal.level.m,
        level_sd=cal.level.sd,
        level_z=cal.level.z,
        level_phi=samples.levels_phi,
        level_tau=samples.levels_tau,
        mean_field_stop_sdf=samples.mean_stop_sdf,
        mean_field_level_dev=samples.mean_level_dev,
    )


# ---------------------------------------------------------------- the view


def view_z(path: Path, cal: Calibration, name: str) -> int:
    """Histograms of z (stops, levels) with the N(0, 1) density; returns the size of the
    PNG in bytes (below MAX_PNG_BYTES)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    grid = np.linspace(-6.0, 6.0, 241)
    dens = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
    for ax, mom, title in (
        (axes[0], cal.stop, "stops"),
        (axes[1], cal.level, "levels"),
    ):
        z = mom.z[np.isfinite(mom.z)]
        zs = zstats(z)
        if z.size:
            out = int((np.abs(z) > 6.0).sum())
            ax.hist(np.clip(z, -6.0, 6.0), bins=60, range=(-6.0, 6.0), density=True,
                    color="0.65")  # fmt: skip
        else:
            out = 0
        ax.plot(grid, dens, "k-", lw=1.2, label="N(0, 1)")
        ax.set_title(
            f"{title}: n={zs['n']}, SD={zs['sd']:.2f}, |z|<1 {zs['frac_lt1']:.2f}, "
            f"|z|>6: {out}",
            fontsize=8,
        )
        ax.set_xlabel("z = (0 - mean) / sd over the samples")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
    fig.suptitle(f"{name}: calibration of the sampled-field spreads", fontsize=9)
    fig.tight_layout()
    size = 0
    for dpi in FR.DPIS:
        fig.savefig(path, dpi=dpi)
        size = path.stat().st_size
        if size < MAX_PNG_BYTES:
            break
    plt.close(fig)
    return int(size)


# ---------------------------------------------------------------- a run directory


def _peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def load_run_grid(
    run_dir: Path, rec: T.RunRecord, shape: tuple[int, int, int]
) -> T.Grid:
    """The grid of a run: a uniform grid from the grid spec (as `ff_run.load_field`
    expects), an octree from the grid_* arrays of field.npz."""
    kind = str(rec.grid.get("kind", "uniform"))
    if kind == "octree":
        z = np.load(run_dir / "field.npz", allow_pickle=False)
        return FOC.OctreeGrid(
            tuple(int(v) for v in z["grid_shape_vox"]),  # type: ignore[arg-type]
            z["grid_level"],
            z["grid_anchor"],
            FR.settings_from_record(rec.settings),
        )
    return FR.make_grid(str(rec.grid["spec"]), shape)


def calibrate_run(
    run_dir: Path,
    truth_arg: Path,
    features: Path,
    *,
    n_samples: int = 32,
    seed: int = 0,
    site: str | None = None,
) -> dict[str, Any]:
    """Read a run directory, draw and trace the sampled fields, and write
    calibration.json, calibration.npz and calibration.png into it. Returns the JSON
    record."""
    t_start = time.perf_counter()
    rec = T.RunRecord.from_json(run_dir / "run.json")
    ph = rec.phantom
    name = str(ph["name"])
    feats = FP.features(site or str(ph["site"]), str(features))  # type: ignore[no-untyped-call]
    phantom = FTR.build_phantom(
        name,
        feats,
        site=site or str(ph["site"]),
        noise=float(ph["noise"]),
        seed=int(ph["seed"]),
        shape=tuple(int(v) for v in ph["shape"]),  # type: ignore[arg-type]
    )
    geo = phantom.geometry
    if geo is None:
        raise ValueError(f"phantom {name!r} has no geometry, so no truth")
    tpath = truth_arg if truth_arg.suffix == ".npz" else truth_arg / name / "truth.npz"
    truth = T.Truth.from_npz(tpath)
    if truth.phantom != name or truth.n == 0:
        raise ValueError("the truth does not belong to this run's phantom")
    if not np.abs(geo.phi(geo.to_local(truth.roots))).max() < 1e-6:
        raise ValueError("the truth's roots are not on phi = 0 of this phantom")
    grid = load_run_grid(run_dir, rec, phantom.shape)
    field, _ = FR.load_field(run_dir / "field.npz", grid)
    settings = FR.settings_from_record(rec.settings)
    samples = sampled_columns(
        field, truth, settings, n_samples, seed, geo=geo, tau_scale=phantom.tau_scale
    )
    cal = calibrate(samples)
    out = report(samples, cal)
    np.savez_compressed(run_dir / "calibration.npz", **arrays(samples, cal))  # type: ignore[arg-type]
    png = view_z(run_dir / "calibration.png", cal, name)
    total = time.perf_counter() - t_start
    out.update(
        phantom=name,
        grid=rec.grid.get("spec"),
        time_s=total,
        time_per_sample_s=float(samples.time_s.mean()),
        peak_rss_bytes=_peak_rss_bytes(),
        png_bytes=png,
    )
    (run_dir / "calibration.json").write_text(
        json.dumps(out, indent=1, allow_nan=True) + "\n"
    )
    return out


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--site", default=None, help="default: the run's site")
    a = ap.parse_args(None if argv is None else list(argv))
    out = calibrate_run(
        a.run_dir, a.truth, a.features, n_samples=a.n, seed=a.seed, site=a.site
    )
    print(
        f"wrote {a.run_dir}/calibration.* ({out['time_s']:.1f} s, "
        f"{out['time_per_sample_s']:.1f} s per sample, "
        f"peak {out['peak_rss_bytes'] / 1e9:.2f} GB)"
    )
    return out


if __name__ == "__main__":
    main()
