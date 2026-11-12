"""The run script of the flow-field phantoms stage (docs/plans/flow-field-phantoms.md,
*Run matrix, outputs and memory* and *Measures*; PR 3, task 3.5).

One run is one phantom, one input source, one noise level and one grid. The script
builds
the phantom, the inputs and the grid, solves the field (`ff_meanfield.solve`), traces
the
columns from the truth's start points (`ff_integrate.integrate`), measures them and
writes
the run directory.

Usage:
    python ff_run.py PHANTOM --source {oracle,image} --noise N --grid uniform:0.16
        --out OUT_DIR --truth TRUTH_DIR --features FEATURES_JSON
        [--site V1V2_banks] [--seed 0] [--shape X,Y,Z] [--profile-dtype float32]
        [--tau-prior] [--origin {top,anchor}] [--anchor-node ID]
        [--set name=value ...] [--quiet]

TRUTH_DIR holds TRUTH_DIR/<phantom>/truth.npz, as `ff_truth.py` writes it (a path to one
truth.npz also works). `--shape` builds a box that is not the plan's; the truth must
come
from the same box (the script checks that the truth's roots lie on phi = 0). `--set`
changes
one field of `ff_types.SolverSettings` for this run (a scan) and records it in
`changed_settings`. The value is JSON text, for example `--set max_sweeps=3`.
`--origin anchor --anchor-node ID` is the anchor comparison run of the plan's question 6
(`ff_meanfield.solve`); the anchor is a node id of the grid. The run directory does not
depend on the origin, so give an anchor run its own `--out`.

Where to add a source or a grid
-------------------------------
- A new input source: one branch in `make_inputs` (PR 4 uses it for the image inputs).
  `run` takes any `ff_types.Inputs`.
- A new grid: one branch in `make_grid` (PR 5 adds "octree"). Everything after it uses
  only
  the `ff_types.Grid` protocol, except the label of the run directory.

The run directory is `ff_types.run_dir(out, phantom, source, noise, grid_label)`, with
the
grid label "uniform0.16" for the grid spec "uniform:0.16". It holds:

- `field.npz`: `Field.to_npz` with the extras of the solver (u, speed, p_jump, ...).
- `columns.npz`: `Columns.to_npz`. Its `levels` are the tau levels of the *Levels*
  measure.
- `column_errors.npz`: per column, the stop error, the path errors and the level
  deviations (the arrays behind the histograms).
- `run.json`: an `ff_types.RunRecord`. `measures` holds one entry for each item of the
  plan's *Measures* that applies to a uniform-grid run (the keys are in `measure_all`),
  and `measures["exact"]` the exact tests.
- `section.png`, `levels.png`, `errors.png`, `free_energy.png`: the views.

Measures
--------
Every distribution is given as n, median, p90, p99 and max (`stats`). Lengths are in mm.

- **stops**: |stop_sdf| at the stop of each column that ended STOPPED (stop_sdf is exact
  only where it is negative; on the other side the contract gives its sign), the signed
  value, the same by true stop kind, and the count of each end reason.
- **paths**: `bent_phantom.path_errors` on the solved and true columns (the largest and
  the mean distance of the solved samples with 0 <= phi <= 1 from the true path, and the
  distance between the two stops).
- **levels**: one monotone map g from solved tau to true tau is fitted by isotonic
  regression (scikit-learn) over all solved path samples with 0 <= phi <= 1, with the
  true
  tau of the sample as target. For each level l (the true tau at relative depth phi =
  0.1 .. 0.9) and each column, the solved position is the first point of the column
  where
  g(solved tau) reaches l. Its deviation is (true tau at the point - l) times the true
  speed there: the distance in mm to the true level surface, to first order. It is
  positive when the solved level lies nearer to the top than the true level.
- **speeds**: solved speed over true speed at the moving nodes whose true label is
  cortex
  (and, apart, those with 0.1 <= phi <= 0.9). For two-bank phantoms, the ratio of the
  median
  solved speeds of the two banks, against tA / tB and against the same ratio of the true
  speeds at those nodes.
- **spacing_near_stop**: for each column, the five true-tau levels below the true tau at
  the true stop (l_m = tau_arrival - m * dtau, m = 1 .. 5). The mean arc distance
  between
  consecutive levels, (arc(l_1) - arc(l_5)) / 4, on the solved column (levels found
  with g)
  and on the true column, and their ratio. `ratio_without_map` gives the same ratio
  from the solved tau alone: levels below the solved column's own arrival tau, no map.
- **topology**: `ff_checks.counts`.
- **direction**: the angle between the posterior-mean direction and the solved grad tau
  at
  moving cortex nodes; the angle between the direction and the true direction; the angle
  between grad tau and the true direction. grad tau is the finite difference of the mean
  tau along each grid axis, central where both neighbours move, one-sided where one
  moves;
  a node needs a difference on each axis.
- **jumps**: the jump at each stopped column, and its error against the true jump.
- **inference**, **cost**: from the solver's log, and the time and peak memory.
- **inputs**: the record of the stop field and the tensor.

Exact tests (`measures["exact"]`)
---------------------------------
Each has the observed value, the tolerance and a pass flag. They apply to noise 0 and
the
oracle source (`conditions`); the flags are also given for other runs.

- **slab**: for each column, the largest angle (over windows of 0.25 mm of path inside
  0.1 <= phi <= 0.9) between the window's chord and the true direction. Pass if the
  largest over all columns is below 1 degree (a column with no window fails). The levels
  are planes if the largest distance of a level's points from their least-squares plane
  is
  below 0.02 mm, for each of the nine levels.
- **tau**: (a) the speeds: the 90th percentile of |speed / true speed - 1| over the
  moving
  nodes with 0.1 <= phi <= 0.9 below 0.05. (b) The flat band: for each column, the mean
  deviation of the levels phi = 0.60 and 0.65 minus that of phi = 0.30 and 0.35; pass if
  the median |difference| is below 0.04 mm. (c) The areal border at u = 0: for each
  level,
  the deviation of the points with |u| <= 1 mm is fitted by a + b u + c [u > 0]; c is
  the
  shift. Pass if the largest |c| over the levels is below 0.04 mm.

The window length of the slab test, the statistic of (a), the levels of (b) and the
window
of (c) are choices of this script. The plan gives no tolerance for the flat band (b);
0.04
mm, the border's, is used.

NumPy or Torch
--------------
NumPy. The measures are one pass over the path samples (a few hundred thousand points),
and
`bent_phantom.path_errors` is a Python loop over the columns. None of it is the cost of
a
run: the solver is.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import bent_phantom as BP  # noqa: E402
import ff_checks as FC  # noqa: E402
import ff_grid as FG  # noqa: E402
import ff_integrate as FI  # noqa: E402
import ff_meanfield as MF  # noqa: E402
import ff_oracle as FO  # noqa: E402
import ff_phantom_views as FV  # noqa: E402
import ff_stop as FS  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

H = T.VOXEL_MM
Array = T.Array

# ---- the tolerances of the plan's *Exact tests* (fixed before any run)
SLAB_ANGLE_DEG = 1.0
SLAB_LEVEL_MM = 0.02
TAU_SPEED_REL = 0.05
TAU_BORDER_MM = 0.04
# ---- choices of this script (module docstring)
TAU_BAND_MM = TAU_BORDER_MM  # the plan gives no tolerance for the flat band
SLAB_WINDOW_MM = 0.25
INTERIOR = (0.1, 0.9)  # relative depths that exclude the white surface and the top
LEVEL_PHI = tuple(round(0.1 * k, 1) for k in range(1, 10))
TAU_BELOW_PHI = (0.30, 0.35)
TAU_ABOVE_PHI = (0.60, 0.65)
TAU_BAND_PHI = (0.40, 0.475, 0.55)
BORDER_WINDOW_MM = 1.0
BORDER_MIN_POINTS = 5  # on each side of the border, for the fit
SPACING_LEVELS = 5
FIT_MAX_POINTS = 200_000
SECTION_COLUMNS = 30
MAX_PNG_BYTES = 900_000  # below the 1 MB of the plan
DPIS = (110, 80, 60, 45)

LogFn = Callable[[str], None]


# ---------------------------------------------------------------- inputs and grid


def make_inputs(
    phantom: T.Phantom,
    source: T.InputSource,
    reference: T.Phantom | FS.SiteStats | None = None,
) -> T.Inputs:
    """The inputs of one run: the stop field, the tensor and the contrast.

    oracle: `ff_oracle.oracle_inputs`. image: `ff_oracle.ImageTensor.of`, and
    `ff_stop.ImageStopField` with `reference`, the same phantom built with noise 0 (or
    its
    `SiteStats`). If the phantom has noise 0, it is its own reference. The contrast is
    `phantom.det` in both. A new source adds one branch here."""
    if source == "oracle":
        return FO.oracle_inputs(phantom)
    if source == "image":
        if reference is None:
            if phantom.noise != 0.0:
                raise ValueError("the image source needs a noise-0 reference phantom")
            reference = phantom
        tensor = FO.ImageTensor.of(phantom)
        stop = FS.ImageStopField(phantom, tensor, reference)
        return T.Inputs(stop=stop, tensor=tensor, contrast=phantom.det)
    raise ValueError(f"unknown input source {source!r}")


def parse_grid(spec: str) -> tuple[str, float]:
    """("uniform", 0.16) for "uniform:0.16"."""
    kind, _, rest = spec.partition(":")
    if kind != "uniform":
        raise ValueError(f"unknown grid {spec!r}; use uniform:<spacing in mm>")
    try:
        spacing = float(rest)
    except ValueError as e:
        raise ValueError(f"unknown grid {spec!r}; use uniform:<spacing in mm>") from e
    if not spacing > 0:
        raise ValueError("the grid spacing must be positive")
    return kind, spacing


def grid_label(spec: str) -> str:
    """The grid part of the run directory name: "uniform:0.16" gives "uniform0.16"."""
    return spec.replace(":", "")


def make_grid(spec: str, shape: Sequence[int]) -> T.Grid:
    """The grid for a spec over a box of `shape` voxels. A new grid adds one branch here
    (PR 5: "octree")."""
    kind, spacing = parse_grid(spec)
    if kind == "uniform":
        return FG.UniformGrid((int(shape[0]), int(shape[1]), int(shape[2])), spacing)
    raise ValueError(f"unknown grid {spec!r}")  # pragma: no cover


def grid_record(grid: T.Grid, spec: str) -> dict[str, Any]:
    """The grid entry of run.json: spec, kind, spacing, node and cell counts per
    level."""
    cells = grid.cells
    sizes = np.asarray(grid.cell_size_mm)
    levels = np.asarray(grid.node_level)
    rec: dict[str, Any] = dict(
        spec=spec,
        kind=grid.kind,
        n_nodes=int(grid.n_nodes),
        n_cells=int(len(cells)),
        nodes_per_level={
            str(int(k)): int((levels == k).sum()) for k in np.unique(levels)
        },
        cells_per_size_mm={f"{s:g}": int((sizes == s).sum()) for s in np.unique(sizes)},
        n_prior_shaped_cells=int(np.asarray(grid.prior_shaped).sum()),
    )
    spacing = getattr(grid, "spacing_mm", None)
    if spacing is not None:
        rec["spacing_mm"] = float(spacing)
    return rec


def parse_settings(
    items: Sequence[str], base: T.SolverSettings = T.DEFAULTS
) -> tuple[T.SolverSettings, dict[str, Any]]:
    """`--set name=value` items to (settings, changed). The value is JSON; it is cast
    to the
    type of the default (a tuple for a tuple). An unknown name is an error."""
    names = {f.name for f in dataclasses.fields(T.SolverSettings)}
    changed: dict[str, Any] = {}
    for item in items:
        name, eq, text = item.partition("=")
        if not eq or name not in names:
            raise ValueError(
                f"--set {item!r}: use name=value with a SolverSettings field"
            )
        default = getattr(base, name)
        value: Any = json.loads(text)
        if isinstance(default, tuple):
            value = tuple(value)
        elif isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        changed[name] = value
    return dataclasses.replace(base, **changed), changed


# ---------------------------------------------------------------- small helpers


def _peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip()


def stats(x: Any) -> dict[str, Any]:
    """n, median, p90, p99 and max of the finite values of x."""
    a = np.asarray(x, float).ravel()
    f = a[np.isfinite(a)]
    if f.size == 0:
        nan = float("nan")
        return dict(n=0, median=nan, p90=nan, p99=nan, max=nan)
    p = np.percentile(f, [50, 90, 99])
    return dict(
        n=int(f.size),
        median=float(p[0]),
        p90=float(p[1]),
        p99=float(p[2]),
        max=float(f.max()),
    )


def _angle_deg(a: Array, b: Array) -> Array:
    """The angle in degrees between unit vectors, row by row (nan if a row is not
    unit)."""
    c = np.sum(a * b, axis=1)
    return np.asarray(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _unit(v: Array) -> Array:
    n = np.linalg.norm(v, axis=1)
    return np.asarray(v / np.where(n > 0, n, np.nan)[:, None])


# ---------------------------------------------------------------- path samples


@dataclass
class PathData:
    """All samples of all solved columns in flat arrays (M samples). Column i holds the
    samples offsets[i] .. offsets[i + 1] - 1."""

    offsets: T.IntArray  # (N + 1,)
    col: T.IntArray  # (M,)
    pos: Array  # (M, 3) voxel space
    tau: Array  # (M,) solved tau
    phi: Array  # (M,) true relative depth
    tau_true: Array  # (M,) (phi - 1) * tau_scale
    arc: Array  # (M,) mm along the column from its start

    @classmethod
    def of(cls, cols: T.Columns, geo: T.Geometry, tau_scale: float) -> PathData:
        n = len(cols.counts)
        counts = np.asarray(cols.counts, np.int64)
        m = cols.paths.shape[1]
        mask = np.arange(m)[None, :] < counts[:, None]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        pos = np.asarray(cols.paths[mask], float)
        tau = np.asarray(cols.tau_path[mask], float)
        col = np.repeat(np.arange(n, dtype=np.int64), counts)
        step = np.zeros(len(pos))
        if len(pos) > 1:
            step[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=1) * H
        step[offsets[:-1][counts > 0]] = 0.0
        c = np.cumsum(step)
        arc = c - c[offsets[:-1][col]] if len(pos) else c
        phi = geo.phi(geo.to_local(pos)) if len(pos) else np.zeros(0)
        return cls(offsets, col, pos, tau, phi, (phi - 1.0) * tau_scale, arc)

    @property
    def n(self) -> int:
        return len(self.offsets) - 1

    def sl(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))


def _bracket(values: Array, levels: Array) -> tuple[T.IntArray, Array, Any]:
    """For values that do not decrease: the index k and the fraction f of the first
    crossing of each level, between sample k - 1 and sample k, and whether there is
    one."""
    n = len(values)
    k = np.searchsorted(values, levels, side="left")
    kk = np.clip(k, 1, max(n - 1, 1))
    ok = (k >= 1) & (k < n)
    if n < 2:
        return kk, np.full(len(levels), np.nan), np.zeros(len(levels), bool)
    d = values[kk] - values[kk - 1]
    ok &= d > 0
    f = (levels - values[kk - 1]) / np.where(d > 0, d, 1.0)
    return kk, np.where(ok, f, np.nan), ok


def fit_tau_map(pd: PathData) -> Any | None:
    """The shared monotone map from solved tau to true tau (isotonic regression) over
    the
    samples with 0 <= phi <= 1; None if there are fewer than two such samples."""
    from sklearn.isotonic import IsotonicRegression

    sel = np.isfinite(pd.tau) & (pd.phi >= 0.0) & (pd.phi <= 1.0)
    x, y = pd.tau[sel], pd.tau_true[sel]
    if x.size < 2:
        return None
    stride = max(1, math.ceil(x.size / FIT_MAX_POINTS))
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(x[::stride], y[::stride])
    return iso


class _Identity:
    """A map that leaves tau as it is (the same interface as the fitted map)."""

    def predict(self, x: Array) -> Array:
        return x


def level_crossings(pd: PathData, g: Any, levels: Array) -> tuple[Array, Array, Any]:
    """The first point of each column where g(solved tau) reaches each level.

    levels: (N, L), the true tau levels of each column. Returns the position (N, L, 3,
    voxel space), the arc length (N, L, mm) and the mask of the levels that were
    found."""
    n, nl = levels.shape
    gt = np.full(len(pd.tau), np.nan)
    fin = np.isfinite(pd.tau)
    if fin.any():
        gt[fin] = g.predict(pd.tau[fin])
    pos = np.full((n, nl, 3), np.nan)
    arc = np.full((n, nl), np.nan)
    found = np.zeros((n, nl), bool)
    for i in range(n):
        s = pd.sl(i)
        idx = np.flatnonzero(fin[s]) + s.start
        if idx.size < 2:
            continue
        gk = np.maximum.accumulate(gt[idx])
        k, f, ok = _bracket(gk, levels[i])
        if not ok.any():
            continue
        a, b = idx[k[ok] - 1], idx[k[ok]]
        fo = f[ok][:, None]
        pos[i, ok] = pd.pos[a] + fo * (pd.pos[b] - pd.pos[a])
        arc[i, ok] = pd.arc[a] + f[ok] * (pd.arc[b] - pd.arc[a])
        found[i] = ok
    return pos, arc, found


# ---------------------------------------------------------------- the measures


def measure_stops(
    cols: T.Columns, truth: T.Truth, geo: T.Geometry, info: dict[str, Any]
) -> dict[str, Any]:
    stopped = cols.end == int(T.EndReason.STOPPED)
    sd = np.full(len(cols.end), np.nan)
    if stopped.any():
        sd[stopped] = geo.stop_sdf(geo.to_local(cols.stops[stopped]))
    by_kind = {
        k.name: stats(np.abs(sd[truth.stop_kind == int(k)]))
        for k in T.StopKind
        if (truth.stop_kind == int(k)).any()
    }
    return dict(
        n_columns=int(len(sd)),
        n_stopped=int(stopped.sum()),
        abs_mm=stats(np.abs(sd)),
        signed_mm=stats(sd),
        by_kind=by_kind,
        end_reasons=dict(info["ends"]),
        _array=sd,
    )


def measure_paths(cols: T.Columns, truth: T.Truth, geo: T.Geometry) -> dict[str, Any]:
    true_paths = [truth.path(i) for i in range(truth.n)]
    emax, emean, eend, _ = BP.path_errors(  # type: ignore[no-untyped-call]
        geo, true_paths, cols.paths, cols.counts
    )
    return dict(
        max_mm=stats(emax),
        mean_mm=stats(emean),
        end_mm=stats(eend),
        _arrays=(emax, emean, eend),
    )


@dataclass
class LevelResult:
    phi_levels: Array
    tau_levels: Array
    pos: Array  # (N, L, 3) voxel space
    arc: Array
    found: Any
    dev: Array  # (N, L) signed mm, nan where not found


def measure_levels(
    pd: PathData,
    g: Any | None,
    geo: T.Geometry,
    tau_scale: float,
    phi_levels: Sequence[float],
    n: int,
) -> tuple[dict[str, Any], LevelResult]:
    phis = np.asarray(phi_levels, float)
    taus = (phis - 1.0) * tau_scale
    nl = len(taus)
    if g is None:
        empty = LevelResult(
            phis, taus, np.full((n, nl, 3), np.nan), np.full((n, nl), np.nan),
            np.zeros((n, nl), bool), np.full((n, nl), np.nan),
        )  # fmt: skip
        return dict(phi_levels=list(phis), tau_levels=list(taus), n_fitted=0), empty
    pos, arc, found = level_crossings(pd, g, np.broadcast_to(taus, (n, nl)))
    dev = np.full((n, nl), np.nan)
    if found.any():
        x = pos[found]
        lv = np.broadcast_to(taus, (n, nl))[found]
        dev[found] = (T.true_tau(geo, tau_scale, x) - lv) * T.true_speed(
            geo, tau_scale, x
        )
    per_level = [
        dict(
            phi=float(phis[j]),
            tau=float(taus[j]),
            n=int(found[:, j].sum()),
            abs_mm=stats(np.abs(dev[:, j])),
            signed_median_mm=stats(dev[:, j])["median"],
        )
        for j in range(nl)
    ]
    knots = np.linspace(-tau_scale, 0.0, 11)
    res = LevelResult(phis, taus, pos, arc, found, dev)
    rec = dict(
        phi_levels=list(phis),
        tau_levels=list(taus),
        n_levels_found=int(found.sum()),
        n_levels=int(found.size),
        abs_mm=stats(np.abs(dev)),
        signed_mm=stats(dev),
        per_level=per_level,
        map_solved_tau=list(knots),
        map_true_tau=list(g.predict(knots)),
    )
    return rec, res


def measure_spacing(
    pd: PathData,
    g: Any | None,
    truth: T.Truth,
    geo: T.Geometry,
    tau_scale: float,
    settings: T.SolverSettings,
    cols_arrival: Array,
) -> dict[str, Any]:
    n = truth.n
    m = np.arange(1, SPACING_LEVELS + 1)
    lv = truth.tau_arrival[:, None] - m[None, :] * settings.dtau
    solved = np.full(n, np.nan)
    true = np.full(n, np.nan)
    if g is not None:
        _, arc, found = level_crossings(pd, g, lv)
        ok = found.all(axis=1)
        solved[ok] = (arc[ok, 0] - arc[ok, -1]) / (SPACING_LEVELS - 1)
    for i in range(n):
        p = truth.path(i)
        if len(p) < 2:
            continue
        t = T.true_tau(geo, tau_scale, p)
        a = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1) * H)]
        )
        k, f, ok_i = _bracket(np.maximum.accumulate(t), lv[i])
        if ok_i.all():
            arc_i = a[k - 1] + f * (a[k] - a[k - 1])
            true[i] = (arc_i[0] - arc_i[-1]) / (SPACING_LEVELS - 1)
    ratio = solved / true
    # without the shared map: levels of the solved tau below the solved arrival
    lv_s = cols_arrival[:, None] - m[None, :] * settings.dtau
    _, arc_s, found_s = level_crossings(pd, _Identity(), lv_s)
    ok_s = found_s.all(axis=1)
    unmapped = np.full(n, np.nan)
    unmapped[ok_s] = (arc_s[ok_s, 0] - arc_s[ok_s, -1]) / (SPACING_LEVELS - 1)
    return dict(
        n_levels=SPACING_LEVELS,
        ratio_without_map=stats(unmapped / true),
        dtau=settings.dtau,
        solved_mm=stats(solved),
        true_mm=stats(true),
        ratio=stats(ratio),
        abs_ratio_error=stats(np.abs(ratio - 1.0)),
        _array=ratio,
    )


def measure_speeds(
    field: T.Field, extras: Mapping[str, Any], phantom: T.Phantom, geo: T.Geometry
) -> tuple[dict[str, Any], dict[str, Any]]:
    X = np.asarray(field.grid.nodes, float)
    Y = geo.to_local(X)
    lab = np.asarray(geo.label(Y))
    moving = field.q_stop < 0.5
    cortex = moving & (lab == T.CORTEX)
    phi = geo.phi(Y)
    interior = cortex & (phi >= INTERIOR[0]) & (phi <= INTERIOR[1])
    speed = np.asarray(extras["speed"], float)
    ratio = np.full(len(X), np.nan)
    ratio[cortex] = speed[cortex] / T.true_speed(geo, phantom.tau_scale, X[cortex])
    out: dict[str, Any] = dict(
        n_cortex_moving=int(cortex.sum()),
        n_interior=int(interior.sum()),
        ratio_cortex=stats(ratio[cortex]),
        rel_error_cortex=stats(np.abs(ratio[cortex] - 1.0)),
        ratio_interior=stats(ratio[interior]),
        rel_error_interior=stats(np.abs(ratio[interior] - 1.0)),
        banks=None,
    )
    bank = np.asarray(geo.bank(Y))
    prm = phantom.params
    if (
        "tA" in prm
        and "tB" in prm
        and (bank[cortex] == 0).any()
        and (bank[cortex] == 1).any()
    ):
        a, b = cortex & (bank == 0), cortex & (bank == 1)
        vt = np.full(len(X), np.nan)
        vt[cortex] = T.true_speed(geo, phantom.tau_scale, X[cortex])
        solved = float(np.median(speed[a]) / np.median(speed[b]))
        out["banks"] = dict(
            solved_ratio_A_over_B=solved,
            thickness_ratio_tA_over_tB=float(prm["tA"] / prm["tB"]),
            true_speed_ratio_A_over_B=float(np.median(vt[a]) / np.median(vt[b])),
            n_A=int(a.sum()),
            n_B=int(b.sum()),
        )
    return out, dict(ratio=ratio, interior=interior, cortex=cortex)


def grad_tau(grid: T.Grid, tau_mean: Array, moving: Any) -> Array:
    """The gradient of the mean tau (per mm) at each node, by finite differences along
    the
    grid axes between moving nodes (module docstring); nan where an axis has none."""
    nb = FG.axis_neighbours(grid)
    X = np.asarray(grid.nodes, float)
    n = len(X)
    out = np.full((n, 3), np.nan)
    ids = np.arange(n)
    for a in range(3):
        p, m = nb[:, 2 * a], nb[:, 2 * a + 1]
        okp = (
            (p >= 0)
            & moving[np.maximum(p, 0)]
            & np.isfinite(tau_mean[np.maximum(p, 0)])
        )
        okm = (
            (m >= 0)
            & moving[np.maximum(m, 0)]
            & np.isfinite(tau_mean[np.maximum(m, 0)])
        )
        pp, mm = np.maximum(p, 0), np.maximum(m, 0)
        both = okp & okm
        one_p = okp & ~okm
        one_m = okm & ~okp
        with np.errstate(invalid="ignore", divide="ignore"):
            cen = (tau_mean[pp] - tau_mean[mm]) / ((X[pp, a] - X[mm, a]) * H)
            fwd = (tau_mean[pp] - tau_mean[ids]) / ((X[pp, a] - X[ids, a]) * H)
            bwd = (tau_mean[ids] - tau_mean[mm]) / ((X[ids, a] - X[mm, a]) * H)
        out[:, a] = np.where(
            both, cen, np.where(one_p, fwd, np.where(one_m, bwd, np.nan))
        )
    out[~moving] = np.nan
    return out


def measure_direction(
    field: T.Field, phantom: T.Phantom, geo: T.Geometry, masks: Mapping[str, Any]
) -> dict[str, Any]:
    grid = field.grid
    X = np.asarray(grid.nodes, float)
    moving = field.q_stop < 0.5
    gt = grad_tau(grid, field.tau_mean(), moving)
    d = field.direction_mean()
    out: dict[str, Any] = {}
    for name in ("cortex", "interior"):
        sel = (
            masks[name] & np.isfinite(gt).all(axis=1) & (np.linalg.norm(d, axis=1) > 0)
        )
        dtrue = T.true_direction(geo, X[sel]) if sel.any() else np.zeros((0, 3))
        ug = _unit(gt[sel])
        out[name] = dict(
            n=int(sel.sum()),
            direction_vs_grad_tau_deg=stats(_angle_deg(d[sel], ug)),
            direction_vs_true_deg=stats(_angle_deg(d[sel], dtrue)),
            grad_tau_vs_true_deg=stats(_angle_deg(ug, dtrue)),
        )
    return out


def measure_jumps(cols: T.Columns, truth: T.Truth) -> dict[str, Any]:
    stopped = cols.end == int(T.EndReason.STOPPED)
    jump = np.where(stopped, cols.jump, np.nan)
    err = np.abs(jump - truth.jump)
    return dict(
        n_stopped=int(stopped.sum()),
        n_positive=int(np.nansum(jump > 0)),
        jump=stats(jump),
        abs_error_vs_truth=stats(err),
        tau_arrival=stats(np.where(stopped, cols.tau_arrival, np.nan)),
    )


# ---------------------------------------------------------------- the exact tests


def _plane_residuals(X_mm: Array) -> tuple[float, float, Array]:
    """(largest |distance|, rms, unit normal) of points from their least-squares
    plane."""
    c = X_mm.mean(axis=0)
    _, _, vt = np.linalg.svd(X_mm - c, full_matrices=False)
    nrm = vt[2]
    r = (X_mm - c) @ nrm
    return float(np.abs(r).max()), float(np.sqrt(np.mean(r * r))), np.asarray(nrm)


def column_window_angles(pd: PathData, geo: T.Geometry, window: int) -> Array:
    """For each column, the largest angle (degrees) between the chord of `window`
    samples
    and the true direction at its middle, over the samples inside INTERIOR; nan if the
    column has no such chord."""
    out = np.full(pd.n, np.nan)
    for i in range(pd.n):
        s = pd.sl(i)
        keep = (pd.phi[s] >= INTERIOR[0]) & (pd.phi[s] <= INTERIOR[1])
        P = pd.pos[s][keep]
        if len(P) <= window:
            continue
        chord = P[window:] - P[:-window]
        mid = 0.5 * (P[window:] + P[:-window])
        ang = _angle_deg(_unit(chord), T.true_direction(geo, mid))
        out[i] = float(np.nanmax(ang)) if np.isfinite(ang).any() else np.nan
    return out


def exact_slab(
    pd: PathData, lev: LevelResult, geo: T.Geometry, settings: T.SolverSettings
) -> dict[str, Any]:
    window = max(1, round(SLAB_WINDOW_MM / settings.rk4_step_mm))
    ang = column_window_angles(pd, geo, window)
    n_short = int((~np.isfinite(ang)).sum())
    worst = float(np.nanmax(ang)) if np.isfinite(ang).any() else float("nan")
    straight = dict(
        window_mm=window * settings.rk4_step_mm,
        angle_deg=stats(ang),
        n_columns=int(len(ang)),
        n_without_window=n_short,
        tolerance_deg=SLAB_ANGLE_DEG,
        **{"pass": bool(n_short == 0 and worst < SLAB_ANGLE_DEG)},
    )
    per = []
    worst_dev = 0.0
    short = False
    for j, phi in enumerate(lev.phi_levels):
        ok = lev.found[:, j]
        if ok.sum() < 4:
            per.append(dict(phi=float(phi), n=int(ok.sum())))
            short = True
            continue
        mx, rms, nrm = _plane_residuals(lev.pos[ok, j] * H)
        worst_dev = max(worst_dev, mx)
        true_n = T.true_direction(geo, lev.pos[ok, j][:1])[0]
        per.append(
            dict(
                phi=float(phi),
                n=int(ok.sum()),
                max_mm=mx,
                rms_mm=rms,
                normal_vs_true_deg=float(
                    np.degrees(np.arccos(min(1.0, abs(float(nrm @ true_n)))))
                ),
            )
        )
    planes = dict(
        per_level=per,
        max_mm=worst_dev,
        tolerance_mm=SLAB_LEVEL_MM,
        **{"pass": bool(not short and worst_dev < SLAB_LEVEL_MM)},
    )
    return {
        "columns_straight_normal": straight,
        "levels_planes": planes,
        "pass": bool(straight["pass"] and planes["pass"]),
    }


def _border_shift(u: Array, dev: Array, window: float) -> tuple[float, int, int]:
    """The step c of the fit dev = a + b u + c [u > 0] over |u| <= window."""
    sel = np.isfinite(dev) & (np.abs(u) <= window)
    uu, dd = u[sel], dev[sel]
    left, right = int((uu <= 0).sum()), int((uu > 0).sum())
    if min(left, right) < BORDER_MIN_POINTS:
        return float("nan"), left, right
    A = np.stack([np.ones_like(uu), uu, (uu > 0).astype(float)], axis=1)
    coef, *_ = np.linalg.lstsq(A, dd, rcond=None)
    return float(coef[2]), left, right


def exact_tau(
    lev: LevelResult,
    geo: T.Geometry,
    speeds: dict[str, Any],
) -> dict[str, Any]:
    p90 = speeds["rel_error_interior"]["p90"]
    speed = dict(
        statistic="p90 of |speed / true speed - 1|, moving nodes, 0.1 <= phi <= 0.9",
        observed=p90,
        median=speeds["rel_error_interior"]["median"],
        p99=speeds["rel_error_interior"]["p99"],
        n=speeds["n_interior"],
        tolerance=TAU_SPEED_REL,
        **{"pass": bool(np.isfinite(p90) and p90 < TAU_SPEED_REL)},
    )

    def cols_of(phis: Sequence[float]) -> list[int]:
        return [int(np.argmin(np.abs(lev.phi_levels - p))) for p in phis]

    below, above, band = (
        cols_of(TAU_BELOW_PHI),
        cols_of(TAU_ABOVE_PHI),
        cols_of(TAU_BAND_PHI),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shift = np.nanmean(lev.dev[:, above], axis=1) - np.nanmean(
            lev.dev[:, below], axis=1
        )
        shift = np.where(
            np.isfinite(lev.dev[:, above + below]).all(axis=1), shift, np.nan
        )
    med = stats(np.abs(shift))["median"]
    band_dev = dict(
        phi=list(TAU_BAND_PHI),
        abs_mm=stats(np.abs(lev.dev[:, band])),
        signed_median_mm=stats(lev.dev[:, band])["median"],
    )
    flat = dict(
        statistic=(
            "median over columns of |mean dev(phi 0.60, 0.65)"
            " - mean dev(phi 0.30, 0.35)|"
        ),
        observed=med,
        shift_mm=stats(np.abs(shift)),
        in_band=band_dev,
        tolerance_mm=TAU_BAND_MM,
        tolerance_from_plan=False,
        **{"pass": bool(np.isfinite(med) and med < TAU_BAND_MM)},
    )
    u = np.full(lev.found.shape, np.nan)
    if lev.found.any():
        u[lev.found] = geo.to_local(lev.pos[lev.found])[:, 1]
    per = []
    for j, phi in enumerate(lev.phi_levels):
        c, nl, nr = _border_shift(u[:, j], lev.dev[:, j], BORDER_WINDOW_MM)
        per.append(dict(phi=float(phi), shift_mm=c, n_left=nl, n_right=nr))
    cs = np.array([p["shift_mm"] for p in per])
    worst = float(np.nanmax(np.abs(cs))) if np.isfinite(cs).any() else float("nan")
    border = dict(
        statistic=(
            "largest |c| over levels of the fit dev = a + b u + c [u > 0], |u| <= 1 mm"
        ),
        window_mm=BORDER_WINDOW_MM,
        per_level=per,
        observed=worst,
        n_levels_without_fit=int((~np.isfinite(cs)).sum()),
        tolerance_mm=TAU_BORDER_MM,
        **{
            "pass": bool(
                np.isfinite(worst)
                and worst < TAU_BORDER_MM
                and not (~np.isfinite(cs)).any()
            )
        },
    )
    return {
        "speeds": speed,
        "flat_band_aligned": flat,
        "border_no_shift": border,
        "pass": bool(speed["pass"] and flat["pass"] and border["pass"]),
    }


# ---------------------------------------------------------------- the views


def _save(fig: Any, path: Path) -> int:
    """Save a figure as PNG below MAX_PNG_BYTES (a lower dpi if it is larger)."""
    size = 0
    for dpi in DPIS:
        fig.savefig(path, dpi=dpi)
        size = path.stat().st_size
        if size < MAX_PNG_BYTES:
            break
    plt.close(fig)
    return int(size)


def _sample_q_stop(field: T.Field, X: Array) -> Array:
    grid = field.grid
    out = np.full(len(X), np.nan)
    cell, loc = grid.locate(X)
    ok = cell >= 0
    if ok.any():
        w = T.trilinear_weights(loc[ok])
        corners = np.asarray(grid.cells)[cell[ok]]
        out[ok] = np.sum(field.q_stop[corners] * w, axis=1)
    return out


def view_section(
    path: Path,
    phantom: T.Phantom,
    field: T.Field,
    cols: T.Columns,
    truth: T.Truth,
    geo: T.Geometry,
) -> int:
    name = phantom.name
    centre = FV.box_centre_local(geo)
    if name.startswith("crown"):
        planes = [FV.crown_plane(geo, truth)]
    else:
        planes = [
            FV.Plane(
                p.h,
                p.v,
                p.fixed,
                FV.snap_value(
                    geo,
                    truth,
                    p,
                    float(centre @ p.basis()[2] if p.value is None else p.value),
                ),
                p.note,
            )
            for p in FV.planes_of(name)
        ]
    fig, axes = plt.subplots(
        1, len(planes), figsize=(5.5 * len(planes), 5.0), squeeze=False
    )
    for ax, plane in zip(axes[0], planes, strict=True):
        sec = FV.make_section(geo, phantom, plane)
        ext = FV.extent_of(sec)
        v = sec.image[sec.inbox]
        lo, hi = np.percentile(v, [1, 99])
        cm = plt.get_cmap("gray").with_extremes(bad="white")
        ax.imshow(
            np.ma.masked_invalid(sec.image), origin="lower", extent=ext, cmap=cm,
            vmin=lo, vmax=hi, interpolation="nearest", zorder=1,
        )  # fmt: skip
        if sec.surface.size:
            ax.scatter(sec.surface[:, 0], sec.surface[:, 1], s=1.6, marker="s",
                       c=FV.SURFACE_COLOUR, linewidths=0, zorder=3)  # fmt: skip
        HH, VV = np.meshgrid(sec.hs, sec.vs)
        Xg = geo.to_vox(sec.local(HH, VV))
        Q = _sample_q_stop(field, Xg).reshape(HH.shape)
        if np.isfinite(Q).any() and np.nanmin(Q) < 0.5 < np.nanmax(Q):
            ax.contour(sec.hs, sec.vs, np.ma.masked_invalid(Q), levels=[0.5],
                       colors="#d62728", linewidths=1.0, zorder=4)  # fmt: skip
        eh, ev, _ = plane.basis()
        pick, nnear, _ = FV.near_columns(geo, truth, sec)
        pick = pick[:SECTION_COLUMNS]
        for i in pick:
            Pt = geo.to_local(truth.path(i))
            ax.plot(Pt @ eh, Pt @ ev, "-", c="#1f77b4", lw=0.7, zorder=5)
            Ps = geo.to_local(cols.paths[i, : cols.counts[i]])
            ax.plot(Ps @ eh, Ps @ ev, "-", c="#ff7f0e", lw=0.7, zorder=6)
            ax.plot(Ps[-1:] @ eh, Ps[-1:] @ ev, "o", ms=2.0, mfc="#ff7f0e", mec="w",
                    mew=0.3, zorder=7)  # fmt: skip
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_xlabel(plane.label("h"), fontsize=8)
        ax.set_ylabel(plane.label("v"), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{plane.note}\n{len(pick)} columns ({nnear} near the plane)",
                     fontsize=8)  # fmt: skip
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([], [], c="#d62728", lw=1.2, label="solved stop (q_stop = 0.5)"),
            Line2D([], [], c=FV.SURFACE_COLOUR, marker="s", ls="", label="true stop"),
            Line2D([], [], c="#ff7f0e", lw=1.2, label="solved column, its stop"),
            Line2D([], [], c="#1f77b4", lw=1.2, label="true column"),
        ],
        loc="lower center", ncol=4, fontsize=7, frameon=False,
    )  # fmt: skip
    fig.suptitle(f"{name}  noise {phantom.noise:g}  detrended echo 2", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _save(fig, path)


def view_levels(
    path: Path, pd: PathData, g: Any | None, lev: LevelResult, name: str
) -> int:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    sel = np.isfinite(pd.tau) & (pd.phi >= 0) & (pd.phi <= 1)
    x, y = pd.tau[sel], pd.tau_true[sel]
    if x.size:
        stride = max(1, math.ceil(x.size / FIT_MAX_POINTS))
        h = a1.hist2d(x[::stride], y[::stride], bins=120, cmap="viridis", cmin=1)
        fig.colorbar(h[3], ax=a1, label="samples")
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        a1.plot(lim, lim, "w--", lw=0.8, label="identity")
        if g is not None:
            xs = np.linspace(x.min(), x.max(), 200)
            a1.plot(xs, g.predict(xs), c="#ff7f0e", lw=1.0, label="monotone map g")
        a1.legend(fontsize=7, loc="upper left")
    a1.set_xlabel("solved tau", fontsize=8)
    a1.set_ylabel("true tau at the sample", fontsize=8)
    a1.set_title(f"{name}: solved tau against true tau (0 <= phi <= 1)", fontsize=9)
    data = [lev.dev[lev.found[:, j], j] for j in range(len(lev.phi_levels))]
    if any(len(d) for d in data):
        a2.boxplot(
            [d if len(d) else [np.nan] for d in data],
            positions=lev.phi_levels,
            widths=0.012,
            whis=(5, 95),
            showfliers=False,
        )
    a2.axhline(0.0, c="0.5", lw=0.6)
    a2.set_xlim(lev.phi_levels.min() - 0.03, lev.phi_levels.max() + 0.03)
    a2.set_xticks(np.round(lev.phi_levels, 3))
    a2.tick_params(axis="x", labelsize=6, labelrotation=90)
    a2.set_xlabel("true level (relative depth phi)", fontsize=8)
    a2.set_ylabel("solved minus true level position (mm)", fontsize=8)
    a2.set_title("level deviation after the shared map (whiskers 5-95 %)",
                 fontsize=9)  # fmt: skip
    fig.tight_layout()
    return _save(fig, path)


def view_errors(path: Path, arrays: Mapping[str, Array], name: str) -> int:
    items = [
        ("stop_abs_mm", "|stop_sdf| at the solved stop (mm)"),
        ("path_max_mm", "largest distance from the true column (mm)"),
        ("path_mean_mm", "mean distance from the true column (mm)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for ax, (key, label) in zip(axes, items, strict=True):
        v = np.asarray(arrays[key], float)
        v = v[np.isfinite(v)]
        if v.size:
            top = float(v.max()) if v.max() > 0 else 1.0
            ax.hist(v, bins=np.linspace(0.0, top * 1.0001, 50), color="#4c78a8")
            for p, ls in zip((50, 90, 99), ("-", "--", ":"), strict=True):
                ax.axvline(np.percentile(v, p), c="k", ls=ls, lw=0.9,
                           label=f"p{p} = {np.percentile(v, p):.3g}")  # fmt: skip
            ax.legend(fontsize=7)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("columns", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{name}: stop and path errors", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def view_free_energy(path: Path, rec: T.RunRecord, name: str) -> int:
    nw = max(1, len(rec.free_energy))
    fig, axes = plt.subplots(1, nw, figsize=(4.2 * nw, 3.6), squeeze=False)
    widths = rec.settings.get("widths", [None] * nw)
    for k, ax in enumerate(axes[0]):
        fl = rec.free_energy[k] if k < len(rec.free_energy) else []
        ax.plot(range(1, len(fl) + 1), fl, "o-", ms=3, lw=0.9, c="#4c78a8")
        for r in rec.free_energy_rises:
            if int(r.get("width", -1)) == widths[k] and 1 <= int(r["sweep"]) <= len(fl):
                ax.plot(int(r["sweep"]), fl[int(r["sweep"]) - 1], "rx", ms=8)
        ax.set_xlabel("sweep", fontsize=8)
        ax.set_ylabel("free energy F", fontsize=8)
        ax.set_title(f"width {widths[k]}: {len(fl)} sweeps", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{name}: free energy per sweep (red x: a rise)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, path)


# ---------------------------------------------------------------- the run


def measure_all(
    phantom: T.Phantom,
    field: T.Field,
    extras: Mapping[str, Any],
    cols: T.Columns,
    info: dict[str, Any],
    truth: T.Truth,
    settings: T.SolverSettings,
    *,
    source: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every measure of the module docstring. Returns (measures, internals): the
    measures
    are JSON-able; the internals hold the arrays that the views and column_errors.npz
    use.
    The keys of measures: stops, paths, levels, speeds, spacing_near_stop, topology,
    direction, jumps, exact (and later inference, cost, inputs, set by `run`)."""
    geo = phantom.geometry
    assert geo is not None
    ts = phantom.tau_scale
    pd = PathData.of(cols, geo, ts)
    g = fit_tau_map(pd)
    is_tau = phantom.name == "tau"
    phis = sorted(
        set(LEVEL_PHI)
        | (set(TAU_BELOW_PHI + TAU_ABOVE_PHI + TAU_BAND_PHI) if is_tau else set())
    )
    stops = measure_stops(cols, truth, geo, info)
    paths = measure_paths(cols, truth, geo)
    levels, lev = measure_levels(pd, g, geo, ts, phis, truth.n)
    spacing = measure_spacing(pd, g, truth, geo, ts, settings, cols.tau_arrival)
    speeds, spd = measure_speeds(field, extras, phantom, geo)
    direction = measure_direction(field, phantom, geo, spd)
    report = FC.topology(field, cols, info, geo, settings)
    jumps = measure_jumps(cols, truth)
    sd = stops.pop("_array")
    emax, emean, eend = paths.pop("_arrays")
    ratio = spacing.pop("_array")
    conditions = dict(noise_zero=bool(phantom.noise == 0.0), oracle=source == "oracle")
    exact: dict[str, Any] = dict(
        conditions=conditions, applies=all(conditions.values())
    )
    if phantom.name == "slab":
        exact["slab"] = exact_slab(pd, lev, geo, settings)
    elif is_tau:
        exact["tau"] = exact_tau(lev, geo, speeds)
    measures = dict(
        stops=stops,
        paths=paths,
        levels=levels,
        speeds=speeds,
        spacing_near_stop=spacing,
        topology=FC.counts(report),
        direction=direction,
        jumps=jumps,
        exact=exact,
    )
    internals = dict(
        pd=pd,
        g=g,
        lev=lev,
        arrays=dict(
            stop_signed_mm=sd,
            stop_abs_mm=np.abs(sd),
            path_max_mm=emax,
            path_mean_mm=emean,
            path_end_mm=eend,
            spacing_ratio=ratio,
            level_dev_mm=lev.dev,
            level_phi=lev.phi_levels,
        ),
    )
    return measures, internals


def run(
    phantom: T.Phantom,
    inputs: T.Inputs,
    truth: T.Truth,
    grid: T.Grid,
    out_dir: Path,
    *,
    source: T.InputSource,
    grid_spec: str,
    settings: T.SolverSettings = T.DEFAULTS,
    changed_settings: Mapping[str, Any] | None = None,
    tau_prior: bool = False,
    profile_dtype: Any = np.float64,
    init: T.Field | None = None,
    origin: str = "top",
    anchor_node: int | None = None,
    versions: Mapping[str, str] | None = None,
    log_fn: LogFn | None = None,
    recentre_at_edge: bool = False,
) -> T.RunRecord:
    """One run into `out_dir` (the module docstring). Returns the record that it
    wrote. init, origin, anchor_node and recentre_at_edge go to `ff_meanfield.solve`
    (init: a start field only, not a prior; recentre_at_edge: the provisional option
    CD-13, default off). measures["inference"] holds the log's edge_repeats (empty with
    the option off). run.json records the origin in RunRecord.origin, and
    measures["origin"] holds the origin, the anchor node, its position (voxel) and
    whether an init field was given. The run directory does not depend on the origin:
    give an anchor run its own `out_dir`."""
    geo = phantom.geometry
    if geo is None:
        raise ValueError(f"phantom {phantom.name!r} has no geometry, so no truth")
    if truth.phantom != phantom.name:
        raise ValueError(
            f"the truth is of {truth.phantom!r}, the phantom is {phantom.name!r}"
        )
    if truth.n == 0:
        raise ValueError("the truth has no roots")
    phi0 = np.abs(geo.phi(geo.to_local(truth.roots)))
    if not phi0.max() < 1e-6:
        raise ValueError(
            "the truth's roots are not on phi = 0 of this phantom: wrong box?"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    noise_sd = [float(s) for s in phantom.image["noise_sd"]]
    t0 = time.perf_counter()
    field, extras, log = MF.solve(
        grid, inputs, noise_sd, settings, tau_prior=tau_prior, init=init,
        origin=origin, anchor_node=anchor_node, profile_dtype=profile_dtype,
        log_fn=log_fn, recentre_at_edge=recentre_at_edge,
    )  # fmt: skip
    t_solve = time.perf_counter() - t0
    t0 = time.perf_counter()
    cols, info = FI.integrate(
        field, truth.starts, truth.roots, settings,
        levels=(np.asarray(LEVEL_PHI) - 1.0) * phantom.tau_scale,
    )  # fmt: skip
    t_integrate = time.perf_counter() - t0
    t0 = time.perf_counter()
    measures, internals = measure_all(
        phantom, field, extras, cols, info, truth, settings, source=source
    )
    t_measure = time.perf_counter() - t0
    field.to_npz(out_dir / "field.npz", extras)
    cols.to_npz(out_dir / "columns.npz")
    np.savez_compressed(out_dir / "column_errors.npz", **internals["arrays"])
    rec = T.RunRecord(
        phantom=phantom.record(),
        source=source,
        grid=grid_record(grid, grid_spec),
        settings=settings.record(),
        changed_settings=dict(changed_settings or {}),
        origin=origin,
        versions=dict(versions or {}),
        free_energy=log["free_energy"],
        sweeps=[int(s) for s in log["sweeps"]],
        free_energy_rises=log["free_energy_rises"],
        pruning=log["pruning"],
        defects=list(log["defects"]),
        measures=measures,
    )
    rec.measures["inference"] = dict(
        sweeps_per_width=rec.sweeps,
        end_reason=log["end_reason"],
        free_energy_last=[fl[-1] if fl else None for fl in log["free_energy"]],
        n_rises=len(log["free_energy_rises"]),
        terms=log["terms"],
        run=log["run"],
        edge_repeats=log["edge_repeats"],
    )
    rec.measures["origin"] = dict(
        origin=origin,
        anchor_node=log["run"]["anchor_node"],
        anchor_position_vox=log["run"]["anchor_position_vox"],
        init=init is not None,
    )
    rec.measures["inputs"] = {
        **inputs.record(),
        **(inputs.stop.record() if hasattr(inputs.stop, "record") else {}),
    }
    pd, g, lev = internals["pd"], internals["g"], internals["lev"]
    views: list[tuple[str, Callable[[Path], int]]] = [
        ("section.png", lambda p: view_section(p, phantom, field, cols, truth, geo)),
        ("levels.png", lambda p: view_levels(p, pd, g, lev, phantom.name)),
        ("errors.png", lambda p: view_errors(p, internals["arrays"], phantom.name)),
        ("free_energy.png", lambda p: view_free_energy(p, rec, phantom.name)),
    ]  # fmt: skip
    sizes: dict[str, int] = {name: fn(out_dir / name) for name, fn in views}
    rec.measures["views"] = sizes
    rec.time_s = time.perf_counter() - t_start
    rec.peak_mem_bytes = _peak_rss_bytes()
    rec.measures["cost"] = dict(
        time_s=rec.time_s,
        solve_s=t_solve,
        solve_s_per_width=log["time_s"],
        integrate_s=t_integrate,
        measure_s=t_measure,
        peak_mem_bytes=rec.peak_mem_bytes,
        n_nodes=int(grid.n_nodes),
        n_columns=int(truth.n),
    )
    rec.to_json(out_dir / "run.json")
    return rec


# ---------------------------------------------------------------- the CLI


def _truth_path(arg: Path, name: str) -> Path:
    return arg if arg.suffix == ".npz" else arg / name / "truth.npz"


def main(argv: Sequence[str] | None = None) -> T.RunRecord:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("phantom", choices=[n for n in FTR.PHANTOMS if n != "noise"])
    ap.add_argument("--source", choices=("oracle", "image"), required=True)
    ap.add_argument("--noise", type=float, required=True)
    ap.add_argument("--grid", default="uniform:0.16")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--site", default="V1V2_banks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape", help="X,Y,Z voxels: a box that is not the plan's")
    ap.add_argument(
        "--profile-dtype", choices=("float32", "float64"), default="float64"
    )
    ap.add_argument("--tau-prior", action="store_true")
    ap.add_argument("--origin", choices=("top", "anchor"), default="top")
    ap.add_argument(
        "--anchor-node", type=int, default=None, help="node id; needs --origin anchor"
    )
    ap.add_argument(
        "--set", dest="sets", action="append", default=[], metavar="NAME=VALUE"
    )
    ap.add_argument(
        "--recentre-at-edge",
        action="store_true",
        help="the provisional option CD-13 of ff_meanfield.solve (default off)",
    )
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    settings, changed = parse_settings(a.sets)
    shape = tuple(int(s) for s in a.shape.split(",")) if a.shape else None
    if shape is not None and len(shape) != 3:
        ap.error("--shape needs three integers")
    feats = FP.features(a.site, str(a.features))  # type: ignore[no-untyped-call]
    phantom = FTR.build_phantom(
        a.phantom,
        feats,
        site=a.site,
        noise=a.noise,
        seed=a.seed,
        shape=shape,
    )
    reference = None
    if a.source == "image" and a.noise != 0.0:
        reference = FTR.build_phantom(
            a.phantom,
            feats,
            site=a.site,
            noise=0.0,
            seed=a.seed,
            shape=shape,
        )
    inputs = make_inputs(phantom, a.source, reference)
    grid = make_grid(a.grid, phantom.shape)
    tpath = _truth_path(a.truth, a.phantom)
    truth = T.Truth.from_npz(tpath)
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    versions = {
        "git_commit": commit or "unknown",
        "git_dirty": "unknown" if dirty is None else str(bool(dirty)).lower(),
        "truth_sha256": _sha256(tpath),
        "features_sha256": _sha256(a.features),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    out = T.run_dir(a.out, phantom.name, a.source, a.noise, grid_label(a.grid))
    log_fn: LogFn | None = None if a.quiet else (lambda s: print(s, flush=True))
    rec = run(
        phantom, inputs, truth, grid, out,
        source=a.source, grid_spec=a.grid, settings=settings, changed_settings=changed,
        tau_prior=a.tau_prior, profile_dtype=np.dtype(a.profile_dtype).type,
        origin=a.origin, anchor_node=a.anchor_node, versions=versions, log_fn=log_fn,
        recentre_at_edge=a.recentre_at_edge,
    )  # fmt: skip
    if not a.quiet:
        print(
            f"wrote {out} ({rec.time_s:.1f} s, peak {rec.peak_mem_bytes / 1e9:.2f} GB)"
        )
    return rec


if __name__ == "__main__":
    main()
