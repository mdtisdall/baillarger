"""The views and numbers of the kissing surfaces (docs/plans/flow-field-phantoms.md,
*PR 4*, task 4.2): questions 1, 3, 4, 5 and 6, and the independence test (R6).

The script reads finished runs of the connected-kiss phantoms (ff_run.py and
ff_kiss_runs.py write them). It does not solve. It writes one JSON file, one markdown
file of tables and a few PNG figures. All numbers are diagnostics, not pass/fail rules.
Each distribution is given as n, median, p90, p99 and max (`ff_run.stats`). Lengths are
in mm; positions in "local" coordinates are in the phantom's local mm frame (bank A at
x > 0, bank B at x < 0, slot half-plane x = 0, y >= 0).

Usage:
    python ff_kiss_views.py --runs RUNS_DIR --truth TRUTH_DIR --figures FIG_DIR
        --out OUT_DIR [--grid-label uniform0.16] [--label TEXT] [--fig-suffix TEXT]
        [--phantoms NAME ...]

--label puts a text in every figure title, at the top of kiss_views.md and in
kiss_views.json ("label"). --fig-suffix goes at the end of each figure's file name
(before .png), so that the figures of two grids do not overwrite each other.

A run directory is RUNS_DIR/<phantom>/<source>_<noise:g>_<grid label>[_coarse|_anchor|
_shiftB]/ with field.npz, columns.npz, column_errors.npz and run.json. A run without
run.json is "not run". TRUTH_DIR holds TRUTH_DIR/<phantom>/truth.npz.

What the script computes, per run
---------------------------------
- **Q1 fused contact.** The columns whose true stop kind is CONTACT. The stop error
  (column_errors.npz) per bank. The end reasons. The fraction of flat-bank columns (true
  stop at local y >= 0) that stop on their own side of the slot plane (the sign of the
  local x of the solved stop). The contact crossings (`contact_crossings`, tolerance
  CROSS_TOL_MM), also on the true paths as a reference. The overlap and focal counts of
  run.json. The width of the solved stop boundary: along the straight line through each
  stop, in the direction of the column's last step, the distance from the nearest point
  before the stop where the interpolated q_stop is 0.2 to the nearest point after the
  stop where it is 0.8 (`ramp_width`). The line is an extension of the column, because
  the column ends at q_stop = 0.5. No sampled fields: this width is the spread.
- **Q3 gap end.** The |stop error| binned by the local z of the true stop (Z_BIN_MM).
  The same in three regions: open (z < −window), the window round the transition (|z| <=
  window: WINDOW_MM) and fused (z > window). The topology places (recounted from the
  saved field and columns with ff_checks and ff_integrate; the recount is compared with
  run.json) in the window and elsewhere, with the moving cortex nodes in each region as
  a reference.
- **Q4 the stopped region.** Sections of q_stop normal to local z at SECTION_Z_MM (the
  figure). The stopped sheet at the fused contact: along lines in local x through the
  slot plane at (y, z) in SHEET_YS_MM x SHEET_ZS_MM, the length where the interpolated
  q_stop > 0.5, against the true 2 g and the grid spacing. At the open slot (local z <
  OPEN_MAX_Z_MM): the fraction of true outside-medium nodes with q_stop > 0.5, and of
  true cortex nodes deeper than one grid spacing (stop_sdf < −spacing) with
  q_stop > 0.5.
- **Q5 τ at the contact.** The bank speed ratio of run.json, against tA / tB.
  tau_arrival per bank at the CONTACT stops (true value 0). The level deviations
  (column_errors.npz level_dev_mm) per bank: |deviation| over all levels, and the
  signed median per level.
- **Inference.** The sweeps per width, the cap (settings.max_sweeps), the end reason per
  width (run.json measures.inference.end_reason), whether the width reached the cap,
  and the count of free-energy rises (`inference_summary`). One figure holds the
  free-energy trace of every run, per width.
- **R5, R7, R11.** The topology counts and the angle between the direction and grad τ
  (run.json, measures.direction.cortex), copied.

What the script computes, per pair of runs
------------------------------------------
- **Q6 origin** (the plain run and the _anchor run of one phantom, source and noise).
  One constant c = mean(τ_anchor − τ_top) over the nodes that move in both runs
  (q_stop < 0.5), and the residual τ_anchor − τ_top − c. The solved levels: for each
  column and each level of columns.npz, the first point of the column where τ reaches
  the level (τ − c in the anchor run); the distance between the two runs' points. The
  sweeps per width of the plain, the anchor and the coarse run, and of the coarse
  solve (coarse.json).
- **R6 independence** (the plain run and the _shiftB run). At the nodes that move in
  both runs, |Δ tau_mean| and the angle between the two direction_mean, binned by the
  distance from the true stop surface (−stop_sdf, bins of one grid spacing up to
  R6_MAX_MM), for bank A (where no change is expected) and bank B (where bank B's image
  changed: the reference). Per column, the largest distance from the shorter of the two
  runs' paths to the other (`path_distance`) and the distance between the two stops,
  by bank and true stop kind. stop_sdf is computed only at nodes with
  d0 = r − g(z) <= R6_MAX_MM + (g_wide − g_thin). At the other nodes the true distance
  is at least d0 − (g_wide − g_thin) > R6_MAX_MM, so they are in no bin; the script
  stores −d0 for them.

Memory: a run's field.npz is read in parts. The full candidate arrays (q, direction) are
read once for the sinks of low resultant length and then freed: at 0.16 mm on the
standard box they are about 1.2 GB.

NumPy or Torch: NumPy. There is no kernel here: the script makes one pass over saved
arrays of finished runs, and the solver, not this script, is the cost of the stage.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import ff_checks as FC  # noqa: E402
import ff_ckiss as CK  # noqa: E402
import ff_grid as FG  # noqa: E402
import ff_integrate as FI  # noqa: E402
import ff_phantom_views as FV  # noqa: E402
import ff_run as FR  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402

H = T.VOXEL_MM
Array = T.Array
stats = FR.stats

# ---- choices of this script, fixed before any run at 0.16 mm
CROSS_TOL_MM = CK.G_THIN
"""A flat-bank column crosses the contact if a sample with local y >= 0 lies farther
than this (mm) on the far side of the slot plane x = 0. The value is g_thin, the
half-width of the fused slot: a sample farther than g_thin past x = 0 is past the far
face of the fused slot, so in the other bank's cortex (fused part) or in the open slot.
A column that stops correctly ends on its own face, at |x| >= g_thin on its own side.
The path step (0.01 mm) is below this value."""
WINDOW_MM = {"smooth": CK.SMOOTH_HALF, "sharp": 0.5}
"""The half-width (mm, in local z) of the window round the gap's end (the brief)."""
Z_BIN_MM = H
SECTION_Z_MM = (-4.0, 0.0, 4.0)  # open slot, gap end, fused part
SECTION_NOTES = ("open slot", "gap end", "fused contact")
SECTION_X_MM = (-2.4, 3.4)  # the section window in local x: bank B (1.2 mm) to bank A
SECTION_Y_MM = (-3.2, 5.0)
SHEET_YS_MM = (0.5, 1.5, 3.0)
SHEET_ZS_MM = (3.0, 4.0, 5.0)
SHEET_HALF_MM = 1.0  # the line goes from x = −1 to x = +1 mm
SHEET_STEP_MM = 0.002  # 20 samples across the true sheet (2 g_thin = 0.04 mm)
OPEN_MAX_Z_MM = -CK.SMOOTH_HALF  # the open slot: g = g_wide there in all four phantoms
RAMP_LO, RAMP_HI = 0.2, 0.8
RAMP_REACH_MM = 1.5  # the line through the stop goes this far each way
RAMP_STEP_MM = 0.005
RAMP_CHUNK = 100  # columns per batch of samples
R6_MAX_MM = 2.0
LEVEL_MAX_DIST_MM = 2.0  # not a filter: only the figure's scale of the level distances
SOURCES: tuple[T.InputSource, ...] = ("oracle", "image")
NOISES = (0.0, 1.0)
VARIANTS = ("", "_coarse", "_anchor", "_shiftB")
EXPECTED_VARIANTS = (("ckiss_smooth", "oracle", 0.0),)  # plan, task 4.1
MOVING = 0.5  # a node moves if q_stop < this (the integrator's stop level)
RUN_COLOURS = {
    ("oracle", 0.0): ("#1f77b4", "-"),
    ("oracle", 1.0): ("#1f77b4", "--"),
    ("image", 0.0): ("#d55e00", "-"),
    ("image", 1.0): ("#d55e00", "--"),
}


# ---------------------------------------------------------------- tested helpers


def contact_crossings(
    paths_local: Array, counts: T.IntArray, bank: Any, tol: float = CROSS_TOL_MM
) -> tuple[T.BoolArray, T.IntArray]:
    """(crosses (N,), first (N,)) for padded local paths (N, m, 3): a column crosses if
    a sample k < counts[i] with local y >= 0 lies on the far side of x = 0 by more than
    tol (x < −tol for bank 0, x > tol for bank 1). first: the index of the first such
    sample, −1 if none."""
    P = np.asarray(paths_local, float)
    n, m = P.shape[:2]
    valid = np.arange(m)[None, :] < np.asarray(counts)[:, None]
    sgn = np.where(np.asarray(bank) == 0, 1.0, -1.0)[:, None]
    with np.errstate(invalid="ignore"):
        far = valid & (P[..., 1] >= 0.0) & (sgn * P[..., 0] < -tol)
    crosses = far.any(axis=1)
    first = np.where(crosses, np.argmax(far, axis=1), -1).astype(np.int64)
    return np.asarray(crosses), first


def own_side(stops_local: Array, bank: Any) -> T.BoolArray:
    """True where the local x of a stop has the sign of its bank (bank 0: x > 0; bank 1:
    x < 0). A stop at x = 0 exactly is on neither side."""
    x = np.asarray(stops_local, float)[:, 0]
    b = np.asarray(bank)
    return np.asarray(np.where(b == 0, x > 0.0, x < 0.0))


def sheet_length(s: Array, q: Array, level: float) -> float:
    """The total length of the parts of a sampled line where q > level. q is linear
    between samples, so each crossing is interpolated. Samples with nan q count as
    below the level."""
    s = np.asarray(s, float)
    q = np.nan_to_num(np.asarray(q, float), nan=-np.inf)
    if len(s) < 2:
        return 0.0
    a, b = q[:-1], q[1:]
    ds = np.diff(s)
    above_a, above_b = a > level, b > level
    frac = np.where(above_a & above_b, 1.0, 0.0)
    one = above_a ^ above_b
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (level - a) / (b - a)  # where the segment crosses the level
    t = np.where(np.isfinite(t), np.clip(t, 0.0, 1.0), 0.5)
    frac = np.where(one & above_a, t, frac)  # above at the start: the part before t
    frac = np.where(one & above_b, 1.0 - t, frac)
    return float(np.sum(frac * ds))


def _cross(s: Array, q: Array, i: int, j: int, level: float) -> float:
    """The interpolated s between samples i and j where q equals level."""
    d = q[j] - q[i]
    if d == 0:
        return float(0.5 * (s[i] + s[j]))
    return float(s[i] + (level - q[i]) / d * (s[j] - s[i]))


def ramp_width(
    s: Array, q: Array, s0: float, lo: float = RAMP_LO, hi: float = RAMP_HI
) -> float:
    """The width of the stop boundary along a line: the s of the nearest point at or
    after s0 where q reaches hi, minus the s of the nearest point at or before s0 where
    q falls to lo (going back). nan if either is not found. s ascending."""
    s = np.asarray(s, float)
    q = np.asarray(q, float)
    n = len(s)
    j0 = (
        int(np.searchsorted(s, s0, side="right")) - 1
    )  # the last sample at or before s0
    i0 = int(np.searchsorted(s, s0, side="left"))  # the first sample at or after s0
    below = np.flatnonzero(q[: j0 + 1] <= lo) if j0 >= 0 else np.zeros(0, int)
    above = np.flatnonzero(q[i0:] >= hi) + i0 if i0 < n else np.zeros(0, int)
    if len(below) == 0 or len(above) == 0:
        return float("nan")
    j, i = int(below[-1]), int(above[0])
    lo_s = _cross(s, q, j, j + 1, lo) if j + 1 < n and q[j + 1] > lo else float(s[j])
    hi_s = _cross(s, q, i - 1, i, hi) if i >= 1 and q[i - 1] < hi else float(s[i])
    return float(hi_s - lo_s)


def fit_constant(a: Array, b: Array) -> tuple[float, Array, int]:
    """(c, residual, n): the least-squares constant c with b ≈ a + c over the pairs
    where both are finite (c = mean(b − a)), the residual b − a − c (nan elsewhere)
    and the count of pairs."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    ok = np.isfinite(d)
    if not ok.any():
        return float("nan"), np.full(d.shape, np.nan), 0
    c = float(d[ok].mean())
    return c, np.where(ok, d - c, np.nan), int(ok.sum())


def bin_stats(values: Array, dist: Array, edges: Array) -> list[dict[str, Any]]:
    """For each bin [edges[k], edges[k + 1]) of dist: lo, hi and `stats` of the finite
    values with dist in the bin."""
    v = np.asarray(values, float)
    d = np.asarray(dist, float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        with np.errstate(invalid="ignore"):
            sel = (d >= lo) & (d < hi)
        out.append(dict(lo=float(lo), hi=float(hi), **stats(v[sel])))
    return out


def distance_edges(spacing_mm: float, max_mm: float) -> Array:
    """Bin edges 0, spacing, 2 spacing, ... below max_mm, and max_mm as the last
    edge."""
    n = int(np.ceil(round(max_mm / spacing_mm, 9)))
    e = np.arange(n + 1) * spacing_mm
    e[-1] = max_mm
    return e


def first_crossings(tau: Array, pos: Array, levels: Array) -> Array:
    """(L, 3): for each level, the first point of a column where τ (made non-decreasing
    by a running maximum) reaches the level, linear between samples; nan if never. The
    first sample counts only if τ there equals the level."""
    t = np.asarray(tau, float)
    P = np.asarray(pos, float)
    ok = np.isfinite(t)
    t, P = t[ok], P[ok]
    out = np.full((len(levels), 3), np.nan)
    if len(t) < 2:
        return out
    tm = np.maximum.accumulate(t)
    k = np.searchsorted(tm, levels, side="left")
    for j, (kk, lv) in enumerate(zip(k, levels, strict=True)):
        if kk == 0 and tm[0] == lv:
            out[j] = P[0]
        elif 1 <= kk < len(tm):
            d = tm[kk] - tm[kk - 1]
            f = (lv - tm[kk - 1]) / d if d > 0 else 0.0
            out[j] = P[kk - 1] + f * (P[kk] - P[kk - 1])
    return out


def sample_nodes(grid: T.Grid, values: Array, X: Array) -> Array:
    """The trilinear interpolation of node values at voxel positions X (nan outside the
    grid). For q_stop this is the integrator's stop probability."""
    out = np.full(len(X), np.nan)
    cell, loc = grid.locate(np.asarray(X, float))
    ok = cell >= 0
    if ok.any():
        w = T.trilinear_weights(loc[ok])
        corners = np.asarray(grid.cells)[cell[ok]]
        out[ok] = np.sum(np.asarray(values)[corners] * w, axis=1)
    return out


def path_distance(A: Array, B: Array) -> float:
    """The largest distance (in the units of A and B) from a sample of the shorter path
    (fewer samples) to the samples of the longer path. So a column that only goes on
    farther than the other does not count: this is the change across the paths, and the
    distance between the stops is a separate measure."""
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    short, long_ = (A, B) if len(A) <= len(B) else (B, A)
    return float(cKDTree(long_).query(short)[0].max())


END_CODES = {"tol_q": "q", "tol_free_energy": "F", "max_sweeps": "CAP"}


def inference_summary(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The solver's inference record of one run: sweeps per width, the cap
    (settings.max_sweeps), the end reason per width, at_cap per width (the width ended
    because it reached the cap: end reason "max_sweeps", or, if the end reason is
    missing, sweeps >= cap), and the count of free-energy rises."""
    sweeps = [int(v) for v in (rec.get("sweeps") or [])]
    cap = (rec.get("settings") or {}).get("max_sweeps")
    inf = (rec.get("measures") or {}).get("inference") or {}
    ends = list(inf.get("end_reason") or [])
    at_cap = []
    for j, sw in enumerate(sweeps):
        e = ends[j] if j < len(ends) else None
        if e is not None:
            at_cap.append(e == "max_sweeps")
        else:
            at_cap.append(cap is not None and sw >= int(cap))
    rises = rec.get("free_energy_rises")
    n_rises = len(rises) if rises is not None else inf.get("n_rises")
    return dict(
        sweeps=sweeps, max_sweeps=cap, end_reason=ends, at_cap=at_cap, n_rises=n_rises
    )


def fig_name(stem: str, suffix: str = "") -> str:
    """The PNG file name of a figure: stem, then the optional suffix."""
    return f"{stem}{suffix}.png"


def with_label(title: str, label: str = "") -> str:
    """A figure title with the run set's label (if any) at its end."""
    return f"{title}  [{label}]" if label else title


# ---------------------------------------------------------------- the runs


@dataclass(frozen=True)
class RunKey:
    phantom: str
    source: str
    noise: float
    variant: str  # "", "_coarse", "_anchor" or "_shiftB"

    def dirname(self, label: str) -> str:
        return f"{self.source}_{self.noise:g}_{label}{self.variant}"

    @property
    def name(self) -> str:
        v = self.variant.lstrip("_") or "top"
        return f"{self.phantom} {self.source} {self.noise:g} {v}"

    @property
    def slug(self) -> str:
        return f"{self.phantom}_{self.source}_{self.noise:g}{self.variant}"


@dataclass
class Run:
    """The parts of one run directory that the views read. The field is light: q_stop,
    tau_mean and direction_mean, on the rebuilt grid."""

    key: RunKey
    path: Path
    rec: dict[str, Any]
    grid: FG.UniformGrid
    q_stop: Array
    tau_mean: Array
    direction_mean: Array
    cols: T.Columns
    errs: dict[str, Array]
    extra: dict[str, Any] | None

    @property
    def spacing(self) -> float:
        return float(self.grid.spacing_mm)

    def light_field(self) -> T.Field:
        """A field with one moving candidate per node (tau_mean, direction_mean, mass
        1 − q_stop). Its tau_mean, direction_mean and q_stop equal the saved ones, so
        the integrator's rules (masked sample, unresolved cells, reversed edges) give
        the same result on it. Its resultant length is 1: the sinks of low resultant
        need the full candidates (`low_resultant_nodes`)."""
        n = len(self.q_stop)
        return T.Field(
            grid=self.grid,
            tau=self.tau_mean[:, None].copy(),
            direction=self.direction_mean[:, None, :].copy(),
            q=(1.0 - self.q_stop)[:, None].copy(),
            q_stop=self.q_stop,
            n_kept=np.ones(n, np.int64),
        )


def find_runs(runs: Path, label: str, phantoms: Sequence[str]) -> dict[RunKey, Path]:
    """Every run directory with a run.json under runs/<phantom>/ for this grid label."""
    pat = re.compile(
        rf"^(oracle|image)_([0-9.eE+-]+)_{re.escape(label)}"
        r"(_coarse|_anchor|_shiftB)?$"
    )
    out: dict[RunKey, Path] = {}
    for ph in phantoms:
        d = runs / ph
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            m = pat.match(sub.name)
            if m and (sub / "run.json").is_file():
                key = RunKey(ph, m.group(1), float(m.group(2)), m.group(3) or "")
                out[key] = sub
    return out


def load_run(key: RunKey, path: Path) -> Run:
    rec = json.loads((path / "run.json").read_text())
    z = np.load(path / "field.npz", allow_pickle=False)
    kind = str(z["grid_kind"])
    if kind != "uniform":
        raise NotImplementedError(
            f"{path}: grid kind {kind!r}; only uniform grids here"
        )
    shape = tuple(int(s) for s in z["grid_shape_vox"])
    grid = FG.UniformGrid((shape[0], shape[1], shape[2]), float(z["grid_spacing_mm"]))
    if tuple(int(d) for d in z["grid_dims"]) != grid.dims:
        raise ValueError(f"{path}: the grid dims do not match the rebuilt grid")
    cz = np.load(path / "columns.npz", allow_pickle=False)
    cols = T.Columns(**{f.name: cz[f.name] for f in dataclasses.fields(T.Columns)})
    ez = np.load(path / "column_errors.npz", allow_pickle=False)
    extra = None
    for mode in ("coarse", "anchor", "shiftB"):
        f = path / f"{mode}.json"
        if f.is_file():
            extra = json.loads(f.read_text())
    return Run(
        key,
        path,
        rec,
        grid,
        np.asarray(z["q_stop"], float),
        np.asarray(z["tau_mean"], float),
        np.asarray(z["direction_mean"], float),
        cols,
        {k: ez[k] for k in ez.files},
        extra,
    )


def low_resultant_nodes(
    path: Path, q_stop: Array, settings: T.SolverSettings
) -> T.IntArray:
    """The moving nodes whose resultant length is below SINK_RESULTANT
    (`ff_checks.field_sinks`), from the full candidates of field.npz."""
    z = np.load(path / "field.npz", allow_pickle=False)
    q = np.asarray(z["q"], float)
    R = np.einsum("nk,nkc->nc", q, np.asarray(z["direction"], float))
    qm = q.sum(1)
    del q
    res = np.sqrt((R * R).sum(1)) / np.where(qm > 0, qm, 1.0)
    mov = q_stop < settings.stop_level
    return np.flatnonzero(mov & (res < FC.SINK_RESULTANT)).astype(np.int64)


# ---------------------------------------------------------------- geometry per phantom


@dataclass
class NodeGeo:
    """The truth at the nodes of one grid of one phantom (shared by its runs)."""

    local: Array
    label: Any
    bank: Any
    sdf: Array  # stop_sdf; below −(R6_MAX_MM) only a bound (module docstring)


class Geo:
    """The geometry and truth of one phantom, with a cache of NodeGeo per grid."""

    def __init__(self, name: str, truth_dir: Path, shape: Sequence[int]) -> None:
        self.name = name
        geo, ts = FTR.geometry_of(name, (int(shape[0]), int(shape[1]), int(shape[2])))
        self.geo = geo
        self.tau_scale = float(ts)
        self.truth = T.Truth.from_npz(truth_dir / name / "truth.npz")
        self.true_stop_local = geo.to_local(self.truth.stops)
        self.flat = self.true_stop_local[:, 1] >= 0.0
        self.kind = "sharp" if "sharp" in name else "smooth"
        self.window = WINDOW_MM[self.kind]
        self._nodes: dict[float, NodeGeo] = {}

    def nodes(self, grid: FG.UniformGrid) -> NodeGeo:
        key = round(grid.spacing_mm, 9)
        if key not in self._nodes:
            geo = self.geo
            Y = geo.to_local(grid.nodes)
            _, _, r, _, _ = geo._polar(Y)
            d0 = r - geo.g(Y[:, 2])
            near = d0 <= R6_MAX_MM + (geo.p["g_wide"] - geo.p["g_thin"])
            sdf = (
                -d0.copy()
            )  # a bound: the true distance is in [d0 − (g_wide − g_thin), d0]
            sdf[near] = geo.stop_sdf(Y[near])
            self._nodes[key] = NodeGeo(
                Y, np.asarray(geo.label(Y)), np.asarray(geo.bank(Y)), sdf
            )
        return self._nodes[key]


# ---------------------------------------------------------------- per run


def _end_counts(end: Any) -> dict[str, int]:
    return {e.name: int((np.asarray(end) == int(e)).sum()) for e in T.EndReason}


def _local_paths(geo: Any, cols: T.Columns) -> Array:
    P = np.asarray(cols.paths, float)
    m = P.shape[1]
    valid = np.arange(m)[None, :] < np.asarray(cols.counts)[:, None]
    out = np.full(P.shape, np.nan)
    out[valid] = geo.to_local(P[valid])
    return out


def _padded_truth(tr: T.Truth, geo: Any) -> Array:
    P = np.asarray(tr.paths, float)
    m = P.shape[1]
    valid = np.arange(m)[None, :] < np.asarray(tr.counts)[:, None]
    out = np.full(P.shape, np.nan)
    out[valid] = geo.to_local(P[valid])
    return out


def stop_ramps(run: Run, sel: T.IntArray) -> tuple[Array, Array]:
    """(width (n,), peak ahead (n,)) for the columns `sel` (`ramp_width` on the line
    through each stop along the last step; peak: the largest q_stop within one grid
    spacing after the stop)."""
    cols = run.cols
    s = np.arange(-RAMP_REACH_MM, RAMP_REACH_MM + 0.5 * RAMP_STEP_MM, RAMP_STEP_MM)
    width = np.full(len(sel), np.nan)
    peak = np.full(len(sel), np.nan)
    ok = cols.counts[sel] >= 2
    idx = sel[ok]
    if len(idx) == 0:
        return width, peak
    c = cols.counts[idx]
    last = cols.paths[idx, c - 1]
    prev = cols.paths[idx, c - 2]
    u = last - prev
    nu = np.linalg.norm(u, axis=1)
    u = u / np.where(nu > 0, nu, 1.0)[:, None]
    q = np.empty((len(idx), len(s)))
    for c0 in range(0, len(idx), RAMP_CHUNK):  # chunks: the (N, 8) arrays stay small
        c1 = c0 + RAMP_CHUNK
        X = last[c0:c1, None, :] + (s[None, :, None] / H) * u[c0:c1, None, :]
        q[c0:c1] = sample_nodes(run.grid, run.q_stop, X.reshape(-1, 3)).reshape(
            -1, len(s)
        )
    ahead = (s > 0) & (s <= run.spacing)
    w = np.array([ramp_width(s, q[i], 0.0) for i in range(len(idx))])
    pk = np.nanmax(np.where(ahead[None, :], q, np.nan), axis=1)
    width[np.flatnonzero(ok)] = w
    peak[np.flatnonzero(ok)] = pk
    return width, peak


def recount_topology(run: Run, G: Geo, settings: T.SolverSettings) -> dict[str, Any]:
    """The topology checks of ff_checks on the saved field and columns, with places.
    Returns {name: places (k, 3) voxel space} and the counts."""
    light = run.light_field()
    cols = run.cols
    geo = G.geo
    sc = FC.solved_stop_crossings(light, cols, settings)
    tc = FC.true_stop_crossings(geo, cols)
    mt = FC.column_meetings(cols, geo)
    fs = FC.field_sinks(light, settings)
    low = low_resultant_nodes(run.path, run.q_stop, settings)
    un = np.flatnonzero(FI.unresolved_cells(light, settings))
    centres = run.grid.cell_origin[un] + 0.5 * (run.grid.cell_size_mm[un] / H)[:, None]
    sink_cols = np.flatnonzero(cols.end == int(T.EndReason.SINK))
    places = dict(
        solved_stop_crossings=sc["places"],
        true_stop_crossings=tc["places"],
        overshoots=tc["overshoot_places"],
        overlap=mt["overlap"]["places"],
        focal_points=mt["focal"]["places"],
        sink_nodes=run.grid.nodes[low],
        reversed_edges=fs["reversed_edges"]["places"],
        sink_columns=cols.stops[sink_cols],
        unresolved_cells=centres,
    )
    return places


def per_run(run: Run, G: Geo, settings: T.SolverSettings) -> dict[str, Any]:
    tr, geo = G.truth, G.geo
    cols, errs = run.cols, run.errs
    m = run.rec["measures"]
    kind, bank, flat = tr.stop_kind, tr.bank, G.flat
    contact = kind == int(T.StopKind.CONTACT)
    end = np.asarray(cols.end)
    stopped = end == int(T.EndReason.STOPPED)
    sabs, ssg = errs["stop_abs_mm"], errs["stop_signed_mm"]
    Ys = geo.to_local(cols.stops)
    own = own_side(Ys, bank)
    out: dict[str, Any] = dict(
        run=run.key.name,
        dir=str(run.path),
        spacing_mm=run.spacing,
        origin=run.rec.get("origin"),
    )
    # ---- Q1
    q1: dict[str, Any] = dict(by_bank={})
    for b, bn in ((0, "A"), (1, "B")):
        sel = contact & (bank == b)
        fsel = flat & (bank == b) & stopped
        q1["by_bank"][bn] = dict(
            n_contact=int(sel.sum()),
            stop_abs_mm=stats(sabs[sel]),
            stop_signed_mm=stats(ssg[sel]),
            end_reasons=_end_counts(end[sel]),
            own_side_contact=_frac(own[sel & stopped]),
            own_side_flat=_frac(own[fsel]),
        )
    Pl = _local_paths(geo, cols)
    fl = np.flatnonzero(flat)
    cr, first = contact_crossings(Pl[fl], cols.counts[fl], bank[fl])
    ci = fl[cr]
    far_pts = cols.paths[ci, first[cr]] if len(ci) else np.zeros((0, 3))
    lab = np.asarray(geo.label(geo.to_local(far_pts))) if len(ci) else np.zeros(0)
    Pt = _padded_truth(tr, geo)
    crt, _ = contact_crossings(Pt[fl], tr.counts[fl], bank[fl])
    q1["contact_crossings"] = dict(
        tol_mm=CROSS_TOL_MM,
        n_flat_columns=int(len(fl)),
        count=int(cr.sum()),
        count_A=int((bank[ci] == 0).sum()),
        count_B=int((bank[ci] == 1).sum()),
        into_tissue=int((lab != T.MEDIUM).sum()),
        count_on_true_paths=int(crt.sum()),
    )
    topo = m.get("topology", {})
    q1["overlap"] = topo.get("overlap")
    q1["focal_points"] = topo.get("focal_points")
    q1["ramp"] = {}
    for kname, kk in (("CONTACT", T.StopKind.CONTACT), ("PIA", T.StopKind.PIA)):
        sel = np.flatnonzero((kind == int(kk)) & flat & stopped)
        w, pk = stop_ramps(run, sel)
        q1["ramp"][kname] = dict(
            width_mm=stats(w),
            n_without_width=int(np.isnan(w).sum()),
            peak_q_stop_ahead=stats(pk),
        )
    out["q1"] = q1
    # ---- Q3
    zt = G.true_stop_local[:, 2]
    zlo = np.floor(np.nanmin(zt) / Z_BIN_MM) * Z_BIN_MM
    zhi = np.ceil(np.nanmax(zt) / Z_BIN_MM) * Z_BIN_MM
    edges = np.arange(zlo, zhi + 0.5 * Z_BIN_MM, Z_BIN_MM)
    bins = bin_stats(sabs, zt, edges)
    nst = bin_stats(np.where(stopped, np.nan, 1.0), zt, edges)
    for b, ns in zip(bins, nst, strict=True):
        b["n_not_stopped"] = ns["n"]
    w = G.window
    regions = dict(open=zt < -w, window=np.abs(zt) <= w, fused=zt > w)
    q3: dict[str, Any] = dict(window_mm=w, z_bins=bins, regions={})
    for rn, rs in regions.items():
        q3["regions"][rn] = dict(
            stop_abs_mm=stats(sabs[rs]),
            n_columns=int(rs.sum()),
            n_not_stopped=int((rs & ~stopped).sum()),
        )
    places = recount_topology(run, G, settings)
    ng = G.nodes(run.grid)
    mov_ctx = (run.q_stop < MOVING) & (ng.label == T.CORTEX)
    zn = ng.local[:, 2]
    q3["moving_cortex_nodes"] = dict(
        window=int((mov_ctx & (np.abs(zn) <= w)).sum()),
        elsewhere=int((mov_ctx & (np.abs(zn) > w)).sum()),
    )
    q3["places"] = {}
    recount = {}
    for name, P in places.items():
        P = np.asarray(P, float).reshape(-1, 3)
        z = geo.to_local(P)[:, 2] if len(P) else np.zeros(0)
        q3["places"][name] = dict(
            window=int((np.abs(z) <= w).sum()), elsewhere=int((np.abs(z) > w).sum())
        )
        recount[name] = int(len(P))
    z = geo.to_local(far_pts)[:, 2] if len(far_pts) else np.zeros(0)
    q3["places"]["contact_crossings"] = dict(
        window=int((np.abs(z) <= w).sum()), elsewhere=int((np.abs(z) > w).sum())
    )
    mismatch = {
        k: (recount[k], v)
        for k, v in topo.items()
        if k in recount and k != "sink_columns" and recount[k] != v
    }
    q3["recount_vs_run_json"] = dict(matches=not mismatch, mismatch=mismatch)
    out["q3"] = q3
    # ---- Q4 numbers
    q4: dict[str, Any] = dict(sheet=[], two_g_mm=2 * CK.G_THIN, spacing_mm=run.spacing)
    s = np.arange(-SHEET_HALF_MM, SHEET_HALF_MM + 0.5 * SHEET_STEP_MM, SHEET_STEP_MM)
    for y in SHEET_YS_MM:
        for zz in SHEET_ZS_MM:
            Y = np.stack([s, np.full_like(s, y), np.full_like(s, zz)], 1)
            X = geo.to_vox(Y)
            if not FV.in_box(geo, X).all():
                q4["sheet"].append(dict(y=y, z=zz, in_box=False))
                continue
            q = sample_nodes(run.grid, run.q_stop, X)
            q4["sheet"].append(
                dict(
                    y=y,
                    z=zz,
                    in_box=True,
                    length_mm=sheet_length(s, q, MOVING),
                    peak_q_stop=float(np.nanmax(q)),
                    true_2g_mm=float(2 * geo.g(np.array([zz]))[0]),
                )
            )
    lens = [d["length_mm"] for d in q4["sheet"] if d.get("in_box")]
    q4["sheet_length_mm"] = stats(lens)
    q4["sheet_resolved"] = int(sum(v > 0 for v in lens))
    openz = zn < OPEN_MAX_Z_MM
    med = openz & (ng.label == T.MEDIUM)
    ctx = openz & (ng.label == T.CORTEX) & (ng.sdf < -run.spacing)
    q4["open_slot"] = dict(
        n_medium_nodes=int(med.sum()),
        medium_stopped=_frac(run.q_stop[med] > MOVING),
        n_cortex_nodes=int(ctx.sum()),
        cortex_stopped=_frac(run.q_stop[ctx] > MOVING),
    )
    out["q4"] = q4
    # ---- Q5
    sp = (m.get("speeds") or {}).get("banks")
    q5: dict[str, Any] = dict(speed_banks=sp, by_bank={})
    dev = np.asarray(errs["level_dev_mm"], float)
    for b, bn in ((0, "A"), (1, "B")):
        sel = contact & (bank == b) & stopped
        q5["by_bank"][bn] = dict(
            tau_arrival_contact=stats(cols.tau_arrival[sel]),
            level_abs_dev_mm=stats(np.abs(dev[bank == b])),
            level_signed_median_mm=[
                _f(np.nanmedian(dev[bank == b][:, j]))
                if np.isfinite(dev[bank == b][:, j]).any()
                else None
                for j in range(dev.shape[1])
            ],
        )
    q5["level_phi"] = [float(v) for v in errs["level_phi"]]
    out["q5"] = q5
    # ---- R5, R7, R11 and the inference
    out["topology"] = topo
    out["direction"] = (m.get("direction") or {}).get("cortex")
    out["inference"] = inference_summary(run.rec)
    out["sweeps"] = out["inference"]["sweeps"]
    out["_free_energy"] = run.rec.get("free_energy") or []
    out["stops_end_reasons"] = (m.get("stops") or {}).get("end_reasons")
    return out


def _frac(b: Any) -> dict[str, Any]:
    b = np.asarray(b, bool)
    return dict(n=int(b.size), fraction=float(b.mean()) if b.size else None)


def _f(x: Any) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------- pairs


def q6_origin(top: Run, anchor: Run | None, coarse: Run | None) -> dict[str, Any]:
    out: dict[str, Any] = dict(top=top.key.name, sweeps_top=top.rec.get("sweeps"))
    out["coarse"] = (
        "not run"
        if coarse is None
        else dict(
            sweeps_fine=coarse.rec.get("sweeps"),
            sweeps_coarse_solve=(coarse.extra or {}).get("sweeps"),
            end_reason=(coarse.rec["measures"].get("inference") or {}).get(
                "end_reason"
            ),
        )
    )
    if anchor is None:
        out["anchor"] = "not run"
        return out
    mov = (top.q_stop < MOVING) & (anchor.q_stop < MOVING)
    a = np.where(mov, top.tau_mean, np.nan)
    b = np.where(mov, anchor.tau_mean, np.nan)
    c, res, n = fit_constant(a, b)
    cols_t, cols_a = top.cols, anchor.cols
    levels = np.asarray(cols_t.levels, float)
    dist = np.full((len(cols_t.counts), len(levels)), np.nan)
    for i in range(len(cols_t.counts)):
        nt, na = int(cols_t.counts[i]), int(cols_a.counts[i])
        Xt = first_crossings(cols_t.tau_path[i, :nt], cols_t.paths[i, :nt], levels)
        Xa = first_crossings(cols_a.tau_path[i, :na] - c, cols_a.paths[i, :na], levels)
        dist[i] = np.linalg.norm(Xt - Xa, axis=1) * H
    out["anchor"] = dict(
        constant=c,
        n_nodes=n,
        residual_abs=stats(np.abs(res)),
        tau_top_range=[_f(np.nanmin(a)), _f(np.nanmax(a))],
        level_distance_mm=stats(dist),
        level_found_both=int(np.isfinite(dist).sum()),
        level_pairs=int(dist.size),
        sweeps_anchor=anchor.rec.get("sweeps"),
        end_reason=(anchor.rec["measures"].get("inference") or {}).get("end_reason"),
        level_dev_abs_top=stats(np.abs(top.errs["level_dev_mm"])),
        level_dev_abs_anchor=stats(np.abs(anchor.errs["level_dev_mm"])),
        anchor_json=anchor.extra,
    )
    out["_arrays"] = dict(a=a, b=b, c=c)
    return out


def r6_independence(top: Run, shift: Run, G: Geo) -> dict[str, Any]:
    if top.grid.dims != shift.grid.dims:
        raise ValueError("R6: the two runs have different grids")
    ng = G.nodes(top.grid)
    mov = (top.q_stop < MOVING) & (shift.q_stop < MOVING)
    dtau = np.abs(shift.tau_mean - top.tau_mean)
    cosang = np.clip(np.sum(top.direction_mean * shift.direction_mean, axis=1), -1, 1)
    ang = np.degrees(np.arccos(cosang))
    dist = -ng.sdf
    edges = distance_edges(top.spacing, R6_MAX_MM)
    out: dict[str, Any] = dict(
        top=top.key.name,
        shift=shift.key.name,
        sweeps_top=top.rec.get("sweeps"),
        sweeps_shift=shift.rec.get("sweeps"),
        end_reason_top=(top.rec["measures"].get("inference") or {}).get("end_reason"),
        end_reason_shift=(shift.rec["measures"].get("inference") or {}).get(
            "end_reason"
        ),
        shift_json=shift.extra,
        nodes={},
    )
    for b, bn in ((0, "A"), (1, "B")):
        sel = mov & (ng.bank == b)
        out["nodes"][bn] = dict(
            n=int(sel.sum()),
            dtau_by_distance=bin_stats(np.where(sel, dtau, np.nan), dist, edges),
            angle_deg_by_distance=bin_stats(np.where(sel, ang, np.nan), dist, edges),
            dtau_all_within=stats(dtau[sel & (dist >= 0) & (dist < edges[-1])]),
        )
    ct, cs = top.cols, shift.cols
    tr = G.truth
    pdist = np.full(tr.n, np.nan)
    for i in range(tr.n):
        A = ct.paths[i, : ct.counts[i]] * H
        B = cs.paths[i, : cs.counts[i]] * H
        pdist[i] = path_distance(A, B)
    sdist = np.linalg.norm(ct.stops - cs.stops, axis=1) * H
    same_end = ct.end == cs.end
    out["columns"] = {}
    for b, bn in ((0, "A"), (1, "B")):
        for k in T.StopKind:
            sel = (tr.bank == b) & (tr.stop_kind == int(k))
            if not sel.any():
                continue
            out["columns"][f"{bn} {k.name}"] = dict(
                n=int(sel.sum()),
                path_distance_mm=stats(pdist[sel]),
                stop_distance_mm=stats(sdist[sel]),
                end_reason_changed=int((sel & ~same_end).sum()),
            )
    return out


# ---------------------------------------------------------------- figures


def _save(fig: Any, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    return FR._save(fig, path)


@dataclass
class SectionTruth:
    hs: Array
    vs: Array
    X: Array  # (nv * nh, 3) voxel positions
    inbox: Any  # (nv, nh)
    label: Any  # (nv, nh)
    surface: Array  # (K, 2)


def section_truth(G: Geo, z: float) -> SectionTruth:
    geo = G.geo
    step = FV.STEP_MM
    hs = np.arange(SECTION_X_MM[0], SECTION_X_MM[1] + 0.5 * step, step)
    vs = np.arange(SECTION_Y_MM[0], SECTION_Y_MM[1] + 0.5 * step, step)
    HH, VV = np.meshgrid(hs, vs)
    plane = FV.Plane(0, 1, 2, z, "")
    sec = FV.Section(
        plane,
        z,
        hs,
        vs,
        np.zeros(HH.shape, bool),
        np.zeros(HH.shape),
        np.zeros(HH.shape, int),
        np.zeros((0, 2)),
    )
    Y = sec.local(HH, VV)
    X = geo.to_vox(Y)
    inb = FV.in_box(geo, X)
    sec.inbox = inb.reshape(HH.shape)
    lab = np.asarray(geo.label(Y)).reshape(HH.shape).astype(float)
    lab[~sec.inbox] = np.nan
    surf = FV.stop_surface(geo, sec, Y, inb)
    return SectionTruth(hs, vs, X, sec.inbox, lab, surf)


def figure_sections(
    name: str,
    G: Geo,
    runs: Mapping[RunKey, tuple[FG.UniformGrid, Array]],
    path: Path,
    label: str = "",
) -> int:
    secs = [section_truth(G, z) for z in SECTION_Z_MM]
    ncol = 2 * len(secs)
    fig, axes = plt.subplots(
        len(NOISES),
        ncol,
        figsize=(2.3 * ncol, 3.3 * len(NOISES) + 0.4),
        squeeze=False,
        layout="constrained",
    )
    im = None
    for r, noise in enumerate(NOISES):
        for p, sec in enumerate(secs):
            for s, src in enumerate(SOURCES):
                ax = axes[r, 2 * p + s]
                run = runs.get(RunKey(name, src, noise, ""))
                ext = (sec.hs[0], sec.hs[-1], sec.vs[0], sec.vs[-1])
                title = (
                    f"z = {SECTION_Z_MM[p]:+g} mm ({SECTION_NOTES[p]})\n"
                    f"{src}, noise {noise:g}"
                )
                ax.set_title(title, fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_xlim(ext[0], ext[1])
                ax.set_ylim(ext[2], ext[3])
                ax.set_aspect("equal")
                if r == len(NOISES) - 1:
                    ax.set_xlabel("local x (mm)", fontsize=7)
                if p == 0 and s == 0:
                    ax.set_ylabel("local y (mm)", fontsize=7)
                if run is None:
                    ax.text(
                        0.5,
                        0.5,
                        "not run",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=9,
                        color="0.4",
                    )
                    continue
                Q = sample_nodes(run[0], run[1], sec.X).reshape(sec.inbox.shape)
                Q[~sec.inbox] = np.nan
                im = ax.imshow(
                    np.ma.masked_invalid(Q),
                    origin="lower",
                    extent=ext,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                if np.nanmin(Q) < MOVING < np.nanmax(Q):
                    ax.contour(
                        sec.hs,
                        sec.vs,
                        np.ma.masked_invalid(Q),
                        levels=[MOVING],
                        colors="#ff3030",
                        linewidths=0.8,
                    )
                ax.contour(
                    sec.hs,
                    sec.vs,
                    np.ma.masked_invalid(sec.label),
                    levels=[0.5, 1.5],
                    colors="white",
                    linewidths=0.5,
                    linestyles=":",
                )
                if sec.surface.size:
                    ax.scatter(
                        sec.surface[:, 0],
                        sec.surface[:, 1],
                        s=0.6,
                        marker="s",
                        c=FV.SURFACE_COLOUR,
                        linewidths=0,
                    )
    handles = [
        Line2D([], [], c="#ff3030", lw=1.2, label="solved stop (q_stop = 0.5)"),
        Line2D(
            [], [], c=FV.SURFACE_COLOUR, marker="s", ls="", label="true stop surface"
        ),
        Line2D(
            [],
            [],
            c="0.5",
            ls=":",
            lw=1.2,
            label="true label borders (WM | cortex | outside medium)",
        ),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=3, fontsize=7, frameon=False
    )
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
        cb.set_label("q_stop", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle(
        with_label(f"{name}: q_stop in sections normal to local z", label), fontsize=9
    )
    return _save(fig, path)


def figure_stop_vs_z(
    results: Mapping[RunKey, dict[str, Any]],
    phantoms: Sequence[str],
    geos: Mapping[str, Geo],
    path: Path,
    label: str = "",
) -> int:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), squeeze=False, sharey=True)
    for ax, name in zip(axes.ravel(), CK.NAMES, strict=True):
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(H, c="0.5", lw=0.6, ls=":")
        if name in geos:
            w = geos[name].window
            ax.axvspan(-w, w, color="0.9", zorder=0)
        n_lines = 0
        for (src, noise), (col, ls) in RUN_COLOURS.items():
            res = results.get(RunKey(name, src, noise, ""))
            if res is None:
                continue
            b = res["q3"]["z_bins"]
            zc = np.array([0.5 * (d["lo"] + d["hi"]) for d in b])
            med = np.array(
                [np.nan if d["median"] is None else d["median"] for d in b], float
            )
            ok = np.isfinite(med)  # the roots lie on a 0.32 mm lattice: empty bins
            ax.plot(
                zc[ok],
                med[ok],
                ls,
                c=col,
                lw=0.8,
                marker=".",
                ms=2.5,
                label=f"{src} noise {noise:g}",
            )
            n_lines += 1
        if n_lines == 0:
            ax.text(
                0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color="0.4"
            )
        else:
            ax.legend(fontsize=6, frameon=False)
        ax.set_xlabel("local z of the true stop (mm)", fontsize=7)
        ax.set_ylabel("median |stop error| (mm)", fontsize=7)
    fig.suptitle(
        with_label(
            "Stop error against the local z of the true stop (bins of 0.16 mm; grey: "
            "the window round the gap's end; dotted: one voxel)",
            label,
        ),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def figure_r6(r6: dict[str, Any], path: Path, label: str = "") -> int:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for bn, col in (("A", "#1f77b4"), ("B", "#d55e00")):
        d = r6["nodes"][bn]
        for ax, key, lab in (
            (axes[0], "dtau_by_distance", "|Δ tau_mean|"),
            (axes[1], "angle_deg_by_distance", "angle (deg)"),
        ):
            b = d[key]
            x = np.array([0.5 * (e["lo"] + e["hi"]) for e in b])
            ax.plot(
                x,
                [e["median"] for e in b],
                "o-",
                c=col,
                ms=3,
                lw=1,
                label=f"bank {bn}, median",
            )
            ax.plot(
                x,
                [e["p90"] for e in b],
                "s--",
                c=col,
                ms=3,
                lw=0.8,
                label=f"bank {bn}, p90",
            )
            ax.set_ylabel(lab, fontsize=8)
    for ax in axes:
        ax.set_xlabel("distance from the true stop surface, −stop_sdf (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle(
        with_label(
            f"R6: {r6['top']} against {r6['shift']} (bank B's image changed), "
            "nodes that move in both",
            label,
        ),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def figure_q6(q6: dict[str, Any], path: Path, label: str = "") -> int:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    arr = q6.get("_arrays")
    if arr is None:
        ax.text(0.5, 0.5, "anchor run: not run", transform=ax.transAxes, ha="center")
    else:
        a, b, c = arr["a"], arr["b"], arr["c"]
        ok = np.flatnonzero(np.isfinite(a) & np.isfinite(b))
        rng = np.random.default_rng(0)
        if len(ok) > 20000:
            ok = rng.choice(ok, 20000, replace=False)
        ax.plot(a[ok], b[ok], ".", ms=1, c="#1f77b4", alpha=0.4)
        lo, hi = np.nanmin(a[ok]), np.nanmax(a[ok])
        ax.plot(
            [lo, hi],
            [lo + c, hi + c],
            "-",
            c="#d55e00",
            lw=1,
            label=f"τ_anchor = τ_top + c, c = {c:.3f}",
        )
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel("tau_mean, top origin", fontsize=8)
        ax.set_ylabel("tau_mean, anchor origin", fontsize=8)
    ax.tick_params(labelsize=7)
    ax = axes[1]
    rows = [("top", q6.get("sweeps_top"))]
    an = q6.get("anchor")
    rows.append(("anchor", an.get("sweeps_anchor") if isinstance(an, dict) else None))
    co = q6.get("coarse")
    rows.append(
        ("coarse: fine", co.get("sweeps_fine") if isinstance(co, dict) else None)
    )
    rows.append(
        (
            "coarse: 0.64 solve",
            co.get("sweeps_coarse_solve") if isinstance(co, dict) else None,
        )
    )
    widths = [f"width {w}" for w in T.DEFAULTS.widths]
    xb = np.arange(len(rows))
    for j, wn in enumerate(widths):
        vals = [(r[1][j] if r[1] is not None and j < len(r[1]) else 0) for r in rows]
        ax.bar(xb + (j - 1) * 0.25, vals, width=0.25, label=wn)
    for k, r in enumerate(rows):
        if r[1] is None:
            ax.text(
                xb[k], 1, "no data", ha="center", fontsize=7, rotation=90, color="0.4"
            )
    ax.set_xticks(xb)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylabel("sweeps", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    fig.suptitle(with_label(f"Q6: the origin of τ ({q6['top']})", label), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def figure_free_energy(
    traces: Mapping[str, Sequence[Sequence[float]]], path: Path, label: str = ""
) -> int:
    """One panel per width: F − min F of each run's trace at that width, log scale.
    The sweep that holds the minimum (F − min F = 0) is not drawn."""
    nw = max([len(t) for t in traces.values()] + [len(T.DEFAULTS.widths)])
    fig, axes = plt.subplots(1, nw, figsize=(4.2 * nw, 4.2), squeeze=False)
    cmap = plt.get_cmap("tab20")
    names = list(traces)
    for j in range(nw):
        ax = axes[0, j]
        w = T.DEFAULTS.widths[j] if j < len(T.DEFAULTS.widths) else j
        ax.set_title(f"width {w}", fontsize=8)
        for i, name in enumerate(names):
            tr = traces[name]
            if j >= len(tr) or len(tr[j]) == 0:
                continue
            F = np.asarray(tr[j], float)
            d = F - np.nanmin(F)
            d = np.where(d > 0, d, np.nan)
            ax.plot(
                np.arange(1, len(F) + 1), d, "-", c=cmap(i % 20), lw=0.8, label=name
            )
        ax.set_yscale("log")
        ax.set_xlabel("sweep", fontsize=7)
        ax.set_ylabel("F − min F (of the run at this width)", fontsize=7)
        ax.tick_params(labelsize=6)
    handles: dict[str, Any] = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
    if handles:
        fig.legend(
            list(handles.values()),
            list(handles),
            loc="lower center",
            ncol=min(4, len(handles)),
            fontsize=6,
            frameon=False,
        )
    else:
        axes[0, 0].text(0.5, 0.5, "no run", transform=axes[0, 0].transAxes, ha="center")
    nrow = (len(handles) + 3) // 4
    fig.suptitle(with_label("Free energy per sweep, per width", label), fontsize=9)
    fig.tight_layout(rect=(0, min(0.45, 0.04 * nrow + 0.02), 1, 0.94))
    return _save(fig, path)


# ---------------------------------------------------------------- tables


def _n(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    return "nan" if not np.isfinite(x) else f"{x:.{nd}g}"


def _st(d: Any, nd: int = 3) -> str:
    if not isinstance(d, Mapping) or not d.get("n"):
        return "–"
    return (
        f"{_n(d['median'], nd)} / {_n(d['p90'], nd)} / {_n(d['max'], nd)} (n {d['n']})"
    )


def summary_rows(
    keys: Sequence[RunKey], results: Mapping[RunKey, dict[str, Any]]
) -> list[list[str]]:
    rows = []
    for k in keys:
        r = results.get(k)
        base = [k.phantom, k.source, f"{k.noise:g}", k.variant.lstrip("_") or "top"]
        if r is None:
            rows.append(base + ["not run"] + [""] * 12)
            continue
        q1 = r["q1"]
        bA, bB = q1["by_bank"]["A"], q1["by_bank"]["B"]
        topo = r["topology"] or {}
        sp = r["q5"]["speed_banks"] or {}
        dirn = r["direction"] or {}
        rows.append(
            base
            + [
                _n(bA["stop_abs_mm"]["median"])
                + " / "
                + _n(bB["stop_abs_mm"]["median"]),
                _n(bA["own_side_flat"]["fraction"])
                + " / "
                + _n(bB["own_side_flat"]["fraction"]),
                str(q1["contact_crossings"]["count"]),
                f"{topo.get('overlap', '–')} / {topo.get('focal_points', '–')}",
                f"{topo.get('sink_nodes', '–')} / {topo.get('sink_columns', '–')}",
                str(topo.get("unresolved_cells", "–")),
                str((r["stops_end_reasons"] or {}).get("UNRESOLVED_CONTACT", "–")),
                _n(sp.get("solved_ratio_A_over_B")),
                _n((dirn.get("direction_vs_grad_tau_deg") or {}).get("median")),
                _n(q1["ramp"]["CONTACT"]["width_mm"]["median"]),
                "/".join(str(s) for s in r["inference"]["sweeps"])
                + f" (cap {r['inference']['max_sweeps']})",
                "/".join(
                    END_CODES.get(str(e), str(e)) for e in r["inference"]["end_reason"]
                ),
                str(r["inference"]["n_rises"]),
            ]
        )
    return rows


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def markdown(
    doc: dict[str, Any],
    keys: Sequence[RunKey],
    results: Mapping[RunKey, dict[str, Any]],
) -> str:
    L: list[str] = [f"# Kissing surfaces: views and numbers ({doc['grid_label']})", ""]
    if doc.get("label"):
        L += [f"**{doc['label']}**", ""]
    L.append(
        "Distributions: median / p90 / max (n). Lengths in mm. Bank A / bank B. "
        "Diagnostics, not pass/fail rules."
    )
    L += ["", "## Summary", ""]
    head = [
        "phantom",
        "source",
        "noise",
        "variant",
        "CONTACT stop error median A / B",
        "own side (flat) A / B",
        "contact crossings",
        "overlap / focal",
        "sink nodes / columns",
        "unresolved cells",
        "columns ended UNRESOLVED",
        "speed ratio A/B (tA/tB 2.08)",
        "dir vs ∇τ median (deg)",
        "stop ramp 0.2→0.8 median",
        "sweeps per width",
        "end per width (F: tol_free_energy, q: tol_q, CAP: max_sweeps)",
        "free-energy rises",
    ]
    L.append(_table(head, summary_rows(keys, results)))
    run_keys = [k for k in keys if k in results]
    L += ["", "## Q1 fused contact (true stop kind CONTACT)", ""]
    rows = []
    for k in run_keys:
        q1 = results[k]["q1"]
        for bn in ("A", "B"):
            b = q1["by_bank"][bn]
            e = b["end_reasons"]
            rows.append(
                [
                    k.name,
                    bn,
                    _st(b["stop_abs_mm"]),
                    _st(b["stop_signed_mm"]),
                    f"{e['STOPPED']} / {e['UNRESOLVED_CONTACT']} / {e['SINK']}",
                    _n(b["own_side_contact"]["fraction"]),
                    _n(b["own_side_flat"]["fraction"]),
                ]
            )
    L.append(
        _table(
            [
                "run",
                "bank",
                "abs stop error",
                "signed stop error",
                "STOPPED / UNRESOLVED / SINK",
                "own side (CONTACT)",
                "own side (flat)",
            ],
            rows,
        )
    )
    L.append("")
    rows = []
    for k in run_keys:
        q1 = results[k]["q1"]
        c = q1["contact_crossings"]
        rows.append(
            [
                k.name,
                f"{c['count']} (A {c['count_A']}, B {c['count_B']})",
                str(c["into_tissue"]),
                str(c["count_on_true_paths"]),
                _st(q1["ramp"]["CONTACT"]["width_mm"]),
                str(q1["ramp"]["CONTACT"]["n_without_width"]),
                _st(q1["ramp"]["CONTACT"]["peak_q_stop_ahead"]),
                _st(q1["ramp"]["PIA"]["width_mm"]),
            ]
        )
    L.append(
        f"Contact crossings: tolerance {CROSS_TOL_MM} mm, flat-bank columns only. "
        "Stop ramp: q_stop 0.2 → 0.8 on the line through the stop."
    )
    L.append("")
    L.append(
        _table(
            [
                "run",
                "crossings",
                "far sample in tissue",
                "on true paths",
                "ramp width CONTACT",
                "CONTACT without width",
                "peak q_stop within one spacing ahead",
                "ramp width PIA",
            ],
            rows,
        )
    )
    L += ["", "## Q3 gap end", ""]
    rows = []
    for k in run_keys:
        q3 = results[k]["q3"]
        rg = q3["regions"]
        pl = q3["places"]
        mc = q3["moving_cortex_nodes"]

        def wp(name: str, _pl: Mapping[str, Any] = pl) -> str:
            return f"{_pl[name]['window']} / {_pl[name]['elsewhere']}"

        rows.append(
            [
                k.name,
                _n(q3["window_mm"]),
                _st(rg["open"]["stop_abs_mm"]),
                _st(rg["window"]["stop_abs_mm"]),
                _st(rg["fused"]["stop_abs_mm"]),
                f"{mc['window']} / {mc['elsewhere']}",
                wp("sink_nodes"),
                wp("reversed_edges"),
                wp("unresolved_cells"),
                wp("overlap"),
                wp("focal_points"),
                wp("contact_crossings"),
                "yes"
                if q3["recount_vs_run_json"]["matches"]
                else str(q3["recount_vs_run_json"]["mismatch"]),
            ]
        )
    L.append("Places: in the window / elsewhere (local z).")
    L.append("")
    L.append(
        _table(
            [
                "run",
                "window ± (mm)",
                "abs stop error: open",
                "window",
                "fused",
                "moving cortex nodes",
                "sink nodes",
                "reversed edges",
                "unresolved cells",
                "overlap",
                "focal",
                "contact crossings",
                "recount = run.json",
            ],
            rows,
        )
    )
    L += ["", "## Q4 the stopped region", ""]
    rows = []
    for k in run_keys:
        q4 = results[k]["q4"]
        o = q4["open_slot"]
        n_in = sum(1 for d in q4["sheet"] if d.get("in_box"))
        rows.append(
            [
                k.name,
                _n(q4["spacing_mm"]),
                _n(q4["two_g_mm"]),
                _st(q4["sheet_length_mm"]),
                f"{q4['sheet_resolved']} of {n_in}",
                f"{_n(o['medium_stopped']['fraction'])} (n {o['n_medium_nodes']})",
                f"{_n(o['cortex_stopped']['fraction'])} (n {o['n_cortex_nodes']})",
            ]
        )
    L.append(
        _table(
            [
                "run",
                "spacing",
                "true 2g",
                "sheet length (q_stop > 0.5)",
                "lines with a sheet",
                "open slot: medium nodes stopped",
                "open slot: cortex nodes (sdf < −spacing) stopped",
            ],
            rows,
        )
    )
    L += ["", "## Q5 τ at the contact", ""]
    rows = []
    for k in run_keys:
        q5 = results[k]["q5"]
        sp = q5["speed_banks"] or {}
        for bn in ("A", "B"):
            b = q5["by_bank"][bn]
            rows.append(
                [
                    k.name,
                    bn,
                    _n(sp.get("solved_ratio_A_over_B")),
                    _n(sp.get("thickness_ratio_tA_over_tB")),
                    _st(b["tau_arrival_contact"]),
                    _st(b["level_abs_dev_mm"]),
                    " ".join(_n(v, 2) for v in b["level_signed_median_mm"]),
                ]
            )
    L.append("Signed level deviation: median per level, φ = 0.1 … 0.9.")
    L.append("")
    L.append(
        _table(
            [
                "run",
                "bank",
                "speed ratio A/B",
                "tA/tB",
                "tau_arrival at CONTACT",
                "abs level deviation",
                "signed median per level",
            ],
            rows,
        )
    )
    L += ["", "## Q6 the origin of τ", ""]
    if not doc["q6"]:
        L.append("No plain run with an anchor or coarse run: not run.")
    for name, q6 in doc["q6"].items():
        an, co = q6["anchor"], q6["coarse"]
        L.append(f"**{name}.** Sweeps per width, top: {q6['sweeps_top']}.")
        if isinstance(an, dict):
            L.append(
                f"Anchor: sweeps {an['sweeps_anchor']}, "
                f"constant c = {_n(an['constant'])}"
                f" over {an['n_nodes']} nodes, |residual| {_st(an['residual_abs'])}; "
                f"level distance (after c) {_st(an['level_distance_mm'])}, "
                f"{an['level_found_both']} of {an['level_pairs']} found in both. "
                f"|level deviation| top {_st(an['level_dev_abs_top'])}, anchor "
                f"{_st(an['level_dev_abs_anchor'])}."
            )
        else:
            L.append("Anchor: not run.")
        if isinstance(co, dict):
            L.append(
                f"Coarse first: fine sweeps {co['sweeps_fine']}, coarse solve "
                f"{co['sweeps_coarse_solve']}."
            )
        else:
            L.append("Coarse first: not run.")
        L.append("")
    L += ["", "## R6 independence across the contact", ""]
    if not doc["r6"]:
        L.append("No plain run with a shiftB run: not run.")
    for name, r6 in doc["r6"].items():
        L.append(
            f"**{name}** against **{r6['shift']}**. Sweeps {r6['sweeps_top']} and "
            f"{r6['sweeps_shift']}."
        )
        L.append("")
        rows = []
        for bn in ("A", "B"):
            d = r6["nodes"][bn]
            for bt, ba in zip(
                d["dtau_by_distance"], d["angle_deg_by_distance"], strict=True
            ):
                rows.append([bn, f"{_n(bt['lo'])}–{_n(bt['hi'])}", _st(bt), _st(ba)])
        L.append(
            _table(["bank", "−stop_sdf (mm)", "abs Δ tau_mean", "angle (deg)"], rows)
        )
        L.append("")
        rows = [
            [
                kk,
                _st(v["path_distance_mm"]),
                _st(v["stop_distance_mm"]),
                str(v["end_reason_changed"]),
            ]
            for kk, v in r6["columns"].items()
        ]
        L.append(
            _table(
                [
                    "bank, true stop kind",
                    "path distance",
                    "stop distance",
                    "end reason changed",
                ],
                rows,
            )
        )
        L.append("")
    L += ["", "## Figures", ""]
    for f, size in doc["figures"].items():
        L.append(f"- {f} ({size} bytes)")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {
            str(k): _jsonable(v) for k, v in x.items() if not str(k).startswith("_")
        }
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return _jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid-label", default="uniform0.16")
    ap.add_argument("--label", default="", help="a label for the titles and the files")
    ap.add_argument("--fig-suffix", default="", help="added to each figure file name")
    ap.add_argument("--phantoms", nargs="*", default=list(CK.NAMES), choices=CK.NAMES)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    found = find_runs(a.runs, a.grid_label, a.phantoms)
    keys = [RunKey(p, s, n, "") for p in a.phantoms for s in SOURCES for n in NOISES]
    for k in sorted(found, key=lambda k: (k.phantom, k.source, k.noise, k.variant)):
        if k not in keys:
            keys.append(k)
    for p, s, n in EXPECTED_VARIANTS:
        for v in VARIANTS[1:]:
            k = RunKey(p, s, n, v)
            if p in a.phantoms and k not in keys:
                keys.append(k)
    geos: dict[str, Geo] = {}
    results: dict[RunKey, dict[str, Any]] = {}
    sec_runs: dict[str, dict[RunKey, tuple[FG.UniformGrid, Array]]] = {}
    figures: dict[str, int] = {}
    for k in keys:
        if k not in found:
            continue
        print(f"{k.name}: {found[k]}", flush=True)
        run = load_run(k, found[k])
        if k.phantom not in geos:
            geos[k.phantom] = Geo(k.phantom, a.truth, run.rec["phantom"]["shape"])
        st = dataclasses.replace(
            T.DEFAULTS, stop_level=float(run.rec["settings"]["stop_level"])
        )
        results[k] = per_run(run, geos[k.phantom], st)
        if k.variant == "":
            # keep only what the section figure needs
            sec_runs.setdefault(k.phantom, {})[k] = (run.grid, run.q_stop)
        del run
    for name in a.phantoms:
        if name in sec_runs:
            f = a.figures / fig_name(f"kiss_q4_sections_{name}", a.fig_suffix)
            figures[f.name] = figure_sections(
                name, geos[name], sec_runs[name], f, a.label
            )
    sec_runs.clear()
    f = a.figures / fig_name("kiss_q1q3_stop_vs_z", a.fig_suffix)
    figures[f.name] = figure_stop_vs_z(results, a.phantoms, geos, f, a.label)
    f = a.figures / fig_name("kiss_free_energy", a.fig_suffix)
    traces = {k.name: results[k]["_free_energy"] for k in keys if k in results}
    figures[f.name] = figure_free_energy(traces, f, a.label)
    q6s: dict[str, Any] = {}
    r6s: dict[str, Any] = {}
    for k in keys:
        if k.variant != "" or k not in found:
            continue
        ka, kc, ks = (
            dataclasses.replace(k, variant=v) for v in ("_anchor", "_coarse", "_shiftB")
        )
        if ka in found or kc in found:
            top = load_run(k, found[k])
            anchor = load_run(ka, found[ka]) if ka in found else None
            coarse = load_run(kc, found[kc]) if kc in found else None
            q6 = q6_origin(top, anchor, coarse)
            f = a.figures / fig_name(f"kiss_q6_{k.slug}", a.fig_suffix)
            figures[f.name] = figure_q6(q6, f, a.label)
            q6s[k.name] = q6
            del top, anchor, coarse
        if ks in found:
            top = load_run(k, found[k])
            shift = load_run(ks, found[ks])
            r6 = r6_independence(top, shift, geos[k.phantom])
            f = a.figures / fig_name(f"kiss_r6_{k.slug}", a.fig_suffix)
            figures[f.name] = figure_r6(r6, f, a.label)
            r6s[k.name] = r6
            del top, shift
    doc: dict[str, Any] = dict(
        grid_label=a.grid_label,
        label=a.label,
        fig_suffix=a.fig_suffix,
        runs_dir=str(a.runs),
        truth_dir=str(a.truth),
        constants=dict(
            CROSS_TOL_MM=CROSS_TOL_MM,
            WINDOW_MM=WINDOW_MM,
            Z_BIN_MM=Z_BIN_MM,
            SECTION_Z_MM=SECTION_Z_MM,
            SHEET_YS_MM=SHEET_YS_MM,
            SHEET_ZS_MM=SHEET_ZS_MM,
            OPEN_MAX_Z_MM=OPEN_MAX_Z_MM,
            RAMP=(RAMP_LO, RAMP_HI, RAMP_REACH_MM),
            R6_MAX_MM=R6_MAX_MM,
        ),
        runs={k.name: ("not run" if k not in results else results[k]) for k in keys},
        q6=q6s,
        r6=r6s,
        figures=figures,
    )
    doc["time_s"] = time.perf_counter() - t0
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "kiss_views.json").write_text(json.dumps(_jsonable(doc), indent=1) + "\n")
    (a.out / "kiss_views.md").write_text(markdown(doc, keys, results))
    print(
        f"wrote {a.out / 'kiss_views.json'}, {a.out / 'kiss_views.md'} and "
        f"{len(figures)} figures in {doc['time_s']:.1f} s",
        flush=True,
    )
    return doc


if __name__ == "__main__":
    main()
