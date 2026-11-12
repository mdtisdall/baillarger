"""The measures, tables and views of PR 6 (docs/plans/flow-field-phantoms.md, tasks 6.1
hairpins, 6.2 jumps, 6.3 the spacing scan; questions 2, 4 and 7; design decisions R2, R3,
R7, R8, R9).

The script reads run directories that other jobs write (`ff_run.py`, `ff_pr6_runs.py`).
It does not solve. All numbers are diagnostics, not pass/fail rules, and no constant here
is tuned: the few view parameters are named below and written into the JSON. Lengths are
in mm; "local" is the phantom's local mm frame (slot half-plane x = 0, y >= 0; the slot's
edge line K is x = y = 0, along z; bank A at x > 0, bank B at x < 0). Distributions are
n, median, p90, p99 and max (`ff_run.stats`).

Usage:
    python ff_pr6_views.py {all,hairpins,jumps,grid} --runs RUNS_DIR --truth TRUTH_DIR
        --features FEATURES_JSON --figures FIG_DIR --tables TABLE_DIR
        [--noises 0 1] [--grid-noise 1] [--extra-adapt LABEL=ADAPT_JSON ...]

Nothing names a fixed path. A run directory is RUNS_DIR/<phantom>/<source>_<noise:g>_<grid>/
with the grid label uniform0.64 .. uniform0.16 or octree. A run counts as present when its
run.json exists (for the phantom ``noise`` octree run: adapt.json). Every other expected run
is skipped and listed (on the screen, in the JSON under "skipped", and in the markdown).
TRUTH_DIR holds TRUTH_DIR/<phantom>/truth.npz. FEATURES_JSON is accepted and recorded in
the output; the geometry needs no image, so it is not read.

Tables (TABLE_DIR, each as .md and .json) and figures (FIG_DIR, each PNG below 1 MB):

A. ``pr6_hairpins``: fundus_g0.48/0.16/0.04 and curved_g0.16/0.04, grids uniform0.16 and
   octree, both sources, the noise levels of --noises. One row per run.
   - Topology from run.json (flow lines across the stop set, overlap, sinks, reversed
     edges, unresolved contact cells) and the focal points inside the cortex recounted
     with `ff_checks.column_meetings` and the geometry (the count is compared with
     run.json's). A focal point's place is its distance r from the slot's edge line K
     (r = hypot(x, y) in local mm; the hairpin radius is g), given as n, median, min, max
     and as the count with r <= g + one voxel.
   - The start check. For each STOPPED column, the bank of the solved stop and the bank
     of the TRUE stop (``geometry.bank`` at the stop), not the root's bank: where tA != tB
     some bank-B true columns end on the x > 0 side. Columns whose solved stop is in the
     other bank are counted, and apart those whose solved stop lies more than half a voxel
     from the plane x = 0 (``BANK_TOL_MM``; a stop at x ~ 0 on the bottom of the hairpin is
     not in either bank in a useful sense). The count of columns whose true stop is not in
     the root's bank is given as a reference.
   - Octree runs: the cells finer than 0.16 mm (``prior_shaped``, read from the arrays that
     field.npz stores: grid_cell_origin, grid_cell_size_mm, grid_prior_shaped), their
     distance from K (the cell centre) against g, and the count of fine cells with
     distance > g + one voxel ("fine cells where no constraint needs them", R8). The count
     with the nearest point of the cell (centre distance minus half the cell diagonal) is
     given too. Where a fine cell's centre lies (cortex, outside medium, WM) is counted.
     This measure is the plan's: it takes the tip as the only place that needs fine cells.
     A fused slot (g < one voxel) also has stop faces at x = +-g along the whole slot; those
     fine cells are counted as "far" by this definition.
   - The hanging step of the final octree (adapt.json final.hanging_step).
   - Figure ``pr6_hairpin_map_<slug>.png`` (octree runs): a section through K normal to the
     slot's edge line (local z = 0) with the cells (the intersection of each cell with the
     plane; colour by cell size), the true stop surface (the zero contour of ``stop_sdf``),
     the solved column samples within ``SECTION_SLAB_MM`` of the plane, and the focal
     points; a zoom round the tip; and the distance of each prior-shaped cell from K
     against its size, with g and g + one voxel.
   - ``pr6_hanging`` (table): the hanging step on every fundus, curved and abraded octree
     run.

B. ``pr6_jumps``: abraded, every octree and uniform run found, both sources and noise
   levels. Per run and per group (intact, intact in bank A, patch 0 = the 0.3 mm patch,
   patch 1 = the 0.9 mm patch; the patches come from the geometry, a column belongs to
   the patch whose cylinder (rho < radius) holds its TRUE stop if its true jump is
   positive):
   the solved jump of the STOPPED columns raw, minus the median solved jump on intact
   cortex of the same run, the true jump, and the error |solved - true| in units of dtau
   (run.json settings.dtau; the plan calls it the tau step), raw and after the intact
   median is removed. p_jump: columns.npz p_jump is nan (ff_run does not pass it), so the
   node array ``p_jump`` of field.npz is read at the nearest MOVING node (q_stop < 0.5) of
   each column's stop; the distance to that node is also given.
   Figures ``pr6_jump_map_<slug>.png`` (the true jump, the solved jump, the solved minus
   the true jump in dtau, and p_jump over bank A's white surface: x = arclength from the
   slot's edge, z = along the slot, root positions; the true patch outlines) and
   ``pr6_jump_scatter.png`` (solved against true jump, one panel per run).

C. ``pr6_grid``: every phantom (the phantoms of ``ff_truth.PHANTOMS``), source oracle,
   noise --grid-noise, grids uniform0.64, 0.48, 0.32, 0.16 and octree. Per run the node
   count, stops (|stop error|) p50 and p90, paths (mean path error) p50 and p90, levels
   p50 and p90 (all from run.json measures), the topology counts, sweeps, time and peak
   memory. Figures ``pr6_grid_<phantom>.png`` (each measure against node count; the octree
   is the star) and ``pr6_grid_overview.png``. ``pr6_octree``: for each octree run (and
   each --extra-adapt file) the nodes and leaves per level, the prior-shaped count and for
   each round whether the refinement trial and the coarsening trial were accepted
   (adapt.json). Any refinement accepted on slab or noise is a failure signature of R9.

NumPy. The only loops over columns are in `ff_checks.column_meetings`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import ff_checks as FC  # noqa: E402
import ff_kiss_views as FK  # noqa: E402
import ff_run as FR  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402

Array = T.Array
H = T.VOXEL_MM

HAIRPIN_PHANTOMS = (
    "fundus_g0.48",
    "fundus_g0.16",
    "fundus_g0.04",
    "curved_g0.16",
    "curved_g0.04",
)
HAIRPIN_GRIDS = ("uniform0.16", "octree")
JUMP_PHANTOM = "abraded"
JUMP_GRIDS = ("octree", "uniform0.64", "uniform0.48", "uniform0.32", "uniform0.16")
GRID_LABELS = ("uniform0.64", "uniform0.48", "uniform0.32", "uniform0.16", "octree")
SOURCES = ("oracle", "image")
NOISES = (0.0, 1.0)
MOVING = 0.5  # a node moves if q_stop < this (the integrator's stop level)

BANK_TOL_MM = 0.5 * H  # see the module docstring (start check)
SECTION_SLAB_MM = H  # column samples within this of the section plane are drawn
SECTION_WIDE_X = (-3.2, 3.2)  # local mm
SECTION_WIDE_Y = (-3.2, 1.0)
SECTION_ZOOM_MM = 0.6  # half-width of the zoom round K
CELL_LEVEL_COLOURS = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860")

# ---------------------------------------------------------------- the runs


@dataclass(frozen=True)
class RunId:
    phantom: str
    source: str
    noise: float
    grid: str

    @property
    def dirname(self) -> str:
        return f"{self.source}_{self.noise:g}_{self.grid}"

    @property
    def name(self) -> str:
        return f"{self.phantom} {self.source} {self.noise:g} {self.grid}"

    @property
    def slug(self) -> str:
        return f"{self.phantom}_{self.source}_{self.noise:g}_{self.grid}"


def expected_ids(
    phantoms: Sequence[str],
    grids: Sequence[str],
    sources: Sequence[str] = SOURCES,
    noises: Sequence[float] = NOISES,
) -> list[RunId]:
    return [
        RunId(p, s, float(n), g)
        for p in phantoms
        for g in grids
        for s in sources
        for n in noises
    ]


def find_runs(
    root: Path, ids: Sequence[RunId], need: str = "run.json"
) -> tuple[dict[RunId, Path], list[RunId]]:
    """(the run directories that exist, the ids that do not). A directory exists for
    this purpose when its file `need` is there (the phantom ``noise`` octree run has no
    run.json: ``need="adapt.json"``)."""
    found: dict[RunId, Path] = {}
    skipped: list[RunId] = []
    for rid in ids:
        d = Path(root) / rid.phantom / rid.dirname
        if (d / need).is_file():
            found[rid] = d
        else:
            skipped.append(rid)
    return found, skipped


def _need(rid: RunId) -> str:
    return "adapt.json" if rid.phantom == "noise" else "run.json"


class Cache:
    """The geometry and truth of each phantom, loaded once."""

    def __init__(self, truth_dir: Path) -> None:
        self.truth_dir = Path(truth_dir)
        self._geo: dict[tuple[str, tuple[int, ...]], tuple[Any, float]] = {}
        self._truth: dict[str, T.Truth] = {}

    def geometry(self, name: str, shape: Sequence[int]) -> tuple[Any, float]:
        key = (name, tuple(int(s) for s in shape))
        if key not in self._geo:
            self._geo[key] = FTR.geometry_of(name, key[1])  # type: ignore[arg-type]
        return self._geo[key]

    def truth(self, name: str) -> T.Truth:
        if name not in self._truth:
            self._truth[name] = T.Truth.from_npz(self.truth_dir / name / "truth.npz")
        return self._truth[name]


@dataclass
class Case:
    """The parts of one run that the views read."""

    rid: RunId
    path: Path
    rec: dict[str, Any]
    cols: T.Columns
    truth: T.Truth
    geo: Any
    tau_scale: float

    @property
    def dtau(self) -> float:
        return float(self.rec["settings"]["dtau"])

    def field_arrays(self, keys: Sequence[str]) -> dict[str, Array]:
        z = np.load(self.path / "field.npz", allow_pickle=False)
        return {k: np.asarray(z[k]) for k in keys if k in z.files}

    def adapt(self) -> dict[str, Any] | None:
        f = self.path / "adapt.json"
        return json.loads(f.read_text()) if f.is_file() else None


def load_case(rid: RunId, path: Path, cache: Cache) -> Case:
    rec = json.loads((path / "run.json").read_text())
    geo, ts = cache.geometry(rid.phantom, rec["phantom"]["shape"])
    cz = np.load(path / "columns.npz", allow_pickle=False)
    cols = T.Columns(**{f.name: cz[f.name] for f in dataclasses.fields(T.Columns)})
    return Case(rid, path, rec, cols, cache.truth(rid.phantom), geo, ts)


# ---------------------------------------------------------------- small helpers


def stats(x: Any) -> dict[str, Any]:
    return FR.stats(x)


def _save(fig: Any, path: Path) -> int:
    size = FK._save(fig, path)
    assert size < 1_000_000, f"{path}: {size} bytes"
    return size


def _g(d: Any, *keys: str) -> Any:
    """d[k0][k1]... or None."""
    for k in keys:
        if not isinstance(d, Mapping) or k not in d:
            return None
        d = d[k]
    return d


def _end_counts(end: Any) -> dict[str, int]:
    return {e.name: int((np.asarray(end) == int(e)).sum()) for e in T.EndReason}


def md_table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return FK._table(head, rows)


def _n(x: Any, nd: int = 3) -> str:
    return FK._n(x, nd)


def _st(d: Any, nd: int = 3) -> str:
    return FK._st(d, nd)


def skipped_md(skipped: Sequence[RunId]) -> str:
    if not skipped:
        return "Skipped runs: none.\n"
    lines = [f"Skipped runs (no run directory): {len(skipped)}", ""]
    lines += [f"- {r.phantom}/{r.dirname}" for r in skipped]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- A: hairpins


def tip_distance(Y: Array) -> Array:
    """Distance (mm) of local points (N, 3) from the slot's edge line K (x = y = 0)."""
    Y = np.asarray(Y, float)
    return np.asarray(np.hypot(Y[:, 0], Y[:, 1]))


def place_report(places_local: Array, g: float, voxel: float = H) -> dict[str, Any]:
    """Distances from K of the places of focal points (local mm), against the hairpin
    radius g: n, the distribution, min and the count with r <= g + one voxel."""
    r = tip_distance(places_local) if len(places_local) else np.zeros(0)
    return dict(
        n=int(len(r)),
        r_mm=stats(r),
        r_min_mm=float(r.min()) if len(r) else None,
        n_within_g_plus_voxel=int((r <= g + voxel).sum()),
        g=float(g),
    )


def bank_check(
    solved_bank: Any,
    true_bank: Any,
    stopped: Any,
    solved_x: Any | None = None,
    tol_mm: float = BANK_TOL_MM,
    root_bank: Any | None = None,
) -> dict[str, Any]:
    """The start check. Compares the bank of each solved stop with the bank of the TRUE
    stop, for the STOPPED columns. n_wrong: the banks differ. n_wrong_clear: they differ
    and the solved stop is farther than tol_mm from the plane x = 0 (solved_x, local
    mm). n_true_stop_not_in_root_bank: reference, columns whose true stop is in the
    other bank than their root (root_bank)."""
    sb, tb, st = (np.asarray(a) for a in (solved_bank, true_bank, stopped))
    wrong = st.astype(bool) & (sb != tb)
    out: dict[str, Any] = dict(
        n_columns=int(len(st)),
        n_stopped=int(st.sum()),
        n_wrong=int(wrong.sum()),
        wrong_columns=np.flatnonzero(wrong),
    )
    if solved_x is not None:
        clear = wrong & (np.abs(np.asarray(solved_x, float)) > tol_mm)
        out["n_wrong_clear"] = int(clear.sum())
        out["tol_mm"] = float(tol_mm)
    if root_bank is not None:
        out["n_true_stop_not_in_root_bank"] = int((np.asarray(root_bank) != tb).sum())
    return out


def fine_cell_report(
    centres_local: Array,
    size_mm: Array,
    fine: Any,
    g: float,
    voxel: float = H,
    label: Any | None = None,
) -> dict[str, Any]:
    """The cells finer than the image against the hairpin (R8). centres_local (C, 3):
    cell centres in local mm; size_mm (C,); fine (C,) bool (prior-shaped). far: the
    centre is farther than g + one voxel from K. far_nearest: the nearest point of the
    cell is (centre distance minus half the cell diagonal). label (C,): the label at the
    centre (CORTEX, MEDIUM, WM), counted for the fine cells."""
    fine = np.asarray(fine, bool)
    d = tip_distance(centres_local)
    size = np.asarray(size_mm, float)
    cut = g + voxel
    far = fine & (d > cut)
    nearest = d - 0.5 * np.sqrt(3.0) * size
    far_nearest = fine & (nearest > cut)
    by_size: dict[str, Any] = {}
    for s in np.unique(np.round(size[fine], 9)):
        m = fine & (np.abs(size - s) < 1e-9)
        by_size[f"{s:g}"] = dict(n=int(m.sum()), n_far=int((m & far).sum()))
    out: dict[str, Any] = dict(
        n_cells=int(len(size)),
        n_fine=int(fine.sum()),
        n_fine_far=int(far.sum()),
        n_fine_far_nearest_point=int(far_nearest.sum()),
        far_cut_mm=float(cut),
        g=float(g),
        d_tip_fine_mm=stats(d[fine]),
        by_size_mm=by_size,
        _d=d[fine],
        _size=size[fine],
    )
    if label is not None:
        lab = np.asarray(label)[fine]
        out["fine_label_counts"] = {
            nm: int((lab == v).sum())
            for nm, v in (("CORTEX", T.CORTEX), ("MEDIUM", T.MEDIUM), ("WM", T.WM))
        }
    return out


def _cell_arrays(case: Case) -> dict[str, Array] | None:
    z = case.field_arrays(
        ["grid_cell_origin", "grid_cell_size_mm", "grid_prior_shaped"]
    )
    return z if len(z) == 3 else None


def cell_centres_local(geo: Any, origin_vox: Array, size_mm: Array) -> Array:
    return np.asarray(geo.to_local(origin_vox + 0.5 * (size_mm / H)[:, None]))


def section_polygons(
    geo: Any,
    origin_vox: Array,
    size_mm: Array,
    z0: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[list[Array], T.IntArray]:
    """The intersection of each cell with the local plane z = z0, as polygons in local
    (x, y), for the cells whose polygon can reach the window (xlim, ylim). Returns the
    polygons and the index of the cell of each."""
    size_vox = size_mm / H
    cen = cell_centres_local(geo, origin_vox, size_mm)
    reach = 0.87 * size_mm  # half the diagonal, rounded up
    near = (
        (np.abs(cen[:, 2] - z0) <= reach)
        & (cen[:, 0] >= xlim[0] - reach)
        & (cen[:, 0] <= xlim[1] + reach)
        & (cen[:, 1] >= ylim[0] - reach)
        & (cen[:, 1] <= ylim[1] + reach)
    )
    idx = np.flatnonzero(near)
    if len(idx) == 0:
        return [], idx
    corner = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], float)
    pts = origin_vox[idx][:, None, :] + corner[None] * size_vox[idx][:, None, None]
    loc = np.asarray(geo.to_local(pts.reshape(-1, 3))).reshape(len(idx), 8, 3)
    edges = [
        (a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1
    ]
    polys: list[Array] = []
    keep: list[int] = []
    for n, L in enumerate(loc):
        P: list[Array] = []
        for a, b in edges:
            za, zb = L[a, 2] - z0, L[b, 2] - z0
            if za * zb < 0:
                t = za / (za - zb)
                P.append(L[a, :2] + t * (L[b, :2] - L[a, :2]))
            elif za == 0.0:
                P.append(L[a, :2])
        if len(P) < 3:
            continue
        Q = np.unique(np.round(np.array(P), 12), axis=0)
        if len(Q) < 3:
            continue
        c = Q.mean(0)
        o = np.argsort(np.arctan2(Q[:, 1] - c[1], Q[:, 0] - c[0]))
        polys.append(Q[o])
        keep.append(n)
    return polys, idx[np.array(keep, np.int64)]


def hanging_row(adapt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    hs = _g(adapt, "final", "hanging_step")
    if not isinstance(hs, Mapping):
        return None
    return dict(
        n_hanging=hs.get("n_hanging"),
        n_moving=hs.get("n_moving"),
        tau_diff_median=_g(hs, "tau_diff", "median"),
        tau_diff_max=_g(hs, "tau_diff", "max"),
        angle_deg_median=_g(hs, "angle_deg", "median"),
        angle_deg_max=_g(hs, "angle_deg", "max"),
        q_stop_diff_max=_g(hs, "q_stop_diff", "max"),
    )


def hairpin_run(case: Case) -> dict[str, Any]:
    geo, cols, truth, rec = case.geo, case.cols, case.truth, case.rec
    g = float(geo.p["g"])
    stopped = cols.end == int(T.EndReason.STOPPED)
    topo = _g(rec, "measures", "topology") or {}
    meet = FC.column_meetings(cols, geo)
    foc = meet["focal"]
    foc_local = geo.to_local(foc["places"]) if foc["count"] else np.zeros((0, 3))
    solved_local = geo.to_local(cols.stops)
    true_local = geo.to_local(truth.stops)
    bc = bank_check(
        geo.bank(solved_local),
        geo.bank(true_local),
        stopped,
        solved_x=solved_local[:, 0],
        root_bank=truth.bank,
    )
    row: dict[str, Any] = dict(
        run=case.rid.name,
        phantom=case.rid.phantom,
        source=case.rid.source,
        noise=case.rid.noise,
        grid=case.rid.grid,
        g=g,
        n_nodes=_g(rec, "measures", "cost", "n_nodes") or _g(rec, "grid", "n_nodes"),
        n_columns=int(truth.n),
        ends=_end_counts(cols.end),
        stop_error_mm=_g(rec, "measures", "stops", "abs_mm"),
        topology_json=topo,
        focal_recount=dict(
            focal_in_cortex=int(foc["count"]),
            focal_all_places=int(foc.get("count_all", foc["count"])),
            overlap_in_cortex=int(meet["overlap"]["count"]),
            overlap_all_places=int(meet["overlap"].get("count_all", 0)),
        ),
        focal_places=place_report(foc_local, g),
        bank_check=bc,
        hanging=None,
        fine_cells=None,
        figure=None,
    )
    adapt = case.adapt()
    row["hanging"] = hanging_row(adapt)
    cells = _cell_arrays(case)
    if case.rid.grid == "octree" and cells is not None:
        cen = cell_centres_local(
            geo, cells["grid_cell_origin"], cells["grid_cell_size_mm"]
        )
        lab = np.asarray(geo.label(cen))
        row["fine_cells"] = fine_cell_report(
            cen,
            cells["grid_cell_size_mm"],
            cells["grid_prior_shaped"].astype(bool),
            g,
            label=lab,
        )
    return row


def figure_hairpin_map(case: Case, row: Mapping[str, Any], path: Path) -> int:
    geo, cols = case.geo, case.cols
    g = float(row["g"])
    cells = _cell_arrays(case)
    assert cells is not None
    org, size = cells["grid_cell_origin"], cells["grid_cell_size_mm"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    sizes = sorted({round(float(s), 9) for s in size})
    colour = {
        s: CELL_LEVEL_COLOURS[i % len(CELL_LEVEL_COLOURS)] for i, s in enumerate(sizes)
    }
    sp = np.asarray(geo.to_local(cols.paths.reshape(-1, 3))).reshape(cols.paths.shape)
    inslab = np.abs(sp[:, :, 2]) <= SECTION_SLAB_MM
    px, py = sp[:, :, 0][inslab][::4], sp[:, :, 1][inslab][::4]
    foc = FC.column_meetings(cols, geo)["focal"]
    fl = geo.to_local(foc["places"]) if foc["count"] else np.zeros((0, 3))
    for a, (xl, yl, ttl) in zip(
        ax[:2],
        (
            (SECTION_WIDE_X, SECTION_WIDE_Y, "section z = 0, all cells"),
            (
                (-SECTION_ZOOM_MM, SECTION_ZOOM_MM),
                (-SECTION_ZOOM_MM, SECTION_ZOOM_MM),
                "zoom round the tip",
            ),
        ),
        strict=True,
    ):
        polys, ii = section_polygons(geo, org, size, 0.0, xl, yl)
        cc = [colour[round(float(size[i]), 9)] for i in ii]
        a.add_collection(
            PolyCollection(
                polys, facecolors=cc, edgecolors="k", linewidths=0.2, alpha=0.45
            )
        )
        xs, ys = np.linspace(*xl, 500), np.linspace(*yl, 500)
        XX, YY = np.meshgrid(xs, ys)
        Y3 = np.stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)], 1)
        sdf = np.asarray(geo.stop_sdf(Y3)).reshape(XX.shape)
        a.contour(XX, YY, sdf, levels=[0.0], colors="k", linewidths=1.4)
        a.plot(px, py, ".", ms=1.2, color="tab:red", alpha=0.6)
        if len(fl):
            a.plot(fl[:, 0], fl[:, 1], "x", color="m", ms=7, label="focal point")
        th = np.linspace(0, 2 * np.pi, 200)
        a.plot((g + H) * np.cos(th), (g + H) * np.sin(th), ":", color="g", lw=1.0)
        a.plot([0], [0], "+", color="g", ms=8)
        a.set_xlim(*xl)
        a.set_ylim(*yl)
        a.set_aspect("equal")
        a.set_xlabel("local x (mm)")
        a.set_ylabel("local y (mm)")
        a.set_title(ttl, fontsize=9)
    ax[0].legend(
        handles=[
            Line2D([], [], marker="s", ls="", color=colour[s], label=f"cell {s:g} mm")
            for s in sizes
        ]
        + [
            Line2D([], [], color="k", label="true stop surface"),
            Line2D([], [], marker=".", ls="", color="tab:red", label="solved columns"),
            Line2D([], [], ls=":", color="g", label="r = g + 1 voxel"),
        ],
        fontsize=7,
        loc="lower right",
    )
    fc = row["fine_cells"]
    a = ax[2]
    if fc is not None and fc["n_fine"]:
        a.scatter(fc["_d"], fc["_size"], s=6, alpha=0.5)
    a.axvline(g, color="k", lw=1, label="g (hairpin radius)")
    a.axvline(g + H, color="g", ls=":", label="g + 1 voxel")
    a.set_xlabel("distance of the cell centre from the slot's edge line K (mm)")
    a.set_ylabel("cell size (mm)")
    a.set_title(
        "prior-shaped cells: "
        + (
            "n/a"
            if fc is None
            else f"{fc['n_fine']} fine, {fc['n_fine_far']} farther than g + voxel"
        ),
        fontsize=9,
    )
    a.legend(fontsize=7)
    fig.suptitle(f"{case.rid.name}: the octree round the hairpin tip", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def hairpin_tables(rows: Sequence[Mapping[str, Any]], skipped: Sequence[RunId]) -> str:
    head1 = [
        "run", "nodes", "cols", "focal (json)", "focal (recount)", "focal r n/min/med/max (mm)",
        "in g+voxel", "overlap", "cross solved/true", "sink nodes/rev edges/sink cols",
        "unresolved", "stop err med/p90",
    ]  # fmt: skip
    r1 = []
    for r in rows:
        t = r["topology_json"]
        fp = r["focal_places"]
        r1.append([
            r["run"], _n(r["n_nodes"]), _n(r["n_columns"]),
            _n(t.get("focal_points")), _n(r["focal_recount"]["focal_in_cortex"]),
            f"{fp['n']} / {_n(fp['r_min_mm'])} / {_n(fp['r_mm']['median'])} / {_n(fp['r_mm']['max'])}",
            _n(fp["n_within_g_plus_voxel"]),
            _n(r["focal_recount"]["overlap_in_cortex"]),
            f"{_n(t.get('solved_stop_crossings'))} / {_n(t.get('true_stop_crossings'))}",
            f"{_n(t.get('sink_nodes'))} / {_n(t.get('reversed_edges'))} / {_n(t.get('sink_columns'))}",
            _n(t.get("unresolved_cells")),
            f"{_n(_g(r, 'stop_error_mm', 'median'))} / {_n(_g(r, 'stop_error_mm', 'p90'))}",
        ])  # fmt: skip
    head2 = [
        "run", "STOPPED", "wrong bank", "wrong bank, |x| > tol", "true stop not in root bank",
        "other ends",
    ]  # fmt: skip
    r2 = []
    for r in rows:
        b = r["bank_check"]
        other = {k: v for k, v in r["ends"].items() if k != "STOPPED" and v}
        r2.append([
            r["run"], _n(b["n_stopped"]), _n(b["n_wrong"]), _n(b.get("n_wrong_clear")),
            _n(b.get("n_true_stop_not_in_root_bank")), json.dumps(other) if other else "none",
        ])  # fmt: skip
    head3 = [
        "run", "g", "leaf cells", "fine (< 0.16 mm)", "fine, centre > g + voxel",
        "same, nearest point", "d from K, fine: med/p90/max", "fine by label", "hanging: n / tau med,max / angle med,max",
    ]  # fmt: skip
    r3 = []
    for r in rows:
        f = r["fine_cells"]
        if f is None:
            continue
        h = r["hanging"] or {}
        r3.append([
            r["run"], _n(r["g"]), _n(f["n_cells"]), _n(f["n_fine"]), _n(f["n_fine_far"]),
            _n(f["n_fine_far_nearest_point"]),
            f"{_n(f['d_tip_fine_mm']['median'])} / {_n(f['d_tip_fine_mm']['p90'])} / {_n(f['d_tip_fine_mm']['max'])}",
            json.dumps(f.get("fine_label_counts", {})),
            f"{_n(h.get('n_hanging'))} / {_n(h.get('tau_diff_median'))},{_n(h.get('tau_diff_max'))} / "
            f"{_n(h.get('angle_deg_median'))},{_n(h.get('angle_deg_max'))}",
        ])  # fmt: skip
    out = [
        "# PR 6 hairpins (task 6.1, question 2, R7, R8)\n",
        "Focal points inside the cortex are recounted from the saved columns "
        "(`ff_checks.column_meetings` with the geometry); `focal (json)` is run.json's count. "
        "r is the distance from the slot's edge line K; g is the hairpin radius.\n",
        "## Topology and focal points\n", md_table(head1, r1), "",
        "## Start check: bank of the solved stop against the bank of the true stop\n",
        f"Tolerance for 'clear': {BANK_TOL_MM:g} mm from the plane x = 0.\n",
        md_table(head2, r2), "",
        "## Octree: cells finer than 0.16 mm and the hanging step\n",
        "Distances are from the cell centre to K, against g + one voxel "
        f"({H:g} mm). 'nearest point' subtracts half the cell diagonal.\n",
        md_table(head3, r3) if r3 else "No octree run found.\n", "",
        skipped_md(skipped),
    ]  # fmt: skip
    return "\n".join(out)


def hanging_table(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    head = [
        "run",
        "hanging nodes",
        "moving hanging",
        "tau diff med / max",
        "angle med / max (deg)",
        "q_stop diff max",
    ]
    body, js = [], []
    for r in rows:
        h = r.get("hanging")
        if h is None:
            continue
        js.append(dict(run=r["run"], **h))
        body.append([
            r["run"], _n(h["n_hanging"]), _n(h["n_moving"]),
            f"{_n(h['tau_diff_median'])} / {_n(h['tau_diff_max'])}",
            f"{_n(h['angle_deg_median'])} / {_n(h['angle_deg_max'])}", _n(h["q_stop_diff_max"]),
        ])  # fmt: skip
    md = "# PR 6: the hanging step of the final octrees\n\n" + (
        md_table(head, body) if body else "No octree run with adapt.json found."
    )
    return md + "\n", js


# ---------------------------------------------------------------- B: jumps


def patch_index(
    local_stops: Array, jump_true: Array, patches: Sequence[Any]
) -> T.IntArray:
    """The patch of each column: k if the true jump is positive and the stop lies inside
    the cylinder of patch k (rho <= radius, with a relative 1e-6 slack), else -1."""
    out = np.full(len(local_stops), -1, np.int64)
    pos = np.asarray(jump_true) > 0
    for k, p in enumerate(patches):
        inside = pos & (np.asarray(p.rho(local_stops)) <= p.radius * (1.0 + 1e-6))
        out[inside & (out < 0)] = k
    return out


def nearest_moving_value(
    nodes: Array, q_stop: Array, values: Array, X: Array, level: float = MOVING
) -> tuple[Array, Array]:
    """The value at the nearest MOVING node (q_stop < level) to each point of X, and the
    distance to that node (voxels). nan where there is no moving node."""
    mov = np.flatnonzero(np.asarray(q_stop) < level)
    X = np.asarray(X, float)
    if len(mov) == 0 or len(X) == 0:
        return np.full(len(X), np.nan), np.full(len(X), np.nan)
    d, i = cKDTree(np.asarray(nodes)[mov]).query(X)
    return np.asarray(values)[mov][i].astype(float), np.asarray(d, float)


def relative_to_intact(jump: Array, intact: Any) -> tuple[float, Array]:
    """(the median of the jump over the intact columns, the jump minus that median).
    nan values (columns that did not stop) are left out of the median and stay nan."""
    j = np.asarray(jump, float)
    sel = np.asarray(intact, bool) & np.isfinite(j)
    m = float(np.median(j[sel])) if sel.any() else float("nan")
    return m, j - m


def jump_groups(
    jump_solved: Array,
    jump_true: Array,
    stopped: Any,
    patch: T.IntArray,
    bank: Any,
    dtau: float,
    p_jump: Array | None = None,
    n_patches: int = 2,
) -> dict[str, Any]:
    """The jump statistics by group. jump_solved is nan for the columns that did not
    stop (or is masked here with `stopped`). Groups: intact (patch = -1), intact_bankA
    (and bank = 0), patch0, patch1, ...; the median solved jump on intact is the
    reference of 'relative'. Errors are in units of dtau: err_raw = |solved - true|,
    err_rel = |solved - median_intact - true|."""
    st = np.asarray(stopped, bool)
    js = np.where(st, np.asarray(jump_solved, float), np.nan)
    jt = np.asarray(jump_true, float)
    patch = np.asarray(patch)
    intact = patch < 0
    m_int, rel = relative_to_intact(js, intact)
    members: dict[str, Any] = {
        "intact": intact,
        "intact_bankA": intact & (np.asarray(bank) == 0),
    }
    for k in range(n_patches):
        members[f"patch{k}"] = patch == k
    out: dict[str, Any] = dict(median_intact_jump=m_int, dtau=float(dtau), groups={})
    for name, m in members.items():
        s = m & st
        e_raw = np.abs(js - jt) / dtau
        e_rel = np.abs(rel - jt) / dtau
        gp: dict[str, Any] = dict(
            n_columns=int(m.sum()),
            n_stopped=int(s.sum()),
            true_jump=stats(jt[m]),
            solved_jump=stats(js[s]),
            solved_minus_intact_median=stats(rel[s]),
            err_raw_dtau=stats(e_raw[s]),
            err_rel_dtau=stats(e_rel[s]),
            frac_solved_positive=float((js[s] > 0).mean()) if s.any() else None,
        )
        if p_jump is not None:
            gp["p_jump"] = stats(np.asarray(p_jump, float)[s])
        out["groups"][name] = gp
    return out


def jump_run(case: Case) -> dict[str, Any]:
    geo, cols, truth = case.geo, case.cols, case.truth
    stopped = cols.end == int(T.EndReason.STOPPED)
    # the patch of a column from its TRUE stop: a solved stop off the patch must not
    # move a patch column into the intact group
    loc = geo.to_local(truth.stops)
    patches = list(getattr(geo, "patches", ()))
    pidx = patch_index(loc, truth.jump, patches)
    pj = np.full(len(cols.stops), np.nan)
    dist = np.full(len(cols.stops), np.nan)
    z = case.field_arrays(["nodes", "q_stop", "p_jump"])
    if len(z) == 3:
        pj, dist = nearest_moving_value(
            z["nodes"], z["q_stop"], z["p_jump"], cols.stops
        )
        pj = np.where(stopped, pj, np.nan)
    out = jump_groups(cols.jump, truth.jump, stopped, pidx, truth.bank, case.dtau, pj)
    out.update(
        run=case.rid.name,
        phantom=case.rid.phantom,
        source=case.rid.source,
        noise=case.rid.noise,
        grid=case.rid.grid,
        n_columns=int(truth.n),
        n_nodes=_g(case.rec, "measures", "cost", "n_nodes")
        or _g(case.rec, "grid", "n_nodes"),
        ends=_end_counts(cols.end),
        p_jump_source="field.npz p_jump at the nearest moving node of the stop",
        p_jump_node_distance_vox=stats(dist[stopped]),
        patches=[
            dict(radius=p.radius, depth=p.depth, centre=list(p.centre)) for p in patches
        ],
        topology_json=_g(case.rec, "measures", "topology"),
        hanging=hanging_row(case.adapt()),
        _patch=pidx,
        _pjump=pj,
    )
    return out


def bank_a_surface(Y: Array) -> tuple[Array, Array]:
    """(s, z) of local points on bank A's white surface: s = y on the flat part
    (y >= 0), and the arclength -r (pi/2 - theta) round the hairpin (y < 0), with r the
    distance from K and theta = arctan2(x, -y); z along the slot."""
    Y = np.asarray(Y, float)
    yq = np.minimum(Y[:, 1], 0.0)
    r = np.hypot(Y[:, 0], yq)
    th = np.arctan2(Y[:, 0], -yq)
    s = np.where(Y[:, 1] >= 0, Y[:, 1], -r * (0.5 * np.pi - th))
    return s, Y[:, 2]


def figure_jump_map(case: Case, res: Mapping[str, Any], path: Path) -> int:
    geo, cols, truth = case.geo, case.cols, case.truth
    stopped = cols.end == int(T.EndReason.STOPPED)
    a_bank = truth.bank == 0
    s, z = bank_a_surface(geo.to_local(truth.roots))
    js = np.where(stopped, cols.jump, np.nan)
    m_int = float(res["median_intact_jump"])
    panels: list[tuple[str, Any, str, Any]] = [
        ("true jump (mm of tau)", truth.jump, "viridis", None),
        ("solved jump", js, "viridis", None),
        (
            f"solved - true, in dtau = {case.dtau:g}",
            (js - truth.jump) / case.dtau,
            "coolwarm",
            "sym",
        ),
        ("p_jump at the stop", res["_pjump"], "magma", (0.0, 1.0)),
    ]
    vmax = float(np.nanmax(np.r_[truth.jump, js, 1e-9]))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8), sharex=True, sharey=True)
    for a, (ttl, v, cmap, lim) in zip(ax, panels, strict=True):
        v = np.asarray(v, float)
        if lim == "sym":
            lm = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 1.0
            vmin, vmx = -max(lm, 1e-9), max(lm, 1e-9)
        elif lim is None:
            vmin, vmx = 0.0, vmax
        else:
            vmin, vmx = lim
        ok = a_bank & np.isfinite(v)
        sc = a.scatter(s[ok], z[ok], c=v[ok], s=9, cmap=cmap, vmin=vmin, vmax=vmx)
        bad = a_bank & ~np.isfinite(v)
        a.plot(s[bad], z[bad], ".", color="0.6", ms=3)
        for p in getattr(geo, "patches", ()):
            th = np.linspace(0, 2 * np.pi, 200)
            a.plot(
                p.centre[1] + p.radius * np.cos(th),
                p.centre[2] + p.radius * np.sin(th),
                "k-",
                lw=1.1,
            )
        a.set_title(ttl, fontsize=9)
        a.set_xlabel("arclength from the slot's edge along bank A (mm)")
        a.set_aspect("equal")
        fig.colorbar(sc, ax=a, shrink=0.8)
    ax[0].set_ylabel("along the slot, local z (mm)")
    fig.suptitle(
        f"{case.rid.name}: jumps over bank A's white surface (roots); "
        f"median solved jump on intact cortex {m_int:.3g}; grey: not stopped; "
        "black: true patch outlines",
        fontsize=9,
    )
    fig.tight_layout()
    return _save(fig, path)


def figure_jump_scatter(
    items: Sequence[tuple[Case, Mapping[str, Any]]], path: Path
) -> int:
    n = len(items)
    nc = min(4, max(n, 1))
    nr = int(np.ceil(max(n, 1) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.0 * nr), squeeze=False)
    colours = {-1: "0.5", 0: "tab:blue", 1: "tab:red"}
    for a in axs.ravel():
        a.set_visible(False)
    for a, (case, res) in zip(axs.ravel(), items, strict=False):
        a.set_visible(True)
        stopped = case.cols.end == int(T.EndReason.STOPPED)
        pidx = res["_patch"]
        for k in sorted(set(pidx.tolist())):
            m = stopped & (pidx == k)
            a.plot(case.truth.jump[m], case.cols.jump[m], ".", ms=3, color=colours.get(k, "k"),
                   label="intact" if k < 0 else f"patch {k}")  # fmt: skip
        mx = float(np.nanmax(np.r_[case.truth.jump, case.cols.jump[stopped], 0.1]))
        a.plot([0, mx], [0, mx], "k-", lw=0.7)
        mi = float(res["median_intact_jump"])
        if np.isfinite(mi):
            a.plot([0, mx], [mi, mx + mi], "k:", lw=0.7)
        a.set_title(case.rid.name, fontsize=8)
        a.set_xlabel("true jump")
        a.set_ylabel("solved jump")
        a.legend(fontsize=6)
    fig.suptitle(
        "solved against true jump (solid: identity; dotted: shifted by the intact median)",
        fontsize=9,
    )
    fig.tight_layout()
    return _save(fig, path)


def jump_tables(results: Sequence[Mapping[str, Any]], skipped: Sequence[RunId]) -> str:
    head = [
        "run", "group", "n / stopped", "true jump med/max", "solved jump med/p90/max",
        "minus intact median: med/p90", "err raw (dtau) med/p90/max", "err after intact median removed: med/p90/max",
        "p_jump med/p90",
    ]  # fmt: skip
    rows = []
    for r in results:
        for name, gp in r["groups"].items():
            rows.append([
                r["run"], name, f"{gp['n_columns']} / {gp['n_stopped']}",
                f"{_n(gp['true_jump']['median'])} / {_n(gp['true_jump']['max'])}",
                f"{_n(gp['solved_jump']['median'])} / {_n(gp['solved_jump']['p90'])} / {_n(gp['solved_jump']['max'])}",
                f"{_n(gp['solved_minus_intact_median']['median'])} / {_n(gp['solved_minus_intact_median']['p90'])}",
                f"{_n(gp['err_raw_dtau']['median'])} / {_n(gp['err_raw_dtau']['p90'])} / {_n(gp['err_raw_dtau']['max'])}",
                f"{_n(gp['err_rel_dtau']['median'])} / {_n(gp['err_rel_dtau']['p90'])} / {_n(gp['err_rel_dtau']['max'])}",
                f"{_n(_g(gp, 'p_jump', 'median'))} / {_n(_g(gp, 'p_jump', 'p90'))}",
            ])  # fmt: skip
    ref = md_table(
        ["run", "median solved jump on intact cortex", "dtau", "STOPPED", "p_jump node distance (vox) med/max"],
        [[r["run"], _n(r["median_intact_jump"]), _n(r["dtau"]), _n(r["ends"]["STOPPED"]),
          f"{_n(r['p_jump_node_distance_vox']['median'])} / {_n(r['p_jump_node_distance_vox']['max'])}"]
         for r in results],
    )  # fmt: skip
    return "\n".join([
        "# PR 6 jumps on the abraded phantom (task 6.2, question 4, R2, R3)\n",
        "Jumps are in units of tau (the plan's tau has the unit of about 1 mm of depth); "
        "errors in units of dtau. 'intact' = true jump 0; 'patch0' = the 0.3 mm patch, "
        "'patch1' = the 0.9 mm patch. p_jump: field.npz p_jump at the nearest moving node "
        "of each stop.\n",
        "## Reference: the intact-cortex jump\n", ref, "",
        "## By group\n", md_table(head, rows) if rows else "No run found.\n", "",
        skipped_md(skipped),
    ])  # fmt: skip


# ---------------------------------------------------------------- C: the grid


def grid_row(
    rid: RunId, rec: Mapping[str, Any], adapt: Mapping[str, Any] | None
) -> dict[str, Any]:
    m = rec.get("measures", {})
    sweeps = list(rec.get("sweeps", []))
    return dict(
        run=rid.name,
        phantom=rid.phantom,
        source=rid.source,
        noise=rid.noise,
        grid=rid.grid,
        n_nodes=_g(m, "cost", "n_nodes") or _g(rec, "grid", "n_nodes"),
        n_cells=_g(rec, "grid", "n_cells"),
        stops=_g(m, "stops", "abs_mm"),
        paths_mean=_g(m, "paths", "mean_mm"),
        levels=_g(m, "levels", "abs_mm"),
        topology=m.get("topology"),
        sweeps=sweeps,
        sweeps_total=int(sum(sweeps)),
        time_s=rec.get("time_s"),
        peak_mem_bytes=rec.get("peak_mem_bytes"),
        adapt=None if adapt is None else adapt_summary(adapt),
    )


def adapt_summary(adapt: Mapping[str, Any]) -> dict[str, Any]:
    """Nodes and leaves per level of the final octree, the prior-shaped count, and per
    round whether the refinement and the coarsening trial were accepted."""
    lc = _g(adapt, "final", "level_counts") or {}
    rounds = []
    for r in adapt.get("rounds", []):
        tr = {t.get("kind"): t for t in r.get("trials", [])}
        rounds.append(
            dict(
                round=r.get("round"),
                n_refine_nominated=r.get("n_refine_nominated"),
                n_coarsen_nominated=r.get("n_coarsen_nominated"),
                refine_accepted=None
                if "refine" not in tr
                else bool(tr["refine"].get("accepted")),
                coarsen_accepted=None
                if "coarsen" not in tr
                else bool(tr["coarsen"].get("accepted")),
                delta_free_energy_refine=_g(tr, "refine", "delta_free_energy"),
                delta_free_energy_coarsen=_g(tr, "coarsen", "delta_free_energy"),
            )
        )
    return dict(
        n_nodes=lc.get("n_nodes"),
        n_cells=lc.get("n_cells"),
        nodes_per_level=lc.get("nodes_per_level"),
        leaves_per_level=lc.get("leaves_per_level"),
        cell_mm_per_level=lc.get("cell_mm_per_level"),
        n_prior_shaped=_g(adapt, "final", "prior_shaped", "count"),
        n_rounds=len(rounds),
        n_refine_accepted=sum(1 for r in rounds if r["refine_accepted"]),
        n_coarsen_accepted=sum(1 for r in rounds if r["coarsen_accepted"]),
        rounds=rounds,
        hanging=hanging_row(adapt),
        inputs=adapt.get("inputs"),
    )


def figure_grid(phantom: str, rows: Sequence[Mapping[str, Any]], path: Path) -> int:
    uni = sorted((r for r in rows if r["grid"] != "octree"), key=lambda r: r["n_nodes"])
    octr = [r for r in rows if r["grid"] == "octree"]
    panels: list[tuple[str, Callable[[Any], Any], Callable[[Any], Any] | None]] = [
        ("stop error |mm|", lambda r: _g(r, "stops", "median"), lambda r: _g(r, "stops", "p90")),
        ("path error (mean, mm)", lambda r: _g(r, "paths_mean", "median"), lambda r: _g(r, "paths_mean", "p90")),
        ("level deviation |mm|", lambda r: _g(r, "levels", "median"), lambda r: _g(r, "levels", "p90")),
        ("sweeps (all widths)", lambda r: r["sweeps_total"], None),
        ("time (s)", lambda r: r["time_s"], None),
        ("peak memory (GB)", lambda r: None if r["peak_mem_bytes"] in (None, -1) else r["peak_mem_bytes"] / 2**30, None),
    ]  # fmt: skip
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for a, (ttl, f50, f90) in zip(axs.ravel(), panels, strict=True):
        for grp, mk in ((uni, "o"), (octr, "*")):
            if not grp:
                continue
            x = [r["n_nodes"] for r in grp]
            y50 = [np.nan if f50(r) is None else f50(r) for r in grp]
            a.plot(x, y50, mk + ("-" if mk == "o" else ""), color="tab:blue",
                   ms=6 if mk == "o" else 14, label=("median" if f90 else ttl) + ("" if mk == "o" else " (octree)"))  # fmt: skip
            if f90:
                y90 = [np.nan if f90(r) is None else f90(r) for r in grp]
                a.plot(x, y90, mk + ("--" if mk == "o" else ""), color="tab:red",
                       ms=6 if mk == "o" else 14, label="p90" + ("" if mk == "o" else " (octree)"))  # fmt: skip
        a.set_xscale("log")
        a.set_xlabel("nodes")
        a.set_title(ttl, fontsize=9)
        a.legend(fontsize=6)
    fig.suptitle(f"{phantom}: measures against node count (star: octree)", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def figure_grid_overview(
    by_phantom: Mapping[str, Sequence[Mapping[str, Any]]], path: Path
) -> int:
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    cm = plt.get_cmap("tab20")
    for a, (ttl, key) in zip(
        axs,
        (
            ("stops p90 (mm)", "stops"),
            ("paths mean p90 (mm)", "paths_mean"),
            ("levels p90 (mm)", "levels"),
        ),
        strict=True,
    ):
        for i, (ph, rows) in enumerate(sorted(by_phantom.items())):
            uni = sorted(
                (r for r in rows if r["grid"] != "octree"), key=lambda r: r["n_nodes"]
            )
            if uni:
                a.plot([r["n_nodes"] for r in uni], [np.nan if _g(r, key, "p90") is None else _g(r, key, "p90") for r in uni],
                       "o-", color=cm(i % 20), ms=4, lw=1, label=ph)  # fmt: skip
            for r in rows:
                if r["grid"] == "octree" and _g(r, key, "p90") is not None:
                    a.plot(
                        r["n_nodes"], _g(r, key, "p90"), "*", color=cm(i % 20), ms=13
                    )
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("nodes")
        a.set_title(ttl, fontsize=9)
    axs[0].legend(fontsize=6, ncol=2)
    fig.suptitle(
        "p90 against node count, every phantom (line: uniform grids; star: octree)",
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, path)


def grid_tables(rows: Sequence[Mapping[str, Any]], skipped: Sequence[RunId]) -> str:
    head = [
        "run", "nodes", "stops med/p90", "paths mean med/p90", "levels med/p90",
        "focal / overlap / crossings / sinks / unresolved", "sweeps", "time (s)", "peak (GB)",
    ]  # fmt: skip
    body = []
    for r in rows:
        t = r["topology"] or {}
        pk = r["peak_mem_bytes"]
        body.append([
            r["run"], _n(r["n_nodes"]),
            f"{_n(_g(r, 'stops', 'median'))} / {_n(_g(r, 'stops', 'p90'))}",
            f"{_n(_g(r, 'paths_mean', 'median'))} / {_n(_g(r, 'paths_mean', 'p90'))}",
            f"{_n(_g(r, 'levels', 'median'))} / {_n(_g(r, 'levels', 'p90'))}",
            f"{_n(t.get('focal_points'))} / {_n(t.get('overlap'))} / {_n(t.get('solved_stop_crossings'))} / "
            f"{_n(t.get('sink_nodes'))} / {_n(t.get('unresolved_cells'))}",
            "+".join(str(s) for s in r["sweeps"]) or "–", _n(r["time_s"], 4),
            "–" if pk in (None, -1) else _n(pk / 2**30, 3),
        ])  # fmt: skip
    return "\n".join([
        "# PR 6 the spacing scan (task 6.3, question 7)\n",
        "Source oracle. Distributions are over the columns of the run (run.json measures). "
        "Levels are the level deviations; paths the mean distance to the true column.\n",
        md_table(head, body) if body else "No run found.\n", "",
        skipped_md(skipped),
    ])  # fmt: skip


def octree_tables(entries: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    head = [
        "run", "rounds", "refinements accepted", "coarsenings accepted", "nodes", "nodes per level",
        "leaves per level", "prior-shaped leaves",
    ]  # fmt: skip
    body, rd = [], []
    for name, ad in entries:
        body.append([
            name, _n(ad["n_rounds"]), _n(ad["n_refine_accepted"]), _n(ad["n_coarsen_accepted"]),
            _n(ad["n_nodes"]), json.dumps(ad["nodes_per_level"]), json.dumps(ad["leaves_per_level"]),
            _n(ad["n_prior_shaped"]),
        ])  # fmt: skip
        for r in ad["rounds"]:
            rd.append([
                name, _n(r["round"]), _n(r["n_refine_nominated"]), str(r["refine_accepted"]),
                _n(r["delta_free_energy_refine"], 4), _n(r["n_coarsen_nominated"]),
                str(r["coarsen_accepted"]), _n(r["delta_free_energy_coarsen"], 4),
            ])  # fmt: skip
    return "\n".join([
        "# PR 6 the octrees: levels, prior-shaped cells and rounds accepted\n",
        "For slab and noise (R9) a refinement accepted is a failure signature of the "
        "free-energy test.\n",
        md_table(head, body) if body else "No adapt.json found.\n", "",
        "## Rounds (delta F: the trial's free energy minus the current one; accepted when negative)\n",
        md_table(["run", "round", "refine nominated", "refine accepted", "delta F", "coarsen nominated",
                  "coarsen accepted", "delta F"], rd) if rd else "",  # fmt: skip
    ])  # fmt: skip


# ---------------------------------------------------------------- orchestration


def write_table(tables: Path, stem: str, md: str, doc: Mapping[str, Any]) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    (tables / f"{stem}.md").write_text(md if md.endswith("\n") else md + "\n")
    (tables / f"{stem}.json").write_text(json.dumps(FK._jsonable(doc), indent=1) + "\n")
    print(f"wrote {tables / (stem + '.md')} and .json", flush=True)


def _skipped_json(skipped: Sequence[RunId]) -> list[str]:
    return [f"{r.phantom}/{r.dirname}" for r in skipped]


def run_hairpins(a: argparse.Namespace, cache: Cache) -> dict[str, Any]:
    ids = expected_ids(HAIRPIN_PHANTOMS, HAIRPIN_GRIDS, SOURCES, a.noises)
    found, skipped = find_runs(a.runs, ids)
    rows: list[dict[str, Any]] = []
    figs: dict[str, int] = {}
    for rid in ids:
        if rid not in found:
            continue
        print(f"hairpins: {rid.name}", flush=True)
        case = load_case(rid, found[rid], cache)
        row = hairpin_run(case)
        if row["fine_cells"] is not None:
            f = a.figures / f"pr6_hairpin_map_{rid.slug}.png"
            row["figure"] = f.name
            figs[f.name] = figure_hairpin_map(case, row, f)
        rows.append(row)
        del case
    md = hairpin_tables(rows, skipped)
    write_table(a.tables, "pr6_hairpins", md,
                dict(runs=rows, skipped=_skipped_json(skipped), figures=figs,
                     constants=dict(BANK_TOL_MM=BANK_TOL_MM, SECTION_SLAB_MM=SECTION_SLAB_MM)))  # fmt: skip
    return dict(rows=rows, figures=figs)


def run_jumps(a: argparse.Namespace, cache: Cache) -> dict[str, Any]:
    ids = expected_ids([JUMP_PHANTOM], JUMP_GRIDS, SOURCES, a.noises)
    found, skipped = find_runs(a.runs, ids)
    results: list[dict[str, Any]] = []
    items: list[tuple[Case, Mapping[str, Any]]] = []
    figs: dict[str, int] = {}
    for rid in ids:
        if rid not in found:
            continue
        print(f"jumps: {rid.name}", flush=True)
        case = load_case(rid, found[rid], cache)
        res = jump_run(case)
        f = a.figures / f"pr6_jump_map_{rid.slug}.png"
        res["figure"] = f.name
        figs[f.name] = figure_jump_map(case, res, f)
        results.append(res)
        items.append((case, res))
    if items:
        f = a.figures / "pr6_jump_scatter.png"
        figs[f.name] = figure_jump_scatter(items, f)
    md = jump_tables(results, skipped)
    hmd, hjs = hanging_table(results)
    write_table(a.tables, "pr6_jumps", md,
                dict(runs=results, skipped=_skipped_json(skipped), figures=figs))  # fmt: skip
    return dict(results=results, figures=figs, hanging=hjs, hanging_md=hmd)


def run_grid(a: argparse.Namespace, cache: Cache) -> dict[str, Any]:
    phantoms = list(FTR.PHANTOMS)
    ids = expected_ids(phantoms, GRID_LABELS, ("oracle",), (a.grid_noise,))
    found: dict[RunId, Path] = {}
    skipped: list[RunId] = []
    for rid in ids:
        f, s = find_runs(a.runs, [rid], _need(rid))
        found.update(f)
        skipped += s
    rows: list[dict[str, Any]] = []
    adapts: list[tuple[str, dict[str, Any]]] = []
    for rid in ids:
        if rid not in found:
            continue
        print(f"grid: {rid.name}", flush=True)
        d = found[rid]
        ad = (
            json.loads((d / "adapt.json").read_text())
            if (d / "adapt.json").is_file()
            else None
        )
        if rid.phantom == "noise":
            if ad is not None:
                adapts.append((rid.name, adapt_summary(ad)))
            continue
        rec = json.loads((d / "run.json").read_text())
        row = grid_row(rid, rec, ad)
        rows.append(row)
        if row["adapt"] is not None:
            adapts.append((rid.name, row["adapt"]))
    for item in a.extra_adapt:
        label, _sep, p = item.partition("=")
        adapts.append(
            (f"{label} (extra)", adapt_summary(json.loads(Path(p).read_text())))
        )
    figs: dict[str, int] = {}
    by_ph: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_ph.setdefault(r["phantom"], []).append(r)
    for ph, rs in by_ph.items():
        f = a.figures / f"pr6_grid_{ph}.png"
        figs[f.name] = figure_grid(ph, rs, f)
    if by_ph:
        f = a.figures / "pr6_grid_overview.png"
        figs[f.name] = figure_grid_overview(by_ph, f)
    write_table(a.tables, "pr6_grid", grid_tables(rows, skipped),
                dict(runs=rows, skipped=_skipped_json(skipped), figures=figs))  # fmt: skip
    write_table(a.tables, "pr6_octree", octree_tables(adapts),
                dict(octrees=[dict(run=n, **ad) for n, ad in adapts]))  # fmt: skip
    return dict(rows=rows, adapts=adapts, figures=figs, skipped=skipped)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("command", choices=("all", "hairpins", "jumps", "grid"))
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--figures", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--noises", type=float, nargs="*", default=list(NOISES),
                    help="noise levels of the hairpin and jump runs")  # fmt: skip
    ap.add_argument("--grid-noise", type=float, default=1.0,
                    help="noise level of the spacing scan")  # fmt: skip
    ap.add_argument("--extra-adapt", action="append", default=[], metavar="LABEL=PATH",
                    help="an adapt.json outside the run layout (for example PR 5's)")  # fmt: skip
    a = ap.parse_args(argv)
    for item in a.extra_adapt:
        if not re.match(r"^[^=]+=.+$", item):
            ap.error("--extra-adapt wants LABEL=PATH")
    return a


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    a = parse_args(argv)
    t0 = time.perf_counter()
    a.figures.mkdir(parents=True, exist_ok=True)
    a.tables.mkdir(parents=True, exist_ok=True)
    cache = Cache(a.truth)
    out: dict[str, Any] = dict(features=None if a.features is None else str(a.features))
    if a.command in ("all", "hairpins"):
        out["hairpins"] = run_hairpins(a, cache)
    if a.command in ("all", "jumps"):
        out["jumps"] = run_jumps(a, cache)
    if a.command in ("all", "hairpins", "jumps"):
        hr = (out.get("hairpins") or {}).get("rows", []) + (out.get("jumps") or {}).get(
            "results", []
        )
        hmd, hjs = hanging_table(hr)
        write_table(a.tables, "pr6_hanging", hmd, dict(rows=hjs))
    if a.command in ("all", "grid"):
        out["grid"] = run_grid(a, cache)
    out["time_s"] = time.perf_counter() - t0
    print(f"done in {out['time_s']:.1f} s", flush=True)
    return out


if __name__ == "__main__":
    main()
