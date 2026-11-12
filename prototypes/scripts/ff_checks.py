"""The topology checks of the flow-field stage (docs/plans/flow-field-phantoms.md,
*Measures*, item *Topology*; PR 3, task 3.4).

Each check gives a count and the places (voxel positions) where it found something. They
are diagnostics, not pass/fail rules. Every tolerance is a named constant below, fixed
before any run on the phantoms.

The checks:

- **Crossings of the solved stop set** (`solved_stop_crossings`): path samples before
  the last one whose interpolated stop probability is above stop_level (0.5).
- **Crossings of the true stop set** (`true_stop_crossings`, with a Geometry): a column
  crosses if a sample before the last one is in the true stopped region (stop_sdf >
  TRUE_STOP_TOL_MM) and a later sample is back in tissue (stop_sdf < −TRUE_STOP_TOL_MM).
  A column whose samples enter the stopped region and do not come back overshoots its
  stop; it is counted apart ("overshoot"), because the *Stops* measure holds that error.
- **Overlap and focal points** (`column_meetings`): two columns meet where a sample of
  one is closer than MEET_TOL_MM to a sample of the other. The samples within
  STOP_EXCLUDE_MM of arc length before each column's end are left out, because a contact
  is a shared stop set: columns from the two banks end on the same sheet and must not
  count as overlap. A meeting of two columns whose roots are more than ROOT_FAR_MM apart
  is an overlap; else it is a focal point. With a Geometry, the counts hold only the
  meetings in the cortex (label CORTEX); the counts at all places are also given.
- **Sinks** (`field_sinks`): moving nodes whose mean direction has resultant length
  below SINK_RESULTANT; grid edges between two moving nodes whose mean directions point
  against each other (d_i · d_j < REVERSE_COS), split into converging (they point at
  each other), diverging (they point away from each other) and other; and the columns
  that ended with SINK.
- **Unresolved contact cells**: the cells of `ff_integrate.unresolved_cells`, and the
  cells that ended a column (the integrator's info).

Limits (not resolved in this PR):

- A fused stopped sheet thinner than 2 · TRUE_STOP_TOL_MM has no sample with stop_sdf
  above the tolerance, so a column that passes through it is not found by
  `true_stop_crossings`. The connected kiss of PR 4 needs a contact check of its own
  (for example, by the bank of the samples on the two sides of the sheet).
- Two columns that pass each other closer than MEET_TOL_MM without a crossing also meet.
- Two columns whose start points are closer than MEET_TOL_MM meet at their starts. The
  meetings are a useful count only when the start points are much farther apart (the
  roots of the truth are at least 0.32 mm apart).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import ff_integrate as FI  # noqa: E402
import ff_types as T  # noqa: E402

TRUE_STOP_TOL_MM = 0.02
"""A sample is in the true stopped region if stop_sdf > this, and back in tissue if
stop_sdf < −this (mm). 1/8 of a voxel: above the 1e-3 mm accuracy of stop_sdf and two
path steps, so that a sample on the stop surface (a correct stop) is not counted."""

MEET_TOL_MM = 0.0075
"""Two columns meet where two of their samples are closer than this (mm). Three quarters
of the 0.01 mm path step: two sampled lines that cross in a plane have samples closer
than about 0.71 step (the half diagonal of a step), and two samples of one column (one
step apart) are never this close, except in the shortened last step, which
STOP_EXCLUDE_MM leaves out."""

STOP_EXCLUDE_MM = 0.08
"""The samples within this arc length (mm) before a column's end do not enter the
meetings (half a voxel). Columns that end on a shared stop set (a contact, or the edge
of a hairpin) come close to each other only near their ends."""

ROOT_FAR_MM = 1.0
"""Two meeting columns whose roots are farther apart than this (mm) are an overlap;
closer, a focal point. This only labels a meeting; the sum of the two counts does not
depend on it."""

SINK_RESULTANT = FI.SINK_RESULTANT
"""A moving node whose mean direction has resultant length |Σ_k q_k d_k| / Σ_k q_k below
this is a sink (the integrator's value)."""

REVERSE_COS = 0.0
"""Two adjacent moving nodes reverse if d_i · d_j < this (an angle above 90°: the
integrator's turn rule)."""


def _labels_at(geometry: T.Geometry, X: T.Array) -> T.IntArray:
    return np.asarray(geometry.label(geometry.to_local(X)), np.int64)


def _samples(cols: T.Columns, last: bool) -> tuple[T.IntArray, T.IntArray]:
    """(column, index) of every path sample; without the last sample if not `last`."""
    k = np.arange(cols.paths.shape[1])[None, :]
    n = cols.counts[:, None] - (0 if last else 1)
    ci, si = np.nonzero(k < n)
    return np.asarray(ci, np.int64), np.asarray(si, np.int64)


def _first_per_column(ci: T.IntArray, si: T.IntArray) -> tuple[T.IntArray, T.IntArray]:
    """The first (column, sample) of each column in the list (by sample index)."""
    if len(ci) == 0:
        return ci, si
    o = np.lexsort((si, ci))
    ci, si = ci[o], si[o]
    first = np.r_[True, ci[1:] != ci[:-1]]
    return ci[first], si[first]


def solved_stop_crossings(
    field: T.Field, cols: T.Columns, settings: T.SolverSettings = T.DEFAULTS
) -> dict[str, Any]:
    """Path samples before the last one whose stop probability (the trilinear
    interpolation of q_stop) is above stop_level. Returns count (the columns with such a
    sample), columns, n_samples, places (the first such sample of each column)."""
    ci, si = _samples(cols, last=False)
    X = cols.paths[ci, si]
    sp = FI.masked_sample(field, X, settings)[2] if len(X) else np.zeros(0)
    hit = np.asarray(sp > settings.stop_level)
    fc, fs = _first_per_column(ci[hit], si[hit])
    return dict(
        count=len(fc), columns=fc, n_samples=int(hit.sum()), places=cols.paths[fc, fs]
    )


def true_stop_crossings(geometry: T.Geometry, cols: T.Columns) -> dict[str, Any]:
    """Crossings of the true stop set, and overshoots (the module docstring). Returns
    count, columns, places (the first sample in the stopped region of each crossing
    column), overshoot_count, overshoot_columns, overshoot_places."""
    ci, si = _samples(cols, last=False)
    X = cols.paths[ci, si]
    sdf = np.asarray(geometry.stop_sdf(geometry.to_local(X))) if len(X) else np.zeros(0)
    into = sdf > TRUE_STOP_TOL_MM
    fc, fs = _first_per_column(ci[into], si[into])
    # a later sample (the last one included) back in tissue
    ca, sa = _samples(cols, last=True)
    Xa = cols.paths[ca, sa]
    sda = (
        np.asarray(geometry.stop_sdf(geometry.to_local(Xa))) if len(Xa) else np.zeros(0)
    )
    back = sda < -TRUE_STOP_TOL_MM
    last_back = np.full(len(cols.counts), -1, np.int64)
    np.maximum.at(last_back, ca[back], sa[back])
    cross = last_back[fc] > fs
    return dict(
        count=int(cross.sum()),
        columns=fc[cross],
        places=cols.paths[fc[cross], fs[cross]],
        overshoot_count=int((~cross).sum()),
        overshoot_columns=fc[~cross],
        overshoot_places=cols.paths[fc[~cross], fs[~cross]],
    )


def column_meetings(
    cols: T.Columns, geometry: T.Geometry | None = None
) -> dict[str, Any]:
    """Overlaps and focal points (the module docstring). Each meeting is one pair of
    columns (i < j); its place is the midpoint of the closest pair of their samples.
    Returns a dict with "overlap" and "focal" (each: count, pairs (k, 2), places (k, 3),
    and with a Geometry count_all, the count at all places) and "root_far_mm"."""
    ci, si = _samples(cols, last=True)
    P = cols.paths
    seg = np.zeros(P.shape[:2])
    if P.shape[1] > 1:
        d = np.diff(P, axis=1)
        seg[:, 1:] = np.nan_to_num(np.sqrt((d * d).sum(-1))) * T.VOXEL_MM
    arc = np.cumsum(seg, axis=1)
    to_end = (
        arc[np.arange(len(cols.counts)), np.maximum(cols.counts - 1, 0)][:, None] - arc
    )
    keep = to_end[ci, si] >= STOP_EXCLUDE_MM
    ci, si = ci[keep], si[keep]
    X = P[ci, si]
    empty = dict(count=0, pairs=np.zeros((0, 2), np.int64), places=np.zeros((0, 3)))
    if len(X) < 2:
        out0: dict[str, Any] = dict(
            overlap=dict(empty), focal=dict(empty), root_far_mm=ROOT_FAR_MM
        )
        if geometry is not None:
            out0["overlap"]["count_all"] = out0["focal"]["count_all"] = 0
        return out0
    pr = cKDTree(X).query_pairs(MEET_TOL_MM / T.VOXEL_MM, output_type="ndarray")
    a, b = pr[:, 0], pr[:, 1]
    other = ci[a] != ci[b]
    a, b = a[other], b[other]
    swap = ci[a] > ci[b]
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    dist = np.sqrt(((X[a] - X[b]) ** 2).sum(-1))
    o = np.lexsort((dist, ci[b], ci[a]))
    a, b = a[o], b[o]
    first = (
        np.r_[True, (ci[a][1:] != ci[a][:-1]) | (ci[b][1:] != ci[b][:-1])]
        if len(a)
        else (np.zeros(0, bool))
    )
    a, b = a[first], b[first]
    pairs = np.stack([ci[a], ci[b]], axis=1) if len(a) else np.zeros((0, 2), np.int64)
    places = 0.5 * (X[a] + X[b])
    rd = (
        np.sqrt(((cols.roots[pairs[:, 0]] - cols.roots[pairs[:, 1]]) ** 2).sum(-1))
        * T.VOXEL_MM
    )
    far = rd > ROOT_FAR_MM
    if geometry is not None and len(places):
        ctx = _labels_at(geometry, places) == T.CORTEX
    else:
        ctx = np.ones(len(places), bool)
    out: dict[str, Any] = dict(root_far_mm=ROOT_FAR_MM)
    for name, sel in (("overlap", far), ("focal", ~far)):
        m = sel & ctx
        r: dict[str, Any] = dict(count=int(m.sum()), pairs=pairs[m], places=places[m])
        if geometry is not None:
            r["count_all"] = int(sel.sum())
        out[name] = r
    return out


def field_sinks(
    field: T.Field, settings: T.SolverSettings = T.DEFAULTS
) -> dict[str, Any]:
    """The sinks of the field (the module docstring). Returns low_resultant (count,
    nodes, places) and reversed_edges (count, edges (k, 2), places (edge midpoints),
    converging, diverging)."""
    mov = field.q_stop < settings.stop_level
    qm = field.q.sum(1)
    R = np.einsum("nk,nkc->nc", field.q, field.direction)
    res = np.sqrt((R * R).sum(1)) / np.where(qm > 0, qm, 1.0)
    low = np.flatnonzero(mov & (res < SINK_RESULTANT))
    X = field.grid.nodes
    d = field.direction_mean()
    E = np.asarray(field.grid.edges, np.int64).reshape(-1, 2)
    i, j = E[:, 0], E[:, 1]
    rev = mov[i] & mov[j] & ((d[i] * d[j]).sum(1) < REVERSE_COS)
    Er = E[rev]
    i, j = Er[:, 0], Er[:, 1]
    dx = X[j] - X[i]
    ai = (d[i] * dx).sum(1)  # > 0: i points toward j
    aj = -(d[j] * dx).sum(1)  # > 0: j points toward i
    return dict(
        low_resultant=dict(count=len(low), nodes=low, places=X[low]),
        reversed_edges=dict(
            count=len(Er),
            edges=Er,
            places=0.5 * (X[i] + X[j]),
            converging=int(((ai > 0) & (aj > 0)).sum()),
            diverging=int(((ai < 0) & (aj < 0)).sum()),
        ),
    )


def topology(
    field: T.Field,
    cols: T.Columns,
    info: dict[str, Any] | None = None,
    geometry: T.Geometry | None = None,
    settings: T.SolverSettings = T.DEFAULTS,
) -> dict[str, Any]:
    """Every check of *Topology*: a dict with solved_stop_crossings, true_stop_crossings
    (None without a Geometry), meetings (overlap and focal), field_sinks, sink_columns
    (count, columns, places: the ends of the columns that ended with SINK) and
    unresolved_cells (count and centres of the field's unresolved contact cells; with
    the integrator's info, also met: the cells that ended a column)."""
    sink = np.flatnonzero(cols.end == int(T.EndReason.SINK))
    un = np.flatnonzero(FI.unresolved_cells(field, settings))
    g = field.grid
    centres = g.cell_origin[un] + 0.5 * (g.cell_size_mm[un] / T.VOXEL_MM)[:, None]
    ur: dict[str, Any] = dict(count=len(un), cells=un, places=centres)
    if info is not None:
        ur["met"] = list(info.get("unresolved_cells_met", []))
        ur["columns_ended"] = int(info.get("ends", {}).get("UNRESOLVED_CONTACT", 0))
    return dict(
        solved_stop_crossings=solved_stop_crossings(field, cols, settings),
        true_stop_crossings=None
        if geometry is None
        else true_stop_crossings(geometry, cols),
        meetings=column_meetings(cols, geometry),
        field_sinks=field_sinks(field, settings),
        sink_columns=dict(count=len(sink), columns=sink, places=cols.stops[sink]),
        unresolved_cells=ur,
    )


def counts(report: dict[str, Any]) -> dict[str, int]:
    """The counts of a `topology` report, for run.json."""
    out = dict(
        solved_stop_crossings=report["solved_stop_crossings"]["count"],
        overlap=report["meetings"]["overlap"]["count"],
        focal_points=report["meetings"]["focal"]["count"],
        sink_nodes=report["field_sinks"]["low_resultant"]["count"],
        reversed_edges=report["field_sinks"]["reversed_edges"]["count"],
        sink_columns=report["sink_columns"]["count"],
        unresolved_cells=report["unresolved_cells"]["count"],
    )
    if report["true_stop_crossings"] is not None:
        out["true_stop_crossings"] = report["true_stop_crossings"]["count"]
        out["overshoots"] = report["true_stop_crossings"]["overshoot_count"]
    return {k: int(v) for k, v in out.items()}
