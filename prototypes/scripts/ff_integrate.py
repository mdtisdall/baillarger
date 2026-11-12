"""The integrator of the flow-field stage (docs/plans/flow-field-phantoms.md, *The
integrator*; PR 3, task 3.4).

The integrator traces one column from each start point through a solved
`ff_types.Field`. It uses only the `ff_types.Grid` protocol, so the octree of PR 5 can
use it without change.

The rules
---------
- **Moving node.** A node is moving if q_stop < stop_level (0.5).
- **Masked interpolation** (`masked_sample`). In a cell, only the moving corners enter
  the interpolation of the direction and of τ. Their trilinear weights are divided by
  the sum of the weights of the moving corners. The stop probability is the trilinear
  interpolation of q_stop over all 8 corners.
- **Unresolved contact cell** (`unresolved_cells`). A cell with two moving corners a and
  b whose mean directions converge, d_a · (x_b − x_a) > 0 and d_b · (x_a − x_b) > 0, and
  point against each other, d_a · d_b < CONTACT_COS. The integrator ends a column before
  any of its evaluation points enters such a cell (EndReason.UNRESOLVED_CONTACT), and it
  records the cell. The second condition (d_a · d_b < CONTACT_COS) is not in the PR 3
  spec. Reason: in a flow that converges on a focus line beyond the cell (the concave
  side of a fold, for example the floor under a fundus), two corners on the two sides of
  the flow's plane of symmetry converge by a small angle. So without the second
  condition, the band of cells along that plane is unresolved, at any distance from the
  focus. `test_ff_integrate.py` shows this on a radial field (16 of 64 cells).
  `unresolved_cells(field, max_cos=2.0)` gives the spec's rule.
- **RK4** on the unit masked direction, step rk4_step_mm, from each start point. The
  positions are in voxels, so one step is rk4_step_mm / VOXEL_MM voxels.
- **Evaluation points.** Each RK4 step evaluates the field at P2 = Y + h/2 k1, P3 = Y +
  h/2 k2, P4 = Y + h k3 and at the step's end Yn. A point is bad if (in this order of
  priority): it is outside the grid (LEFT_BOX); its stop probability is ≥ stop_level (a
  stop); its cell is an unresolved contact cell (UNRESOLVED_CONTACT); the resultant
  length of its masked direction is below SINK_RESULTANT (SINK). The step also ends with
  SINK if the unit direction at Yn turns by more than 90° from the direction at Y. The
  first bad point in the order P2, P3, P4, Yn gives the end reason.
- **The stop.** If the first bad point is a stop, the step crosses the stop set. The
  integrator then bisects the step length s in [0, h], stop_bisections (6) times. A
  sub-step of length s is good if none of its own evaluation points and its end is bad.
  The stop is the end of the longest good sub-step: it lies on the moving side of the
  0.5 crossing, within h / 2⁶ of it along the path. This is the bisection of
  `ff_truth._last_step` with a fixed count, as the plan asks (`bent_phantom._last_step`
  uses Newton steps on φ, which a field does not have).
- **Other ends.** LEFT_BOX, UNRESOLVED_CONTACT and SINK end the column at the last
  accepted point Y (no new point). MAX_LENGTH ends it after floor(max_column_mm /
  rk4_step_mm) steps. A start point that is bad ends the column at the start (one
  point).
- **τ.** tau_path is the masked interpolation of τ_mean at each path point. tau_arrival
  is τ at the last point of each column. jump = max(0, −tau_arrival) for a STOPPED
  column, nan for the other ends. p_jump, stop_sd and level_sd are nan unless the caller
  gives them (the sampled fields are PR 6).

Determinism
-----------
Every operation on the columns is an elementwise NumPy ufunc or an integer gather. The
sums over the 8 corners and over the 3 axes are written as additions in a fixed order
(no `np.sum`, `einsum` or `linalg.norm`, whose reductions can group terms in a different
way). So the result for one column does not depend on the other columns in the batch.
The columns are also sorted (np.lexsort of the roots and the start points) before the
loop and put back in the caller's order after it. So the same input gives the same
columns to the last bit on the CPU, in any order of the roots. This needs `grid.locate`
to be elementwise too (the uniform grid's floor is).

NumPy or Torch (the plan's *NumPy or Torch*)
-------------------------------------------
The kernel: one evaluation of the field at a point is `grid.locate` (about 15
operations), 8 trilinear weights (about 30), a gather of 8 corners × 6 values (q_stop,
moving, 3 direction, τ: 48 float64, 384 B, plus 64 B of corner ids) and about 100
multiply-adds. One RK4 step needs 4 evaluations (k1 is the previous step's end): about
600 operations and 1.8 KB of gathers per column and step. In NumPy each operation is a
pass over the active columns, so the traffic of the temporaries (about 600 passes of
8–24 B per column) is about 10 KB per column and step.

- Phantom size: 10³–10⁴ roots × up to 1200 steps = 1.2·10⁶–1.2·10⁷ column-steps: about
  10⁹–10¹⁰ operations and 10–100 GB of temporary traffic, most of it in cache (one
  active array of 10⁴ float64 is 80 KB). The Python loop is over the steps (at most 1200
  iterations, plus 6 bisections at the stop), vectorised over the active columns.
- Target size (`column-solver.md`): 4·10⁶–4·10⁷ grid nodes and about 3.5·10⁸ columns (50
  µm whole brain), 800–1200 steps each: about 3·10¹¹–4·10¹¹ column-steps, 2·10¹⁴
  operations and 7·10¹⁴ B of gathers. These are estimates from the counts above, not
  measurements. The paths at every step (3·10¹¹ × 12 B in float32, about 4 TB) cannot be
  stored at that size: a target run must keep only the τ levels, by superblock.
- The work is regular and data-parallel: every column does the same operations, with no
  scatter and no reduction across columns. The only irregularity is that columns end at
  different steps; the loop compacts the active set at each step.

The choice: NumPy, for this PR. At the target size a GPU would give a large gain (the
kernel is gather-bound and data-parallel). But every evaluation calls `grid.locate`,
which the Grid protocol defines on NumPy arrays; a Torch integrator would need a Torch
`locate` in the grid (ff_grid.py now, the octree in PR 5), which is outside this task.
With a NumPy `locate`, each evaluation would copy the points to the host and back. The
kernel uses only elementwise operations and gathers (no `index_add_`, no
`scatter_add_`), so a port to Torch is the same algorithm once the grid gives a Torch
`locate`."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402

SINK_RESULTANT = 0.5
"""A point with a masked direction whose resultant length is below this is a sink (the
spec's value). The resultant length is |Σ w_c d_c| / Σ w_c over the moving corners, with
d_c the unit mean directions of the nodes."""

CONTACT_COS = 0.0
"""Two converging moving corners of a cell make it an unresolved contact cell only if
their directions point against each other: d_a · d_b < CONTACT_COS (an angle above 90°).
The same angle as the integrator's turn rule. Fixed before any run on the phantoms."""

# the codes of a bad evaluation point; the order is the priority
_GOOD, _OUTSIDE, _STOP, _UNRESOLVED, _SINK = 0, 1, 2, 3, 4
_REASON = {
    _OUTSIDE: T.EndReason.LEFT_BOX,
    _STOP: T.EndReason.STOPPED,
    _UNRESOLVED: T.EndReason.UNRESOLVED_CONTACT,
    _SINK: T.EndReason.SINK,
}

# the corner pairs of a cell (a < b), for unresolved_cells
_PAIRS = [(a, b) for a in range(8) for b in range(a + 1, 8)]


@dataclass(frozen=True)
class _Nodes:
    """The node values that the integrator reads, computed once per field."""

    grid: T.Grid
    cells: T.IntArray  # (C, 8)
    q_stop: T.Array  # (n,)
    moving: T.Array  # (n,) 1.0 for a moving node, 0.0 else
    direction: T.Array  # (n, 3) unit mean direction, 0 for a node that is not moving
    tau: T.Array  # (n,) τ_mean, 0 for a node that is not moving
    unresolved: T.BoolArray  # (C,)


def _nodes(field: T.Field, settings: T.SolverSettings, max_cos: float) -> _Nodes:
    mov = field.q_stop < settings.stop_level
    d = np.where(mov[:, None], field.direction_mean(), 0.0)
    tau = np.where(mov, np.nan_to_num(field.tau_mean()), 0.0)
    return _Nodes(
        grid=field.grid,
        cells=np.asarray(field.grid.cells, np.int64),
        q_stop=np.asarray(field.q_stop, float),
        moving=mov.astype(float),
        direction=np.ascontiguousarray(d, float),
        tau=np.asarray(tau, float),
        unresolved=_unresolved(field, settings, max_cos),
    )


def _unresolved(
    field: T.Field, settings: T.SolverSettings, max_cos: float
) -> T.BoolArray:
    mov = field.q_stop < settings.stop_level
    d = field.direction_mean()
    X = field.grid.nodes
    cells = np.asarray(field.grid.cells, np.int64)
    out = np.zeros(len(cells), bool)
    for a, b in _PAIRS:
        ia, ib = cells[:, a], cells[:, b]
        dx = X[ib] - X[ia]
        da, db = d[ia], d[ib]
        conv = (_dot(da, dx) > 0) & (_dot(db, -dx) > 0) & (_dot(da, db) < max_cos)
        out |= mov[ia] & mov[ib] & conv
    return out


def unresolved_cells(
    field: T.Field,
    settings: T.SolverSettings = T.DEFAULTS,
    max_cos: float = CONTACT_COS,
) -> T.BoolArray:
    """(C,) True for each unresolved contact cell of the field (the module docstring).
    With max_cos > 1, the rule is the spec's rule (convergence only)."""
    return _unresolved(field, settings, max_cos)


def _dot(a: T.Array, b: T.Array) -> T.Array:
    """Row dot products of (N, 3) arrays, in a fixed order of additions."""
    return np.asarray(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2])


def _sample(
    nd: _Nodes, X: T.Array
) -> tuple[T.Array, T.Array, T.Array, T.IntArray, T.BoolArray]:
    """The masked sample at X (N, 3). Returns (resultant direction (N, 3), τ (N,), stop
    probability (N,), cell (N,), unresolved (N,)). nan and cell −1 outside the grid.
    Where no moving corner has weight, the direction is 0 and τ is nan."""
    n = len(X)
    cell, loc = nd.grid.locate(X)
    cell = np.asarray(cell, np.int64)
    inside = cell >= 0
    c = np.where(inside, cell, 0)
    corners = nd.cells[c]
    W = T.trilinear_weights(np.where(inside[:, None], loc, 0.0))
    sp, ws, tau = np.zeros(n), np.zeros(n), np.zeros(n)
    dx, dy, dz = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(8):  # a fixed order of additions
        ids = corners[:, k]
        w = W[:, k]
        sp = sp + w * nd.q_stop[ids]
        wm = w * nd.moving[ids]
        ws = ws + wm
        D = nd.direction[ids]
        dx = dx + wm * D[:, 0]
        dy = dy + wm * D[:, 1]
        dz = dz + wm * D[:, 2]
        tau = tau + wm * nd.tau[ids]
    has = ws > 0
    den = np.where(has, ws, 1.0)
    D = np.stack([dx / den, dy / den, dz / den], axis=1)
    D = np.where(has[:, None], D, 0.0)
    tau = np.where(has, tau / den, np.nan)
    D = np.where(inside[:, None], D, np.nan)
    tau = np.where(inside, tau, np.nan)
    sp = np.where(inside, sp, np.nan)
    unres = inside & nd.unresolved[c]
    return D, tau, sp, np.where(inside, cell, -1), unres


def masked_sample(
    field: T.Field,
    X: T.Array,
    settings: T.SolverSettings = T.DEFAULTS,
    max_cos: float = CONTACT_COS,
) -> tuple[T.Array, T.Array, T.Array, T.IntArray, T.BoolArray]:
    """The masked interpolation of the field at voxel positions X (N, 3).

    Returns (direction (N, 3), τ (N,), stop probability (N,), cell (N,), unresolved
    (N,)):

    - direction: Σ w_c d_c / Σ w_c over the moving corners c of the point's cell, with
      d_c the node's unit mean direction. It is not normalised: its length is the
      resultant length. Zero where no moving corner has weight.
    - τ: the same masked interpolation of τ_mean. nan where no moving corner has weight.
    - stop probability: the trilinear interpolation of q_stop over all 8 corners.
    - cell: the cell of the point (−1 outside the grid; then the values are nan).
    - unresolved: the cell is an unresolved contact cell (`unresolved_cells`)."""
    nd = _nodes(field, settings, max_cos)
    return _sample(nd, np.asarray(X, float).reshape(-1, 3))


npt_int8 = np.ndarray[Any, np.dtype[np.int8]]


@dataclass
class _Eval:
    """The field at one set of evaluation points."""

    unit: T.Array  # (N, 3) unit direction (finite even at a bad point)
    tau: T.Array
    bad: npt_int8  # (N,) the code of the bad point, _GOOD if good
    cell: T.IntArray


def _evaluate(nd: _Nodes, X: T.Array, level: float, fallback: T.Array) -> _Eval:
    """The field at X, with the code of each bad point. `fallback` (N, 3) replaces the
    unit direction at a bad point, so that later evaluation points stay finite."""
    D, tau, sp, cell, unres = _sample(nd, X)
    Dn = np.nan_to_num(D)
    r = np.sqrt(Dn[:, 0] * Dn[:, 0] + Dn[:, 1] * Dn[:, 1] + Dn[:, 2] * Dn[:, 2])
    bad = np.full(len(X), _GOOD, np.int8)
    bad = np.where(r < SINK_RESULTANT, _SINK, bad)
    bad = np.where(unres, _UNRESOLVED, bad)
    bad = np.where(sp >= level, _STOP, bad)
    bad = np.where(cell < 0, _OUTSIDE, bad).astype(np.int8)
    ok = bad == _GOOD
    unit = Dn / np.where(ok, r, 1.0)[:, None]
    unit = np.where(ok[:, None], unit, fallback)
    return _Eval(unit=unit, tau=tau, bad=bad, cell=cell)


def _first_bad(*codes: npt_int8) -> npt_int8:
    """The first code that is not _GOOD, in the order given."""
    out = np.full(len(codes[0]), _GOOD, np.int8)
    for c in reversed(codes):
        out = np.where(c != _GOOD, c, out).astype(np.int8)
    return out


def _rk4_step(
    nd: _Nodes, Y: T.Array, k1: T.Array, h: float | T.Array, level: float
) -> tuple[T.Array, _Eval, npt_int8, T.IntArray]:
    """One RK4 step of length h (voxels; a scalar or (N,)) from Y with the unit
    direction k1 at Y. Returns (the end Yn, the field at Yn, the first bad code of the
    step, the cell of the first bad point)."""
    hh = np.asarray(h, float)
    hc = hh[:, None] if hh.ndim else hh
    e2 = _evaluate(nd, Y + (0.5 * hc) * k1, level, k1)
    e3 = _evaluate(nd, Y + (0.5 * hc) * e2.unit, level, k1)
    e4 = _evaluate(nd, Y + hc * e3.unit, level, k1)
    incr = ((k1 + 2.0 * e2.unit) + 2.0 * e3.unit) + e4.unit
    Yn = Y + (hc / 6.0) * incr
    en = _evaluate(nd, Yn, level, k1)
    turn = (en.bad == _GOOD) & (_dot(en.unit, k1) < 0.0)
    bn = np.where(turn, _SINK, en.bad).astype(np.int8)
    first = _first_bad(e2.bad, e3.bad, e4.bad, bn)
    cell = np.full(len(Y), -1, np.int64)
    for e, b in ((en, bn), (e4, e4.bad), (e3, e3.bad), (e2, e2.bad)):
        cell = np.where(b != _GOOD, e.cell, cell)
    return Yn, en, first, cell


def _last_step(
    nd: _Nodes,
    Y: T.Array,
    k1: T.Array,
    tau0: T.Array,
    h: float,
    level: float,
    n_bis: int,
) -> tuple[T.Array, T.Array, T.Array]:
    """The stop in a step that crosses the stop set: n_bis bisections of the step length
    s in [0, h]. A sub-step is good if none of its evaluation points and its end is bad.
    Returns (the end of the longest good sub-step, its τ, its length)."""
    n = len(Y)
    lo, hi = np.zeros(n), np.full(n, h)
    Ye, te = Y.copy(), tau0.copy()
    for _ in range(n_bis):
        mid = 0.5 * (lo + hi)
        Ym, em, first, _cell = _rk4_step(nd, Y, k1, mid, level)
        good = first == _GOOD
        lo = np.where(good, mid, lo)
        hi = np.where(good, hi, mid)
        Ye = np.where(good[:, None], Ym, Ye)
        te = np.where(good, em.tau, te)
    return Ye, te, lo


def _order(starts: T.Array, roots: T.Array) -> T.IntArray:
    """A fixed order of the columns, from their values only (np.lexsort, stable)."""
    keys = (
        starts[:, 2],
        starts[:, 1],
        starts[:, 0],
        roots[:, 2],
        roots[:, 1],
        roots[:, 0],
    )
    return np.asarray(np.lexsort(keys), np.int64)


def integrate(
    field: T.Field,
    starts: T.Array,
    roots: T.Array,
    settings: T.SolverSettings = T.DEFAULTS,
    *,
    max_cos: float = CONTACT_COS,
    p_jump: T.Array | None = None,
    stop_sd: T.Array | None = None,
    level_sd: T.Array | None = None,
    levels: T.Array | None = None,
) -> tuple[T.Columns, dict[str, Any]]:
    """Trace one column from each start point (N, 3) through the field (the module
    docstring).

    roots (N, 3) are stored in the Columns and fix the order of the work; they do not
    enter the trace. Returns (Columns, info). info holds: "ends" (the count of each
    EndReason, by name), "unresolved_cells_met" (the sorted ids of the unresolved
    contact cells that ended a column), "n_unresolved_cells" (the count in the whole
    field), "steps" (the iterations of the loop), "max_steps"."""
    S0 = np.asarray(starts, float).reshape(-1, 3)
    R0 = np.asarray(roots, float).reshape(-1, 3)
    assert len(S0) == len(R0)
    N = len(S0)
    order = _order(S0, R0)
    S = S0[order]
    nd = _nodes(field, settings, max_cos)
    level = settings.stop_level
    h = settings.rk4_step_mm / T.VOXEL_MM
    max_steps = int(np.floor(settings.max_column_mm / settings.rk4_step_mm + 1e-9))

    end = np.full(N, -1, np.int8)
    met = np.full(N, -1, np.int64)
    # the start points
    e0 = _evaluate(nd, S, level, np.zeros((N, 3)))
    Y = S.copy()
    K1 = e0.unit
    TAU = e0.tau.copy()
    rec_ids = [np.arange(N)]
    rec_pos = [S.copy()]
    rec_tau = [e0.tau]
    for code, reason in _REASON.items():
        end[e0.bad == code] = int(reason)
    met[e0.bad == _UNRESOLVED] = e0.cell[e0.bad == _UNRESOLVED]
    counts = np.ones(N, np.int64)
    active = np.flatnonzero(e0.bad == _GOOD)
    steps = 0
    while active.size and steps < max_steps:
        steps += 1
        Ya, k1 = Y[active], K1[active]
        Yn, en, first, cell = _rk4_step(nd, Ya, k1, h, level)
        good = first == _GOOD
        g = active[good]
        Y[g] = Yn[good]
        K1[g] = en.unit[good]
        TAU[g] = en.tau[good]
        rec_ids.append(g)
        rec_pos.append(Yn[good])
        rec_tau.append(en.tau[good])
        counts[g] += 1
        st = first == _STOP
        if st.any():
            s = active[st]
            Ye, te, lo = _last_step(
                nd, Ya[st], k1[st], TAU[s], h, level, settings.stop_bisections
            )
            new = lo > 0  # a zero-length sub-step adds no point
            rec_ids.append(s[new])
            rec_pos.append(Ye[new])
            rec_tau.append(te[new])
            counts[s[new]] += 1
            end[s] = int(T.EndReason.STOPPED)
        for code in (_OUTSIDE, _UNRESOLVED, _SINK):
            m = first == code
            end[active[m]] = int(_REASON[code])
        mu = first == _UNRESOLVED
        met[active[mu]] = cell[mu]
        active = g
    end[active] = int(T.EndReason.MAX_LENGTH)
    assert (end >= 0).all()

    m = int(counts.max()) if N else 1
    paths = np.full((N, m, 3), np.nan)
    tau_path = np.full((N, m), np.nan)
    fill = np.zeros(N, np.int64)
    for ids, pos, tau in zip(rec_ids, rec_pos, rec_tau, strict=True):
        paths[ids, fill[ids]] = pos
        tau_path[ids, fill[ids]] = tau
        fill[ids] += 1
    assert (fill == counts).all()

    def back(a: Any) -> Any:  # the caller's order
        out = np.empty_like(a)
        out[order] = a
        return out

    paths, tau_path, counts, end, met = (
        back(a) for a in (paths, tau_path, counts, end, met)
    )
    idx = np.arange(N)
    stops = paths[idx, counts - 1] if N else np.zeros((0, 3))
    tau_arrival = tau_path[idx, counts - 1] if N else np.zeros(0)
    stopped = end == int(T.EndReason.STOPPED)
    jump = np.where(stopped, np.maximum(0.0, T.TAU_TOP - tau_arrival), np.nan)
    lv = np.zeros(0) if levels is None else np.asarray(levels, float)
    cols = T.Columns(
        roots=R0.copy(),
        paths=paths,
        counts=counts,
        tau_path=tau_path,
        stops=stops,
        end=end,
        tau_arrival=tau_arrival,
        jump=jump,
        p_jump=np.full(N, np.nan) if p_jump is None else np.asarray(p_jump, float),
        stop_sd=np.full(N, np.nan) if stop_sd is None else np.asarray(stop_sd, float),
        level_sd=np.full((N, len(lv)), np.nan)
        if level_sd is None
        else np.asarray(level_sd),
        levels=lv,
    )
    info: dict[str, Any] = dict(
        ends={r.name: int((end == int(r)).sum()) for r in T.EndReason},
        unresolved_cells_met=sorted({int(c) for c in met[met >= 0]}),
        n_unresolved_cells=int(nd.unresolved.sum()),
        steps=steps,
        max_steps=max_steps,
    )
    return cols, info
