"""The mean field of the flow-field solver (docs/plans/flow-field-phantoms.md,
*Inference*; PR 3 task 3.3). The spec of PR 3, section *The mean field*, makes it
concrete; this module follows it.

The entry point is `solve`. It takes a grid, the inputs (ff_types.Inputs: the stop
field, the tensor and the contrast) and the noise SD of the two echoes. It returns the
solved `ff_types.Field`, the extras for field.npz and a log that is JSON-able.

The loop
--------

1. The initial field: the moving region is the set of nodes with stop log-odds < 0
   (stop probability < 0.5). τ_init = −(the distance in mm to the nearest stopped
   node); d_init = the tensor's eigenvector, signed to agree with the central
   difference of τ_init; u = 0; q_stop = sigmoid(log-odds). With `init` (a solved
   Field on another grid), its q_stop, mean τ and mean direction, interpolated to the
   nodes, replace these values (`initial_from_field`).
2. For each width w of settings.widths: the predictions (τ_pred, d_pred) (at the
   first width the initial field, after it `recentre`); the moving candidates around
   them (ff_terms.moving_candidates); q carried over from the previous state with a
   Gaussian kernel of Δτ in τ and of settings.dir_cone_deg in angle (`carry_over`),
   q_stop kept; s_ref = exp(−u); the profile lines of the candidates (computed one
   time per width).
3. One sweep: for each colour c = 0 .. n_colours − 1 in order, q_i ∝ exp(−(unary +
   messages)) over the valid states of the nodes of colour c, then the moments of these
   nodes. Then u by colour (ff_terms.update_u). Then the free energy F = Σ
   expected_energy − H(q).
4. After the first sweep of each width: `prune`. Then the width continues from the
   pruned state, and the next sweep is compared with F after the pruning.
5. A width ends at max |Δq| < tol_q, or |ΔF| < tol_free_energy_rel · |F|, or
   max_sweeps sweeps.
6. A rise of F in a sweep above 1e-9 · |F| is a defect: the log keeps it, and the
   loop continues. F is compared only within one width (the energy changes with w,
   the candidates and the profiles).
7. Only with `solve(recentre_at_edge=True)` (the provisional option CD-13; default
   off): see *Re-centring at the edge* below.

Re-centring at the edge (CD-13, provisional)
--------------------------------------------

A finding: a solve reaches only the minimum of F over its last candidate sets, and
the predictions move only at the re-centrings between widths. The option
recentre_at_edge (default False) adds re-centrings inside a width. At the end of
each pass of a width, `edge_counts` gives the final choices at the edge. A final
choice is at the edge if tau_at_kept_edge counts it or dir_at_ring counts it. A node
that kept one τ index only (tau_single) is at both τ edges, and tau_at_kept_edge
counts it, so it is at the edge too: a re-centring moves its window. If a final
choice is at the edge and the width has used fewer than max_sweeps sweeps over its
passes, the width does a new pass: `recentre`, new candidates round the new
predictions, the carry-over as at a width change, the pruning after the first sweep,
the same stop rules, and at most max_sweeps minus the sweeps used. The width ends
when no final choice is at the edge or when its passes used max_sweeps sweeps in
total. F is compared only within one pass (the rise rule above). The option adds no
constant. With the option off, the loop is the plan's schedule, with the same bits.

The origin of τ is the top (τ_top = 0, the default) or an anchor node (the comparison
run of the plan's question 6; see `solve`).

Every coupling term is multilinear in the node distributions, and no two nodes of one
colour share a term. So each colour update is the exact minimiser of F over the q of
these nodes, and the u update is the minimiser over u of these nodes (update_u never
gives a larger energy). So F cannot rise in a sweep, except by rounding. The pruning
can raise F: it is not a sweep, and the log gives its change of F separately.

Pruning: per node, the moving candidates sorted by q (ties: the lower candidate index
first) are kept until they hold settings.prune_mass of the moving mass, at least
prune_min, at most prune_max, and never more than the valid ones. The stopped state is
always kept. The node's distribution is then divided by its kept mass. The state is
then compacted to prune_max slots: the kept candidates first, in the order of their
candidate index (a · n_dir + m).

Re-centring between widths: τ_pred_i = the mean over the axis neighbours j with
q_stop_j < stop_level of τ̄_j + exp(−u_j) · d̄_j · (x_i − x_j) (mm), with τ̄_j the
conditional mean τ and d̄_j the normalised posterior-mean direction of j. Where no
neighbour is moving: τ̄_i if the node has moving mass, else TAU_TOP. d_pred_i = the
node's posterior-mean direction (the previous prediction where it is zero).

Profiles (CD-7): the state holds the profile lines of ff_terms.profile_lines, one line
of L + n_tau − 1 = 17 levels per direction, 7 per node, computed one time per width.
Each slot reads its 9 levels from its line (ff_terms.line_slots: line = m, first line
index n_tau − 1 − a). After the pruning, the kept slots keep their line and first index
(`prune` compacts prof_line and prof_off, not the lines). The map is exact; the two
forms differ only by rounding (ff_terms.profile_lines). On the slab run below, F of the
two forms agreed to 5e-15 of |F| in every sweep.

Memory per node (the state that the loop holds):

- the grid arrays (GridArrays, Q_u, the edge and pair tables): about 1.3 kB;
- the candidates at K slots: cand_tau, q (8 K bytes each), cand_dir (24 K), valid (K),
  the candidate index (2 K), prof_line and prof_off (K each): 44 K bytes, so 2.8 kB at
  K = 63;
- the profile lines: 238 · b bytes (b = 8 in float64, 4 in float32): 1.9 kB in
  float64, 0.95 kB in float32;
- the moments (Moments): about 0.35 kB.

The pruning compacts every (n, K) array of the state in place (`_compact`): it does not
allocate a second state, but the buffers of 63 slots stay allocated until the next
width frees them. So the sweep-1 size is the peak of a width. At the next width the
profile lines are freed first; the other old arrays are freed after the carry-over,
before the new lines are allocated. The colour update, the moments, the free energy
(ff_terms.expected_energy and `entropy`), update_u, the pruning rule, the posterior
means and `edge_counts` work in node chunks, so no full (n, K) temporary occurs. The
output Field (K = prune_max) is allocated after the lines are freed.

The macOS allocator keeps freed large blocks resident (they count in the RSS that
scripts/memguard reads) and reuses them only for requests of the same size. So the
RSS peak is above the live peak.

Measured (this machine, CPU, float32 profiles, oracle inputs with noise 0,
OMP_NUM_THREADS=4, scripts/memguard; max_sweeps and widths set only for the
measurement; "before" = the form of 63 profiles of 9 levels, run in the same session):

- the slab, 64 × 64 × 48 voxels at 0.16 mm (1.97·10⁵ nodes), widths (4, 2, 1),
  max_sweeps 4: peak 2482 MiB, 249 s, 13.5 to 14.2 s per sweep at K = 32, 38 to 40 s
  from the end of one width to the end of sweep 1 of the next. Before: 3232 MiB, 272 s,
  13.1 to 13.9 s per sweep, 44 to 46 s.
- crown_w1.28, 81³ voxels at 0.16 mm (5.3·10⁵ nodes), width 4 only, max_sweeps 2:
  peak 4876 MiB, 139 s, 36.2 s for sweep 2 (K = 32), 91 s to the end of sweep 1.
  Before: 6443 MiB, 161 s, 36.2 s for sweep 2, 110 s to the end of sweep 1.
- a profile (cProfile) of the slab, width 4, 4 sweeps (76 s): the colour updates 46 s
  (ff_terms.messages 42 s), expected_energy 19 s (6 calls), update_u 6 s. Inside them:
  the own octants 18 s, the octants of the neighbours 16 s, jump_energy 10 s, the
  masked gathers of the candidates (ff_terms._cand) 8 s, the gathers of the profiles
  from the lines 3 s.

NumPy or Torch (the plan's *NumPy or Torch*)
--------------------------------------------

Symbols: n nodes, K candidate slots (63 in the first sweep of a width, then
prune_max = 32). Sizes: PR 3 runs up to 5.3·10⁵ nodes, PR 4 1.2·10⁶ nodes, the target
4·10⁶ to 4·10⁷ nodes.

Operations per sweep (estimates from the loop structure, not measured): 8 calls of
ff_terms.messages on n / 8 nodes each (about 2.4·10³ · K flops per node, from the
analysis in ff_terms.py), 8 calls of update_u (the own octants, 8 · K · 40 flops, and a
search of about 350 values of a 4-term function: about 3·10³ flops per node), one
expected_energy (the own octants and the edges once) and the entropy (n · K). In total
about 4·10³ · K flops per node: about 7·10¹⁰ flops per sweep at 5.3·10⁵ nodes and K =
32, 1.5·10¹¹ at 1.2·10⁶ nodes, 5·10¹¹ to 5·10¹² at the target size. A run has up to
3 · 200 sweeps. At the measured rate (14 s per sweep at 1.97·10⁵ nodes, about 70 µs per
node and sweep), 600 sweeps take about 2.3 h on the slab, 6 h on the crown and about
20 days at 4·10⁷ nodes.

Memory traffic per sweep: by estimate the octant temporaries of ff_terms (about 10
(m, K) arrays per octant combination) and the gathered chunk profiles, not the stored
lines (2 · n · 238 · b bytes per sweep), set the time of a sweep.

Loop structure: Python loops over the widths, the sweeps, the 8 colours, the chunks
and fixed small sets (octants, edge and pair slots). Each kernel inside is whole-array
element-wise arithmetic, gathers by fixed padded tables and small dense contractions.
The work is regular and data-parallel, and a Torch version could be the same
algorithm (the CSR products with Q_u and Q_tau would become padded tables of up to 27
neighbours).

The measurement (a line-for-line Torch port of ff_terms._own_octants, the largest
kernel, on one colour; 4 CPU threads; the port is not part of this module):

- 1.97·10⁵ nodes, K = 32, one colour: NumPy float64 0.17 s; Torch CPU float64 0.13 s;
  Torch CPU float32 0.06 s; MPS float32 0.036 s (chunk 32768; 0.14 s at chunk 4096).
- 1.05·10⁶ nodes, K = 32, one colour: NumPy float64 0.90 to 0.99 s; Torch CPU float64
  0.62 s; Torch CPU float32 0.30 s; MPS float32 0.16 s (chunk 131072).

The free energy in float32 (every float array of the solved slab state at width 4,
sweep 3, cast to float32, NumPy): |F32 − F64| = 0.22, with |F| = 3.66·10⁵; 0.21 of it
from the consistency term, which is a difference of large terms (F0 − 2 s F1 +
s² F2). The stop rule needs |ΔF| < 1e-7 · |F| = 0.037, and the rise test
1e-9 · |F| = 3.7·10⁻⁴.

Choice: NumPy. Reasons: (1) on the CPU in float64 (the exact path) Torch gives 1.3 to
1.6 times on the largest kernel with 4 threads: not a large gain. (2) MPS gives about
5 to 6 times, but only in float32 (MPS has no float64). In float32 the error of F is
about 6 times the stop threshold of the free energy and 600 times the rise test, so an
MPS run would not keep the stop rules and would log false rises. (3) At the target
size neither backend makes a run possible here: the state does not fit in memory
(ff_terms.py, *NumPy or Torch*), and a run takes days even with the MPS gain. The
target needs blocks with a halo first (column-solver.md, *Coarse to fine*); a CUDA
device with float64 is the place to compare the backends again.
`profile_dtype=np.float32` halves the profile lines for the runs; the tests use
float64.

Determinism: the loop has no random numbers, every sort is stable or has an explicit
tie rule, and every sum is over a fixed order. The same input gives the same bits on
the CPU (tested).
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import ff_grid as FG  # noqa: E402
import ff_terms as FT  # noqa: E402
import ff_types as T  # noqa: E402

Array = T.Array
IntArray = T.IntArray
BoolArray = T.BoolArray

CHUNK = 2048  # nodes per chunk in the carry-over, the profiles and the compaction
RISE_REL = 1e-9  # a rise of F above RISE_REL · |F| is a defect (float64 rounding)
TAU_PRIOR_SD = 1.0  # the SD of the FEM prior on τ (only if tau_prior; not in the spec)

LogFn = Callable[[str], None]


# ---------------------------------------------------------------- the grid arrays


def grid_arrays(
    grid: T.Grid, settings: T.SolverSettings = T.DEFAULTS, tau_prior: bool = False
) -> FT.GridArrays:
    """The GridArrays of a grid, from the ff_grid helpers. dir_pairs: the pairs (i, j),
    i < j, with K_ij < 0 of the Q1 stiffness, and dir_pair_w = −K_ij. Q_u =
    fem_precision(grid, fem_length_mm, speed_log_sd). Q_tau = fem_precision(grid,
    fem_length_mm, TAU_PRIOR_SD) if tau_prior, else None."""
    K, M = FG.fem_matrices(grid)  # one assembly for the pairs, Q_u, Q_tau and node_w
    up = sp.triu(K, k=1).tocoo()
    neg = up.data < 0
    rows = np.asarray(up.row[neg], np.int64)
    cols = np.asarray(up.col[neg], np.int64)
    order = np.lexsort((cols, rows))
    dir_pairs = np.stack([rows[order], cols[order]], axis=1)
    dir_pair_w = np.asarray(-up.data[neg][order], np.float64)
    del up, rows, cols, order, neg
    Q_u = FG.precision_from(K, M, settings.fem_length_mm, settings.speed_log_sd)
    Q_tau = (
        FG.precision_from(K, M, settings.fem_length_mm, TAU_PRIOR_SD)
        if tau_prior
        else None
    )
    node_w = FG.node_weights_from_mass(M)
    del K, M
    return FT.GridArrays(
        nodes=np.asarray(grid.nodes, np.float64),
        nbr6=FG.axis_neighbours(grid),
        edges=np.asarray(grid.edges, np.int64),
        edge_vec_mm=FG.edge_vectors_mm(grid),
        node_w=node_w,
        edge_w=FG.edge_weights_vox(grid),
        face_w=FG.face_weights_vox(grid),
        dir_pairs=dir_pairs.reshape(-1, 2),
        dir_pair_w=dir_pair_w,
        Q_u=Q_u,
        Q_tau=Q_tau,
    )


def grid_spacing_mm(grid: T.Grid) -> float:
    """The spacing that sets σ_c: grid.spacing_mm if the grid has it, else the smallest
    cell side."""
    s = getattr(grid, "spacing_mm", None)
    if s is not None:
        return float(s)
    return float(np.min(grid.cell_size_mm))


# ---------------------------------------------------------------- the inputs

NO_DIRECTION = np.array([1.0, 0.0, 0.0])  # where the tensor gives no direction


def tensor_at_nodes(tensor: T.Tensor, nodes: Array) -> tuple[Array, Array, int]:
    """(direction (n, 3), coherence (n,), count): the tensor at the nodes. Where the
    direction is not finite or zero, or the coherence is not finite, the direction is
    NO_DIRECTION and the coherence 0 (so the alignment term is 0 there); count is the
    number of these nodes. (The oracle tensor gives no direction where ∇φ = 0, for
    example on the axis of a crown.)"""
    d, c = tensor.at(nodes)
    d = np.asarray(d, np.float64)
    c = np.asarray(c, np.float64)
    nr = np.linalg.norm(d, axis=1)
    bad = ~(np.isfinite(nr) & (nr > 0) & np.isfinite(c))
    d = np.where(bad[:, None], NO_DIRECTION, d / np.where(bad, 1.0, nr)[:, None])
    c = np.where(bad, 0.0, c)
    return d, c, int(bad.sum())


# ---------------------------------------------------------------- the initial field


def _central_gradient(ga: FT.GridArrays, f: Array) -> Array:
    """(n, 3) the gradient of a node function (per mm) by differences along the axis
    neighbours: central where both exist, one-sided where one exists, 0 where none."""
    g = np.zeros((len(f), 3))
    x = ga.nodes * T.VOXEL_MM
    for a in range(3):
        jp = ga.nbr6[:, 2 * a]
        jm = ga.nbr6[:, 2 * a + 1]
        ip = np.where(jp >= 0, jp, np.arange(len(f)))
        im = np.where(jm >= 0, jm, np.arange(len(f)))
        dx = x[ip, a] - x[im, a]
        ok = dx != 0
        g[ok, a] = (f[ip[ok]] - f[im[ok]]) / dx[ok]
    return g


def initial_field(
    ga: FT.GridArrays, stop_log_odds: Array, tensor_dir: Array
) -> tuple[Array, Array, Array]:
    """(τ_init (n,), d_init (n, 3), q_stop (n,)) of the spec: τ_init = −(the distance in
    mm to the nearest node with log-odds ≥ 0); 0 on those nodes, and 0 everywhere if
    there is none. d_init = the tensor direction with the sign of its dot product with
    the central gradient of τ_init (+ where that is 0). q_stop = sigmoid(log-odds)."""
    n = len(stop_log_odds)
    stopped = stop_log_odds >= 0.0
    tau = np.zeros(n)
    if stopped.any() and not stopped.all():
        x = ga.nodes * T.VOXEL_MM
        tree = cKDTree(x[stopped])
        dist, _ = tree.query(x[~stopped], k=1, workers=1)
        tau[~stopped] = -np.asarray(dist, np.float64)
    g = _central_gradient(ga, tau)
    d = np.asarray(tensor_dir, np.float64)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    sgn = np.where(np.einsum("nc,nc->n", d, g) < 0.0, -1.0, 1.0)
    return tau, d * sgn[:, None], T.sigmoid(stop_log_odds)


def initial_from_field(
    init: T.Field,
    nodes: Array,
    settings: T.SolverSettings = T.DEFAULTS,
    tau_max: float | None = T.TAU_TOP,
) -> tuple[Array, Array, Array, BoolArray]:
    """(τ (n,), d (n, 3), q_stop (n,), has (n,)) of a solved Field on another grid,
    interpolated to the nodes (voxel positions). The nodes are clipped to the box of
    init's nodes. q_stop: the trilinear interpolation over the 8 corners. τ and d: the
    trilinear weights times the corners' moving probability, renormalised (d then
    normalised). has = False where no corner has moving mass or the mean direction is
    zero; there τ and d have no meaning. τ is clipped to tau_max (no clip if None)."""
    g = init.grid
    lo = g.nodes.min(0)
    hi = g.nodes.max(0)
    X = np.clip(np.asarray(nodes, np.float64), lo, hi)
    cell, local = g.locate(X)
    if np.any(cell < 0):
        raise ValueError("a node is outside the grid of init")
    w = T.trilinear_weights(local)
    corners = g.cells[cell]
    qs = (w * init.q_stop[corners]).sum(1)
    pm = w * init.p_moving[corners]
    tm = np.nan_to_num(init.tau_mean())[corners]
    dm = init.direction_mean()[corners]
    sw = pm.sum(1)
    tau = np.where(sw > 0, (pm * tm).sum(1) / np.maximum(sw, 1e-300), T.TAU_TOP)
    if tau_max is not None:
        tau = np.minimum(tau, tau_max)
    D = np.einsum("nk,nkc->nc", pm, dm)
    nr = np.linalg.norm(D, axis=1)
    has = (sw > 0) & (nr > 0)
    d = np.where(has[:, None], D / np.maximum(nr, 1e-300)[:, None], 0.0)
    return tau, d, np.clip(qs, 0.0, 1.0), has


# ---------------------------------------------------------------- carry-over


def carry_over(
    old_tau: Array,
    old_dir: Array,
    old_q: Array,
    old_valid: BoolArray,
    new_tau: Array,
    new_dir: Array,
    new_valid: BoolArray,
    dtau: float,
    cone_deg: float,
    chunk: int = CHUNK,
) -> Array:
    """(n, K_new) the moving q on the new candidates. Each old candidate k gives its
    mass q_k to the valid new candidates c in proportion to exp(−½ ((τ_c − τ_k) / Δτ)²
    − ½ (θ_kc / cone)²), θ_kc the angle between d_k and d_c. So the moving mass of each
    node is kept (to rounding), and a point mass goes mostly to its nearest new
    candidate in this metric. Each node must have at least one valid new candidate."""
    n, Ko = old_tau.shape
    Kn = new_tau.shape[1]
    out = np.zeros((n, Kn))
    th = math.radians(cone_deg)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        ov = old_valid[a:b]
        ot = np.where(ov, old_tau[a:b], 0.0)
        od = np.where(ov[..., None], old_dir[a:b], 0.0)
        oq = np.where(ov, old_q[a:b], 0.0)
        nv = new_valid[a:b]
        dt = (new_tau[a:b][:, None, :] - ot[:, :, None]) / dtau
        cos = np.clip(np.einsum("mkc,mjc->mkj", od, new_dir[a:b]), -1.0, 1.0)
        ang = np.arccos(cos) / th
        logk = -0.5 * (dt * dt + ang * ang)
        logk = np.where(nv[:, None, :], logk, -np.inf)
        mx = logk.max(2, keepdims=True)
        wk = np.exp(logk - mx)
        wk /= wk.sum(2, keepdims=True)
        out[a:b] = np.einsum("mk,mkj->mj", oq, wk)
    return out


# ---------------------------------------------------------------- re-centring


def posterior_means(st: FT.TermState, chunk: int = CHUNK) -> tuple[Array, Array, Array]:
    """(p (n,), τ̄ (n,), d̄ (n, 3)): the moving mass, the conditional mean τ (TAU_TOP
    where p = 0) and the normalised posterior-mean direction (0 where it is zero). By
    node chunks."""
    n = st.n
    p = np.empty(n)
    tbar = np.empty(n)
    dbar = np.empty((n, 3))
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        v = st.valid[a:b]
        q = np.where(v, st.q[a:b], 0.0)
        pc = q.sum(1)
        tau = np.where(v, st.cand_tau[a:b], 0.0)
        tbar[a:b] = np.where(
            pc > 0, (q * tau).sum(1) / np.maximum(pc, 1e-300), T.TAU_TOP
        )
        D = np.einsum("nk,nkc->nc", q, np.where(v[..., None], st.cand_dir[a:b], 0.0))
        nr = np.linalg.norm(D, axis=1)
        dbar[a:b] = np.where(nr[:, None] > 0, D / np.maximum(nr, 1e-300)[:, None], 0.0)
        p[a:b] = pc
    return p, tbar, dbar


def recentre(
    st: FT.TermState,
    ga: FT.GridArrays,
    dir_prev: Array,
    settings: T.SolverSettings = T.DEFAULTS,
) -> tuple[Array, Array]:
    """(τ_pred (n,), d_pred (n, 3)) for the next width (module docstring)."""
    p, tbar, dbar = posterior_means(st)
    n = st.n
    x = ga.nodes * T.VOXEL_MM
    s = np.exp(-st.u)
    num = np.zeros(n)
    cnt = np.zeros(n)
    for c in range(6):
        j = ga.nbr6[:, c]
        jj = np.where(j >= 0, j, 0)
        ok = (j >= 0) & (st.q_stop[jj] < settings.stop_level) & (p[jj] > 0)
        val = tbar[jj] + s[jj] * np.einsum("nc,nc->n", dbar[jj], x - x[jj])
        num += np.where(ok, val, 0.0)
        cnt += ok
    own = np.where(p > 0, tbar, T.TAU_TOP)
    tau_pred = np.where(cnt > 0, num / np.maximum(cnt, 1.0), own)
    has = np.linalg.norm(dbar, axis=1) > 0
    d_pred = np.where(has[:, None], dbar, dir_prev)
    return tau_pred, d_pred


# ---------------------------------------------------------------- pruning


@dataclass
class PruneResult:
    """order (n, K): the slots sorted for the compaction (kept first, in the order of
    their slot); n_keep (n,): the kept moving candidates; kept_mass (n,): their mass
    before the renormalisation; moving_mass (n,): the moving mass before the pruning."""

    order: IntArray
    n_keep: IntArray
    kept_mass: Array
    moving_mass: Array


def prune_choice(
    q: Array,
    valid: BoolArray,
    settings: T.SolverSettings = T.DEFAULTS,
    k_out: int | None = None,
    chunk: int = CHUNK,
) -> PruneResult:
    """The kept moving candidates of each node (module docstring, *Pruning*). order
    holds the first k_out columns (all K if None). By node chunks: the rule acts on
    each node alone."""
    n, K = q.shape
    ko = K if k_out is None else min(int(k_out), K)
    order = np.empty((n, ko), np.int64)
    n_keep = np.empty(n, np.int64)
    kept_mass = np.empty(n)
    moving = np.empty(n)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        o, nk, km, p = _prune_rows(q[a:b], valid[a:b], settings)
        order[a:b] = o[:, :ko]
        n_keep[a:b] = nk
        kept_mass[a:b] = km
        moving[a:b] = p
    return PruneResult(order, n_keep, kept_mass, moving)


def _prune_rows(
    q: Array, valid: BoolArray, settings: T.SolverSettings
) -> tuple[IntArray, IntArray, Array, Array]:
    """prune_choice for one chunk of rows: (order, n_keep, kept_mass, moving_mass)."""
    n, K = q.shape
    qm = np.where(valid, q, 0.0)
    p = qm.sum(1)
    idx = np.broadcast_to(np.arange(K), (n, K))
    by_q = np.lexsort((idx, ~valid, -qm), axis=-1)
    qs = np.take_along_axis(qm, by_q, axis=1)
    cum = np.cumsum(qs, axis=1)
    need = settings.prune_mass * p
    count = (cum < need[:, None]).sum(1) + 1
    count = np.where(p > 0, count, settings.prune_min)
    n_valid = valid.sum(1)
    n_keep = np.minimum(
        np.clip(count, settings.prune_min, settings.prune_max), n_valid
    ).astype(np.int64)
    rank = np.empty_like(by_q)
    np.put_along_axis(rank, by_q, idx, axis=1)
    keep = rank < n_keep[:, None]
    order = np.lexsort((idx, ~keep), axis=-1).astype(np.int64)
    kept_mass = np.where(keep, qm, 0.0).sum(1)
    return order, n_keep, kept_mass, p


def _compact(
    P: npt.NDArray[Any],
    idx: IntArray,
    chunk: int,
    fill: Callable[[int, int, npt.NDArray[Any]], npt.NDArray[Any]] | None = None,
) -> Any:
    """The array P (n, K, ...) gathered to the slots idx (n, Kn), Kn ≤ K, written in
    place into the first n · Kn · r elements of P's buffer (r = the size of the
    trailing axes), row chunk by row chunk. The write of rows [a, b) ends at
    b · Kn · r ≤ b · K · r, the start of the rows that are not yet read, so no unread
    value is overwritten. fill(a, b, rows): an optional change of the gathered rows
    [a, b) before the write. Returns a view of the buffer (n, Kn, ...)."""
    n, K = P.shape[:2]
    rest = P.shape[2:]
    Kn = idx.shape[1]
    if not P.flags.c_contiguous:
        P = np.ascontiguousarray(P)
    flat = P.reshape(-1)
    row = Kn * int(np.prod(rest, dtype=np.int64))
    ix = (slice(None), slice(None)) + (None,) * len(rest)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        tmp = np.take_along_axis(P[a:b], idx[a:b][ix], axis=1)
        if fill is not None:
            tmp = fill(a, b, tmp)
        flat[a * row : b * row] = tmp.reshape(-1)
    return flat[: n * row].reshape((n, Kn) + rest)


def _compact_profiles(P: npt.NDArray[Any], idx: IntArray, chunk: int) -> Any:
    """The profiles P (n, K, L, E) gathered to the slots idx (n, Kn) in place
    (`_compact`). Returns a view of the buffer (n, Kn, L, E)."""
    return _compact(P, idx, chunk)


def prune(
    st: FT.TermState,
    cand_index: npt.NDArray[np.int16],
    settings: T.SolverSettings = T.DEFAULTS,
    chunk: int = CHUNK,
) -> tuple[FT.TermState, npt.NDArray[np.int16], PruneResult]:
    """Prune and compact the state to settings.prune_max slots (module docstring).
    Every (n, K) array of `st` and cand_index is compacted in place (`_compact`), so
    the pruning allocates no second state. The returned state and index share the
    buffers of `st` and cand_index: st and cand_index must not be used after."""
    Kn = min(settings.prune_max, st.K)
    res = prune_choice(st.q, st.valid, settings, k_out=Kn, chunk=chunk)
    idx = res.order
    slot_ok = np.arange(Kn)[None, :] < res.n_keep[:, None]

    def zero_dropped(a: int, b: int, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.where(slot_ok[a:b], x, 0.0)

    def keep_valid(a: int, b: int, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.asarray(slot_ok[a:b] & x)

    def index_or_none(a: int, b: int, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.where(slot_ok[a:b], x, -1).astype(np.int16)

    q = _compact(st.q, idx, chunk, zero_dropped)
    z = st.q_stop + q.sum(1)
    q /= z[:, None]
    q_stop = st.q_stop / z
    cand_tau = _compact(st.cand_tau, idx, chunk)
    cand_dir = _compact(st.cand_dir, idx, chunk)
    valid = _compact(st.valid, idx, chunk, keep_valid)
    ci = _compact(cand_index, idx, chunk, index_or_none)
    if st.prof_line is None:  # direct form: one profile per slot
        prof = _compact(st.profiles, idx, chunk)
        pl, po = None, None
    else:  # line form: the kept slots keep reading their lines
        assert st.prof_off is not None
        prof = st.profiles
        pl = _compact(st.prof_line, idx, chunk)
        po = _compact(st.prof_off, idx, chunk)
    new = FT.TermState(
        cand_tau=cand_tau,
        cand_dir=cand_dir,
        valid=valid,
        q=q,
        q_stop=q_stop,
        u=st.u,
        s_ref=st.s_ref,
        profiles=prof,
        level_base=st.level_base,
        stop_log_odds=st.stop_log_odds,
        tensor_dir=st.tensor_dir,
        coherence=st.coherence,
        prof_line=pl,
        prof_off=po,
        n_levels=st.n_levels,
    )
    return new, ci, res


# ---------------------------------------------------------------- free energy


def entropy(st: FT.TermState, chunk: int = CHUNK) -> float:
    """H(q) = −Σ q log q over the valid moving states and the stopped state. By node
    chunks, summed in a fixed order."""
    hq = 0.0
    for a in range(0, st.n, chunk):
        b = min(a + chunk, st.n)
        q = np.where(st.valid[a:b], st.q[a:b], 0.0)
        hq += float(np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0).sum())
    qs = st.q_stop
    hs = np.where(qs > 0, qs * np.log(np.where(qs > 0, qs, 1.0)), 0.0).sum()
    return float(-(hq + hs))


def free_energy(
    st: FT.TermState,
    ga: FT.GridArrays,
    tw: FT.TermWeights,
    mom: FT.Moments | None = None,
) -> tuple[float, dict[str, float]]:
    """(F, terms): F = Σ expected_energy − H(q); terms holds each expected term and
    "entropy". mom: the current moments of all nodes (computed if None)."""
    terms = FT.expected_energy(st, ga, tw, mom=mom)
    H = entropy(st)
    F = float(sum(terms[k] for k in sorted(terms)) - H)
    out = {k: float(v) for k, v in terms.items()}
    out["entropy"] = H
    return F, out


# ---------------------------------------------------------------- one colour


def _update_colour(
    st: FT.TermState,
    ga: FT.GridArrays,
    tw: FT.TermWeights,
    sel: IntArray,
    mom: FT.Moments,
    chunk: int = FT.CHUNK,
    no_stop: BoolArray | None = None,
) -> float:
    """q_i ∝ exp(−(unary + messages)) over the valid states of the nodes sel; refresh
    their moments. Returns max |Δq| of these nodes. The nodes go in chunks of `chunk`.
    No two nodes of one colour share a term, so the messages of a chunk do not use the
    moments of the other chunks of sel, and the result is the same as in one step.
    no_stop (n,): the nodes that have no stopped state (the anchor); None: no such
    node."""
    dq = 0.0
    for s0 in range(0, len(sel), chunk):
        c = sel[s0 : s0 + chunk]
        dq = max(dq, _update_rows(st, ga, tw, c, mom, no_stop))
    return dq


def _update_rows(
    st: FT.TermState,
    ga: FT.GridArrays,
    tw: FT.TermWeights,
    sel: IntArray,
    mom: FT.Moments,
    no_stop: BoolArray | None = None,
) -> float:
    """_update_colour for one chunk of nodes."""
    M, Ms = FT.messages(st, ga, tw, sel, mom)
    e_move, e_stop = FT.unary(st, ga, tw, sel)
    v = st.valid[sel]
    lm = np.where(v, -(e_move + M), -np.inf)
    ls = -(e_stop + Ms)
    if no_stop is not None:
        ls = np.where(no_stop[sel], -np.inf, ls)
    mx = np.maximum(lm.max(1), ls)
    wm = np.where(v, np.exp(lm - mx[:, None]), 0.0)
    ws = np.exp(ls - mx)
    z = wm.sum(1) + ws
    q_new = wm / z[:, None]
    qs_new = ws / z
    dq = 0.0
    if len(sel):
        dq = float(
            max(
                np.abs(q_new - st.q[sel]).max(initial=0.0),
                np.abs(qs_new - st.q_stop[sel]).max(initial=0.0),
            )
        )
    st.q[sel] = q_new
    st.q_stop[sel] = qs_new
    FT.refresh_moments(mom, st, sel)
    return dq


# ---------------------------------------------------------------- profiles


def _profile_lines(
    ga: FT.GridArrays,
    c: Array,
    dirs: Array,
    s_ref: Array,
    lb: IntArray,
    volumes: Sequence[Array],
    settings: T.SolverSettings,
    dtype: Any,
    chunk: int = CHUNK,
) -> npt.NDArray[Any]:
    """(n, n_dir, L + n_tau − 1, 2) of ff_terms.profile_lines, computed by node chunks
    into an array of `dtype` (so a float32 run never holds the float64 array of all
    nodes)."""
    n, M = dirs.shape[:2]
    L = FT.N_LEVELS
    out = np.empty((n, M, L + settings.n_tau - 1, len(volumes)), dtype=dtype)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        out[a:b] = FT.profile_lines(
            ga.nodes[a:b],
            c[a:b],
            dirs[a:b],
            s_ref[a:b],
            lb[a:b],
            volumes,
            L,
            settings.n_tau,
            settings.dtau,
        )
    return out


# ---------------------------------------------------------------- diagnostics


def p_jump(st: FT.TermState, ga: FT.GridArrays, tw: FT.TermWeights) -> Array:
    """(n,) the probability that the node is moving next to a stopped node with J > 0:
    Σ_k q_k (1 − Π_j (1 − q_stop_j · 1[J_kj > 0])), over the edges (i, j) of the node,
    with independent nodes."""
    n = st.n
    out = np.zeros(n)
    for a in range(0, n, CHUNK):
        ids = np.arange(a, min(a + CHUNK, n))
        v = st.valid[ids]
        tau = np.where(v, st.cand_tau[ids], 0.0)
        d = np.where(v[..., None], st.cand_dir[ids], 0.0)
        q = np.where(v, st.q[ids], 0.0)
        none = np.ones_like(q)
        for k in range(ga.inc_edge.shape[1]):
            e = ga.inc_edge[ids, k]
            ok = e >= 0
            ee = np.where(ok, e, 0)
            j = np.where(ok, ga.inc_edge_nbr[ids, k], 0)
            delta = ga.inc_edge_sign[ids, k][:, None] * ga.edge_vec_mm[ee]
            _, J = FT.jump_energy(tau, d, st.s_ref[ids][:, None], delta[:, None, :], tw)
            qs = np.where(ok, st.q_stop[j], 0.0)
            none *= 1.0 - qs[:, None] * (J > 0.0)
        out[ids] = (q * (1.0 - none)).sum(1)
    return out


def edge_counts(
    st: FT.TermState,
    cand_index: npt.NDArray[np.int16],
    settings: T.SolverSettings = T.DEFAULTS,
    chunk: int = CHUNK,
) -> dict[str, int]:
    """The final choices at the edge of the kept set, over the moving nodes (q_stop <
    stop_level). The final choice is the kept candidate of the largest q (the lower
    slot on a tie). tau_at_kept_edge: its τ index is the smallest or the largest of the
    node's kept τ indices; tau_single: the node kept one τ index only (counted in
    tau_at_kept_edge too); tau_at_set_edge: its τ index is 0 or n_tau − 1;
    dir_at_ring: its direction is one of the ring (m ≥ 1). By node chunks."""
    counts = dict(
        moving_nodes=0,
        tau_at_kept_edge=0,
        tau_single=0,
        tau_at_set_edge=0,
        dir_at_ring=0,
    )
    for a in range(0, st.n, chunk):
        b = min(a + chunk, st.n)
        valid = st.valid[a:b]
        mov = st.q_stop[a:b] < settings.stop_level
        q = np.where(valid, st.q[a:b], -1.0)
        k = np.argmax(q, axis=1)
        ci = cand_index[a:b].astype(np.int64)
        best = ci[np.arange(b - a), k]
        a_best = best // settings.n_dir
        m_best = best % settings.n_dir
        a_all = np.where(valid, ci // settings.n_dir, -1)
        a_max = a_all.max(1)
        a_min = np.where(valid, a_all, settings.n_tau).min(1)
        ok = mov & (best >= 0)
        counts["moving_nodes"] += int(ok.sum())
        counts["tau_at_kept_edge"] += int(
            (ok & ((a_best == a_min) | (a_best == a_max))).sum()
        )
        counts["tau_single"] += int((ok & (a_min == a_max)).sum())
        counts["tau_at_set_edge"] += int(
            (ok & ((a_best == 0) | (a_best == settings.n_tau - 1))).sum()
        )
        counts["dir_at_ring"] += int((ok & (m_best >= 1)).sum())
    return counts


def _stats(x: npt.NDArray[Any]) -> dict[str, float]:
    if len(x) == 0:
        return dict(min=float("nan"), median=float("nan"), max=float("nan"))
    return dict(min=float(np.min(x)), median=float(np.median(x)), max=float(np.max(x)))


# ---------------------------------------------------------------- the solver


def _new_state(
    ga: FT.GridArrays,
    tau_pred: Array,
    dir_pred: Array,
    old: list[npt.NDArray[Any]],
    q_stop: Array,
    u: Array,
    lo: Array,
    tdir: Array,
    coh: Array,
    volumes: Sequence[Array],
    settings: T.SolverSettings,
    profile_dtype: Any,
    tau_max: float | None = T.TAU_TOP,
    anchor_node: int | None = None,
) -> tuple[FT.TermState, npt.NDArray[np.int16]]:
    """The state of a new width: candidates round the predictions, q carried over from
    `old` = [τ, d, q, valid], q_stop kept, s_ref = exp(−u), the profiles. The list `old`
    is emptied after the carry-over, so that the old arrays can be freed before the
    profiles are allocated. tau_max: the upper bound of τ (None: no bound). anchor_node:
    only the candidates of the centre τ index (τ = τ_pred = 0 there) are valid at this
    node."""
    cand_tau, cand_dir, valid = FT.moving_candidates(
        tau_pred, dir_pred, settings, tau_max
    )
    if anchor_node is not None:
        a_idx = np.arange(cand_tau.shape[1]) // settings.n_dir
        valid[anchor_node] = a_idx == (settings.n_tau - 1) // 2
    q = carry_over(
        old[0],
        old[1],
        old[2],
        old[3],
        cand_tau,
        cand_dir,
        valid,
        dtau=settings.dtau,
        cone_deg=settings.dir_cone_deg,
    )
    old.clear()
    q = np.where(valid, q, 0.0)
    s_ref = np.exp(-u)
    lb = FT.level_base(tau_pred, settings, FT.N_LEVELS, tau_max)
    c = np.asarray(tau_pred, np.float64)
    if tau_max is not None:
        c = np.minimum(c, tau_max)
    # the directions of the candidates a = 0: m = 0 .. n_dir − 1
    prof = _profile_lines(
        ga,
        c,
        cand_dir[:, : settings.n_dir],
        s_ref,
        lb,
        volumes,
        settings,
        profile_dtype,
    )
    K = cand_tau.shape[1]
    n = len(q)
    ci = np.broadcast_to(np.arange(K, dtype=np.int16), (n, K)).copy()
    line, off = FT.line_slots(settings.n_tau, settings.n_dir)
    st = FT.TermState(
        cand_tau=cand_tau,
        cand_dir=cand_dir,
        valid=valid,
        q=q,
        q_stop=q_stop.copy(),
        u=u.copy(),
        s_ref=s_ref,
        profiles=prof,
        level_base=lb,
        stop_log_odds=lo,
        tensor_dir=tdir,
        coherence=coh,
        prof_line=np.broadcast_to(line, (n, K)).copy(),
        prof_off=np.broadcast_to(off, (n, K)).copy(),
        n_levels=FT.N_LEVELS,
    )
    return st, ci


def _field(
    grid: T.Grid,
    st: FT.TermState,
    n_kept: IntArray,
    K_out: int,
) -> T.Field:
    """The Field of the state, padded or cut to K_out slots (kept slots first)."""
    n, K = st.cand_tau.shape
    Kc = min(K, K_out)
    tau = np.full((n, K_out), np.nan)
    d = np.zeros((n, K_out, 3))
    q = np.zeros((n, K_out))
    ok = st.valid[:, :Kc]
    # copyto with a mask: no temporary of the output's size
    np.copyto(tau[:, :Kc], st.cand_tau[:, :Kc], where=ok)
    np.copyto(d[:, :Kc], st.cand_dir[:, :Kc], where=ok[..., None])
    np.copyto(q[:, :Kc], st.q[:, :Kc], where=ok)
    return T.Field(
        grid=grid,
        tau=tau,
        direction=d,
        q=q,
        q_stop=st.q_stop.copy(),
        n_kept=n_kept.astype(np.int64),
    )


def solve(
    grid: T.Grid,
    inputs: T.Inputs,
    noise_sd: Sequence[float],
    settings: T.SolverSettings = T.DEFAULTS,
    *,
    tau_prior: bool = False,
    init: T.Field | None = None,
    origin: str = "top",
    anchor_node: int | None = None,
    profile_dtype: Any = np.float64,
    log_fn: LogFn | None = None,
    recentre_at_edge: bool = False,
) -> tuple[T.Field, dict[str, npt.NDArray[Any]], dict[str, Any]]:
    """Solve the field on a grid by the mean field (module docstring).

    grid: a ff_types.Grid without hanging nodes. inputs: the stop field, the tensor and
    the contrast (the detrended echoes {1: echo 2, 2: echo 3}). noise_sd: the matched
    noise SD of the two echoes (the image's "noise_sd"); both must be positive.
    tau_prior: the switch of the FEM prior on τ. init: a solved Field on another grid
    that replaces the initial field (a start field only, not a prior). origin: "top"
    (τ_top = 0, the default) or "anchor" (the comparison run of the plan's question 6:
    see *The anchor origin* below); anchor_node: the node id of the anchor (only with
    origin "anchor"). profile_dtype: the dtype of the stored profile lines (np.float32
    halves their memory; the arithmetic stays float64). log_fn: called with one line of
    text per sweep. recentre_at_edge: the provisional option CD-13 (default False: the
    plan's schedule). True: a width does more passes while a final choice is at the
    edge, up to max_sweeps sweeps in total (module docstring, *Re-centring at the
    edge*).

    The anchor origin: the anchor node is a moving node with τ = 0, and every other τ
    is free. In detail:

    - The jump term is off (TermWeights.jump = False): the stopped state gives no τ
      value to its moving neighbours.
    - No upper bound on τ: τ can be positive away from the anchor. The candidates are
      τ_pred + (a − 4) Δτ (ff_terms.moving_candidates, tau_max None), and the profiles
      use c = τ_pred.
    - The anchor node: τ_pred = 0 at every width, only the 7 candidates of the centre
      τ index (τ = 0) are valid, and it has no stopped state (q_stop = 0). So its τ is
      a point mass at 0 and it is moving. The stop field must give it log-odds < 0 (a
      moving node of the initial field); else solve raises ValueError.
    - The initial τ (from the stop field, or from init) is shifted by one constant so
      that the anchor's τ is 0. init's τ is not clipped.
    - All else is as with the top origin. p_jump in extras is nan (no jump term).

    The energy with the anchor origin changes by no constant shift of every τ, except
    for two parts: the contrast term samples the profiles at the absolute levels ℓ Δτ,
    so only a shift by an integer multiple of Δτ keeps it (with the levels shifted by
    the same integer); and the FEM prior on τ (tau_prior) has a mass term that no
    shift keeps.

    Returns (field, extras, log). field: K = settings.prune_max slots. extras (n,)
    arrays: u, speed (= exp(u)), p_jump, tau_pred and the (n, 3) dir_pred of the last
    width, and cand_index (n, K): the candidate index a · n_dir + m of each slot (−1 on
    padding). log: see `_new_log`.
    """
    t_all = time.perf_counter()
    if len(noise_sd) != 2 or not all(float(s) > 0 for s in noise_sd):
        raise ValueError("noise_sd must hold 2 positive values")
    if origin not in ("top", "anchor"):
        raise ValueError(f'origin must be "top" or "anchor", not {origin!r}')
    if (origin == "anchor") != (anchor_node is not None):
        raise ValueError('origin "anchor" needs anchor_node, and only it takes one')

    def say(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    ga = grid_arrays(grid, settings, tau_prior)
    nodes = ga.nodes
    n = len(nodes)
    lo = np.asarray(inputs.stop.log_odds(nodes), np.float64)
    if not np.all(np.isfinite(lo)):
        raise ValueError("the stop field gave a log-odds that is not finite")
    tdir, coh, n_bad_tensor = tensor_at_nodes(inputs.tensor, nodes)
    volumes = FT.smoothed_echoes(inputs.contrast, settings.band_scale_vox)
    spacing = grid_spacing_mm(grid)

    anchor = None if anchor_node is None else int(anchor_node)
    tau_max: float | None = T.TAU_TOP if anchor is None else None
    if anchor is not None:
        if not 0 <= anchor < n:
            raise ValueError(f"anchor_node {anchor} is not a node id (n = {n})")
        if lo[anchor] >= 0.0:
            raise ValueError(
                f"the stop field gives the anchor node {anchor} a log-odds "
                f"{lo[anchor]:.6g} >= 0: the anchor must be a moving node"
            )

    tau0, dir0, qs0 = initial_field(ga, lo, tdir)
    if init is not None:
        ti, di, qi, has = initial_from_field(init, nodes, settings, tau_max)
        tau0 = np.where(has, ti, tau0)
        dir0 = np.where(has[:, None], di, dir0)
        qs0 = qi
    no_stop: BoolArray | None = None
    if anchor is not None:
        tau0 = tau0 - tau0[anchor]
        qs0 = qs0.copy()
        qs0[anchor] = 0.0
        no_stop = np.zeros(n, bool)
        no_stop[anchor] = True
    u = np.zeros(n)
    colours = np.asarray(grid.colour, np.int64)
    groups = [np.flatnonzero(colours == c) for c in range(grid.n_colours)]

    log = _new_log(settings, noise_sd, tau_prior, profile_dtype, grid, n)
    log["run"]["tensor_not_finite"] = n_bad_tensor
    log["run"]["origin"] = origin
    log["run"]["recentre_at_edge"] = bool(recentre_at_edge)
    log["run"]["anchor_node"] = anchor
    log["run"]["anchor_position_vox"] = (
        None if anchor is None else [float(v) for v in nodes[anchor]]
    )
    old: list[npt.NDArray[Any]] = [
        tau0[:, None],
        dir0[:, None, :],
        (1.0 - qs0)[:, None],
        np.ones((n, 1), bool),
    ]
    q_stop = qs0
    tau_pred, dir_pred = tau0, dir0
    st: FT.TermState | None = None
    ci: npt.NDArray[np.int16] | None = None
    for w in settings.widths:
        t0 = time.perf_counter()
        tw = FT.term_weights(
            spacing, noise_sd, w, settings, tau_prior, jump=anchor is None
        )
        used = 0  # the sweeps of this width, over all its passes
        rep = 0  # the pass of this width: 0 is the plan's pass, 1, 2, ... repeats
        fl: list[float] = []
        while True:
            if st is not None:
                tau_pred, dir_pred = recentre(st, ga, dir_pred, settings)
                if anchor is not None:
                    tau_pred[anchor] = 0.0
                # views of the old buffers: only the profiles are freed here; the
                # rest goes when _new_state empties `old` after the carry-over
                old = [st.cand_tau, st.cand_dir, st.q, st.valid]
                q_stop = st.q_stop
                u = st.u
                st = None  # free the profiles before the new ones
                ci = None
            st, ci = _new_state(
                ga,
                tau_pred,
                dir_pred,
                old,
                q_stop,
                u,
                lo,
                tdir,
                coh,
                volumes,
                settings,
                profile_dtype,
                tau_max,
                anchor,
            )
            mom = FT.node_moments(st)
            F_old, terms = free_energy(st, ga, tw, mom)
            F_start = F_old
            F_end = F_old
            fr: list[float] = []
            prec: dict[str, Any] = {}
            end = "max_sweeps"
            for sweep in range(1, settings.max_sweeps - used + 1):
                dq = 0.0
                for sel in groups:
                    dq = max(dq, _update_colour(st, ga, tw, sel, mom, no_stop=no_stop))
                for sel in groups:
                    st.u[sel] = FT.update_u(st, ga, tw, sel, mom)
                F_new, terms = free_energy(st, ga, tw, mom)
                F_end = F_new
                fr.append(F_new)
                if F_new - F_old > RISE_REL * abs(F_old):
                    rise: dict[str, Any] = dict(
                        width=float(w), sweep=sweep, f_before=F_old, f_after=F_new
                    )
                    at = ""
                    if recentre_at_edge:
                        rise["repeat"] = rep
                        at = f" (repeat {rep})"
                    log["free_energy_rises"].append(rise)
                    log["defects"].append(
                        f"free energy rose at width {w:g}, sweep {sweep}{at}: "
                        f"{F_old:.17g} -> {F_new:.17g}"
                    )
                dF = abs(F_new - F_old)
                say(f"w={w:g} sweep={sweep} F={F_new:.12g} dF={dF:.3g} max_dq={dq:.3g}")
                if sweep == 1:
                    del mom  # the moments of the pruned state replace it
                    st, ci, pr = prune(st, ci, settings)
                    mom = FT.node_moments(st)
                    F_pruned, terms = free_energy(st, ga, tw, mom)
                    F_end = F_pruned
                    prec = dict(
                        width=float(w),
                        kept=_stats(pr.n_keep),
                        at_prune_max=int((pr.n_keep >= settings.prune_max).sum()),
                        removed_mass_max=float(
                            np.max(pr.moving_mass - pr.kept_mass, initial=0.0)
                        ),
                        free_energy_before=F_new,
                        free_energy_after=F_pruned,
                    )
                    if rep == 0:
                        log["pruning"]["per_width"].append(prec)
                    say(f"w={w:g} pruned: kept {_stats(pr.n_keep)} F={F_pruned:.12g}")
                    del pr
                    F_cmp = F_pruned
                else:
                    F_cmp = F_new
                if dq < settings.tol_q:
                    end = "tol_q"
                elif dF < settings.tol_free_energy_rel * abs(F_new):
                    end = "tol_free_energy"
                if end != "max_sweeps":
                    break
                F_old = F_cmp
            del mom
            used += len(fr)
            fl.extend(fr)
            if not recentre_at_edge:
                break
            assert ci is not None
            ec = edge_counts(st, ci, settings)
            at_edge = ec["tau_at_kept_edge"] > 0 or ec["dir_at_ring"] > 0
            nxt = "repeat" if at_edge and used < settings.max_sweeps else "stop"
            log["edge_repeats"].append(
                dict(
                    width=float(w),
                    repeat=rep,
                    sweeps=len(fr),
                    sweeps_of_width=used,
                    end_reason=end,
                    free_energy_start=F_start,
                    free_energy_end=F_end,
                    free_energy=fr,
                    pruning=prec,
                    edge_counts=ec,
                    at_edge=bool(at_edge),
                    next=nxt,
                )
            )
            say(
                f"w={w:g} repeat {rep}: {len(fr)} sweeps ({end}), F={F_end:.12g}, "
                f"tau_at_kept_edge={ec['tau_at_kept_edge']}, "
                f"dir_at_ring={ec['dir_at_ring']}: {nxt}"
            )
            if nxt == "stop":
                break
            rep += 1
        log["free_energy"].append(fl)
        log["sweeps"].append(used)
        log["end_reason"].append(end)
        log["terms"].append(dict(width=float(w), free_energy=F_end, **terms))
        log["time_s"].append(time.perf_counter() - t0)
        say(f"w={w:g} done: {used} sweeps ({end}), F={F_end:.12g}")
    assert st is not None and ci is not None
    # the outputs do not use the profiles: free their buffer first
    st = replace(st, profiles=np.zeros((n, st.profiles.shape[1], 0, 2)))
    n_kept = st.valid.sum(1).astype(np.int64)
    field = _field(grid, st, n_kept, settings.prune_max)
    log["pruning"]["final"] = edge_counts(st, ci, settings)
    log["pruning"]["final"]["kept"] = _stats(n_kept)
    ci_out = np.full((n, settings.prune_max), -1, np.int16)
    kc = min(st.K, settings.prune_max)
    ci_out[:, :kc] = ci[:, :kc]
    extras: dict[str, npt.NDArray[Any]] = dict(
        u=st.u.copy(),
        speed=np.exp(st.u),
        p_jump=p_jump(st, ga, tw) if anchor is None else np.full(n, np.nan),
        tau_pred=np.asarray(tau_pred, np.float64),
        dir_pred=np.asarray(dir_pred, np.float64),
        cand_index=ci_out,
    )
    log["time_total_s"] = time.perf_counter() - t_all
    return field, extras, log


def _new_log(
    settings: T.SolverSettings,
    noise_sd: Sequence[float],
    tau_prior: bool,
    profile_dtype: Any,
    grid: T.Grid,
    n: int,
) -> dict[str, Any]:
    """The log of a solve (JSON-able): free_energy (per width, per sweep, after the
    sweep; the first sweep's value is before the pruning), sweeps (per width),
    end_reason (per width: tol_q, tol_free_energy or max_sweeps), free_energy_rises
    (width, sweep, f_before, f_after), defects (text), pruning (per_width: the kept
    counts min, median, max, the count at prune_max, the largest removed mass, F before
    and after; final: `edge_counts` and the kept counts), terms (per width at its end:
    each expected term, the entropy and F), time_s (per width), time_total_s, settings
    (SolverSettings.record()) and run (noise_sd, tau_prior, profile_dtype, grid kind,
    spacing, nodes, rise_rel; `solve` adds tensor_not_finite, origin, recentre_at_edge,
    anchor_node and anchor_position_vox).

    With recentre_at_edge (CD-13) on: free_energy and sweeps of a width hold all its
    passes in order (F can jump between two passes); end_reason is that of the last
    pass; pruning.per_width is of the first pass; terms is of the last pass; a rise
    also holds its pass ("repeat"); edge_repeats holds one record per pass (width,
    repeat, sweeps, sweeps_of_width, end_reason, free_energy_start after the
    carry-over, free_energy_end, free_energy per sweep, pruning, edge_counts, at_edge,
    next: "repeat" or "stop"). With the option off, edge_repeats is empty."""
    return dict(
        free_energy=[],
        sweeps=[],
        end_reason=[],
        free_energy_rises=[],
        defects=[],
        edge_repeats=[],
        pruning=dict(per_width=[], final={}),
        terms=[],
        time_s=[],
        time_total_s=float("nan"),
        settings=settings.record(),
        run=dict(
            noise_sd=[float(s) for s in noise_sd],
            tau_prior=bool(tau_prior),
            profile_dtype=str(np.dtype(profile_dtype)),
            grid_kind=str(grid.kind),
            spacing_mm=grid_spacing_mm(grid),
            n_nodes=int(n),
            rise_rel=RISE_REL,
        ),
    )
