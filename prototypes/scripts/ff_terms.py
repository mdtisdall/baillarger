"""The terms of the field form on a grid (docs/plans/flow-field-phantoms.md, *Terms on
the grid*, PR 3 task 3.2). The spec of PR 3 makes each term concrete; this module
follows it.

Each node has a distribution q over its moving candidates (τ, d) and the stopped state
(τ = TAU_TOP, no direction). This module gives:

- the moving candidates of *Node states* (`moving_candidates`) and the profiles of the
  contrast match (`smoothed_echoes`, `candidate_profiles`, `contrast_sigma`,
  `level_base`);
- the unary terms (stop, alignment) per state (`unary`);
- the jump prior (`jump_energy`). A negative jump is not a state: J = max(0, −reach)
  is never negative;
- the energy of given joint states (`state_energy`), written directly from the terms;
- the expectation of each term under q (`expected_energy`);
- the mean-field messages (`messages`): the partial derivatives of the expected coupling
  energy with respect to the q of each state of the selected nodes;
- the update of the log speed u (`update_u`).

Every coupling term is multilinear in the node distributions: each term is a product of
functions of the states of distinct nodes. So the expected energy is a polynomial that
is linear in the q of each node, and the message of a state is the expected energy of
the terms of that node with the node fixed in that state. The code uses the unnormalised
moments of each node (p = Σ q, Σ q τ, Σ q τ², Σ q d, Σ q P, Σ q P²), not the conditional
moments, so no division by p occurs and an expectation is exact also where p = 0.

Width w: the energies of items 3 to 7 of the spec are divided by w². The unary terms and
the jump are not.

NumPy or Torch (the plan's *NumPy or Torch*)
--------------------------------------------

Symbols: n nodes, K candidate slots (63 before the first pruning, then at most 32), L =
9 levels, 2 echoes, 6 axis edges per node, up to 20 direction pairs per node (the
off-diagonal entries of the Q1 stiffness that are negative), 8 octants per node. One
sweep updates each node once (8 colours).

Operations per node and sweep (estimates from the loop structure, not measured):

- consistency, own octants: 8 octants × K × about 40 flops;
- consistency, the octants of the 6 axis neighbours that hold the node: 24 × K × about
  50 flops (the candidates of the neighbour are summed to 3 coefficients of a quadratic
  in τ first, so no K × K array occurs);
- contrast: 6 edges × K × L × 2 × about 4 flops;
- bending: 20 pairs × K × about 8 flops; jump: 6 edges × 2 × K × about 20 flops.

In total about 2.4·10³ · K flops per node. Phantom size (n ≤ 1.2·10⁶): about 1.8·10¹¹
flops per sweep at K = 63, 9·10¹⁰ at K = 32. Target size (n = 4·10⁶ to 4·10⁷, K ≤ 32):
3·10¹¹ to 3·10¹² flops per sweep.

Profiles (CD-7): the state holds one line of L + n_tau − 1 = 17 levels per direction
(7 lines, `profile_lines`), 238 values per node, and each slot reads its 9 levels from
its line (`candidate_profile_rows`). The form before CD-7 held 63 × 9 × 2 = 1134 values
per node. The messages and refresh_moments gather the (chunk, K, L, 2) profiles of a
chunk from the lines; this gather is a temporary, not a state.

Memory traffic per sweep: the lines are read once for the own contrast messages and
once to refresh the moments: 2 · n · 238 · b bytes (b = 4 in float32), about 2.3 GB at
n = 1.2·10⁶. The gathered chunk profiles (n · K · 18 values in all) are then read again
for each of the 6 edge slots (P and P²). The other node arrays are smaller. NumPy also
writes about 10 temporary (m, K) arrays per octant combination (32 combinations); by
estimate this traffic, not the flops, sets the time of a NumPy sweep.

Memory: the messages, expected_energy, refresh_moments and update_u process the nodes
(and expected_energy the edges) in chunks (`chunk`, default 4096), so the temporaries
are O(chunk · K · L), no full (n, K) copy occurs and no n × K × K array occurs. The
state itself is the large item: the lines take 952 bytes per node in float32 (38 GB at
4·10⁷ nodes) and the candidate arrays of K = 32 slots about 1.4 kB per node (58 GB). At
the target size the state cannot be stored on this machine; that is a design question
for the solver (blocks with a halo), not a question of the backend.

Loop structure and regularity: Python loops only over fixed small sets (8 octants, 6 × 4
neighbour octants, the padded edge and pair slots of each node) and over chunks. Inside,
every operation is a gather by a fixed neighbour table, element-wise arithmetic or a
small dense contraction over (L, 2) or 3. The work is regular and data-parallel. The
sums over the edges of a node are over padded slots (a dense sum), not a scatter, so a
Torch version needs no index_add_, no scatter_add_ and no sparse tensors (except the
products with Q_u and Q_tau, which are row-sliced CSR products here and would need a
padded form of up to 27 neighbours in Torch).

Choice: NumPy. The measurements and the reasons are in the module docstring of
ff_meanfield.py (*NumPy or Torch*): on the CPU in float64 Torch gave 1.3 to 1.6 times
on the largest kernel; MPS gave about 5 to 6 times, but only in float32, and float32
cannot resolve the stop rule of the free energy.
"""

from __future__ import annotations

import itertools
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402

Array = T.Array
IntArray = T.IntArray
BoolArray = T.BoolArray

N_LEVELS = 9  # L: the levels of each profile
CHUNK = 4096  # nodes per chunk in the messages
OCTANTS: tuple[tuple[int, int, int], ...] = tuple(
    (int(a), int(b), int(c)) for a, b, c in itertools.product((1, -1), repeat=3)
)
U_RANGE = (-6.0, 6.0)  # the search range of update_u (log speed)


def _col(axis: int, sign: int) -> int:
    """The column of nbr6 for the neighbour along `axis` on the side `sign` (+x, −x, +y,
    ...)."""
    return 2 * axis + (0 if sign > 0 else 1)


def _softplus(x: Array) -> Array:
    return np.asarray(np.logaddexp(0.0, x))


# ---------------------------------------------------------------- the grid arrays


def _incidence(pairs: IntArray, n: int) -> tuple[IntArray, IntArray]:
    """(slot (n, D), other (n, D)): for each node the ids of the pairs that hold it and
    the other node of each pair, padded with −1. Sorted by pair id (deterministic)."""
    P = len(pairs)
    if P == 0:
        return np.full((n, 1), -1, np.int64), np.full((n, 1), -1, np.int64)
    pid = np.concatenate([np.arange(P), np.arange(P)]).astype(np.int64)
    ends = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
    other = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int64)
    order = np.lexsort((pid, ends))
    pid, ends, other = pid[order], ends[order], other[order]
    deg = np.bincount(ends, minlength=n)
    D = max(int(deg.max()), 1)
    start = np.cumsum(deg) - deg
    slot = np.arange(2 * P) - start[ends]
    s = np.full((n, D), -1, np.int64)
    o = np.full((n, D), -1, np.int64)
    s[ends, slot] = pid
    o[ends, slot] = other
    return s, o


@dataclass
class GridArrays:
    """The arrays of a grid that the terms use (ff_meanfield builds it with the ff_grid
    helpers).

    nodes (n, 3) voxel positions; nbr6 (n, 6) the node ids at +x, −x, +y, −y, +z, −z (−1
    where none); edges (E, 2) with edge_vec_mm (E, 3) = x_j − x_i in mm; node_w (n,),
    edge_w (E,), face_w (E,) in voxel units; dir_pairs (P, 2) and dir_pair_w (P,) =
    −K_ij > 0 of the Q1 stiffness, each pair once; Q_u the FEM precision on u (CSR, n ×
    n); Q_tau the FEM precision on τ (CSR or None), used only if TermWeights.tau_prior
    is True.

    The fields after Q_tau are derived in __post_init__: h_axis (n, 6) the distance in
    mm to each axis neighbour (nan where none), n_oct (n,) the count of octants whose 3
    neighbours exist, and the padded incidence of the edges and of the direction pairs.
    """

    nodes: Array
    nbr6: IntArray
    edges: IntArray
    edge_vec_mm: Array
    node_w: Array
    edge_w: Array
    face_w: Array
    dir_pairs: IntArray
    dir_pair_w: Array
    Q_u: Any
    Q_tau: Any = None
    h_axis: Array = field(init=False)
    n_oct: IntArray = field(init=False)
    inc_edge: IntArray = field(init=False)
    inc_edge_nbr: IntArray = field(init=False)
    inc_edge_sign: Array = field(init=False)
    inc_pair: IntArray = field(init=False)
    inc_pair_nbr: IntArray = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.nodes)
        assert self.nbr6.shape == (n, 6)
        h = np.full((n, 6), np.nan)
        for c in range(6):
            j = self.nbr6[:, c]
            ok = j >= 0
            h[ok, c] = (
                np.abs(self.nodes[j[ok], c // 2] - self.nodes[ok, c // 2]) * T.VOXEL_MM
            )
        self.h_axis = h
        n_oct = np.zeros(n, np.int64)
        for o in OCTANTS:
            cols = [_col(a, o[a]) for a in range(3)]
            n_oct += (self.nbr6[:, cols] >= 0).all(1)
        self.n_oct = n_oct
        self.inc_edge, self.inc_edge_nbr = _incidence(self.edges, n)
        e = np.where(self.inc_edge >= 0, self.inc_edge, 0)
        first = self.edges[e, 0] == np.arange(n)[:, None]
        self.inc_edge_sign = np.where(
            self.inc_edge >= 0, np.where(first, 1.0, -1.0), 0.0
        )
        self.inc_pair, self.inc_pair_nbr = _incidence(self.dir_pairs, n)

    @property
    def n(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------- the state and the
# weights


@dataclass
class TermState:
    """The node states and the shared state, for n nodes and K candidate slots.

    cand_tau (n, K), cand_dir (n, K, 3) unit, valid (n, K): the moving candidates. q (n,
    K) and q_stop (n,): the distributions (q = 0 on slots that are not valid). u (n,):
    the log speed, a point estimate per node. s_ref (n,): the slowness frozen at the
    start of a width (profiles, jump). stop_log_odds (n,), tensor_dir (n, 3), coherence
    (n,): the inputs at the nodes.

    The profiles have two forms. The profile of slot k is always given at the absolute
    levels level_base + 0 .. L − 1 (`candidate_profile_rows` reads it).

    - Direct form (prof_line None): profiles (n, K, L, 2), one profile per slot.
    - Line form (CD-7): profiles (n, M, L_line, 2), one line per direction (M = n_dir,
      L_line = L + n_tau − 1, from `profile_lines`). Slot k reads line prof_line[i, k]
      at the line indices prof_off[i, k] + 0 .. L − 1 (`line_slots`). L = n_levels.
    """

    cand_tau: Array
    cand_dir: Array
    valid: BoolArray
    q: Array
    q_stop: Array
    u: Array
    s_ref: Array
    profiles: Array
    level_base: IntArray
    stop_log_odds: Array
    tensor_dir: Array
    coherence: Array
    prof_line: npt.NDArray[np.int8] | None = None
    prof_off: npt.NDArray[np.int8] | None = None
    n_levels: int = N_LEVELS

    def __post_init__(self) -> None:
        n, K = self.cand_tau.shape
        assert self.cand_dir.shape == (n, K, 3) and self.valid.shape == (n, K)
        assert self.q.shape == (n, K) and self.q_stop.shape == (n,)
        assert self.profiles.shape[0] == n and self.profiles.shape[3] == 2
        if self.prof_line is None:
            assert self.prof_off is None and self.profiles.shape[1] == K
        else:
            assert self.prof_off is not None
            assert self.prof_line.shape == (n, K) and self.prof_off.shape == (n, K)

    @property
    def n(self) -> int:
        return int(self.cand_tau.shape[0])

    @property
    def K(self) -> int:
        return int(self.cand_tau.shape[1])

    @property
    def L(self) -> int:
        """The levels of the profile of one slot."""
        if self.prof_line is None:
            return int(self.profiles.shape[2])
        return int(self.n_levels)


@dataclass(frozen=True)
class TermWeights:
    """The weights of the terms. sigma_c: consistency σ (τ per mm); sigma_e (2,):
    contrast σ per echo; sigma_align: σ of the alignment (sin 15°); lambda_dir: λ of the
    direction prior; tau_prior: the switch of the FEM prior on τ; w: the width (4, 2 or
    1); jump_p_zero and jump_mean: the jump prior."""

    sigma_c: float
    sigma_e: tuple[float, float]
    sigma_align: float
    lambda_dir: float
    tau_prior: bool = False
    w: float = 1.0
    jump_p_zero: float = T.DEFAULTS.jump_p_zero
    jump_mean: float = T.DEFAULTS.jump_mean


def term_weights(
    spacing_mm: float,
    noise_sd: Sequence[float],
    w: float,
    settings: T.SolverSettings = T.DEFAULTS,
    tau_prior: bool = False,
) -> TermWeights:
    """The weights of the spec: σ_c = Δτ / h, σ_e = contrast_sigma, σ_a = sin 15°,
    λ_d = 1 / (sin² 15° · ℓ)."""
    sa = math.sin(math.radians(settings.dir_cone_deg))
    se = contrast_sigma(noise_sd, settings.band_scale_vox)
    return TermWeights(
        sigma_c=settings.dtau / spacing_mm,
        sigma_e=(float(se[0]), float(se[1])),
        sigma_align=sa,
        lambda_dir=1.0 / (sa * sa * settings.fem_length_mm),
        tau_prior=tau_prior,
        w=float(w),
        jump_p_zero=settings.jump_p_zero,
        jump_mean=settings.jump_mean,
    )


# ---------------------------------------------------------------- candidates and
# profiles


def moving_candidates(
    tau_pred: Array, dir_pred: Array, settings: T.SolverSettings = T.DEFAULTS
) -> tuple[Array, Array, BoolArray]:
    """The moving candidates of each node: (cand_tau (n, 63), cand_dir (n, 63, 3), valid
    (n, 63)).

    τ index a = 0 .. 8: τ = c + (a − 4) Δτ, c = min(τ_pred, TAU_TOP). Direction index m
    = 0 .. 6: m = 0 is the predicted direction p; m = 1 .. 6 are at 15° from p, at the
    azimuth 60° (m − 1) in the basis u = cross(p, e) / |cross(p, e)|, w = cross(p, u),
    with e the voxel axis least aligned with p (the first such axis on a tie). Candidate
    c = a · 7 + m. A candidate with τ > TAU_TOP is not valid. A τ within 1e-9 Δτ above
    TAU_TOP (rounding) is valid and set to TAU_TOP.
    """
    tau_pred = np.asarray(tau_pred, float)
    n = len(tau_pred)
    na, nm = settings.n_tau, settings.n_dir
    c = np.minimum(tau_pred, T.TAU_TOP)
    a = np.arange(na) - (na - 1) // 2
    tau = c[:, None] + a[None, :] * settings.dtau  # (n, na)
    valid_a = tau <= T.TAU_TOP + 1e-9 * settings.dtau
    tau = np.where(valid_a, np.minimum(tau, T.TAU_TOP), tau)
    p = np.asarray(dir_pred, float)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    e = np.zeros((n, 3))
    e[np.arange(n), np.argmin(np.abs(p), axis=1)] = 1.0
    u = np.cross(p, e)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    wv = np.cross(p, u)
    th = math.radians(settings.dir_cone_deg)
    dirs = [p]
    for m in range(1, nm):
        az = 2.0 * math.pi * (m - 1) / (nm - 1)
        dirs.append(
            math.cos(th) * p + math.sin(th) * (math.cos(az) * u + math.sin(az) * wv)
        )
    D = np.stack(dirs, axis=1)  # (n, nm, 3)
    cand_tau = np.repeat(tau, nm, axis=1)  # c = a · nm + m
    cand_dir = np.tile(D, (1, na, 1))
    valid = np.repeat(valid_a, nm, axis=1)
    return cand_tau, cand_dir, valid


def level_base(
    tau_pred: Array, settings: T.SolverSettings = T.DEFAULTS, L: int = N_LEVELS
) -> IntArray:
    """The first absolute level of each node's profiles: rint(c / Δτ) − (L − 1) / 2,
    c = min(τ_pred, TAU_TOP). np.rint rounds a half to the even integer."""
    c = np.minimum(np.asarray(tau_pred, float), T.TAU_TOP)
    return np.asarray(np.rint(c / settings.dtau) - (L - 1) // 2, dtype=np.int64)


def _gauss_sigma(band_scale_vox: float) -> float:
    return band_scale_vox / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def smoothed_echoes(
    det: Mapping[int, Array], band_scale_vox: float
) -> tuple[Array, Array]:
    """The two detrended echoes (det[1]: echo 2, det[2]: echo 3), each smoothed by a
    Gaussian of FWHM band_scale_vox (scipy.ndimage.gaussian_filter, mode "nearest")."""
    s = _gauss_sigma(band_scale_vox)
    v1 = ndi.gaussian_filter(np.asarray(det[1], float), s, mode="nearest")
    v2 = ndi.gaussian_filter(np.asarray(det[2], float), s, mode="nearest")
    return np.asarray(v1), np.asarray(v2)


def contrast_sigma(noise_sd: Sequence[float], band_scale_vox: float) -> Array:
    """(2,) the contrast σ per echo: the noise SD times the noise reduction of the
    smoothing Gaussian of `smoothed_echoes` on white noise (the root of the sum of the
    squared weights of its 3D kernel, away from the box edges)."""
    s = _gauss_sigma(band_scale_vox)
    r = int(4.0 * s + 0.5) + 2
    delta = np.zeros((2 * r + 1,) * 3)
    delta[r, r, r] = 1.0
    k = ndi.gaussian_filter(delta, s, mode="constant")
    f = float(np.sqrt((k * k).sum()))
    return np.asarray(noise_sd, float) * f


def candidate_profiles(
    nodes: Array,
    cand_tau: Array,
    cand_dir: Array,
    s_ref: Array,
    level_base: IntArray,
    volumes: Sequence[Array],
    L: int = N_LEVELS,
    dtau: float = T.DEFAULTS.dtau,
    chunk: int = CHUNK,
) -> Array:
    """(n, K, L, 2) the profile of each candidate: the two volumes sampled
    (T.sample_volume) at x_i + d (ℓ Δτ − τ) / (s_ref_i · VOXEL_MM), ℓ = level_base_i + 0
    .. L − 1. A candidate with a τ or a direction that is not finite is sampled at the
    node."""
    n, K = cand_tau.shape
    out = np.empty((n, K, L, len(volumes)))
    lev = np.arange(L)
    for st in range(0, n, chunk):
        sl = slice(st, min(st + chunk, n))
        m = sl.stop - sl.start
        ell = (level_base[sl][:, None] + lev[None, :]) * dtau  # (m, L)
        dist = (ell[:, None, :] - cand_tau[sl][:, :, None]) / (
            s_ref[sl][:, None, None] * T.VOXEL_MM
        )
        X = nodes[sl][:, None, None, :] + cand_dir[sl][:, :, None, :] * dist[..., None]
        X = np.where(
            np.isfinite(X), X, np.broadcast_to(nodes[sl][:, None, None, :], X.shape)
        )
        Xf = X.reshape(-1, 3)
        for e, vol in enumerate(volumes):
            out[sl, :, :, e] = T.sample_volume(vol, Xf).reshape(m, K, L)
    return out


def line_slots(
    n_tau: int = T.DEFAULTS.n_tau, n_dir: int = T.DEFAULTS.n_dir
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """(line (n_tau · n_dir,), off (n_tau · n_dir,)) int8: the line and the first line
    index of each moving candidate c = a · n_dir + m (`moving_candidates`) in the line
    form of `profile_lines`: line = m, off = (n_tau − 1) − a."""
    c = np.arange(n_tau * n_dir)
    line = (c % n_dir).astype(np.int8)
    off = ((n_tau - 1) - c // n_dir).astype(np.int8)
    return line, off


def profile_lines(
    nodes: Array,
    c: Array,
    dirs: Array,
    s_ref: Array,
    level_base: IntArray,
    volumes: Sequence[Array],
    L: int = N_LEVELS,
    n_tau: int = T.DEFAULTS.n_tau,
    dtau: float = T.DEFAULTS.dtau,
    chunk: int = CHUNK,
) -> Array:
    """(n, M, L + n_tau − 1, 2) the profile lines of CD-7: one line per direction m of
    dirs (n, M, 3). Line index t samples the volumes (T.sample_volume) at
    x_i + d_m ((level_base_i + t − h) Δτ − c_i) / (s_ref_i · VOXEL_MM), h = (n_tau − 1)
    / 2, with c_i = min(τ_pred_i, TAU_TOP) the centre of the τ candidates. A point that
    is not finite is sampled at the node.

    The candidate (a, m) of `moving_candidates` has τ = c_i + (a − h) Δτ. Its profile
    at level ℓ = level_base_i + l (l = 0 .. L − 1) is at the distance (ℓ Δτ − τ) =
    ((level_base_i + l − a + h) Δτ − c_i), that is line index t = l + (n_tau − 1) − a
    of line m (`line_slots`). The τ offsets and the levels are both integer multiples
    of Δτ, so this map is exact for every c_i; c_i need not be on the level lattice.
    The two forms differ only by rounding: the product (ℓ − a + h) Δτ is rounded in a
    different order, and `moving_candidates` sets a τ within 1e-9 Δτ above TAU_TOP to
    TAU_TOP (a shift of at most 1e-9 Δτ of the sample point along d, in τ units).
    """
    n_tau = int(n_tau)
    assert n_tau % 2 == 1
    h = (n_tau - 1) // 2
    n, M = dirs.shape[:2]
    Lt = L + n_tau - 1
    out = np.empty((n, M, Lt, len(volumes)))
    t = np.arange(Lt)
    for st in range(0, n, chunk):
        sl = slice(st, min(st + chunk, n))
        m = sl.stop - sl.start
        ell = (level_base[sl][:, None] + t[None, :] - h) * dtau  # (m, Lt)
        dist = (ell - c[sl][:, None]) / (s_ref[sl][:, None] * T.VOXEL_MM)
        X = (
            nodes[sl][:, None, None, :]
            + dirs[sl][:, :, None, :] * dist[:, None, :, None]
        )
        X = np.where(
            np.isfinite(X), X, np.broadcast_to(nodes[sl][:, None, None, :], X.shape)
        )
        Xf = X.reshape(-1, 3)
        for e, vol in enumerate(volumes):
            out[sl, :, :, e] = T.sample_volume(vol, Xf).reshape(m, M, Lt)
    return out


# ---------------------------------------------------------------- the jump


def jump_energy(
    tau: Array, dir: Array, s_ref: Array, delta_mm: Array, settings: Any = T.DEFAULTS
) -> tuple[Array, Array]:
    """(energy, J) of the jump (spec item 8), broadcast over the leading axes.

    tau (...), dir (..., 3), s_ref (...), delta_mm (..., 3): the edge vector from the
    moving node to the stopped node in mm. reach = τ + s_ref · max(d · δ, 0); J = max(0,
    −reach). Energy: −log p0 if J = 0, else −log(1 − p0) + log(jump_mean) + J /
    jump_mean. `settings` is a SolverSettings or a TermWeights (fields jump_p_zero and
    jump_mean). Not scaled by face_w."""
    p0 = float(settings.jump_p_zero)
    mean = float(settings.jump_mean)
    dd = np.sum(np.asarray(dir) * np.asarray(delta_mm), axis=-1)
    reach = np.asarray(tau) + np.asarray(s_ref) * np.maximum(dd, 0.0)
    J = np.maximum(0.0, -reach)
    e0 = -math.log(p0)
    e1 = -math.log(1.0 - p0) + math.log(mean)
    E = np.where(J > 0.0, e1 + J / mean, e0)
    return np.asarray(E, float), np.asarray(J, float)


# ---------------------------------------------------------------- moments


@dataclass
class Moments:
    """The unnormalised moments of each node's moving states: p = Σ q, s1 = Σ q τ, s2 =
    Σ q τ², dsum = Σ q d (n, 3), m1 = Σ q P and m2 = Σ q P² (n, L, 2)."""

    p: Array
    s1: Array
    s2: Array
    dsum: Array
    m1: Array
    m2: Array


def _cand(st: TermState, ids: IntArray) -> tuple[Array, Array, Array]:
    """(τ, d, q) of the nodes ids, with 0 on the slots that are not valid."""
    v = st.valid[ids]
    tau = np.where(v, st.cand_tau[ids], 0.0)
    d = np.where(v[..., None], st.cand_dir[ids], 0.0)
    q = np.where(v, st.q[ids], 0.0)
    return tau, d, q


def candidate_profile_rows(st: TermState, ids: IntArray, slots: IntArray) -> Array:
    """The profiles (S..., L, 2) of the slots `slots` of the nodes `ids` (two integer
    arrays that broadcast to the shape S). In the line form, a gather from the lines:
    no array of all slots of all nodes occurs."""
    if st.prof_line is None:
        return np.asarray(st.profiles[ids, slots])
    assert st.prof_off is not None
    n, M, Lt, E = st.profiles.shape
    ids_ = np.asarray(ids, np.intp)
    # the flat row of (node, line, line index) in the profiles seen as (n · M · Lt, E)
    row = (ids_ * M + st.prof_line[ids, slots]) * Lt + st.prof_off[ids, slots]
    flat = st.profiles.reshape(n * M * Lt, E)
    return np.asarray(np.take(flat, row[..., None] + np.arange(st.L), axis=0))


def _prof(st: TermState, ids: IntArray) -> Array:
    P = candidate_profile_rows(st, ids[:, None], np.arange(st.K)[None, :])
    return np.where(st.valid[ids][..., None, None], P, 0.0)


def node_moments(st: TermState, chunk: int = CHUNK) -> Moments:
    """The moments of all nodes."""
    n = st.n
    mom = Moments(
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros((n, 3)),
        np.zeros((n, st.L, 2)),
        np.zeros((n, st.L, 2)),
    )
    for s in range(0, n, chunk):
        refresh_moments(mom, st, np.arange(s, min(s + chunk, n)))
    return mom


def refresh_moments(
    mom: Moments, st: TermState, ids: IntArray, chunk: int = CHUNK
) -> None:
    """Recompute the moments of the nodes ids in place (the mean field calls it after it
    updates one colour). The only function of this module that changes its argument.
    The nodes go in chunks of `chunk`, so the float64 copies of the profiles are
    O(chunk · K · L)."""
    ids = np.asarray(ids, np.int64)
    for s0 in range(0, len(ids), chunk):
        c = ids[s0 : s0 + chunk]
        tau, d, q = _cand(st, c)
        P = _prof(st, c)
        mom.p[c] = q.sum(1)
        mom.s1[c] = (q * tau).sum(1)
        mom.s2[c] = (q * tau * tau).sum(1)
        mom.dsum[c] = np.einsum("nk,nkc->nc", q, d)
        mom.m1[c] = np.einsum("nk,nkle->nle", q, P)
        mom.m2[c] = np.einsum("nk,nkle->nle", q, P * P)


# ---------------------------------------------------------------- unary terms


def unary(
    st: TermState, ga: GridArrays, tw: TermWeights, ids: IntArray | None = None
) -> tuple[Array, Array]:
    """(E_move (n, K), E_stop (n,)): the stop term (item 1) and the alignment (item 2).
    With ids: the rows of these nodes only ((len(ids), K), (len(ids),)).

    E_stop = node_w softplus(−lo); E_move = node_w [softplus(lo) + coh (1 − (d · e)²) /
    (2 σ_a²)].
    """
    sl: Any = slice(None) if ids is None else np.asarray(ids, np.int64)
    lo = st.stop_log_odds[sl]
    nw = ga.node_w[sl]
    e_stop = nw * _softplus(-lo)
    cos = np.einsum("nkc,nc->nk", st.cand_dir[sl], st.tensor_dir[sl])
    align = (
        (nw * st.coherence[sl])[:, None] * (1.0 - cos * cos) / (2.0 * tw.sigma_align**2)
    )
    e_move = (nw * _softplus(lo))[:, None] + align
    return np.asarray(e_move), np.asarray(e_stop)


# ---------------------------------------------------------------- consistency


def _cons_coef(ga: GridArrays, tw: TermWeights, ids: IntArray) -> Array:
    """node_w / (n_oct · 2 σ_c² w²), 0 where the node has no octant."""
    no = ga.n_oct[ids]
    c = ga.node_w[ids] / (np.maximum(no, 1) * 2.0 * tw.sigma_c**2 * tw.w**2)
    return np.asarray(np.where(no > 0, c, 0.0))


def _rows(ok: BoolArray) -> Any:
    """The rows where ok: slice(None) if all rows are (a view, no copy), else their
    indices."""
    return slice(None) if ok.all() else np.nonzero(ok)[0]


def _octant_poly(
    Z0: list[Array], Z1: list[Array], Z2: list[Array]
) -> tuple[Array, Array, Array]:
    """For 3 independent neighbours with Z0 = E[1], Z1 = E[1 y], Z2 = E[1 y²] (1 = the
    neighbour moves): E[Π 1 (Σ y − s)²] = F0 − 2 s F1 + s² F2. Returns (F0, F1, F2)."""
    a, b, c = Z0
    F0 = (
        Z2[0] * b * c
        + Z2[1] * a * c
        + Z2[2] * a * b
        + 2.0 * (Z1[0] * Z1[1] * c + Z1[0] * Z1[2] * b + Z1[1] * Z1[2] * a)
    )
    F1, F2 = _octant_poly_f1_f2(Z0, Z1)
    return F0, F1, F2


def _octant_poly_f1_f2(Z0: list[Array], Z1: list[Array]) -> tuple[Array, Array]:
    """(F1, F2) of `_octant_poly` (they do not use Z2)."""
    a, b, c = Z0
    return Z1[0] * b * c + Z1[1] * a * c + Z1[2] * a * b, a * b * c


def _own_octants(
    ga: GridArrays,
    mom: Moments,
    ids: IntArray,
    tau: Array,
    dirs: Array,
    need_f0: bool = True,
) -> tuple[Array, Array, Array]:
    """For each node of ids in each of its candidates (τ (m, K), d (m, K, 3)): the sums
    over its octants of (F0, F1, F2) of `_octant_poly` ((m, K), (m, K), (m,)). The
    expected own consistency energy of the candidate is coef (F0 − 2 s F1 + s² F2).
    With need_f0 False, F0 is not computed (it stays 0); F1 and F2 do not change."""
    m, K = tau.shape
    F0 = np.zeros((m, K))
    F1 = np.zeros((m, K))
    F2 = np.zeros(m)
    nb = ga.nbr6[ids]
    for o in OCTANTS:
        cols = [_col(a, o[a]) for a in range(3)]
        J = nb[:, cols]
        ok = (J >= 0).all(1)
        if not ok.any():
            continue
        r = _rows(ok)
        Z0: list[Array] = []
        Z1: list[Array] = []
        Z2: list[Array] = []
        t = tau[r]
        for a in range(3):
            j = J[r, a]
            h = ga.h_axis[ids[r], cols[a]]
            c = dirs[r, :, a] * o[a] / h[:, None]
            p = mom.p[j][:, None]
            s1 = mom.s1[j][:, None]
            Z0.append(p)
            Z1.append(c * (s1 - t * p))
            if need_f0:
                s2 = mom.s2[j][:, None]
                Z2.append(c * c * (s2 - 2.0 * t * s1 + t * t * p))
        if need_f0:
            f0, f1, f2 = _octant_poly(Z0, Z1, Z2)
            F0[r] += f0
        else:
            f1, f2 = _octant_poly_f1_f2(Z0, Z1)
        F1[r] += f1
        F2[r] += f2[:, 0]
    return F0, F1, F2


def _nbr_octant_messages(
    st: TermState,
    ga: GridArrays,
    tw: TermWeights,
    mom: Moments,
    ids: IntArray,
    tau_i: Array,
) -> Array:
    """(m, K): the derivative of the octant terms of the axis neighbours r that hold
    node i, with respect to q of each moving candidate of i. Node i is the neighbour of
    r along axis a on the side σ; the term is quadratic in τ_i, so the candidates of r
    sum to 3 coefficients first."""
    out = np.zeros_like(tau_i)
    nb = ga.nbr6[ids]
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        for sg in (1, -1):
            r_all = nb[:, _col(a, -sg)]
            has_r = r_all >= 0
            if not has_r.any():
                continue
            # the candidates of r: one gather for the 4 octants of r that hold i.
            # The factors that depend on one side only (of a, of b or of c) are
            # computed once on all rows of `base`; each octant takes its rows.
            base = np.nonzero(has_r)[0]
            RB = r_all[base]
            trB, drB, qrB = _cand(st, RB)
            srB = np.exp(-st.u[RB])[:, None]
            coefB = _cons_coef(ga, tw, RB)
            caB = drB[:, :, a] * sg / ga.h_axis[RB, _col(a, sg)][:, None]
            beB = -caB * trB
            wqB = qrB * coefB[:, None]
            wq_al_alB = wqB * caB * caB
            two_al_beB = 2.0 * caB * beB
            be_beB = beB * beB
            side = {}
            for ax, sx in ((b, 1), (b, -1), (c, 1), (c, -1)):
                jx = ga.nbr6[RB, _col(ax, sx)]
                jj = np.where(jx >= 0, jx, 0)
                cx = drB[:, :, ax] * sx / ga.h_axis[RB, _col(ax, sx)][:, None]
                px = mom.p[jj][:, None]
                s1x = mom.s1[jj][:, None]
                s2x = mom.s2[jj][:, None]
                Z1x = cx * (s1x - trB * px)
                Z2x = cx * cx * (s2x - 2.0 * trB * s1x + trB * trB * px)
                side[ax, sx] = (jx >= 0, px, Z1x, Z2x)
            for sb in (1, -1):
                for sc in (1, -1):
                    okb, pbB, Z1bB, Z2bB = side[b, sb]
                    okc, pcB, Z1cB, Z2cB = side[c, sc]
                    ok = okb & okc
                    if not ok.any():
                        continue
                    subb = _rows(ok)
                    sub = base[subb]
                    pb, Z1b, Z2b = pbB[subb], Z1bB[subb], Z2bB[subb]
                    pc, Z1c, Z2c = pcB[subb], Z1cB[subb], Z2cB[subb]
                    sr = srB[subb]
                    P = pb * pc
                    S = Z1b * pc + Z1c * pb
                    G = 2.0 * S - 2.0 * sr * P
                    H = (
                        Z2b * pc
                        + Z2c * pb
                        + 2.0 * Z1b * Z1c
                        - 2.0 * sr * S
                        + sr * sr * P
                    )
                    wq = wqB[subb]
                    be = beB[subb]
                    A2 = (wq_al_alB[subb] * P).sum(1)
                    A1 = (wq * (two_al_beB[subb] * P + caB[subb] * G)).sum(1)
                    A0 = (wq * (be_beB[subb] * P + be * G + H)).sum(1)
                    t = tau_i[sub]
                    out[sub] += A2[:, None] * t * t + A1[:, None] * t + A0[:, None]
    return out


# ---------------------------------------------------------------- the messages


def _shifted(
    arr: Array, lb_i: IntArray, lb_j: IntArray, L: int
) -> tuple[Array, BoolArray]:
    """arr (s, L, 2) of the nodes j in the level frame of the nodes i, and the mask of
    the shared levels (s, L)."""
    idx = np.arange(L)[None, :] + (lb_i - lb_j)[:, None]
    mask = (idx >= 0) & (idx < L)
    idxc = np.clip(idx, 0, L - 1)
    out = np.take_along_axis(arr, idxc[:, :, None], axis=1) * mask[:, :, None]
    return out, mask


def _shifted2(
    a1: Array, a2: Array, lb_i: IntArray, lb_j: IntArray, L: int
) -> tuple[Array, Array, BoolArray]:
    """`_shifted` of two arrays with the same level frames: (a1 shifted, a2 shifted,
    mask). The same values as two calls of `_shifted`."""
    idx = np.arange(L)[None, :] + (lb_i - lb_j)[:, None]
    mask = (idx >= 0) & (idx < L)
    idxc = np.clip(idx, 0, L - 1)[:, :, None]
    m3 = mask[:, :, None]
    o1 = np.take_along_axis(a1, idxc, axis=1) * m3
    o2 = np.take_along_axis(a2, idxc, axis=1) * m3
    return o1, o2, mask


def messages(
    st: TermState,
    ga: GridArrays,
    tw: TermWeights,
    sel: IntArray,
    mom: Moments | None = None,
    chunk: int = CHUNK,
) -> tuple[Array, Array]:
    """(M_move (len(sel), K), M_stop (len(sel),)): for each selected node, the
    derivative of the expected coupling energy (items 3, 4, 5, 7, 8) with respect to the
    q of each of its states, including the terms of the neighbours that hold the node.
    `mom`: the moments of all nodes (node_moments); computed here if None. The values of
    slots that are not valid have no use."""
    if mom is None:
        mom = node_moments(st)
    sel = np.asarray(sel, np.int64)
    K = st.K
    L = st.L
    M = np.zeros((len(sel), K))
    Ms = np.zeros(len(sel))
    w2 = tw.w**2
    inv_se = 1.0 / np.asarray(tw.sigma_e, float) ** 2
    diag_tau = ga.Q_tau.diagonal() if (tw.tau_prior and ga.Q_tau is not None) else None
    for s0 in range(0, len(sel), chunk):
        ids = sel[s0 : s0 + chunk]
        out = np.zeros((len(ids), K))
        out_s = np.zeros(len(ids))
        tau, d, _ = _cand(st, ids)
        # 3. consistency: own octants and the octants of the neighbours
        F0, F1, F2 = _own_octants(ga, mom, ids, tau, d)
        s = np.exp(-st.u[ids])[:, None]
        out += _cons_coef(ga, tw, ids)[:, None] * (
            F0 - 2.0 * s * F1 + s * s * F2[:, None]
        )
        out += _nbr_octant_messages(st, ga, tw, mom, ids, tau)
        # 4. contrast and 8. jump, over the edges of each node
        P = _prof(st, ids)
        P2 = P * P
        for k in range(ga.inc_edge.shape[1]):
            e_all = ga.inc_edge[ids, k]
            has = e_all >= 0
            if not has.any():
                continue
            sub = _rows(has)
            e = e_all[sub]
            i = ids[sub]
            j = ga.inc_edge_nbr[i, k]
            m1j, m2j, mask = _shifted2(
                mom.m1[j], mom.m2[j], st.level_base[i], st.level_base[j], L
            )
            nsh = mask.sum(1)
            W = np.where(nsh > 0, ga.edge_w[e] / (2.0 * w2 * np.maximum(nsh, 1)), 0.0)
            Wl = W[:, None, None] * mask[:, :, None] * inv_se[None, None, :]
            out[sub] += (
                np.einsum("skle,sle->sk", P2[sub], Wl * mom.p[j][:, None, None])
                - 2.0 * np.einsum("skle,sle->sk", P[sub], Wl * m1j)
                + (Wl * m2j).sum((1, 2))[:, None]
            )
            delta = ga.inc_edge_sign[i, k][:, None] * ga.edge_vec_mm[e]
            Ei, _ = jump_energy(
                tau[sub], d[sub], st.s_ref[i][:, None], delta[:, None, :], tw
            )
            out[sub] += (ga.face_w[e] * st.q_stop[j])[:, None] * Ei
            tj, dj, qj = _cand(st, j)
            Ej, _ = jump_energy(tj, dj, st.s_ref[j][:, None], -delta[:, None, :], tw)
            out_s[sub] += ga.face_w[e] * (qj * Ej).sum(1)
        # 5. the FEM prior on the direction
        for k in range(ga.inc_pair.shape[1]):
            pa = ga.inc_pair[ids, k]
            has = pa >= 0
            if not has.any():
                continue
            sub = _rows(has)
            j = ga.inc_pair_nbr[ids[sub], k]
            cw = tw.lambda_dir / (2.0 * w2) * ga.dir_pair_w[pa[sub]]
            dot = np.einsum("skc,sc->sk", d[sub], mom.dsum[j])
            out[sub] += cw[:, None] * (2.0 * mom.p[j][:, None] - 2.0 * dot)
        # 7. the FEM prior on τ
        if diag_tau is not None:
            off = np.asarray(ga.Q_tau[ids] @ mom.s1) - diag_tau[ids] * mom.s1[ids]
            out += (
                diag_tau[ids][:, None] * tau * tau / (2.0 * w2)
                + tau * off[:, None] / w2
            )
        M[s0 : s0 + len(ids)] = out
        Ms[s0 : s0 + len(ids)] = out_s
    return M, Ms


# ---------------------------------------------------------------- the expected energy


def expected_energy(
    st: TermState,
    ga: GridArrays,
    tw: TermWeights,
    chunk: int = CHUNK,
    mom: Moments | None = None,
) -> dict[str, float]:
    """The expectation under q of each term: stop, align, consistency, contrast,
    bending, speed, tau_prior, jump. The free energy is the sum of these minus the
    entropy of q. Written from the terms, not from `messages`.

    `mom`: the moments of all nodes (node_moments); computed here if None. The nodes
    and the edges go in chunks of `chunk`, so no (n, K) temporary occurs. The sums
    over the chunks are in a fixed order."""
    n = st.n
    w2 = tw.w**2
    if mom is None:
        mom = node_moments(st, chunk)
    lo = st.stop_log_odds
    out: dict[str, float] = {}
    out["stop"] = float(
        (ga.node_w * (mom.p * _softplus(lo) + st.q_stop * _softplus(-lo))).sum()
    )
    # 2. alignment and 3. consistency, by node chunks
    align = 0.0
    cons = 0.0
    for s0 in range(0, n, chunk):
        ids = np.arange(s0, min(s0 + chunk, n))
        tau, d, q = _cand(st, ids)
        e_move, _ = unary(st, ga, tw, ids)
        e_lo = (ga.node_w[ids] * _softplus(lo[ids]))[:, None]
        align += float((q * (e_move - e_lo)).sum())
        F0, F1, F2 = _own_octants(ga, mom, ids, tau, d)
        s = np.exp(-st.u[ids])[:, None]
        val = F0 - 2.0 * s * F1 + s * s * F2[:, None]
        cons += float((_cons_coef(ga, tw, ids)[:, None] * q * val).sum())
    out["align"] = align
    out["consistency"] = cons
    # 4. contrast, 8. jump: over the edges
    inv_se = 1.0 / np.asarray(tw.sigma_e, float) ** 2
    con = 0.0
    jump = 0.0
    E = len(ga.edges)
    for s0 in range(0, E, chunk):
        es = np.arange(s0, min(s0 + chunk, E))
        i, j = ga.edges[es, 0], ga.edges[es, 1]
        m1j, m2j, mask = _shifted2(
            mom.m1[j], mom.m2[j], st.level_base[i], st.level_base[j], st.L
        )
        nsh = mask.sum(1)
        W = np.where(nsh > 0, ga.edge_w[es] / (2.0 * w2 * np.maximum(nsh, 1)), 0.0)
        Wl = W[:, None, None] * mask[:, :, None] * inv_se[None, None, :]
        con += float(
            (
                Wl
                * (
                    mom.p[j][:, None, None] * mom.m2[i]
                    - 2.0 * mom.m1[i] * m1j
                    + mom.p[i][:, None, None] * m2j
                )
            ).sum()
        )
        ti, di, qi = _cand(st, i)
        tj, dj, qj = _cand(st, j)
        delta = ga.edge_vec_mm[es][:, None, :]
        Ei, _ = jump_energy(ti, di, st.s_ref[i][:, None], delta, tw)
        Ej, _ = jump_energy(tj, dj, st.s_ref[j][:, None], -delta, tw)
        jump += float(
            (
                ga.face_w[es]
                * ((qi * Ei).sum(1) * st.q_stop[j] + (qj * Ej).sum(1) * st.q_stop[i])
            ).sum()
        )
    out["contrast"] = con
    # 5. bending
    pi, pj = ga.dir_pairs[:, 0], ga.dir_pairs[:, 1]
    bend = 2.0 * mom.p[pi] * mom.p[pj] - 2.0 * (mom.dsum[pi] * mom.dsum[pj]).sum(1)
    out["bending"] = float((tw.lambda_dir / (2.0 * w2) * ga.dir_pair_w * bend).sum())
    # 6. speed
    out["speed"] = float(0.5 * st.u @ np.asarray(ga.Q_u @ st.u) / w2)
    # 7. τ prior
    tp = 0.0
    if tw.tau_prior and ga.Q_tau is not None:
        dg = ga.Q_tau.diagonal()
        Qs = np.asarray(ga.Q_tau @ mom.s1)
        tp = float(
            0.5 / w2 * (mom.s1 @ Qs - (dg * mom.s1**2).sum() + (dg * mom.s2).sum())
        )
    out["tau_prior"] = tp
    out["jump"] = jump
    return out


COUPLING = ("consistency", "contrast", "bending", "tau_prior", "jump")
UNARY = ("stop", "align")


# ---------------------------------------------------------------- the energy of given
# states


def state_energy(
    st: TermState, ga: GridArrays, tw: TermWeights, choice: IntArray
) -> dict[str, Array]:
    """The energy of each term for given joint states, written directly from the terms.

    choice (B, n) or (n,): the slot of each node's moving candidate, or −1 for the
    stopped state. Returns a dict of (B,) arrays (the same keys as expected_energy)."""
    ch = np.atleast_2d(np.asarray(choice, np.int64))
    B, n = ch.shape
    w2 = tw.w**2
    mov = ch >= 0
    k = np.where(mov, ch, 0)
    rows = np.arange(n)[None, :]
    tau = np.where(mov, st.cand_tau[rows, k], T.TAU_TOP)  # (B, n)
    d = np.where(mov[..., None], st.cand_dir[rows, k], 0.0)  # (B, n, 3)
    prof = candidate_profile_rows(st, np.broadcast_to(rows, k.shape), k)  # (B, n, L, 2)
    lo = st.stop_log_odds
    out: dict[str, Array] = {}
    out["stop"] = (ga.node_w * np.where(mov, _softplus(lo), _softplus(-lo))).sum(1)
    cos = np.einsum("bnc,nc->bn", d, st.tensor_dir)
    al = ga.node_w * st.coherence * (1.0 - cos * cos) / (2.0 * tw.sigma_align**2)
    out["align"] = (mov * al).sum(1)
    # 3. consistency
    s = np.exp(-st.u)
    coef = _cons_coef(ga, tw, np.arange(n))
    cons = np.zeros(B)
    for o in OCTANTS:
        cols = [_col(a, o[a]) for a in range(3)]
        J = ga.nbr6[:, cols]
        ok = (J >= 0).all(1)
        if not ok.any():
            continue
        r = np.nonzero(ok)[0]
        g = np.stack(
            [
                o[a] * (tau[:, J[r, a]] - tau[:, r]) / ga.h_axis[r, cols[a]]
                for a in range(3)
            ],
            axis=-1,
        )  # (B, r, 3)
        val = ((d[:, r] * g).sum(-1) - s[r]) ** 2
        allm = mov[:, r] & mov[:, J[r, 0]] & mov[:, J[r, 1]] & mov[:, J[r, 2]]
        cons += (coef[r] * allm * val).sum(1)
    out["consistency"] = cons
    # 4. contrast
    i, j = ga.edges[:, 0], ga.edges[:, 1]
    L = st.L
    inv_se = 1.0 / np.asarray(tw.sigma_e, float) ** 2
    con = np.zeros(B)
    for b in range(B):
        pj, mask = _shifted(prof[b, j], st.level_base[i], st.level_base[j], L)
        nsh = mask.sum(1)
        W = np.where(nsh > 0, ga.edge_w / (2.0 * w2 * np.maximum(nsh, 1)), 0.0)
        diff2 = (prof[b, i] - pj) ** 2 * mask[:, :, None] * inv_se[None, None, :]
        con[b] = (W * mov[b, i] * mov[b, j] * diff2.sum((1, 2))).sum()
    out["contrast"] = con
    # 5. bending
    pi, pj2 = ga.dir_pairs[:, 0], ga.dir_pairs[:, 1]
    dd = ((d[:, pi] - d[:, pj2]) ** 2).sum(-1)
    cw = tw.lambda_dir / (2.0 * w2) * ga.dir_pair_w
    out["bending"] = (cw * mov[:, pi] * mov[:, pj2] * dd).sum(1)
    # 6. speed
    out["speed"] = np.full(B, float(0.5 * st.u @ np.asarray(ga.Q_u @ st.u) / w2))
    # 7. τ prior
    if tw.tau_prior and ga.Q_tau is not None:
        t = np.where(mov, tau, 0.0)
        Qt = np.asarray(ga.Q_tau @ t.T).T
        out["tau_prior"] = 0.5 * (t * Qt).sum(1) / w2
    else:
        out["tau_prior"] = np.zeros(B)
    # 8. jump
    delta = ga.edge_vec_mm[None, :, :]
    Ei, _ = jump_energy(tau[:, i], d[:, i], st.s_ref[i][None, :], delta, tw)
    Ej, _ = jump_energy(tau[:, j], d[:, j], st.s_ref[j][None, :], -delta, tw)
    out["jump"] = (
        ga.face_w * (mov[:, i] * ~mov[:, j] * Ei + mov[:, j] * ~mov[:, i] * Ej)
    ).sum(1)
    return out


# ---------------------------------------------------------------- the update of u


def u_energy_coeffs(
    st: TermState,
    ga: GridArrays,
    tw: TermWeights,
    sel: IntArray,
    mom: Moments | None = None,
    diag_u: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """(a, b, A, B) per selected node: the u-dependent expected energy (items 3 and 6)
    of node i, with the other nodes fixed, is a u² + b u + A e^(−2u) − 2 B e^(−u) +
    const. diag_u: the diagonal of Q_u (computed here if None)."""
    if mom is None:
        mom = node_moments(st)
    if diag_u is None:
        diag_u = ga.Q_u.diagonal()
    sel = np.asarray(sel, np.int64)
    w2 = tw.w**2
    tau, d, q = _cand(st, sel)
    _, F1, F2 = _own_octants(ga, mom, sel, tau, d, need_f0=False)
    coef = _cons_coef(ga, tw, sel)
    A = coef * mom.p[sel] * F2
    Bc = coef * (q * F1).sum(1)
    dg = diag_u[sel]
    a = 0.5 * dg / w2
    b = (np.asarray(ga.Q_u[sel] @ st.u) - dg * st.u[sel]) / w2
    return np.asarray(a), np.asarray(b), np.asarray(A), np.asarray(Bc)


def _u_fun(a: Array, b: Array, A: Array, B: Array, u: Array) -> Array:
    return np.asarray(
        a[:, None] * u * u
        + b[:, None] * u
        + A[:, None] * np.exp(-2.0 * u)
        - 2.0 * B[:, None] * np.exp(-u)
    )


def update_u(
    st: TermState,
    ga: GridArrays,
    tw: TermWeights,
    sel: IntArray,
    mom: Moments | None = None,
    chunk: int = CHUNK,
) -> Array:
    """u_new (len(sel),): the minimiser over u_i of the u-dependent expected energy
    (items 3 and 6) of each selected node, with the other nodes fixed.

    The search: a grid of step 0.05 on U_RANGE, then 5 refinements (21 points round the
    best, the step / 10 each time), and the closed-form candidates −b / (2a) (if A = 0
    and a > 0) and −log(B / A) (if A, B > 0). The result is the best of these and the
    current u, so the energy never rises. The nodes go in chunks of `chunk`; the result
    of a node does not depend on the other selected nodes."""
    sel = np.asarray(sel, np.int64)
    if mom is None:
        mom = node_moments(st)
    diag_u = ga.Q_u.diagonal()
    out = np.empty(len(sel))
    for s0 in range(0, len(sel), chunk):
        c = sel[s0 : s0 + chunk]
        out[s0 : s0 + len(c)] = _update_u_rows(st, ga, tw, c, mom, diag_u)
    return out


def _update_u_rows(
    st: TermState,
    ga: GridArrays,
    tw: TermWeights,
    sel: IntArray,
    mom: Moments,
    diag_u: Array,
) -> Array:
    """update_u for one chunk of nodes."""
    a, b, A, B = u_energy_coeffs(st, ga, tw, sel, mom, diag_u)
    m = len(sel)
    grid = np.arange(U_RANGE[0], U_RANGE[1] + 1e-12, 0.05)[None, :]
    f = _u_fun(a, b, A, B, np.broadcast_to(grid, (m, grid.shape[1])))
    u = grid[0, np.argmin(f, axis=1)]
    step = 0.05
    off = np.linspace(-1.0, 1.0, 21)[None, :]
    for _ in range(5):
        cand = u[:, None] + step * off
        u = cand[np.arange(m), np.argmin(_u_fun(a, b, A, B, cand), axis=1)]
        step /= 10.0
    u0 = st.u[sel]
    quad = np.where((A == 0) & (a > 0), -b / (2.0 * np.where(a > 0, a, 1.0)), u0)
    ok = (A > 0) & (B > 0)
    lg = np.where(ok, -np.log(np.where(ok, B, 1.0) / np.where(ok, A, 1.0)), u0)
    cands = np.stack([u, u0, quad, lg], axis=1)
    best = np.argmin(_u_fun(a, b, A, B, cands), axis=1)
    return np.asarray(cands[np.arange(m), best])
