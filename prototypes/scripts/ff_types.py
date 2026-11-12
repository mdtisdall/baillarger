"""The contracts of the flow-field phantoms stage (docs/plans/flow-field-phantoms.md).

Every later PR of the stage builds on this file. After PR 1 is open, a change to it needs the
coordinator's permission. So this file holds only types, constants, the stage's solver settings
and small pure helpers. It holds no algorithm.

Conventions (all modules of the stage):

- **Positions** are in voxel index space of the phantom's box: voxel centres at integers, shape
  (N, 3), float64. The voxels are isotropic, VOXEL_MM on a side.
- **Directions** are unit vectors in the voxel axes, shape (N, 3). Because the voxels are
  isotropic, a unit vector in voxel axes is also a unit vector in mm.
- **Lengths** (distances, thicknesses, steps, spacings) are in mm, unless a name ends in `_vox`.
- **Local frame.** Each phantom has a local mm frame (x, y, z), in which its geometry is closed
  form. `Geometry.to_local` and `Geometry.to_vox` convert, as `fold_phantoms.Fold` does.
- **Labels** are those of `fold_phantoms`: CORTEX 0, MEDIUM 1 (the outside medium; never call it
  CSF), WM 2.
- **Relative depth φ**: 0 on the white surface, 1 on the top (the pia or the contact). φ < 0 is
  WM, φ > 1 is the stopped region (except where a phantom removes tissue, see `Geometry.stopped`).
- **τ** (`docs/plans/column-solver.md`, *Columns in τ*): the origin is the top, τ_top = 0. For
  every phantom of this stage the true τ is (φ − 1) · tau_scale, with tau_scale the phantom's mean
  bank thickness (mm). So τ ≤ 0 in the cortex, and the unit is near one mm of depth.
- **Speed** v = ds/dτ (mm per unit τ) along the column. The true speed is 1 / (tau_scale · |∇φ|).
- **Jump** at a stop: τ_top − τ(arrival) ≥ 0. Zero at a pia or a contact, positive where top
  layers are missing (abrasion). Never negative.
- **Determinism.** Every function here is pure. Arrays are float64 unless a field says
  otherwise.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

VOXEL_MM = 0.16  # fold_phantoms.H
CORTEX, MEDIUM, WM = 0, 1, 2
TAU_TOP = 0.0

InputSource = Literal["image", "oracle"]


class StopKind(IntEnum):
    """Where a true or solved column stops. The truth adapters set it for each true column."""

    PIA = 0  # an open top: the column meets the outside medium at a slot or a flat pia
    CONTACT = 1  # a fused contact: the slot is thinner than one voxel (g < VOXEL_MM)
    HAIRPIN = 2  # the round end of the slot at the fundus (the slot's edge, local y < 0)
    ABRADED = 3  # the face of an abraded patch (a positive jump)


class EndReason(IntEnum):
    """Why the integrator ended a solved column (`Columns.end`)."""

    STOPPED = 0  # crossed the stop probability 0.5: a normal stop
    UNRESOLVED_CONTACT = 1  # entered a cell with moving nodes on both sides of a stopped sheet
    LEFT_BOX = 2  # left the grid
    MAX_LENGTH = 3  # reached SolverSettings.max_column_mm without a stop
    SINK = 4  # the direction field fell to zero or reversed away from the stop set


# ---------------------------------------------------------------- the phantom


@runtime_checkable
class Geometry(Protocol):
    """The closed-form geometry of a phantom. A phantom module implements it, usually as a
    subclass of `fold_phantoms.Fold` (so that `fold_phantoms.make` can render it).

    All point arguments Y are local mm points (N, 3). All returns are per point.
    """

    shape: tuple[int, int, int]

    def to_local(self, X: Array) -> Array:
        """Voxel positions (N, 3) to local mm points."""
        ...

    def to_vox(self, Y: Array) -> Array:
        """Local mm points (N, 3) to voxel positions."""
        ...

    def dir_to_vox(self, U: Array) -> Array:
        """Local unit directions (N, 3) to voxel-axis unit directions."""
        ...

    def label(self, Y: Array) -> npt.NDArray[np.int8]:
        """The closed-form label (CORTEX, MEDIUM, WM) at each point. The image's labels are this
        function at the voxel centres."""
        ...

    def phi(self, Y: Array) -> Array:
        """The relative depth φ. Defined and finite at every point of the box (also outside the
        cortex, by the same formula), so that the true columns can be traced from start points in
        the WM. Continuous; continuously differentiable in the cortex."""
        ...

    def grad_phi(self, Y: Array) -> Array:
        """∇φ (1/mm), (N, 3) in the local frame. Never zero in the cortex."""
        ...

    def stopped(self, Y: Array) -> BoolArray:
        """True in the true stopped region: the outside medium, and tissue removed by abrasion.
        Equal to label == MEDIUM at every point."""
        ...

    def stop_sdf(self, Y: Array) -> Array:
        """The signed distance (mm) to the true stop surface: negative in tissue, positive in the
        stopped region, zero on the stop surface. Exact (to 1e-3 mm) where it is negative and
        above −1 mm; elsewhere only its sign is required. A sheet thinner than a voxel (a fused
        contact) is resolved: the function is evaluated in closed form, not from voxels."""
        ...

    def bank(self, Y: Array) -> npt.NDArray[np.int8]:
        """The bank of each point (0: A, 1: B). Single-bank phantoms give 0 everywhere."""
        ...


@dataclass(frozen=True)
class Phantom:
    """One phantom of the plan's *Phantoms* table, with its image.

    image: the dict of `fold_phantoms.make` (det {1: echo 2, 2: echo 3} detrended, ctx, wm, rim,
    miss, vox, dface, labels, noise_sd, fold, lo). Code that reads the site's data in the same
    form (`baillarger.orient`, `top_confidence.py`) reads this dict.
    """

    name: str  # the name in the plan's table, for example "ckiss_smooth"
    geometry: Geometry | None  # None only for "noise" (no structure, no truth)
    image: Mapping[str, Any]
    features: Mapping[str, Any]  # the site features used for the image (fold_measure.py form)
    site: str  # the features' site, "V1V2_banks" for the runs; "synthetic" in tests
    noise: float  # the scale of the site's matched noise SD (0 or 1.0)
    seed: int
    tau_scale: float  # mm: the mean bank thickness. True τ = (φ − 1) · tau_scale
    params: Mapping[str, Any]  # every parameter of the phantom (thicknesses, g, box, ...), JSON-able

    @property
    def shape(self) -> tuple[int, int, int]:
        lab = self.image["labels"]
        return (int(lab.shape[0]), int(lab.shape[1]), int(lab.shape[2]))

    @property
    def labels(self) -> npt.NDArray[np.int16]:
        return np.asarray(self.image["labels"])

    @property
    def det(self) -> Mapping[int, Array]:
        """The detrended echoes: {1: echo 2, 2: echo 3}."""
        return self.image["det"]  # type: ignore[no-any-return]

    def record(self) -> dict[str, Any]:
        """The phantom's entry in a run record."""
        return dict(name=self.name, site=self.site, noise=self.noise, seed=self.seed,
                    tau_scale=self.tau_scale, shape=list(self.shape), params=_jsonable(self.params))


# ---------------------------------------------------------------- the truth


@dataclass(frozen=True)
class Truth:
    """The truth of one phantom under the stop rule (the plan's *Truth under the stop rule*).

    The true columns are the orthogonal trajectories of φ, traced with RK4 at `step_mm`, from each
    root on φ = 0 to the true stop surface (stop_sdf = 0). Each column stops on the stop surface:
    on the pia (φ = 1), on a fused contact (each bank on its own face), or on an abraded face.

    The paths are padded: paths[i, :counts[i]] is column i, nan after it. paths[i, 0] is the root,
    paths[i, counts[i] − 1] the stop. All positions in voxel space.
    """

    phantom: str
    roots: Array  # (N, 3) on φ = 0
    normals: Array  # (N, 3) unit, voxel axes: the white surface's normal at the root, toward the cortex
    starts: Array  # (N, 3): root − START_DEPTH_MM along the normal (the column's fixed start point)
    paths: Array  # (N, m, 3), nan-padded
    counts: IntArray  # (N,)
    stops: Array  # (N, 3) == paths[i, counts[i] − 1]
    stop_kind: npt.NDArray[np.int8]  # (N,) StopKind
    bank: npt.NDArray[np.int8]  # (N,)
    tau_arrival: Array  # (N,): the true τ at the stop, (φ(stop) − 1) · tau_scale
    jump: Array  # (N,): TAU_TOP − tau_arrival ≥ 0; zero except on abraded patches
    offset: Array  # (N,): the true τ at the start point (φ there extended into the WM)
    tau_scale: float
    step_mm: float
    root_spacing_mm: float

    def __post_init__(self) -> None:
        n = len(self.roots)
        for k in ("normals", "starts", "stops"):
            assert getattr(self, k).shape == (n, 3), k
        assert self.paths.shape[0] == n and self.paths.shape[2] == 3
        for k in ("counts", "stop_kind", "bank", "tau_arrival", "jump", "offset"):
            assert getattr(self, k).shape == (n,), k

    @property
    def n(self) -> int:
        return len(self.roots)

    def path(self, i: int) -> Array:
        return self.paths[i, : self.counts[i]]

    def to_npz(self, path: Path) -> None:
        np.savez_compressed(path, **{k: v for k, v in _fields(self).items()})

    @classmethod
    def from_npz(cls, path: Path) -> Truth:
        z = np.load(path, allow_pickle=False)
        kw: dict[str, Any] = {k: z[k] for k in z.files}
        for k in ("phantom",):
            kw[k] = str(kw[k])
        for k in ("tau_scale", "step_mm", "root_spacing_mm"):
            kw[k] = float(kw[k])
        return cls(**kw)


def true_tau(geometry: Geometry, tau_scale: float, X: Array) -> Array:
    """The true τ at voxel positions X: (φ − 1) · tau_scale."""
    return (geometry.phi(geometry.to_local(X)) - 1.0) * tau_scale


def true_speed(geometry: Geometry, tau_scale: float, X: Array) -> Array:
    """The true speed ds/dτ (mm per unit τ) at voxel positions X: 1 / (tau_scale · |∇φ|)."""
    g = np.linalg.norm(geometry.grad_phi(geometry.to_local(X)), axis=1)
    return 1.0 / (tau_scale * g)


def true_direction(geometry: Geometry, X: Array) -> Array:
    """The true column direction at voxel positions X: ∇φ / |∇φ| in voxel axes."""
    G = geometry.grad_phi(geometry.to_local(X))
    return geometry.dir_to_vox(G / np.linalg.norm(G, axis=1)[:, None])


# ---------------------------------------------------------------- the inputs


@runtime_checkable
class StopField(Protocol):
    """The stop field: the log-odds that a point is stopped (the plan's *Inputs*). Evaluated at
    any points, so that an octree finer than the image can read it."""

    source: InputSource

    def log_odds(self, X: Array) -> Array:
        """Log-odds of the stopped state at voxel positions X (N, 3)."""
        ...

    def terms(self, X: Array) -> dict[str, Array]:
        """The separate terms at X, for the views (image source: label, sheet, mirror,
        partial_volume; oracle source: {"oracle": log-odds}). Their combination is log_odds."""
        ...


@runtime_checkable
class Tensor(Protocol):
    """The orientation input: the leading eigenvector (sign-free) and its coherence."""

    source: InputSource

    def at(self, X: Array) -> tuple[Array, Array]:
        """(direction (N, 3) unit in voxel axes, coherence (N,) in [0, 1]) at voxel positions X.
        The direction's sign is arbitrary; terms must use it sign-free."""
        ...


@dataclass(frozen=True)
class Inputs:
    """One run's inputs. The contrast is the phantom's detrended echoes (no oracle)."""

    stop: StopField
    tensor: Tensor
    contrast: Mapping[int, Array]  # Phantom.det

    def record(self) -> dict[str, Any]:
        return dict(stop=self.stop.source, tensor=self.tensor.source)


def log_odds(p: Array, lo: float = 0.02, hi: float = 0.98) -> Array:
    """The log-odds of a probability clipped to [lo, hi] (the oracle's clip)."""
    q = np.clip(p, lo, hi)
    return np.asarray(np.log(q / (1.0 - q)))


def sigmoid(z: Array) -> Array:
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, float))))


# ---------------------------------------------------------------- the grid


@runtime_checkable
class Grid(Protocol):
    """The grid interface. PR 3 writes a uniform grid behind it, PR 5 an octree. Code above the
    grid (terms, mean field, integrator, views) uses only this interface.

    Nodes are at cell corners. Cells are axis-aligned cubes in voxel space. A node that lies on
    the face or edge of a larger neighbour cell (a hanging node, octree only) is constrained: its
    value is the weighted sum of its parents' values, and it has no free state.
    """

    kind: Literal["uniform", "octree"]

    @property
    def n_nodes(self) -> int: ...

    @property
    def nodes(self) -> Array:
        """(n, 3) node positions, voxel space."""
        ...

    @property
    def node_level(self) -> IntArray:
        """(n,) the refinement level of the finest cell that touches the node: 0 at the
        coarsest spacing, +1 per halving. A uniform grid has one level."""
        ...

    @property
    def cells(self) -> IntArray:
        """(C, 8) node ids of each leaf cell's corners, in the order (i, j, k) with corner index
        i + 2 j + 4 k (i, j, k ∈ {0, 1} along the voxel axes)."""
        ...

    @property
    def cell_origin(self) -> Array:
        """(C, 3) the corner 0 of each cell, voxel space."""
        ...

    @property
    def cell_size_mm(self) -> Array:
        """(C,) the side of each cell (mm)."""
        ...

    @property
    def prior_shaped(self) -> BoolArray:
        """(C,) True for a cell finer than the image (side < VOXEL_MM): its shape comes mostly
        from the prior and the constraints (the plan's *The octree*)."""
        ...

    @property
    def edges(self) -> IntArray:
        """(E, 2) node ids of each cell edge between two free or hanging nodes, i < j, each edge
        once, sorted. The pair terms of *Terms on the grid* act on these."""
        ...

    @property
    def hanging(self) -> tuple[IntArray, IntArray, Array]:
        """(ids (H,), parents (H, 4), weights (H, 4)): each hanging node's value is
        Σ weights · values[parents]. Unused parent slots have weight 0 (an edge node has 2
        parents, a face node 4). A uniform grid gives empty arrays."""
        ...

    @property
    def colour(self) -> IntArray:
        """(n,) colour groups for the mean field: no two corners of one cell share a colour, so
        nodes of one colour do not interact. A uniform grid uses the parity of the node indices
        (8 colours)."""
        ...

    @property
    def n_colours(self) -> int: ...

    def locate(self, X: Array) -> tuple[IntArray, Array]:
        """(cell id (N,), local coordinates (N, 3) in [0, 1]) of each voxel position X. Cell −1
        for a point outside the grid. Deterministic on a face between cells: the cell with the
        larger corner-0 index along each axis."""
        ...

    def morton(self) -> IntArray:
        """(C,) the Morton key of each leaf cell (the order of marking in the octree)."""
        ...

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Every array that defines the grid, for field.npz (keys prefixed "grid_")."""
        ...


def trilinear_weights(local: Array) -> Array:
    """(N, 8) Q1 weights of the corners (order i + 2 j + 4 k) at local coordinates (N, 3)."""
    u = np.clip(local, 0.0, 1.0)
    w = np.ones((len(u), 8))
    for c in range(8):
        for a in range(3):
            b = (c >> a) & 1
            w[:, c] *= u[:, a] if b else 1.0 - u[:, a]
    return w


def sample_volume(vol: Array, X: Array) -> Array:
    """Trilinear sample of a voxel volume at voxel positions X (N, 3), with the edge values held
    outside the box. The contrast terms read the image at full resolution through this."""
    from scipy import ndimage as ndi

    return np.asarray(ndi.map_coordinates(np.asarray(vol, float), np.asarray(X, float).T, order=1,
                                          mode="nearest"))


# ---------------------------------------------------------------- the field


@dataclass
class Field:
    """The solved field on a grid (the plan's *Node states*): for each node, the kept moving
    candidates (τ, direction) and the stopped state, with their probabilities q.

    Slot k < n_kept[i] of node i is a kept moving candidate; the slots after it are padding (nan
    τ, zero direction, q 0). q_stop[i] is the probability of the stopped state (τ = TAU_TOP,
    speed 0). For each node, q_stop + Σ_k q[i, k] = 1. Hanging nodes carry the values their
    parents give them.
    """

    grid: Grid
    tau: Array  # (n, K) candidate τ, nan-padded; K = SolverSettings.prune_max
    direction: Array  # (n, K, 3) candidate unit direction, voxel axes, zero-padded
    q: Array  # (n, K) probabilities of the moving candidates
    q_stop: Array  # (n,) probability of the stopped state
    n_kept: IntArray  # (n,)

    def __post_init__(self) -> None:
        n, K = self.tau.shape
        assert self.direction.shape == (n, K, 3) and self.q.shape == (n, K)
        assert self.q_stop.shape == (n,) and self.n_kept.shape == (n,)

    @property
    def p_moving(self) -> Array:
        return np.asarray(1.0 - self.q_stop)

    def tau_mean(self) -> Array:
        """(n,) the mean τ over the moving candidates (conditional on moving); nan where the node
        is stopped with probability 1."""
        qm = self.q.sum(1)
        num = np.nansum(self.q * np.nan_to_num(self.tau), 1)
        return np.where(qm > 0, num / np.maximum(qm, 1e-300), np.nan)

    def direction_mean(self) -> Array:
        """(n, 3) the posterior-mean direction over the moving candidates, normalised (the
        integrator's direction). Zero where no moving mass. Directions are signed here: the
        candidates of one node point the same way (toward the stop)."""
        D = np.einsum("nk,nkc->nc", self.q, self.direction)
        nr = np.linalg.norm(D, axis=1)
        return np.where(nr[:, None] > 0, D / np.maximum(nr, 1e-300)[:, None], 0.0)

    def tau_map(self) -> Array:
        """(n,) τ of the most probable state (TAU_TOP if the stopped state is the most probable)."""
        k = np.argmax(self.q, 1)
        t = self.tau[np.arange(len(k)), k]
        return np.where(self.q_stop >= self.q.max(1), TAU_TOP, t)

    def to_npz(self, path: Path, extra: Mapping[str, npt.NDArray[Any]] | None = None) -> None:
        """field.npz: the candidates, q, the stop probability, the means, and the grid arrays."""
        d: dict[str, npt.NDArray[Any]] = dict(
            tau=self.tau, direction=self.direction, q=self.q, q_stop=self.q_stop,
            n_kept=self.n_kept, tau_mean=self.tau_mean(), direction_mean=self.direction_mean(),
            tau_map=self.tau_map(), nodes=self.grid.nodes)
        d.update(self.grid.arrays())
        if extra:
            d.update(extra)
        np.savez_compressed(path, **d)


# ---------------------------------------------------------------- the solved columns


@dataclass(frozen=True)
class Columns:
    """The solved columns of one run (columns.npz), from the integrator.

    paths[i, :counts[i]] is column i (voxel space, step SolverSettings.rk4_step_mm), from its
    start point to its stop; tau_path the interpolated τ along it. The sample statistics (sd) are
    nan when no fields were sampled.
    """

    roots: Array  # (N, 3) the same roots as Truth.roots
    paths: Array  # (N, m, 3), nan-padded
    counts: IntArray  # (N,)
    tau_path: Array  # (N, m), nan-padded
    stops: Array  # (N, 3)
    end: npt.NDArray[np.int8]  # (N,) EndReason
    tau_arrival: Array  # (N,) τ at the stop, before the jump
    jump: Array  # (N,) TAU_TOP − tau_arrival, ≥ 0
    p_jump: Array  # (N,) the probability that the jump is positive
    stop_sd: Array  # (N,) mm, from sampled fields
    level_sd: Array  # (N, L) mm per τ level (Measures: Uncertainty), nan if none
    levels: Array  # (L,) the τ levels that level_sd and the level measures use

    def to_npz(self, path: Path) -> None:
        np.savez_compressed(path, **_fields(self))


# ---------------------------------------------------------------- the stage's settings


@dataclass(frozen=True)
class SolverSettings:
    """The defaults of the plan's *The solver for this stage*. Every run writes them into its
    record. A scan of a setting (× 0.5 and × 2) is a finding, never a setting to tune: a run
    that changes one records the change through `dataclasses.replace` and says so."""

    # inputs
    oracle_blur_vox: float = 0.5  # Gaussian SD of the oracle stop field
    oracle_clip: tuple[float, float] = (0.02, 0.98)
    band_scale_vox: float = 2.0  # the sheet term's Hessian scale and the contrast smoothing
    mirror_mm: float = 1.0
    # grid
    uniform_spacings_mm: tuple[float, ...] = (0.64, 0.48, 0.32, 0.16)
    octree_coarsest_mm: float = 0.64
    octree_finest_mm: float = 0.02
    prior_shaped_below_mm: float = VOXEL_MM
    # node states
    n_tau: int = 9
    dtau: float = 0.04
    n_dir: int = 7  # the predicted direction and 6 round it
    dir_cone_deg: float = 15.0
    prune_mass: float = 0.999
    prune_min: int = 4
    prune_max: int = 32
    # terms
    speed_typical: float = 1.0  # log v near log 1
    speed_log_sd: float = 0.3
    jump_p_zero: float = 0.999
    jump_mean: float = 0.5  # the exponential above 0
    fem_length_mm: float = 0.5  # ℓ of λ (K + M / ℓ²)
    start_depth_mm: float = 2.0  # the start point behind the white surface
    # inference
    widths: tuple[int, ...] = (4, 2, 1)
    max_sweeps: int = 200
    tol_q: float = 1e-4
    tol_free_energy_rel: float = 1e-7
    coarse_first_mm: float = 0.64  # question 6 only
    # integrator
    rk4_step_mm: float = 0.01
    stop_level: float = 0.5
    stop_bisections: int = 6
    max_column_mm: float = 12.0
    # octree
    refine_factor: float = 2.0  # nominate above 2 × median
    coarsen_factor: float = 0.5  # nominate below 0.5 × median
    octree_rounds: int = 6
    # sampled fields
    n_samples: int = 32

    def record(self) -> dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))


DEFAULTS = SolverSettings()
START_DEPTH_MM = DEFAULTS.start_depth_mm
TRUTH_STEP_MM = 0.01  # bent_phantom.STEP


# ---------------------------------------------------------------- the run record


@dataclass
class RunRecord:
    """run.json: everything needed to read and repeat one run (the plan's *Run matrix*)."""

    phantom: dict[str, Any]  # Phantom.record()
    source: InputSource
    grid: dict[str, Any]  # kind, spacing or octree levels, node and cell counts per level
    settings: dict[str, Any] = field(default_factory=DEFAULTS.record)
    changed_settings: dict[str, Any] = field(default_factory=dict)  # name -> value, if a scan
    origin: str = "top"  # "top" (τ_top = 0) or "anchor" (question 6 comparison)
    versions: dict[str, str] = field(default_factory=dict)  # git commit, input files and hashes
    free_energy: list[list[float]] = field(default_factory=list)  # per width, per sweep
    sweeps: list[int] = field(default_factory=list)  # per width
    free_energy_rises: list[dict[str, Any]] = field(default_factory=list)  # defects
    pruning: dict[str, Any] = field(default_factory=dict)  # counts at the edge of the kept set
    defects: list[str] = field(default_factory=list)
    measures: dict[str, Any] = field(default_factory=dict)
    time_s: float = float("nan")
    peak_mem_bytes: int = -1

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(_jsonable(dataclasses.asdict(self)), indent=1,
                                         allow_nan=True) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> RunRecord:
        return cls(**json.loads(Path(path).read_text()))


def run_dir(out: Path, phantom: str, source: InputSource, noise: float, grid: str) -> Path:
    """<out>/<phantom>/<source>_<noise>_<grid>/ (the plan's *Run matrix, outputs and memory*);
    out is an argument of the script, never a fixed path."""
    d = Path(out) / phantom / f"{source}_{noise:g}_{grid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- helpers


def _fields(obj: Any) -> dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (tuple, list)) or (isinstance(x, Sequence) and not isinstance(x, str)):
        return [_jsonable(v) for v in x]
    return x
