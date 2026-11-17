"""The continuous column energy on a 1D line in the S1 form: ONE continuous τ field, the top is
the level set τ = 0, no banks, no stop unknown, no jump (docs/plans/continuous-column-solver.md,
draft; the user's decisions for contact1d).

The field
---------
τ is piecewise linear on the nodes x_0 < ... < x_n (MFEM H1 order 1 on a 1D mesh; the mesh may
be non-uniform). Optionally one cell is ENRICHED (`enrich_cell`, the kink form K3a′): it gets one
extra interior node with a free position x_k and a free value τ_k (both unknowns). This is the
XFEM kink enrichment |x − x_k| with a free kink position. Where ∇τ reverses inside a cell (a
sheet thinner than a cell), the reversal is at x_k, so the field is not interpolated across the
sheet. `solve_kink` decides from the field where the enrichment is (or that there is none: a
reversal on a node needs none).

Orientation (K2): the speed term is sign-free; the sign comes from the roots. `roots` gives
each root's position and the white surface's normal there (+1 or −1 along x): the hard condition
n·∇τ > 0 on the cell that holds the root (Q1, a hard constraint: feasibility in the line
search). Two rules of the model, as feasibility: τ has no local minimum except at the roots
(Q3d: no sources inside the domain; every point is reached by ascending from a root), and a local
maximum (a ridge) only where τ ≥ 0 (Q2c: inside tissue a column cannot turn back). They are
checked at every node where the slope changes sign (the extra node of the enrichment, or a fixed
node). The ridge is where the virtual continuations of τ from two tops meet in the outside
medium: a contact, found from the field and the roots' orientations, not from the start.
The line search rejects a trial that breaks a constraint. An active-set step is not coded: the
solver records when a solve ends with a constraint active (`active_constraints`).

The energy (per voxel column of cross-section A = VOXEL_MM², as ffc_solve1d)
---------------------------------------------------------------------------
- Speed (K1: over the whole line; K2: sign-free): u_e = −log |τ'_e| on each cell e.
  E_speed = (λ A / 2) [Σ_e L_e u_e² / ℓ² + Σ_f (u_{e+1} − u_e)² / d_f], d_f = (L_e + L_{e+1}) / 2
  (the two-point form of ∫ u'²). K4 (`skip_reversal`): the dual cell of a node where the slope
  changes sign is left out, so the speeds of the two sides of a sheet are not tied.
- Stop: (1 / VOXEL_MM) ∫ [τ < 0: softplus(ℓo); τ > 0: softplus(−ℓo)] dx over the line, each
  cell split at its crossing of τ = 0 (Gauss rule on each part).
- No jump (J = 0 by form) and no alignment term (the oracle tensor is along the line).

The top probability (Laplace)
-----------------------------
At the minimum z* the posterior of the unknowns is approximated by N(z*, H⁻¹), H the Hessian of
the energy. τ(x) is linearised in z: σ²(x) = a(x)ᵀ H⁻¹ a(x), a(x) = ∂τ(x)/∂z. The probability
that x is beyond the top is P(τ(x) > 0) = Φ(τ*(x) / σ(x)).

The τ representation is behind the interface ffc_basis.TauBasis (the user's decision of
2026-11-12): the energy, the feasibility rules and the readouts read τ only through it. The P1
forms above are those of ffc_basis.P1Basis (the speed prior's two-point form is its
`derived_norms`; the cut cells are its `integrate_cut`). The kink switch (`solve_kink`) is
P1-only.

Derivatives: Torch autograd (float64) on the quadrature-point energy, as ffc_solve1d. Newton
with Levenberg damping and an Armijo line search; stop at a Newton decrement below NEWTON_TOL.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402
import ffc_basis as B  # noqa: E402

F64 = torch.float64
QUAD_ORDER = B.QUAD_ORDER
NEWTON_TOL = 1e-14
MAX_NEWTON = 200
MAX_LINE = 60


# ---------------------------------------------------------------- the problem

ENERGY_FORMS = ("speed", "equivolume_abs", "equivolume_convex")
# The equivolume weight: the old speed prior's λA (1 / (speed_log_sd² ℓ) · VOXEL_MM² = 0.569),
# a PROVISIONAL placeholder (the user's E5; a literature check runs separately).
EQUIVOLUME_LAMBDA_A_PROVISIONAL = 1.0 / (T.DEFAULTS.speed_log_sd**2 * T.DEFAULTS.fem_length_mm) * T.VOXEL_MM**2
MM_PULL_LOG_SD = math.log(2.0)  # the prior SD of log speed behind the mm pull (the user's E6)


def mm_pull_coefficient(st: T.SolverSettings) -> float:
    """c in c ∫ (|τ'| − 1)² dx: the old speed prior's mass-term normalisation (λA/2)/ℓ², with
    speed_log_sd replaced by log 2: 0.5 · VOXEL_MM² / ((log 2)² ℓ³)."""
    return 0.5 * T.VOXEL_MM**2 / (MM_PULL_LOG_SD**2 * st.fem_length_mm**3)


mfem_nodes = B.mfem_nodes
gauss_rule = B.gauss_rule


@dataclass
class Problem:
    """One 1D line in the S1 form. log_odds_t: the stop field's log-odds at x (Torch,
    differentiable). τ is read only through `basis` (ffc_basis.TauBasis); by default the MFEM
    P1 basis on the nodes x, enriched in `enrich_cell` (K3a′) if it is not None."""

    x: T.Array  # (n + 1,) the fixed nodes
    log_odds_t: Callable[[torch.Tensor], torch.Tensor]
    settings: T.SolverSettings = T.DEFAULTS
    enrich_cell: int | None = None  # the cell with the extra free node (K3a′), or None
    roots: tuple[tuple[float, float], ...] = ()  # (x_root, n): the white surface's root and its normal (±1 along x)
    skip_reversal: bool = True
    # "speed": the old speed prior. "equivolume_abs": equivolume in tissue + c (|τ'| − 1)² everywhere
    # (non-convex in τ' at 0). "equivolume_convex": equivolume in tissue + c (n τ' − 1)² in tissue
    # (n the orientation of the tissue's root) + c max(|τ'| − 1, 0)² beyond the top.
    energy_form: str = "speed"
    mm_scale: float = 1.0  # a factor on the mm-pull weight (an insensitivity check only)
    slope_bounds: bool = False  # Q1 and Q3d as linear bounds in the active set (S1), not only line-search checks
    basis: B.P1Basis = field(init=False)

    def __post_init__(self) -> None:
        if self.energy_form not in ENERGY_FORMS:
            raise ValueError(f"energy_form: {self.energy_form!r}")
        self.basis = B.P1Basis(np.asarray(self.x, float), self.enrich_cell)
        self.x = self.basis.x
        st = self.settings
        lam = 1.0 / (st.speed_log_sd**2 * st.fem_length_mm)
        self.lam_a = lam * T.VOXEL_MM**2
        self.ell = st.fem_length_mm
        self.inv_hv = 1.0 / T.VOXEL_MM
        # the equivolume model: the weights (module docstring)
        self.eq_weight = EQUIVOLUME_LAMBDA_A_PROVISIONAL
        self.mm_coeff = self.mm_scale * mm_pull_coefficient(st)

    @property
    def qx(self) -> T.Array:
        return self.basis.qx

    @property
    def qw(self) -> T.Array:
        return self.basis.qw

    @property
    def n_unknowns(self) -> int:
        return self.basis.n_unknowns

    def split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.basis.split(z)

    def pack(self, tau: T.Array, tau_k: float | None = None, x_k: float | None = None) -> T.Array:
        return self.basis.pack(tau, tau_k, x_k)

    def fixed_tau(self, z: T.Array) -> T.Array:
        return self.basis.fixed_tau(z)

    # ---- the energy

    def _quad(self, a: torch.Tensor, b: torch.Tensor, sign: float) -> torch.Tensor:
        """∫_a^b softplus(sign · ℓo) dx by the Gauss rule (a check of the cut quadrature)."""
        qx = torch.as_tensor(self.qx, dtype=F64)
        qw = torch.as_tensor(self.qw, dtype=F64)
        xs = a + (b - a) * qx
        return (b - a) * (qw * torch.nn.functional.softplus(sign * self.log_odds_t(xs))).sum()

    def _quad_v(self, a: torch.Tensor, b: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """Per interval: ∫_a^b softplus(sign · ℓo) dx by the Gauss rule (vectors a, b, sign)."""
        qx = torch.as_tensor(self.qx, dtype=F64)
        qw = torch.as_tensor(self.qw, dtype=F64)
        xq = a[:, None] + (b - a)[:, None] * qx[None, :]
        f = torch.nn.functional.softplus(sign[:, None] * self.log_odds_t(xq))
        return (b - a) * (f * qw[None, :]).sum(1)

    def energy_terms(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        # the stop term (evidence): tissue (τ < 0) softplus(ℓo), beyond the top softplus(−ℓo),
        # on the basis's quadrature split at τ = 0
        if self.energy_form in ("equivolume_abs", "equivolume_convex"):
            # equivolume in tissue: v = 1/τ' per cell, (λA/2) Σ frac (v_{e+1} − v_e)² / d
            _, e_dv = self.basis.derived_norms(z, lambda g: 1.0 / g, self.skip_reversal, tissue_only=True)
            e_eq = 0.5 * self.eq_weight * e_dv
            if self.energy_form == "equivolume_abs":
                # the mm pull everywhere: c ∫ (|τ'| − 1)² dx
                e_pull, _ = self.basis.derived_norms(z, lambda g: torch.abs(g) - 1.0, False)
            else:
                n_of = self.orientation(z)

                def pull(g: torch.Tensor, side: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                    tissue = (n_of(x) * g - 1.0) ** 2
                    beyond = torch.clamp(torch.abs(g) - 1.0, min=0.0) ** 2
                    return torch.where(side < 0, tissue, beyond)

                e_pull = self.basis.integrate_grad_cut(z, pull)
            e_mm = self.mm_coeff * e_pull
            e_stop = self.basis.integrate_cut(
                z, lambda x, side: torch.nn.functional.softplus(-side * self.log_odds_t(x))) * self.inv_hv
            return dict(equivolume=e_eq, mm_pull=e_mm, stop=e_stop, total=e_eq + e_mm + e_stop)
        # the speed prior on the derived field u = −log |∇τ| (K1 over the whole line, K2
        # sign-free, K4 the reversal parts left out)
        e_u2, e_du = self.basis.derived_norms(z, lambda g: -torch.log(torch.abs(g)), self.skip_reversal)
        e_speed = 0.5 * self.lam_a * (e_u2 / (self.ell * self.ell) + e_du)
        e_stop = self.basis.integrate_cut(
            z, lambda x, side: torch.nn.functional.softplus(-side * self.log_odds_t(x)))
        e_stop = e_stop * self.inv_hv
        return dict(speed=e_speed, stop=e_stop, total=e_speed + e_stop)

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        return self.energy_terms(z)["total"]

    def root_slopes(self, z: T.Array) -> T.Array:
        """n · ∇τ at each root."""
        if not self.roots:
            return np.zeros(0)
        xr = np.array([r for r, _ in self.roots])
        nr = np.array([n for _, n in self.roots])
        with torch.no_grad():
            g = self.basis.gradients(torch.as_tensor(z, dtype=F64), xr).numpy()
        return np.asarray(nr * g)

    def valid(self, z: T.Array) -> bool:
        """The field is admissible, n·∇τ > 0 at the roots (Q1), no interior local minimum
        (Q3d) and every local maximum (a ridge) has τ ≥ 0 (Q2c)."""
        if not self.basis.admissible(z, flat_beyond_top=self.energy_form != "speed"):
            return False
        for ex in self.basis.extrema(z):
            if ex.kind == "min" or ex.value < 0:
                return False
        return bool(np.all(self.root_slopes(z) > 0))

    def ridge_x(self, z: T.Array) -> float | None:
        mx = [ex.x for ex in self.basis.extrema(z) if ex.kind == "max"]
        return mx[0] if mx else None

    def orientation(self, z: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """n(x): the orientation of the column at x, from its root (the root on the same side
        of the ridge; with one root, or with no ridge, the nearest root). A constant of the
        energy's evaluation (from the detached field)."""
        zd = z.detach().numpy()
        xr = self.ridge_x(zd)
        roots = sorted(self.roots)

        def n_of(x: torch.Tensor) -> torch.Tensor:
            xn = x.detach().numpy()
            if len(roots) == 1:
                return torch.full_like(x, roots[0][1])
            if xr is None:
                d = np.stack([np.abs(xn - r) for r, _ in roots])
                k = np.argmin(d, axis=0)
            else:
                k = (xn > xr).astype(int)
            return torch.as_tensor(np.array([n for _, n in roots])[k], dtype=F64)

        return n_of

    def bounds(self, z: T.Array) -> list[T.Array]:
        """The linear bounds c·z ≥ 0: each ridge's τ ≥ 0 (Q2c); with slope_bounds also the slope
        signs (Q1 and Q3d): each cell rises away from its root (n · (τ_end − τ_start) ≥ 0, the
        side from the ridge), except the cells next to the ridge (their sign may change: the ridge
        moves)."""
        out = [ex.c for ex in self.basis.extrema(z) if ex.kind == "max"]
        if not self.slope_bounds:
            return out
        mx = [ex for ex in self.basis.extrema(z) if ex.kind == "max"]
        roots = sorted(self.roots)
        tau, xs = self.basis.nodal(z)
        lo, hi = len(xs), -1  # the nodes of the ridge (one node, or a flat run)
        if mx:
            i = int(np.argmin(np.abs(xs - mx[0].x)))
            lo = hi = i
            while lo > 0 and tau[lo - 1] == tau[i]:
                lo -= 1
            while hi < len(xs) - 1 and tau[hi + 1] == tau[i]:
                hi += 1
        for e, (xm, c) in enumerate(self.basis.cell_rise_forms()):
            if lo - 1 <= e <= hi:
                continue  # touches the ridge: free (the ridge may move)
            if len(roots) == 1:
                n = roots[0][1]
            elif mx:
                n = roots[0][1] if e < lo else roots[1][1]
            else:
                n = roots[0][1] if abs(xm - roots[0][0]) < abs(xm - roots[1][0]) else roots[1][1]
            out.append(n * c)
        return out

    def active_constraints(self, z: T.Array, tol: float = 1e-9) -> dict[str, int]:
        """The constraints within tol of their bound at z (root slopes and extremum values)."""
        n_root = int(np.sum(self.root_slopes(z) < tol))
        return dict(root=n_root, reversal=int(sum(abs(ex.value) < tol for ex in self.basis.extrema(z))))

    def n_reversals(self, z: T.Array) -> int:
        return len(self.basis.extrema(z))

    def total(self, z: T.Array) -> float:
        if not self.valid(z):
            return math.inf
        with torch.no_grad():
            E = float(self.energy(torch.as_tensor(z, dtype=F64)))
        return E if math.isfinite(E) else math.inf

    # ---- the field at points

    def tau_at(self, z: torch.Tensor, xq: T.Array) -> torch.Tensor:
        """τ at the points xq (Torch, differentiable in z)."""
        return self.basis.values(z, xq)


@dataclass
class SolveLog:
    energy: list[float] = field(default_factory=list)
    decrement: list[float] = field(default_factory=list)
    damping: list[float] = field(default_factory=list)
    step: list[float] = field(default_factory=list)
    active: list[int] = field(default_factory=list)  # the active bounds in each iteration
    converged: bool = False
    reason: str = ""

    @property
    def iterations(self) -> int:
        return len(self.decrement)


def gradient_hessian(p: Problem, z: T.Array) -> tuple[T.Array, T.Array]:
    zt = torch.as_tensor(z, dtype=F64).clone().requires_grad_(True)
    g = torch.autograd.grad(p.energy(zt), zt)[0].detach().numpy()
    H = torch.autograd.functional.hessian(p.energy, torch.as_tensor(z, dtype=F64)).numpy()
    return g, H


FREE_NODE_CUT = 0.99  # a step moves the free node at most this fraction of the way to its cell's end
EDGE_TOL = 1e-6  # the free node within this fraction of the cell length of an end: the kink is on that node
MIN_STEP = 1e-12  # a line-search step below this ends the solve (stalled)
ACTIVE_TOL = 1e-12  # a ridge with |τ| below this is on its bound τ = 0


def ridge_index(p: Problem, z: T.Array) -> int | None:
    """P1 (the kink switch): the index in z of the τ value of the first ridge, or None."""
    for ex in p.basis.extrema(z):
        if ex.kind == "max":
            return int(np.argmax(ex.c))
    return None


def _newton_step(g: T.Array, H: T.Array, active: list[T.Array]) -> tuple[T.Array, float]:
    """The Newton step in the null space of the active bounds (c·step = 0 for each), with the
    Levenberg shift μ that makes the reduced Hessian positive definite. A bound on one unknown
    (c a unit vector) fixes that unknown."""
    n = len(g)
    if all(np.count_nonzero(c) == 1 for c in active):
        free = np.ones(n, bool)
        for c in active:
            free[int(np.flatnonzero(c)[0])] = False
        Z = np.eye(n)[:, free]
    else:
        Z = scipy.linalg.null_space(np.array(active))
    gf, Hf = Z.T @ g, Z.T @ H @ Z
    mu = 0.0
    scale = max(1e-12, float(np.max(np.abs(np.diag(Hf)))))
    while True:
        try:
            C = np.linalg.cholesky(Hf + mu * np.eye(len(gf)))
            break
        except np.linalg.LinAlgError:
            mu = max(2.0 * mu, 1e-10 * scale)
    return Z @ -np.linalg.solve(C.T, np.linalg.solve(C, gf)), mu


def _project(z: T.Array, c: T.Array) -> T.Array:
    """z moved onto the bound c·z = 0 (for a unit vector c: that unknown set to 0)."""
    out = z - (c @ z) / (c @ c) * c
    if np.count_nonzero(c) == 1:
        out[int(np.flatnonzero(c)[0])] = 0.0
    return out


def solve(p: Problem, z0: T.Array, deadline: float | None = None) -> tuple[T.Array, SolveLog]:
    """Newton from z0 (module docstring), with an active set for the ridge bounds c·z ≥ 0.

    Each iteration: a bound on its boundary (|c·z| < ACTIVE_TOL) is active if the gradient
    pushes it out (c·g > 0, the multiplier is positive) or the Newton step on the other
    unknowns would (c·step < 0); Newton then runs in the null space of the active bounds. A
    step that would break an inactive bound is cut at the bound (the fraction to the boundary),
    and z is put on the bound exactly there. The other constraints (the root slopes, no interior
    minimum) are feasibility checks in the line search; the basis limits the step for its
    geometry (`step_limit`). deadline: a time.time() value; the solve stops there ("wall
    time")."""
    z = np.asarray(z0, float).copy()
    E = p.total(z)
    if not math.isfinite(E):
        raise ValueError("the start is not valid")
    log = SolveLog(energy=[E])
    for _ in range(MAX_NEWTON):
        if deadline is not None and time.time() > deadline:
            log.reason = "wall time"
            break
        g, H = gradient_hessian(p, z)
        cons = p.bounds(z)
        active: list[T.Array] = []
        for c in cons:
            if abs(c @ z) < ACTIVE_TOL and c @ g > 0:
                active.append(c)
                z = _project(z, c)
        step, mu = _newton_step(g, H, active)
        late = [c for c in cons if not any(c is a for a in active) and abs(c @ z) < ACTIVE_TOL and c @ step < 0]
        if late:
            # on the bound and the step leaves it: the bound is active for this step
            for c in late:
                active.append(c)
                z = _project(z, c)
            step, mu = _newton_step(g, H, active)
        dec = 0.5 * float(-g @ step)
        log.decrement.append(dec)
        log.damping.append(mu)
        log.active.append(len(active))
        if dec < NEWTON_TOL:
            log.converged, log.reason = True, "decrement"
            log.step.append(0.0)
            break
        t = 1.0
        t_bound = math.inf
        c_bound: T.Array | None = None
        for c in cons:
            if any(c is a for a in active):
                continue
            if c @ step < 0 and c @ z >= 0:
                tb = (c @ z) / -(c @ step)
                if tb < t_bound:
                    t_bound, c_bound = tb, c
        t = min(1.0, t_bound)
        t_geo = p.basis.step_limit(z, step, FREE_NODE_CUT)
        if t_geo < t:
            t = t_geo
            t_bound = math.inf if t < t_bound else t_bound
        if t <= 0.0:
            log.reason = "zero step at a bound"
            log.step.append(0.0)
            break
        for _ls in range(MAX_LINE):
            zt = z + t * step
            if t == t_bound and c_bound is not None:
                zt = _project(zt, c_bound)
            Et = p.total(zt)
            if Et <= E + 1e-4 * t * float(g @ step):
                break
            t *= 0.5
        else:
            log.reason = "line search failed"
            log.step.append(0.0)
            break
        z = zt
        E = Et
        log.energy.append(E)
        log.step.append(t)
        if t < MIN_STEP:
            log.reason = "stalled (step below MIN_STEP)"
            break
        if p.basis.free_node_at_end(z, EDGE_TOL):
            log.reason = "free node at a cell end"
            break
    else:
        log.reason = "max iterations"
    return z, log


# ---------------------------------------------------------------- readouts


def crossings(p: Problem, z: T.Array) -> list[tuple[float, int]]:
    """The crossings of τ = 0: (position, sign of the slope there: +1 when τ rises through 0,
    that is a top reached from the left; −1 a top reached from the right)."""
    return p.basis.crossings(z)


@dataclass
class Laplace:
    x: T.Array  # the points
    mean: T.Array  # τ*(x)
    sd: T.Array  # σ(x)
    p_beyond: T.Array  # P(τ(x) > 0)
    min_eig: float  # the smallest eigenvalue of H (> 0 for a defined approximation)


def laplace(p: Problem, z: T.Array, xq: T.Array) -> Laplace:
    _, H = gradient_hessian(p, z)
    w = np.linalg.eigvalsh(H)
    C = np.linalg.inv(H)
    zt = torch.as_tensor(z, dtype=F64)
    A = torch.autograd.functional.jacobian(lambda zz: p.tau_at(zz, xq), zt).numpy()  # (m, nz)
    mean = p.tau_at(zt, xq).detach().numpy()
    var = np.einsum("mi,ij,mj->m", A, C, A)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        pb = ndtr(np.where(sd > 0, mean / sd, np.sign(mean) * np.inf))
    return Laplace(np.asarray(xq), mean, sd, pb, float(w[0]))


def reversal_x(p: Problem, z: T.Array) -> float | None:
    """The position of the first extremum of τ (where ∇τ changes sign), or None."""
    ex = p.basis.extrema(z)
    return ex[0].x if ex else None


# ---------------------------------------------------------------- the kink enrichment (K3a′)

MAX_SWITCH = 40  # passes of the enrichment switch (a ridge moves one cell per pass or two)


@dataclass
class KinkLog:
    passes: list[dict[str, object]] = field(default_factory=list)

    @property
    def newton(self) -> int:
        return int(sum(int(q["newton"]) for q in self.passes))  # type: ignore[call-overload]


def _with_cell(p: Problem, cell: int | None) -> Problem:
    return Problem(p.x, p.log_odds_t, p.settings, enrich_cell=cell, roots=p.roots,
                   skip_reversal=p.skip_reversal, energy_form=p.energy_form, mm_scale=p.mm_scale,
                   slope_bounds=p.slope_bounds)


def _field(p: Problem, z: T.Array) -> tuple[T.Array, T.Array]:
    tau, xs = p.split(torch.as_tensor(z, dtype=F64))
    return tau.numpy(), xs.numpy()


def solve_kink(p0: Problem, tau0: T.Array, cell0: int | None,
               tau_k: float | None = None, x_k: float | None = None,
               deadline: float | None = None) -> tuple[Problem, T.Array, KinkLog]:
    """Solve with the kink enrichment decided from the field (module docstring, K3a′).

    Start: τ0 at the fixed nodes, the enrichment in cell0 (or none) with (tau_k, x_k).
    Loop: solve. If enriched and the ridge is on a fixed node (not on the free node), drop the
    enrichment. If enriched and the free node is at a cell end, the kink is on that node:
    remove the enrichment and solve again. If not enriched and the field has a reversal at a
    fixed node r, try the enrichment in the cells r − 1 and r (the extra node at the cell's
    middle, with the kink moved there), solve each, and keep the lowest energy if it is lower than
    the plain one by more than 1e-12 (if its free node ended at a cell end, the next pass moves
    the kink to that node). Else stop. Each kept change lowers the energy, so the loop ends."""
    log = KinkLog()
    p = _with_cell(p0, cell0)
    z = p.pack(tau0, tau_k, x_k)
    for _ in range(MAX_SWITCH):
        if deadline is not None and time.time() > deadline:
            log.passes.append(dict(cell=p.enrich_cell, newton=0, reason="wall time", energy=p.total(z),
                                   converged=False))
            return p, z, log
        z, sl = solve(p, z, deadline)
        log.passes.append(dict(cell=p.enrich_cell, newton=sl.iterations, reason=sl.reason, active=sl.active[-1] if sl.active else 0,
                               energy=sl.energy[-1], converged=sl.converged))
        if p.enrich_cell is not None and ridge_index(p, z) not in (None, len(p.x)):
            # the ridge is on a fixed node, not on the free node: the enrichment is in the wrong
            # cell; drop it (the next pass tries the two cells next to that node)
            p = _with_cell(p, None)
            z = p.fixed_tau(z).copy()
            continue
        if p.enrich_cell is not None:
            e = p.enrich_cell
            xk = float(z[-1])
            L = p.x[e + 1] - p.x[e]
            at_left = xk - p.x[e] < EDGE_TOL * L
            at_right = p.x[e + 1] - xk < EDGE_TOL * L
            if not (at_left or at_right):
                return p, z, log
            # the kink is on a fixed node: drop the extra node (the field is unchanged to the
            # precision of the node's distance from the cell end)
            # the merged node takes the free node's value (they are within EDGE_TOL · L)
            tk = float(z[len(p.x)])
            z = p.fixed_tau(z).copy()
            z[e if at_left else e + 1] = tk
            p = _with_cell(p, None)
            continue
        tau, _ = _field(p, z)
        d = np.diff(tau)
        rev = np.nonzero(d[1:] * d[:-1] < 0)[0]
        if len(rev) == 0:
            return p, z, log
        r = int(rev[0]) + 1
        E_plain = sl.energy[-1]
        best: tuple[float, Problem, T.Array] | None = None
        for cell in (r - 1, r):
            q = _with_cell(p, cell)
            # the extra node at the cell's middle, with the kink moved there: the slope of the
            # other side of node r is continued to it, so node r is on a straight part
            xk = 0.5 * (q.x[cell] + q.x[cell + 1])
            g_other = (tau[r] - tau[r - 1]) / (q.x[r] - q.x[r - 1]) if cell == r else \
                (tau[r + 1] - tau[r]) / (q.x[r + 1] - q.x[r])
            zq0 = q.pack(tau, tau[r] + g_other * (xk - q.x[r]), xk)
            if not math.isfinite(q.total(zq0)):
                log.passes.append(dict(cell=cell, newton=0, reason="trial start not feasible",
                                       energy=math.inf, converged=False, trial=True))
                continue
            zq, slq = solve(q, zq0, deadline)
            log.passes.append(dict(cell=cell, newton=slq.iterations, reason=slq.reason,
                                   energy=slq.energy[-1], converged=slq.converged, trial=True))
            if slq.energy[-1] < E_plain - 1e-12 and (best is None or slq.energy[-1] < best[0]):
                best = (slq.energy[-1], q, zq)
        if best is None:
            return p, z, log
        _, p, z = best
        # a trial whose free node ended at a cell end is handled by the next pass (the
        # enrichment is removed there and the kink is on that node)
    return p, z, log
