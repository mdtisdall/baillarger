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

import mfem.ser as mfem
import numpy as np
import torch
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402

F64 = torch.float64
QUAD_ORDER = 39  # 20 Gauss points on each part of a cell
NEWTON_TOL = 1e-14
MAX_NEWTON = 200
MAX_LINE = 60


# ---------------------------------------------------------------- the mesh (MFEM)


def mfem_nodes(x: T.Array) -> T.Array:
    """Build MFEM's 1D mesh on the nodes x (a segment mesh, vertices moved to x) and return its
    vertex positions; checks that the H1 order 1 dofs are the vertices in order."""
    n = len(x) - 1
    mesh = mfem.Mesh.MakeCartesian1D(n, 1.0)
    for i, v in enumerate(mesh.GetVertexArray()):
        v[0] = float(x[i])
    xv = np.array([float(v[0]) for v in mesh.GetVertexArray()])
    h1 = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(1, 1))
    for e in range(n):
        if list(h1.GetElementDofs(e)) != [e, e + 1]:
            raise ValueError("unexpected H1 dof order")
    return xv


def gauss_rule() -> tuple[T.Array, T.Array]:
    ir = mfem.IntRules.Get(mfem.Geometry.SEGMENT, QUAD_ORDER)
    qx = np.array([ir.IntPoint(i).x for i in range(ir.GetNPoints())])
    qw = np.array([ir.IntPoint(i).weight for i in range(ir.GetNPoints())])
    return qx, qw


# ---------------------------------------------------------------- the problem


@dataclass
class Problem:
    """One 1D line in the S1 form. log_odds_t: the stop field's log-odds at x (Torch,
    differentiable)."""

    x: T.Array  # (n + 1,) the fixed nodes
    log_odds_t: Callable[[torch.Tensor], torch.Tensor]
    settings: T.SolverSettings = T.DEFAULTS
    enrich_cell: int | None = None  # the cell with the extra free node (K3a′), or None
    roots: tuple[tuple[float, float], ...] = ()  # (x_root, n): the white surface's root and its normal (±1 along x)
    skip_reversal: bool = True
    qx: T.Array = field(init=False)
    qw: T.Array = field(init=False)

    def __post_init__(self) -> None:
        self.x = mfem_nodes(np.asarray(self.x, float))
        self.qx, self.qw = gauss_rule()
        st = self.settings
        lam = 1.0 / (st.speed_log_sd**2 * st.fem_length_mm)
        self.lam_a = lam * T.VOXEL_MM**2
        self.ell = st.fem_length_mm
        self.inv_hv = 1.0 / T.VOXEL_MM

    @property
    def n_unknowns(self) -> int:
        return len(self.x) + (2 if self.enrich_cell is not None else 0)

    def split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(τ at all nodes, all node positions), the extra node inserted, from the unknowns z =
        [τ at the fixed nodes, (τ_k, x_k) if enriched]."""
        n1 = len(self.x)
        tau = z[:n1]
        xs = torch.as_tensor(self.x, dtype=F64)
        if self.enrich_cell is not None:
            c = self.enrich_cell + 1
            tau = torch.cat([tau[:c], z[n1 : n1 + 1], tau[c:]])
            xs = torch.cat([xs[:c], z[n1 + 1 : n1 + 2], xs[c:]])
        return tau, xs

    def pack(self, tau: T.Array, tau_k: float | None = None, x_k: float | None = None) -> T.Array:
        """The unknowns from τ at the fixed nodes (and the extra node's value and position;
        by default the cell's midpoint, on the line)."""
        tau = np.asarray(tau, float)
        if self.enrich_cell is None:
            return tau.copy()
        e = self.enrich_cell
        if x_k is None:
            x_k = 0.5 * (self.x[e] + self.x[e + 1])
        if tau_k is None:
            tau_k = float(np.interp(x_k, self.x[e : e + 2], tau[e : e + 2]))
        return np.r_[tau, tau_k, x_k]

    def fixed_tau(self, z: T.Array) -> T.Array:
        return np.asarray(z[: len(self.x)], float)

    # ---- the energy

    def _quad(self, a: torch.Tensor, b: torch.Tensor, sign: float) -> torch.Tensor:
        qx = torch.as_tensor(self.qx, dtype=F64)
        qw = torch.as_tensor(self.qw, dtype=F64)
        xs = a + (b - a) * qx
        return (b - a) * (qw * torch.nn.functional.softplus(sign * self.log_odds_t(xs))).sum()

    def _quad_v(self, a: torch.Tensor, b: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """Per cell: ∫_a^b softplus(sign · ℓo) dx by the Gauss rule (vectors a, b, sign)."""
        qx = torch.as_tensor(self.qx, dtype=F64)
        qw = torch.as_tensor(self.qw, dtype=F64)
        xq = a[:, None] + (b - a)[:, None] * qx[None, :]
        f = torch.nn.functional.softplus(sign[:, None] * self.log_odds_t(xq))
        return (b - a) * (f * qw[None, :]).sum(1)

    def energy_terms(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        tau, xs = self.split(z)
        L = xs[1:] - xs[:-1]
        g = (tau[1:] - tau[:-1]) / L
        u = -torch.log(torch.abs(g))
        e_u2 = (L * u * u).sum() / (self.ell * self.ell)
        du = u[1:] - u[:-1]
        d = 0.5 * (L[1:] + L[:-1])
        keep = torch.ones_like(du)
        if self.skip_reversal:
            gs = g.detach()
            keep = torch.where(gs[1:] * gs[:-1] < 0, torch.zeros_like(du), keep)
        e_du = (keep * du * du / d).sum()
        e_speed = 0.5 * self.lam_a * (e_u2 + e_du)
        # the stop term, all cells at once: each cell is split at its crossing c (c = b when τ
        # does not change sign); the first part has the sign of τ at a, the second the other
        a_, b_ = xs[:-1], xs[1:]
        ta, tb = tau[:-1], tau[1:]
        cross = (ta.detach() < 0) != (tb.detach() < 0)
        frac = torch.where(cross, -ta / torch.where(cross, tb - ta, torch.ones_like(ta)), torch.ones_like(ta))
        c_ = a_ + (b_ - a_) * frac
        s1 = torch.where(ta.detach() < 0, 1.0, -1.0).to(F64)  # moving (+ℓo) where τ < 0
        e_stop = (self._quad_v(a_, c_, s1) + self._quad_v(c_, b_, -s1)).sum()
        e_stop = e_stop * self.inv_hv
        return dict(speed=e_speed, stop=e_stop, total=e_speed + e_stop)

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        return self.energy_terms(z)["total"]

    def valid(self, z: T.Array) -> bool:
        """The cells have positive length, no slope is 0, and n·∇τ > 0 on the root cells."""
        tau, xs = self.split(torch.as_tensor(z, dtype=F64))
        L = (xs[1:] - xs[:-1]).numpy()
        dt = (tau[1:] - tau[:-1]).numpy()
        ok = bool(np.all(L > 0) and np.all(dt != 0) and np.all(np.isfinite(dt)))
        if ok:
            # Q2c and Q3d (rules of the model): no interior local minimum of τ (columns start
            # only at the roots), and a local maximum (a ridge) only where τ ≥ 0
            tn = tau.numpy()
            rev = np.nonzero(dt[1:] * dt[:-1] < 0)[0] + 1
            if np.any(dt[rev - 1] < 0) or np.any(tn[rev] < 0):
                return False
            xn = xs.numpy()
            for xr, nr in self.roots:
                e = int(np.clip(np.searchsorted(xn, xr, side="right") - 1, 0, len(L) - 1))
                if not nr * dt[e] / L[e] > 0:
                    return False
        return ok

    def active_constraints(self, z: T.Array, tol: float = 1e-9) -> dict[str, int]:
        """The constraints within tol of their bound at z (root slopes and reversal values)."""
        tau, xs = self.split(torch.as_tensor(z, dtype=F64))
        tn, xn = tau.numpy(), xs.numpy()
        L = np.diff(xn)
        dt = np.diff(tn)
        rev = np.nonzero(dt[1:] * dt[:-1] < 0)[0] + 1
        n_root = 0
        for xr, nr in self.roots:
            e = int(np.clip(np.searchsorted(xn, xr, side="right") - 1, 0, len(L) - 1))
            n_root += int(nr * dt[e] / L[e] < tol)
        return dict(root=n_root, reversal=int(np.sum(np.abs(tn[rev]) < tol)))

    def n_reversals(self, z: T.Array) -> int:
        tau, _ = self.split(torch.as_tensor(z, dtype=F64))
        d = np.diff(tau.numpy())
        return int(np.sum(d[1:] * d[:-1] < 0))

    def total(self, z: T.Array) -> float:
        if not self.valid(z):
            return math.inf
        with torch.no_grad():
            E = float(self.energy(torch.as_tensor(z, dtype=F64)))
        return E if math.isfinite(E) else math.inf

    # ---- the field at points

    def tau_at(self, z: torch.Tensor, xq: T.Array) -> torch.Tensor:
        """The P1 interpolant at the points xq (Torch, differentiable in z)."""
        tau, xs = self.split(z)
        xsn = xs.detach().numpy()
        e = np.clip(np.searchsorted(xsn, xq, side="right") - 1, 0, len(xsn) - 2)
        xq_t = torch.as_tensor(xq, dtype=F64)
        t = (xq_t - xs[e]) / (xs[e + 1] - xs[e])
        return tau[e] * (1 - t) + tau[e + 1] * t


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
    """The index in z of the τ value of the ridge (the node where the slope changes from + to
    −), or None. The full node arrays hold the extra node at enrich_cell + 1."""
    tau, _ = p.split(torch.as_tensor(z, dtype=F64))
    d = np.diff(tau.numpy())
    k = np.nonzero((d[:-1] > 0) & (d[1:] < 0))[0]
    if len(k) == 0:
        return None
    i = int(k[0]) + 1
    if p.enrich_cell is None:
        return i
    c = p.enrich_cell + 1
    return i if i < c else (len(p.x) if i == c else i - 1)


def _newton_step(g: T.Array, H: T.Array, free: T.BoolArray) -> tuple[T.Array, float]:
    """The Newton step on the free unknowns (0 on the others), with the Levenberg shift μ
    that makes the free Hessian positive definite."""
    gf, Hf = g[free], H[np.ix_(free, free)]
    mu = 0.0
    scale = max(1e-12, float(np.max(np.abs(np.diag(Hf)))))
    while True:
        try:
            C = np.linalg.cholesky(Hf + mu * np.eye(len(gf)))
            break
        except np.linalg.LinAlgError:
            mu = max(2.0 * mu, 1e-10 * scale)
    step = np.zeros(len(g))
    step[free] = -np.linalg.solve(C.T, np.linalg.solve(C, gf))
    return step, mu


def solve(p: Problem, z0: T.Array, deadline: float | None = None) -> tuple[T.Array, SolveLog]:
    """Newton from z0 (module docstring), with an active set for the ridge's bound τ ≥ 0.

    Each iteration: if the ridge's τ is on its bound (|τ| < ACTIVE_TOL) and the gradient
    pushes it down (∂E/∂τ_r > 0, the multiplier is positive) or the free Newton step would move
    it below 0, τ_r is fixed at 0 and Newton runs
    on the other unknowns; else it is free. A step that would make the ridge's τ negative is
    cut at the bound (the fraction to the boundary), and τ_r is set to 0 exactly there. The
    other constraints (the root slopes, no interior minimum) are feasibility checks in the
    line search. deadline: a time.time() value; the solve stops there ("wall time")."""
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
        free = np.ones(len(z), bool)
        r = ridge_index(p, z)
        if r is not None and abs(z[r]) < ACTIVE_TOL and g[r] > 0:
            free[r] = False
            z[r] = 0.0
        step, mu = _newton_step(g, H, free)
        if r is not None and free[r] and abs(z[r]) < ACTIVE_TOL and step[r] < 0:
            # on the bound and the free step leaves it: the bound is active for this step
            free[r] = False
            z[r] = 0.0
            step, mu = _newton_step(g, H, free)
        dec = 0.5 * float(-g @ step)
        log.decrement.append(dec)
        log.damping.append(mu)
        log.active.append(int((~free).sum()))
        if dec < NEWTON_TOL:
            log.converged, log.reason = True, "decrement"
            log.step.append(0.0)
            break
        t = 1.0
        t_bound = math.inf
        if r is not None and free[r] and step[r] < 0 and z[r] >= 0:
            t_bound = z[r] / -step[r]
            t = min(1.0, t_bound)
        if p.enrich_cell is not None:
            # the free node stays inside its cell: cut the step at FREE_NODE_CUT of the way to
            # the cell end (the switch in solve_kink moves the node when it reaches an end)
            e = p.enrich_cell
            xk, dx = z[-1], step[-1]
            room = (p.x[e + 1] - xk) if dx > 0 else (xk - p.x[e])
            if dx != 0 and abs(dx) * t > FREE_NODE_CUT * room:
                t = FREE_NODE_CUT * room / abs(dx)
                t_bound = math.inf if t < t_bound else t_bound
        if t <= 0.0:
            log.reason = "zero step at a bound"
            log.step.append(0.0)
            break
        for _ls in range(MAX_LINE):
            zt = z + t * step
            if t == t_bound:
                zt[r] = 0.0
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
        if p.enrich_cell is not None:
            e = p.enrich_cell
            L = p.x[e + 1] - p.x[e]
            if min(z[-1] - p.x[e], p.x[e + 1] - z[-1]) < EDGE_TOL * L:
                log.reason = "free node at a cell end"
                break
    else:
        log.reason = "max iterations"
    return z, log


# ---------------------------------------------------------------- readouts


def crossings(p: Problem, z: T.Array) -> list[tuple[float, int]]:
    """The crossings of τ = 0: (position, sign of the slope there: +1 when τ rises through 0,
    that is a top reached from the left; −1 a top reached from the right)."""
    tau, xs = (a.numpy() for a in p.split(torch.as_tensor(z, dtype=F64)))
    out = []
    for e in range(len(xs) - 1):
        a, b = tau[e], tau[e + 1]
        if (a < 0) != (b < 0):
            c = xs[e] + (xs[e + 1] - xs[e]) * (-a) / (b - a)
            out.append((float(c), 1 if b > a else -1))
    return out


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


def reversal_node(p: Problem, z: T.Array) -> int | None:
    """The node where the slope changes sign (the first one), or None."""
    tau, _ = p.split(torch.as_tensor(z, dtype=F64))
    d = np.diff(tau.numpy())
    k = np.nonzero(d[1:] * d[:-1] < 0)[0]
    return int(k[0]) + 1 if len(k) else None


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
                   skip_reversal=p.skip_reversal)


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
