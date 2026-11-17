"""The continuous column energy on a 1D line, and its Newton solver (route R1, stop form S2;
docs/plans/continuous-column-solver.md, draft).

The unknowns
------------
- τ: continuous, piecewise linear (MFEM H1, order 1) on the line's cells. Its nodal values are
  the unknowns.
- s: the stop position (the free boundary between the moving and the stopped region). The
  moving region is [x_start, s], the stopped region (s, x_end].
- J: the jump at the stop, J = τ_top − τ(s) = −τ(s) ≥ 0 (S2).

The speed is not an unknown (R1): on a P1 cell the slope g_e = τ'_e is constant, so the log
speed u_e = log v_e = −log g_e is eliminated exactly, per cell. No σ_c and no separate u field.

The energy (the 3D energy of a laterally uniform field, per voxel column of cross-section
A = VOXEL_MM², choice N)
----------------------------------------------------------------------------------------
E = E_speed + E_stop + E_jump, with

- E_speed = (λ A / 2) ∫_{x_start}^{s} (u'² + u² / ℓ²) dx, λ = 1 / (speed_log_sd² ℓ),
  ℓ = fem_length_mm (as ff_grid.precision_from; the mean of u is log speed_typical = 0).
  u is per cell (P0). The u² part integrates u_e² over the moving part of each cell. The u'²
  part is the two-point form: on the dual cell [m_e, m_e+1] between the cell midpoints, u' is
  (u_e+1 − u_e) / h; it is integrated over the moving part of the dual cell. So E_speed is a
  continuous function of s. Before the first midpoint u' = 0 (the natural boundary).
- E_stop = (1 / VOXEL_MM) [∫_{x_start}^{s} softplus(ℓo) dx + ∫_{s}^{x_end} softplus(−ℓo) dx],
  ℓo the oracle's log-odds. softplus(ℓo) = −log(1 − p) (moving), softplus(−ℓo) = −log p
  (stopped): ff_terms.unary per unit length (node_w = h / VOXEL_MM in 1D).
- E_jump: −log p0 if J = 0; −log(1 − p0) + log(mean) + J / mean if J > 0 (ff_terms.jump_energy).
- The alignment term is zero (the oracle tensor is along the line, coherence 1). There is no
  contrast term (one column has no neighbour).

The two branches of the jump. The jump prior has a point mass at 0, so the energy has two
smooth branches:

- "J0": J = 0. s is the crossing of τ = 0 (τ is increasing). Unknowns: the nodal τ. `solve`
  minimises this branch.
- "J+": J > 0, J = −τ(s), s free. On one line the offset of τ is free (no neighbour and no
  fixed start value), and only J reads it. So for each (slopes, s) the J+ energy is the J0-like
  energy + e1 + J / mean, which falls linearly as J -> 0+: the branch has no minimum, its
  infimum is min(J0) − e0 + e1, and Newton has a flat direction there. `solve` does not
  minimise it; `jump_branch` evaluates it along the shifts of the J0 solution.

Derivatives and the solver
--------------------------
MFEM gives the mesh, the P1 space, the slope operator (a MixedBilinearForm of H1 and L2 order 0
with MixedScalarDerivativeIntegrator, divided by the cell length) and the Gauss rule of the
cut-cell integrals. The pointwise energy is written in Torch (float64) and its exact gradient
and Hessian come from autograd: this is the "quadrature-point" form (MFEM's dfem and libCEED use
it with AD), not a hand-written Jacobian.

Newton with the exact Hessian of the active unknowns. If the Hessian is not positive definite,
a multiple of the identity is added (Levenberg) until the Cholesky factorisation succeeds. A
backtracking line search on the energy (Armijo, c = 1e-4) keeps every slope > 0 (else the energy
is +∞), the crossing inside the line, and J > 0 in the J+ branch. The iteration stops when the
Newton decrement ½ gᵀ H⁻¹ g is below NEWTON_TOL, or at MAX_NEWTON iterations.

Active unknowns. The energy reads the nodes of the cells and dual cells that hold a part of the
moving region. The other nodes are not unknowns; after each step they are set by a linear
extrapolation of the last active slope, so that a later step can move s into their cells.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mfem.ser as mfem
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402
import ffc_line as LN  # noqa: E402

Branch = Literal["J0", "J+"]

QUAD_ORDER = 39  # the Gauss rule on each cut interval (20 points): the test checks it
NEWTON_TOL = 1e-14  # the Newton decrement at which the solve stops (energy units, nats)
MAX_NEWTON = 100
MAX_LINE = 60  # halvings of the line search
J_FLOOR = 1e-10  # J+ branch: below this J the branch is at its boundary J -> 0+


# ---------------------------------------------------------------- the discretisation (MFEM)


@dataclass
class Discretisation:
    """The mesh and the operators of one line at one spacing."""

    line: LN.Line
    h: float
    x: T.Array  # (n + 1,) node positions (mm)
    G: T.Array  # (n, n + 1) slope operator: g = G τ
    qx: T.Array  # (Q,) Gauss points on [0, 1]
    qw: T.Array  # (Q,) their weights (sum 1)
    mesh: object = field(repr=False, default=None)

    @property
    def n_cells(self) -> int:
        return len(self.x) - 1


def discretise(line: LN.Line, h: float) -> Discretisation:
    """MFEM's 1D mesh of the line, the H1 order 1 space and the slope operator."""
    x = line.nodes(h)
    n = len(x) - 1
    mesh = mfem.Mesh.MakeCartesian1D(n, float(x[-1] - x[0]))
    xv = np.array([float(v[0]) for v in mesh.GetVertexArray()]) + x[0]
    if not np.allclose(xv, x, atol=1e-12):
        raise ValueError("MFEM's vertices are not the line's nodes")
    h1 = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(1, 1))
    l2 = mfem.FiniteElementSpace(mesh, mfem.L2_FECollection(0, 1))
    b = mfem.MixedBilinearForm(h1, l2)
    b.AddDomainIntegrator(mfem.MixedScalarDerivativeIntegrator())
    b.Assemble()
    b.Finalize()
    B = b.SpMat()
    rows = np.asarray(B.GetIArray())
    cols = np.asarray(B.GetJArray())
    vals = np.asarray(B.GetDataArray())
    Bd = np.zeros((B.Height(), B.Width()))
    for r in range(B.Height()):
        Bd[r, cols[rows[r] : rows[r + 1]]] = vals[rows[r] : rows[r + 1]]
    # the H1 order 1 dofs of a 1D mesh are the vertices, in vertex order (checked)
    for e in range(n):
        if list(h1.GetElementDofs(e)) != [e, e + 1]:
            raise ValueError("unexpected H1 dof order")
    G = Bd / np.diff(x)[:, None]  # ∫_e τ' / |e|: the slope (L2 order-0 mass is diag |e|)
    ir = mfem.IntRules.Get(mfem.Geometry.SEGMENT, QUAD_ORDER)
    qx = np.array([ir.IntPoint(i).x for i in range(ir.GetNPoints())])
    qw = np.array([ir.IntPoint(i).weight for i in range(ir.GetNPoints())])
    return Discretisation(line, h, x, G, qx, qw, mesh)


# ---------------------------------------------------------------- the energy (Torch)


def _log_odds_t(line: LN.Line, x: torch.Tensor) -> torch.Tensor:
    """The oracle log-odds at x (Torch, differentiable in x): the formula of ffc_line."""
    st = line.settings
    sigma = st.oracle_blur_vox * T.VOXEL_MM
    sdf = x - line.x_stop_true  # slab: stop_sdf = x − t (ff_simple.SlabGeometry)
    p = torch.special.ndtr(sdf / sigma)
    lo, hi = st.oracle_clip
    p = torch.clamp(p, lo, hi)
    return torch.log(p) - torch.log1p(-p)


def _quad(d: Discretisation, a: torch.Tensor, b: torch.Tensor, sign: float) -> torch.Tensor:
    """∫_a^b softplus(sign · ℓo) dx by the Gauss rule (a, b scalars, b ≥ a)."""
    qx = torch.as_tensor(d.qx)
    qw = torch.as_tensor(d.qw)
    xs = a + (b - a) * qx
    return (b - a) * (qw * torch.nn.functional.softplus(sign * _log_odds_t(d.line, xs))).sum()


@dataclass(frozen=True)
class Constants:
    lam_a: float  # λ A
    ell: float
    inv_hv: float
    e0: float  # −log p0
    e1: float  # −log(1 − p0) + log(mean)
    jump_mean: float


def constants(st: T.SolverSettings) -> Constants:
    lam = 1.0 / (st.speed_log_sd**2 * st.fem_length_mm)
    return Constants(
        lam_a=lam * T.VOXEL_MM**2,
        ell=st.fem_length_mm,
        inv_hv=1.0 / T.VOXEL_MM,
        e0=-math.log(st.jump_p_zero),
        e1=-math.log(1.0 - st.jump_p_zero) + math.log(st.jump_mean),
        jump_mean=st.jump_mean,
    )


def stop_cell(x: T.Array, s: float) -> int:
    """The cell c with x_c <= s < x_c+1 (the last cell if s = x_end)."""
    return int(min(max(np.searchsorted(x, s, side="right") - 1, 0), len(x) - 2))


def crossing(x: T.Array, tau: T.Array) -> float | None:
    """The first crossing of τ = 0 by the P1 interpolant, or None (τ < 0 at the end, or
    τ(x_start) >= 0)."""
    if tau[0] >= 0.0:
        return None
    k = np.nonzero(tau >= 0.0)[0]
    if len(k) == 0:
        return None
    c = int(k[0]) - 1
    return float(x[c] + (x[c + 1] - x[c]) * (-tau[c]) / (tau[c + 1] - tau[c]))


def n_active(x: T.Array, s: float) -> int:
    """The number of leading nodes that the energy reads when the stop is at s: the nodes of the
    cells and of the dual cells that hold a part of [x_start, s]."""
    c = stop_cell(x, s)
    mid_c = 0.5 * (x[c] + x[c + 1])
    k = c + 1 if (s > mid_c and c + 1 < len(x) - 1) else c  # the last cell read
    return k + 2


def energy_t(
    d: Discretisation, k: Constants, tau: torch.Tensor, s: torch.Tensor, branch: Branch
) -> dict[str, torch.Tensor]:
    """The energy terms (Torch scalars) at nodal τ (all n + 1 nodes) and stop s.

    Only the nodes that `n_active` names are read. The caller has checked that s is inside
    the line and that the slopes that are read are > 0."""
    x = d.x
    h = d.h
    sv = float(s.detach())
    c = stop_cell(x, sv)
    na = n_active(x, sv)
    ncell = na - 1  # cells 0 .. ncell − 1 are read
    G = torch.as_tensor(d.G[:ncell, :na])
    g = G @ tau[:na]
    u = -torch.log(g)
    xt = torch.as_tensor(x)
    # u² on the moving part of each cell
    w = torch.zeros(ncell, dtype=torch.float64)
    w[:c] = h
    w = w + torch.where(torch.arange(ncell) == c, s - xt[c], torch.zeros((), dtype=torch.float64))
    e_u2 = (w * u * u).sum() / (k.ell * k.ell)
    # u'² on the moving part of each dual cell [m_e, m_e+1]
    e_du = torch.zeros((), dtype=torch.float64)
    if ncell >= 2:
        m = 0.5 * (xt[:ncell] + xt[1:na])
        lo_ = m[:-1]
        hi_ = m[1:]
        part = torch.clamp(torch.minimum(hi_, s) - lo_, min=0.0)
        du = (u[1:] - u[:-1]) / h
        e_du = (part * du * du).sum()
    e_speed = 0.5 * k.lam_a * (e_u2 + e_du)
    # the stop term: whole cells before c (moving), after c (stopped), and the cut cell
    a = torch.as_tensor(float(x[c]), dtype=torch.float64)
    b = torch.as_tensor(float(x[c + 1]), dtype=torch.float64)
    e_stop = _quad(d, a, s, 1.0) + _quad(d, s, b, -1.0) + torch.as_tensor(_whole_cells(d, c), dtype=torch.float64)
    e_stop = e_stop * k.inv_hv
    # the jump
    tau_s = tau[c] + (s - xt[c]) / h * (tau[c + 1] - tau[c])
    if branch == "J0":
        e_jump = torch.as_tensor(k.e0, dtype=torch.float64)
        J = torch.zeros((), dtype=torch.float64)
    else:
        J = -tau_s
        e_jump = k.e1 + J / k.jump_mean
    return dict(speed=e_speed, stop=e_stop, jump=e_jump, total=e_speed + e_stop + e_jump,
                J=J, tau_s=tau_s)


_WHOLE_CACHE: dict[tuple[int, float, int], float] = {}


def _whole_cells(d: Discretisation, c: int) -> float:
    """Σ over the whole cells before c of ∫ softplus(ℓo), plus Σ over the cells after c of
    ∫ softplus(−ℓo) (a constant for a fixed c)."""
    key = (id(d), d.h, c)
    if key not in _WHOLE_CACHE:
        tot = 0.0
        with torch.no_grad():
            for e in range(d.n_cells):
                if e == c:
                    continue
                a = torch.as_tensor(float(d.x[e]), dtype=torch.float64)
                b = torch.as_tensor(float(d.x[e + 1]), dtype=torch.float64)
                tot += float(_quad(d, a, b, 1.0 if e < c else -1.0))
        _WHOLE_CACHE[key] = tot
    return _WHOLE_CACHE[key]


# ---------------------------------------------------------------- the state and the solver


@dataclass
class State:
    tau: T.Array  # (n + 1,)
    s: float


def _valid(d: Discretisation, st: State, branch: Branch) -> bool:
    if not (d.x[0] < st.s < d.x[-1]):
        return False
    na = n_active(d.x, st.s)
    if not np.all(np.diff(st.tau[:na]) > 0.0):
        return False
    if branch == "J0":
        cr = crossing(d.x, st.tau)
        return cr is not None and abs(cr - st.s) < 1e-9
    c = stop_cell(d.x, st.s)
    tau_s = st.tau[c] + (st.s - d.x[c]) / d.h * (st.tau[c + 1] - st.tau[c])
    return bool(-tau_s > 0.0)


def _extrapolate(d: Discretisation, st: State) -> None:
    """Set the nodes after the active ones by the last active slope (in place)."""
    na = n_active(d.x, st.s)
    if na < len(d.x):
        g = (st.tau[na - 1] - st.tau[na - 2]) / d.h
        st.tau[na:] = st.tau[na - 1] + g * (d.x[na:] - d.x[na - 1])


def total_energy(d: Discretisation, k: Constants, st: State, branch: Branch) -> float:
    """The energy of a state (+∞ if it is not valid)."""
    if not _valid(d, st, branch):
        return math.inf
    with torch.no_grad():
        return float(energy_t(d, k, torch.as_tensor(st.tau), torch.as_tensor(st.s, dtype=torch.float64), branch)["total"])


def _fun(d: Discretisation, k: Constants, st: State, branch: Branch, na: int):  # type: ignore[no-untyped-def]
    """The energy as a function of the active unknowns z (J0: τ[:na]; J+: τ[:na] and s)."""
    tau_rest = torch.as_tensor(st.tau[na:])

    def f(z: torch.Tensor) -> torch.Tensor:
        tau = torch.cat([z[:na], tau_rest])
        if branch == "J0":
            # s is the crossing of τ = 0 in the cell where it is now (a smooth function of τ)
            c = stop_cell(d.x, st.s)
            s = d.x[c] + d.h * (-tau[c]) / (tau[c + 1] - tau[c])
        else:
            s = z[na]
        return energy_t(d, k, tau, s, branch)["total"]

    return f


@dataclass
class SolveLog:
    energy: list[float] = field(default_factory=list)
    decrement: list[float] = field(default_factory=list)
    damping: list[float] = field(default_factory=list)
    step: list[float] = field(default_factory=list)
    s: list[float] = field(default_factory=list)
    converged: bool = False
    reason: str = ""

    @property
    def iterations(self) -> int:
        return len(self.decrement)


def jump_branch(d: Discretisation, k: Constants, st: State, Js: T.Array) -> T.Array:
    """The J+ energy of the J0 state `st` with τ shifted by −J (the same slopes and s), for
    each J of Js (> 0)."""
    out = []
    for J in Js:
        sh = State(st.tau - float(J), st.s)
        out.append(total_energy(d, k, sh, "J+"))
    return np.asarray(out)


def solve(d: Discretisation, k: Constants, start: State, branch: Branch = "J0") -> tuple[State, SolveLog]:
    """Newton on the J0 branch from `start` (module docstring). The J+ code path is kept for
    the test of its flat direction; on one line it does not converge."""
    st = State(start.tau.copy(), float(start.s))
    if branch == "J0":
        cr = crossing(d.x, st.tau)
        if cr is None:
            raise ValueError("the start has no crossing of τ = 0")
        st.s = cr
    _extrapolate(d, st)
    log = SolveLog()
    E = total_energy(d, k, st, branch)
    if not math.isfinite(E):
        raise ValueError("the start state is not valid")
    log.energy.append(E)
    log.s.append(st.s)
    for _ in range(MAX_NEWTON):
        na = n_active(d.x, st.s)
        z0 = torch.as_tensor(st.tau[:na].copy())
        if branch == "J+":
            z0 = torch.cat([z0, torch.as_tensor([st.s], dtype=torch.float64)])
        f = _fun(d, k, st, branch, na)
        z0.requires_grad_(True)
        g = torch.autograd.grad(f(z0), z0)[0].detach().numpy()
        H = torch.autograd.functional.hessian(f, z0.detach()).numpy()
        mu = 0.0
        scale = max(1e-12, float(np.max(np.abs(np.diag(H)))))
        while True:
            try:
                L = np.linalg.cholesky(H + mu * np.eye(len(g)))
                break
            except np.linalg.LinAlgError:
                mu = max(2.0 * mu, 1e-10 * scale)
        p = -np.linalg.solve(L.T, np.linalg.solve(L, g))
        dec = 0.5 * float(-g @ p)
        log.decrement.append(dec)
        log.damping.append(mu)
        if dec < NEWTON_TOL:
            log.converged = True
            log.reason = "decrement"
            log.step.append(0.0)
            break
        t = 1.0
        z = z0.detach().numpy()
        for _ls in range(MAX_LINE):
            zt = z + t * p
            trial = State(st.tau.copy(), st.s)
            trial.tau[:na] = zt[:na]
            if branch == "J+":
                trial.s = float(zt[na])
            else:
                cr = crossing(d.x, trial.tau)
                if cr is None:
                    t *= 0.5
                    continue
                trial.s = cr
            Et = total_energy(d, k, trial, branch)
            if Et <= E + 1e-4 * t * float(g @ p):
                break
            t *= 0.5
        else:
            log.reason = "line search failed"
            log.step.append(0.0)
            break
        st = trial
        _extrapolate(d, st)
        E = total_energy(d, k, st, branch)
        log.energy.append(E)
        log.step.append(t)
        log.s.append(st.s)
        if branch == "J+":
            c = stop_cell(d.x, st.s)
            J = -(st.tau[c] + (st.s - d.x[c]) / d.h * (st.tau[c + 1] - st.tau[c]))
            if J < J_FLOOR:
                log.reason = "J+ at its boundary J -> 0+"
                break
    else:
        log.reason = "max iterations"
    return st, log


def terms(d: Discretisation, k: Constants, st: State, branch: Branch) -> dict[str, float]:
    with torch.no_grad():
        e = energy_t(d, k, torch.as_tensor(st.tau), torch.as_tensor(st.s, dtype=torch.float64), branch)
    return {key: float(v) for key, v in e.items()}


def true_state(d: Discretisation) -> State:
    """The truth in the discrete space: τ at the nodes and the true stop (exact for slab1d,
    because the true τ is linear)."""
    return State(d.line.tau_true(d.x), d.line.x_stop_true)
