"""The τ representation behind one interface (the user's decision of 2026-11-12).

The energy, the evidence terms, the refinement indicator and the measures read τ only through
`TauBasis`. A later implementation (for example multiscale B-splines) can replace `P1Basis`
with no change to them.

`P1Basis` is the only implementation: MFEM H1 order 1 (ordinary continuous elements) on a 1D
mesh, with the optional kink enrichment of one cell (K3a′: one extra node with a free position
and a free value). Its unknowns z are [τ at the fixed nodes, (τ_k, x_k) if enriched].
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Protocol

import mfem.ser as mfem
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402

F64 = torch.float64
QUAD_ORDER = 39  # 20 Gauss points on each part of a cell


# The density of an evidence term: f(x, side) at the points x, side = +1 beyond the top (τ > 0)
# and −1 in tissue (τ < 0), a constant of the quadrature.
Density = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# The density of a term on ∇τ: f(∇τ, side, x) at the points x.
GradDensity = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class Extremum:
    """A local extremum of τ: its position, its value, "max" or "min", and the linear form c
    with c·z = τ at the extremum (the bound τ ≥ 0 of a ridge is c·z ≥ 0)."""

    x: float
    value: float
    kind: str
    c: T.Array


class TauBasis(Protocol):
    """The τ field: its unknowns z, its values and gradients at points, the assembly of the
    terms, the extrema, and the refinement. The energy, the evidence terms, the refinement
    indicator and the measures read τ only through these methods."""

    @property
    def n_unknowns(self) -> int:
        """The number of unknowns z."""
        ...

    @property
    def n_cells(self) -> int:
        """The number of cells (the units of marking and refinement)."""
        ...

    def cell_sizes(self, z: T.Array) -> T.Array:
        """The length of each cell."""
        ...

    def values(self, z: torch.Tensor, xq: T.Array) -> torch.Tensor:
        """τ at the points xq (differentiable in z)."""
        ...

    def gradients(self, z: torch.Tensor, xq: T.Array) -> torch.Tensor:
        """∇τ at the points xq (differentiable in z)."""
        ...

    def admissible(self, z: T.Array, flat_beyond_top: bool = False) -> bool:
        """z gives a valid field: valid geometry, finite values, ∇τ ≠ 0 (with flat_beyond_top:
        except where τ > 0 on the whole cell)."""
        ...

    def integrate_cut(self, z: torch.Tensor, density: Density, per_cell: bool = False) -> torch.Tensor:
        """∫ density(x, side(x)) dx over the line, by a quadrature split at τ = 0 (the side is
        constant on each part); per cell if per_cell (for an indicator), else the total.
        Differentiable in z."""
        ...

    def integrate_grad_cut(self, z: torch.Tensor, f: GradDensity) -> torch.Tensor:
        """∫ f(∇τ(x), side(x), x) dx over the line, split at τ = 0 (side +1 beyond the top, −1 in
        tissue). Differentiable in z."""
        ...

    def cell_rise_forms(self) -> list[tuple[float, T.Array]]:
        """For each cell: (its midpoint, the linear form c with c·z = τ(cell end) − τ(cell start)):
        the slope-sign rules as linear bounds."""
        ...

    def derived_norms(
        self, z: torch.Tensor, w_of_grad: Callable[[torch.Tensor], torch.Tensor], skip_reversal: bool,
        tissue_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(∫ w² dx, ∫ |∇w|² dx) for the derived field w = w_of_grad(∇τ). skip_reversal: leave
        out of ∫ |∇w|² the parts where ∇τ changes sign (K4b). tissue_only: ∫ |∇w|² only where
        τ ≤ 0 (each part weighted by its tissue fraction)."""
        ...

    def extrema(self, z: T.Array) -> list[Extremum]:
        """The local extrema of τ."""
        ...

    def crossings(self, z: T.Array) -> list[tuple[float, int]]:
        """The zeros of τ: (position, the sign of ∇τ there)."""
        ...

    def step_limit(self, z: T.Array, step: T.Array, cut: float) -> float:
        """The largest step length t such that z + t·step keeps a valid geometry, at most the
        fraction `cut` of the room (inf if the geometry does not depend on z)."""
        ...

    def cells_at(self, x: float) -> T.IntArray:
        """The cells that hold the point x (two if x is on a cell boundary): for marking."""
        ...

    def refine(self, cells: T.Array) -> TauBasis:
        """The basis with the given cells refined (neighbours at most one level apart)."""
        ...

    def prolong(self, z: T.Array, fine: TauBasis) -> T.Array:
        """z on a refined basis."""
        ...


# ---------------------------------------------------------------- helpers


def _seg_tissue(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The fraction of a segment where the linear τ from a to b is ≤ 0."""
    ad, bd = a.detach(), b.detach()
    one, zero = torch.ones_like(a), torch.zeros_like(a)
    den_ab = torch.where(ad > bd, a - b, one)
    den_ba = torch.where(bd > ad, b - a, one)
    return torch.where((ad <= 0) & (bd <= 0), one, torch.where((ad > 0) & (bd > 0), zero,
                       torch.where(ad <= 0, -a / den_ba, -b / den_ab)))


def _dual_tissue_fraction(tau: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """For each dual cell [m_e, m_{e+1}] (around node e + 1): the fraction where τ ≤ 0."""
    tm = 0.5 * (tau[1:] + tau[:-1])  # τ at the cell midpoints
    L = xs[1:] - xs[:-1]
    h1, h2 = 0.5 * L[:-1], 0.5 * L[1:]
    tj = tau[1:-1]
    f1 = _seg_tissue(tm[:-1], tj)
    f2 = _seg_tissue(tj, tm[1:])
    return (f1 * h1 + f2 * h2) / (h1 + h2)


# ---------------------------------------------------------------- MFEM


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


def graded(x: T.Array, cells: T.Array) -> T.Array:
    """Bisect the given cells, then bisect cells until neighbouring cells differ in length by
    at most a factor 2 (neighbours at most one level apart)."""
    cells = np.asarray(cells, int)
    x = np.sort(np.r_[x, 0.5 * (x[cells] + x[cells + 1])])
    while True:
        L = np.diff(x)
        bad = [e for e in range(len(L) - 1) if L[e] > 2 * L[e + 1] + 1e-15 or L[e + 1] > 2 * L[e] + 1e-15]
        if not bad:
            return x
        add = []
        for e in bad:
            k = e if L[e] > L[e + 1] else e + 1
            add.append(0.5 * (x[k] + x[k + 1]))
        x = np.unique(np.r_[x, add])


# ---------------------------------------------------------------- P1


@dataclass
class P1Basis:
    """MFEM H1 order 1 on the nodes x, with the optional kink enrichment of one cell."""

    x: T.Array  # (n + 1,) the fixed nodes
    enrich_cell: int | None = None
    qx: T.Array = field(init=False, repr=False)
    qw: T.Array = field(init=False, repr=False)
    _last: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.x = mfem_nodes(np.asarray(self.x, float))
        self.qx, self.qw = gauss_rule()

    @property
    def n_unknowns(self) -> int:
        return len(self.x) + (2 if self.enrich_cell is not None else 0)

    @property
    def n_cells(self) -> int:
        return len(self.x) - 1 + (1 if self.enrich_cell is not None else 0)

    def with_cell(self, cell: int | None) -> P1Basis:
        return P1Basis(self.x, cell)

    def cell_sizes(self, z: T.Array) -> T.Array:
        return np.diff(self.nodal(z)[1])

    def unknown_index(self, node: int) -> int:
        """The index in z of the τ value of a node of the full node array (the extra node of the
        enrichment is at enrich_cell + 1 in the full array and at len(x) in z)."""
        if self.enrich_cell is None:
            return node
        c = self.enrich_cell + 1
        return node if node < c else (len(self.x) if node == c else node - 1)

    # ---- the unknowns and the nodal field (P1 only)

    def split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(τ at all nodes, all node positions), the extra node inserted, from the unknowns z =
        [τ at the fixed nodes, (τ_k, x_k) if enriched]. The result for the last z (the same
        tensor object) is reused, so that the terms of one energy share one node field."""
        if self._last is not None and self._last[0] is z:
            return self._last[1], self._last[2]
        tau, xs = self._split(z)
        self._last = (z, tau, xs)
        return tau, xs

    def _split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n1 = len(self.x)
        tau = z[:n1]
        xs = torch.as_tensor(self.x, dtype=F64)
        if self.enrich_cell is not None:
            c = self.enrich_cell + 1
            tau = torch.cat([tau[:c], z[n1 : n1 + 1], tau[c:]])
            xs = torch.cat([xs[:c], z[n1 + 1 : n1 + 2], xs[c:]])
        return tau, xs

    def nodal(self, z: T.Array) -> tuple[T.Array, T.Array]:
        tau, xs = self.split(torch.as_tensor(z, dtype=F64))
        return tau.detach().numpy(), xs.detach().numpy()

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

    # ---- the field at points

    def _locate(self, xs: torch.Tensor, xq: T.Array) -> T.IntArray:
        xsn = xs.detach().numpy()
        return np.clip(np.searchsorted(xsn, xq, side="right") - 1, 0, len(xsn) - 2)

    def values(self, z: torch.Tensor, xq: T.Array) -> torch.Tensor:
        """The P1 interpolant at the points xq (Torch, differentiable in z)."""
        tau, xs = self.split(z)
        e = self._locate(xs, xq)
        xq_t = torch.as_tensor(xq, dtype=F64)
        t = (xq_t - xs[e]) / (xs[e + 1] - xs[e])
        return tau[e] * (1 - t) + tau[e + 1] * t

    def gradients(self, z: torch.Tensor, xq: T.Array) -> torch.Tensor:
        """dτ/dx at the points xq (the slope of the cell that holds each point)."""
        tau, xs = self.split(z)
        e = self._locate(xs, xq)
        return (tau[e + 1] - tau[e]) / (xs[e + 1] - xs[e])

    def admissible(self, z: T.Array, flat_beyond_top: bool = False) -> bool:
        """The cells have positive length, no slope is not finite, and no slope is 0; with
        flat_beyond_top, a slope of 0 is allowed on a cell with τ > 0 at both ends (beyond the
        top: no tissue in the cell)."""
        tau, xs = self.nodal(z)
        dt = np.diff(tau)
        if not (np.all(np.diff(xs) > 0) and np.all(np.isfinite(dt))):
            return False
        flat = dt == 0
        if flat_beyond_top:
            flat &= np.minimum(tau[:-1], tau[1:]) <= 0
        return not bool(np.any(flat))

    def extrema(self, z: T.Array) -> list[Extremum]:
        """The nodes where the slope changes sign. A run of flat cells (slope 0) between a
        rise and a fall is one maximum (between a fall and a rise one minimum); its position
        is the run's middle, its value τ on the run, and its linear form the mean of the run's
        nodes."""
        tau, xs = self.nodal(z)
        dt = np.diff(tau)
        out = []
        if not np.any(dt == 0):
            for i in np.nonzero(dt[1:] * dt[:-1] < 0)[0] + 1:
                c = np.zeros(self.n_unknowns)
                c[self.unknown_index(int(i))] = 1.0
                out.append(Extremum(float(xs[i]), float(tau[i]), "max" if dt[i - 1] > 0 else "min", c))
            return out
        nz = np.nonzero(dt)[0]  # the cells with a slope
        for a, b in zip(nz[:-1], nz[1:], strict=True):
            if dt[a] * dt[b] < 0:
                run = np.arange(a + 1, b + 1)  # the nodes from the end of cell a to the start of cell b
                c = np.zeros(self.n_unknowns)
                for i in run:
                    c[self.unknown_index(int(i))] = 1.0 / len(run)
                out.append(Extremum(float(0.5 * (xs[run[0]] + xs[run[-1]])), float(tau[run[0]]),
                                    "max" if dt[a] > 0 else "min", c))
        return out

    def derived_norms(
        self, z: torch.Tensor, w_of_grad: Callable[[torch.Tensor], torch.Tensor], skip_reversal: bool,
        tissue_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """w is constant on each cell (∇τ is). ∫ w² = Σ_e L_e w_e². ∫ |∇w|² is the two-point
        form on the dual cells between the cell midpoints: Σ_f (w_{e+1} − w_e)² / d_f,
        d_f = (L_e + L_{e+1}) / 2; with skip_reversal, the dual cells where the slope changes
        sign are left out. With tissue_only, each dual cell's term is weighted by the fraction of
        the dual cell where τ ≤ 0 (continuous in τ); w is then not evaluated on a flat cell
        (slope 0, allowed only beyond the top), and a dual cell with tissue next to a flat cell
        gives +∞. The mass part is not weighted."""
        tau, xs = self.split(z)
        L = xs[1:] - xs[:-1]
        g = (tau[1:] - tau[:-1]) / L
        if not tissue_only:
            w = w_of_grad(g)
            mass = (L * w * w).sum()
            dw = w[1:] - w[:-1]
            d = 0.5 * (L[1:] + L[:-1])
            keep = torch.ones_like(dw)
            if skip_reversal:
                gs = g.detach()
                keep = torch.where(gs[1:] * gs[:-1] < 0, torch.zeros_like(dw), keep)
            return mass, (keep * dw * dw / d).sum()
        flat = g.detach() == 0
        w = w_of_grad(torch.where(flat, torch.ones_like(g), g))
        mass = (L * w * w).sum()
        dw = w[1:] - w[:-1]
        d = 0.5 * (L[1:] + L[:-1])
        frac = _dual_tissue_fraction(tau, xs)
        keep = torch.ones_like(dw)
        if skip_reversal:
            gs = g.detach()
            keep = torch.where(gs[1:] * gs[:-1] < 0, torch.zeros_like(dw), keep)
        weight = keep * frac
        if bool(torch.any((flat[1:] | flat[:-1]) & (weight.detach() > 0))):
            return mass, torch.full((), math.inf, dtype=F64)
        return mass, (weight * dw * dw / d).sum()

    def step_limit(self, z: T.Array, step: T.Array, cut: float) -> float:
        """Enriched: the free node moves at most the fraction `cut` of the way to its cell's
        end. Plain: inf."""
        if self.enrich_cell is None:
            return float("inf")
        e = self.enrich_cell
        xk, dx = float(z[-1]), float(step[-1])
        if dx == 0:
            return float("inf")
        room = (self.x[e + 1] - xk) if dx > 0 else (xk - self.x[e])
        return cut * room / abs(dx)

    def free_node_at_end(self, z: T.Array, tol: float) -> bool:
        """Enriched: the free node is within tol · (cell length) of an end of its cell."""
        if self.enrich_cell is None:
            return False
        e = self.enrich_cell
        L = self.x[e + 1] - self.x[e]
        return bool(min(z[-1] - self.x[e], self.x[e + 1] - z[-1]) < tol * L)

    def cells_at(self, x: float) -> T.IntArray:
        if self.enrich_cell is not None:
            raise ValueError("cells_at: plain P1 only")
        i = np.nonzero(np.abs(self.x - x) < 1e-12)[0]
        if len(i):
            k = int(i[0])
            return np.array([c for c in (k - 1, k) if 0 <= c < len(self.x) - 1], dtype=np.int64)
        return np.array([int(np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 2))], dtype=np.int64)

    # ---- the quadrature

    def integrate_cut(self, z: torch.Tensor, density: Density, per_cell: bool = False) -> torch.Tensor:
        """Each cell split at its crossing c of τ = 0 (c = the cell's end when τ does not
        change sign there); the Gauss rule on each part. The first part has the side of τ at
        the cell's start, the second the other side."""
        tau, xs = self.split(z)
        a_, b_ = xs[:-1], xs[1:]
        ta, tb = tau[:-1], tau[1:]
        cross = (ta.detach() < 0) != (tb.detach() < 0)
        frac = torch.where(cross, -ta / torch.where(cross, tb - ta, torch.ones_like(ta)), torch.ones_like(ta))
        c_ = a_ + (b_ - a_) * frac
        s1 = torch.where(ta.detach() < 0, -1.0, 1.0).to(F64)
        qx = torch.as_tensor(self.qx, dtype=F64)
        qw = torch.as_tensor(self.qw, dtype=F64)

        def part(a: torch.Tensor, b: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
            xq = a[:, None] + (b - a)[:, None] * qx[None, :]
            return (b - a) * (density(xq, side[:, None]) * qw[None, :]).sum(1)

        v = part(a_, c_, s1) + part(c_, b_, -s1)
        return v if per_cell else v.sum()

    def integrate_grad_cut(self, z: torch.Tensor, f: GradDensity) -> torch.Tensor:
        """As integrate_cut, with ∇τ (constant on each cell) passed to f."""
        tau, xs = self.split(z)
        a_, b_ = xs[:-1], xs[1:]
        ta, tb = tau[:-1], tau[1:]
        g = (tb - ta) / (b_ - a_)
        cross = (ta.detach() < 0) != (tb.detach() < 0)
        frac = torch.where(cross, -ta / torch.where(cross, tb - ta, torch.ones_like(ta)), torch.ones_like(ta))
        c_ = a_ + (b_ - a_) * frac
        s1 = torch.where(ta.detach() < 0, -1.0, 1.0).to(F64)
        qx = torch.as_tensor(self.qx, dtype=F64)
        qw = torch.as_tensor(self.qw, dtype=F64)

        def part(a: torch.Tensor, b: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
            xq = a[:, None] + (b - a)[:, None] * qx[None, :]
            vals = f(g[:, None].expand_as(xq), side[:, None].expand_as(xq), xq)
            return (b - a) * (vals * qw[None, :]).sum(1)

        return (part(a_, c_, s1) + part(c_, b_, -s1)).sum()

    def cell_rise_forms(self) -> list[tuple[float, T.Array]]:
        if self.enrich_cell is not None:
            raise ValueError("cell_rise_forms: plain P1 only")
        out = []
        for e in range(len(self.x) - 1):
            c = np.zeros(self.n_unknowns)
            c[e + 1], c[e] = 1.0, -1.0
            out.append((float(0.5 * (self.x[e] + self.x[e + 1])), c))
        return out

    def crossings(self, z: T.Array) -> list[tuple[float, int]]:
        """The crossings of τ = 0: (position, sign of the slope there: +1 when τ rises through
        0, that is a top reached from the left; −1 a top reached from the right)."""
        tau, xs = self.nodal(z)
        out = []
        for e in range(len(xs) - 1):
            a, b = tau[e], tau[e + 1]
            if (a < 0) != (b < 0):
                c = xs[e] + (xs[e + 1] - xs[e]) * (-a) / (b - a)
                out.append((float(c), 1 if b > a else -1))
        return out

    # ---- the refinement

    def refine(self, cells: T.Array) -> P1Basis:
        """Bisect the given cells (with 2:1 grading). Plain P1 only."""
        if self.enrich_cell is not None:
            raise ValueError("refine an enriched basis: not defined")
        return P1Basis(graded(self.x, cells))

    def prolong(self, z: T.Array, fine: TauBasis) -> T.Array:
        """z on a refined basis (exact: the P1 meshes are nested)."""
        if not isinstance(fine, P1Basis) or self.enrich_cell is not None or fine.enrich_cell is not None:
            raise ValueError("prolong: plain P1 to plain P1 only")
        return np.interp(fine.x, self.x, np.asarray(z, float))
