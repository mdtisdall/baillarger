"""Antialiasing a binary mask's surface: a smooth field with the mask's signs.

A binary mask fixes only which voxel centres are inside; its surface is a staircase. The
field phi here keeps every voxel's sign (phi >= eps inside the mask, phi <= -eps
outside) and is otherwise as smooth as it can be: it minimises the thin-plate energy,
the sum of its squared second derivatives,

    E(phi) = sum over voxels of phi_xx^2 + phi_yy^2 + phi_zz^2
                                + 2 (phi_xy^2 + phi_xz^2 + phi_yz^2)

(Lempitsky, "Surface extraction from binary volumes with higher-order smoothness", CVPR
2010; Whitaker, "Reducing aliasing artifacts in iso-surfaces of binary volumes", 2000).
Marching tetrahedra of phi at 0 has the binary mesh's triangles and topology (they
follow from the signs alone), with each vertex moved along its own grid edge between an
inside and an outside voxel centre, where the true surface also crosses: no self-
intersections (one piecewise-linear level set), and a smooth surface where phi is near
linear. The problem is convex with one solution, so devices agree to the solver's
tolerance.

Only a band |sd| <= band round the boundary is solved (sd the signed distance,
signed_distance); outside it phi stays sd. The solver is accelerated projected gradient
(FISTA) with adaptive restart (O'Donoghue and Candes 2015) in PyTorch, on any
baillarger.backend device (the numpy device runs it on the CPU). The six stencils are
shifted slices of a padded copy (on MPS about 120 times faster than conv3d), each
applied, then applied again as its own transpose (they are symmetric); the first pass's
output is zero outside the volume. It stops when the largest mean change per iteration
over a window of 25 iterations, at the voxels next to the boundary (they set the
vertices), is below tol: tol 3e-6 leaves every vertex within about 1e-3 voxel of the
fully converged solution (prototypes/scripts/antialias_profile.py; the plan).
"""

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as Fn
from scipy import ndimage as ndi

from baillarger.backend import Device
from baillarger.timing import log

type Bool = npt.NDArray[np.bool_]
type F32 = npt.NDArray[np.float32]

#: the band solved round the boundary (voxels of signed distance)
BAND = 5.0
#: every voxel keeps |phi| >= EPS on its own side
EPS = 0.02
#: stop when the mean change per iteration at the boundary is below this
TOL = 3e-6
#: the most iterations
MAX_ITER = 6000
#: the stopping test's window (iterations)
WINDOW = 25
_WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)
_MIXED = ((0, 1), (0, 2), (1, 2))


def signed_distance(mask: Bool, clip: float) -> F32:
    """The signed distance of a mask's voxel centres to the other side: +d - 1/2 inside,
    -(d - 1/2) outside, so the voxels next to the boundary are at +-1/2; clipped to
    +-clip."""
    m = np.asarray(mask, bool)
    din = ndi.distance_transform_edt(m)
    dout = ndi.distance_transform_edt(~m)
    out: F32 = np.clip(np.where(m, din - 0.5, 0.5 - dout), -clip, clip).astype(
        np.float32
    )
    return out


def _stencil(x: torch.Tensor, i: int) -> torch.Tensor:
    """Stencil i of the thin-plate energy (phi_xx, phi_yy, phi_zz, phi_xy, phi_xz,
    phi_yz) of x (..., D, H, W), zero outside."""
    D, H, W = x.shape[-3:]
    p = Fn.pad(x, (1, 1, 1, 1, 1, 1))

    def sl(d: tuple[int, int, int]) -> torch.Tensor:
        return p[
            ...,
            1 + d[0] : 1 + d[0] + D,
            1 + d[1] : 1 + d[1] + H,
            1 + d[2] : 1 + d[2] + W,
        ]

    if i < 3:
        e = [0, 0, 0]
        e[i] = 1
        return sl((e[0], e[1], e[2])) - 2.0 * x + sl((-e[0], -e[1], -e[2]))
    a, b = _MIXED[i - 3]

    def at(sa: int, sb: int) -> torch.Tensor:
        d = [0, 0, 0]
        d[a], d[b] = sa, sb
        return sl((d[0], d[1], d[2]))

    return 0.25 * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))


def energy_gradient(x: torch.Tensor) -> torch.Tensor:
    """The gradient of the thin-plate energy, 2 sum_s w_s D_s^T D_s x (D_s^T = D_s)."""
    g = torch.zeros_like(x)
    for i, w in enumerate(_WEIGHTS):
        g += (2.0 * w) * _stencil(_stencil(x, i), i)
    return g


def energy(x: torch.Tensor) -> float:
    """The thin-plate energy of x (zero outside)."""
    return float(
        sum(w * torch.sum(_stencil(x, i) ** 2) for i, w in enumerate(_WEIGHTS))
    )


def _lipschitz(dev: torch.device) -> float:
    """An upper bound on the gradient's Lipschitz constant (power iteration, +5 %)."""
    z = torch.as_tensor(
        np.random.default_rng(0).standard_normal((24, 24, 24)),
        dtype=torch.float32,
        device=dev,
    )
    lam = 1.0
    for _ in range(60):
        z = energy_gradient(z)
        lam = float(torch.linalg.vector_norm(z))
        z = z / lam
    return 1.05 * lam


def next_to_boundary(mask: Bool) -> Bool:
    """The voxels with a 6-neighbour on the other side of the mask's boundary."""
    P = np.pad(np.asarray(mask, bool), 1)
    c = (slice(1, -1),) * 3
    out = np.zeros(np.shape(mask), bool)
    for ax in range(3):
        for d in (-1, 1):
            sh = [slice(1, -1)] * 3
            sh[ax] = slice(1 + d, P.shape[ax] - 1 + d)
            out |= P[c] != P[tuple(sh)]
    return out


@dataclass(frozen=True)
class Solve:
    """How the solver went."""

    iterations: int
    restarts: int
    last_change: float
    converged: bool
    seconds: float

    def summary(self) -> dict[str, Any]:
        return dict(
            Iterations=self.iterations,
            Restarts=self.restarts,
            LastChange=float(self.last_change),
            Converged=self.converged,
        )


def thin_plate(
    mask: Bool,
    dev: Device,
    band: float = BAND,
    eps: float = EPS,
    tol: float = TOL,
    max_iter: int = MAX_ITER,
) -> tuple[F32, Solve]:
    """The antialiased field of mask (module docstring): phi with phi >= eps on the
    mask, phi <= -eps off it, minimising the thin-plate energy in the band |sd| <= band,
    phi = sd outside the band (sd = signed_distance(mask, band + 3)). Returns phi
    (float32) and how the solve went."""
    m = np.asarray(mask, bool)
    tdev = dev.torch if dev.torch is not None else torch.device("cpu")
    sd = signed_distance(m, band + 3.0)
    free = torch.as_tensor(np.abs(sd) <= band, device=tdev)
    lo = torch.as_tensor(m, device=tdev) & free
    hi = torch.as_tensor(~m, device=tdev) & free
    watch = torch.as_tensor(next_to_boundary(m), device=tdev) & free
    fixed = torch.as_tensor(sd, device=tdev)
    L = _lipschitz(tdev)

    def project(x: torch.Tensor) -> torch.Tensor:
        x = torch.where(free, x, fixed)
        x = torch.where(lo, torch.clamp(x, min=eps), x)
        return torch.where(hi, torch.clamp(x, max=-eps), x)

    t0 = time.perf_counter()
    x = project(fixed.clone())
    y, t = x.clone(), 1.0
    snap = x.clone()
    step, restarts, k = math.inf, 0, 0
    for k in range(max_iter):
        xn = project(y - energy_gradient(y) / L)
        # adaptive restart: the step went against the momentum
        if k > 0 and float(torch.sum((y - xn) * (xn - x))) > 0:
            t, restarts = 1.0, restarts + 1
        tn = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
        if k > 0 and k % WINDOW == 0:
            w = torch.abs(x - snap)[watch]
            step = float(torch.max(w)) / WINDOW if w.numel() else 0.0
            snap = x.clone()
            if k % 100 == 0:
                # the time per iteration is the same all through a solve: a
                # slowdown in this log is the machine's, not the solver's
                log.info(
                    f"  antialias iteration {k}: change {step:.2e}, "
                    f"{time.perf_counter() - t0:.1f} s"
                )
            if step < tol:
                break
    out: F32 = x.cpu().numpy().astype(np.float32)
    return out, Solve(k + 1, restarts, step, step < tol, time.perf_counter() - t0)
