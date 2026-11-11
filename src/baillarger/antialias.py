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
fully converged solution (prototypes/scripts/antialias_profile.py). Its cost at
whole-brain scale, and the roadmap to a faster solver: docs/scaling.md.
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
from baillarger.blocks import Source
from baillarger.timing import log

type Bool = npt.NDArray[np.bool_]
type F32 = npt.NDArray[np.float32]
type Int = npt.NDArray[np.int64]

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
    start: F32 | None = None,
    held: Bool | None = None,
) -> tuple[F32, Solve]:
    """The antialiased field of mask (module docstring): phi with phi >= eps on the
    mask, phi <= -eps off it, minimising the thin-plate energy in the band |sd| <= band,
    phi = sd outside the band (sd = signed_distance(mask, band + 3)). Returns phi
    (float32) and how the solve went.

    start: the field the solve starts from (default sd). held: voxels that keep their
    start value and do not move (default none); the other voxels outside the band take
    sd. A local solve (local_solve) gives a crop of an earlier field as start and holds
    the voxels far from an edit. With neither, the solve is the full solve above."""
    m = np.asarray(mask, bool)
    tdev = dev.torch if dev.torch is not None else torch.device("cpu")
    sd = signed_distance(m, band + 3.0)
    free_np = np.abs(sd) <= band
    fixed_np = sd
    if held is not None:
        h = np.asarray(held, bool)
        if h.shape != m.shape:
            raise ValueError(f"held {h.shape} is not on the mask's grid {m.shape}")
        if start is None:
            raise ValueError("held voxels need a start field")
        free_np &= ~h
    if start is not None:
        s0 = np.asarray(start, np.float32)
        if s0.shape != m.shape:
            raise ValueError(f"start {s0.shape} is not on the mask's grid {m.shape}")
        if held is not None:
            fixed_np = np.where(h, s0, sd).astype(np.float32)
    free = torch.as_tensor(free_np, device=tdev)
    lo = torch.as_tensor(m, device=tdev) & free
    hi = torch.as_tensor(~m, device=tdev) & free
    watch = torch.as_tensor(next_to_boundary(m), device=tdev) & free
    fixed = torch.as_tensor(fixed_np, device=tdev)
    x0 = fixed if start is None else torch.as_tensor(s0, device=tdev)
    L = _lipschitz(tdev)

    def project(x: torch.Tensor) -> torch.Tensor:
        x = torch.where(free, x, fixed)
        x = torch.where(lo, torch.clamp(x, min=eps), x)
        return torch.where(hi, torch.clamp(x, max=-eps), x)

    t0 = time.perf_counter()
    x = project(x0.clone())
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


# ------------------------------------------------- the local solve of a preview

#: the preview's error near an edit (voxels; docs/preview-radius.md)
PREVIEW_TAU = 0.01
#: the smallest preview radius (voxels)
PREVIEW_MIN_RADIUS = 16
#: the decay length of the change of the field is c / DECAY (the slowest root of the
#: clamped strip, docs/preview-radius.md)
DECAY = 2.106
#: c is measured within this many voxels of the changed voxels
C_REACH = 4


def preview_radius(c: float, delta0: float, tau: float = PREVIEW_TAU) -> int:
    """The radius R (voxels) of a local solve (docs/preview-radius.md):
    R = S + (c / DECAY) ln(delta0 / tau), S = BAND + 3, rounded up, at least
    PREVIEW_MIN_RADIUS. c: the band's half-width at the edit; delta0: the size of the
    change of the field."""
    s = BAND + 3.0
    r = s + (c / DECAY) * math.log(max(delta0, tau) / tau)
    return max(PREVIEW_MIN_RADIUS, math.ceil(r))


def _box(
    lo: Int, hi: Int, margin: int, shape: tuple[int, ...]
) -> tuple[slice, slice, slice]:
    """[lo - margin, hi + margin) on each axis, clipped to the grid."""
    a = np.maximum(lo - margin, 0)
    b = np.minimum(hi + margin, np.asarray(shape))
    return (
        slice(int(a[0]), int(b[0])),
        slice(int(a[1]), int(b[1])),
        slice(int(a[2]), int(b[2])),
    )


def _near(changed: Int, box: tuple[slice, ...]) -> F32:
    """The distance (voxels) of each voxel of box to the nearest changed voxel."""
    seed = np.ones(tuple(s.stop - s.start for s in box), bool)
    o = np.array([s.start for s in box])
    seed[tuple((changed - o).T)] = False
    out: F32 = ndi.distance_transform_edt(seed).astype(np.float32)
    return out


def band_half_width(mask: Bool, near: F32, band: float = BAND) -> float:
    """c (docs/preview-radius.md): the largest distance from a free voxel (|sd| <=
    band) to a voxel that is not free, over the voxels within C_REACH of the changed
    voxels (near <= C_REACH). mask needs BAND + 3 + c voxels round those voxels."""
    free = np.abs(signed_distance(mask, band + 3.0)) <= band
    half = ndi.distance_transform_edt(free)
    z = half[near <= C_REACH]
    return float(z.max()) if z.size else 0.0


@dataclass(frozen=True)
class LocalSolve:
    """A local solve of the antialiased field round an edit: the new field phi on box
    (grid voxels), and how it was chosen and went. changed: the voxels (n, 3) whose
    side of the mask changed."""

    box: tuple[slice, slice, slice]
    phi: F32
    changed: Int
    radius: int
    c: float
    delta0: float
    solve: Solve


def local_solve(
    field: Source,
    mask: Source,
    voxels: npt.ArrayLike,
    dev: Device,
    tau: float = PREVIEW_TAU,
    tol: float = TOL,
) -> LocalSolve | None:
    """The antialiased field of an edited mask, solved only near the edit
    (docs/preview-radius.md). field: the earlier field on the whole grid (a Source:
    for a field on a crop, an EmbeddedSource with fill -(BAND + 3)); its sign gives
    the earlier mask. mask: the edited mask on the whole grid. voxels (n, 3): grid
    voxels that the edit may have changed; the changed voxels are those whose mask
    value differs from the field's sign. None if there are none.

    R follows the rule of preview_radius: c measured at the changed voxels
    (band_half_width), delta0 the larger of 1 and half the largest side of their box.
    The solve is thin_plate on the box of the changed voxels plus R + BAND + 5 (so sd
    is exact within R of them), started from the earlier field, with the voxels
    farther than R from the changed voxels held. The result is the new field on that
    box; field is not changed."""
    shape = tuple(int(n) for n in field.shape)
    if tuple(mask.shape) != shape:
        raise ValueError(f"mask {mask.shape} and field {shape} are on other grids")
    vx = np.asarray(voxels, np.int64).reshape(-1, 3)
    if np.any(vx < 0) or np.any(vx >= np.asarray(shape)):
        raise ValueError("a voxel is off the grid")
    if not len(vx):
        return None
    vx = np.unique(vx, axis=0)
    lo, hi = vx.min(0), vx.max(0) + 1
    small = _box(lo, hi, 0, shape)
    o = np.array([s.start for s in small])
    was = field.read(small)[tuple((vx - o).T)] > 0
    now = np.asarray(mask.read(small), bool)[tuple((vx - o).T)]
    changed = vx[was != now]
    if not len(changed):
        return None
    lo, hi = changed.min(0), changed.max(0) + 1
    s = int(BAND + 3)
    # c: the free voxels near the edit, and their distance to non-free voxels (up to
    # about 2 BAND + 1 in merged bands), need sd exact there
    cbox = _box(lo, hi, C_REACH + 2 * s + 4, shape)
    c = band_half_width(np.asarray(mask.read(cbox), bool), _near(changed, cbox))
    delta0 = max(1.0, float((hi - lo).max()) / 2.0)
    radius = preview_radius(c, delta0, tau)
    box = _box(lo, hi, radius + s + 2, shape)
    m = np.ascontiguousarray(mask.read(box), bool)
    start = np.asarray(field.read(box), np.float32)
    held = _near(changed, box) > radius
    phi, solve = thin_plate(m, dev, tol=tol, start=start, held=held)
    return LocalSolve(box, phi, changed, radius, c, delta0, solve)
