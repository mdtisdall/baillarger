"""The column-volume sampler: the echoes along the columns of a superblock
(spec: docs/formats/column-volume.md, column-volume/1).

A sampler computes its output on demand from the rays of the ray sampler
(ray-sampler/1) and the echoes; nothing is stored. Each sample of a column
keeps the value of every echo (trilinear, as acquired), a validity flag and
the partial-volume mark. A column is the same, bit for bit, whichever
superblock computes it.

This version gives the types and the call; the sampler itself comes with
Part B of phase 3.0.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from baillarger import backend
from baillarger.blocks import Source
from baillarger.rays import Rays

VERSION = "column-volume/1"

PENDING = "column-volume/1: the sampler comes in phase 3.0, Part B"

Bool = npt.NDArray[np.bool_]
F32 = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ColumnVolume:
    """The echoes along the columns of one call of the ray sampler, as flat
    arrays: column c has the rows rays.sample_start[c] to
    rays.sample_start[c + 1]."""

    #: the rays of the call (edge key, owned flag, s, position, intervals)
    rays: Rays
    #: (N, E) the value of each echo at each sample, NaN where not valid
    echo: F32
    #: (N,) true where the interpolation rule gives a value
    valid: Bool
    #: (N,) true for a partial-volume sample
    partial: Bool
    #: (n,) w of each column, mm
    reach: F32


#: the largest difference between two affines, as a fraction of the largest
#: entry: float32 rounding
AFFINE_RTOL = 1e-5


def reach_of(normal: F32, voxel_to_ras: npt.NDArray[np.float64]) -> F32:
    """w of each column, mm: the chord of a voxel along the unit normal."""
    m_inv = np.linalg.inv(voxel_to_ras[:3, :3])
    n = normal.astype(np.float64)
    # elementwise sums, not a matrix product: the result must not depend on
    # the number of columns
    u = np.stack(
        [
            (m_inv[a, 0] * n[:, 0] + m_inv[a, 1] * n[:, 1]) + m_inv[a, 2] * n[:, 2]
            for a in range(3)
        ],
        axis=1,
    )
    with np.errstate(divide="ignore"):
        chord = np.where(u != 0.0, 1.0 / np.abs(u), np.inf)
    return chord.min(axis=1).astype(np.float32)


class ColumnVolumeSampler:
    """The column-volume sampler on echoes (one source for each echo, on the
    grid of wm-dseg/1), the VoxelToRAS affine of that grid, the header
    affine of each echo, and the GridShape of white-surf/1."""

    def __init__(
        self,
        echoes: Sequence[Source],
        voxel_to_ras: npt.ArrayLike,
        echo_affines: Sequence[npt.ArrayLike],
        grid_shape: tuple[int, int, int],
    ) -> None:
        self.echoes = tuple(echoes)
        self.voxel_to_ras = np.asarray(voxel_to_ras, dtype=np.float64)
        self.echo_affines = tuple(np.asarray(a, dtype=np.float64) for a in echo_affines)
        self.grid_shape = grid_shape
        if len(self.echo_affines) != len(self.echoes):
            raise ValueError("one header affine is needed for each echo")
        tol = AFFINE_RTOL * float(np.abs(self.voxel_to_ras).max())
        for k, (src, aff) in enumerate(
            zip(self.echoes, self.echo_affines, strict=True)
        ):
            if tuple(src.shape) != tuple(grid_shape):
                raise ValueError(
                    f"echo {k}: shape {src.shape} is not the grid {grid_shape}"
                )
            if src.dtype.kind not in "iuf":
                raise ValueError(
                    f"echo {k}: dtype {src.dtype} is not an integer or float"
                )
            if (
                aff.shape != (4, 4)
                or float(np.abs(aff - self.voxel_to_ras).max()) > tol
            ):
                raise ValueError(f"echo {k}: its affine is not voxel_to_ras")
        self._inverse = np.linalg.inv(self.voxel_to_ras)

    def voxel_coordinates(self, position: F32) -> npt.NDArray[np.float64]:
        """A^-1 applied to positions (scanner mm), float64, as elementwise
        sums so that the result does not depend on the number of samples."""
        p = position.astype(np.float64)
        ai = self._inverse
        return np.stack(
            [
                ((ai[a, 0] * p[:, 0] + ai[a, 1] * p[:, 1]) + ai[a, 2] * p[:, 2])
                + ai[a, 3]
                for a in range(3)
            ],
            axis=1,
        )

    def sample(self, rays: Rays, device: str | None = None) -> ColumnVolume:
        """The echoes at the samples of rays, with the validity and the
        partial-volume mark of each sample. device is a name of
        baillarger.backend ("numpy" is the reference; the default is
        $BAILLARGER_DEVICE, else the best device here). Only the weighted
        sum runs on the device, in float32; the coordinates, the validity,
        the weights' inputs and the mark are float64 on the host, so they
        are the same on every device."""
        dev = backend.device(device)
        v = self.voxel_coordinates(rays.position)
        top = np.asarray(self.grid_shape, dtype=np.float64) - 1.0
        valid = np.all((v >= 0.0) & (v <= top), axis=1)
        echo = np.full((len(v), len(self.echoes)), np.nan, dtype=np.float32)
        if valid.any():
            echo[valid] = self._interpolate(v[valid], dev)
        reach = reach_of(rays.roots.normal, self.voxel_to_ras)
        partial = _partial(rays, reach)
        return ColumnVolume(
            rays=rays, echo=echo, valid=valid, partial=partial, reach=reach
        )

    def _interpolate(self, v: npt.NDArray[np.float64], dev: backend.Device) -> F32:
        """The trilinear values at valid voxel coordinates (spec,
        "Interpolation"). One box read for each echo."""
        n = np.asarray(self.grid_shape, dtype=np.int64)
        i0 = np.minimum(np.floor(v).astype(np.int64), n - 2)
        i0 = np.maximum(i0, 0)
        t = v - i0
        lo = i0.min(axis=0)
        hi = i0.max(axis=0) + 2
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
        rel = i0 - lo
        out = np.empty((len(v), len(self.echoes)), dtype=np.float32)
        for e, src in enumerate(self.echoes):
            if dev.torch is not None:
                out[:, e] = _weighted_sum_torch(
                    np.asarray(src.read(box)), rel, t, dev.torch
                )
                continue
            data = np.asarray(src.read(box)).astype(np.float64)
            acc = np.zeros(len(v), dtype=np.float64)
            for d in range(8):
                di, dj, dk = (d >> 2) & 1, (d >> 1) & 1, d & 1
                w = (
                    (t[:, 0] if di else 1.0 - t[:, 0])
                    * (t[:, 1] if dj else 1.0 - t[:, 1])
                ) * (t[:, 2] if dk else 1.0 - t[:, 2])
                acc = acc + w * data[rel[:, 0] + di, rel[:, 1] + dj, rel[:, 2] + dk]
            out[:, e] = acc.astype(np.float32)
        return out


def _partial(rays: Rays, reach: F32) -> Bool:
    """The partial-volume mark (spec, "The partial-volume mark")."""
    n = len(rays.sample_start) - 1
    counts = np.diff(rays.sample_start)
    column = np.repeat(np.arange(n), counts)
    s = rays.s.astype(np.float64)
    w = reach.astype(np.float64)[column]
    first = rays.interval_start[:-1]
    n_changes = np.diff(rays.interval_start) - 1
    partial = np.zeros(len(s), dtype=np.bool_)
    s0 = rays.interval_s0.astype(np.float64)
    for j in range(int(n_changes.max(initial=0))):
        has = n_changes > j
        b = np.full(n, np.inf)
        b[has] = s0[first[has] + 1 + j]
        partial |= np.abs(s - b[column]) <= w
    return partial


def _weighted_sum_torch(
    data: npt.NDArray[np.generic],
    rel: npt.NDArray[np.int64],
    t: npt.NDArray[np.float64],
    tdev: torch.device,
) -> F32:
    """The trilinear sum on a device, in float32: the 8 corners of each
    sample are gathered by index and added in the order of the spec (a gather,
    no atomic addition, so a run gives the same result every time)."""
    flat = torch.as_tensor(np.ascontiguousarray(data, np.float32), device=tdev).reshape(
        -1
    )
    nj, nk = data.shape[1], data.shape[2]
    base = (rel[:, 0] * nj + rel[:, 1]) * nk + rel[:, 2]
    tt = torch.as_tensor(t.astype(np.float32), device=tdev)
    acc = torch.zeros(len(t), dtype=torch.float32, device=tdev)
    for d in range(8):
        di, dj, dk = (d >> 2) & 1, (d >> 1) & 1, d & 1
        idx = torch.as_tensor(base + (di * nj + dj) * nk + dk, device=tdev)
        w = (
            (tt[:, 0] if di else 1.0 - tt[:, 0]) * (tt[:, 1] if dj else 1.0 - tt[:, 1])
        ) * (tt[:, 2] if dk else 1.0 - tt[:, 2])
        acc = acc + w * flat[idx]
    result: F32 = acc.cpu().numpy()
    return result
