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


class ColumnVolumeSampler:
    """The column-volume sampler on echoes (one source for each echo, on the
    grid of wm-dseg/1), the VoxelToRAS affine of that grid, and the header
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

    def sample(self, rays: Rays, device: str | None = None) -> ColumnVolume:
        """The echoes at the samples of rays, with the validity and the
        partial-volume mark of each sample."""
        raise NotImplementedError(PENDING)
