"""The ray sampler: the columns of a superblock (spec:
docs/formats/ray-sampler.md, ray-sampler/1).

A sampler computes its output on demand from published files only
(white-surf/1, wm-dseg/1); nothing per column is stored. A column is the
straight ray of one white-surface vertex (its root) along the root's unit
normal, from s = -e_in to s = e_out (mm; s = 0 is the root), with the label
intervals that the ray crosses over the full caps, -CAP_IN to CAP_OUT. A
column is the same, bit for bit, whichever superblock computes it.

This version gives the types and the call; the sampler itself comes with
the normal method (phase 2.0, Part D).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

VERSION = "ray-sampler/1"

#: the caps, mm: the largest inward and outward extents
CAP_IN = 2.0
CAP_OUT = 8.0
#: the default step between samples, mm
STEP = 0.04
#: the methods of this version (phase 2.1 adds "equivolume")
METHODS = ("normal",)

PENDING = "ray-sampler/1: the sampler comes in phase 2.0, Part D"

#: the classes of a label interval
OUTSIDE = 0
WM = 1
CORTEX = 2
OTHER = 3
OFF_GRID = 4
#: the label of an interval off the grid
OFF_GRID_LABEL = -1

#: the derived classes of a column (diagnostics only)
OPEN = 1
IN_WM = 2
NEITHER = 3

Bool = npt.NDArray[np.bool_]
I32 = npt.NDArray[np.int32]
I64 = npt.NDArray[np.int64]
F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Roots:
    """The columns of a call, without their rays: the owned columns, then
    the halo columns, each in vertex order. Each edge key occurs once."""

    #: (n,) the root's grid-edge key, the key of the column
    edge_key: I64
    #: (n,) the root's row in the vertex array of white-surf/1
    vertex: I64
    #: (n,) true for an owned column, false for a halo column
    owned: Bool
    #: (n, 3) the root's position, scanner mm
    root: F32
    #: (n, 3) the root's unit normal, scanner space, out of the WM
    normal: F32
    #: (n,) |n̄|, the smoothed normal's length before its last division
    normal_length: F32


@dataclass(frozen=True)
class Rays:
    """The columns of a call with their samples and label intervals, as
    flat arrays: column c has the samples sample_start[c] to
    sample_start[c + 1] (k from -k_in[c] to k_out[c]) and the intervals
    interval_start[c] to interval_start[c + 1] (in order of s, tiling
    [-CAP_IN, CAP_OUT])."""

    roots: Roots
    method: str
    #: mm
    step: float
    #: (n,) K_in and K_out of each column
    k_in: I32
    k_out: I32
    #: (n + 1,) offsets into the samples
    sample_start: I64
    #: (N,) s of each sample, mm
    s: F32
    #: (N, 3) the position of each sample, scanner mm
    position: F32
    #: (n + 1,) offsets into the intervals
    interval_start: I64
    #: (Q,) the start and the end s of each interval, mm
    interval_s0: F32
    interval_s1: F32
    #: (Q,) the label of each interval, OFF_GRID_LABEL off the grid
    interval_label: I32
    #: (Q,) the class of each interval (OUTSIDE, WM, CORTEX, OTHER, OFF_GRID)
    interval_class: U8
    #: (n,) the derived class of each column (OPEN, IN_WM, NEITHER)
    derived_class: U8


class RaySampler:
    """The ray sampler on one white surface and its segmentation: surf, the
    _white.surf.gii of white-surf/1; dseg, the _desc-wmtopo_dseg of
    wm-dseg/1 that the surface was made from."""

    def __init__(self, surf: Path, dseg: Path) -> None:
        self.surf = Path(surf)
        self.dseg = Path(dseg)

    def roots(
        self, superblock: tuple[int, int, int], level: int, halo_mm: float = 0.0
    ) -> Roots:
        """The columns of superblock (I, J, K) at level m: the roots its
        blocks own, then the roots in its halo box (halo_mm), without rays."""
        raise NotImplementedError(PENDING)

    def sample(
        self,
        superblock: tuple[int, int, int],
        level: int,
        *,
        method: str = "normal",
        e_in: float | npt.ArrayLike = CAP_IN,
        e_out: float | npt.ArrayLike = CAP_OUT,
        step: float = STEP,
        halo_mm: float = 0.0,
    ) -> Rays:
        """The columns of roots(superblock, level, halo_mm), in that order,
        with their samples from -e_in to e_out at step (one value, or one
        per column) and their label intervals over the full caps."""
        raise NotImplementedError(PENDING)
