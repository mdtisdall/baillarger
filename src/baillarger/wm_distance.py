"""The WM-distance sampler: the distance through tissue from the WM, capped,
for a superblock and its halo (spec: docs/formats/wm-distance.md,
wm-distance/1).

A sampler computes its output on demand from published files only
(wm-dseg/1); nothing is stored. The distance is the signed distance (mm)
from the WM/cortex interface, measured through the free voxels (the filled
WM and the cortex), negative in the WM. Outside and every other label are
obstacles. Beyond the cap the value is +inf (-inf in the WM); obstacles are
NaN.

The distance is second-order fast marching (scikit-fmm, BSD) in float64 on
the call's read box: the superblock with its halo, plus the cap, plus the
stencil's reach.
The read box adds MARGIN x cap, not the cap: fast marching's upwind cone
reaches up to about sqrt(3) x the distance along an axis. The output is
float32. Two superblocks give the same value within TOLERANCE (the
contract, "Determinism").
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import skfmm

from baillarger import labels as L
from baillarger.bids import read_sidecar, sidecar_path
from baillarger.blocks import (
    CACHE_BYTES,
    CHUNK,
    ChunkedSource,
    open_volume,
    superblock_box,
    voxel_size,
)
from baillarger.mesh import BLOCK
from baillarger.white_dseg import DSEG_SPEC

VERSION = "wm-distance/1"

#: the largest cap, mm: the column length cap
CAP = 8.0
#: the stencil's reach, voxels: second-order fast marching reads two
#: neighbours on each axis
REACH = 2
#: the read box's margin over the cap: fast marching's upwind cone reaches
#: up to sqrt(3) x the distance along an axis (a front oblique to all three
#: axes), so the read box holds the halo plus MARGIN x cap
MARGIN = math.sqrt(3.0)
#: mm: two superblocks give the same voxel within this (the cone's tail past
#: MARGIN x cap has very small weights; docs/formats/wm-distance.md,
#: "Determinism")
TOLERANCE = 1e-5

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
Box = tuple[slice, slice, slice]


@dataclass(frozen=True)
class WMDistance:
    """The distance of one call on its box of grid points."""

    #: the box's shape: the distance, mm; negative in the WM, +inf (-inf)
    #: beyond the cap in cortex (WM), NaN at obstacles
    distance: F32
    #: the box: grid points, the superblock with its halo
    box: Box
    #: the cap of the call, mm
    cap: float


def _has_interface(wm: Bool, ctx: Bool) -> bool:
    """Whether a WM voxel and a cortex voxel share a face."""
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis] = slice(None, -1)
        b[axis] = slice(1, None)
        ta, tb = tuple(a), tuple(b)
        if np.any(wm[ta] & ctx[tb]) or np.any(ctx[ta] & wm[tb]):
            return True
    return False


def wm_distance(wm: Bool, ctx: Bool, voxel_mm: npt.ArrayLike, cap: float = CAP) -> F64:
    """The capped signed distance (mm, float64) on a whole array: from the
    interface of the WM voxels (wm) and the cortex voxels (ctx), through
    wm | ctx, negative in the WM; +inf in cortex and -inf in the WM where no
    path of length <= cap exists; NaN at the other voxels (obstacles). The
    edge of the array is a boundary: no path leaves it."""
    _check_cap(cap)
    wm = np.asarray(wm, bool)
    ctx = np.asarray(ctx, bool)
    if wm.shape != ctx.shape or wm.ndim != 3:
        raise ValueError(f"wm {wm.shape} and ctx {ctx.shape}: not one 3-D shape")
    if np.any(wm & ctx):
        raise ValueError("wm and ctx overlap")
    v = np.asarray(voxel_mm, np.float64)
    free = wm | ctx
    far = np.where(wm, -np.inf, np.inf)
    if not _has_interface(wm, ctx):
        out: F64 = np.where(free, far, np.nan)
        return out
    phi = np.ma.MaskedArray(np.where(wm, -1.0, 1.0), mask=~free)
    d = skfmm.distance(phi, dx=v, order=2, narrow=cap)
    dm = np.ma.MaskedArray(d)
    vals = np.ma.filled(dm.astype(np.float64), np.nan)
    reached = ~np.ma.getmaskarray(dm) & free
    vals = np.where(reached & (np.abs(vals) <= cap), vals, far)
    out = np.where(free, vals, np.nan)
    return out


def _check_cap(cap: float) -> None:
    if not 0 < cap <= CAP:
        raise ValueError(f"cap {cap}: not in (0, {CAP}]")


class WMDistanceSampler:
    """The WM-distance sampler on one segmentation: dseg, the
    _desc-wmtopo_dseg of wm-dseg/1. The filled labels and the hemisphere
    come from its sidecar. Superblocks are 2^level blocks of `cells` voxels
    a side (wm-dseg/1 has no block size; the default is the mesh's). A call
    reads the labels of its read box through a chunk cache of cache_bytes,
    so that calls on superblocks in Morton order read each chunk about
    once."""

    def __init__(
        self, dseg: Path, cells: int = BLOCK, cache_bytes: int = CACHE_BYTES
    ) -> None:
        if cells < 1:
            raise ValueError(f"cells {cells}: not >= 1")
        self.dseg = Path(dseg)
        self.cells = int(cells)
        side = read_sidecar(sidecar_path(self.dseg), DSEG_SPEC)
        self.filled = np.asarray(side["FilledLabels"], np.int64)
        self.cortex = L.CORTEX_LABEL[side["Hemisphere"]]
        vol = open_volume(self.dseg)
        self.shape: tuple[int, int, int] = tuple(int(n) for n in vol.shape)  # type: ignore[assignment]
        self.affine = np.asarray(vol.affine, np.float64)
        self.voxel_mm = voxel_size(self.affine)
        self.labels = ChunkedSource(vol, CHUNK, cache_bytes)

    def boxes(
        self,
        superblock: tuple[int, int, int],
        level: int,
        halo_mm: float = 0.0,
        cap: float = CAP,
    ) -> tuple[Box, Box]:
        """The output box (the superblock's grid points with the halo,
        blocks.superblock_box) and the read box (the halo plus MARGIN x cap,
        rounded outward to whole voxels, plus REACH voxels), both clipped to
        the grid."""
        _check_call(superblock, level, halo_mm)
        _check_cap(cap)
        out = superblock_box(
            superblock, level, self.cells, halo_mm, self.affine, self.shape
        )
        wide = superblock_box(
            superblock,
            level,
            self.cells,
            halo_mm + MARGIN * cap,
            self.affine,
            self.shape,
        )
        read = tuple(
            slice(max(s.start - REACH, 0), min(s.stop + REACH, n))
            for s, n in zip(wide, self.shape, strict=True)
        )
        return out, read  # type: ignore[return-value]

    def sample64(
        self,
        superblock: tuple[int, int, int],
        level: int,
        halo_mm: float = 0.0,
        cap: float = CAP,
    ) -> tuple[F64, Box]:
        """As sample, in float64 before the cast: for the tests of
        determinism. Not part of the contract."""
        out, read = self.boxes(superblock, level, halo_mm, cap)
        lab = self.labels.read(read)
        wm = np.isin(lab, self.filled)
        ctx = lab == self.cortex
        d = wm_distance(wm, ctx, self.voxel_mm, cap)
        crop = tuple(
            slice(o.start - r.start, o.stop - r.start)
            for o, r in zip(out, read, strict=True)
        )
        return np.ascontiguousarray(d[crop]), out

    def sample(
        self,
        superblock: tuple[int, int, int],
        level: int,
        halo_mm: float = 0.0,
        cap: float = CAP,
    ) -> WMDistance:
        """The distance on the grid points of superblock (I, J, K) at level
        m with its halo (halo_mm), capped at cap (mm, at most CAP)."""
        d, out = self.sample64(superblock, level, halo_mm, cap)
        return WMDistance(distance=d.astype(np.float32), box=out, cap=float(cap))


def read_factor(block_mm: float, halo_mm: float) -> float:
    """((B + 2h) / B)^3: the voxels a call reads over the voxels it owns,
    for a superblock of side block_mm and a reach of halo_mm."""
    return float(((block_mm + 2 * halo_mm) / block_mm) ** 3)


def _check_call(superblock: tuple[int, int, int], level: int, halo_mm: float) -> None:
    if len(superblock) != 3 or min(superblock) < 0:
        raise ValueError(f"superblock {superblock}: not 3 indices >= 0")
    if level < 0:
        raise ValueError(f"level {level}: not >= 0")
    if not (halo_mm >= 0 and math.isfinite(halo_mm)):
        raise ValueError(f"halo_mm {halo_mm}: not a finite number >= 0")
