"""The volume as bricks at levels of detail, for the viewer's own volume pane.

A volume is served as a pyramid of levels: level 0 is the image as stored, and level
L + 1 is level L averaged over blocks of 2 x 2 x 2 voxels (the blocks at the far
edge of an odd axis hold fewer voxels, and their mean is over the voxels there), so
its shape is ceil(shape_L / 2). The levels stop at the first one that fits in one
brick. Each level is cut into bricks of `brick` voxels a side, from voxel (0, 0, 0);
the bricks at the far edges are smaller (the rest of the axis). The page asks only
for the bricks that its slices cross, at the level that its zoom needs.

Voxel and world mm: level L's voxel c is the centre of its block of level 0, level-0
voxel 2^L c + (2^L - 1) / 2, so its affine is A_0 @ [[s, 0, 0, (s - 1)/2], ...] with
s = 2^L (level_affine). A_0 is the image's affine (the sform when its code is set,
nibabel's rule). The voxel grid of every level is regular: the edge voxel of a coarse
level has the same spacing, though its block may hold fewer voxels.

Values: integer images keep int16 (rounded to the nearest, ties to even, in the
coarse levels) when every value fits; any other image is float32. Brick bytes are
little-endian, x fastest, then y, then z (the order of a WebGL 3D texture), with no
header: the page knows each brick's shape from the level's shape.

BrickSource is what the server needs: `levels` (shape and affine of each) and
`read(level, b)`. ArrayPyramid holds the levels in memory (the hemisphere at 160 um:
about 0.7 GB as int16); a NIfTI-Zarr store with its own multi-resolution chunks can
take its place later with the same interface.
"""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from baillarger.data import crop_affine
from baillarger.viewer.data import nifti

type Arr = npt.NDArray[Any]
type Float = npt.NDArray[np.float64]
# the finest level with at most this many voxels gives the default window
WINDOW_VOXELS = 1 << 22
WINDOW_PERCENTILES = (0.5, 99.5)


@dataclass(frozen=True)
class Level:
    shape: tuple[int, int, int]
    affine: Float


class BrickSource(Protocol):
    """A volume served as bricks: the module docstring."""

    brick: int
    dtype: np.dtype[Any]

    @property
    def levels(self) -> list[Level]: ...

    def read(self, level: int, b: Sequence[int]) -> Arr: ...


def level_affine(affine: npt.ArrayLike, level: int) -> Float:
    """The affine of level `level` from the affine of level 0."""
    s = float(2**level)
    m = np.eye(4)
    m[:3, :3] *= s
    m[:3, 3] = (s - 1) / 2
    out: Float = np.asarray(affine, np.float64) @ m
    return out


def level_shapes(shape: Sequence[int], brick: int) -> list[tuple[int, int, int]]:
    """The shape of each level: halved (rounded up) until one brick holds it."""
    s = tuple(int(n) for n in shape)
    if len(s) != 3 or min(s) < 1:
        raise ValueError(f"a 3D volume is needed, not shape {s}")
    out = [(s[0], s[1], s[2])]
    while max(out[-1]) > brick:
        out.append((-(-out[-1][0] // 2), -(-out[-1][1] // 2), -(-out[-1][2] // 2)))
    return out


def wire_dtype(a: Arr) -> np.dtype[Any]:
    """int16 for an integer image whose values fit, else float32."""
    if np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_:
        info = np.iinfo(np.int16)
        if a.size == 0 or (int(a.min()) >= info.min and int(a.max()) <= info.max):
            return np.dtype("<i2")
    return np.dtype("<f4")


def halve(a: Arr, dtype: np.dtype[Any]) -> Arr:
    """Level L + 1 from level L: the mean of each 2 x 2 x 2 block (fewer voxels at the
    far edge of an odd axis), rounded half to even for int16. One pair of x-slices at a
    time, so the float temporaries stay at two slices."""
    nx, ny, nz = a.shape
    out = np.empty((-(-nx // 2), -(-ny // 2), -(-nz // 2)), dtype)
    iy, iz = np.arange(0, ny, 2), np.arange(0, nz, 2)
    cy = np.minimum(2, ny - iy).astype(np.float64)
    cz = np.minimum(2, nz - iz).astype(np.float64)
    for i, x in enumerate(range(0, nx, 2)):
        slab = np.asarray(a[x : x + 2], np.float64)
        s = np.add.reduceat(np.add.reduceat(slab.sum(0), iy, 0), iz, 1)
        m = s / (len(slab) * cy[:, None] * cz[None, :])
        out[i] = np.rint(m) if dtype.kind == "i" else m
    return out


class ArrayPyramid:
    """The levels of an image in memory (level 0: the image, in its wire dtype)."""

    def __init__(self, volume: Arr, affine: npt.ArrayLike, brick: int = 32) -> None:
        if brick < 1:
            raise ValueError("a brick needs at least one voxel a side")
        self.brick = int(brick)
        self.dtype = wire_dtype(volume)
        a0 = np.asarray(volume).astype(self.dtype, copy=False)
        shapes = level_shapes(a0.shape, self.brick)
        self.arrays: list[Arr] = [a0]
        for _ in shapes[1:]:
            self.arrays.append(halve(self.arrays[-1], self.dtype))
        A = np.asarray(affine, np.float64)
        self._levels = [Level(s, level_affine(A, k)) for k, s in enumerate(shapes)]

    @property
    def levels(self) -> list[Level]:
        return self._levels

    def read(self, level: int, b: Sequence[int]) -> Arr:
        if not 0 <= level < len(self.arrays):
            raise ValueError(f"no level {level}: levels 0 to {len(self.arrays) - 1}")
        a, B = self.arrays[level], self.brick
        lo = [int(k) * B for k in b]
        if len(lo) != 3 or any(
            k < 0 or k >= n for k, n in zip(lo, a.shape, strict=True)
        ):
            raise ValueError(f"no brick {list(b)} on level {level} of shape {a.shape}")
        out: Arr = a[lo[0] : lo[0] + B, lo[1] : lo[1] + B, lo[2] : lo[2] + B]
        return out

    @classmethod
    def from_nifti(
        cls,
        path: Path,
        lo: npt.ArrayLike | None = None,
        shape: Sequence[int] | None = None,
        brick: int = 32,
    ) -> "ArrayPyramid":
        """The image at path, or its box from voxel lo of the given shape (the crop's
        affine: crop_affine)."""
        img = nifti(path)
        if lo is None:
            data = np.asanyarray(img.dataobj)
            return cls(data, img.affine, brick)
        a = np.asarray(lo, np.int64)
        n = np.asarray(shape, np.int64)
        b = a + n
        data = np.asanyarray(img.dataobj[a[0] : b[0], a[1] : b[1], a[2] : b[2]])
        box = (int(a[0]), int(a[1]), int(a[2]))
        return cls(data, crop_affine(img.affine, box), brick)


def brick_shape(level: Level, b: Sequence[int], brick: int) -> tuple[int, ...]:
    """The shape of brick b of a level: `brick` a side, less at the far edges."""
    return tuple(
        min(brick, n - int(k) * brick) for k, n in zip(b, level.shape, strict=True)
    )


def encode(src: BrickSource, level: int, bricks: Sequence[Sequence[int]]) -> bytes:
    """The bricks one after another, each x fastest (the module docstring)."""
    parts = [
        np.asarray(src.read(level, b), src.dtype).tobytes(order="F") for b in bricks
    ]
    return b"".join(parts)


def window(src: BrickSource) -> list[float]:
    """The default display window: percentiles WINDOW_PERCENTILES of the finest level
    with at most WINDOW_VOXELS voxels, read whole."""
    L = next(
        k for k, lv in enumerate(src.levels) if int(np.prod(lv.shape)) <= WINDOW_VOXELS
    )
    lo, hi = np.percentile(_whole(src, L), WINDOW_PERCENTILES)
    return [float(lo), float(hi) if hi > lo else float(lo) + 1.0]


def _whole(src: BrickSource, level: int) -> Arr:
    lv, B = src.levels[level], src.brick
    out = np.empty(lv.shape, src.dtype)
    for i in range(0, lv.shape[0], B):
        for j in range(0, lv.shape[1], B):
            for k in range(0, lv.shape[2], B):
                out[i : i + B, j : j + B, k : k + B] = src.read(
                    level, (i // B, j // B, k // B)
                )
    return out


def meta(src: BrickSource, name: str) -> dict[str, Any]:
    """What the page needs: the levels (shape, affine), the brick size, the wire dtype
    and the default window."""
    return dict(
        name=name,
        brick=src.brick,
        dtype="int16" if src.dtype == np.dtype("<i2") else "float32",
        levels=[
            dict(shape=list(lv.shape), affine=lv.affine.tolist()) for lv in src.levels
        ],
        window=window(src),
    )


class LazyVolume:
    """A brick source built on first use (the pane may never be opened), once."""

    def __init__(self, make: Callable[[], BrickSource], name: str) -> None:
        self.make, self.name = make, name
        self._src: BrickSource | None = None
        self._meta: dict[str, Any] | None = None
        self.lock = threading.Lock()

    def source(self) -> BrickSource:
        with self.lock:
            if self._src is None:
                self._src = self.make()
                self._meta = meta(self._src, self.name)
            return self._src

    def meta(self) -> dict[str, Any]:
        self.source()
        assert self._meta is not None
        return self._meta

    def bricks(self, level: int, spec: str) -> bytes:
        """Bricks of a level from "i,j,k;i,j,k;...", in that order."""
        bs = [[int(x) for x in t.split(",")] for t in spec.split(";") if t]
        return encode(self.source(), level, bs)
