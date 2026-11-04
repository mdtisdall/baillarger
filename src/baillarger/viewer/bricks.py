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
coarse levels) when every value fits; any other image is float32. A label image uses
the reducer `mode` in place of `mean`: level L + 1 holds the most common label of each
block (at the far edge of an odd axis, of the voxels there; a tie goes to the smallest
label). Labels stay integers: int16 when every label fits, else int32, never float.
Brick bytes are little-endian, x fastest, then y, then z (the order of a WebGL 3D
texture), with no header: the page knows each brick's shape from the level's shape.

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


def _fits(a: Arr, t: type[np.integer[Any]]) -> bool:
    info = np.iinfo(t)
    return a.size == 0 or (int(a.min()) >= info.min and int(a.max()) <= info.max)


def wire_dtype(a: Arr, labels: bool = False) -> np.dtype[Any]:
    """int16 for an integer image whose values fit, else float32. Labels: int16 when
    they fit, else int32; labels that are not integers or do not fit int32 raise."""
    integer = np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_
    if integer and _fits(a, np.int16):
        return np.dtype("<i2")
    if not labels:
        return np.dtype("<f4")
    if not integer or not _fits(a, np.int32):
        raise ValueError(f"labels must be integers that fit int32, not {a.dtype}")
    return np.dtype("<i4")


#: the dtype names of the wire dtypes (meta)
WIRE_NAMES = {
    np.dtype("<i2"): "int16",
    np.dtype("<i4"): "int32",
    np.dtype("<f4"): "float32",
}


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


def mode(a: Arr, dtype: np.dtype[Any]) -> Arr:
    """Level L + 1 from level L of a label image: the most common label of each
    2 x 2 x 2 block (of the voxels there, at the far edge of an odd axis); a tie goes
    to the smallest label. One pair of x-slices at a time.

    Method: the (up to) 8 labels of each block are sorted; a label's count is the run
    length of its value in the sorted list. The first run of the greatest length is
    the answer, and in sorted order that is the smallest of the tied labels. The
    missing voxels of an edge block are padded with a sentinel above every label
    (sorted last) and never counted."""
    nx, ny, nz = a.shape
    out = np.empty((-(-nx // 2), -(-ny // 2), -(-nz // 2)), dtype)
    my, mz = out.shape[1], out.shape[2]
    big = np.iinfo(np.int64).max
    for i, x in enumerate(range(0, nx, 2)):
        slab = np.full((2, 2 * my, 2 * mz), big, np.int64)
        part = np.asarray(a[x : x + 2], np.int64)
        slab[: len(part), :ny, :nz] = part
        # (my, mz, 8): the voxels of each block
        blk = slab.reshape(2, my, 2, mz, 2).transpose(1, 3, 0, 2, 4).reshape(my, mz, 8)
        s = np.sort(blk, axis=2)
        # run length ending at each position: start a new run where the value changes
        new = np.ones(s.shape, bool)
        new[..., 1:] = s[..., 1:] != s[..., :-1]
        idx = np.arange(8)
        start = np.maximum.accumulate(np.where(new, idx, 0), axis=2)
        run = idx - start + 1
        run[s == big] = 0  # padding is never the answer
        best = np.argmax(run, axis=2)  # the first greatest run: the smallest label
        out[i] = np.take_along_axis(s, best[..., None], axis=2)[..., 0]
    return out


#: the reducers of ArrayPyramid, by name
REDUCERS: dict[str, Callable[[Arr, np.dtype[Any]], Arr]] = dict(mean=halve, mode=mode)


class ArrayPyramid:
    """The levels of an image in memory (level 0: the image, in its wire dtype).
    reducer: "mean" (an image) or "mode" (a label image: the module docstring)."""

    def __init__(
        self,
        volume: Arr,
        affine: npt.ArrayLike,
        brick: int = 32,
        reducer: str = "mean",
    ) -> None:
        if brick < 1:
            raise ValueError("a brick needs at least one voxel a side")
        if reducer not in REDUCERS:
            raise ValueError(f"no reducer {reducer!r}: {sorted(REDUCERS)}")
        self.brick = int(brick)
        self.reducer = reducer
        self.dtype = wire_dtype(np.asarray(volume), labels=reducer == "mode")
        a0 = np.asarray(volume).astype(self.dtype, copy=False)
        shapes = level_shapes(a0.shape, self.brick)
        self.arrays: list[Arr] = [a0]
        reduce = REDUCERS[reducer]
        for _ in shapes[1:]:
            self.arrays.append(reduce(self.arrays[-1], self.dtype))
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
        reducer: str = "mean",
    ) -> "ArrayPyramid":
        """The image at path, or its box from voxel lo of the given shape (the crop's
        affine: crop_affine)."""
        img = nifti(path)
        if lo is None:
            data = np.asanyarray(img.dataobj)
            return cls(data, img.affine, brick, reducer)
        a = np.asarray(lo, np.int64)
        n = np.asarray(shape, np.int64)
        b = a + n
        data = np.asanyarray(img.dataobj[a[0] : b[0], a[1] : b[1], a[2] : b[2]])
        box = (int(a[0]), int(a[1]), int(a[2]))
        return cls(data, crop_affine(img.affine, box), brick, reducer)


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
        dtype=WIRE_NAMES[np.dtype(src.dtype)],
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
