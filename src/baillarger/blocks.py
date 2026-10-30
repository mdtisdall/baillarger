"""Reading a volume on the grid block by block, with a halo, onto a device.

Every stage, and the viewer, reads volumes (segmentations, FLASH echoes)
only through a Source (open_volume for a file; a box of voxels at a time)
and, for work by block, a BlockReader, so that their storage can change
under them: now NIfTI (memory-mapped .nii, or a .nii.gz read once); at
whole-brain scale a chunked store with chunks aligned to the blocks, read
chunk by chunk.

The grid's cells are divided into blocks of `cells` cells a side from
voxel (0, 0, 0), as in the mesh files (docs/formats/white-surf.md). Block
(i, j, k) starts at grid point origin = (i, j, k) * cells; read gives the
grid points from origin - before to origin + cells + after on each axis,
clipped to the grid, with the first point's coordinates. Marching
tetrahedra reads with before 0, after 1 (a block's cells need one more
point); a filter of radius r reads with r on both sides.

On a PyTorch device, float64 volumes are sent as float32 (MPS has no
float64; the device steps work in float32). The reader can hold the whole
volume there (resident), sent once, so that each block is a slice on the
device and no copy goes through the host per block; it does so when the
volume is at most RESIDENT_BYTES.
"""

import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import nibabel as nib
import numpy as np
import numpy.typing as npt
import torch

from baillarger.backend import Device

#: the largest volume (bytes) a reader holds on a device by default
RESIDENT_BYTES = 4 << 30


def morton(ijk: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """The Morton (Z-order) code of block indices (n, 3) (or one (3,)):
    their bits interleaved, i's bit before j's before k's at each level, so
    that blocks are in C order within each 2 x 2 x 2 group, and any aligned
    cube of 2^m blocks a side is a run of consecutive codes. Indices up to
    2^21 per axis."""
    a = np.asarray(ijk, np.int64)
    out = np.zeros(a.shape[:-1], np.int64)
    for b in range(21):
        for axis, shift in ((0, 2), (1, 1), (2, 0)):
            out |= ((a[..., axis] >> b) & 1) << (3 * b + shift)
    return out


def _for_device(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """a as a PyTorch device takes it: contiguous, float64 as float32 (MPS
    has no float64, and the device steps work in float32)."""
    if a.dtype == np.float64:
        return np.ascontiguousarray(a, np.float32)
    return np.ascontiguousarray(a)


class Source(Protocol):
    """Where a volume's voxels come from: its shape, dtype, and a box of it
    as a numpy array."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    def read(self, box: tuple[slice, ...]) -> npt.NDArray[Any]: ...


class ArraySource:
    """A volume in memory."""

    def __init__(self, a: npt.ArrayLike) -> None:
        self.a = np.asarray(a)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.a.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.a.dtype

    def read(self, box: tuple[slice, ...]) -> npt.NDArray[Any]:
        return self.a[box]


class NiftiSource:
    """A NIfTI file on disk. An uncompressed .nii is memory-mapped, so a box
    reads only its own voxels; a .nii.gz is read whole on the first read
    and kept (a box of it would otherwise decompress from the start every
    time). One object per file: keep it while the file is in use (the
    viewer, a stage)."""

    def __init__(self, path: Path) -> None:
        img = nib.load(path)
        if not isinstance(img, nib.nifti1.Nifti1Image):
            raise ValueError(f"{Path(path).name}: not a NIfTI-1 image")
        self.path = Path(path)
        self.img = img
        self.affine: npt.NDArray[np.float64] = np.asarray(img.affine, np.float64)
        self._whole: npt.NDArray[Any] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.img.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        dt: np.dtype[Any] = np.dtype(self.img.get_data_dtype())
        return dt

    def read(self, box: tuple[slice, ...]) -> npt.NDArray[Any]:
        if self.path.name.endswith(".gz"):
            if self._whole is None:
                self._whole = np.asarray(self.img.dataobj)
            return self._whole[box]
        return np.asarray(self.img.dataobj[box])


class EmbeddedSource:
    """An array placed at lo in a larger grid of shape, fill elsewhere: e.g. a field
    solved on a crop, read on the whole grid (so blocks and keys follow the grid)."""

    def __init__(
        self, a: npt.ArrayLike, lo: npt.ArrayLike, shape: tuple[int, ...], fill: float
    ) -> None:
        self.a = np.asarray(a)
        self.lo = np.asarray(lo, np.int64)
        self._shape = tuple(int(n) for n in shape)
        self.fill = fill

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.a.dtype

    def read(self, box: tuple[slice, ...]) -> npt.NDArray[Any]:
        out = np.full(
            tuple(int(s.stop) - int(s.start) for s in box), self.fill, self.a.dtype
        )
        src, dst = [], []
        for s, lo, n in zip(box, self.lo, self.a.shape, strict=True):
            a, b = max(int(s.start), int(lo)), min(int(s.stop), int(lo) + n)
            if a >= b:
                return out
            src.append(slice(a - int(lo), b - int(lo)))
            dst.append(slice(a - int(s.start), b - int(s.start)))
        out[tuple(dst)] = self.a[tuple(src)]
        return out


def open_volume(path: Path) -> NiftiSource:
    """The source of a volume file: the one entry point, for the pipeline
    and the viewer alike, so that a change of storage (chunked stores at
    whole-brain scale) changes only this."""
    return NiftiSource(path)


class BlockReader:
    """Blocks of a volume, with a halo, as numpy arrays or tensors on dev."""

    def __init__(
        self,
        source: Source,
        cells: int,
        before: int = 0,
        after: int = 0,
        dev: Device | None = None,
        resident: bool | None = None,
    ) -> None:
        self.source = source
        self.cells = cells
        self.before, self.after = before, after
        self.dev = dev
        self._all: torch.Tensor | None = None
        tdev = None if dev is None else dev.torch
        if tdev is not None:
            nbytes = int(np.prod(source.shape)) * source.dtype.itemsize
            if resident is None:
                resident = nbytes <= RESIDENT_BYTES
            if resident:
                whole = tuple(slice(0, n) for n in source.shape)
                self._all = torch.as_tensor(
                    _for_device(source.read(whole)), device=tdev
                )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source.shape

    @property
    def grid(self) -> tuple[int, ...]:
        """Blocks along each axis (of the cells, one fewer than the points)."""
        return tuple(-(-max(n - 1, 0) // self.cells) for n in self.shape)

    def indices(self) -> Iterator[tuple[int, ...]]:
        """Every block index, in Morton order (the order of the mesh files:
        neighbouring blocks near each other)."""
        ijk = np.array(list(itertools.product(*(range(g) for g in self.grid))))
        if not len(ijk):
            return iter(())
        o = np.argsort(morton(ijk), kind="stable")
        return (tuple(int(x) for x in ijk[i]) for i in o)

    def box(self, ijk: tuple[int, ...]) -> tuple[slice, ...]:
        """The grid points block ijk reads, with its halo, clipped."""
        return tuple(
            slice(
                max(i * self.cells - self.before, 0),
                min(i * self.cells + self.cells + self.after, n),
            )
            for i, n in zip(ijk, self.shape, strict=True)
        )

    def read(
        self, ijk: tuple[int, ...]
    ) -> tuple[npt.NDArray[Any] | torch.Tensor, npt.NDArray[np.int64]]:
        """Block ijk with its halo (a numpy array, or a tensor on the
        reader's PyTorch device), and its first grid point."""
        b = self.box(ijk)
        origin = np.array([s.start for s in b], np.int64)
        if self._all is not None:
            return self._all[b], origin
        a = self.source.read(b)
        tdev = None if self.dev is None else self.dev.torch
        if tdev is not None:
            return torch.as_tensor(_for_device(a), device=tdev), origin
        return np.ascontiguousarray(a), origin
