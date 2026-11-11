"""Reading a volume on the grid block by block, with a halo, onto a device.

Every stage, and the viewer, reads volumes (segmentations, FLASH echoes)
only through a Source (open_volume for a file; a box of voxels at a time)
and, for work by block, a BlockReader, so that their storage can change
under them. No large volume is read whole: a box of a file reads only its
own voxels (an uncompressed .nii is memory-mapped; a .nii.gz is decompressed
once, as a stream, to a .nii: decompress), and a ChunkedSource reads
through 64^3 chunks kept in a byte-limited LRU cache. At whole-brain scale
a chunked store (NIfTI-Zarr, 64^3 chunks sharded in Morton order) takes the
place of the memory map: it is a Source whose natural unit is a chunk, and
ChunkedSource over it reads each chunk once.

The grid's cells are divided into blocks of `cells` cells a side from
voxel (0, 0, 0), as in the mesh files (docs/formats/white-surf.md). Block
(i, j, k) starts at grid point origin = (i, j, k) * cells; read gives the
grid points from origin - before to origin + cells + after on each axis,
clipped to the grid, with the first point's coordinates. Marching
tetrahedra reads with before 0, after 1 (a block's cells need one more
point); a filter of radius r reads with r on both sides.

Stages with long reaches work on superblocks of 2^m blocks a side: a
superblock is a run of consecutive Morton codes (superblock_range), and its
box of grid points, with a halo stated in mm, is superblock_box. The halo
is a set of reads through the chunk cache, not a copy.

On a PyTorch device, float64 blocks are sent as float32 (MPS has no
float64; the device steps work in float32). Every block goes from the
chunk cache to the device; the volume is never held on the device.

The command line:

    python -m baillarger.blocks decompress GZ [--out NII]
"""

import argparse
import gzip
import hashlib
import itertools
import json
import math
import operator
import os
import shutil
import tempfile
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import nibabel as nib
import numpy as np
import numpy.typing as npt
import torch

from baillarger.backend import Device

#: the default byte limit of a ChunkedSource's cache
CACHE_BYTES = 1 << 30
#: the chunk size (voxels a side) of the chunked reads
CHUNK = 64
#: the piece size (bytes) of the streamed copies and hashes
PIECE = 1 << 24


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


def superblock_range(sb: tuple[int, int, int], m: int) -> tuple[int, int]:
    """The Morton codes [start, stop) of the blocks of superblock sb: the
    2^m x 2^m x 2^m blocks with indices sb * 2^m to sb * 2^m + 2^m - 1 on
    each axis. They are a run of consecutive codes: morton(sb) << 3m to
    (morton(sb) + 1) << 3m."""
    c = int(morton(sb))
    return c << (3 * m), (c + 1) << (3 * m)


def voxel_size(affine: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The voxel sizes (mm) along the grid axes: the column norms of the
    affine's 3 x 3 part."""
    a = np.asarray(affine, np.float64)
    out: npt.NDArray[np.float64] = np.linalg.norm(a[:3, :3], axis=0)
    return out


def superblock_box(
    sb: tuple[int, int, int],
    m: int,
    cells: int,
    halo_mm: float,
    affine: npt.ArrayLike,
    shape: tuple[int, ...],
) -> tuple[slice, ...]:
    """The grid points that the cells of superblock sb touch, from
    sb * 2^m * cells to (sb + 1) * 2^m * cells + 1 on each axis (the last
    cell's far point), extended by halo_mm on each side, rounded outward to
    whole grid points (ceil(halo_mm / voxel size); a tolerance keeps float
    error from adding a point), and clipped to the grid."""
    vs = voxel_size(affine)
    out = []
    for axis, (i, n) in enumerate(zip(sb, shape, strict=True)):
        h = max(math.ceil(halo_mm / float(vs[axis]) - 1e-9), 0)
        lo = i * (1 << m) * cells
        hi = (i + 1) * (1 << m) * cells + 1
        out.append(slice(max(lo - h, 0), min(hi + h, n)))
    return tuple(out)


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
    """An uncompressed NIfTI file on disk, memory-mapped: a box reads only
    its own voxels (the image must have no scaling, or the read still
    touches only that box but returns scaled values). A .nii.gz is refused:
    open_volume or decompress. One object per file: keep it while the file
    is in use (the viewer, a stage)."""

    def __init__(self, path: Path) -> None:
        if Path(path).name.endswith(".gz"):
            raise ValueError(
                f"{Path(path).name}: a compressed file cannot be read by box; "
                "use open_volume (which decompresses it) or decompress"
            )
        img = nib.load(path, mmap=True)
        if not isinstance(img, nib.nifti1.Nifti1Image):
            raise ValueError(f"{Path(path).name}: not a NIfTI-1 image")
        self.path = Path(path)
        self.img = img
        self.affine: npt.NDArray[np.float64] = np.asarray(img.affine, np.float64)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.img.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        dt: np.dtype[Any] = np.dtype(self.img.get_data_dtype())
        return dt

    def read(self, box: tuple[slice, ...]) -> npt.NDArray[Any]:
        return np.asarray(self.img.dataobj[box])


def _is_source(a: object) -> bool:
    """Whether a is a Source (it has shape, dtype and read), not an array."""
    return not isinstance(a, np.ndarray) and all(
        hasattr(a, k) for k in ("shape", "dtype", "read")
    )


class EmbeddedSource:
    """An array or a Source placed at lo in a larger grid of shape, fill elsewhere:
    e.g. a field solved on a crop, read on the whole grid (so blocks and keys follow
    the grid)."""

    def __init__(
        self,
        a: "npt.ArrayLike | Source",
        lo: npt.ArrayLike,
        shape: tuple[int, ...],
        fill: float,
    ) -> None:
        self.a: npt.NDArray[Any] | Source = (
            a if _is_source(a) else np.asarray(a)  # type: ignore[assignment]
        )
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
            tuple(int(s.stop) - int(s.start) for s in box), self.fill, self.dtype
        )
        src, dst = [], []
        for s, lo, n in zip(box, self.lo, self.a.shape, strict=True):
            a, b = max(int(s.start), int(lo)), min(int(s.stop), int(lo) + n)
            if a >= b:
                return out
            src.append(slice(a - int(lo), b - int(lo)))
            dst.append(slice(a - int(s.start), b - int(s.start)))
        arr = self.a
        out[tuple(dst)] = (
            arr[tuple(src)] if isinstance(arr, np.ndarray) else arr.read(tuple(src))
        )
        return out


class PatchedSource:
    """A Source with the values of patch on box (grid voxels, within the grid): e.g.
    an earlier field with a local solve written in, read with no copy of the
    field."""

    def __init__(
        self, base: Source, box: tuple[slice, ...], patch: npt.ArrayLike
    ) -> None:
        self.base = base
        self.box = tuple(slice(int(s.start), int(s.stop)) for s in box)
        self.patch = np.asarray(patch, base.dtype)
        want = tuple(s.stop - s.start for s in self.box)
        if self.patch.shape != want or not all(
            0 <= s.start <= s.stop <= n
            for s, n in zip(self.box, base.shape, strict=True)
        ):
            raise ValueError(f"patch {self.patch.shape} does not fit box {self.box}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.base.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.base.dtype

    def read(self, box: tuple[slice, ...]) -> npt.NDArray[Any]:
        out = np.array(self.base.read(box))
        src, dst = [], []
        for s, p in zip(box, self.box, strict=True):
            a, b = max(int(s.start), p.start), min(int(s.stop), p.stop)
            if a >= b:
                return out
            src.append(slice(a - p.start, b - p.start))
            dst.append(slice(a - int(s.start), b - int(s.start)))
        out[tuple(dst)] = self.patch[tuple(src)]
        return out


def _source_path(nii: Path) -> Path:
    """The file beside a decompressed nii that says where it came from."""
    return nii.with_name(nii.name + ".source.json")


class _Hashing:
    """A binary file read through a hash of the bytes read."""

    def __init__(self, f: BinaryIO) -> None:
        self.f = f
        self.hash = hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        b = self.f.read(n)
        self.hash.update(b)
        return b


def sha256_file(path: Path) -> str:
    """The SHA-256 (hex) of a file's bytes, read in pieces."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while piece := f.read(PIECE):
            h.update(piece)
    return h.hexdigest()


def decompress(gz: Path, nii: Path) -> Path:
    """Decompress a .nii.gz to nii as a stream, in pieces of PIECE bytes
    (never the whole volume in memory), and write `<nii name>.source.json`
    beside it: Source (the absolute path of gz), SourceSHA256 (of the gz's
    bytes) and SHA256 (of the nii written). Both files are written under a
    temporary name and renamed at the end. Returns nii."""
    gz, nii = Path(gz), Path(nii)
    part = nii.with_name(nii.name + ".part")
    side = _source_path(nii)
    side_part = side.with_name(side.name + ".part")
    out_hash = hashlib.sha256()
    try:
        with open(gz, "rb") as raw, open(part, "wb") as dst:
            src = _Hashing(raw)
            with gzip.GzipFile(fileobj=src) as z:  # type: ignore[call-overload]
                while piece := z.read(PIECE):
                    out_hash.update(piece)
                    dst.write(piece)
            while src.read(PIECE):  # whatever follows the gzip stream
                pass
        meta = {
            "Source": str(gz.resolve()),
            "SourceSHA256": src.hash.hexdigest(),
            "SHA256": out_hash.hexdigest(),
        }
        side_part.write_text(json.dumps(meta, indent=2) + "\n")
        os.replace(part, nii)
        os.replace(side_part, side)
    finally:
        part.unlink(missing_ok=True)
        side_part.unlink(missing_ok=True)
    return nii


def open_volume(path: Path, uncompressed: Path | None = None) -> NiftiSource:
    """The source of a volume file: the one entry point, for the pipeline
    and the viewer alike, so that a change of storage (chunked stores at
    whole-brain scale) changes only this.

    A .nii is memory-mapped. For a .nii.gz, the decompressed copy
    (`uncompressed`, else the path without .gz) is used if it exists and its
    .source.json (decompress) records the SHA-256 of this gz; a copy with
    another SHA, or none recorded, is a ValueError (the gz or the copy has
    changed). With no copy, the gz is decompressed into a temporary
    directory, removed when the source is collected or at exit."""
    path = Path(path)
    if not path.name.endswith(".gz"):
        return NiftiSource(path)
    nii = Path(uncompressed) if uncompressed is not None else path.with_suffix("")
    if nii.exists():
        side = _source_path(nii)
        want = json.loads(side.read_text())["SourceSHA256"] if side.exists() else None
        if want != sha256_file(path):
            why = (
                "no " + side.name if want is None else "another SHA-256 in " + side.name
            )
            raise ValueError(
                f"{nii} is not a decompressed copy of {path} ({why}); "
                "remove it or decompress again"
            )
        return NiftiSource(nii)
    tmp = tempfile.mkdtemp(prefix="baillarger-")
    try:
        src = NiftiSource(decompress(path, Path(tmp) / path.with_suffix("").name))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    weakref.finalize(src, shutil.rmtree, tmp, True)
    return src


class ChunkedSource:
    """Any Source read through chunks of `chunk` voxels a side, aligned at
    voxel (0, 0, ...), with a least-recently-used cache of at most
    `max_bytes`. A chunk is read from the wrapped source once (its box
    clipped to the grid) and kept until it is the oldest and the cache is
    over the limit; a read gathers the chunks its box meets and returns a
    new array. A read that alone needs more than max_bytes is made in full,
    and the cache is trimmed after it (so it may hold less than the read's
    chunks). Counters for tests: `reads` (chunk reads from the wrapped
    source) and `nbytes` (bytes kept).

    A chunked store whose natural unit is a chunk (NIfTI-Zarr, 64^3
    chunks) is a Source that this wraps with the store's chunk size: each
    chunk is then one store read."""

    def __init__(
        self, source: Source, chunk: int = CHUNK, max_bytes: int = CACHE_BYTES
    ) -> None:
        if chunk < 1:
            raise ValueError("a chunk needs at least one voxel a side")
        self.source = source
        self.chunk = int(chunk)
        self.max_bytes = int(max_bytes)
        self.reads = 0
        self.nbytes = 0
        self._cache: OrderedDict[tuple[int, ...], npt.NDArray[Any]] = OrderedDict()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.source.dtype

    def _chunk(self, idx: tuple[int, ...]) -> npt.NDArray[Any]:
        """The chunk with index idx, from the cache or the wrapped source."""
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        c = self.chunk
        box = tuple(
            slice(i * c, min((i + 1) * c, n))
            for i, n in zip(idx, self.shape, strict=True)
        )
        a = np.array(self.source.read(box))  # a copy: a memory map is not kept
        self.reads += 1
        self.nbytes += a.nbytes
        self._cache[idx] = a
        return a

    def read(self, box: tuple[slice, ...]) -> npt.NDArray[Any]:
        shape = self.shape
        if len(box) != len(shape):
            raise ValueError(f"box of {len(box)} axes for a volume of {len(shape)}")
        lim = []
        for s, n in zip(box, shape, strict=True):
            if s.step is not None:
                raise ValueError("a box has no step")
            a, b = operator.index(s.start), operator.index(s.stop)
            if not 0 <= a <= b <= n:
                raise ValueError(f"box {a}:{b} outside the grid of {n} points")
            lim.append((a, b))
        out = np.empty(tuple(b - a for a, b in lim), self.dtype)
        if out.size:
            c = self.chunk
            for idx in itertools.product(
                *(range(a // c, (b - 1) // c + 1) for a, b in lim)
            ):
                ch = self._chunk(idx)
                src, dst = [], []
                for i, (a, b), k in zip(idx, lim, ch.shape, strict=True):
                    lo = i * c
                    p, q = max(a, lo), min(b, lo + k)
                    src.append(slice(p - lo, q - lo))
                    dst.append(slice(p - a, q - a))
                out[tuple(dst)] = ch[tuple(src)]
        while self.nbytes > self.max_bytes and self._cache:
            _, old = self._cache.popitem(last=False)
            self.nbytes -= old.nbytes
        return out


class BlockReader:
    """Blocks of a volume, with a halo, as numpy arrays or tensors on dev.
    Every block is read through a ChunkedSource round source (a given
    ChunkedSource is used as it is)."""

    def __init__(
        self,
        source: Source,
        cells: int,
        before: int = 0,
        after: int = 0,
        dev: Device | None = None,
        resident: bool | None = None,  # deprecated, ignored (no volume is resident)
    ) -> None:
        self.source: Source = (
            source if isinstance(source, ChunkedSource) else ChunkedSource(source)
        )
        self.cells = cells
        self.before, self.after = before, after
        self.dev = dev

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
        a = self.source.read(b)
        tdev = None if self.dev is None else self.dev.torch
        if tdev is not None:
            return torch.as_tensor(_for_device(a), device=tdev), origin
        return np.ascontiguousarray(a), origin


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser(
        "decompress", help="a .nii.gz as a .nii, with its .source.json beside it"
    )
    d.add_argument("gz", type=Path, help="a .nii.gz")
    d.add_argument("--out", type=Path, help="the .nii (default: the gz without .gz)")
    a = ap.parse_args(argv)
    out = a.out if a.out is not None else a.gz.with_suffix("")
    print(decompress(a.gz, out))


if __name__ == "__main__":
    main()
