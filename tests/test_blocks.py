"""Tests for baillarger.blocks (the block reader) and baillarger.timing."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from baillarger import backend, timing
from baillarger.blocks import (
    CHUNK,
    ArraySource,
    BlockReader,
    ChunkedSource,
    EmbeddedSource,
    PatchedSource,
    decompress,
    main,
    morton,
    open_volume,
    sha256_file,
    superblock_box,
    superblock_range,
    voxel_size,
)

DEVICES = [d for d in backend.available() if d != "numpy"]


def volume() -> np.ndarray:
    return np.arange(23 * 17 * 9, dtype=np.int16).reshape(23, 17, 9)


def test_blocks_cover_the_grid_with_their_halo() -> None:
    a = volume()
    r = BlockReader(ArraySource(a), 4, before=2, after=1)
    assert r.grid == (6, 4, 2)  # cells: 22, 16, 8
    seen = np.zeros(a.shape, bool)
    for ijk in r.indices():
        blk, o = r.read(ijk)
        assert isinstance(blk, np.ndarray)
        lo = np.array(ijk) * 4
        # the halo, clipped to the grid
        assert np.array_equal(o, np.maximum(lo - 2, 0))
        hi = np.minimum(lo + 4 + 1, a.shape)
        assert blk.shape == tuple(hi - o)
        assert np.array_equal(
            blk, a[tuple(slice(p, q) for p, q in zip(o, hi, strict=True))]
        )
        seen[tuple(slice(p, q) for p, q in zip(lo, hi, strict=True))] = True
    assert seen.all()


@pytest.mark.parametrize("name", DEVICES)
def test_a_device_reads_the_same_blocks(name: str) -> None:
    a = volume()
    ref = BlockReader(ArraySource(a), 4, 0, 1)
    r = BlockReader(ArraySource(a), 4, 0, 1, backend.device(name))
    for ijk in r.indices():
        blk, o = r.read(ijk)
        want, o2 = ref.read(ijk)
        assert isinstance(blk, torch.Tensor) and blk.device.type == name
        assert np.array_equal(o, o2) and np.array_equal(blk.cpu().numpy(), want)


@pytest.mark.parametrize("name", DEVICES)
def test_float64_goes_to_a_device_as_float32(name: str) -> None:
    a = volume().astype(np.float64) / 7
    r = BlockReader(ArraySource(a), 8, 0, 1, backend.device(name))
    blk, o = r.read((1, 1, 0))
    assert isinstance(blk, torch.Tensor) and blk.dtype == torch.float32
    want = a[8:17, 8:17, 0:9].astype(np.float32)
    assert np.array_equal(blk.cpu().numpy(), want)


def test_block_reader_on_numpy_and_torch_cpu_match_the_array() -> None:
    a = volume()
    for dev in (None, backend.device("cpu")):
        r = BlockReader(ArraySource(a), 4, 2, 1, dev)
        for ijk in r.indices():
            blk, o = r.read(ijk)
            box = r.box(ijk)
            got = blk.numpy() if isinstance(blk, torch.Tensor) else blk
            assert np.array_equal(got, a[box])
            assert list(o) == [s.start for s in box]


def test_laps_log_each_step(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="baillarger")
    lap = timing.Laps(backend.device("cpu"))
    lap("first")
    with timing.step("second"):
        pass
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 2
    assert msgs[0].startswith("first") and msgs[1].startswith("second")
    assert all(" s   peak " in m and m.endswith("GB") for m in msgs)


def test_a_nifti_source_reads_boxes(tmp_path: Path) -> None:
    import nibabel as nib

    a = volume()
    aff = np.diag([0.2, 0.16, 0.2, 1.0])
    box = (slice(3, 11), slice(0, 17), slice(4, 9))
    for name in ("plain/v.nii", "zipped/v.nii.gz"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        nib.save(nib.nifti1.Nifti1Image(a, aff), tmp_path / name)
        src = open_volume(tmp_path / name)
        assert src.shape == a.shape and src.dtype == a.dtype
        assert np.allclose(src.affine, aff)
        assert np.array_equal(src.read(box), a[box])
        r = BlockReader(src, 8, 0, 1)
        blk, o = r.read((1, 0, 0))
        assert np.array_equal(blk, a[8:17, 0:9, 0:9]) and list(o) == [8, 0, 0]


def box_of(rng: np.random.Generator, shape: tuple[int, ...]) -> tuple[slice, ...]:
    out = []
    for n in shape:
        a = int(rng.integers(0, n))
        b = int(rng.integers(a, n + 1))
        out.append(slice(a, b))
    return tuple(out)


@pytest.mark.parametrize(
    "chunk, shape", [(4, (13, 9, 10)), (8, (20, 17, 9)), (64, (70, 30, 5))]
)
def test_a_chunked_read_equals_the_array(chunk: int, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(chunk)
    a = rng.standard_normal(shape)
    c = ChunkedSource(ArraySource(a), chunk)
    assert c.shape == a.shape and c.dtype == a.dtype
    for _ in range(40):
        box = box_of(rng, shape)
        assert np.array_equal(c.read(box), a[box])


def test_a_box_over_chunk_edges_and_the_grid_edge() -> None:
    a = np.arange(11 * 10 * 9).reshape(11, 10, 9)  # not multiples of 4
    c = ChunkedSource(ArraySource(a), 4)
    for box in (
        (slice(3, 11), slice(2, 10), slice(1, 9)),  # to the far edge
        (slice(0, 11), slice(0, 10), slice(0, 9)),  # all
        (slice(4, 5), slice(7, 9), slice(8, 9)),  # one chunk's corner
        (slice(2, 2), slice(0, 3), slice(0, 3)),  # empty
    ):
        assert np.array_equal(c.read(box), a[box])
    # a chunk at the edge is clipped: 3 x 2 x 1 voxels
    assert c.nbytes <= a.nbytes
    for bad in (
        (slice(-1, 3), slice(0, 3), slice(0, 3)),
        (slice(0, 12), slice(0, 3), slice(0, 3)),
        (slice(0, 3, 2), slice(0, 3), slice(0, 3)),
        (slice(0, 3), slice(0, 3)),
    ):
        with pytest.raises(ValueError):
            c.read(bad)


def test_the_cache_keeps_its_byte_limit_and_drops_the_oldest() -> None:
    a = np.arange(16 * 4 * 4, dtype=np.int64).reshape(16, 4, 4)
    chunk_bytes = 4 * 4 * 4 * 8
    c = ChunkedSource(ArraySource(a), 4, max_bytes=2 * chunk_bytes)

    def get(i: int) -> None:
        assert np.array_equal(
            c.read((slice(4 * i, 4 * i + 4), slice(0, 4), slice(0, 4))),
            a[4 * i : 4 * i + 4],
        )
        assert c.nbytes <= c.max_bytes

    get(0)
    get(1)
    assert c.reads == 2
    get(0)  # recent: 0 is now newer than 1
    assert c.reads == 2
    get(2)  # drops 1, the oldest
    assert c.reads == 3 and c.nbytes == 2 * chunk_bytes
    get(0)
    assert c.reads == 3  # survived
    get(1)
    assert c.reads == 4  # was dropped
    # a single read that needs more than the limit is made, then trimmed
    whole = tuple(slice(0, n) for n in a.shape)
    assert np.array_equal(c.read(whole), a)
    assert c.nbytes <= c.max_bytes
    # a repeated read of what fits costs no new chunk reads
    get(3)
    n = c.reads
    get(3)
    assert c.reads == n


def test_superblock_range_is_a_run_of_consecutive_codes() -> None:
    for m in (0, 1, 2):
        k = 1 << m
        for sb in [(0, 0, 0), (1, 0, 0), (0, 1, 1), (2, 3, 1), (3, 3, 3), (5, 2, 7)]:
            ijk = np.array(
                [
                    (sb[0] * k + i, sb[1] * k + j, sb[2] * k + q)
                    for i in range(k)
                    for j in range(k)
                    for q in range(k)
                ]
            )
            start, stop = superblock_range(sb, m)
            assert sorted(morton(ijk).tolist()) == list(range(start, stop))


def test_superblock_box_on_anisotropic_voxels() -> None:
    aff = np.diag([0.16, 0.2, 0.5, 1.0])
    assert np.allclose(voxel_size(aff), [0.16, 0.2, 0.5])
    shape = (400, 300, 120)
    # cells 8, m = 1: a superblock is 16 cells; halo 1 mm: 7, 5, 2 points
    box = superblock_box((1, 1, 1), 1, 8, 1.0, aff, shape)
    assert box == (slice(16 - 7, 33 + 7), slice(16 - 5, 33 + 5), slice(16 - 2, 33 + 2))
    # clipped at the low edges, and at the far ones
    box = superblock_box((0, 0, 0), 1, 8, 1.0, aff, shape)
    assert box == (slice(0, 17 + 7), slice(0, 17 + 5), slice(0, 17 + 2))
    box = superblock_box((24, 18, 7), 1, 8, 1.0, aff, shape)
    assert box == (slice(384 - 7, 400), slice(288 - 5, 300), slice(112 - 2, 120))
    # float error: 8 mm at 0.16 mm is 50 points, not 51
    box = superblock_box(
        (10, 10, 10), 0, 8, 8.0, np.diag([0.16, 0.16, 0.16, 1.0]), (999,) * 3
    )
    assert box[0] == slice(80 - 50, 89 + 50)


def write_gz(tmp_path: Path, a: np.ndarray) -> Path:
    import nibabel as nib

    gz = tmp_path / "x_dseg.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(a, np.diag([0.2, 0.2, 0.2, 1.0])), gz)
    return gz


def test_decompress_gives_nibabels_array_and_a_source_file(tmp_path: Path) -> None:
    import json

    import nibabel as nib

    a = volume()
    gz = write_gz(tmp_path, a)
    main(["decompress", str(gz)])
    nii = tmp_path / "x_dseg.nii"
    side = tmp_path / "x_dseg.nii.source.json"
    meta = json.loads(side.read_text())
    assert set(meta) == {"Source", "SourceSHA256", "SHA256"}
    assert meta["Source"] == str(gz.resolve())
    assert meta["SourceSHA256"] == sha256_file(gz)
    assert meta["SHA256"] == sha256_file(nii)
    assert not list(tmp_path.glob("*.part"))
    whole = tuple(slice(0, n) for n in a.shape)
    src = open_volume(gz)  # the copy beside it
    assert src.path == nii
    img = nib.load(gz)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    ref = np.asarray(img.dataobj)
    got = src.read(whole)
    assert got.dtype == ref.dtype and np.array_equal(got, ref)
    # --out
    out = tmp_path / "sub" / "y.nii"
    out.parent.mkdir()
    assert decompress(gz, out) == out and out.exists()
    assert (tmp_path / "sub" / "y.nii.source.json").exists()


def test_open_volume_refuses_a_copy_of_another_gz(tmp_path: Path) -> None:
    a = volume()
    gz = write_gz(tmp_path, a)
    decompress(gz, tmp_path / "x_dseg.nii")
    write_gz(tmp_path, a + 1)  # the gz changes after decompressing
    with pytest.raises(ValueError, match=r"x_dseg\.nii.*x_dseg\.nii\.gz"):
        open_volume(gz)
    (tmp_path / "x_dseg.nii.source.json").unlink()
    with pytest.raises(ValueError, match="x_dseg.nii"):
        open_volume(gz)
    with pytest.raises(ValueError, match="open_volume"):
        from baillarger.blocks import NiftiSource

        NiftiSource(gz)


def test_open_volume_decompresses_a_gz_with_no_copy(tmp_path: Path) -> None:
    import gc

    a = volume()
    gz = write_gz(tmp_path, a)
    src = open_volume(gz)
    assert src.path.parent != tmp_path and src.path.exists()
    whole = tuple(slice(0, n) for n in a.shape)
    assert np.array_equal(src.read(whole), a)
    tmpdir = src.path.parent
    del src
    gc.collect()
    assert not tmpdir.exists()
    assert not (tmp_path / "x_dseg.nii").exists()


def test_an_embedded_source_over_a_source_equals_one_over_its_array() -> None:
    a = volume().astype(np.float32)
    grid = (40, 30, 20)
    arr = EmbeddedSource(a, (5, 3, 2), grid, -8.0)
    src = EmbeddedSource(ArraySource(a), (5, 3, 2), grid, -8.0)
    for box in (
        (slice(0, 40), slice(0, 30), slice(0, 20)),
        (slice(0, 7), slice(2, 4), slice(10, 15)),
        (slice(30, 40), slice(0, 3), slice(0, 2)),  # outside the array
    ):
        assert np.array_equal(arr.read(box), src.read(box))
    assert src.dtype == np.float32 and src.shape == grid


def test_a_patched_source_reads_the_patch_on_its_box_only() -> None:
    a = volume()
    box = (slice(4, 10), slice(0, 5), slice(3, 9))
    patch = -np.arange(6 * 5 * 6, dtype=np.int16).reshape(6, 5, 6)
    src = PatchedSource(ArraySource(a), box, patch)
    want = a.copy()
    want[box] = patch
    for b in (
        (slice(0, 23), slice(0, 17), slice(0, 9)),
        (slice(8, 20), slice(3, 17), slice(0, 4)),
        (slice(12, 20), slice(0, 17), slice(0, 9)),  # misses the box
    ):
        assert np.array_equal(src.read(b), want[b])
    assert np.array_equal(a, volume())  # the base is not changed
    with pytest.raises(ValueError):
        PatchedSource(ArraySource(a), (slice(20, 26), slice(0, 5), slice(3, 9)), patch)


class CountingSource(ArraySource):
    """An ArraySource that counts its reads."""

    def __init__(self, a: np.ndarray) -> None:
        super().__init__(a)
        self.reads = 0

    def read(self, box: tuple[slice, ...]) -> np.ndarray:
        self.reads += 1
        return super().read(box)


Paint = tuple[tuple[slice, ...], np.ndarray]


def paints(n: int, shape: tuple[int, ...], seed: int = 0) -> list[Paint]:
    """n random boxes in a grid of shape, each with a random patch."""
    rng = np.random.default_rng(seed)
    out: list[Paint] = []
    for _ in range(n):
        lo = [int(rng.integers(0, s - 1)) for s in shape]
        hi = [
            int(rng.integers(a + 1, min(s, a + 12) + 1))
            for a, s in zip(lo, shape, strict=True)
        ]
        box = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
        out.append((box, rng.normal(size=[b - a for a, b in zip(lo, hi, strict=True)])))
    return out


def blocks_of(
    boxes: list[tuple[slice, ...]], shape: tuple[int, ...], block: int
) -> set[tuple[int, ...]]:
    """The blocks of the grid (block voxels a side) that meet the union of boxes."""
    union = np.zeros(shape, bool)
    for box in boxes:
        union[box] = True
    return {tuple(int(x) for x in v) for v in np.argwhere(union) // block}


def test_repeated_paints_merge_into_one_patch_and_equal_the_stacked_reads() -> None:
    """Paint after paint gives one PatchedSource over the base (no chain): its values
    are those of the patches written in order (the stacked version), and a read
    reads the base one time, after one paint as after many."""
    shape = (40, 35, 30)
    a = np.random.default_rng(1).normal(size=shape).astype(np.float32)
    base = CountingSource(a)
    work = [paints(25, shape, seed) for seed in (2, 3)]
    for steps in work:
        stacked = a.copy()  # the reads of a chain of one wrapper per paint
        merged = PatchedSource(base, block=8)
        flat = PatchedSource(base, block=8)
        for box, patch in steps:
            stacked[box] = patch
            merged.write(box, patch)
            flat = PatchedSource(flat, box, patch)  # a new source, not a chain
            assert merged.base is base and flat.base is base
        everything = tuple(slice(0, n) for n in shape)
        for src in (merged, flat):
            assert np.array_equal(src.read(everything), stacked)
            for box, _ in steps:
                base.reads = 0
                assert np.array_equal(src.read(box), stacked[box])
                assert base.reads == 1
    assert np.array_equal(a, base.a)  # the base is not changed


def test_where_patches_overlap_the_later_patch_wins() -> None:
    a = np.zeros((20, 20, 20), np.float32)
    src = PatchedSource(ArraySource(a), block=8)
    first = (slice(2, 12), slice(2, 12), slice(2, 12))
    later = (slice(7, 17), slice(5, 9), slice(0, 20))
    src.write(first, np.full((10, 10, 10), 1.0))
    src.write(later, np.full((10, 4, 20), 2.0))
    want = a.copy()
    want[first] = 1.0
    want[later] = 2.0
    got = src.read((slice(0, 20),) * 3)
    assert np.array_equal(got, want)
    assert got[8, 6, 6] == 2.0 and got[3, 3, 3] == 1.0 and got[8, 10, 6] == 1.0


def test_the_patch_memory_is_the_blocks_that_meet_the_painted_boxes() -> None:
    """The merged patch keeps one array per block of the grid that a painted box
    meets (clipped to the grid), however many paints wrote it."""
    shape = (40, 35, 30)
    a = np.zeros(shape, np.float32)
    src = PatchedSource(ArraySource(a), block=8)
    steps = paints(30, shape, 4)
    for box, patch in steps:
        src.write(box, patch)
        src.write(box, patch)  # again: no more memory
    want = blocks_of([b for b, _ in steps], shape, 8)
    assert set(src.blocks) == want
    size = 0
    for ijk in want:
        size += int(
            np.prod([min(8, n - 8 * i) for i, n in zip(ijk, shape, strict=True)])
        )
    assert src.nbytes == size * 4


def test_a_source_made_from_a_patched_source_leaves_it_unchanged() -> None:
    """A preview: a new PatchedSource from the working field with one more patch;
    the working field reads as before."""
    a = volume().astype(np.float32)
    box0 = (slice(0, 10), slice(0, 10), slice(0, 9))
    work = PatchedSource(ArraySource(a), box0, np.ones((10, 10, 9)))
    whole = (slice(0, 23), slice(0, 17), slice(0, 9))
    before = work.read(whole)
    box = (slice(5, 15), slice(5, 15), slice(0, 9))
    view = PatchedSource(work, box, np.full((10, 10, 9), 7.0))
    want = before.copy()
    want[box] = 7.0
    assert np.array_equal(view.read(whole), want)
    assert np.array_equal(work.read(whole), before)


def test_a_patched_source_of_a_patched_source_keeps_its_block() -> None:
    """The block of a PatchedSource of a PatchedSource is the base's: another block
    is refused; the same block, or no block, works."""
    a = volume().astype(np.float32)
    box = (slice(0, 10), slice(0, 10), slice(0, 9))
    work = PatchedSource(ArraySource(a), box, np.ones((10, 10, 9)), block=8)
    with pytest.raises(ValueError, match="block"):
        PatchedSource(work, block=16)
    whole = (slice(0, 23), slice(0, 17), slice(0, 9))
    for view in (PatchedSource(work, block=8), PatchedSource(work)):
        assert view.block == 8
        assert np.array_equal(view.read(whole), work.read(whole))
    assert PatchedSource(ArraySource(a)).block == CHUNK
