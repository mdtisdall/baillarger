"""Tests for baillarger.blocks (the block reader) and baillarger.timing."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from baillarger import backend, timing
from baillarger.blocks import ArraySource, BlockReader

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
@pytest.mark.parametrize("resident", [True, False])
def test_a_device_reads_the_same_blocks(name: str, resident: bool) -> None:
    a = volume()
    ref = BlockReader(ArraySource(a), 4, 0, 1)
    r = BlockReader(ArraySource(a), 4, 0, 1, backend.device(name), resident)
    assert (r._all is not None) == resident
    for ijk in r.indices():
        blk, o = r.read(ijk)
        want, o2 = ref.read(ijk)
        assert isinstance(blk, torch.Tensor) and blk.device.type == name
        assert np.array_equal(o, o2) and np.array_equal(blk.cpu().numpy(), want)


@pytest.mark.parametrize("name", DEVICES)
def test_float64_goes_to_a_device_as_float32(name: str) -> None:
    a = volume().astype(np.float64) / 7
    for resident in (True, False):
        r = BlockReader(ArraySource(a), 8, 0, 1, backend.device(name), resident)
        blk, o = r.read((1, 1, 0))
        assert isinstance(blk, torch.Tensor) and blk.dtype == torch.float32
        want = a[8:17, 8:17, 0:9].astype(np.float32)
        assert np.array_equal(blk.cpu().numpy(), want)


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

    from baillarger.blocks import open_volume

    a = volume()
    aff = np.diag([0.2, 0.16, 0.2, 1.0])
    box = (slice(3, 11), slice(0, 17), slice(4, 9))
    for name in ("v.nii", "v.nii.gz"):
        nib.save(nib.nifti1.Nifti1Image(a, aff), tmp_path / name)
        src = open_volume(tmp_path / name)
        assert src.shape == a.shape and src.dtype == a.dtype
        assert np.allclose(src.affine, aff)
        assert np.array_equal(src.read(box), a[box])
        assert np.array_equal(src.read(box), a[box])  # again, from the cache
        r = BlockReader(src, 8, 0, 1)
        blk, o = r.read((1, 0, 0))
        assert np.array_equal(blk, a[8:17, 0:9, 0:9]) and list(o) == [8, 0, 0]
