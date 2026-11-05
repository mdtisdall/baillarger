"""Tests for baillarger.fix_diff: the diff's order and format, apply and
revert, and the fix table."""

from pathlib import Path

import numpy as np
import pytest

from baillarger import fix_diff as D
from baillarger.blocks import morton


def labels(shape: tuple[int, int, int] = (70, 66, 65)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.choice(np.array([0, 41, 42], np.int16), size=shape)


def test_the_diff_is_in_morton_order_and_16_bytes_a_voxel(tmp_path: Path) -> None:
    old = labels()
    new = old.copy()
    idx = np.array([0, 64, 64 * 66 * 65, 5, 65 * 66 * 65 - 1], np.int64)
    new.reshape(-1)[idx] = 41 + (new.reshape(-1)[idx] == 41)  # 41 -> 42, else 41
    rows = D.make_diff(old, new, idx, [1, 2, 3, 4, 5])
    assert D.DIFF_DTYPE.itemsize == 16
    ijk = np.stack(np.unravel_index(rows["index"], old.shape), 1)
    codes = morton(ijk // 64)
    key = list(zip(codes.tolist(), rows["index"].tolist(), strict=True))
    assert key == sorted(key) and len(rows) == 5
    p = tmp_path / "d.npy"
    D.write_diff(p, rows)
    back = D.read_diff(p, old.shape)
    assert back.tobytes() == rows.tobytes()
    assert p.stat().st_size <= 256 + 16 * len(rows)  # the .npy header
    with pytest.raises(ValueError, match="Morton"):
        D.check_diff(rows[::-1].copy(), old.shape)


def test_unchanged_voxels_are_left_out() -> None:
    old = labels((8, 8, 8))
    rows = D.make_diff(old, old, [1, 2], [1, 1])
    assert len(rows) == 0


def test_apply_reproduces_and_revert_changes_one_fix_only() -> None:
    old = labels((70, 66, 65))
    new = old.copy()
    flat = new.reshape(-1)
    a = np.arange(100, 200, dtype=np.int64)
    b = np.arange(300000, 300050, dtype=np.int64)
    flat[a] = np.where(flat[a] == 41, 42, 41)
    flat[b] = np.where(flat[b] == 0, 41, 0)
    rows = D.make_diff(old, new, np.r_[a, b], np.r_[[1] * len(a), [2] * len(b)])
    got = D.apply_diff(old, rows)
    assert got.tobytes() == new.tobytes()
    r = D.revert_fix(new, rows, 2)
    changed = np.flatnonzero(r.reshape(-1) != new.reshape(-1))
    assert changed.tolist() == b.tolist()
    assert (r.reshape(-1)[b] == old.reshape(-1)[b]).all()
    with pytest.raises(ValueError, match="old label"):
        D.apply_diff(new, rows)


def test_same_fix_and_overlap() -> None:
    old = np.zeros((4, 4, 4), np.int16)
    new = old.copy()
    new.reshape(-1)[[1, 2, 3]] = 41
    rows = D.make_diff(old, new, [1, 2, 3], [1, 1, 2])
    f1, f2 = D.fix_voxels(rows, 1), D.fix_voxels(rows, 2)
    assert f1[0].tolist() == [1, 2] and f1[1].tolist() == [41, 41]
    assert D.same_fix(f1, D.fix_voxels(rows, 1)) and not D.same_fix(f1, f2)
    assert not D.overlap(f1, f2)
    moved = (np.array([2, 5]), np.array([41, 41]))
    assert D.overlap(f1, moved) and not D.same_fix(f1, moved)


def test_the_table_round_trip(tmp_path: Path) -> None:
    shape = (10, 10, 10)
    aff = np.diag([0.5, 0.25, 1.0, 1.0])
    aff[:3, 3] = [1, 2, 3]
    idx = np.ravel_multi_index(([1, 3], [2, 2], [4, 4]), shape)
    fixes = [
        D.Fix(1, "manual_edit", np.zeros(0, np.int64)),
        D.Fix(2, "handle_cut", idx, wall=2, overrides=1),
    ]
    rows = D.table_rows(fixes, shape, aff)
    assert rows[1]["voxels"] == 2 and rows[1]["x"] == pytest.approx(2.0)
    assert (rows[1]["i"], rows[1]["j"], rows[1]["k"]) == (1, 2, 4)
    assert rows[1]["x_min"] == pytest.approx(1.5) and rows[1]["x_max"] == 2.5
    assert rows[1]["y"] == pytest.approx(2.5) and rows[1]["z"] == pytest.approx(7)
    p = tmp_path / "t.tsv"
    D.write_table(p, rows)
    back = D.read_table(p)
    assert back[0]["i"] is None and back[0]["wall"] is None
    assert back[1]["overrides"] == 1 and back[1]["kind"] == "handle_cut"
    assert back[1]["x"] == pytest.approx(2.0)
