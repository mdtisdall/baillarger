"""Tests for the label edits (baillarger.label_edits): the files, the lock and
the atomic replacement, apply, paint, the eraser, undo, differ, and their use
by the dseg command (baillarger.white_dseg.wm_topo, write_wm_topo)."""

import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import threading
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from edits_helpers import ROD, ring_edits, write_edits
from edits_helpers import edits as build_edits
from test_white_dseg import FILLED, VOX, synthetic_aseg

from baillarger import bids, white_dseg
from baillarger import label_edits as LE
from baillarger.fileops import flock
from baillarger.handles import kuhn_betti

SHAPE = (12, 10, 8)
AFFINE = np.diag([0.16, 0.16, 0.16, 1.0])


def edits(
    pairs: dict[tuple[int, int, int], int], shape: tuple[int, ...] = SHAPE
) -> LE.Edits:
    """A diff from {voxel: label}, on SHAPE by default."""
    return build_edits(pairs, shape)


def same(a: LE.Edits, b: LE.Edits) -> bool:
    return bool(np.array_equal(a.index, b.index) and np.array_equal(a.label, b.label))


def write(path: Path, e: LE.Edits, shape: tuple[int, ...] = SHAPE) -> None:
    write_edits(path, e, shape, AFFINE)


# ------------------------------------------------------------------ the files


def test_write_and_read_give_the_same_diff(tmp_path: Path) -> None:
    p = LE.bids_path(tmp_path, dict(sub="T01", ses="S1"), "R")
    assert p.name == "sub-T01_ses-S1_hemi-R_desc-labeledits_edits.npy"
    e = edits({(1, 2, 3): 41, (0, 0, 0): 0, (11, 9, 7): 42})
    write(p, e)
    got, side, sha = LE.read_with_sidecar(p)
    assert same(got, e) and got.index.dtype == np.int64 and got.label.dtype == np.int32
    assert side["SpecVersion"] == "label-edits/1" and side["Count"] == 3
    assert side["Shape"] == list(SHAPE) and side["Hemisphere"] == "R"
    assert side["SHA256"] == sha == bids.file_sha256(p)
    assert np.allclose(np.reshape(side["Affine"], (4, 4)), AFFINE)
    rec = np.load(p, allow_pickle=False)
    assert rec.dtype == LE.EDITS_DTYPE and rec["index"].tolist() == sorted(
        rec["index"].tolist()
    )
    write(p, LE.empty())
    assert len(LE.read(p)) == 0
    # no file of the writes is left in the directory
    assert sorted(q.name for q in p.parent.iterdir()) == sorted(
        [p.name, bids.sidecar_path(p).name]
    )


def test_a_reader_checks_the_sha256(tmp_path: Path) -> None:
    p = tmp_path / "e_edits.npy"
    write(p, edits({(1, 1, 1): 41}))
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 1  # the label
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA-256"):
        LE.read(p)


def test_a_bad_diff_is_not_written(tmp_path: Path) -> None:
    p = tmp_path / "e_edits.npy"
    unsorted = LE.Edits(np.array([5, 2], np.int64), np.array([41, 41], np.int32))
    twice = LE.Edits(np.array([2, 2], np.int64), np.array([41, 42], np.int32))
    off = LE.Edits(np.array([int(np.prod(SHAPE))], np.int64), np.array([41], np.int32))
    bad = LE.Edits(np.array([2], np.int64), np.array([7], np.int32))
    for e, msg in (
        (unsorted, "sorted"),
        (twice, "twice"),
        (off, "outside"),
        (bad, "label-set"),
    ):
        with pytest.raises(ValueError, match=msg):
            write(p, e)
    with pytest.raises(ValueError, match="hemisphere"):
        LE.write(p, LE.empty(), hemisphere="X", shape=SHAPE, affine=AFFINE)
    assert not p.exists() and not list(tmp_path.iterdir())


def _writer(path: str, n: int) -> None:
    """Write n diffs in turn: the k-th has k + 1 voxels, each with label 41
    or 42 by k's parity."""
    for k in range(n):
        idx = np.arange(k + 1, dtype=np.int64)
        lab = np.full(k + 1, 41 if k % 2 else 42, np.int32)
        LE.write(
            Path(path),
            LE.Edits(idx, lab),
            hemisphere="R",
            shape=(64, 64, 64),
            affine=AFFINE,
        )


def test_a_reader_sees_the_old_or_the_new_diff_not_a_mix(tmp_path: Path) -> None:
    p = tmp_path / "e_edits.npy"
    _writer(str(p), 1)
    n = 300
    proc = mp.get_context("spawn").Process(target=_writer, args=(str(p), n))
    proc.start()
    seen: set[int] = set()
    while proc.is_alive() or not seen:
        e = LE.read(p)  # a mix would not have the labels of one k
        k = len(e) - 1
        want = 41 if k % 2 else 42
        assert e.index.tolist() == list(range(k + 1))
        assert set(e.label.tolist()) == {want}
        seen.add(k)
    proc.join()
    assert proc.exitcode == 0 and len(LE.read(p)) == n
    assert len(seen) > 1


def test_a_stop_between_the_two_renames_is_repaired_by_the_next_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "e_edits.npy"
    write(p, edits({(1, 1, 1): 41}))
    new = edits({(1, 1, 1): 0, (2, 3, 4): 42})
    real = os.replace
    calls: list[Any] = []

    def stop(a: Any, b: Any) -> None:
        calls.append(b)
        if len(calls) == 2:  # the sidecar's rename
            raise KeyboardInterrupt("stopped")
        real(a, b)

    monkeypatch.setattr(os, "replace", stop)  # fileops.write_atomic
    with pytest.raises(KeyboardInterrupt):
        write(p, new)
    monkeypatch.undo()
    # the new records, the old sidecar: no file of the write is left
    assert bids.file_sha256(p) != json.loads(bids.sidecar_path(p).read_text())["SHA256"]
    assert sorted(q.name for q in tmp_path.iterdir()) == sorted(
        [p.name, bids.sidecar_path(p).name]
    )
    with caplog.at_level(logging.WARNING, logger="baillarger.label_edits"):
        got, side, sha = LE.read_with_sidecar(p, shape=SHAPE, hemisphere="R")
    assert same(got, new) and "older than the records" in caplog.text
    assert side["SHA256"] == sha == bids.file_sha256(p) and side["Count"] == 2
    assert side["Shape"] == list(SHAPE) and side["Hemisphere"] == "R"
    # the new sidecar is on disk: the next read needs no repair
    caplog.clear()
    assert same(LE.read(p), new) and not caplog.text


def test_an_update_repairs_a_stopped_write_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "e_edits.npy"
    real = os.replace
    calls: list[Any] = []

    def stop(a: Any, b: Any) -> None:
        calls.append(b)
        if len(calls) == 2:
            raise KeyboardInterrupt("stopped")
        real(a, b)

    # the first write of the diff stops: records, and no sidecar
    monkeypatch.setattr(os, "replace", stop)  # fileops.write_atomic
    with pytest.raises(KeyboardInterrupt):
        write(p, edits({(1, 1, 1): 41}))
    monkeypatch.undo()
    assert p.exists() and not bids.sidecar_path(p).exists()
    seen: list[LE.Edits] = []

    def look(c: LE.Current) -> None:
        seen.append(c.edits)

    cur = LE.update(
        p,
        look,
        hemisphere="R",
        shape=SHAPE,
        affine=AFFINE,
    )
    assert same(seen[0], edits({(1, 1, 1): 41})) and cur.sha == bids.file_sha256(p)
    assert same(LE.read(p, shape=SHAPE, affine=AFFINE, hemisphere="R"), seen[0])


def test_the_reader_checks_the_grid_affine_and_hemisphere(tmp_path: Path) -> None:
    p = tmp_path / "e_edits.npy"
    write(p, edits({(1, 1, 1): 41}))
    LE.read(p, shape=SHAPE, affine=AFFINE, hemisphere="R")
    with pytest.raises(ValueError, match="grid"):
        LE.read(p, shape=(12, 10, 9))
    with pytest.raises(ValueError, match="grid"):
        LE.read(p, affine=np.eye(4))
    with pytest.raises(ValueError, match="hemisphere"):
        LE.read(p, hemisphere="L")
    with pytest.raises(ValueError, match="grid"):
        LE.update(p, lambda c: None, hemisphere="R", shape=(4, 4, 4), affine=AFFINE)


class _NoLock:
    """A stand-in for fileops.flock that takes no lock."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def _painter(
    path: str, k: int, n: int, start: Any, order: Any, count: Any, lock: bool
) -> None:
    """Process k paints the voxels 1000 k, ..., 1000 k + n - 1 (linear indices
    of a 64^3 grid) with 42, one update for each. It waits at the barrier
    start after its imports, so that the updates of the two processes start
    together. Inside each update (under the lock, if there is one) it puts k
    in the next entry of the shared array order (count is the next index).
    With lock False, update takes no lock (label_edits.flock is a no-op)."""
    if not lock:
        pytest.MonkeyPatch().setattr(LE, "flock", _NoLock)
    shape = (64, 64, 64)
    lab = np.zeros(shape, np.int16)
    start.wait()
    for i in range(1000 * k, 1000 * k + n):
        v = np.array([np.unravel_index(i, shape)], np.int64)

        def one(c: LE.Current, v: npt.NDArray[np.int64] = v) -> LE.Edits:
            with count.get_lock():
                order[count.value] = k
                count.value += 1
            return LE.paint(c.edits, lab, v, [42])[0]

        LE.update(Path(path), one, hemisphere="R", shape=shape, affine=AFFINE)


def _paint_in_two_processes(p: Path, lock: bool, n: int = 100) -> list[int]:
    """Run _painter in two "spawn" processes, each with n updates, and
    return the order of the updates (the process of each)."""
    ctx = mp.get_context("spawn")
    # the timeout stops a process that waits for one that failed to start
    start = ctx.Barrier(2, timeout=120)
    order = ctx.Array("b", 2 * n)
    count = ctx.Value("i", 0)
    procs = [
        ctx.Process(target=_painter, args=(str(p), k, n, start, order, count, lock))
        for k in (0, 1)
    ]
    for q in procs:
        q.start()
    for q in procs:
        q.join()
    assert [q.exitcode for q in procs] == [0, 0]
    assert count.value == 2 * n
    return list(order[:])


def _switches(order: list[int]) -> int:
    """The number of times that the process changes from one update to the
    next."""
    return sum(a != b for a, b in itertools.pairwise(order))


def test_two_processes_that_paint_keep_all_their_voxels(tmp_path: Path) -> None:
    p = tmp_path / "e_edits.npy"
    order = _paint_in_two_processes(p, lock=True)
    s = _switches(order)
    assert s >= 10, f"the updates of the two processes did not overlap ({s} switches)"
    e = LE.read(p)
    assert e.index.tolist() == [*range(100), *range(1000, 1100)]
    assert set(e.label.tolist()) == {42}


def test_two_processes_that_paint_without_the_lock_lose_voxels(
    tmp_path: Path,
) -> None:
    """The control for the test above: the same scenario with no lock loses
    voxels, so the test above can see a missing lock."""
    p = tmp_path / "e_edits.npy"
    order = _paint_in_two_processes(p, lock=False)
    s = _switches(order)
    assert s >= 10, f"the updates of the two processes did not overlap ({s} switches)"
    e = LE.read(p)
    assert len(e.index) < 200, "with no lock, no update was lost"


def test_an_update_reads_only_the_sidecar_when_its_copy_is_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "e_edits.npy"
    kw: dict[str, Any] = dict(hemisphere="R", shape=SHAPE, affine=AFFINE)
    cur = LE.update(p, lambda c: edits({(1, 1, 1): 41}), **kw)
    reads: list[Path] = []
    real = LE._read_records

    def counted(q: Path) -> bytes:
        reads.append(q)
        return real(q)

    monkeypatch.setattr(LE, "_read_records", counted)
    seen: list[LE.Edits] = []

    def add(c: LE.Current) -> LE.Edits:
        seen.append(c.edits)
        return LE._set(c.edits, np.array([7], np.int64), [42])

    cur = LE.update(p, add, known=cur, **kw)
    assert reads == [] and same(seen[-1], edits({(1, 1, 1): 41}))
    # another writer: the copy is not current, and the records are read
    write(p, edits({(2, 2, 2): 0}))
    LE.update(p, add, known=cur, **kw)
    assert reads == [p] and same(seen[-1], edits({(2, 2, 2): 0}))


def test_an_update_whose_change_raises_writes_nothing(tmp_path: Path) -> None:
    p = tmp_path / "e_edits.npy"
    write(p, edits({(1, 1, 1): 41}))
    before = p.read_bytes(), bids.sidecar_path(p).read_bytes()

    def fail(c: LE.Current) -> LE.Edits:
        raise ValueError("refused")

    with pytest.raises(ValueError, match="refused"):
        LE.update(p, fail, hemisphere="R", shape=SHAPE, affine=AFFINE)
    assert (p.read_bytes(), bids.sidecar_path(p).read_bytes()) == before


def test_an_update_whose_after_raises_writes_the_diff_back(tmp_path: Path) -> None:
    p = tmp_path / "e_edits.npy"
    kw: dict[str, Any] = dict(hemisphere="R", shape=SHAPE, affine=AFFINE)
    write(p, edits({(1, 1, 1): 41}))
    before = p.read_bytes()
    seen: list[tuple[LE.Edits, bool]] = []

    def refuse(c: LE.Current) -> None:
        # after runs after the write, with the lock held (a read would wait):
        # the new diff is on disk
        on_disk = hashlib.sha256(p.read_bytes()).hexdigest() == c.sha
        seen.append((c.edits, on_disk))
        raise OSError("the log refuses")

    with pytest.raises(OSError, match="the log refuses"):
        LE.update(p, lambda c: edits({(2, 2, 2): 42}), after=refuse, **kw)
    assert same(seen[0][0], edits({(2, 2, 2): 42})) and seen[0][1]
    assert p.read_bytes() == before and same(LE.read(p), edits({(1, 1, 1): 41}))
    # no file before the update: the empty diff is written back
    q = tmp_path / "f_edits.npy"
    with pytest.raises(OSError, match="the log refuses"):
        LE.update(q, lambda c: edits({(2, 2, 2): 42}), after=refuse, **kw)
    assert len(LE.read(q)) == 0
    # after that does not raise: the update returns the diff it wrote
    cur = LE.update(p, lambda c: edits({(3, 3, 3): 0}), after=lambda c: None, **kw)
    assert same(cur.edits, edits({(3, 3, 3): 0})) and same(LE.read(p), cur.edits)


def test_after_holds_the_lock_so_another_update_waits(tmp_path: Path) -> None:
    p = tmp_path / "e_edits.npy"
    kw: dict[str, Any] = dict(hemisphere="R", shape=SHAPE, affine=AFFINE)
    write(p, edits({(1, 1, 1): 41}))
    started, done = threading.Event(), threading.Event()
    order: list[str] = []

    def other() -> None:
        # a second writer (another viewer process: its own open of the lock)
        started.set()
        LE.update(p, lambda c: LE._set(c.edits, np.array([7], np.int64), [0]), **kw)
        order.append("other")
        done.set()

    t = threading.Thread(target=other)

    def refuse(c: LE.Current) -> None:
        t.start()
        assert started.wait(10)
        # the other update cannot take the lock while after runs
        assert not done.wait(0.5)
        with pytest.raises(BlockingIOError):
            with flock(p.parent, wait=False):
                pass
        order.append("after")
        raise OSError("the log refuses")

    with pytest.raises(OSError, match="the log refuses"):
        LE.update(p, lambda c: edits({(2, 2, 2): 42}), after=refuse, **kw)
    t.join(10)
    assert order == ["after", "other"]
    # the other writer wrote on the diff as it was before the refused update
    assert same(LE.read(p), LE._set(edits({(1, 1, 1): 41}), np.array([7]), [0]))


def test_set_merges_like_a_dict() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        old = dict(
            zip(
                rng.choice(200, 40, replace=False).tolist(),
                rng.choice([0, 41, 42], 40).tolist(),
                strict=True,
            )
        )
        idx = np.sort(rng.choice(200, 30, replace=False)).astype(np.int64)
        lab = rng.choice([LE.INPUT, 0, 41, 42], 30)
        want = dict(old)
        for i, v in zip(idx.tolist(), lab.tolist(), strict=True):
            if v == LE.INPUT:
                want.pop(i, None)
            else:
                want[i] = v
        e = LE._edits(sorted(old), [old[k] for k in sorted(old)])
        got = LE._set(e, idx, lab)
        got.check()
        assert dict(zip(got.index.tolist(), got.label.tolist(), strict=True)) == want


# ------------------------------------------------------------------ the changes


def test_apply_gives_the_labels() -> None:
    a = np.zeros(SHAPE, np.int16)
    a[2:5, 2:5, 2:5] = 41
    e = edits({(1, 1, 1): 41, (3, 3, 3): 42, (2, 2, 2): 41})
    out = LE.apply(a, e)
    assert out[1, 1, 1] == 41 and out[3, 3, 3] == 42 and out[2, 2, 2] == 41
    assert (out != a).sum() == 2 and not np.shares_memory(out, a)
    assert np.array_equal(LE.apply(a, LE.empty()), a)
    with pytest.raises(ValueError, match="outside"):
        LE.apply(np.zeros((2, 2, 2), np.int16), e)
    with pytest.raises(ValueError, match="fit"):
        LE.apply(np.zeros(SHAPE, np.int8), edits({(1, 1, 1): 192}))


def test_apply_to_a_crop_writes_only_the_voxels_in_it() -> None:
    """A crop of the grid at offset (as in paint): the diff's voxels in the
    crop are written at their crop voxel, the others are left out; the grid
    check is on the grid's shape, not the crop's."""
    a = np.arange(SHAPE[0] * SHAPE[1] * SHAPE[2], dtype=np.int32).reshape(SHAPE)
    e = edits({(1, 1, 1): 41, (3, 4, 5): 42, (11, 9, 7): 0})
    off, crop = (2, 3, 4), a[2:6, 3:7, 4:8]
    out = LE.apply(crop, e, offset=off, shape=SHAPE)
    assert np.array_equal(out, LE.apply(a, e)[2:6, 3:7, 4:8])
    assert (out != crop).sum() == 1 and out[1, 1, 1] == 42
    assert not np.shares_memory(out, crop)
    with pytest.raises(ValueError, match="outside"):
        LE.apply(crop, e, offset=off, shape=(4, 4, 4))


def labels() -> npt.NDArray[np.int16]:
    a = np.full(SHAPE, 42, np.int16)
    a[2:8, 2:8, 2:6] = 41
    return a


def test_paint_keeps_only_the_voxels_whose_label_changes() -> None:
    a = labels()
    e, undo = LE.paint(LE.empty(), a, [(3, 3, 3), (0, 0, 0), (4, 4, 4)], [41, 41, 0])
    # (3, 3, 3) is WM already: painting it WM is not an edit
    assert same(e, edits({(0, 0, 0): 41, (4, 4, 4): 0}))
    assert same(undo, edits({(0, 0, 0): -1, (4, 4, 4): -1}))
    # the current label counts the diff: painting (0, 0, 0) WM again is no edit
    e2, undo2 = LE.paint(e, a, [(0, 0, 0)], [41])
    assert same(e2, e) and len(undo2) == 0
    # a voxel with an entry painted with its input label keeps an entry
    e3, undo3 = LE.paint(e, a, [(4, 4, 4)], [41])
    assert same(e3, edits({(0, 0, 0): 41, (4, 4, 4): 41}))
    assert same(undo3, edits({(4, 4, 4): 0}))


def test_a_paint_against_the_correction_is_an_edit() -> None:
    """A paint compares with the dseg (the labels that the user sees), not
    with the input labels: a voxel that the correction changed (input 0,
    dseg 41), painted back to 0, gets an entry 0, and the undo removes it."""
    a = labels()
    a[0, 0, 0] = 0
    d = a.copy()
    d[0, 0, 0] = 41  # the correction's fill
    e, undo = LE.paint(LE.empty(), a, [(0, 0, 0)], [0], dseg=d)
    assert same(e, edits({(0, 0, 0): 0}))
    assert same(undo, edits({(0, 0, 0): -1}))
    assert len(LE.restore(e, undo)) == 0
    # painted 41, its dseg label, it is no edit
    e2, undo2 = LE.paint(LE.empty(), a, [(0, 0, 0)], [41], dseg=d)
    assert len(e2) == len(undo2) == 0
    # the eraser changes no voxel without an entry, also where dseg != input
    e3, undo3 = LE.paint(LE.empty(), a, [(0, 0, 0)], [-1], dseg=d)
    assert len(e3) == len(undo3) == 0
    with pytest.raises(ValueError, match="differ in shape"):
        LE.paint(LE.empty(), a, [(0, 0, 0)], [0], dseg=d[:-1])


def test_a_paint_does_not_depend_on_the_history() -> None:
    """Paint 41 and then 0 gives the same diff as paint 0 alone, on a voxel
    that the correction changed and on one that it did not."""
    a = labels()
    a[0, 0, 0] = 0
    d = a.copy()
    d[0, 0, 0] = 41
    vox = [(0, 0, 0), (1, 1, 1)]
    once, _ = LE.paint(LE.empty(), a, vox, [0, 0], dseg=d)
    first, _ = LE.paint(LE.empty(), a, vox, [41, 41], dseg=d)
    twice, _ = LE.paint(first, a, vox, [0, 0], dseg=d)
    assert same(once, twice)
    assert same(once, edits({(0, 0, 0): 0, (1, 1, 1): 0}))


def test_the_eraser_removes_voxels_from_the_diff() -> None:
    a = labels()
    e = edits({(0, 0, 0): 41, (4, 4, 4): 41, (5, 5, 5): 0})
    out, undo = LE.paint(e, a, [(0, 0, 0), (4, 4, 4), (1, 1, 1)], [-1, -1, -1])
    # (4, 4, 4) has its input label: erased all the same; (1, 1, 1) had no entry
    assert same(out, edits({(5, 5, 5): 0}))
    assert same(undo, edits({(0, 0, 0): 41, (4, 4, 4): 41}))
    assert np.array_equal(LE.apply(a, out)[0, 0, 0], a[0, 0, 0])


def test_paint_refuses_bad_voxels_and_labels() -> None:
    a = labels()
    for vox, lab, msg in (
        ([(12, 0, 0)], [41], "outside the grid"),
        ([(-1, 0, 0)], [41], "outside the grid"),
        ([(1, 1, 1), (1, 1, 1)], [41, 42], "twice"),
        ([(1, 1, 1)], [7], "label-set"),
        ([(1, 1, 1)], [41, 42], "length"),
    ):
        with pytest.raises(ValueError, match=msg):
            LE.paint(LE.empty(), a, vox, lab)


def test_paint_on_a_crop_of_the_input_labels() -> None:
    a = labels()
    e0 = edits({(9, 9, 7): 0})
    vox = [(3, 3, 3), (6, 7, 5), (2, 2, 2)]
    new = [42, 0, 41]
    whole = LE.paint(e0, a, vox, new)
    box = (slice(2, 8), slice(1, 9), slice(1, 7))
    crop = LE.paint(e0, a[box], vox, new, offset=(2, 1, 1), shape=SHAPE)
    assert all(same(x, y) for x, y in zip(whole, crop, strict=True))
    with pytest.raises(ValueError, match="outside the labels"):
        LE.paint(e0, a[box], [(0, 0, 0)], [41], offset=(2, 1, 1), shape=SHAPE)


def test_undo_restores_the_diff_exactly_after_overlapping_paints() -> None:
    a = labels()
    e0 = edits({(1, 1, 1): 41, (3, 3, 3): 42})
    e1, u1 = LE.paint(e0, a, [(1, 1, 1), (2, 2, 2), (3, 3, 3)], [0, 0, -1])
    e2, u2 = LE.paint(e1, a, [(2, 2, 2), (3, 3, 3), (4, 4, 4)], [42, 0, 0])
    assert same(LE.restore(e2, u2), e1)
    assert same(LE.restore(LE.restore(e2, u2), u1), e0)


def test_differ_gives_the_voxels_and_the_labels_of_a() -> None:
    a = edits({(1, 1, 1): 41, (2, 2, 2): 42, (3, 3, 3): 0})
    b = edits({(1, 1, 1): 41, (2, 2, 2): 0, (4, 4, 4): 41})
    idx, lab = LE.differ(a, b)
    want = {(2, 2, 2): 42, (3, 3, 3): 0, (4, 4, 4): -1}
    assert same(LE.Edits(idx, lab), edits(want))
    i0, l0 = LE.differ(a, a)
    assert len(i0) == len(l0) == 0


# ------------------------------------------------------------------ with dseg

DRILL = [(18, 16, z) for z in range(6, 18)]  # a 1 x 1 tunnel of cortex through WM
ASEG_SHAPE = synthetic_aseg().shape


def written_aseg(tmp_path: Path) -> Path:
    src = tmp_path / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(exist_ok=True)
    nib.save(nib.nifti1.Nifti1Image(synthetic_aseg(), np.diag([*VOX, 1.0])), src)
    return src


def aseg_edits(pairs: dict[tuple[int, int, int], int]) -> LE.Edits:
    return edits(pairs, ASEG_SHAPE)


def test_an_edit_replaces_the_automatic_cut() -> None:
    """Removing the rod by hand leaves the ring without its handle: the
    automatic correction no longer cuts it, and the fixes show the edit."""
    t = white_dseg.wm_topo(synthetic_aseg(), VOX, aseg_edits(dict.fromkeys(ROD, 0)))
    assert t.betti_before == (2, 1, 1) and t.betti_after == (2, 0, 0)
    assert kuhn_betti(np.isin(t.dseg, FILLED)) == (2, 0, 0)
    assert all(t.dseg[v] == 0 and t.fixes[v] == 4 for v in ROD)
    assert t.manual_voxels == 2 and (t.fixes == 1).sum() == 0
    assert t.moves[0] == dict(Kind="manual", Voxels=2, Centre=[27, 16, 25])
    assert sorted(m["Kind"] for m in t.moves) == ["fill", "manual"]
    assert t.fix_list[0].kind == "manual_edit" and t.fix_list[0].overrides is None
    assert all(f.overrides is None for f in t.fix_list)


def test_a_manual_edit_fix_is_a_26_connected_piece() -> None:
    """Two separate pieces of edited voxels are two fixes; voxels that touch
    at a corner are one; an edited voxel with its input label is in none."""
    a = synthetic_aseg()
    e = aseg_edits(
        {ROD[0]: 0, ROD[1]: 0, (2, 2, 2): 42, (3, 3, 3): 42, (34, 30, 2): 42}
        | {(15, 15, 10): int(a[15, 15, 10])}
    )
    t = white_dseg.wm_topo(a, VOX, e)
    man = [f for f in t.fix_list if f.kind == "manual_edit"]
    assert [f.id for f in man] == [1, 2, 3]
    pieces = sorted(len(f.index) for f in man)
    assert pieces == [1, 2, 2] and t.manual_voxels == 5


def test_the_correction_avoids_edited_voxels_where_it_can() -> None:
    """An edit that drills a tunnel through the WM makes a handle. Without
    edits, the correction plugs such a tunnel inside it; here the tunnel is
    edited, so the plug goes in the cortex above it, and no fix overrides
    the edit."""
    t = white_dseg.wm_topo(synthetic_aseg(), VOX, aseg_edits(dict.fromkeys(DRILL, 42)))
    assert t.betti_before == (2, 3, 1) and t.betti_after == (2, 0, 0)
    assert all(t.dseg[v] == 42 for v in DRILL)  # every edited voxel kept
    assert all(f.overrides is None for f in t.fix_list)
    assert t.manual_voxels == len(DRILL)


def test_a_fix_that_changes_edited_voxels_gives_their_number() -> None:
    t = white_dseg.wm_topo(synthetic_aseg(), VOX, ring_edits(synthetic_aseg()))
    assert t.betti_after == (2, 0, 0)
    over = [f for f in t.fix_list if f.overrides is not None]
    assert len(over) == 1 and over[0].overrides == 1
    assert over[0].kind == "handle_cut" and len(over[0].index) == 1
    (v,) = np.stack(np.unravel_index(over[0].index, t.dseg.shape), 1).tolist()
    assert tuple(v) in ROD and t.dseg[tuple(v)] == 0
    # the edits changed no voxel, so they have no fix of their own
    assert t.manual_voxels == 0 and t.fix_list[0].kind != "manual_edit"


def test_overrides_and_manual_voxels_match_their_definition() -> None:
    """Per fix, from its own voxels: overrides is the number of edited
    voxels that it changes, and a manual piece keeps its voxels outside the
    diff."""
    a = synthetic_aseg()
    e = ring_edits(a)
    rng = np.random.default_rng(0)
    lab = rng.choice(np.array([0, 41, 42], np.int32), len(e))
    t = white_dseg.wm_topo(
        a, VOX, LE.Edits(e.index, np.where(rng.random(len(e)) < 0.1, lab, e.label))
    )
    man = [f for f in t.fix_list if f.kind == "manual_edit"]
    assert len(man) > 1 and any(f.overrides for f in t.fix_list)
    for f in t.fix_list:
        n = int(np.isin(e.index, f.index).sum())
        assert f.overrides == (None if f.kind == "manual_edit" or not n else n)
        assert not np.isin(f.index, t.diff["index"]).any() or f.kind != "manual_edit"


def test_a_paint_back_to_the_input_against_a_fix_is_verified() -> None:
    """The cavity fill turns 8 voxels of cortex (input 42) into WM in the
    dseg. Painted back to 42 against the dseg, they get entries 42: they are
    verified (the fill that changes them reports overrides), and they are
    not a manual_edit piece (their label is the input label)."""
    a = synthetic_aseg()
    t0 = white_dseg.wm_topo(a, VOX)
    (fill,) = [f for f in t0.fix_list if f.kind == "cavity_fill"]
    vox = np.stack(np.unravel_index(fill.index, a.shape), 1)
    assert np.all(a.reshape(-1)[fill.index] == 42)
    assert np.all(t0.dseg.reshape(-1)[fill.index] == 41)
    e, _ = LE.paint(LE.empty(), a, vox, np.full(len(vox), 42), dseg=t0.dseg)
    assert np.array_equal(e.index, np.sort(fill.index)) and np.all(e.label == 42)
    t = white_dseg.wm_topo(a, VOX, e)
    assert t.manual_voxels == 0
    assert all(f.kind != "manual_edit" for f in t.fix_list)
    (over,) = [f for f in t.fix_list if f.overrides is not None]
    assert over.kind == "cavity_fill" and over.overrides == len(vox)


def test_edits_apply_to_another_input_unchanged() -> None:
    a = synthetic_aseg()
    a[0, 0, 0] = 42  # another input on the same grid
    t = white_dseg.wm_topo(a, VOX, aseg_edits(dict.fromkeys(ROD, 0)))
    assert all(t.dseg[v] == 0 for v in ROD) and t.dseg[0, 0, 0] == 42


def test_replay_with_edits_reproduces_the_dseg() -> None:
    a = synthetic_aseg()
    e = aseg_edits(dict.fromkeys(ROD, 0))
    t = white_dseg.wm_topo(a, VOX, e)
    assert t.fix_list[0].kind == "manual_edit" and len(t.fix_list[0].index) == 2
    assert white_dseg.replay_dseg(a, t.diff, e).tobytes() == t.dseg.tobytes()
    assert (t.fixes == 4).sum() == 2


def test_the_dseg_command_takes_the_edits(tmp_path: Path) -> None:
    src = written_aseg(tmp_path)
    ep = LE.bids_path(tmp_path / "edits", bids.entities(src), "R")
    e = aseg_edits(dict.fromkeys(ROD, 0))
    LE.write(ep, e, hemisphere="R", shape=ASEG_SHAPE, affine=np.diag([*VOX, 1.0]))
    out = tmp_path / "deriv"
    white_dseg.main([str(src), "--edits", str(ep), "--out", str(out)])
    anat = out / "sub-T01" / "ses-S1" / "anat"
    side = bids.read_sidecar(
        anat / "sub-T01_ses-S1_desc-wmtopo_dseg.json", white_dseg.DSEG_SPEC
    )
    copy = anat / "sub-T01_ses-S1_desc-wmtopo_edits.npy"
    want = [str(p.resolve()) for p in (src, copy)]
    assert side["Sources"] == want and set(side["SourcesSHA256"]) == set(want)
    # the copy is the diff that the run used
    got, cside, sha = LE.read_with_sidecar(copy)
    assert same(got, e) and sha == bids.file_sha256(ep)
    assert cside["Hemisphere"] == "R" and cside["Shape"] == list(ASEG_SHAPE)
    assert side["ManualVoxels"] == 2 and side["HandleCutVoxels"] == 0
    assert side["Moves"][0] == dict(Kind="manual", Voxels=2, Centre=[27, 16, 25])
    fimg = nib.load(anat / "sub-T01_ses-S1_desc-wmtopofixes_dseg.nii.gz")
    assert isinstance(fimg, nib.nifti1.Nifti1Image)
    assert (np.asarray(fimg.dataobj) == 4).sum() == 2
    fx = bids.read_sidecar(
        anat / "sub-T01_ses-S1_desc-wmtopofixes_fixes.json", white_dseg.FIXES_SPEC
    )
    assert fx["DiffInputs"]["Edits"] == bids.file_sha256(ep)
    # without edits: ManualVoxels 0, and the copy is the empty diff
    white_dseg.main([str(src), "--out", str(out)])
    plain = bids.read_sidecar(
        anat / "sub-T01_ses-S1_desc-wmtopo_dseg.json", white_dseg.DSEG_SPEC
    )
    assert plain["ManualVoxels"] == 0 and plain["Sources"][0] == str(src.resolve())
    assert len(LE.read(copy)) == 0
    # the output's copy is not an input
    with pytest.raises(ValueError, match="copy"):
        white_dseg.write_wm_topo(src, out, copy)


def test_edits_on_another_grid_or_hemisphere_are_refused(tmp_path: Path) -> None:
    src = written_aseg(tmp_path)
    ep = tmp_path / "e_edits.npy"
    e = aseg_edits(dict.fromkeys(ROD, 0))
    A = np.diag([*VOX, 1.0])
    LE.write(ep, e, hemisphere="R", shape=ASEG_SHAPE, affine=A)
    other = tmp_path / "in" / "sub-T01_ses-S2_aseg.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(synthetic_aseg()[:-1], A), other)
    with pytest.raises(ValueError, match="grid"):
        white_dseg.write_wm_topo(other, tmp_path / "deriv2", ep)
    shifted = A.copy()
    shifted[0, 3] = 1.0
    LE.write(ep, e, hemisphere="R", shape=ASEG_SHAPE, affine=shifted)
    with pytest.raises(ValueError, match="grid"):
        white_dseg.write_wm_topo(src, tmp_path / "deriv2", ep)
    LE.write(ep, e, hemisphere="L", shape=ASEG_SHAPE, affine=A)
    with pytest.raises(ValueError, match="hemisphere"):
        white_dseg.write_wm_topo(src, tmp_path / "deriv2", ep)
