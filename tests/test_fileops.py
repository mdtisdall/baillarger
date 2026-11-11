"""Tests for the shared file helpers (baillarger.fileops): the lock and the
atomic write."""

import os
from pathlib import Path
from typing import Any

import pytest

from baillarger import fileops


def test_write_atomic_replaces_the_file_and_syncs_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "a.txt"
    p.write_bytes(b"old")
    synced: list[int] = []
    real = os.fsync

    def fsync(fd: int) -> None:
        synced.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    fileops.write_atomic(p, b"new")
    assert p.read_bytes() == b"new" and [q.name for q in tmp_path.iterdir()] == [
        "a.txt"
    ]
    assert len(synced) == 2  # the file, then the directory


def test_a_failed_rename_leaves_the_old_file_and_no_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "a.txt"
    p.write_bytes(b"old")

    def fail(a: Any, b: Any) -> None:
        raise OSError("stopped")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="stopped"):
        fileops.write_atomic(p, b"new")
    assert p.read_bytes() == b"old" and [q.name for q in tmp_path.iterdir()] == [
        "a.txt"
    ]


def test_flock_on_a_directory_and_on_a_file(tmp_path: Path) -> None:
    f = tmp_path / "x.lock"
    for path in (tmp_path, f):
        with fileops.flock(path, exclusive=True):
            # a second open of the same path: its lock conflicts
            with pytest.raises(BlockingIOError):
                with fileops.flock(path, exclusive=False, wait=False):
                    pass
        with fileops.flock(path, exclusive=False):
            with fileops.flock(path, exclusive=False, wait=False):
                pass
        with fileops.flock(path, exclusive=True, wait=False):
            pass
    assert f.is_file()  # a file path is made
