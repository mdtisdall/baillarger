"""File helpers shared by the writers: one lock and one atomic write.

flock holds an advisory lock (fcntl.flock) on a directory or a file. A flock
belongs to an open file description, so a second flock on the same path in
the same process, from another open, waits for the first one (or fails, with
wait False). write_atomic replaces a file so that a reader sees the old file
or the new file, never a part of one, also after a power loss.
"""

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def flock(path: Path, exclusive: bool = True, wait: bool = True) -> Iterator[None]:
    """A lock (fcntl.flock) on path, held in the block: exclusive (LOCK_EX)
    or shared (LOCK_SH). path is a directory, or a file (made empty if it
    does not exist). With wait False, raises BlockingIOError at once if
    another open of path holds a lock that conflicts."""
    flags = os.O_RDONLY if path.is_dir() else os.O_RDONLY | os.O_CREAT
    fd = os.open(path, flags, 0o666)
    try:
        op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, op if wait else op | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def fsync_dir(directory: Path) -> None:
    """Flush the entries of directory (a rename in it) to the disk."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path: Path, data: bytes) -> None:
    """Write data to a new file in path's directory, flush it to the disk,
    rename it over path, then flush the directory. If a step fails, the new
    file is removed and path is unchanged."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)
