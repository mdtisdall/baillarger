"""The viewer's job pattern: a computation that takes too long for one request
runs as a job, one job at a time, in a thread.

- A job has a key (a string that names its inputs) and a label (for the page).
- One job runs at a time. A second job is refused while one runs (RuntimeError).
- The state (status, key, label, the last LOG_LINES lines of the job's log,
  the error) is kept under a lock. The page reads it (status) and polls until
  the status is "done" or "error".
- The last keep_in_memory results are kept in memory. With a cache directory
  and a save and a load function, each result is also saved there, as
  <prefix>_<key><suffix>, so that a request for the same key later, or after a
  restart, loads it and does not compute it again.

The module reads no data files. Each user gives the work and the format of
its results: baillarger.viewer.columns (the columns near a vertex).
"""

import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: the number of log lines that the state keeps
LOG_LINES = 50

type Log = Callable[[str], object]


class Jobs[R]:
    """Jobs whose results are of type R; see the module docstring. save(result,
    path) and load(path) are needed together with cache."""

    def __init__(
        self,
        cache: Path | None = None,
        prefix: str = "job",
        suffix: str = ".npz",
        save: Callable[[R, Path], None] | None = None,
        load: Callable[[Path], R] | None = None,
        log: Log = print,
        keep_in_memory: int = 3,
    ) -> None:
        if cache is not None and (save is None or load is None):
            raise ValueError("a cache directory needs a save and a load function")
        self.cache = None if cache is None else Path(cache)
        self.prefix, self.suffix = prefix, suffix
        self.save, self.load = save, load
        self.log_fn = log
        self.keep_in_memory = keep_in_memory
        self.results: dict[str, R] = {}
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.state: dict[str, Any] = dict(status="idle", key=None, label=None, log=[])

    def path(self, key: str) -> Path | None:
        """The file of the result with this key, or None without a cache."""
        if self.cache is None:
            return None
        return self.cache / f"{self.prefix}_{key}{self.suffix}"

    def files(self) -> list[Path]:
        """The result files in the cache, by name (no partial files)."""
        if self.cache is None:
            return []
        return sorted(
            f
            for f in self.cache.glob(f"{self.prefix}_*{self.suffix}")
            if not f.name.endswith(".tmp" + self.suffix)
        )

    def log(self, msg: str) -> None:
        with self.lock:
            self.state["log"] = (self.state["log"] + [msg])[-LOG_LINES:]
        self.log_fn(msg)

    def status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state, log=list(self.state["log"]))

    def has(self, key: str) -> bool:
        """Whether the result with this key is in memory or in the cache."""
        with self.lock:
            if key in self.results:
                return True
        p = self.path(key)
        return p is not None and p.exists()

    def get(self, key: str) -> R:
        """A finished result, from memory or the cache (KeyError: none)."""
        with self.lock:
            if key in self.results:
                return self.results[key]
        p = self.path(key)
        if p is None or not p.exists() or self.load is None:
            raise KeyError(f"no result {key}")
        r = self.load(p)
        self.remember(key, r)
        return r

    def remember(self, key: str, r: R) -> None:
        with self.lock:
            self.results.pop(key, None)
            self.results[key] = r
            while len(self.results) > self.keep_in_memory:
                self.results.pop(next(iter(self.results)))

    def run(
        self, key: str, label: str, work: Callable[[Log], R], wait: bool = False
    ) -> dict[str, Any]:
        """Start work (it gets the job's log function) as the job `key`, or find
        it done. Returns {key, label, status}; status "done" when the result is
        ready now."""
        if self.has(key):
            with self.lock:
                self.state.update(status="done", key=key, label=label)
            return dict(key=key, label=label, status="done")
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError(
                    f"a job is already running ({self.state['label']}): one at a time"
                )
            self.state = dict(status="running", key=key, label=label, log=[])
            self.thread = threading.Thread(
                target=self._work, args=(key, work), daemon=True
            )
            self.thread.start()
            th = self.thread
        if wait:
            th.join()
        return dict(key=key, label=label, status=self.status()["status"])

    def _work(self, key: str, work: Callable[[Log], R]) -> None:
        try:
            r = work(self.log)
            p = self.path(key)
            if p is not None and self.save is not None:
                self.save(r, p)
            self.remember(key, r)
            with self.lock:
                self.state["status"] = "done"
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state["status"] = "error"
                self.state["error"] = f"{type(exc).__name__}: {exc}"
