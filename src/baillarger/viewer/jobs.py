"""The viewer's job pattern: a computation that takes too long for one request
runs as a job, one job at a time, in a thread.

- A job has a key (a string that names its inputs) and a label (for the page).
- One job runs at a time. A second job is refused while one runs (RuntimeError).
- The state (status, key, label, the last LOG_LINES lines of the job's log,
  the error) is kept under a lock. The page reads it (status) and polls until
  the status is "done" or "error".
- A failed job's error is reported one time, to the request with its key that
  finds it (run, or wait after run). That request does not start the job
  again. The next request with that key starts it again: a failure can be
  transient, so it is not kept as a result. status() keeps the error for the
  page until the next job starts.
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
        #: whether the error in state was reported to a request (take_error)
        self.reported = False

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
        it done or failed. Returns {key, label, status}: status "done" when the
        result is ready now, "running", or "error" (with "error") when the last
        job with this key failed and its error was not reported yet (the work
        does not run again: see the module docstring). wait: wait for the job
        (wait)."""
        if self.has(key):
            with self.lock:
                self.state.update(status="done", key=key, label=label)
            return dict(key=key, label=label, status="done")
        with self.lock:
            err = self.take_error(key)
            if err is not None:
                return dict(key=key, label=label, status="error", error=err)
            if self.state["status"] == "running":
                raise RuntimeError(
                    f"a job is already running ({self.state['label']}): one at a time"
                )
            self.state = dict(status="running", key=key, label=label, log=[])
            self.reported = False
            self.thread = threading.Thread(
                target=self._work, args=(key, work), daemon=True
            )
            self.thread.start()
        if wait:
            return self.wait(key, None)
        return dict(key=key, label=label, status="running")

    def wait(self, key: str, timeout: float | None) -> dict[str, Any]:
        """Wait up to timeout s (None: until it ends) for the running job, then
        the outcome for key, as run returns it: "done", "error" (reported now,
        one time) or else the state's status ("running" while the job of key
        runs)."""
        th = self.thread
        if th is not None:
            th.join(timeout)
        done = self.has(key)
        with self.lock:
            label = self.state["label"] if self.state["key"] == key else None
            if done:
                return dict(key=key, label=label, status="done")
            err = self.take_error(key)
            if err is not None:
                return dict(key=key, label=label, status="error", error=err)
            status = self.state["status"] if self.state["key"] == key else "idle"
            return dict(key=key, label=label, status=status)

    def take_error(self, key: str) -> str | None:
        """The error of the last job if it had this key, failed, and its error
        was not reported yet; then it is reported. Call it under the lock."""
        st = self.state
        if st["key"] != key or st["status"] != "error" or self.reported:
            return None
        self.reported = True
        return str(st.get("error", "the job failed"))

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
