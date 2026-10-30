"""Per-step time and memory, logged, so that a run shows where its time and
memory go (the scaling work decides what to change from these).

    with step("marching", dev):
        ...

or, along a function, a lap timer (one line at the end of each step):

    lap = Laps(dev)
    ...
    lap("marching")

logs, at INFO on the "baillarger" logger, the step's wall time, the
process's peak resident memory so far, and for a PyTorch GPU the memory
the device holds. A GPU step is synchronized before its time is read, so
the time includes the device's work. The command lines turn the log on
(log_to_stderr).
"""

import logging
import resource
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from baillarger.backend import Device

log = logging.getLogger("baillarger")


def peak_rss_gb() -> float:
    """The process's peak resident memory so far, GB (ru_maxrss is in bytes
    on macOS, kB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(r) / (2**30 if sys.platform == "darwin" else 2**20)


def _sync(dev: Device | None) -> str:
    """Wait for dev's queued work; its memory in use, as text."""
    if dev is None or dev.torch is None:
        return ""
    if dev.name == "cuda":
        torch.cuda.synchronize()
        return f"   cuda {torch.cuda.max_memory_allocated() / 2**30:5.1f} GB peak"
    if dev.name == "mps":
        torch.mps.synchronize()
        return f"   mps {torch.mps.driver_allocated_memory() / 2**30:5.1f} GB"
    return ""


@contextmanager
def step(name: str, dev: Device | None = None) -> Iterator[None]:
    """Log name's wall time and memory when the block ends."""
    t0 = time.perf_counter()
    yield
    gpu = _sync(dev)
    dt = time.perf_counter() - t0
    log.info(f"{name:<24} {dt:8.1f} s   peak {peak_rss_gb():5.1f} GB{gpu}")


class Laps:
    """A lap timer: each call logs the step that just ended (as step)."""

    def __init__(self, dev: Device | None = None) -> None:
        self.dev = dev
        self.t = time.perf_counter()

    def __call__(self, name: str) -> None:
        gpu = _sync(self.dev)
        now = time.perf_counter()
        log.info(
            f"{name:<24} {now - self.t:8.1f} s   peak {peak_rss_gb():5.1f} GB{gpu}"
        )
        self.t = now


def log_to_stderr() -> None:
    """Send the baillarger log, INFO and up, to stderr (command lines)."""
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
