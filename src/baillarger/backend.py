"""Where the array work runs: numpy (the reference) or PyTorch on a device.

device(name) turns a name into a Device:
- "numpy": the numpy reference path, float64 or float32 as each step asks;
- "cpu", "cuda", "mps": PyTorch on that device, float32 (MPS has no
  float64); "cpu" is multi-threaded;
- "auto" (the default): cuda if available, then mps, then cpu.

The environment variable BAILLARGER_DEVICE sets the default name. Asking
for a device that is not available is an error, not a silent fall back to
the CPU. The PyTorch kernels use only operations that MPS supports (no
float64, no CSR sparse tensors) and no atomic additions, so a run gives the
same result every time on a given device; tests compare each device with
numpy.
"""

import os
from dataclasses import dataclass

import torch

NAMES = ("auto", "numpy", "cpu", "cuda", "mps")


@dataclass(frozen=True)
class Device:
    """A resolved device: name is "numpy", "cpu", "cuda" or "mps"; torch is
    the torch.device, None for numpy."""

    name: str
    torch: torch.device | None

    @property
    def is_numpy(self) -> bool:
        return self.torch is None


def available() -> list[str]:
    """The device names usable here, best first."""
    out = []
    if torch.cuda.is_available():
        out.append("cuda")
    if torch.backends.mps.is_available():
        out.append("mps")
    return [*out, "cpu", "numpy"]


def device(name: str | None = None) -> Device:
    """The Device for name (default: $BAILLARGER_DEVICE, else "auto").
    Raises ValueError for an unknown name or a device that is not
    available."""
    n = name or os.environ.get("BAILLARGER_DEVICE") or "auto"
    if n not in NAMES:
        raise ValueError(f"device {n!r}: not one of {NAMES}")
    if n == "auto":
        n = available()[0]
    if n == "numpy":
        return Device("numpy", None)
    if n not in available():
        raise ValueError(f"device {n!r} is not available here: {available()}")
    return Device(n, torch.device(n))
