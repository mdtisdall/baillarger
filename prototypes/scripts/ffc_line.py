"""The 1D line problems of the continuous column solver (docs/plans/continuous-column-solver.md,
draft): slab1d, with oracle inputs.

slab1d is the slab of ff_simple (WM for x < 0, cortex 0 <= x <= t, outside medium for x > t,
t = ff_simple.T_DEFAULT = 2.4 mm) along one line: the local x axis, which is the cortex normal.
The line is in the slab's local mm frame, so the box rotation R_DEFAULT does not apply. Every
value comes from `ff_simple.SlabGeometry` and `ff_types`:

- the true τ = (φ − 1) · tau_scale (`ff_types`), tau_scale = t; so τ = x − t, and τ_top = 0 at
  the pia;
- the true speed 1 / (tau_scale |∇φ|) = 1 everywhere;
- the start point: START_DEPTH_MM behind the root (x = 0) along the normal, so x = −2 mm;
- the oracle stop field: the formula of ff_oracle.OracleStopField (origin/feature/ff-inputs,
  CD-4): p = Φ(stop_sdf / σ), σ = oracle_blur_vox · VOXEL_MM = 0.08 mm, clipped to
  oracle_clip, as log-odds. (ff_oracle is not on this branch's base, so the formula is
  repeated here; the test checks it against the closed form.)

The mesh. Nodes x_i = x_start + i h, from the start point to x_end = t + margin. With h = 0.16
mm and t = 2.4 mm the pia is at (t − x_start) / h = 27.5 cells: in the middle of a cell, which
is the worst case for a node-based stop (±½ h).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
import ff_simple as S  # noqa: E402
import ff_types as T  # noqa: E402

MARGIN_MM = 1.0  # the stopped region kept beyond the true pia (mm)


@dataclass(frozen=True)
class Line:
    """One 1D line problem. All lengths in mm, along the local x axis."""

    name: str
    geometry: S.SlabGeometry
    tau_scale: float
    x_start: float  # the column's fixed start point
    x_end: float  # the end of the line, in the stopped region
    x_stop_true: float  # the true stop (stop_sdf = 0)
    settings: T.SolverSettings

    def local(self, x: T.Array) -> T.Array:
        x = np.asarray(x, float)
        Y = np.zeros((x.size, 3))
        Y[:, 0] = x.ravel()
        return Y

    def tau_true(self, x: T.Array) -> T.Array:
        return np.asarray((self.geometry.phi(self.local(x)) - 1.0) * self.tau_scale)

    def speed_true(self, x: T.Array) -> T.Array:
        g = np.linalg.norm(self.geometry.grad_phi(self.local(x)), axis=1)
        return np.asarray(1.0 / (self.tau_scale * g))

    def stop_probability(self, x: T.Array) -> T.Array:
        """The oracle's p before the clip: Φ(stop_sdf / σ)."""
        sigma = self.settings.oracle_blur_vox * T.VOXEL_MM
        return np.asarray(ndtr(self.geometry.stop_sdf(self.local(x)) / sigma))

    def stop_log_odds(self, x: T.Array) -> T.Array:
        """The oracle stop field: log-odds of the clipped p."""
        return T.log_odds(self.stop_probability(x), *self.settings.oracle_clip)

    def nodes(self, h: float) -> T.Array:
        """The mesh nodes x_start + i h, i = 0 .. n, with the last node at or beyond x_end."""
        n = int(np.ceil((self.x_end - self.x_start) / h - 1e-9))
        return self.x_start + h * np.arange(n + 1)


def slab1d(settings: T.SolverSettings = T.DEFAULTS) -> Line:
    """slab1d: the slab of ff_simple along its normal (module docstring)."""
    geo, tau_scale, _ = S.geometry("slab")
    t = float(geo.p["t"])
    return Line(
        name="slab1d",
        geometry=geo,
        tau_scale=float(tau_scale),
        x_start=-settings.start_depth_mm,
        x_end=t + MARGIN_MM,
        x_stop_true=t,
        settings=settings,
    )
