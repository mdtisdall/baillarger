"""contact1d: the line through the fused part of a connected kiss (ff_ckiss), in closed form.

Along the local x axis: WM | cortex B (thickness tB) | outside medium for |x| < g | cortex A
(thickness tA) | WM. g is the slot's HALF-width, as in ff_ckiss (the slot is 2g wide). The
line runs from bank B's start point (START_DEPTH_MM behind its root, x = −g − tB − 2) to bank
A's start point (x = g + tA + 2).

The closed forms (ff_ckiss with r = |x|, one tau_scale = (tA + tB) / 2):
- φ = 1 − (|x| − g) / t, t = tA for x > 0, tB for x < 0;
- the true τ = (φ − 1) · tau_scale = −(|x| − g) · tau_scale / t in tissue. In the outside
  medium τ > 0 is the virtual continuation beyond the top; the two sides' continuations meet at
  x_ridge = g (tB − tA) / (tA + tB) (not at 0 when tA ≠ tB), and the true τ there is the lower
  of the two (continuous, one ridge);
- the true speed |dx/dτ| = t / tau_scale on each side;
- the true tops: x = −g (side B) and x = +g (side A);
- stop_sdf = g − |x| (exact in 1D);
- the oracle stop field: p = Φ(stop_sdf / σ), σ = oracle_blur_vox · VOXEL_MM, clipped to
  oracle_clip, as log-odds (the formula of ff_oracle.OracleStopField, CD-4).

The mesh: nodes c + i h. `centre` = "mid" puts x = 0 in the middle of a cell, "node" on a node.
The first node is at or before B's start point, the last at or after A's start point.

No code here labels the sides for the solver: the solver sees only the line, the nodes and the
stop field. The sides appear only in the truth and the measures.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
import ff_ckiss as CK  # noqa: E402
import ff_types as T  # noqa: E402

THICKNESS = {"ckiss": (CK.TA, CK.TB), "equal": (2.4, 2.4)}  # (tA, tB) mm


@dataclass(frozen=True)
class Contact:
    tA: float
    tB: float
    g: float
    settings: T.SolverSettings = T.DEFAULTS

    @property
    def tau_scale(self) -> float:
        return 0.5 * (self.tA + self.tB)

    @property
    def x_start_b(self) -> float:
        return -self.g - self.tB - self.settings.start_depth_mm

    @property
    def x_start_a(self) -> float:
        return self.g + self.tA + self.settings.start_depth_mm

    def thickness(self, x: T.Array) -> T.Array:
        return np.where(np.asarray(x) > 0, self.tA, self.tB)

    @property
    def x_ridge(self) -> float:
        """Where the two virtual continuations of the true τ meet in the outside medium:
        (g + x) / tB = (g − x) / tA, so x = g (tB − tA) / (tA + tB). 0 for equal banks."""
        return self.g * (self.tB - self.tA) / (self.tA + self.tB)

    def tau_true(self, x: T.Array) -> T.Array:
        """The true τ in tissue; in the outside medium the lower of the two sides' virtual
        continuations, so τ is continuous with its ridge at x_ridge (the S1 form)."""
        x = np.asarray(x, float)
        tb = (self.g + x) * self.tau_scale / self.tB
        ta = (self.g - x) * self.tau_scale / self.tA
        return np.asarray(np.minimum(ta, tb))

    def speed_true(self, x: T.Array) -> T.Array:
        return np.asarray(self.thickness(x) / self.tau_scale, float)

    def stop_sdf(self, x: T.Array) -> T.Array:
        return self.g - np.abs(np.asarray(x, float))

    def stop_probability(self, x: T.Array) -> T.Array:
        return np.asarray(ndtr(self.stop_sdf(x) / (self.settings.oracle_blur_vox * T.VOXEL_MM)))

    def stop_log_odds(self, x: T.Array) -> T.Array:
        return T.log_odds(self.stop_probability(x), *self.settings.oracle_clip)

    def log_odds_t(self, x: torch.Tensor) -> torch.Tensor:
        """The oracle log-odds in Torch (differentiable in x)."""
        st = self.settings
        sdf = self.g - torch.abs(x)
        p = torch.special.ndtr(sdf / (st.oracle_blur_vox * T.VOXEL_MM))
        lo, hi = st.oracle_clip
        p = torch.clamp(p, lo, hi)
        return torch.log(p) - torch.log1p(-p)

    def nodes(self, h: float, centre: str) -> T.Array:
        off = 0.5 * h if centre == "mid" else 0.0
        i0 = int(np.floor((self.x_start_b - off) / h + 1e-9))
        i1 = int(np.ceil((self.x_start_a - off) / h - 1e-9))
        return off + h * np.arange(i0, i1 + 1)
