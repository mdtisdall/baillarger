"""A white surface from the mesh command, viewed alone (no column volume): its overview
(baillarger.viewer.overview) in the 3D surface pane, and the WM selection on it, with
the true vertex that each overview vertex is.

Run: python -m baillarger.viewer --white <..._white.surf.gii> --cache DIR (the
overview is written by the mesh command, or by python -m baillarger.white overview)
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from baillarger.viewer.overview import overview_surface, surface_name

type Float = npt.NDArray[np.float64]


class WhiteView:
    """The overview's Surface (S), the overview (ov) and the true surface (W) of the
    surface at surf; see the module docstring."""

    def __init__(
        self,
        surf: Path,
        cache: Path,
        log: Callable[[str], object] = print,
    ) -> None:
        self.surf = Path(surf)
        self.name = surface_name(self.surf)
        self.S, self.ov, self.W = overview_surface(self.surf, cache, log)
        self.no_trace = -np.ones(len(self.S.v_vox), np.int32)

    def meta(self) -> dict[str, Any]:
        """The page's meta in surface-only mode: no traces, layers or flat map."""
        return dict(
            site=f"{self.name} (white surface)",
            surface_only=True,
            n=0,
            uv=[],
            rv=[],
            levels=self.S.infl_names,
            layers=[],
            classes=[],
            points=0,
            points_world=[],
            vmax=1.0,
            has_levels=False,
            tracer="none",
            wm_label="none",
            overview=dict(
                true_vertices=len(self.W.v),
                true_faces=len(self.W.f),
                vertices=len(self.ov.vertex_ids),
                faces=len(self.ov.f),
                eps_mm=self.ov.sidecar["EpsMM"],
                max_distance_mm=round(self.ov.sidecar["MaxDistanceMM"], 4),
            ),
        )

    def vertex(self, k: int) -> dict[str, Any]:
        """The WM selection at overview vertex k: its world point and the true
        vertex it is (an index into the surface's files)."""
        k = int(k)
        p = self.S.world(self.S.v_vox[k])[0]
        return dict(
            vertex=k,
            true_vertex=int(self.ov.vertex_ids[k]),
            world_mm=[round(float(x), 3) for x in p],
            trace=None,
            root_mm=None,
        )

    def snap(self, mm: npt.ArrayLike) -> dict[str, Any]:
        """The WM selection at the overview vertex nearest world point mm."""
        return self.vertex(self.S.nearest(mm))

    def mesh_payload(self, level: int, faces: bool) -> bytes:
        """The 3D pane's overview at inflated level `level`; with faces, every
        vertex without a trace."""
        return self.S.mesh_payload(level, self.no_trace if faces else None)

    def white_bytes(self) -> bytes:
        return self.S.white_bytes()
