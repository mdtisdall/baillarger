"""The viewer's overview of a white surface written by the mesh command: the
overview that the command writes with the surface (baillarger.white.WhiteOverview,
docs/formats/white-overview.md), and a Surface (baillarger.viewer.surface) on it.

The true surface is too large for the panes and for the viewer's whole-mesh steps (the
graph, the inflation, the trees): about 25 M faces for a hemisphere. The overview keeps
its topology, every true vertex within its EpsMM, and only true vertices, so each
overview vertex is a true vertex (vertex_ids). It is read, not made, here; it is checked
against the true surface's data (the two sidecars' SHA-256). The Surface's inflation is
cached in the cache directory, keyed by the overview file's SHA-256 and VERSION.

The overview is closed, so its Surface has no flat map (flat=False) until the flat map
handles closed pieces.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from baillarger.bids import file_sha256
from baillarger.viewer.surface import Surface
from baillarger.white import WhiteOverview, WhiteSurface, overview_paths

#: bumped when the viewer's use of the overview changes, so that caches are remade
VERSION = 1


def surface_name(surf: Path) -> str:
    """The name of a .surf.gii without its suffix (the viewer's cache files)."""
    return surf.name.removesuffix(".surf.gii").removesuffix("_white")


def overview_surface(
    surf: Path, cache: Path, log: Callable[[str], object] = print
) -> tuple[Surface, WhiteOverview, WhiteSurface]:
    """A Surface on the overview of the white surface at surf (its graph, inflation
    and picking; the inflation cached), the overview and the true surface
    (memory-mapped). Raises FileNotFoundError if the overview was not written
    (python -m baillarger.white overview SURF), ValueError if it was made from
    other data."""
    W = WhiteSurface.load(surf)
    path = overview_paths(surf)["overview"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name}: no overview; write it with "
            f"python -m baillarger.white overview {surf}"
        )
    ov = WhiteOverview.load(path)
    ov.check(W)
    key = hashlib.sha1(f"{file_sha256(path)} v{VERSION}".encode()).hexdigest()[:16]
    vox = np.linalg.norm(W.affine[:3, :3], axis=0)
    S = Surface(
        np.asarray(ov.v, np.float64),
        ov.f,
        vox,
        W.affine,
        cache,
        f"{surface_name(surf)}_overview",
        key=key,
        log=log,
        flat=False,
    )
    return S, ov, W
