"""Named volumes for the brick pane: a registry of brick pyramids
(baillarger.viewer.bricks) by name, each an image (the echo: block means) or a label
image (the segmentation, the fixes image: block modes), with the colour table of each
label image.

Routes (served in every mode that has the registry; see route):
- /volume/list: the names, their kinds ("image" or "labels"), the dtype on disk and
  the wire dtype (null until the pyramid is built), and the colour table's name.
- /volume/<name>/meta: the pyramid's meta (bricks.meta), with the kind and, for a
  label image, its colour table {value: {name, rgba}}. The pyramid is built on the
  first call.
- /volume/<name>/bricks?level=L&b=i,j,k;...: as /volume/bricks.
/volume, /volume/meta and /volume/bricks stay the default volume (the echo).

The pyramids are built on first use, one at a time (one lock for the registry), so
two builds never hold their temporaries at the same time.

With --white, the segmentation is the first entry of the white surface's `Sources`
(white-surf/1), and the fixes image is its `desc-wmtopofixes` neighbour (wm-dseg/1).
When a source path is not a file (the derivatives were moved), the file of the same
name next to the surface is used. --dseg and --fixes give other paths.

Colours: the labels of label-set/1 take their FreeSurferColorLUT colours
(LABEL_RGB). The fixes image takes FIX_RGB (our own: FreeSurfer has no such image).
"""

import argparse
import gzip
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baillarger.labels import LUT_NAMES
from baillarger.viewer.bricks import (
    WIRE_NAMES,
    ArrayPyramid,
    BrickSource,
    LazyVolume,
    nifti,
)

#: FreeSurferColorLUT colours (R, G, B) of the labels of label-set/1 (LUT_NAMES)
LABEL_RGB: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    2: (245, 245, 245),
    3: (205, 62, 78),
    4: (120, 18, 134),
    5: (196, 58, 250),
    10: (0, 118, 14),
    11: (122, 186, 220),
    12: (236, 13, 176),
    13: (12, 48, 255),
    17: (220, 216, 20),
    18: (103, 255, 255),
    26: (255, 165, 0),
    28: (165, 42, 42),
    30: (160, 32, 240),
    31: (0, 200, 200),
    41: (245, 245, 245),
    42: (205, 62, 78),
    43: (120, 18, 134),
    44: (196, 58, 250),
    49: (0, 118, 14),
    50: (122, 186, 220),
    51: (236, 13, 176),
    52: (13, 48, 255),
    53: (220, 216, 20),
    54: (103, 255, 255),
    58: (255, 165, 0),
    60: (165, 42, 42),
    62: (160, 32, 240),
    63: (0, 200, 221),
    192: (0, 0, 160),
}

#: the values of the fixes image (wm-dseg-fixes/1; 5 is phase 1.0.5's small piece)
FIX_NAMES = {
    1: "handle_cut",
    2: "handle_fill",
    3: "cavity_fill",
    4: "manual_edit",
    5: "small_piece",
}
FIX_RGB: dict[int, tuple[int, int, int]] = {
    1: (255, 80, 40),
    2: (40, 200, 255),
    3: (255, 220, 0),
    4: (60, 255, 90),
    5: (180, 110, 255),
}

#: names that the routes use, so no volume can have them
RESERVED = {"list", "meta", "bricks"}


def colour_table(kind: str) -> dict[str, dict[str, Any]]:
    """The colour table of a label image: "labels" (label-set/1) or "fixes". Value 0
    is transparent (alpha 0); every other value is opaque."""
    names, rgb = (LUT_NAMES, LABEL_RGB) if kind == "labels" else (FIX_NAMES, FIX_RGB)
    out: dict[str, dict[str, Any]] = {}
    for v, n in sorted(names.items()):
        if v == 0:
            continue
        out[str(v)] = dict(name=n, rgba=[*rgb[v], 255])
    return out


@dataclass
class Named:
    volume: LazyVolume
    kind: str  # "image" or "labels"
    table: str | None  # the colour table of a label image: "labels" or "fixes"
    disk_dtype: str


class Volumes:
    """The named volumes (the module docstring)."""

    def __init__(self) -> None:
        self.items: dict[str, Named] = {}
        self.build = threading.Lock()  # one pyramid built at a time

    def add(
        self,
        name: str,
        make: Callable[[], BrickSource],
        kind: str = "image",
        table: str | None = None,
        disk_dtype: str = "",
    ) -> None:
        if name in RESERVED or "/" in name or not name:
            raise ValueError(f"a volume cannot be named {name!r}")
        if kind not in ("image", "labels") or (kind == "labels") != (table is not None):
            raise ValueError("an image has no colour table; a label image needs one")

        def built() -> BrickSource:
            with self.build:
                return make()

        self.items[name] = Named(LazyVolume(built, name), kind, table, disk_dtype)

    def add_lazy(self, name: str, volume: LazyVolume, disk_dtype: str = "") -> None:
        """An image already made as a LazyVolume (the default volume), shared."""
        if name in RESERVED or "/" in name:
            raise ValueError(f"a volume cannot be named {name!r}")
        make = volume.make

        def built() -> BrickSource:
            with self.build:
                return make()

        volume.make = built
        self.items[name] = Named(volume, "image", None, disk_dtype)

    def add_nifti(
        self, name: str, path: Path, kind: str, table: str | None, brick: int = 32
    ) -> None:
        """A NIfTI image (read on first use): an image, or a label image (mode)."""
        p = Path(path)
        dt = str(nifti(p).get_data_dtype())
        reducer = "mode" if kind == "labels" else "mean"
        self.add(
            name,
            lambda: ArrayPyramid.from_nifti(p, brick=brick, reducer=reducer),
            kind,
            table,
            dt,
        )

    def list(self) -> list[dict[str, Any]]:
        out = []
        for name, n in self.items.items():
            src = n.volume._src
            out.append(
                dict(
                    name=name,
                    kind=n.kind,
                    table=n.table,
                    dtype=n.disk_dtype,
                    wire=None if src is None else WIRE_NAMES[np.dtype(src.dtype)],
                )
            )
        return out

    def meta(self, name: str) -> dict[str, Any]:
        n = self.items[name]
        out = dict(n.volume.meta(), kind=n.kind)
        if n.table is not None:
            out["table"] = colour_table(n.table)
        return out


def route(h: Any, path: str, q: dict[str, str]) -> bool:
    """Serve /volume/list and /volume/<name>/(meta|bricks) for handler h (its
    `volumes`); False for any other path (the old /volume routes)."""
    vs: Volumes | None = getattr(h, "volumes", None)
    parts = path.split("/")  # "", "volume", ...
    if len(parts) == 3 and parts[2] == "list":
        if vs is None:
            h.send_error(404)
        else:
            h.send(json.dumps(vs.list()).encode(), "application/json")
        return True
    if len(parts) != 4 or parts[2] in RESERVED:
        return False
    name, what = parts[2], parts[3]
    if vs is None or name not in vs.items or what not in ("meta", "bricks"):
        h.send_error(404)
        return True
    if what == "meta":
        h.send_json(lambda: vs.meta(name))
        return True
    try:
        body = vs.items[name].volume.bricks(int(q["level"]), q.get("b", ""))
    except (KeyError, ValueError) as exc:
        h.send(str(exc).encode(), "text/plain", 400)
        return True
    h.send_response(200)
    h.send_header("Content-Type", "application/octet-stream")
    if "gzip" in (h.headers.get("Accept-Encoding") or ""):
        body = gzip.compress(body, compresslevel=1)
        h.send_header("Content-Encoding", "gzip")
    h.send_header("Content-Length", str(len(body)))
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    h.wfile.write(body)
    return True


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--dseg",
        type=Path,
        help="with --white: the segmentation of the label overlay (default: the white "
        "surface's sidecar Sources)",
    )
    ap.add_argument(
        "--fixes",
        type=Path,
        help="with --white: the fixes image of the overlay (default: the "
        "desc-wmtopofixes neighbour of the segmentation)",
    )


def _found(p: Path, surf: Path) -> Path:
    """p, or the file of the same name next to the surface when p is not a file."""
    if p.is_file():
        return p
    q = surf.with_name(p.name)
    if q.is_file():
        return q
    raise FileNotFoundError(f"{p.name}: not at {p.parent} nor next to {surf.name}")


def white_paths(
    surf: Path, sources: list[str], dseg: Path | None, fixes: Path | None
) -> tuple[Path, Path]:
    """The segmentation and the fixes image of a white surface (the module
    docstring)."""
    d = Path(dseg) if dseg is not None else _found(Path(sources[0]), surf)
    if fixes is not None:
        return d, Path(fixes)
    name = d.name.replace("desc-wmtopo_", "desc-wmtopofixes_")
    return d, _found(d.with_name(name), surf)


def for_white(
    surf: Path,
    sources: list[str],
    echo: LazyVolume | None,
    echo_dtype: str,
    dseg: Path | None = None,
    fixes: Path | None = None,
) -> Volumes:
    """The registry of --white mode: the echo (if any), the labels and the fixes."""
    vs = Volumes()
    if echo is not None:
        vs.add_lazy("echo", echo, echo_dtype)
    d, f = white_paths(surf, sources, dseg, fixes)
    vs.add_nifti("labels", d, "labels", "labels")
    vs.add_nifti("fixes", f, "labels", "fixes")
    return vs
