"""The sites where the white surface folds tightest, for review (plan:
PLAN_WM_REVIEW.md; format: docs/formats/wm-sites.md).

A face of the white surface (baillarger.white.write_white_surface) is
misfacing when its normal points away from the summed normals of its own
corners (each vertex's faces' cross products, summed:
baillarger.decimate.normals). The surface is a single level set, so no face is
flipped; a misfacing face marks a place where the surface folds back within a
few faces: a pinch, the tip of a thin strand, the stub of a handle cut. Faces
whose centroids are within GROUP voxels of each other are one site.

Per site, from the corrected segmentation (desc-wmtopo_dseg) and its fixes
(desc-wmtopofixes: 1 handle cut, 2 handle fill, 3 cavity fill):

- the topology fixes of each kind within FIX voxels (a cube) of its centre;
- the tissue outside the WM mask within TISSUE voxels: grey matter (cortex
  labels), fluid (background or CSF), mixed (both) or other;
- in a cube of CONTACT voxels round it, the pieces of the mask and of the rest
  with face adjacency and with any adjacency: more with face adjacency means
  pieces that touch only at an edge or a corner;
- the WM thickness and the gap width: the largest distance (voxels) to the
  other side, in the mask and outside it, within 2 voxels of the centre.

The queue order: grey matter, mixed, fluid, other; within each, more faces
first, then more fixes, then by the centre voxel.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from nibabel.gifti.gifti import GiftiImage
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    provenance,
    read_sidecar,
    sidecar_path,
    sources,
    write_sidecar,
)
from baillarger.blocks import open_volume
from baillarger.decimate import CHUNK, normals

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]
type Bool = npt.NDArray[np.bool_]

GROUP = 3.0  # voxels: misfacing faces this close are one site
FIX = 4  # voxels (a cube): fixes counted round a site
TISSUE = 3  # voxels (a cube): labels outside the mask that name its tissue
CONTACT = 4  # voxels (a cube): the window of the piece counts
CORE = 2  # voxels (a cube): where thickness and gap are read
CORTEX = (3, 42)  # FreeSurfer LUT: left and right cerebral cortex
FLUID = (0, 24)  # background and CSF
TISSUES = ("grey matter", "mixed", "fluid", "other")
FIX_KINDS = ("cut", "fill", "cavity")  # codes 1, 2, 3 of desc-wmtopofixes

#: the table's columns
COLUMNS = (
    "rank",
    "i",
    "j",
    "k",
    "faces",
    "tissue",
    "cut",
    "fill",
    "cavity",
    "wm_face_pieces",
    "wm_any_pieces",
    "gap_face_pieces",
    "gap_any_pieces",
    "wm_thickness",
    "gap_width",
)

#: docs/formats/wm-sites.md
SITES_SPEC = SidecarSpec(
    "wm-sites/1",
    COMMON_KEYS
    + (
        Key("Hemisphere", "string", True, "R or L"),
        Key("FilledLabels", "array of integers", True, "labels of the WM mask"),
        Key("MisfacingFaces", "integer", True, "misfacing faces of the surface"),
        Key("Sites", "integer", True, "rows of the table"),
        Key("GroupVoxels", "number", True, "faces this close are one site"),
        Key("FixVoxels", "integer", True, "half-side of the fix window"),
        Key("TissueVoxels", "integer", True, "half-side of the tissue window"),
        Key("ContactVoxels", "integer", True, "half-side of the piece window"),
        Key("CoreVoxels", "integer", True, "half-side of the thickness window"),
    ),
)


def misfacing(v: Float, f: Int) -> Bool:
    """Per face of mesh (v, f): whether its normal points away from (or square
    to) the summed normals of its corners (normals). CHUNK faces at a time."""
    N = normals(v, f)
    out = np.zeros(len(f), bool)
    for a in range(0, len(f), CHUNK):
        fc = f[a : a + CHUNK]
        n = np.cross(v[fc[:, 1]] - v[fc[:, 0]], v[fc[:, 2]] - v[fc[:, 0]])
        S = N[fc[:, 0]] + N[fc[:, 1]] + N[fc[:, 2]]
        out[a : a + len(fc)] = np.sum(n * S, 1) <= 0
    return out


def group(points: Float, radius: float = GROUP) -> Int:
    """Labels 0, 1, ... of the groups of points linked by distances of at most
    radius (single linkage)."""
    if not len(points):
        return np.zeros(0, np.int64)
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    A = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2
    )
    lab: Int = connected_components(A, directed=False)[1].astype(np.int64)
    return lab


def _window(a: npt.NDArray[Any], c: Int, h: int) -> npt.NDArray[Any]:
    lo = np.maximum(c - h, 0)
    return a[lo[0] : c[0] + h + 1, lo[1] : c[1] + h + 1, lo[2] : c[2] + h + 1]


def _pieces(m: Bool) -> tuple[int, int]:
    face = ndi.label(m, structure=ndi.generate_binary_structure(3, 1))[1]
    anyc = ndi.label(m, structure=ndi.generate_binary_structure(3, 3))[1]
    return int(face), int(anyc)


def tissue(labels: npt.ArrayLike) -> str:
    """The tissue named by the labels outside the WM mask round a site."""
    s = set(np.unique(np.asarray(labels)).tolist())
    gm, fl = bool(s & set(CORTEX)), bool(s & set(FLUID))
    return "mixed" if gm and fl else "grey matter" if gm else "fluid" if fl else "other"


def measure(
    seg: npt.NDArray[np.integer[Any]],
    mask: Bool,
    fixes: npt.NDArray[np.integer[Any]],
    c: npt.ArrayLike,
) -> dict[str, Any]:
    """The per-site measures (module docstring) at voxel c."""
    c = np.asarray(c, np.int64)
    fx = _window(fixes, c, FIX)
    out: dict[str, Any] = {k: int((fx == i + 1).sum()) for i, k in enumerate(FIX_KINDS)}
    sw, mw = _window(seg, c, TISSUE), _window(mask, c, TISSUE)
    out["tissue"] = tissue(sw[~mw])
    win = _window(mask, c, CONTACT)
    out["wm_face_pieces"], out["wm_any_pieces"] = _pieces(win)
    out["gap_face_pieces"], out["gap_any_pieces"] = _pieces(~win)
    pad = CORE + 8
    big = _window(mask, c, pad)
    at = c - np.maximum(c - pad, 0)  # c in big
    core = tuple(slice(max(a - CORE, 0), a + CORE + 1) for a in at)
    out["wm_thickness"] = round(float(ndi.distance_transform_edt(big)[core].max()), 2)
    out["gap_width"] = round(float(ndi.distance_transform_edt(~big)[core].max()), 2)
    return out


@dataclass(frozen=True)
class Sites:
    """The sites of a white surface, in queue order: one dict per site (the
    table's columns) and the number of misfacing faces."""

    rows: list[dict[str, Any]]
    misfacing_faces: int


def find_sites(
    v_vox: Float,
    f: Int,
    v_mm: Float,
    seg: npt.NDArray[np.integer[Any]],
    filled: tuple[int, ...],
    fixes: npt.NDArray[np.integer[Any]],
) -> Sites:
    """The sites of the surface (v_mm in mm for the normals; v_vox the same
    vertices in voxels of seg's grid), from the corrected segmentation seg (its
    WM mask: filled) and its fixes."""
    bad = np.flatnonzero(misfacing(v_mm, f))
    cen = v_vox[f[bad]].mean(1)
    lab = group(cen)
    mask = np.isin(seg, filled)
    rows = []
    for s in range(int(lab.max()) + 1 if len(lab) else 0):
        m = lab == s
        c = np.round(cen[m].mean(0)).astype(np.int64)
        row = dict(i=int(c[0]), j=int(c[1]), k=int(c[2]), faces=int(m.sum()))
        row.update(measure(seg, mask, fixes, c))
        rows.append(row)
    order = sorted(
        range(len(rows)),
        key=lambda r: (
            TISSUES.index(rows[r]["tissue"]),
            -rows[r]["faces"],
            -(rows[r]["cut"] + rows[r]["fill"] + rows[r]["cavity"]),
            (rows[r]["i"], rows[r]["j"], rows[r]["k"]),
        ),
    )
    out = []
    for rank, r in enumerate(order, 1):
        out.append(dict(rank=rank, **rows[r]))
    return Sites(out, len(bad))


def sites_paths(surf: Path) -> dict[str, Path]:
    """The sites' files next to a white surface's .surf.gii."""
    stem = surf.name.removesuffix("_white.surf.gii")
    return dict(
        table=surf.with_name(f"{stem}_desc-wmsites_sites.tsv"),
        sidecar=surf.with_name(f"{stem}_desc-wmsites_sites.json"),
    )


def write_sites(surf: Path) -> dict[str, Path]:
    """Find the sites of the white surface at surf (its corrected segmentation
    and fixes from its sidecar's Sources) and write them next to it
    (docs/formats/wm-sites.md). Returns the paths written."""
    from baillarger.white import SURF_SPEC, WhiteSurface

    side = read_sidecar(sidecar_path(surf), SURF_SPEC)
    dseg = Path(side["Sources"][0])
    fixes = dseg.with_name(dseg.name.replace("desc-wmtopo_", "desc-wmtopofixes_"))
    W = WhiteSurface.load(surf)
    pts = GiftiImage.from_filename(str(surf), mmap=True).agg_data("pointset")
    dv, fv = open_volume(dseg), open_volume(fixes)
    seg = dv.read(tuple(slice(0, n) for n in dv.shape))
    fx = fv.read(tuple(slice(0, n) for n in fv.shape))
    filled = tuple(int(x) for x in side["FilledLabels"])
    S = find_sites(
        np.asarray(W.v, np.float64), W.f, np.asarray(pts, np.float64), seg, filled, fx
    )
    p = sites_paths(surf)
    dat = surf.with_name(surf.name.removesuffix(".surf.gii") + ".dat")
    with p["table"].open("w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in S.rows:
            fh.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")
    write_sidecar(
        p["sidecar"],
        SITES_SPEC,
        dict(
            SpecVersion=SITES_SPEC.version,
            Description="Where the white surface folds tightest, for review: "
            "sites of misfacing faces, in queue order.",
            **sources(surf, dat, dseg, fixes),
            Provenance=provenance("baillarger.wm_sites.write_sites"),
            Hemisphere=side["Hemisphere"],
            FilledLabels=list(filled),
            MisfacingFaces=S.misfacing_faces,
            Sites=len(S.rows),
            GroupVoxels=GROUP,
            FixVoxels=FIX,
            TissueVoxels=TISSUE,
            ContactVoxels=CONTACT,
            CoreVoxels=CORE,
        ),
    )
    return p


def read_sites(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The sites' table at path (a list of rows, typed) and its sidecar."""
    side = read_sidecar(sidecar_path(path), SITES_SPEC)
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    if tuple(head) != COLUMNS:
        raise ValueError(f"{path.name}: columns {head} are not {list(COLUMNS)}")
    rows = []
    for line in lines[1:]:
        d: dict[str, Any] = dict(zip(head, line.split("\t"), strict=True))
        for k in COLUMNS:
            if k == "tissue":
                continue
            d[k] = float(d[k]) if k in ("wm_thickness", "gap_width") else int(d[k])
        rows.append(d)
    return rows, side
