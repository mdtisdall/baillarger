"""The sites where the white surface folds tightest, for review (plan:
docs/version-2-plan.md, §3c step 6; format: docs/formats/wm-sites.md).

The command line:

    python -m baillarger.wm_sites SURF [--out DIR]

It writes the sites of a white surface (a hemi-<h>_white.surf.gii that
baillarger.white_surf wrote) in DIR, or next to the surface, and prints
"role: path" for each file.

A face of the white surface (baillarger.white_surf.write_white_surface) is
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
  labels), fluid (background, CSF, the ventricles, the choroid plexus),
  mixed (both) or other;
- in a cube of CONTACT voxels round it, the pieces of the mask and of the rest
  with face adjacency and with any adjacency: more with face adjacency means
  pieces that touch only at an edge or a corner;
- the WM thickness and the gap width: the largest distance (voxels) to the
  other side, in the mask and outside it, within 2 voxels of the centre.

The queue order: grey matter, mixed, fluid, other; within each, more faces
first, then more fixes, then by the centre voxel.

Random sites are a second kind of site (random_sites, read_random_sites):
vertices of the surface that are drawn with equal weight, not ranked. They are
not written to a file: n and the seed give the same sites again.
"""

import argparse
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
from baillarger.timing import log_to_stderr
from baillarger.white_surf import SURF_SPEC, WhiteSurface

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]
type Bool = npt.NDArray[np.bool_]

GROUP = 3.0  # voxels: misfacing faces this close are one site
FIX = 4  # voxels (a cube): fixes counted round a site
TISSUE = 3  # voxels (a cube): labels outside the mask that name its tissue
CONTACT = 4  # voxels (a cube): the window of the piece counts
CORE = 2  # voxels (a cube): where thickness and gap are read
CORTEX = (3, 42)  # FreeSurfer LUT: left and right cerebral cortex
#: background, CSF, and the ventricles and the choroid plexus (left: 4, 5, 31;
#: right: 43, 44, 63). label-set/1 puts the ventricles outside the WM mask:
#: ex vivo they hold outside medium.
FLUID = (0, 24, 4, 5, 31, 43, 44, 63)
TISSUES = ("grey matter", "mixed", "fluid", "other")
FIX_KINDS = ("cut", "fill", "cavity")  # codes 1, 2, 3 of desc-wmtopofixes

#: the table's columns
COLUMNS = (
    "rank",
    "kind",
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
        out.append(dict(rank=rank, kind="misfacing", **rows[r]))
    return Sites(out, len(bad))


def sites_paths(surf: Path, out: Path | None = None) -> dict[str, Path]:
    """The sites' files of a white surface's .surf.gii: in the directory out,
    or next to the surface if out is None."""
    stem = surf.name.removesuffix("_white.surf.gii")
    d = surf.parent if out is None else out
    return dict(
        table=d / f"{stem}_desc-wmsites_sites.tsv",
        sidecar=d / f"{stem}_desc-wmsites_sites.json",
    )


@dataclass(frozen=True)
class _Loaded:
    side: dict[str, Any]
    dseg: Path
    fixes: Path
    v_vox: Float
    v_mm: Float
    f: Int
    seg: npt.NDArray[np.integer[Any]]
    fx: npt.NDArray[np.integer[Any]]
    filled: tuple[int, ...]


def _load(surf: Path) -> _Loaded:
    """The surface at surf, its corrected segmentation and its fixes (named in
    its sidecar's Sources)."""
    side = read_sidecar(sidecar_path(surf), SURF_SPEC)
    dseg = Path(side["Sources"][0])
    fixes = dseg.with_name(dseg.name.replace("desc-wmtopo_", "desc-wmtopofixes_"))
    W = WhiteSurface.load(surf)
    pts = GiftiImage.from_filename(str(surf), mmap=True).agg_data("pointset")
    dv, fv = open_volume(dseg), open_volume(fixes)
    seg = dv.read(tuple(slice(0, n) for n in dv.shape))
    fx = fv.read(tuple(slice(0, n) for n in fv.shape))
    filled = tuple(int(x) for x in side["FilledLabels"])
    return _Loaded(
        side,
        dseg,
        fixes,
        np.asarray(W.v, np.float64),
        np.asarray(pts, np.float64),
        W.f,
        seg,
        fx,
        filled,
    )


def random_sites(
    v_vox: Float,
    n: int,
    seed: int,
    seg: npt.NDArray[np.integer[Any]],
    filled: tuple[int, ...],
    fixes: npt.NDArray[np.integer[Any]],
) -> list[dict[str, Any]]:
    """n random sites: n different vertices of the surface (v_vox, in voxels
    of seg's grid), each with the same weight. The vertices are
    np.random.default_rng(seed).choice(len(v_vox), size=n, replace=False). A
    site is the vertex position rounded to the nearest voxel (np.rint),
    clipped into the grid. The rows have the columns of COLUMNS, with kind
    "random", rank 0 (no rank) and faces 0. Raises ValueError if n is more
    than the number of vertices."""
    if n > len(v_vox):
        raise ValueError(f"{n} sites asked for, the surface has {len(v_vox)} vertices")
    idx = np.random.default_rng(seed).choice(len(v_vox), size=n, replace=False)
    hi = np.array(seg.shape, np.int64) - 1
    mask = np.isin(seg, filled)
    rows = []
    for c in np.clip(np.rint(v_vox[idx]).astype(np.int64), 0, hi):
        row: dict[str, Any] = dict(
            rank=0, kind="random", i=int(c[0]), j=int(c[1]), k=int(c[2]), faces=0
        )
        row.update(measure(seg, mask, fixes, c))
        rows.append(row)
    return rows


def read_random_sites(surf: Path, n: int, seed: int) -> list[dict[str, Any]]:
    """The random sites (random_sites) of the white surface at surf, with its
    corrected segmentation and fixes loaded as write_sites loads them."""
    L = _load(surf)
    return random_sites(L.v_vox, n, seed, L.seg, L.filled, L.fx)


def write_sites(surf: Path, out: Path | None = None) -> dict[str, Path]:
    """Find the sites of the white surface at surf (its corrected segmentation
    and fixes from its sidecar's Sources) and write them in the directory out,
    or next to the surface if out is None (docs/formats/wm-sites.md). Returns
    the paths written."""
    L = _load(surf)
    dseg, fixes, side = L.dseg, L.fixes, L.side
    filled = L.filled
    S = find_sites(L.v_vox, L.f, L.v_mm, L.seg, filled, L.fx)
    p = sites_paths(surf, out)
    p["table"].parent.mkdir(parents=True, exist_ok=True)
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
            if k in ("tissue", "kind"):
                continue
            d[k] = float(d[k]) if k in ("wm_thickness", "gap_width") else int(d[k])
        rows.append(d)
    return rows, side


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    log_to_stderr()
    ap = argparse.ArgumentParser(
        description="where a white surface folds tightest (the review queue)"
    )
    ap.add_argument("surf", type=Path, help="a hemi-<h>_white.surf.gii")
    ap.add_argument(
        "--out", type=Path, help="the directory to write to (default: the surface's)"
    )
    a = ap.parse_args(argv)
    for role, path in write_sites(a.surf, a.out).items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
