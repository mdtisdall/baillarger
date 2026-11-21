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

A crease is an edge of the surface whose two faces are folded back on each
other: the cosine of the angle between their normals is at most
baillarger.mesh_checks.FOLD_COS (-0.9, 154 degrees; the shared test,
mesh_checks.edge_folds). The surface stays embedded there, but its normals
are not well defined. On the first hemisphere every crease sat on a strand
or a gap one voxel thick: the two vertices opposite the crease edge lie
close to one voxel centre or to two neighbouring ones, the tip. Creases
within GROUP voxels of each other are one group. A group within GROUP
voxels of a misfacing face joins that face's site; any other group is a
crease site, centred on its tip. A site with creases proposes a choice
(baillarger.wm_choices) for the reviewer to accept: remove-strand if the
tip is in the WM mask, fill-gap if not; the tip is that of the crease with
the largest angle.

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
from baillarger.fix_diff import read_table
from baillarger.labels import OUTSIDE, TO_OUTSIDE
from baillarger.mesh import edge_points
from baillarger.mesh_checks import FOLD_COS, edge_face_pairs, edge_folds, pair_cosines
from baillarger.timing import log_to_stderr
from baillarger.white_dseg import output_paths, surface_dseg
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
#: outside (0) and the labels that the dseg step sets to outside (TO_OUTSIDE:
#: CSF, the ventricles and the choroid plexus), which a dseg written before
#: that step can still hold. Ex vivo they hold outside medium.
FLUID = (OUTSIDE, *TO_OUTSIDE)
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
    "fix",
    "wall",
    "creases",
    "crease_angle",
    "proposal",
)
#: the integer columns of the table
_INTS = tuple(
    c
    for c in COLUMNS
    if c
    not in ("kind", "tissue", "wm_thickness", "gap_width", "crease_angle", "proposal")
)
#: the columns that can be n/a (None in a row)
_OPTIONAL = ("crease_angle", "proposal")
#: the crease columns of a site with no crease
NO_CREASE: dict[str, Any] = dict(creases=0, crease_angle=None, proposal=None)
#: the proposal of a crease site, by whether its tip is in the WM mask
PROPOSALS = {True: "remove-strand", False: "fill-gap"}

#: voxels: a handle fix with a wall of this size or more is a site
MIN_WALL = 10
#: the kinds of site, in queue order (random sites are not in the file)
KINDS = ("override", "misfacing", "crease", "handle-fix")

#: docs/formats/wm-sites.md
SITES_SPEC = SidecarSpec(
    "wm-sites/1",
    COMMON_KEYS
    + (
        Key("Hemisphere", "string", True, "R or L"),
        Key("FilledLabels", "array of integers", True, "labels of the WM mask"),
        Key("MisfacingFaces", "integer", True, "misfacing faces of the surface"),
        Key("CreaseEdges", "integer", True, "crease edges of the surface"),
        Key("CreaseCosine", "number", True, "the crease test's cut-off"),
        Key("Sites", "integer", True, "rows of the table"),
        Key("GroupVoxels", "number", True, "faces this close are one site"),
        Key("FixVoxels", "integer", True, "half-side of the fix window"),
        Key("TissueVoxels", "integer", True, "half-side of the tissue window"),
        Key("ContactVoxels", "integer", True, "half-side of the piece window"),
        Key("CoreVoxels", "integer", True, "half-side of the thickness window"),
        Key("MinWallVoxels", "integer", True, "smallest wall of a handle-fix site"),
        Key("SitesByKind", "object", True, "rows of the table, by kind"),
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


@dataclass(frozen=True)
class Creases:
    """The crease edges of a surface (module docstring), k of them: their
    vertices (k, 2), their two faces (k, 2), the cosine of the angle
    between the faces' normals (k,), the edge's midpoint in voxels (k, 3)
    and the tip (k, 3): a voxel of the grid."""

    edges: Int
    faces: Int
    cos: Float
    mid: Float
    tip: Int

    @property
    def degrees(self) -> Float:
        """The angle between the two faces' normals, in degrees."""
        out: Float = np.degrees(np.arccos(np.clip(self.cos, -1.0, 1.0)))
        return out


def tips(v_vox: Float, keys: Int, shape: tuple[int, ...]) -> tuple[Int, Float]:
    """Per vertex (v_vox in voxels, keys its grid-edge keys,
    baillarger.mesh.edge_key, on a grid of shape): the end of its grid edge
    that is nearer to it, and the distance to that end (voxels)."""
    lo, hi = edge_points(keys, shape)
    d = (hi - lo).astype(np.float64)
    t = np.einsum("ij,ij->i", v_vox - lo, d) / np.einsum("ij,ij->i", d, d)
    near: Int = np.where((t < 0.5)[:, None], lo, hi)
    dist: Float = np.minimum(t, 1 - t) * np.linalg.norm(d, axis=1)
    return near, dist


def creases(
    v_vox: Float,
    f: Int,
    v_mm: Float,
    keys: Int,
    shape: tuple[int, ...],
    fold_cos: float = FOLD_COS,
) -> Creases:
    """The crease edges of the surface (v_mm in mm for the normals; v_vox
    the same vertices in voxels of a grid of shape; keys their grid-edge
    keys): the shared test, mesh_checks.edge_folds at fold_cos. The tip of
    a crease is the nearer grid point (tips) of whichever of the two
    vertices opposite the edge is closer to its own."""
    E, P = edge_face_pairs(f, len(v_mm))
    fold = edge_folds(v_mm, f, P, fold_cos)
    E, P = E[fold], P[fold]
    cos = pair_cosines(v_mm, f, P)
    opp = f[P].sum(2) - E.sum(1)[:, None]  # (k, 2): the vertex off the edge
    near, dist = tips(v_vox[opp.ravel()], keys[opp.ravel()], shape)
    pick = np.argmin(dist.reshape(-1, 2), 1)
    tip = near.reshape(-1, 2, 3)[np.arange(len(E)), pick]
    mid = v_vox[E].mean(1)
    return Creases(E, P, cos, mid, tip.astype(np.int64))


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
    """The sites of a white surface: the misfacing sites and the crease
    sites, each in queue order (one dict per site, the table's columns), the
    number of misfacing faces and of crease edges."""

    rows: list[dict[str, Any]]
    misfacing_faces: int
    crease_rows: list[dict[str, Any]]
    crease_edges: int


def _crease_columns(C: Creases, members: Int, mask: Bool) -> dict[str, Any]:
    """The crease columns of a site whose creases are members (indices
    into C), and the tip of the largest angle."""
    if not len(members):
        return dict(NO_CREASE)
    top = members[np.argmin(C.cos[members])]
    tip = C.tip[top]
    return dict(
        creases=len(members),
        crease_angle=round(float(C.degrees[top]), 1),
        proposal=PROPOSALS[bool(mask[tuple(tip)])],
    )


def find_sites(
    v_vox: Float,
    f: Int,
    v_mm: Float,
    keys: Int,
    seg: npt.NDArray[np.integer[Any]],
    filled: tuple[int, ...],
    fixes: npt.NDArray[np.integer[Any]],
) -> Sites:
    """The sites of the surface (v_mm in mm for the normals; v_vox the same
    vertices in voxels of seg's grid; keys their grid-edge keys), from the
    corrected segmentation seg (its WM mask: filled) and its fixes."""
    bad = np.flatnonzero(misfacing(v_mm, f))
    cen = v_vox[f[bad]].mean(1)
    lab = group(cen)
    mask = np.isin(seg, filled)
    C = creases(v_vox, f, v_mm, keys, seg.shape)
    # crease groups; a group near a misfacing face joins that face's site
    clab = group(C.mid)
    joins: dict[int, list[int]] = {}
    alone: list[Int] = []
    tree = cKDTree(cen) if len(cen) else None
    for g in range(int(clab.max()) + 1 if len(clab) else 0):
        members = np.flatnonzero(clab == g)
        if tree is not None:
            q = tree.query(C.mid[members])
            d, at = np.asarray(q[0]), np.asarray(q[1])
            if d.min() <= GROUP:
                site = int(lab[at[np.argmin(d)]])
                joins.setdefault(site, []).extend(members.tolist())
                continue
        alone.append(members)
    rows = []
    for s in range(int(lab.max()) + 1 if len(lab) else 0):
        m = lab == s
        c = np.round(cen[m].mean(0)).astype(np.int64)
        row: dict[str, Any] = dict(
            i=int(c[0]), j=int(c[1]), k=int(c[2]), faces=int(m.sum())
        )
        row.update(measure(seg, mask, fixes, c))
        row.update(_crease_columns(C, np.array(joins.get(s, []), np.int64), mask))
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
        out.append(dict(rank=rank, kind="misfacing", **rows[r], fix=0, wall=0))
    crows = []
    for members in alone:
        cc = _crease_columns(C, members, mask)
        c = C.tip[members[np.argmin(C.cos[members])]]
        row = dict(rank=0, kind="crease", i=int(c[0]), j=int(c[1]), k=int(c[2]))
        row.update(faces=0, **measure(seg, mask, fixes, c), fix=0, wall=0, **cc)
        crows.append(row)
    crows.sort(
        key=lambda r: (
            TISSUES.index(r["tissue"]),
            -r["creases"],
            -r["crease_angle"],
            (r["i"], r["j"], r["k"]),
        )
    )
    return Sites(out, len(bad), crows, len(C.cos))


def fix_sites(
    table: list[dict[str, Any]],
    seg: npt.NDArray[np.integer[Any]],
    filled: tuple[int, ...],
    fixes: npt.NDArray[np.integer[Any]],
    min_wall: int = MIN_WALL,
) -> list[dict[str, Any]]:
    """The sites of the fix table (baillarger.fix_diff.read_table), with no
    rank: kind override for each fix that changes an edited voxel (its
    overrides is not n/a); kind handle-fix for each other handle fix with a
    wall of min_wall voxels or more. The site is the fix's i, j, k. To
    accept the fix of an override site, the user writes the fix's labels into
    the label edits: the next dseg then has no override there."""
    mask = np.isin(seg, filled)
    out = []
    for r in table:
        if r["i"] is None:
            continue
        if r["overrides"] is not None:
            kind = "override"
        elif (
            r["kind"] in ("handle_cut", "handle_fill") and (r["wall"] or 0) >= min_wall
        ):
            kind = "handle-fix"
        else:
            continue
        c = np.array([r["i"], r["j"], r["k"]], np.int64)
        row: dict[str, Any] = dict(
            rank=0, kind=kind, i=int(c[0]), j=int(c[1]), k=int(c[2]), faces=0
        )
        row.update(measure(seg, mask, fixes, c))
        row.update(fix=int(r["id"]), wall=int(r["wall"] or 0), **NO_CREASE)
        out.append(row)
    return out


def queue(
    misfacing_rows: list[dict[str, Any]],
    fix_rows: list[dict[str, Any]],
    crease_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The queue: override sites (by fix id), then the misfacing sites (in
    their order), then the crease sites (in their order), then the
    handle-fix sites (larger walls first; ties: by fix id), ranked from 1."""
    over = sorted(
        (r for r in fix_rows if r["kind"] == "override"), key=lambda r: r["fix"]
    )
    hand = sorted(
        (r for r in fix_rows if r["kind"] == "handle-fix"),
        key=lambda r: (-r["wall"], r["fix"]),
    )
    rows = over + misfacing_rows + list(crease_rows or []) + hand
    return [dict(r, rank=k) for k, r in enumerate(rows, 1)]


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
    keys: Int
    seg: npt.NDArray[np.integer[Any]]
    fx: npt.NDArray[np.integer[Any]]
    filled: tuple[int, ...]


def _load(surf: Path) -> _Loaded:
    """The surface at surf, its corrected segmentation and its fixes (named in
    its sidecar's Sources)."""
    side = read_sidecar(sidecar_path(surf), SURF_SPEC)
    dseg = surface_dseg(side)
    fixes = output_paths(dseg)["fixes"]
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
        W.vertex_key,
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
        row.update(fix=0, wall=0, **NO_CREASE)
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
    S = find_sites(L.v_vox, L.f, L.v_mm, L.keys, L.seg, filled, L.fx)
    table_path = output_paths(dseg)["table"]
    extra: list[Path] = []
    fix_rows: list[dict[str, Any]] = []
    if table_path.exists():
        fix_rows = fix_sites(read_table(table_path), L.seg, filled, L.fx)
        extra = [table_path]
    rows = queue(S.rows, fix_rows, S.crease_rows)
    p = sites_paths(surf, out)
    p["table"].parent.mkdir(parents=True, exist_ok=True)
    dat = surf.with_name(surf.name.removesuffix(".surf.gii") + ".dat")
    with p["table"].open("w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join(_cell(r[c]) for c in COLUMNS) + "\n")
    write_sidecar(
        p["sidecar"],
        SITES_SPEC,
        dict(
            SpecVersion=SITES_SPEC.version,
            Description="The review queue of the WM mask: the sites that "
            "need judgment, in queue order.",
            **sources(surf, dat, dseg, fixes, *extra),
            Provenance=provenance("baillarger.wm_sites.write_sites"),
            Hemisphere=side["Hemisphere"],
            FilledLabels=list(filled),
            MisfacingFaces=S.misfacing_faces,
            CreaseEdges=S.crease_edges,
            CreaseCosine=FOLD_COS,
            Sites=len(rows),
            GroupVoxels=GROUP,
            FixVoxels=FIX,
            TissueVoxels=TISSUE,
            ContactVoxels=CONTACT,
            CoreVoxels=CORE,
            MinWallVoxels=MIN_WALL,
            SitesByKind={k: sum(r["kind"] == k for r in rows) for k in KINDS},
        ),
    )
    return p


def _cell(x: Any) -> str:
    return "n/a" if x is None else str(x)


def read_sites(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The sites' table at path (a list of rows, typed) and its sidecar. The
    review pane reads the queue of a hemisphere with it."""
    side = read_sidecar(sidecar_path(path), SITES_SPEC)
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    if tuple(head) != COLUMNS:
        raise ValueError(f"{path.name}: columns {head} are not {list(COLUMNS)}")
    rows = []
    for line in lines[1:]:
        d: dict[str, Any] = dict(zip(head, line.split("\t"), strict=True))
        for k in COLUMNS:
            if k in _OPTIONAL and d[k] == "n/a":
                d[k] = None
            elif k in _INTS:
                d[k] = int(d[k])
            elif k in ("wm_thickness", "gap_width", "crease_angle"):
                d[k] = float(d[k])
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
