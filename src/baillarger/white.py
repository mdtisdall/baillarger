"""The WM topology correction and the white surface of a hemisphere.

The WM mask (FILLED_LABELS: the WM and the deep structures it encloses) must
be cavity- and handle-free, so that the white mesh has genus-0 pieces. The
correction is made once, on the whole mask (baillarger.handles,
remove_handles), because a box crop of a handle-free solid has planar surface
pieces, while a box alone cannot tell a handle from a fold that closes outside
it. wm_topo makes the corrected segmentation: the aseg's labels, with the
mask's cavities and handle fills set to the WM label and each handle cut
given the label of its nearest neighbour outside the mask. Every later stage
reads its labels from the corrected segmentation, never the aseg plus a list
of changes: the whole-hemisphere surface (write_white_surface) and the box
stages (baillarger.column_traces.load_tissue, baillarger.pial_labels, whose
WM is the corrected mask of their box: filled_labels).

white_mesh makes the white mesh of a box's label: marching tetrahedra at 0.5
of the binary label (so the mesh has the label's Kuhn topology), then
tangential relaxation and Gaussian smoothing without shrinkage
(baillarger.mesh_smoothing), which move vertices but keep the faces.
white_mesh_normals adds the smoothed vertex normals (out of the WM) that the
column traces start from.

The command line:

    python -m baillarger.white dseg ASEG [--edits EDITS] --out DERIVATIVES
    python -m baillarger.white mesh DSEG --out DERIVATIVES [--block-cells N]
        [--device DEVICE] [--overview-eps MM] [--workers N]
    python -m baillarger.white overview SURF [--eps MM] [--workers N]

dseg writes the corrected segmentation (desc-wmtopo_dseg) and the changed
voxels (desc-wmtopofixes_dseg), with their .tsv and .json sidecars
(docs/formats/wm-dseg.md); EDITS, manual edits of the aseg
(desc-wmedits_edits.tsv, baillarger.wm_edits), are made before the automatic
correction. mesh writes the white surface of a corrected
segmentation as GIfTI, with its normals, edge keys and block index
(docs/formats/white-surf.md), then its overview (write_white_overview): the
surface decimated for viewing, as an ordinary GIfTI with embedded data
(docs/formats/white-overview.md). overview writes the overview of a surface
that mesh wrote.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage, GiftiMetaData
from scipy import ndimage as ndi

from baillarger import antialias as AA
from baillarger import backend
from baillarger import mesh_smoothing as S
from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    dataset_description,
    derivative_path,
    entities,
    file_sha256,
    provenance,
    read_sidecar,
    sidecar_path,
    sources,
    write_sidecar,
)
from baillarger.blocks import EmbeddedSource, open_volume
from baillarger.decimate import closest_points, csr_take, decimate, reorder
from baillarger.gifti import DataArray, write_external
from baillarger.handles import kuhn_betti, kuhn_structure, remove_handles
from baillarger.mesh import (
    BLOCK,
    MeshBlocks,
    marching_tetrahedra,
    marching_tetrahedra_blocks,
    topology,
)
from baillarger.mesh_torch import MeshOps
from baillarger.timing import Laps, log, log_to_stderr
from baillarger.wm_edits import Edits, apply_edits, read_edits, sources_match

type Bool = npt.NDArray[np.bool_]
type Betti = tuple[int, int, int]
type Float = npt.NDArray[np.float64]
type F32 = npt.NDArray[np.float32]
type Int = npt.NDArray[np.int64]

# ------------------------------------------------- the corrected segmentation

#: the labels filled into the WM mask, by hemisphere (FreeSurfer LUT): WM,
#: lateral ventricle, thalamus, caudate, putamen, pallidum, corpus callosum.
#: The hippocampus is not filled: its laminar axis does not run out from the
#: WM below it.
FILLED_LABELS = {"R": (41, 43, 49, 50, 51, 52, 192), "L": (2, 4, 10, 11, 12, 13, 192)}
WM_LABEL = {"R": 41, "L": 2}

#: FreeSurfer LUT names of the labels the corrected segmentation's .tsv lists
LUT_NAMES = {
    0: "Unknown",
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    192: "Corpus_Callosum",
}

#: values of the fixes image: a handle cut, a handle fill, a cavity fill, a
#: manual edit (baillarger.wm_edits)
FIXES = {1: "handle_cut", 2: "handle_fill", 3: "cavity_fill", 4: "manual_edit"}

#: docs/formats/wm-dseg.md
DSEG_SPEC = SidecarSpec(
    "wm-dseg/1",
    COMMON_KEYS
    + (
        Key("Hemisphere", "string", True, "R or L"),
        Key("FilledLabels", "array of integers", True, "labels of the WM mask"),
        Key("WMLabel", "integer", True, "label given to cavity and handle fills"),
        Key("BettiBefore", "array of integers", True, "b0, b1, b2 of the mask"),
        Key("BettiAfter", "array of integers", True, "b0, b1, b2 after"),
        Key("CavityVoxels", "integer", True, "voxels set to WM: cavities"),
        Key("HandleFillVoxels", "integer", True, "voxels set to WM: handle fills"),
        Key("HandleCutVoxels", "integer", True, "voxels cut from the mask"),
        Key("CutLabels", "object", True, "new label of the cut voxels: count"),
        Key("ManualVoxels", "integer", True, "voxels changed by manual edits"),
        Key("Moves", "array of objects", True, "each manual edit and handle move"),
    ),
)
#: docs/formats/wm-dseg.md
FIXES_SPEC = SidecarSpec("wm-dseg-fixes/1", COMMON_KEYS)


def hemisphere(labels: npt.ArrayLike) -> str:
    """R or L: the hemisphere whose WM label is in the segmentation. Raises
    ValueError for both (a whole brain needs a midline cut first) or
    neither."""
    present = [h for h, w in WM_LABEL.items() if np.any(np.asarray(labels) == w)]
    if len(present) != 1:
        raise ValueError(
            f"WM labels present for hemispheres {present}: need exactly one"
        )
    return present[0]


def cavities(obj: Bool) -> Bool:
    """The background voxels that do not reach the array's border (Kuhn
    adjacency, as baillarger.handles.kuhn_betti counts cavities)."""
    lab, _ = ndi.label(np.pad(~obj, 1, constant_values=True), kuhn_structure())
    out: Bool = (lab[1:-1, 1:-1, 1:-1] != lab[0, 0, 0]) & ~obj
    return out


def relabel_cuts(
    labels: npt.NDArray[np.integer[Any]],
    cut: Bool,
    donor: Bool,
    voxel_mm: npt.ArrayLike,
    margin: int = 4,
) -> None:
    """Give each cut voxel the label of the nearest donor voxel (Euclidean, in
    mm), in place. Each connected piece of the cut is done in its bounding
    box plus margin voxels, widened until the nearest donor found is nearer
    than the box's edge."""
    vox = np.asarray(voxel_mm, np.float64)
    lab, _ = ndi.label(cut, np.ones((3, 3, 3), bool))
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        m = margin
        while True:
            box = tuple(
                slice(max(t.start - m, 0), min(t.stop + m, n))
                for t, n in zip(sl, labels.shape, strict=True)
            )
            d = donor[box]
            whole = all(
                b.start == 0 and b.stop == n
                for b, n in zip(box, labels.shape, strict=True)
            )
            if d.any():
                dist, idx = ndi.distance_transform_edt(
                    ~d, sampling=vox, return_indices=True
                )
                sel = lab[box] == k
                if whole or dist[sel].max() <= m * vox.min():
                    break
            elif whole:
                raise ValueError("no donor voxel for a cut")
            m *= 2
        src = tuple(i[sel] for i in idx)
        sub = labels[box]
        sub[sel] = sub[src]


@dataclass(frozen=True)
class WMTopo:
    """The corrected segmentation and what was changed."""

    #: the segmentation with the WM mask made cavity- and handle-free
    dseg: npt.NDArray[np.integer[Any]]
    #: FIXES values at the changed voxels, 0 elsewhere
    fixes: npt.NDArray[np.uint8]
    hemi: str
    betti_before: Betti
    betti_after: Betti
    #: label: count, for the cut voxels
    cut_labels: dict[int, int]
    #: per manual edit (Answer: its Id) and handle move: Kind, Voxels, Centre
    #: (0-based voxel of the grid)
    moves: list[dict[str, Any]]
    #: voxels changed by the manual edits
    manual_voxels: int = 0


def wm_topo(
    aseg: npt.ArrayLike, voxel_mm: npt.ArrayLike, edits: Edits | None = None
) -> WMTopo:
    """The corrected segmentation of an aseg: the manual edits made first
    (baillarger.wm_edits.apply_edits; it raises ValueError for edits that do
    not fit the aseg), then the mask of its FILLED_LABELS,
    with its cavities filled and its handles removed
    (baillarger.handles.remove_handles). Cavity and handle fills become the
    WM label; each handle cut takes the label of the nearest voxel that is
    outside the corrected mask and not a cut. The corrected segmentation's
    voxels with a filled label are exactly the corrected mask. Raises
    RuntimeError if that mask still has a handle or a cavity (b1 or b2 not 0),
    which a fill of one piece of the mask touching another can make."""
    lap = Laps()
    a0 = np.asarray(aseg)
    manual = None
    if edits is not None:
        a0, manual = apply_edits(a0, edits)
        lap("manual edits")
    hemi = hemisphere(a0)
    filled, wm_label = FILLED_LABELS[hemi], WM_LABEL[hemi]
    m = np.isin(a0, filled)
    nz = [
        np.flatnonzero(m.any(axis=tuple(j for j in range(3) if j != i)))
        for i in range(3)
    ]
    lo = np.array([max(int(t[0]) - 2, 0) for t in nz])
    hi = np.array([min(int(t[-1]) + 3, s) for t, s in zip(nz, m.shape, strict=True)])
    box = tuple(slice(int(p), int(q)) for p, q in zip(lo, hi, strict=True))
    x = np.ascontiguousarray(m[box])
    del m
    lap("mask")
    before = kuhn_betti(x)
    lap("Betti before")
    cav = cavities(x)
    x |= cav
    lap("cavities")
    r = remove_handles(x)
    obj = r.obj
    lap("handles")
    cut, fill = x & ~obj, obj & ~x
    a = a0[box]
    labels = a.copy()
    labels[cav | fill] = wm_label
    relabel_cuts(labels, cut, ~obj & ~cut, voxel_mm)
    lap("relabel cuts")
    if not np.array_equal(np.isin(labels, filled), obj):
        raise AssertionError("the corrected labels do not give the corrected mask")
    after = kuhn_betti(obj)
    lap("Betti after")
    if after[1] or after[2]:
        # remove_handles fixes each piece with the others as background: a fill
        # of one piece can touch another and make a handle or a cavity
        raise RuntimeError(
            f"the corrected WM mask has Betti numbers {after} (before: {before}): "
            "a handle fill of one piece of the mask touched another"
        )
    dseg = a0.copy()
    dseg[box] = labels
    fixes = np.zeros(a0.shape, np.uint8)
    if manual is not None:
        fixes[manual] = 4  # an edited voxel the correction then changes shows that
    fx = fixes[box]
    fx[cut] = 1
    fx[fill] = 2
    fx[cav] = 3
    new, cnt = np.unique(labels[cut], return_counts=True)
    moves = []
    if edits is not None:
        for ans in edits.answers:
            v = edits.voxels[edits.answer == ans.id]
            if len(v):
                moves.append(
                    dict(
                        Kind="manual",
                        Voxels=len(v),
                        Centre=np.round(v.mean(0)).astype(int).tolist(),
                        Answer=ans.id,
                    )
                )
    moves += [
        dict(
            Kind=mv.kind,
            Voxels=len(mv.voxels),
            Centre=np.round(mv.voxels.mean(0) + lo).astype(int).tolist(),
        )
        for mv in r.moves
    ]
    return WMTopo(
        dseg,
        fixes,
        hemi,
        before,
        after,
        {int(k): int(c) for k, c in zip(new, cnt, strict=True)},
        moves,
        0 if manual is None else int(manual.sum()),
    )


def _write_tsv(path: Path, names: dict[int, str]) -> None:
    rows = "".join(f"{k}\t{v}\n" for k, v in sorted(names.items()))
    path.write_text("index\tname\n" + rows)


def write_wm_topo(aseg: Path, out: Path, edits: Path | None = None) -> dict[str, Path]:
    """The dseg command: wm_topo of the aseg at aseg, with the manual edits in
    the table at edits (docs/formats/wm-edits.md) if given, written to the BIDS
    derivatives root out (docs/formats/wm-dseg.md). Raises ValueError for edits
    made against another aseg, or that do not fit it. Returns the paths
    written, by role."""
    img = nib.load(aseg)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{aseg.name}: not a NIfTI-1 image")
    ents = entities(aseg)
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    e = None
    if edits is not None:
        e, side = read_edits(edits)
        if not sources_match(side, aseg):
            raise ValueError(f"{edits.name}: the edits are not against {aseg.name}")
    lap = Laps()
    a = np.asarray(img.dataobj)
    lap("read")
    t = wm_topo(a, vox, e)
    lap = Laps()
    dataset_description(out)
    p = dict(
        dseg=derivative_path(out, ents, "dseg", ".nii.gz", desc="wmtopo"),
        fixes=derivative_path(out, ents, "dseg", ".nii.gz", desc="wmtopofixes"),
    )
    p["dseg"].parent.mkdir(parents=True, exist_ok=True)
    hdr = img.header.copy()
    hdr.set_data_dtype(t.dseg.dtype)
    nib.save(nib.nifti1.Nifti1Image(t.dseg, img.affine, hdr), p["dseg"])
    nib.save(nib.nifti1.Nifti1Image(t.fixes, img.affine), p["fixes"])
    present = np.unique(t.dseg).tolist()
    _write_tsv(
        p["dseg"].with_name(p["dseg"].name.replace(".nii.gz", ".tsv")),
        {k: LUT_NAMES.get(k, f"label-{k}") for k in present},
    )
    _write_tsv(p["fixes"].with_name(p["fixes"].name.replace(".nii.gz", ".tsv")), FIXES)
    lap("write")
    src = sources(aseg) if edits is None else sources(aseg, edits, sidecar_path(edits))
    prov = provenance("baillarger.white.write_wm_topo")
    write_sidecar(
        sidecar_path(p["dseg"]),
        DSEG_SPEC,
        dict(
            SpecVersion=DSEG_SPEC.version,
            Description="The segmentation with its WM mask (FilledLabels) made "
            "cavity- and handle-free.",
            **src,
            Provenance=prov,
            Hemisphere=t.hemi,
            FilledLabels=list(FILLED_LABELS[t.hemi]),
            WMLabel=WM_LABEL[t.hemi],
            BettiBefore=list(t.betti_before),
            BettiAfter=list(t.betti_after),
            CavityVoxels=int((t.fixes == 3).sum()),
            HandleFillVoxels=int((t.fixes == 2).sum()),
            HandleCutVoxels=int((t.fixes == 1).sum()),
            CutLabels={str(k): v for k, v in t.cut_labels.items()},
            ManualVoxels=t.manual_voxels,
            Moves=t.moves,
        ),
    )
    write_sidecar(
        sidecar_path(p["fixes"]),
        FIXES_SPEC,
        dict(
            SpecVersion=FIXES_SPEC.version,
            Description="The voxels that the WM topology correction changed: "
            "1 handle cut, 2 handle fill, 3 cavity fill, 4 manual edit.",
            **src,
            Provenance=prov,
        ),
    )
    return p


def filled_labels(dseg: Path) -> tuple[int, ...]:
    """The FilledLabels of a corrected segmentation (write_wm_topo), from its
    sidecar: its voxels with one of these labels are the corrected WM mask."""
    side = read_sidecar(sidecar_path(dseg), DSEG_SPEC)
    return tuple(int(k) for k in side["FilledLabels"])


# ---------------------------------------------------------------- the mesh


def white_mesh(wm: Bool, sigma: float = 2.0) -> tuple[Float, Int]:
    """The white mesh of a WM label: vertices (voxel coordinates) and faces
    oriented out of the WM. Marching tetrahedra at 0.5 of the binary label,
    then relax and twice (sigma in voxels)."""
    v0, f = marching_tetrahedra(np.asarray(wm, np.float64), 0.5)
    f = f[:, [0, 2, 1]].copy()  # marching_tetrahedra points into the WM
    if not len(f):
        return v0, f
    return S.twice(S.relax(v0, f), f, sigma), f


def white_mesh_normals(wm: Bool, sigma: float = 2.0) -> tuple[Float, Int, Float]:
    """white_mesh and its smoothed unit vertex normals, out of the WM
    (baillarger.mesh_smoothing.smooth_normals)."""
    v, f = white_mesh(wm, sigma)
    return v, f, S.smooth_normals(v, f)


def mesh_pieces(
    v: npt.NDArray[np.floating[Any]], f: Int, voxel_mm: tuple[float, float, float]
) -> list[dict[str, Any]]:
    """Per connected piece, largest first (by vertices): V, F, boundary loops,
    chi, genus and area in mm^2."""
    _, flab = S.pieces(f, len(v))
    vox = np.asarray(voxel_mm, float)
    out = []
    for p in range(int(flab.max()) + 1 if len(f) else 0):
        fp = f[flab == p]
        t = topology(fp, len(v))
        e1 = (v[fp[:, 1]] - v[fp[:, 0]]) * vox
        e2 = (v[fp[:, 2]] - v[fp[:, 0]]) * vox
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()
        out.append(
            dict(
                V=int(np.unique(fp).size),
                F=int(len(fp)),
                boundary_loops=int(t["boundary_loops"]),
                chi=int(t["chi"]),
                genus=int(t["genus"]),
                area_mm2=round(float(area), 3),
            )
        )
    return sorted(out, key=lambda t: -t["V"])


# ------------------------------------------------- the whole-hemisphere surface

#: docs/formats/white-surf.md
SURF_SPEC = SidecarSpec(
    "white-surf/1",
    COMMON_KEYS
    + (
        Key("Hemisphere", "string", True, "R or L"),
        Key("FilledLabels", "array of integers", True, "labels of the WM mask"),
        Key("BlockCells", "integer", True, "block size (cells a side)"),
        Key("VoxelToRAS", "array of numbers", True, "4 x 4 affine, row-major"),
        Key("Vertices", "integer", True, "number of vertices"),
        Key("Faces", "integer", True, "number of faces"),
        Key("Blocks", "integer", True, "rows of the block index"),
        Key("Pieces", "array of objects", True, "per connected piece"),
        Key("NormalsFile", "string", True, "the normals' GIfTI (name)"),
        Key("EdgeKeysFile", "string", True, "the vertices' edge keys (name)"),
        Key("GridShape", "array of integers", True, "the segmentation's grid"),
        Key("BlocksFile", "string", True, "the block index (name)"),
        Key("DataSHA256", "object", True, "SHA-256 of each .dat, by name"),
        Key("Antialias", "object", True, "how the field was solved"),
        Key("Device", "string", True, "where the mesh steps ran"),
    ),
)

#: columns of the block index
BLOCK_COLUMNS = (
    "block_i",
    "block_j",
    "block_k",
    "vertex_start",
    "vertex_stop",
    "face_start",
    "face_stop",
    "min_i",
    "min_j",
    "min_k",
    "max_i",
    "max_j",
    "max_k",
)

_STRUCTURE = {"R": "CortexRight", "L": "CortexLeft"}


def surface_paths(out: Path, ents: dict[str, str], hemi: str) -> dict[str, Path]:
    """The files of a white surface in the derivatives root out."""
    return dict(
        surf=derivative_path(out, ents, "white", ".surf.gii", hemi=hemi),
        normals=derivative_path(
            out, ents, "white", ".func.gii", hemi=hemi, desc="normals"
        ),
        keys=derivative_path(
            out, ents, "white", ".func.gii", hemi=hemi, desc="edgekeys"
        ),
        blocks=derivative_path(out, ents, "white", ".tsv", hemi=hemi, desc="blocks"),
        sidecar=derivative_path(out, ents, "white", ".json", hemi=hemi),
    )


def mm_to_voxel(mm: npt.ArrayLike, affine: npt.ArrayLike) -> Float:
    """Points in scanner mm to voxel coordinates of the grid (float32), as
    WhiteSurface.load computes them: the inverse affine in float64, then
    rounded."""
    inv = np.linalg.inv(np.asarray(affine, np.float64))
    out: Float = (np.asarray(mm, np.float64) @ inv[:3, :3].T + inv[:3, 3]).astype(
        np.float32
    )
    return out


def block_bounds(v: Float, f: Int, blocks: MeshBlocks) -> Float:
    """(k, 6): per block, the bounding box (min, max) of the vertices its
    faces use."""
    out = np.zeros((len(blocks.origin), 6), np.float64)
    if not len(f):
        return out
    p = v[f]  # (m, 3, 3)
    lo, hi = p.min(1), p.max(1)
    out[:, :3] = np.minimum.reduceat(lo, blocks.face_start, axis=0)
    out[:, 3:] = np.maximum.reduceat(hi, blocks.face_start, axis=0)
    return out


def write_white_surface(
    dseg: Path,
    out: Path,
    block: int = BLOCK,
    device: str | None = None,
) -> "SurfaceFiles":
    """The mesh command: the white mesh of the WM mask of a corrected segmentation
    (write_wm_topo), as GIfTI with external binary data, the normals, the block index
    and the sidecar (docs/formats/white-surf.md). The surface is marching tetrahedra of
    the antialiased field of the mask (baillarger.antialias): the binary mesh's
    triangles, each vertex moved along its own grid edge, so no self-intersections by
    construction. Raises ValueError if the mask has a handle or a cavity. Returns the
    paths written."""
    dev = backend.device(device)
    log.info(f"device: {dev.name}")
    lap = Laps(dev)
    side = read_sidecar(sidecar_path(dseg), DSEG_SPEC)
    hemi, filled = side["Hemisphere"], tuple(side["FilledLabels"])
    vol = open_volume(dseg)
    img = vol.img
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    mask = np.isin(vol.read(tuple(slice(0, n) for n in vol.shape)), filled)
    nz = [
        np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != i)))
        for i in range(3)
    ]
    if not all(len(t) for t in nz):
        raise ValueError(f"{dseg.name}: the WM mask is empty")
    crop = tuple(slice(int(t[0]), int(t[-1]) + 1) for t in nz)
    lap("read, mask")
    b = kuhn_betti(mask[crop])
    lap("Betti check")
    if b[1] or b[2]:
        raise ValueError(
            f"{dseg.name}: the WM mask has Betti numbers {b}, not (b0, 0, 0)"
        )
    # the antialiased field on the mask's box plus a margin past the band
    m = int(np.ceil(AA.BAND)) + 4
    lo = np.array([max(c.start - m, 0) for c in crop])
    hi = np.array([min(c.stop + m, n) for c, n in zip(crop, mask.shape, strict=True)])
    sub = np.ascontiguousarray(
        mask[tuple(slice(p, q) for p, q in zip(lo, hi, strict=True))]
    )
    shape = mask.shape
    del mask
    phi, solve = AA.thin_plate(sub, dev)
    del sub
    if not solve.converged:
        log.warning(
            f"antialias: not converged after {solve.iterations} iterations "
            f"(last change {solve.last_change:.1e})"
        )
    lap("antialias")
    # marching on the whole grid, so blocks and edge keys follow it; outside the box
    # the field is outside the mask
    field = EmbeddedSource(phi, lo, shape, -(AA.BAND + 3.0))
    v64, f, blocks = marching_tetrahedra_blocks(field, 0.0, block, np.float32, dev)
    v: F32 = v64.astype(np.float32, copy=False)  # already float32: typing only
    del phi, field, v64
    f = f[:, [0, 2, 1]].copy()  # marching_tetrahedra points into the WM
    lap("marching")
    aff = np.asarray(img.affine, np.float64)

    def as_written(x: npt.NDArray[np.floating[Any]]) -> Float:
        out: Float = (x @ aff[:3, :3].T + aff[:3, 3]).astype("<f4").astype(np.float64)
        return out

    if dev.torch is None:
        n = S.smooth_normals(v, f)
    else:
        n = MeshOps(f, len(v), dev.torch).normals(v)
    lap("normals")
    v_mm = as_written(v).astype("<f4")
    paths = surface_paths(out, entities(dseg), hemi)
    dataset_description(out)
    surf_meta = dict(
        AnatomicalStructurePrimary=_STRUCTURE[hemi],
        AnatomicalStructureSecondary="GrayWhite",
        GeometricType="Anatomical",
    )
    dat = write_external(
        paths["surf"],
        [
            DataArray(v_mm, "NIFTI_INTENT_POINTSET", surf_meta, np.eye(4)),
            DataArray(f.astype("<i4"), "NIFTI_INTENT_TRIANGLE"),
        ],
        dict(AnatomicalStructurePrimary=_STRUCTURE[hemi]),
    )
    ndat = write_external(
        paths["normals"],
        [DataArray(n.astype("<f4"), "NIFTI_INTENT_VECTOR", dict(Name="normals"))],
        dict(AnatomicalStructurePrimary=_STRUCTURE[hemi]),
    )
    # int64 keys as (low, high) int32 words: GIfTI has no 64-bit integers
    kdat = write_external(
        paths["keys"],
        [
            DataArray(
                np.ascontiguousarray(blocks.vertex_key, "<i8")
                .view("<i4")
                .reshape(-1, 2),
                "NIFTI_INTENT_NONE",
                dict(Name="edge keys, int64 as (low, high) int32 words"),
            )
        ],
        dict(AnatomicalStructurePrimary=_STRUCTURE[hemi]),
    )
    lap("write GIfTI")
    # the boxes hold the vertices as a reader gets them back from the file
    bb = block_bounds(mm_to_voxel(v_mm, aff), f, blocks)
    bb[:, :3] = np.floor(bb[:, :3].astype(np.float64) * 1000) / 1000
    bb[:, 3:] = np.ceil(bb[:, 3:].astype(np.float64) * 1000) / 1000
    rows = np.c_[
        blocks.origin // block,
        blocks.vertex_start,
        blocks.vertex_stop,
        blocks.face_start,
        blocks.face_stop,
    ]
    with paths["blocks"].open("w") as fh:
        fh.write("\t".join(BLOCK_COLUMNS) + "\n")
        for r, q in zip(rows, bb, strict=True):
            ints = "\t".join(str(int(x)) for x in r)
            fh.write(ints + "\t" + "\t".join(f"{x:.3f}" for x in q) + "\n")
    lap("block index")
    pieces = mesh_pieces(v, f, (vox[0], vox[1], vox[2]))
    lap("pieces")
    write_sidecar(
        paths["sidecar"],
        SURF_SPEC,
        dict(
            SpecVersion=SURF_SPEC.version,
            Description="The white (WM) surface of the corrected segmentation's "
            "WM mask (FilledLabels), smoothed.",
            **sources(dseg),
            Provenance=provenance("baillarger.white.write_white_surface"),
            Hemisphere=hemi,
            FilledLabels=list(filled),
            BlockCells=int(block),
            VoxelToRAS=aff.ravel().tolist(),
            Vertices=len(v),
            Faces=len(f),
            Blocks=len(rows),
            Pieces=pieces,
            NormalsFile=paths["normals"].name,
            EdgeKeysFile=paths["keys"].name,
            GridShape=[int(x) for x in img.shape[:3]],
            BlocksFile=paths["blocks"].name,
            DataSHA256={d.name: file_sha256(d) for d in (dat, ndat, kdat)},
            Antialias=dict(
                Method="thin-plate",
                Band=AA.BAND,
                Eps=AA.EPS,
                Tolerance=AA.TOL,
                **solve.summary(),
            ),
            Device=dev.name,
        ),
    )
    lap("sidecar (hashes)")
    return SurfaceFiles(dict(paths, dat=dat, normals_dat=ndat, keys_dat=kdat))


@dataclass(frozen=True)
class SurfaceFiles:
    """What write_white_surface wrote: the paths, by role."""

    paths: dict[str, Path]


@dataclass(frozen=True)
class WhiteSurface:
    """A white surface read from its files (write_white_surface), whole or
    one block: vertices (float32) in voxel coordinates of the
    segmentation's grid, faces into them, normals (float32, out of the WM),
    vertex_ids, the vertices' indices in the whole mesh (for one block: the
    vertices its faces use), and vertex_key, their grid-edge keys
    (baillarger.mesh.MeshBlocks.vertex_key), which survive a remesh where
    the surface did not change."""

    v: Float
    f: Int
    normals: Float
    vertex_ids: Int
    vertex_key: Int
    affine: Float
    sidecar: dict[str, Any]

    @classmethod
    def load(
        cls, surf: Path, block: tuple[int, int, int] | None = None
    ) -> "WhiteSurface":
        """Read the .surf.gii at surf (and its sidecar, normals and block
        index), memory-mapped; block: (i, j, k) of one block (a block the
        surface does not cross gives an empty mesh)."""
        side = read_sidecar(sidecar_path(surf), SURF_SPEC)
        aff = np.asarray(side["VoxelToRAS"], np.float64).reshape(4, 4)
        g = GiftiImage.from_filename(str(surf), mmap=True)
        pts, tri = g.agg_data(("pointset", "triangle"))
        ng = GiftiImage.from_filename(
            str(surf.with_name(side["NormalsFile"])), mmap=True
        )
        nrm = ng.darrays[0].data
        kg = GiftiImage.from_filename(
            str(surf.with_name(side["EdgeKeysFile"])), mmap=True
        )
        words = kg.darrays[0].data  # (n, 2) int32: low, high
        if block is None:
            f = np.asarray(tri, np.int64)
            ids = np.arange(len(pts))
        else:
            idx = np.loadtxt(surf.with_name(side["BlocksFile"]), skiprows=1, ndmin=2)
            hit = np.flatnonzero(np.all(idx[:, :3] == np.asarray(block), axis=1))
            if not len(hit):
                f = np.zeros((0, 3), np.int64)
            else:
                fs, fe = int(idx[hit[0], 5]), int(idx[hit[0], 6])
                f = np.asarray(tri[fs:fe], np.int64)
            ids, f = np.unique(f, return_inverse=True)
            f = f.reshape(-1, 3)
        v = mm_to_voxel(pts[ids], aff)
        key = np.ascontiguousarray(words[ids], "<i4").view("<i8").ravel()
        return cls(
            v, f, np.asarray(nrm[ids], np.float32), ids, key.astype(np.int64), aff, side
        )

    def check(self, dseg: Path) -> None:
        """Raises ValueError unless dseg is the segmentation the surface was
        made from (its SHA-256)."""
        if file_sha256(dseg) not in self.sidecar["SourcesSHA256"].values():
            raise ValueError(f"{dseg.name} is not the surface's segmentation")


# ------------------------------------------------- the surface's overview

#: the overview's default distance bound (mm) and decimation block (mm)
OVERVIEW_EPS_MM = 0.1
OVERVIEW_BLOCK_MM = 10.0

#: docs/formats/white-overview.md
OVERVIEW_SPEC = SidecarSpec(
    "white-overview/1",
    COMMON_KEYS
    + (
        Key("Hemisphere", "string", True, "R or L"),
        Key("VoxelToRAS", "array of numbers", True, "4 x 4 affine, row-major"),
        Key("EpsMM", "number", True, "the distance bound (mm)"),
        Key("BlockMM", "number", True, "the decimation's block side (mm)"),
        Key("MaxDistanceMM", "number", True, "largest distance measured (mm)"),
        Key("Vertices", "integer", True, "number of vertices"),
        Key("Faces", "integer", True, "number of faces"),
        Key("TrueVertices", "integer", True, "the white surface's vertices"),
        Key("TrueFaces", "integer", True, "the white surface's faces"),
        Key("Pieces", "array of objects", True, "per connected piece"),
        Key("IdsFile", "string", True, "the true vertex ids' GIfTI (name)"),
        Key("Repairs", "array of objects", True, "per repair round"),
        Key("CarryFile", "string", True, "the carried vertices' GIfTI (name)"),
        Key("Carried", "integer", True, "true vertices carried by faces"),
        Key("DataSHA256", "object", True, "SHA-256 of the carry .dat, by name"),
        Key("Passes", "array of objects", True, "per decimation pass"),
    ),
)


def overview_paths(surf: Path) -> dict[str, Path]:
    """The overview's files next to a white surface's .surf.gii."""
    stem = surf.name.removesuffix("_white.surf.gii")
    return dict(
        overview=surf.with_name(f"{stem}_desc-overview_white.surf.gii"),
        ids=surf.with_name(f"{stem}_desc-overviewids_white.func.gii"),
        carry=surf.with_name(f"{stem}_desc-overviewcarry_white.func.gii"),
        sidecar=surf.with_name(f"{stem}_desc-overview_white.json"),
    )


def _embedded(arrays: list[tuple[npt.NDArray[Any], str]], hemi: str) -> GiftiImage:
    g = GiftiImage(meta=GiftiMetaData(AnatomicalStructurePrimary=_STRUCTURE[hemi]))
    for a, intent in arrays:
        g.add_gifti_data_array(
            GiftiDataArray(a, intent=intent, encoding="GIFTI_ENCODING_B64GZ")
        )
    return g


def write_white_overview(
    surf: Path,
    eps_mm: float = OVERVIEW_EPS_MM,
    block_mm: float = OVERVIEW_BLOCK_MM,
    workers: int | None = None,
) -> SurfaceFiles:
    """The overview of the white surface at surf (write_white_surface): the
    surface decimated (baillarger.decimate) so that it keeps its topology,
    every true vertex is within eps_mm of it and its vertices are true
    vertices; written next to surf as an ordinary GIfTI with embedded data,
    with the true vertex of each overview vertex, the true vertices each face
    carries (external binary data) and a sidecar
    (docs/formats/white-overview.md). The faces are sorted by their true
    vertex ids, smallest first, so that faces near each other are near each
    other in the files. Returns the paths written."""
    lap = Laps()
    side = read_sidecar(sidecar_path(surf), SURF_SPEC)
    hemi = side["Hemisphere"]
    aff = np.asarray(side["VoxelToRAS"], np.float64).reshape(4, 4)
    g = GiftiImage.from_filename(str(surf), mmap=True)
    pts, tri = g.agg_data(("pointset", "triangle"))
    v = np.asarray(pts, np.float64)
    f = np.asarray(tri, np.int64)
    lap("read the surface")
    D = decimate(v, f, eps=eps_mm, block=block_mm, workers=workers, log=log.info)
    fs = np.sort(D.faces, 1)  # vertex order = true id order
    D = reorder(D, np.lexsort((fs[:, 2], fs[:, 1], fs[:, 0])))
    lap("decimate")
    paths = overview_paths(surf)
    _embedded(
        [
            (np.asarray(pts[D.vertex_ids], "<f4"), "NIFTI_INTENT_POINTSET"),
            (D.faces.astype("<i4"), "NIFTI_INTENT_TRIANGLE"),
        ],
        hemi,
    ).to_filename(str(paths["overview"]))
    _embedded([(D.vertex_ids.astype("<i4"), "NIFTI_INTENT_NONE")], hemi).to_filename(
        str(paths["ids"])
    )
    cdat = write_external(
        paths["carry"],
        [
            DataArray(
                D.cptr.astype("<i4")[:, None],
                "NIFTI_INTENT_NONE",
                dict(Name="per face, where its carried true vertices start"),
            ),
            DataArray(
                D.cids.astype("<i4")[:, None],
                "NIFTI_INTENT_NONE",
                dict(Name="carried true vertices, by face"),
            ),
        ],
        dict(AnatomicalStructurePrimary=_STRUCTURE[hemi]),
    )
    vox = np.linalg.norm(aff[:3, :3], axis=0)
    pieces = mesh_pieces(
        mm_to_voxel(pts[D.vertex_ids], aff), D.faces, (vox[0], vox[1], vox[2])
    )
    dat = surf.with_name(surf.name.removesuffix(".surf.gii") + ".dat")
    write_sidecar(
        paths["sidecar"],
        OVERVIEW_SPEC,
        dict(
            SpecVersion=OVERVIEW_SPEC.version,
            Description="The white surface decimated for viewing: its topology, "
            "every true vertex within EpsMM of it, its vertices true vertices.",
            **sources(surf, dat),
            Provenance=provenance("baillarger.white.write_white_overview"),
            Hemisphere=hemi,
            VoxelToRAS=aff.ravel().tolist(),
            EpsMM=float(eps_mm),
            BlockMM=float(block_mm),
            MaxDistanceMM=float(D.max_distance),
            Vertices=len(D.vertex_ids),
            Faces=len(D.faces),
            TrueVertices=len(v),
            TrueFaces=len(f),
            Pieces=pieces,
            IdsFile=paths["ids"].name,
            Repairs=D.repairs,
            CarryFile=paths["carry"].name,
            Carried=len(D.cids),
            DataSHA256={cdat.name: file_sha256(cdat)},
            Passes=D.passes,
        ),
    )
    lap("write")
    return SurfaceFiles(dict(paths, carry_dat=cdat))


def surface_points(surf: Path) -> npt.NDArray[np.float32]:
    """The white surface's vertices as its file holds them (scanner mm,
    float32), memory-mapped."""
    pts: npt.NDArray[np.float32] = GiftiImage.from_filename(
        str(surf), mmap=True
    ).agg_data("pointset")
    return pts


@dataclass(frozen=True)
class WhiteOverview:
    """A white surface's overview read from its files (write_white_overview):
    vertices (float32) in voxel coordinates of the grid, as WhiteSurface.load
    gives the same vertices; faces into them; vertex_ids, the true vertex of
    each; and per face the true vertices it carries: they start at ptr[k]
    in carried (memory-mapped)."""

    v: Float
    f: Int
    vertex_ids: Int
    affine: Float
    sidecar: dict[str, Any]
    ptr: Int
    carried_ids: npt.NDArray[np.int32]

    @classmethod
    def load(cls, path: Path) -> "WhiteOverview":
        side = read_sidecar(sidecar_path(path), OVERVIEW_SPEC)
        aff = np.asarray(side["VoxelToRAS"], np.float64).reshape(4, 4)
        pts, tri = GiftiImage.from_filename(str(path)).agg_data(
            ("pointset", "triangle")
        )
        ids = GiftiImage.from_filename(str(path.with_name(side["IdsFile"])))
        cg = GiftiImage.from_filename(str(path.with_name(side["CarryFile"])), mmap=True)
        return cls(
            mm_to_voxel(pts, aff),
            np.asarray(tri, np.int64),
            np.asarray(ids.darrays[0].data, np.int64),
            aff,
            side,
            np.asarray(cg.darrays[0].data, np.int64).ravel(),
            cg.darrays[1].data.reshape(-1),
        )

    def carried(self, faces: npt.ArrayLike) -> tuple[Int, Int]:
        """The true vertices that the overview faces `faces` carry, and the face
        (of `faces`) that carries each; only their part of the file is read."""
        fa = np.asarray(faces, np.int64).ravel()
        cptr, ids = csr_take(self.ptr, self.carried_ids, fa)
        return ids.astype(np.int64), np.repeat(fa, np.diff(cptr))

    def carry(
        self, faces: npt.ArrayLike, pts: npt.NDArray[np.floating[Any]]
    ) -> tuple[Int, Int, Float, Float, Float, Float]:
        """For the true vertices that `faces` carry: their ids, carrying faces,
        the barycentric coordinates (b0, b1, b2) of their closest point on that
        face and the distance to it (mm). pts: the white surface's vertices as
        its file holds them (mm, float32, surface_points), the coordinates the
        decimation used, so the distances are its own."""
        ids, fc = self.carried(faces)
        t = self.vertex_ids[self.f[fc]]
        x = np.asarray(pts[ids], np.float64)
        a, b, c = (np.asarray(pts[t[:, k]], np.float64) for k in range(3))
        d, b0, b1, b2 = closest_points(x, a, b, c)
        return ids, fc, b0, b1, b2, d

    def check(self, surf: WhiteSurface) -> None:
        """Raises ValueError unless the overview was made from surf's data (its
        .dat's SHA-256, from the two sidecars: no hashing)."""
        sha = [
            v
            for k, v in surf.sidecar["DataSHA256"].items()
            if k.endswith("_white.dat") and "_desc-" not in k
        ]
        if not sha or sha[0] not in self.sidecar["SourcesSHA256"].values():
            raise ValueError("the overview was not made from this surface's data")


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    log_to_stderr()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser(
        "dseg", help="the segmentation with a cavity- and handle-free WM mask"
    )
    d.add_argument("aseg", type=Path, help="a BIDS-named segmentation (NIfTI)")
    d.add_argument(
        "--edits", type=Path, help="manual edits of the aseg (desc-wmedits_edits.tsv)"
    )
    d.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    m = sub.add_parser("mesh", help="the white surface of a corrected segmentation")
    m.add_argument("dseg", type=Path, help="a desc-wmtopo_dseg (the dseg command)")
    m.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    m.add_argument("--block-cells", type=int, default=BLOCK, help="block size")
    m.add_argument(
        "--device",
        choices=backend.NAMES,
        help="where the mesh steps run (default: $BAILLARGER_DEVICE, else auto)",
    )
    m.add_argument(
        "--overview-eps",
        type=float,
        default=OVERVIEW_EPS_MM,
        help="the overview's distance bound (mm)",
    )
    m.add_argument("--workers", type=int, default=4, help="processes to decimate")
    o = sub.add_parser("overview", help="the overview of a surface that mesh wrote")
    o.add_argument("surf", type=Path, help="a hemi-<h>_white.surf.gii (mesh)")
    o.add_argument("--eps", type=float, default=OVERVIEW_EPS_MM, help="bound (mm)")
    o.add_argument("--workers", type=int, default=4, help="processes to decimate")
    a = ap.parse_args(argv)
    if a.cmd == "dseg":
        for role, path in write_wm_topo(a.aseg, a.out, a.edits).items():
            print(f"{role}: {path}")
    elif a.cmd == "mesh":
        w = write_white_surface(a.dseg, a.out, a.block_cells, a.device)
        ov = write_white_overview(w.paths["surf"], a.overview_eps, workers=a.workers)
        for role, path in w.paths.items():
            print(f"{role}: {path}")
        for role, path in ov.paths.items():
            print(f"overview {role}: {path}")
    else:
        ov = write_white_overview(a.surf, a.eps, workers=a.workers)
        for role, path in ov.paths.items():
            print(f"overview {role}: {path}")


if __name__ == "__main__":
    main()
