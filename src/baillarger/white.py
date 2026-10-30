"""The handle-free WM label: the aseg's WM label (41) with every handle
removed, so that the white mesh of any box has genus-0 pieces.

The correction is made once, on the whole label (baillarger.handles,
remove_handles), because a box crop of a handle-free solid has planar surface
pieces, while a box alone cannot tell a handle from a fold that closes outside
it. wm_changes computes it (for the example hemisphere: a few minutes and
about 20 GB) and returns only the changed voxels, in full-grid coordinates.
Save them once (WMChanges.save); white_matter_box then gives the corrected
label of any box from the aseg and the saved changes.

white_mesh makes the white mesh of a label: marching tetrahedra at 0.5 of
the binary label (so the mesh has the label's Kuhn topology, which
baillarger.handles makes handle-free), then tangential relaxation and
Gaussian smoothing without shrinkage (baillarger.mesh_smoothing), which move
vertices but keep the faces. white_mesh_normals adds the smoothed vertex
normals (out of the WM) that the column traces start from.

A box's white mesh can be made once and saved (WhiteMesh: box_mesh, save),
so that a series of experiments uses one surface. The file holds the box,
its affine, sigma, the SHA-256 of the WM label the mesh was made from and the
provenance of that label; WhiteMesh.check refuses a box or a WM label that
is not the mesh's own (baillarger.column_traces.trace_columns, mesh=).

The reports: changes_summary (the moves on the full grid), box_report (the
white mesh of a box before and after, per piece, and the moves in it) and
overlay (an ITK-SNAP label image of the moves). The command line:

    python -m baillarger.white changes --out OUTDIR
    python -m baillarger.white box --changes OUTDIR/wm_changes.npz --out OUTDIR
        --name NAME --lo X Y Z --hi X Y Z
    python -m baillarger.white mesh [--changes OUTDIR/wm_changes.npz]
        --out OUTDIR --name NAME (--patch P | --lo X Y Z --hi X Y Z)
        [--sigma S]

changes writes wm_changes.npz (the changes), wm_changes.json
(changes_summary) and wm_changes_overlay.nii.gz (full grid: 1 cut, 2 fill,
3 stranded). box writes wm_box_<NAME>.json (box_report),
wm_box_<NAME>_overlay.nii.gz and wm_box_<NAME>_wm_fixed.nii.gz (the box's
handle-free label), on the box grid. mesh writes wm_mesh_<NAME>.npz (the
WhiteMesh, the file that other tools load), wm_mesh_<NAME>.json (the box,
mesh_pieces and the provenance) and wm_mesh_<NAME>.vtk (the mesh in RAS mm,
for viewing); without --changes, the mesh is that of the aseg's WM label.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.gifti.gifti import GiftiImage
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
from baillarger.data import ASEG_WM, Box, aseg_path, crop_affine
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

type Bool = npt.NDArray[np.bool_]
type Betti = tuple[int, int, int]
type Float = npt.NDArray[np.float64]
type F32 = npt.NDArray[np.float32]
type Int = npt.NDArray[np.int64]

#: kinds of change: WM voxels removed (a cut), voxels added (a fill), WM voxels
#: removed that no front could claim (stranded: each would make a handle)
CUT, FILL, STRANDED = 1, 2, 3
KINDS = {"cut": CUT, "fill": FILL, "stranded": STRANDED}


@dataclass(frozen=True)
class WMChanges:
    """The changes that make the WM label handle-free."""

    #: (n, 3) int32, voxels of the full grid
    voxels: npt.NDArray[np.int32]
    #: (n,) int8, CUT, FILL or STRANDED
    kind: npt.NDArray[np.int8]
    #: (n,) int32, the move (one handle, or the stranded voxels) of each voxel
    move: npt.NDArray[np.int32]
    #: (b0, b1, b2) of the WM label before and after (Kuhn adjacency)
    betti_before: Betti
    betti_after: Betti

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            voxels=self.voxels,
            kind=self.kind,
            move=self.move,
            betti_before=np.array(self.betti_before),
            betti_after=np.array(self.betti_after),
        )

    @classmethod
    def load(cls, path: Path) -> "WMChanges":
        z = np.load(path)
        b0, b1 = z["betti_before"].tolist(), z["betti_after"].tolist()
        return cls(
            voxels=z["voxels"].astype(np.int32),
            kind=z["kind"].astype(np.int8),
            move=z["move"].astype(np.int32),
            betti_before=(b0[0], b0[1], b0[2]),
            betti_after=(b1[0], b1[1], b1[2]),
        )


def _aseg(directory: Path | None) -> nib.nifti1.Nifti1Image:
    img = nib.load(aseg_path(directory))
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{aseg_path(directory).name}: not a NIfTI-1 image")
    return img


def wm_changes(directory: Path | None = None) -> WMChanges:
    """Remove the handles of the aseg's WM label, on its bounding box with a
    margin of 2 voxels, and return the changes."""
    img = _aseg(directory)
    wm = np.asarray(img.dataobj) == ASEG_WM
    nz = [
        np.flatnonzero(wm.any(axis=tuple(j for j in range(3) if j != i)))
        for i in range(3)
    ]
    lo = np.array([max(int(t[0]) - 2, 0) for t in nz])
    hi = np.array([min(int(t[-1]) + 3, s) for t, s in zip(nz, wm.shape, strict=True)])
    x = wm[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    del wm
    before = kuhn_betti(x)
    r = remove_handles(x)
    after = kuhn_betti(r.obj)
    n = [len(m.voxels) for m in r.moves]
    if r.moves:
        voxels = np.concatenate([m.voxels for m in r.moves]) + lo
    else:
        voxels = np.zeros((0, 3), np.int64)
    kind = np.repeat([KINDS[m.kind] for m in r.moves], n)
    move = np.repeat(np.arange(len(r.moves)), n)
    return WMChanges(
        voxels.astype(np.int32),
        kind.astype(np.int8),
        move.astype(np.int32),
        before,
        after,
    )


def white_matter_box(
    lo: Box, hi: Box, changes: WMChanges, directory: Path | None = None
) -> Bool:
    """The handle-free WM label of the box [lo, hi) of the full grid (bool).
    Raises ValueError if the box is not inside the grid."""
    img = _aseg(directory)
    if not all(0 <= a < b <= n for a, b, n in zip(lo, hi, img.shape, strict=True)):
        raise ValueError(f"box {lo}-{hi} is not inside the grid {img.shape}")
    sl = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    wm: Bool = np.asarray(img.dataobj[sl]) == ASEG_WM
    p = changes.voxels.astype(np.int64) - np.asarray(lo)
    inside = np.all((p >= 0) & (p < np.subtract(hi, lo)), axis=1)
    p, k = p[inside], changes.kind[inside]
    wm[tuple(p[k == FILL].T)] = True
    wm[tuple(p[k != FILL].T)] = False
    return wm


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

#: values of the fixes image: a handle cut, a handle fill, a cavity fill
FIXES = {1: "handle_cut", 2: "handle_fill", 3: "cavity_fill"}

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
        Key("Moves", "array of objects", True, "each handle move"),
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
    #: per handle move: Kind, Voxels, Centre (0-based voxel of the grid)
    moves: list[dict[str, Any]]


def wm_topo(aseg: npt.ArrayLike, voxel_mm: npt.ArrayLike) -> WMTopo:
    """The corrected segmentation of an aseg: the mask of its FILLED_LABELS,
    with its cavities filled and its handles removed
    (baillarger.handles.remove_handles). Cavity and handle fills become the
    WM label; each handle cut takes the label of the nearest voxel that is
    outside the corrected mask and not a cut. The corrected segmentation's
    voxels with a filled label are exactly the corrected mask."""
    lap = Laps()
    a0 = np.asarray(aseg)
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
    dseg = a0.copy()
    dseg[box] = labels
    fixes = np.zeros(a0.shape, np.uint8)
    fx = fixes[box]
    fx[cut] = 1
    fx[fill] = 2
    fx[cav] = 3
    new, cnt = np.unique(labels[cut], return_counts=True)
    moves = [
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
    )


def _write_tsv(path: Path, names: dict[int, str]) -> None:
    rows = "".join(f"{k}\t{v}\n" for k, v in sorted(names.items()))
    path.write_text("index\tname\n" + rows)


def write_wm_topo(aseg: Path, out: Path) -> dict[str, Path]:
    """The dseg command: wm_topo of the aseg at aseg, written to the BIDS
    derivatives root out (docs/formats/wm-dseg.md). Returns the paths
    written, by role."""
    img = nib.load(aseg)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{aseg.name}: not a NIfTI-1 image")
    ents = entities(aseg)
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    lap = Laps()
    a = np.asarray(img.dataobj)
    lap("read")
    t = wm_topo(a, vox)
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
    src = sources(aseg)
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
            Moves=t.moves,
        ),
    )
    write_sidecar(
        sidecar_path(p["fixes"]),
        FIXES_SPEC,
        dict(
            SpecVersion=FIXES_SPEC.version,
            Description="The voxels that the WM topology correction changed: "
            "1 handle cut, 2 handle fill, 3 cavity fill.",
            **src,
            Provenance=prov,
        ),
    )
    return p


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


def label_sha256(wm: Bool) -> str:
    """SHA-256 of a binary label: its shape and its voxels in C order."""
    a = np.ascontiguousarray(np.asarray(wm, bool))
    h = hashlib.sha256(repr(a.shape).encode())
    h.update(np.packbits(a, axis=None).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class WhiteMesh:
    """The white mesh of one box, made once and loaded by the tools that use
    it."""

    #: (n, 3) vertices, voxel coordinates of the box
    v: Float
    #: (m, 3) faces, oriented out of the WM
    f: Int
    #: (n, 3) smoothed unit vertex normals, out of the WM
    normals: Float
    lo: Box
    hi: Box
    #: voxel to RAS (mm) of the box
    affine: Float
    sigma: float
    #: label_sha256 of the box's WM label that the mesh was made from
    wm_sha256: str
    #: where the WM label came from, and how and when the mesh was made
    provenance: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            v=self.v,
            f=self.f,
            normals=self.normals,
            lo=np.array(self.lo),
            hi=np.array(self.hi),
            affine=self.affine,
            sigma=np.array(self.sigma),
            wm_sha256=np.array(self.wm_sha256),
            provenance=np.array(json.dumps(self.provenance)),
        )

    @classmethod
    def load(cls, path: Path) -> "WhiteMesh":
        z = np.load(path)
        lo, hi = z["lo"].tolist(), z["hi"].tolist()
        return cls(
            v=z["v"].astype(np.float64),
            f=z["f"].astype(np.int64),
            normals=z["normals"].astype(np.float64),
            lo=(lo[0], lo[1], lo[2]),
            hi=(hi[0], hi[1], hi[2]),
            affine=z["affine"].astype(np.float64),
            sigma=float(z["sigma"]),
            wm_sha256=str(z["wm_sha256"]),
            provenance=json.loads(str(z["provenance"])),
        )

    def check(self, lo: Box, hi: Box, wm: Bool) -> None:
        """Raises ValueError unless [lo, hi) is the mesh's box and wm the WM
        label that it was made from."""
        if tuple(lo) != self.lo or tuple(hi) != self.hi:
            raise ValueError(
                f"box {tuple(lo)}-{tuple(hi)} is not the mesh's {self.lo}-{self.hi}"
            )
        if label_sha256(wm) != self.wm_sha256:
            raise ValueError("the WM label is not the one the mesh was made from")


def box_mesh(
    wm: Bool,
    lo: Box,
    hi: Box,
    affine: npt.ArrayLike,
    sigma: float = 2.0,
    provenance: dict[str, Any] | None = None,
) -> WhiteMesh:
    """The WhiteMesh of the WM label wm of the box [lo, hi)."""
    shape = tuple(int(b - a) for a, b in zip(lo, hi, strict=True))
    if wm.shape != shape:
        raise ValueError(f"WM label {wm.shape} is not on the box {shape}")
    v, f, n = white_mesh_normals(wm, sigma)
    return WhiteMesh(
        v,
        f,
        n,
        lo,
        hi,
        np.asarray(affine, np.float64),
        float(sigma),
        label_sha256(wm),
        dict(provenance or {}),
    )


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


# ---------------------------------------------------------------- reports

_NAMES = {v: k for k, v in KINDS.items()}


def changes_summary(
    changes: WMChanges, voxel_mm: tuple[float, float, float]
) -> dict[str, Any]:
    """The Betti numbers, the counts by kind, and per move its kind, voxels,
    volume, extent and centre (full grid, 0-based, and ITK-SNAP, 1-based)."""
    vmm = float(np.prod(voxel_mm))
    moves = []
    for i in np.unique(changes.move).tolist():
        sel = changes.move == i
        p = changes.voxels[sel]
        c = np.round(p.mean(0)).astype(int)
        moves.append(
            dict(
                move=int(i),
                kind=_NAMES[int(changes.kind[sel][0])],
                voxels=int(len(p)),
                volume_mm3=round(len(p) * vmm, 4),
                extent_vox=(p.max(0) - p.min(0) + 1).tolist(),
                centre_full0=c.tolist(),
                centre_itksnap=(c + 1).tolist(),
            )
        )
    return dict(
        betti_before=list(changes.betti_before),
        betti_after=list(changes.betti_after),
        voxels_removed=int((changes.kind != FILL).sum()),
        voxels_added=int((changes.kind == FILL).sum()),
        stranded=int((changes.kind == STRANDED).sum()),
        by_kind={
            k: dict(
                moves=sum(m["kind"] == k for m in moves),
                voxels=sum(m["voxels"] for m in moves if m["kind"] == k),
            )
            for k in KINDS
        },
        moves=moves,
    )


def overlay(changes: WMChanges, lo: Box, hi: Box) -> npt.NDArray[np.int16]:
    """The kind (1 cut, 2 fill, 3 stranded) of each changed voxel of the box
    [lo, hi), 0 elsewhere."""
    p = changes.voxels.astype(np.int64) - np.asarray(lo)
    shape = tuple(int(b - a) for a, b in zip(lo, hi, strict=True))
    inside = np.all((p >= 0) & (p < np.array(shape)), axis=1)
    ov = np.zeros(shape, np.int16)
    ov[tuple(p[inside].T)] = changes.kind[inside]
    return ov


def box_report(
    lo: Box,
    hi: Box,
    changes: WMChanges,
    directory: Path | None = None,
    sigma: float = 2.0,
) -> dict[str, Any]:
    """The white mesh of the box [lo, hi) before and after the changes (per
    piece: mesh_pieces), the voxels removed and added, the total areas, and
    the moves in the box: kind, voxels in the box, centre (box, 0-based;
    ITK-SNAP on the box grid, 1-based; full grid, 0-based) and the centre's
    distance to the nearest box face in mm."""
    img = _aseg(directory)
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    voxel_mm = (vox[0], vox[1], vox[2])
    sl = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    wm0: Bool = np.asarray(img.dataobj[sl]) == ASEG_WM
    wm1 = white_matter_box(lo, hi, changes, directory)
    before = mesh_pieces(*white_mesh(wm0, sigma), voxel_mm)
    after = mesh_pieces(*white_mesh(wm1, sigma), voxel_mm)
    p = changes.voxels.astype(np.int64) - np.asarray(lo)
    shape = np.array(wm0.shape)
    inside = np.all((p >= 0) & (p < shape), axis=1)
    moves = []
    for i in np.unique(changes.move[inside]).tolist():
        sel = inside & (changes.move == i)
        c = np.round(p[sel].mean(0)).astype(int)
        face = np.min(np.r_[c, shape - 1 - c] * np.r_[voxel_mm, voxel_mm])
        moves.append(
            dict(
                move=int(i),
                kind=_NAMES[int(changes.kind[sel][0])],
                voxels_in_box=int(sel.sum()),
                centre_box0=c.tolist(),
                centre_itksnap_box=(c + 1).tolist(),
                centre_full0=(c + np.asarray(lo)).tolist(),
                face_distance_mm=round(float(face), 2),
            )
        )
    return dict(
        lo=list(lo),
        hi=list(hi),
        voxels_removed=int((wm0 & ~wm1).sum()),
        voxels_added=int((wm1 & ~wm0).sum()),
        area_mm2_before=round(sum(t["area_mm2"] for t in before), 3),
        area_mm2_after=round(sum(t["area_mm2"] for t in after), 3),
        pieces_before=before,
        pieces_after=after,
        moves=moves,
    )


# ---------------------------------------------------------------- command line


def write_changes(changes: WMChanges, out: Path, stem: str = "wm_changes") -> None:
    """<stem>.npz, <stem>.json (changes_summary) and <stem>_overlay.nii.gz
    (full grid) in out."""
    out.mkdir(parents=True, exist_ok=True)
    img = _aseg(None)
    changes.save(out / f"{stem}.npz")
    vox = img.header.get_zooms()[:3]
    s = changes_summary(changes, (float(vox[0]), float(vox[1]), float(vox[2])))
    (out / f"{stem}.json").write_text(json.dumps(s, indent=1))
    ov = overlay(changes, (0, 0, 0), (img.shape[0], img.shape[1], img.shape[2]))
    nib.save(
        nib.nifti1.Nifti1Image(ov, img.affine, img.header),
        out / f"{stem}_overlay.nii.gz",
    )


def write_box(
    lo: Box, hi: Box, changes: WMChanges, out: Path, stem: str
) -> dict[str, Any]:
    """<stem>.json (box_report), <stem>_overlay.nii.gz and
    <stem>_wm_fixed.nii.gz on the box grid, in out. Returns the report."""
    out.mkdir(parents=True, exist_ok=True)
    img = _aseg(None)
    rep = box_report(lo, hi, changes)
    (out / f"{stem}.json").write_text(json.dumps(rep, indent=1))
    aff = crop_affine(img.affine, lo)
    nib.save(
        nib.nifti1.Nifti1Image(overlay(changes, lo, hi), aff),
        out / f"{stem}_overlay.nii.gz",
    )
    wm = white_matter_box(lo, hi, changes).astype(np.int16)
    nib.save(nib.nifti1.Nifti1Image(wm, aff), out / f"{stem}_wm_fixed.nii.gz")
    return rep


def main(argv: list[str] | None = None) -> None:
    log_to_stderr()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser(
        "dseg", help="the segmentation with a cavity- and handle-free WM mask"
    )
    d.add_argument("aseg", type=Path, help="a BIDS-named segmentation (NIfTI)")
    d.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    c = sub.add_parser("changes", help="remove the handles of the WM label")
    c.add_argument("--out", required=True, type=Path)
    b = sub.add_parser("box", help="report one box before and after")
    b.add_argument("--changes", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--name", required=True)
    b.add_argument("--lo", required=True, type=int, nargs=3)
    b.add_argument("--hi", required=True, type=int, nargs=3)
    m = sub.add_parser("mesh", help="the white surface of a corrected segmentation")
    m.add_argument("dseg", type=Path, help="a desc-wmtopo_dseg (the dseg command)")
    m.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    m.add_argument("--block-cells", type=int, default=BLOCK, help="block size")
    m.add_argument(
        "--device",
        choices=backend.NAMES,
        help="where the mesh steps run (default: $BAILLARGER_DEVICE, else auto)",
    )
    a = ap.parse_args(argv)
    if a.cmd == "dseg":
        for role, path in write_wm_topo(a.aseg, a.out).items():
            print(f"{role}: {path}")
    elif a.cmd == "mesh":
        w = write_white_surface(a.dseg, a.out, a.block_cells, a.device)
        for role, path in w.paths.items():
            print(f"{role}: {path}")
    elif a.cmd == "changes":
        ch = wm_changes()
        write_changes(ch, a.out)
        b0, b1 = ch.betti_before, ch.betti_after
        print(f"(b0, b1, b2) {b0} -> {b1}, {len(ch.voxels)} voxels")
    else:
        lo, hi = (a.lo[0], a.lo[1], a.lo[2]), (a.hi[0], a.hi[1], a.hi[2])
        rep = write_box(lo, hi, WMChanges.load(a.changes), a.out, f"wm_box_{a.name}")
        g = [p["genus"] for p in rep["pieces_after"]]
        n0, n1 = rep["voxels_removed"], rep["voxels_added"]
        print(f"{a.name}: removed {n0}, added {n1}, genus {g}")


if __name__ == "__main__":
    main()
