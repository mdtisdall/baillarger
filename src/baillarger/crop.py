"""Crops of the whole-hemisphere derivatives: one box of voxels cut from the
segmentation, the echoes, other volumes on the same grid, the white surface
(with its antialiased field, and an overview made again) and the label
edits, for tests and prototypes (docs/formats/crop.md).

The box is [lo, hi) in voxels of the segmentation's grid (Box). Three
functions give one:

- voxel_box(lo, hi, shape): the box as given;
- box_at_mm(centre_mm, size_mm, affine, shape): the voxel nearest the
  centre, and half = ceil(size_mm / 2 / voxel size) voxels on each side of
  it, per axis (an odd count: 10 mm at 0.16 mm gives 65 voxels);
- box_at_vertex(surf, vertex, size_mm): box_at_mm at a white-surface vertex.

A box that is not inside the grid is a ValueError: the caller moves the
centre or makes the size smaller.

crop() reads only the box from each file (the white surface through its
block index) into a Crop in memory. write_crop() writes the crop as a BIDS
derivatives set: each file with the name of its source, on the crop's grid,
and one crop/1 sidecar (desc-crop_box.json) that records the box, the
sources and the outputs. load_crop() reads the set back.

The white surface of a crop is a white-surf/1 set on the crop's grid: the
faces of the grid cells inside the box (each face comes from one cell), so
the grid edge of each vertex is in the box. These are the faces that
marching the cropped antialiased field gives. The vertex coordinates,
normals and normal lengths are those of the whole surface, bit for bit (not
smoothed again at the box faces). The edge keys and the block index follow
the white-surf/1 rules on the crop's grid.
"""

import argparse
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.gifti.gifti import GiftiImage

from baillarger import label_edits
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
from baillarger.blocks import morton
from baillarger.mesh import MeshBlocks, edge_key, edge_points
from baillarger.white_overview import write_white_overview
from baillarger.white_surf import (
    SURF_SPEC,
    field_path,
    mesh_pieces,
    mm_to_voxel,
    read_field,
    surface_paths,
    write_block_index,
    write_field,
    write_surface_arrays,
)

type Float = npt.NDArray[np.float64]
type F32 = npt.NDArray[np.float32]
type Int = npt.NDArray[np.int64]
type Triple = tuple[int, int, int]

#: docs/formats/crop.md
CROP_SPEC = SidecarSpec(
    "crop/1",
    COMMON_KEYS
    + (
        Key("BoxLo", "array of integers", True, "first voxel of the box"),
        Key("BoxHi", "array of integers", True, "one past the last voxel"),
        Key("SourceGridShape", "array of integers", True, "the source grid"),
        Key("SourceVoxelToRAS", "array of numbers", True, "the source affine"),
        Key("GridShape", "array of integers", True, "the crop's grid"),
        Key("VoxelToRAS", "array of numbers", True, "the crop's affine"),
        Key("Request", "object", True, "how the box was named"),
        Key("Outputs", "object", True, "the role of each output, by name"),
        Key("OutputSources", "object", True, "the source of each output, by name"),
        Key("OutputsSHA256", "object", True, "SHA-256 of each output, by name"),
    ),
)

#: the crop/1 sidecar's suffix and desc
CROP_SUFFIX, CROP_DESC = "box", "crop"


# ---------------------------------------------------------------- the box


@dataclass(frozen=True)
class Box:
    """A box of voxels [lo, hi) of a grid, and how it was named (request:
    Method voxel-box, centre-mm or vertex, with its values)."""

    lo: Triple
    hi: Triple
    request: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Triple:
        return (
            self.hi[0] - self.lo[0],
            self.hi[1] - self.lo[1],
            self.hi[2] - self.lo[2],
        )

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.lo[0], self.hi[0]),
            slice(self.lo[1], self.hi[1]),
            slice(self.lo[2], self.hi[2]),
        )


def _triple(x: npt.ArrayLike) -> Triple:
    a = [int(t) for t in np.asarray(x).ravel()]
    if len(a) != 3:
        raise ValueError(f"{a}: not three values")
    return (a[0], a[1], a[2])


def _check_inside(box: Box, shape: Sequence[int]) -> None:
    n = _triple(shape)
    if not all(0 <= p < q <= m for p, q, m in zip(box.lo, box.hi, n, strict=True)):
        raise ValueError(f"box {box.lo}-{box.hi} is not inside the grid {n}")


def voxel_box(
    lo: npt.ArrayLike,
    hi: npt.ArrayLike,
    shape: Sequence[int],
    request: dict[str, Any] | None = None,
) -> Box:
    """The box [lo, hi) of a grid of shape. Raises ValueError unless
    0 <= lo < hi <= shape on each axis."""
    a, b = _triple(lo), _triple(hi)
    box = Box(a, b, request or dict(Method="voxel-box", Lo=list(a), Hi=list(b)))
    _check_inside(box, shape)
    return box


def voxel_size(affine: npt.ArrayLike) -> Float:
    """The voxel size (mm) per axis: the norms of the affine's columns."""
    out: Float = np.linalg.norm(np.asarray(affine, np.float64)[:3, :3], axis=0)
    return out


def box_at_mm(
    centre_mm: npt.ArrayLike,
    size_mm: float,
    affine: npt.ArrayLike,
    shape: Sequence[int],
    request: dict[str, Any] | None = None,
) -> Box:
    """The box round the voxel nearest centre_mm (scanner mm), with
    ceil(size_mm / 2 / voxel size) voxels on each side of it, per axis.
    Raises ValueError if it is not inside the grid."""
    if not size_mm > 0:
        raise ValueError("size_mm must be > 0")
    A = np.asarray(affine, np.float64)
    c = np.asarray(centre_mm, np.float64).reshape(3)
    cv = np.rint(np.linalg.inv(A)[:3] @ np.r_[c, 1.0]).astype(np.int64)
    half = np.ceil(size_mm / 2 / voxel_size(A)).astype(np.int64)
    req = request or dict(
        Method="centre-mm", CentreMM=c.tolist(), SizeMM=float(size_mm)
    )
    return voxel_box(cv - half, cv + half + 1, shape, req)


def _surface_side(surf: Path) -> dict[str, Any]:
    return read_sidecar(sidecar_path(surf), SURF_SPEC)


def box_at_vertex(surf: Path, vertex: int, size_mm: float) -> Box:
    """box_at_mm at a vertex of the white surface at surf (white-surf/1),
    on the surface's grid."""
    side = _surface_side(surf)
    pts = GiftiImage.from_filename(str(surf), mmap=True).agg_data("pointset")
    if not 0 <= vertex < len(pts):
        raise ValueError(f"vertex {vertex} is not in the surface ({len(pts)})")
    p = np.asarray(pts[vertex], np.float64)
    req = dict(
        Method="vertex", Vertex=int(vertex), CentreMM=p.tolist(), SizeMM=float(size_mm)
    )
    A = np.asarray(side["VoxelToRAS"], np.float64).reshape(4, 4)
    return box_at_mm(p, size_mm, A, side["GridShape"], req)


def crop_affine(affine: npt.ArrayLike, lo: npt.ArrayLike) -> Float:
    """The affine of a crop whose voxel (0, 0, 0) is voxel lo of the grid."""
    shift = np.eye(4)
    shift[:3, 3] = np.asarray(lo, np.float64)
    out: Float = np.asarray(affine, np.float64) @ shift
    return out


# ---------------------------------------------------------------- the white mesh


@dataclass(frozen=True)
class WhiteCrop:
    """The white surface of a crop: vertices v_mm (float32 scanner mm, as the
    whole surface's file holds them), faces f, the whole surface's normals
    and normal lengths at the vertices, source_ids (the index of each vertex
    in the whole surface; -1 where not known), keys (edge keys on the crop's
    grid) and blocks (the block index on the crop's grid)."""

    v_mm: F32
    f: Int
    normals: F32
    normlen: F32
    source_ids: Int
    keys: Int
    blocks: MeshBlocks
    sidecar: dict[str, Any]


def _gii(surf: Path, name: str) -> GiftiImage:
    g: GiftiImage = GiftiImage.from_filename(str(surf.with_name(name)), mmap=True)
    return g


def crop_white(surf: Path, box: Box) -> WhiteCrop:
    """The faces of the white surface at surf (white-surf/1) that come from
    grid cells inside box, in the white-surf/1 order of a surface on the
    crop's grid. Reads only the blocks whose bounding box meets the box."""
    side = _surface_side(surf)
    grid = _triple(side["GridShape"])
    cells = int(side["BlockCells"])
    _check_inside(box, grid)
    lo, hi = np.asarray(box.lo), np.asarray(box.hi)
    idx = np.loadtxt(surf.with_name(side["BlocksFile"]), skiprows=1, ndmin=2)
    # the grid points of the cells inside the box: lo to hi - 1
    meet = np.all(idx[:, 7:10] <= hi - 1, axis=1) & np.all(idx[:, 10:13] >= lo, axis=1)
    pts, tri = GiftiImage.from_filename(str(surf), mmap=True).agg_data(
        ("pointset", "triangle")
    )
    parts = [np.asarray(tri[int(r[5]) : int(r[6])], np.int64) for r in idx[meet]]
    f0 = np.concatenate(parts) if parts else np.zeros((0, 3), np.int64)
    words = _gii(surf, side["EdgeKeysFile"]).darrays[0].data
    ids0, inv = np.unique(f0, return_inverse=True)
    inv = inv.reshape(-1, 3)
    key0 = np.ascontiguousarray(words[ids0], "<i4").view("<i8").ravel()
    a, b = edge_points(key0, grid)
    # a face's cell is the least lower end of its edges, and the cell + 1
    # their greatest upper end: so the face is inside iff its vertices are
    inside = np.all(a >= lo, axis=1) & np.all(b <= hi - 1, axis=1)
    used, local = np.unique(inv[inside[inv].all(axis=1)], return_inverse=True)
    local = local.reshape(-1, 3)
    a, b = a[used] - lo, b[used] - lo
    ids = ids0[used]
    keys = edge_key(np.ravel_multi_index(tuple(a.T), box.shape), b - a)
    # each face's block on the crop's grid, from its cell; blocks in Morton order
    blk = a[local].min(axis=1) // cells
    order = np.argsort(morton(blk), kind="stable")
    local, blk = local[order], blk[order]
    code = morton(blk)
    _, first, nface = np.unique(code, return_index=True, return_counts=True)
    rank = np.repeat(np.arange(len(first)), nface)
    # a vertex's owner: the first block whose faces use it; then by edge key
    owner = np.full(len(ids), len(first), np.int64)
    np.minimum.at(owner, local.ravel(), np.repeat(rank, 3))
    vorder = np.lexsort((keys, owner))
    new = np.empty(len(vorder), np.int64)
    new[vorder] = np.arange(len(vorder))
    ids, keys = ids[vorder], keys[vorder]
    nvert = np.bincount(owner, minlength=len(first))
    vstop, fstop = np.cumsum(nvert), np.cumsum(nface)
    mb = MeshBlocks(
        cells,
        (blk[first] * cells).reshape(-1, 3).astype(np.int64),
        (vstop - nvert).astype(np.int64),
        vstop.astype(np.int64),
        (fstop - nface).astype(np.int64),
        fstop.astype(np.int64),
        keys,
    )
    nrm = _gii(surf, side["NormalsFile"]).darrays[0].data
    nlen = _gii(surf, side["NormLenFile"]).darrays[0].data.reshape(-1)
    return WhiteCrop(
        v_mm=np.asarray(pts[ids], np.float32),
        f=new[local],
        normals=np.asarray(nrm[ids], np.float32),
        normlen=np.asarray(nlen[ids], np.float32),
        source_ids=ids,
        keys=keys,
        blocks=mb,
        sidecar=side,
    )


# ---------------------------------------------------------------- the crop


@dataclass
class Crop:
    """A crop in memory: the box, the crop's affine (its voxel (0, 0, 0) is
    voxel lo of the source grid) and the cropped data, each with the path of
    its source: the labels (dseg), the echoes and other volumes (stored data
    type, by source path; echoes lists the echoes among them, in order), the
    white surface and its antialiased field (float32; the field's Fill
    outside its box), and the label edits on the crop's grid (linear indices
    of the crop, C order)."""

    box: Box
    affine: Float
    source_affine: Float
    source_shape: Triple
    dseg_path: Path
    dseg: npt.NDArray[Any]
    volumes: dict[Path, npt.NDArray[Any]] = field(default_factory=dict)
    echoes: tuple[Path, ...] = ()
    white_path: Path | None = None
    white: WhiteCrop | None = None
    field: F32 | None = None
    edits_path: Path | None = None
    edits: label_edits.Edits | None = None
    edits_hemisphere: str | None = None

    def echo_stack(self) -> F32:
        """The echoes, float32, (n_echoes, *shape), in their order."""
        return np.stack([np.asarray(self.volumes[p], np.float32) for p in self.echoes])


def _load_nifti(path: Path) -> nib.nifti1.Nifti1Image:
    img = nib.load(path)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{path.name}: not a NIfTI-1 image")
    return img


def _read_box(img: nib.nifti1.Nifti1Image, box: Box) -> npt.NDArray[Any]:
    """The box of the image: its stored data type where it has no scaling
    (nibabel scales only with a slope or intercept other than 1, 0)."""
    return np.asarray(img.dataobj[box.slices])


def crop(
    box: Box,
    dseg: Path,
    echoes: Sequence[Path] = (),
    volumes: Sequence[Path] = (),
    white: Path | None = None,
    edits: Path | None = None,
) -> Crop:
    """Read the box of each file: the segmentation dseg (which sets the
    grid), the echoes and other volumes (each on the dseg's grid), the white
    surface at white (white-surf/1 on this grid) with its antialiased field,
    and the label edits at edits (label-edits/1 on this grid). Raises
    ValueError if the box or a file is not on the dseg's grid."""
    dseg = Path(dseg)
    dimg = _load_nifti(dseg)
    shape = _triple(dimg.shape[:3])
    aff = np.asarray(dimg.affine, np.float64)
    _check_inside(box, shape)
    out = Crop(
        box=box,
        affine=crop_affine(aff, box.lo),
        source_affine=aff,
        source_shape=shape,
        dseg_path=dseg,
        dseg=_read_box(dimg, box),
        echoes=tuple(Path(p) for p in echoes),
    )
    for p in (*out.echoes, *(Path(q) for q in volumes)):
        img = _load_nifti(p)
        if img.shape[:3] != shape or not np.allclose(img.affine, aff):
            raise ValueError(f"{p.name}: not on the grid of {dseg.name}")
        out.volumes[p] = _read_box(img, box)
    if white is not None:
        white = Path(white)
        side = _surface_side(white)
        A = np.asarray(side["VoxelToRAS"], np.float64).reshape(4, 4)
        if tuple(side["GridShape"]) != shape or not np.allclose(A, aff):
            raise ValueError(f"{white.name}: not on the grid of {dseg.name}")
        out.white_path = white
        out.white = crop_white(white, box)
        src, _ = read_field(field_path(white), verify=False)
        out.field = np.asarray(src.read(box.slices), np.float32)
    if edits is not None:
        edits = Path(edits)
        e, eside, _ = label_edits.read_with_sidecar(edits, shape=shape, affine=aff)
        ijk = np.stack(np.unravel_index(e.index, shape), 1)
        lo, hi = np.asarray(box.lo), np.asarray(box.hi)
        ins = np.all((ijk >= lo) & (ijk < hi), axis=1)
        # the C order of the box keeps the C order of the grid: still sorted
        index = np.ravel_multi_index(tuple((ijk[ins] - lo).T), box.shape)
        out.edits_path = edits
        out.edits = label_edits.Edits(
            np.asarray(index, np.int64), np.asarray(e.label[ins], np.int32)
        )
        out.edits_hemisphere = str(eside["Hemisphere"])
    return out


# ---------------------------------------------------------------- the files


def _volume_name(path: Path) -> str:
    """A volume's output name: its source's name, as .nii.gz."""
    return path.name.removesuffix(".gz").removesuffix(".nii") + ".nii.gz"


def _write_volume(path: Path, data: npt.NDArray[Any], src: Path, aff: Float) -> None:
    """data as a NIfTI with the source's header (data type, scaling, units)
    and the crop's affine."""
    hdr = _load_nifti(src).header.copy()
    img = nib.nifti1.Nifti1Image(data, aff, header=hdr)
    img.set_qform(aff, code=int(hdr["qform_code"]))
    img.set_sform(aff, code=int(hdr["sform_code"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)


def _companions(src: Path, dst: Path) -> list[tuple[Path, Path]]:
    """The JSON sidecar and the .tsv (a dseg's label table) of src, if there,
    and their places next to dst."""
    out = []
    for ext in (".json", ".tsv"):
        s = src.with_name(src.name.split(".", 1)[0] + ext)
        if s.is_file():
            out.append((s, dst.with_name(dst.name.split(".", 1)[0] + ext)))
    return out


def crop_path(root: Path, ents: dict[str, str]) -> Path:
    """The crop/1 sidecar of a crop in the derivatives root."""
    return derivative_path(root, ents, CROP_SUFFIX, ".json", desc=CROP_DESC)


type _Add = Callable[[str, Path, Path], None]


def write_crop(c: Crop, root: Path, overview: bool = True) -> dict[str, Path]:
    """Write the crop as a BIDS derivatives set in root (docs/formats/crop.md):
    each file with its source's name, in root/sub-<sub>/ses-<ses>/anat/, on
    the crop's grid; the white-surf/1 set with its antialiased field
    (white-antialias/1) and, with overview, an overview made again from the
    crop (white-overview/1); the label edits (label-edits/1); and the crop/1
    sidecar. Returns the paths written, by role."""
    ents = entities(c.dseg_path)
    dataset_description(root)
    roles: dict[str, str] = {}
    srcs: dict[str, str] = {}
    written: dict[str, Path] = {}

    def add(role: str, p: Path, src: Path) -> None:
        written[role] = p
        roles[p.name] = role
        srcs[p.name] = str(src.resolve())

    anat = crop_path(root, ents).parent
    dpath = anat / _volume_name(c.dseg_path)
    _write_volume(dpath, c.dseg, c.dseg_path, c.affine)
    add("dseg", dpath, c.dseg_path)
    for s, d in _companions(c.dseg_path, dpath):
        shutil.copyfile(s, d)
        add("dseg" + d.suffix, d, s)
    n_other = 0
    for src, data in c.volumes.items():
        p = anat / _volume_name(src)
        if p.name in roles:
            raise ValueError(f"two outputs named {p.name}")
        if src in c.echoes:
            role = f"echo{c.echoes.index(src) + 1}"
        else:
            role, n_other = f"volume{n_other + 1}", n_other + 1
        _write_volume(p, data, src, c.affine)
        add(role, p, src)
        for s, d in _companions(src, p):
            shutil.copyfile(s, d)
            add(role + d.suffix, d, s)
    if c.white is not None:
        _write_white(c, root, ents, dpath, add)
        if overview:
            assert c.white_path is not None
            for r, p in write_white_overview(written["white"]).paths.items():
                add("overview." + r, p, c.white_path)
    if c.edits is not None and c.edits_path is not None:
        assert c.edits_hemisphere is not None
        ep = anat / c.edits_path.name
        label_edits.write(
            ep,
            c.edits,
            hemisphere=c.edits_hemisphere,
            shape=c.box.shape,
            affine=c.affine,
            description="The label edits of a hemisphere inside a crop's box, "
            "on the crop's grid: each edited voxel and its label.",
        )
        add("edits", ep, c.edits_path)
        add("edits.json", sidecar_path(ep), sidecar_path(c.edits_path))
    inputs = [c.dseg_path, *c.volumes]
    inputs += [p for p in (c.white_path, c.edits_path) if p is not None]
    side = crop_path(root, ents)
    write_sidecar(
        side,
        CROP_SPEC,
        dict(
            SpecVersion=CROP_SPEC.version,
            Description="A crop of the derivatives: the box BoxLo to BoxHi of "
            "the source grid; each output on the crop's grid.",
            **sources(*inputs),
            Provenance=provenance("baillarger.crop.write_crop"),
            BoxLo=list(c.box.lo),
            BoxHi=list(c.box.hi),
            SourceGridShape=list(c.source_shape),
            SourceVoxelToRAS=c.source_affine.ravel().tolist(),
            GridShape=list(c.box.shape),
            VoxelToRAS=c.affine.ravel().tolist(),
            Request=c.box.request,
            Outputs=roles,
            OutputSources=srcs,
            OutputsSHA256={p.name: file_sha256(p) for p in written.values()},
        ),
    )
    written["crop"] = side
    return written


def _write_white(
    c: Crop, root: Path, ents: dict[str, str], dpath: Path, add: _Add
) -> None:
    """The white-surf/1 set of the crop and its antialiased field; their
    Sources are the crop's dseg (dpath)."""
    w, src = c.white, c.white_path
    assert w is not None and src is not None and c.field is not None
    side = w.sidecar
    hemi = side["Hemisphere"]
    paths = surface_paths(root, ents, hemi)
    paths["surf"].parent.mkdir(parents=True, exist_ok=True)
    dat, ndat, ldat, kdat = write_surface_arrays(
        paths, w.v_mm, w.f, w.normals, w.normlen, w.keys, hemi
    )
    rows = write_block_index(paths["blocks"], w.v_mm, w.f, w.blocks, c.affine)
    vox = voxel_size(c.affine)
    pieces = mesh_pieces(mm_to_voxel(w.v_mm, c.affine), w.f, (vox[0], vox[1], vox[2]))
    fpath = field_path(src)
    _, fside = read_field(fpath, verify=False)
    write_field(
        paths["field"],
        c.field,
        (0, 0, 0),
        c.box.shape,
        c.affine,
        dict(
            **sources(dpath),
            Provenance=provenance("baillarger.crop.write_crop"),
            Hemisphere=hemi,
            FilledLabels=list(fside["FilledLabels"]),
            Antialias=fside["Antialias"],
        ),
    )
    write_sidecar(
        paths["sidecar"],
        SURF_SPEC,
        dict(
            SpecVersion=SURF_SPEC.version,
            Description="A crop of the white (WM) surface: the faces of the "
            "grid cells inside the crop's box, with the whole surface's "
            "vertices, smoothed normals and normal lengths; edge keys and "
            "blocks on the crop's grid (docs/formats/crop.md).",
            **sources(dpath),
            Provenance=provenance("baillarger.crop.write_crop"),
            Hemisphere=hemi,
            FilledLabels=list(side["FilledLabels"]),
            BlockCells=int(side["BlockCells"]),
            VoxelToRAS=c.affine.ravel().tolist(),
            Vertices=len(w.v_mm),
            Faces=len(w.f),
            Blocks=len(rows),
            Pieces=pieces,
            NormalsFile=paths["normals"].name,
            EdgeKeysFile=paths["keys"].name,
            NormLenFile=paths["normlen"].name,
            NormalSmoothingSteps=int(side["NormalSmoothingSteps"]),
            GridShape=list(c.box.shape),
            BlocksFile=paths["blocks"].name,
            DataSHA256={d.name: file_sha256(d) for d in (dat, ndat, kdat, ldat)},
            Antialias=side["Antialias"],
            Device=side["Device"],
        ),
    )
    for role, p, s in (
        ("white", paths["surf"], src),
        ("white.dat", dat, src),
        ("white.json", paths["sidecar"], sidecar_path(src)),
        ("normals", paths["normals"], src.with_name(side["NormalsFile"])),
        ("normals.dat", ndat, src.with_name(side["NormalsFile"])),
        ("normlen", paths["normlen"], src.with_name(side["NormLenFile"])),
        ("normlen.dat", ldat, src.with_name(side["NormLenFile"])),
        ("keys", paths["keys"], src.with_name(side["EdgeKeysFile"])),
        ("keys.dat", kdat, src.with_name(side["EdgeKeysFile"])),
        ("blocks", paths["blocks"], src.with_name(side["BlocksFile"])),
        ("field", paths["field"], fpath),
        ("field.json", paths["field_sidecar"], sidecar_path(fpath)),
    ):
        add(role, p, s)


def load_crop(path: Path, verify: bool = True) -> Crop:
    """The crop that write_crop wrote, read back from its crop/1 sidecar at
    path. Its white surface (WhiteCrop, source_ids -1: the files do not keep
    them) is also readable with baillarger.white_surf.WhiteSurface.load.
    verify: raise ValueError if an output's SHA-256 is not the sidecar's."""
    side = read_sidecar(path, CROP_SPEC)
    d = path.parent
    if verify:
        for name, sha in side["OutputsSHA256"].items():
            if file_sha256(d / name) != sha:
                raise ValueError(f"{name}: the SHA-256 is not the crop sidecar's")
    by_role = {r: d / n for n, r in side["Outputs"].items()}
    box = Box(_triple(side["BoxLo"]), _triple(side["BoxHi"]), dict(side["Request"]))
    whole = tuple(slice(0, n) for n in box.shape)
    out = Crop(
        box=box,
        affine=np.asarray(side["VoxelToRAS"], np.float64).reshape(4, 4),
        source_affine=np.asarray(side["SourceVoxelToRAS"], np.float64).reshape(4, 4),
        source_shape=_triple(side["SourceGridShape"]),
        dseg_path=by_role["dseg"],
        dseg=_read_box(_load_nifti(by_role["dseg"]), Box((0, 0, 0), box.shape)),
    )
    vol_roles = [r for r in by_role if r[:4] in ("echo", "volu") and "." not in r]
    echo_roles = sorted((r for r in vol_roles if r.startswith("echo")), key=_rank)
    out.echoes = tuple(by_role[r] for r in echo_roles)
    for r in vol_roles:
        img = _load_nifti(by_role[r])
        out.volumes[by_role[r]] = _read_box(img, Box((0, 0, 0), box.shape))
    if "white" in by_role:
        out.white_path = by_role["white"]
        out.white = _read_white(by_role["white"])
        fsrc, _ = read_field(by_role["field"], verify=False)
        out.field = np.asarray(fsrc.read(whole), np.float32)
    if "edits" in by_role:
        e, eside, _ = label_edits.read_with_sidecar(by_role["edits"])
        out.edits_path, out.edits = by_role["edits"], e
        out.edits_hemisphere = str(eside["Hemisphere"])
    return out


def _rank(role: str) -> int:
    return int(role.removeprefix("echo"))


def _read_white(surf: Path) -> WhiteCrop:
    side = _surface_side(surf)
    pts, tri = GiftiImage.from_filename(str(surf)).agg_data(("pointset", "triangle"))
    words = _gii(surf, side["EdgeKeysFile"]).darrays[0].data
    keys = np.ascontiguousarray(words, "<i4").view("<i8").ravel().astype(np.int64)
    idx = np.loadtxt(surf.with_name(side["BlocksFile"]), skiprows=1, ndmin=2)
    ii = idx[:, :7].astype(np.int64)
    cells = int(side["BlockCells"])
    return WhiteCrop(
        v_mm=np.asarray(pts, np.float32),
        f=np.asarray(tri, np.int64),
        normals=np.asarray(_gii(surf, side["NormalsFile"]).darrays[0].data, np.float32),
        normlen=np.asarray(
            _gii(surf, side["NormLenFile"]).darrays[0].data, np.float32
        ).reshape(-1),
        source_ids=np.full(len(pts), -1, np.int64),
        keys=keys,
        blocks=MeshBlocks(
            cells, ii[:, :3] * cells, ii[:, 3], ii[:, 4], ii[:, 5], ii[:, 6], keys
        ),
        sidecar=side,
    )


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="crop the whole-hemisphere derivatives to a box "
        "(docs/formats/crop.md)"
    )
    ap.add_argument("--dseg", required=True, type=Path, help="the segmentation")
    ap.add_argument(
        "--echo", type=Path, action="append", default=[], help="an echo (in order)"
    )
    ap.add_argument(
        "--volume", type=Path, action="append", default=[], help="another volume"
    )
    ap.add_argument("--white", type=Path, help="a white-surf/1 .surf.gii")
    ap.add_argument("--edits", type=Path, help="a label-edits/1 .tsv")
    ap.add_argument("--no-overview", action="store_true", help="write no overview")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--centre-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    g.add_argument("--vertex", type=int, help="a vertex of --white")
    g.add_argument("--lo", type=int, nargs=3, metavar=("I", "J", "K"))
    ap.add_argument("--hi", type=int, nargs=3, metavar=("I", "J", "K"))
    ap.add_argument("--size-mm", type=float, help="the box's edge (mm)")
    ap.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    a = ap.parse_args(argv)
    img = _load_nifti(a.dseg)
    shape = img.shape[:3]
    if a.lo is not None:
        if a.hi is None:
            ap.error("--lo needs --hi")
        box = voxel_box(a.lo, a.hi, shape)
    elif a.size_mm is None:
        ap.error("--centre-mm and --vertex need --size-mm")
    elif a.vertex is not None:
        if a.white is None:
            ap.error("--vertex needs --white")
        box = box_at_vertex(a.white, a.vertex, a.size_mm)
    else:
        box = box_at_mm(a.centre_mm, a.size_mm, img.affine, shape)
    c = crop(box, a.dseg, a.echo, a.volume, a.white, a.edits)
    w = write_crop(c, a.out, overview=not a.no_overview)
    print(json.dumps({k: str(v) for k, v in w.items()}, indent=1))


if __name__ == "__main__":
    main()
