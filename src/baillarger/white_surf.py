"""The white surface of a corrected segmentation (mesh): the second of the
three white-surface commands. Run them in this order:

    python -m baillarger.white_dseg ASEG [--edits EDITS] --out DERIVATIVES
    python -m baillarger.white_surf DSEG --out DERIVATIVES [--block-cells N]
        [--device DEVICE]
    python -m baillarger.white_overview SURF [--eps MM] [--workers N]

The command writes the white surface of the WM mask of a corrected
segmentation (baillarger.white_dseg) as GIfTI, with its normals, edge keys
and block index (docs/formats/white-surf.md). WhiteSurface reads it.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from nibabel.gifti.gifti import GiftiImage

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
from baillarger.gifti import DataArray, write_external
from baillarger.handles import kuhn_betti
from baillarger.mesh import BLOCK, MeshBlocks, marching_tetrahedra_blocks, topology
from baillarger.mesh_torch import MeshOps
from baillarger.timing import Laps, log, log_to_stderr
from baillarger.white_dseg import DSEG_SPEC

type Float = npt.NDArray[np.float64]
type F32 = npt.NDArray[np.float32]
type Int = npt.NDArray[np.int64]


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

STRUCTURE = {"R": "CortexRight", "L": "CortexLeft"}


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
    """The command: the white mesh of the WM mask of a corrected
    segmentation (baillarger.white_dseg.write_wm_topo), as GIfTI with external
    binary data, the normals, the block index and the sidecar
    (docs/formats/white-surf.md). The surface is marching tetrahedra of the
    antialiased field of the mask (baillarger.antialias): the binary mesh's
    triangles, each vertex moved along its own grid edge, so no
    self-intersections by construction. Raises ValueError if the mask has a
    handle or a cavity. Returns the paths written."""
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
        AnatomicalStructurePrimary=STRUCTURE[hemi],
        AnatomicalStructureSecondary="GrayWhite",
        GeometricType="Anatomical",
    )
    dat = write_external(
        paths["surf"],
        [
            DataArray(v_mm, "NIFTI_INTENT_POINTSET", surf_meta, np.eye(4)),
            DataArray(f.astype("<i4"), "NIFTI_INTENT_TRIANGLE"),
        ],
        dict(AnatomicalStructurePrimary=STRUCTURE[hemi]),
    )
    ndat = write_external(
        paths["normals"],
        [DataArray(n.astype("<f4"), "NIFTI_INTENT_VECTOR", dict(Name="normals"))],
        dict(AnatomicalStructurePrimary=STRUCTURE[hemi]),
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
        dict(AnatomicalStructurePrimary=STRUCTURE[hemi]),
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
            Provenance=provenance("baillarger.white_surf.write_white_surface"),
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


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    log_to_stderr()
    ap = argparse.ArgumentParser(
        description="the white surface of a corrected segmentation"
    )
    ap.add_argument("dseg", type=Path, help="a desc-wmtopo_dseg (white_dseg)")
    ap.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    ap.add_argument("--block-cells", type=int, default=BLOCK, help="block size")
    ap.add_argument(
        "--device",
        choices=backend.NAMES,
        help="where the mesh steps run (default: $BAILLARGER_DEVICE, else auto)",
    )
    a = ap.parse_args(argv)
    w = write_white_surface(a.dseg, a.out, a.block_cells, a.device)
    for role, path in w.paths.items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
