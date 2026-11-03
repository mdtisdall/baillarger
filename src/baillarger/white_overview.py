"""The overview of a white surface: the third of the three white-surface
commands. Run them in this order:

    python -m baillarger.white_dseg ASEG [--edits EDITS] --out DERIVATIVES
    python -m baillarger.white_surf DSEG --out DERIVATIVES [--block-cells N]
        [--device DEVICE]
    python -m baillarger.white_overview SURF [--eps MM] [--workers N]

The command writes the overview of a surface that baillarger.white_surf
wrote (write_white_overview): the surface decimated for viewing, as an
ordinary GIfTI with embedded data (docs/formats/white-overview.md).
WhiteOverview reads it.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage, GiftiMetaData

from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    file_sha256,
    provenance,
    read_sidecar,
    sidecar_path,
    sources,
    write_sidecar,
)
from baillarger.decimate import closest_points, csr_take, decimate, reorder
from baillarger.gifti import DataArray, write_external
from baillarger.timing import Laps, log, log_to_stderr
from baillarger.white_surf import (
    STRUCTURE,
    SURF_SPEC,
    SurfaceFiles,
    WhiteSurface,
    mesh_pieces,
    mm_to_voxel,
)

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]

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
    g = GiftiImage(meta=GiftiMetaData(AnatomicalStructurePrimary=STRUCTURE[hemi]))
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
    """The overview of the white surface at surf
    (baillarger.white_surf.write_white_surface): the surface decimated
    (baillarger.decimate) so that it keeps its topology, every true vertex is
    within eps_mm of it and its vertices are true vertices; written next to
    surf as an ordinary GIfTI with embedded data, with the true vertex of each
    overview vertex, the true vertices each face carries (external binary
    data) and a sidecar (docs/formats/white-overview.md). The faces are sorted
    by their true vertex ids, smallest first, so that faces near each other
    are near each other in the files. Returns the paths written."""
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
        dict(AnatomicalStructurePrimary=STRUCTURE[hemi]),
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
            Provenance=provenance("baillarger.white_overview.write_white_overview"),
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
    ap = argparse.ArgumentParser(
        description="the overview of a surface that white_surf wrote"
    )
    ap.add_argument("surf", type=Path, help="a hemi-<h>_white.surf.gii (white_surf)")
    ap.add_argument("--eps", type=float, default=OVERVIEW_EPS_MM, help="bound (mm)")
    ap.add_argument("--workers", type=int, default=4, help="processes to decimate")
    a = ap.parse_args(argv)
    ov = write_white_overview(a.surf, a.eps, workers=a.workers)
    for role, path in ov.paths.items():
        print(f"overview {role}: {path}")


if __name__ == "__main__":
    main()
