"""Tests for the mesh command (baillarger.white.write_white_surface) and its
reader (WhiteSurface): GIfTI with external binary data, the block index and
the sidecar, on a synthetic segmentation."""

import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from nibabel.gifti.gifti import GiftiImage

from baillarger import bids, white
from baillarger.mesh import topology

VOX = (0.2, 0.16, 0.2)
AFF = np.array([[0.2, 0, 0, -3.0], [0, 0.16, 0, 2.0], [0, 0, 0.2, 5.0], [0, 0, 0, 1.0]])


def write_dseg(d: Path) -> Path:
    """A right hemisphere: a WM ball round a ventricle, in cortex, and a
    separate WM block; through the dseg command. Returns the dseg's path."""
    x = np.indices((44, 40, 36)).astype(float)
    r = np.sqrt(((x - np.array([20.3, 19.6, 17.8])[:, None, None, None]) ** 2).sum(0))
    a = np.zeros(x.shape[1:], np.int16)
    a[r < 15] = 42
    a[r < 12] = 41
    a[r < 4] = 43
    a[36:42, 4:12, 4:30] = 41
    src = d / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(parents=True)
    nib.save(nib.nifti1.Nifti1Image(a, AFF), src)
    white.main(["dseg", str(src), "--out", str(d / "deriv")])
    return d / "deriv/sub-T01/ses-S1/anat/sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"


@pytest.fixture(scope="module")
def surface(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("surf")
    dseg = write_dseg(d)
    out = d / "deriv"
    white.main(["mesh", str(dseg), "--out", str(out), "--block-cells", "8"])
    surf = out / "sub-T01/ses-S1/anat/sub-T01_ses-S1_hemi-R_white.surf.gii"
    return dseg, surf


def test_the_surface_and_its_sidecar(surface: tuple[Path, Path]) -> None:
    dseg, surf = surface
    s = white.WhiteSurface.load(surf)
    s.check(dseg)
    t = topology(s.f, len(s.v))
    assert t["components"] == 2 and t["genus"] == 0 and t["boundary_loops"] == 0
    side = s.sidecar
    assert side["Vertices"] == len(s.v) and side["Faces"] == len(s.f)
    assert [p["genus"] for p in side["Pieces"]] == [0, 0]
    aa = side["Antialias"]
    assert aa["Method"] == "thin-plate" and aa["Converged"] and aa["Iterations"] > 0
    assert side["BlockCells"] == 8 and np.allclose(s.affine, AFF)
    for name, sha in side["DataSHA256"].items():
        assert bids.file_sha256(surf.with_name(name)) == sha
    # the ball's surface lies between the ventricle and the cortex, round its
    # centre; normals out of the WM
    ball = s.v[:, 0] < 34
    c = np.array([20.3, 19.6, 17.8])
    rad = np.linalg.norm(s.v[ball] - c, axis=1)
    assert rad.min() > 10 and rad.max() < 13
    assert np.all(np.sum(s.normals[ball] * (s.v[ball] - c), 1) > 0)
    assert np.allclose(np.linalg.norm(s.normals, axis=1), 1, atol=1e-5)


def test_nibabel_reads_the_external_data(surface: tuple[Path, Path]) -> None:
    _, surf = surface
    s = white.WhiteSurface.load(surf)
    g = GiftiImage.from_filename(str(surf), mmap=True)
    pts, tri = g.agg_data(("pointset", "triangle"))
    assert isinstance(pts, np.memmap)
    assert pts.dtype == np.float32 and tri.dtype == np.int32
    mm = s.v @ AFF[:3, :3].T + AFF[:3, 3]
    assert np.allclose(pts, mm, atol=1e-4) and np.array_equal(tri, s.f)
    pa = g.darrays[0]
    assert pa.meta["AnatomicalStructurePrimary"] == "CortexRight"
    assert pa.meta["GeometricType"] == "Anatomical"
    assert pa.coordsys is not None and np.allclose(pa.coordsys.xform, np.eye(4))
    for da in g.darrays:
        assert da.ext_fname == surf.name.replace(".surf.gii", ".dat")
    n = GiftiImage.from_filename(
        str(surf.with_name(s.sidecar["NormalsFile"])), mmap=True
    )
    assert np.allclose(n.darrays[0].data, s.normals, atol=1e-6)
    # the edge keys: int64 as (low, high) int32 words, one per vertex
    k = (
        GiftiImage.from_filename(
            str(surf.with_name(s.sidecar["EdgeKeysFile"])), mmap=True
        )
        .darrays[0]
        .data
    )
    assert k.dtype == np.int32 and k.shape == (len(s.v), 2)
    assert np.array_equal(np.ascontiguousarray(k).view("<i8").ravel(), s.vertex_key)
    assert len(np.unique(s.vertex_key)) == len(s.v)


def test_blocks_cover_the_mesh(surface: tuple[Path, Path]) -> None:
    _, surf = surface
    whole = white.WhiteSurface.load(surf)
    idx = np.loadtxt(surf.with_name(whole.sidecar["BlocksFile"]), skiprows=1)
    assert len(idx) == whole.sidecar["Blocks"] > 8
    header = surf.with_name(whole.sidecar["BlocksFile"]).read_text().split("\n")[0]
    assert tuple(header.split("\t")) == white.BLOCK_COLUMNS
    faces = []
    for row in idx:
        blk = (int(row[0]), int(row[1]), int(row[2]))
        b = white.WhiteSurface.load(surf, block=blk)
        g = b.vertex_ids[b.f]  # global indices
        assert np.array_equal(g, whole.f[int(row[5]) : int(row[6])])
        assert np.allclose(b.v, whole.v[b.vertex_ids])
        assert np.allclose(b.normals, whole.normals[b.vertex_ids])
        assert np.array_equal(b.vertex_key, whole.vertex_key[b.vertex_ids])
        # the box holds the vertices as read, with no tolerance
        assert np.all(b.v >= row[7:10]) and np.all(b.v <= row[10:13])
        assert np.all(row[10:13] - row[7:10] >= 0)
        faces.append(g)
    assert np.array_equal(np.concatenate(faces), whole.f)
    empty = white.WhiteSurface.load(surf, block=(100, 100, 100))
    assert len(empty.f) == 0 and len(empty.v) == 0


def test_the_files_move_together(surface: tuple[Path, Path], tmp_path: Path) -> None:
    _, surf = surface
    for p in surf.parent.iterdir():
        shutil.copy(p, tmp_path / p.name)
    a = white.WhiteSurface.load(surf)
    b = white.WhiteSurface.load(tmp_path / surf.name)
    assert np.array_equal(a.v, b.v) and np.array_equal(a.f, b.f)


def test_the_mesh_command_refuses_a_handle(tmp_path: Path) -> None:
    dseg = write_dseg(tmp_path)
    img = nib.load(dseg)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    a: npt.NDArray[np.int16] = np.asarray(img.dataobj).copy()
    a[16:26, 15:24, 12:24] = 42  # a tunnel through the ball, round the ventricle
    a[19:22, 18:21, 12:24] = 0
    nib.save(nib.nifti1.Nifti1Image(a, img.affine, img.header), dseg)
    with pytest.raises(ValueError, match="Betti"):
        white.write_white_surface(dseg, tmp_path / "deriv2")
    s = white.WhiteSurface.load(
        white.write_white_surface(
            write_dseg(tmp_path / "again"), tmp_path / "deriv3"
        ).paths["surf"]
    )
    with pytest.raises(ValueError, match="not the surface's segmentation"):
        s.check(dseg)


def test_float32_matches_the_float64_reference() -> None:
    """The mesh steps in float32 (as on a GPU) against float64: the same
    faces and blocks; vertices and normals within a stated tolerance."""
    from baillarger import mesh_smoothing as S
    from baillarger.mesh import marching_tetrahedra_blocks

    x = np.indices((48, 44, 40)).astype(float)
    r = np.sqrt(((x - np.array([23.3, 21.6, 19.8])[:, None, None, None]) ** 2).sum(0))
    # a ball with a bumpy surface, so smoothing has work to do
    mask = (r + 1.5 * np.sin(x[0] / 2.1) * np.cos(x[2] / 1.7) < 15).view(np.uint8)
    out = {}
    for dt in (np.float64, np.float32):
        v0, f, b = marching_tetrahedra_blocks(mask, 0.5, 8, dt)
        v = S.twice(S.relax(v0, f), f, 2.0)
        n = S.smooth_normals(v, f)
        assert v0.dtype == v.dtype == n.dtype == dt
        out[dt] = (v0, f, b, v, n)
    v0a, fa, ba, va, na = out[np.float64]
    v0b, fb, bb, vb, nb = out[np.float32]
    assert np.array_equal(fa, fb) and np.array_equal(ba.face_stop, bb.face_stop)
    assert np.array_equal(v0b, v0a.astype(np.float32))  # rounded per vertex
    assert np.abs(vb - va).max() < 1e-3  # voxels
    assert np.abs(nb - na).max() < 1e-3


def test_the_reader_gives_float32(surface: tuple[Path, Path]) -> None:
    _, surf = surface
    s = white.WhiteSurface.load(surf)
    assert s.v.dtype == np.float32 and s.normals.dtype == np.float32
    b = white.WhiteSurface.load(surf, block=(0, 0, 0))
    assert b.v.dtype == np.float32
