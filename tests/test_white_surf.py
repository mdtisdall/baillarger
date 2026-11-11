"""Tests for the mesh command (baillarger.white_surf.write_white_surface) and its
reader (WhiteSurface): GIfTI with external binary data, the block index and
the sidecar, on a synthetic segmentation."""

import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from nibabel.gifti.gifti import GiftiImage

from baillarger import antialias as AA
from baillarger import backend, bids, white_dseg, white_surf
from baillarger import mesh_smoothing as S
from baillarger.blocks import ArraySource, PatchedSource
from baillarger.mesh import marching_tetrahedra_blocks, topology

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
    white_dseg.main([str(src), "--out", str(d / "deriv")])
    return d / "deriv/sub-T01/ses-S1/anat/sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"


@pytest.fixture(scope="module")
def surface(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("surf")
    dseg = write_dseg(d)
    out = d / "deriv"
    white_surf.main([str(dseg), "--out", str(out), "--block-cells", "8"])
    surf = out / "sub-T01/ses-S1/anat/sub-T01_ses-S1_hemi-R_white.surf.gii"
    return dseg, surf


def test_the_surface_and_its_sidecar(surface: tuple[Path, Path]) -> None:
    dseg, surf = surface
    s = white_surf.WhiteSurface.load(surf)
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
    # the normal length: in [0, 1], near 1 on the smooth ball and block, and
    # the length that mesh_smoothing gives for the vertices as read. On the
    # ball (radius r about 11 voxels) the unit normals turn by about s / r
    # over the smoothing support (s about 2 voxels), so the length is about
    # 1 - (s / r)**2 / 2, near 0.98
    assert s.normlen.dtype == np.float32 and s.normlen.shape == (len(s.v),)
    assert s.normlen.min() >= 0 and s.normlen.max() <= 1
    assert np.median(s.normlen) > 0.98
    _, ref = S.smooth_normals_and_length(s.v, s.f)
    assert np.abs(s.normlen - np.minimum(ref, 1)).max() < 1e-4


def test_nibabel_reads_the_external_data(surface: tuple[Path, Path]) -> None:
    _, surf = surface
    s = white_surf.WhiteSurface.load(surf)
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
    nl = GiftiImage.from_filename(
        str(surf.with_name(s.sidecar["NormLenFile"])), mmap=True
    ).darrays[0]
    # one value per vertex: a 1-D array, shape (n,)
    assert nl.dims == [len(s.v)]
    assert nl.data.dtype == np.float32 and nl.data.shape == (len(s.v),)
    assert np.array_equal(nl.data, s.normlen)
    assert nl.ext_fname == surf.name.replace(
        "_white.surf.gii", "_desc-smoothnormlen_white.dat"
    )
    # the names and the sidecar say that the normals are smoothed, and how much
    assert s.sidecar["NormalsFile"] == surf.name.replace(
        "_white.surf.gii", "_desc-smoothnormals_white.func.gii"
    )
    assert s.sidecar["NormalSmoothingSteps"] == white_surf.NORMAL_STEPS == 10
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
    whole = white_surf.WhiteSurface.load(surf)
    idx = np.loadtxt(surf.with_name(whole.sidecar["BlocksFile"]), skiprows=1)
    assert len(idx) == whole.sidecar["Blocks"] > 8
    header = surf.with_name(whole.sidecar["BlocksFile"]).read_text().split("\n")[0]
    assert tuple(header.split("\t")) == white_surf.BLOCK_COLUMNS
    faces = []
    for row in idx:
        blk = (int(row[0]), int(row[1]), int(row[2]))
        b = white_surf.WhiteSurface.load(surf, block=blk)
        g = b.vertex_ids[b.f]  # global indices
        assert np.array_equal(g, whole.f[int(row[5]) : int(row[6])])
        assert np.allclose(b.v, whole.v[b.vertex_ids])
        assert np.allclose(b.normals, whole.normals[b.vertex_ids])
        assert np.array_equal(b.normlen, whole.normlen[b.vertex_ids])
        assert np.array_equal(b.vertex_key, whole.vertex_key[b.vertex_ids])
        # the box holds the vertices as read, with no tolerance
        assert np.all(b.v >= row[7:10]) and np.all(b.v <= row[10:13])
        assert np.all(row[10:13] - row[7:10] >= 0)
        faces.append(g)
    assert np.array_equal(np.concatenate(faces), whole.f)
    empty = white_surf.WhiteSurface.load(surf, block=(100, 100, 100))
    assert len(empty.f) == 0 and len(empty.v) == 0


def test_the_files_move_together(surface: tuple[Path, Path], tmp_path: Path) -> None:
    _, surf = surface
    for p in surf.parent.iterdir():
        shutil.copy(p, tmp_path / p.name)
    a = white_surf.WhiteSurface.load(surf)
    b = white_surf.WhiteSurface.load(tmp_path / surf.name)
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
        white_surf.write_white_surface(dseg, tmp_path / "deriv2")
    s = white_surf.WhiteSurface.load(
        white_surf.write_white_surface(
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
    s = white_surf.WhiteSurface.load(surf)
    assert s.v.dtype == np.float32 and s.normals.dtype == np.float32
    assert s.normlen.dtype == np.float32
    b = white_surf.WhiteSurface.load(surf, block=(0, 0, 0))
    assert b.v.dtype == np.float32


# ------------------------------------------------------- the antialiased field


def test_the_stored_field_reads_back_as_the_field_the_surface_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    used = []
    solve = AA.thin_plate

    def keep(*a: Any, **k: Any) -> tuple[npt.NDArray[np.float32], AA.Solve]:
        phi, s = solve(*a, **k)
        used.append(phi.copy())
        return phi, s

    monkeypatch.setattr(AA, "thin_plate", keep)
    dseg = write_dseg(tmp_path)
    w = white_surf.write_white_surface(dseg, tmp_path / "deriv", 8)
    field, side = white_surf.read_field(w.paths["field"])
    assert len(used) == 1
    phi = used[0]
    lo = np.array(side["BoxOrigin"])
    box = tuple(slice(int(a), int(a) + n) for a, n in zip(lo, phi.shape, strict=True))
    whole = field.read(tuple(slice(0, n) for n in field.shape))
    assert field.dtype == np.float32 and np.array_equal(whole[box], phi)
    rest = np.ones(whole.shape, bool)
    rest[box] = False
    assert white_surf.FIELD_FILL == side["Fill"] == -(AA.BAND + 3)
    assert np.all(whole[rest] == side["Fill"])
    assert w.paths["field"].name == "sub-T01_ses-S1_hemi-R_desc-antialias_white.nii"
    assert side["GridShape"] == list(field.shape) and side["BoxShape"] == list(
        phi.shape
    )
    assert np.allclose(np.reshape(side["VoxelToRAS"], (4, 4)), AFF)
    img = nib.load(w.paths["field"])
    assert isinstance(img, nib.nifti1.Nifti1Image)
    assert np.allclose(img.affine[:3, 3], AFF[:3, :3] @ lo + AFF[:3, 3])
    assert (
        side["Antialias"]
        == white_surf.WhiteSurface.load(w.paths["surf"]).sidecar["Antialias"]
    )
    # the surface is the march of the stored field
    s = white_surf.WhiteSurface.load(w.paths["surf"])
    _, _, mb = marching_tetrahedra_blocks(field, 0.0, 8, np.float32)
    assert np.array_equal(np.sort(mb.vertex_key), np.sort(s.vertex_key))
    # a changed file is refused
    with w.paths["field"].open("r+b") as fh:
        fh.seek(-4, 2)
        fh.write(b"\0\0\0\0")
    with pytest.raises(ValueError, match="SHA-256"):
        white_surf.read_field(w.paths["field"])


def wm_mask(dseg: Path) -> npt.NDArray[np.bool_]:
    img = nib.load(dseg)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    side = bids.read_sidecar(bids.sidecar_path(dseg), white_dseg.DSEG_SPEC)
    out: npt.NDArray[np.bool_] = np.isin(np.asarray(img.dataobj), side["FilledLabels"])
    return out


def face_keys(
    keys: npt.NDArray[np.int64], f: npt.NDArray[np.int64]
) -> set[tuple[int, ...]]:
    """The faces as sets of edge keys (independent of vertex numbers)."""
    return {tuple(sorted(t)) for t in keys[f].tolist()}


def test_an_edit_with_no_change_of_the_mask_gives_no_new_blocks(
    surface: tuple[Path, Path],
) -> None:
    dseg, surf = surface
    field, _ = white_surf.read_field(
        surf.with_name(
            surf.name.replace("_white.surf.gii", "_desc-antialias_white.nii")
        )
    )
    mask = wm_mask(dseg)
    dev = backend.device("cpu")
    vox = np.argwhere(mask)[:50]
    assert AA.local_solve(field, ArraySource(mask), vox, dev) is None
    box = (slice(10, 30), slice(10, 30), slice(10, 30))
    same = AA.LocalSolve(
        box,
        field.read(box),
        np.zeros((0, 3), np.int64),
        16,
        5.0,
        1.0,
        AA.Solve(0, 0, 0.0, True, 0.0),
    )
    r = white_surf.remarch(field, same, 8)
    assert len(r.blocks) == 0 and len(r.v) == 0 and len(r.f) == 0


def test_the_remarched_blocks_replace_the_surfaces_blocks(
    surface: tuple[Path, Path],
) -> None:
    """After a local solve of an edit, the march of the whole patched field is the
    stored surface's blocks where the field did not change and the re-marched blocks
    where it did."""
    dseg, surf = surface
    field, _ = white_surf.read_field(
        surf.with_name(
            surf.name.replace("_white.surf.gii", "_desc-antialias_white.nii")
        )
    )
    mask = wm_mask(dseg)
    edited = mask.copy()
    x = np.indices(mask.shape).astype(float)
    q = np.array([20, 19, 29])  # on the ball's surface (radius about 12)
    bump = np.sqrt(((x - q[:, None, None, None]) ** 2).sum(0)) <= 2.0
    edited |= bump
    dev = backend.device("cpu")
    local = AA.local_solve(field, ArraySource(edited), np.argwhere(bump), dev)
    assert local is not None and len(local.changed)
    r = white_surf.remarch(field, local, 8)
    assert len(r.blocks) and r.v.dtype == np.float32
    s = white_surf.WhiteSurface.load(surf)
    idx = np.loadtxt(surf.with_name(s.sidecar["BlocksFile"]), skiprows=1, ndmin=2)
    touched = {tuple(b) for b in r.blocks.tolist()}
    want: set[tuple[int, ...]] = set()
    for row in idx:
        if (int(row[0]), int(row[1]), int(row[2])) not in touched:
            want |= face_keys(s.vertex_key, s.f[int(row[5]) : int(row[6])])
    want |= face_keys(r.mesh.vertex_key, r.f)
    patched = PatchedSource(field, local.box, local.phi)
    v, f, mb = marching_tetrahedra_blocks(patched, 0.0, 8, np.float32)
    assert face_keys(mb.vertex_key, f) == want
    # the re-marched vertices are those of the full march
    pos = dict(zip(mb.vertex_key.tolist(), map(tuple, v), strict=True))
    for key, p in zip(r.mesh.vertex_key.tolist(), r.v, strict=True):
        assert pos[key] == tuple(p)
    # orientation as the surface: normals out of the WM
    c = np.array([20.3, 19.6, 17.8])
    n = np.cross(r.v[r.f[:, 1]] - r.v[r.f[:, 0]], r.v[r.f[:, 2]] - r.v[r.f[:, 0]])
    ball = r.v[r.f].mean(1)[:, 0] < 34
    assert np.mean(np.sum(n[ball] * (r.v[r.f].mean(1)[ball] - c), 1) > 0) > 0.95


def test_touched_blocks_include_the_block_before_on_a_shared_face() -> None:
    pts = np.array([[8, 3, 3], [5, 16, 0], [12, 12, 12]])
    got = {tuple(b) for b in white_surf.touched_blocks(pts, 8).tolist()}
    assert got == {(1, 0, 0), (0, 0, 0), (0, 2, 0), (0, 1, 0), (1, 1, 1)}
    assert len(white_surf.touched_blocks(np.zeros((0, 3), np.int64), 8)) == 0
