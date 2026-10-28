"""Tests for baillarger.white: the handle-free WM label of a small synthetic
aseg, its changes on the full grid, and boxes cropped from it."""

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

from baillarger import data, white
from baillarger import mesh_smoothing as S
from baillarger.handles import kuhn_betti
from baillarger.mesh import marching_tetrahedra, topology

SHAPE = (40, 36, 30)
type Bool = npt.NDArray[np.bool_]


def write_aseg(d: Path) -> npt.NDArray[np.int16]:
    """An aseg with two WM components: a block drilled by a 2 x 2 tunnel (a
    fill of 4 voxels), and a ring 5 voxels wide closed by a one-voxel rod (a
    cut of 1)."""
    a = np.zeros(SHAPE, np.int16)
    a[8:24, 6:22, 4:20] = data.ASEG_CORTEX
    a[10:22, 8:20, 6:18] = data.ASEG_WM
    a[15:17, 13:15, 6:18] = data.ASEG_CORTEX
    a[25:37, 3:33, 22:28] = data.ASEG_WM  # 5 wide round a 2 x 20 hole
    a[30:32, 8:28, 22:28] = 0
    a[25:30, 16:18, 22:28] = 0
    a[27, 16:18, 25] = data.ASEG_WM
    nib.save(
        nib.nifti1.Nifti1Image(a, np.diag([0.2, 0.16, 0.2, 1.0])), data.aseg_path(d)
    )
    return a


def test_changes_on_a_phantom_aseg(tmp_path: Path) -> None:
    a = write_aseg(tmp_path)
    ch = white.wm_changes(tmp_path)
    assert ch.betti_before == (2, 2, 0) and ch.betti_after == (2, 0, 0)
    fill = ch.voxels[ch.kind == white.FILL]
    cut = ch.voxels[ch.kind == white.CUT]
    assert len(ch.voxels) == 5 and not (ch.kind == white.STRANDED).any()
    # the plug of the tunnel: 2 x 2 voxels in one slice, full-grid coordinates
    assert len(fill) == 4 and len(np.unique(fill[:, 2])) == 1
    assert {tuple(p[:2]) for p in fill} == {(15, 13), (15, 14), (16, 13), (16, 14)}
    assert all(a[tuple(p)] == data.ASEG_CORTEX for p in fill)
    # one voxel of the rod
    assert len(cut) == 1 and cut[0][0] == 27 and cut[0][2] == 25
    assert a[tuple(cut[0])] == data.ASEG_WM
    assert len(np.unique(ch.move)) == 2


def test_boxes_apply_the_changes(tmp_path: Path) -> None:
    a = write_aseg(tmp_path)
    ch = white.wm_changes(tmp_path)
    wm = white.white_matter_box((0, 0, 0), SHAPE, ch, tmp_path)
    expected = a == data.ASEG_WM
    expected[tuple(ch.voxels[ch.kind == white.FILL].T)] = True
    expected[tuple(ch.voxels[ch.kind != white.FILL].T)] = False
    assert np.array_equal(wm, expected)
    assert kuhn_betti(wm) == (2, 0, 0)
    v, f = marching_tetrahedra(wm.astype(np.float64), 0.5)
    assert topology(f, len(v))["genus"] == 0
    for lo, hi in (((12, 10, 4), (20, 30, 25)), ((26, 0, 20), (40, 36, 30))):
        box = white.white_matter_box(lo, hi, ch, tmp_path)
        assert np.array_equal(
            box, expected[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        )


def test_save_and_load(tmp_path: Path) -> None:
    write_aseg(tmp_path)
    ch = white.wm_changes(tmp_path)
    p = tmp_path / "wm_changes.npz"
    ch.save(p)
    back = white.WMChanges.load(p)
    assert back.betti_before == ch.betti_before and back.betti_after == ch.betti_after
    for k in ("voxels", "kind", "move"):
        assert np.array_equal(getattr(back, k), getattr(ch, k))


def test_box_outside_the_grid_raises(tmp_path: Path) -> None:
    write_aseg(tmp_path)
    ch = white.wm_changes(tmp_path)
    with pytest.raises(ValueError):
        white.white_matter_box((30, 0, 0), (41, 10, 10), ch, tmp_path)


# ---------------------------------------------------------------- the mesh


def test_white_mesh_of_a_ball() -> None:
    n, c, r = 30, np.array([14.3, 14.6, 15.1]), 9.0
    g = np.indices((n, n, n)).astype(float)
    wm = ((g - c[:, None, None, None]) ** 2).sum(0) <= r * r
    v, f = white.white_mesh(wm)
    v0, f0 = marching_tetrahedra(wm.astype(np.float64), 0.5)
    np.testing.assert_array_equal(f, f0[:, [0, 2, 1]])
    np.testing.assert_array_equal(v, S.twice(S.relax(v0, f), f, 2.0))
    assert topology(f, len(v))["genus"] == 0
    assert np.all(np.sum(S.vertex_normals(v, f) * (v - c), 1) > 0)  # out of the WM


def test_mesh_pieces_on_the_phantom(tmp_path: Path) -> None:
    a = write_aseg(tmp_path)
    vox = (0.2, 0.16, 0.2)
    v, f = white.white_mesh(a == data.ASEG_WM)
    ps = white.mesh_pieces(v, f, vox)
    assert [p["genus"] for p in ps] == [1, 1] and [p["boundary_loops"] for p in ps] == [
        0,
        0,
    ]
    assert ps[0]["V"] >= ps[1]["V"] and sum(p["F"] for p in ps) == len(f)
    e1, e2 = (v[f[:, 1]] - v[f[:, 0]]) * vox, (v[f[:, 2]] - v[f[:, 0]]) * vox
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()
    assert abs(sum(p["area_mm2"] for p in ps) - area) < 1e-2


# ---------------------------------------------------------------- the reports


def test_changes_summary(tmp_path: Path) -> None:
    write_aseg(tmp_path)
    ch = white.wm_changes(tmp_path)
    s = white.changes_summary(ch, (0.2, 0.16, 0.2))
    assert s["betti_before"] == [2, 2, 0] and s["betti_after"] == [2, 0, 0]
    assert (s["voxels_removed"], s["voxels_added"], s["stranded"]) == (1, 4, 0)
    assert s["by_kind"]["fill"] == dict(moves=1, voxels=4)
    fill = next(m for m in s["moves"] if m["kind"] == "fill")
    assert fill["extent_vox"][:2] == [2, 2] and fill["extent_vox"][2] == 1
    assert fill["centre_itksnap"] == [c + 1 for c in fill["centre_full0"]]
    assert fill["volume_mm3"] == pytest.approx(4 * 0.2 * 0.16 * 0.2, abs=1e-4)


def test_box_report(tmp_path: Path) -> None:
    write_aseg(tmp_path)
    ch = white.wm_changes(tmp_path)
    r = white.box_report((0, 0, 0), SHAPE, ch, tmp_path)
    assert (r["voxels_removed"], r["voxels_added"]) == (1, 4)
    assert [p["genus"] for p in r["pieces_before"]] == [1, 1]
    assert [p["genus"] for p in r["pieces_after"]] == [0, 0]
    assert sorted(m["kind"] for m in r["moves"]) == ["cut", "fill"]
    # a box round the block only: its plug, and a face distance in mm
    lo, hi = (5, 3, 2), (25, 25, 22)
    r = white.box_report(lo, hi, ch, tmp_path)
    assert [m["kind"] for m in r["moves"]] == ["fill"] and r["voxels_added"] == 4
    m = r["moves"][0]
    assert m["centre_full0"] == [
        p + q for p, q in zip(m["centre_box0"], lo, strict=True)
    ]
    assert m["face_distance_mm"] > 0
    ov = white.overlay(ch, lo, hi)
    assert ov.shape == (20, 22, 20) and (ov == white.FILL).sum() == 4 and ov.max() == 2


def test_command_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_aseg(tmp_path)
    monkeypatch.setenv("BAILLARGER_DATA", str(tmp_path))
    out = tmp_path / "out"
    white.main(["changes", "--out", str(out)])
    for name in ("wm_changes.npz", "wm_changes.json", "wm_changes_overlay.nii.gz"):
        assert (out / name).is_file()
    white.main(
        [
            "box",
            "--changes",
            str(out / "wm_changes.npz"),
            "--out",
            str(out),
            "--name",
            "blk",
            "--lo",
            "5",
            "3",
            "2",
            "--hi",
            "25",
            "25",
            "22",
        ]
    )
    rep = json.loads((out / "wm_box_blk.json").read_text())
    assert rep["voxels_added"] == 4
    img = nib.load(out / "wm_box_blk_wm_fixed.nii.gz")
    assert isinstance(img, nib.nifti1.Nifti1Image)
    fixed = np.asarray(img.dataobj)
    assert fixed.shape == (20, 22, 20)
    assert np.array_equal(
        fixed.astype(bool),
        white.white_matter_box(
            (5, 3, 2), (25, 25, 22), white.WMChanges.load(out / "wm_changes.npz")
        ),
    )


def ball(shape: tuple[int, int, int] = (20, 22, 24), r: float = 6.0) -> Bool:
    c = (np.array(shape) - 1) / 2
    idx = np.indices(shape).reshape(3, -1).T
    out: Bool = (np.linalg.norm(idx - c, axis=1) <= r).reshape(shape)
    return out


def test_label_sha256_tells_labels_apart() -> None:
    a = ball()
    assert white.label_sha256(a) == white.label_sha256(a.copy())
    b = a.copy()
    b[0, 0, 0] = True
    assert white.label_sha256(b) != white.label_sha256(a)
    # the same voxels in C order, on another shape
    assert white.label_sha256(a.reshape(22, 20, 24)) != white.label_sha256(a)


def test_box_mesh_is_the_traces_mesh_and_survives_save_and_load(
    tmp_path: Path,
) -> None:
    wm = ball()
    lo, hi = (3, 4, 5), (23, 26, 29)
    aff = np.diag([0.2, 0.16, 0.2, 1.0])
    m = white.box_mesh(wm, lo, hi, aff, provenance={"wm_label": "test"})
    v, f, n = white.white_mesh_normals(wm)
    assert np.array_equal(m.v, v) and np.array_equal(m.f, f)
    assert np.array_equal(m.normals, n)
    m.save(tmp_path / "m.npz")
    r = white.WhiteMesh.load(tmp_path / "m.npz")
    for k in ("v", "f", "normals", "affine"):
        assert np.array_equal(getattr(r, k), getattr(m, k))
    assert (r.lo, r.hi, r.sigma) == (lo, hi, 2.0)
    assert r.wm_sha256 == white.label_sha256(wm)
    assert r.provenance == {"wm_label": "test"}
    with pytest.raises(ValueError, match="not on the box"):
        white.box_mesh(wm, lo, (23, 26, 30), aff)


def test_check_refuses_another_box_or_label() -> None:
    wm = ball()
    lo, hi = (0, 0, 0), wm.shape
    m = white.box_mesh(wm, lo, hi, np.eye(4))
    m.check(lo, hi, wm.copy())
    with pytest.raises(ValueError, match="not the mesh's"):
        m.check((1, 0, 0), (21, 22, 24), wm)
    other = wm.copy()
    other[10, 11, 12] = False
    with pytest.raises(ValueError, match="WM label"):
        m.check(lo, hi, other)


def test_mesh_command_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = write_aseg(tmp_path)
    monkeypatch.setenv("BAILLARGER_DATA", str(tmp_path))
    out = tmp_path / "out"
    white.main(["changes", "--out", str(out)])
    ch_path = out / "wm_changes.npz"
    lo, hi = (5, 3, 2), (25, 25, 22)
    box = ["--lo", "5", "3", "2", "--hi", "25", "25", "22"]
    white.main(
        ["mesh", "--changes", str(ch_path), "--out", str(out)] + ["--name", "blk", *box]
    )
    m = white.WhiteMesh.load(out / "wm_mesh_blk.npz")
    wm = white.white_matter_box(lo, hi, white.WMChanges.load(ch_path))
    m.check(lo, hi, wm)
    v, f, n = white.white_mesh_normals(wm)
    assert np.array_equal(m.v, v) and np.array_equal(m.f, f)
    assert np.array_equal(m.normals, n)
    assert np.allclose(m.affine, data.crop_affine(np.diag([0.2, 0.16, 0.2, 1.0]), lo))
    p = m.provenance
    assert p["wm_label"] == "handle-free"
    assert p["changes"]["sha256"] == hashlib.sha256(ch_path.read_bytes()).hexdigest()
    rep = json.loads((out / "wm_mesh_blk.json").read_text())
    assert [t["genus"] for t in rep["pieces"]] == [0]
    assert rep["vertices"] == len(v) and rep["provenance"] == p
    assert (out / "wm_mesh_blk.vtk").read_text().count("POINTS") == 1
    # without --changes: the aseg's WM label
    white.main(["mesh", "--out", str(out), "--name", "aseg", *box])
    m0 = white.WhiteMesh.load(out / "wm_mesh_aseg.npz")
    sl = tuple(slice(x, y) for x, y in zip(lo, hi, strict=True))
    m0.check(lo, hi, a[sl] == data.ASEG_WM)
    assert m0.provenance["wm_label"] == "aseg" and m0.provenance["changes"] is None
    with pytest.raises(SystemExit):
        white.main(["mesh", "--out", str(out), "--name", "x", "--patch", "V1V2", *box])
