"""Tests for baillarger.white: the white mesh of a box's label, its pieces,
and the WM mask of a corrected segmentation (filled_labels). write_data
writes a small hemisphere (three echoes, an aseg and its corrected
segmentation) for the tests of the box stages that read it."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt

from baillarger import data, white
from baillarger import mesh_smoothing as S
from baillarger.handles import kuhn_betti
from baillarger.mesh import marching_tetrahedra, topology

SHAPE = (40, 36, 30)
VOX = (0.2, 0.16, 0.2)


def phantom_aseg() -> npt.NDArray[np.int16]:
    """An aseg with two WM components: a block drilled by a 2 x 2 tunnel (a
    fill of 4 voxels) and holding a ventricle, inside cortex; and a ring 5
    voxels wide closed by a one-voxel rod across a slit of background (a cut
    of 1)."""
    a = np.zeros(SHAPE, np.int16)
    a[8:24, 6:22, 4:20] = data.ASEG_CORTEX
    a[10:22, 8:20, 6:18] = data.ASEG_WM
    a[15:17, 13:15, 6:18] = data.ASEG_CORTEX  # the tunnel
    a[11:13, 9:11, 8:10] = 43  # a ventricle inside the WM
    a[25:37, 3:33, 22:28] = data.ASEG_WM  # 5 wide round a 2 x 20 hole
    a[30:32, 8:28, 22:28] = 0
    a[25:30, 16:18, 22:28] = 0  # the slit
    a[27, 16:18, 25] = data.ASEG_WM  # the rod across it
    return a


def write_data(d: Path) -> tuple[npt.NDArray[np.int16], Path]:
    """Three echoes (with sidecars) and phantom_aseg in d, as the example
    data, and its corrected segmentation in d / deriv. Returns the aseg and
    the corrected segmentation's path."""
    rng = np.random.default_rng(0)
    aff = np.diag([*VOX, 1.0])
    for k, te in enumerate((0.00937, 0.02071, 0.03205)):
        p = data.echo_path(k + 1, d)
        e = rng.integers(500, 3000, SHAPE).astype(np.int16)
        nib.save(nib.nifti1.Nifti1Image(e, aff), p)
        side = p.with_name(p.name.removesuffix(".nii.gz") + ".json")
        side.write_text(json.dumps({"EchoTime": te, "EchoNumber": k + 1}))
    a = phantom_aseg()
    nib.save(nib.nifti1.Nifti1Image(a, aff), data.aseg_path(d))
    return a, white.write_wm_topo(data.aseg_path(d), d / "deriv")["dseg"]


def read_labels(p: Path) -> npt.NDArray[np.int16]:
    img = nib.load(p)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    return np.asarray(img.dataobj).astype(np.int16)


def test_filled_labels_give_the_corrected_mask(tmp_path: Path) -> None:
    a, p = write_data(tmp_path)
    filled = white.filled_labels(p)
    assert filled == white.FILLED_LABELS["R"]
    mask = np.isin(read_labels(p), filled)
    assert kuhn_betti(np.isin(a, filled)) == (2, 2, 0)
    assert kuhn_betti(mask) == (2, 0, 0)
    v, f = marching_tetrahedra(mask.astype(np.float64), 0.5)
    assert topology(f, len(v))["genus"] == 0


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


def test_mesh_pieces_on_the_phantom() -> None:
    a = phantom_aseg()
    v, f = white.white_mesh(np.isin(a, white.FILLED_LABELS["R"]))
    ps = white.mesh_pieces(v, f, VOX)
    assert [p["genus"] for p in ps] == [1, 1]
    assert [p["boundary_loops"] for p in ps] == [0, 0]
    assert ps[0]["V"] >= ps[1]["V"] and sum(p["F"] for p in ps) == len(f)
    e1, e2 = (v[f[:, 1]] - v[f[:, 0]]) * VOX, (v[f[:, 2]] - v[f[:, 0]]) * VOX
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()
    assert abs(sum(p["area_mm2"] for p in ps) - area) < 1e-2
