"""Tests for the white surface's overview (baillarger.white.write_white_overview,
WhiteOverview): written by the mesh command from a synthetic segmentation; its
guarantees, its files as ordinary GIfTI, its reader and check, and the overview
command."""

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.gifti.gifti import GiftiImage

from baillarger import decimate as D
from baillarger import white
from baillarger.mesh import topology

AFF = np.array([[0.2, 0, 0, -3.0], [0, 0.16, 0, 2.0], [0, 0, 0.2, 5.0], [0, 0, 0, 1.0]])
EPS = 0.05


@pytest.fixture(scope="module")
def surf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A right hemisphere: a WM ball round a ventricle, in cortex, and a separate
    WM block, through the dseg and mesh commands (which writes the overview)."""
    d = tmp_path_factory.mktemp("ov")
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
    anat = d / "deriv/sub-T01/ses-S1/anat"
    dseg = anat / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    out = str(d / "deriv")
    args = ["--block-cells", "8", "--overview-eps", str(EPS), "--workers", "1"]
    white.main(["mesh", str(dseg), "--out", out, *args])
    return anat / "sub-T01_ses-S1_hemi-R_white.surf.gii"


def true_mesh(surf: Path) -> tuple[np.ndarray, np.ndarray]:
    pts, tri = GiftiImage.from_filename(str(surf)).agg_data(("pointset", "triangle"))
    return np.asarray(pts), np.asarray(tri, np.int64)


def test_the_files_and_the_guarantees(surf: Path) -> None:
    p = white.overview_paths(surf)
    assert all(x.exists() for x in p.values())
    side = json.loads(p["sidecar"].read_text())
    white.OVERVIEW_SPEC.check(side)
    # ordinary GIfTI: embedded data, read by nibabel with nothing special
    g = GiftiImage.from_filename(str(p["overview"]))
    assert all(da.encoding == 3 for da in g.darrays)  # GZipBase64Binary
    pts, tri = g.agg_data(("pointset", "triangle"))
    ids = GiftiImage.from_filename(str(p["ids"])).darrays[0].data.astype(np.int64)
    tv, tf = true_mesh(surf)
    assert (side["Vertices"], side["Faces"]) == (len(pts), len(tri))
    assert (side["TrueVertices"], side["TrueFaces"]) == (len(tv), len(tf))
    assert len(tri) < 0.5 * len(tf)
    # the vertices are true vertices, bit for bit, in increasing order
    assert np.all(np.diff(ids) > 0)
    np.testing.assert_array_equal(pts, tv[ids])
    # the topology of the true surface
    tri = np.asarray(tri, np.int64)
    assert topology(tri, len(pts)) == topology(tf, len(tv))
    assert [q["genus"] for q in side["Pieces"]] == [0, 0]
    # every true vertex within EPS of the overview (brute force, mm)
    v = tv.astype(np.float64)
    w = np.asarray(pts, np.float64)
    best = np.full(len(v), np.inf)
    for t in tri:
        a, b, c = (np.broadcast_to(w[i], v.shape) for i in t)
        best = np.minimum(best, D.triangle_distances(v, a, b, c))
    assert best.max() <= EPS + 1e-6
    assert side["MaxDistanceMM"] <= side["EpsMM"] == EPS


def test_the_reader_and_its_check(surf: Path, tmp_path: Path) -> None:
    ov = white.WhiteOverview.load(white.overview_paths(surf)["overview"])
    W = white.WhiteSurface.load(surf)
    ov.check(W)
    np.testing.assert_array_equal(ov.v, W.v[ov.vertex_ids])
    assert np.allclose(ov.affine, AFF)
    # another surface's data: refused
    d = tmp_path / "other"
    shutil.copytree(surf.parent, d)
    sc = d / surf.name.replace(".surf.gii", ".json")
    side = json.loads(sc.read_text())
    side["DataSHA256"] = {k: "0" * 64 for k in side["DataSHA256"]}
    sc.write_text(json.dumps(side))
    with pytest.raises(ValueError, match="not made from"):
        ov.check(white.WhiteSurface.load(d / surf.name))


def test_the_overview_command_writes_the_same_overview(
    surf: Path, tmp_path: Path
) -> None:
    d = tmp_path / "again"
    shutil.copytree(surf.parent, d)
    s2 = d / surf.name
    for x in white.overview_paths(s2).values():
        x.unlink()
    white.main(["overview", str(s2), "--eps", str(EPS), "--workers", "1"])
    a = white.WhiteOverview.load(white.overview_paths(surf)["overview"])
    b = white.WhiteOverview.load(white.overview_paths(s2)["overview"])
    np.testing.assert_array_equal(a.f, b.f)
    np.testing.assert_array_equal(a.vertex_ids, b.vertex_ids)
    np.testing.assert_array_equal(a.v, b.v)
