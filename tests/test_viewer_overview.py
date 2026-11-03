"""Tests of baillarger.viewer.overview and baillarger.viewer.white_view on a
surface and overview written by the mesh command from a synthetic segmentation:
the Surface on the overview, its cache, and the refusals; the server viewing the
surface alone (--white)."""

import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from baillarger import white
from baillarger.viewer import overview as OV
from baillarger.viewer import server as VS
from baillarger.viewer.white_view import WhiteView

AFF = np.array([[0.2, 0, 0, -3.0], [0, 0.16, 0, 2.0], [0, 0, 0.2, 5.0], [0, 0, 0, 1.0]])
EPS = 0.05


@pytest.fixture(scope="module")
def surf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A right hemisphere: a WM ball round a ventricle, in cortex, and a separate
    WM block, through the dseg and mesh commands (with its overview)."""
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
    args = ["--block-cells", "8", "--overview-eps", str(EPS), "--workers", "1"]
    white.main(["mesh", str(dseg), "--out", str(d / "deriv"), *args])
    return anat / "sub-T01_ses-S1_hemi-R_white.surf.gii"


def test_a_surface_on_the_overview(surf: Path, tmp_path: Path) -> None:
    S, ov, W = OV.overview_surface(surf, tmp_path, log=lambda m: None)
    # its vertices are the true vertices, in the true surface's coordinates
    np.testing.assert_array_equal(S.v_vox, np.asarray(W.v[ov.vertex_ids], float))
    assert np.allclose(S.vox, [0.2, 0.16, 0.2]) and np.allclose(S.affine, AFF)
    # closed: inflated without collapse, and no flat map
    assert len(S.infl) == 4 and np.isnan(S.uv_v).all()
    # picking a vertex's position gives that vertex
    k = 7
    assert S.nearest(S.world(S.v_vox[k])[0]) == k
    # the inflation is cached under the overview file's hash
    assert (tmp_path / f"inflate_{OV.surface_name(surf)}_overview.npz").exists()
    S2, _, _ = OV.overview_surface(surf, tmp_path, log=lambda m: None)
    assert S2.key == S.key
    for a, b in zip(S.infl, S2.infl, strict=True):
        np.testing.assert_array_equal(a, b)


def test_no_overview_or_another_surface_is_refused(surf: Path, tmp_path: Path) -> None:
    d = tmp_path / "copy"
    shutil.copytree(surf.parent, d)
    s2 = d / surf.name
    # another surface's data
    sc = d / surf.name.replace(".surf.gii", ".json")
    side = json.loads(sc.read_text())
    good = dict(side["DataSHA256"])
    side["DataSHA256"] = {k: "0" * 64 for k in good}
    sc.write_text(json.dumps(side))
    with pytest.raises(ValueError, match="not made from"):
        OV.overview_surface(s2, tmp_path, log=lambda m: None)
    side["DataSHA256"] = good
    sc.write_text(json.dumps(side))
    # no overview written
    white.overview_paths(s2)["overview"].unlink()
    with pytest.raises(FileNotFoundError, match="baillarger.white overview"):
        OV.overview_surface(s2, tmp_path, log=lambda m: None)


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=10) as r:
        out: bytes = r.read()
    return out


def test_the_server_views_a_white_surface_alone(surf: Path, tmp_path: Path) -> None:
    w = WhiteView(surf, tmp_path, log=lambda m: None)
    srv = VS.serve(None, 0, white=w)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        meta = json.loads(get(base + "/meta"))
        assert meta["surface_only"] and meta["n"] == 0 and meta["layers"] == []
        assert meta["levels"][0] == "true surface" and len(meta["levels"]) == 4
        assert meta["overview"]["faces"] == len(w.ov.f)
        assert meta["overview"]["true_faces"] == len(w.W.f)
        b = get(base + "/mesh?level=2&faces=1")
        hl = int(np.frombuffer(b[:4], np.uint32)[0])
        hdr = json.loads(b[4 : 4 + hl])
        nv, nf = len(w.ov.vertex_ids), len(w.ov.f)
        assert (hdr["nv"], hdr["nf"], hdr["faces"]) == (nv, nf, True)
        vt = np.frombuffer(b, np.int32, nv, 4 + hl + 4 * 7 * nv)
        assert (vt == -1).all()  # no traces
        # the WM selection: an overview vertex and the true vertex it is
        k = 5
        sel = json.loads(get(base + f"/vertex?v={k}"))
        t = sel["true_vertex"]
        assert sel["vertex"] == k and t == int(w.ov.vertex_ids[k])
        assert np.allclose(sel["world_mm"], w.S.world(w.W.v[t])[0], atol=1e-3)
        mm = ",".join(str(x) for x in sel["world_mm"])
        assert json.loads(get(base + f"/snap?mm={mm}"))["true_vertex"] == t
        assert len(get(base + "/white.gii")) > 0
        assert meta["box"]["half"] == w.half and len(meta["box"]["centre"]) == 3
        assert b"<html" in get(base + "/")[:200].lower()
        for path in ("/section?i=0", "/layer?name=x", "/filters", "/echo2.nii.gz"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(base + path)
            assert e.value.code == 404
    finally:
        srv.shutdown()
        VS.Handler.white = None


def echoes(d: Path, shape: tuple[int, ...]) -> Path:
    """Echo 1-3 on the surface's grid in d: echo k's value is k times its linear
    index (int32)."""
    from baillarger.data import echo_path

    d.mkdir(parents=True, exist_ok=True)
    for k in (1, 2, 3):
        a = (k * np.arange(int(np.prod(shape)))).reshape(shape).astype(np.int32)
        nib.save(nib.nifti1.Nifti1Image(a, AFF), echo_path(k, d))
    return d


def test_the_volume_pane_box(surf: Path, tmp_path: Path) -> None:
    """The box round a voxel: the echoes cut from their grid with the box's
    affine, the overlay of the mask, the true surface's faces with a corner in
    it (all of them), and the box moved to a selection's voxel."""
    import gzip

    from baillarger import white

    W = white.WhiteSurface.load(surf)
    shape = tuple(W.sidecar["GridShape"])
    w = WhiteView(
        surf,
        tmp_path / "cache",
        log=lambda m: None,
        echo_dir=echoes(tmp_path / "echo", shape),
        box=6,
    )
    c = [int(x) for x in np.round(np.asarray(W.v[len(W.v) // 2], float))]
    lo, hi = w.box(c)
    np.testing.assert_array_equal(lo, np.clip(np.array(c) - 6, 0, shape))
    np.testing.assert_array_equal(hi, np.clip(np.array(c) + 7, 0, shape))
    assert w.box([0, 39, 35])[0].tolist() == [0, 33, 29]  # clipped to the grid
    sl = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi, strict=True))
    img = nib.Nifti1Image.from_bytes(gzip.decompress(w.echo_bytes(2, c)))
    full = (2 * np.arange(int(np.prod(shape)))).reshape(shape)
    np.testing.assert_array_equal(np.asarray(img.dataobj), full[sl])
    np.testing.assert_allclose(
        img.affine[:3, 3], (AFF @ [*lo.tolist(), 1])[:3], atol=1e-6
    )
    dseg = surf.with_name("sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz")
    seg = np.asarray(nib.nifti1.Nifti1Image.from_filename(dseg).dataobj)
    mask = np.isin(seg, W.sidecar["FilledLabels"])[sl]
    ov = w.overlay(c)
    np.testing.assert_array_equal(ov > 0, mask | (ov > 1))
    # every face of the true surface with a corner in the box, and no other
    f = w.surface_faces(c)
    v = np.asarray(W.v, float)
    corner_in = np.all((v >= lo - 0.5) & (v <= hi - 0.5), axis=1)
    want = W.f[corner_in[W.f].any(1)]
    assert len(f) == len(want) > 0
    assert {tuple(x) for x in f.tolist()} == {tuple(x) for x in want.tolist()}
    # a selection reports its voxel
    sel = w.vertex(0)
    assert sel["voxel"] == [int(x) for x in np.round(v[sel["true_vertex"]])]


def test_an_echo_on_another_grid_is_refused(surf: Path, tmp_path: Path) -> None:
    from baillarger.data import echo_path

    d = tmp_path / "echo"
    d.mkdir()
    nib.save(
        nib.nifti1.Nifti1Image(np.zeros((10, 10, 10), np.int16), AFF), echo_path(2, d)
    )
    w = WhiteView(surf, tmp_path / "cache", log=lambda m: None, echo_dir=d)
    with pytest.raises(ValueError, match="grid"):
        w.echo_bytes(2, [5, 5, 5])
