"""Tests of baillarger.viewer.data and baillarger.viewer.server on a synthetic site,
with exact answers."""

import gzip
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial import cKDTree

from baillarger.data import crop_affine
from baillarger.viewer import data as VD
from baillarger.viewer import flatmap as FM
from baillarger.viewer import server as VS
from baillarger.viewer import streamlines as SV

type Float = npt.NDArray[np.float64]

LO = np.array([5, 6, 7])
SHAPE = (20, 20, 20)
VOX = np.array([0.5, 0.5, 0.5])
STEP = 0.04


def write_site(tmp: Path, z0: float = 4.0) -> tuple[Path, Path]:
    """A plane of white mesh at box z = z0 (voxels), one root per vertex, traces
    straight up 1 mm (box +z); the cortex label above the plane; echo 2 of a full grid
    of 40^3 whose voxel values are their own linear index. Returns (column volume,
    echo)."""
    nx = ny = 9
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    v = np.c_[i.ravel() * 2.0 + 1, j.ravel() * 2.0 + 1, np.full(nx * ny, z0)]
    F = []
    for a in range(nx - 1):
        for b in range(ny - 1):
            p, q, r, s = (
                a * ny + b,
                (a + 1) * ny + b,
                (a + 1) * ny + b + 1,
                a * ny + b + 1,
            )
            F += [(p, q, r), (p, r, s)]
    f = np.array(F, np.int64)
    n = len(v)
    ns = np.full(n, 26)
    s = np.arange(201) * STEP / VOX[2]
    pos = v[:, None, :] + s[None, :, None] * np.array([0, 0, 1.0])
    pos[np.arange(201)[None, :] >= ns[:, None]] = np.nan
    ctx = np.zeros(SHAPE, bool)
    ctx[:, :, int(z0) + 1 : int(z0) + 4] = True
    cv = tmp / "column_volume_synth.npz"
    np.savez_compressed(
        cv,
        v=v.astype(np.float32),
        f=f,
        nrm=np.tile([0, 0, 1.0], (n, 1)).astype(np.float32),
        tv=np.arange(n),
        pos=pos.astype(np.float32),
        ns=ns,
        fv=np.ones((2, n, 201), np.float32),
        cls=(np.arange(n) % 5).astype(np.int8),
        tracer=(np.arange(n) % 2).astype(np.int8),
        ctx=ctx,
        wm=np.zeros(SHAPE, bool),
        P=np.array([[3.0, 3.0, z0 + 1.0]]),
        vox=VOX,
        lo=LO,
        H_top=np.full(n, np.nan, np.float32),
        tracer_name=np.array("fitted_blend"),
        wm_label=np.array("aseg"),
    )
    full = np.arange(40**3, dtype=np.float32).reshape(40, 40, 40)
    aff = np.diag([0.5, 0.5, 0.5, 1.0])
    aff[:3, 3] = [-10.0, 3.0, 1.5]
    ep = tmp / "echo2.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(full, aff), ep)
    return cv, ep


@pytest.fixture
def site(tmp_path: Path) -> VD.Site:
    cv, ep = write_site(tmp_path)
    return VD.Site(cv, tmp_path / "cache", echo=ep, log=lambda m: None)


def test_site_reads_the_column_volume_and_names_its_tracer(site: VD.Site) -> None:
    assert site.name == "synth" and site.n == 81
    assert site.tracer_name == "fitted_blend" and site.wm_label == "aseg"
    assert site.z is None and site.meta()["has_levels"] is False
    assert (site.vtrace >= 0).all()  # every vertex is a root


def test_flat_map_and_inflation_are_cached_by_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cv, ep = write_site(tmp_path)
    a = VD.Site(cv, tmp_path / "cache", echo=ep, log=lambda m: None)
    assert (tmp_path / "cache" / "flat_map_synth.npz").exists()

    def fail(*args: Any, **kw: Any) -> Any:
        raise AssertionError("recomputed")

    monkeypatch.setattr(FM, "site_flat_map", fail)
    b = VD.Site(cv, tmp_path / "cache", echo=ep, log=lambda m: None)
    assert np.array_equal(a.uv, b.uv, equal_nan=True)
    # another mesh: the cache does not match, so it is computed again
    cv2, _ = write_site(tmp_path, z0=5.0)
    with pytest.raises(AssertionError, match="recomputed"):
        VD.Site(cv2, tmp_path / "cache", echo=ep, log=lambda m: None)


def test_world_is_the_box_crop_of_the_echo_affine(
    site: VD.Site, tmp_path: Path
) -> None:
    aff = nib.load(tmp_path / "echo2.nii.gz").affine  # type: ignore[attr-defined]
    A = crop_affine(aff, (5, 6, 7))
    p = site.v_vox[7]
    assert np.allclose(site.world(p)[0], A[:3, :3] @ p + A[:3, 3])
    root = site.world(site.v_vox[site.rv[7]])[0]
    assert site.locate(root + [0.3, 0, 0]) == 7
    assert site.locate(root + [0, 0, 5.0]) is None


def test_snap_is_the_nearest_surface_vertex_and_its_trace(site: VD.Site) -> None:
    # vertices are at odd box voxels (1, 3, ..., 17) in x and y, z = 4: a click above
    # (5.2, 7.1, 9) is nearest (5, 7, 4), vertex 2 * 9 + 3, the root of trace 21
    sel = site.snap(site.world([5.2, 7.1, 9.0])[0])
    assert sel["vertex"] == 21 and sel["trace"] == 21
    assert np.allclose(sel["world_mm"], site.world([5.0, 7.0, 4.0])[0], atol=1e-3)
    assert sel["root_mm"] == pytest.approx(0.0)
    assert site.vertex(21) == sel


def test_a_vertex_without_a_near_trace_takes_the_nearest_root(site: VD.Site) -> None:
    site.vtrace[:] = -1  # as if no shown trace were rooted within NEAREST_MM
    sel = site.vertex(21)
    assert sel["trace"] == 21 and sel["root_mm"] == pytest.approx(0.0)
    site.root_tree = cKDTree(site.world(site.v_vox[site.rv[:1]]))  # only trace 0
    far = site.vertex(80)  # (17, 17, 4) voxels, 8 * sqrt(2) * 2 * 0.5 mm from trace 0
    assert far["trace"] is None and far["root_mm"] is None


def test_volume_is_the_box_of_the_full_grid(site: VD.Site, tmp_path: Path) -> None:
    img = nib.nifti1.Nifti1Image.from_bytes(gzip.decompress(site.volume_bytes()))
    full = np.arange(40**3, dtype=np.float32).reshape(40, 40, 40)
    a, b = LO, LO + np.array(SHAPE)
    assert np.array_equal(img.get_fdata(), full[a[0] : b[0], a[1] : b[1], a[2] : b[2]])
    assert np.allclose(img.affine, site.affine)


def test_sections_and_streamlines_without_a_joint_result(site: VD.Site) -> None:
    i = 4 * 9 + 4  # the plane's centre vertex
    S = site.section_at(i, half=3.0)
    assert i in S["u"]["traces"] and i in S["v"]["traces"]
    assert S["u"]["levels"] == [] and S["info"]["pia_mm"] is None
    st = site.streamlines(i, radius=1.0)
    fr = [t for k, ts in st["_groups"].items() if k.startswith("from") for t in ts]
    near = np.linalg.norm(site.vmm - site.vmm[i], axis=1) <= 1.0
    assert set(fr) <= set(np.flatnonzero(near).tolist()) and i in fr
    assert st["n_to"] == 0  # every trace goes straight up: none ends on the plane
    lines = SV.read_tck(site.tracts([i]))
    want = site.world(site.pos_all[i, : site.ns_all[i]])
    assert len(lines) == 1 and np.allclose(lines[0], want, atol=1e-5)
    # the point (3, 3, 5) voxels is 0.5 mm up the trace from vertex (3, 3, 4), between
    # samples 12 and 13 (0.02 mm from each): the first is kept
    hits = site.point_hits()
    assert list(hits) == [10] and hits[10][0][0] == 1
    assert hits[10][0][1] == pytest.approx(0.48)


def test_rotated_sections_swap_u_and_v_and_follow_their_curves(
    site: VD.Site,
) -> None:
    i = 4 * 9 + 4  # the plane's centre vertex
    S0, S90 = site.section_at(i, half=3.0), site.section_at(i, half=3.0, rot=90.0)
    assert S90["u"]["traces"] == S0["v"]["traces"]
    assert S90["u"]["dist"] == S0["v"]["dist"]
    assert S90["v"]["traces"] == S0["u"]["traces"][::-1]
    assert S90["v"]["dist"] == [-d for d in S0["u"]["dist"][::-1]]
    # the curves lie on the plane (world z of box z = 4 voxels) and pass by every
    # trace of their section, within the section's 1.5 band
    band = 1.5 * 0.75 * float(site.vox.min())
    zw = site.world([0, 0, 4.0])[0, 2]
    for S in (S0, S90):
        for a in ("u", "v"):
            c = np.array(S["curves"][a])
            roots = site.world(site.v_vox[site.rv[S[a]["traces"]]])
            assert np.allclose(c[:, 2], zw, atol=1e-4)
            assert np.max(cKDTree(c).query(roots)[0]) <= band + 1e-9
    # the plane's normal is box z: the axial plane is the tangent plane (no snap);
    # the coronal plane holds the axis along box x
    assert S0["snap"]["axial"] is None
    tu = np.array(S0["curves"]["u"][-1]) - np.array(S0["curves"]["u"][0])
    along_x = abs(tu[0]) > abs(tu[1])
    co, sa = S0["snap"]["coronal"], S0["snap"]["sagittal"]
    assert co["tilt_deg"] == 0 and sa["tilt_deg"] == 0
    assert (co["axis"] == "u") == along_x and (sa["axis"] == "u") != along_x


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=5) as r:
        out: bytes = r.read()
        return out


def test_server_shares_state_between_pages(site: VD.Site) -> None:
    srv = VS.serve(site, 0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    try:
        meta = json.loads(get(base + "/meta"))
        assert meta["site"] == "synth" and meta["tracer"] == "fitted_blend"
        es = urllib.request.urlopen(base + "/events", timeout=5)
        hello = [es.readline() for _ in range(3)]
        assert hello[0] == b"event: hello\n"
        req2 = urllib.request.Request(
            base + "/state",
            data=json.dumps({"from": "pageA", "patch": dict(trace=40)}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req2, timeout=5).read()
        line = es.readline()
        msg = json.loads(line.decode().removeprefix("data: "))
        assert msg["sender"] == "pageA" and msg["patch"] == {"trace": 40}
        assert json.loads(get(base + "/state"))["state"]["trace"] == 40
        snap = json.loads(
            get(base + "/snap?mm=" + ",".join(map(str, site.world([5.2, 7.1, 9.0])[0])))
        )
        assert snap["vertex"] == 21 and snap["trace"] == 21
        assert json.loads(get(base + "/vertex?v=21")) == snap
        es.close()
        S = json.loads(get(base + "/section?i=40&half=3&rot=90"))
        assert S["rot"] == 90 and "curves" not in S
        k, npts = S["axes"]["id"], S["axes"]["points"]
        for a in ("u", "v"):
            c = SV.read_tck(get(base + f"/axes/{k}/{a}.tck"))
            assert len(c) == 1 and len(c[0]) == npts[a] > 2
        lines = SV.read_tck(get(base + "/overview/0/1.tck"))
        assert len(lines) == int((site.cls_all[site.sub] == 0).sum())
        with pytest.raises(urllib.error.HTTPError):
            get(base + "/static/../server.py")
    finally:
        srv.shutdown()
        srv.server_close()
