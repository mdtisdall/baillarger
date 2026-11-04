"""Tests of baillarger.viewer.volumes: the colour tables, the registry and its routes
on a real server (two volumes of different shapes, /volume/list, an unknown name, the
old /volume routes unchanged), and the paths of --white mode."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from baillarger.labels import LUT_NAMES
from baillarger.viewer import bricks as VB
from baillarger.viewer import server as VS
from baillarger.viewer import volumes as VV

A0 = np.diag([0.5, 0.5, 0.5, 1.0])


def test_the_label_table_has_every_label_of_the_set() -> None:
    t = VV.colour_table("labels")
    assert set(t) == {str(v) for v in LUT_NAMES if v != 0}
    for v, n in LUT_NAMES.items():
        if v:
            assert t[str(v)]["name"] == n
            assert t[str(v)]["rgba"] == [*VV.LABEL_RGB[v], 255]
    # FreeSurferColorLUT values (7.4.1)
    assert VV.LABEL_RGB[41] == (245, 245, 245)
    assert VV.LABEL_RGB[42] == (205, 62, 78)
    assert VV.LABEL_RGB[192] == (0, 0, 160)
    assert VV.LABEL_RGB[52] == (13, 48, 255) and VV.LABEL_RGB[13] == (12, 48, 255)
    assert VV.LABEL_RGB[63] == (0, 200, 221) and VV.LABEL_RGB[31] == (0, 200, 200)


def test_the_fixes_table() -> None:
    t = VV.colour_table("fixes")
    assert list(t) == ["1", "2", "3", "4", "5"]
    assert [t[k]["name"] for k in t] == [
        "handle_cut",
        "handle_fill",
        "cavity_fill",
        "manual_edit",
        "small_piece",
    ]
    assert len({tuple(t[k]["rgba"]) for k in t}) == 5


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=10) as r:
        out: bytes = r.read()
        return out


def test_named_volumes_on_the_server(tmp_path: Path) -> None:
    x, y, z = np.indices((5, 3, 7))
    echo = (21 * x + 7 * y + z).astype(np.int16)
    labels = np.where(x < 2, 41, 42).astype(np.uint8)[:4, :, :6]  # another shape
    built: list[str] = []

    def make_echo() -> VB.BrickSource:
        built.append("echo")
        return VB.ArrayPyramid(echo, A0, brick=2)

    lazy = VB.LazyVolume(make_echo, "echo test")
    vs = VV.Volumes()
    vs.add_lazy("echo", lazy, "int16")
    p = tmp_path / "l.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(labels, A0), p)
    vs.add_nifti("labels", p, "labels", "labels", brick=2)
    srv = VS.serve(None, 0, white=object(), volume=lazy, volumes=vs)  # type: ignore[arg-type]
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ls = json.loads(get(base + "/volume/list"))
        assert ls == [
            dict(name="echo", kind="image", table=None, dtype="int16", wire=None),
            dict(
                name="labels", kind="labels", table="labels", dtype="uint8", wire=None
            ),
        ]
        assert built == []
        old = get(base + "/volume/bricks?level=0&b=0,0,0;2,1,3")
        assert get(base + "/volume/echo/bricks?level=0&b=0,0,0;2,1,3") == old
        assert np.frombuffer(old, "<i2").tolist() == [0, 21, 7, 28, 1, 22, 8, 29, 104]
        assert built == ["echo"]  # one pyramid, shared by the old and new routes
        assert json.loads(get(base + "/volume/meta")) | {"kind": "image"} == json.loads(
            get(base + "/volume/echo/meta")
        )
        me = json.loads(get(base + "/volume/echo/meta"))
        assert me["kind"] == "image" and me["levels"][0]["shape"] == [5, 3, 7]
        assert "table" not in me
        ml = json.loads(get(base + "/volume/labels/meta"))
        assert ml["kind"] == "labels" and ml["dtype"] == "int16"
        assert [lv["shape"] for lv in ml["levels"]] == [[4, 3, 6], [2, 2, 3], [1, 1, 2]]
        assert ml["table"]["41"]["name"] == "Right-Cerebral-White-Matter"
        b = np.frombuffer(get(base + "/volume/labels/bricks?level=1&b=0,0,0"), "<i2")
        # level 1 brick (0, 0, 0): x 0..1 of level 1 (level 0 x 0..3): 41 then 42
        assert b.reshape(2, 2, 2, order="F")[:, 0, 0].tolist() == [41, 42]
        ls = json.loads(get(base + "/volume/list"))
        assert [v["wire"] for v in ls] == ["int16", "int16"]
        for bad, code in [
            ("/volume/nothing/meta", 404),
            ("/volume/labels/other", 404),
            ("/volume/labels/bricks?level=0&b=9,0,0", 400),
        ]:
            with pytest.raises(urllib.error.HTTPError) as e:
                get(base + bad)
            assert e.value.code == code
    finally:
        srv.shutdown()
        VS.Handler.white = None
        VS.Handler.volume = None
        VS.Handler.volumes = None


def test_no_registry_is_404() -> None:
    srv = VS.serve(None, 0, white=object())  # type: ignore[arg-type]
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for path in ("/volume/list", "/volume/labels/meta", "/outline?axis=0"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(base + path)
            assert e.value.code == 404
    finally:
        srv.shutdown()
        VS.Handler.white = None


def test_reserved_names_and_kinds() -> None:
    vs = VV.Volumes()
    for name in ("list", "meta", "bricks", "a/b", ""):
        with pytest.raises(ValueError):
            vs.add(name, lambda: VB.ArrayPyramid(np.zeros((2, 2, 2)), A0))
    with pytest.raises(ValueError):
        vs.add("x", lambda: VB.ArrayPyramid(np.zeros((2, 2, 2)), A0), "labels")


def test_white_paths_from_sources_or_next_to_the_surface(tmp_path: Path) -> None:
    d = tmp_path / "anat"
    d.mkdir()
    surf = d / "s_hemi-R_white.surf.gii"
    dseg = d / "s_desc-wmtopo_dseg.nii.gz"
    fixes = d / "s_desc-wmtopofixes_dseg.nii.gz"
    for f in (dseg, fixes):
        f.write_bytes(b"")
    # the source as written
    assert VV.white_paths(surf, [str(dseg)], None, None) == (dseg, fixes)
    # a moved tree: the same name next to the surface
    moved = tmp_path / "gone" / dseg.name
    assert VV.white_paths(surf, [str(moved)], None, None) == (dseg, fixes)
    # given paths win
    o = tmp_path / "o.nii.gz"
    assert VV.white_paths(surf, [str(moved)], o, o) == (o, o)
    with pytest.raises(FileNotFoundError):
        VV.white_paths(tmp_path / "x.surf.gii", [str(moved)], None, None)
