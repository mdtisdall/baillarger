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
from edits_helpers import write_dseg_sidecar

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


def test_the_wm_table() -> None:
    assert VV.colour_table("wm") == {"1": dict(name="wm_mask", rgba=[255, 149, 0, 255])}


def _blocks(a: np.ndarray) -> np.ndarray:
    """The 2 x 2 x 2 blocks of an array of even shape, as (nx/2, ny/2, nz/2, 8)."""
    nx, ny, nz = a.shape
    return (
        a.reshape(nx // 2, 2, ny // 2, 2, nz // 2, 2)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(nx // 2, ny // 2, nz // 2, 8)
    )


def _brick(base: str, name: str, level: int, shape: tuple[int, ...]) -> np.ndarray:
    """Brick (0, 0, 0) of a level whose shape fits one brick, as an array."""
    meta = json.loads(get(f"{base}/volume/{name}/meta"))
    dt = {"int16": "<i2", "int32": "<i4", "float32": "<f4"}[meta["dtype"]]
    b = np.frombuffer(get(f"{base}/volume/{name}/bricks?level={level}&b=0,0,0"), dt)
    return b.reshape(shape, order="F")


def test_the_echo_choice_and_the_wm_mask(tmp_path: Path) -> None:
    """--white mode's registry with echoes 1 to 3 and FilledLabels: each echo's
    pyramid is the file at level 0 and the block means at level 1; the echo that
    is the default volume is a second name of it (one pyramid, built once); the
    WM mask is 1 on the filled labels at level 0 and the block modes at level 1
    (a tie of 4 and 4 goes to 0)."""
    rng = np.random.default_rng(7)
    shape = (40, 34, 36)  # level 1 (20, 17, 18) fits one brick of 32
    d = tmp_path / "anat"
    d.mkdir()
    echoes: dict[int, Path] = {}
    data: dict[int, np.ndarray] = {}
    for k in (1, 2, 3):
        data[k] = rng.integers(0, 3000, shape).astype(np.int16)
        echoes[k] = d / f"echo-{k}.nii"
        nib.save(nib.nifti1.Nifti1Image(data[k], A0), echoes[k])
    seg = rng.choice(np.array([0, 2, 41, 42, 49, 60], np.int16), shape)
    filled = (41, 49, 60)
    dseg = d / "s_desc-wmtopo_dseg.nii.gz"
    fixes = d / "s_desc-wmtopofixes_dseg.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(seg, A0), dseg)
    nib.save(nib.nifti1.Nifti1Image(np.zeros(shape, np.uint8), A0), fixes)
    built: list[str] = []

    def make() -> VB.BrickSource:
        built.append("echo")
        return VB.ArrayPyramid.from_nifti(echoes[2])

    lazy = VB.LazyVolume(make, echoes[2].name)
    vs = VV.for_white(
        d / "s_hemi-R_white.surf.gii",
        [str(dseg)],
        lazy,
        "int16",
        filled=filled,
        echoes=echoes,
        echo_path=echoes[2],
    )
    srv = VS.serve(None, 0, white=object(), volume=lazy, volumes=vs)  # type: ignore[arg-type]
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ls = {v["name"]: v for v in json.loads(get(base + "/volume/list"))}
        assert list(ls) == ["echo", "echo1", "echo2", "echo3", "labels", "wm", "fixes"]
        assert ls["echo2"]["same_as"] == "echo"
        assert all("same_as" not in ls[n] for n in ls if n != "echo2")
        assert ls["wm"]["kind"] == "labels" and ls["wm"]["table"] == "wm"
        for k in (1, 2, 3):
            n = f"echo{k}"
            lv = json.loads(get(f"{base}/volume/{n}/meta"))["levels"]
            assert [x["shape"] for x in lv] == [list(shape), [20, 17, 18]]
            a0 = _brick(base, n, 0, (32, 32, 32))
            assert np.array_equal(a0, data[k][:32, :32, :32])
            a1 = _brick(base, n, 1, (20, 17, 18))
            mean = _blocks(data[k].astype(np.float64)).mean(-1)
            assert np.array_equal(a1, np.rint(mean).astype(np.int16))
        assert built == ["echo"]  # echo 2 and the default volume: one pyramid
        assert get(base + "/volume/echo2/bricks?level=1&b=0,0,0") == get(
            base + "/volume/bricks?level=1&b=0,0,0"
        )
        mw = json.loads(get(base + "/volume/wm/meta"))
        assert mw["table"] == VV.colour_table("wm")
        m0 = _brick(base, "wm", 0, (32, 32, 32))
        mask = np.isin(seg, filled)
        assert np.array_equal(m0, mask[:32, :32, :32].astype(m0.dtype))
        m1 = _brick(base, "wm", 1, (20, 17, 18))
        ones = _blocks(mask.astype(np.int64)).sum(-1)
        assert np.array_equal(m1, (ones > 4).astype(m1.dtype))
        assert (ones == 4).any() and (ones > 4).any()  # the tie rule is tested
    finally:
        srv.shutdown()
        VS.Handler.white = None
        VS.Handler.volume = None
        VS.Handler.volumes = None


def test_no_filled_labels_no_mask_and_no_alias_without_the_default(
    tmp_path: Path,
) -> None:
    """Without FilledLabels there is no WM mask; an echo that is not the default
    volume's file is its own pyramid; an alias needs a free name."""
    d = tmp_path
    dseg = d / "s_desc-wmtopo_dseg.nii.gz"
    fixes = d / "s_desc-wmtopofixes_dseg.nii.gz"
    e1 = d / "e1.nii"
    for f in (dseg, fixes, e1):
        nib.save(nib.nifti1.Nifti1Image(np.zeros((2, 2, 2), np.int16), A0), f)
    vs = VV.for_white(
        d / "s.surf.gii", [str(dseg)], None, "", echoes={1: e1}, echo_path=None
    )
    assert [v["name"] for v in vs.list()] == ["echo1", "labels", "fixes"]
    assert vs.same_as == {}
    with pytest.raises(ValueError):
        vs.alias("labels", "echo1")


def test_the_input_labels_are_a_volume(tmp_path: Path) -> None:
    """The input labels (the first of the dseg sidecar's Sources) are the volume
    "input", with the colours of "labels"; when they were moved, the file of the
    same name next to the surface; a dseg with no sidecar has no "input"."""
    d = tmp_path / "anat"
    d.mkdir()
    rng = np.random.default_rng(3)
    shape = (6, 5, 4)
    seg = np.full(shape, 41, np.int16)
    inp = rng.choice(np.array([0, 41, 42], np.int16), shape)
    dseg = d / "s_desc-wmtopo_dseg.nii.gz"
    fixes = d / "s_desc-wmtopofixes_dseg.nii.gz"
    src = tmp_path / "excised" / "s_desc-excised_dseg.nii.gz"
    src.parent.mkdir()
    nib.save(nib.nifti1.Nifti1Image(seg, A0), dseg)
    nib.save(nib.nifti1.Nifti1Image(np.zeros(shape, np.uint8), A0), fixes)
    nib.save(nib.nifti1.Nifti1Image(inp, A0), src)
    surf = d / "s.surf.gii"
    assert "input" not in VV.for_white(surf, [str(dseg)], None, "").items
    write_dseg_sidecar(dseg, src)
    vs = VV.for_white(surf, [str(dseg)], None, "")
    assert [v["name"] for v in vs.list()] == ["labels", "input", "fixes"]
    meta = vs.meta("input")
    assert meta["kind"] == "labels" and meta["table"] == VV.colour_table("labels")
    a = vs.items["input"].volume.source().arrays[0]  # type: ignore[attr-defined]
    assert np.array_equal(np.asarray(a), inp)
    # moved: the file of the same name next to the surface
    src.rename(d / src.name)
    vs = VV.for_white(surf, [str(dseg)], None, "")
    a = vs.items["input"].volume.source().arrays[0]  # type: ignore[attr-defined]
    assert np.array_equal(np.asarray(a), inp)
