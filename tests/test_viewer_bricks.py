"""Tests of baillarger.viewer.bricks, with exact answers: the voxel <-> world mm mapping
of each level, the levels' shapes and values (the edge blocks of odd axes, int16
rounding), the bricks (inside, at the edges, on coarse levels) and their bytes; and the
server's /volume routes."""

import gzip
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from baillarger.viewer import bricks as VB
from baillarger.viewer import server as VS

# exact in binary floating point: every product and sum below is
A0 = np.array([[0.5, 0, 0, -10.0], [0, 0.25, 0, 5.0], [0, 0, 1.0, 2.0], [0, 0, 0, 1.0]])


def world(A: np.ndarray, v: list[float]) -> list[float]:
    return [float(x) for x in (A @ np.r_[v, 1.0])[:3]]


def test_voxel_to_mm_on_each_level() -> None:
    # level 0: the affine itself
    assert world(VB.level_affine(A0, 0), [1, 2, 3]) == [-9.5, 5.5, 5.0]
    # level 1: voxel (1, 2, 3) is the centre of level-0 voxels 2..3, 4..5, 6..7,
    # level-0 voxel (2.5, 4.5, 6.5)
    assert world(VB.level_affine(A0, 1), [1, 2, 3]) == [-8.75, 6.125, 8.5]
    # level 2: voxel (0, 0, 0) is level-0 voxel (1.5, 1.5, 1.5)
    assert world(VB.level_affine(A0, 2), [0, 0, 0]) == [-9.25, 5.375, 3.5]
    # and back: mm -> voxel by the inverse
    A1 = VB.level_affine(A0, 1)
    assert np.linalg.solve(A1, [-8.75, 6.125, 8.5, 1.0]).tolist() == [1, 2, 3, 1]


def test_an_oblique_affine_keeps_the_block_centres() -> None:
    R = np.array([[0, -0.5, 0, 1.0], [0.5, 0, 0, 2.0], [0, 0, 0.5, 3.0], [0, 0, 0, 1]])
    # level-1 voxel (1, 0, 0) is level-0 voxel (2.5, 0.5, 0.5)
    assert world(VB.level_affine(R, 1), [1, 0, 0]) == world(R, [2.5, 0.5, 0.5])
    assert world(R, [2.5, 0.5, 0.5]) == [0.75, 3.25, 3.25]


def test_level_shapes_halve_up_to_one_brick() -> None:
    assert VB.level_shapes((5, 3, 7), 2) == [(5, 3, 7), (3, 2, 4), (2, 1, 2)]
    assert VB.level_shapes((4, 4, 4), 4) == [(4, 4, 4)]
    assert VB.level_shapes((700, 384, 1148), 32)[-1] == (11, 6, 18)
    assert len(VB.level_shapes((700, 384, 1148), 32)) == 7
    with pytest.raises(ValueError):
        VB.level_shapes((4, 4), 2)


def vol() -> np.ndarray:
    """(5, 3, 7) int16 with value 21 x + 7 y + z at voxel (x, y, z)."""
    x, y, z = np.indices((5, 3, 7))
    out: np.ndarray = (21 * x + 7 * y + z).astype(np.int16)
    return out


def test_coarse_levels_are_block_means() -> None:
    P = VB.ArrayPyramid(vol(), A0, brick=2)
    assert P.dtype == np.dtype("<i2")
    L1, L2 = P.arrays[1], P.arrays[2]
    assert L1.shape == (3, 2, 4) and L2.shape == (2, 1, 2)
    # a full block: the mean of 21 x + 7 y + z over x, y, z in {0, 1} is 14.5, which
    # rounds to 14 (ties to even)
    assert L1[0, 0, 0] == 14
    # the block at (1, 0, 0): x in {2, 3}: 21 * 2.5 + 3.5 + 0.5 = 56.5 -> 56
    assert L1[1, 0, 0] == 56
    # the corner block at the far edges holds one voxel, (4, 2, 6): 84 + 14 + 6
    assert L1[2, 1, 3] == 104
    # an edge block of x only: x = 4, y in {0, 1}, z in {2, 3}: 84 + 3.5 + 2.5 = 90
    assert L1[2, 0, 1] == 90
    # level 2 from level 1 (not from level 0): block x {0,1}, y {0,1}, z {0,1}
    assert L2[0, 0, 0] == np.rint(L1[0:2, 0:2, 0:2].astype(float).mean())


def test_float_images_keep_float_means() -> None:
    a = np.arange(8, dtype=np.float64).reshape(2, 2, 2) + 0.25
    P = VB.ArrayPyramid(a, A0, brick=1)
    assert P.dtype == np.dtype("<f4")
    assert P.arrays[1].tolist() == [[[3.75]]]


def test_integers_out_of_int16_are_float32() -> None:
    a = np.zeros((2, 2, 2), np.int32)
    a[0, 0, 0] = 40000
    assert VB.wire_dtype(a) == np.dtype("<f4")
    assert VB.wire_dtype(a.astype(np.uint8)) == np.dtype("<i2")


def test_bricks_inside_at_the_edges_and_coarse() -> None:
    v = vol()
    P = VB.ArrayPyramid(v, A0, brick=2)
    lv0, lv1 = P.levels[0], P.levels[1]
    # inside
    assert (P.read(0, (1, 0, 1)) == v[2:4, 0:2, 2:4]).all()
    assert VB.brick_shape(lv0, (1, 0, 1), 2) == (2, 2, 2)
    # the far corner: one voxel
    assert P.read(0, (2, 1, 3)).tolist() == [[[104]]]
    assert VB.brick_shape(lv0, (2, 1, 3), 2) == (1, 1, 1)
    # an edge in z only
    assert P.read(0, (0, 0, 3)).shape == (2, 2, 1)
    # a coarse level: brick (1, 0, 1) of level 1 (shape (3, 2, 4)) is x = 2 only
    assert VB.brick_shape(lv1, (1, 0, 1), 2) == (1, 2, 2)
    assert (P.read(1, (1, 0, 1)) == P.arrays[1][2:3, 0:2, 2:4]).all()
    # its first voxel, level-1 (2, 0, 2): x 4, y {0, 1}, z {4, 5}: 84 + 3.5 + 4.5
    assert P.read(1, (1, 0, 1))[0, 0, 0] == 92
    # the coarsest level is one brick
    assert P.read(2, (0, 0, 0)).shape == (2, 1, 2)
    for bad in [(3, 0, 0), (0, 2, 0), (-1, 0, 0)]:
        with pytest.raises(ValueError):
            P.read(0, bad)
    with pytest.raises(ValueError):
        P.read(3, (0, 0, 0))


def test_brick_bytes_are_x_fastest_and_in_order() -> None:
    P = VB.ArrayPyramid(vol(), A0, brick=2)
    b = VB.encode(P, 0, [(0, 0, 0), (2, 1, 3)])
    a = np.frombuffer(b, "<i2")
    # brick (0, 0, 0): x fastest, then y, then z
    assert a[:8].tolist() == [0, 21, 7, 28, 1, 22, 8, 29]
    # then the one-voxel corner brick
    assert a[8:].tolist() == [104]


def test_meta_and_window() -> None:
    P = VB.ArrayPyramid(vol(), A0, brick=2)
    m = VB.meta(P, "test")
    assert m["brick"] == 2 and m["dtype"] == "int16"
    assert [lv["shape"] for lv in m["levels"]] == [[5, 3, 7], [3, 2, 4], [2, 1, 2]]
    assert m["levels"][1]["affine"] == VB.level_affine(A0, 1).tolist()
    # level 0 has 105 voxels, so it gives the window
    assert m["window"] == np.percentile(vol(), [0.5, 99.5]).tolist()


def test_from_nifti_crops_with_the_crop_affine(tmp_path: Path) -> None:
    p = tmp_path / "v.nii.gz"
    nib.save(nib.nifti1.Nifti1Image(vol(), A0), p)
    P = VB.ArrayPyramid.from_nifti(p, (1, 1, 2), (3, 2, 4), brick=2)
    assert (P.arrays[0] == vol()[1:4, 1:3, 2:6]).all()
    # the crop's voxel 0 is voxel (1, 1, 2) of the image
    assert world(P.levels[0].affine, [0, 0, 0]) == world(A0, [1, 1, 2])
    whole = VB.ArrayPyramid.from_nifti(p, brick=2)
    assert (whole.arrays[0] == vol()).all()
    assert np.array_equal(whole.levels[0].affine, A0)


def get(url: str, gz: bool = False) -> tuple[bytes, str | None]:
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"} if gz else {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read(), r.headers.get("Content-Encoding")


def test_the_server_sends_the_bricks(tmp_path: Path) -> None:
    built: list[int] = []

    def make() -> VB.BrickSource:
        built.append(1)
        return VB.ArrayPyramid(vol(), A0, brick=2)

    lazy = VB.LazyVolume(make, "test")
    # a server needs a mode: a WhiteView stand-in is enough for these routes
    srv = VS.serve(None, 0, white=object(), volume=lazy)  # type: ignore[arg-type]
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert json.loads(get(base + "/volume")[0]) == dict(name="test")
        assert built == []  # /volume does not read it
        m = json.loads(get(base + "/volume/meta")[0])
        assert m["levels"][2]["shape"] == [2, 1, 2] and built == [1]
        b, enc = get(base + "/volume/bricks?level=0&b=0,0,0;2,1,3")
        assert enc is None
        assert np.frombuffer(b, "<i2").tolist() == [0, 21, 7, 28, 1, 22, 8, 29, 104]
        gb, enc = get(base + "/volume/bricks?level=1&b=1,0,1", gz=True)
        assert enc == "gzip"
        assert np.frombuffer(gzip.decompress(gb), "<i2").tolist() == [
            int(x)
            for x in VB.ArrayPyramid(vol(), A0, 2)
            .arrays[1][2:3, 0:2, 2:4]
            .ravel(order="F")
        ]
        assert built == [1]
        for bad in ("level=0&b=9,0,0", "level=7&b=0,0,0", "b=0,0,0"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(base + "/volume/bricks?" + bad)
            assert e.value.code == 400
    finally:
        srv.shutdown()
        VS.Handler.white = None
        VS.Handler.volume = None


def test_no_volume_is_404() -> None:
    srv = VS.serve(None, 0, white=object())  # type: ignore[arg-type]
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for path in ("/volume", "/volume/meta"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(base + path)
            assert e.value.code == 404
    finally:
        srv.shutdown()
        VS.Handler.white = None
