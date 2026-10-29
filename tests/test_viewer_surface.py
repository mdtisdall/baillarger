"""Tests of baillarger.viewer.surface: a Surface made from arrays alone, with no
column volume."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from baillarger.viewer import inflation as IN
from baillarger.viewer.surface import Surface, mesh_key

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def plane(n: int = 9) -> tuple[Float, Int]:
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.c_[i.ravel() * 2.0 + 1, j.ravel() * 2.0 + 1, np.full(n * n, 4.0)]
    F = []
    for a in range(n - 1):
        for b in range(n - 1):
            p, q, r, s = a * n + b, (a + 1) * n + b, (a + 1) * n + b + 1, a * n + b + 1
            F += [(p, q, r), (p, r, s)]
    return v, np.array(F, np.int64)


def make(tmp: Path, key: str | None = None) -> Surface:
    v, f = plane()
    aff = np.diag([0.5, 0.5, 0.5, 1.0])
    aff[:3, 3] = [10.0, -20.0, 30.0]
    return Surface(
        v, f, np.full(3, 0.5), aff, tmp, "plane", key=key, log=lambda m: None
    )


def test_coordinates_and_picking(tmp_path: Path) -> None:
    S = make(tmp_path)
    assert np.allclose(S.vmm, S.v_vox * 0.5)
    assert np.allclose(S.world(S.v_vox), S.v_vox * 0.5 + [10.0, -20.0, 30.0])
    k = 40
    assert S.nearest(S.world(S.v_vox[k])[0] + [0.1, -0.1, 0.05]) == k


def test_mesh_payload_with_and_without_faces(tmp_path: Path) -> None:
    S = make(tmp_path)
    nv, nf = len(S.v_vox), len(S.f)
    for vtrace in (None, np.arange(nv, dtype=np.int32)):
        b = S.mesh_payload(0, vtrace)
        hl = int(np.frombuffer(b[:4], np.uint32)[0])
        hdr = json.loads(b[4 : 4 + hl])
        assert (hdr["nv"], hdr["nf"], hdr["faces"]) == (nv, nf, vtrace is not None)
        body = len(b) - 4 - hl
        assert body == 4 * (7 * nv + (nv + 3 * nf if vtrace is not None else 0))


def test_the_cache_is_keyed(tmp_path: Path) -> None:
    S = make(tmp_path)
    assert S.key == mesh_key(S.v_vox, S.f)
    np.testing.assert_array_equal(make(tmp_path).uv_v, S.uv_v)  # loaded again
    other = make(tmp_path, key="given")
    assert other.key == "given"
    with np.load(tmp_path / "inflate_plane.npz") as Q:
        assert str(Q["key"]) == f"given-v{IN.VERSION}"
