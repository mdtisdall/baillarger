"""Tests of baillarger.normal_maps (wave 2.1, Part A), with exact answers: a
plane, a sphere, a phantom white surface from white_dseg and white_surf, a
sphere with one face turned over, and the ranked list against a brute-force
greedy reference."""

from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from nibabel.gifti.gifti import GiftiImage

from baillarger import mesh_smoothing as S
from baillarger import normal_maps as NM
from baillarger import white_dseg, white_surf, wm_sites
from baillarger.white_surf import WhiteSurface

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def plane(k: int = 12, h: float = 0.3) -> tuple[Float, Int]:
    """A flat k x k grid of squares of side h in z = 0, two triangles each,
    oriented to +z."""
    i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
    v = np.c_[i.ravel() * h, j.ravel() * h, np.zeros(i.size)]
    idx = (i * (k + 1) + j)[:-1, :-1].ravel()
    a, b, c, d = idx, idx + (k + 1), idx + (k + 2), idx + 1
    f = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
    return v, f.astype(np.int64)


def icosphere(level: int, radius: float) -> tuple[Float, Int]:
    """A subdivided icosahedron on a sphere of the radius, faces outward."""
    t = (1 + 5**0.5) / 2
    v = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]  # fmt: skip
    f = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]  # fmt: skip
    vs = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces: list[tuple[int, int, int]] = [(a, b, c) for a, b, c in f]
    for _ in range(level):
        mid: dict[tuple[int, int], int] = {}

        def m(a: int, b: int, mid: dict[tuple[int, int], int] = mid) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = vs[a] + vs[b]
                vs.append(p / np.linalg.norm(p))
                mid[key] = len(vs) - 1
            return mid[key]

        nf = []
        for a, b, c in faces:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    vv = np.array(vs) * radius
    ff = np.array(faces, np.int64)
    # faces outward
    fn = np.cross(vv[ff[:, 1]] - vv[ff[:, 0]], vv[ff[:, 2]] - vv[ff[:, 0]])
    flip = np.sum(fn * vv[ff].mean(1), 1) < 0
    ff[flip] = ff[flip][:, [0, 2, 1]]
    return vv, ff


def maps_of(v: Float, f: Int, steps: int = 10) -> dict[str, npt.NDArray[np.float32]]:
    """The maps of a mesh in mm whose voxel frame is mm (identity affine),
    with its normals smoothed as white_surf smooths them."""
    n, nlen = S.smooth_normals_and_length(v.astype(np.float32), f, steps)
    return NM.normal_maps(v, v, f, n, np.minimum(nlen, 1), steps)


def test_a_plane() -> None:
    v, f = plane()
    mp = maps_of(v, f)
    for k in NM.MAPS:
        want = 1.0 if k == "normlen" else 0.0
        np.testing.assert_allclose(mp[k], want, atol=1e-5, err_msg=k)


def test_a_sphere() -> None:
    """With the radial normals, nbrmaxpermm is the arc's angle over the chord:
    (180 / pi) / R to second order in h / R. With the smoothed normals, the
    same away from the 12 vertices of valence 5: there the smoothing pulls
    the neighbours' normals toward the centre vertex's, and the angle is
    about 20 % smaller (a property of the smoothing, not of the map).
    renorm is about 0 (at most 0.0025 degrees here). smoothing has no exact
    value: the umbrella mean is not weighted, so faces of different sizes
    bend the smoothed normal from the radial direction (up to 1.0 degree,
    median 0.24, on this mesh; the raw normals up to 0.34)."""
    R = 5.0
    v, f = icosphere(4, R)
    E = S.edges(f)
    ln = np.linalg.norm(v[E[:, 0]] - v[E[:, 1]], axis=1)
    assert ln.mean() < R / 10  # a fine mesh
    want = np.degrees(1 / R)  # (180 / pi) / R
    _, per = NM.neighbour_maps(v / R, v, E)
    np.testing.assert_allclose(per, want, rtol=0.005)
    mp = maps_of(v, f)
    deg = np.bincount(E.ravel(), minlength=len(v))
    six = deg == 6
    assert (~six).sum() == 12
    np.testing.assert_allclose(np.median(mp["nbrmaxpermm"]), want, rtol=0.03)
    np.testing.assert_allclose(mp["nbrmaxpermm"][six], want, rtol=0.15)
    assert np.all(mp["nbrmaxpermm"][~six] < want)
    assert float(np.max(mp["renorm"])) < 0.05
    assert not mp["misfacingraw"].any() and not mp["misfacingsmooth"].any()
    assert np.all(mp["normlen"] < 1)  # curved: the mean of unit normals is shorter


def test_one_face_turned_over() -> None:
    v, f = icosphere(4, 5.0)
    g = f.copy()
    g[100] = g[100][[0, 2, 1]]
    mp = maps_of(v, g)
    for k in ("misfacingraw", "misfacingsmooth"):
        want = np.zeros(len(v), np.float32)
        want[g[100]] = 1
        np.testing.assert_array_equal(mp[k], want, err_msg=k)


def test_misfacingraw_is_the_wm_sites_rule() -> None:
    v, f = icosphere(3, 4.0)
    g = f.copy()
    g[[7, 300]] = g[[7, 300]][:, [0, 2, 1]]
    bad = wm_sites.misfacing(v, g)
    assert bad.sum() >= 2
    mp = maps_of(v, g)
    np.testing.assert_array_equal(
        mp["misfacingraw"], np.bincount(g[bad].ravel(), minlength=len(v))
    )


def test_a_vertex_in_no_face_is_nan() -> None:
    v, f = plane(4)
    v = np.r_[v, [[9.0, 9.0, 9.0]]]
    mp = maps_of(v, f)
    for k in NM.MAPS:
        assert np.isnan(mp[k][-1]), k
        assert np.isfinite(mp[k][:-1]).all(), k


def test_neighbour_maps_by_hand() -> None:
    """Three vertices on a line, normals at 0, 10 and 40 degrees; edges of
    1 and 2 mm."""
    n = np.array(
        [[np.cos(np.radians(a)), np.sin(np.radians(a)), 0] for a in (0, 10, 40)]
    )
    v = np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0]])
    E = np.array([[0, 1], [1, 2]])
    mx, per = NM.neighbour_maps(n, v, E)
    np.testing.assert_allclose(mx, [10, 30, 30], rtol=1e-6)
    np.testing.assert_allclose(per, [10, 15, 15], rtol=1e-6)


# ------------------------------------------------------------ the phantom white surface
AFF = np.array([[0.2, 0, 0, -3.0], [0, 0.16, 0, 2.0], [0, 0, 0.2, 5.0], [0, 0, 0, 1.0]])


@pytest.fixture(scope="module")
def surf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A right hemisphere's white surface from a synthetic segmentation: a WM
    ball round a ventricle, in cortex (as tests/test_viewer_columns.py)."""
    d = tmp_path_factory.mktemp("nmaps")
    shape = (44, 40, 36)
    x = np.indices(shape).astype(float)
    r = np.sqrt(((x - np.array([20.3, 19.6, 17.8])[:, None, None, None]) ** 2).sum(0))
    a = np.zeros(shape, np.int16)
    a[r < 15] = 42
    a[r < 12] = 41
    a[r < 4] = 43
    src = d / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(parents=True)
    nib.save(nib.nifti1.Nifti1Image(a, AFF), src)
    white_dseg.main([str(src), "--out", str(d / "deriv")])
    anat = d / "deriv/sub-T01/ses-S1/anat"
    dseg = anat / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    white_surf.main([str(dseg), "--out", str(d / "deriv"), "--device", "cpu"])
    return anat / "sub-T01_ses-S1_hemi-R_white.surf.gii"


def test_the_length_is_the_published_one(surf: Path) -> None:
    W = WhiteSurface.load(surf)
    steps = int(W.sidecar["NormalSmoothingSteps"])
    m = NM.linear_smoothed(W.v.astype(np.float32), W.f, steps)
    ln = np.minimum(np.linalg.norm(m, axis=1), 1)
    np.testing.assert_allclose(ln, W.normlen, atol=1e-5)
    _, ref = S.smooth_normals_and_length(W.v.astype(np.float32), W.f, steps)
    np.testing.assert_allclose(np.linalg.norm(m, axis=1), ref, atol=1e-5)


def test_the_maps_of_the_phantom(surf: Path) -> None:
    W = WhiteSurface.load(surf)
    pts = GiftiImage.from_filename(str(surf)).agg_data("pointset")
    mp = NM.normal_maps(
        W.v, pts, W.f, W.normals, W.normlen, int(W.sidecar["NormalSmoothingSteps"])
    )
    assert list(mp) == list(NM.MAPS)
    for k in NM.MAPS:
        assert mp[k].dtype == np.float32 and mp[k].shape == (len(W.v),), k
        assert np.isfinite(mp[k]).all(), k
    np.testing.assert_array_equal(mp["normlen"], W.normlen)
    np.testing.assert_array_equal(
        mp["misfacingraw"],
        np.bincount(
            W.f[wm_sites.misfacing(np.asarray(pts, float), W.f)].ravel(),
            minlength=len(W.v),
        ),
    )
    # n is m / |m| renormalised at each step: close, not equal, on a smooth surface
    assert float(np.median(mp["renorm"])) < 1.0


# ---------------------------------------------------------------- the ranked list
def greedy(values: Float, pts: Float, n: int, spacing: float, order: str) -> list[int]:
    """The rule, one candidate at a time against every kept site."""
    ok = np.flatnonzero(np.isfinite(values))
    key = -values[ok] if order == "high" else values[ok]
    kept: list[int] = []
    for v in ok[np.argsort(key, kind="stable")]:
        if len(kept) == n:
            break
        if kept and np.min(np.linalg.norm(pts[kept] - pts[v], axis=1)) <= spacing:
            continue
        kept.append(int(v))
    return kept


@pytest.mark.parametrize("order", ["high", "low"])
@pytest.mark.parametrize("spacing", [0.0, 0.3, 1.0, 2.5])
def test_ranked_sites_is_the_greedy_rule(order: str, spacing: float) -> None:
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 10, (3000, 3))
    val = np.round(rng.normal(size=3000), 1)  # many ties
    val[rng.random(3000) < 0.1] = np.nan
    got = NM.ranked_sites(val, pts, 200, spacing, order)
    assert got.tolist() == greedy(val, pts, 200, spacing, order)
    assert np.isfinite(val[got]).all()
    fin = val[np.isfinite(val)]
    assert val[got[0]] == (fin.max() if order == "high" else fin.min())
    if spacing > 0 and len(got) > 1:
        d = np.linalg.norm(pts[got][:, None] - pts[got][None], axis=-1)
        assert d[np.triu_indices(len(got), 1)].min() > spacing


def test_ranked_sites_across_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NM, "SITE_CHUNK", 7)
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 6, (2000, 3))
    val = rng.random(2000)
    got = NM.ranked_sites(val, pts, 50, 0.8)
    assert got.tolist() == greedy(val, pts, 50, 0.8, "high")


def test_ranked_sites_runs_out() -> None:
    pts = np.zeros((5, 3))
    val = np.array([1.0, np.nan, 3.0, 2.0, np.nan])
    assert NM.ranked_sites(val, pts, 10, 1.0).tolist() == [2]
    assert NM.ranked_sites(val, pts + np.arange(5)[:, None] * 5, 10, 1.0).tolist() == [
        2,
        3,
        0,
    ]
    assert NM.ranked_sites(np.full(4, np.nan), np.zeros((4, 3)), 3, 1.0).tolist() == []


def test_ranked_table() -> None:
    v, f = icosphere(2, 3.0)
    mp = maps_of(v, f)
    rows = NM.ranked_table(mp, "nbrmax", v, np.arange(len(v)) * 10, n=5, spacing=1.0)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    r = rows[0]
    assert r["value"] == float(np.nanmax(mp["nbrmax"]))
    assert r["edge_key"] == 10 * r["vertex"]
    assert set(r) == {"rank", "vertex", "edge_key", "x", "y", "z", "value"} | (
        set(NM.MAPS) - {"nbrmax"}
    )
