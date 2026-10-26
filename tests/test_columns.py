"""Tests for baillarger.columns (and frame.path_exit) on the phantoms, and on
the example data when data/ has it (skipped otherwise)."""

import numpy as np
import numpy.typing as npt
import pytest
from test_refine import two_areas

from baillarger import columns as cl
from baillarger import data
from baillarger import frame as fr
from baillarger import phantoms as ph

VOX = (0.16, 0.16, 0.16)
type F64 = npt.NDArray[np.float64]


@pytest.fixture(scope="module")
def slab() -> tuple[ph.Rim, F64, fr.Frame]:
    """P3's flat slab (WM z 2..9, cortex z 10..24, outside above), with three
    synthetic echoes."""
    rim, flash, _ = ph.p3_stria_band()
    echoes = np.stack([flash, 0.6 * flash, 0.4 * flash], -1).astype(np.float64)
    return rim, echoes, fr.build_frame(rim, VOX)


def test_path_exit_on_the_slab(slab: tuple[ph.Rim, F64, fr.Frame]) -> None:
    rim, _, F = slab
    T = fr.path_exit(F, rim == 3, rim == 1)
    top = F.verts[:, 2] > 9
    # the pial surface is between z = 24 and 25: 15 voxels above z = 9.5
    assert np.allclose(T[top], 15 * 0.16, atol=0.01)
    # the bottom face of the WM has no cortex: its streamlines never exit
    assert np.isnan(T[~top]).all()


def test_fused_streamlines_do_not_exit() -> None:
    rim, _, e = ph.p1_kissing_banks()
    F = fr.build_frame(rim, VOX)
    T = fr.path_exit(F, rim == 3, rim == 1)
    patch = e["patch_mask"]
    vp = F.voxel_path[patch & (F.voxel_path >= 0)]
    ridge = F.path_end[vp] == fr.END_RIDGE
    # the streamlines of the touching patch meet at the ridge inside the
    # cortex: no pial depth
    assert ridge.mean() > 0.9
    assert np.isnan(T[vp[ridge]]).all()
    keep, why = cl.model_columns(F, rim == 3, T)
    assert sum(why[k] for k in why if k != "columns") == why["columns"]
    assert why["fused"] > 0
    pc = np.unique(F.voxel_column[patch])
    assert keep[pc].mean() < 0.5


def test_column_graph_has_no_links_across_fused_banks() -> None:
    rim, _, e = ph.p1_kissing_banks()
    F = fr.build_frame(rim, VOX)
    cortex = rim == 3
    allc = np.bincount(F.voxel_column[cortex], minlength=F.n_columns) > 0
    a = set(np.unique(F.voxel_column[e["bank_a_mask"] & e["patch_mask"]]).tolist())
    b = set(np.unique(F.voxel_column[e["bank_b_mask"] & e["patch_mask"]]).tolist())

    def crossing(E: npt.NDArray[np.int64]) -> int:
        return sum(1 for i, j in E if (i in a and j in b) or (i in b and j in a))

    Eb, Wb = cl.column_graph(F, cortex, allc, base_mm=1.0)
    Ef, Wf = cl.column_graph(F, cortex, allc, links="faces")
    assert crossing(Ef) > 0 and crossing(Eb) == 0
    for E, W in ((Eb, Wb), (Ef, Wf)):
        assert (E[:, 0] < E[:, 1]).all() and (W > 0).all()
        assert len(np.unique(E[:, 0] * F.n_columns + E[:, 1])) == len(E)
    # a base link never weighs more than its plain face count
    kf = dict(zip((Ef[:, 0] * F.n_columns + Ef[:, 1]).tolist(), Wf, strict=True))
    for (i, j), w in zip(Eb, Wb, strict=True):
        assert w <= kf[i * F.n_columns + j] + 1e-9


def test_column_graph_weights_by_brute_force(
    slab: tuple[ph.Rim, F64, fr.Frame],
) -> None:
    rim, _, F = slab
    cortex = rim == 3
    allc = np.ones(F.n_columns, bool)
    E, W = cl.column_graph(F, cortex, allc, base_mm=1.0)
    c, d = F.voxel_column, F.depth
    want: dict[tuple[int, int], float] = {}
    for i, j, k in np.argwhere(cortex & (c >= 0)):
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            u = (i + di, j + dj, k + dk)
            if u[0] >= rim.shape[0] or u[1] >= rim.shape[1] or u[2] >= rim.shape[2]:
                continue
            if not cortex[u] or c[u] < 0 or c[u] == c[i, j, k]:
                continue
            key = (min(c[i, j, k], c[u]), max(c[i, j, k], c[u]))
            dd = 0.5 * (d[i, j, k] + d[u])
            want[key] = want.get(key, 0.0) + max(0.0, 1.0 - dd / 1.0)
    # every link with weight is mesh-adjacent on the flat slab's top face
    got = {(int(a), int(b)): float(w) for (a, b), w in zip(E, W, strict=True)}
    for key, w in got.items():
        assert abs(want[key] - w) < 1e-6
    missing = [k for k, w in want.items() if w > 0 and k not in got]
    # pairs that touch but are not neighbours on the mesh (next-but-one)
    assert len(missing) < 0.1 * len(got)


def test_column_data_on_the_slab(slab: tuple[ph.Rim, F64, fr.Frame]) -> None:
    rim, echoes, F = slab
    D, why = cl.column_data(F, rim, echoes)
    assert why["model"] == D.n > 100
    assert D.f.shape == (len(D.index), 3) and D.sigma2.shape == (D.n, 3)
    ijk = np.stack(np.unravel_index(D.index, rim.shape), 1)
    inner = np.all((ijk[:, :2] >= 3) & (ijk[:, :2] < 45), 1)
    # cortex: r is the height above the white surface over the thickness
    c = D.ctx & inner
    # (the exit is found within 1/64 of a streamline segment: 0.0025 mm)
    assert np.abs(D.m[c] - (ijk[c, 2] - 9.5) / 15).max() < 2e-3
    assert np.allclose(D.thickness, 15 * 0.16, atol=0.01)
    # WM extension: within 0.5 mm below the white surface, r < 0
    w = ~D.ctx & inner
    assert w.any() and (D.m[w] < 0).all()
    assert np.abs(D.m[w] - (ijk[w, 2] - 9.5) / 15).max() < 2e-3
    assert ((9.5 - ijk[w, 2]) * 0.16 <= 0.5 + 1e-9).all()
    # the echoes of each voxel
    assert np.array_equal(D.f, echoes.reshape(-1, 3)[D.index])
    img = D.image(D.col, fill=-1, dtype=np.int64)
    assert np.array_equal(img.ravel()[D.index], D.col)
    assert (img.ravel()[np.setdiff1d(np.arange(img.size), D.index)] == -1).all()


def test_grid_sums_by_brute_force(slab: tuple[ph.Rim, F64, fr.Frame]) -> None:
    rim, echoes, F = slab
    D, _ = cl.column_data(F, rim, echoes)
    h = 1 / 20
    cnt, sx, sxx = cl.grid_sums(D, h)
    G = 20
    want_c = np.zeros((D.n, G))
    want_x = np.zeros((D.n, G, 3))
    want_xx = np.zeros((D.n, G, 3))
    for v in np.flatnonzero(D.ctx):
        g = min(max(int(D.m[v] / h), 0), G - 1)
        want_c[D.col[v], g] += 1
        want_x[D.col[v], g] += D.f[v]
        want_xx[D.col[v], g] += D.f[v] ** 2
    assert np.array_equal(cnt, want_c)
    assert np.allclose(sx, want_x) and np.allclose(sxx, want_xx)


def test_noise_is_the_within_layer_variance() -> None:
    rng = np.random.default_rng(0)
    n, per = 4, 13 * 30
    col = np.repeat(np.arange(n), per)
    m = np.tile(np.repeat((np.arange(13) + 0.5) / 13, 30), n)
    sd = np.array([[1.0, 2.0], [0.5, 0.5], [3.0, 1.0], [1.0, 1.0]])
    # a different mean in every layer, and the column's own noise
    f = 10 * m[:, None] + rng.standard_normal((len(col), 2)) * sd[col]
    var = cl._noise(col, m, f, n, 13)
    assert np.allclose(np.sqrt(var), sd, rtol=0.15)


# ---------------------------------------------------------------- the example data

need_data = pytest.mark.skipif(not data.data_available(), reason="no example data")


@need_data
def test_v1v2_column_data() -> None:
    p = data.load_patch("V1V2")
    miss = fr.missing_tissue(p.rim, p.flash, p.voxel_mm)
    F = fr.build_frame(p.rim, p.voxel_mm, miss)
    D, why = cl.column_data(F, p.rim, p.echoes, miss)
    assert D.n > 5000 and D.n_echoes == 3
    r = D.m[D.ctx]
    assert (r > 1.2).mean() < 0.03
    # cortex voxels whose centres are just inside the smoothed white surface
    # (0.06 % on V1V2) have a small negative depth, under one voxel
    assert r.min() > -0.1 and (r < 0).mean() < 0.005
    assert 1.5 < np.median(D.thickness) < 3.0
    deg = np.bincount(D.edges.ravel(), minlength=D.n)
    assert (deg == 0).mean() < 0.01
    # no voxel of the table is missing tissue
    assert not miss.ravel()[D.index].any()


def test_drop_small_pieces() -> None:
    data, _ = two_areas()
    # cut column 0 and the pair (1, 7) off from the rest
    keep_e = ~np.isin(data.edges, [0, 1, 7]).any(1)
    e = np.vstack([data.edges[keep_e], [[1, 7]]])
    w = np.r_[data.w[keep_e], 1.0]
    cut = cl.ColumnData(**{**data.__dict__, "edges": e, "w": w})
    out, dropped = cl.drop_small_pieces(cut, 5)
    assert dropped == 3 and out.n == data.n - 3
    assert set(out.labels.tolist()) == set(range(data.n)) - {0, 1, 7}
    assert len(out.index) == np.isin(cut.col, out.labels).sum()
    assert out.edges.max() < out.n and len(out.edges) == len(e) - 1
    # each voxel keeps its values and its column
    assert np.array_equal(out.labels[out.col], cut.col[np.isin(cut.col, out.labels)])
    same, none = cl.drop_small_pieces(data, 5)
    assert same is data and none == 0


def ball(R0: float = 1.6, R1: float = 3.6) -> tuple[ph.Rim, F64]:
    """A WM ball of radius R0 (mm) in a cortex shell to R1, fluid outside;
    and each voxel's distance from the centre (mm)."""
    n = 64
    c = np.array([31.7, 32.2, 31.9])
    X = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    r = np.linalg.norm((X - c) * 0.16, axis=-1)
    rim = np.ones((n, n, n), np.int16)
    rim[r <= R1] = 3
    rim[r <= R0] = 2
    return rim, r


def test_equivolume_depth_on_a_ball() -> None:
    """Round a ball of WM, the cross-section of a radial flow tube grows as the
    square of the radius, so the equivolume depth of a voxel at radius R is
    (R³ − R0³) / (R1³ − R0³), below the equidistant (R − R0) / (R1 − R0)."""
    R0, R1 = 1.6, 3.6
    rim, r = ball(R0, R1)
    F = fr.build_frame(rim, VOX)
    f = np.zeros(rim.shape + (1,))
    out = {}
    for mode in ("equidistant", "equivolume"):
        D, _ = cl.column_data(F, rim, f, depth=mode)
        rr = r.ravel()[D.index]
        sel = D.ctx & (rr > R0 + 0.1) & (rr < R1 - 0.1)
        out[mode] = (D.m[sel], rr[sel])
    m_ed, rr = out["equidistant"]
    m_ev, rr2 = out["equivolume"]
    assert np.array_equal(rr, rr2)
    exact = (rr**3 - R0**3) / (R1**3 - R0**3)
    err_ev = np.abs(m_ev - exact)
    err_ed = np.abs(m_ed - exact)
    assert np.median(err_ev) < 0.04 and np.percentile(err_ev, 90) < 0.08
    assert np.median(err_ed) > 4 * np.median(err_ev)
    assert np.mean(m_ev - m_ed) < -0.1


def test_equivolume_is_equidistant_on_a_flat_slab(
    slab: tuple[ph.Rim, F64, fr.Frame],
) -> None:
    rim, echoes, F = slab
    D1, _ = cl.column_data(F, rim, echoes)
    D2, _ = cl.column_data(F, rim, echoes, depth="equivolume")
    assert np.array_equal(D1.index, D2.index)
    ijk = np.stack(np.unravel_index(D1.index, rim.shape), 1)
    inner = np.all((ijk[:, :2] >= 4) & (ijk[:, :2] < 44), 1)
    assert np.abs(D2.m[inner] - D1.m[inner]).max() < 0.01


def test_path_volume() -> None:
    # one straight streamline through a field whose div u is constant: A grows
    # as exp(k s), and V(s) = (exp(k s) - 1) / k
    k = 0.8
    s = np.linspace(0, 2, 41).astype(np.float32)
    F = fr.Frame(
        voxel_mm=VOX, verts=np.zeros((1, 3)), faces=np.zeros((0, 3), np.int64),
        normals=np.zeros((1, 3)), vertex_column=np.zeros(1, np.int64),
        column_seed=np.zeros(1, np.int64), distance=np.zeros((1, 1, 1), np.float32),
        path_start=np.array([0, len(s)]), path_points=np.zeros((len(s), 3), np.float32),
        path_arc=s, path_end=np.zeros(1, np.int8), path_cut=np.zeros(1, bool),
        column_cut=np.zeros(1, bool), path_div=np.full(len(s), k, np.float32),
        voxel_column=np.zeros((1, 1, 1), np.int32),
        voxel_path=np.zeros((1, 1, 1), np.int32),
        depth=np.zeros((1, 1, 1), np.float32),
    )  # fmt: skip
    A, V = fr.path_volume(F)
    assert np.allclose(A, np.exp(k * s), rtol=1e-6)
    assert np.allclose(V, (np.exp(k * s) - 1) / k, rtol=1e-3)
    d = np.array([-0.3, 0.5, 1.37, 2.5])
    got = fr.volume_at(F, A, V, np.zeros(4, np.int64), d)
    assert got[0] == -0.3
    assert np.allclose(got[1:3], (np.exp(k * d[1:3]) - 1) / k, rtol=2e-3)
    # past the last point A stays constant
    assert np.isclose(got[3], V[-1] + A[-1] * 0.5)
