"""Tests of ff_grid.py: the uniform grid, the Q1 FEM matrices, the generic helpers."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).parent))
import ff_grid as G  # noqa: E402
import ff_types as T  # noqa: E402

# ---------------------------------------------------------------- element matrices


def _shared(a: int, b: int) -> int:
    """The number of coordinates that corners a and b of a cell share."""
    return sum(((a >> ax) & 1) == ((b >> ax) & 1) for ax in range(3))


def test_element_matrices_match_closed_form():
    k_ref, m_ref = G.q1_reference()
    for h in (0.16, 0.64):
        Ke, Me = h * k_ref, h**3 * m_ref
        for a, b in itertools.product(range(8), range(8)):
            s = _shared(a, b)
            k_expect = {3: h / 3, 2: 0.0, 1: -h / 12, 0: -h / 12}[s]
            assert Ke[a, b] == pytest.approx(k_expect, abs=1e-15)
            assert Me[a, b] == pytest.approx(h**3 / 216 * 2**s, rel=1e-14)
        assert Me[0, 0] == pytest.approx(h**3 / 27)
        assert Me[0, 1] == pytest.approx(h**3 / 54)
        assert Me[0, 3] == pytest.approx(h**3 / 108)
        assert Me[0, 7] == pytest.approx(h**3 / 216)


def test_element_matrices_from_a_small_grid():
    """A grid of one cell gives the element matrices."""
    g = G.UniformGrid((5, 5, 5), 0.64)  # one cell: 4 voxels = 0.64 mm
    assert g.dims == (2, 2, 2)
    K, M = G.fem_matrices(g)
    k_ref, m_ref = G.q1_reference()
    c = g.cells[0]
    assert np.allclose(K.toarray()[np.ix_(c, c)], 0.64 * k_ref, atol=1e-15)
    assert np.allclose(M.toarray()[np.ix_(c, c)], 0.64**3 * m_ref, atol=1e-15)


@pytest.mark.parametrize("spacing", [0.16, 0.32, 0.48, 0.64])
def test_stiffness_rows_sum_to_zero_and_mass_sums_to_volume(spacing):
    g = G.UniformGrid((17, 13, 21), spacing)
    K, M = G.fem_matrices(g)
    assert K.shape == M.shape == (g.n_nodes, g.n_nodes)
    assert np.abs(np.asarray(K.sum(axis=1))).max() < 1e-12
    assert np.abs(K - K.T).max() < 1e-15 and np.abs(M - M.T).max() < 1e-15
    side = (np.array(g.dims) - 1) * spacing
    assert M.sum() == pytest.approx(float(np.prod(side)), rel=1e-12)
    # K is exact for a linear function: u = x gives uᵀ K u = ∫ |∇u|² = volume
    u = g.nodes[:, 0] * T.VOXEL_MM
    assert u @ (K @ u) == pytest.approx(float(np.prod(side)), rel=1e-12)


def test_fem_matrices_reject_hanging_nodes():
    g = G.UniformGrid((9, 9, 9), 0.64)

    class Hanging:
        def __getattr__(self, name):
            return getattr(g, name)

        @property
        def hanging(self):
            return (np.array([0]), np.zeros((1, 4), np.int64), np.zeros((1, 4)))

    with pytest.raises(NotImplementedError):
        G.fem_matrices(Hanging())  # type: ignore[arg-type]


def test_fem_precision_is_lambda_times_k_plus_m_over_l2():
    g = G.UniformGrid((9, 9, 9), 0.32)
    K, M = G.fem_matrices(g)
    ell, sd = 0.5, 0.3
    Q = G.fem_precision(g, ell, sd)
    assert isinstance(Q, sp.csr_matrix)
    lam = 1.0 / (sd**2 * ell)
    assert np.abs(Q - lam * (K + M / ell**2)).max() < 1e-12


# ---------------------------------------------------------------- the Grid protocol


def test_protocol_members_and_shapes():
    g = G.UniformGrid((10, 7, 12), 0.32)  # h_vox 2: dims ceil(9/2)+1, ceil(6/2)+1, ...
    assert isinstance(g, T.Grid)
    assert g.kind == "uniform"
    assert g.dims == (6, 4, 7)
    n, C = g.n_nodes, g.n_cells
    assert n == 6 * 4 * 7 and C == 5 * 3 * 6
    assert g.nodes.shape == (n, 3) and g.nodes.dtype == np.float64
    assert g.nodes.max(0).tolist() == [10.0, 6.0, 12.0]  # covers every voxel centre
    assert g.node_level.shape == (n,) and not g.node_level.any()
    assert g.cells.shape == (C, 8) and g.cells.dtype == np.int64
    assert g.cell_origin.shape == (C, 3)
    assert np.allclose(g.cell_size_mm, 0.32) and g.cell_size_mm.shape == (C,)
    assert g.prior_shaped.shape == (C,) and not g.prior_shaped.any()
    assert G.UniformGrid((5, 5, 5), 0.08).prior_shaped.all()
    assert not G.UniformGrid((5, 5, 5), 0.16).prior_shaped.any()
    ids, par, w = g.hanging
    assert ids.shape == (0,) and par.shape == (0, 4) and w.shape == (0, 4)
    assert g.n_colours == 8 and g.colour.shape == (n,)
    assert g.morton().shape == (C,)
    arr = g.arrays()
    assert all(k.startswith("grid_") for k in arr)


def test_node_ids_are_c_order():
    g = G.UniformGrid((10, 7, 12), 0.32)
    idx = np.rint(g.nodes / g.h_vox).astype(int)
    assert np.array_equal(
        np.ravel_multi_index(tuple(idx.T), g.dims), np.arange(g.n_nodes)
    )


def test_dims_for_spacings_that_are_not_exact_in_float():
    # 0.48 / 0.16 is 2.9999999999999996 in float64; 96 voxels give 95 / 3 → 33 nodes
    assert G.UniformGrid((96, 96, 128), 0.48).dims == (33, 33, 44)
    assert G.UniformGrid((96, 96, 128), 0.16).dims == (96, 96, 128)
    assert G.UniformGrid((96, 96, 128), 0.64).dims == (25, 25, 33)


def test_cell_corners_follow_the_corner_order():
    g = G.UniformGrid((10, 7, 12), 0.32)
    for c in range(8):
        off = np.array([(c >> a) & 1 for a in range(3)]) * g.h_vox
        assert np.allclose(g.nodes[g.cells[:, c]], g.cell_origin + off)


def test_edges_unique_sorted_and_along_cell_edges():
    g = G.UniformGrid((10, 7, 12), 0.32)
    e = g.edges
    ni, nj, nk = g.dims
    expect = (ni - 1) * nj * nk + ni * (nj - 1) * nk + ni * nj * (nk - 1)
    assert e.shape == (expect, 2) and e.dtype == np.int64
    assert np.all(e[:, 0] < e[:, 1])
    key = e[:, 0] * g.n_nodes + e[:, 1]
    assert np.all(np.diff(key) > 0)  # sorted and unique
    d = np.abs(g.nodes[e[:, 1]] - g.nodes[e[:, 0]])
    assert np.allclose(np.sort(d, axis=1), [0, 0, g.h_vox])
    # the set of edges is the set of the 12 edges of all cells
    pairs = set()
    for a, b in G._cell_edge_pairs():
        pairs |= set(zip(g.cells[:, a].tolist(), g.cells[:, b].tolist(), strict=True))
    assert pairs == set(map(tuple, e.tolist()))


def test_colours_differ_on_the_corners_of_each_cell():
    g = G.UniformGrid((10, 7, 12), 0.32)
    cc = g.colour[g.cells]
    assert np.all(np.sort(cc, axis=1) == np.arange(8))
    assert set(np.unique(g.colour).tolist()) == set(range(8))


def test_morton_is_unique_and_interleaves_bits():
    g = G.UniformGrid((10, 7, 12), 0.32)
    m = g.morton()
    assert len(np.unique(m)) == g.n_cells
    ci = np.rint(g.cell_origin / g.h_vox).astype(int)
    i = np.flatnonzero((ci == [1, 0, 0]).all(1))[0]
    j = np.flatnonzero((ci == [0, 1, 0]).all(1))[0]
    k = np.flatnonzero((ci == [0, 0, 1]).all(1))[0]
    ijk = np.flatnonzero((ci == [3, 2, 1]).all(1))[0]
    assert (m[i], m[j], m[k]) == (1, 2, 4)
    # i = 0b11, j = 0b10, k = 0b01 → bits 0 (i), 3 (i), 4 (j), 2 (k)
    assert m[ijk] == 1 + 8 + 16 + 4


# ---------------------------------------------------------------- locate


def test_locate_interior_points():
    g = G.UniformGrid((10, 7, 12), 0.32)
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (200, 3)) * (np.array(g.dims) - 1) * g.h_vox
    cid, loc = g.locate(X)
    assert np.all(cid >= 0)
    assert np.all((loc >= 0) & (loc <= 1))
    assert np.allclose(g.cell_origin[cid] + loc * g.h_vox, X)
    # trilinear interpolation of a linear function is exact
    f = g.nodes @ np.array([1.0, -2.0, 0.5])
    w = T.trilinear_weights(loc)
    assert np.allclose((w * f[g.cells[cid]]).sum(1), X @ np.array([1.0, -2.0, 0.5]))


def test_locate_on_faces_takes_the_larger_cell():
    g = G.UniformGrid((10, 7, 12), 0.32)  # h_vox 2
    cid, loc = g.locate(np.array([[2.0, 1.0, 1.0], [4.0, 2.0, 6.0], [0.0, 0.0, 0.0]]))
    ci = np.rint(g.cell_origin[cid] / g.h_vox).astype(int)
    assert ci.tolist() == [[1, 0, 0], [2, 1, 3], [0, 0, 0]]
    assert np.allclose(loc, [[0, 0.5, 0.5], [0, 0, 0], [0, 0, 0]])
    # on the last node plane: the last cell, local 1
    top = (np.array(g.dims) - 1) * g.h_vox
    cid, loc = g.locate(top[None, :])
    assert cid[0] == g.n_cells - 1 and np.allclose(loc, 1.0)


def test_locate_outside_gives_minus_one():
    g = G.UniformGrid((10, 7, 12), 0.32)
    top = (np.array(g.dims) - 1) * g.h_vox
    X = np.array([[-1e-9, 1, 1], [1, -0.5, 1], [top[0] + 1e-9, 1, 1], [1, 1, 100]])
    cid, loc = g.locate(X)
    assert np.all(cid == -1)
    assert np.all(loc == 0.0)


# ---------------------------------------------------------------- generic helpers


def test_axis_neighbours_order():
    g = G.UniformGrid((10, 7, 12), 0.32)
    nb = G.axis_neighbours(g)
    assert nb.shape == (g.n_nodes, 6) and nb.dtype == np.int64
    steps = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )
    idx = np.rint(g.nodes / g.h_vox).astype(int)
    for s in range(6):
        t = idx + steps[s]
        ok = np.all((t >= 0) & (t < np.array(g.dims)), axis=1)
        expect = np.full(g.n_nodes, -1)
        expect[ok] = np.ravel_multi_index(tuple(t[ok].T), g.dims)
        assert np.array_equal(nb[:, s], expect)


def test_edge_vectors_mm():
    g = G.UniformGrid((10, 7, 12), 0.32)
    v = G.edge_vectors_mm(g)
    assert v.shape == (len(g.edges), 3)
    assert np.allclose(np.linalg.norm(v, axis=1), 0.32)
    assert np.all(v.sum(1) > 0)  # i < j: the second node is along +axis


def _interior_node(g: G.UniformGrid) -> int:
    return int(np.ravel_multi_index(tuple(np.array(g.dims) // 2), g.dims))


def test_node_weights_interior_and_total():
    g = G.UniformGrid((12, 12, 12), 0.16)
    w = G.node_weights_vox(g)
    assert w[_interior_node(g)] == pytest.approx(1.0, rel=1e-12)
    assert w[0] == pytest.approx(1.0 / 8, rel=1e-12)  # a corner of the box
    assert w.sum() == pytest.approx(11.0**3, rel=1e-12)


def test_edge_weights_interior_edge():
    g = G.UniformGrid((12, 12, 12), 0.16)
    w = G.edge_weights_vox(g)
    c = _interior_node(g)
    nb = G.axis_neighbours(g)
    key = g.edges[:, 0] * g.n_nodes + g.edges[:, 1]
    for s in (0, 2, 4):
        e = int(np.searchsorted(key, c * g.n_nodes + nb[c, s]))
        assert w[e] == pytest.approx(1.0 / 3, rel=1e-12)
    # an edge of the box's corner is in one cell
    assert w[0] == pytest.approx(1.0 / 12, rel=1e-12)
    # at spacing h: (h / VOXEL_MM)³ / 3
    g2 = G.UniformGrid((13, 13, 13), 0.32)
    w2 = G.edge_weights_vox(g2)
    c2 = _interior_node(g2)
    key2 = g2.edges[:, 0] * g2.n_nodes + g2.edges[:, 1]
    e2 = int(np.searchsorted(key2, c2 * g2.n_nodes + G.axis_neighbours(g2)[c2, 0]))
    assert w2[e2] == pytest.approx(8.0 / 3, rel=1e-12)


def _old_face_weights(g):
    """The face weight before the fix: (edge length / VOXEL_MM)² on every edge."""
    return (np.linalg.norm(G.edge_vectors_mm(g), axis=1) / G.VOXEL_MM) ** 2


@pytest.mark.parametrize(
    "shape,h", [((9, 9, 9), 0.16), ((9, 9, 9), 0.32), ((30, 2, 2), 0.16), ((11, 7, 5), 0.48)]
)
def test_face_weights_dual_faces(shape, h):
    """The dual faces of the edges of one axis tile the box: Σ face_w · edge length (in
    voxels) over the edges of each axis equals the box volume of the nodes in voxels³
    (1e-12 relative). Inside the box the weight is the full face, bit for bit the old
    value; ½ on a box face, ¼ on a box edge."""
    g = G.UniformGrid(shape, h)
    fw = G.face_weights_vox(g)
    vec = G.edge_vectors_mm(g) / G.VOXEL_MM
    length = np.linalg.norm(vec, axis=1)
    axis = np.argmax(np.abs(vec), axis=1)
    side = (np.array(g.dims) - 1) * g.h_vox
    vol = float(np.prod(side))
    for a in range(3):
        k = axis == a
        assert (fw[k] * length[k]).sum() == pytest.approx(vol, rel=1e-12)
    count = G.edge_cell_counts(g)
    old = _old_face_weights(g)
    inner = count == 4
    assert np.array_equal(fw[inner], old[inner])
    assert np.allclose(fw[count == 2], old[count == 2] / 2, rtol=1e-15, atol=0)
    assert np.allclose(fw[count == 1], old[count == 1] / 4, rtol=1e-15, atol=0)
    assert set(np.unique(count)) <= {1, 2, 4}


def test_face_weights_full_face_inside():
    """An interior edge has the full face: 1 at 0.16 mm, 4 at 0.32 mm (1e-12)."""
    for h, val in ((0.16, 1.0), (0.32, 4.0)):
        g = G.UniformGrid((9, 9, 9), h)
        fw = G.face_weights_vox(g)
        inner = G.edge_cell_counts(g) == 4
        assert inner.any()
        assert np.allclose(fw[inner], val, rtol=1e-12, atol=0)


# ---------------------------------------------------------------- the prior's length
#
# The prior λ (K + M / ℓ²) is the α = 1 Matérn case in 3D: the covariance is near
# sd² ℓ e^(−r/ℓ) / (4π r), with no pointwise variance. So the test compares the decay
# length of r·C(r), not the node variance. C is the column of the covariance at the
# centre node, from a solve of Q c = e_centre (conjugate gradients).
#
# Box: 57 voxels on a side (8.96 mm). The centre node is 4.48 mm = 9 ℓ from each face.
# Fit range: r = 0.64 mm to 1.92 mm along +x (1 to 3 cells at 0.64 mm, about 1.3 ℓ to
# 3.8 ℓ). Both grids have nodes at these distances.
#
# Tolerances, fixed before the first run. The estimate came from the decay rate of the
# 1D Q1 stencil: 0.461 mm at 0.64 mm spacing, 0.498 mm at 0.16 mm. The first run showed
# that this estimate does not hold at 0.64 mm: there the covariance along the axis
# changes sign (the axis neighbours have K_ij = 0 and M_ij > 0, so Q_ij > 0).
# - decay lengths at the two spacings agree to 10 % of ℓ;
# - the decay length at 0.16 mm is within 10 % of ℓ;
# - C(1.28 mm) agrees between the spacings to 25 %;
# - C at 0.16 mm is within 10 % of the continuous form at r = 0.96 and 1.28 mm.

ELL = T.DEFAULTS.fem_length_mm
SD = 0.3
BOX = (57, 57, 57)


def _covariance_along_x(spacing: float) -> tuple[np.ndarray, np.ndarray, float]:
    g = G.UniformGrid(BOX, spacing)
    Q = G.fem_precision(g, ELL, SD)
    c = _interior_node(g)
    assert np.allclose(g.nodes[c], 28.0)
    e = np.zeros(g.n_nodes)
    e[c] = 1.0
    col, info = spla.cg(Q, e, rtol=1e-12, maxiter=5000)
    assert info == 0
    nb = G.axis_neighbours(g)
    ids, i = [c], c
    while nb[i, 0] >= 0:
        i = int(nb[i, 0])
        ids.append(i)
    r = (g.nodes[ids, 0] - g.nodes[c, 0]) * T.VOXEL_MM
    return r, col[ids], float(col[c])


def _decay_length(r: np.ndarray, C: np.ndarray) -> float:
    sel = (r > 0.64 - 1e-9) & (r < 1.92 + 1e-9)
    assert np.all(C[sel] > 0), f"the covariance changes sign in the fit range: {C[sel]}"
    slope = np.polyfit(r[sel], np.log(r[sel] * C[sel]), 1)[0]
    return float(-1.0 / slope)


@pytest.fixture(scope="module")
def covariances():
    return {s: _covariance_along_x(s) for s in (0.64, 0.32, 0.16)}


# Finding (coordinator decision CD-6): at 0.64 mm (h / ℓ = 1.28) the Q1 prior cannot
# hold ℓ = 0.5 mm. The axis neighbours have K_ij = 0 and M_ij > 0, so Q_ij > 0 and the
# covariance along an axis changes sign (C = 2.33e-2, -1.13e-3, 3.25e-4, -1.14e-5 at
# r = 0, 0.64, 1.28, 1.92 mm). A lumped mass gives a decay length of 1.64 mm there. The
# tests stay as written (the tolerances are not changed) and are expected to fail at
# 0.64 mm. strict=True: if they pass, the finding no longer holds and the test fails.
COARSE = [
    pytest.param(
        0.64,
        marks=pytest.mark.xfail(
            strict=True,
            reason="finding: Q1 cannot hold ℓ = 0.5 mm at h = 0.64 mm (h > ℓ/2); "
            "the covariance along an axis changes sign",
        ),
    ),
    0.32,
]


def _at(covariances, s: float, r0: float) -> float:
    r, C, _ = covariances[s]
    return float(C[np.argmin(np.abs(r - r0))])


@pytest.mark.parametrize("coarse", COARSE)
def test_prior_decay_length_is_the_same_at_both_spacings(covariances, coarse):
    L = {s: _decay_length(*covariances[s][:2]) for s in (coarse, 0.16)}
    print(f"decay length of r·C(r): {coarse} mm {L[coarse]:.4f}, 0.16 mm {L[0.16]:.4f}")
    assert abs(L[coarse] - L[0.16]) <= 0.10 * ELL


@pytest.mark.parametrize("coarse", COARSE)
def test_prior_covariance_at_fixed_distance_converges(covariances, coarse):
    cc, c16 = _at(covariances, coarse, 1.28), _at(covariances, 0.16, 1.28)
    print(f"C(1.28 mm): {coarse} mm {cc:.4e}, 0.16 mm {c16:.4e}")
    assert abs(cc - c16) <= 0.25 * c16


def test_prior_at_the_image_spacing_matches_the_continuous_form(covariances):
    L16 = _decay_length(*covariances[0.16][:2])
    print(f"decay length of r·C(r) at 0.16 mm: {L16:.4f}")
    assert abs(L16 - ELL) <= 0.10 * ELL
    for r0 in (0.96, 1.28):
        cont = SD**2 * ELL * np.exp(-r0 / ELL) / (4 * np.pi * r0)
        c = _at(covariances, 0.16, r0)
        print(f"C({r0} mm) at 0.16 mm {c:.4e}, continuous {cont:.4e}")
        assert abs(c - cont) <= 0.10 * cont


def test_prior_node_variance_rises_as_spacing_falls(covariances):
    # α = 1 in 3D: no pointwise variance, so the node variance is larger on a finer grid
    assert covariances[0.16][2] > covariances[0.64][2]


# ---------------------------------------------------------------- chunked assembly


def test_fem_matrices_chunks_match_a_dense_assembly(monkeypatch):
    """Chunks of 7 cells (summed in a tree) give the matrices of a dense assembly over
    all cells, to rounding."""
    g = G.UniformGrid((9, 7, 11), 0.32)
    n = g.n_nodes
    k_ref, m_ref = G.q1_reference()
    Kd = np.zeros((n, n))
    Md = np.zeros((n, n))
    for c, h in zip(g.cells, g.cell_size_mm, strict=True):
        Kd[np.ix_(c, c)] += h * k_ref
        Md[np.ix_(c, c)] += h**3 * m_ref
    monkeypatch.setattr(G, "CHUNK_CELLS", 7)
    K, M = G.fem_matrices(g)
    assert g.n_cells > 7 * 4
    assert np.abs(K.toarray() - Kd).max() < 1e-15
    assert np.abs(M.toarray() - Md).max() < 1e-17
    assert K.has_sorted_indices and M.has_sorted_indices
    assert not np.any(K.data == 0.0)
    Q = G.precision_from(K, M, 0.5, 0.3)
    assert np.abs(Q - G.fem_precision(g, 0.5, 0.3)).max() < 1e-12
    assert np.array_equal(G.node_weights_from_mass(M), G.node_weights_vox(g))
