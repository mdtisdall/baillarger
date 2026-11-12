"""Tests of ff_octree_fem.py: the Q1 FEM prior on the octree, in the non-conforming
form (every node free; the solve uses it, CD-10) and the conforming form (hanging
nodes constrained by P)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).parent))
import ff_grid as G  # noqa: E402
import ff_octree as FO  # noqa: E402
import ff_octree_fem as F  # noqa: E402
import ff_types as T  # noqa: E402

U_MM = T.DEFAULTS.octree_finest_mm  # mm per octree unit (0.02)


def _units(mm: float) -> int:
    return int(round(mm / U_MM))


def refine_box(g: FO.OctreeGrid, lo_mm, hi_mm, level: int) -> FO.OctreeGrid:
    """Refine every leaf that overlaps the open box (lo_mm, hi_mm) until it has at least
    `level` (`refine` restores the balance)."""
    lo = np.array([_units(v) for v in lo_mm])
    hi = np.array([_units(v) for v in hi_mm])
    while True:
        a, s = g.anchor, g.size_units[:, None]
        hit = np.all((a < hi) & (a + s > lo), axis=1) & (g.level < level)
        if not hit.any():
            return g
        g = g.refine(np.flatnonzero(hit))


# ---------------------------------------------------------------- small octrees


def _small_octrees() -> dict[str, FO.OctreeGrid]:
    """Octrees of a box of 17 × 13 × 21 voxels: the base, one leaf refined twice (2:1
    interfaces at two levels), and a corner region at level 3 (interfaces at three
    levels)."""
    shape = (17, 13, 21)
    b = FO.OctreeGrid.base(shape)
    one = b.refine([5])
    one = one.refine(np.flatnonzero(one.level == 1)[:3])
    deep = refine_box(b, (0.0, 0.0, 0.0), (0.3, 0.3, 0.3), 3)
    return {"base": b, "two_levels": one, "deep": deep}


SMALL = _small_octrees()


def _forms(g: FO.OctreeGrid):
    """[(name, K, M, P)]: P is the identity for the non-conforming form."""
    K, M = F.nonconforming_matrices(g)
    K_c, M_c, P, _ = F.conforming_matrices(g)
    eye = sp.identity(g.n_nodes, format="csr")
    return [("nonconforming", K, M, eye), ("conforming", K_c, M_c, P)]


def _volume_mm3(g: FO.OctreeGrid) -> float:
    return float(np.prod(g.extent * U_MM))


def test_the_small_octrees_have_hanging_nodes_but_no_chains():
    """In these balanced octrees no parent of a hanging node is hanging. (A parent p
    is a corner of the large leaf B. The small leaves on B's face or edge come as 8
    siblings, so one of them touches p. If p were hanging on a leaf B' one level
    coarser than B, that small leaf would touch B' at p, two levels apart: not 2:1
    balanced across a corner.) So the chains in `constraint_matrix` are tested on a
    stub below."""
    for name in ("two_levels", "deep"):
        ids, par, w = SMALL[name].hanging
        assert len(ids) > 0
        assert not np.isin(par[w > 0], ids).any(), name
    for case in CASES:
        ids, par, w = _box_octree(case).hanging
        assert not np.isin(par[w > 0], ids).any(), case


def test_constraint_resolves_a_chain():
    """A stub with a chain: node 3 = (0 + 1) / 2, node 4 = (3 + 2) / 2, node 5 = the
    mean of 4, 0, 1, 2. So node 4 = 0.25, 0.25, 0.5 of nodes 0, 1, 2."""

    class Stub:
        n_nodes = 6
        hanging = (
            np.array([3, 4, 5]),
            np.array([[0, 1, 0, 0], [3, 2, 0, 0], [4, 0, 1, 2]]),
            np.array([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0.25] * 4]),
        )

    P, free = F.constraint_matrix(Stub())
    assert free.tolist() == [0, 1, 2]
    expect = np.zeros((6, 3))
    expect[:3] = np.eye(3)
    expect[3] = [0.5, 0.5, 0]
    expect[4] = [0.25, 0.25, 0.5]
    expect[5] = 0.25 * (expect[4] + expect[0] + expect[1] + expect[2])
    assert np.array_equal(P.toarray(), expect)


@pytest.mark.parametrize("name", list(SMALL))
def test_exactness_of_both_forms(name):
    """Tolerances (fixed before the first run): symmetry and K's row sums to 1e-12
    absolute (entries are near 0.1); M's sum and uᵀKu to 1e-12 relative.

    For the conforming form, Σ M_c = (P 1)ᵀ M (P 1) and P 1 = 1 (the weights of each
    row of P sum to 1), so Σ M_c is the box volume. K_c 1 = Pᵀ K P 1 = Pᵀ K 1 = 0. For
    u = x, P u_free gives x at every node (also the hanging nodes), so
    u_freeᵀ K_c u_free = xᵀ K x = ∫ |∇x|² = the volume."""
    g = SMALL[name]
    vol = _volume_mm3(g)
    x_mm = g.nodes[:, 0] * T.VOXEL_MM
    for form, K, M, P in _forms(g):
        assert abs(K - K.T).max() < 1e-12 and abs(M - M.T).max() < 1e-12, form
        assert np.abs(np.asarray(K.sum(axis=1))).max() < 1e-12, form
        assert M.sum() == pytest.approx(vol, rel=1e-12), form
        u = x_mm if form == "nonconforming" else x_mm[F.constraint_matrix(g)[1]]
        assert np.allclose(P @ u, x_mm, rtol=0, atol=1e-12), form
        assert u @ (K @ u) == pytest.approx(vol, rel=1e-12), form


def test_constraint_reproduces_the_node_positions():
    for name, g in SMALL.items():
        P, free = F.constraint_matrix(g)
        ids = g.hanging[0]
        assert np.array_equal(free, np.setdiff1d(np.arange(g.n_nodes), ids))
        assert P.shape == (g.n_nodes, len(free))
        assert np.array_equal(P @ g.nodes[free], g.nodes), name  # exact: weights 2^-k
        assert np.array_equal(np.asarray(P.sum(axis=1)).ravel(), np.ones(g.n_nodes))
        assert P.data.min() > 0  # no explicit zeros, no negative weights
        # deterministic
        P2, free2 = F.constraint_matrix(g)
        assert (P != P2).nnz == 0 and np.array_equal(free, free2)


@pytest.mark.parametrize("shape", [(57, 57, 57), (17, 13, 21)])
def test_base_octree_equals_the_uniform_grid_exactly(shape):
    b = FO.OctreeGrid.base(shape)
    u = G.UniformGrid(shape, 0.64)
    assert b.n_nodes == u.n_nodes and np.array_equal(b.nodes, u.nodes)
    K0, M0 = G.fem_matrices(u)
    for K, M in (F.nonconforming_matrices(b), F.conforming_matrices(b)[:2]):
        assert (K != K0).nnz == 0 and (M != M0).nnz == 0
        assert np.array_equal(K.indptr, K0.indptr) and np.array_equal(
            K.indices, K0.indices
        )


def test_octree_at_one_finer_level_equals_the_uniform_grid_exactly():
    shape = (17, 13, 21)
    b = FO.OctreeGrid.base(shape)
    g = b.refine(np.arange(b.n_cells))
    assert len(g.hanging[0]) == 0
    u = G.UniformGrid(shape, 0.32)
    assert np.array_equal(g.nodes, u.nodes)
    K0, M0 = G.fem_matrices(u)
    K, M = F.nonconforming_matrices(g)
    assert (K != K0).nnz == 0 and (M != M0).nnz == 0


def test_precisions_are_lambda_times_k_plus_m_over_l2():
    g = SMALL["two_levels"]
    ell, sd = 0.5, 0.3
    lam = 1.0 / (sd**2 * ell)
    K, M = F.nonconforming_matrices(g)
    assert (
        abs(F.nonconforming_precision(g, ell, sd) - lam * (K + M / ell**2)).max()
        < 1e-12
    )
    K_c, M_c, _, _ = F.conforming_matrices(g)
    assert (
        abs(F.conforming_precision(g, ell, sd) - lam * (K_c + M_c / ell**2)).max()
        < 1e-12
    )


def test_octree_fem_matrices_is_the_nonconforming_form():
    g = SMALL["two_levels"]
    K, M = g.fem_matrices()
    K1, M1 = F.nonconforming_matrices(g)
    assert (K != K1).nnz == 0 and (M != M1).nnz == 0


# ---------------------------------------------------------------- the prior's length
#
# The measure of test_ff_grid.py, section "the prior's length": the decay length of
# r·C(r) along +x from the centre node, fit range r = 0.64–1.92 mm, box 57³ voxels,
# ℓ = 0.5 mm, SD 0.3, C from a CG solve of Q c = e_centre. The reference is the uniform
# grid at 0.16 mm.
#
# Conforming form: Q_c c_free = Pᵀ e_centre (the centre node is free, so Pᵀ e_centre is
# its unit vector), and C = P c_free at every node. So a hanging node on the line takes
# its value through P. Non-conforming form: Q c = e_centre on all nodes.
#
# The octrees of the box (the centre node is at 4.48 mm on each axis; the fit line is
# x = 4.48–6.40 mm at y = z = 4.48 mm):
# - fine: level 2 (0.16 mm) on x 2.56–7.68, y and z 2.56–6.40 mm; coarser outside.
# - mid: level 1 (0.32 mm) on the same region; coarser outside.
# - cross: level 1 on the region, and level 2 on x < 5.44 mm (r < 0.96 mm). The fit
#   line crosses a 0.16/0.32 interface (a plane across the line) at r = 0.96 mm.
# - side: level 1 on the region, and level 2 on y < 4.48 mm. The line lies on the
#   interface: its nodes at odd multiples of 0.16 mm are hanging (edge middles of the
#   0.32-mm leaves on y > 4.48).
# - base: the base octree at 0.64 mm (no hanging nodes; it equals the uniform grid at
#   0.64 mm, finding CD-6 of PR 3).
#
# Tolerances, fixed before the first run (those of test_ff_grid.py): the decay lengths
# agree with the reference to 10 % of ℓ; C(1.28 mm) agrees with the reference to 25 %.

ELL = T.DEFAULTS.fem_length_mm
SD = 0.3
BOX = (57, 57, 57)
CENTRE_MM = 4.48
R_AT = (0.64, 1.28, 1.92)
REGION_LO = (2.56, 2.56, 2.56)
REGION_HI = (7.68, 6.40, 6.40)


def _box_octree(case: str) -> FO.OctreeGrid:
    b = FO.OctreeGrid.base(BOX)
    if case == "base":
        return b
    if case == "fine":
        return refine_box(b, REGION_LO, REGION_HI, 2)
    g = refine_box(b, REGION_LO, REGION_HI, 1)
    if case == "mid":
        return g
    if case == "cross":
        return refine_box(g, REGION_LO, (CENTRE_MM + 0.96, *REGION_HI[1:]), 2)
    if case == "side":
        return refine_box(g, REGION_LO, (REGION_HI[0], CENTRE_MM, REGION_HI[2]), 2)
    raise ValueError(case)


def _cg(Q: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
    x, info = spla.cg(Q, b, rtol=1e-12, maxiter=5000)
    assert info == 0
    return np.asarray(x)


def _uniform_line() -> tuple[np.ndarray, np.ndarray]:
    g = G.UniformGrid(BOX, 0.16)
    c = int(np.ravel_multi_index(tuple(np.array(g.dims) // 2), g.dims))
    assert np.allclose(g.nodes[c] * T.VOXEL_MM, CENTRE_MM)
    e = np.zeros(g.n_nodes)
    e[c] = 1.0
    col = _cg(G.fem_precision(g, ELL, SD), e)
    return _line(g.nodes, c, col)


def _line(
    nodes_vox: np.ndarray, c: int, col: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(r, C) at the nodes on the half line +x from node c, sorted by r."""
    p = nodes_vox * T.VOXEL_MM
    on = np.all(np.abs(p[:, 1:] - p[c, 1:]) < 1e-9, axis=1) & (
        p[:, 0] >= p[c, 0] - 1e-9
    )
    ids = np.flatnonzero(on)
    ids = ids[np.argsort(p[ids, 0])]
    return p[ids, 0] - p[c, 0], col[ids]


def _octree_line(g: FO.OctreeGrid, form: str) -> tuple[np.ndarray, np.ndarray]:
    c = int(g.node_id(np.full(3, _units(CENTRE_MM)))[()])
    assert c >= 0
    if form == "nonconforming":
        e = np.zeros(g.n_nodes)
        e[c] = 1.0
        col = _cg(F.nonconforming_precision(g, ELL, SD), e)
    else:
        P, free = F.constraint_matrix(g)
        j = int(np.searchsorted(free, c))
        assert free[j] == c
        e = np.zeros(len(free))
        e[j] = 1.0
        col = P @ _cg(F.conforming_precision(g, ELL, SD), e)
    return _line(g.nodes, c, col)


def _decay_length(r: np.ndarray, C: np.ndarray) -> float:
    """As test_ff_grid._decay_length; NaN if C changes sign in the fit range."""
    sel = (r > 0.64 - 1e-9) & (r < 1.92 + 1e-9)
    if not np.all(C[sel] > 0):
        return float("nan")
    slope = np.polyfit(r[sel], np.log(r[sel] * C[sel]), 1)[0]
    return float(-1.0 / slope)


def _at(r: np.ndarray, C: np.ndarray, r0: float) -> float:
    i = int(np.argmin(np.abs(r - r0)))
    assert abs(r[i] - r0) < 1e-9
    return float(C[i])


CASES = ("fine", "mid", "cross", "side", "base")
FORMS = ("conforming", "nonconforming")


@pytest.fixture(scope="module")
def lines():
    """{(case, form): (r, C), ("uniform", "0.16"): (r, C)}, and a printed table."""
    out = {("uniform", "0.16"): _uniform_line()}
    info = {"uniform": (G.UniformGrid(BOX, 0.16).n_nodes, 0)}
    for case in CASES:
        g = _box_octree(case)
        ids = g.hanging[0]
        info[case] = (g.n_nodes, len(ids))
        for form in FORMS:
            out[(case, form)] = _octree_line(g, form)
    print()
    print(f"{'case':8s} {'form':14s} {'nodes':>7s} {'hang':>6s} {'decay mm':>9s}"
          + "".join(f" {'C(' + str(r0) + ')':>10s}" for r0 in R_AT))  # fmt: skip
    for key, (r, C) in out.items():
        n, h = info[key[0]]
        row = f"{key[0]:8s} {key[1]:14s} {n:7d} {h:6d} {_decay_length(r, C):9.4f}"
        print(row + "".join(f" {_at(r, C, r0):10.3e}" for r0 in R_AT))
    return out


def test_the_side_case_has_hanging_nodes_on_the_line():
    g = _box_octree("side")
    ids = g.hanging[0]
    p = g.node_units[ids]
    c = _units(CENTRE_MM)
    on = (p[:, 1] == c) & (p[:, 2] == c) & (p[:, 0] > c) & (p[:, 0] < c + _units(1.92))
    assert on.sum() == 6  # r = 0.16, 0.48, …, 1.76 mm


def test_the_cross_case_crosses_an_interface_on_the_line():
    g = _box_octree("cross")
    c = _units(CENTRE_MM)
    p = g.node_units
    on = (
        (p[:, 1] == c) & (p[:, 2] == c) & (p[:, 0] >= c) & (p[:, 0] <= c + _units(1.92))
    )
    r = np.sort(p[on, 0] - c) * U_MM
    assert np.allclose(np.diff(r[r < 0.96 + 1e-9]), 0.16)
    assert np.allclose(np.diff(r[r > 0.96 - 1e-9]), 0.32)


# Finding (CD-6 of PR 3): the base octree is the uniform grid at 0.64 mm, where the Q1
# prior cannot hold ℓ = 0.5 mm: the covariance along an axis changes sign, because the
# axis neighbours have K_ij = 0 and M_ij > 0 (test_ff_grid.py). Here (both forms, no
# hanging nodes): C = -1.126e-3, 3.247e-4, -1.137e-5 at r = 0.64, 1.28, 1.92 mm; the
# decay length is not defined. strict=True: if it passes, the finding no longer holds.
_XFAIL_BASE = pytest.mark.xfail(
    strict=True,
    reason="finding CD-6: Q1 cannot hold ℓ = 0.5 mm at h = 0.64 mm; the covariance "
    "along an axis changes sign",
)
# Finding (first run): in the non-conforming form, an interface across the line
# (cross: 0.16 mm for r < 0.96 mm, 0.32 mm beyond) lowers the covariance beyond it.
# Decay length 0.4151 mm (reference 0.5017; conforming 0.4872); C(1.28 mm) 1.364e-4
# (reference 2.073e-4, -34 %; conforming 2.003e-4); C(1.92 mm) 2.506e-5 (reference
# 3.843e-5). C / C_reference along the line: 1.04 at 0.64 mm, 1.18 at 0.80 mm, then a
# step to 0.67 at the interface (0.96 mm) and 0.65–0.66 beyond it. The conforming form
# has no step (0.93–1.01). The large leaves do not see the hanging nodes on their
# faces, so the non-conforming field is not continuous across the interface.
# The tolerances are not changed. strict=True: if it passes, the finding no longer
# holds.
_XFAIL_CROSS_NC = pytest.mark.xfail(
    strict=True,
    reason="finding: the non-conforming form lowers the covariance across a "
    "coarse/fine interface that crosses the line",
)
CASE_FORM = [
    pytest.param(case, form, marks=mark, id=f"{case}-{form}")
    for case in CASES
    for form in FORMS
    for mark in [
        _XFAIL_BASE
        if case == "base"
        else _XFAIL_CROSS_NC
        if (case, form) == ("cross", "nonconforming")
        else ()
    ]
]


@pytest.mark.parametrize(("case", "form"), CASE_FORM)
def test_prior_decay_length_matches_the_uniform_grid(lines, case, form):
    L0 = _decay_length(*lines[("uniform", "0.16")])
    L = _decay_length(*lines[(case, form)])
    print(f"decay length: {case} {form} {L:.4f}, uniform 0.16 mm {L0:.4f}")
    assert abs(L - L0) <= 0.10 * ELL


@pytest.mark.parametrize(("case", "form"), CASE_FORM)
def test_prior_covariance_at_1_28_mm_matches_the_uniform_grid(lines, case, form):
    c0 = _at(*lines[("uniform", "0.16")], 1.28)
    c = _at(*lines[(case, form)], 1.28)
    print(f"C(1.28 mm): {case} {form} {c:.4e}, uniform 0.16 mm {c0:.4e}")
    assert abs(c - c0) <= 0.25 * c0
