"""Tests of ffc_refine1d (the indicator, the marking and the start) and of the ffc_basis
interface.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ffc_refine1d.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402
import ffc_basis as B  # noqa: E402
import ffc_contact as CT  # noqa: E402
import ffc_contact1d as R  # noqa: E402
import ffc_refine1d as RF  # noqa: E402

H = 0.16
F64 = torch.float64


def test_dorfler_marks_the_fewest_cells() -> None:
    eta = np.array([0.1, 3.0, 0.0, 1.0, 1.0])
    assert list(RF.dorfler(eta, 0.5)) == [1]  # 3.0 ≥ 0.5 · 5.1
    assert list(RF.dorfler(eta, 0.7)) == [1, 3]
    assert len(RF.dorfler(np.zeros(4), 0.5)) == 0


def test_indicator_is_zero_where_the_sides_agree() -> None:
    """Equal banks, the sheet centre on a node: the true τ (ridge on that node) has the side of
    the evidence everywhere, so η = 0; a ridge collapsed 0.8 mm away has η > 0 only in the
    cells that hold the sheet."""
    c = CT.Contact(2.4, 2.4, 0.32)
    x = c.nodes(H, "node")
    p = R.base_problem(c, x)
    assert np.all(RF.indicator(p, c.tau_true(x)) == 0.0)
    eta = RF.indicator(p, -np.abs(x - 0.8))
    mid = 0.5 * (x[:-1] + x[1:])
    assert np.all(eta[np.abs(mid) > c.g + H] == 0.0) and eta.sum() > 0


def test_start_peak_put_on_a_node() -> None:
    c = CT.Contact(2.5, 1.2, 0.02)
    x = c.nodes(H, "mid")
    fB, fA = R.starts(c, x)["good"]
    p, z, n = RF.start_state(c, x, fB, fA)
    assert np.any(np.abs(p.x - 0.5 * (fA + fB)) < 1e-12) and n >= 0
    L = np.diff(p.x)
    assert np.all(L[1:] <= 2 * L[:-1] + 1e-15) and np.all(L[:-1] <= 2 * L[1:] + 1e-15)
    assert p.valid(z)


def test_basis_refine_and_prolong_are_exact() -> None:
    b = B.P1Basis(np.linspace(0.0, 1.6, 11))
    z = np.sin(b.x)
    f = b.refine(np.array([4]))
    zf = b.prolong(z, f)
    xq = np.linspace(0.0, 1.6, 97)
    v0 = b.values(torch.as_tensor(z, dtype=F64), xq).numpy()
    v1 = f.values(torch.as_tensor(zf, dtype=F64), xq).numpy()
    assert np.max(np.abs(v0 - v1)) < 1e-14
    assert f.n_cells == 11  # 10 cells, one bisected (its neighbours are within 2:1)
    lv = RF.levels(R.base_problem(CT.Contact(2.4, 2.4, 0.32), f.x), zf)
    assert sorted(lv.tolist()) == [0] * 9 + [1, 1]


@pytest.mark.parametrize("enrich", [None, 7])
def test_basis_gradients_and_extrema(enrich: int | None) -> None:
    b = B.P1Basis(np.linspace(-1.6, 1.6, 21), enrich)
    tau = -np.abs(b.x - 0.05)
    z = b.pack(tau)
    zt = torch.as_tensor(z, dtype=F64)
    g = b.gradients(zt, np.array([-1.0, 1.0])).numpy()
    assert g[0] == pytest.approx(1.0) and g[1] == pytest.approx(-1.0)
    ex = b.extrema(z)
    assert [e.kind for e in ex] == ["max"]
    assert float(ex[0].c @ z) == pytest.approx(ex[0].value)


# ---- the equivolume model (2026-11-13)


def test_mm_pull_coefficient() -> None:
    import ff_types as T
    import ffc_tau1d as P

    assert P.mm_pull_coefficient(T.DEFAULTS) == pytest.approx(0.5 * 0.16**2 / (np.log(2.0) ** 2 * 0.5**3))
    assert P.EQUIVOLUME_LAMBDA_A_PROVISIONAL == pytest.approx(0.16**2 / (0.3**2 * 0.5))


def test_dual_tissue_fraction() -> None:
    xs = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=F64)
    tau = torch.tensor([-2.0, -1.0, 1.0, 3.0], dtype=F64)
    # dual cells [0.5, 1.5] (τ −1.5 → −1 → 0: all tissue) and [1.5, 2.5] (τ 0 → 1 → 2: none)
    f = B._dual_tissue_fraction(tau, xs).numpy()
    assert f[0] == pytest.approx(1.0) and f[1] == pytest.approx(0.0)
    tau2 = torch.tensor([-2.0, -0.5, 0.5, 3.0], dtype=F64)
    # [0.5, 1.5]: τ −1.25 → −0.5 → 0: all tissue; [1.5, 2.5]: τ 0 → 0.5 → 1.75: none
    f2 = B._dual_tissue_fraction(tau2, xs).numpy()
    assert f2[0] == pytest.approx(1.0) and f2[1] == pytest.approx(0.0)
    tau3 = torch.tensor([-2.0, -1.0, 0.0, 3.0], dtype=F64)
    # [1.5, 2.5]: τ −0.5 → 0 → 1.5: the first half is tissue
    assert B._dual_tissue_fraction(tau3, xs).numpy()[1] == pytest.approx(0.5)


def test_flat_cells_and_extrema() -> None:
    b = B.P1Basis(np.linspace(0.0, 1.0, 6))
    z = np.array([-0.4, -0.1, 0.2, 0.2, -0.1, -0.4])
    assert not b.admissible(z) and b.admissible(z, flat_beyond_top=True)
    ex = b.extrema(z)
    assert len(ex) == 1 and ex[0].kind == "max" and ex[0].x == pytest.approx(0.5)
    assert float(ex[0].c @ z) == pytest.approx(0.2)
    zt = np.array([-0.4, -0.1, -0.1, 0.2, -0.1, -0.4])  # a flat cell in tissue
    assert not b.admissible(zt, flat_beyond_top=True)


@pytest.mark.parametrize("form", ["equivolume_abs", "equivolume_convex"])
def test_equivolume_truth_is_stationary_and_gradient_exact(form: str) -> None:
    """Equal banks (slope 1, the ridge on a node): the true τ is stationary for the new energy;
    and the autograd gradient matches central differences at a perturbed state."""
    import ffc_tau1d as P

    c = CT.Contact(2.4, 2.4, 0.32)
    x = c.nodes_roots(H, "node", "merge")
    p = P.Problem(x, c.log_odds_t, c.settings, roots=R.roots(c), energy_form=form)
    z = c.tau_true(x)
    assert p.valid(z)
    g, _ = P.gradient_hessian(p, z)
    assert np.max(np.abs(g)) < 1e-6
    z1 = 1.05 * z + 0.01 + 0.05 * np.sin(3.0 * x)  # monotone on each side, the ridge kept
    assert p.valid(z1)
    g1, _ = P.gradient_hessian(p, z1)
    eps = 1e-6
    for i in (0, len(z) // 2 - 3, len(z) // 2, len(z) - 1):
        e = np.zeros(len(z))
        e[i] = eps
        fd = (p.total(z1 + e) - p.total(z1 - e)) / (2 * eps)
        assert fd == pytest.approx(g1[i], rel=1e-5, abs=1e-7)


def test_init_variants_and_envelope() -> None:
    import ffc_init1d as I

    c = CT.Contact(2.5, 1.2, 0.02)
    assert I.t_hat(c, "label") == (1.2, 2.5)
    assert I.t_hat(c, "label_B_in0.16") == pytest.approx((1.2 - 0.16, 2.5))
    assert I.t_hat(c, "label_BA_out0.5") == pytest.approx((1.7, 3.0))
    m = 0.5 * (c.root_a - c.root_b)
    assert I.t_hat(c, "fused") == (m, m)
    tb, ta = I.t_hat(c, "probe")  # type: ignore[misc]
    assert abs(tb - c.tB) < 0.002 and abs(ta - c.tA) < 0.002
    # the envelope with the true faces: zero at both true faces, its peak at x_ridge
    assert I.envelope(c, np.array([-c.g, c.g]), c.tB, c.tA) == pytest.approx([0.0, 0.0], abs=1e-12)
    assert I.envelope_peak(c, c.tB, c.tA) == pytest.approx(0.5 * (c.root_a + c.root_b + c.tB - c.tA))
    assert len(I.VARIANTS) == 15


def test_nodes_at_the_roots() -> None:
    c = CT.Contact(2.5, 1.2, 0.02)
    for centre in ("mid", "node"):
        for ends in ("keep", "merge"):
            x = c.nodes_roots(H, centre, ends)
            assert x[0] == pytest.approx(c.root_b) and x[-1] == pytest.approx(c.root_a)
            assert np.all(np.diff(x) > 0)
            if ends == "merge":
                L = np.diff(x)
                assert L[0] >= H / 2 - 1e-12 and L[-1] >= H / 2 - 1e-12 and max(L[0], L[-1]) <= 1.5 * H + 1e-12


def test_convex_pull_split_at_the_top() -> None:
    """The convex form: c (n τ' − 1)² in tissue (n from the root on the cell's side of the ridge),
    c max(|τ'| − 1, 0)² beyond the top. A tent with slope ±2, its top inside a cell: the pull is
    c ((2 − 1)² · tissue length + (2 − 1)² · beyond length) = c · line length; with slope ±0.5:
    tissue (0.5 − 1)² only."""
    import ffc_tau1d as P

    x = np.linspace(-1.0, 1.0, 9)
    roots = ((-1.0, 1.0), (1.0, -1.0))
    for s_, want in ((2.0, lambda Lt: 1.0 * 2.0), (0.5, lambda Lt: 0.25 * Lt)):
        z = s_ * (0.3 - np.abs(x))  # the ridge at 0 (a node), τ = 0 at |x| = 0.3 (inside a cell)
        p = P.Problem(x, lambda xx: torch.zeros_like(xx), T.DEFAULTS, roots=roots, energy_form="equivolume_convex")
        e = p.energy_terms(torch.as_tensor(z, dtype=F64))
        lt = 2.0 * (1.0 - 0.3)
        assert float(e["mm_pull"]) == pytest.approx(p.mm_coeff * want(lt), rel=1e-12)


def test_slope_bounds() -> None:
    """With slope_bounds: one rise form per cell away from the ridge, signed by the root on
    its side; all hold on a valid tent; the two cells at the ridge are free."""
    import ffc_tau1d as P

    x = np.linspace(-1.0, 1.0, 9)
    roots = ((-1.0, 1.0), (1.0, -1.0))
    z = 0.3 - np.abs(x)
    p0 = P.Problem(x, lambda xx: torch.zeros_like(xx), T.DEFAULTS, roots=roots, energy_form="equivolume_convex")
    p1 = P.Problem(x, lambda xx: torch.zeros_like(xx), T.DEFAULTS, roots=roots, energy_form="equivolume_convex",
                   slope_bounds=True)
    assert len(p0.bounds(z)) == 1
    b = p1.bounds(z)
    assert len(b) == 1 + (len(x) - 1) - 2
    assert all(float(c @ z) >= 0 for c in b)
    zb = z.copy()
    zb[1] = zb[0] - 0.01  # the first cell falls toward the ridge: Q1 broken, a bound < 0
    assert min(float(c @ zb) for c in p1.bounds(zb)) < 0
