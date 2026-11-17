"""Tests of ffc_contact and ffc_tau1d (the S1 form: one τ field, the top is {τ = 0}).

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ffc_contact1d.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402
import ffc_contact as CT  # noqa: E402
import ffc_contact1d as R  # noqa: E402
import ffc_tau1d as P  # noqa: E402

H = 0.16
F64 = torch.float64
# Tolerances, fixed before any run.
TOL_QUAD = 1e-12  # the vectorised Gauss rule against the scalar one
TOL_GRAD = 1e-6  # the gradient at the true state (equal banks: the truth is the minimum)
TOL_TOP = 1e-6  # the solved tops against the true faces (mm)


def test_contact_truth_and_oracle() -> None:
    c = CT.Contact(2.5, 1.2, 0.02)
    assert c.tau_scale == pytest.approx(1.85)
    x = np.array([-c.g - c.tB, -c.g, c.x_ridge, c.g, c.g + c.tA])
    tt = c.tau_true(x)
    assert tt[0] == pytest.approx(-c.tau_scale) and tt[4] == pytest.approx(-c.tau_scale)
    assert tt[1] == pytest.approx(0.0, abs=1e-15) and tt[3] == pytest.approx(0.0, abs=1e-15)
    # the two virtual continuations meet at the ridge
    assert (c.g + c.x_ridge) / c.tB == pytest.approx((c.g - c.x_ridge) / c.tA)
    assert np.allclose(c.stop_probability(np.array([-c.g, c.g])), 0.5)
    xs = np.linspace(-1, 1, 101)
    assert np.allclose(c.stop_log_odds(xs), T.log_odds(ndtr((c.g - np.abs(xs)) / 0.08), 0.02, 0.98))
    lo_t = c.log_odds_t(torch.as_tensor(xs, dtype=F64)).numpy()
    assert np.allclose(lo_t, c.stop_log_odds(xs), atol=1e-12)
    for centre, frac in (("mid", 0.5), ("node", 0.0)):
        x = c.nodes(H, centre)
        assert np.min(np.abs(x / H - np.round(x / H - frac) - frac)) < 1e-9
        assert x[0] <= c.x_start_b and x[-1] >= c.x_start_a


def test_vectorised_gauss_rule() -> None:
    c = CT.Contact(2.4, 2.4, 0.02)
    p = R.base_problem(c, c.nodes(H, "mid"))
    a = torch.tensor([-0.1, -0.02, 0.0], dtype=F64)
    b = torch.tensor([0.05, 0.03, 0.08], dtype=F64)
    sg = torch.tensor([1.0, -1.0, 1.0], dtype=F64)
    v = p._quad_v(a, b, sg).numpy()
    for i in range(3):
        assert v[i] == pytest.approx(float(p._quad(a[i], b[i], float(sg[i]))), abs=TOL_QUAD)


def test_enrichment_inserts_a_node() -> None:
    c = CT.Contact(2.4, 2.4, 0.02)
    x = c.nodes(H, "mid")
    e = int(np.searchsorted(x, 0.0) - 1)
    p = P.Problem(x, c.log_odds_t, c.settings, enrich_cell=e, roots=R.roots(c))
    z = p.pack(c.tau_true(x), 0.05, 0.01)
    tau, xs = (a.numpy() for a in p.split(torch.as_tensor(z, dtype=F64)))
    assert len(xs) == len(x) + 1 and xs[e + 1] == pytest.approx(0.01) and tau[e + 1] == pytest.approx(0.05)
    assert np.all(np.diff(xs) > 0)


def test_feasibility_rules() -> None:
    c = CT.Contact(2.4, 2.4, 0.32)
    x = c.nodes(H, "node")
    p = R.base_problem(c, x)
    assert p.valid(np.asarray(c.tau_true(x)))
    # Q3d: two ridges with a valley between them (an interior minimum) are not feasible, even
    # with τ > 0 everywhere between them
    w = 1.0 - np.minimum(np.abs(x + 1.0), np.abs(x - 1.0))
    assert not p.valid(w)
    # Q2c: one ridge with τ < 0 is not feasible
    assert not p.valid(-np.abs(x) - 0.1)
    assert p.valid(-np.abs(x) + 0.1)
    # Q1: τ must rise from each root along its normal; a monotone field breaks it at one root
    assert not p.valid(x.copy())


def test_true_state_is_stationary_for_equal_banks() -> None:
    c = CT.Contact(2.4, 2.4, 0.02)
    for centre in ("mid", "node"):
        q, z = R.true_state(c, R.base_problem(c, c.nodes(H, centre)))
        g, _ = P.gradient_hessian(q, z)
        assert np.max(np.abs(g)) < TOL_GRAD


@pytest.mark.parametrize("g", [0.02, 0.32])
@pytest.mark.parametrize("centre", ["mid", "node"])
def test_good_start_finds_both_tops(g: float, centre: str) -> None:
    c = CT.Contact(2.4, 2.4, g)
    x = c.nodes(H, centre)
    fB, fA = R.starts(c, x)["good"]
    tau0, cell, tk, xk = R.kink_start(x, fB, fA)
    p, z, _ = P.solve_kink(R.base_problem(c, x), tau0, cell, tk, xk)
    m = R.side_measures(c, p, z)
    assert abs(m["B"]["top_error_mm"]) < TOL_TOP and abs(m["A"]["top_error_mm"]) < TOL_TOP
    assert p.n_reversals(z) == 1


def test_slab_s1_from_one_mm_out() -> None:
    s = R.slab_s1()
    for k in ("good", "wrong+1mm", "wrong-1mm"):
        assert abs(s[k]["stop_error_mm"]) < TOL_TOP
        assert s[k]["energy"] == pytest.approx(s["energy_true"], abs=1e-9)


def test_collapsed_ridge_is_a_plateau() -> None:
    """A τ > 0 region started away from the sheet shrinks to a point (the ridge on τ = 0); the
    energy of that state does not depend on where the ridge is (the finding of contact1d)."""
    c = CT.Contact(2.4, 2.4, 0.02)
    x = c.nodes(H, "node")
    p = R.base_problem(c, x)
    E = []
    for xr in (-0.8, -0.48, 0.48, 0.8):
        tau = -np.abs(x - xr)
        E.append(p.total(tau))
    assert np.ptp(E) < 1e-9
