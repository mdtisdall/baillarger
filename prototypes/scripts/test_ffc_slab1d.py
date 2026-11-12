"""Tests of ffc_line and ffc_solve1d on slab1d (the continuous column energy, route R1, stop
form S2).

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ffc_slab1d.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import integrate
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402
import ffc_line as LN  # noqa: E402
import ffc_solve1d as F  # noqa: E402

H = 0.16
# Tolerances, fixed before any run.
TOL_QUAD = 1e-10  # the Gauss rule against adaptive quadrature (nats per mm)
TOL_GRAD = 1e-6  # the gradient at the true state (nats per unit τ)
TOL_STOP = 1e-6  # the solved stop against the true stop (mm)
TOL_TAU = 1e-6  # the solved τ against the true τ
TOL_CONT = 1e-9  # the energy on both sides of a node or a midpoint


@pytest.fixture(scope="module")
def setup() -> tuple[LN.Line, F.Discretisation, F.Constants]:
    line = LN.slab1d()
    d = F.discretise(line, H)
    return line, d, F.constants(line.settings)


def test_line_oracle_and_truth(setup: tuple[LN.Line, F.Discretisation, F.Constants]) -> None:
    line, d, _ = setup
    x = np.array([-1.0, 0.0, 1.2, 2.4, 2.45, 3.0])
    assert np.allclose(line.tau_true(x), x - 2.4, atol=1e-12)
    assert np.allclose(line.speed_true(x), 1.0)
    p = ndtr((x - 2.4) / 0.08)
    assert np.allclose(line.stop_probability(x), p, atol=1e-15)
    assert np.allclose(line.stop_log_odds(x), T.log_odds(p, 0.02, 0.98))
    assert line.stop_log_odds(np.array([2.4]))[0] == pytest.approx(0.0, abs=1e-12)
    # the true stop lies in the middle of a cell
    assert (line.x_stop_true - line.x_start) / H == pytest.approx(27.5)
    assert d.x[0] == line.x_start and d.x[-1] >= line.x_end


def test_slope_operator(setup: tuple[LN.Line, F.Discretisation, F.Constants]) -> None:
    _, d, _ = setup
    tau = np.sin(d.x)
    assert np.allclose(d.G @ tau, np.diff(tau) / np.diff(d.x), atol=1e-12)


def test_gauss_rule(setup: tuple[LN.Line, F.Discretisation, F.Constants]) -> None:
    line, d, _ = setup
    for a, b in ((2.32, 2.48), (2.32, 2.41), (2.39, 2.48), (2.0, 2.16)):
        for sign in (1.0, -1.0):
            got = float(F._quad(d, torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64), sign))
            ref = integrate.quad(
                lambda x, s=sign: float(np.logaddexp(0.0, s * line.stop_log_odds(np.array([x]))[0])),
                a, b, epsabs=1e-13, epsrel=1e-13, limit=200,
            )[0]
            assert got == pytest.approx(ref, abs=TOL_QUAD)


def test_true_state_is_stationary(setup: tuple[LN.Line, F.Discretisation, F.Constants]) -> None:
    line, d, k = setup
    st = F.true_state(d)
    na = F.n_active(d.x, st.s)
    f = F._fun(d, k, st, "J0", na)
    z = torch.as_tensor(st.tau[:na].copy()).requires_grad_(True)
    g = torch.autograd.grad(f(z), z)[0].numpy()
    assert np.max(np.abs(g)) < TOL_GRAD
    e = F.terms(d, k, st, "J0")
    assert e["speed"] == pytest.approx(0.0, abs=1e-20)
    assert e["jump"] == pytest.approx(-math.log(0.999))


@pytest.mark.parametrize("s", [2.32, 2.40])  # a node and a cell midpoint
def test_energy_continuous_in_s(setup: tuple[LN.Line, F.Discretisation, F.Constants], s: float) -> None:
    _, d, k = setup
    slope = 1.3  # not the prior's mean, so that u != 0
    E = []
    for ds in (-1e-12, 1e-12):
        st = F.State(slope * (d.x - (s + ds)), s + ds)
        E.append(F.total_energy(d, k, st, "J0"))
    assert E[0] == pytest.approx(E[1], abs=TOL_CONT)


@pytest.mark.parametrize("s0", [2.32, 3.4, 1.4])
def test_solve_reaches_the_truth(setup: tuple[LN.Line, F.Discretisation, F.Constants], s0: float) -> None:
    line, d, k = setup
    st, log = F.solve(d, k, F.State(d.x - s0, s0), "J0")
    assert log.converged
    assert abs(st.s - line.x_stop_true) < TOL_STOP
    mov = d.x <= st.s
    assert np.max(np.abs(st.tau[mov] - line.tau_true(d.x[mov]))) < TOL_TAU
    assert all(b <= a + 1e-12 for a, b in zip(log.energy, log.energy[1:], strict=False))


def test_jump_branch_is_linear(setup: tuple[LN.Line, F.Discretisation, F.Constants]) -> None:
    _, d, k = setup
    st = F.true_state(d)
    E0 = F.total_energy(d, k, st, "J0")
    Js = np.array([1e-6, 0.01, 0.3])
    got = F.jump_branch(d, k, st, Js)
    want = E0 - k.e0 + k.e1 + Js / k.jump_mean
    assert np.allclose(got, want, atol=1e-12)
