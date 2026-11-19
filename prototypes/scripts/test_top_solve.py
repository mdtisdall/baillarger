"""Exact tests of the pieces of T5 (top_solve.py): the crossing t and n_a
as functions of the face's vertices, the stop term and its derivative on a
flat sheet, ∂E_τ/∂gdist, and the constrained move. The whole loop runs on
real site crops (diagnostics), not here."""

from __future__ import annotations

import global_tau_energy as GE
import jax.numpy as jnp
import numpy as np
import top_solve as TSV
import top_surface as TS
import top_tau as TT
from test_global_tau import small_problem
from test_top_tau import H, sheet

from baillarger import mesh_checks as MC


def test_crossing_derivatives_match_finite_differences() -> None:
    rng = np.random.default_rng(0)
    Vf = np.array([[0.2, 0.1, 3.0], [2.9, 0.4, 3.6], [0.5, 3.1, 4.2]]) + rng.normal(scale=0.05, size=(3, 3))
    P = Vf.copy()
    F = np.array([[0, 1, 2]])
    vox = np.full(3, H)
    U, W, T, S, Fx = TT._axis_crossings(P, F, 2, np.array([5, 5, 8]))
    assert len(U) > 0
    uw = np.c_[U, W]
    val, jac = TSV.crossing_derivs(P, F, vox, np.full(len(U), 2), Fx, uw)
    np.testing.assert_allclose(val[:, 0], T, atol=1e-12)
    eps = 1e-6
    for i in range(3):
        for c in range(3):
            Pp = P.copy()
            Pp[i, c] += eps / H  # eps mm
            vp, _ = TSV.crossing_derivs(Pp, F, vox, np.full(len(U), 2), Fx, uw)
            np.testing.assert_allclose((vp - val) / eps, jac[:, :, i, c], rtol=1e-4, atol=1e-6)


def test_stop_energy_on_a_flat_sheet() -> None:
    """Σ at z = 5.3 (normal +z), cortex everywhere (c = -log 19): f = 1 for
    z = 0..4, 0.8 at z = 5 (0.5 + 0.3), 0 above. Raising Σ by δ mm changes E
    by c δ / h per column: the z derivatives sum to n² c / h."""
    n = 6
    v, f = sheet(n, 5.3, True)
    c = np.full((n, n, 10), -GE.LOG_ODDS)
    from types import SimpleNamespace

    st = SimpleNamespace(labels=np.full((n, n, 10), 42, np.int16), vox=np.full(3, H),
                         inside=np.zeros((n, n, 10), bool), dist=np.ones((n, n, 10)))
    sig, _ = TT.sigma_top(st, v * H, f, None, TSV.GS.Settings())
    e, sd = TSV.stop_energy(c, sig.geo, v, f, np.full(3, H))
    assert e == np.float64(-GE.LOG_ODDS * n * n * 5.8) or abs(e + GE.LOG_ODDS * n * n * 5.8) < 1e-9
    assert sd["ramp_voxels"] == n * n
    _, jac = TSV.crossing_derivs(v, f, np.full(3, H), sd["a"], sd["face"], TSV._line_uw(sd["ijk"], sd["a"]))
    g = np.zeros_like(v)
    gv = sd["dE_dt"][:, None, None] * jac[:, 0] + sd["dE_dn"][:, None, None] * jac[:, 1]
    np.add.at(g, f[sd["face"]].ravel(), gv.reshape(-1, 3))
    np.testing.assert_allclose(g[:, 2].sum(), -GE.LOG_ODDS * n * n / H, rtol=1e-9)
    np.testing.assert_allclose(g[:, :2].sum(0), 0.0, atol=1e-9)


def test_energy_derivative_in_gdist() -> None:
    """∂E_τ/∂gdist (JAX) against a finite difference, with contact ghosts."""
    G, P, x, unk, _ = small_problem(faces_z=3)
    P = dict(P)
    P["c"] = jnp.zeros_like(P["c"])
    rng = np.random.default_rng(5)
    gd = rng.uniform(0.2, 1.0, size=(len(x), 3, 2))
    P["gdist"] = jnp.asarray(gd)
    xj = jnp.asarray(x)
    e, dg = TSV._energy_and_dgdist(xj, P)
    dg = np.asarray(dg)
    gh = np.asarray(P["ghost"])
    s, a, d = np.argwhere(gh)[0]
    eps = 1e-6
    gp = gd.copy()
    gp[s, a, d] += eps
    ep, _ = TSV._energy_and_dgdist(xj, {**P, "gdist": jnp.asarray(gp)})
    np.testing.assert_allclose((float(ep) - float(e)) / eps, dg[s, a, d], rtol=1e-4)
    assert np.all(dg[~gh] == 0.0)


def test_constrained_move_keeps_the_gap() -> None:
    """Two facing sheets 0.5 voxels apart; moving the lower one up by 0.4
    voxels would leave 0.1 < the gap 0.25: the move is halved (to 0.2, then
    0.1, leaving 0.3 and 0.4 voxels)."""
    n = 6
    v1, f1 = sheet(n, 5.0, True)
    v2, f2 = sheet(n, 5.5, False)
    V = np.r_[v1, v2] * H
    F = np.r_[f1, f2 + len(v1)]
    m = TS.Mesh(V.copy(), F, V.copy(), np.zeros(len(V), np.int64), np.zeros(len(V)))
    chk = MC.Checker(m.top, m.V, 0.25 * H, slack=0.5 * H)
    d = np.zeros_like(V)
    d[: len(v1), 2] = 0.4 * H
    dd, info = TSV.constrained_move(m, d, chk, MC.FOLD_COS)
    gapz = (V[len(v1):, 2].min() - (V[: len(v1), 2] + dd[: len(v1), 2]).max()) / H
    assert gapz >= 0.25 - 1e-9
    assert info["halved"] > 0 and info["refused"] == 0
    assert dd[: len(v1), 2].max() > 0
