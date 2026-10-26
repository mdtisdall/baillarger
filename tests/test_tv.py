"""Tests for baillarger.tv: plan test 3 (λ = 0 gives x̂; large λ gives the
mean over the connected columns; λ_l = 0 against a direct convex solve)."""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from baillarger import tv
from baillarger.bands import band_of

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def setup(E: int = 1, seed: int = 4) -> dict[str, Any]:
    """3 x 2 columns of 20 voxels, two bands split at 0.5, a step profile with
    a gain that grows along x, and noise."""
    rng = np.random.default_rng(seed)
    nx, ny, per = 3, 2, 20
    n = nx * ny
    col = np.repeat(np.arange(n), per).astype(np.int64)
    m = np.tile((np.arange(per) + 0.5) / per, n)
    x = col // ny
    prof = np.where(m < 0.5, 0.0, 1.0) * (1 + 0.2 * x)
    f = prof[:, None] * np.linspace(1, 0.7, E)[None, :] + rng.normal(
        0, 0.3, (len(m), E)
    )
    e = []
    for c in range(n):
        if c % ny < ny - 1:
            e.append((c, c + 1))
        if c + ny < n:
            e.append((c, c + ny))
    edges = np.array(e, np.int64)
    z = np.full((n, 1), 0.5)
    Kc = np.full(n, 2)
    return dict(
        col=col,
        m=m,
        f=f,
        a=np.zeros((n, E)),
        s=np.ones((n, E)),
        bandv=band_of(m, z[col]),
        z=z,
        Kc=Kc,
        edges=edges,
        ew=np.linspace(1, 2, len(edges)),
        h=1 / 50,
    )


def run(
    d: dict[str, Any], lg: float, ll: float, it: int = 2000, tol: float = 1e-5
) -> Any:
    return tv.tv_bands(
        d["col"], d["m"], d["f"], d["a"], d["s"], d["bandv"], d["z"], d["Kc"],
        d["edges"], d["ew"], d["h"], lg, ll, it, tol,
    )  # fmt: skip


def test_tv_zero() -> None:
    d = setup(E=3)
    u, xhat, _ = run(d, 0.0, 0.0)
    assert np.allclose(u, xhat)


def test_tv_large_group() -> None:
    d = setup(E=3)
    u, xhat, n = run(d, 1e4, 0.0, 20000, 1e-10)
    mean = (n[..., None] * xhat).sum(0) / n.sum(0)[:, None]
    assert np.allclose(u, mean[None], atol=1e-3)


def test_tv_large_per_depth() -> None:
    d = setup(E=1)
    u, xhat, n = run(d, 0.0, 1e4, 20000, 1e-10)
    mean = (n[..., None] * xhat).sum(0) / n.sum(0)[:, None]
    assert np.allclose(u, mean[None], atol=1e-3)


def objective(uflat: F64, d: dict[str, Any], xhat: F64, n: F64, lg: float) -> float:
    """½ Σ n (u − x̂)² + λ_g Σ w ||P_c u_c − P_c' u_c'||₂ (smoothed by 1e-12)."""
    nc, K, E = xhat.shape
    u = uflat.reshape(nc, K, E)
    G = 50
    h = d["h"]
    centres = (np.arange(G) + 0.5) * h
    gb = np.stack([band_of(centres, d["z"][i]) for i in range(nc)])
    gv = np.clip((d["m"] / h).astype(int), 0, G - 1)
    has = np.zeros((nc, G), bool)
    has[d["col"], gv] = True
    val = 0.5 * float(np.sum(n[..., None] * (u - xhat) ** 2))
    for (i, j), w in zip(d["edges"], d["ew"], strict=True):
        g = has[i] & has[j]
        diff = np.sqrt(h) * (u[i, gb[i, g]] - u[j, gb[j, g]])
        val += lg * w * float(np.sqrt(np.sum(diff**2) + 1e-12))
    return val


def test_group_tv_equals_a_direct_convex_solve() -> None:
    """Plan test 3: with λ_l = 0, Chambolle-Pock reaches the minimum that a
    general solver finds on the (smoothed) objective."""
    for E in (1, 3):
        d = setup(E=E, seed=7)
        lg = 0.3
        u, xhat, n = run(d, lg, 0.0, 50000, 1e-12)
        res = minimize(
            objective,
            xhat.ravel(),
            args=(d, xhat, n, lg),
            method="L-BFGS-B",
            options=dict(maxiter=20000, ftol=1e-15, gtol=1e-12),
        )
        ours = objective(u.ravel(), d, xhat, n, lg)
        assert ours <= res.fun + 1e-6 * abs(res.fun)
        assert np.allclose(u.ravel(), res.x, atol=2e-3)
