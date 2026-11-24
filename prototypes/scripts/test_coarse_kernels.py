"""The torch kernels of the coarse scan against the numpy reference: float64
on the CPU must agree to rounding; float32 on MPS (when present) closely."""

import coarse_kernels as K
import numpy as np
import pytest
import torch

RNG = np.random.default_rng(7)
MPS = torch.backends.mps.is_available()


def t64(a):
    return torch.as_tensor(np.asarray(a, np.float64))


def nan_close(a, b, atol):
    a, b = np.asarray(a), np.asarray(b)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    assert np.allclose(a[m], b[m], atol=atol, rtol=0)


def spd(n, gap):
    """Symmetric matrices with eigenvalues (1 + gap, 1, 0.5) in random frames."""
    Q, _ = np.linalg.qr(RNG.normal(size=(n, 3, 3)))
    w = np.array([0.5, 1.0, 1.0 + gap])
    return np.einsum("nij,j,nkj->nik", Q, w, Q), Q[:, :, 2]


def test_sample_matches_map_coordinates_inside_outside_and_at_nodes() -> None:
    vol = RNG.normal(size=(2, 10, 12, 9))
    q = np.r_[RNG.uniform(-2, 13, size=(300, 3)), RNG.integers(0, 9, size=(20, 3)).astype(float)]
    nan_close(K.sample_t(t64(vol), t64(q)).numpy(), K.sample_np(vol, q), 1e-12)


def test_sample_nan_corners() -> None:
    vol = RNG.normal(size=(1, 8, 8, 8))
    vol[0, 4:, :, :] = np.nan
    q = np.r_[RNG.uniform(0, 7, size=(400, 3)), [[3.0, 2.5, 2.5], [3.5, 2.0, 2.0]]]
    nan_close(K.sample_t(t64(vol), t64(q)).numpy()[0], K.sample_np(vol[0], q), 1e-12)


@pytest.mark.parametrize("gap", [0.05, 0.5, 5.0])
def test_leading_matches_eigh(gap: float) -> None:
    M, v_true = spd(500, gap)
    v, w1, w2 = K.leading_t(t64(M))
    w = np.linalg.eigh(M)[0]
    assert np.allclose(w1.numpy(), w[:, 2], atol=1e-9)
    assert np.allclose(w2.numpy(), w[:, 1], atol=1e-9)
    assert np.allclose(np.abs((v.numpy() * v_true).sum(1)), 1.0, atol=1e-9)


def field_box():
    """A smooth tensor field whose leading direction turns along z, with a
    NaN corner, on a 20^3 box."""
    X, Y, Z = np.meshgrid(*[np.arange(20.0)] * 3, indexing="ij")
    ang = 0.08 * Z
    d = np.stack([np.sin(ang), np.zeros_like(ang), np.cos(ang)], -1)
    M = 0.3 * np.eye(3) + 2.0 * d[..., :, None] * d[..., None, :]
    M[..., 1, 1] += 0.2 * (1 + np.sin(0.3 * X))
    T = np.stack([M[..., i, j] for i, j in K.PAIRS])
    T[:, :3, :3, :3] = np.nan
    return T


def test_field_matches_numpy() -> None:
    T = field_box()
    q = RNG.uniform(0, 19, size=(400, 3))
    D = np.diag([0.16, 0.16, 0.16]) @ np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1.0]])
    v_n, c_n, ok_n = K.field_np(T, q, D)
    v_t, c_t, ok_t = K.field_t(t64(T), t64(q), t64(D))
    assert np.array_equal(ok_t.numpy(), ok_n)
    assert np.allclose(c_t.numpy(), c_n, atol=1e-9)
    hi = ok_n & (c_n > 0.1)
    assert np.allclose(np.abs((v_t.numpy()[hi] * v_n[hi]).sum(1)), 1.0, atol=1e-9)


def trace_case():
    T = field_box()
    out0 = np.zeros((20, 20, 20))
    out0[:, :, 15:] = 1.0
    Ainv = np.eye(3) / 0.16
    b = np.zeros(3)
    D = np.eye(3) * 0.16
    R = np.c_[RNG.uniform(0.8, 2.4, size=(60, 2)), np.full(60, 0.7)]
    N = np.c_[RNG.normal(scale=0.2, size=(60, 2)), np.ones(60)]
    s_grid = np.arange(int(round(2.4 / 0.04)) + 1) * 0.04
    return T, out0, R, N, Ainv, b, D, s_grid


@pytest.mark.parametrize("early", [0.0, 0.5])
def test_trace_matches_numpy(early: float) -> None:
    T, out0, R, N, Ainv, b, D, s_grid = trace_case()
    P_n, sh_n = K.trace_np(T, out0, R, N, Ainv, b, D, 2.4, 0.16, early, s_grid)
    P_t, sh_t = K.trace_t(t64(T), t64(out0[None]), R, N, Ainv, b, D, 2.4, 0.16, early, s_grid)
    assert sh_t == sh_n
    assert np.allclose(P_t, P_n, atol=1e-9)


def test_profile_matches_numpy() -> None:
    echo = np.exp(RNG.normal(3, 0.5, size=(20, 20, 20)))
    Pq = RNG.uniform(0, 19, size=(30, 50, 3))
    p_n = K.profile_np(echo, Pq, 12.5)
    p_t = K.profile_t(t64(echo[None]), t64(Pq), 12.5).numpy()
    assert np.allclose(p_t, p_n, atol=1e-9)


def curves_case():
    m, ns = 40, 120
    sp = np.arange(ns) * 0.04
    fa, gb = RNG.normal(size=(m, ns)), RNG.normal(size=(m, ns))
    top = RNG.uniform(0.2, 4.5, size=m)
    top[3] = np.nan
    pairs = np.c_[RNG.integers(0, m, 80), RNG.integers(0, m, 80)]
    return fa, gb, pairs, top, sp


def test_curves_match_numpy() -> None:
    fa, gb, pairs, top, sp = curves_case()
    sh = np.arange(-4, 5)
    c_n = K.curves_np(fa, gb, pairs, top, sp, 0.3, sh, 10, 1)
    c_t = K.curves_t(t64(fa), t64(gb), pairs, top, sp, 0.3, sh, 10, 1)
    nan_close(c_t, c_n, 1e-9)
    assert np.isfinite(c_n).any() and np.isnan(c_n).any()


@pytest.mark.skipif(not MPS, reason="no MPS device")
def test_mps_float32_close_to_numpy() -> None:
    dev = torch.device("mps")

    def f32(a):
        return torch.as_tensor(np.asarray(a, np.float32), device=dev)

    T, out0, R, N, Ainv, b, D, s_grid = trace_case()
    P_n, _ = K.trace_np(T, out0, R, N, Ainv, b, D, 2.4, 0.16, 0.5, s_grid)
    P_t, _ = K.trace_t(f32(T), f32(out0[None]), R, N, Ainv, b, D, 2.4, 0.16, 0.5, s_grid)
    assert np.abs(P_t - P_n).max() < 1e-4
    fa, gb, pairs, top, sp = curves_case()
    sh = np.arange(-4, 5)
    c_n = K.curves_np(fa, gb, pairs, top, sp, 0.3, sh, 10, 1)
    c_t = K.curves_t(f32(fa), f32(gb), pairs, top, sp, 0.3, sh, 10, 1)
    nan_close(c_t, c_n, 1e-4)
