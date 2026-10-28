"""Tests for baillarger.streamlines: the direction fields and the integrators,
on fields with known answers."""

from typing import Any

import numpy as np
import numpy.typing as npt

from baillarger import streamlines as S

type F64 = npt.NDArray[np.float64]
VOX = np.full(3, 0.16)


def constant_tensor(shape: tuple[int, int, int], axis: int) -> F64:
    """The field e e^T along one axis (coherence 1)."""
    T = np.zeros((6, *shape))
    T[(0, 3, 5)[axis]] = 1.0
    for k in (0, 3, 5):
        if k != (0, 3, 5)[axis]:
            T[k] = 1e-9
    return T


def slab(shape: tuple[int, int, int] = (40, 16, 16)) -> dict[str, Any]:
    """WM below x = 4.5, cortex above, layers parallel to the WM surface."""
    idx = np.indices(shape).astype(float)
    img = np.sin(idx[0] * 0.16 / 0.2)
    labels = np.zeros(shape, np.int16)
    labels[:5] = 2
    ctx = labels == 0
    return dict(img=img * ctx, ctx=ctx, wm=labels == 2, labels=labels)


def test_masked_smooth_keeps_a_constant_inside_the_mask() -> None:
    mask = np.zeros((20, 20, 20), bool)
    mask[5:15, 5:15, 5:15] = True
    out = S.masked_smooth(np.full(mask.shape, 3.0), mask, 2.0)
    assert np.allclose(out[mask], 3.0)


def test_gradient_of_a_linear_field_is_its_slope_over_the_spacing() -> None:
    idx = np.indices((8, 9, 10)).astype(float)
    f = 2.0 * idx[0] - 1.0 * idx[1] + 0.5 * idx[2]
    g = S.gradient(f, (0.5, 1.0, 2.0))
    for gk, want in zip(g, (4.0, -1.0, 0.25), strict=True):
        assert np.allclose(gk, want)


def test_structure_tensor_of_a_ramp_is_rank_one_along_its_slope() -> None:
    idx = np.indices((36, 36, 36)).astype(float)
    a = np.array([1.0, 2.0, -2.0]) / 3.0
    img = np.tensordot(a, idx, 1)
    T = S.structure_tensor(img, np.ones(img.shape, bool))
    n, c = S.eigen(T)
    core = (slice(13, -13),) * 3  # beyond the reach of the two Gaussians' edges
    assert np.allclose(np.abs(n[core] @ a), 1.0, atol=1e-5)
    assert np.allclose(c[core], 1.0, atol=1e-5)


def test_eigen_of_a_known_tensor() -> None:
    T = np.zeros((6, 2, 1, 1))
    T[0], T[3], T[5] = 1.0, 3.0, 2.0  # leading axis y, coherence (3 - 2) / (3 + 2)
    n, c = S.eigen(T)
    assert np.allclose(np.abs(n[..., 1]), 1.0)
    assert np.allclose(c, 0.2)


def test_wm_direction_points_away_from_the_wm() -> None:
    wm = np.zeros((20, 12, 12), bool)
    wm[:6] = True
    g = S.wm_direction(wm, VOX)
    assert np.allclose(g[8:14, 3:-3, 3:-3], [1.0, 0.0, 0.0], atol=1e-6)


def test_oriented_normal_is_signed_away_from_the_wm_and_falls_back_to_g() -> None:
    g = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    n = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = S.oriented_normal(n, np.array([1.0, 0.0]), g, lam=0.1)
    assert np.allclose(F[0], [1.0, 0.0, 0.0])  # the data's normal, flipped
    assert np.allclose(F[1], [1.0, 0.0, 0.0])  # no coherence: g alone


def test_fitted_field_on_a_slab_is_the_distance_from_the_wm() -> None:
    """Layers parallel to the WM, coherence 1: F is the unit normal, so T is
    the distance (mm) from the WM's last voxel row, to solver precision."""
    b = slab()
    m = b["ctx"].astype(float)
    T, info = S.fitted_field(
        S.structure_tensor(b["img"], m) + 0.0, b["ctx"], b["wm"], VOX
    )
    want = (np.arange(40) - 4) * 0.16
    assert np.max(np.abs(T[5:-3, 4:-4, 4:-4] - want[5:-3, None, None])) < 0.01
    assert info["resid"] < 1e-6


def test_fitted_field_is_the_same_on_every_run() -> None:
    """The multigrid setup must not depend on numpy's global generator."""
    b = slab((24, 12, 12))
    tensor = S.structure_tensor(b["img"], b["ctx"].astype(float))
    np.random.seed(1)
    T1, _ = S.fitted_field(tensor, b["ctx"], b["wm"], VOX)
    np.random.seed(2)
    T2, _ = S.fitted_field(tensor, b["ctx"], b["wm"], VOX)
    assert np.array_equal(T1, T2)


def test_gradient_tensor_of_a_linear_field_is_its_direction() -> None:
    idx = np.indices((10, 10, 10)).astype(float)
    T = (3.0 * idx[1] + 4.0 * idx[2]) * 0.16
    R = S.gradient_tensor(T, VOX)
    u = np.array([0.0, 0.6, 0.8])
    want = [u[i] * u[j] for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
    assert np.allclose(R[:, 2, 2, 2], want)


def test_trace_in_a_constant_field_is_a_straight_line_to_the_fluid() -> None:
    """Steps of exactly STEP_MM along the field, signed by the start
    direction; the first sample in the fluid ends the trace there."""
    shape = (60, 20, 20)
    labels = np.zeros(shape, np.int16)
    labels[45:] = 1
    x0 = np.array([[10.1, 10.0, 10.0], [40.0, 10.0, 10.0]])
    d0 = np.array([[0.9, 0.1, 0.0], [-1.0, 0.0, 0.0]])
    T = constant_tensor(shape, 0)
    pos, ns, out = S.trace(T, labels, np.full(shape, 99.0), x0, d0, VOX, 300)
    h = S.STEP_MM / 0.16
    assert out[0] == S.FLUID and out[1] == S.FACE  # the second runs out at x = 0
    assert ns[0] == int(np.ceil((44.5 - 10.1) / h)) + 1  # first sample rounding to 45
    assert np.allclose(np.diff(pos[0, : ns[0], 0]), h, atol=1e-4)
    assert np.allclose(pos[0, 1 : ns[0], 1:], 10.0)
    assert np.allclose(np.diff(pos[1, : ns[1], 0]), -h, atol=1e-4)


def test_trace_stops_where_the_field_turns() -> None:
    """The field turns by 90 degrees at a plane: TURN once the direction
    lag steps back differs by more than turn_deg."""
    shape = (40, 40, 20)
    T = constant_tensor(shape, 0)
    T[:, 20:] = constant_tensor((20, 40, 20), 1)
    pos, ns, out = S.trace(
        T,
        np.zeros(shape, np.int16),
        np.full(shape, 99.0),
        np.array([[10.0, 20.0, 10.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        VOX,
        300,
    )
    assert out[0] == S.TURN
    assert 19.0 < pos[0, ns[0] - 1, 0] < 21.5


def test_blend_turns_from_the_mesh_normal_to_the_data_normal_as_prescribed() -> None:
    """A field e1 e1^T (coherence 1) and a mesh normal e2: each step goes
    along normalize(w e1 + (1 - w) e2), w = min(1, s / blend_mm), s the
    midpoint's arc length."""
    shape = (60, 60, 60)
    pos, ns, _ = S.trace(
        constant_tensor(shape, 0),
        np.zeros(shape, np.int16),
        np.full(shape, 99.0),
        np.array([[30.0, 20.0, 30.0]]),
        np.array([[0.0, 1.0, 0.0]]),
        VOX,
        20,
        blend_mm=0.3,
    )
    step = np.diff(pos[0, : ns[0]], axis=0) * VOX
    for i, st in enumerate(step):
        w = min(1.0, (i + 0.5) * S.STEP_MM / 0.3)
        want = np.array([w, 1 - w, 0.0])
        assert np.allclose(
            st / np.linalg.norm(st), want / np.linalg.norm(want), atol=1e-4
        )


def test_trace_stop_codes() -> None:
    shape = (40, 12, 12)
    labels = np.zeros(shape, np.int16)
    T = constant_tensor(shape, 0)
    x0 = np.array([[5.0, 6.0, 6.0]])
    d0 = np.array([[1.0, 0.0, 0.0]])
    dface = np.full(shape, 99.0)
    for code, lab in ((S.WM, 2), (S.MISSING, 3), (S.OTHER, 4)):
        labels[:] = 0
        labels[20:] = lab
        out = S.trace(T, labels, dface, x0, d0, VOX, 300)[2]
        assert out[0] == code
    labels[:] = 0
    labels[:6] = 2  # WM within wm_after of the root does not stop it
    near = np.full(shape, 99.0)
    near[30:] = 1.0
    out = S.trace(T, labels, near, x0, d0, VOX, 300)[2]
    assert out[0] == S.FACE
    assert S.trace(T, labels, dface, x0, d0, VOX, 5)[2][0] == S.LENGTH


def test_ascend_climbs_the_field_and_stops_at_contact_or_ridge() -> None:
    shape = (60, 12, 12)
    idx = np.indices(shape).astype(float)
    labels = np.zeros(shape, np.int16)
    dface = np.full(shape, 99.0)
    x0 = np.array([[5.0, 6.0, 6.0]])
    never = lambda i, q: np.zeros(len(i), bool)  # noqa: E731
    # u rises to x = 30, then falls: a ridge
    u = np.minimum(idx[0], 60.0 - idx[0])
    pos, ns, out = S.ascend(u, labels, dface, x0, VOX, 400, never)
    h = S.STEP_MM / 0.16
    assert out[0] == S.RIDGE
    assert abs(pos[0, ns[0] - 1, 0] - 30.0) < 2 * h
    assert np.allclose(np.diff(pos[0, : ns[0] - 1, 0]), h, atol=1e-4)
    # the caller's contact test
    beyond = lambda i, q: q[:, 0] >= 20  # noqa: E731
    pos, ns, out = S.ascend(idx[0], labels, dface, x0, VOX, 400, beyond)
    assert out[0] == S.CONTACT and round(float(pos[0, ns[0] - 1, 0])) == 20


def test_ascend_prefers_a_label_to_a_flat_field_to_a_face() -> None:
    shape = (30, 12, 12)
    labels = np.zeros(shape, np.int16)
    never = lambda i, q: np.zeros(len(i), bool)  # noqa: E731
    flat = np.zeros(shape)
    x0 = np.array([[10.0, 6.0, 6.0]])
    near = np.full(shape, 1.0)  # every voxel near a face
    assert S.ascend(flat, labels, near, x0, VOX, 10, never)[2][0] == S.RIDGE
    labels[:] = 1
    assert S.ascend(flat, labels, near, x0, VOX, 10, never)[2][0] == S.FLUID


def test_descend_reaches_the_wm_down_a_linear_field() -> None:
    """T = distance from the WM plane x = 4.5: from x = 20, unit steps of
    STEP_MM straight down, ending on the first sample in the WM."""
    shape = (30, 12, 12)
    labels = np.zeros(shape, np.int16)
    labels[:5] = 2
    T = (np.indices(shape)[0] - 4.5) * 0.16
    pos, ns, out = S.descend(
        T, labels, np.full(shape, 99.0), np.array([[20.1, 6.0, 6.0]]), VOX, 300
    )
    h = S.STEP_MM / 0.16
    assert out[0] == S.WM
    assert np.round(pos[0, ns[0] - 1, 0]) == 4
    assert np.allclose(np.diff(pos[0, : ns[0], 0]), -h, atol=1e-4)
