"""Tests of ff_oracle.py: the plan's exact test "the oracle stop field is 0.5 on the true stop
surface to within 0.02 mm" (docs/plans/flow-field-phantoms.md), and the oracle tensor and the image
tensor.

Every tolerance here was fixed when the test was written, before any run.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ff_oracle as FO  # noqa: E402
import ff_truth as FT  # noqa: E402
import ff_types as T  # noqa: E402

FEATS = {
    e: dict(
        profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))),
        wm_level=-0.3,
        fluid_level=-2.0,
        hp_stat=0.0,
    )
    for e in ("echo2", "echo3")
}
FEATS_N = {
    e: dict(FEATS[e], hp_stat=3.0) for e in FEATS
}  # a noise level above the image's own

CROSS_TOL_MM = 0.02  # the plan: p = 0.5 within 0.02 mm of the true stop
TANGENT_COS_TOL = (
    1e-6  # 1 - |cos| of the tensor and the path tangent, away from the stop
)
TANGENT_COS_TOL_ALL = 1e-4  # the same at every interior sample
NEAR_STOP_MM = (
    0.05  # the samples nearer than this to the stop are in the looser test only
)
MEDIAN_ANGLE_DEG = 2.0  # the image tensor against the true direction, slab, noise 0
REBUILD_TOL = 1e-12  # the rebuilt, detrended echoes against image["det"]

# name: (root spacing mm, the stop kinds the truth has)
K = T.StopKind
CASES = {
    "slab": (0.64, {K.PIA}),
    "fundus_g0.04": (0.64, {K.CONTACT, K.HAIRPIN}),
    "ckiss_smooth": (0.64, {K.PIA, K.CONTACT, K.HAIRPIN}),
    "abraded": (0.64, {K.PIA, K.HAIRPIN, K.ABRADED}),
}


@functools.cache
def case(name: str) -> tuple[object, T.Truth]:
    geo, ts = FT.geometry_of(name)
    return geo, FT.truth(geo, ts, name, root_spacing_mm=CASES[name][0])


def local_direction(geo: object, Y: np.ndarray) -> np.ndarray:
    G = geo.grad_phi(Y)  # type: ignore[attr-defined]
    return np.asarray(G / np.linalg.norm(G, axis=1)[:, None])


# ---------------------------------------------------------------- the oracle stop field


def test_protocols() -> None:
    geo, _ = case("slab")
    assert isinstance(FO.OracleStopField(geo), T.StopField)  # type: ignore[arg-type]
    assert isinstance(FO.OracleTensor(geo), T.Tensor)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", list(CASES))
def test_half_level_on_the_true_stop_surface(name: str) -> None:
    """Along each true column's direction at its stop, the first point where p reaches 0.5 is
    within CROSS_TOL_MM of the stop. Every StopKind of the phantom is covered."""
    geo, tr = case(name)
    kinds = {T.StopKind(k) for k in np.unique(tr.stop_kind)}
    assert kinds == CASES[name][1]
    stop_field = FO.OracleStopField(geo)  # type: ignore[arg-type]
    Ys = geo.to_local(tr.stops)  # type: ignore[attr-defined]
    u = local_direction(geo, Ys)
    n = len(Ys)

    def lo(s: np.ndarray) -> np.ndarray:
        X = geo.to_vox(Ys + s[:, None] * u)  # type: ignore[attr-defined]
        return stop_field.log_odds(X)

    grid = np.linspace(-CROSS_TOL_MM, CROSS_TOL_MM, 81)
    assert (lo(np.full(n, grid[0])) < 0).all()  # p < 0.5 at the near end
    first = np.full(n, np.nan)
    for s in grid:
        hit = np.isnan(first) & (lo(np.full(n, s)) >= 0)
        first[hit] = s
    assert np.isfinite(first).all(), "p does not reach 0.5 within the tolerance"
    # bisect between the last sample below 0.5 and the first at or above it
    a, b = first - (grid[1] - grid[0]), first.copy()
    for _ in range(40):
        m = 0.5 * (a + b)
        below = lo(m) < 0
        a, b = np.where(below, m, a), np.where(below, b, m)
    cross = 0.5 * (a + b)
    for k in kinds:
        sel = tr.stop_kind == k
        assert sel.any()
        assert np.abs(cross[sel]).max() <= CROSS_TOL_MM, k


@pytest.mark.parametrize("name", list(CASES))
def test_columns_stay_below_half_before_the_stop(name: str) -> None:
    geo, tr = case(name)
    stop_field = FO.OracleStopField(geo)  # type: ignore[arg-type]
    for i in range(tr.n):
        P = tr.path(i)[:-1]  # every sample before the stop
        assert (stop_field.probability(P) < 0.5).all(), (name, i)


def test_log_odds_is_clipped_and_terms_agree() -> None:
    geo, _ = case("slab")
    f = FO.OracleStopField(geo)  # type: ignore[arg-type]
    X = np.array([[10.0, 30.0, 20.0], [30.0, 30.0, 20.0], [50.0, 30.0, 20.0]])
    lo = f.log_odds(X)
    z = np.log(0.98 / 0.02)
    assert np.all(np.abs(lo) <= z + 1e-12)
    np.testing.assert_array_equal(f.terms(X)["oracle"], lo)
    assert f.source == "oracle"


# ---------------------------------------------------------------- the oracle tensor


@pytest.mark.parametrize("name", list(CASES))
def test_oracle_tensor_is_parallel_to_the_path(name: str) -> None:
    """At the interior samples whose two neighbouring steps are full (not the first sample, not the
    last two: the last step is shortened by the bisection), the oracle direction is parallel to the
    central difference of the path. The central difference has an error that grows where the
    curvature grows: the columns that converge on the slot's edge line (the hairpin stops). So the
    strict bound TANGENT_COS_TOL holds at the samples at least NEAR_STOP_MM from the stop, and the
    looser bound TANGENT_COS_TOL_ALL at every interior sample."""
    geo, tr = case(name)
    ten = FO.OracleTensor(geo)  # type: ignore[arg-type]
    worst = worst_all = 0.0
    for i in range(tr.n):
        P = tr.path(i)
        if len(P) < 4:
            continue
        X = P[1 : len(P) - 2]
        tang = P[2 : len(P) - 1] - P[0 : len(P) - 3]
        tang /= np.linalg.norm(tang, axis=1)[:, None]
        d, c = ten.at(X)
        assert (c == 1.0).all()
        err = 1.0 - np.abs(np.sum(d * tang, axis=1))
        far = np.linalg.norm(X - P[-1], axis=1) * T.VOXEL_MM >= NEAR_STOP_MM
        worst_all = max(worst_all, float(err.max()))
        if far.any():
            worst = max(worst, float(err[far].max()))
    assert worst <= TANGENT_COS_TOL
    assert worst_all <= TANGENT_COS_TOL_ALL


# ---------------------------------------------------------------- the image tensor


@functools.cache
def slab_image() -> tuple[T.Phantom, FO.ImageTensor]:
    ph = FT.build_phantom("slab", FEATS_N, noise=0.0, seed=0)
    return ph, FO.ImageTensor.of(ph)


def test_image_tensor_on_the_slab() -> None:
    ph, it = slab_image()
    assert isinstance(it, T.Tensor)
    assert it.source == "image"
    assert it.direction.shape == ph.shape + (3,)
    cortex = ph.labels == 0
    inside = cortex & np.isfinite(it.coherence)
    assert inside.sum() > 1000
    X = np.argwhere(inside).astype(float)
    d, c = it.at(X)
    assert ((c >= 0) & (c <= 1)).all()
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
    true = T.true_direction(ph.geometry, X)  # type: ignore[arg-type]
    ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(d * true, axis=1)), 0, 1)))
    assert np.median(ang) < MEDIAN_ANGLE_DEG


def test_image_tensor_at_voxel_centres_is_the_voxel_array() -> None:
    ph, it = slab_image()
    ijk = np.argwhere(it.valid)[::97]
    d, c = it.at(ijk.astype(float))
    ref = it.direction[tuple(ijk.T)]
    np.testing.assert_allclose(np.abs(np.sum(d * ref, axis=1)), 1.0, atol=1e-9)
    np.testing.assert_allclose(c, it.coherence[tuple(ijk.T)], atol=1e-9)


def test_image_tensor_without_valid_voxels() -> None:
    ph, it = slab_image()
    far = np.array(
        [[0.0, 0.0, 0.0], [-50.0, 3.0, 3.0], [1e3, 1e3, 1e3]]
    )  # in the reach, outside
    d, c = it.at(far)
    np.testing.assert_array_equal(c, 0.0)
    np.testing.assert_array_equal(d, np.tile(FO.NO_DIRECTION, (3, 1)))
    d0, c0 = it.at(np.zeros((0, 3)))
    assert d0.shape == (0, 3) and c0.shape == (0,)


# ---------------------------------------------------------------- the rebuilt echoes

SMALL = {
    "slab": (28, 28, 24),
    "noise": (16, 16, 16),
    "fundus_g0.04": (24, 24, 24),
    "curved_g0.16": (24, 24, 24),
    "abraded": (24, 24, 24),
    "ckiss_smooth": (24, 24, 24),
    "ckiss_smooth_flat": (24, 24, 24),
    "tau": (24, 24, 24),
}


@pytest.mark.parametrize("name", list(SMALL))
def test_rebuilt_echoes_are_the_phantoms_own(name: str) -> None:
    ph = FT.build_phantom(name, FEATS_N, noise=1.0, seed=3, shape=SMALL[name])
    labels, L, sds = FO.phantom_log_echoes(ph)
    assert FO.rebuild_error(ph, L, labels) <= REBUILD_TOL
    assert len(sds) == 2 and min(sds) > 0


def test_a_wrong_seed_is_not_accepted() -> None:
    ph = FT.build_phantom("slab", FEATS_N, noise=1.0, seed=3, shape=SMALL["slab"])
    other = T.Phantom(**{**ph.__dict__, "seed": 4})
    labels, L, _ = FO.phantom_log_echoes(other)
    assert FO.rebuild_error(ph, L, labels) > 1e-3
    with pytest.raises(ValueError):
        FO.ImageTensor.of(other)


def test_oracle_inputs() -> None:
    ph = FT.build_phantom("slab", FEATS_N, noise=0.0, shape=SMALL["slab"])
    inp = FO.oracle_inputs(ph)
    assert inp.record() == dict(stop="oracle", tensor="oracle")
    assert inp.contrast is ph.det
    noise = FT.build_phantom("noise", FEATS_N, noise=1.0, shape=SMALL["noise"])
    with pytest.raises(ValueError):
        FO.oracle_inputs(noise)


LITERAL_FLAT_TOL = 0.03  # the literal option against the closed form on a flat surface


def test_literal_blur_option_matches_the_closed_form_on_the_slab() -> None:
    """On a flat stop surface the literal blur is the closed form (to the quadrature)."""
    geo, tr = case("slab")
    f_cf = FO.OracleStopField(geo)
    f_lit = FO.OracleStopField(geo, literal=True)
    d = tr.paths[:, 1] - tr.paths[:, 0]
    d /= np.linalg.norm(d, axis=1)[:, None]
    s = np.linspace(-0.2, 0.2, 21) / T.VOXEL_MM
    X = (tr.stops[:, None, :] + s[None, :, None] * d[:, None, :]).reshape(-1, 3)
    np.testing.assert_allclose(
        f_lit.probability(X), f_cf.probability(X), rtol=0, atol=LITERAL_FLAT_TOL
    )
    assert np.array_equal(f_lit.log_odds(X), T.log_odds(f_lit.probability(X)))
