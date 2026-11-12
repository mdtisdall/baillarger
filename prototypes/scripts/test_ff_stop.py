"""Tests of ff_stop.py (docs/plans/flow-field-phantoms.md, task 2.1).

The plan's test: on the slab (noise 0), the stop probability is above 0.9 in the
outside medium and below 0.1 deeper than 0.5 mm in the cortex. Every tolerance here
was fixed when the test was written, before any run. The features are synthetic (no
site data): a profile with a weak dark band at mid depth and a strong dark band at the
top, as the site's profile has.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ff_simple as S  # noqa: E402
import ff_stop as FS  # noqa: E402
import ff_types as T  # noqa: E402

_X = (np.arange(50) + 0.5) / 50
_PROF = (
    0.2 * _X
    - 0.05 * np.exp(-0.5 * ((_X - 0.35) / 0.05) ** 2)
    - 0.8 * np.exp(-0.5 * ((_X - 0.95) / 0.04) ** 2)
)
FEATS = {
    "echo2": dict(profile=list(_PROF), wm_level=-0.3, fluid_level=-2.2, hp_stat=0.0),
    "echo3": dict(
        profile=list(1.1 * _PROF), wm_level=-0.37, fluid_level=-1.8, hp_stat=0.0
    ),
}
SHAPE = (40, 40, 40)
N_POINTS = 20000

P_MEDIUM_MIN = 0.9  # the plan
P_DEEP_MAX = 0.1  # the plan
DEEP_MM = 0.5  # the plan
SUM_TOL = 1e-12
CHUNK_TOL = 1e-12
SIGN_TOL = 1e-12


class ConstantTensor:
    """A constant direction with coherence 1."""

    source: T.InputSource = "oracle"

    def __init__(self, d: np.ndarray) -> None:
        self.d = np.asarray(d, float) / np.linalg.norm(d)

    def at(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        return np.tile(self.d, (n, 1)), np.ones(n)


@functools.cache
def slab(noise: float = 0.0) -> T.Phantom:
    feats = FEATS
    if noise:
        feats = {e: dict(FEATS[e], hp_stat=0.1) for e in FEATS}
    return S.slab(feats, noise=noise, seed=0, shape=SHAPE)


def slab_tensor(ph: T.Phantom, sign: float = 1.0) -> ConstantTensor:
    assert ph.geometry is not None
    d = T.true_direction(ph.geometry, np.zeros((1, 3)))[0]
    return ConstantTensor(sign * d)


@functools.cache
def field() -> FS.ImageStopField:
    ph = slab()
    return FS.ImageStopField(ph, slab_tensor(ph), ph)


def points() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, np.array(SHAPE) - 1, (N_POINTS, 3))


def test_the_field_is_a_stop_field() -> None:
    f = field()
    assert isinstance(f, T.StopField) and f.source == "image"
    t = f.terms(points()[:100])
    assert set(t) == {"label", "sheet", "mirror", "partial_volume"}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The plan's slab test fails with the forms and weights of top_confidence.py "
        "(coordinator decision CD-5; limits not changed). Slab, noise 0, 20000 points: "
        "the outside medium within 0.04 mm of the pia has p down to 0.745 (1.6 % of "
        "medium points below 0.9). Deep cortex without the mirror gate: max p 0.272, "
        "40 % of points at p >= 0.1. With the gate (mirror only where S >= S_floor): "
        "max p 0.121, every point at p near 0.12, because the clip bounds give at "
        "best sheet -1 + partial volume -1 = -2. The terms are ranking scores, not "
        "calibrated log-odds: a plan gap for the checkpoint."
    ),
)
def test_slab_medium_and_deep_cortex() -> None:
    """The plan's test."""
    ph, f = slab(), field()
    assert ph.geometry is not None
    X = points()
    Y = ph.geometry.to_local(X)
    sdf = ph.geometry.stop_sdf(Y)
    lab = ph.geometry.label(Y)
    med = sdf > 0
    deep = (lab == T.CORTEX) & (sdf < -DEEP_MM)
    assert med.sum() > 1000 and deep.sum() > 1000
    p = f.probability(X)
    assert p[med].min() > P_MEDIUM_MIN, np.sort(p[med])[:10]
    assert p[deep].max() < P_DEEP_MAX, np.sort(p[deep])[-10:]


def test_terms_sum_to_log_odds() -> None:
    f = field()
    X = points()[:3000]
    t = f.terms(X)
    want = FS.PRIOR + t["label"] + t["sheet"] + t["mirror"] + t["partial_volume"]
    np.testing.assert_allclose(f.log_odds(X), want, rtol=0, atol=SUM_TOL)


def test_deterministic() -> None:
    ph = slab()
    X = points()[:3000]
    a = FS.ImageStopField(ph, slab_tensor(ph), ph).log_odds(X)
    b = FS.ImageStopField(ph, slab_tensor(ph), ph).log_odds(X)
    np.testing.assert_array_equal(a, b)


def test_chunks_do_not_change_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    f = field()
    X = points()[:3000]
    whole = f.log_odds(X)
    parts = np.concatenate(
        [f.log_odds(X[s:e]) for s, e in ((0, 1), (1, 777), (777, 3000))]
    )
    np.testing.assert_allclose(parts, whole, rtol=0, atol=CHUNK_TOL)
    monkeypatch.setattr(FS, "CHUNK", 101)
    np.testing.assert_allclose(f.log_odds(X), whole, rtol=0, atol=CHUNK_TOL)


def test_sign_free_in_the_direction() -> None:
    ph = slab()
    X = points()[:3000]
    a = FS.ImageStopField(ph, slab_tensor(ph, 1.0), ph).log_odds(X)
    b = FS.ImageStopField(ph, slab_tensor(ph, -1.0), ph).log_odds(X)
    np.testing.assert_allclose(a, b, rtol=0, atol=SIGN_TOL)


def test_log_echoes_reproduce_det_with_noise() -> None:
    ph = slab(1.0)
    assert ph.image["noise_sd"][0] > 0
    L2, L3 = FS.log_echoes(ph)  # raises if the detrend differs from det
    assert L2.shape == SHAPE and L3.shape == SHAPE


def test_medium_fraction_of_exact_mixtures() -> None:
    levels = np.array([[-0.3, -0.37], [0.0, 0.05], [-2.2, -1.8]])
    rng = np.random.default_rng(1)
    fr = rng.dirichlet([1, 1, 1], 100)  # WM, cortex, medium
    L = fr @ levels
    f = FS.medium_fraction(L[:, 0], L[:, 1], levels)
    np.testing.assert_allclose(f, fr[:, 2], rtol=0, atol=1e-12)
