"""Tests for baillarger.bottom: a region whose areal contrast continues into
the WM gets a deep bottom; one whose contrast stops at the WM gets b = 0;
and the stop reasons."""

from typing import Any

import numpy as np
import pytest

from baillarger import bottom as bt


def patch(decay: float, nwm: int = 12, seed: int = 0, E: int = 1) -> dict[str, Any]:
    """6 x 4 columns of 40 cortex voxels and nwm WM voxels: two areas
    (x < 3 and x >= 3) with different profiles, whose difference fades into
    the WM with the rate decay (per unit of r)."""
    rng = np.random.default_rng(seed)
    nx, ny, per = 6, 4, 40
    n = nx * ny
    col = np.repeat(np.arange(n), per + nwm).astype(np.int64)
    zz = np.tile(np.arange(per + nwm), n)
    m = (zz - nwm + 0.5) / per
    x = col // ny
    prof = np.where(x < 3, np.sin(6 * m), np.cos(5 * m))
    prof = prof * np.exp(np.minimum(m, 0) * decay)
    f = prof[:, None] * np.ones((1, E)) + rng.normal(0, 0.1, (len(m), E))
    e = []
    for c in range(n):
        if c % ny < ny - 1:
            e.append((c, c + 1))
        if c + ny < n:
            e.append((c, c + ny))
    reg = (np.arange(n) // ny >= 3).astype(np.int64)
    return dict(
        col=col,
        m=m,
        f=f,
        a=np.zeros((n, E)),
        s=np.ones((n, E)),
        sigma2=np.full((n, E), 0.01),
        reg=reg,
        regions={0: list(np.flatnonzero(reg == 0)), 1: list(np.flatnonzero(reg == 1))},
        nbrs={0: {1}, 1: {0}},
        tmpl_v=np.where(m[:, None] >= 0, prof[:, None], 0.0) * np.ones((1, E)),
        edges=np.array(e, np.int64),
    )


def bottoms(d: dict[str, Any], **kw: Any) -> dict[int, tuple[float, str]]:
    return bt.bottom_depths(
        d["col"], d["m"], d["f"], d["a"], d["s"], d["sigma2"], d["reg"],
        d["regions"], d["nbrs"], d["tmpl_v"], d["edges"], 1 / 40, **kw,
    )  # fmt: skip


def test_contrast_that_continues_into_the_wm_gives_a_deep_bottom() -> None:
    for E in (1, 3):
        out = bottoms(patch(decay=0.0, E=E))
        for b, why in out.values():
            assert why == "end" and b == pytest.approx(-12 / 40)


def test_the_bottom_is_shallower_when_the_contrast_stops_sooner() -> None:
    b = {d: bottoms(patch(decay=d))[0][0] for d in (0.0, 30.0, 200.0)}
    assert b[0.0] < b[30.0] < b[200.0] <= 0
    # test A stops it, a few bins below where the contrast ends
    out = bottoms(patch(decay=200.0))
    assert {why for _, why in out.values()} == {"A"}
    assert all(-0.1 < bb < 0 for bb, _ in out.values())


def test_test_b_stops_noisy_wm() -> None:
    # WM voxels much noisier than the cortex: the template explains less and
    # less of the variance as they are added
    d = patch(decay=200.0)
    wm = d["m"] < 0
    rng = np.random.default_rng(1)
    d["f"][wm] += rng.normal(0, 2.0, d["f"][wm].shape)
    out = bottoms(d, delta_a=1e9)
    assert {why for _, why in out.values()} == {"B"}
    # test B alone does not stop noise-only WM bins as noisy as the cortex
    out = bottoms(patch(decay=200.0), delta_a=1e9)
    assert {why for _, why in out.values()} == {"end"}


def test_too_few_voters() -> None:
    # only one column of each region has WM voxels
    d = patch(decay=0.0)
    keep = (d["m"] >= 0) | np.isin(d["col"], [0, 12])
    for k in ("col", "m", "f", "tmpl_v"):
        d[k] = d[k][keep]
    out = bottoms(d)
    assert {why for _, why in out.values()} == {"voters"}
    assert all(b == 0.0 for b, _ in out.values())
