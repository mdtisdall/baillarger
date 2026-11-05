"""Tests of baillarger.viewer.filters, with exact answers."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from baillarger.viewer import filters as VF

type Float = npt.NDArray[np.float64]

STEP = 0.04


def pattern(t: Any) -> Float:
    """Bands 0.3-0.5 mm wide, alternating sign (as tests/test_flatten.py)."""
    t = np.asarray(t, float)
    c = [(0.3, 1.0, 0.10), (0.8, -1.2, 0.12), (1.3, 0.8, 0.10), (1.8, -0.9, 0.14),
         (2.4, 0.7, 0.12), (3.0, -0.8, 0.12)]  # fmt: skip
    out: Float = sum(
        (a * np.exp(-0.5 * ((t - m) / s) ** 2) for m, a, s in c), np.zeros_like(t)
    )
    return out


def grid_inputs(
    a: float = 0.0, nx: int = 10, ny: int = 8, sp: float = 0.16, n: int = 101
) -> VF.Inputs:
    """Traces straight up from a plane grid of roots (sp mm apart); echo 2 and echo 3
    (= 0.5 echo 2) are the pattern with horizons z_v(tau) = tau (1 + a (x - xbar)),
    x the root's position (mm): layers thicken across the grid, all meeting the
    roots at tau = 0; every trace 4 mm long."""
    xs, ys = np.meshgrid(np.arange(nx) * sp, np.arange(ny) * sp, indexing="ij")
    root = np.c_[xs.ravel(), ys.ravel(), np.zeros(nx * ny)]
    N = len(root)
    z = np.arange(n) * STEP
    ns = np.full(N, n, np.int64)
    g = 1 + a * (root[:, 0] - root[:, 0].mean())
    f = np.stack([pattern(z / g[i]) for i in range(N)])
    return VF.Inputs(
        fv=np.stack([f, 0.5 * f]),
        ns=ns,
        keep=np.arange(N, dtype=np.int64),
        root_mm=root,
        nrm=np.tile([0, 0, 1.0], (N, 1)),
    )


def quiet(m: str) -> None:
    pass


def test_params_parse_and_key() -> None:
    p = VF.Params.parse(dict(echo="2+3", flatten="dyn", smax=0.6))
    assert p == VF.Params(echo="2+3", flatten="dyn", smax=0.6)
    assert p.key("a") == VF.Params.parse(p.as_dict()).key("a") != p.key("b")
    assert p.key("a") != VF.Params().key("a")
    for bad in (dict(echo="4"), dict(flatten="x"), dict(smax=0), dict(other=1)):
        with pytest.raises(ValueError):
            VF.Params.parse(bad)


def test_echo_choice_is_each_echo_over_its_sd() -> None:
    inp = grid_inputs()
    r2 = VF.compute(inp, VF.Params(echo="2"), "k", quiet)
    r3 = VF.compute(inp, VF.Params(echo="3"), "k", quiet)
    rb = VF.compute(inp, VF.Params(echo="2+3"), "k", quiet)
    sd = float(np.std(inp.fv[0]))
    assert np.allclose(r2.prof, inp.fv[0] / sd)
    # echo 3 is half echo 2: over its own SD it is the same
    assert np.allclose(r3.prof, r2.prof) and np.allclose(rb.prof, r2.prof)
    assert r2.frame == "depth" and r2.z is None and r2.layers == {}


def test_smoothing_keeps_a_constant_and_each_trace_end() -> None:
    inp = grid_inputs()
    inp.fv[:] = 3.0
    inp.ns[0] = 50
    inp.fv[:, 0, 50:] = np.nan
    r = VF.compute(inp, VF.Params(smooth_mm=0.16), "k", quiet)
    assert np.isnan(r.prof[0, 50:]).all()
    # a constant over its own samples stays constant, also at the end (normalized)
    fin = r.prof[np.isfinite(r.prof)]
    assert np.allclose(fin, fin[0])


def test_stack_weights_and_average() -> None:
    """Two traces 0.2 mm apart, the second 2 x the first + 1 (correlation 1); a third
    anticorrelated (weight 0)."""
    a = pattern(np.arange(60) * STEP)
    G = np.stack([a, 2 * a + 1, -a])
    pairs = np.array([[0, 1], [0, 2]])
    d = np.array([0.2, 0.2])
    out, wsum = VF.stack(G, pairs, d, 0.5)
    w = np.exp(-(0.2**2) / (2 * 0.25**2))
    assert np.allclose(out[0], (a + w * (2 * a + 1)) / (1 + w))
    assert np.allclose(out[1], (2 * a + 1 + w * a) / (1 + w))
    assert np.allclose(out[2], -a)
    assert np.allclose(wsum, [w, w, 0.0])


@pytest.mark.parametrize("est", ["lk", "dyn"])
def test_flattening_levels_thickening_layers(est: str) -> None:
    """Layers 7 % thicker at one side of the grid than the other (horizons fan out
    from the roots): in tau every profile is the same (the mean horizon's), the warp
    is the true one, nothing folds, and the layers are there."""
    inp = grid_inputs(a=0.1)
    r = VF.compute(inp, VF.Params(flatten=est), "k", quiet)
    assert r.frame == "tau" and r.z is not None and r.tau is not None
    inner = slice(int(0.4 / STEP), int(3.0 / STEP))
    P = r.prof[:, inner]
    raw = VF.compute(inp, VF.Params(), "k", quiet).prof[:, inner]
    spread_flat = np.nanmax(np.abs(P - np.nanmedian(P, 0)))
    spread_raw = np.nanmax(np.abs(raw - np.nanmedian(raw, 0)))
    assert spread_flat < 0.1 * spread_raw
    x = inp.root_mm[:, 0]
    true = r.tau[None] * (1 + 0.1 * (x - x.mean()))[:, None]
    lv = (r.tau > 0.3) & (r.tau < 3.0)
    assert np.max(np.abs(r.z[:, lv] - true[:, lv])) < 0.01
    for k in ("residual rms (mm)", "fold fraction", "shifts at the bound (fraction)"):
        assert r.layers[k].shape == (len(x),)
    assert np.nanmax(r.layers["fold fraction"]) == 0
    # a depth on a trace maps to its tau: the horizon at tau = 2.0 on each trace
    j = int(round(2.0 / VF.DTAU))
    for row in (0, len(x) - 1):
        assert abs(r.depth_to_frame(row, r.z[row, j : j + 1])[0] - 2.0) < 1e-9


def test_result_round_trip(tmp_path: Path) -> None:
    inp = grid_inputs(a=0.1)
    p = VF.Params(flatten="lk", stack_mm=0.2)
    r = VF.compute(inp, p, p.key("d"), quiet)
    r.save(tmp_path / "x.npz")
    q = VF.Result.load(tmp_path / "x.npz")
    assert q.params == p and q.key == r.key and q.frame == "tau"
    assert np.allclose(q.prof, r.prof, equal_nan=True)
    assert q.z is not None and r.z is not None and np.allclose(q.z, r.z, atol=1e-5)
    assert set(q.layers) == set(r.layers) and "stack weight" in q.layers
