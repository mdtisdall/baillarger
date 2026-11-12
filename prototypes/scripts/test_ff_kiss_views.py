"""Tests of the helpers of ff_kiss_views (PR 4, the views and numbers of the kissing
surfaces). Small synthetic inputs only: no solver run, no data files.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_kiss_views.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ff_grid as FG  # noqa: E402
import ff_kiss_views as KV  # noqa: E402

NAN = np.nan


def _padded(paths: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    m = max(len(p) for p in paths)
    out = np.full((len(paths), m, 3), NAN)
    for i, p in enumerate(paths):
        out[i, : len(p)] = p
    return out, np.array([len(p) for p in paths], np.int64)


# ---------------------------------------------------------------- contact crossings


def test_contact_crossings_by_bank_and_tolerance() -> None:
    tol = KV.CROSS_TOL_MM
    s = np.linspace(0.0, 1.0, 11)
    # bank A (x > 0) column that goes to x = -2 tol at y = 1: crosses
    a_cross = np.stack([1.0 - s * (1.0 + 2 * tol), np.ones(11), np.zeros(11)], 1)
    # bank A column that stops at x = -0.5 tol: inside the tolerance, no crossing
    a_near = np.stack([1.0 - s * (1.0 + 0.5 * tol), np.ones(11), np.zeros(11)], 1)
    # bank A column that goes far to x < 0, but only where y < 0: not counted
    a_low = np.stack([1.0 - 2 * s, -np.ones(11), np.zeros(11)], 1)
    # bank B (x < 0) column that goes to x = +3 tol at y = 2: crosses
    b_cross = np.stack([-1.0 + s * (1.0 + 3 * tol), 2 * np.ones(11), np.zeros(11)], 1)
    # bank B column that stays on its own side
    b_ok = np.stack([-1.0 + 0.9 * s, np.ones(11), np.zeros(11)], 1)
    P, counts = _padded([a_cross, a_near, a_low, b_cross, b_ok])
    bank = np.array([0, 0, 0, 1, 1], np.int8)
    crosses, first = KV.contact_crossings(P, counts, bank, tol)
    assert crosses.tolist() == [True, False, False, True, False]
    # the first far sample: x < -tol for bank A, x > tol for bank B
    assert P[0, first[0], 0] < -tol and P[0, first[0] - 1, 0] >= -tol
    assert P[3, first[3], 0] > tol and P[3, first[3] - 1, 0] <= tol
    assert (first[~crosses] == -1).all()


def test_contact_crossings_ignores_padding() -> None:
    # the padding after counts holds far-side values: they must not count
    p = np.array([[0.5, 1.0, 0.0], [0.2, 1.0, 0.0], [-5.0, 1.0, 0.0]])
    P = p[None]
    crosses, _ = KV.contact_crossings(P, np.array([2]), np.array([0], np.int8), 0.02)
    assert not crosses[0]


def test_own_side() -> None:
    Y = np.array([[0.1, 1, 0], [-0.1, 1, 0], [-0.1, 1, 0], [0.1, 1, 0], [0.0, 1, 0]])
    bank = np.array([0, 0, 1, 1, 0], np.int8)
    assert KV.own_side(Y, bank).tolist() == [True, False, True, False, False]


# ---------------------------------------------------------------- lengths along a line


def test_sheet_length_triangle() -> None:
    # q is a triangle of height 1 and base 2 centred at s = 0: q > 0.5 over |s| < 0.5
    s = np.linspace(-2.0, 2.0, 4001)
    q = np.clip(1.0 - np.abs(s), 0.0, None)
    assert KV.sheet_length(s, q, 0.5) == pytest.approx(1.0, abs=1e-9)
    # a coarse sampling: the crossings are linear between samples, so still exact
    s2 = np.array([-1.0, -0.25, 0.25, 1.0])
    q2 = np.clip(1.0 - np.abs(s2), 0.0, None)
    assert KV.sheet_length(s2, q2, 0.5) == pytest.approx(1.0, abs=1e-12)


def test_sheet_length_two_parts_and_none() -> None:
    s = np.arange(0.0, 10.0, 1.0)
    q = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0], float)
    # each part: from the crossing at 0.5 before to the crossing after
    assert KV.sheet_length(s, q, 0.5) == pytest.approx(2.0 + 1.0)
    assert KV.sheet_length(s, np.zeros(10), 0.5) == 0.0
    assert KV.sheet_length(s, np.ones(10), 0.5) == pytest.approx(9.0)


def test_ramp_width_linear() -> None:
    # q rises linearly from 0 at s = 0 to 1 at s = 1: 0.2 at 0.2, 0.8 at 0.8
    s = np.linspace(-1.0, 2.0, 301)
    q = np.clip(s, 0.0, 1.0)
    w = KV.ramp_width(s, q, s0=0.5, lo=0.2, hi=0.8)
    assert w == pytest.approx(0.6, abs=1e-9)


def test_ramp_width_nearest_crossings_and_nan() -> None:
    s = np.linspace(0.0, 4.0, 401)
    # an earlier bump above 0.2 (s near 0.5) must not move the low crossing
    q = np.where(s < 1.0, 0.3 * np.exp(-((s - 0.5) ** 2) / 0.01), 0.0)
    q = q + np.clip(s - 2.0, 0.0, 1.0)  # the ramp: 0 at s = 2, 1 at s = 3
    assert KV.ramp_width(s, q, s0=2.5, lo=0.2, hi=0.8) == pytest.approx(0.6, abs=1e-9)
    # the high level is never reached ahead of the stop: nan
    q2 = np.clip(s - 2.0, 0.0, 0.7)
    assert np.isnan(KV.ramp_width(s, q2, s0=2.5, lo=0.2, hi=0.8))


# ---------------------------------------------------------------- the one-constant fit


def test_fit_constant() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = a + 0.7 + 1e-3 * rng.normal(size=500)
    b[::50] = NAN  # nan pairs are left out
    c, res, n = KV.fit_constant(a, b)
    assert c == pytest.approx(0.7, abs=1e-3)
    assert n == 490
    assert np.nanmax(np.abs(res)) < 5e-3
    # the residual of a constant shift alone is zero to rounding
    c2, res2, _ = KV.fit_constant(a, a - 2.0)
    assert c2 == pytest.approx(-2.0)
    assert np.nanmax(np.abs(res2)) < 1e-12


def test_fit_constant_empty() -> None:
    c, res, n = KV.fit_constant(np.array([NAN]), np.array([1.0]))
    assert n == 0 and np.isnan(c) and np.isnan(res).all()


# ---------------------------------------------------------------- binning


def test_bin_stats() -> None:
    d = np.array([0.05, 0.1, 0.15, 0.2, 0.35, 0.9, 5.0, NAN])
    v = np.array([1.0, 2.0, 3.0, 10.0, 4.0, 7.0, 99.0, 1.0])
    edges = np.array([0.0, 0.16, 0.32, 0.48])
    out = KV.bin_stats(v, d, edges)
    assert [b["n"] for b in out] == [3, 1, 1]
    assert out[0]["median"] == pytest.approx(2.0)
    assert out[0]["max"] == pytest.approx(3.0)
    assert out[1]["median"] == pytest.approx(10.0)
    assert out[0]["lo"] == 0.0 and out[2]["hi"] == pytest.approx(0.48)
    # values that are nan do not count
    out2 = KV.bin_stats(np.array([NAN, 1.0]), np.array([0.1, 0.1]), edges)
    assert out2[0]["n"] == 1


def test_distance_edges() -> None:
    e = KV.distance_edges(0.64, 2.0)
    assert e.tolist() == pytest.approx([0.0, 0.64, 1.28, 1.92, 2.0])
    e2 = KV.distance_edges(0.5, 2.0)
    assert e2.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])


def test_path_distance_ignores_extra_length() -> None:
    s = np.linspace(0.0, 1.0, 101)[:, None]
    A = np.hstack([s, 0 * s, 0 * s])  # along x from 0 to 1
    B = np.vstack([A, A[-1] + np.hstack([s, 0 * s, 0 * s])[1:]])  # goes on to x = 2
    assert KV.path_distance(A, B) == pytest.approx(0.0)
    C = A + np.array([0.0, 0.3, 0.0])  # moved sideways
    assert KV.path_distance(C, B) == pytest.approx(0.3)


# ---------------------------------------------------------------- levels along a column


def test_first_crossings() -> None:
    tau = np.array([-2.0, -1.5, -1.0, -1.2, -0.5, 0.0])  # one step down: running max
    pos = np.stack([np.arange(6.0), np.zeros(6), np.zeros(6)], 1)
    X = KV.first_crossings(tau, pos, np.array([-1.75, -0.75, 0.5]))
    assert X[0, 0] == pytest.approx(0.5)
    assert X[1, 0] == pytest.approx(3.5)  # between -1.0 (index 2, held) and -0.5
    assert np.isnan(X[2]).all()


# ---------------------------------------------------------------- sampling on a grid


def test_sample_nodes_trilinear() -> None:
    grid = FG.UniformGrid((5, 5, 5), 0.32)  # h_vox = 2, nodes at 0, 2, 4
    v = grid.nodes[:, 0] * 0.25  # linear in x: trilinear is exact
    X = np.array([[1.0, 1.0, 1.0], [3.5, 0.0, 4.0], [10.0, 0.0, 0.0]])
    out = KV.sample_nodes(grid, v, X)
    assert out[0] == pytest.approx(0.25)
    assert out[1] == pytest.approx(0.875)
    assert np.isnan(out[2])


# ---------------------------------------------------------------- inference and names


def test_inference_summary() -> None:
    rec = dict(
        sweeps=[200, 60, 20],
        settings=dict(max_sweeps=200),
        measures=dict(
            inference=dict(end_reason=["max_sweeps", "tol_free_energy", "tol_q"])
        ),
        free_energy_rises=[dict(width=4), dict(width=2)],
    )
    s = KV.inference_summary(rec)
    assert s["sweeps"] == [200, 60, 20] and s["max_sweeps"] == 200
    assert s["at_cap"] == [True, False, False]
    assert s["n_rises"] == 2
    # without end reasons: the sweep count against the cap decides
    rec2 = dict(sweeps=[10, 3], settings=dict(max_sweeps=10), measures={})
    s2 = KV.inference_summary(rec2)
    assert s2["at_cap"] == [True, False] and s2["end_reason"] == []


def test_fig_name_and_label() -> None:
    assert KV.fig_name("kiss_free_energy") == "kiss_free_energy.png"
    assert KV.fig_name("kiss_q4_sections_ckiss_smooth", "_u0.32") == (
        "kiss_q4_sections_ckiss_smooth_u0.32.png"
    )
    assert KV.with_label("Title") == "Title"
    assert "0.32 mm stand-in" in KV.with_label("Title", "0.32 mm stand-in")


def test_find_runs_grid_label_and_variants(tmp_path: Path) -> None:
    ph = tmp_path / "ckiss_smooth"
    for d in (
        "oracle_0_uniform0.32",
        "oracle_0_uniform0.32_shiftB",
        "oracle_0_uniform0.32_anchor",
        "image_1_uniform0.32",
        "oracle_0_uniform0.16",  # another grid label: not found
        "oracle_0_uniform0.32_other",  # an unknown suffix: not found
    ):
        (ph / d).mkdir(parents=True)
        (ph / d / "run.json").write_text("{}")
    (ph / "image_0_uniform0.32").mkdir()  # no run.json: not run
    found = KV.find_runs(tmp_path, "uniform0.32", ["ckiss_smooth", "ckiss_sharp"])
    names = sorted(k.dirname("uniform0.32") for k in found)
    assert names == [
        "image_1_uniform0.32",
        "oracle_0_uniform0.32",
        "oracle_0_uniform0.32_anchor",
        "oracle_0_uniform0.32_shiftB",
    ]
    k = KV.RunKey("ckiss_smooth", "oracle", 0.0, "_shiftB")
    assert found[k] == ph / "oracle_0_uniform0.32_shiftB"


def test_speed_vs_true() -> None:
    true = np.array([1.0, 2.0, 4.0, NAN, 1.0])
    solved = np.array([0.5, 1.0, 2.0, 3.0, NAN])  # half the true speed where both exist
    d = KV.speed_vs_true(solved, true)
    assert d["n"] == 3
    assert d["ratio"]["median"] == pytest.approx(0.5)
    assert d["ratio"]["max"] == pytest.approx(0.5)
    assert d["solved"]["median"] == pytest.approx(1.0)
    assert d["true"]["max"] == pytest.approx(4.0)
