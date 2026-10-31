"""Exact-answer tests for pial_height.py (plan: reports/Pial editing and edit priority.md, §5).

- test 5, click mapping: on the wavy body (finding 28's phantom, columns along the true
  normals), the root and arc length interpolated from the march's carried fields at a
  click equal the exact foot point and distance. Tolerances: finding 28's measured error
  of the march from the twicing 0.32 mm surface at the top of the wavy body (arrival
  spread IQR 0.017 / 0.013 mm at ratio 1 / 3; root median / p95 0.052 / 0.171 mm at
  ratio 1 and 0.009 / 0.024 at ratio 3), with room for the trilinear interpolation.
- the columns follow the characteristics: on the wavy body a traced column passes through
  foot + s n at arc length s (same tolerances as the root).
- test 7, column extension on P4 (baillarger.phantoms: in the fused patch the contact is
  the plane x = 25.5, the WM surfaces x = 9.5 and 55.5, so equal arrival is at 32.5):
  bank B's columns meet the ridge 23 voxels up, carry on along the known layer normal,
  and pass x = 25.5 at arc length 30, on their own straight line.

Run: nix develop --command uv run pytest prototypes/scripts/test_pial_height.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import mesh_smoothing as M  # noqa: E402
import pial_height as P  # noqa: E402

from baillarger import phantoms  # noqa: E402

H = P.H_MM
MARGIN = 8.0


def _wavy_case(ratio):
    ph = M.wavy()
    wm, ctx, tr = ph.labels(1)
    fluid = ~wm & ~ctx
    nfield = tr.n.reshape(wm.shape + (3,))
    v, f, n = P.white_surface(wm)
    g = P.march_from_mesh(v, f, wm, ctx, fluid, nfield, ratio)
    return ph, wm, ctx, fluid, v, f, n, g


@pytest.fixture(scope="module", params=[1.0, 3.0], ids=["ratio1", "ratio3"])
def wavy(request):
    return request.param, _wavy_case(request.param)


def _away(ph, X):
    hi = np.array(ph.shape) - 1 - MARGIN
    return np.all((X >= MARGIN) & (X <= hi), 1)


TOL = {  # mm: (|s error| median, p95), (root error median, p95)
    1.0: ((0.02, 0.06), (0.06, 0.20)),
    3.0: ((0.02, 0.06), (0.015, 0.04)),
}


def test_white_surface_outward(wavy):
    _, (ph, wm, ctx, fluid, v, f, n, g) = wavy
    tr = ph.truth(v)
    assert np.mean(np.sum(n * tr.n, 1) > 0.9) > 0.99


def test_click_mapping(wavy):
    ratio, (ph, wm, ctx, fluid, v, f, n, g) = wavy
    rng = np.random.default_rng(5)
    X = rng.uniform(0, np.array(ph.shape) - 1, size=(4000, 3))
    tr = ph.truth(X)
    keep = _away(ph, X) & (tr.d > 2.0) & (tr.d < ph.T - 2.0) & _away(ph, tr.foot)
    X, tr = X[keep][:800], ph.truth(X[keep][:800])
    R, s = P.click_column(g, X)
    es = np.abs(s - tr.d) * H
    er = np.linalg.norm(R - tr.foot, axis=1) * H
    (sm, s95), (rm, r95) = TOL[ratio]
    print(
        f"\nratio {ratio}: |s err| median {np.median(es):.4f} p95 {np.percentile(es, 95):.4f} mm; "
        f"root median {np.median(er):.4f} p95 {np.percentile(er, 95):.4f} mm"
    )
    assert np.median(es) <= sm and np.percentile(es, 95) <= s95
    assert np.median(er) <= rm and np.percentile(er, 95) <= r95


def test_columns_follow_characteristics(wavy):
    ratio, (ph, wm, ctx, fluid, v, f, n, g) = wavy
    C = P.characteristic_field(g)
    nfield = ph.truth(ph.grid(1).reshape(-1, 3)).n.reshape(wm.shape + (3,))
    sel = np.flatnonzero(_away(ph, v))[::7]
    cols = P.trace_columns(v[sel], n[sel], g, C, np.where(ctx[..., None], nfield, 0.0))
    foot = ph.truth(v[sel])
    err = {}
    for s in (5.0, 10.0):
        p = P.point_at(cols, np.full(len(sel), s))
        exact = (
            foot.foot + (s + foot.d)[:, None] * foot.n
        )  # v may sit off the true surface by d
        e = np.linalg.norm(p - exact, axis=1) * H
        err[s] = e[np.isfinite(e)]
        assert np.mean(np.isfinite(e)) > 0.99
    (_, _), (rm, r95) = TOL[ratio]
    for s, e in err.items():
        print(
            f"\nratio {ratio}, s = {s:g}: column error median {np.median(e):.4f} p95 {np.percentile(e, 95):.4f} mm"
        )
        # the vertex's own position error (finding 28: |d| p95 0.027 mm) enters once more here
        assert np.median(e) <= rm + 0.01 and np.percentile(e, 95) <= r95 + 0.03


def test_label_top_open(wavy):
    ratio, (ph, wm, ctx, fluid, v, f, n, g) = wavy
    C = P.characteristic_field(g)
    nfield = ph.truth(ph.grid(1).reshape(-1, 3)).n.reshape(wm.shape + (3,))
    sel = np.flatnonzero(_away(ph, v))[::7]
    cols = P.trace_columns(v[sel], n[sel], g, C, np.where(ctx[..., None], nfield, 0.0))
    t, kind = P.top_label(cols, fluid)
    d0 = ph.truth(v[sel]).d
    # the label's top is the voxel staircase of d <= T: within half a voxel diagonal
    e = t + d0 - ph.T
    print(
        f"\nratio {ratio}: label top - T median {np.median(e):+.3f}, IQR {np.subtract(*np.percentile(e, [75, 25])):.3f} voxels"
    )
    assert np.mean(kind == P.TOP_FLUID) > 0.99
    assert abs(np.median(e)) < 0.25 and np.percentile(np.abs(e), 95) < 0.9


@pytest.fixture(scope="module", params=[1.0, 3.0], ids=["ratio1", "ratio3"])
def p4(request):
    ratio = request.param
    rim, _, e = phantoms.p4_kissing_dark_line()
    wm, ctx, fluid = rim == 2, rim == 3, rim == 1
    xdir = np.zeros(wm.shape + (3,))
    xdir[..., 0] = 1.0
    v, f, n = P.white_surface(wm)
    g = P.march_from_mesh(v, f, wm, ctx, fluid, xdir, ratio)
    C = P.characteristic_field(g)
    # bank B's inner WM face inside the fused patch, away from the patch's edges
    sel = np.flatnonzero(
        (np.abs(v[:, 0] - 55.5) < 0.3)
        & (v[:, 1] >= 16)
        & (v[:, 1] <= 23)
        & (v[:, 2] >= 38)
        & (v[:, 2] <= 43)
    )
    cols = P.trace_columns(v[sel], n[sel], g, C, np.where(ctx[..., None], xdir, 0.0))
    return ratio, v[sel], n[sel], cols


def test_p4_vertices_found(p4):
    _, v, n, _ = p4
    assert len(v) > 50
    assert np.all(n[:, 0] < -0.99)


def test_p4_ridge_then_extension(p4):
    ratio, v, n, cols = p4
    print(
        f"\nratio {ratio}: ridge at {np.median(cols.ridge_s):.2f} voxels (exact 23), "
        f"range {np.min(cols.ridge_s):.2f}-{np.max(cols.ridge_s):.2f}"
    )
    assert np.all(cols.extended)
    np.testing.assert_allclose(cols.ridge_s, 23.0, atol=1.0)


def test_p4_reaches_contact_straight(p4):
    ratio, v, n, cols = p4
    p = P.point_at(cols, np.full(len(v), 30.0))
    assert np.all(np.isfinite(p))
    print(
        f"\nratio {ratio}: x at arc length 30: {np.median(p[:, 0]):.3f} (exact 25.5); "
        f"max off-line {np.max(np.abs(p[:, 1:] - v[:, 1:])):.3f} voxels"
    )
    np.testing.assert_allclose(p[:, 0], 25.5, atol=0.1)
    np.testing.assert_allclose(p[:, 1:], v[:, 1:], atol=0.1)
    assert np.all(cols.stop_why == P.CAP)
