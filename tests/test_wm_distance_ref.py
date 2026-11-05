"""Quick checks of the brute-force WM distance reference."""

import numpy as np
import pytest
from wm_distance_ref import (
    STENCIL_BOUND,
    geodesic_reference,
    obstacle_label_phantom,
    plane_exact,
    plane_phantom,
    reference_bound,
    stencil_bound,
    sulcus_phantom,
)

VOXELS = [(0.16, 0.16, 0.16), (0.16, 0.24, 0.12)]


def test_stencil_bound_constant() -> None:
    assert stencil_bound(2) == pytest.approx(STENCIL_BOUND, rel=1e-9)
    assert 0.04 < STENCIL_BOUND < 0.06
    assert stencil_bound(1) > STENCIL_BOUND > stencil_bound(3)


@pytest.mark.parametrize("vox", VOXELS)
@pytest.mark.parametrize("axis", [0, 1])
def test_plane_matches_exact(vox: tuple[float, float, float], axis: int) -> None:
    shape = (14, 12, 10)
    wm, ctx = plane_phantom(shape, axis, 5)
    ref = geodesic_reference(wm, ctx, vox)
    exact = np.abs(plane_exact(shape, axis, 5, vox))
    tol = np.array([reference_bound(d, vox, 3, 2) for d in exact.ravel()])
    assert np.all(np.abs(ref - exact).ravel() <= tol)


def test_sulcus_detour_longer_than_straight() -> None:
    vox = (0.16, 0.24, 0.12)
    wm, ctx, detour = sulcus_phantom((16, 14, 6))
    ref = geodesic_reference(wm, ctx, vox)
    assert detour.any()
    x = np.indices(wm.shape)[0]
    straight = (x - 5.5) * vox[0]
    assert np.all(ref[detour] > straight[detour])
    assert np.all(np.isfinite(ref[detour]))
    assert np.all(np.isnan(ref[~(wm | ctx)]))


def test_obstacle_blocked_longer() -> None:
    vox = (0.16, 0.16, 0.16)
    shape = (16, 12, 12)
    wm, ctx, blocked = obstacle_label_phantom(shape)
    ref = geodesic_reference(wm, ctx, vox)
    straight = (np.indices(shape)[0] - 5.5) * vox[0]
    assert blocked.any()
    assert np.all(ref[blocked] > straight[blocked])
    assert np.max(ref[blocked] - straight[blocked]) > 0.1
