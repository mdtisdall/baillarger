"""orient, Part E: the agreement of the surface's normals with the data
normal."""

import numpy as np

from baillarger.orient_diag import surface_agreement

AFF = np.diag([0.2, 0.2, 0.2, 1.0])
AFF[:3, 3] = [1.0, -2.0, 3.0]


def field(d: tuple[float, float, float], shape: tuple[int, int, int] = (10, 10, 10)):  # type: ignore[no-untyped-def]
    nf = np.broadcast_to(
        np.asarray(d, np.float64)[:, None, None, None], (3, *shape)
    ).copy()
    return nf, np.ones(shape), np.ones(shape, np.int8)


def test_the_angle_to_a_uniform_field() -> None:
    """A vertex in the box, the field tilted 30 degrees from its normal:
    30 degrees, weight 1 (the samples at 0.5 and 1 voxel fall in one
    voxel, coherence 1)."""
    t = np.radians(30)
    nf, c, s = field((np.sin(t), 0.0, np.cos(t)))
    box = (slice(0, 10), slice(0, 10), slice(0, 10))
    x = (AFF @ [4.0, 4.0, 4.0, 1.0])[:3][None]
    ang, w = surface_agreement(x, [[0.0, 0.0, 1.0]], nf, c, s, box, AFF)
    assert abs(ang[0] - 30.0) < 1e-9 and w[0] == 1.0


def test_signed_and_unsigned_normals() -> None:
    nf, c, s = field((0.0, 0.0, -1.0))
    box = (slice(0, 10), slice(0, 10), slice(0, 10))
    x = (AFF @ [4.0, 4.0, 4.0, 1.0])[:3][None]
    ang, _ = surface_agreement(x, [[0.0, 0.0, 1.0]], nf, c, s, box, AFF)
    assert abs(ang[0] - 180.0) < 1e-9  # a signed normal facing the other way
    ang, _ = surface_agreement(x, [[0.0, 0.0, 1.0]], nf, c, 0 * s, box, AFF)
    assert abs(ang[0]) < 1e-9  # unsigned: compared as an axis


def test_coherence_weights_and_the_box() -> None:
    """Two voxels with different normals: the coherence-weighted mean.
    A vertex outside the box: NaN, weight 0."""
    nf, c, s = field((0.0, 0.0, 1.0))
    box = (slice(2, 12), slice(0, 10), slice(0, 10))
    # vertex at grid (6, 4, 4) -> box index (4, 4, 4); normal +k: samples at
    # 0.5 and 1 voxel -> k = 4.5 (rounds to 5) and 5: one voxel, (4, 4, 5)
    nf[:, 4, 4, 5] = [1.0, 0.0, 0.0]
    c[4, 4, 5] = 0.5
    x = (AFF @ [6.0, 4.0, 4.0, 1.0])[:3]
    far = (AFF @ [40.0, 4.0, 4.0, 1.0])[:3]
    ang, w = surface_agreement(
        np.stack([x, far]), [[0.0, 0.0, 1.0]] * 2, nf, c, s, box, AFF
    )
    assert abs(ang[0] - 90.0) < 1e-9 and w[0] == 0.5
    assert np.isnan(ang[1]) and w[1] == 0.0
