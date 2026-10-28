"""The image's GM/WM edge against the white mesh: phantoms with a known edge.

The phantom image is a blurred step, erf(d / (sqrt(2) s)), of the signed
distance d to a known surface, so the steepest point along any ray across a
plane is exactly where the ray crosses it. The measure smooths the image's
gradient at scale_mm and samples it by cubic spline; the only error left is
the spline's on a step a voxel or so wide, which depends on the edge's
sub-voxel position (test_interpolation_error records its size).
"""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.special import erf

from baillarger.mesh_smoothing import smooth_normals
from baillarger.white import white_mesh
from baillarger.wm_edge import edge_offsets, tissue_contrast

type F64 = npt.NDArray[np.float64]

VOX = np.array([0.16, 0.16, 0.16])


def _step(d: F64, s: float) -> F64:
    out: F64 = erf(d / (np.sqrt(2) * s))
    return out


def _plane(n: int, normal: F64, x0: F64, s: float, vox: F64 = VOX) -> F64:
    """Image of a plane through x0 (mm) with unit normal (mm), dark on the
    negative side; edge blur s (mm)."""
    X = np.indices((n, n, n)).transpose(1, 2, 3, 0) * vox
    return _step((X - x0) @ normal, s)


def _rot() -> F64:
    a = np.deg2rad([17.0, 31.0])
    nz = np.array(
        [np.sin(a[0]) * np.cos(a[1]), np.sin(a[0]) * np.sin(a[1]), np.cos(a[0])]
    )
    out: F64 = nz / np.linalg.norm(nz)
    return out


def test_plane_exact_offset() -> None:
    """Points at known signed distances from an oblique plane, normals along
    the plane's normal: the offset is the distance to the plane. A wide edge
    (2 voxels) leaves the spline almost nothing: 0.002 mm."""
    nrm = _rot()
    x0 = np.full(3, 32 * 0.16 + 0.037)
    img = _plane(64, nrm, x0, 2 * 0.16)
    rng = np.random.default_rng(0)
    d_true = rng.uniform(-0.25, 0.25, 200)
    tang = rng.normal(size=(200, 3))
    tang -= np.outer(tang @ nrm, nrm)
    p_mm = x0 + 0.5 * tang - np.outer(d_true, nrm)
    e = edge_offsets(img, p_mm / VOX, np.tile(nrm, (200, 1)), VOX)
    assert e.sign == 1.0
    assert not e.at_end.any()
    np.testing.assert_allclose(e.offset_mm, d_true, atol=0.002)


def test_interpolation_error() -> None:
    """A step a voxel wide (s = 0.16 mm), at many sub-voxel positions: the
    spline error after the 0.1 mm derivative scale stays below 0.1 voxel
    (0.016 mm), a fifth of the half voxel at which the label is corrected.
    Without the derivative scale the same step gives 0.085 voxel at s = 0.8
    voxel and 0.155 voxel at s = 0.5 voxel (measured while writing this)."""
    nrm = np.array([1.0, 0.0, 0.0])
    errs = []
    for frac in np.linspace(0.0, 1.0, 21):
        x0 = np.array([20.0 + frac, 8.0, 8.0]) * 0.16
        img = _plane(40, nrm, x0, 0.16)
        p = np.array([[19.6, 8.0, 8.0], [20.3, 8.0, 8.0]])
        e = edge_offsets(img, p, np.tile(nrm, (2, 1)), VOX)
        errs.append(e.offset_mm - (x0[0] - p[:, 0] * 0.16))
    assert np.abs(errs).max() < 0.1 * 0.16


def test_sign_from_the_data() -> None:
    """Bright WM instead of dark: the same offsets, sign -1."""
    nrm = _rot()
    x0 = np.full(3, 32 * 0.16)
    img = _plane(64, nrm, x0, 0.32)
    p = (x0 - np.outer([0.1, -0.05, 0.2], nrm)) / VOX
    a = edge_offsets(img, p, np.tile(nrm, (3, 1)), VOX)
    b = edge_offsets(-img, p, np.tile(nrm, (3, 1)), VOX)
    assert (a.sign, b.sign) == (1.0, -1.0)
    np.testing.assert_allclose(a.offset_mm, b.offset_mm, atol=1e-9)
    np.testing.assert_allclose(a.slope, b.slope, atol=1e-9)
    assert (a.slope > 0).all()


def test_edge_outside_the_window() -> None:
    """An edge 0.5 mm away with a 0.3 mm window: flagged at the window end."""
    nrm = np.array([0.0, 0.0, 1.0])
    x0 = np.full(3, 32 * 0.16)
    img = _plane(64, nrm, x0, 0.32)
    p = (x0 - np.outer([0.5, 0.1], nrm)) / VOX
    e = edge_offsets(img, p, np.tile(nrm, (2, 1)), VOX, window_mm=0.3)
    assert e.at_end.tolist() == [True, False]
    assert e.offset_mm[0] == pytest.approx(0.3, abs=1e-9)


def test_anisotropic_voxels() -> None:
    """Voxels 0.16 x 0.16 x 0.32 mm, an oblique plane: positions and normals
    are converted to mm, so the offset is still the distance to the plane."""
    vox = np.array([0.16, 0.16, 0.32])
    nrm = _rot()
    x0 = np.array([32 * 0.16, 32 * 0.16, 32 * 0.32]) + 0.021
    img = _plane(64, nrm, x0, 0.4, vox)
    d_true = np.array([-0.2, -0.07, 0.0, 0.13, 0.24])
    p_mm = x0 - np.outer(d_true, nrm)
    # the mesh normal in voxel coordinates: a covector, n_mm * vox
    n_vox = nrm * vox
    n_vox /= np.linalg.norm(n_vox)
    e = edge_offsets(img, p_mm / vox, np.tile(n_vox, (5, 1)), vox)
    np.testing.assert_allclose(e.offset_mm, d_true, atol=0.003)


@pytest.mark.parametrize("R", [6.3, 12.3])
@pytest.mark.parametrize("scale", [0.1, 0.15])
def test_ball_shifted_edge(R: float, scale: float) -> None:
    """The white mesh of a ball label (radius R voxels) under an image whose
    edge is a ball 0.6 voxel larger: each vertex's offset is the distance
    along its normal to the image's sphere, less scale² H. That term is the
    derivative scale's own shift on a curved edge (a Gaussian-smoothed step
    is steepest nearer the concave side, by scale² times the mean curvature
    H = 1/radius): 0.005 mm on the larger ball at 0.1 mm. What is left,
    within 0.005 mm (second order in scale / radius on the 1 mm ball), is the
    spline and the mesh normal's tilt."""
    n = 48
    c = np.array([23.7, 24.2, 23.9])
    delta = 0.6
    X = np.indices((n, n, n)).transpose(1, 2, 3, 0)
    r = np.linalg.norm(X - c, axis=-1)
    v, f = white_mesh(r < R)
    nr = smooth_normals(v, f)
    img = _step((r - R - delta) * 0.16, 1.5 * 0.16)
    e = edge_offsets(img, v, nr, VOX, scale_mm=scale)
    b = np.einsum("ij,ij->i", v - c, nr)
    t = -b + np.sqrt(b * b - (np.sum((v - c) ** 2, 1) - (R + delta) ** 2))
    err = e.offset_mm - (t * 0.16 - scale**2 / ((R + delta) * 0.16))
    assert np.abs(err).max() < 0.005


def test_tissue_contrast_keeps_the_step() -> None:
    """Intensity over its normalized convolution over the WM: a
    multiplicative bias field is removed, and the WM/GM step stays put (a
    plane step under a smooth bias that also varies across it: the measured
    edge is where it is on the bare step, within 0.001 mm (the bias's own
    gradient across the edge, 1 %/mm, moves it by about 2 s² g)."""
    n = 48
    nrm = np.array([1.0, 0.0, 0.0])
    x0 = np.array([24.3, 0, 0]) * 0.16
    X = np.indices((n, n, n)).transpose(1, 2, 3, 0) * 0.16
    step = 0.5 * (1 + _plane(n, nrm, x0, 0.24))
    bias = np.exp(0.3 * np.sin(X[..., 1] / 2.0) + 0.2 * X[..., 2] + 0.01 * X[..., 0])
    echo = (100 + 40 * step) * bias
    wm = X[..., 0] < x0[0]
    E = tissue_contrast(echo, wm)
    p = np.array([[23.0, 20.0, 24.0], [25.0, 30.0, 10.0]])
    e = edge_offsets(E, p, np.tile(nrm, (2, 1)), VOX)
    bare = edge_offsets(step, p, np.tile(nrm, (2, 1)), VOX)
    np.testing.assert_allclose(e.offset_mm, bare.offset_mm, atol=0.001)
