"""orient, Part C: the whitened tensor, the data normal, its coherence and
its sign (docs/formats/orient.md)."""

import itertools
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import orient_phantoms as op
import pytest
import torch
from test_orient import AFF, write_files

from baillarger import backend, white_dseg
from baillarger import orient as O

DEVICES = [d for d in ("cpu", "mps") if d in backend.available()]
F64 = npt.NDArray[np.float64]


@dataclass
class Model:
    """A noise model (orient.NoiseModel)."""

    sigma: F64
    R: F64
    gain: float
    rest: F64


def diag_model(rest: list[float]) -> Model:
    """No thermal part; C = diag(rest)."""
    e = len(rest)
    return Model(np.zeros(e), np.eye(e), 1.0, np.diag(np.asarray(rest)))


VOX = (0.16, 0.16, 0.16)
CORE = (slice(10, -10),) * 3


def plane(noise: float = 0.0, seed: int = 0) -> tuple[F64, npt.NDArray[np.bool_], F64]:
    shape = (48, 48, 48)
    n = np.array([1.0, 2.0, 2.0]) / 3.0
    e = op.laminae_plane(
        shape, VOX, n.tolist(), 0.8, (0.3, 0.2, 0.1), (600.0, 350.0, 200.0)
    )
    if noise:
        rng = np.random.default_rng(seed)
        e = e * np.exp(op.log_noise(rng, shape, np.eye(3) * noise**2))
    return e, np.ones(shape, bool), n


def discrete_normal(n: F64, period_mm: float) -> F64:
    """The direction of the discrete gradient of a plane wave along n: on
    axis a, sin(k_a) / vs_a, with k_a its wavenumber in radians per voxel
    (the central difference's response; the separable Gaussian scales
    every component by the same factor)."""
    k = 2 * np.pi * n * np.asarray(VOX) / period_mm
    g = np.sin(k) / np.asarray(VOX)
    out: F64 = g / np.linalg.norm(g)
    return out


def test_a_plane_of_laminae() -> None:
    """The normal is the plane's normal as the discrete filters see it
    (discrete_normal: 1.7 degrees from n at a 5-voxel period, 0.8 mm), to
    1e-6; the coherence is near 1."""
    e, ctx, n = plane()
    J = O.orient_arrays(e, ctx, diag_model([1.0, 1.0, 1.0]), VOX)["tensor"]
    v, c = O.leading(J)
    want = discrete_normal(n, 0.8)
    assert np.min(np.abs(v[CORE] @ want)) > 1 - 1e-6
    assert np.degrees(np.arccos(want @ n)) == pytest.approx(1.7, abs=0.1)
    assert np.min(c[CORE]) > 0.99


def test_whitening_with_a_diagonal_c_is_the_weighted_sum() -> None:
    e, ctx, _ = plane(0.02)
    w = [2.0, 0.5, 4.0]
    res = O.orient_arrays(e, ctx, diag_model(w), VOX, echo_tensors=True)
    want = sum(res["echo_tensor"][i] / w[i] for i in range(3))
    np.testing.assert_allclose(
        res["tensor"][(slice(None), *CORE)],
        want[(slice(None), *CORE)],
        rtol=1e-10,
        atol=0,
    )


def test_an_echo_of_pure_noise_has_no_weight() -> None:
    """Echo 3 is replaced by strong noise in another direction's laminae;
    with C_33 large it has no weight, and the normal is echoes 1 and 2's."""
    e, ctx, n = plane()
    other = op.laminae_plane(e.shape[1:], VOX, (1.0, 0.0, 0.0), 0.5, (1.0,), (200.0,))
    e[2] = other[0]
    v, _ = O.leading(
        O.orient_arrays(e, ctx, diag_model([1.0, 1.0, 1e12]), VOX)["tensor"]
    )
    assert np.min(np.abs(v[CORE] @ discrete_normal(n, 0.8))) > 1 - 1e-6
    v2, _ = O.leading(
        O.orient_arrays(e, ctx, diag_model([1.0, 1.0, 1e-3]), VOX)["tensor"]
    )
    assert np.median(np.abs(v2[CORE] @ n)) < 0.9


@pytest.mark.parametrize("dev", DEVICES)
def test_numpy_and_torch_agree(dev: str) -> None:
    """float32 on the device against float64 numpy, on a noisy plane: J
    within 1e-4 relative to its trace, the normal within 1e-3 rad where the
    coherence is above 0.2, the coherence within 1e-4."""
    e, ctx, _ = plane(0.05, 1)
    m = Model(
        np.array([30.0, 28.0, 27.0]), np.eye(3) * 0.9 + 0.1, 2.0, np.eye(3) * 1e-4
    )
    a = O.orient_arrays(e, ctx, m, VOX)["tensor"]
    d = backend.device(dev)
    b = O.orient_arrays(
        torch.as_tensor(e.astype(np.float32), device=d.torch),
        torch.as_tensor(ctx, device=d.torch),
        m,
        VOX,
    )["tensor"]
    va, ca = O.leading_any(a[(slice(None), *CORE)])
    vb, cb = O.leading_any(b[(slice(None), *CORE)].contiguous())
    tr = a[0] + a[3] + a[5]
    bn = b.cpu().numpy()
    assert np.max(np.abs(bn - a)[(slice(None), *CORE)] / tr[CORE]) < 1e-4
    sel = ca > 0.2
    assert sel.mean() > 0.5
    ang = np.arccos(np.clip(np.abs(np.sum(va * vb, -1)), 0, 1))
    assert np.max(ang[sel]) < 1e-3
    assert np.max(np.abs(ca - cb)) < 1e-4
    assert np.array_equal(np.sign(va[sel]), np.sign(vb[sel]))  # canonical


def test_the_distance_gradient() -> None:
    x = np.arange(10, dtype=np.float64)
    d = np.broadcast_to(x[:, None, None] * 0.16, (10, 4, 4)).copy()
    d[0] = np.nan
    d[9] = np.inf
    g = O.distance_gradient(d, VOX)
    np.testing.assert_allclose(g[1:9, :, :, 0], 1.0)
    np.testing.assert_allclose(g[1:9, :, :, 1:], 0.0)
    assert np.all(np.isnan(g[0])) and np.all(np.isnan(g[9]))


def test_the_sign_rule() -> None:
    n = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    g = np.array([[2.0, 0, 0], [-2.0, 0, 0], [np.nan] * 3, [5e-4, 1.0, 0]])
    assert O.sign_of(n, g).tolist() == [1, -1, 0, 1]
    assert O.sign_of(np.array([[1.0, 0, 0]]), np.array([[1e-4, 1.0, 0]])).tolist() == [
        0
    ]


# ------------------------------------------------------------ on the files


@pytest.fixture(scope="module")
def sphere(tmp_path_factory: pytest.TempPathFactory) -> O.OrientSampler:
    paths, dseg = write_files(tmp_path_factory.mktemp("sphere"), period_mm=1.0)
    s = O.OrientSampler(paths, dseg, cells=8)
    s.noise = Model(
        np.array([3.0, 3.0, 3.0]), np.eye(3), O.gradient_gain(), np.eye(3) * 1e-4
    )
    return s


def test_a_sphere_of_laminae_is_radial_and_outward(sphere: O.OrientSampler) -> None:
    r = sphere.orient((0, 0, 0), 3)
    shape = r.coherence.shape
    x = np.indices(shape).astype(np.float64) * sphere.voxel_mm[:, None, None, None]
    rad = x - np.array([3.8, 4.7, 2.9])[:, None, None, None]
    dist = np.linalg.norm(rad, axis=0)
    u = rad / dist
    inner = (dist > 1.9) & (dist < 3.3) & np.isfinite(r.coherence)
    assert inner.sum() > 1000
    cos = np.sum(r.normal * u, 0)
    # the period is 1 mm, 5 to 8 voxels: the central difference turns the
    # normal by up to about 2 degrees (test_a_plane_of_laminae pins it)
    assert np.median(cos[inner]) > 0.99
    assert np.mean(cos[inner] > 0.95) > 0.95
    assert np.all(r.sign[inner] != 0)
    assert np.all(r.sign[~np.isfinite(r.coherence)] == 0)


def test_the_output_is_the_same_from_two_superblocks(sphere: O.OrientSampler) -> None:
    """numpy: normal, coherence and sign the same, bit for bit, from every
    superblock that holds a voxel and from the whole grid."""
    whole = sphere.orient((0, 0, 0), 3, tensor=True)
    for i, j, k in itertools.product(range(1, 5), repeat=3):
        r = sphere.orient((i, j, k), 0, halo_mm=0.4, tensor=True)
        b = r.box
        assert np.array_equal(r.normal, whole.normal[(slice(None), *b)], equal_nan=True)
        assert np.array_equal(r.coherence, whole.coherence[b], equal_nan=True)
        assert np.array_equal(r.sign, whole.sign[b])
        assert r.tensor is not None and whole.tensor is not None
        assert np.array_equal(r.tensor, whole.tensor[(slice(None), *b)], equal_nan=True)


@pytest.mark.parametrize("dev", DEVICES)
def test_the_sampler_on_a_device(sphere: O.OrientSampler, dev: str) -> None:
    a = sphere.orient((0, 0, 0), 3)
    b = sphere.orient((0, 0, 0), 3, device=dev)
    sel = np.nan_to_num(a.coherence) > 0.2
    ang = np.arccos(np.clip(np.sum(a.normal * b.normal, 0), -1, 1))
    assert np.max(ang[sel]) < 1e-3
    assert np.array_equal(a.sign[sel], b.sign[sel])


def test_the_sign_does_not_flip_beside_an_open_sulcus(tmp_path: Path) -> None:
    """WM slabs at x < 6 and x >= 30, cortex between, a slot of outside at
    x = 21-22 (as tests/test_streamlines.py): through tissue, the left bank's
    cortex is nearer its own WM, so its normal points +x up to the slot, and
    the right bank's points -x. Laminae parallel to the slabs."""
    shape = (36, 24, 24)
    lab = np.zeros(shape, np.int16)
    lab[:, 2:-2, 2:-2] = 42
    lab[:6, 2:-2, 2:-2] = 41
    lab[30:, 2:-2, 2:-2] = 41
    lab[21:23] = 0
    x = np.arange(shape[0])[:, None, None] * 0.16
    e = np.stack(
        [
            b * np.exp(a * np.sin(2 * np.pi * x / 0.7)) * np.ones(shape)
            for a, b in ((0.3, 600.0), (0.2, 350.0), (0.1, 200.0))
        ]
    )
    src = tmp_path / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(parents=True)
    aff = np.diag([0.16, 0.16, 0.16, 1.0])
    nib.save(nib.nifti1.Nifti1Image(lab, aff), src)
    white_dseg.main([str(src), "--out", str(tmp_path / "deriv")])
    dseg = tmp_path / "deriv/sub-T01/ses-S1/anat/sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    paths = []
    for i in range(3):
        p = tmp_path / f"sub-T01_ses-S1_echo-{i + 1}_part-mag_FLASH.nii"
        nib.save(nib.nifti1.Nifti1Image(e[i].astype(np.float32), aff), p)
        paths.append(p)
    s = O.OrientSampler(paths, dseg, cells=8)
    s.noise = diag_model([1.0, 1.0, 1.0])
    r = s.orient((0, 0, 0), 3)
    core = (slice(None), slice(6, -6), slice(6, -6))
    nx = r.normal[0][core]
    assert np.all(nx[8:20] > 0.99)
    assert np.all(nx[24:29] < -0.99)
    # the sign multiplies the canonical eigenvector (+x here)
    assert np.all(r.sign[core][8:20] == 1) and np.all(r.sign[core][24:29] == -1)


def test_orient_needs_the_noise(tmp_path: Path) -> None:
    paths, dseg = write_files(tmp_path)
    with pytest.raises(ValueError):
        O.OrientSampler(paths, dseg, cells=8).orient((0, 0, 0), 0)
    _ = AFF
