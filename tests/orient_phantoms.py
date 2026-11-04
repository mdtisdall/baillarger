"""Phantoms for the exact tests of the `orient` sampler: multi-echo magnitude
images with a known data normal (laminae parallel to a plane, or concentric
around a point), and noise generators for the log echoes, for the background
of a multi-channel acquisition and for a signal in complex Gaussian noise.

Coordinates: voxel index i along each axis; the position in mm of the centre of
voxel i is i * vox[axis] (vox: three voxel sizes in mm, possibly anisotropic).
Labels follow FreeSurfer, right hemisphere: 0 outside, 41 WM, 42 cortex.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

type Arr = npt.NDArray[Any]
type Float = npt.NDArray[np.float64]


def _positions(shape: tuple[int, int, int], vox: tuple[float, float, float]) -> Float:
    """Voxel centres in mm, shape (3, *shape)."""
    axes = [np.arange(n, dtype=np.float64) * v for n, v in zip(shape, vox, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"))


def laminae_plane(
    shape: tuple[int, int, int],
    vox: tuple[float, float, float],
    normal: Sequence[float],
    period_mm: float,
    amplitude: Sequence[float],
    base: Sequence[float],
) -> Float:
    """Magnitude images (E, *shape) with laminae parallel to a plane.

    Echo e is I_e = base[e] * exp(amplitude[e] * sin(2 pi (n . x) / period_mm)),
    with n the unit vector along `normal` (a 3-vector in mm space) and x the
    voxel centre in mm. So log I_e = log base[e] + amplitude[e] * sin(...):
    every echo has the same pattern with its own amplitude.

    The exact normal is +-normal (normalised, in mm space).
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    x = _positions(shape, vox)
    phase = 2.0 * np.pi * np.tensordot(n, x, axes=1) / period_mm
    amp = np.asarray(amplitude, dtype=np.float64)
    b = np.asarray(base, dtype=np.float64)
    pattern = np.sin(phase)
    out = b[:, None, None, None] * np.exp(amp[:, None, None, None] * pattern)
    return np.asarray(out, dtype=np.float64)


def laminae_sphere(
    shape: tuple[int, int, int],
    vox: tuple[float, float, float],
    centre_mm: Sequence[float],
    r_wm: float,
    r_pial: float,
    period_mm: float,
    amplitude: Sequence[float],
    base: Sequence[float],
) -> tuple[Float, npt.NDArray[np.int16]]:
    """Concentric laminae in a spherical shell of cortex: (echoes, labels).

    r is the distance in mm of the voxel centre from centre_mm. Labels: 41 (WM)
    where r < r_wm, 42 (cortex) where r_wm <= r < r_pial, 0 beyond.
    Echoes: in cortex I_e = base[e] * exp(amplitude[e] * sin(2 pi (r - r_wm) /
    period_mm)); in WM the constant base[e] * 1.3; outside the constant 1.0
    (dark).

    The exact normal in cortex is radial, outward (pointing away from the WM).
    """
    x = _positions(shape, vox)
    c = np.asarray(centre_mm, dtype=np.float64)
    r = np.sqrt(np.sum((x - c[:, None, None, None]) ** 2, axis=0))
    labels = np.zeros(shape, dtype=np.int16)
    labels[r < r_wm] = 41
    labels[(r >= r_wm) & (r < r_pial)] = 42
    amp = np.asarray(amplitude, dtype=np.float64)[:, None, None, None]
    b = np.asarray(base, dtype=np.float64)[:, None, None, None]
    cortex = b * np.exp(amp * np.sin(2.0 * np.pi * (r - r_wm) / period_mm))
    echoes = np.where(
        labels == 42, cortex, np.where(labels == 41, b * 1.3, 1.0)
    ).astype(np.float64)
    return echoes, labels


def log_noise(rng: np.random.Generator, shape: tuple[int, ...], C: Arr) -> Float:
    """Gaussian noise (E, *shape), white in space (independent voxels), with
    covariance C (E x E, symmetric positive definite) across echoes at each
    voxel. To be added to the log images."""
    cov = np.asarray(C, dtype=np.float64)
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((cov.shape[0], *shape))
    return np.asarray(np.tensordot(chol, z, axes=1), dtype=np.float64)


def _correlated(
    rng: np.random.Generator, shape: tuple[int, ...], sigma: Arr, R: Arr
) -> Float:
    """Real Gaussian field (E, *shape): sd sigma[e], correlation R across echoes."""
    chol = np.linalg.cholesky(np.asarray(R, dtype=np.float64))
    s = np.asarray(sigma, dtype=np.float64)
    z = rng.standard_normal((s.shape[0], *shape))
    return np.asarray(s.reshape((-1,) + (1,) * len(shape)) * np.tensordot(chol, z, 1))


def chi_moments(sigma: Arr, channels: int) -> tuple[Float, Float]:
    """Mean and variance per echo of the background magnitude (see `background`)."""
    s = np.asarray(sigma, dtype=np.float64)
    n = channels
    mean = s * math.sqrt(2.0) * math.exp(math.lgamma(n + 0.5) - math.lgamma(n))
    var = 2.0 * n * s**2 - mean**2
    return mean, var


def background(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    sigma: Arr,
    R: Arr,
    channels: int,
) -> Float:
    """Magnitude noise (E, *shape) of a multi-channel acquisition with no signal.

    For each of `channels` receive channels, a complex Gaussian per echo whose
    real and imaginary parts each have standard deviation sigma[e] and whose
    correlation across echoes is R (E x E correlation matrix; the same R for
    the real parts and for the imaginary parts; real and imaginary parts
    independent; channels independent). The magnitude is
    sqrt(sum over channels of |z|^2). With channels = 1 this is Rayleigh noise.

    For N channels, M / sigma is chi distributed with 2N degrees of freedom
    (scaled chi): mean = sigma * sqrt(2) * Gamma(N + 1/2) / Gamma(N),
    E[M^2] = 2 N sigma^2, variance = E[M^2] - mean^2 (see `chi_moments`).
    The correlation of the magnitudes across echoes is not R.
    """
    acc = np.zeros((np.asarray(sigma).shape[0], *shape), dtype=np.float64)
    for _ in range(channels):
        re = _correlated(rng, shape, sigma, R)
        im = _correlated(rng, shape, sigma, R)
        acc += re**2 + im**2
    return np.sqrt(acc)


def signal_with_noise(
    rng: np.random.Generator, signal: Arr, sigma: Arr, R: Arr
) -> Float:
    """Magnitude of a real positive signal (E, *shape) plus one-channel complex
    Gaussian noise (real and imaginary parts with sd sigma[e], correlated across
    echoes by R, real and imaginary independent).

    At high SNR the noise of log(magnitude) is about sigma[e] / signal_e, with
    correlation R across echoes.
    """
    s = np.asarray(signal, dtype=np.float64)
    shape = s.shape[1:]
    re = _correlated(rng, shape, sigma, R)
    im = _correlated(rng, shape, sigma, R)
    return np.asarray(np.hypot(s + re, im), dtype=np.float64)
