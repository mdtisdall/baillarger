"""Pure parts of the column top stage (top_features.py, top_confidence.py)."""
import numpy as np


def two_mode_log_prior(dens: np.ndarray, floor: float = -6.0) -> tuple[np.ndarray, np.ndarray, int]:
    """The global thickness term from a smoothed density of clear label tops.

    Two modes (near-0 thickness and cortical thickness): split at the trough
    between the two highest peaks; each side is the log density relative to its
    own peak, so neither mode penalises the other. With fewer than two peaks,
    relative to the one peak. Returns (log term per sample, peak indices,
    trough index or -1)."""
    dens = np.asarray(dens, np.float64)
    n = len(dens)
    pk = np.flatnonzero((dens > 0) & (dens >= np.r_[0, dens[:-1]]) & (dens > np.r_[dens[1:], 0]))
    pk = np.sort(pk[np.argsort(dens[pk])[-2:]])
    if len(pk) == 2:
        trough = int(pk[0] + np.argmin(dens[pk[0] : pk[1] + 1]))
        peak_of = np.where(np.arange(n) <= trough, dens[pk[0]], dens[pk[1]])
    else:
        trough, peak_of = -1, np.full(n, dens.max())
    logg = np.maximum(np.log(np.maximum(dens / peak_of, 1e-6)), floor)
    return logg, pk, trough


def nearest_class(x: np.ndarray, means: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """Index of the nearest class mean for each sample (last axis: echoes),
    with the squared distance weighted by ci (the inverse noise covariance)."""
    d = [np.einsum("...e,ef,...f->...", x - mu, ci, x - mu) for mu in means]
    return np.argmin(np.stack(d), 0)


def behind_root_lengths(cls: np.ndarray, nin: int, step: float) -> tuple[np.ndarray, np.ndarray]:
    """GM-like (class 0) and fluid-like (class 2) lengths in the first nin
    samples of each row (the samples behind the root), in mm."""
    b = cls[:, :nin]
    return (b == 0).sum(1) * step, (b == 2).sum(1) * step


def gaussian_log_term(x: np.ndarray, sigma: float, cap: float = -8.0) -> np.ndarray:
    """-0.5 (x / sigma)^2, not below cap; NaN stays NaN (the term does not apply)."""
    x = np.asarray(x, np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(x), np.maximum(-0.5 * (x / sigma) ** 2, cap), np.nan)
