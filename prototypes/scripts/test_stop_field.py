"""Tests of stop_field.py: the stop energy on a sparse band, against the
dense form (stop_energy below: top_solve.stop_energy of the prototype branch
claude/tau-shell-cc08e1, on sigma_band's dense signed distance).

test_stop_field_along_a_real_run needs a saved run of stop_run.py with
snapshots (snapshots/solve-000.npz to solve-040.npz) and its crop set, given
by the environment variables STOP_FIELD_RUN and STOP_FIELD_CROP; it is
skipped without them."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import crop_site as CS
import numpy as np
import pytest
import sigma_band as SB
import stop_field as SF
import stop_solve as SV

RUN = Path(os.environ.get("STOP_FIELD_RUN", "/nonexistent"))
CROP = Path(os.environ.get("STOP_FIELD_CROP", "/nonexistent"))


def _icosphere(r: float, levels: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5**0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(levels):
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        u, inv = np.unique(E, axis=0, return_inverse=True)
        m = len(V) + inv.reshape(3, -1).T  # midpoints of edges 01, 12, 20
        V = np.vstack([V, 0.5 * (V[u[:, 0]] + V[u[:, 1]])])
        F = np.concatenate([np.stack([F[:, 0], m[:, 0], m[:, 2]], 1), np.stack([F[:, 1], m[:, 1], m[:, 0]], 1),
                            np.stack([F[:, 2], m[:, 2], m[:, 1]], 1), m])
    return r * V / np.linalg.norm(V, axis=1, keepdims=True), F


def stop_weights(labels: np.ndarray) -> np.ndarray:
    c = np.zeros(labels.shape)
    c[labels == CS.CORTEX] = -SV.LOG_ODDS
    c[labels == CS.OUTSIDE] = SV.LOG_ODDS
    return c


def stop_energy(c: np.ndarray, geo: dict[str, Any], F: np.ndarray, nv: int, h: float) -> tuple[float, np.ndarray]:
    """The dense E_stop = Σ c f, f = clip(0.5 - s/h, 0, 1) in the
    signed-distance band and 1 inside Σ, 0 outside, off it; and ∂E_stop/∂V
    (nv, 3). On the ramp (0 < f < 1), ∂f/∂s = -1/h and ∂s/∂V_i = -w_i u
    (sigma_band.signed_distance_band), so ∂E/∂V_i = c w_i u / h. Only voxels
    with c ≠ 0 count."""
    band = geo["band"]
    live = c != 0
    f = np.where(geo["inside"] & live, 1.0, 0.0)
    X = tuple(band["ijk"].T)
    cv = c[X]
    fr = 0.5 - band["s"] / h
    f[X] = np.where(cv != 0, np.clip(fr, 0.0, 1.0), 0.0)
    ramp = (fr > 0.0) & (fr < 1.0) & (cv != 0)
    e = float((c * f).sum())
    grad = np.zeros((nv, 3))
    coef = (cv[ramp] / h)[:, None] * band["u"][ramp]  # (k, 3)
    fr_ = F[band["face"][ramp]]
    w = band["w"][ramp]
    for k in range(3):
        np.add.at(grad, fr_[:, k], w[:, k : k + 1] * coef)
    return e, grad


def test_band_sparse_is_the_dense_band() -> None:
    V, F = _icosphere(1.0, 2)
    vox = np.array([0.11, 0.1, 0.09])
    shape = (30, 32, 34)
    V = V + 1.55
    a = SB.signed_distance_band(V, F, vox, shape)
    b = SF.band_sparse(V, F, vox, shape)
    assert np.array_equal(a["ijk"], b["ijk"])
    assert np.array_equal(a["face"], b["face"])
    for k in ("s", "w", "u"):
        assert np.allclose(a[k], b[k], atol=1e-13)


def _dense(labels: np.ndarray, vox: np.ndarray, V: np.ndarray, F: np.ndarray) -> tuple[float, np.ndarray]:
    band = SB.signed_distance_band(V, F, vox, labels.shape)
    inside, _ = SB.inside_from_band(band, labels.shape)
    return stop_energy(stop_weights(labels), {"band": band, "inside": inside}, F, len(V), float(vox.mean()))


def test_stop_field_on_a_growing_sphere() -> None:
    """Σ grows in steps of h/2 through a shell of cortex into the outside
    medium: E and ∂E/∂V are the dense stop_energy's at every step, with the
    inside kept from step to step."""
    n, h = 40, 0.1
    vox = np.array([h, h, h])
    x = (np.indices((n, n, n)).transpose(1, 2, 3, 0) - (n - 1) / 2) * h
    r = np.linalg.norm(x, axis=-1)
    labels = np.where(r < 0.8, 41, np.where(r < 1.4, CS.CORTEX, CS.OUTSIDE)).astype(np.int16)
    V0, F = _icosphere(1.0, 3)
    c = (n - 1) / 2 * h
    ev = SF.StopField(SimpleNamespace(labels=labels, vox=vox))
    for k, rad in enumerate(np.arange(0.8, 1.61, 0.05)):
        V = rad * V0 + c
        st = ev(V, F)
        e, g = _dense(labels, vox, V, F)
        assert st.E == pytest.approx(e, abs=1e-8), k
        assert np.allclose(st.grad, g, atol=1e-10), k


@pytest.mark.skipif(not RUN.exists() or not CROP.exists(), reason="no saved stop run (STOP_FIELD_RUN, STOP_FIELD_CROP)")
def test_stop_field_along_a_real_run() -> None:
    """Along the saved Σ of a no-τ run (a step and a remesh between two
    snapshots): ∂E/∂V is the dense one's. E is not compared here: on a crop
    Σ is open at the box, and the dense inside (inside_from_band's flood)
    leaks through the opening (-639,358 against -77,105 at step 4)."""
    z = CS.site_arrays(CROP)
    labels = z["labels"]
    vox = np.abs(np.diag(np.asarray(z["affine"], float))[:3])
    ev = SF.StopField(SimpleNamespace(labels=labels, vox=vox))
    c = stop_weights(labels)
    e_last = np.inf
    for k in range(0, 41):
        s = np.load(RUN / "snapshots" / f"solve-{k:03d}.npz")
        V, F = np.asarray(s["V"], float), np.asarray(s["F"], np.int64)
        st = ev(V, F)
        assert st.E < e_last  # the run's own E fell at every step
        e_last = st.E
        if k % 10 == 0:
            band = SB.signed_distance_band(V, F, vox, labels.shape)
            _, g = stop_energy(c, {"band": band, "inside": np.zeros(labels.shape, bool)}, F, len(V), float(vox.mean()))
            assert np.allclose(st.grad, g, atol=1e-9), k
