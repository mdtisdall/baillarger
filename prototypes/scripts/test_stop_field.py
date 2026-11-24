"""Tests of stop_field.py: the stop energy on a sparse band, against the
dense form (stop_energy below: top_solve.stop_energy of the prototype branch
claude/tau-shell-cc08e1, on sigma_band's dense signed distance), and the
band on threads against the band as it was first written (band_pairs_sorted
below: every pair collected, then sorted).

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
import numba as nb
import numpy as np
import pytest
import sigma_band as SB
import stop_field as SF
import stop_solve as SV

RUN = Path(os.environ.get("STOP_FIELD_RUN", "/nonexistent"))
CROP = Path(os.environ.get("STOP_FIELD_CROP", "/nonexistent"))


def _icosphere(r: float, levels: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5**0.5) / 2
    V = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        float,
    )
    F = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
    )
    for _ in range(levels):
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        u, inv = np.unique(E, axis=0, return_inverse=True)
        m = len(V) + inv.reshape(3, -1).T  # midpoints of edges 01, 12, 20
        V = np.vstack([V, 0.5 * (V[u[:, 0]] + V[u[:, 1]])])
        F = np.concatenate(
            [
                np.stack([F[:, 0], m[:, 0], m[:, 2]], 1),
                np.stack([F[:, 1], m[:, 1], m[:, 0]], 1),
                np.stack([F[:, 2], m[:, 2], m[:, 1]], 1),
                m,
            ]
        )
    return r * V / np.linalg.norm(V, axis=1, keepdims=True), F


def stop_weights(labels: np.ndarray) -> np.ndarray:
    c = np.zeros(labels.shape)
    c[labels == CS.CORTEX] = -SV.LOG_ODDS
    c[labels == CS.OUTSIDE] = SV.LOG_ODDS
    return c


def stop_energy(
    c: np.ndarray, geo: dict[str, Any], F: np.ndarray, nv: int, h: float
) -> tuple[float, np.ndarray]:
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


@nb.njit(cache=True)
def _band_pairs(
    V: np.ndarray,
    F: np.ndarray,
    vox: np.ndarray,
    shape: np.ndarray,
    band: float,
    lin: np.ndarray,
    face: np.ndarray,
    d2: np.ndarray,
) -> int:
    """Every (voxel, face) pair with the voxel centre within band (mm) of
    the face, face by face: the voxel's linear index, the face, the squared
    distance. With arrays of length 0: the count only."""
    n = 0
    fill = len(lin) > 0
    b2 = band * band
    lo = np.empty(3, np.int64)
    hi = np.empty(3, np.int64)
    p = np.empty(3)
    for f in range(len(F)):
        a, b, c = V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]
        for k in range(3):
            mn = min(a[k], b[k], c[k]) - band
            mx = max(a[k], b[k], c[k]) + band
            lo[k] = max(0, int(np.ceil(mn / vox[k])))
            hi[k] = min(shape[k] - 1, int(np.floor(mx / vox[k])))
        for i in range(lo[0], hi[0] + 1):
            p[0] = i * vox[0]
            for j in range(lo[1], hi[1] + 1):
                p[1] = j * vox[1]
                for k in range(lo[2], hi[2] + 1):
                    p[2] = k * vox[2]
                    w0, w1, w2 = SB._closest_on_triangle(p, a, b, c)
                    q0 = w0 * a[0] + w1 * b[0] + w2 * c[0] - p[0]
                    q1 = w0 * a[1] + w1 * b[1] + w2 * c[1] - p[1]
                    q2 = w0 * a[2] + w1 * b[2] + w2 * c[2] - p[2]
                    dd = q0 * q0 + q1 * q1 + q2 * q2
                    if dd < b2:
                        if fill:
                            lin[n] = (i * shape[1] + j) * shape[2] + k
                            face[n] = f
                            d2[n] = dd
                        n += 1
    return n


def band_pairs_sorted(
    V_mm: np.ndarray,
    F: np.ndarray,
    vox: np.ndarray,
    shape: tuple[int, ...],
    band_vox: float = SB.BAND_VOX,
) -> tuple[dict[str, np.ndarray], int]:
    """stop_field.band_sparse as it was before the slabs and threads (the
    reference): every (voxel, face) pair within the band, sorted by voxel,
    distance and face, the first of each voxel taken. Also returns the
    number of voxels with two or more faces at exactly the nearest distance."""
    vox = np.asarray(vox, float)
    h = float(vox.mean())
    shp = np.asarray(shape, np.int64)
    Vc = np.ascontiguousarray(V_mm, np.float64)
    Fc = np.ascontiguousarray(F, np.int64)
    e = np.zeros(0, np.int64)
    n = _band_pairs(Vc, Fc, vox, shp, band_vox * h, e, e, np.zeros(0))
    lin, fc, d2 = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n)
    _band_pairs(Vc, Fc, vox, shp, band_vox * h, lin, fc, d2)
    o = np.lexsort((fc, d2, lin))
    lin, fc, d2 = lin[o], fc[o], d2[o]
    first = np.r_[True, lin[1:] != lin[:-1]] if n else np.zeros(0, bool)
    ties = int((first[:-1] & ~first[1:] & (d2[1:] == d2[:-1])).sum()) if n else 0
    lin, fc = lin[first], fc[first]
    ijk = np.stack(np.unravel_index(lin, tuple(int(x) for x in shp)), 1).astype(
        np.int64
    )
    x = ijk * vox
    A, B, C = Vc[Fc[fc, 0]], Vc[Fc[fc, 1]], Vc[Fc[fc, 2]]
    w = SB._closest_weights(x, A, B, C)
    q = w[:, :1] * A + w[:, 1:2] * B + w[:, 2:] * C
    fn, en, vn = SB._pseudonormals(Vc, Fc)
    nz = w > 0.0
    cnt = nz.sum(1)
    pn = fn[fc].copy()
    one = cnt == 1
    pn[one] = vn[Fc[fc[one], nz[one].argmax(1)]]
    two = cnt == 2
    pn[two] = en[fc[two], (np.argmin(nz[two], 1) + 1) % 3]
    d = x - q
    dn = np.linalg.norm(d, axis=1)
    s = np.where((d * pn).sum(1) < 0.0, -1.0, 1.0) * dn
    u = fn[fc].copy()
    ok = dn > 1e-12 * h
    u[ok] = d[ok] / s[ok, None]
    return {"ijk": ijk, "s": s, "face": fc, "w": w, "u": u, "lin": lin}, ties


def _same_band(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k


def _cube(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """A cube's surface, two triangles per side, outward."""
    V = np.array(
        [[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)], float
    )
    F = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
    )
    return V, F


@pytest.mark.parametrize("workers", [1, 2, 5, None])
def test_band_sparse_is_the_sorted_pairs_band(workers: int | None) -> None:
    """The slabs on threads give the band of the reference, bit for bit, for
    any number of threads: on a sphere, on a sphere that leaves the grid on
    three sides, and on a cube with its sides on voxel centres (many voxels
    at exactly the same distance from two or more faces: the lowest face is
    taken)."""
    vox = np.array([0.11, 0.1, 0.09])
    shape = (30, 32, 34)
    V, F = _icosphere(1.0, 2)
    for shift in (1.55, 0.6):
        ref, _ = band_pairs_sorted(V + shift, F, vox, shape)
        assert len(ref["face"]) > 1000
        _same_band(ref, SF.band_sparse(V + shift, F, vox, shape, workers=workers))
    h = 0.1
    Vc, Fc = _cube(5 * h, 20 * h)
    ref, ties = band_pairs_sorted(Vc, Fc, np.array([h, h, h]), (26, 26, 26))
    assert ties > 1000
    _same_band(
        ref, SF.band_sparse(Vc, Fc, np.array([h, h, h]), (26, 26, 26), workers=workers)
    )


def test_band_sparse_of_no_face_in_the_grid_is_empty() -> None:
    V, F = _icosphere(1.0, 1)
    b = SF.band_sparse(V + 100.0, F, np.array([0.1, 0.1, 0.1]), (20, 20, 20), workers=3)
    assert b["ijk"].shape == (0, 3) and all(len(b[k]) == 0 for k in b)


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


def _dense(
    labels: np.ndarray, vox: np.ndarray, V: np.ndarray, F: np.ndarray
) -> tuple[float, np.ndarray]:
    band = SB.signed_distance_band(V, F, vox, labels.shape)
    inside, _ = SB.inside_from_band(band, labels.shape)
    return stop_energy(
        stop_weights(labels),
        {"band": band, "inside": inside},
        F,
        len(V),
        float(vox.mean()),
    )


def test_stop_field_on_a_growing_sphere() -> None:
    """Σ grows in steps of h/2 through a shell of cortex into the outside
    medium: E and ∂E/∂V are the dense stop_energy's at every step, with the
    inside kept from step to step."""
    n, h = 40, 0.1
    vox = np.array([h, h, h])
    x = (np.indices((n, n, n)).transpose(1, 2, 3, 0) - (n - 1) / 2) * h
    r = np.linalg.norm(x, axis=-1)
    labels = np.where(r < 0.8, 41, np.where(r < 1.4, CS.CORTEX, CS.OUTSIDE)).astype(
        np.int16
    )
    V0, F = _icosphere(1.0, 3)
    c = (n - 1) / 2 * h
    ev = SF.StopField(SimpleNamespace(labels=labels, vox=vox))
    for k, rad in enumerate(np.arange(0.8, 1.61, 0.05)):
        V = rad * V0 + c
        st = ev(V, F)
        e, g = _dense(labels, vox, V, F)
        assert st.E == pytest.approx(e, abs=1e-8), k
        assert np.allclose(st.grad, g, atol=1e-10), k


@pytest.mark.skipif(
    not RUN.exists() or not CROP.exists(),
    reason="no saved stop run (STOP_FIELD_RUN, STOP_FIELD_CROP)",
)
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
            _, g = stop_energy(
                c,
                {"band": band, "inside": np.zeros(labels.shape, bool)},
                F,
                len(V),
                float(vox.mean()),
            )
            assert np.allclose(st.grad, g, atol=1e-9), k
