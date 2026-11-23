"""Σ moved by the stop energy alone, on a sparse band (user, 2026-11-23: a
first stage "more like FreeSurfer", whose point is the topology of Σ; "build
it" for a whole hemisphere).

The evaluator of stop_solve.SolveRun, with no array of the grid's size but
two: the labels, and one byte per voxel for "inside Σ". From stop_field.py
of the prototype branch claude/tau-shell-cc08e1, with the same results.

E_stop = Σ c f over the voxels with c ≠ 0 (cortex: -LOG_ODDS, outside
medium: +LOG_ODDS): f = clip(1/2 - s/h, 0, 1) within
BAND_VOX voxels of Σ (s the signed distance), 1 inside Σ and 0 outside off
that band.

The band is found per face (band_sparse: each face visits the voxels of its
widened bounding box; the pairs within the band are kept and the nearest
face of each voxel taken), so the cost follows Σ's area, not the grid.

Inside Σ off the band is kept, not found again: a voxel in the band has the
sign of its s; a voxel that leaves the band keeps the side it was last seen
on. That is right while two Σ evaluated one after the other are less than
BAND_VOX voxels apart (a step is at most h/2): a voxel that changes side
between them is within that distance of both, so in both bands. At the
start, every cortex or outside-medium voxel off the band is taken as
outside (the start Σ is the white surface: no cortex behind it but in the
band). E is kept the same way: the sum over the band, plus the sum of c
over the inside voxels off it, updated by what the band wrote.

The energy and its derivative are those of the dense form (top_solve.
stop_energy of the prototype branch, on sigma_band.signed_distance_band and
inside_from_band; tested against a copy of it, test_stop_field.py).
"""

from __future__ import annotations

import time
from typing import Any

import crop_site as CS
import numba as nb
import numpy as np
import sigma_band as SB
import stop_solve as SV


@nb.njit(cache=True)
def _band_pairs(V: np.ndarray, F: np.ndarray, vox: np.ndarray, shape: np.ndarray, band: float,
                lin: np.ndarray, face: np.ndarray, d2: np.ndarray) -> int:
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


def band_sparse(V_mm: np.ndarray, F: np.ndarray, vox: np.ndarray, shape: tuple[int, ...],
                band_vox: float = SB.BAND_VOX) -> dict[str, np.ndarray]:
    """sigma_band.signed_distance_band with no array of the grid's size: the
    band voxels (ijk, in the order of the grid), s, the nearest face, its
    weights w and u."""
    vox = np.asarray(vox, float)
    h = float(vox.mean())
    shp = np.asarray(shape, np.int64)
    Vc = np.ascontiguousarray(V_mm, np.float64)
    Fc = np.ascontiguousarray(F, np.int64)
    e = np.zeros(0, np.int64)
    n = _band_pairs(Vc, Fc, vox, shp, band_vox * h, e, e, np.zeros(0))
    lin, fc, d2 = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n)
    _band_pairs(Vc, Fc, vox, shp, band_vox * h, lin, fc, d2)
    # the nearest face of each voxel; of equal ones the first (the lowest face, as the dense kernel)
    o = np.lexsort((fc, d2, lin))
    lin, fc = lin[o], fc[o]
    first = np.r_[True, lin[1:] != lin[:-1]] if n else np.zeros(0, bool)
    lin, fc = lin[first], fc[first]
    ijk = np.stack(np.unravel_index(lin, tuple(int(x) for x in shp)), 1).astype(np.int64)
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
    return {"ijk": ijk, "s": s, "face": fc, "w": w, "u": u, "lin": lin}


class StopField:
    """E = E_stop on a sparse band (see the module docstring), as a
    stop_solve evaluator: site.labels (any integer array of the grid) and
    site.vox (mm per axis)."""

    def __init__(self, site: Any, progress: Any = None):
        self.labels = site.labels
        self.vox = np.asarray(site.vox, float)
        self.h = float(self.vox.mean())
        self.shape = tuple(int(x) for x in site.labels.shape)
        self.inside = np.zeros(self.shape, bool)  # the cortex and outside-medium voxels behind Σ, off the band too
        self.e_inside = 0.0  # Σ c over inside
        self.progress = progress or (lambda m: None)
        self.setup: dict[str, Any] = {"kind": "stop energy on a sparse band"}

    def weights(self, ijk: np.ndarray) -> np.ndarray:
        """c at voxels: -LOG_ODDS in cortex, +LOG_ODDS in outside medium."""
        lab = self.labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        return np.where(lab == CS.CORTEX, -SV.LOG_ODDS, np.where(lab == CS.OUTSIDE, SV.LOG_ODDS, 0.0))

    def __call__(self, V: np.ndarray, F: np.ndarray) -> SV.State:
        t0 = time.time()
        h = self.h
        band = band_sparse(V, F, self.vox, self.shape)
        ijk = band["ijk"]
        X = (ijk[:, 0], ijk[:, 1], ijk[:, 2])
        cv = self.weights(ijk)
        neg = (band["s"] < 0.0) & (cv != 0)
        # the band writes its sides; E's inside part follows
        self.e_inside += float((cv * (neg.astype(float) - self.inside[X])).sum())
        self.inside[X] = neg
        fr = 0.5 - band["s"] / h
        f = np.where(cv != 0, np.clip(fr, 0.0, 1.0), 0.0)
        e = self.e_inside + float((cv * (f - neg)).sum())
        ramp = (fr > 0.0) & (fr < 1.0) & (cv != 0)
        grad = np.zeros((len(V), 3))
        coef = (cv[ramp] / h)[:, None] * band["u"][ramp]
        fr_ = F[band["face"][ramp]]
        w = band["w"][ramp]
        for k in range(3):
            np.add.at(grad, fr_[:, k], w[:, k:k + 1] * coef)
        info = {"ramp_voxels": int(ramp.sum()), "band_voxels": len(ijk), "s": time.time() - t0}
        return SV.State(V.copy(), e, grad, info)
