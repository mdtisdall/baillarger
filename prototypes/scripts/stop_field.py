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
widened bounding box, and each voxel within the band keeps its nearest
face), so the cost follows Σ's area, not the grid. The grid is taken slab by
slab along its first axis: a slab's faces write into one plane of distances,
which is read in the order of the grid and cleared, so nothing is sorted and
each (voxel, face) pair is computed once. Runs of slabs go to threads (the
kernel holds no lock); each thread has its own plane. The result is the same
for any number of threads, and the same, bit for bit, as collecting every
pair and sorting them (test_stop_field.py keeps that form as the reference).

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

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import crop_site as CS
import numba as nb
import numpy as np
import sigma_band as SB
import stop_solve as SV

#: runs of slabs per worker thread (band_sparse)
PIECES = 4


@nb.njit(cache=True)
def _slab_faces(V: np.ndarray, F: np.ndarray, vox: np.ndarray, shape: np.ndarray, band: float) -> tuple[np.ndarray, np.ndarray]:
    """For each slab of the grid (one index i of the first axis): the faces
    whose bounding box, widened by band (mm), reaches it, in the order of
    the faces. Returns (start, faces): slab i has faces[start[i]:start[i + 1]]."""
    n0 = shape[0]
    lo = np.empty(len(F), np.int64)
    hi = np.empty(len(F), np.int64)
    start = np.zeros(n0 + 1, np.int64)
    for f in range(len(F)):
        a, b, c = V[F[f, 0], 0], V[F[f, 1], 0], V[F[f, 2], 0]
        mn = min(a, b, c) - band
        mx = max(a, b, c) + band
        lo[f] = max(0, int(np.ceil(mn / vox[0])))
        hi[f] = min(n0 - 1, int(np.floor(mx / vox[0])))
        for i in range(lo[f], hi[f] + 1):
            start[i + 1] += 1
    for i in range(n0):
        start[i + 1] += start[i]
    faces = np.empty(start[n0], np.int64)
    at = start[:n0].copy()
    for f in range(len(F)):
        for i in range(lo[f], hi[f] + 1):
            faces[at[i]] = f
            at[i] += 1
    return start, faces


@nb.njit(cache=True, nogil=True)
def _band_slabs(V: np.ndarray, F: np.ndarray, vox: np.ndarray, shape: np.ndarray, band: float,
                i0: int, i1: int, start: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The band voxels of the slabs i0 <= i < i1, in the order of the grid:
    ijk (n, 3), the nearest face of each (of equal ones the lowest) and the
    weights of the closest point on it (n, 3).

    Slab by slab: each face of the slab (_slab_faces) visits the voxels of
    its widened bounding box in that slab and writes its squared distance
    where it is within band and below the one there (D, N: one value per
    voxel of a slab); then the rows that a box touched are read in order
    and cleared. Holds no lock: several run at once on separate slabs."""
    n1, n2 = shape[1], shape[2]
    b2 = band * band
    D = np.full((n1, n2), np.inf)  # the squared distance to the nearest face so far, inf: not in the band
    N = np.empty((n1, n2), np.int64)  # that face
    klo = np.full(n1, n2, np.int64)  # per row j: the range of k that a box touched in this slab
    khi = np.full(n1, -1, np.int64)
    cap = 1 << 16
    ijk = np.empty((cap, 3), np.int64)
    fc = np.empty(cap, np.int64)
    w = np.empty((cap, 3))
    n = 0
    p = np.empty(3)
    for i in range(i0, i1):
        p[0] = i * vox[0]
        jlo, jhi = n1, -1
        for t in range(start[i], start[i + 1]):
            f = faces[t]
            a, b, c = V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]
            j0 = max(0, int(np.ceil((min(a[1], b[1], c[1]) - band) / vox[1])))
            j1 = min(n1 - 1, int(np.floor((max(a[1], b[1], c[1]) + band) / vox[1])))
            k0 = max(0, int(np.ceil((min(a[2], b[2], c[2]) - band) / vox[2])))
            k1 = min(n2 - 1, int(np.floor((max(a[2], b[2], c[2]) + band) / vox[2])))
            if j0 > j1 or k0 > k1:
                continue
            jlo, jhi = min(jlo, j0), max(jhi, j1)
            for j in range(j0, j1 + 1):
                p[1] = j * vox[1]
                klo[j], khi[j] = min(klo[j], k0), max(khi[j], k1)
                for k in range(k0, k1 + 1):
                    p[2] = k * vox[2]
                    w0, w1, w2 = SB._closest_on_triangle(p, a, b, c)
                    q0 = w0 * a[0] + w1 * b[0] + w2 * c[0] - p[0]
                    q1 = w0 * a[1] + w1 * b[1] + w2 * c[1] - p[1]
                    q2 = w0 * a[2] + w1 * b[2] + w2 * c[2] - p[2]
                    dd = q0 * q0 + q1 * q1 + q2 * q2
                    # the faces come in their order: of equal distances the lowest face stays
                    if dd < b2 and dd < D[j, k]:
                        D[j, k] = dd
                        N[j, k] = f
        for j in range(jlo, jhi + 1):
            p[1] = j * vox[1]
            for k in range(klo[j], khi[j] + 1):
                if D[j, k] < np.inf:
                    D[j, k] = np.inf
                    if n == cap:
                        cap *= 2
                        ijk2, fc2, w2_ = np.empty((cap, 3), np.int64), np.empty(cap, np.int64), np.empty((cap, 3))
                        ijk2[:n], fc2[:n], w2_[:n] = ijk, fc, w
                        ijk, fc, w = ijk2, fc2, w2_
                    f = N[j, k]
                    p[2] = k * vox[2]
                    w[n, 0], w[n, 1], w[n, 2] = SB._closest_on_triangle(p, V[F[f, 0]], V[F[f, 1]], V[F[f, 2]])
                    ijk[n, 0], ijk[n, 1], ijk[n, 2] = i, j, k
                    fc[n] = f
                    n += 1
            klo[j], khi[j] = n2, -1
    return ijk[:n], fc[:n], w[:n]


def band_sparse(V_mm: np.ndarray, F: np.ndarray, vox: np.ndarray, shape: tuple[int, ...],
                band_vox: float = SB.BAND_VOX, workers: int | None = None) -> dict[str, np.ndarray]:
    """sigma_band.signed_distance_band with no array of the grid's size: the
    band voxels (ijk, in the order of the grid), s, the nearest face, its
    weights w and u. On `workers` threads (default: the CPU count); the
    result does not depend on their number."""
    vox = np.asarray(vox, float)
    h = float(vox.mean())
    shp = np.asarray(shape, np.int64)
    Vc = np.ascontiguousarray(V_mm, np.float64)
    Fc = np.ascontiguousarray(F, np.int64)
    band = band_vox * h
    start, faces = _slab_faces(Vc, Fc, vox, shp, band)
    nw = max(1, (os.cpu_count() or 1) if workers is None else workers)
    # runs of slabs with about the same number of (slab, face) visits each, PIECES per worker: a thread that
    # ends a run takes the next, so a run with more voxels per face than another does not leave threads idle
    at = np.searchsorted(start, np.linspace(0, start[-1], PIECES * nw + 1)[1:-1]) if nw > 1 else []
    cuts = np.unique(np.r_[0, at, shp[0]]).tolist()

    def run(r: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _band_slabs(Vc, Fc, vox, shp, band, r[0], r[1], start, faces)

    runs = list(zip(cuts[:-1], cuts[1:]))
    if nw > 1:
        with ThreadPoolExecutor(max_workers=nw) as ex:
            parts = list(ex.map(run, runs))  # in the order of the runs: the order of the grid
    else:
        parts = [run(r) for r in runs]
    ijk, fc, w = (np.concatenate([p[k] for p in parts]) for k in range(3))
    lin = (ijk[:, 0] * shp[1] + ijk[:, 1]) * shp[2] + ijk[:, 2]
    x = ijk * vox
    A, B, C = Vc[Fc[fc, 0]], Vc[Fc[fc, 1]], Vc[Fc[fc, 2]]
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
    site.vox (mm per axis). workers: the threads of band_sparse."""

    def __init__(self, site: Any, progress: Any = None, workers: int | None = None):
        self.labels = site.labels
        self.vox = np.asarray(site.vox, float)
        self.h = float(self.vox.mean())
        self.shape = tuple(int(x) for x in site.labels.shape)
        self.inside = np.zeros(self.shape, bool)  # the cortex and outside-medium voxels behind Σ, off the band too
        self.e_inside = 0.0  # Σ c over inside
        self.progress = progress or (lambda m: None)
        self.workers = workers  # threads of band_sparse (None: the CPU count)
        self.setup: dict[str, Any] = {"kind": "stop energy on a sparse band"}

    def weights(self, ijk: np.ndarray) -> np.ndarray:
        """c at voxels: -LOG_ODDS in cortex, +LOG_ODDS in outside medium."""
        lab = self.labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        return np.where(lab == CS.CORTEX, -SV.LOG_ODDS, np.where(lab == CS.OUTSIDE, SV.LOG_ODDS, 0.0))

    def __call__(self, V: np.ndarray, F: np.ndarray) -> SV.State:
        t0 = time.time()
        h = self.h
        band = band_sparse(V, F, self.vox, self.shape, workers=self.workers)
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
