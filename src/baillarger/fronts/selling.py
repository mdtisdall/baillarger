"""Selling's decomposition of 3 x 3 symmetric positive definite matrices.

Every 3 x 3 symmetric positive definite matrix D is a sum of six terms
rho_i e_i e_i^T with rho_i >= 0 and integer vectors e_i (Selling 1874;
Conway and Sloane 1992). They come from an obtuse superbase: four integer
vectors b_0..b_3 that sum to zero, span the lattice, and satisfy
b_i^T D b_j <= 0 for i != j. For each pair (i, j), with k and l the other two
indices, the offset is e = b_k x b_l and its weight is -b_i^T D b_j.

Selling's algorithm finds an obtuse superbase from the canonical one
(e_1, e_2, e_3, -(e_1 + e_2 + e_3)): while some pair has b_i^T D b_j > 0, it
replaces (b_i, b_k, b_l) by (-b_i, b_k + b_i, b_l + b_i) (in the code, l is
q). Each step lowers sum_i b_i^T D b_i, so it ends. The offsets stay short
when D is moderately anisotropic, which keeps the fast-marching stencils of
baillarger.fronts local; the decomposition is exact, so no direction is
favoured by the grid.
"""

import numpy as np
import numpy.typing as npt

from baillarger.fronts._jit import njit

type Float = npt.NDArray[np.float64]
type Int8 = npt.NDArray[np.int8]

# for each of the six pairs (i, j): the other two indices (k, l)
_PAIRS = np.array(
    [
        [0, 1, 2, 3],
        [0, 2, 1, 3],
        [0, 3, 1, 2],
        [1, 2, 0, 3],
        [1, 3, 0, 2],
        [2, 3, 0, 1],
    ],
    dtype=np.int64,
)
_MAX_STEPS = 500


@njit
def _selling_flat(d: Float, offsets: npt.NDArray[np.int64], weights: Float) -> int:
    """Decompose each d[n] (3 x 3) into offsets[n] (6 x 3) and weights[n] (6).

    Returns the number of matrices whose algorithm did not end in _MAX_STEPS
    (their weights are left at zero)."""
    failed = 0
    b = np.zeros((4, 3), dtype=np.int64)
    for n in range(d.shape[0]):
        m = d[n]
        b[:] = 0
        b[0, 0] = 1
        b[1, 1] = 1
        b[2, 2] = 1
        b[3, 0] = -1
        b[3, 1] = -1
        b[3, 2] = -1
        tol = 1e-12 * (m[0, 0] + m[1, 1] + m[2, 2])
        done = False
        for _ in range(_MAX_STEPS):
            moved = False
            for p in range(6):
                i, j, k, q = _PAIRS[p, 0], _PAIRS[p, 1], _PAIRS[p, 2], _PAIRS[p, 3]
                s = 0.0
                for a in range(3):
                    for c in range(3):
                        s += b[i, a] * m[a, c] * b[j, c]
                if s > tol:
                    for a in range(3):
                        b[k, a] += b[i, a]
                        b[q, a] += b[i, a]
                        b[i, a] = -b[i, a]
                    moved = True
                    break
            if not moved:
                done = True
                break
        if not done:
            failed += 1
            continue
        for p in range(6):
            i, j, k, q = _PAIRS[p, 0], _PAIRS[p, 1], _PAIRS[p, 2], _PAIRS[p, 3]
            s = 0.0
            for a in range(3):
                for c in range(3):
                    s += b[i, a] * m[a, c] * b[j, c]
            offsets[n, p, 0] = b[k, 1] * b[q, 2] - b[k, 2] * b[q, 1]
            offsets[n, p, 1] = b[k, 2] * b[q, 0] - b[k, 0] * b[q, 2]
            offsets[n, p, 2] = b[k, 0] * b[q, 1] - b[k, 1] * b[q, 0]
            weights[n, p] = max(-s, 0.0)
    return failed


def selling(d: npt.ArrayLike) -> tuple[Int8, Float]:
    """Selling's decomposition of each symmetric positive definite d[..., 3, 3].

    Returns offsets (..., 6, 3), integer vectors, and weights (..., 6) >= 0,
    with d = sum_i weights[i] * outer(offsets[i], offsets[i]). Raises
    ValueError if a matrix is not finite and symmetric positive definite, or if
    an offset does not fit in an int8 (a matrix far too anisotropic for local
    stencils).
    """
    a = np.asarray(d, np.float64)
    if a.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3) matrices, got shape {a.shape}")
    flat = np.ascontiguousarray(a.reshape(-1, 3, 3))
    if not np.all(np.isfinite(flat)):
        raise ValueError("matrices are not finite")
    scale = np.abs(flat).max(axis=(1, 2), initial=0.0)[:, None, None]
    if np.any(np.abs(flat - flat.transpose(0, 2, 1)) > 1e-10 * scale):
        raise ValueError("matrices are not symmetric")
    flat = 0.5 * (flat + flat.transpose(0, 2, 1))
    if flat.shape[0] and np.linalg.eigvalsh(flat).min() <= 0:
        raise ValueError("matrices are not positive definite")
    off = np.zeros((flat.shape[0], 6, 3), np.int64)
    w = np.zeros((flat.shape[0], 6), np.float64)
    failed = _selling_flat(flat, off, w)
    if failed:
        raise ValueError(f"Selling's algorithm did not end for {failed} matrices")
    if np.abs(off).max(initial=0) > 127:
        raise ValueError(
            "an offset does not fit in an int8: the metric is too anisotropic"
        )
    return off.astype(np.int8).reshape(*a.shape[:-2], 6, 3), w.reshape(*a.shape[:-2], 6)
