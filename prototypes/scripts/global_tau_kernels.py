"""Numba kernels of the global sparse τ solve.

The grid is padded by PAD voxels on each side (flat index on the padded
grid); slot_of maps a padded flat index to its slot (domain voxel) or -1.
Every kernel is linear in the number of slots (the march and the τ sort:
N log N).

- contact_faces: 6-neighbour slot pairs whose owners are not in the same
  bank (same_ptr / same_idx: the patches within SAME_MM on the patch graph).
- upwind_owner: voxels in increasing τ take the owner of the 26-neighbour
  with lower τ that is best aligned with -∇τ (G8).
- column_lines: chains along ∇τ (G6).
- the Jacobian of the voxel residuals in stencil storage Jst (S, NOFF, NRES):
  Jst[s, o, k] = ∂r[s, k] / ∂τ[slot at s + offset o]; matrix products,
  the diagonal of JᵀJ, and dense JᵀJ blocks on the column lines.
"""

from __future__ import annotations

import numba as nb
import numpy as np

PAD = 2
# offsets of the residual stencil: L1 distance <= 2 (25 offsets)
OFFS = np.array(
    [
        (a, b, c)
        for a in range(-2, 3)
        for b in range(-2, 3)
        for c in range(-2, 3)
        if abs(a) + abs(b) + abs(c) <= 2
    ],
    np.int64,
)
NOFF = len(OFFS)
O26 = np.array(
    [
        (a, b, c)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        if (a, b, c) != (0, 0, 0)
    ],
    np.int64,
)


def find_colouring(maxdist: int = 4) -> tuple[int, int, int]:
    """Smallest m and (a, b) such that colour = (x + a y + b z) mod m differs
    for any two distinct points within L1 distance maxdist (so two unknowns of
    one colour never share a residual whose stencil is L1 <= maxdist / 2)."""
    ds = np.array(
        [
            (i, j, k)
            for i in range(-maxdist, maxdist + 1)
            for j in range(-maxdist, maxdist + 1)
            for k in range(-maxdist, maxdist + 1)
            if 0 < abs(i) + abs(j) + abs(k) <= maxdist
        ]
    )
    for m in range(1, 200):
        for a in range(m):
            for b in range(m):
                if np.all((ds[:, 0] + a * ds[:, 1] + b * ds[:, 2]) % m != 0):
                    return m, a, b
    raise RuntimeError("no colouring found")


@nb.njit(cache=True)
def same_bank(a: int, b: int, ptr: np.ndarray, idx: np.ndarray) -> bool:
    if a < 0 or b < 0 or a == b:
        return True
    lo, hi = ptr[a], ptr[a + 1]
    while lo < hi:
        mid = (lo + hi) // 2
        v = idx[mid]
        if v == b:
            return True
        if v < b:
            lo = mid + 1
        else:
            hi = mid
    return False


@nb.njit(cache=True)
def contact_faces(
    slot_flat: np.ndarray,
    slot_of: np.ndarray,
    strides: np.ndarray,
    owner: np.ndarray,
    ptr: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    """Contact faces (n, 3): minus slot, plus slot, axis."""
    out = []
    for s in range(len(slot_flat)):
        f = slot_flat[s]
        for a in range(3):
            u = slot_of[f + strides[a]]
            if u >= 0 and not same_bank(owner[s], owner[u], ptr, idx):
                out.append((s, u, a))
    res = np.zeros((len(out), 3), np.int64)
    for i in range(len(out)):
        res[i, 0], res[i, 1], res[i, 2] = out[i]
    return res


@nb.njit(cache=True)
def neighbours(
    slot_flat: np.ndarray, slot_of: np.ndarray, strides: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """nb (S1, 3, 2) slot of each 6-neighbour (dummy S if none), nbv (a real
    neighbour, not across a contact face), ghost (a contact face)."""
    S = len(slot_flat)
    nbr = np.full((S + 1, 3, 2), S, np.int64)
    nbv = np.zeros((S + 1, 3, 2), np.bool_)
    gh = np.zeros((S + 1, 3, 2), np.bool_)
    for s in range(S):
        f = slot_flat[s]
        for a in range(3):
            for k in range(2):
                u = slot_of[f + (2 * k - 1) * strides[a]]
                if u >= 0:
                    nbr[s, a, k] = u
                    nbv[s, a, k] = True
    for i in range(len(faces)):
        s, u, a = faces[i, 0], faces[i, 1], faces[i, 2]
        nbv[s, a, 1] = False
        gh[s, a, 1] = True
        nbv[u, a, 0] = False
        gh[u, a, 0] = True
    return nbr, nbv, gh


@nb.njit(cache=True)
def upwind_owner(
    order: np.ndarray,
    slot_flat: np.ndarray,
    slot_of: np.ndarray,
    strides: np.ndarray,
    o26: np.ndarray,
    h: np.ndarray,
    tau: np.ndarray,
    ghat: np.ndarray,
    owner0: np.ndarray,
    fixed: np.ndarray,
    ptr: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    """Owners by one pass in increasing τ (order): each free slot takes the
    owner of the 26-neighbour with lower τ best aligned with -ĝ. A neighbour
    whose owner before the pass is not in the same bank as this slot's owner
    before the pass is across a contact and is not used. Fixed slots and
    slots without a candidate keep their owner."""
    own = owner0.copy()
    no = len(o26)
    for i in range(len(order)):
        s = order[i]
        if fixed[s]:
            continue
        f = slot_flat[s]
        best, bo = -2.0, -1
        for j in range(no):
            u = slot_of[
                f
                + o26[j, 0] * strides[0]
                + o26[j, 1] * strides[1]
                + o26[j, 2] * strides[2]
            ]
            if u < 0 or tau[u] >= tau[s]:
                continue
            if not same_bank(owner0[s], owner0[u], ptr, idx):
                continue
            dx = o26[j, 0] * h[0]
            dy = o26[j, 1] * h[1]
            dz = o26[j, 2] * h[2]
            L = np.sqrt(dx * dx + dy * dy + dz * dz)
            al = -(dx * ghat[s, 0] + dy * ghat[s, 1] + dz * ghat[s, 2]) / L
            if al > best:
                best, bo = al, own[u]
        if bo >= 0:
            own[s] = bo
    return own


@nb.njit(cache=True)
def column_lines(
    nbr: np.ndarray, nbv: np.ndarray, ghat: np.ndarray, lmax: int
) -> tuple[np.ndarray, np.ndarray]:
    """Chains along ∇τ: each slot links to the real 6-neighbour best aligned
    with ĝ (alignment > 0); each slot keeps only its best-aligned incoming
    link. Chains longer than lmax are split. Returns (ptr, slots) in CSR
    form; every slot is in exactly one chain (a slot without links alone)."""
    S = nbr.shape[0] - 1
    nxt = -np.ones(S, np.int64)
    nal = np.zeros(S)
    for s in range(S):
        best, bu = 0.0, -1
        for a in range(3):
            for k in range(2):
                if nbv[s, a, k]:
                    al = (2 * k - 1) * ghat[s, a]
                    if al > best:
                        best, bu = al, nbr[s, a, k]
        nxt[s] = bu
        nal[s] = best
    inc = -np.ones(S, np.int64)
    inal = np.zeros(S)
    for s in range(S):
        u = nxt[s]
        if u >= 0 and nal[s] > inal[u]:
            inal[u] = nal[s]
            inc[u] = s
    for s in range(S):  # drop links that lost
        u = nxt[s]
        if u >= 0 and inc[u] != s:
            nxt[s] = -1
    ptr = [0]
    out = np.empty(S, np.int64)
    n = 0
    seen = np.zeros(S, np.bool_)
    for s in range(S):
        if inc[s] >= 0 and nxt[inc[s]] == s:
            continue  # not a head
        cur = s
        ln = 0
        while cur >= 0 and not seen[cur]:
            if ln == lmax:
                ptr.append(n)
                ln = 0
            seen[cur] = True
            out[n] = cur
            n += 1
            ln += 1
            cur = nxt[cur]
        ptr.append(n)
    for s in range(S):  # cycles (none expected: ∇τ ascends) as singletons
        if not seen[s]:
            out[n] = s
            n += 1
            ptr.append(n)
    return np.array(ptr, np.int64), out


@nb.njit(cache=True)
def stencil_cols(
    slot_flat: np.ndarray, slot_of: np.ndarray, strides: np.ndarray, offs: np.ndarray
) -> np.ndarray:
    """col (S, NOFF): the slot at s + offset (-1 if none)."""
    S = len(slot_flat)
    col = -np.ones((S, len(offs)), np.int64)
    for s in range(S):
        f = slot_flat[s]
        for o in range(len(offs)):
            col[s, o] = slot_of[
                f
                + offs[o, 0] * strides[0]
                + offs[o, 1] * strides[1]
                + offs[o, 2] * strides[2]
            ]
    return col


@nb.njit(cache=True)
def scatter_colour(
    Jst: np.ndarray,
    jv: np.ndarray,
    col: np.ndarray,
    colour: np.ndarray,
    c: int,
    unk: np.ndarray,
) -> None:
    """Jst[s, o, :] = jv[s, :] for the offsets whose slot has colour c."""
    S = Jst.shape[0]
    for s in range(S):
        for o in range(Jst.shape[1]):
            u = col[s, o]
            if u >= 0 and colour[u] == c and unk[u]:
                for k in range(Jst.shape[2]):
                    Jst[s, o, k] = jv[s, k]


@nb.njit(cache=True, parallel=True)
def jst_apply(Jst: np.ndarray, col: np.ndarray, v: np.ndarray) -> np.ndarray:
    """r = J v (S, NRES)."""
    S, no, nr = Jst.shape
    r = np.zeros((S, nr))
    for s in nb.prange(S):
        for o in range(no):
            u = col[s, o]
            if u >= 0:
                vu = v[u]
                if vu != 0.0:
                    for k in range(nr):
                        r[s, k] += Jst[s, o, k] * vu
    return r


@nb.njit(cache=True)
def jst_apply_t(Jst: np.ndarray, col: np.ndarray, r: np.ndarray, n: int) -> np.ndarray:
    """Jᵀ r (n,)."""
    S, no, nr = Jst.shape
    out = np.zeros(n)
    for s in range(S):
        for o in range(no):
            u = col[s, o]
            if u >= 0:
                acc = 0.0
                for k in range(nr):
                    acc += Jst[s, o, k] * r[s, k]
                out[u] += acc
    return out


@nb.njit(cache=True)
def jst_diag(Jst: np.ndarray, col: np.ndarray, n: int) -> np.ndarray:
    S, no, nr = Jst.shape
    d = np.zeros(n)
    for s in range(S):
        for o in range(no):
            u = col[s, o]
            if u >= 0:
                acc = 0.0
                for k in range(nr):
                    acc += Jst[s, o, k] ** 2
                d[u] += acc
    return d


@nb.njit(cache=True)
def _entry(
    i: int,
    j: int,
    Jst: np.ndarray,
    col: np.ndarray,
    slot_flat: np.ndarray,
    slot_of: np.ndarray,
    strides: np.ndarray,
    offs: np.ndarray,
    olut: np.ndarray,
    coords: np.ndarray,
) -> float:
    """(JᵀJ)_ij from the stencil storage: sum over the residual slots s whose
    stencil holds both i and j."""
    fi = slot_flat[i]
    acc = 0.0
    for oi in range(len(offs)):
        fs = fi - (
            offs[oi, 0] * strides[0]
            + offs[oi, 1] * strides[1]
            + offs[oi, 2] * strides[2]
        )
        s = slot_of[fs]
        if s < 0:
            continue
        dx = coords[j, 0] - coords[s, 0]
        dy = coords[j, 1] - coords[s, 1]
        dz = coords[j, 2] - coords[s, 2]
        if abs(dx) > 2 or abs(dy) > 2 or abs(dz) > 2:
            continue
        oj = olut[dx + 2, dy + 2, dz + 2]
        if oj < 0:
            continue
        for k in range(Jst.shape[2]):
            acc += Jst[s, oi, k] * Jst[s, oj, k]
    return acc


@nb.njit(cache=True)
def line_blocks(
    ptr: np.ndarray,
    slots: np.ndarray,
    Jst: np.ndarray,
    col: np.ndarray,
    slot_flat: np.ndarray,
    slot_of: np.ndarray,
    strides: np.ndarray,
    offs: np.ndarray,
    olut: np.ndarray,
    coords: np.ndarray,
    diag_add: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cholesky factors of the dense JᵀJ blocks of the lines (principal
    submatrices, so positive definite when diag_add > 0). Returns the flat
    factor storage and its offsets (bptr[c] for chain c, size L²)."""
    nc = len(ptr) - 1
    bptr = np.zeros(nc + 1, np.int64)
    for c in range(nc):
        L = ptr[c + 1] - ptr[c]
        bptr[c + 1] = bptr[c] + L * L
    B = np.zeros(bptr[nc])
    for c in range(nc):
        p0 = ptr[c]
        L = ptr[c + 1] - p0
        M = np.zeros((L, L))
        for p in range(L):
            i = slots[p0 + p]
            for q in range(p, L):
                j = slots[p0 + q]
                l1 = (
                    abs(coords[i, 0] - coords[j, 0])
                    + abs(coords[i, 1] - coords[j, 1])
                    + abs(coords[i, 2] - coords[j, 2])
                )
                if l1 > 4:
                    continue
                e = _entry(
                    i, j, Jst, col, slot_flat, slot_of, strides, offs, olut, coords
                )
                M[p, q] = e
                M[q, p] = e
            M[p, p] += diag_add[i]
        # Cholesky
        for p in range(L):
            acc = M[p, p]
            for k in range(p):
                acc -= M[p, k] ** 2
            acc = np.sqrt(max(acc, 1e-300))
            M[p, p] = acc
            for q in range(p + 1, L):
                a2 = M[q, p]
                for k in range(p):
                    a2 -= M[q, k] * M[p, k]
                M[q, p] = a2 / acc
        for p in range(L):
            for q in range(L):
                B[bptr[c] + p * L + q] = M[p, q] if q <= p else 0.0
    return B, bptr


@nb.njit(cache=True, parallel=True)
def line_solve(
    ptr: np.ndarray, slots: np.ndarray, B: np.ndarray, bptr: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """z = M⁻¹ r with M the block diagonal of the lines' Cholesky factors."""
    z = np.zeros_like(r)
    nc = len(ptr) - 1
    for c in nb.prange(nc):
        p0 = ptr[c]
        L = ptr[c + 1] - p0
        b0 = bptr[c]
        y = np.zeros(L)
        for p in range(L):
            acc = r[slots[p0 + p]]
            for k in range(p):
                acc -= B[b0 + p * L + k] * y[k]
            y[p] = acc / B[b0 + p * L + p]
        for p in range(L - 1, -1, -1):
            acc = y[p]
            for k in range(p + 1, L):
                acc -= B[b0 + k * L + p] * y[k]
            y[p] = acc / B[b0 + p * L + p]
        for p in range(L):
            z[slots[p0 + p]] = y[p]
    return z
