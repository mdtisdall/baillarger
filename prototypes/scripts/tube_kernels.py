"""numba kernels of the tube solver on a flat, padded voxel grid.

The grid is padded by PAD voxels on every side (never band), so every
neighbour index of a band voxel is inside the array. Flat C-order indices;
strides (sx, sy, 1).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from baillarger.topology import is_simple

PAD = 4  # >= the L1-4 neighbour offsets of the colouring


@njit(cache=True)
def same_bank(i, j, ptr, idx):
    """True when patch j is in i's same-bank list (sorted) or j == i."""
    if i == j:
        return True
    lo, hi = ptr[i], ptr[i + 1]
    while lo < hi:
        mid = (lo + hi) // 2
        v = idx[mid]
        if v == j:
            return True
        if v < j:
            lo = mid + 1
        else:
            hi = mid
    return False


@njit(cache=True)
def offsets_l1(r, strides):
    out = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if abs(dx) + abs(dy) + abs(dz) <= r:
                    out.append(dx * strides[0] + dy * strides[1] + dz * strides[2])
    return np.array(out, np.int64)


@njit(cache=True)
def build_problem(
    members, tvptr, tvidx, tau, owner, inside, depth, sptr, sidx, strides,
    off2, localid, roots, rootw,
):
    """Local problem for the union of the member tubes' windows.

    U: member voxels with tau >= -depth. E: band voxels within L1 distance 2
    of U (inner voxels too: they carry the mm pull only). S: band voxels within L1 distance 2 of
    E, plus the members' root corners. Returns slots (global), unknown slot
    ids, eval flags, neighbour slots (S, 3, 2), validity, valid, root slots
    and weights (T, 8), hasroot (T,). localid is a scratch array of -1 and is
    reset before returning.
    """
    slots = []
    unk = []
    # U
    for t in members:
        for k in range(tvptr[t], tvptr[t + 1]):
            v = tvidx[k]
            if tau[v] >= -depth:
                if localid[v] < 0:
                    localid[v] = len(slots)
                    slots.append(v)
                    unk.append(localid[v])
    nU = len(slots)
    # E
    isev = []
    for i in range(nU):
        isev.append(False)
    for i in range(nU):
        u = slots[i]
        for o in off2:
            w = u + o
            if owner[w] >= 0:
                if localid[w] < 0:
                    localid[w] = len(slots)
                    slots.append(w)
                    isev.append(True)
                else:
                    isev[localid[w]] = True
    nE = len(slots)
    # S: ring around every eval slot
    for i in range(nE):
        if not isev[i]:
            continue
        e = slots[i]
        for o in off2:
            w = e + o
            if owner[w] >= 0 and localid[w] < 0:
                localid[w] = len(slots)
                slots.append(w)
                isev.append(False)
    # root corners
    T = len(members)
    rslot = np.zeros((T, 8), np.int64)
    rw = np.zeros((T, 8))
    hasroot = np.zeros(T)
    for a in range(T):
        t = members[a]
        tot = 0.0
        for c in range(8):
            v = roots[t, c]
            if owner[v] >= 0:
                tot += rootw[t, c]
        if tot <= 0.0:
            continue
        hasroot[a] = 1.0
        for c in range(8):
            v = roots[t, c]
            if owner[v] >= 0:
                if localid[v] < 0:
                    localid[v] = len(slots)
                    slots.append(v)
                    isev.append(False)
                rslot[a, c] = localid[v]
                rw[a, c] = rootw[t, c] / tot
    n = len(slots)
    sl = np.empty(n, np.int64)
    ev = np.zeros(n, np.uint8)
    for i in range(n):
        sl[i] = slots[i]
        ev[i] = 1 if isev[i] else 0
    nb = np.empty((n, 3, 2), np.int64)
    nbv = np.zeros((n, 3, 2), np.bool_)
    valid = np.zeros(n, np.uint8)
    for i in range(n):
        v = sl[i]
        oi = owner[v]
        allax = True
        for a in range(3):
            anyax = False
            for s in range(2):
                w = v + (strides[a] if s == 1 else -strides[a])
                nb[i, a, s] = i
                if owner[w] >= 0 and localid[w] >= 0 and same_bank(oi, owner[w], sptr, sidx):
                    nb[i, a, s] = localid[w]
                    nbv[i, a, s] = True
                    anyax = True
            allax = allax and anyax
        valid[i] = 1 if allax else 0
    for i in range(n):
        localid[sl[i]] = -1
    un = np.empty(len(unk), np.int64)
    for i in range(len(unk)):
        un[i] = unk[i]
    return sl, un, ev, nb, nbv, valid, rslot, rw, hasroot


@njit(cache=True)
def apply_signs(obj3, shape_p, gidx, old, new, inside):
    """Apply the sign changes of tau at the voxels gidx (old -> new) to the
    object (inside or tau < 0), each checked with the (26, 6) simple-point
    test, retreats first, then advances by new tau ascending. On the first
    non-simple change, undo the applied ones and return False."""
    n = len(gidx)
    order = np.argsort(new)
    done = []
    sy = shape_p[1] * shape_p[2]
    sz = shape_p[2]
    for pass_ in range(2):
        for kk in range(n):
            k = order[n - 1 - kk] if pass_ == 0 else order[kk]
            v = gidx[k]
            if inside[v]:
                continue
            was = old[k] < 0
            now = new[k] < 0
            if was == now:
                continue
            if pass_ == 0 and now:
                continue
            if pass_ == 1 and not now:
                continue
            x, r = v // sy, v % sy
            y, z = r // sz, r % sz
            if not is_simple(obj3, x, y, z, 26):
                for d in done:
                    xx, rr = d // sy, d % sy
                    obj3[xx, rr // sz, rr % sz] = 1 - obj3[xx, rr // sz, rr % sz]
                return False
            obj3[x, y, z] = 1 if now else 0
            done.append(v)
    return True


@njit(cache=True)
def predict(obj3, shape_p, tvptr, tvidx, tau, inside, go, delta, block_eps):
    """Shift tau by -delta in the tubes with go[t]; sign changes checked
    (simple-point, ascending new tau); a blocked voxel is set to +block_eps.
    Returns the number of blocked voxels per tube."""
    ntube = len(go)
    blocked = np.zeros(ntube, np.int64)
    ch_v = []
    ch_t = []
    for t in range(ntube):
        if not go[t]:
            continue
        for k in range(tvptr[t], tvptr[t + 1]):
            v = tvidx[k]
            was = tau[v] < 0
            tau[v] -= delta
            if not inside[v] and (tau[v] < 0) != was:
                ch_v.append(v)
                ch_t.append(t)
    m = len(ch_v)
    vals = np.empty(m)
    for i in range(m):
        vals[i] = tau[ch_v[i]]
    order = np.argsort(vals)
    sy = shape_p[1] * shape_p[2]
    sz = shape_p[2]
    for kk in range(m):
        v = ch_v[order[kk]]
        x, r = v // sy, v % sy
        y, z = r // sz, r % sz
        if is_simple(obj3, x, y, z, 26):
            obj3[x, y, z] = 1
        else:
            tau[v] = block_eps
            blocked[ch_t[order[kk]]] += 1
    return blocked


@njit(cache=True)
def rebuild_shell(tau, owner, inside, allowed, band_list, off26, dist26, block_eps):
    """Drop shell voxels (tau >= 0, not inside) with no 26-neighbour tissue
    voxel (tau < 0, not inside) of the same tube; then every allowed unowned
    voxel next to a tissue voxel becomes shell of the tube with the lowest
    extrapolation tau_x + |y - x| (clamped to >= block_eps). Returns the
    number of clamped voxels."""
    for v in band_list:
        if owner[v] < 0 or inside[v] or tau[v] < 0:
            continue
        keep = False
        for k in range(len(off26)):
            w = v + off26[k]
            if owner[w] == owner[v] and tau[w] < 0 and not inside[w]:
                keep = True
                break
        if not keep:
            owner[v] = -1
            tau[v] = np.nan
    newv = []
    newt = []
    newo = []
    for v in band_list:
        if owner[v] < 0 or inside[v] or not (tau[v] < 0):
            continue
        for k in range(len(off26)):
            w = v + off26[k]
            if allowed[w] and owner[w] < 0:
                newv.append(w)
                newt.append(tau[v] + dist26[k])
                newo.append(owner[v])
    clamped = 0
    # lowest extrapolation wins
    m = len(newv)
    best = {}
    for i in range(m):
        w = newv[i]
        if w in best:
            j = best[w]
            if newt[i] < newt[j]:
                best[w] = i
        else:
            best[w] = i
    for w, i in best.items():
        t = newt[i]
        if t < block_eps:
            t = block_eps
            clamped += 1
        tau[w] = t
        owner[w] = newo[i]
    return clamped


@njit(cache=True)
def tube_pairs(tvptr, tvidx, owner, offs, ntube):
    """Pairs (i, j), i != j, with voxels within the offsets (both orders)."""
    stamp = -np.ones(ntube, np.int64)
    out_i = []
    out_j = []
    for t in range(ntube):
        for k in range(tvptr[t], tvptr[t + 1]):
            v = tvidx[k]
            for o in offs:
                j = owner[v + o]
                if j >= 0 and j != t and stamp[j] != t:
                    stamp[j] = t
                    out_i.append(t)
                    out_j.append(j)
    a = np.empty((len(out_i), 2), np.int64)
    for i in range(len(out_i)):
        a[i, 0] = out_i[i]
        a[i, 1] = out_j[i]
    return a


@njit(cache=True)
def greedy_colour(ptr, idx, order):
    n = len(ptr) - 1
    col = -np.ones(n, np.int64)
    used = -np.ones(n + 1, np.int64)
    for t in order:
        for k in range(ptr[t], ptr[t + 1]):
            c = col[idx[k]]
            if c >= 0:
                used[c] = t
        c = 0
        while used[c] == t:
            c += 1
        col[t] = c
    return col


@njit(cache=True)
def contacts(band_list, tau, owner, inside, strides, sptr, sidx):
    """Pairs (i, j) where a tissue voxel of i (tau < 0, not inside) is
    6-adjacent to a band voxel of j and j is not in i's same-bank list.
    Returns (i, j, voxel) rows (one per tissue voxel and direction)."""
    out = []
    for v in band_list:
        i = owner[v]
        if i < 0 or inside[v] or not (tau[v] < 0):
            continue
        for a in range(3):
            for s in (-1, 1):
                w = v + s * strides[a]
                j = owner[w]
                if j >= 0 and j != i and not same_bank(i, j, sptr, sidx):
                    out.append((i, j, v))
    a = np.empty((len(out), 3), np.int64)
    for k in range(len(out)):
        a[k, 0], a[k, 1], a[k, 2] = out[k]
    return a


@njit(cache=True)
def voxel_gradient(band_list, tau, owner, strides, h, sptr, sidx):
    """Contact-aware stencil gradient (as in tube_energy) at band voxels."""
    g = np.zeros((len(band_list), 3))
    for k in range(len(band_list)):
        v = band_list[k]
        i = owner[v]
        for a in range(3):
            wm, wp = v - strides[a], v + strides[a]
            okm = owner[wm] >= 0 and same_bank(i, owner[wm], sptr, sidx)
            okp = owner[wp] >= 0 and same_bank(i, owner[wp], sptr, sidx)
            if okm and okp:
                g[k, a] = (tau[wp] - tau[wm]) / (2 * h[a])
            elif okp:
                g[k, a] = (tau[wp] - tau[v]) / h[a]
            elif okm:
                g[k, a] = (tau[v] - tau[wm]) / h[a]
    return g


@njit(cache=True)
def trace_down(start, shape_p, gidx_of, G, owner, inside, sptr, sidx, step, nmax):
    """From each start voxel, follow -ĝ (nearest-voxel stencil gradient) by
    `step` voxels until a voxel inside the white surface. Returns the end
    position (voxel coordinates, padded grid) and a status: 0 reached,
    1 left the band, 2 zero gradient, 3 crossed to another bank, 4 too long."""
    n = len(start)
    endp = np.zeros((n, 3))
    status = np.zeros(n, np.int64)
    sy = shape_p[1] * shape_p[2]
    sz = shape_p[2]
    for k in range(n):
        v = start[k]
        t0 = owner[v]
        x, r = v // sy, v % sy
        p = np.array([float(x), float(r // sz), float(r % sz)])
        st = 4
        for it in range(nmax):
            q0 = int(np.round(p[0]))
            q1 = int(np.round(p[1]))
            q2 = int(np.round(p[2]))
            q = q0 * sy + q1 * sz + q2
            if inside[q]:
                st = 0
                break
            if owner[q] < 0:
                st = 1
                break
            if not same_bank(t0, owner[q], sptr, sidx):
                st = 3
                break
            gi = gidx_of[q]
            gx, gy, gz = G[gi, 0], G[gi, 1], G[gi, 2]
            nn = np.sqrt(gx * gx + gy * gy + gz * gz)
            if nn < 1e-12:
                st = 2
                break
            p[0] -= step * gx / nn
            p[1] -= step * gy / nn
            p[2] -= step * gz / nn
        endp[k] = p
        status[k] = st
    return endp, status
