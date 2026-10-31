"""Topology of a voxel object on the Kuhn tetrahedralization, and the
automatic removal of its handles.

baillarger.mesh.marching_tetrahedra contours the piecewise-linear
interpolant on the Kuhn tetrahedralization of the grid (every cube split into
6 tetrahedra round its main diagonal). For a binary field contoured at 0.5,
the region inside the surface deformation-retracts onto the full subcomplex
of that tetrahedralization spanned by the object voxels, and the region
outside onto the full subcomplex spanned by the background voxels. So the
topology of the mesh is the topology of the object with the Kuhn adjacency:
two voxels are neighbours when their offset is one of the 14 vectors with
entries in {0, 1} or in {0, -1} (the edges of the tetrahedra). This adjacency
is the same for object and background, so there is no (26, 6) or (6, 26)
choice to make, and the 6- and 26-connected answers of baillarger.topology
can disagree with the mesh.

Betti numbers. kuhn_euler counts the simplices of the full subcomplex
(vertices - edges + triangles - tetrahedra). kuhn_betti gives (b0, b1, b2):
components, handles and cavities, with b1 = b0 + b2 - chi. When b1 = 0, every
closed component of the boundary surface is a sphere (for a compact region of
space, the boundary's total genus equals b1), so every piece of that surface
cut out by a box is planar: genus 0.

Simple points. A voxel is simple when adding it to the object (or removing
it) changes no Betti number of the object or of the background. Its link (the
14 neighbours, triangulated as a 2-sphere by the 24 triangles opposite the
voxel in its tetrahedra) decides: the voxel is simple exactly when the object
neighbours form one connected piece of the link and the background
neighbours form one connected piece. A table of the 2^14 cases gives the test.

remove_handles makes b1 = 0 and keeps b0 and b2. It treats each object
component on its own, with the other components as background (one
background front could not separate two components without changing its own
topology), and returns the union. For one component it grows two regions,
each only by simple points, so each keeps its topology:

1. The component's cavities (background components that do not reach the
   array border) are filled first and emptied again at the end, and every
   other background voxel belongs to one region, the background front, grown
   from the border. The object front is grown from the deepest voxel of the
   component. Object voxels are claimed by the object front, background
   voxels by the background front, deepest first (squared distance to the
   other phase, in voxel units: the voxels are taken as cubes). A front can
   never close a loop, so round each handle the object front meets itself at
   a wall across the handle's bridge, near its thinnest place, and the background
   front meets itself at a wall across its tunnel. The walls stay unclaimed.
2. The unclaimed voxels are split into walls (connected pieces of one phase).
   The smallest wall, in voxels, is claimed by the other phase's front: an
   object wall by the background (a cut), a background wall by the object
   front (a fill). Then both phases claim what they can of their own
   unclaimed voxels again: the partner wall of the handle just resolved now
   joins its own phase, so nothing else changes there. Repeat until no wall
   is left. Ties go to the cut, then to the wall with the lowest voxel index.

3. Where an object wall and a background wall touch, they can block each
   other: no single voxel of either is simple for either front (and in a
   simple pair, no single voxel of one wall is). When no wall can be claimed
   voxel by voxel, sets are claimed whole, smallest first: a wall, two
   touching walls of opposite phase, or a whole cluster of touching walls,
   by the front that changes fewer voxels, if the front's homology stays the
   same (_set_simple).
4. When nothing more can be claimed, the unclaimed background stays
   background and the unclaimed object voxels join the object front where
   they are simple for it. The rest are removed and reported as a 'stranded'
   move: each of them, put back alone, would make a handle. (In a ring of
   walls, keeping some of them could change fewer voxels; that search is not
   made.)

So each handle is resolved by whichever of cut or fill changes fewer voxels,
measured on the walls where the fronts met. The result is the object front
voxels with the cavities emptied again.
Every step of every front is a simple point, so the object front stays a
ball: the result has no handle by construction, whatever the order of the
steps. The union of the components keeps that unless a fill of one component
touches another (not checked here; baillarger.white.wm_topo checks it).

Variations, for testing their effect (the dseg command's options): fill_cost
weighs a fill's voxels against a cut's in the choice; after the removal,
thicken widens a cut's gap or thickens a fill's plug by simple points, and
trim_threads removes thin dangling protrusions whole. Each keeps the Betti
numbers.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from baillarger._jit import njit

type Bool = npt.NDArray[np.bool_]
type Int = npt.NDArray[np.int64]

_U, _OBJ, _BG = 0, 1, 2  # voxel states: unclaimed, object front, background front
_IDX_BITS = 30  # flat index bits in a heap key


def _kuhn_tets() -> npt.NDArray[np.int_]:
    """Corner offsets of the 6 tetrahedra of the unit cube, (6, 4, 3): (0,0,0),
    +e_a, +e_a+e_b, (1,1,1) for each order of the axes (the split of
    baillarger.mesh)."""
    tets = []
    for perm in itertools.permutations(range(3)):
        c = [np.zeros(3, int)]
        for ax in perm:
            nxt = c[-1].copy()
            nxt[ax] = 1
            c.append(nxt)
        tets.append(np.array(c))
    return np.array(tets)


_TETS = _kuhn_tets()


def _link() -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """The 14 neighbour offsets (14, 3) and the 24 triangles of the link of a
    vertex (24, 3), as indices into the offsets."""
    tris = []
    for tet in _TETS:
        for i in range(4):
            t = tet - tet[i]  # the tetrahedron moved so that corner i is at 0
            tris.append([tuple(p) for k, p in enumerate(t) if k != i])
    off = sorted({p for t in tris for p in t})
    pos = {p: k for k, p in enumerate(off)}
    return np.array(off), np.array([[pos[p] for p in t] for t in tris])


KUHN_OFFSETS, LINK_TRIANGLES = _link()


def kuhn_structure() -> Bool:
    """The 3x3x3 structuring element of the Kuhn adjacency (centre included),
    for scipy.ndimage.label."""
    s = np.zeros((3, 3, 3), bool)
    s[1, 1, 1] = True
    for o in KUHN_OFFSETS:
        s[tuple(o + 1)] = True
    return s


def _simple_table() -> Bool:
    edges = {
        tuple(sorted((int(t[a]), int(t[b]))))
        for t in LINK_TRIANGLES
        for a, b in ((0, 1), (1, 2), (2, 0))
    }
    adj = np.zeros(14, np.int64)
    for a, b in edges:
        adj[a] |= 1 << b
        adj[b] |= 1 << a

    def pieces(m: int) -> int:
        n, left = 0, m
        while left:
            n += 1
            stack = left & -left
            left &= ~stack
            while stack:
                b = (stack & -stack).bit_length() - 1
                stack &= ~(1 << b)
                nb = int(adj[b]) & left
                left &= ~nb
                stack |= nb
        return n

    full = (1 << 14) - 1
    return np.array([pieces(m) == 1 and pieces(full & ~m) == 1 for m in range(1 << 14)])


SIMPLE = _simple_table()
"""SIMPLE[m]: whether a voxel is simple when its object neighbours are the
bits of m (bit k: KUHN_OFFSETS[k])."""


def _simplex_types() -> list[list[npt.NDArray[np.int_]]]:
    """For each dimension 0..3, the simplices of the Kuhn tetrahedralization up
    to translation, each as its vertex offsets (all entries 0 or 1)."""
    out: list[set[tuple[tuple[int, ...], ...]]] = [set() for _ in range(4)]
    for tet in _TETS:
        for k in range(1, 5):
            for sub in itertools.combinations(range(4), k):
                p = sorted(tuple(int(c) for c in tet[i]) for i in sub)
                base = np.array(p[0])
                out[k - 1].add(
                    tuple(tuple(int(c) for c in np.array(q) - base) for q in p)
                )
    return [[np.array(s) for s in sorted(d)] for d in out]


_SIMPLICES = _simplex_types()


def kuhn_euler(obj: npt.ArrayLike) -> int:
    """Euler characteristic of the full subcomplex of the Kuhn
    tetrahedralization spanned by the object voxels (3D bool)."""
    a = np.pad(np.asarray(obj, bool), (0, 1))
    n = np.array(a.shape) - 1
    chi = 0
    for dim, types in enumerate(_SIMPLICES):
        for t in types:
            m = np.ones(tuple(n), bool)
            for o in t:
                m &= a[o[0] : o[0] + n[0], o[1] : o[1] + n[1], o[2] : o[2] + n[2]]
            chi += (-1) ** dim * int(m.sum())
    return chi


def kuhn_betti(obj: npt.ArrayLike) -> tuple[int, int, int]:
    """(b0, b1, b2) of the object (3D bool) with the Kuhn adjacency:
    components, handles, cavities. Outside the array is background."""
    a = np.pad(np.asarray(obj, bool), 1)
    s = kuhn_structure()
    b0 = int(ndi.label(a, s)[1])
    b2 = int(ndi.label(~a, s)[1]) - 1
    return b0, b0 + b2 - kuhn_euler(a), b2


# ---------------------------------------------------------------- growth


@njit
def _push(heap: Int, size: int, key: int) -> tuple[Int, int]:
    if size == heap.size:
        bigger = np.empty(2 * heap.size, np.int64)
        bigger[:size] = heap[:size]
        heap = bigger
    i = size
    heap[i] = key
    while i > 0:
        j = (i - 1) // 2
        if heap[j] <= heap[i]:
            break
        heap[i], heap[j] = heap[j], heap[i]
        i = j
    return heap, size + 1


@njit
def _pop(heap: Int, size: int) -> tuple[int, int]:
    top = heap[0]
    size -= 1
    heap[0] = heap[size]
    i = 0
    while True:
        a = 2 * i + 1
        b = a + 1
        s = i
        if a < size and heap[a] < heap[s]:
            s = a
        if b < size and heap[b] < heap[s]:
            s = b
        if s == i:
            break
        heap[i], heap[s] = heap[s], heap[i]
        i = s
    return top, size


@njit
def _link_mask(state: npt.NDArray[np.uint8], i: int, front: int, offs: Int) -> int:
    m = 0
    for k in range(14):
        if state[i + offs[k]] == front:
            m |= 1 << k
    return m


@njit
def _grow(
    state: npt.NDArray[np.uint8],
    phase: npt.NDArray[np.uint8],
    d2: npt.NDArray[np.int32],
    cand: Int,
    offs: Int,
    simple: Bool,
    dmax: int,
    queued: Bool,
) -> int:
    """Each unclaimed voxel joins the front of its own phase (phase[i]: _OBJ or
    _BG), deepest first, when it is simple for that front. cand: the voxels to
    start from (those next to their front are queued). queued: all False, the
    size of state; all False again on return (every voxel queued is popped).
    Returns the number of voxels claimed."""
    heap = np.empty(1024, np.int64)
    size = 0
    for i in cand:
        if state[i] != _U or queued[i]:
            continue
        for k in range(14):
            if state[i + offs[k]] == phase[i]:
                heap, size = _push(heap, size, ((dmax - d2[i]) << _IDX_BITS) | i)
                queued[i] = True
                break
    n = 0
    mask = (1 << _IDX_BITS) - 1
    while size > 0:
        key, size = _pop(heap, size)
        i = key & mask
        queued[i] = False
        if state[i] != _U:
            continue
        f = phase[i]
        if not simple[_link_mask(state, i, f, offs)]:
            continue  # tested again when a neighbour joins f
        state[i] = f
        n += 1
        for k in range(14):
            j = i + offs[k]
            if state[j] == _U and phase[j] == f and not queued[j]:
                heap, size = _push(heap, size, ((dmax - d2[j]) << _IDX_BITS) | j)
                queued[j] = True
    return n


@njit
def _claim(
    state: npt.NDArray[np.uint8], wall: Int, front: int, offs: Int, simple: Bool
) -> int:
    """Claim the unclaimed voxels of wall for front, by simple points, sweeping
    until nothing more joins. Returns the number claimed."""
    n = 0
    progress = True
    while progress:
        progress = False
        for i in wall:
            if state[i] != _U:
                continue
            m = _link_mask(state, i, front, offs)
            if m != 0 and simple[m]:
                state[i] = front
                n += 1
                progress = True
    return n


@dataclass(frozen=True)
class Move:
    """One change: kind 'cut' (object voxels removed: a wall claimed by the
    background), 'fill' (background voxels added: a wall claimed by an object
    front) or 'stranded' (object voxels no front could claim, removed), and its
    voxels (n, 3), in the coordinates of the input array."""

    kind: str
    voxels: npt.NDArray[np.intp]


@dataclass(frozen=True)
class HandleRemoval:
    """The result of remove_handles."""

    obj: Bool
    moves: list[Move]
    #: object voxels that no front could claim (removed; also the 'stranded' move)
    stranded: int


def _walls(
    u: Int,
    phase: npt.NDArray[np.uint8],
    offs: Int,
    mixed: bool = False,
    fill_cost: float = 1.0,
) -> list[Int]:
    """Connected pieces (Kuhn adjacency, one phase, or both phases together
    when mixed) of the voxels u (sorted flat indices: the unclaimed voxels),
    cheapest first: an object wall (a cut) costs its voxels, a background wall
    (a fill) fill_cost times its voxels; ties: object walls first, then lowest
    index."""
    if not u.size:
        return []
    ph = phase[u]
    rows, cols = [], []
    for o in offs:
        j = np.searchsorted(u, u + o)
        j = np.minimum(j, u.size - 1)
        hit = (u[j] == u + o) & (mixed | (ph[j] == ph))
        rows.append(np.flatnonzero(hit))
        cols.append(j[hit])
    r, c = np.concatenate(rows), np.concatenate(cols)
    G = sparse.coo_matrix((np.ones(r.size), (r, c)), shape=(u.size, u.size))
    _, lab = connected_components(G, directed=False)
    order = np.argsort(lab, kind="stable")
    cuts = np.flatnonzero(np.diff(lab[order])) + 1
    walls = np.split(u[order], cuts)

    def key(w: Int) -> tuple[float, int, int]:
        bg = int(phase[w[0]] != _OBJ)
        return (w.size * (fill_cost if bg else 1.0), bg, int(w[0]))

    return sorted(walls, key=key)


def _set_simple(state: npt.NDArray[np.uint8], front: int, p: Int) -> bool:
    """Whether front can claim the voxels p (flat indices into the 3-D state)
    all at once without changing its homology. With N the front voxels next
    to p, the front with p is the union of the front and the full subcomplex
    on p and N, which meet in the full subcomplex on N; when both of those are
    acyclic, Mayer-Vietoris gives the front's homology unchanged. For one
    voxel this is the simple-point test; it also takes the clusters of which
    no single voxel is simple (a simple pair, or an object wall and a
    background wall that block each other)."""
    q = np.stack(np.unravel_index(p, state.shape), 1)
    lo, hi = q.min(0) - 2, q.max(0) + 3
    sub = state[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    pm = np.zeros(sub.shape, bool)
    pm[tuple((q - lo).T)] = True
    n = ndi.binary_dilation(pm, kuhn_structure()) & (sub == front) & ~pm
    return bool(n.any()) and kuhn_betti(n) == kuhn_betti(n | pm) == (1, 0, 0)


def _resolve_cluster(
    state: npt.NDArray[np.uint8],
    phase: npt.NDArray[np.uint8],
    offs: Int,
    shape: tuple[int, ...],
    u: Int,
    fill_cost: float = 1.0,
) -> Move | None | bool:
    """When no wall can be claimed voxel by voxel: claim the smallest set of
    unclaimed voxels that one front can take whole (_set_simple), by the front
    that changes fewer voxels, a fill's counted fill_cost times (ties: the
    background, a cut). The sets tried
    are the walls, the pairs of touching walls of opposite phase and the
    clusters (Kuhn-connected pieces of both phases; a ring of walls is one
    that no front can take whole). u: the unclaimed voxels (sorted flat
    indices). Ties in size: lowest index. Returns the
    move, None if nothing changed (a set of one phase claimed by its own
    front), or False if no set can be claimed."""
    s3 = state.reshape(shape)
    walls = _walls(u, phase, offs)
    lab = np.full(state.size, -1, np.int64)
    for k, w in enumerate(walls):
        lab[w] = k
    pairs = set()
    for k, w in enumerate(walls):
        nb = lab[(w[:, None] + offs[None, :]).ravel()]
        for j in np.unique(nb[nb > k]).tolist():
            if phase[walls[j][0]] != phase[w[0]]:
                pairs.add((k, j))
    sets = walls + [np.sort(np.r_[walls[k], walls[j]]) for k, j in sorted(pairs)]
    sets += [c for c in _walls(u, phase, offs, mixed=True) if len(c) > 1]
    for c in sorted(sets, key=lambda c: (c.size, int(c[0]))):
        n_obj = int((phase[c] == _OBJ).sum())
        n_bg = c.size - n_obj
        options = sorted(
            (
                (float(n_obj), n_obj, _BG, _OBJ, "cut"),
                (fill_cost * n_bg, n_bg, _OBJ, _BG, "fill"),
            )
        )
        for _, changed, front, lose, kind in options:
            if _set_simple(s3, front, c):
                state[c] = front
                if not changed:
                    return None
                taken = c[phase[c] == lose]
                return Move(kind, np.stack(np.unravel_index(taken, shape), 1) - 2)
    return False


def _remove_one(obj: Bool, fill_cost: float = 1.0) -> tuple[Bool, list[Move]]:
    """remove_handles for an object with one component."""
    x = np.pad(obj, 2)
    if x.size >= 1 << _IDX_BITS:
        raise ValueError("array too large for the heap keys")
    s = kuhn_structure()
    bg_lab, _ = ndi.label(~x, s)
    cavity = (bg_lab != bg_lab[0, 0, 0]) & (bg_lab > 0)
    del bg_lab
    xf = x | cavity
    d2a = np.where(
        xf,
        ndi.distance_transform_edt(xf) ** 2,
        ndi.distance_transform_edt(~xf) ** 2,
    )
    d2 = np.rint(d2a).astype(np.int32)
    del d2a
    dmax = int(d2.max()) + 1
    phase = np.where(xf, _OBJ, _BG).astype(np.uint8)
    state = np.zeros(x.shape, np.uint8)
    state[:2], state[-2:], state[:, :2], state[:, -2:] = _BG, _BG, _BG, _BG
    state[:, :, :2], state[:, :, -2:] = _BG, _BG
    state.flat[int(np.argmax(np.where(xf, d2, -1)))] = _OBJ

    offs = (
        KUHN_OFFSETS[:, 0] * x.shape[1] * x.shape[2]
        + KUHN_OFFSETS[:, 1] * x.shape[2]
        + KUHN_OFFSETS[:, 2]
    ).astype(np.int64)
    st, ph, dd = state.ravel(), phase.ravel(), d2.ravel()
    queued = np.zeros(st.size, np.bool_)
    _grow(st, ph, dd, np.flatnonzero(st == _U), offs, SIMPLE, dmax, queued)

    # u: the unclaimed voxels, sorted. Voxels are only ever claimed, so each
    # pass refreshes u from its last value instead of from the whole array.
    u = np.flatnonzero(st == _U)
    moves: list[Move] = []
    while True:
        u = u[st[u] == _U]
        walls = _walls(u, ph, offs, fill_cost=fill_cost)
        done = False
        for w in walls:
            own = int(ph[w[0]])
            other = _BG if own == _OBJ else _OBJ
            got = np.array([i for i in w if st[i] == _U], np.int64)
            if not _claim(st, w, other, offs, SIMPLE):
                continue
            taken = got[st[got] == other]
            vox = np.stack(np.unravel_index(taken, x.shape), 1) - 2
            moves.append(Move("cut" if own == _OBJ else "fill", vox))
            u = u[st[u] == _U]
            _grow(st, ph, dd, u, offs, SIMPLE, dmax, queued)
            done = True
            break
        if not done:
            m = _resolve_cluster(st, ph, offs, x.shape, u, fill_cost)
            if m is False:
                break
            if isinstance(m, Move):
                moves.append(m)
            u = u[st[u] == _U]
            _grow(st, ph, dd, u, offs, SIMPLE, dmax, queued)

    # the rest of the background stays background; the rest of the object
    # joins the object front where it can (simple for it, alone or as a
    # cluster), and what cannot is removed
    st[(st == _U) & (ph == _BG)] = _BG
    _claim(st, np.flatnonzero(st == _U), _OBJ, offs, SIMPLE)
    for c in _walls(np.flatnonzero(st == _U), ph, offs):
        if _set_simple(state, _OBJ, c):
            st[c] = _OBJ
    left = np.argwhere((state == _U) & x) - 2
    if len(left):
        moves.append(Move("stranded", left))
    out = (state == _OBJ) & ~cavity
    return out[2:-2, 2:-2, 2:-2], moves


def remove_handles(obj: npt.ArrayLike, fill_cost: float = 1.0) -> HandleRemoval:
    """Remove every handle of the object (3D bool) with the Kuhn adjacency,
    keeping its components and cavities: each handle is cut or filled,
    whichever changes fewer voxels, a fill's voxels counted fill_cost times
    (module docstring). Outside the array is background."""
    a = np.asarray(obj, bool)
    lab, n = ndi.label(a, kuhn_structure())
    out = np.zeros_like(a)
    moves: list[Move] = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        lo = np.array([max(t.start - 2, 0) for t in sl])
        hi = np.array([min(t.stop + 2, m) for t, m in zip(sl, a.shape, strict=True)])
        box = tuple(slice(int(p), int(q)) for p, q in zip(lo, hi, strict=True))
        got, mv = _remove_one(lab[box] == k, fill_cost)
        out[box] |= got
        moves += [Move(m.kind, m.voxels + lo) for m in mv]
    stranded = sum(len(m.voxels) for m in moves if m.kind == "stranded")
    return HandleRemoval(out, moves, stranded)


# ---------------------------------------------------------------- variations
# Changes after the handles are removed, each made of simple points (or of sets
# that keep the homology, _set_simple), so the Betti numbers stay as they are.


@njit
def _sweep(
    state: npt.NDArray[np.uint8], cand: Int, offs: Int, simple: Bool, add: bool
) -> int:
    """Flip the voxels cand (flat indices, in order) of state (1 object, 0
    background): add background voxels to the object (add) or remove object
    voxels, each only when it is a simple point, sweeping in order until none
    flips. Returns the number flipped."""
    n = 0
    want = 0 if add else 1
    progress = True
    while progress:
        progress = False
        for i in cand:
            if state[i] != want:
                continue
            m = _link_mask(state, i, 1, offs)
            if m != 0 and simple[m]:
                state[i] = 1 - want
                n += 1
                progress = True
    return n


def thicken(obj: npt.ArrayLike, seeds: npt.ArrayLike, r: float, add: bool) -> Bool:
    """The object (3D bool) with the background voxels (add) or the object
    voxels within distance r (voxels, Euclidean) of the seeds (3D bool) made
    object or background, one simple point at a time, nearest the seeds first
    (ties: lowest index), as far as simple points allow; the seeds themselves
    are not changed. So the topology is kept: a cut's gap widened (add False,
    the cut's voxels as seeds), a fill's plug thickened (add True). Outside the
    array is background and is not changed."""
    a = np.asarray(obj, bool)
    sd = np.asarray(seeds, bool)
    if r <= 0 or not sd.any():
        return a.copy()
    d = ndi.distance_transform_edt(~sd)
    near = (d <= r) & ~sd & (~a if add else a)
    idx = np.flatnonzero(near)
    idx = idx[np.argsort(d.ravel()[idx], kind="stable")]
    state = np.pad(a, 1).astype(np.uint8)
    pidx = np.ravel_multi_index(
        tuple(c + 1 for c in np.unravel_index(idx, a.shape)), state.shape
    ).astype(np.int64)
    offs = _flat_offsets(state.shape)
    flat = state.ravel()
    _sweep(flat, pidx, offs, SIMPLE, add)
    out: Bool = flat.reshape(state.shape)[1:-1, 1:-1, 1:-1].astype(bool)
    return out


def _flat_offsets(shape: tuple[int, ...]) -> Int:
    return (
        KUHN_OFFSETS[:, 0] * shape[1] * shape[2]
        + KUHN_OFFSETS[:, 1] * shape[2]
        + KUHN_OFFSETS[:, 2]
    ).astype(np.int64)


def trim_threads(obj: npt.ArrayLike, r: float) -> Bool:
    """The object (3D bool) without its thin protrusions: the voxels outside
    its opening by a ball of radius r (voxels: those no ball of radius r
    inside the object covers, so features thinner than about 2r + 1 voxels),
    taken in Kuhn-connected pieces, each removed whole when that keeps the
    homology (_set_simple: a piece that hangs from the rest by one
    attachment). A thin piece that joins two thick parts is kept whole, not
    thinned; components with no part thicker than r are not changed. Outside
    the array is background."""
    a = np.asarray(obj, bool)
    if r <= 0 or not a.any():
        return a.copy()
    p = np.pad(a, 2)  # room for _set_simple's box round a piece
    core = ndi.distance_transform_edt(p) > r
    if not core.any():
        return a.copy()
    opened = ndi.distance_transform_edt(~core) <= r
    s = kuhn_structure()
    comp, _ = ndi.label(p, s)
    keep_comps = np.unique(comp[core])
    thin = p & ~opened & np.isin(comp, keep_comps)
    lab, n = ndi.label(thin, s)
    if not n:
        return a.copy()
    state = np.where(p, _OBJ, _BG).astype(np.uint8)
    flat = state.ravel()
    offs = _flat_offsets(state.shape)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    # single voxels: the simple-point test; pieces are not Kuhn-adjacent to each
    # other, so each test sees none of the others' changes
    one = np.flatnonzero((sizes[lab.ravel()] == 1) & (lab.ravel() > 0))
    m = np.zeros(one.size, np.int64)
    for k in range(14):
        m |= (flat[one + offs[k]] == _OBJ).astype(np.int64) << k
    flat[one[(m != 0) & SIMPLE[m]]] = _BG
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None or sizes[k] == 1:
            continue
        sub = lab[sl] == k
        pts = np.stack(np.nonzero(sub), 1) + np.array([t.start for t in sl])
        pf = np.ravel_multi_index(tuple(pts.T), state.shape).astype(np.int64)
        if _set_simple(state, _BG, pf):
            flat[pf] = _BG
    out: Bool = state[2:-2, 2:-2, 2:-2] == _OBJ
    return out
