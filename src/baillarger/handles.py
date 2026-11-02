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
) -> list[Int]:
    """Connected pieces (Kuhn adjacency, one phase, or both phases together
    when mixed) of the voxels u (sorted flat indices: the unclaimed voxels),
    smallest first; ties: object walls first, then lowest index."""
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
    return sorted(walls, key=lambda w: (w.size, int(phase[w[0]] != _OBJ), int(w[0])))


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
    nofill: Bool | None = None,
) -> Move | None | bool:
    """When no wall can be claimed voxel by voxel: claim the smallest set of
    unclaimed voxels that one front can take whole (_set_simple), by the front
    that changes fewer voxels (ties: the background, a cut); a fill that would
    take a voxel of nofill (flat, the size of state) is not tried. The sets tried
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
        options = sorted(
            ((n_obj, _BG, _OBJ, "cut"), (c.size - n_obj, _OBJ, _BG, "fill"))
        )
        for changed, front, lose, kind in options:
            if (
                kind == "fill"
                and nofill is not None
                and nofill[c[phase[c] == _BG]].any()
            ):
                continue
            if _set_simple(s3, front, c):
                state[c] = front
                if not changed:
                    return None
                taken = c[phase[c] == lose]
                return Move(kind, np.stack(np.unravel_index(taken, shape), 1) - 2)
    return False


def _remove_one(obj: Bool, nofill: Bool | None = None) -> tuple[Bool, list[Move]]:
    """remove_handles for an object with one component."""
    x = np.pad(obj, 2)
    forb = None if nofill is None else np.pad(nofill, 2).ravel()
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
        walls = _walls(u, ph, offs)
        done = False
        for w in walls:
            own = int(ph[w[0]])
            other = _BG if own == _OBJ else _OBJ
            if own == _BG and forb is not None and forb[w].any():
                continue  # a fill may not take these voxels
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
            m = _resolve_cluster(st, ph, offs, x.shape, u, forb)
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


def remove_handles(
    obj: npt.ArrayLike, nofill: npt.ArrayLike | None = None
) -> HandleRemoval:
    """Remove every handle of the object (3D bool) with the Kuhn adjacency,
    keeping its components and cavities: each handle is cut or filled,
    whichever changes fewer voxels (module docstring). A fill never takes a
    voxel of nofill (3D bool, the object's shape): a handle whose fill would
    is cut. Outside the array is background."""
    nf = None if nofill is None else np.asarray(nofill, bool)
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
        got, mv = _remove_one(lab[box] == k, None if nf is None else nf[box])
        out[box] |= got
        moves += [Move(m.kind, m.voxels + lo) for m in mv]
    stranded = sum(len(m.voxels) for m in moves if m.kind == "stranded")
    return HandleRemoval(out, moves, stranded)


# ---------------------------------------------------------------- thickening


@dataclass(frozen=True)
class Thickening:
    """The result of thicken: the object, the voxels added, the target voxels
    that could not be added without changing the topology (blocked), and the
    passes the growth took."""

    obj: Bool
    added: Bool
    blocked: Bool
    passes: int


def thin_voxels(
    obj: npt.ArrayLike, s: float, sampling: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Bool:
    """The object's (3D bool) voxels that no ball of radius s (in the units of
    sampling, the voxel size) inside the object holds: where it is thinner
    than about 2s. A ball holds a voxel when the voxel is within s of its
    centre, and the centre is more than s from the background. Distances
    within a millionth of a voxel of s count as s (grid distances often equal
    it, and voxel sizes read as float32 would otherwise decide)."""
    a = np.asarray(obj, bool)
    s = s + 1e-6 * min(sampling)
    core = ndi.distance_transform_edt(a, sampling=sampling) > s
    if not core.any():
        return a.copy()
    out: Bool = a & (ndi.distance_transform_edt(~core, sampling=sampling) > s)
    return out


def thicken(
    obj: npt.ArrayLike,
    s: float,
    sampling: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_passes: int = 1000,
) -> Thickening:
    """Thicken the object (3D bool) where it is thinner than about 2s,
    keeping its topology. The target is the object with a ball of radius s
    (in the units of sampling, the voxel size) round each of its thin voxels
    (thin_voxels), so every target voxel lies in a ball of radius s inside
    the target, closed by radius s near the balls (a fillet where they meet
    the object). The object grows into the target by simple points only (so
    its Betti numbers and its background's stay), in passes: each pass adds
    every simple target voxel next to the object, one parity class
    (i % 2, j % 2, k % 2) at a time. Voxels of one class are never in each
    other's neighbourhoods, so adding them together is adding them one by
    one. Passes repeat until one adds nothing. The target voxels left out
    are blocked: adding them would join two parts of the object or close a
    handle or a cavity (a gap narrower than the target). Outside the array
    is background; the object is not grown within 2 voxels of the border."""
    a = np.asarray(obj, bool)
    thin = thin_voxels(a, s, sampling)
    s = s + 1e-6 * min(sampling)  # ties as in thin_voxels
    target = a | (ndi.distance_transform_edt(~thin, sampling=sampling) <= s)
    # the union of balls has notches below the scale s where it meets the
    # object (a voxel of background between a ball and the object's side);
    # the closing by radius s, near what is added, fills them with a fillet
    near = ndi.distance_transform_edt(~(target & ~a), sampling=sampling) <= s
    grown = ndi.distance_transform_edt(~target, sampling=sampling) <= s
    target |= near & (ndi.distance_transform_edt(grown, sampling=sampling) > s)
    del near, grown
    pad = 2
    target[:pad] = target[-pad:] = False
    target[:, :pad] = target[:, -pad:] = False
    target[:, :, :pad] = target[:, :, -pad:] = False
    target |= a
    flat = np.ascontiguousarray(a).ravel().copy()
    want = (target & ~a).ravel()
    offs = _flat_offsets(a.shape)
    cand = np.flatnonzero(want)
    i, j, k = np.unravel_index(cand, a.shape)
    cls = ((i % 2) * 4 + (j % 2) * 2 + k % 2).astype(np.uint8)
    del i, j, k
    passes = 0
    while cand.size and passes < max_passes:
        passes += 1
        grew = False
        for c in range(8):
            v = cand[(cls == c) & ~flat[cand]]
            if not v.size:
                continue
            m = np.zeros(v.size, np.int64)
            for b in range(14):
                m |= flat[v + offs[b]].astype(np.int64) << b
            ok = SIMPLE[m]
            if ok.any():
                flat[v[ok]] = True
                grew = True
        keep = ~flat[cand]
        cand, cls = cand[keep], cls[keep]
        if not grew:
            break
    out = flat.reshape(a.shape)
    return Thickening(out, out & ~a, target & ~out, passes)


# ---------------------------------------------------------------- well-composed


def _square_configs() -> Int:
    """The 12 squares of 2 x 2 voxels (in the three axis planes) that hold a
    voxel, as offsets from it (12, 4, 3): the diagonal pair first, then the
    other pair."""
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for oa in (-1, 0):
            for ob in (-1, 0):
                corners = []
                for da, db in ((0, 0), (1, 1), (1, 0), (0, 1)):
                    o = [0, 0, 0]
                    o[a], o[b] = oa + da, ob + db
                    corners.append(o)
                out.append(corners)
    return np.array(out, np.int64)


def _cube_configs() -> Int:
    """The 8 cubes of 2 x 2 x 2 voxels that hold a voxel, as offsets from it
    (8, 8, 3), corner c at bit pattern c (so c and 7 - c are opposite)."""
    out = []
    for o in itertools.product((-1, 0), repeat=3):
        out.append(
            [
                [o[0] + (c >> 2 & 1), o[1] + (c >> 1 & 1), o[2] + (c & 1)]
                for c in range(8)
            ]
        )
    return np.array(out, np.int64)


_SQUARES = _square_configs()
_CUBES = _cube_configs()


def _bad_square(x: list[Bool]) -> Bool:
    """A checkerboard: the diagonal pair alike, the other pair alike, the two
    pairs different."""
    out: Bool = (x[0] == x[1]) & (x[2] == x[3]) & (x[0] != x[2])
    return out


def _bad_cube(x: list[Bool]) -> Bool:
    """Two opposite corners of one phase and the other six of the other."""
    n = np.zeros(x[0].shape, np.uint8)
    for c in x:
        n += c
    out = np.zeros(n.shape, bool)
    for u in range(4):
        pair = x[u] & x[7 - u]
        out |= (n == 2) & pair
        out |= (n == 6) & ~x[u] & ~x[7 - u]
    return out


def pinched(obj: npt.ArrayLike) -> Bool:
    """The voxels of every diagonal-only contact of the object (3D bool): a
    2 x 2 square in a checkerboard, or a 2 x 2 x 2 cube whose two voxels of
    one phase touch only at a corner. Either phase can be the pinched one
    (the Kuhn adjacency joins one diagonal of each square and one of each
    cube); either way the surface between them has a pinch of zero width."""
    a = np.asarray(obj, bool)
    out = np.zeros_like(a)
    n = a.shape
    for cfgs, bad in ((_SQUARES, _bad_square), (_CUBES, _bad_cube)):
        # each configuration whose least corner is (0, 0, 0)
        for corners in cfgs[(cfgs.min(1) == 0).all(1)]:
            ext = corners.max(0)
            sl = [
                tuple(
                    slice(int(c[d]), n[d] - int(ext[d]) + int(c[d])) for d in range(3)
                )
                for c in corners
            ]
            hit = bad([a[s] for s in sl])
            for s in sl:
                out[s] |= hit
    return out


def _contacts(flat: Bool, v: Int, shape: tuple[int, ...]) -> Int:
    """The number of diagonal-only contacts that hold each voxel v (flat
    indices)."""
    st = np.array([shape[1] * shape[2], shape[2], 1], np.int64)
    out = np.zeros(v.size, np.int64)
    for cfgs, bad in ((_SQUARES, _bad_square), (_CUBES, _bad_cube)):
        for cfg in cfgs:
            out += bad([flat[v + int(o @ st)] for o in cfg])
    return out


@dataclass(frozen=True)
class WellComposed:
    """The result of well_compose: the object, the voxels added and removed,
    the voxels of the diagonal-only contacts left, and the passes taken."""

    obj: Bool
    added: Bool
    removed: Bool
    left: Bool
    passes: int


def well_compose(obj: npt.ArrayLike, max_passes: int = 100) -> WellComposed:
    """Remove the object's (3D bool) diagonal-only contacts (pinched), as
    FreeSurfer's mri_pretess does, keeping its topology. Only voxels of a
    contact change, and only by simple points, one parity class
    (i % 2, j % 2, k % 2) at a time (as in thicken), and each voxel at most
    once, so the passes end. A pass first adds the background voxels, then
    removes the object voxels, whose change leaves no contact at that voxel.
    When that changes nothing, it adds the background voxels whose addition
    leaves no more contacts at that voxel than before: a contact through the
    Kuhn diagonal of a cube takes two additions in a row, and the first only
    turns it into a contact along an edge. Passes repeat until one changes
    nothing; the contacts left are reported. Voxels within 2 of the array
    border are not changed."""
    a = np.asarray(obj, bool)
    shape = a.shape
    inner = np.zeros(shape, bool)
    inner[2:-2, 2:-2, 2:-2] = True
    flat = np.ascontiguousarray(a).ravel().copy()
    fixed = ~inner.ravel()
    offs = _flat_offsets(shape)

    def flip(phase: bool, strict: bool) -> bool:
        cand = np.flatnonzero(pinched(flat.reshape(shape)).ravel() & ~fixed)
        cand = cand[flat[cand] != phase]
        i, j, k = np.unravel_index(cand, shape)
        cls = (i % 2) * 4 + (j % 2) * 2 + k % 2
        changed = False
        for c in range(8):
            v = cand[cls == c]
            if not v.size:
                continue
            before = _contacts(flat, v, shape)
            v, before = v[before > 0], before[before > 0]
            m = np.zeros(v.size, np.int64)
            for b in range(14):
                m |= flat[v + offs[b]].astype(np.int64) << b
            v, before = v[SIMPLE[m]], before[SIMPLE[m]]
            flat[v] = phase
            after = _contacts(flat, v, shape)
            ok = after == 0 if strict else after <= before
            flat[v[~ok]] = not phase
            if ok.any():
                fixed[v[ok]] = True
                changed = True
        return changed

    passes = 0
    while passes < max_passes:
        passes += 1
        changed = flip(True, True)
        changed |= flip(False, True)
        if not changed and not flip(True, False):
            break
    out = flat.reshape(shape)
    return WellComposed(out, out & ~a, a & ~out, pinched(out), passes)


def _flat_offsets(shape: tuple[int, ...]) -> Int:
    """The flat-index offsets of the 14 Kuhn neighbours in an array of shape."""
    return (
        KUHN_OFFSETS[:, 0] * shape[1] * shape[2]
        + KUHN_OFFSETS[:, 1] * shape[2]
        + KUHN_OFFSETS[:, 2]
    ).astype(np.int64)
