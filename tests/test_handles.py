"""Tests for baillarger.handles: the Kuhn topology against the marching
tetrahedra mesh and a global check, and handle removal on phantoms with known
answers (a drilled hole is filled, a thin bridge and a pinhole contact are
cut, cavities and islands are kept)."""

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

from baillarger import handles as H
from baillarger.mesh import _TETS, marching_tetrahedra, topology

type Bool3 = npt.NDArray[np.bool_]


def mesh_topology(obj: Bool3) -> dict[str, float]:
    v, f = marching_tetrahedra(obj.astype(np.float64), 0.5)
    return topology(f, len(v))


def ball(n: int, r: float, c: tuple[float, float, float] | None = None) -> Bool3:
    g = np.indices((n, n, n)).astype(float)
    c = c or (n // 2, n // 2, n // 2)
    out: Bool3 = ((g - np.array(c)[:, None, None, None]) ** 2).sum(0) <= r * r
    return out


def ring() -> Bool3:
    """A square ring, 5 voxels wide and 6 high, round a 14 x 14 hole."""
    a = np.zeros((30, 30, 12), bool)
    a[3:27, 3:27, 3:9] = True
    a[8:22, 8:22, :] = False
    return a


def blobs(seed: int, n: int = 18, sigma: float = 1.2) -> Bool3:
    rng = np.random.default_rng(seed)
    a = ndi.gaussian_filter(rng.random((n, n, n)), sigma) > 0.5
    a[:2], a[-2:], a[:, :2], a[:, -2:], a[:, :, :2], a[:, :, -2:] = (False,) * 6
    return a


def changes(a: Bool3, r: H.HandleRemoval) -> tuple[int, int]:
    """(voxels added, voxels removed)."""
    return int((r.obj & ~a).sum()), int((a & ~r.obj).sum())


# ---------------------------------------------------------------- Kuhn topology


def test_tetrahedra_and_link_match_the_mesher() -> None:
    assert np.array_equal(H._TETS, _TETS)
    assert H.KUHN_OFFSETS.shape == (14, 3)
    assert set(map(tuple, H.KUHN_OFFSETS)) == set(map(tuple, -H.KUHN_OFFSETS))
    edges = {
        tuple(sorted((int(t[a]), int(t[b]))))
        for t in H.LINK_TRIANGLES
        for a, b in ((0, 1), (1, 2), (2, 0))
    }
    # a triangulated sphere: V - E + F = 14 - 36 + 24 = 2
    assert (len(edges), len(H.LINK_TRIANGLES)) == (36, 24)
    assert [len(t) for t in H._SIMPLICES] == [1, 7, 12, 6]


def test_euler_matches_the_mesh() -> None:
    for seed in range(8):
        a = blobs(seed)
        # the boundary of a compact 3-manifold: chi(surface) = 2 chi(solid)
        assert mesh_topology(a)["chi"] == 2 * H.kuhn_euler(a)


def test_betti_of_known_solids() -> None:
    assert H.kuhn_betti(ball(15, 5)) == (1, 0, 0)
    assert H.kuhn_betti(ball(17, 7) & ~ball(17, 3)) == (1, 0, 1)
    assert H.kuhn_betti(ring()) == (1, 1, 0)
    two = ring()
    two[3:27, 14:16, 3:9] = True  # a crossbar: genus 2
    assert H.kuhn_betti(two) == (1, 2, 0)
    assert H.kuhn_betti(ball(21, 4, (5, 5, 5)) | ball(21, 4, (15, 15, 15))) == (2, 0, 0)


def link_contractible(m: int) -> bool:
    """Whether the full subcomplex of the link on the bits of m is contractible:
    a planar complex is, exactly when it is connected and chi = 1."""
    on = [k for k in range(14) if m >> k & 1]
    if not on:
        return False
    tris = [t for t in H.LINK_TRIANGLES if all(m >> int(k) & 1 for k in t)]
    edges = {
        tuple(sorted((int(t[a]), int(t[b]))))
        for t in H.LINK_TRIANGLES
        for a, b in ((0, 1), (1, 2), (2, 0))
        if m >> int(t[a]) & 1 and m >> int(t[b]) & 1
    }
    g = np.zeros((14, 14), bool)
    for a, b in edges:
        g[a, b] = g[b, a] = True
    seen, stack = {on[0]}, [on[0]]
    while stack:
        for j in np.flatnonzero(g[stack.pop()]).tolist():
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == len(on) and len(on) - len(edges) + len(tris) == 1


def test_simple_table_matches_the_link_condition() -> None:
    # adding a vertex to a full subcomplex keeps its homotopy type exactly when
    # the vertex's link in the subcomplex is contractible
    for m in range(1 << 14):
        assert bool(H.SIMPLE[m]) == link_contractible(m), m


def test_simple_points_keep_the_betti_numbers() -> None:
    rng = np.random.default_rng(1)
    n_simple = 0
    for _ in range(400):
        a = np.zeros((5, 5, 5), bool)
        a[1:4, 1:4, 1:4] = rng.random((3, 3, 3)) < rng.uniform(0.2, 0.8)
        a[2, 2, 2] = False
        m = sum(1 << k for k, o in enumerate(H.KUHN_OFFSETS) if a[tuple(o + 2)])
        b = a.copy()
        b[2, 2, 2] = True
        if H.SIMPLE[m]:
            n_simple += 1
            assert H.kuhn_betti(a) == H.kuhn_betti(b)
    assert n_simple > 50


# ---------------------------------------------------------------- handle removal


def test_no_handle_no_change() -> None:
    for a in (ball(15, 5), ball(17, 7) & ~ball(17, 3)):
        r = H.remove_handles(a)
        assert np.array_equal(r.obj, a) and r.moves == [] and r.stranded == 0


def test_drilled_hole_is_filled() -> None:
    a = np.zeros((20, 20, 20), bool)
    a[4:16, 4:16, 4:16] = True
    a[9:11, 9:11, :] = False  # a 2 x 2 tunnel through a 12-voxel block
    assert H.kuhn_betti(a) == (1, 1, 0)
    r = H.remove_handles(a)
    assert H.kuhn_betti(r.obj) == (1, 0, 0)
    assert [m.kind for m in r.moves] == ["fill"]
    assert changes(a, r) == (4, 0)


def test_thin_bridge_is_cut() -> None:
    a = ring()
    a[3:8, 13:17, 3:9] = False
    a[5, 13:17, 6] = True  # the ring closed by a rod one voxel across
    assert H.kuhn_betti(a) == (1, 1, 0)
    r = H.remove_handles(a)
    assert H.kuhn_betti(r.obj) == (1, 0, 0)
    assert [m.kind for m in r.moves] == ["cut"]
    assert changes(a, r) == (0, 1)


def test_thick_ring_is_cut_not_filled() -> None:
    # cut: 5 x 6 = 30 voxels across the ring; fill: about 14 x 14 = 196
    a = ring()
    r = H.remove_handles(a)
    added, removed = changes(a, r)
    assert H.kuhn_betti(r.obj) == (1, 0, 0)
    assert [m.kind for m in r.moves] == ["cut"]
    assert added == 0 and removed <= 2 * 30


def test_pinhole_between_two_sheets_is_cut() -> None:
    # two sheets joined along one edge (a fold), touching at one column
    a = np.zeros((30, 30, 16), bool)
    a[3:27, 3:27, 3:6] = True
    a[3:27, 3:27, 9:12] = True
    a[3:6, 3:27, 3:12] = True
    a[18, 15, 6:9] = True
    assert H.kuhn_betti(a) == (1, 1, 0)
    r = H.remove_handles(a)
    assert H.kuhn_betti(r.obj) == (1, 0, 0)
    assert [m.kind for m in r.moves] == ["cut"]
    added, removed = changes(a, r)
    assert added == 0 and 1 <= removed <= 3
    assert all(a[tuple(p)] and p[2] in (6, 7, 8) for m in r.moves for p in m.voxels)


def test_cavities_and_islands_are_kept() -> None:
    n = 40
    a = ball(n, 9, (14, 14, 14)) & ~ball(n, 4, (14, 14, 14))  # a hollow ball
    a |= ball(n, 1.5, (14, 14, 14))  # an island in its cavity
    a |= ball(n, 2.5, (34, 6, 34))  # an island outside
    a[7:22, 22:37, 12:17] = True  # a thick square ring fused to the shell
    a[11:18, 26:33, :] = False
    assert H.kuhn_betti(a) == (3, 1, 1)
    r = H.remove_handles(a)
    assert H.kuhn_betti(r.obj) == (3, 0, 1)
    assert [m.kind for m in r.moves] == ["cut"]
    assert r.obj[14, 14, 14] and not r.obj[14, 14, 11]  # island and cavity kept


def test_random_blobs_become_handle_free_and_the_mesh_genus_zero() -> None:
    for seed in range(12):
        a = blobs(seed, 22)
        b0, b1, b2 = H.kuhn_betti(a)
        r = H.remove_handles(a)
        assert H.kuhn_betti(r.obj) == (b0, 0, b2), seed
        added, removed = changes(a, r)
        assert added == sum(len(m.voxels) for m in r.moves if m.kind == "fill")
        assert removed == sum(len(m.voxels) for m in r.moves if m.kind != "fill")
        assert r.stranded == sum(len(m.voxels) for m in r.moves if m.kind == "stranded")
        t = mesh_topology(r.obj)
        assert t["genus"] == 0 and t["nonmanifold_edges"] == 0
        # any box crop of a handle-free solid gives planar pieces
        for lo in ((0, 0, 0), (5, 3, 7), (8, 8, 2)):
            crop = r.obj[lo[0] : lo[0] + 12, lo[1] : lo[1] + 13, lo[2] : lo[2] + 11]
            assert mesh_topology(crop)["genus"] == 0, (seed, lo)


def test_stranded_voxels_cannot_be_put_back() -> None:
    # finer blobs: object and background walls that block each other, of
    # which no single voxel is simple for either front, are claimed as whole
    # clusters; a voxel left over is removed only when it cannot be put back
    # alone without changing the topology
    for seed in range(40):
        a = blobs(seed, 18, 1.0)
        b0, _, b2 = H.kuhn_betti(a)
        r = H.remove_handles(a)
        assert H.kuhn_betti(r.obj) == (b0, 0, b2), seed
        added, removed = changes(a, r)
        assert added == sum(len(m.voxels) for m in r.moves if m.kind == "fill")
        assert removed == sum(len(m.voxels) for m in r.moves if m.kind != "fill")
        for m in r.moves:
            for v in m.voxels if m.kind == "stranded" else ():
                b = r.obj.copy()
                b[tuple(v)] = True
                assert H.kuhn_betti(b) != H.kuhn_betti(r.obj), (seed, v)
