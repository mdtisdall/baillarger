"""Tests for baillarger.handles: the Kuhn topology against the marching
tetrahedra mesh and a global check, and handle removal on phantoms with known
answers (a drilled hole is filled, a thin bridge and a pinhole contact are
cut, cavities and islands are kept)."""

import hashlib

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


# ---------------------------------------------------------------- exact output


def perforated_slab(k: int, seed: int) -> Bool3:
    """A slab 8 voxels thick with k tunnels (radius 1.5 to 3) through it on a
    jittered grid 12 voxels apart: b1 = k."""
    rng = np.random.default_rng(seed)
    m = int(np.ceil(np.sqrt(k)))
    n = m * 12 + 8
    a = np.zeros((n, n, 24), bool)
    a[2:-2, 2:-2, 8:16] = True
    i, j = np.mgrid[:n, :n]
    for t in range(k):
        cx = 10 + (t // m) * 12 + int(rng.integers(-2, 3))
        cy = 10 + (t % m) * 12 + int(rng.integers(-2, 3))
        r = rng.uniform(1.5, 3.0)
        a[(i - cx) ** 2 + (j - cy) ** 2 <= r * r, :] = False
    return a


def digest(obj: Bool3, moves: list[H.Move]) -> str:
    """A hash of an output and its moves, in order."""
    h = hashlib.sha256(np.packbits(obj).tobytes() + str(obj.shape).encode())
    for m in moves:
        h.update(m.kind.encode() + np.ascontiguousarray(m.voxels, np.int64).tobytes())
    return h.hexdigest()[:16]


def reference_digests() -> dict[str, str]:
    """The digests of remove_handles (and _remove_one on a padded array) on
    the phantoms of the tests above and on perforated slabs."""
    hole = np.zeros((20, 20, 20), bool)
    hole[4:16, 4:16, 4:16] = True
    hole[9:11, 9:11, :] = False
    bridge = ring()
    bridge[3:8, 13:17, 3:9] = False
    bridge[5, 13:17, 6] = True
    sheets = np.zeros((30, 30, 16), bool)
    sheets[3:27, 3:27, 3:6] = True
    sheets[3:27, 3:27, 9:12] = True
    sheets[3:6, 3:27, 3:12] = True
    sheets[18, 15, 6:9] = True
    shell = ball(40, 9, (14, 14, 14)) & ~ball(40, 4, (14, 14, 14))
    shell |= ball(40, 1.5, (14, 14, 14)) | ball(40, 2.5, (34, 6, 34))
    shell[7:22, 22:37, 12:17] = True
    shell[11:18, 26:33, :] = False
    cases: dict[str, Bool3] = {
        "hole": hole,
        "bridge": bridge,
        "ring": ring(),
        "sheets": sheets,
        "shell": shell,
        "slab": perforated_slab(16, 16),
    }
    cases |= {f"blobs22-{s}": blobs(s, 22) for s in range(12)}
    cases |= {f"blobs18-{s}": blobs(s, 18, 1.0) for s in range(40)}
    out = {}
    for name, a in cases.items():
        r = H.remove_handles(a)
        out[name] = digest(r.obj, r.moves)
    # one component in a padded box, straight to _remove_one
    out["slab-padded"] = digest(*H._remove_one(np.pad(perforated_slab(16, 16), 20)))
    return out


REFERENCE = {
    "hole": "6bb2a14cf71aa076",
    "bridge": "2d9cb6266a893ebe",
    "ring": "b247ab13bb5a183a",
    "sheets": "7adb4f5782cd5134",
    "shell": "1fa6bccb5bf47802",
    "slab": "e81c8ad7b20f11bd",
    "blobs22-0": "5673df3065670b06",
    "blobs22-1": "98f0e222786171d1",
    "blobs22-2": "aff212d1ad2c8b1f",
    "blobs22-3": "e29df872e9ec7a20",
    "blobs22-4": "82eb5c91baec57ae",
    "blobs22-5": "e5081443a14332a7",
    "blobs22-6": "5d4fad58185a29d3",
    "blobs22-7": "db1dc3f40b170420",
    "blobs22-8": "fdf203da521885b5",
    "blobs22-9": "2bd60250327a2725",
    "blobs22-10": "83e9014149b0c8d5",
    "blobs22-11": "8018dadc543e1883",
    "blobs18-0": "bfe38d0d6f035b64",
    "blobs18-1": "3e9bff85b0e8291a",
    "blobs18-2": "9ff03e5bccccbc66",
    "blobs18-3": "d0e115446705a81a",
    "blobs18-4": "51c3ca2ab0ed2c68",
    "blobs18-5": "482f1cdcf5419822",
    "blobs18-6": "6893501b011d8247",
    "blobs18-7": "dbfa1b1f8f1e953b",
    "blobs18-8": "7b43dcfc4ca98ba5",
    "blobs18-9": "316bd1e3744d2abb",
    "blobs18-10": "b2665ce2d28d3a11",
    "blobs18-11": "75b33dc19824d9d0",
    "blobs18-12": "217d719645f17cee",
    "blobs18-13": "e17f00eb648c0fae",
    "blobs18-14": "67b87800b004b5fd",
    "blobs18-15": "4468f51eeea6ba3d",
    "blobs18-16": "c7bd6402ca992519",
    "blobs18-17": "a57459889829062c",
    "blobs18-18": "8fcda9bc35402642",
    "blobs18-19": "a2875ed1f8130702",
    "blobs18-20": "bb17183e247ab543",
    "blobs18-21": "f0a75ae29c9b02e9",
    "blobs18-22": "2604b72dd6bfad66",
    "blobs18-23": "3fc2ae77e096a5f3",
    "blobs18-24": "b74f9b70364ac30d",
    "blobs18-25": "9df049d4f0f563f1",
    "blobs18-26": "d931dff70d02fd93",
    "blobs18-27": "051c06b3b97282ca",
    "blobs18-28": "613043606be67eee",
    "blobs18-29": "bd7eb2ab91442b4b",
    "blobs18-30": "3bbea9134c92512b",
    "blobs18-31": "ccc59d37f1520e8f",
    "blobs18-32": "69fa5e680b94a894",
    "blobs18-33": "46146bcb90ce5a43",
    "blobs18-34": "f297f2186472d5a9",
    "blobs18-35": "60cccedb275942c8",
    "blobs18-36": "d8c5a29cde5ffccf",
    "blobs18-37": "6be0ba64a51a85bc",
    "blobs18-38": "cedc038049eb5bb5",
    "blobs18-39": "9d9079dfdff62599",
    "slab-padded": "25078be5e6bfb0d8",
}


def test_output_is_the_reference_output() -> None:
    # exact output of the one-wall-per-pass loop, pinned before its
    # per-pass work was cut down: any change in the order of claims shows here
    assert H.kuhn_betti(perforated_slab(16, 16)) == (1, 16, 0)
    assert reference_digests() == REFERENCE
