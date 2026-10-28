"""Tests of baillarger.viewer.flatmap with exact answers (or a stated tolerance
where the continuum answer is only reached in the limit)."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial import ConvexHull, Delaunay

from baillarger.viewer import flatmap as FM

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def rotation(rng: np.random.Generator) -> Float:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    out: Float = q * np.sign(np.linalg.det(q))
    return out


def cross2(a: Float, b: Float) -> Float:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def similarity_residual(a: Float, b: Float) -> float:
    """Smallest max |s R a + t - b| over similarities (no reflection) of the plane."""
    a0, b0 = a - a.mean(0), b - b.mean(0)
    za, zb = a0[:, 0] + 1j * a0[:, 1], b0[:, 0] + 1j * b0[:, 1]
    c = np.vdot(za, zb) / np.vdot(za, za)
    return float(np.max(np.abs(c * za - zb)))


def grid(nx: int, ny: int, wrap_x: bool = False, wrap_y: bool = False) -> Int:
    """Faces of an nx x ny vertex grid (index i * ny + j), counter-clockwise in
    (i, j)."""
    F = []
    for i in range(nx if wrap_x else nx - 1):
        for j in range(ny if wrap_y else ny - 1):
            a, b = i * ny + j, ((i + 1) % nx) * ny + j
            c, d = ((i + 1) % nx) * ny + (j + 1) % ny, i * ny + (j + 1) % ny
            F += [(a, b, c), (a, c, d)]
    return np.array(F)


def torus(n: int, m: int, R: float = 2.0, r: float = 0.5) -> tuple[Float, Int]:
    u, w = np.meshgrid(
        np.arange(n) * 2 * np.pi / n, np.arange(m) * 2 * np.pi / m, indexing="ij"
    )
    v = np.c_[
        ((R + r * np.cos(w)) * np.cos(u)).ravel(),
        ((R + r * np.cos(w)) * np.sin(u)).ravel(),
        (r * np.sin(w)).ravel(),
    ]
    return v, grid(n, m, True, True)


def cap(n: int, zmin: float = 0.0) -> tuple[Float, Int]:
    """A unit sphere cap z > zmin from a Fibonacci point set, faces oriented outward."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + 5**0.5) * k
    p = np.c_[np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z]
    f = ConvexHull(p).simplices
    nrm = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    f = np.where((np.einsum("ij,ij->i", nrm, p[f].mean(1)) < 0)[:, None], f[:, ::-1], f)
    f = f[p[f, 2].min(1) > zmin]
    used = np.unique(f)
    loc = -np.ones(n, int)
    loc[used] = np.arange(used.size)
    return p[used], loc[f]


# ------------------------------------------------------------- topology and cuts


def test_topology_of_known_surfaces() -> None:
    v, f = torus(12, 8)
    assert FM.topology(f) == dict(V=96, E=288, F=192, b=0, chi=0, genus=1)
    assert FM.orientable_check(f)
    f_cyl = grid(12, 5, wrap_x=True)
    t = FM.topology(f_cyl)
    assert (t["b"], t["genus"], t["chi"]) == (2, 0, 0)
    loops = FM.boundary_loops(f_cyl)
    assert sorted(lp.size for lp in loops) == [12, 12]
    pc, cap_f = cap(400, 0.2)
    assert FM.topology(cap_f)["b"] == 1 and FM.topology(cap_f)["genus"] == 0


def test_cut_along_a_line_splits_a_grid() -> None:
    f = grid(6, 5)
    line = [(i * 5 + 2, (i + 1) * 5 + 2) for i in range(5)]  # the full line j = 2
    f2, orig = FM.cut(f, line)
    lab, _ = FM.pieces(f2, int(f2.max()) + 1)
    assert lab.max() + 1 == 2
    assert len(orig) == 30 + 6  # the six vertices of the line, each twice
    assert np.array_equal(np.sort(orig[f2], axis=None), np.sort(f, axis=None))


def test_shortest_handle_of_a_torus_is_its_meridian() -> None:
    n, m, R, r = 24, 10, 2.0, 0.5
    v, f = torus(n, m, R, r)
    lp, L, _ = FM.shortest_handle(v, f)
    assert L == pytest.approx(m * 2 * r * np.sin(np.pi / m), abs=1e-9)
    assert lp is not None
    assert lp.size == m and np.unique(lp // m).size == 1  # one ring u = const


def test_genus_one_with_a_hole_is_cut_to_genus_zero_then_to_a_disk() -> None:
    n, m = 24, 10
    v, f = torus(n, m)
    hole = np.isin(f, [5 * m + 3]).any(
        1
    )  # remove the faces round one vertex: one boundary loop
    f = f[~hole]
    used = np.unique(f)
    loc = -np.ones(len(v), int)
    loc[used] = np.arange(used.size)
    v, f = v[used], loc[f]
    assert (FM.topology(f)["genus"], FM.topology(f)["b"]) == (1, 1)
    f2, orig, loops = FM.cut_handles(v, f)
    t = FM.topology(f2)
    assert len(loops) == 1
    # a cut loop adds two boundary loops, or one if it runs through the hole's boundary
    # (the two new loops and the hole's are then one)
    touches = np.isin(loops[0][0], np.concatenate(FM.boundary_loops(f))).any()
    assert (t["genus"], t["b"]) == (0, 2 if touches else 3) and FM.pieces(
        f2, len(orig)
    )[0].max() == 0
    f3, o3 = FM.join_boundaries(v[orig], f2)
    t = FM.topology(f3)
    assert (t["genus"], t["b"], t["chi"]) == (0, 1, 1)  # a disk
    uv = FM.lscm(v[orig][o3], f3)
    assert np.all(FM.tangent_maps(v[orig][o3], f3, uv)[3] > 0)  # no face flipped


# ------------------------------------------------------------- LSCM


def test_plane_maps_isometrically_up_to_similarity() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 5, (400, 2))
    f = Delaunay(p).simplices
    det = cross2(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    f = np.where((det < 0)[:, None], f[:, ::-1], f)
    v = np.c_[p, np.zeros(len(p))] @ rotation(rng).T + rng.normal(size=3)
    uv = FM.lscm(v, f)
    assert similarity_residual(p, uv) < 1e-9
    T, s1, s2, _ = FM.tangent_maps(v, f, uv)
    assert np.ptp(s1) < 1e-9 and np.max(np.abs(s1 - s2)) < 1e-9


def test_cylinder_strip_unrolls_exactly() -> None:
    # a polygonal cylinder strip is developable (each quad is a planar rectangle): the
    # map is the unrolled grid, exactly, up to similarity
    nt, nz = 40, 15
    th, z = np.meshgrid(np.linspace(0, np.pi, nt), np.linspace(0, 2, nz), indexing="ij")
    v = np.c_[np.cos(th).ravel(), np.sin(th).ravel(), z.ravel()]
    f = grid(nt, nz)
    uv = FM.lscm(v, f)
    chord = 2 * np.sin(np.pi / (nt - 1) / 2)
    flat = np.c_[(np.arange(nt)[:, None] * chord + 0 * z).ravel(), z.ravel()]
    assert similarity_residual(flat, uv) < 1e-9


def test_hemisphere_cap_keeps_angles() -> None:
    # a sphere cap has conformal maps to the plane (stereographic) but no isometry. On
    # an irregular mesh the per-face angle error of the discrete map is first order in
    # the spacing: the area-weighted mean of s1/s2 - 1 is under 0.025 at 4000 sphere
    # points and halves when the spacing halves. (Which conformal map, of the many, the
    # free boundary picks is not fixed, so the area factor is not tested.)
    qc = []
    for n in (4000, 16000):
        v, f = cap(n, 0.0)
        uv = FM.lscm(v, f)
        T, s1, s2, sa = FM.tangent_maps(v, f, uv)
        assert np.all(sa > 0)
        a = FM.face_areas(v, f)
        qc.append(np.sum(a * (s1 / s2 - 1)) / a.sum())
    assert qc[0] < 0.025 and qc[0] / qc[1] > 1.8, qc


def test_a_long_tube_needs_the_disk_cut() -> None:
    # a torus cut along its meridian is a tube 8 times longer than round: its conformal
    # image is an annulus with radius ratio ~ exp(-2 pi 8 / 2): free-boundary LSCM folds
    # instead. Joined to a disk (cut lengthwise) it unrolls to a strip without a flipped
    # face.
    vt, ft = torus(24, 10)
    f2, o, _ = FM.cut_handles(vt, ft)
    assert np.any(FM.tangent_maps(vt[o], f2, FM.lscm(vt[o], f2))[3] < 0)
    f3, o3 = FM.join_boundaries(vt[o], f2)
    uv = FM.lscm(vt[o][o3], f3)
    assert np.all(FM.tangent_maps(vt[o][o3], f3, uv)[3] > 0)


def test_flatten_packs_pieces_and_scales_to_area() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 3, (200, 2))
    f = Delaunay(p).simplices
    det = cross2(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    f = np.where((det < 0)[:, None], f[:, ::-1], f)
    v1 = np.c_[p, np.zeros(len(p))]
    vt, ft = torus(24, 10)
    v = np.vstack([v1, vt + 10])
    f_all = np.vstack([f, ft + len(v1)])
    with pytest.raises(AssertionError):  # a closed mesh: LSCM needs a boundary
        FM.lscm(vt, ft)
    out = FM.flatten(v, f_all, to_disk=True)
    assert len(out["pieces"]) == 2 and len(out["loops"]) == 1
    assert out["pieces"][1]["after"]["genus"] == 0
    assert np.all(np.isfinite(out["uv_v"]))
    # each piece's 2D area equals its 3D area
    for pc in range(2):
        fk = out["piece"][out["f_cut"][:, 0]] == pc
        assert np.abs(out["signed_area"][fk]).sum() == pytest.approx(
            out["area3"][fk].sum(), rel=1e-9
        )
    assert np.all(out["signed_area"] > 0)
    # the plane piece: isometric, so T has orthonormal columns
    T = out["T_v"][: len(v1)]
    G = np.einsum("nij,nik->njk", T, T)
    assert np.max(np.abs(G - np.eye(2))) < 1e-9


def octahedron() -> tuple[Float, Int]:
    """Vertices +x, -x, +y, -y, +z, -z (ids 0-5), faces oriented outward."""
    v = np.r_[np.eye(3), -np.eye(3)][[0, 3, 1, 4, 2, 5]]
    f = np.array(
        [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4)]
        + [(2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    )
    return v, f


def test_slit_opens_a_closed_sphere_along_a_shortest_path() -> None:
    # the far pair is +x and -x (2 sqrt 2 mm apart over the surface; the rest are
    # sqrt 2 from +x), joined by a path of 2 edges through one of the four others:
    # cut, the middle vertex has two copies and the boundary is a 4-loop
    v, f = octahedron()
    assert FM.topology(f) == dict(V=6, E=12, F=8, b=0, chi=2, genus=0)
    f2, o = FM.slit(v, f)
    assert FM.topology(f2) == dict(V=7, E=14, F=8, b=1, chi=1, genus=0)
    (lp,) = FM.boundary_loops(f2)
    ends, mids = o[lp[::2]], o[lp[1::2]]
    if ends[0] not in (0, 1):
        ends, mids = mids, ends
    assert sorted(ends.tolist()) == [0, 1]
    assert mids[0] == mids[1] and mids[0] in (2, 3, 4, 5)
    assert np.array_equal(o[f2], f)


def test_flatten_opens_a_closed_piece_next_to_an_open_one() -> None:
    # a small closed WM island (no boundary, genus 0) next to an open sheet: LSCM has
    # no boundary on the island, so flatten slits it open first; every vertex gets a
    # map position, and the sheet still maps isometrically
    vo, fo = octahedron()
    p = np.array([(i, j) for i in range(4) for j in range(4)], float)
    fg = grid(4, 4)
    v = np.vstack([np.c_[p, np.zeros(len(p))], vo + 10])
    f = np.vstack([fg, fo + len(p)])
    out = FM.flatten(v, f)
    sheet, island = out["pieces"]
    assert (
        sheet["before"] == sheet["after"] == dict(V=16, E=33, F=18, b=1, chi=1, genus=0)
    )
    assert not sheet["opened"]
    assert island["before"] == dict(V=6, E=12, F=8, b=0, chi=2, genus=0)
    assert island["after"] == dict(V=7, E=14, F=8, b=1, chi=1, genus=0)
    assert island["opened"] and island["handles"] == 0
    assert island["area_mm2"] == round(4 * 3**0.5, 2)
    assert out["loops"] == []
    assert np.all(np.isfinite(out["uv_v"])) and np.all(np.isfinite(out["T_v"]))
    T = out["T_v"][: len(p)]
    G = np.einsum("nij,nik->njk", T, T)
    assert np.max(np.abs(G - np.eye(2))) < 1e-9
    fk = out["piece"][out["f_cut"][:, 0]] == 1
    assert np.abs(out["signed_area"][fk]).sum() == pytest.approx(4 * 3**0.5, rel=1e-9)
