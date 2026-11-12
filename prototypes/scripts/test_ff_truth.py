"""Tests of ff_truth.py: the plan's exact test *Truth adapters* (docs/plans/flow-field-phantoms.md):
"on each phantom, the true stops lie on the true stop surface to within 0.01 mm, and no two true
columns cross".

Every tolerance here was fixed when the test was written, before any run. The cases use coarse
root spacings, so that the file runs in well under 60 s, but every family is covered: the slab, a
crown, the sub-voxel fundus, the curved fundus, the smooth and the sharp connected kiss, the
abraded phantom and the τ phantom.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay, cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import bent_phantom as BP  # noqa: E402
import ff_truth as FT  # noqa: E402
import ff_types as T  # noqa: E402

FEATS = {e: dict(profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))), wm_level=-0.3, fluid_level=-2.0,
                 hp_stat=0.0) for e in ("echo2", "echo3")}

# name: root spacing (mm). The plan's boxes.
CASES = {
    "slab": 0.64,
    "crown_w0.64": 0.32,
    "fundus_g0.04": 0.64,
    "curved_g0.16": 0.64,
    "ckiss_smooth": 0.64,
    "ckiss_sharp": 0.64,
    "abraded": 0.64,
    "tau": 0.64,
}
NAMES = list(CASES)
TWO_BANKS = ("fundus_g0.04", "curved_g0.16", "ckiss_smooth", "ckiss_sharp", "abraded")

STOP_TOL_MM = 0.01  # the plan: the stops lie on the stop surface to within 0.01 mm
IN_TISSUE_TOL_MM = 1e-6  # stop_sdf at every path point
ROOT_PHI_TOL = 1e-9
STEP_REL_TOL = 1e-3  # |step length / step_mm - 1| at every full step
TANGENT_TOL_RAD = 1e-3
JUMP_ZERO_TOL = 1e-6
ABRADED_JUMP_TOL = 1e-3

# The abraded phantom's closed form (the plan's table; ff_simple): bank A of the curved fundus,
# g 0.16, tA 2.5, tB 1.2, tau_scale (tA + tB) / 2; two discs on the pia x = g, centres y = 3 and
# y = 8 at z = 0, radius 2, depths 0.3 and 0.9.
AB_G, AB_TA, AB_TS = 0.16, 2.5, 1.85
AB_DISCS = ((3.0, 0.3), (8.0, 0.9))  # (centre y, depth)
AB_RADIUS = 2.0


@functools.lru_cache(maxsize=None)
def case(name: str) -> tuple[object, float, T.Truth]:
    geo, ts = FT.geometry_of(name)
    return geo, ts, FT.truth(geo, ts, name, root_spacing_mm=CASES[name])


def local_paths(geo: object, tr: T.Truth) -> np.ndarray:
    P = tr.paths
    return np.asarray(geo.to_local(P.reshape(-1, 3)).reshape(P.shape))  # type: ignore[attr-defined]


def ckiss_g(z: np.ndarray, half: float) -> np.ndarray:
    """g(z) of the plan's table: 0.32 below the transition, 0.02 above, the quintic between."""
    u = np.clip((z + half) / (2 * half), 0.0, 1.0)
    S = u**3 * (10 - 15 * u + 6 * u**2)
    return 0.32 + (0.02 - 0.32) * S


# ---------------------------------------------------------------- the registry


def test_registry_holds_the_plans_table() -> None:
    want = {"slab", "noise", "crown_w0.32", "crown_w0.64", "crown_w1.28", "ckiss_smooth", "ckiss_sharp",
            "ckiss_smooth_flat", "ckiss_sharp_flat", "fundus_g0.48", "fundus_g0.16", "fundus_g0.04",
            "curved_g0.16", "curved_g0.04", "abraded", "tau"}
    assert set(FT.PHANTOMS) == want and len(FT.PHANTOMS) == len(want)
    for n in want - {"noise"}:
        geo, ts = FT.geometry_of(n)
        assert isinstance(geo, T.Geometry) and ts > 0
    with pytest.raises(KeyError):
        FT.geometry_of("noise")


@pytest.mark.parametrize("name", ["slab", "ckiss_sharp_flat", "tau", "abraded"])
def test_build_phantom_matches_geometry_of(name: str) -> None:
    shape = (24, 24, 24)
    ph = FT.build_phantom(name, FEATS, shape=shape)
    geo, ts = FT.geometry_of(name, shape)
    assert ph.name == name and ph.shape == shape and ph.tau_scale == ts
    assert type(ph.geometry) is type(geo)
    assert ph.geometry.p == geo.p and ph.geometry.origin_mm == geo.origin_mm  # type: ignore[union-attr]


# ---------------------------------------------------------------- roots


@pytest.mark.parametrize("name", NAMES)
def test_roots_starts_and_offsets(name: str) -> None:
    geo, ts, tr = case(name)
    assert tr.n > 20
    Yr = geo.to_local(tr.roots)  # type: ignore[attr-defined]
    assert np.abs(geo.phi(Yr)).max() < ROOT_PHI_TOL  # type: ignore[attr-defined]
    d, _ = cKDTree(Yr).query(Yr, k=2)
    assert d[:, 1].min() >= CASES[name] * (1 - 1e-6)
    assert np.allclose(np.linalg.norm(tr.normals, axis=1), 1.0, atol=1e-12)
    assert np.allclose(tr.normals, T.true_direction(geo, tr.roots), atol=1e-12)  # type: ignore[arg-type]
    assert np.allclose(tr.starts, tr.roots - (T.START_DEPTH_MM / T.VOXEL_MM) * tr.normals, atol=1e-12)
    assert np.allclose(tr.offset, T.true_tau(geo, ts, tr.starts), atol=1e-12)  # type: ignore[arg-type]
    lo, hi = 1.0 / T.VOXEL_MM, np.array(geo.shape) - 1 - 1.0 / T.VOXEL_MM  # type: ignore[attr-defined]
    for X in (tr.starts, tr.stops, tr.paths.reshape(-1, 3)):
        X = X[~np.isnan(X[:, 0])]
        assert np.all((X >= lo - 1e-9) & (X <= hi + 1e-9))


# ---------------------------------------------------------------- stops


@pytest.mark.parametrize("name", NAMES)
def test_stops_lie_on_the_stop_surface(name: str) -> None:
    """|stop_sdf| ≤ 0.01 mm at every stop, and the family's closed form of the stop surface."""
    geo, ts, tr = case(name)
    Ys = geo.to_local(tr.stops)  # type: ignore[attr-defined]
    assert np.allclose(tr.stops, tr.paths[np.arange(tr.n), tr.counts - 1])
    assert np.abs(geo.stop_sdf(Ys)).max() <= STOP_TOL_MM  # type: ignore[attr-defined]
    x, y, z = Ys.T
    if name == "slab":
        assert np.abs(x - 2.4).max() <= STOP_TOL_MM
    elif name.startswith("crown"):
        r = np.linalg.norm(np.stack([x, y, np.maximum(z, 0)], 1), axis=1)
        assert np.abs(r - (0.64 + 2.4)).max() <= STOP_TOL_MM
    elif name in ("fundus_g0.04", "curved_g0.16"):
        g = 0.04 if name == "fundus_g0.04" else 0.16
        assert np.abs(np.hypot(x, np.minimum(y, 0)) - g).max() <= STOP_TOL_MM
    elif name.startswith("ckiss"):
        half = 2.5 if name == "ckiss_smooth" else 0.08
        assert np.abs(np.hypot(x, np.minimum(y, 0)) - ckiss_g(z, half)).max() <= STOP_TOL_MM
    elif name == "tau":
        t = 2.4 * (1 + 0.25 * np.sin(2 * np.pi * y / 6.0))
        assert np.abs(x - t).max() <= STOP_TOL_MM
    elif name == "abraded":
        ab = tr.stop_kind == T.StopKind.ABRADED
        r = np.hypot(x, np.minimum(y, 0))
        assert np.abs(r[~ab] - AB_G).max() <= STOP_TOL_MM
        on = np.zeros(ab.sum(), bool)
        for cy, depth in AB_DISCS:
            rho = np.hypot(y[ab] - cy, z[ab])
            on |= (np.abs(x[ab] - AB_G - depth) <= STOP_TOL_MM) & (rho <= AB_RADIUS + STOP_TOL_MM)
            on |= np.abs(rho - AB_RADIUS) <= STOP_TOL_MM
        assert on.all()


@pytest.mark.parametrize("name", NAMES)
def test_jumps(name: str) -> None:
    """jump = TAU_TOP − tau_arrival ≥ −1e-6 everywhere and |jump| < 1e-6 at every stop that is
    not ABRADED."""
    geo, ts, tr = case(name)
    assert np.allclose(tr.tau_arrival, T.true_tau(geo, ts, tr.stops), atol=1e-12)  # type: ignore[arg-type]
    assert np.allclose(tr.jump, T.TAU_TOP - tr.tau_arrival, atol=0)
    assert tr.jump.min() >= -JUMP_ZERO_TOL
    assert np.abs(tr.jump[tr.stop_kind != T.StopKind.ABRADED]).max() < JUMP_ZERO_TOL


# ---------------------------------------------------------------- paths


FD_BIAS = ("the central-difference tangent and the chord length have an O(h^2) bias where the "
           "true column bends strongly: near the thin slot's edge line (curvature about 1/r, "
           "ckiss_smooth) and in the one-voxel z transition (ckiss_sharp). The paths match a "
           "10x finer trace: test_paths_match_a_finer_trace")
PATH_CASES = [pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=FD_BIAS))
              if n in ("ckiss_smooth", "ckiss_sharp") else n for n in NAMES]


@pytest.mark.parametrize("name", PATH_CASES)
def test_path_steps_and_tangent(name: str) -> None:
    """The first point is the root (|φ| < 1e-9). Every full step has length step_mm (relative
    1e-3); the last step is not longer. At every interior point i (1 ≤ i ≤ count − 3) the central
    chord P[i+1] − P[i−1] makes an angle below 1e-3 rad with ∇φ(P[i])."""
    geo, ts, tr = case(name)
    P = local_paths(geo, tr)
    assert np.array_equal(tr.paths[:, 0], tr.roots)
    for i in range(tr.n):
        c = int(tr.counts[i])
        Q = P[i, :c]
        assert not np.isnan(Q).any() and (c == P.shape[1] or np.isnan(P[i, c:]).all())
        d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        assert np.abs(d[:-1] / tr.step_mm - 1).max() <= STEP_REL_TOL, (i, np.abs(d[:-1] / tr.step_mm - 1).max())
        assert d[-1] <= tr.step_mm * (1 + STEP_REL_TOL)
        if c >= 4:
            ch = Q[2:c - 1] - Q[: c - 3]
            ch /= np.linalg.norm(ch, axis=1)[:, None]
            G = geo.grad_phi(Q[1:c - 2])  # type: ignore[attr-defined]
            G /= np.linalg.norm(G, axis=1)[:, None]
            ang = np.arccos(np.clip(np.sum(ch * G, 1), -1, 1))
            assert ang.max() < TANGENT_TOL_RAD, (i, ang.max())


FINE_TOL_MM = 1e-4  # 1 % of the truth step


@pytest.mark.parametrize("name", NAMES)
def test_paths_match_a_finer_trace(name: str) -> None:
    """For up to 40 columns (every k-th), the same tracer at step_mm / 10 from the same root:
    every point of the true column is within 1e-4 mm of the fine polyline, and the two stops are
    within 1e-4 mm."""
    geo, ts, tr = case(name)
    P = local_paths(geo, tr)
    sel = np.arange(0, tr.n, max(1, tr.n // 40))
    F, c = FT.trace(geo, P[sel, 0], h=tr.step_mm / 10)
    for j, i in enumerate(sel):
        Q = P[i, : tr.counts[i]]
        Fi = F[j, : c[j]]
        assert np.linalg.norm(Fi[-1] - Q[-1]) <= FINE_TOL_MM
        assert BP._dist_to_polyline(Q, Fi).max() <= FINE_TOL_MM


@pytest.mark.parametrize("name", NAMES)
def test_paths_stay_in_tissue_and_bank(name: str) -> None:
    """(a) stop_sdf ≤ 1e-6 mm at every path point: no column enters the stopped region, so none
    crosses a slot or a contact. For the two-bank phantoms, every path point with local y ≥ 0
    (the flat banks, which the slot separates) is in the root's bank. At the round end (y < 0) the
    bank split θ = 0 is a convention, not a barrier: where tA ≠ tB the true columns cross it."""
    geo, ts, tr = case(name)
    P = local_paths(geo, tr)
    m = ~np.isnan(P[..., 0])
    Q = P[m]
    assert geo.stop_sdf(Q).max() <= IN_TISSUE_TOL_MM  # type: ignore[attr-defined]
    if name in TWO_BANKS:
        b = np.broadcast_to(tr.bank[:, None], m.shape)[m]
        flat = Q[:, 1] >= 0
        assert np.array_equal(geo.bank(Q[flat]), b[flat])  # type: ignore[attr-defined]
        assert set(np.unique(tr.bank)) == {0, 1}


def _triangles(R: np.ndarray, N: np.ndarray, s: float) -> np.ndarray:
    """Triangles of the roots R (local mm) with normals N: for each root, a 2D Delaunay of it and
    its 8 nearest roots, projected on its tangent plane; the triangles that hold it. Kept if every
    edge is below 2.5 s and the three normals agree pairwise (dot > 0.5)."""
    tree = cKDTree(R)
    _, nb = tree.query(R, k=min(9, len(R)))
    tris = set()
    for i in range(len(R)):
        n = N[i]
        a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        Q = R[nb[i]] - R[i]
        D = Delaunay(np.stack([Q @ a, Q @ b], 1))
        for t in D.simplices:
            if 0 in t:
                tris.add(tuple(sorted(int(nb[i][k]) for k in t)))
    T3 = np.array(sorted(tris), int)
    E = np.stack([np.linalg.norm(R[T3[:, j]] - R[T3[:, k]], axis=1) for j, k in ((0, 1), (1, 2), (0, 2))], 1)
    Nd = np.stack([np.sum(N[T3[:, j]] * N[T3[:, k]], 1) for j, k in ((0, 1), (1, 2), (0, 2))], 1)
    return T3[(E.max(1) < 2.5 * s) & (Nd.min(1) > 0.5)]


@pytest.mark.parametrize("name", NAMES)
def test_columns_do_not_cross(name: str) -> None:
    """(b) The exact rule. Triangulate the roots (`_triangles`). Sample each column at the φ
    levels 0, 0.01, ..., 1 (linear interpolation in φ along the path), but only at levels up to φ
    of its point count − 3, which is at least one full step (0.01 mm) before its stop. For each
    triangle (a, b, c) and each level that all three columns reach, D = det[b − a, c − a, g], with
    g the sum of the unit ∇φ at the three points. At φ = 0, D0 uses the sum of the root normals;
    triangles with |D0| < 1e-3 s² (s the root spacing) are dropped as degenerate. The test: D · sign(D0) > 0 at every such
    level. A crossing of two neighbouring columns flips or zeroes D."""
    geo, ts, tr = case(name)
    s = CASES[name]
    P = local_paths(geo, tr)
    R = P[:, 0]
    N = geo.grad_phi(R)  # type: ignore[attr-defined]
    N /= np.linalg.norm(N, axis=1)[:, None]
    tris = _triangles(R, N, s)
    a, b, c = (tris[:, k] for k in range(3))
    D0 = np.einsum("ij,ij->i", np.cross(R[b] - R[a], R[c] - R[a]), N[a] + N[b] + N[c])
    keep = np.abs(D0) >= 1e-3 * s * s
    tris, D0 = tris[keep], D0[keep]
    assert len(tris) > tr.n  # a surface of n points has about 2n triangles
    L = np.linspace(0.0, 1.0, 101)
    Qs = np.full((tr.n, len(L), 3), np.nan)
    for i in range(tr.n):
        cnt = int(tr.counts[i])
        ph = geo.phi(P[i, :cnt])  # type: ignore[attr-defined]
        assert np.all(np.diff(ph) > 0)
        ok = L <= ph[cnt - 3]
        for k in range(3):
            Qs[i, ok, k] = np.interp(L[ok], ph, P[i, :cnt, k])
    m = ~np.isnan(Qs[..., 0])
    G = np.zeros_like(Qs)
    Gm = geo.grad_phi(Qs[m])  # type: ignore[attr-defined]
    G[m] = Gm / np.linalg.norm(Gm, axis=1)[:, None]
    a, b, c = (tris[:, k] for k in range(3))
    D = np.einsum("tlj,tlj->tl", np.cross(Qs[b] - Qs[a], Qs[c] - Qs[a]), G[a] + G[b] + G[c])
    both = m[a] & m[b] & m[c]
    assert both[:, 1:].sum() > 10 * len(tris)  # levels above φ = 0 are checked
    bad = both & ~(D * np.sign(D0)[:, None] > 0)
    assert not bad.any(), (int(bad.sum()), tris[bad.any(1)][:5])


# ---------------------------------------------------------------- stop kinds


@pytest.mark.parametrize("name", ["slab", "crown_w0.64", "tau", "curved_g0.16"])
def test_open_phantoms_stop_at_the_pia(name: str) -> None:
    geo, ts, tr = case(name)
    Ys = geo.to_local(tr.stops)  # type: ignore[attr-defined]
    if name == "curved_g0.16":  # g = 0.16 is not below one voxel: PIA on the banks
        hair = Ys[:, 1] < 0
        assert np.all(tr.stop_kind[hair] == T.StopKind.HAIRPIN) and hair.any()
        assert np.all(tr.stop_kind[~hair] == T.StopKind.PIA) and (~hair).any()
    else:
        assert np.all(tr.stop_kind == T.StopKind.PIA)


@pytest.mark.parametrize("name", ["ckiss_smooth", "ckiss_sharp"])
def test_ckiss_stop_kinds(name: str) -> None:
    """CONTACT on the fused side (z > 2.5), PIA on the open side (z < −2.5), HAIRPIN at the round
    end (y < 0); each of them present for both banks."""
    geo, ts, tr = case(name)
    x, y, z = geo.to_local(tr.stops).T  # type: ignore[attr-defined]
    k = tr.stop_kind
    assert np.all((k == T.StopKind.HAIRPIN) == (y < 0))
    for bank in (0, 1):
        on = (tr.bank == bank) & (y >= 0)
        assert np.all(k[on & (z > 2.5)] == T.StopKind.CONTACT) and (on & (z > 2.5)).any()
        assert np.all(k[on & (z < -2.5)] == T.StopKind.PIA) and (on & (z < -2.5)).any()
    assert (k == T.StopKind.HAIRPIN).any()
    assert not (k == T.StopKind.ABRADED).any()


def test_subvoxel_fundus_stop_kinds() -> None:
    geo, ts, tr = case("fundus_g0.04")
    y = geo.to_local(tr.stops)[:, 1]  # type: ignore[attr-defined]
    assert np.all(tr.stop_kind[y < 0] == T.StopKind.HAIRPIN) and (y < 0).any()
    assert np.all(tr.stop_kind[y >= 0] == T.StopKind.CONTACT) and (y >= 0).any()


def test_abraded_stop_kinds_and_jumps() -> None:
    """ABRADED stops only on bank A, inside the discs, with the jump depth / tA · tau_scale
    (1e-3). Each disc has ABRADED stops. Every bank-A column whose root is inside a disc by
    0.05 mm stops ABRADED (the flat bank's columns run along x). All other jumps are 0."""
    geo, ts, tr = case("abraded")
    assert ts == AB_TS
    Ys = geo.to_local(tr.stops)  # type: ignore[attr-defined]
    Yr = geo.to_local(tr.roots)  # type: ignore[attr-defined]
    ab = tr.stop_kind == T.StopKind.ABRADED
    assert np.all(tr.bank[ab] == 0)
    seen = np.zeros(ab.sum(), bool)
    for cy, depth in AB_DISCS:
        inside = (np.hypot(Ys[ab, 1] - cy, Ys[ab, 2]) <= AB_RADIUS + 1e-6) & (Ys[ab, 0] > 0)
        assert inside.any()
        assert np.allclose(tr.jump[ab][inside], depth / AB_TA * AB_TS, atol=ABRADED_JUMP_TOL)
        seen |= inside
        deep = (tr.bank == 0) & (Yr[:, 1] >= 0) & (np.hypot(Yr[:, 1] - cy, Yr[:, 2]) < AB_RADIUS - 0.05)
        assert deep.any() and np.all(ab[deep])
    assert seen.all()
    assert np.all(tr.jump[ab] > 0)
    assert np.abs(tr.jump[~ab]).max() < JUMP_ZERO_TOL


# ---------------------------------------------------------------- determinism and files


def test_determinism() -> None:
    for name in ("curved_g0.16", "ckiss_sharp"):
        geo, ts = FT.geometry_of(name)
        a = FT.truth(geo, ts, name, root_spacing_mm=1.0)
        b = FT.truth(geo, ts, name, root_spacing_mm=1.0)
        for k in ("roots", "normals", "starts", "paths", "counts", "stops", "stop_kind", "bank",
                  "tau_arrival", "jump", "offset"):
            assert np.array_equal(getattr(a, k), getattr(b, k), equal_nan=k in ("paths",)), k


def test_npz_round_trip(tmp_path: Path) -> None:
    geo, ts, tr = case("abraded")
    tr.to_npz(tmp_path / "truth.npz")
    back = T.Truth.from_npz(tmp_path / "truth.npz")
    assert back.phantom == tr.phantom and back.tau_scale == tr.tau_scale
    assert back.step_mm == tr.step_mm and back.root_spacing_mm == tr.root_spacing_mm
    for k in ("roots", "normals", "starts", "paths", "counts", "stops", "stop_kind", "bank",
              "tau_arrival", "jump", "offset"):
        assert np.array_equal(getattr(back, k), getattr(tr, k), equal_nan=k == "paths"), k
        assert getattr(back, k).dtype == getattr(tr, k).dtype, k


def test_cli_writes_truth(tmp_path: Path) -> None:
    import json

    FT.main([str(tmp_path), "--names", "slab", "noise", "--spacing", "1.0"])
    assert not (tmp_path / "noise").exists()
    rec = json.loads((tmp_path / "slab" / "truth.json").read_text())
    tr = T.Truth.from_npz(tmp_path / "slab" / "truth.npz")
    assert rec["n_roots"] == tr.n and rec["stop_kinds"]["PIA"] == tr.n
