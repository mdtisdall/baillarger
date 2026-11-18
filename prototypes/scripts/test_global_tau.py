"""Tests of the global sparse τ solve (global_tau_*.py) on small synthetic
arrays: exact answers for the kernels, and two small slabs for the whole
loop (one bank with a label-0 top; two banks that meet with no label 0)."""

from __future__ import annotations

import global_tau_energy as GE
import global_tau_kernels as K
import global_tau_prep as TP
import global_tau_solver as GS
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import csgraph

H = 0.16


# ------------------------------------------------------------ helpers


def small_problem(shape=(6, 6, 7), faces_z=None, seed=0):
    """A box domain, all unknowns, with random τ and inputs; optional contact
    faces between z = faces_z and faces_z + 1."""
    rng = np.random.default_rng(seed)
    dom = np.ones(shape, bool)
    G = GS.Grid(dom)
    S = G.S
    faces = np.zeros((0, 3), np.int64)
    if faces_z is not None:
        c = G.coords - GS.PAD
        m = c[:, 2] == faces_z
        sm = np.flatnonzero(m)
        sp = G.slot_of[G.slot_flat[sm] + G.strides[2]]
        faces = np.c_[sm, sp, np.full(len(sm), 2)]
    nbr, nbv, gh = K.neighbours(G.slot_flat, G.slot_of, G.strides, faces)
    S1 = GS._pow2(S + 1)

    def pad(a, fill):
        out = np.full((S1, *a.shape[1:]), fill, dtype=a.dtype)
        out[: len(a)] = a
        return out

    nb_p = pad(nbr[:S], S)
    nbv_p = pad(nbv[:S], False)
    gh_p = pad(gh[:S], False)
    nh = rng.normal(size=(S, 3))
    nh /= np.linalg.norm(nh, axis=1, keepdims=True)
    T, Q = 4, 4
    P = {
        "nb": nb_p,
        "nbv": nbv_p,
        "ghost": gh_p,
        "h": np.full(3, H),
        "htau": H,
        "valid": np.all((nbv_p | gh_p).any(2), 1).astype(float),
        "c": pad(rng.choice([-GE.LOG_ODDS, GE.LOG_ODDS], S), 0.0),
        "coh": pad(rng.uniform(0, 1, S), 0.0),
        "nhat": pad(nh, 0.0),
        "inner": pad(np.zeros(S), 1.0),
        "wq": pad(np.full(S, 1.4), 0.0),
        "evalw": pad(np.ones(S), 0.0),
        "rslot": np.full((T, 8), S),
        "rw": np.zeros((T, 8)),
        "hasroot": np.zeros(T),
        "logmed": np.zeros(T),
        "pi": np.zeros(Q, np.int64),
        "pj": np.zeros(Q, np.int64),
        "psd": np.ones(Q),
        "pm": np.zeros(Q),
    }
    P = {k: jnp.asarray(v) for k, v in P.items()}
    c = G.coords - GS.PAD
    x = np.zeros(S1)
    x[:S] = -(shape[2] - c[:, 2]) * H + rng.normal(scale=0.03, size=S)
    unk = np.zeros(S1, bool)
    unk[:S] = True
    return G, P, x, unk, faces


def flat_site(
    two_banks: bool, nx: int = 14, nz: int = 26
) -> tuple[TP.Site, np.ndarray]:
    """A synthetic slab site (voxel 0.16 mm). One bank: WM z <= 4, cortex
    5..15, label 0 above. Two banks: WM at both ends, cortex between with no
    label 0. White surface(s): flat grids half a voxel outside the WM."""
    lab = np.full((nx, nx, nz), TP.OUTSIDE, np.int16)
    vox = np.full(3, H)
    lab[:, :, :5] = TP.WM
    if two_banks:
        lab[:, :, 5 : nz - 5] = TP.CORTEX
        lab[:, :, nz - 5 :] = TP.WM
    else:
        lab[:, :, 5:16] = TP.CORTEX

    def sheet(z0: float, up: bool) -> tuple[np.ndarray, np.ndarray]:
        g = np.arange(-1.0, nx + 0.01, 0.5)
        X, Y = np.meshgrid(g, g, indexing="ij")
        v = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z0)]
        n = len(g)
        idx = np.arange(n * n).reshape(n, n)
        a, b, c, d = (
            idx[:-1, :-1].ravel(),
            idx[1:, :-1].ravel(),
            idx[:-1, 1:].ravel(),
            idx[1:, 1:].ravel(),
        )
        f = np.r_[np.c_[a, b, d], np.c_[a, d, c]]
        if not up:
            f = f[:, ::-1]
        return v, f

    v, f = sheet(4.5, True)
    if two_banks:
        v2, f2 = sheet(nz - 5.5, False)
        f = np.r_[f, f2 + len(v)]
        v = np.r_[v, v2]
    n, area = TP.vertex_normals_areas(v * vox, f)
    z = np.arange(nz)[None, None, :] * np.ones((nx, nx, 1))
    d1 = (z - 4.5) * H
    dist = d1.copy()
    inside = lab == TP.WM
    if two_banks:
        d2 = (nz - 5.5 - z) * H
        dist = np.where(z < nz / 2, d1, d2)
    nhat = np.zeros((nx, nx, nz, 3), np.float32)
    nhat[..., 2] = 1.0
    coh = np.where(lab == TP.CORTEX, 1.0, 0.0).astype(np.float32)
    echoes = np.ones((3, nx, nx, nz), np.float32)
    site = TP.Site(
        echoes,
        lab,
        np.diag([H, H, H, 1.0]),
        vox,
        v,
        f,
        n,
        area,
        np.zeros(len(v)),
        dist,
        inside,
        nhat,
        coh,
    )
    return site, np.array([nx / 2, nx / 2, nz / 2])


# ------------------------------------------------------------ kernels


def test_colouring_separates_stencils() -> None:
    m, a, b = GS.COLOURING
    ds = [
        (i, j, k)
        for i in range(-4, 5)
        for j in range(-4, 5)
        for k in range(-4, 5)
        if 0 < abs(i) + abs(j) + abs(k) <= 4
    ]
    assert all((i + a * j + b * k) % m != 0 for i, j, k in ds)
    assert m == 27  # the minimum for L1 <= 4 is at least 25 (the L1 ball of radius 2)


def test_same_bank_csr() -> None:
    ptr = np.array([0, 2, 4, 5])
    idx = np.array([0, 1, 0, 1, 2])
    assert K.same_bank(0, 1, ptr, idx) and K.same_bank(1, 0, ptr, idx)
    assert not K.same_bank(0, 2, ptr, idx) and not K.same_bank(2, 1, ptr, idx)
    assert K.same_bank(-1, 2, ptr, idx) and K.same_bank(2, 2, ptr, idx)


def test_ghost_gradient_at_contact() -> None:
    G, P, x, unk, faces = small_problem(faces_z=3)
    g = np.asarray(GE.gradient(jnp.asarray(x), P))
    c = G.coords - GS.PAD
    for s in np.flatnonzero(c[:, 2] == 3)[:5]:  # minus side: (0 - τ)/(h/2) ...
        m = G.slot_of[G.slot_flat[s] - G.strides[2]]
        assert g[s, 2] == pytest.approx((0.0 - x[m]) / (1.5 * H))
    for s in np.flatnonzero((c[:, 2] == 4) & (c[:, 0] == 2)):  # plus side
        p = G.slot_of[G.slot_flat[s] + G.strides[2]]
        assert g[s, 2] == pytest.approx((x[p] - 0.0) / (1.5 * H))
    # a slot at the face with no other z neighbour: one-sided to the ghost
    G, P, x, unk, faces = small_problem(shape=(4, 4, 2), faces_z=0)
    g = np.asarray(GE.gradient(jnp.asarray(x), P))
    assert g[0, 2] == pytest.approx((0.0 - x[0]) / (0.5 * H))


def test_stencil_jacobian_matches_dense() -> None:
    G, P, x, unk, faces = small_problem(faces_z=3)
    xj = jnp.asarray(x)
    g0 = GE.j_g0(xj, P)
    fw, _ = GE.j_weights(xj, P)
    col = K.stencil_cols(G.slot_flat, G.slot_of, G.strides, K.OFFS)
    Jst = GS.stencil_jacobian(xj, P, g0, fw, G.colour, G.ncolour, col, unk, G.S)
    Jd = np.asarray(jax.jacfwd(lambda y: GE.voxel_residuals(y, P, g0, fw))(xj))[
        : G.S, :, : G.S
    ]  # (S, NRES, S)
    v = np.random.default_rng(1).normal(size=G.S)
    r = K.jst_apply(Jst, col, v)
    assert np.allclose(r, np.einsum("skt,t->sk", Jd, v), atol=1e-9)
    u = np.random.default_rng(2).normal(size=(G.S, GE.NRES))
    assert np.allclose(
        K.jst_apply_t(Jst, col, u, G.S), np.einsum("skt,sk->t", Jd, u), atol=1e-9
    )
    A = np.einsum("skt,skq->tq", Jd, Jd)
    assert np.allclose(K.jst_diag(Jst, col, G.S), np.diag(A), atol=1e-9)
    olut = -np.ones((5, 5, 5), np.int64)
    for o, (a, b, c) in enumerate(K.OFFS):
        olut[a + 2, b + 2, c + 2] = o
    for i, j in [(0, 1), (10, 10 + 7), (30, 30 + 49), (40, 41)]:
        e = K._entry(
            i, j, Jst, col, G.slot_flat, G.slot_of, G.strides, K.OFFS, olut, G.coords
        )
        assert e == pytest.approx(A[i, j], abs=1e-9)


def test_line_preconditioner_is_exact_on_the_lines() -> None:
    G, P, x, unk, faces = small_problem(shape=(3, 3, 8))
    xj = jnp.asarray(x)
    g0 = GE.j_g0(xj, P)
    fw, _ = GE.j_weights(xj, P)
    col = K.stencil_cols(G.slot_flat, G.slot_of, G.strides, K.OFFS)
    Jst = GS.stencil_jacobian(xj, P, g0, fw, G.colour, G.ncolour, col, unk, G.S)
    Jd = np.asarray(jax.jacfwd(lambda y: GE.voxel_residuals(y, P, g0, fw))(xj))[
        : G.S, :, : G.S
    ]
    A = np.einsum("skt,skq->tq", Jd, Jd) + 0.1 * np.eye(G.S)
    ghat = np.zeros((G.S, 3))
    ghat[:, 2] = 1.0
    nbr = np.asarray(P["nb"])[: G.S + 1].copy()
    nbr[nbr >= G.S] = G.S
    nbv = np.asarray(P["nbv"])[: G.S + 1].copy()
    nbv[G.S] = False
    ptr, sl = K.column_lines(nbr, nbv, ghat, 32)
    lines = [sl[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]
    assert len(lines) == 9 and all(len(li) == 8 for li in lines)  # the z columns
    c = G.coords
    for li in lines:
        assert np.all(c[li, 0] == c[li[0], 0]) and np.all(np.diff(c[li, 2]) == 1)
    olut = -np.ones((5, 5, 5), np.int64)
    for o, (a, b, cc) in enumerate(K.OFFS):
        olut[a + 2, b + 2, cc + 2] = o
    B, bptr = K.line_blocks(
        ptr,
        sl,
        Jst,
        col,
        G.slot_flat,
        G.slot_of,
        G.strides,
        K.OFFS,
        olut,
        G.coords,
        np.full(G.S, 0.1),
    )
    r = np.random.default_rng(3).normal(size=G.S)
    z = K.line_solve(ptr, sl, B, bptr, r)
    for li in lines:
        assert np.allclose(z[li], np.linalg.solve(A[np.ix_(li, li)], r[li]), atol=1e-8)


def test_contact_faces_and_upwind_owner() -> None:
    # two owners in different banks on the two halves of a box along z
    G = GS.Grid(np.ones((3, 3, 8), bool))
    c = G.coords - GS.PAD
    owner = np.where(c[:, 2] < 4, 0, 1).astype(np.int64)
    ptr = np.array([0, 1, 2])
    idx = np.array([0, 1])
    f = K.contact_faces(G.slot_flat, G.slot_of, G.strides, owner, ptr, idx)
    assert len(f) == 9 and np.all(f[:, 2] == 2) and np.all(c[f[:, 0], 2] == 3)
    # one bank, two patches: τ rises with z; the owner of z = 0 spreads up
    ptr = np.array([0, 2, 4])
    idx = np.array([0, 1, 0, 1])
    tau = c[:, 2] * H - 1.0
    ghat = np.zeros((G.S, 3))
    ghat[:, 2] = 1.0
    own0 = np.where(c[:, 2] == 0, 0, 1).astype(np.int64)
    fixed = c[:, 2] == 0
    own = K.upwind_owner(
        np.argsort(tau),
        G.slot_flat,
        G.slot_of,
        G.strides,
        K.O26,
        np.full(3, H),
        tau,
        ghat,
        own0,
        fixed,
        ptr,
        idx,
    )
    assert np.all(own == 0)
    # different banks: no owner crosses
    ptr = np.array([0, 1, 2])
    idx = np.array([0, 1])
    own = K.upwind_owner(
        np.argsort(tau),
        G.slot_flat,
        G.slot_of,
        G.strides,
        K.O26,
        np.full(3, H),
        tau,
        ghat,
        own0,
        fixed,
        ptr,
        idx,
    )
    assert np.all(own == own0)


def test_march_metric_is_anisotropic() -> None:
    G = GS.Grid(np.ones((15, 15, 3), bool))
    c = G.coords - GS.PAD
    nhat = np.zeros((G.S, 3))
    nhat[:, 0] = 1.0
    coh = np.ones(G.S)
    s0 = int(np.flatnonzero((c[:, 0] == 7) & (c[:, 1] == 7) & (c[:, 2] == 1))[0])
    A = GS.march_graph(
        G, np.full(3, H), nhat, coh, 8.0, np.array([s0]), np.array([0.0])
    )
    d = csgraph.dijkstra(A, directed=False, indices=[G.S])[0]
    sx = int(np.flatnonzero((c[:, 0] == 12) & (c[:, 1] == 7) & (c[:, 2] == 1))[0])
    sy = int(np.flatnonzero((c[:, 0] == 7) & (c[:, 1] == 12) & (c[:, 2] == 1))[0])
    assert d[sx] == pytest.approx(5 * H)
    assert d[sy] == pytest.approx(3 * 5 * H)  # sqrt(1 + 8) times
    A0 = GS.march_graph(
        G, np.full(3, H), nhat, np.zeros(G.S), 8.0, np.array([s0]), np.array([0.0])
    )
    d0 = csgraph.dijkstra(A0, directed=False, indices=[G.S])[0]
    assert d0[sy] == pytest.approx(5 * H)


# ------------------------------------------------------------ whole loop


def test_one_bank_slab() -> None:
    site, cv = flat_site(two_banks=False)
    pt = TP.make_patches(site)
    s = GS.Settings(outer_iters=1, lm_iters=8, face_test=True)
    gt = GS.GlobalTau(site, pt, s, cv)
    assert len(gt.faces) == 0 and np.all(gt.owner >= 0)
    gt.run()
    c = gt.G.coords - GS.PAD
    mid = (c[:, 0] >= 4) & (c[:, 0] < 10) & (c[:, 1] >= 4) & (c[:, 1] < 10)
    top = mid & (c[:, 2] == 15)
    above = mid & (c[:, 2] == 16)
    assert np.all(gt.tau[top] < 0) and np.all(
        gt.tau[above] > 0
    )  # the top stays at the label face
    tz = [np.median(gt.tau[mid & (c[:, 2] == z)]) for z in range(6, 15)]
    assert np.all(np.diff(tz) > 0)
    assert np.allclose(np.diff(tz), H, rtol=0.25)  # τ ≈ mm
    assert gt.log.outer[-1]["faces_after"] == 0 and not gt.log.events["topology"]


def test_two_banks_meet_at_a_contact() -> None:
    nx = 6
    site, cv = flat_site(two_banks=True, nx=nx)
    pt = TP.make_patches(site)
    s = GS.Settings(outer_iters=1, lm_iters=8, face_test=True, face_range=1)
    gt = GS.GlobalTau(site, pt, s, cv)
    c = gt.G.coords - GS.PAD
    zf = c[gt.faces[:, 0], 2]
    assert (
        len(gt.faces) == nx * nx and np.all(gt.faces[:, 2] == 2) and np.all(zf == zf[0])
    )  # one flat contact
    # a move of +2 gives side A two layers of side B's voxels in each column
    units, nfb = gt.face_units(gt.owner, gt.faces)
    assert nfb == 0 and sum(len(u["a"]) for u in units) == nx * nx
    un = units[0]
    ra, rb = int(gt.owner[un["a"][0]]), int(gt.owner[un["b"][0]])
    moved = gt._moved_owner(gt.owner, un, 2, ra, rb)
    ch = np.flatnonzero(moved != gt.owner)
    assert len(ch) == 2 * len(un["a"]) and all(gt.same(int(o), ra) for o in moved[ch])
    gt.run()
    rec = gt.log.outer[0]
    assert rec["faces_before"] == nx * nx
    f = gt.faces
    assert np.all(gt.tau[f[:, 0]] < 0) and np.all(
        gt.tau[f[:, 1]] < 0
    )  # both sides are tissue up to the face
    assert len(gt.face_records) == len(units) and all(
        r["move"] in (-1, 0, 1) for r in gt.face_records
    )


def test_snapshots_round_trip(tmp_path) -> None:  # noqa: ANN001
    import global_tau_movie as GM

    site, cv = flat_site(two_banks=True, nx=6)
    pt = TP.make_patches(site)
    s = GS.Settings(outer_iters=1, lm_iters=3, face_test=False)
    gt = GS.GlobalTau(site, pt, s, cv)
    gt.snap_dir = tmp_path
    gt.run()
    snaps = sorted(tmp_path.glob("*.npz"))
    names = [p.stem.split("-", 1)[1] for p in snaps]
    assert names[0] == "outer0-lm00" and names[-1] == "outer0-owner"
    assert len(names) == len(gt.log.outer) + 1 + gt.log.outer[0]["lm_iters"]
    tau, own, faces, meta = GM.load_snap(snaps[-2], site.labels.shape)
    c = gt.G.coords - GS.PAD
    assert np.allclose(tau[tuple(c.T)], gt.tau, atol=1e-6)
    assert np.isnan(tau[~gt.domain]).all() and len(faces) == 6 * 6
    assert meta["tag"] == "outer0" and meta["lm_iter"] == len(snaps) - 2
