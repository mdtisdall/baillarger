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
        "ridge": np.float64(0.0),
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
    two_banks: bool, nx: int = 14, nz: int = 26, gap: int = 0
) -> tuple[TP.Site, np.ndarray]:
    """A synthetic slab site (voxel 0.16 mm). One sheet: WM z <= 4, cortex
    5..15, label 0 above. Two sheets: WM at both ends, cortex between with no
    label 0, or with gap layers of label 0 in the middle. White surface(s):
    flat grids half a voxel outside the WM."""
    lab = np.full((nx, nx, nz), TP.OUTSIDE, np.int16)
    vox = np.full(3, H)
    lab[:, :, :5] = TP.WM
    if two_banks:
        lab[:, :, 5 : nz - 5] = TP.CORTEX
        lab[:, :, nz - 5 :] = TP.WM
        if gap:
            z0 = nz // 2 - gap // 2
            lab[:, :, z0 : z0 + gap] = TP.OUTSIDE
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


def _ridge_P(P):  # noqa: ANN001, ANN202
    Q = dict(P)
    Q["ridge"] = jnp.asarray(1.0)
    return Q


@pytest.mark.parametrize("ridge", [False, True])
def test_stencil_jacobian_matches_dense(ridge: bool) -> None:
    G, P, x, unk, faces = small_problem(faces_z=None if ridge else 3)
    if ridge:  # the ridge form: upwind split, div v weighted by f
        x[: G.S] += 0.5
        P = _ridge_P(P)
        assert (x[: G.S] > 0).any() and (x[: G.S] < 0).any()
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


def test_energy_gradient_is_twice_jt_r() -> None:
    """In tissue (f = 1) with no stop term, the voxel energy is Σ r², so
    ∇E = 2 Jᵀr and its Hessian ≈ 2 JᵀJ: the Gauss-Newton system of lm_solve
    is (JᵀJ + λD) dx = -∇E / 2 (an LM step of -∇E is twice the step)."""
    G, P, x, unk, _ = small_problem()
    x[: G.S] -= H  # f = 1 everywhere
    P = dict(P)
    P["c"] = jnp.zeros_like(P["c"])
    xj = jnp.asarray(x)
    g0 = GE.j_g0(xj, P)
    fw, _ = GE.j_weights(xj, P)
    assert np.all(np.asarray(fw)[: G.S] == 1.0)
    r = np.asarray(GE.voxel_residuals(xj, P, g0, fw))
    e = float(GE.energy(xj, P, g0))
    assert e == pytest.approx(float((r**2).sum()), rel=1e-8)  # EPS in |∇τ|
    col = K.stencil_cols(G.slot_flat, G.slot_of, G.strides, K.OFFS)
    Jst = GS.stencil_jacobian(xj, P, g0, fw, G.colour, G.ncolour, col, unk, G.S)
    grad = np.asarray(jax.grad(GE.energy)(xj, P, g0))[: G.S]
    np.testing.assert_allclose(
        grad, 2 * K.jst_apply_t(Jst, col, r[: G.S], G.S), rtol=1e-7, atol=1e-7
    )


def test_upwind_gradient_sees_the_checkerboard() -> None:
    G, P, x, unk, _ = small_problem(shape=(6, 6, 8))
    c = G.coords - GS.PAD
    lin = np.zeros_like(x)
    lin[: G.S] = c[:, 2] * H - 2.0
    chk = np.zeros_like(x)
    chk[: G.S] = 0.01 * (-1.0) ** c.sum(1)
    inner = np.all((c > 0) & (c < np.array([5, 5, 7])), 1)
    gl = np.asarray(GE.gradient(jnp.asarray(lin), P))[: G.S]
    gc = np.asarray(GE.gradient(jnp.asarray(lin + chk), P))[: G.S]
    assert np.allclose(gc[inner], gl[inner])  # central differences: blind to it
    Pu = _ridge_P(P)
    gu = np.asarray(GE.gradient(jnp.asarray(lin + chk), Pu))[: G.S]
    assert not np.allclose(gu[inner], gl[inner])
    s = int(np.flatnonzero(inner)[0])
    m = G.slot_of[G.slot_flat[s] - G.strides[2]]
    t = lin + chk
    assert gu[s, 2] == pytest.approx((t[s] - t[m]) / H)


@pytest.mark.parametrize(
    "peak,cls", [(0.05, "contact"), (-0.3, "below"), (H, "beyond")]
)
def test_ridge_classes(peak: float, cls: str) -> None:
    # τ is a tent along z with its peak on the face between z = 4 and z = 5
    G = GS.Grid(np.ones((3, 3, 10), bool))
    c = G.coords - GS.PAD
    nbr, nbv, _ = K.neighbours(
        G.slot_flat, G.slot_of, G.strides, np.zeros((0, 3), np.int64)
    )
    x = np.zeros(G.S + 1)
    x[: G.S] = peak - np.abs(c[:, 2] - 4.5) * H
    up = GS.upwind_stencil(x, nbr, nbv)
    g = np.zeros((G.S, 3))
    g[:, 2] = np.where(c[:, 2] <= 4, 1.0, -1.0)  # the upwind gradient of the tent
    r = GS.ridge_faces(x, nbr, nbv, up, G.S, g)
    assert len(r[cls]) == 9 and sum(len(v) for v in r.values()) == 9
    assert np.all(c[r[cls][:, 0], 2] == 4) and np.all(r[cls][:, 2] == 2)


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


def test_ridge_one_sheet_slab() -> None:
    site, cv = flat_site(two_banks=False, nx=6)
    pt = TP.make_patches(site)
    s = GS.Settings(outer_iters=1, lm_iters=8, contacts="ridge")
    gt = GS.GlobalTau(site, pt, s, cv)
    assert len(gt.faces) == 0
    gt.run()
    c = gt.G.coords - GS.PAD
    assert np.all(gt.tau[c[:, 2] == 15] < 0) and np.all(gt.tau[c[:, 2] == 16] > 0)
    rec = gt.log.outer[0]
    assert rec["ridge_faces"] == {"contact": 0, "below": 0, "beyond": 0}
    assert rec["owner_changes_upwind_bank"] is None and rec["face_patches"] == 0


def test_ridge_energy_is_continuous_and_rim_fixed() -> None:
    # τ is a tent along z with its peak on the voxel plane z = 3 (exact ties of
    # the two neighbours): the ridge-form energy changes by O(ε) under a step ε
    G, P, x, unk, _ = small_problem(shape=(5, 5, 8))
    c = G.coords - GS.PAD
    x[: G.S] = -0.1 - np.abs(c[:, 2] - 3) * H
    P = _ridge_P(P)
    g0 = GE.j_g0(jnp.asarray(x), P)
    e = float(GE.j_energy(jnp.asarray(x), P, g0))
    d = np.zeros_like(x)
    d[: G.S] = np.random.default_rng(0).normal(size=G.S)
    de = [
        abs(float(GE.j_energy(jnp.asarray(x + eps * d), P, g0)) - e)
        for eps in (1e-9, 1e-15)
    ]
    # continuous, but very steep: at the ridge plane the split gives ∇τ = 0,
    # and v = ∇τ / (|∇τ|² + EPS) has a spike of height ~1/(2 √EPS) there
    # (site 1, 2026-11-18: a step of 1e-5 mm raises E by 8e10 in this tent)
    assert de[0] > 0 and de[1] < 1e-3 * de[0]
    # the rim voxels are not unknowns in the ridge form
    site, cv = flat_site(two_banks=False, nx=6)
    pt = TP.make_patches(site)
    gt = GS.GlobalTau(site, pt, GS.Settings(contacts="ridge"), cv)
    pb = gt.global_problem(gt.owner)
    valid = np.asarray(pb.P["valid"])[: pb.S] > 0
    assert np.array_equal(pb.unk[: pb.S], valid)  # the slab has no rim voxel


def test_ridge_two_sheets_with_medium_between() -> None:
    # two sheets with 2 layers of label 0 between them: a ridge of τ in the
    # medium (beyond the top), no contact
    site, cv = flat_site(two_banks=True, nx=6, gap=2)
    pt = TP.make_patches(site)
    s = GS.Settings(outer_iters=1, lm_iters=8, contacts="ridge")
    gt = GS.GlobalTau(site, pt, s, cv)
    gt.run()
    c = gt.G.coords - GS.PAD
    assert np.all(gt.tau[c[:, 2] == 11] < 0) and np.all(gt.tau[c[:, 2] == 14] < 0)
    assert np.all(gt.tau[(c[:, 2] == 12) | (c[:, 2] == 13)] > 0)
    r = gt.log.outer[0]["ridge_faces"]
    assert r["contact"] == 0 and r["beyond"] == 36
    assert np.all(c[gt.ridges["beyond"][:, 0], 2] == 12)


def test_ridge_seeded_tau0(tmp_path) -> None:  # noqa: ANN001
    # two sheets with no label 0: the midplane faces as τ0 seeds give
    # τ0 = -(distance to the midplane), -h/2 next to it on both sides
    nx = 6
    site, cv = flat_site(two_banks=True, nx=nx)
    pt = TP.make_patches(site)
    X, Y = np.meshgrid(np.arange(nx), np.arange(nx), indexing="ij")
    f = np.c_[X.ravel(), Y.ravel(), np.full(nx * nx, 12), np.full(nx * nx, 2)]
    np.savez(tmp_path / "seed.npz", faces=f)
    s = GS.Settings(contacts="ridge", tau0_faces=str(tmp_path / "seed.npz"))
    gt = GS.GlobalTau(site, pt, s, cv)
    c = gt.G.coords - GS.PAD
    assert gt.n_seed_faces == nx * nx and len(gt.faces) == 0
    for z, d in [(12, 0.5), (13, 0.5), (10, 2.5), (15, 2.5)]:
        assert np.allclose(gt.tau[c[:, 2] == z], -d * H)


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


# ------------------------------------------------------------ lagged quadratic form


def _lagged_problem():  # noqa: ANN202
    G, P, x, unk, _ = small_problem()
    x[: G.S] -= H  # f = 1 everywhere, as inside a fixed Σ
    P = dict(P)
    P["c"] = jnp.zeros_like(P["c"])
    return G, P, x, unk


def test_lagged_residuals_equal_the_original_at_the_lag_point() -> None:
    """At x0 the lagged align, equi and pull residuals are the original ones
    (pull with ĝ0 = ĝ(x0)); smooth uses the derivative of ∇τ, not of ĝ, so
    it is only close."""
    G, P, x, unk = _lagged_problem()
    xj = jnp.asarray(x)
    g0 = GE.j_g0(xj, P)
    fw, _ = GE.j_weights(xj, P)
    r = np.asarray(GE.voxel_residuals(xj, P, g0, fw))[: G.S]
    L = GE.j_lag_state(xj, P)
    rl = np.asarray(GE.j_lagged_residuals(xj, P, L))[: G.S]
    np.testing.assert_allclose(rl[:, :3], r[:, :3], atol=1e-9)  # align
    np.testing.assert_allclose(rl[:, 12], r[:, 12], rtol=1e-7, atol=1e-7)  # equi
    np.testing.assert_allclose(rl[:, 13], r[:, 13], atol=1e-9)  # pull
    sm, sml = (r[:, 3:12] ** 2).sum(), (rl[:, 3:12] ** 2).sum()
    assert abs(sml - sm) < 0.2 * sm


def test_lagged_residuals_are_affine() -> None:
    G, P, x, unk = _lagged_problem()
    xj = jnp.asarray(x)
    L = GE.j_lag_state(xj, P)
    t = np.zeros_like(x)
    t[: G.S] = np.random.default_rng(3).normal(size=G.S)
    r = [np.asarray(GE.j_lagged_residuals(jnp.asarray(x + a * t), P, L)) for a in (0, 1, 2)]
    np.testing.assert_allclose(r[2] - 2 * r[1] + r[0], 0.0, atol=1e-6)
    jt = np.asarray(GE.j_lagged_jvp(xj, P, L, jnp.asarray(t)))
    np.testing.assert_allclose(r[1] - r[0], jt, rtol=1e-9, atol=1e-6)


def test_lagged_stencil_jacobian_matches_dense() -> None:
    G, P, x, unk = _lagged_problem()
    xj = jnp.asarray(x)
    L = GE.j_lag_state(xj, P)
    col = K.stencil_cols(G.slot_flat, G.slot_of, G.strides, K.OFFS)
    Jst = GS.stencil_jacobian(
        xj, P, None, None, G.colour, G.ncolour, col, unk, G.S,
        jvp=lambda t: GE.j_lagged_jvp(xj, P, L, t),
    )
    assert Jst.shape[2] == GE.NRES_L
    Jd = np.asarray(jax.jacfwd(lambda y: GE.lagged_voxel_residuals(y, P, L))(xj))[
        : G.S, :, : G.S
    ]
    v = np.random.default_rng(4).normal(size=G.S)
    np.testing.assert_allclose(
        K.jst_apply(Jst, col, v), np.einsum("skt,t->sk", Jd, v), atol=1e-9
    )


def test_lagged_patch_terms_majorise_at_the_lag_point() -> None:
    """t / t0 is the tangent of log t (equal at t0, above elsewhere), and the
    IRLS weight makes tw0 d² / 2 the tangent quadratic of the Student-t."""
    t0, t = 2.0, np.array([0.5, 1.0, 2.0, 3.0, 6.0])
    tang = np.log(t0) + t / t0 - 1.0
    assert np.all(tang >= np.log(t) - 1e-12)
    psd, d0 = 0.3, 0.4
    st = lambda d: (GE.NU + 1) / 2 * np.log1p(d**2 / (GE.NU * psd**2))  # noqa: E731
    tw = (GE.NU + 1) / (GE.NU * psd**2 + d0**2)
    d = np.linspace(-2, 2, 41)
    maj = st(d0) + tw / 2 * (d**2 - d0**2)
    assert np.all(maj >= st(d) - 1e-12)
