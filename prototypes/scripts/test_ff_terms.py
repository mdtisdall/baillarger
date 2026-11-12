"""Tests of ff_terms: each term on the smallest grid that holds it against values
computed by hand, the messages against finite differences of the expected energy, the
expected energy against an exact enumeration of the joint states, the jump, the
candidates, the update of u and the switch of the τ prior.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_terms.py
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy import optimize, sparse

sys.path.insert(0, str(Path(__file__).parent))
import ff_terms as FT  # noqa: E402
import ff_types as T  # noqa: E402

V = T.VOXEL_MM
SA = math.sin(math.radians(15.0))


# ---------------------------------------------------------------- builders


def uniform_ga(dims, h_vox=1.0, rng=None, q_tau=False):
    """A GridArrays of a full box of dims nodes at spacing h_vox voxels. With rng:
    random positive weights; Q_u (and Q_tau) SPD, coupling only nodes of one cell
    (different colours)."""
    ni, nj, nk = dims
    idx = np.indices(dims).reshape(3, -1).T
    n = len(idx)
    nodes = idx * h_vox
    nbr6 = np.full((n, 6), -1, np.int64)
    for c in range(6):
        a, s = c // 2, (1 if c % 2 == 0 else -1)
        o = idx.copy()
        o[:, a] += s
        ok = (o[:, a] >= 0) & (o[:, a] < dims[a])
        nbr6[ok, c] = np.ravel_multi_index(o[ok].T, dims)
    edges = np.array(
        sorted(
            {
                (i, int(nbr6[i, c]))
                for i in range(n)
                for c in (0, 2, 4)
                if nbr6[i, c] >= 0
            }
        ),
        np.int64,
    ).reshape(-1, 2)
    pairs = set()
    for i in range(n):
        for off in itertools.product((0, 1), repeat=3):
            if sum(off) < 2:
                continue
            o = idx[i] + np.array(off)
            if (o < np.array(dims)).all():
                pairs.add((i, int(np.ravel_multi_index(o, dims))))
        for off in (
            (1, -1, 0),
            (1, 0, -1),
            (0, 1, -1),
            (1, 1, -1),
            (1, -1, 1),
            (-1, 1, 1),
        ):
            o = idx[i] + np.array(off)
            if (o >= 0).all() and (o < np.array(dims)).all():
                j = int(np.ravel_multi_index(o, dims))
                pairs.add((min(i, j), max(i, j)))
    dir_pairs = np.array(sorted(pairs), np.int64).reshape(-1, 2)
    E, P = len(edges), len(dir_pairs)
    if rng is None:
        node_w, edge_w, pair_w = np.ones(n), np.full(E, h_vox**3 / 3), np.full(P, 0.1)
    else:
        node_w = rng.uniform(0.5, 1.5, n)
        edge_w = rng.uniform(0.2, 0.6, E)
        pair_w = rng.uniform(0.05, 0.3, P)
    vec = (nodes[edges[:, 1]] - nodes[edges[:, 0]]) * V
    face_w = (np.linalg.norm(vec, axis=1) / V) ** 2

    def spd(wp, d):
        Q = sparse.coo_array(
            (
                np.concatenate([wp, wp, -wp, -wp]),
                (
                    np.concatenate(
                        [
                            dir_pairs[:, 0],
                            dir_pairs[:, 1],
                            dir_pairs[:, 0],
                            dir_pairs[:, 1],
                        ]
                    ),
                    np.concatenate(
                        [
                            dir_pairs[:, 0],
                            dir_pairs[:, 1],
                            dir_pairs[:, 1],
                            dir_pairs[:, 0],
                        ]
                    ),
                ),
            ),
            shape=(n, n),
        )
        return (Q + sparse.diags_array(d)).tocsr()

    r = rng if rng is not None else np.random.default_rng(0)
    Q_u = spd(r.uniform(0.5, 1.5, P), r.uniform(0.5, 1.0, n))
    Q_t = spd(r.uniform(0.5, 1.5, P), r.uniform(0.5, 1.0, n)) if q_tau else None
    return FT.GridArrays(
        nodes=nodes.astype(float),
        nbr6=nbr6,
        edges=edges,
        edge_vec_mm=vec,
        node_w=node_w,
        edge_w=edge_w,
        face_w=face_w,
        dir_pairs=dir_pairs,
        dir_pair_w=pair_w,
        Q_u=Q_u,
        Q_tau=Q_t,
    )


def two_node_ga(pair_w=0.5, Q_u=None, Q_tau=None):
    nbr6 = np.full((2, 6), -1, np.int64)
    nbr6[0, 0] = 1
    nbr6[1, 1] = 0
    return FT.GridArrays(
        nodes=np.array([[0.0, 0, 0], [1.0, 0, 0]]),
        nbr6=nbr6,
        edges=np.array([[0, 1]]),
        edge_vec_mm=np.array([[V, 0.0, 0.0]]),
        node_w=np.array([1.0, 2.0]),
        edge_w=np.array([0.7]),
        face_w=np.array([1.0]),
        dir_pairs=np.array([[0, 1]]),
        dir_pair_w=np.array([pair_w]),
        Q_u=Q_u if Q_u is not None else sparse.csr_array(np.zeros((2, 2))),
        Q_tau=Q_tau,
    )


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def random_state(n, K, L, rng, invalid=0.0, stop_frac=0.3):
    tau = -rng.uniform(0.0, 1.0, (n, K))
    d = unit(rng.normal(size=(n, K, 3)) + np.array([1.0, 0.5, 0.2]))
    valid = rng.uniform(size=(n, K)) >= invalid
    valid[:, 0] = True
    q = rng.uniform(0.1, 1.0, (n, K)) * valid
    qs = rng.uniform(0.0, 1.0, n) * stop_frac
    z = q.sum(1) + qs
    return FT.TermState(
        cand_tau=np.where(valid, tau, 0.5),
        cand_dir=d,
        valid=valid,
        q=q / z[:, None],
        q_stop=qs / z,
        u=rng.normal(0.0, 0.3, n),
        s_ref=rng.uniform(0.6, 1.5, n),
        profiles=rng.normal(size=(n, K, L, 2)),
        level_base=rng.integers(0, L + 1, n),
        stop_log_odds=rng.normal(0.0, 2.0, n),
        tensor_dir=unit(rng.normal(size=(n, 3))),
        coherence=rng.uniform(0.0, 1.0, n),
    )


def weights(w=1.0, tau_prior=False, sigma_c=0.25):
    return FT.TermWeights(
        sigma_c=sigma_c,
        sigma_e=(0.3, 0.5),
        sigma_align=SA,
        lambda_dir=1.0 / (SA * SA * 0.5),
        tau_prior=tau_prior,
        w=w,
    )


def one_state(n, K, L=3):
    """A state with K = 1 deterministic candidate per node (for the hand tests)."""
    return FT.TermState(
        cand_tau=np.zeros((n, K)),
        cand_dir=np.tile([1.0, 0, 0], (n, K, 1)),
        valid=np.ones((n, K), bool),
        q=np.ones((n, K)) / K,
        q_stop=np.zeros(n),
        u=np.zeros(n),
        s_ref=np.ones(n),
        profiles=np.zeros((n, K, L, 2)),
        level_base=np.zeros(n, np.int64),
        stop_log_odds=np.zeros(n),
        tensor_dir=np.tile([1.0, 0, 0], (n, 1)),
        coherence=np.zeros(n),
    )


def coupling(e):
    return sum(e[k] for k in FT.COUPLING)


# ---------------------------------------------------------------- 1. terms by hand


def test_unary_by_hand():
    ga = two_node_ga()
    st = one_state(2, 1)
    st = replace(
        st,
        stop_log_odds=np.array([1.5, -0.5]),
        coherence=np.array([0.8, 0.0]),
        tensor_dir=np.array([[0.0, 1.0, 0.0], [1.0, 0, 0]]),
        cand_dir=np.array([[[math.cos(0.3), math.sin(0.3), 0.0]], [[1.0, 0, 0]]]),
    )
    tw = weights()
    em, es = FT.unary(st, ga, tw)
    sp = lambda x: math.log1p(math.exp(x))  # noqa: E731
    assert es[0] == pytest.approx(sp(-1.5))
    assert es[1] == pytest.approx(2.0 * sp(0.5))
    align0 = 0.8 * (1 - math.sin(0.3) ** 2) / (2 * SA**2)
    assert em[0, 0] == pytest.approx(sp(1.5) + align0)
    assert em[1, 0] == pytest.approx(2.0 * sp(-0.5))
    e = FT.state_energy(st, ga, tw, np.array([0, -1]))
    assert e["stop"][0] == pytest.approx(sp(1.5) + 2.0 * sp(0.5))
    assert e["align"][0] == pytest.approx(align0)


def test_contrast_by_hand():
    ga = two_node_ga()
    st = one_state(2, 1, L=3)
    P = np.zeros((2, 1, 3, 2))
    P[0, 0, :, 0] = [1.0, 2.0, 3.0]
    P[0, 0, :, 1] = [0.0, 1.0, 0.0]
    P[1, 0, :, 0] = [5.0, 1.0, 7.0]  # levels 1, 2, 3 (level_base 1)
    P[1, 0, :, 1] = [2.0, 2.0, 2.0]
    st = replace(st, profiles=P, level_base=np.array([0, 1]))
    for w in (1.0, 2.0):
        tw = weights(w=w)
        # shared levels 1, 2: node 0 slots 1, 2 against node 1 slots 0, 1
        s = ((2 - 5) ** 2 + (3 - 1) ** 2) / 0.3**2 + (
            (1 - 2) ** 2 + (0 - 2) ** 2
        ) / 0.5**2
        hand = 0.7 / (2 * w * w * 2) * s
        e = FT.state_energy(st, ga, tw, np.array([0, 0]))
        assert e["contrast"][0] == pytest.approx(hand)
        assert FT.expected_energy(st, ga, tw)["contrast"] == pytest.approx(hand)
        assert FT.state_energy(st, ga, tw, np.array([0, -1]))["contrast"][0] == 0.0
    st2 = replace(st, level_base=np.array([0, 3]))  # no shared level: no term
    assert FT.state_energy(st2, ga, weights(), np.array([0, 0]))["contrast"][0] == 0.0


def test_bending_by_hand():
    ga = two_node_ga(pair_w=0.5)
    st = one_state(2, 1)
    st = replace(st, cand_dir=np.array([[[1.0, 0, 0]], [[0.0, 1.0, 0]]]))
    tw = weights(w=2.0)
    hand = tw.lambda_dir / (2 * 4.0) * 0.5 * 2.0
    assert FT.state_energy(st, ga, tw, np.array([0, 0]))["bending"][0] == pytest.approx(
        hand
    )
    assert FT.state_energy(st, ga, tw, np.array([0, -1]))["bending"][0] == 0.0
    assert tw.lambda_dir == pytest.approx(1 / (math.sin(math.radians(15)) ** 2 * 0.5))


def test_speed_and_tau_prior_by_hand():
    Q = sparse.csr_array(np.array([[2.0, -0.5], [-0.5, 1.0]]))
    ga = two_node_ga(Q_u=Q, Q_tau=Q)
    st = replace(
        one_state(2, 1), u=np.array([0.3, -0.2]), cand_tau=np.array([[-0.4], [-0.1]])
    )
    tw = weights(w=2.0, tau_prior=True)
    u = st.u
    speed = 0.5 * (2 * u[0] ** 2 - 2 * 0.5 * u[0] * u[1] + u[1] ** 2) / 4.0
    tp = 0.5 * (2 * 0.16 - 2 * 0.5 * 0.04 + 0.01) / 4.0
    e = FT.state_energy(st, ga, tw, np.array([0, 0]))
    assert e["speed"][0] == pytest.approx(speed)
    assert e["tau_prior"][0] == pytest.approx(tp)
    # node 1 stopped: only node 0's diagonal term
    e = FT.state_energy(st, ga, tw, np.array([0, -1]))
    assert e["tau_prior"][0] == pytest.approx(0.5 * 2 * 0.16 / 4.0)
    assert FT.expected_energy(st, ga, tw)["speed"] == pytest.approx(speed)


def test_jump_by_hand_two_nodes():
    ga = two_node_ga()
    st = replace(
        one_state(2, 1), cand_tau=np.array([[-0.3], [-0.1]]), s_ref=np.array([1.0, 2.0])
    )
    tw = weights()
    # node 0 moving toward node 1 (stopped): reach = -0.3 + 1.0 * 0.16 = -0.14
    e = FT.state_energy(st, ga, tw, np.array([0, -1]))
    assert e["jump"][0] == pytest.approx(-math.log(0.001) + math.log(0.5) + 0.14 / 0.5)
    # node 1 moving (d = +x, away from node 0): reach = τ = -0.1, J = 0.1; face_w 1
    e = FT.state_energy(st, ga, tw, np.array([-1, 0]))
    assert e["jump"][0] == pytest.approx(-math.log(0.001) + math.log(0.5) + 0.1 / 0.5)
    assert FT.state_energy(st, ga, tw, np.array([0, 0]))["jump"][0] == 0.0
    assert FT.state_energy(st, ga, tw, np.array([-1, -1]))["jump"][0] == 0.0


def test_consistency_by_hand():
    ga = uniform_ga((2, 2, 2), h_vox=2.0)  # h = 0.32 mm; each node has one octant
    n = 8
    st = one_state(n, 1)
    x = ga.nodes * V
    grad = np.array([0.5, -0.25, 1.0])
    d = unit([1.0, 0.2, 0.4])
    u = np.linspace(-0.2, 0.3, n)
    st = replace(
        st, cand_tau=(-2.0 + x @ grad)[:, None], cand_dir=np.tile(d, (n, 1, 1)), u=u
    )
    for w in (1.0, 4.0):
        tw = weights(w=w, sigma_c=0.04 / 0.32)
        hand = sum(
            1.0 / (2 * tw.sigma_c**2 * w * w) * (d @ grad - math.exp(-u[i])) ** 2
            for i in range(n)
        )
        e = FT.state_energy(st, ga, tw, np.zeros(n, np.int64))
        assert e["consistency"][0] == pytest.approx(hand)
        assert FT.expected_energy(st, ga, tw)["consistency"] == pytest.approx(hand)
    # one node stopped: the 4 octants that hold it vanish (its own and those of its 3
    # neighbours along the axes)
    ch = np.zeros(n, np.int64)
    ch[0] = -1
    tw = weights(sigma_c=0.04 / 0.32)
    rest = [i for i in range(n) if i not in (0, 1, 2, 4)]
    hand = sum(
        1.0 / (2 * tw.sigma_c**2) * (d @ grad - math.exp(-u[i])) ** 2 for i in rest
    )
    assert FT.state_energy(st, ga, tw, ch)["consistency"][0] == pytest.approx(hand)


# ---------------------------------------------------------------- 2. messages = ∂E/∂q


@pytest.mark.parametrize("seed", [0, 1])
def test_messages_finite_differences(seed):
    rng = np.random.default_rng(seed)
    ga = uniform_ga((3, 3, 3), rng=rng, q_tau=True)
    st = random_state(ga.n, 4, 3, rng, invalid=0.2)
    tw = weights(w=2.0, tau_prior=True)
    M, Ms = FT.messages(st, ga, tw, np.arange(ga.n))
    eps = 1e-4
    for i in range(ga.n):
        for k in [*np.nonzero(st.valid[i])[0], -1]:
            vals = []
            for sgn in (1, -1):
                if k >= 0:
                    q = st.q.copy()
                    q[i, k] += sgn * eps
                    s2 = replace(st, q=q)
                else:
                    qs = st.q_stop.copy()
                    qs[i] += sgn * eps
                    s2 = replace(st, q_stop=qs)
                vals.append(coupling(FT.expected_energy(s2, ga, tw)))
            fd = (vals[0] - vals[1]) / (2 * eps)
            m = M[i, k] if k >= 0 else Ms[i]
            assert m == pytest.approx(fd, rel=1e-6, abs=1e-6 * max(1.0, abs(fd))), (
                i,
                k,
            )
    # the unary terms: their derivative is `unary`
    em, es = FT.unary(st, ga, tw)
    q = st.q.copy()
    q[5, 0] += eps
    e1 = FT.expected_energy(replace(st, q=q), ga, tw)
    e0 = FT.expected_energy(st, ga, tw)
    du = sum(e1[k] - e0[k] for k in FT.UNARY) / eps
    assert du == pytest.approx(em[5, 0], rel=1e-6)


def test_messages_chunks_and_subsets():
    rng = np.random.default_rng(3)
    ga = uniform_ga((3, 3, 3), rng=rng)
    st = random_state(ga.n, 5, 4, rng, invalid=0.1)
    tw = weights(w=4.0)
    M, Ms = FT.messages(st, ga, tw, np.arange(ga.n))
    sel = np.array([26, 3, 13, 0, 7])
    M2, Ms2 = FT.messages(st, ga, tw, sel, mom=FT.node_moments(st), chunk=2)
    np.testing.assert_allclose(M2, M[sel], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Ms2, Ms[sel], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------- 3. enumeration


@pytest.mark.parametrize("tau_prior", [False, True])
def test_expected_energy_by_enumeration(tau_prior):
    rng = np.random.default_rng(7)
    ga = uniform_ga((2, 2, 2), rng=rng, q_tau=True)
    st = random_state(ga.n, 3, 3, rng, invalid=0.25)
    tw = weights(w=2.0, tau_prior=tau_prior)
    states = [[*np.nonzero(st.valid[i])[0], -1] for i in range(ga.n)]
    ch = np.array(list(itertools.product(*states)), np.int64)  # (B, n)
    rows = np.arange(ga.n)[None, :]
    prob = np.where(ch >= 0, st.q[rows, np.maximum(ch, 0)], st.q_stop[rows]).prod(1)
    assert prob.sum() == pytest.approx(1.0)
    se = FT.state_energy(st, ga, tw, ch)
    ee = FT.expected_energy(st, ga, tw)
    for k in (*FT.COUPLING, *FT.UNARY, "speed"):
        assert ee[k] == pytest.approx(float(prob @ se[k]), rel=1e-10, abs=1e-12), k


def test_one_hot_matches_state_energy():
    rng = np.random.default_rng(11)
    ga = uniform_ga((3, 3, 2), rng=rng, q_tau=True)
    st = random_state(ga.n, 4, 3, rng)
    tw = weights(tau_prior=True)
    ch = rng.integers(-1, 4, ga.n)
    q = np.zeros_like(st.q)
    q[np.arange(ga.n)[ch >= 0], ch[ch >= 0]] = 1.0
    s1 = replace(st, q=q, q_stop=(ch < 0).astype(float), valid=np.ones_like(st.valid))
    ee = FT.expected_energy(s1, ga, tw)
    se = FT.state_energy(s1, ga, tw, ch)
    for k in ee:
        assert ee[k] == pytest.approx(float(se[k][0]), rel=1e-10, abs=1e-12), k


# ---------------------------------------------------------------- 4. the jump


def test_jump_energy():
    p0, mean = 0.999, 0.5
    d = np.array([1.0, 0.0, 0.0])
    delta = np.array([0.16, 0.0, 0.0])
    # the boundary lies within the edge: reach = -0.1 + 0.16 >= 0
    E, J = FT.jump_energy(np.array(-0.1), d, np.array(1.0), delta)
    assert J == 0.0 and E == pytest.approx(-math.log(p0))
    E, J = FT.jump_energy(
        np.array(-0.16), d, np.array(1.0), delta
    )  # exactly at the far end
    assert J == 0.0
    E, J = FT.jump_energy(np.array(-0.3), d, np.array(1.0), delta)
    assert J == pytest.approx(0.14)
    assert E == pytest.approx(-math.log(1 - p0) - math.log(1 / mean) + 0.14 / mean)
    # pointing away from the stopped node: reach = τ
    E, J = FT.jump_energy(np.array(-0.05), -d, np.array(1.0), delta)
    assert J == pytest.approx(0.05)
    # slowness scales the reach
    E, J = FT.jump_energy(
        np.array(-0.3), d, np.array(2.0), delta
    )  # reach = -0.3 + 0.32
    assert J == 0.0
    # never negative
    _, J = FT.jump_energy(np.linspace(-1, 0, 11), d, np.array(1.0), delta)
    assert (J >= 0).all()


def test_tau_above_top_is_not_a_state():
    s = T.DEFAULTS
    tau, _, valid = FT.moving_candidates(
        np.array([-0.05, 0.3, -1.0]), np.array([[1.0, 0, 0]] * 3), s
    )
    a = np.arange(63) // 7
    assert (valid[0] == (-0.05 + (a - 4) * 0.04 <= 0)).all()
    assert (valid[1] == (a <= 4)).all() and (tau[1, valid[1]] <= 0).all()
    assert tau[1, 4 * 7] == 0.0  # c = min(τ_pred, TAU_TOP)
    assert valid[2].all()
    assert (tau[valid] <= T.TAU_TOP).all()
    # rounding at the top: -0.08 + 2 Δτ is the top, and valid
    tau, _, valid = FT.moving_candidates(
        np.array([-0.08]), np.array([[0.0, 0, 1.0]]), s
    )
    assert valid[0, 6 * 7] and tau[0, 6 * 7] == 0.0 and not valid[0, 7 * 7]


# ---------------------------------------------------------------- 5. the candidates


def test_moving_candidates():
    rng = np.random.default_rng(5)
    p = unit(rng.normal(size=(20, 3)))
    tp = -rng.uniform(0.2, 2.0, 20)
    tau, d, valid = FT.moving_candidates(tp, p)
    assert tau.shape == (20, 63) and d.shape == (20, 63, 3) and valid.shape == (20, 63)
    np.testing.assert_allclose(np.linalg.norm(d, axis=2), 1.0, atol=1e-14)
    for a in range(9):
        for m in range(7):
            c = a * 7 + m
            np.testing.assert_allclose(tau[:, c], tp + (a - 4) * 0.04, atol=1e-15)
            np.testing.assert_allclose(d[:, c], d[:, m], atol=0)
    np.testing.assert_allclose(d[:, 0], p, atol=1e-15)
    ang = np.degrees(np.arccos(np.clip(np.einsum("nmc,nc->nm", d[:, 1:7], p), -1, 1)))
    np.testing.assert_allclose(ang, 15.0, atol=1e-9)
    # azimuths 0, 60, ..., 300 in the basis u, w (e: the axis least aligned with p)
    for i in range(20):
        e = np.zeros(3)
        e[np.argmin(np.abs(p[i]))] = 1.0
        u = unit(np.cross(p[i], e))
        w = np.cross(p[i], u)
        az = np.arctan2(d[i, 1:7] @ w, d[i, 1:7] @ u)
        want = np.radians([0, 60, 120, 180, 240, 300])
        np.testing.assert_allclose(np.cos(az - want), 1.0, atol=1e-12)
    t2, d2, v2 = FT.moving_candidates(tp.copy(), p.copy())
    assert (
        np.array_equal(t2, tau) and np.array_equal(d2, d) and np.array_equal(v2, valid)
    )


def test_level_base():
    lb = FT.level_base(np.array([-1.0, -0.02, 0.5, -0.06]))
    assert lb.tolist() == [-25 - 4, 0 - 4, 0 - 4, -2 - 4]


# ---------------------------------------------------------------- profiles


def test_candidate_profiles_linear_volume():
    shape = (20, 20, 20)
    X = np.indices(shape).astype(float)
    vols = [X[0] + 2 * X[1] - X[2], 0.5 * X[2]]
    nodes = np.array([[10.0, 10, 10], [8.0, 11, 9]])
    tau = np.array([[-0.3, -0.25], [-0.1, -0.12]])
    d = np.array([[[1.0, 0, 0], unit([1.0, 1, 0])], [[0, 0, 1.0], unit([0, 1, 1.0])]])
    s_ref = np.array([1.0, 1.5])
    lb = np.array([-10, -4])
    P = FT.candidate_profiles(nodes, tau, d, s_ref, lb, vols, L=5, dtau=0.04)
    for i in range(2):
        for k in range(2):
            for ell in range(5):
                dist = ((lb[i] + ell) * 0.04 - tau[i, k]) / (s_ref[i] * V)
                x = nodes[i] + d[i, k] * dist
                assert P[i, k, ell, 0] == pytest.approx(x[0] + 2 * x[1] - x[2])
                assert P[i, k, ell, 1] == pytest.approx(0.5 * x[2])


def test_contrast_sigma_and_smoothing():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((64, 64, 64))
    v1, v2 = FT.smoothed_echoes({1: noise, 2: 3.0 * noise}, 2.0)
    f = FT.contrast_sigma([1.0, 3.0], 2.0)
    core = (slice(8, -8),) * 3
    assert v1[core].std() == pytest.approx(f[0], rel=0.03)
    assert v2[core].std() == pytest.approx(f[1], rel=0.03)
    s = 2.0 / 2.3548
    assert np.allclose(v1, ndi.gaussian_filter(noise, s, mode="nearest"), atol=1e-3)
    tw = FT.term_weights(0.16, [0.1, 0.2], 4.0)
    assert tw.sigma_c == pytest.approx(0.25) and tw.sigma_align == pytest.approx(SA)
    assert tw.sigma_e == pytest.approx((0.1 * f[0], 0.2 * f[0]))


# ---------------------------------------------------------------- 6. update of u


def _u_energy(st, ga, tw):
    e = FT.expected_energy(st, ga, tw)
    return e["consistency"] + e["speed"]


def test_update_u_never_raises():
    rng = np.random.default_rng(2)
    ga = uniform_ga((3, 3, 3), rng=rng)
    st = random_state(ga.n, 4, 3, rng, stop_frac=0.1)
    tw = weights(w=1.0)
    colour = (np.indices((3, 3, 3)).reshape(3, -1) % 2).T @ np.array([1, 2, 4])
    for _sweep in range(2):
        for c in range(8):
            sel = np.nonzero(colour == c)[0]
            e0 = _u_energy(st, ga, tw)
            u = st.u.copy()
            u[sel] = FT.update_u(st, ga, tw, sel)
            st = replace(st, u=u)
            assert _u_energy(st, ga, tw) <= e0 + 1e-12 * abs(e0)
    # the result is the minimum of the 1D energy over a fine grid of the search range
    sel = np.nonzero(colour == 0)[0]
    a, b, A, B = FT.u_energy_coeffs(st, ga, tw, sel)
    f = lambda u: a * u * u + b * u + A * np.exp(-2 * u) - 2 * B * np.exp(-u)  # noqa: E731
    un = FT.update_u(st, ga, tw, sel)
    fine = np.linspace(*FT.U_RANGE, 240001)[:, None]
    assert (f(un) <= f(fine).min(0) + 1e-12).all()


def test_update_u_by_hand():
    ga = ga_with(uniform_ga((2, 2, 2), h_vox=1.0), Q_u=sparse.csr_array((8, 8)))
    st = one_state(8, 1)
    g = 3.90625  # d · ∇τ in τ per mm (∇τ along +x; d = +x)
    tau = -1.0 + g * ga.nodes[:, 0] * V
    st = replace(st, cand_tau=tau[:, None], cand_dir=np.tile([1.0, 0, 0], (8, 1, 1)))
    tw = weights()
    u = FT.update_u(st, ga, tw, np.array([0]))
    assert u[0] == pytest.approx(-math.log(g), abs=1e-9)  # s = d · ∇τ exactly
    # with the prior: E = (Q / 2) u² + A e^(-2u) - 2 B e^(-u), A = coef, B = coef g
    Q = 3.0
    ga2 = ga_with(ga, Q_u=sparse.csr_array(np.diag(np.full(8, Q))))
    coef = 1.0 / (2 * tw.sigma_c**2)
    A, B = coef, coef * g
    ustar = optimize.brentq(
        lambda v: Q * v - 2 * A * math.exp(-2 * v) + 2 * B * math.exp(-v),
        -10,
        10,
        xtol=1e-14,
    )
    u = FT.update_u(st, ga2, tw, np.array([0]))
    assert u[0] == pytest.approx(ustar, abs=1e-6)
    # neighbours stopped: no octant term, the prior alone; minimum -b / (2a)
    Qm = np.diag(np.full(8, Q))
    Qm[0, 4] = Qm[4, 0] = -1.0
    ga3 = ga_with(ga, Q_u=sparse.csr_array(Qm))
    st3 = replace(
        st,
        q_stop=np.r_[0.0, np.ones(7)],
        q=np.r_[[[1.0]], np.zeros((7, 1))],
        u=np.full(8, 0.7),
    )
    a, b, A0, B0 = FT.u_energy_coeffs(st3, ga3, tw, np.array([0]))
    assert A0[0] == 0.0 and B0[0] == 0.0
    assert FT.update_u(st3, ga3, tw, np.array([0]))[0] == pytest.approx(
        0.7 / Q, abs=1e-12
    )


def ga_with(ga, **kw):
    names = (
        "nodes",
        "nbr6",
        "edges",
        "edge_vec_mm",
        "node_w",
        "edge_w",
        "face_w",
        "dir_pairs",
        "dir_pair_w",
        "Q_u",
        "Q_tau",
    )
    return FT.GridArrays(**{f: kw.get(f, getattr(ga, f)) for f in names})


# ---------------------------------------------------------------- 7. the τ prior switch


def test_tau_prior_switch():
    rng = np.random.default_rng(9)
    ga = uniform_ga((3, 3, 2), rng=rng, q_tau=True)
    st = random_state(ga.n, 4, 3, rng)
    off = weights(tau_prior=False)
    on = weights(tau_prior=True)
    e_off = FT.expected_energy(st, ga, off)
    e_on = FT.expected_energy(st, ga, on)
    assert e_off["tau_prior"] == 0.0 and e_on["tau_prior"] != 0.0
    for k in e_off:
        if k != "tau_prior":
            assert e_on[k] == pytest.approx(e_off[k], rel=1e-14)
    sel = np.arange(ga.n)
    M0, S0 = FT.messages(st, ga, off, sel)
    M1, S1 = FT.messages(st, ga, on, sel)
    np.testing.assert_array_equal(S0, S1)
    # the difference is the τ prior's own message: Q_ii τ² / 2 + τ Σ_j≠i Q_ij E[1 τ_j]
    Q = ga.Q_tau.toarray()
    s1 = (st.q * st.cand_tau).sum(1)
    off_d = Q @ s1 - np.diag(Q) * s1
    hand = np.diag(Q)[:, None] * st.cand_tau**2 / 2 + st.cand_tau * off_d[:, None]
    np.testing.assert_allclose(
        (M1 - M0)[st.valid], hand[st.valid], rtol=1e-10, atol=1e-12
    )
    # Q_tau None with the switch on: nothing
    assert FT.expected_energy(st, ga_with(ga, Q_tau=None), on)["tau_prior"] == 0.0


# ---------------------------------------------------------------- chunks


@pytest.mark.parametrize("tau_prior", [False, True])
def test_expected_energy_chunks_match_one_chunk(tau_prior):
    """The node and edge chunks of expected_energy give the sums of one chunk, to
    float64 rounding; the moments given by the caller give the same values."""
    rng = np.random.default_rng(13)
    ga = uniform_ga((4, 3, 3), rng=rng, q_tau=True)
    st = random_state(ga.n, 5, 4, rng, invalid=0.2)
    tw = weights(w=2.0, tau_prior=tau_prior)
    big = FT.expected_energy(st, ga, tw, chunk=10**6)
    for chunk in (1, 5, 7):
        small = FT.expected_energy(st, ga, tw, chunk=chunk)
        given = FT.expected_energy(st, ga, tw, chunk=chunk, mom=FT.node_moments(st))
        assert small.keys() == big.keys() == given.keys()
        for k in big:
            assert small[k] == pytest.approx(big[k], rel=1e-12, abs=1e-12), k
            assert given[k] == small[k], k


def test_unary_rows_match_all_nodes():
    rng = np.random.default_rng(17)
    ga = uniform_ga((3, 3, 2), rng=rng)
    st = random_state(ga.n, 4, 3, rng)
    tw = weights()
    em, es = FT.unary(st, ga, tw)
    ids = np.array([5, 0, 17, 3])
    em2, es2 = FT.unary(st, ga, tw, ids)
    assert np.array_equal(em2, em[ids]) and np.array_equal(es2, es[ids])


def test_update_u_and_moments_chunks_give_the_same_bits():
    rng = np.random.default_rng(19)
    ga = uniform_ga((3, 3, 3), rng=rng)
    st = random_state(ga.n, 4, 3, rng)
    tw = weights(w=2.0)
    sel = np.array([0, 2, 6, 8, 18, 20, 24, 26])  # colour 0
    mom = FT.node_moments(st)
    u1 = FT.update_u(st, ga, tw, sel, mom, chunk=10**6)
    u3 = FT.update_u(st, ga, tw, sel, mom, chunk=3)
    assert u1.tobytes() == u3.tobytes()
    m1 = FT.node_moments(st, chunk=10**6)
    m3 = FT.node_moments(st)
    FT.refresh_moments(m3, st, np.arange(ga.n)[::-1].copy(), chunk=4)
    for k in ("p", "s1", "s2", "dsum", "m1", "m2"):
        assert getattr(m1, k).tobytes() == getattr(m3, k).tobytes(), k


# ---------------------------------------------------------------- the profile lines
# (CD-7) against the reference form of 63 profiles of 9 levels

# The reference: the profile computation and the contrast energy and messages of the
# form before CD-7, copied here without change of the arithmetic. Each candidate holds
# its own profile of L levels.


def _ref_candidate_profiles(nodes, cand_tau, cand_dir, s_ref, level_base, volumes, L,
                            dtau):  # fmt: skip
    """(n, K, L, 2): the volumes sampled at x_i + d (ℓ Δτ − τ) / (s_ref_i · VOXEL_MM),
    ℓ = level_base_i + 0 .. L − 1 (the form before CD-7)."""
    n, K = cand_tau.shape
    lev = np.arange(L)
    ell = (level_base[:, None] + lev[None, :]) * dtau
    dist = (ell[:, None, :] - cand_tau[:, :, None]) / (s_ref[:, None, None] * V)
    X = nodes[:, None, None, :] + cand_dir[:, :, None, :] * dist[..., None]
    X = np.where(np.isfinite(X), X, np.broadcast_to(nodes[:, None, None, :], X.shape))
    Xf = X.reshape(-1, 3)
    out = np.empty((n, K, L, len(volumes)))
    for e, vol in enumerate(volumes):
        out[:, :, :, e] = T.sample_volume(vol, Xf).reshape(n, K, L)
    return out


def _ref_moments(valid, q, prof):
    qv = np.where(valid, q, 0.0)
    P = np.where(valid[..., None, None], prof, 0.0)
    return (
        qv.sum(1),
        np.einsum("nk,nkle->nle", qv, P),
        np.einsum("nk,nkle->nle", qv, P * P),
    )


def _ref_shift(arr, lb_i, lb_j, L):
    idx = np.arange(L)[None, :] + (lb_i - lb_j)[:, None]
    mask = (idx >= 0) & (idx < L)
    out = np.take_along_axis(arr, np.clip(idx, 0, L - 1)[:, :, None], axis=1)
    return out * mask[:, :, None], mask


def _ref_contrast_energy(ga, tw, valid, q, prof, lb):
    """The expected contrast energy of the form before CD-7."""
    p, m1, m2 = _ref_moments(valid, q, prof)
    L = prof.shape[2]
    i, j = ga.edges[:, 0], ga.edges[:, 1]
    m1j, mask = _ref_shift(m1[j], lb[i], lb[j], L)
    m2j, _ = _ref_shift(m2[j], lb[i], lb[j], L)
    nsh = mask.sum(1)
    W = np.where(nsh > 0, ga.edge_w / (2.0 * tw.w**2 * np.maximum(nsh, 1)), 0.0)
    inv_se = 1.0 / np.asarray(tw.sigma_e, float) ** 2
    Wl = W[:, None, None] * mask[:, :, None] * inv_se[None, None, :]
    val = p[j][:, None, None] * m2[i] - 2.0 * m1[i] * m1j + p[i][:, None, None] * m2j
    return float((Wl * val).sum())


def _ref_contrast_messages(ga, tw, valid, q, prof, lb):
    """(n, K): the contrast messages of the form before CD-7."""
    p, m1, m2 = _ref_moments(valid, q, prof)
    n, K, L, _ = prof.shape
    P = np.where(valid[..., None, None], prof, 0.0)
    inv_se = 1.0 / np.asarray(tw.sigma_e, float) ** 2
    out = np.zeros((n, K))
    for k in range(ga.inc_edge.shape[1]):
        e = ga.inc_edge[:, k]
        has = np.nonzero(e >= 0)[0]
        e = e[has]
        j = ga.inc_edge_nbr[has, k]
        m1j, mask = _ref_shift(m1[j], lb[has], lb[j], L)
        m2j, _ = _ref_shift(m2[j], lb[has], lb[j], L)
        nsh = mask.sum(1)
        W = np.where(nsh > 0, ga.edge_w[e] / (2.0 * tw.w**2 * np.maximum(nsh, 1)), 0.0)
        Wl = W[:, None, None] * mask[:, :, None] * inv_se[None, None, :]
        out[has] += (
            np.einsum("skle,sle->sk", P[has] ** 2, Wl * p[j][:, None, None])
            - 2.0 * np.einsum("skle,sle->sk", P[has], Wl * m1j)
            + (Wl * m2j).sum((1, 2))[:, None]
        )
    return out


# The tolerance of the comparison, fixed before the first run: the two forms differ only
# by the rounding of the sample positions (ff_terms.profile_lines), so the profiles, the
# contrast energy and the contrast messages agree to 1e-9 of their scale.
LINE_REL = 1e-9


def test_profile_lines_match_the_reference_form():
    rng = np.random.default_rng(29)
    ga0 = uniform_ga((4, 4, 3), rng=rng)
    ga = ga_with(ga0, nodes=ga0.nodes + 6.0)  # inside a 16³ volume
    n = ga.n
    vols = [
        ndi.gaussian_filter(rng.standard_normal((16, 16, 16)), 1.5) * 10.0
        for _ in range(2)
    ]
    tau_pred = -rng.uniform(0.05, 1.0, n)
    tau_pred[0] = -0.08  # a candidate at the top (the rounding rule of the top)
    tau_pred[1] = -0.4  # on the level lattice
    tau_pred[2] = 0.03  # above the top: c = TAU_TOP
    dir_pred = unit(rng.normal(size=(n, 3)))
    s = T.DEFAULTS
    cand_tau, cand_dir, valid = FT.moving_candidates(tau_pred, dir_pred, s)
    s_ref = rng.uniform(0.6, 1.5, n)
    lb = FT.level_base(tau_pred, s)
    ref = _ref_candidate_profiles(
        ga.nodes, cand_tau, cand_dir, s_ref, lb, vols, FT.N_LEVELS, s.dtau
    )
    c = np.minimum(tau_pred, T.TAU_TOP)
    lines = FT.profile_lines(ga.nodes, c, cand_dir[:, : s.n_dir], s_ref, lb, vols)
    assert lines.shape == (n, s.n_dir, FT.N_LEVELS + s.n_tau - 1, 2)
    line, off = FT.line_slots(s.n_tau, s.n_dir)
    K = cand_tau.shape[1]
    q = rng.uniform(0.1, 1.0, (n, K)) * valid
    qs = rng.uniform(0.0, 0.3, n)
    z = q.sum(1) + qs
    st = FT.TermState(
        cand_tau=cand_tau,
        cand_dir=cand_dir,
        valid=valid,
        q=q / z[:, None],
        q_stop=qs / z,
        u=rng.normal(0.0, 0.3, n),
        s_ref=s_ref,
        profiles=lines,
        level_base=lb,
        stop_log_odds=rng.normal(0.0, 2.0, n),
        tensor_dir=unit(rng.normal(size=(n, 3))),
        coherence=rng.uniform(0.0, 1.0, n),
        prof_line=np.broadcast_to(line, (n, K)).copy(),
        prof_off=np.broadcast_to(off, (n, K)).copy(),
    )
    # the profiles of every slot
    P = FT.candidate_profile_rows(st, np.arange(n)[:, None], np.arange(K)[None, :])
    scale = np.abs(ref).max()
    assert np.abs(P - ref).max() <= LINE_REL * scale
    tw = weights(w=2.0)
    # the contrast energy
    e_ref = _ref_contrast_energy(ga, tw, valid, st.q, ref, lb)
    e_new = FT.expected_energy(st, ga, tw)["contrast"]
    assert e_ref > 0.0
    assert abs(e_new - e_ref) <= LINE_REL * abs(e_ref)
    # the contrast messages: the messages less those of zero lines (no contrast term)
    sel = np.arange(n)
    M, Ms = FT.messages(st, ga, tw, sel)
    M0, Ms0 = FT.messages(replace(st, profiles=np.zeros_like(lines)), ga, tw, sel)
    np.testing.assert_array_equal(Ms, Ms0)  # the stopped state has no contrast term
    m_ref = _ref_contrast_messages(ga, tw, valid, st.q, ref, lb)
    tol = LINE_REL * np.abs(m_ref[valid]).max() + 1e-12 * np.abs(M[valid]).max()
    assert np.abs((M - M0)[valid] - m_ref[valid]).max() <= tol
    _line_report["profile"] = float(np.abs(P - ref).max() / scale)
    _line_report["energy"] = float(abs(e_new - e_ref) / abs(e_ref))
    _line_report["messages"] = float(
        np.abs((M - M0)[valid] - m_ref[valid]).max() / np.abs(m_ref[valid]).max()
    )
    print("profile lines vs reference (relative):", _line_report)


_line_report: dict[str, float] = {}


def test_line_slots_index_map():
    line, off = FT.line_slots(9, 7)
    for a in range(9):
        for m in range(7):
            c = a * 7 + m
            assert line[c] == m and off[c] == 8 - a
    assert off.min() == 0 and off.max() + FT.N_LEVELS == 17


# ---------------------------------------------------------------- the anchor origin


def test_no_upper_bound_on_tau_with_the_anchor_origin() -> None:
    s = T.DEFAULTS
    tp = np.array([0.3, -0.05])
    p = np.array([[1.0, 0, 0]] * 2)
    tau, d, valid = FT.moving_candidates(tp, p, s, tau_max=None)
    t0, d0, _ = FT.moving_candidates(np.array([-0.3, -1.0]), p, s)
    a = np.arange(63) // 7
    assert valid.all()
    np.testing.assert_allclose(tau, tp[:, None] + (a - 4)[None, :] * s.dtau, atol=1e-15)
    assert np.array_equal(d, d0)
    lb = FT.level_base(np.array([0.5, -1.0]), s, tau_max=None)
    assert lb.tolist() == [round(0.5 / 0.04) - 4, -25 - 4]


def test_jump_switch_off() -> None:
    """TermWeights.jump = False: the jump term is 0 in the expected energy, in the
    state energy and in the messages (the messages equal those with face_w = 0); the
    other terms do not change."""
    rng = np.random.default_rng(41)
    ga = uniform_ga((3, 3, 2), rng=rng)
    st = random_state(ga.n, 4, 3, rng, invalid=0.2)
    on = weights(w=2.0)
    off = replace(on, jump=False)
    e_on = FT.expected_energy(st, ga, on)
    e_off = FT.expected_energy(st, ga, off)
    assert e_on["jump"] > 0.0 and e_off["jump"] == 0.0
    for k in e_on:
        if k != "jump":
            assert e_off[k] == e_on[k], k
    ch = rng.integers(-1, 4, (5, ga.n))
    assert np.all(FT.state_energy(st, ga, off, ch)["jump"] == 0.0)
    M, Ms = FT.messages(st, ga, off, np.arange(ga.n))
    M0, Ms0 = FT.messages(st, ga_with(ga, face_w=np.zeros_like(ga.face_w)), on,
                          np.arange(ga.n))  # fmt: skip
    np.testing.assert_allclose(M, M0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(Ms, Ms0, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("k", [3, -7])
def test_energy_is_invariant_under_a_shift_of_every_tau(k: int) -> None:
    """With the jump term off and no τ prior (the anchor origin), a shift of every τ by
    k Δτ, with every level_base shifted by k, changes no expected term and no message
    (to rounding); and the sampled profiles at τ + k Δτ and the levels + k are the
    profiles at τ. So a shift by an integer multiple of Δτ is a symmetry of the
    discrete energy."""
    s = T.DEFAULTS
    rng = np.random.default_rng(43)
    ga = uniform_ga((3, 3, 3), rng=rng)
    st = random_state(ga.n, 4, 3, rng, invalid=0.2)
    tw = replace(weights(w=2.0), jump=False)
    sh = replace(st, cand_tau=st.cand_tau + k * s.dtau, level_base=st.level_base + k)
    e0 = FT.expected_energy(st, ga, tw)
    e1 = FT.expected_energy(sh, ga, tw)
    for key in e0:
        assert e1[key] == pytest.approx(e0[key], rel=1e-12, abs=1e-12), key
    M0, Ms0 = FT.messages(st, ga, tw, np.arange(ga.n))
    M1, Ms1 = FT.messages(sh, ga, tw, np.arange(ga.n))
    v = st.valid  # the values of slots that are not valid have no use
    np.testing.assert_allclose(M1[v], M0[v], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(Ms1, Ms0, rtol=1e-12, atol=1e-10)
    shape = (20, 20, 20)
    vols = [rng.normal(size=shape), rng.normal(size=shape)]
    nodes = rng.uniform(6.0, 14.0, (5, 3))
    tau = -rng.uniform(0.0, 0.5, (5, 4))
    d = unit(rng.normal(size=(5, 4, 3)))
    s_ref = rng.uniform(0.8, 1.2, 5)
    lb = rng.integers(-12, 0, 5)
    P0 = FT.candidate_profiles(nodes, tau, d, s_ref, lb, vols, L=9, dtau=s.dtau)
    P1 = FT.candidate_profiles(
        nodes, tau + k * s.dtau, d, s_ref, lb + k, vols, L=9, dtau=s.dtau
    )
    np.testing.assert_allclose(P1, P0, rtol=0, atol=1e-9)


# ---------------------------------------------------------------- the R2 change: the
# jump at the stop crossing (TermWeights.jump_at "crossing")


def _lo(p):
    return math.log(p / (1.0 - p))


def test_stop_crossing_cases() -> None:
    """t* = (0.5 − p_from) / (p_to − p_from) inside; 0 if p_from ≥ 0.5; 1 if p_to <
    0.5."""
    t = FT.stop_crossing(
        np.array([_lo(0.2), _lo(0.3), _lo(0.6), _lo(0.1), _lo(0.5)]),
        np.array([_lo(0.8), _lo(0.6), _lo(0.9), _lo(0.4), _lo(0.9)]),
    )
    np.testing.assert_allclose(t, [0.5, 2.0 / 3.0, 0.0, 1.0, 0.0], rtol=0, atol=1e-12)


def test_jump_at_crossing_by_hand_two_nodes() -> None:
    """Two nodes 0.16 mm apart on x. Node 0 moving (τ = −0.2, d = +x, slowness 1),
    node 1 stopped. The stop field: p = 0.3 at node 0, 0.6 at node 1, so the crossing
    is at t* = 2/3 of the edge. The level there: −0.2 + 0.16 · 2/3 = −0.09333; J =
    0.09333. With jump_at "node" (PR 3): −0.2 + 0.16 = −0.04, J = 0.04."""
    ga = two_node_ga()
    st = replace(
        one_state(2, 1),
        cand_tau=np.array([[-0.2], [-0.1]]),
        s_ref=np.array([1.0, 2.0]),
        stop_log_odds=np.array([_lo(0.3), _lo(0.6)]),
    )
    e1 = -math.log(0.001) + math.log(0.5)
    J_cross = 0.2 - 0.16 * 2.0 / 3.0
    tw_c = replace(weights(), jump_at="crossing")
    e = FT.state_energy(st, ga, tw_c, np.array([0, -1]))
    assert e["jump"][0] == pytest.approx(e1 + J_cross / 0.5, rel=1e-12)
    e = FT.state_energy(st, ga, weights(), np.array([0, -1]))
    assert e["jump"][0] == pytest.approx(e1 + 0.04 / 0.5, rel=1e-12)
    # node 1 moving toward −x is not the direction (d = +x, away from node 0): the
    # level is τ = −0.1 at any crossing, J = 0.1 in both forms
    for tw in (weights(), tw_c):
        e = FT.state_energy(st, ga, tw, np.array([-1, 0]))
        assert e["jump"][0] == pytest.approx(e1 + 0.1 / 0.5, rel=1e-12)
    # the same J from jump_energy with the fraction
    _, J = FT.jump_energy(
        np.array(-0.2), np.array([1.0, 0, 0]), np.array(1.0),
        np.array([V, 0, 0]), frac=np.array(2.0 / 3.0),
    )  # fmt: skip
    assert float(J) == pytest.approx(J_cross, rel=1e-12)
    # the level at the crossing reaches 0: no jump
    st0 = replace(st, cand_tau=np.array([[-0.16 * 2.0 / 3.0 + 1e-9], [-0.1]]))
    assert FT.state_energy(st0, ga, tw_c, np.array([0, -1]))["jump"][0] == (
        pytest.approx(-math.log(0.999), rel=1e-12)
    )


def test_term_weights_jump_at() -> None:
    assert FT.term_weights(0.16, [0.1, 0.2], 1.0).jump_at == "node"
    assert FT.term_weights(0.16, [0.1, 0.2], 1.0, jump_at="crossing").jump_at == (
        "crossing"
    )
    with pytest.raises(ValueError):
        FT.term_weights(0.16, [0.1, 0.2], 1.0, jump_at="edge")


@pytest.mark.parametrize("seed", [0, 1])
def test_crossing_messages_finite_differences(seed: int) -> None:
    """With jump_at "crossing", the messages are the derivatives of the expected
    energy (the same tolerance as test_messages_finite_differences)."""
    rng = np.random.default_rng(seed)
    ga = uniform_ga((3, 3, 3), rng=rng)
    st = random_state(ga.n, 4, 3, rng, invalid=0.2)
    tw = replace(weights(w=2.0), jump_at="crossing")
    M, Ms = FT.messages(st, ga, tw, np.arange(ga.n))
    eps = 1e-4
    for i in range(ga.n):
        for k in [*np.nonzero(st.valid[i])[0], -1]:
            vals = []
            for sgn in (1, -1):
                if k >= 0:
                    q = st.q.copy()
                    q[i, k] += sgn * eps
                    s2 = replace(st, q=q)
                else:
                    qs = st.q_stop.copy()
                    qs[i] += sgn * eps
                    s2 = replace(st, q_stop=qs)
                vals.append(coupling(FT.expected_energy(s2, ga, tw)))
            fd = (vals[0] - vals[1]) / (2 * eps)
            m = M[i, k] if k >= 0 else Ms[i]
            assert m == pytest.approx(fd, rel=1e-6, abs=1e-6 * max(1.0, abs(fd)))


def test_crossing_expected_energy_by_enumeration() -> None:
    """With jump_at "crossing", the expected jump energy is the exact expectation over
    the joint states, and it differs from the "node" form."""
    rng = np.random.default_rng(7)
    ga = uniform_ga((2, 2, 2), rng=rng)
    st = random_state(ga.n, 3, 3, rng, invalid=0.25)
    tw = replace(weights(w=2.0), jump_at="crossing")
    states = [[*np.nonzero(st.valid[i])[0], -1] for i in range(ga.n)]
    ch = np.array(list(itertools.product(*states)), np.int64)
    rows = np.arange(ga.n)[None, :]
    prob = np.where(ch >= 0, st.q[rows, np.maximum(ch, 0)], st.q_stop[rows]).prod(1)
    se = FT.state_energy(st, ga, tw, ch)
    ee = FT.expected_energy(st, ga, tw)
    assert ee["jump"] == pytest.approx(float(prob @ se["jump"]), rel=1e-10, abs=1e-12)
    assert ee["jump"] != FT.expected_energy(st, ga, weights(w=2.0))["jump"]
