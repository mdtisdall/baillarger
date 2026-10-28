"""Tests of edit_resolve with exact answers.
    nix develop --command uv run --with thinqpbo pytest -q prototypes/scripts/test_edit_resolve.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).parent))
import baillarger.surfcut as WC  # noqa: E402
import edit_resolve as ER  # noqa: E402
import joint_levels as JL  # noqa: E402


def chain_data(rng, n, tau=JL.TAU, m=120, ties=((1, 4),)):
    pairs = np.array([(i, i + 1) for i in range(n - 1)] + [(0, 2)])
    top = np.where(rng.random(n) < 0.5, rng.uniform(1.8, 2.2, n), np.nan)
    ties = np.array(ties, int).reshape(-1, 2)
    D = JL.Data(step=0.04, ns=np.full(n, m), top=top, mu=rng.normal(0, 0.1, n), pairs=pairs,
                S=rng.normal(0, 0.05, (len(pairs), m)), W=rng.uniform(0, 1, (len(pairs), m)), ties=ties,
                tie_L=np.full(len(ties), 4.1), tie_w=np.full(len(ties), 0.8))
    D.c0 = np.full(n, 2.0)
    D.tau = tau
    return D


def grid(z, h):
    return h * np.round(z / h)


@pytest.mark.parametrize("seed", range(6))
def test_everything_free_equals_a_full_solve(seed):
    """R covering every trace: no ring, nothing fixed; the local solve is qpbo_pruned on the full
    model from the same start."""
    rng = np.random.default_rng(seed)
    n, h, half = 8, 0.08, 0.24
    D = chain_data(rng, n)
    z = grid(np.stack([t * rng.uniform(1.8, 2.2, n) for t in D.tau]), h)
    out, info = ER.local_solve(D, z, np.ones(n, bool), h=h, half=half, recentre=0)
    M, o = JL.build(D, h, z, np.full(z.shape, half))
    lab0 = np.round(z.ravel() / h).astype(int) - np.asarray(M.o)
    r = WC.qpbo_pruned(M, start_labels=lab0)
    ref = h * (np.asarray(M.o) + r.labels).reshape(z.shape)
    assert info["ring_traces"] == 0
    k_out = np.round(out.ravel() / h).astype(int) - np.asarray(M.o)
    assert abs(WC.energy(M, k_out) - r.energy) < 1e-9
    if r.certified.all():
        assert np.allclose(out, ref)


def brute_with_fixed(M, fixed, lab):
    best, arg = np.inf, None
    free = np.flatnonzero(~fixed)
    for kk in itertools.product(*[range(int(M.K[i])) for i in free]):
        k = lab.copy()
        k[free] = kk
        e = WC.energy(M, k)
        if e < best - 1e-12:
            best, arg = e, k
    return best, arg


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("edit", [False, True])
def test_part_free_equals_brute_force(seed, edit):
    """Two levels, 6 traces, a chain; traces 2-3 free, the rest fixed at their levels. The local
    solve (sub-model of the free traces and their ring, the ring fixed by reduce), with or without
    an edit on trace 2, reaches the brute-force minimum of the full model over the free variables,
    the edit's unary included."""
    rng = np.random.default_rng(100 + seed)
    n, h, half = 6, 0.08, 0.16
    D = chain_data(rng, n, tau=(0.5, 1.0), ties=((2, 5),))
    z = grid(np.stack([t * rng.uniform(1.8, 2.2, n) for t in D.tau]), h)
    free = np.zeros(n, bool)
    free[[2, 3]] = True
    edits = {2: float(z[-1, 2] + 0.16)} if edit else {}
    out, info = ER.local_solve(D, z, free, h=h, half=half, edits=edits, recentre=0)
    assert info["ring_traces"] == 4  # 1, 4 and 0 by pairs (0 by the pair (0, 2)); 5 by the tie
    hw = np.where(free[None], half, 0.0) * np.ones((2, 1))
    for t, d in edits.items():
        hw[-1, t] = max(hw[-1, t], abs(d - z[-1, t]) + 2 * h)
    M, o = JL.build(D, h, z, hw)
    for t, d in edits.items():
        i = n + t
        M.unary[i] = M.unary[i] + ER.W_EDIT * np.abs(h * (M.o[i] + np.arange(M.K[i])) - d)
    lab = np.round(z.ravel() / h).astype(int) - np.asarray(M.o)
    fixed = np.repeat(~free[None], 2, 0).ravel()
    best, _ = brute_with_fixed(M, fixed, lab)
    k_out = np.round(out.ravel() / h).astype(int) - np.asarray(M.o)
    assert np.all(k_out[fixed] == lab[fixed])
    assert abs(WC.energy(M, k_out) - best) < 1e-9


def test_a_strong_edit_moves_the_pia_to_the_point():
    rng = np.random.default_rng(7)
    n, h = 8, 0.04
    D = chain_data(rng, n)
    z = grid(np.stack([t * np.full(n, 2.0) for t in D.tau]), h)
    out, _ = ER.local_solve(D, z, np.ones(n, bool), h=h, half=0.32, edits={3: 2.24})
    assert abs(out[-1, 3] - 2.24) < 1e-9


def test_neighbourhood_is_geodesic():
    """A path graph of 6 vertices 1 mm apart; traces rooted at vertices 0, 2, 5 and a seed at
    vertex 2: R = 1.5 reaches only itself, R = 2 also 0, R = 3 also 5."""
    w = np.ones(5)
    G = coo_matrix((w, (np.arange(5), np.arange(1, 6))), shape=(6, 6)).tocsr()
    G = G.maximum(G.T).tocsr()
    rv = np.array([0, 2, 5])
    assert list(ER.neighbourhood(G, rv, [1], 1.5)) == [False, True, False]
    assert list(ER.neighbourhood(G, rv, [1], 2.0)) == [True, True, False]
    assert list(ER.neighbourhood(G, rv, [1], 3.0)) == [True, True, True]


def test_edits_persist_and_only_free_edits_count():
    """Two edits in turn whose neighbourhoods overlap: the second solve keeps the first edit
    (passed again, as main does); an edit on a trace outside the free set changes nothing."""
    rng = np.random.default_rng(9)
    n, h = 8, 0.04
    D = chain_data(rng, n)
    z = grid(np.stack([t * np.full(n, 2.0) for t in D.tau]), h)
    free = np.ones(n, bool)
    z1, _ = ER.local_solve(D, z, free, h=h, half=0.32, edits={3: 2.24})
    z2, _ = ER.local_solve(D, z1, free, h=h, half=0.32, edits={3: 2.24, 5: 1.84})
    assert abs(z2[-1, 3] - 2.24) < 1e-9 and abs(z2[-1, 5] - 1.84) < 1e-9
    part = np.zeros(n, bool)
    part[[0, 1]] = True
    z3, _ = ER.local_solve(D, z2, part, h=h, half=0.32, edits={3: 2.24, 5: 1.84, 7: 1.0})
    z4, _ = ER.local_solve(D, z2, part, h=h, half=0.32, edits={})
    assert np.allclose(z3, z4)
