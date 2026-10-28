"""Tests of edit_priority with exact answers.
    nix develop --command uv run --with thinqpbo pytest -q prototypes/scripts/test_edit_priority.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import baillarger.surfcut as WC  # noqa: E402
import edit_priority as EP  # noqa: E402
import joint_levels as JL  # noqa: E402
from baillarger.surfcut.model import ints  # noqa: E402


def random_model(rng, n=5):
    """Every kind of term, on random pairs (no self-terms), including a hard bound and tables
    with nan padding."""
    K = ints(rng.integers(2, 7, n))
    o = ints(rng.integers(-3, 4, n))
    unary = [np.round(rng.normal(size=int(k)), 3) for k in K]
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    pick = lambda m: [pairs[i] for i in rng.choice(len(pairs), m, replace=False)]  # noqa: E731
    hard = [(ints([p[0] for p in pp]), ints([p[1] for p in pp]), ints([-2, -1]), ints([1, 3]))
            for pp in [pick(2)]]
    hinge = [(ints([p[0] for p in pp]), ints([p[1] for p in pp]), ints(rng.integers(-2, 3, 4)),
              np.round(rng.uniform(0, 1.5, 4), 3)) for pp in [pick(4)]]
    tie = [(ints([p[0] for p in pp]), ints([p[1] for p in pp]), ints(rng.integers(0, 6, 3)),
            np.round(rng.uniform(0.2, 1.5, 3), 3)) for pp in [pick(3)]]
    tp = pick(3)
    T = np.full((3, int(K.max()), int(K.max())), np.nan)
    for p_, (a, b) in enumerate(tp):
        T[p_, :K[a], :K[b]] = np.round(rng.normal(size=(K[a], K[b])), 3)
    table = [(ints([p[0] for p in tp]), ints([p[1] for p in tp]), T)]
    return WC.Model(K, o, unary, hard, hinge, tie, table)


def reference(M, k, d):
    """energy() difference of moving each variable alone by d labels (inf outside its window or
    where a hard bound breaks)."""
    e0 = WC.energy(M, k)
    out = np.full(M.n, np.inf)
    for i in range(M.n):
        if 0 <= k[i] + d < M.K[i]:
            k2 = k.copy()
            k2[i] += d
            out[i] = WC.energy(M, k2) - e0
    return out


@pytest.mark.parametrize("seed", range(60))
def test_margins_equal_energy_differences(seed):
    rng = np.random.default_rng(seed)
    M = random_model(rng)
    for _ in range(2000):  # labellings where the bounds hold
        k = ints([rng.integers(0, K) for K in M.K])
        if np.isfinite(WC.energy(M, k)):
            break
    else:
        pytest.skip("no feasible labelling drawn")
    deltas = [-3, -2, -1, 1, 2, 3]
    got = EP.margins(M, k, deltas)
    for r, d in enumerate(deltas):
        ref = reference(M, k, d)
        assert np.array_equal(np.isinf(got[r]), np.isinf(ref)), (d, got[r], ref)
        f = np.isfinite(ref)
        assert np.allclose(got[r][f], ref[f], atol=1e-9)


def test_margins_on_a_joint_model():
    """A small joint_levels model (all of its term kinds: anchors, smoothness, ordering, the
    vertical prior and conservation tables, ties), every variable, every move of up to 4 labels."""
    rng = np.random.default_rng(3)
    n, m = 6, 120
    pairs = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2)])
    D = JL.Data(step=0.04, ns=np.full(n, m), top=np.array([2.0, np.nan, 2.2, np.nan, 1.9, 2.1]),
                mu=rng.normal(0, 0.1, n), pairs=pairs, S=rng.normal(0, 0.05, (len(pairs), m)),
                W=rng.uniform(0, 1, (len(pairs), m)), ties=np.array([(1, 4)]), tie_L=np.array([4.1]),
                tie_w=np.array([0.8]))
    D.c0 = np.full(n, 2.0)
    centre = np.stack([t * np.full(n, 2.0) for t in JL.TAU])
    M, _ = JL.build(D, 0.08, centre, np.full(centre.shape, 0.24))
    k = WC.qpbo(M).labels
    for d in [-4, -2, -1, 1, 3]:
        got = EP.margins(M, k, [d])[0]
        ref = reference(M, k, d)
        f = np.isfinite(ref)
        assert np.array_equal(np.isinf(got), ~f)
        assert np.allclose(got[f], ref[f], atol=1e-9)


def test_margin_at_the_minimum_is_not_negative():
    """At a brute-force minimum no single move lowers the energy."""
    rng = np.random.default_rng(11)
    M = random_model(rng, n=4)
    best, arg = np.inf, None
    for k in itertools.product(*[range(int(K)) for K in M.K]):
        e = WC.energy(M, np.array(k))
        if e < best:
            best, arg = e, np.array(k)
    assert np.all(EP.margins(M, ints(arg), [-1, 1, -2, 2]) >= -1e-9)


def test_ranks_give_no_doubt_rank_zero():
    """No doubt (0, or nan where a doubt is not defined) ranks 0 however many traces share it;
    a larger doubt ranks by the fraction of all traces with a defined doubt strictly below it."""
    x = np.array([0.0, 0.0, 0.0, np.nan, 0.5, 0.2, 0.5, 1.0])
    assert np.allclose(EP.ranks(x), [0, 0, 0, 0, 4 / 8, 3 / 8, 4 / 8, 6 / 8])
    assert np.allclose(EP.ranks(np.array([-2.0, -1.0, -3.0])), [1 / 3, 2 / 3, 0])
