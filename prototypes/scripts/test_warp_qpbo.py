"""Tests of warp_cut.qpbo with exact answers: brute force on small models, the dynamic
programme of tests/test_surfcut.py (fold_min) on larger folded ones.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import warp_cut as WC  # noqa: E402
from test_warp_cut import brute, random_model  # noqa: E402


def check_against(M, best, restricted_best):
    r = WC.qpbo(M)
    assert r["chain_ok"]
    assert abs(WC.energy(M, r["labels"]) - r["energy"]) < 1e-9
    assert r["energy"] >= best - 1e-9  # a labeling, so never below the optimum
    assert r["lower_bound"] <= best + 1e-6  # a valid bound
    # persistency: fixing the certified labels keeps the optimum reachable
    if r["certified"].any():
        assert abs(restricted_best(r["certified"], r["certified_labels"]) - best) < 1e-6
    return r


@pytest.mark.parametrize("seed", range(60))
def test_qpbo_against_brute_force(seed):
    rng = np.random.default_rng(7000 + seed)
    M = random_model(rng, ties=True)
    best, _ = brute(M)
    if not np.isfinite(best):
        return

    def restricted(cert, lab):
        b = np.inf
        for k in itertools.product(*[range(K) if not c else [l] for K, c, l in zip(M.K, cert, lab)]):
            b = min(b, WC.energy(M, np.array(k)))
        return b

    check_against(M, best, restricted)


@pytest.mark.parametrize("seed", range(20))
def test_qpbo_without_ties_is_exact(seed):
    """Submodular (no ties): everything certified, the optimum found."""
    rng = np.random.default_rng(8000 + seed)
    M = random_model(rng, ties=False)
    best, _ = brute(M)
    if not np.isfinite(best):
        return
    r = WC.qpbo(M)
    assert r["certified"].all()
    assert abs(r["energy"] - best) < 1e-6
    assert abs(WC.energy(M, r["certified_labels"]) - best) < 1e-6


def fold_min_wc(M, fix=None):
    """The exact optimum of a folded_model by the dynamic programme of
    tests/test_surfcut.py (fold_min); CP-SAT took up to 10 s here to prove it."""
    sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))
    from test_surfcut import fold_min

    from baillarger.surfcut import Model

    return fold_min(Model(np.asarray(M.K), np.asarray(M.o), M.unary, M.hard, M.hinge, M.tie), fix)


def folded_model(rng, n_side=12, K=14):
    """Two banks of n_side columns in a chain, joined at one end (the fold): lateral hinges toward
    their trend, ties between facing columns (a sum), random data costs, ordering-free."""
    n = 2 * n_side
    Ks = np.full(n, K)
    o = rng.integers(0, 3, n)
    unary = [np.round(rng.normal(size=K) * 0.6, 3) for _ in range(n)]
    hinge = []
    for i in range(n - 1):  # A: 0..n_side-1, then B continues round the fold
        hinge.append((np.array([i]), np.array([i + 1]), np.array([int(rng.integers(-1, 2))]),
                      np.array([round(float(rng.uniform(0.2, 0.8)), 3)])))
    hard = [(np.array([i]), np.array([i + 1]), np.array([-3]), np.array([3])) for i in range(n - 1)]
    tie = []
    for i in range(n_side - 2):  # facing columns: A's i against B's mirror, away from the fold
        j = n - 1 - i
        tie.append((np.array([i]), np.array([j]), np.array([int(rng.integers(K - 2, K + 4))]),
                    np.array([round(float(rng.uniform(0.3, 1.0)), 3)])))
    return WC.Model(Ks, o, unary, hard, hinge, tie)


@pytest.mark.parametrize("seed", range(8))
def test_qpbo_against_exact_folded(seed):
    rng = np.random.default_rng(9000 + seed)
    M = folded_model(rng)
    best = fold_min_wc(M)
    r = check_against(M, best, lambda cert, lab: fold_min_wc(M, (cert, lab)))
    print(f"seed {seed}: optimum {best:.3f}, qpbo {r['energy']:.3f}, bound {r['lower_bound']:.3f}, "
          f"certified {r['certified'].mean():.0%}")


@pytest.mark.parametrize("seed", range(30))
def test_reduce_is_exact(seed):
    """For random fixings and random labelings of the rest: reduced energy + constant equals the
    full energy (where the fixed variables' own hard bounds hold)."""
    rng = np.random.default_rng(11000 + seed)
    M = random_model(rng, ties=True)
    fixed = rng.random(M.n) < 0.5
    lab = np.array([rng.integers(0, K) for K in M.K])
    R, free, const = WC.reduce(M, fixed, lab, big=1e6)
    x = M.o + lab
    a, b, lo, hi = WC._cat(M.hard, "hard")
    ff = fixed[a] & fixed[b] if a.size else np.zeros(0, bool)
    if np.any((x[b][ff] - x[a][ff] < lo[ff]) | (x[b][ff] - x[a][ff] > hi[ff])):
        return  # the fixed labels break a bound between fixed variables: outside reduce's contract
    for _ in range(20):
        k = lab.copy()
        k[free] = [rng.integers(0, K) for K in R.K]
        e = WC.energy(M, k)
        er = WC.energy(R, k[free]) + const
        if np.isfinite(e):
            assert abs(e - er) < 1e-6
        else:
            assert er >= 1e5  # a broken bound touching a free variable shows as the big cost


@pytest.mark.parametrize("seed", range(20))
def test_pruned_never_worse_than_its_bound(seed):
    rng = np.random.default_rng(12000 + seed)
    M = random_model(rng, ties=True)
    best, _ = brute(M)
    if not np.isfinite(best):
        return
    r = WC.qpbo_pruned(M)
    assert r["energy"] >= best - 1e-9
    assert r["lower_bound"] <= best + 1e-6
    assert abs(WC.energy(M, r["labels"]) - r["energy"]) < 1e-9


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("kind", ["sub", "any"])
def test_qpbo_with_tables(seed, kind):
    rng = np.random.default_rng(14000 + seed)
    M = random_model(rng, ties=True, tables=kind)
    best, _ = brute(M)
    if not np.isfinite(best):
        return

    def restricted(cert, lab):
        b = np.inf
        for k in itertools.product(*[range(K) if not c else [l] for K, c, l in zip(M.K, cert, lab)]):
            b = min(b, WC.energy(M, np.array(k)))
        return b

    check_against(M, best, restricted)


@pytest.mark.parametrize("seed", range(20))
def test_qpbo_submodular_tables_exact(seed):
    rng = np.random.default_rng(15000 + seed)
    M = random_model(rng, ties=False, tables="sub")
    best, _ = brute(M)
    if not np.isfinite(best):
        return
    r = WC.qpbo(M)
    assert r["certified"].all() and abs(r["energy"] - best) < 1e-6


@pytest.mark.parametrize("seed", range(20))
def test_reduce_with_tables(seed):
    rng = np.random.default_rng(16000 + seed)
    M = random_model(rng, ties=True, tables="any")
    M.hard = []
    fixed = rng.random(M.n) < 0.5
    lab = np.array([rng.integers(0, K) for K in M.K])
    R, free, const = WC.reduce(M, fixed, lab, big=1e6)
    for _ in range(20):
        k = lab.copy()
        k[free] = [rng.integers(0, K) for K in R.K]
        assert abs(WC.energy(M, k) - (WC.energy(R, k[free]) + const)) < 1e-6
