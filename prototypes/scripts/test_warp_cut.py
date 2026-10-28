"""Tests of warp_cut with exact answers (brute force on small models).

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_warp_cut.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import warp_cut as WC  # noqa: E402


def brute(M):
    best, arg = np.inf, None
    for k in itertools.product(*[range(K) for K in M.K]):
        e = WC.energy(M, np.array(k))
        if e < best - 1e-12:
            best, arg = e, np.array(k)
    return best, arg


def random_tables(rng, K, pairs, submodular):
    """Random pairwise tables; submodular ones as w |x_b - g(x_a)| with g non-decreasing (the
    conservation term's form), others as random tables."""
    if not pairs:
        return []
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    T = np.full((len(pairs), K.max(), K.max()), np.nan)
    for p_, (aa, bb) in enumerate(pairs):
        if submodular:
            g = np.cumsum(rng.integers(0, 3, K[aa])) - int(rng.integers(0, 3))
            T[p_, :K[aa], :K[bb]] = round(float(rng.uniform(0.1, 1.0)), 3) * np.abs(np.arange(K[bb])[None] - g[:, None])
        else:
            T[p_, :K[aa], :K[bb]] = np.round(rng.normal(size=(K[aa], K[bb])), 3)
    return [(a, b, T)]


def random_model(rng, ties=False, tables=None):
    n = int(rng.integers(2, 5))
    K = rng.integers(2, 6, n)
    o = rng.integers(-2, 3, n)
    unary = [np.round(rng.normal(size=k), 3) for k in K]
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(pairs)
    hard, hinge, tie = [], [], []
    for a, b in pairs[: int(rng.integers(0, len(pairs) + 1))]:
        lo = int(rng.integers(-3, 2))
        hard.append((np.array([a]), np.array([b]), np.array([lo]), np.array([lo + int(rng.integers(0, 4))])))
    for a, b in pairs[: int(rng.integers(1, len(pairs) + 1))]:
        hinge.append((np.array([a]), np.array([b]), np.array([int(rng.integers(-2, 3))]),
                      np.array([round(float(rng.uniform(0, 1.5)), 3)])))
    if ties:
        for a, b in pairs[: int(rng.integers(1, 3))]:
            tie.append((np.array([a]), np.array([b]), np.array([int(rng.integers(0, 6))]),
                        np.array([round(float(rng.uniform(0.2, 1.5)), 3)])))
    tab = random_tables(rng, K, pairs[: int(rng.integers(1, len(pairs) + 1))], tables == "sub") if tables else []
    return WC.Model(K, o, unary, hard, hinge, tie, tab)


@pytest.mark.parametrize("seed", range(40))
def test_cut_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    M = random_model(rng)
    best, _ = brute(M)
    if not np.isfinite(best):
        with pytest.raises(ValueError):
            WC.cut(M)
        return
    k = WC.cut(M)
    assert abs(WC.energy(M, k) - best) < 1e-6


def test_hinge_negative_offsets_and_origins():
    """A hinge whose target lies outside both label ranges (the j <= 0 part)."""
    M = WC.Model(np.array([3, 3]), np.array([5, 0]), [np.zeros(3), np.array([0.0, 0.1, 0.2])], [],
                 [(np.array([0]), np.array([1]), np.array([-9]), np.array([1.0]))])
    best, _ = brute(M)
    assert abs(WC.energy(M, WC.cut(M)) - best) < 1e-9


@pytest.mark.parametrize("seed", range(30))
def test_cut_with_submodular_tables(seed):
    rng = np.random.default_rng(13000 + seed)
    M = random_model(rng, tables="sub")
    best, _ = brute(M)
    if not np.isfinite(best):
        return
    assert abs(WC.energy(M, WC.cut(M)) - best) < 1e-6
