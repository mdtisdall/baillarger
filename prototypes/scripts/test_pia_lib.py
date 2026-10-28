"""Tests of pia_lib with exact answers (brute force on small graphs).

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_pia_lib.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import pia_lib as PL  # noqa: E402


def brute(D, pairs, lo, hi):
    n, K = D.shape
    best, arg = np.inf, None
    for c in itertools.product(range(K), repeat=n):
        c = np.array(c)
        d = c[pairs[:, 1]] - c[pairs[:, 0]]
        if np.all((d >= lo) & (d <= hi)):
            cost = D[np.arange(n), c].sum()
            if cost < best - 1e-12:
                best, arg = cost, c
    return best, arg


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n, K = int(rng.integers(2, 5)), int(rng.integers(3, 7))
    D = np.round(rng.normal(size=(n, K)), 3)
    allp = np.array([(a, b) for a in range(n) for b in range(a + 1, n)])
    pairs = allp[rng.random(len(allp)) < 0.7]
    if not len(pairs):
        pairs = allp[:1]
    lo = rng.integers(-2, 1, len(pairs))
    hi = lo + rng.integers(0, 3, len(pairs))
    best, _ = brute(D, pairs, lo, hi)
    if not np.isfinite(best):
        with pytest.raises(ValueError):
            PL.min_surface(D, pairs, lo, hi)
        return
    c = PL.min_surface(D, pairs, lo, hi)
    d = c[pairs[:, 1]] - c[pairs[:, 0]]
    assert np.all((d >= lo) & (d <= hi))
    assert abs(D[np.arange(n), c].sum() - best) < 1e-6


def test_chain_carries_an_anchor():
    """A chain of 6 columns anchored at label 2 on the first. A bound of 1 label per step keeps
    the dark label 9 on the last column out of reach (at most 2 + 5 = 7)."""
    K = 10
    D = np.zeros((6, K))
    D[0] = 5.0
    D[0, 2] = -5.0  # the anchor: column 0 at label 2
    D[5, 9] = -1.0  # a tempting dark label far away
    pairs = np.array([(i, i + 1) for i in range(5)])
    lo, hi = -np.ones(5, int), np.ones(5, int)
    c = PL.min_surface(D, pairs, lo, hi)
    assert c[0] == 2
    assert c[5] <= 7
    assert np.all(np.abs(np.diff(c)) <= 1)


def test_bound_labels_outward():
    lo, hi = PL.bound_labels(np.array([0.05]), np.array([0.1]), 0.04)
    assert lo[0] * 0.04 <= 0.05 - 0.1 and hi[0] * 0.04 >= 0.05 + 0.1


def test_anchor_penalty():
    d = np.arange(0, 3, 0.04)
    p = PL.anchor_penalty(d, 2.0, 0.1, 0.2, 99.0)
    inside = (d >= 1.9 - 1e-9) & (d <= 2.3 + 1e-9)
    assert np.all(p[inside] == 0) and np.all(p[~inside] == 99.0)


def brute_multi(grids, costs, cons):
    n = costs[0].shape[0]
    S = len(grids)
    best, arg = np.inf, None
    ranges = [range(len(g)) for g in grids for _ in range(n)]
    for lab in itertools.product(*ranges):
        L = np.array(lab).reshape(S, n)
        ok = True
        for sa, sb, ia, ib, lo, hi in cons:
            d = np.asarray(grids[sb])[L[sb, ib]] - np.asarray(grids[sa])[L[sa, ia]]
            if np.any(d < np.asarray(lo) - 1e-9) or np.any(d > np.asarray(hi) + 1e-9):
                ok = False
                break
        if ok:
            cost = sum(costs[s][np.arange(n), L[s]].sum() for s in range(S))
            if cost < best - 1e-12:
                best, arg = cost, L
    return best, arg


@pytest.mark.parametrize("seed", range(20))
def test_multi_matches_brute_force(seed):
    """Two surfaces on different grids (0.04 and 0.08 steps on one lattice) over 2-3 columns:
    neighbour bounds on each surface, an ordering band between them per column."""
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(2, 4))
    g0 = 0.4 + 0.04 * np.arange(int(rng.integers(3, 5)))
    g1 = 0.4 + 0.08 * np.arange(int(rng.integers(3, 5)))
    costs = [np.round(rng.normal(size=(n, g.size)), 3) for g in (g0, g1)]
    pairs = np.array([(a, a + 1) for a in range(n - 1)])
    cons = []
    for s_ in (0, 1):
        lo = np.round(rng.uniform(-0.12, 0.0, len(pairs)), 2)
        cons.append((s_, s_, pairs[:, 0], pairs[:, 1], lo, lo + np.round(rng.uniform(0, 0.16, len(pairs)), 2)))
    lo = np.round(rng.uniform(-0.1, 0.1, n), 2)
    cons.append((0, 1, np.arange(n), np.arange(n), lo, lo + np.round(rng.uniform(0.0, 0.2, n), 2)))
    best, _ = brute_multi([g0, g1], costs, cons)
    if not np.isfinite(best):
        with pytest.raises(ValueError):
            PL.min_surfaces([g0, g1], costs, cons)
        return
    out = PL.min_surfaces([g0, g1], costs, cons)
    L = np.stack(out)
    for sa, sb, ia, ib, lo_, hi_ in cons:
        d = [g0, g1][sb][L[sb, ib]] - [g0, g1][sa][L[sa, ia]]
        assert np.all(d >= lo_ - 1e-9) and np.all(d <= hi_ + 1e-9)
    assert abs(sum(costs[s_][np.arange(n), L[s_]].sum() for s_ in (0, 1)) - best) < 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_multi_one_surface_matches_single(seed):
    rng = np.random.default_rng(200 + seed)
    n, K = 4, 6
    D = np.round(rng.normal(size=(n, K)), 3)
    pairs = np.array([(0, 1), (1, 2), (2, 3), (0, 3)])
    lo = rng.integers(-2, 1, len(pairs))
    hi = lo + rng.integers(0, 3, len(pairs))
    g = 0.04 * np.arange(K)
    cons = [(0, 0, pairs[:, 0], pairs[:, 1], lo * 0.04, hi * 0.04)]
    try:
        c1 = PL.min_surface(D, pairs, lo, hi)
    except ValueError:
        with pytest.raises(ValueError):
            PL.min_surfaces([g], [D], cons)
        return
    (c2,) = PL.min_surfaces([g], [D], cons)
    assert np.isclose(D[np.arange(n), c1].sum(), D[np.arange(n), c2].sum())


def test_ordering_rejects_the_decoy():
    """The ordering's mechanics, one column: dark bands at 1.0 (near stria) and 2.7 (far stria).
    A darkness-only contact takes one of them. With its own faint evidence at 2.0, a near stria
    below it (0.6-1.2 short of it) and a far stria beyond it (0.6-2.1 past it), the contact stays
    at 2.0 and each stria takes its band. (With darkness as the contact's cost too, the decoy is
    as cheap as the truth: the ordering alone does not decide it.)"""
    g = 0.04 * np.arange(10, 126)  # 0.4-5.0 mm
    dark = lambda m: -np.exp(-0.5 * ((g - m) / 0.08) ** 2)  # noqa: E731
    D = (dark(1.0) + dark(2.7))[None]
    contact_alone = g[PL.min_surface(D, np.zeros((0, 2), int), np.zeros(0, int), np.zeros(0, int))[0]]
    assert abs(contact_alone - 2.7) < 0.05 or abs(contact_alone - 1.0) < 0.05
    pref = np.where(np.abs(g - 2.0) < 0.3, -0.05, 0.0)[None]  # a faint contact
    one = np.array([0])
    cons = [(0, 1, one, one, [0.6], [1.2]), (1, 2, one, one, [0.6], [2.1])]
    near, con, far = PL.min_surfaces([g, g, g], [D, pref, D], cons)
    assert abs(g[near[0]] - 1.0) < 0.05 and abs(g[far[0]] - 2.7) < 0.05
    assert 1.6 <= g[con[0]] <= 2.1
