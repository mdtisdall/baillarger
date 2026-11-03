"""Solver-independent tests of baillarger.surfcut.qpbo: exact answers by brute
force and by hand, persistency, and determinism."""

import dataclasses
import itertools
import random

import numpy as np
import pytest
from test_surfcut import brute, folded_model, random_model

from baillarger.surfcut import Model, Result, cut, energy, qpbo, qpbo_pruned
from baillarger.surfcut._flow import capacities
from baillarger.surfcut.model import Float, floats, ints


def binary_model(rng: np.random.Generator, n: int, coarse: bool) -> tuple[Model, bool]:
    """Binary variables, random unaries, tables on random pairs, a mix of
    submodular and not. coarse: values rounded to 0.5, so that ties are
    likely. Also returns whether every table is submodular."""
    step = 0.5 if coarse else 0.001
    unary: list[Float] = [np.round(rng.normal(size=2) / step) * step for _ in range(n)]
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    pick = rng.permutation(len(pairs))[: int(rng.integers(1, len(pairs) + 1))]
    T = np.zeros((len(pick), 2, 2))
    sub = True
    for p in range(len(pick)):
        t = np.round(rng.normal(size=(2, 2)) / step) * step
        if rng.random() < 0.5:  # force D = T00 - T01 - T10 + T11 <= 0
            t[1, 1] = t[0, 1] + t[1, 0] - t[0, 0] - abs(round(float(rng.normal()), 3))
        T[p] = t
        sub &= bool(t[0, 0] - t[0, 1] - t[1, 0] + t[1, 1] <= 0)
    a = ints([pairs[i][0] for i in pick])
    b = ints([pairs[i][1] for i in pick])
    M = Model(ints(np.full(n, 2)), ints(np.zeros(n)), unary, table=[(a, b, T)])
    return M, sub


def all_energies(M: Model) -> dict[tuple[int, ...], float]:
    return {
        k: energy(M, np.array(k))
        for k in itertools.product(*[range(int(K)) for K in M.K])
    }


@pytest.mark.parametrize("seed", range(40))
def test_binary_models(seed: int) -> None:
    """Valid bound, valid labelling, weak persistency; exact if submodular."""
    rng = np.random.default_rng(seed)
    n = 2 + seed % 11
    M, sub = binary_model(rng, n, coarse=seed % 4 == 3)
    E = all_energies(M)
    best = min(E.values())
    r = qpbo(M)
    assert r.lower_bound <= best + 1e-6
    assert r.energy >= best - 1e-9
    assert abs(energy(M, r.labels) - r.energy) < 1e-9
    fixed = {
        k: e
        for k, e in E.items()
        if all(k[i] == r.certified_labels[i] for i in np.flatnonzero(r.certified))
    }
    assert abs(min(fixed.values()) - best) < 1e-6
    unique = sum(e <= best + 1e-9 for e in E.values()) == 1
    if sub and unique:  # strong persistency: the unique minimum is found
        assert r.certified.all()
        assert abs(r.energy - best) < 1e-9


@pytest.mark.parametrize("seed", range(30))
def test_submodular_equals_cut(seed: int) -> None:
    """On submodular models QPBO is the minimum cut."""
    M = random_model(np.random.default_rng(500 + seed), tables="sub")
    best, _ = brute(M)
    if not np.isfinite(best):
        pytest.skip("infeasible")
    r = qpbo(M)
    assert abs(r.energy - energy(M, cut(M))) < 1e-9
    assert r.certified.all()


def triangle(unary0: list[float]) -> Model:
    """Three binary variables; every pair costs 1 if equal, 0 if different:
    T = [[1, 0], [0, 1]], D = 2 > 0, not submodular."""
    T = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]]), (3, 1, 1))
    unary: list[Float] = [
        np.array(unary0),
        np.zeros(2),
        np.zeros(2),
    ]
    return Model(
        ints([2, 2, 2]),
        ints([0, 0, 0]),
        unary,
        table=[(ints([0, 0, 1]), ints([1, 2, 2]), T)],
    )


def test_frustrated_triangle() -> None:
    """Three variables cannot all differ pairwise, so at least one pair is
    equal: the minimum is 1, reached by the six labellings that are not
    constant. By symmetry (flipping all labels) no variable has a label that
    every optimum shares, so none can be certified."""
    M = triangle([0.0, 0.0])
    best, _ = brute(M)
    assert best == 1.0
    r = qpbo(M)
    assert not r.certified.any()
    assert r.lower_bound <= 1.0 + 1e-9
    assert r.energy == 1.0


def test_triangle_with_unary() -> None:
    """Unary [0, 5] on variable 0. With x0 = 1 the cost is 5 + at least 1 = 6.
    With x0 = 0 the pairs (0,1), (0,2) cost 1 when x1 = 0, x2 = 0, and pair
    (1,2) costs 1 when x1 = x2: (x1, x2) = (1,1), (1,0), (0,1) give 1, and
    (0,0) gives 3. The minimum is 1, with x0 = 0 in every optimum and three
    optima 011, 001, 010 (x0 x1 x2). Roof duality resolves the cycle: the
    certified labels must extend to an optimum, so x0 = 0 and (x1, x2) is
    not (0, 0); which optimum is the solver's choice."""
    M = triangle([0.0, 5.0])
    best, arg = brute(M)
    assert abs(best - 1.0) < 1e-12
    assert arg is not None and arg[0] == 0
    r = qpbo(M)
    assert abs(r.energy - 1.0) < 1e-9
    assert abs(r.lower_bound - 1.0) < 1e-9
    assert r.certified.all()
    assert r.certified_labels[0] == 0
    assert tuple(r.certified_labels) in {(0, 1, 1), (0, 0, 1), (0, 1, 0)}


def test_submodular_chain() -> None:
    """Chain of 4 binary variables, a cost 1 for each change of label between
    neighbours (D = -2 <= 0), unaries (cost of label 0, label 1):
    x0 (0, 3), x1 (1, 0.2), x2 (0.5, 1), x3 (2, 0). By hand: 0111 costs
    0 + 0.2 + 1 + 0 + 1 change = 2.2; 0001 costs 0 + 1 + 0.5 + 0 + 1 = 2.5;
    0011 costs 3; 0000 costs 3.5; 1111 costs 4.2; labellings with more
    changes cost at least 3.2 + 2. So 0111 is the unique minimum, 2.2."""
    T = np.tile(np.array([[0.0, 1.0], [1.0, 0.0]]), (3, 1, 1))
    unary: list[Float] = [
        np.array([0.0, 3.0]),
        np.array([1.0, 0.2]),
        np.array([0.5, 1.0]),
        np.array([2.0, 0.0]),
    ]
    M = Model(
        ints([2] * 4),
        ints([0] * 4),
        unary,
        table=[(ints([0, 1, 2]), ints([1, 2, 3]), T)],
    )
    r = qpbo(M)
    assert r.certified.all()
    assert list(r.certified_labels) == [0, 1, 1, 1]
    assert list(r.labels) == [0, 1, 1, 1]
    assert abs(r.energy - 2.2) < 1e-9
    assert abs(r.lower_bound - 2.2) < 1e-9


def same(a: Result, b: Result) -> None:
    for f in dataclasses.fields(Result):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y), f.name
        else:
            assert x == y, f.name


def test_deterministic_across_global_rng() -> None:
    """The result does not depend on numpy's or Python's global random state."""
    M = folded_model(np.random.default_rng(1))
    for solve in (qpbo, qpbo_pruned):
        np.random.seed(7)
        random.seed(8)
        first = solve(M)
        np.random.seed(123)
        random.seed(456)
        same(first, solve(M))


def test_other_seed_still_valid() -> None:
    """Another visiting order gives a valid labelling, no worse than the cut."""
    M = folded_model(np.random.default_rng(1))
    ref = energy(M, cut(M))
    for solve in (qpbo, qpbo_pruned):
        r = solve(M, seed=5)
        assert abs(energy(M, r.labels) - r.energy) < 1e-9
        assert r.energy <= ref + 1e-9


# ------------------------------------------- integer capacities


def test_capacities_at_the_limit() -> None:
    """The hard cost is 1 + the sum of the finite capacities; it must be below
    2^61. Arcs 2^1 .. 2^60 sum to 2^61 - 2 (hard cost 2^61 - 1); with 2^0 too,
    the hard cost is 2^61."""
    p = [2.0**k for k in range(1, 61)]
    c = capacities(floats(p + [np.inf]), scale=1.0)
    assert c.hard == 2**61 - 1 and int(c.caps[-1]) == 2**61 - 1
    with pytest.raises(ValueError, match="2\\^61"):
        capacities(floats([1.0] + p + [np.inf]), scale=1.0)


def test_capacities_round_down() -> None:
    c = capacities(floats([0.0014, 0.0026]), scale=1000.0)
    assert list(c.caps) == [1, 2]
    assert c.max_error == pytest.approx(6e-4)
    assert c.total_error == pytest.approx(1e-3)


def test_cut_at_the_capacity_limit() -> None:
    """One variable, K = 2, unary [0, c]: one finite arc of c * 10^9 units
    (2^61 is 2.306e18)."""
    M = Model(ints([2]), ints([0]), [floats([0.0, 2.30e9])])
    assert list(cut(M)) == [0]
    M = Model(ints([2]), ints([0]), [floats([0.0, 2.31e9])])
    with pytest.raises(ValueError, match="2\\^61"):
        cut(M)


def three_neighbours(a: float, b: float) -> Model:
    """x1 = x2 = 0 and x3 = 1 are forced (a cost of 1 otherwise); hinges a |x1 -
    x0|, a |x2 - x0| and b |x3 - x0|: x0 = 1 costs 2a, x0 = 0 costs b."""
    one = [floats([0.0, 1.0]), floats([0.0, 1.0]), floats([1.0, 0.0])]
    hinge = [
        (ints(0), ints(k), ints(0), floats([w])) for k, w in ((1, a), (2, a), (3, b))
    ]
    return Model(
        ints([2, 2, 2, 2]), ints([0, 0, 0, 0]), [floats([0.0, 0.0])] + one, hinge=hinge
    )


@pytest.mark.parametrize("a, b", [(0.0004, 0.0006), (0.0009, 0.0011)])
def test_cut_with_small_costs(a: float, b: float) -> None:
    """2a > b, so the only minimum has x0 = 0. Costs rounded to 1/1000 change
    it: to the nearest unit, 0.0004 is 0 and 0.0006 is 0.001; rounded down,
    0.0009 is 0 and 0.0011 is 0.001. Both give x0 = 1."""
    M = three_neighbours(a, b)
    best, arg = brute(M)
    assert arg is not None and list(arg) == [0, 0, 0, 1]
    assert list(cut(M)) == [0, 0, 0, 1]


def test_qpbo_bound_with_a_cost_that_rounds_up() -> None:
    """x1 = 1 is forced; x0 = 0 pays a table entry of 0.0006, x0 = 1 a unary cost
    of 0.0007. The minimum is 0.0006; to the nearest 1/1000 its active cost is
    0.001, and a bound of the rounded energy would be above the minimum."""
    T = np.array([[[0.0, 0.0006], [0.0, 0.0]]])
    M = Model(
        ints([2, 2]),
        ints([0, 0]),
        [floats([0.0, 0.0007]), floats([1.0, 0.0])],
        table=[(ints([0]), ints([1]), T)],
    )
    best, _ = brute(M)
    assert best == pytest.approx(0.0006, abs=1e-15)
    r = qpbo(M)
    assert r.lower_bound <= best + 1e-12
    assert list(r.labels) == [0, 1]
