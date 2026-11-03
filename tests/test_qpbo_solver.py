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
from baillarger.surfcut._roof import Binary, _sink_first_rank, binary_energy, roof
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


# ------------------------------------------- the binary solver (_roof, _flow)


def random_binary(rng: np.random.Generator, n: int) -> Binary:
    """Binary energies with general tables (every entry random, 3 decimals),
    some pairs repeated or reversed, and some forbidden configurations."""
    m = int(rng.integers(1, 2 * n + 1))
    i, j = ints(rng.integers(0, n, m)), ints(rng.integers(0, n, m))
    tab = np.round(rng.normal(size=(4, m)), 3)
    tab[rng.random((4, m)) < 0.05] = np.inf
    U = np.round(rng.normal(size=(2, n)), 3)
    return Binary(U[0], U[1], i, j, tab[0], tab[1], tab[2], tab[3])


@pytest.mark.parametrize("seed", range(60))
def test_roof_persistency(seed: int) -> None:
    """Strong labels agree with every minimum, weak labels with some minimum,
    and the bound is at most the minimum."""
    rng = np.random.default_rng(31000 + seed)
    P = random_binary(rng, int(rng.integers(1, 11)))
    ys = [np.array(y) for y in itertools.product([0, 1], repeat=P.n)]
    E = np.array([binary_energy(P, y) for y in ys])
    best = float(E.min())
    if not np.isfinite(best):
        with pytest.raises(ValueError):
            roof(P)
        return
    r = roof(P)
    assert r.lower_bound <= best + 1e-9
    argmin = [y for y, e in zip(ys, E, strict=True) if e <= best + 1e-9]
    s = r.strong >= 0
    assert all(np.array_equal(y[s], r.strong[s]) for y in argmin)
    w = r.weak >= 0
    assert np.array_equal(r.weak[s], r.strong[s])  # strong labels are weak labels
    assert any(np.array_equal(y[w], r.weak[w]) for y in argmin)


@pytest.mark.parametrize("seed", range(40))
def test_roof_submodular_labels_all(seed: int) -> None:
    """With every table submodular (A + D <= B + C) and a unique minimum, every
    variable is labelled, at the minimum, and the bound equals it."""
    rng = np.random.default_rng(32000 + seed)
    P = random_binary(rng, int(rng.integers(1, 11)))
    P.A, P.B, P.C, P.D = (np.where(np.isinf(x), 1.0, x) for x in (P.A, P.B, P.C, P.D))
    P.D = np.minimum(P.D, P.B + P.C - P.A)
    ys = [np.array(y) for y in itertools.product([0, 1], repeat=P.n)]
    E = np.array([binary_energy(P, y) for y in ys])
    best = float(E.min())
    if np.sum(E <= best + 1e-9) > 1:
        return
    r = roof(P)
    assert np.all(r.weak >= 0) and np.all(r.strong >= 0)
    assert abs(binary_energy(P, r.weak) - best) < 1e-9
    assert abs(r.lower_bound - best) < 1e-9


def test_roof_known_answers() -> None:
    """By hand.
    - Two variables with D = 2 (non-submodular) and a unary cost of -1 on
      label 1 of each: E(00) = 0, E(01) = E(10) = -1, E(11) = 0. The two minima
      differ on both variables, so no label is strong. The bound is -1: it
      cannot exceed the minimum, and the cut of either minimum and its mirror
      is twice -1.
    - A chain y0 - y1 - y2, B = C = 1 (submodular), a cost of 3 on y0 = 1 and 2
      on y2 = 0: E = 3 y0 + 2 (1 - y2) + [y0 != y1] + [y1 != y2].
    """
    P = Binary(
        floats([0, 0]), floats([-1, -1]), ints([0]), ints([1]),
        floats([0]), floats([0]), floats([0]), floats([2]),
    )  # fmt: skip
    r = roof(P)
    assert list(r.strong) == [-1, -1]
    assert r.lower_bound == -1.0
    # the chain: E(y) = 3 y0 + 2 (1 - y2) + [y0 != y1] + [y1 != y2]
    P = Binary(
        floats([0, 0, 2]), floats([3, 0, 0]), ints([0, 1]), ints([1, 2]),
        floats([0, 0]), floats([1, 1]), floats([1, 1]), floats([0, 0]),
    )  # fmt: skip
    # 000: 2; 001: 1; 011: 1; 111: 3; 010: 2 + 2 = 4; 100: 3+2+1 = 6; ...
    # two minima, 001 and 011: y0 = 0 and y2 = 1 strong, y1 free
    r = roof(P)
    assert list(r.strong) == [0, -1, 1]
    assert r.lower_bound == 1.0
    assert binary_energy(P, np.where(r.weak < 0, 0, r.weak)) == 1.0


def test_capacities_at_the_limit() -> None:
    """The hard cost is 1 + the sum of the finite capacities; it must be below
    2^30."""
    c = capacities(floats([2**30 - 2, np.inf]), scale=1.0)
    assert c.hard == 2**30 - 1 and list(c.caps) == [2**30 - 2, 2**30 - 1]
    with pytest.raises(ValueError, match="2\\^30"):
        capacities(floats([2**30 - 1, np.inf]), scale=1.0)
    c = capacities(floats([0.0014, 0.0026]), scale=1000.0)
    assert list(c.caps) == [1, 3] and c.max_error == pytest.approx(4e-4)


def test_cut_at_the_capacity_limit() -> None:
    """One variable, K = 2, unary [0, c]: one finite arc of c * 1000. At
    2^30 - 2 units (hard cost 2^30 - 1) it solves; one unit more raises."""
    M = Model(ints([2]), ints([0]), [floats([0.0, 1073741.822])])
    assert list(cut(M)) == [0]
    M = Model(ints([2]), ints([0]), [floats([0.0, 1073741.823])])
    with pytest.raises(ValueError, match="2\\^30"):
        cut(M)


def test_qpbo_logs_scale_and_rounding() -> None:
    lines: list[str] = []
    qpbo(folded_model(np.random.default_rng(0)), log=lines.append)
    assert any("scale 1000" in m and "rounding error" in m for m in lines)


def test_sink_first_rank() -> None:
    """Arcs 0 -> 1 -> 2 and 0 -> 2 with labels that break the order: the rank is
    computed again, and every arc goes from a higher rank to a lower one."""
    cu, cv = ints([0, 1, 0]), ints([1, 2, 2])
    labels = ints([0, 1, 2])
    rank = _sink_first_rank(3, cu, cv, labels)
    assert np.all(rank[cu] > rank[cv])
    assert np.array_equal(_sink_first_rank(3, cu, cv, ints([2, 1, 0])), [2, 1, 0])
