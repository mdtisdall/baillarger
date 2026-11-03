"""Exact tests of the binary QPBO solver (baillarger.surfcut._roof): brute
force on small energies, answers by hand, and the order of components."""

import itertools

import dimod
import numpy as np
import numpy.typing as npt
import pytest
from dwave.preprocessing import roof_duality
from scipy.optimize import linprog
from test_surfcut import folded_model

from baillarger.surfcut import qpbo
from baillarger.surfcut._roof import Binary, _sink_first_rank, binary_energy, roof
from baillarger.surfcut.model import Float, floats, ints


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
    variable is labelled, at the minimum, and the bound equals it up to the
    rounding: costs are rounded down to 1e-9, so the bound is at most the
    minimum and below it by at most the total rounding error (< 1e-7 here)."""
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
    assert r.lower_bound <= best + 1e-12
    assert best - r.lower_bound < 1e-7


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


def test_qpbo_logs_scale_and_rounding() -> None:
    lines: list[str] = []
    qpbo(folded_model(np.random.default_rng(0)), log=lines.append)
    assert any("scale 1e+09" in m and "rounding error" in m for m in lines)


def test_sink_first_rank() -> None:
    """Arcs 0 -> 1 -> 2 and 0 -> 2 with labels that break the order: the rank is
    computed again, and every arc goes from a higher rank to a lower one."""
    cu, cv = ints([0, 1, 0]), ints([1, 2, 2])
    labels = ints([0, 1, 2])
    rank = _sink_first_rank(3, cu, cv, labels)
    assert np.all(rank[cu] > rank[cv])
    assert np.array_equal(_sink_first_rank(3, cu, cv, ints([2, 1, 0])), [2, 1, 0])


# ------------------------------------------- the bound and labels of the true energy


def forced(n: int) -> tuple[Float, Float]:
    """Unary costs: variable 0 free, variables 1.. forced to 1 (a cost of 1 at 0)."""
    U0, U1 = np.zeros(n), np.zeros(n)
    U0[1:] = 1.0
    return U0, U1


def test_bound_is_below_the_true_minimum_when_costs_round_up() -> None:
    """y1 = 1 is forced; y0 = 0 and y0 = 1 cost 0.0006 (B, submodular) and 0.0007
    (unary): the minimum is 0.0006 at y = (0, 1). Its active cost, 0.0006, is
    above half a unit of 1/1000: rounded to the nearest unit it is 0.001, and a
    bound of the rounded energy would be above the true minimum."""
    U0, U1 = forced(2)
    U1[0] = 0.0007
    P = Binary(U0, U1, ints([0]), ints([1]), *(floats([c]) for c in (0, 0.0006, 0, 0)))
    best = min(binary_energy(P, y) for y in itertools.product([0, 1], repeat=2))
    assert best == pytest.approx(0.0006, abs=1e-15)
    assert roof(P).lower_bound <= best + 1e-12


def test_small_non_submodular_terms_do_not_change_the_labels() -> None:
    """y1, y2, y3 = 1 are forced. y0 = 1 pays two non-submodular terms D = 0.0004
    (with y1 and y2); y0 = 0 pays one submodular term B = 0.0006 (with y3). The
    only minimum has y0 = 0 (0.0006 < 0.0008). Rounded to 1/1000 the two 0.0004
    terms are 0 and the 0.0006 term is 0.001, and y0 = 1 would look optimal."""
    U0, U1 = forced(4)
    i, j = ints([0, 0, 0]), ints([1, 2, 3])
    z = floats([0, 0, 0])
    P = Binary(U0, U1, i, j, z, floats([0, 0, 0.0006]), z, floats([0.0004, 0.0004, 0]))
    E = {y: binary_energy(P, y) for y in itertools.product([0, 1], repeat=4)}
    best = min(E.values())
    assert [y for y, e in E.items() if e <= best + 1e-15] == [(0, 1, 1, 1)]
    r = roof(P)
    assert r.weak[0] in (-1, 0) and r.strong[0] in (-1, 0)
    assert r.lower_bound <= best + 1e-12


def test_small_terms_do_not_change_the_labels_when_rounded_down() -> None:
    """As above with 0.0009 (twice, non-submodular) against 0.0011: the only
    minimum has y0 = 0 (0.0011 < 0.0018). Rounded down to 1/1000 the 0.0009
    terms are 0 and the 0.0011 term is 0.001, and y0 = 1 would look optimal."""
    U0, U1 = forced(4)
    i, j = ints([0, 0, 0]), ints([1, 2, 3])
    z = floats([0, 0, 0])
    P = Binary(U0, U1, i, j, z, floats([0, 0, 0.0011]), z, floats([0.0009, 0.0009, 0]))
    E = {y: binary_energy(P, y) for y in itertools.product([0, 1], repeat=4)}
    best = min(E.values())
    assert [y for y, e in E.items() if e <= best + 1e-15] == [(0, 1, 1, 1)]
    r = roof(P)
    assert r.weak[0] in (-1, 0) and r.strong[0] in (-1, 0)


# ------------------------------------------- an independent roof duality: dwave's


def bqm_of(P: Binary) -> dimod.BinaryQuadraticModel:
    """The same energy as a dimod binary quadratic model (finite tables only)."""
    b = dimod.BinaryQuadraticModel("BINARY")
    off = float(np.sum(P.U0))
    for v in range(P.n):
        b.add_linear(v, float(P.U1[v] - P.U0[v]))
    for i, j, A, B, C, D in zip(
        P.i.tolist(), P.j.tolist(), P.A, P.B, P.C, P.D, strict=True
    ):
        # A (1 - yi)(1 - yj) + B (1 - yi) yj + C yi (1 - yj) + D yi yj
        off += float(A)
        b.add_linear(i, float(C - A))
        b.add_linear(j, float(B - A))
        if i == j:
            b.add_linear(i, float(A - B - C + D))
        else:
            b.add_quadratic(i, j, float(A - B - C + D))
    b.offset = off
    return b


@pytest.mark.parametrize("seed", range(40))
def test_roof_against_dwave(seed: int) -> None:
    """dwave-preprocessing's roof_duality (Apache-2.0, its own implementation)
    on random models of up to 60 variables, too many for brute force: the same
    bound and the same strong labels. The costs are integers, so that no
    rounding error can break a tie between two minima: with costs such as
    0.123, the two solvers round differently, and where two minima differ
    only on a variable, each can label it strongly, to different values."""
    rng = np.random.default_rng(33000 + seed)
    n = int(rng.integers(2, 60))
    m = int(rng.integers(1, 3 * n))
    i, j = ints(rng.integers(0, n, m)), ints(rng.integers(0, n, m))
    tab = floats(rng.integers(-1000, 1001, size=(4, m)))
    U = floats(rng.integers(-1000, 1001, size=(2, n)))
    P = Binary(U[0], U[1], i, j, tab[0], tab[1], tab[2], tab[3])
    r = roof(P)
    lb, fixed = roof_duality(bqm_of(P), strict=True)
    assert abs(r.lower_bound - lb) < 1e-6  # dwave sums in floating point
    assert {v: int(r.strong[v]) for v in range(n) if r.strong[v] >= 0} == fixed


def local_polytope_bound(P: Binary) -> float:
    """The minimum of the local-polytope LP relaxation (finite tables only),
    by scipy's HiGHS. Variables: q_v = P(y_v = 1), and the four entries of each
    pair's table, which sum to 1 and have the q of the two variables as
    marginals. The tables of a pair are merged first: separate tables for one
    pair would each get their own entries, a looser relaxation. For binary
    energies this LP's minimum is the roof-duality bound (Hammer, Hansen and
    Simeone 1984)."""
    n = P.n
    const = float(np.sum(P.U0))
    lin = P.U1 - P.U0
    tables: dict[tuple[int, int], npt.NDArray[np.float64]] = {}
    for a, b, A, B, C, D in zip(
        P.i.tolist(), P.j.tolist(), P.A, P.B, P.C, P.D, strict=True
    ):
        if a == b:  # a term on one variable: A at y = 0, D at y = 1
            const += float(A)
            lin[a] += float(D - A)
            continue
        t = np.array([[A, B], [C, D]])  # t[y_a, y_b]
        key, t = ((a, b), t) if a < b else ((b, a), t.T)
        tables[key] = tables.get(key, np.zeros((2, 2))) + t
    m = len(tables)
    c = np.concatenate([lin, np.zeros(4 * m)])
    A_eq = np.zeros((3 * m, n + 4 * m))
    b_eq = np.zeros(3 * m)
    for k, ((a, b), t) in enumerate(tables.items()):
        cols = n + 4 * k + np.arange(4)  # entries 00, 01, 10, 11
        c[cols] = t.ravel()
        A_eq[3 * k, cols] = 1  # sum 1
        b_eq[3 * k] = 1
        A_eq[3 * k + 1, cols[[2, 3]]] = 1  # P(y_a = 1) = q_a
        A_eq[3 * k + 1, a] = -1
        A_eq[3 * k + 2, cols[[1, 3]]] = 1  # P(y_b = 1) = q_b
        A_eq[3 * k + 2, b] = -1
    res = linprog(
        c,
        A_eq=A_eq if m else None,
        b_eq=b_eq if m else None,
        bounds=(0, 1),
        method="highs",
    )
    assert res.status == 0 and res.fun is not None
    return const + float(res.fun)


@pytest.mark.parametrize("seed", range(20))
def test_roof_bound_is_the_local_polytope_bound(seed: int) -> None:
    """The roof bound equals the minimum of the local-polytope LP, up to the
    rounding (below it by at most the total rounding error, never above)."""
    rng = np.random.default_rng(34000 + seed)
    n = int(rng.integers(2, 40))
    m = int(rng.integers(1, 3 * n))
    i, j = ints(rng.integers(0, n, m)), ints(rng.integers(0, n, m))
    tab = np.round(rng.normal(size=(4, m)), 3)
    U = np.round(rng.normal(size=(2, n)), 3)
    P = Binary(U[0], U[1], i, j, tab[0], tab[1], tab[2], tab[3])
    lp = local_polytope_bound(P)
    lb = roof(P).lower_bound
    assert lb <= lp + 1e-7 and lp - lb < 1e-6
