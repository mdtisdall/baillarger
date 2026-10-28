"""Tests of baillarger.surfcut with exact answers: brute force on small
models, OR-Tools CP-SAT on larger ones."""

import itertools
from collections.abc import Callable

import numpy as np
import pytest

from baillarger.surfcut import Model, cut, energy, qpbo, qpbo_pruned, reduce
from baillarger.surfcut.model import (
    Float,
    Hard,
    Hinge,
    Int,
    Table,
    Tie,
    hard_terms,
    hinge_terms,
    ints,
    tie_terms,
)


def brute(M: Model) -> tuple[float, Int | None]:
    best, arg = float("inf"), None
    for k in itertools.product(*[range(int(K)) for K in M.K]):
        e = energy(M, np.array(k))
        if e < best - 1e-12:
            best, arg = e, np.array(k)
    return best, arg


def random_tables(
    rng: np.random.Generator, K: Int, pairs: list[tuple[int, int]], submodular: bool
) -> list[Table]:
    """Random pairwise tables; submodular ones as w |k_b - g(k_a)| with g
    non-decreasing, the others random."""
    if not pairs:
        return []
    T = np.full((len(pairs), int(K.max()), int(K.max())), np.nan)
    for p_, (aa, bb) in enumerate(pairs):
        if submodular:
            g = np.cumsum(rng.integers(0, 3, K[aa])) - int(rng.integers(0, 3))
            w = round(float(rng.uniform(0.1, 1.0)), 3)
            T[p_, : K[aa], : K[bb]] = w * np.abs(np.arange(K[bb])[None] - g[:, None])
        else:
            T[p_, : K[aa], : K[bb]] = np.round(rng.normal(size=(K[aa], K[bb])), 3)
    return [(ints([p[0] for p in pairs]), ints([p[1] for p in pairs]), T)]


def random_model(
    rng: np.random.Generator, ties: bool = False, tables: str | None = None
) -> Model:
    n = int(rng.integers(2, 5))
    K = ints(rng.integers(2, 6, n))
    o = ints(rng.integers(-2, 3, n))
    unary = [np.round(rng.normal(size=int(k)), 3) for k in K]
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    hard: list[Hard] = []
    hinge: list[Hinge] = []
    tie: list[Tie] = []
    for a, b in pairs[: int(rng.integers(0, len(pairs) + 1))]:
        lo = int(rng.integers(-3, 2))
        hard.append((ints(a), ints(b), ints(lo), ints(lo + int(rng.integers(0, 4)))))
    for a, b in pairs[: int(rng.integers(1, len(pairs) + 1))]:
        hinge.append(
            (
                ints(a),
                ints(b),
                ints(int(rng.integers(-2, 3))),
                np.array([round(float(rng.uniform(0, 1.5)), 3)]),
            )
        )
    if ties:
        for a, b in pairs[: int(rng.integers(1, 3))]:
            tie.append(
                (
                    ints(a),
                    ints(b),
                    ints(int(rng.integers(0, 6))),
                    np.array([round(float(rng.uniform(0.2, 1.5)), 3)]),
                )
            )
    tab = (
        random_tables(
            rng, K, pairs[: int(rng.integers(1, len(pairs) + 1))], tables == "sub"
        )
        if tables
        else []
    )
    return Model(K, o, unary, hard, hinge, tie, tab)


def restricted_brute(M: Model) -> Callable[[np.ndarray, Int], float]:
    def f(cert: np.ndarray, lab: Int) -> float:
        b = float("inf")
        for k in itertools.product(
            *[
                range(int(K)) if not c else [int(lv)]
                for K, c, lv in zip(M.K, cert, lab, strict=True)
            ]
        ):
            b = min(b, energy(M, np.array(k)))
        return b

    return f


def check_against(
    M: Model, best: float, restricted: Callable[[np.ndarray, Int], float]
) -> None:
    r = qpbo(M)
    assert r.chain_ok
    assert abs(energy(M, r.labels) - r.energy) < 1e-9
    assert r.energy >= best - 1e-9  # a labelling, never below the optimum
    assert r.lower_bound <= best + 1e-6  # a valid bound
    if r.certified.any():  # persistency: fixing the certified labels keeps the optimum
        assert abs(restricted(r.certified, r.certified_labels) - best) < 1e-6


# ---------------------------------------------------------------- the cut


@pytest.mark.parametrize("seed", range(40))
def test_cut_matches_brute_force(seed: int) -> None:
    M = random_model(np.random.default_rng(seed))
    best, _ = brute(M)
    if not np.isfinite(best):
        with pytest.raises(ValueError):
            cut(M)
        return
    assert abs(energy(M, cut(M)) - best) < 1e-6


def test_hinge_target_outside_both_ranges() -> None:
    M = Model(
        ints([3, 3]),
        ints([5, 0]),
        [np.zeros(3), np.array([0.0, 0.1, 0.2])],
        [],
        [(ints(0), ints(1), ints(-9), np.array([1.0]))],
    )
    best, _ = brute(M)
    assert abs(energy(M, cut(M)) - best) < 1e-9


@pytest.mark.parametrize("seed", range(30))
def test_cut_with_submodular_tables(seed: int) -> None:
    M = random_model(np.random.default_rng(13000 + seed), tables="sub")
    best, _ = brute(M)
    if np.isfinite(best):
        assert abs(energy(M, cut(M)) - best) < 1e-6


# ---------------------------------------------------------------- QPBO


@pytest.mark.parametrize("seed", range(60))
def test_qpbo_against_brute_force(seed: int) -> None:
    M = random_model(np.random.default_rng(7000 + seed), ties=True)
    best, _ = brute(M)
    if np.isfinite(best):
        check_against(M, best, restricted_brute(M))


@pytest.mark.parametrize("seed", range(20))
def test_qpbo_without_ties_is_exact(seed: int) -> None:
    M = random_model(np.random.default_rng(8000 + seed))
    best, _ = brute(M)
    if not np.isfinite(best):
        return
    r = qpbo(M)
    assert r.certified.all()
    assert abs(r.energy - best) < 1e-6
    assert abs(energy(M, r.certified_labels) - best) < 1e-6


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("kind", ["sub", "any"])
def test_qpbo_with_tables(seed: int, kind: str) -> None:
    M = random_model(np.random.default_rng(14000 + seed), ties=True, tables=kind)
    best, _ = brute(M)
    if np.isfinite(best):
        check_against(M, best, restricted_brute(M))


@pytest.mark.parametrize("seed", range(20))
def test_qpbo_submodular_tables_exact(seed: int) -> None:
    M = random_model(np.random.default_rng(15000 + seed), tables="sub")
    best, _ = brute(M)
    if not np.isfinite(best):
        return
    r = qpbo(M)
    assert r.certified.all() and abs(r.energy - best) < 1e-6


@pytest.mark.parametrize("seed", range(20))
def test_qpbo_never_ends_above_its_start(seed: int) -> None:
    rng = np.random.default_rng(17000 + seed)
    M = random_model(rng, ties=True, tables="any")
    best, _ = brute(M)
    if not np.isfinite(best):
        return
    start = cut(M)
    assert qpbo(M, start_labels=start).energy <= energy(M, start) + 1e-12


# ---------------------------------------------------------------- reduce and pruning


@pytest.mark.parametrize("seed", range(30))
def test_reduce_is_exact(seed: int) -> None:
    """Reduced energy + constant equals the full energy, wherever the bounds
    between fixed variables hold (reduce's contract)."""
    rng = np.random.default_rng(11000 + seed)
    M = random_model(rng, ties=True, tables="any")
    fixed = rng.random(M.n) < 0.5
    lab = ints([rng.integers(0, K) for K in M.K])
    R, free, const = reduce(M, fixed, lab, big=1e6)
    x = M.o + lab
    a, b, lo, hi = hard_terms(M)
    ff = fixed[a] & fixed[b]
    if np.any((x[b][ff] - x[a][ff] < lo[ff]) | (x[b][ff] - x[a][ff] > hi[ff])):
        return
    for _ in range(20):
        k = lab.copy()
        k[free] = [rng.integers(0, K) for K in R.K]
        e = energy(M, k)
        er = energy(R, k[free]) + const
        if np.isfinite(e):
            assert abs(e - er) < 1e-6
        else:
            assert er >= 1e5  # a broken bound touching a free variable costs `big`


@pytest.mark.parametrize("seed", range(20))
def test_pruned_brackets_the_optimum(seed: int) -> None:
    M = random_model(np.random.default_rng(12000 + seed), ties=True)
    best, _ = brute(M)
    if not np.isfinite(best):
        return
    r = qpbo_pruned(M)
    assert r.energy >= best - 1e-9
    assert r.lower_bound <= best + 1e-6
    assert abs(energy(M, r.labels) - r.energy) < 1e-9


# ------------------------------------------------ CP-SAT on larger models


def cpsat(M: Model, fix: tuple[np.ndarray, Int] | None = None) -> float:
    """The exact optimum by CP-SAT (costs are multiples of 0.001)."""
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    S = 1000
    k = [m.new_int_var(0, int(K) - 1, f"k{i}") for i, K in enumerate(M.K)]
    obj: list[cp_model.LinearExpr] = []
    for i in range(M.n):
        u = m.new_int_var(-(10**9), 10**9, f"u{i}")
        m.add_element(k[i], [round(float(c) * S) for c in M.unary[i]], u)
        obj.append(u)
        if fix is not None and fix[0][i]:
            m.add(k[i] == int(fix[1][i]))
    for a, b, lo, hi in zip(*hard_terms(M), strict=True):
        d = k[b] - k[a] + int(M.o[b] - M.o[a])
        m.add(d >= int(lo))
        m.add(d <= int(hi))
    for j, (a, b, t, w) in enumerate(zip(*hinge_terms(M), strict=True)):
        d = m.new_int_var(-(10**6), 10**6, f"d{j}")
        m.add(d == k[b] - k[a] + int(M.o[b] - M.o[a]) - int(t))
        ab = m.new_int_var(0, 10**6, f"ab{j}")
        m.add_abs_equality(ab, d)
        obj.append(round(float(w) * S) * ab)
    for j, (a, b, L, w) in enumerate(zip(*tie_terms(M), strict=True)):
        d = m.new_int_var(-(10**6), 10**6, f"t{j}")
        m.add(d == k[a] + k[b] + int(M.o[a] + M.o[b]) - int(L))
        ab = m.new_int_var(0, 10**6, f"at{j}")
        m.add_abs_equality(ab, d)
        obj.append(round(float(w) * S) * ab)
    m.minimize(sum(obj))
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = 60
    sol.parameters.num_workers = 4
    assert sol.solve(m) == cp_model.OPTIMAL
    return float(sol.objective_value) / S


def folded_model(rng: np.random.Generator, n_side: int = 12, K: int = 14) -> Model:
    """Two banks of n_side columns in one chain, joined at the fold: lateral
    hinges and hard bounds, and ties (sums) between facing columns."""
    n = 2 * n_side
    unary: list[Float] = [np.round(rng.normal(size=K) * 0.6, 3) for _ in range(n)]
    hinge: list[Hinge] = [
        (
            ints(i),
            ints(i + 1),
            ints(int(rng.integers(-1, 2))),
            np.array([round(float(rng.uniform(0.2, 0.8)), 3)]),
        )
        for i in range(n - 1)
    ]
    hard: list[Hard] = [(ints(i), ints(i + 1), ints(-3), ints(3)) for i in range(n - 1)]
    tie: list[Tie] = [
        (
            ints(i),
            ints(n - 1 - i),
            ints(int(rng.integers(K - 2, K + 4))),
            np.array([round(float(rng.uniform(0.3, 1.0)), 3)]),
        )
        for i in range(n_side - 2)
    ]
    return Model(
        ints(np.full(n, K)), ints(rng.integers(0, 3, n)), unary, hard, hinge, tie
    )


@pytest.mark.parametrize("seed", range(8))
def test_qpbo_against_cpsat_folded(seed: int) -> None:
    M = folded_model(np.random.default_rng(9000 + seed))
    check_against(M, cpsat(M), lambda cert, lab: cpsat(M, (cert, lab)))


# ------------------------------------------------ ambiguity intervals


def box_brute(M: Model, lo: Int, hi: Int) -> float:
    """The minimum over labellings with lo <= k <= hi."""
    b = float("inf")
    for k in itertools.product(
        *[range(int(a), int(c) + 1) for a, c in zip(lo, hi, strict=True)]
    ):
        b = min(b, energy(M, np.array(k)))
    return b


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("tables", [None, "any"])
def test_intervals_hold_a_minimum(seed: int, tables: str | None) -> None:
    """Weak persistency: some global minimum lies in every variable's
    interval at once; certified variables have a zero-width interval at
    their label."""
    M = random_model(np.random.default_rng(18000 + seed), ties=True, tables=tables)
    best, _ = brute(M)
    if not np.isfinite(best):
        return
    r = qpbo(M)
    assert np.all((0 <= r.interval_lo) & (r.interval_lo <= r.interval_hi))
    assert np.all(r.interval_hi < M.K)
    c = r.certified
    assert np.array_equal(r.interval_lo[c], r.certified_labels[c])
    assert np.array_equal(r.interval_hi[c], r.certified_labels[c])
    assert abs(box_brute(M, r.interval_lo, r.interval_hi) - best) < 1e-6


def test_intervals_of_an_unlabelled_variable() -> None:
    """Three binary variables, each pair tied by w |x_a + x_b - 1| (prefers
    differing labels): an odd frustrated cycle, where roof duality labels no
    node, so each interval is its whole window."""
    pairs = [(0, 1), (1, 2), (0, 2)]
    M = Model(
        ints([2, 2, 2]),
        ints([0, 0, 0]),
        [np.zeros(2) for _ in range(3)],
        tie=[(ints(a), ints(b), ints(1), np.array([1.0])) for a, b in pairs],
    )
    r = qpbo(M)
    assert not r.certified.any()
    assert list(r.interval_lo) == [0, 0, 0] and list(r.interval_hi) == [1, 1, 1]


@pytest.mark.parametrize("seed", range(8))
def test_intervals_hold_a_minimum_folded(seed: int) -> None:
    M = folded_model(np.random.default_rng(9000 + seed))
    r = qpbo(M)
    fix = r.interval_lo == r.interval_hi
    assert np.array_equal(fix, r.certified)
    assert abs(cpsat_box(M, r.interval_lo, r.interval_hi) - cpsat(M)) < 1e-6


def cpsat_box(M: Model, lo: Int, hi: Int) -> float:
    """cpsat() with lo <= k <= hi: unary costs outside the box made large."""
    big = 1e4
    U = [
        np.where((np.arange(K) < a) | (np.arange(K) > c), big, u)
        for u, K, a, c in zip(M.unary, M.K, lo, hi, strict=True)
    ]
    return cpsat(Model(M.K, M.o, U, M.hard, M.hinge, M.tie, M.table))


@pytest.mark.parametrize("pruned", [False, True])
def test_qpbo_does_not_depend_on_the_c_rand_state(pruned: bool) -> None:
    """QPBO-I's order of variables comes from C rand(); a model on which it matters
    (folded, seed 1: 3 labellings under 5 srand values) gives the same labels
    whatever the process's rand state was before the call."""
    import ctypes

    libc = ctypes.CDLL(None)
    M = folded_model(np.random.default_rng(1))
    solve = qpbo_pruned if pruned else qpbo
    out = []
    for s in (1, 2, 3, 7, 12345):
        libc.srand(s)
        out.append(tuple(solve(M).labels))
    assert len(set(out)) == 1
