"""Tests of baillarger.surfcut with exact answers: brute force on small
models, dynamic programming on larger folded ones (checked against brute force
and OR-Tools CP-SAT)."""

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
    floats,
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


def pair_costs(M: Model) -> dict[tuple[int, int], Float]:
    """Every pairwise term of M as a cost matrix over (k_a, k_b), a < b, summed
    per pair; inf where a hard bound is broken."""
    x = [int(M.o[i]) + np.arange(int(K)) for i, K in enumerate(M.K)]
    P: dict[tuple[int, int], Float] = {}

    def add(a: int, b: int, C: Float) -> None:
        key, C = ((a, b), C) if a < b else ((b, a), C.T)
        P[key] = P.get(key, np.zeros_like(C, dtype=float)) + C

    for a, b, lo, hi in zip(*hard_terms(M), strict=True):
        d = x[b][None, :] - x[int(a)][:, None]
        add(int(a), int(b), np.where((d < lo) | (d > hi), np.inf, 0.0))
    for a, b, t, w in zip(*hinge_terms(M), strict=True):
        d = x[b][None, :] - x[a][:, None] - t
        add(int(a), int(b), w * np.abs(d))
    for a, b, L, w in zip(*tie_terms(M), strict=True):
        add(int(a), int(b), w * np.abs(x[a][:, None] + x[b][None, :] - L))
    for ta, tb, T in M.table:
        for j, (a, b) in enumerate(zip(ints(ta), ints(tb), strict=True)):
            add(int(a), int(b), T[j, : M.K[a], : M.K[b]].astype(float))
    return P


def ladder_min(M: Model, stages: list[tuple[int, ...]]) -> float:
    """The exact minimum energy by dynamic programming over stages (a partition
    of the variables into groups, in order) when every pairwise term lies within
    a stage or between consecutive stages: the state of a stage is the joint
    label of its variables. Checks that the minimiser's energy() agrees."""
    assert sorted(i for s in stages for i in s) == list(range(M.n))
    where = {i: p for p, s in enumerate(stages) for i in s}
    P = pair_costs(M)
    assert all(abs(where[a] - where[b]) <= 1 for a, b in P)
    labs = [
        np.array(list(itertools.product(*(range(int(M.K[i])) for i in s))))
        for s in stages
    ]

    def within(p: int) -> Float:
        s, lab = stages[p], labs[p]
        c = sum(floats(M.unary[i])[lab[:, q]] for q, i in enumerate(s))
        for q, a in enumerate(s):
            for r, b in enumerate(s):
                if (a, b) in P:
                    c = c + P[a, b][lab[:, q], lab[:, r]]
        return np.asarray(c, dtype=float)

    def between(p: int) -> Float:
        C = np.zeros((len(labs[p]), len(labs[p + 1])))
        for q, a in enumerate(stages[p]):
            for r, b in enumerate(stages[p + 1]):
                if (a, b) in P:
                    C += P[a, b][labs[p][:, q][:, None], labs[p + 1][:, r][None, :]]
                if (b, a) in P:
                    C += P[b, a][labs[p + 1][:, r][None, :], labs[p][:, q][:, None]]
        return C

    V, back = within(0), []
    for p in range(len(stages) - 1):
        T = V[:, None] + between(p)
        back.append(np.argmin(T, axis=0))
        V = within(p + 1) + T.min(axis=0)
    s = int(np.argmin(V))
    best = float(V[s])
    k = np.zeros(M.n, dtype=int)
    for p in range(len(stages) - 1, -1, -1):
        k[list(stages[p])] = labs[p][s]
        if p:
            s = int(back[p - 1][s])
    assert abs(energy(M, k) - best) < 1e-9 or not np.isfinite(best)
    return best


def fold_stages(M: Model) -> list[tuple[int, ...]]:
    """The stages of folded_model: facing columns (i, n - 1 - i)."""
    return [(i, M.n - 1 - i) for i in range(M.n // 2)]


def fold_min(M: Model, fix: tuple[np.ndarray, Int] | None = None) -> float:
    """The exact optimum of a folded_model, with the labels fix[1] kept where
    fix[0]: unary costs elsewhere set to inf."""
    if fix is not None:
        U = [
            np.where(f & (np.arange(len(u)) != lab), np.inf, u)
            for u, f, lab in zip(M.unary, fix[0], fix[1], strict=True)
        ]
        M = Model(M.K, M.o, U, M.hard, M.hinge, M.tie, M.table)
    return ladder_min(M, fold_stages(M))


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


@pytest.mark.parametrize("seed", range(6))
def test_fold_min_matches_brute_force(seed: int) -> None:
    M = folded_model(np.random.default_rng(9100 + seed), n_side=3, K=5)
    assert abs(fold_min(M) - brute(M)[0]) < 1e-9


@pytest.mark.parametrize("seed", range(4))
def test_fold_min_matches_cpsat(seed: int) -> None:
    """On folded models small enough that CP-SAT proves the optimum fast (under
    0.1 s here; the proof takes 2 to 11 s at the full size of 24 x 14)."""
    M = folded_model(np.random.default_rng(9200 + seed), n_side=6, K=8)
    assert abs(fold_min(M) - cpsat(M)) < 1e-6


@pytest.mark.parametrize("seed", range(8))
def test_qpbo_against_exact_folded(seed: int) -> None:
    M = folded_model(np.random.default_rng(9000 + seed))
    check_against(M, fold_min(M), lambda cert, lab: fold_min(M, (cert, lab)))


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
    assert abs(fold_box(M, r.interval_lo, r.interval_hi) - fold_min(M)) < 1e-6


def fold_box(M: Model, lo: Int, hi: Int) -> float:
    """fold_min() with lo <= k <= hi: unary costs outside the box set to inf."""
    U = [
        np.where((np.arange(K) < a) | (np.arange(K) > c), np.inf, u)
        for u, K, a, c in zip(M.unary, M.K, lo, hi, strict=True)
    ]
    return fold_min(Model(M.K, M.o, U, M.hard, M.hinge, M.tie, M.table))


def overlapping_levels(
    rng: np.random.Generator, n: int, L: int = 4, K: int = 13
) -> Model:
    """n columns in a chain, L levels each on overlapping windows (all origins 0), so
    that the ordering bounds force binary nodes: lateral hinges, ordering lo = 1
    between levels."""
    N = n * L
    unary: list[Float] = [np.round(rng.normal(size=K), 3) for _ in range(N)]
    a = np.arange(n - 1)
    hinge: list[Hinge] = []
    for lev in range(L):
        t = ints(rng.integers(-1, 2, n - 1))
        w = np.round(rng.uniform(0.1, 1, n - 1), 3)
        hinge.append((ints(lev * n + a), ints(lev * n + a + 1), t, w))
    c = np.arange(n)
    hard: list[Hard] = []
    for lev in range(L - 1):
        lo, hi = ints(np.full(n, 1)), ints(np.full(n, 10**6))
        hard.append((ints(lev * n + c), ints((lev + 1) * n + c), lo, hi))
    return Model(ints(np.full(N, K)), ints(np.zeros(N)), unary, hard, hinge)


def test_qpbo_bound_is_precise_with_forced_nodes() -> None:
    """Certified everywhere (submodular), so the bound equals the optimum up to
    rounding; a hard cost 1000 times too large made it 1.3e-2 off here."""
    M = overlapping_levels(np.random.default_rng(2000), 2000)
    r = qpbo(M)
    assert r.certified.all()
    assert abs(r.lower_bound - r.energy) <= 1e-6 * abs(r.energy)


@pytest.mark.parametrize("seed", range(30))
def test_reduce_default_big_with_large_tables(seed: int) -> None:
    """With reduce's own big cost and tables 100 times larger than the other terms,
    the reduced model's optimum plus its constant equals the full optimum with the
    same variables fixed."""
    rng = np.random.default_rng(18000 + seed)
    M = random_model(rng, ties=True, tables="any")
    M.table = [(a, b, 100.0 * T) for a, b, T in M.table]
    fixed = rng.random(M.n) < 0.5
    lab = ints([rng.integers(0, K) for K in M.K])
    full = restricted_brute(M)(fixed, lab)
    if not np.isfinite(full):
        return
    R, free, const = reduce(M, fixed, lab)
    best_r, _ = brute(R) if R.n else (energy(R, ints([])), None)
    assert abs(best_r + const - full) < 1e-6
