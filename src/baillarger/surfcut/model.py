"""The model: ordered labels on one depth lattice, and its energy.

Variable i takes a label k in [0, K_i), at absolute position x_i = o_i + k_i
on a lattice shared by all variables (its own integer origin o_i). Terms, on
absolute positions:

- unary costs, one per label of each variable;
- hard bounds lo <= x_b - x_a <= hi;
- L1 hinges w |x_b - x_a - t|, w >= 0;
- ties w |x_a + x_b - L|, w >= 0: a sum, not a difference, so not
  submodular wherever ties and differences close a loop;
- pairwise tables theta(k_a, k_b), on labels. A table is submodular where
  every second difference
  D(i, j) = theta(i, j) - theta(i-1, j) - theta(i, j-1) + theta(i-1, j-1)
  is <= 0; w |x_b - g(x_a)| with g non-decreasing is.

Differences, hinges and submodular tables are exact by a minimum cut
(baillarger.surfcut.cut); ties and the rest of the tables need QPBO
(baillarger.surfcut.qpbo).
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

type Bool = npt.NDArray[np.bool_]
type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]
type Hard = tuple[Int, Int, Int, Int]  # a, b, lo, hi
type Hinge = tuple[Int, Int, Int, Float]  # a, b, t, w
type Tie = tuple[Int, Int, Int, Float]  # a, b, L, w
type Table = tuple[Int, Int, Float]  # a, b, theta (m, Ka_max, Kb_max), nan padding


@dataclass
class Model:
    """Variables (labels K, origins o, unary costs) and terms, each a list of
    arrays over terms (several lists are concatenated)."""

    K: Int
    o: Int
    unary: list[Float]
    hard: list[Hard] = field(default_factory=list)
    hinge: list[Hinge] = field(default_factory=list)
    tie: list[Tie] = field(default_factory=list)
    table: list[Table] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.K)


def ints(x: npt.ArrayLike) -> Int:
    return np.atleast_1d(np.asarray(x, dtype=np.int64))


def floats(x: npt.ArrayLike) -> Float:
    return np.atleast_1d(np.asarray(x, dtype=np.float64))


def hard_terms(M: Model) -> Hard:
    if not M.hard:
        e = ints([])
        return e, e, e, e
    return (
        np.concatenate([ints(t[0]) for t in M.hard]),
        np.concatenate([ints(t[1]) for t in M.hard]),
        np.concatenate([ints(t[2]) for t in M.hard]),
        np.concatenate([ints(t[3]) for t in M.hard]),
    )


def _sum_terms(terms: list[Hinge]) -> Hinge:
    if not terms:
        e = ints([])
        return e, e, e, floats([])
    return (
        np.concatenate([ints(t[0]) for t in terms]),
        np.concatenate([ints(t[1]) for t in terms]),
        np.concatenate([ints(t[2]) for t in terms]),
        np.concatenate([floats(t[3]) for t in terms]),
    )


def hinge_terms(M: Model) -> Hinge:
    return _sum_terms(M.hinge)


def tie_terms(M: Model) -> Tie:
    return _sum_terms(M.tie)


def energy(M: Model, k: npt.ArrayLike) -> float:
    """Total energy of labels k (n,), every term included; inf if a hard bound
    is broken."""
    lab = ints(k) if M.n else ints([])
    x = ints(M.o) + lab
    start = np.r_[0, np.cumsum(ints(M.K))]
    e = 0.0
    if M.n:
        e = float(np.concatenate([floats(u) for u in M.unary])[start[:-1] + lab].sum())
    a, b, lo, hi = hard_terms(M)
    if a.size and np.any((x[b] - x[a] < lo) | (x[b] - x[a] > hi)):
        return float("inf")
    a, b, t, w = hinge_terms(M)
    e += float(np.sum(w * np.abs(x[b] - x[a] - t)))
    a, b, L, w = tie_terms(M)
    e += float(np.sum(w * np.abs(x[a] + x[b] - L)))
    for ta, tb, T in M.table:
        ta, tb = ints(ta), ints(tb)
        e += float(np.sum(T[np.arange(ta.size), lab[ta], lab[tb]]))
    return e


def table_parts(a: Int, b: Int, T: Float, K: Int) -> tuple[Bool, Float, Float, Float]:
    """For tables over variables a, b: the valid entries, the unary parts
    theta(i, 0) - theta(0, 0) on a and theta(0, j) - theta(0, 0) on b, and
    the second differences D (i, j >= 1; 0 elsewhere), so that

        theta(i, j) = theta(0, 0) + ua(i) + ub(j) + sum_{i' <= i, j' <= j} D(i', j').
    """
    Ka, Kb = K[a][:, None, None], K[b][:, None, None]
    ii = np.arange(T.shape[1])[None, :, None]
    jj = np.arange(T.shape[2])[None, None, :]
    valid = (ii < Ka) & (jj < Kb)
    Tz = np.where(valid, T, 0.0)
    ua = Tz[:, :, 0] - Tz[:, :1, 0]
    ub = Tz[:, 0, :] - Tz[:, 0, :1]
    D = np.zeros_like(Tz)
    D[:, 1:, 1:] = Tz[:, 1:, 1:] - Tz[:, :-1, 1:] - Tz[:, 1:, :-1] + Tz[:, :-1, :-1]
    return valid, ua, ub, np.where(valid, D, 0.0)
