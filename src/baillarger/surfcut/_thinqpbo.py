"""thinqpbo's QPBODouble with a typed interface.

thinqpbo (Kolmogorov's QPBO, double precision) has no type stubs, so mypy
sees it as untyped. The protocol lists the methods used here.
"""

import ctypes
from typing import Protocol, cast

_LIBC = ctypes.CDLL(None)


class QPBOGraph(Protocol):
    def add_node(self, num: int) -> int: ...
    def add_unary_term(self, i: int, e0: float, e1: float) -> None: ...
    def add_pairwise_term(
        self, i: int, j: int, e00: float, e01: float, e10: float, e11: float
    ) -> int: ...
    def merge_parallel_edges(self) -> None: ...
    def solve(self) -> None: ...
    def compute_weak_persistencies(self) -> None: ...
    def get_label(self, i: int) -> int: ...
    def set_label(self, i: int, label: int) -> None: ...
    def improve(self) -> bool: ...
    def compute_twice_lower_bound(self) -> float: ...


def new_graph(nodes: int, edges: int) -> QPBOGraph:
    import thinqpbo

    return cast(QPBOGraph, thinqpbo.QPBODouble(nodes, edges))


def seed_rand(seed: int) -> None:
    """Seed C rand(), from which QPBO's Improve() (called without an order, the
    only form thinqpbo exposes) draws its order of variables."""
    _LIBC.srand(ctypes.c_uint(seed & 0xFFFFFFFF))
