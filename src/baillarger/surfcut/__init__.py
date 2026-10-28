"""Layered surfaces as labels: exact minimum cuts, and QPBO with ties.

A self-contained tool: it imports nothing from the rest of baillarger, so it
can become a package of its own.

Each variable is a depth on a lattice (its own window of labels); terms are
unary costs, hard bounds and L1 hinges on differences of two depths, ties on
their sum, and pairwise tables (baillarger.surfcut.model).

- cut: the exact minimum of the submodular terms by one minimum s-t cut
  (Ishikawa 2003), scipy's maximum flow.
- qpbo: every term, ties included, by QPBO (thinqpbo): a partial labelling
  that agrees with some global minimum, each variable's interval of labels
  it could not rule out, a lower bound, and QPBO-I's completion. Exact on
  submodular models.
- reduce, qpbo_pruned: fix variables exactly and solve the rest.
"""

from baillarger.surfcut.cut import cut
from baillarger.surfcut.model import Model, energy
from baillarger.surfcut.qpbo import Result, qpbo, qpbo_pruned, reduce

__all__ = ["Model", "Result", "cut", "energy", "qpbo", "qpbo_pruned", "reduce"]
