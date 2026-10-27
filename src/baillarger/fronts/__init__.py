"""Anisotropic fast marching from labelled sources.

A self-contained tool: it imports nothing from the rest of baillarger, so it
can become a package of its own.

- march: arrival from labelled sources under a Riemannian metric, with each
  voxel's source label, the Euclidean length of its path, and the best
  arrival of any other label (for the meeting surface between fronts).
- riemann_from_direction: a metric with unit cost along a direction field.
- selling: Selling's decomposition of 3 x 3 symmetric positive definite
  matrices, which gives the adaptive stencils.
"""

from baillarger.fronts.march import Fronts, march, riemann_from_direction
from baillarger.fronts.selling import selling

__all__ = ["Fronts", "march", "riemann_from_direction", "selling"]
