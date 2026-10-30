# Reference solvers for the tests

An environment of its own (`pyproject.toml`, `uv.lock`) for solvers that the tests compare
baillarger's own solvers with, but that must not share a process with the main environment:
CVXPY installs `highspy` (HiGHS), and once it is loaded OR-Tools' CP-SAT, which bundles a
different HiGHS, fails to import (a symbol clash). The tests run these scripts as
subprocesses:

    uv run --frozen --project tests/reference python tests/reference/solve.py IN.npz OUT.npz

- `solve.py`: the antialiasing problem (`baillarger.antialias`) as an explicit quadratic
  program, solved by CVXPY with Clarabel (an interior-point method) at tight tolerances.
