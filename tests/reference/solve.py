"""The antialiasing problem (baillarger.antialias) as an explicit quadratic program,
solved by CVXPY with Clarabel at tight tolerances: the reference for
tests/test_antialias.py, run in this environment of its own (README.md).

Input (npz): fixed_idx, fixed_val (voxels held at a value), pos_idx (voxels >= eps),
neg_idx (voxels <= -eps), eps, n (voxels), and per stencil s its CSR matrix (D{s}_data,
D{s}_indices, D{s}_indptr) and weight w{s}, for s in 0..count-1, count.
Output (npz): phi (n,), value (the optimal energy), status.

Usage: solve.py IN.npz OUT.npz
"""

import sys

import cvxpy as cp
import numpy as np
from scipy import sparse


def main(src: str, dst: str) -> None:
    z = np.load(src)
    n = int(z["n"])
    eps = float(z["eps"])
    mats = []
    for s in range(int(z["count"])):
        D = sparse.csr_matrix(
            (z[f"D{s}_data"], z[f"D{s}_indices"], z[f"D{s}_indptr"]), shape=(n, n)
        )
        mats.append((D, float(z[f"w{s}"])))
    phi = cp.Variable(n)
    cons = [
        phi[z["fixed_idx"]] == z["fixed_val"],
        phi[z["pos_idx"]] >= eps,
        phi[z["neg_idx"]] <= -eps,
    ]
    obj = cp.Minimize(sum(w * cp.sum_squares(D @ phi) for D, w in mats))
    prob = cp.Problem(obj, cons)
    # tight tolerances: at Clarabel's defaults the field stops short of the optimum by
    # enough to move a vertex on an edge between two clamped voxels by 7e-3 voxel
    prob.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-12,
        tol_gap_rel=1e-12,
        tol_feas=1e-12,
        max_iter=500,
    )
    np.savez(dst, phi=np.asarray(phi.value), value=prob.value, status=prob.status)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
