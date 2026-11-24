"""Test 10 of the plan (reports/Pial editing and edit priority.md, §5): timing of the
correction field on the patches' white meshes. A measurement, not a pass criterion.

For each mesh (finding 28's twicing 0.32 mm meshes, world mm): the matrices, the
factorisation (load, and again for a change of range), and per edit: one observation
added (its column Q^-1 a) and the correction over all vertices, for 20 edits at random
mesh points; the vertices the edit moved by more than 0.001 mm (what the page must
update). Peak memory.

Usage: solver_timing.py <out json> <mesh.vtk> [<mesh.vtk> ...]
"""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import solver as S  # noqa: E402
from server import read_vtk  # noqa: E402


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main() -> None:
    out_path, *paths = sys.argv[1:]
    res = {}
    rng = np.random.default_rng(10)
    for p in paths:
        v, f = read_vtk(p)
        v = v.astype(np.float64)
        t = time.perf_counter()
        pr = S.Prior(v, f.astype(np.int64), 2.0, 0.5)
        load = time.perf_counter() - t
        # correctness of the factor: relative residual of L x = b for a random b
        from scipy import sparse as sp

        Lw = (pr.kappa**2 * sp.diags(pr.mass) + pr.K).tocsr()
        b = rng.normal(size=Lw.shape[0])
        resid = float(np.linalg.norm(Lw @ pr.lu.solve(b) - b) / np.linalg.norm(b))
        r = dict(
            vertices=len(v),
            welded_away=pr.n_welded,
            relative_residual=resid,
            faces=len(f),
            seconds_matrices=round(pr.seconds_matrices, 2),
            seconds_factor=round(pr.seconds_factor, 2),
            seconds_load=round(load, 2),
            factor_nnz=int(pr.lu.L.nnz + pr.lu.U.nnz),
            peak_gb_after_load=round(peak_gb(), 2),
        )
        fd = S.Field(pr, np.full(len(v), 2.5))
        add_ms, corr_ms, moved = [], [], []
        prev = np.zeros(len(v))
        for _ in range(20):
            tri = rng.integers(len(f))
            w = rng.dirichlet(np.ones(3))
            t = time.perf_counter()
            fd.add(f[tri], w, 2.5 + rng.normal(0, 0.3))
            t1 = time.perf_counter()
            c = fd.correction()
            t2 = time.perf_counter()
            add_ms.append(1e3 * (t1 - t))
            corr_ms.append(1e3 * (t2 - t1))
            moved.append(int(np.sum(np.abs(c - prev) > 1e-3)))
            prev = c
        t = time.perf_counter()
        pr.set_range(1.0, 0.5)
        fd.refresh()
        r.update(
            add_ms_median=round(float(np.median(add_ms)), 1),
            add_ms_max=round(float(np.max(add_ms)), 1),
            correction_ms_at_1_10_20=[round(corr_ms[i], 1) for i in (0, 9, 19)],
            moved_vertices_median=int(np.median(moved)),
            moved_vertices_max=int(np.max(moved)),
            seconds_change_range_with_20_edits=round(time.perf_counter() - t, 2),
            peak_gb=round(peak_gb(), 2),
        )
        res[Path(p).name] = r
        print(Path(p).name, json.dumps(r), flush=True)
    Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
