# ruff: noqa: E501
"""Throwaway CI diagnostic for PR #247 (not to merge): where the finger
test's result changes. Fails on purpose so that the log shows the report."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from baillarger import surface_remesh as SR

BODY = r"""
import hashlib, json, sys
import numpy as np
sys.path.insert(0, "tests")
import test_surface_remesh as T
from baillarger import surface_remesh as SR
V0, F0 = T.finger()
ref = SR.Reference(V0, F0, np.full(len(V0), 4.0))
rm = SR.Remesher(ref, V0.copy(), F0.copy(), T.GAP, theta_deg=30.0, d_out=0.5)
start = rm.measures(1.0)
log = SR.remesh(rm, 1.0, max_passes=4)
V, F = rm.chk.V, rm.chk.F
h = hashlib.sha1(np.ascontiguousarray(V).tobytes() + np.ascontiguousarray(F).tobytes()).hexdigest()[:12]
print(json.dumps(dict(dot=SR.DOT_ORDER, hits=rm.memo_hits,
    angle=[start["face_angle_p50_90_99_max"][3], log[-1]["face_angle_p50_90_99_max"][3]],
    passes=[[r.get("splits"), r.get("collapses"), r.get("flips")] for r in log], hash=h)))
"""


def machine() -> dict[str, object]:
    out: dict[str, object] = {}
    try:
        info = open("/proc/cpuinfo").read().split("\n\n")[0]
        for line in info.splitlines():
            k, _, v = line.partition(":")
            if k.strip() == "model name":
                out["cpu"] = v.strip()
            if k.strip() == "flags":
                fl = set(v.split())
                out["flags"] = sorted(
                    fl & {"avx", "avx2", "fma", "avx512f", "avx512vl", "avx512dq"}
                )
    except OSError:
        pass
    import llvmlite.binding as llvm

    out["llvm_cpu"] = llvm.get_host_cpu_name()
    from numpy._core._multiarray_umath import __cpu_features__ as cf  # type: ignore[import-not-found]

    out["np_simd"] = sorted(
        k for k, v in cf.items() if v and k.startswith(("AVX", "FMA", "SSE4"))
    )
    out["dot_order"] = SR.DOT_ORDER
    try:
        out["blas"] = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception as e:  # noqa: BLE001
        out["blas"] = repr(e)
    return out


@pytest.mark.parametrize("case", ["a", "b", "c", "d"])
def test_ci_diag(case: str) -> None:
    rep: dict[str, object] = {"case": case, "machine": machine()}
    runs = []
    for seed in range(3):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        p = subprocess.run(
            [sys.executable, "-c", BODY], env=env, capture_output=True, text=True
        )
        runs.append(
            p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-500:]
        )
    rep["runs"] = runs
    rep["same_in_all"] = len(set(runs)) == 1
    pytest.fail("CIDIAG " + json.dumps(rep))
