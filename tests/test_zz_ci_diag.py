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
import os
if os.environ.get("FIXNORM"):
    _norm = np.linalg.norm
    def norm(x, *a, **k):
        x = np.asarray(x)
        if x.ndim == 1 and not a and not k:
            return np.sqrt((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2])
        return _norm(x, *a, **k)
    np.linalg.norm = norm
V0, F0 = T.finger()
ref = SR.Reference(V0, F0, np.full(len(V0), 4.0))
rm = SR.Remesher(ref, V0.copy(), F0.copy(), T.GAP, theta_deg=30.0, d_out=0.5)
start = rm.measures(1.0)
log = SR.remesh(rm, 1.0, max_passes=4)
V, F, _ = rm.chk.export()
h = hashlib.sha1(np.ascontiguousarray(V).tobytes() + np.ascontiguousarray(F).tobytes()).hexdigest()[:12]
print(json.dumps(dict(dot=getattr(SR, "DOT_ORDER", None), hits=getattr(rm, "memo_hits", None), src=SR.__file__,
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


AVX512 = "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR AVX512VL AVX512BW AVX512DQ AVX512VNNI AVX512IFMA AVX512VBMI AVX512VBMI2 AVX512BITALG AVX512VPOPCNTDQ AVX512BF16 AVX512FP16"


def main_src(tmp: str) -> str:
    """A copy of src/ with main's versions of the three changed modules."""
    import shutil

    here = os.path.dirname(__file__)
    dst = os.path.join(tmp, "mainsrc")
    shutil.copytree(
        os.path.join(here, "..", "src"),
        dst,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for f in ("surface_remesh", "mesh_checks", "_jit"):
        shutil.copy(
            os.path.join(here, "_diag_main", f + ".py.txt"),
            os.path.join(dst, "baillarger", f + ".py"),
        )
    return dst


def test_ci_diag(tmp_path: object) -> None:
    tmp = str(tmp_path)
    rep: dict[str, object] = {"machine": machine()}
    ms = main_src(tmp)
    probe = subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "_diag_main", "probe.py.txt"),
        ],
        capture_output=True,
        text=True,
    )
    rep["blas_probe"] = probe.stdout.strip() or probe.stderr[-400:]
    variants = {
        "pr": {},
        "pr_fixnorm": {"FIXNORM": "1"},
        "main_fixnorm": {"PYTHONPATH": ms, "FIXNORM": "1"},
        "pr_openblas_haswell": {"OPENBLAS_CORETYPE": "Haswell"},
        "pr_np_no_avx512": {"NPY_DISABLE_CPU_FEATURES": AVX512},
        "main": {"PYTHONPATH": ms},
        "main_np_no_avx512": {"PYTHONPATH": ms, "NPY_DISABLE_CPU_FEATURES": AVX512},
    }
    for name, extra in variants.items():
        env = dict(os.environ, **extra)
        p = subprocess.run(
            [sys.executable, "-c", BODY], env=env, capture_output=True, text=True
        )
        rep[name] = (
            p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-800:]
        )
    pytest.fail("CIDIAG " + json.dumps(rep))
