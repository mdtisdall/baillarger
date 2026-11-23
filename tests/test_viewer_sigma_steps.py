"""Tests for baillarger.viewer.sigma_steps: Σ by step as outlines."""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from test_decimate import sphere

from baillarger.viewer import outlines as VO
from baillarger.viewer import sigma_steps as VS


def _write(d: Path, step: int, v: Any, f: Any) -> None:
    G: Any = nib.gifti
    img = G.GiftiImage(
        darrays=[
            G.GiftiDataArray(
                np.asarray(v, np.float32),
                intent="NIFTI_INTENT_POINTSET",
                datatype="NIFTI_TYPE_FLOAT32",
            ),
            G.GiftiDataArray(
                np.asarray(f, np.int32),
                intent="NIFTI_INTENT_TRIANGLE",
                datatype="NIFTI_TYPE_INT32",
            ),
        ]
    )
    nib.save(img, d / f"sub-X_hemi-R_desc-sigma_step-{step:03d}.surf.gii")


def test_steps_are_listed_in_order_and_cut_like_the_outline(tmp_path: Path) -> None:
    v, f = sphere()
    _write(tmp_path, 10, v * 1.1, f)
    _write(tmp_path, 2, v, f)
    (tmp_path / "other.surf.gii").write_text("not a step")
    s = VS.SigmaSteps(tmp_path)
    assert [r["step"] for r in s.list()] == [2, 10]
    lo, hi = v.min(0) - 1, v.max(0) + 1
    mm = float(v[:, 2].mean())
    o = s.cut(2, 2, mm, lo, hi)
    want = VO.plane_cut(np.asarray(v, np.float32).astype(np.float64), f, 2, mm)
    assert o.source == "sigma" and len(o.segments) == len(want) > 0
    assert np.allclose(
        np.sort(o.segments.reshape(-1, 3), 0),
        np.sort(want.reshape(-1, 3), 0),
        atol=1e-5,
    )
    # the larger sphere of step 10 gives a longer outline; a small box keeps
    # only the segments in it
    assert (
        np.linalg.norm(
            np.diff(s.cut(10, 2, mm * 1.1, lo * 1.2, hi * 1.2).segments, axis=1), axis=2
        ).sum()
        > np.linalg.norm(np.diff(o.segments, axis=1), axis=2).sum()
    )
    c = v.mean(0)
    part = s.cut(2, 2, mm, c - [20, 20, 20], [c[0], c[1], c[2] + 20])
    assert 0 < len(part.segments) < len(o.segments)
    assert s.list()[0]["faces"] == len(f)


def test_route_answers(tmp_path: Path) -> None:
    v, f = sphere()
    _write(tmp_path, 0, v, f)

    class H:
        sigma = VS.SigmaSteps(tmp_path)
        sent: list[Any] = []

        def json(self, fn: Any) -> None:
            self.sent.append(fn())

        def send_error(self, code: int) -> None:
            self.sent.append(code)

        def send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.sent.append((code, body))

    h = H()
    assert not VS.route(h, "/outline", {})
    assert VS.route(h, "/sigma/steps", {}) and h.sent[-1][0]["step"] == 0
    assert (
        VS.route(
            h,
            "/sigma/outline",
            {"step": "7", "axis": "0", "mm": "1", "box": "0,0,0,1,1,1"},
        )
        and h.sent[-1] == 404
    )
    assert (
        VS.route(
            h,
            "/sigma/outline",
            {"step": "0", "axis": "5", "mm": "1", "box": "0,0,0,1,1,1"},
        )
        and h.sent[-1][0] == 400
    )
