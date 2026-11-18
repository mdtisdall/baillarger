"""Tests of the grow brush (label_grow)."""

from typing import Any

import numpy as np
import pytest
from scipy import ndimage as ndi

from baillarger import label_grow as LG

WM, CX = 41, 42
SHAPE = (40, 40, 40)
X0 = 20  # the true boundary: WM at x < X0


def _image(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Echoes 2 and 3 of two classes: WM bright, cortex darker, with noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(SHAPE[0])[:, None, None]
    wm = np.broadcast_to(x < X0, SHAPE)
    e2 = np.where(wm, 2000.0, 1000.0) * np.exp(0.05 * rng.standard_normal(SHAPE))
    e3 = np.where(wm, 1500.0, 900.0) * np.exp(0.05 * rng.standard_normal(SHAPE))
    return e2, e3


def _labels_with_bump() -> np.ndarray:
    """The labels: WM at x < X0, and a bump of WM into the cortex (x < 26)
    over y, z in [12, 28)."""
    lab = np.full(SHAPE, CX, np.int32)
    lab[:X0] = WM
    lab[X0:26, 12:28, 12:28] = WM
    return lab


def _cortex_strokes(step: int = 4) -> list[tuple[int, np.ndarray]]:
    """Cortex strokes on one z slice in step, over the bump."""
    out = []
    for z in list(range(12, 28, step)) + [27]:
        ii, jj = np.meshgrid(np.arange(X0 + 1, 26), np.arange(13, 27), indexing="ij")
        v = np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, z)], 1)
        out.append((CX, v))
    return out


def _run(
    lab: np.ndarray, strokes: list[tuple[int, np.ndarray]], **kw: Any
) -> LG.Proposal:
    e2, e3 = _image()

    def crop(a: np.ndarray, box: LG.Box) -> np.ndarray:
        (a0, b0, c0), (a1, b1, c1) = box
        return a[a0:a1, b0:b1, c0:c1]

    return LG.grow(
        strokes,
        lambda b: crop(lab, b),
        lambda b: (crop(e2, b), crop(e3, b)),
        shape=SHAPE,
        wm=WM,
        **kw,
    )


def _apply(lab: np.ndarray, p: LG.Proposal) -> np.ndarray:
    out = lab.copy()
    out.reshape(-1)[p.index] = p.label
    return out


def test_region_fills_between_slices() -> None:
    strokes = _cortex_strokes(4)
    reg = LG.region(strokes, np.zeros(3, np.int64), SHAPE)
    # every slice from the first stroke to the last is filled over the bump
    assert reg[X0 + 1 : 26, 14:26, 12:28].all()
    assert not reg[:, :, :12].any() and not reg[:, :, 28:].any()


def test_bump_is_removed_to_the_true_boundary() -> None:
    lab = _labels_with_bump()
    p = _run(lab, _cortex_strokes(4))
    out = _apply(lab, p)
    # inside the bump's footprint (away from its sides), the WM ends at X0
    core = out[:, 14:26, 14:26]
    wm_end = (core == WM).sum(axis=0)  # WM is x < wm_end on each column
    assert np.all(np.abs(wm_end - X0) <= 1)


def test_no_change_outside_the_area() -> None:
    lab = _labels_with_bump()
    strokes = _cortex_strokes(4)
    p = _run(lab, strokes)
    v = p.voxels(SHAPE)
    reg = LG.region(strokes, np.zeros(3, np.int64), SHAPE)
    dist = ndi.distance_transform_edt(~reg)
    assert np.all(dist[tuple(v.T)] <= LG.Settings().band)


def test_no_thin_wm_near_the_change() -> None:
    lab = _labels_with_bump()
    p = _run(lab, _cortex_strokes(4))
    out = _apply(lab, p)
    w = out == WM
    near = np.zeros(SHAPE, bool)
    near.reshape(-1)[p.index] = True
    near = ndi.binary_dilation(near, iterations=LG.Settings().zone)
    thin = w & ~ndi.binary_opening(w, ndi.generate_binary_structure(3, 1))
    assert not (thin & near).any()


def test_strokes_that_agree_give_no_proposal() -> None:
    lab = np.full(SHAPE, CX, np.int32)
    lab[:X0] = WM
    strokes = [(CX, np.array([[X0 + 3, 20, z] for z in (15, 19, 23)]))]
    p = _run(lab, strokes)
    assert len(p.index) == 0


def test_deterministic() -> None:
    lab = _labels_with_bump()
    a = _run(lab, _cortex_strokes(4))
    b = _run(lab, _cortex_strokes(4))
    assert np.array_equal(a.index, b.index) and np.array_equal(a.label, b.label)


def test_area_above_the_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="paint strokes closer together"):
        _run(_labels_with_bump(), _cortex_strokes(4), box_limit=100)


def test_later_stroke_wins_and_strokes_keep_their_labels() -> None:
    lab = _labels_with_bump()
    v = np.array([[22, 20, 20]])
    p = _run(lab, [(WM, v), (CX, v)] + _cortex_strokes(4))
    out = _apply(lab, p)
    assert out[22, 20, 20] == CX


def test_off_grid_and_empty_are_refused() -> None:
    with pytest.raises(ValueError, match="outside the grid"):
        _run(_labels_with_bump(), [(CX, np.array([[40, 0, 0]]))])
    with pytest.raises(ValueError, match="at least one"):
        _run(_labels_with_bump(), [(CX, np.zeros((0, 3), np.int64))])
