"""Exact tests of the site answers of the WM review."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import scipy.ndimage as ndi

from baillarger.wm_choices import (
    CHOICES,
    Change,
    apply,
    box_round,
    fill_gap,
    leave,
    preview,
    remove_strand,
    run,
    support,
    thicken,
    widen_gap,
)

WM, CTX = 41, 42
Site = tuple[int, int, int]


def _cortex(n: int = 21) -> Any:
    return np.full((n, n, n), CTX, np.int64)


def _ball_set(site: Site, reach: int, cells: set[Site]) -> set[Site]:
    return {
        c
        for c in cells
        if sum((a - b) ** 2 for a, b in zip(c, site, strict=True)) <= reach * reach
    }


def _local(site: Site, off: Any) -> Site:
    a, b, c = (int(k) for k in np.array(site) - off)
    return (a, b, c)


def _rows(ch: Change) -> dict[Site, int]:
    return {
        (int(v[0]), int(v[1]), int(v[2])): int(label)
        for v, label in zip(ch.voxels, ch.labels, strict=True)
    }


def _check_form(ch: Change, labels: Any) -> None:
    assert ch.voxels.dtype == np.int64 and ch.labels.dtype == np.int64
    assert ch.voxels.shape == (len(ch.labels), 3)
    if len(ch.labels):
        lin = np.ravel_multi_index(tuple(ch.voxels.T), labels.shape)
        assert (np.diff(lin) > 0).all()
        assert (labels[tuple(ch.voxels.T)] != ch.labels).all()


def test_remove_strand() -> None:
    a = _cortex()
    a[2:7, :, :] = WM
    a[7:15, 10, 10] = WM  # strand, one voxel thick, joined to the slab
    before = a.copy()
    site = (10, 10, 10)
    ch = remove_strand(a, site, "R")
    # the strand voxel at x = 7 stays: the opening restores it from the
    # eroded slab voxel at x = 6, which has the strand as a neighbour
    expected = _ball_set(site, 4, {(x, 10, 10) for x in range(8, 15)})
    assert _rows(ch) == dict.fromkeys(expected, CTX)
    assert ch.params == {"radius": 1, "reach": 4}
    _check_form(ch, a)
    assert (a == before).all()
    out = apply(a, ch)
    assert (out[8:15, 10, 10] == CTX).all()
    assert out[7, 10, 10] == WM
    assert (out[2:7] == WM).all()


def test_remove_strand_none() -> None:
    a = _cortex()
    a[2:7, :, :] = WM
    ch = remove_strand(a, (15, 10, 10), "R")
    assert len(ch.voxels) == 0 and ch.voxels.shape == (0, 3)


def _gap_phantom() -> Any:
    a = _cortex()
    a[2:7] = WM
    a[8:13] = WM  # one-voxel gap at x = 7
    return a


def _gap_cells(site: Site, reach: int, x: int) -> set[Site]:
    return _ball_set(
        site, reach, {(x, y, z) for y in range(1, 20) for z in range(1, 20)}
    )


def test_fill_gap() -> None:
    a = _gap_phantom()
    before = a.copy()
    site = (7, 10, 10)
    ch = fill_gap(a, site, "R")
    assert _rows(ch) == dict.fromkeys(_gap_cells(site, 4, 7), WM)
    _check_form(ch, a)
    assert (a == before).all()
    assert (apply(a, ch)[7, 6:15, 6:15] == WM).sum() > 0


def test_widen_gap() -> None:
    a = _gap_phantom()
    before = a.copy()
    site = (7, 10, 10)
    ch = widen_gap(a, site, "R")
    gap = _gap_cells(site, 4, 7)
    walls = _ball_set(site, 4, {(x, y, z) for (_, y, z) in gap for x in (6, 8)})
    assert _rows(ch) == dict.fromkeys(walls, CTX)
    _check_form(ch, a)
    assert (a == before).all()
    out = apply(a, ch)
    assert (out[:, 10, 10] == CTX)[6:9].all()
    assert out[5, 10, 10] == WM and out[9, 10, 10] == WM


@pytest.mark.parametrize(("radius", "width"), [(1, 4), (2, 6)])
def test_thicken(radius: int, width: int) -> None:
    a = _cortex()
    a[:, :, 9:11] = WM  # blade, 2 voxels thick
    before = a.copy()
    site = (10, 10, 10)
    ch = thicken(a, site, "R", radius=radius)
    cells = {
        (x, y, z)
        for x in range(21)
        for y in range(21)
        for z in range(10 - radius - 1, 10 + radius + 2)
        if (z < 9 and 9 - z <= radius) or (z > 10 and z - 10 <= radius)
    }
    assert _rows(ch) == dict.fromkeys(_ball_set(site, 4, cells), WM)
    assert ch.params == {"radius": radius, "reach": 4}
    _check_form(ch, a)
    assert (a == before).all()
    assert (apply(a, ch)[10, 10, :] == WM).sum() == width


def test_leave() -> None:
    a = _gap_phantom()
    before = a.copy()
    ch = leave(a, (7, 10, 10), "R")
    assert len(ch.voxels) == 0 and len(ch.labels) == 0
    assert ch.params == {"radius": 1, "reach": 4}
    assert (a == before).all()
    assert (apply(a, ch) == a).all()


def test_support() -> None:
    assert support(1, 4) == 7
    assert box_round((10, 10, 10), (1, 5, 9), 3) == (
        slice(0, 5),
        slice(2, 9),
        slice(6, 10),
    )


Choice = Callable[..., Change]
FUNCTIONS: list[Choice] = [remove_strand, fill_gap, widen_gap, thicken, leave]


def _random_phantom(seed: int, n: int = 40) -> Any:
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.standard_normal((n, n, n)), 1.5)
    g = ndi.gaussian_filter(rng.standard_normal((n, n, n)), 3.0)
    a = np.zeros((n, n, n), np.int64)
    a[g > -0.02 * g.std()] = CTX
    a[f > 0.15 * f.std()] = WM
    a[rng.random((n, n, n)) < 0.02] = 43  # some other label
    return a


@pytest.mark.parametrize("choice", FUNCTIONS)
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_crop_equals_whole(choice: Choice, radius: int, seed: int) -> None:
    a = _random_phantom(seed)
    before = a.copy()
    reach = 3
    sites: list[Site] = [(20, 20, 20), (3, 5, 4), (36, 2, 21), (0, 39, 39), (12, 30, 8)]
    changed = 0
    for site in sites:
        whole = choice(a, site, "R", radius=radius, reach=reach)
        _check_form(whole, a)
        sl = box_round(a.shape, site, support(radius, reach))
        off = np.array([s.start for s in sl])
        sub = choice(
            a[sl],
            _local(site, off),
            "R",
            radius=radius,
            reach=reach,
        )
        assert (sub.voxels + off == whole.voxels).all()
        assert (sub.labels == whole.labels).all()
        assert sub.params == whole.params
        changed += len(whole.voxels)
    assert (a == before).all()
    if choice is not leave:
        assert changed > 0


def test_crop_site_near_edge_of_larger_volume() -> None:
    a = _cortex(30)
    a[:, :, 1:4] = WM
    a[:, :, 5:8] = WM  # gap at z = 4, near the edge of the volume
    a[9:14, 1, 2] = WM
    site = (11, 1, 4)
    for choice in (fill_gap, widen_gap, thicken, remove_strand):
        whole = choice(a, site, "R")
        sl = box_round(a.shape, site, support(1, 4))
        off = np.array([s.start for s in sl])
        sub = choice(a[sl], _local(site, off), "R")
        assert (sub.voxels + off == whole.voxels).all()
        assert (sub.labels == whole.labels).all()
    assert len(fill_gap(a, site, "R").voxels) > 0


def test_run_names_the_choice() -> None:
    a = _cortex()
    a[2:7, :, :] = WM
    a[7:15, 10, 10] = WM
    for name, f in CHOICES.items():
        ch = run(name, a, (10, 10, 10), "R", radius=1, reach=3)
        want = f(a, (10, 10, 10), "R", radius=1, reach=3)
        assert ch.choice == name
        assert (ch.voxels == want.voxels).all() and (ch.labels == want.labels).all()
    with pytest.raises(KeyError):
        run("paint", a, (10, 10, 10), "R")


# ------------------------------------------------------------------ preview

PVOX = (0.2, 0.16, 0.2)


def _ring(slit: int) -> Any:
    """A WM ring (label 41) round a 2 x 20 hole, open at a slit of
    background `slit` voxels wide across one side, in background 0."""
    a = np.zeros((44, 40, 34), np.int16)
    a[25:37, 3:33, 22:28] = 41
    a[30:32, 8:28, 22:28] = 0  # the hole
    a[25:30, 16 : 16 + slit, 22:28] = 0  # the slit
    return a


def test_preview_of_a_choice_that_makes_a_handle() -> None:
    a = _ring(1)
    p = preview(a, (27, 16, 25), "R", "fill-gap", voxel_mm=PVOX, halo_mm=2.0)
    assert p.answer.choice == "fill-gap" and len(p.answer.voxels)
    assert set(p.answer.voxels[:, 1].tolist()) == {16}  # in the slit
    assert not p.cut_by_box
    # the answer closes the ring; the correction removes the handle, and
    # not through the verified voxels of the answer
    assert len(p.correction.voxels) and not p.overrides
    ans = {tuple(v) for v in p.answer.voxels.tolist()}
    assert not ans & {tuple(v) for v in p.correction.voxels.tolist()}
    assert a.tobytes() == _ring(1).tobytes()  # the input does not change


def test_preview_of_an_answer_that_leaves_a_handle_shows_the_override() -> None:
    a = _ring(2)
    rod = [(27, 16, 25), (27, 17, 25)]
    for v in rod:
        a[v] = 41  # a one-voxel rod across the slit closes the ring
    box = np.zeros(a.shape, bool)
    box[25:37, 3:33, 22:28] = True
    vox = np.argwhere(box).astype(np.int64)
    paint = Change(vox, a[box].astype(np.int64), choice="manual-paint")
    p = preview(a, rod[0], "R", paint, voxel_mm=PVOX)
    assert p.overrides and not p.cut_by_box
    assert len(p.correction.voxels) == 1
    assert tuple(p.correction.voxels[0].tolist()) in rod
    assert p.correction.labels.tolist() == [0]


def test_preview_box_cut() -> None:
    a = _ring(1)
    p = preview(a, (27, 16, 25), "R", "fill-gap", voxel_mm=PVOX, halo_mm=0.2)
    assert p.cut_by_box  # the ring leaves the box: the handle is not seen
    assert not len(p.correction.voxels)
