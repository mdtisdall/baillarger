"""Tests for baillarger.refine: steps 1-6 end to end on a synthetic patch of
two areas with different three-echo band templates."""

import numpy as np
import numpy.typing as npt
import pytest

from baillarger import refine as rf
from baillarger.columns import ColumnData

type Tree = list[tuple[int, int, int, int, float, int]]
type Run = tuple[ColumnData, npt.NDArray[np.int64], Tree, dict[int, rf.Level]]


def two_areas(seed: int = 0) -> tuple[ColumnData, npt.NDArray[np.int64]]:
    """10 x 6 columns of 60 cortex voxels and 6 WM voxels. Area A (x < 5)
    has three bands, the middle one with a faster decay over the echoes (as
    the stria); area B (x >= 5) has two bands. Each column has its own gain
    and offset per echo, and noise. Returns the table and the true area of
    each column."""
    rng = np.random.default_rng(seed)
    nx, ny, per, nwm = 10, 6, 60, 6
    n = nx * ny
    col = np.repeat(np.arange(n), per + nwm).astype(np.int64)
    zz = np.tile(np.arange(per + nwm), n)
    m = (zz - nwm + 0.5) / per
    x = col // ny
    A = x < 5
    E = 3
    lev_a = np.array([[0.0, 0.0, 0.0], [-1.0, -1.4, -1.8], [0.6, 0.6, 0.6]])
    lev_b = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    band_a = np.searchsorted([0.35, 0.55], m, side="right")
    band_b = np.searchsorted([0.5], m, side="right")
    prof = np.where(A[:, None], lev_a[band_a], lev_b[np.minimum(band_b, 1)])
    prof[m < 0] = -0.5
    gain = rng.uniform(80, 120, (n, E))
    off = rng.uniform(400, 600, (n, E))
    f = off[col] + gain[col] * (prof + rng.normal(0, 0.15, (len(m), E)))
    e = []
    for c in range(n):
        if c % ny < ny - 1:
            e.append((c, c + 1))
        if c + ny < n:
            e.append((c, c + ny))
    edges = np.array(e, np.int64)
    data = ColumnData(
        shape=(len(m), 1, 1),
        index=np.arange(len(m), dtype=np.int64),
        col=col,
        m=m,
        f=f,
        ctx=m >= 0,
        labels=np.arange(n, dtype=np.int64),
        thickness=np.full(n, 2.0),
        edges=edges,
        w=np.full(len(edges), 10.0),
        sigma2=(0.15 * gain) ** 2,
    )
    return data, (np.arange(n) // ny >= 5).astype(np.int64)


@pytest.fixture(scope="module")
def run() -> Run:
    data, truth = two_areas()
    tree, levels = rf.Refiner(data).run([8, 2])
    return data, truth, tree, levels


def test_two_regions_are_the_two_areas(run: Run) -> None:
    data, truth, tree, levels = run
    lv = levels[2]
    a, b = (lv.reg == lv.reg[0]), (lv.reg != lv.reg[0])
    assert np.array_equal(a, truth == 0) or np.array_equal(a, truth == 1)
    assert b.any()


def test_bands_of_each_area(run: Run) -> None:
    data, truth, tree, levels = run
    lv = levels[2]
    RA = int(lv.reg[np.flatnonzero(truth == 0)[0]])
    RB = int(lv.reg[np.flatnonzero(truth == 1)[0]])
    assert lv.K[RA] == 3 and lv.K[RB] == 2
    assert np.allclose(lv.Z[RA], [0.35, 0.55], atol=0.03)
    assert np.allclose(lv.Z[RB], [0.5], atol=0.03)
    # each column's boundaries near the truth after the nudge
    za = lv.z[truth == 0, :2]
    assert np.median(np.abs(za - [0.35, 0.55])) < 0.03


def test_energy_does_not_increase_within_a_level(run: Run) -> None:
    _, _, _, levels = run
    for lv in levels.values():
        tot = [e["total"] for e in lv.energies]
        assert all(b <= a * (1 + 1e-6) for a, b in zip(tot[:-1], tot[1:], strict=True))


def test_merge_tree_and_bottoms(run: Run) -> None:
    data, truth, tree, levels = run
    assert len(tree) == data.n - 1
    lv = levels[2]
    # the WM is the same flat value in both areas, so the areal border ends
    # at the WM; but the bottom tests react late (plan, step 5), and with 6
    # WM bins the bottom reaches the deepest one
    for b, why in lv.bottoms.values():
        assert b == pytest.approx(-0.1) and why == "end"
    assert lv.u.shape == lv.xhat.shape and lv.u.shape[0] == data.n
