"""Tests for baillarger.surfaces: plan test 1 (optimal surfaces against brute
force), plan test 5 (a column copied with a gain and an offset gets the same
boundaries, through steps 2-4), and the min-cut itself."""

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from baillarger import bands as bd
from baillarger import gain as gn
from baillarger import surfaces as sf

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def sticks(n: int, per: int, profile: F64, seed: int = 2) -> tuple[I64, F64]:
    """n columns of per voxels each, depth (z + 0.5) / per."""
    col = np.repeat(np.arange(n), per).astype(np.int64)
    m = np.tile((np.arange(per) + 0.5) / per, n)
    return col, m


@pytest.mark.parametrize("E", [1, 3])
def test_surfaces_brute_force(E: int) -> None:
    """Plan test 1: 3 columns, 2 surfaces, every combination of depths."""
    rng = np.random.default_rng(1)
    n, per = 3, 20
    col, m = sticks(n, per, np.zeros(0))
    x = np.arange(n)[col]
    base = np.where(m < 0.3 + 0.05 * x, 0.0, 1.0) + np.where(m > 0.7, 1.0, 0.0)
    f = base[:, None] * np.linspace(1, 0.6, E)[None, :] + rng.normal(
        0, 0.3, (len(m), E)
    )
    sigma2 = rng.uniform(0.05, 0.2, (n, E))
    edges = np.array([[0, 1], [1, 2]], np.int64)
    ew = np.array([3.0, 5.0])
    zstep, t_min, eta, lz = 0.05, 0.1, 3.0, 2.0
    Z = np.array([0.35, 0.7])
    mu = np.stack([np.array([-1.0, 0.2, 1.1]) * (1 + 0.1 * e) for e in range(E)], 1)
    a = rng.normal(0, 0.1, (n, E))
    s = 1 + rng.normal(0, 0.1, (n, E))
    cols = np.arange(n)
    z = sf.surfaces_region(
        cols, col, m, f, sigma2, Z, mu, a, s, edges, ew, zstep, eta, lz, t_min
    )
    cand = sf.candidates(Z, zstep)

    def energy(zz: F64) -> float:
        e = 0.0
        for c in range(n):
            v = col == c
            t = mu[bd.band_of(m[v], zz[c])]
            e += float(np.sum((f[v] - a[c] - s[c] * t) ** 2 / sigma2[c]))
            e += eta * float(np.sum((zz[c] - Z) ** 2))
        for (i, j), w in zip(edges, ew, strict=True):
            e += lz * w * float(np.sum(np.abs(zz[i] - zz[j])))
        return e

    per_col = [(z0, z1) for z0 in cand[0] for z1 in cand[1] if z1 - z0 >= t_min - 1e-9]
    best = min(
        energy(np.array(combo)) for combo in itertools.product(per_col, repeat=n)
    )
    assert energy(z) == pytest.approx(best, rel=1e-9, abs=1e-9)
    assert np.all(np.diff(z, axis=1) >= t_min - 1e-9)


def test_one_band_has_no_surfaces() -> None:
    col, m = sticks(2, 10, np.zeros(0))
    z = sf.surfaces_region(
        np.arange(2),
        col,
        m,
        np.zeros((20, 1)),
        np.ones((2, 1)),
        np.zeros(0),
        np.zeros((1, 1)),
        np.zeros((2, 1)),
        np.ones((2, 1)),
        np.zeros((0, 2), np.int64),
        np.zeros(0),
        0.05,
        1.0,
        1.0,
        0.05,
    )
    assert z.shape == (2, 0)


def test_copy_with_gain_offset() -> None:
    """Plan test 5: a column copied with a gain and an offset gets the same
    boundaries, through the bands, templates, gain/offset and surfaces."""
    nx, ny, per = 6, 3, 40
    rng = np.random.default_rng(6)
    noise = rng.standard_normal((nx // 2, ny, per)) * 0.2
    n = nx * ny
    col = np.repeat(np.arange(n), per).astype(np.int64)
    zi = np.tile(np.arange(per), n)
    m = (zi + 0.5) / per
    x, y = col // ny, col % ny
    base = np.where(m < 0.3 + 0.03 * (x % 3), 0.0, 1.0) + np.where(m > 0.75, -1.5, 0.0)
    out = base + noise[x % 3, y, zi]
    f = np.where(x >= 3, 3.0 * out + 10.0, out)[:, None]
    # graph: neighbours in x and in y
    e = []
    for c in range(n):
        if c % ny < ny - 1:
            e.append((c, c + 1))
        if c + ny < n:
            e.append((c, c + ny))
    edges = np.array(e, np.int64)
    ew = np.ones(len(edges))
    sigma2 = np.ones((n, 1)) * 0.04
    cnt = np.bincount(col, minlength=n)
    a = (np.bincount(col, f[:, 0], n) / cnt)[:, None]
    s = np.sqrt(np.bincount(col, f[:, 0] ** 2, n) / cnt - a[:, 0] ** 2)[:, None]
    reg = np.zeros(n, np.int64)
    xh = (f - a[col]) / s[col]
    w = s[col] ** 2 / sigma2[col]
    K, Z, _ = bd.bands_for_region(
        col, m, f, xh, w, sigma2, n, 1 / 50, 8, 0.05, np.random.default_rng(0)
    )
    assert K >= 2
    z = np.tile(Z, (n, 1))
    for _ in range(3):
        bandv = bd.band_of(m, z[col])
        mus = gn.update_templates(col, reg, bandv, f, a, s, sigma2, {0: K})
        t = mus[0][bandv]
        a, s = gn.solve_gain(col, f, t, sigma2, edges, ew, 0.0, a, s)
        mus = gn.update_templates(col, reg, bandv, f, a, s, sigma2, {0: K})
        z = sf.surfaces_region(
            np.arange(n), col, m, f, sigma2, Z, mus[0], a, s, edges, ew,
            1 / 200, 0.1, 0.0, 0.05,
        )  # fmt: skip
    # column (x, y) and its copy (x + 3, y) have the same boundaries
    for xx in range(3):
        for yy in range(ny):
            assert np.allclose(z[xx * ny + yy], z[(xx + 3) * ny + yy])


def cut_cost(
    sd: npt.NDArray[np.bool_], src: F64, snk: F64, cap: F64, tail: I64, head: I64
) -> float:
    """The cost of the cut with source side sd."""
    c = float(np.sum(src[~sd]) + np.sum(snk[sd]))
    return c + float(np.sum(cap[sd[tail] & ~sd[head]]))


def test_st_min_cut_brute_force() -> None:
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = 6
        ne = 10
        tail = rng.integers(0, n, ne)
        head = rng.integers(0, n, ne)
        keep = tail != head
        tail, head = tail[keep], head[keep]
        cap = rng.uniform(0, 5, len(tail))
        src = rng.uniform(0, 3, n) * (rng.random(n) < 0.6)
        snk = rng.uniform(0, 3, n) * (rng.random(n) < 0.6)
        side = sf.st_min_cut(n, tail, head, cap, src, snk)

        args = (src, snk, cap, tail, head)
        best = min(
            cut_cost(np.array(bits, bool), *args)
            for bits in itertools.product([0, 1], repeat=n)
        )
        assert cut_cost(side, *args) == pytest.approx(best, rel=1e-6, abs=1e-6)


def test_st_min_cut_infinite_edges() -> None:
    # node 0 wants the source side (cost 5 on the sink side), but an infinite
    # edge 0 -> 1 and an infinite sink edge on 1 force it to the sink side
    side = sf.st_min_cut(
        2,
        np.array([0]),
        np.array([1]),
        np.array([0.0]),
        np.array([5.0, 0.0]),
        np.array([0.0, np.inf]),
        np.array([True]),
    )
    assert not side.any()
