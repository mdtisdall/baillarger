"""The column-volume sampler (docs/formats/column-volume.md): the spec
against the code."""

import dataclasses
import typing
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from baillarger import backend, column_volume
from baillarger import rays as R
from baillarger.blocks import ArraySource, ChunkedSource

DOC = Path(__file__).resolve().parents[1] / "docs" / "formats" / "column-volume.md"


def doc_rows(heading: str) -> list[list[str]]:
    """The cells of the table rows (first cell in backticks) under heading."""
    lines = DOC.read_text().splitlines()
    i = lines.index(heading)
    out: list[list[str]] = []
    for line in lines[i + 1 :]:
        if line.startswith("#") and out:
            break
        if line.startswith("| `"):
            out.append([c.strip().strip("`") for c in line.strip("|").split("|")])
    return out


def field_type(t: object) -> str:
    """float32 for npt.NDArray[np.float32], the class name otherwise."""
    args = typing.get_args(t)
    if args:
        return str(args[-1].__name__)
    return str(getattr(t, "__name__", t))


def test_the_spec_lists_the_output_fields() -> None:
    hints = typing.get_type_hints(column_volume.ColumnVolume)
    want = [
        [f.name, field_type(hints[f.name])]
        for f in dataclasses.fields(column_volume.ColumnVolume)
    ]
    got = [row[:2] for row in doc_rows("### `ColumnVolume`")]
    assert got == want


def test_the_spec_gives_the_version() -> None:
    assert f"SpecVersion `{column_volume.VERSION}`" in DOC.read_text()


# --- the sampler: exact tests ---

Interval = tuple[float, float, int]


def make_rays(
    roots: npt.ArrayLike,
    normals: npt.ArrayLike,
    intervals: list[list[Interval]] | None = None,
    e_in: float = 0.4,
    e_out: float = 0.8,
    step: float = 0.04,
) -> R.Rays:
    """Rays of straight columns, built from the spec of ray-sampler/1."""
    x0 = np.asarray(roots, np.float32).reshape(-1, 3)
    nrm = np.asarray(normals, np.float64).reshape(-1, 3)
    nrm = (nrm / np.linalg.norm(nrm, axis=1, keepdims=True)).astype(np.float32)
    n = len(x0)
    if intervals is None:
        intervals = [[(-R.CAP_IN, R.CAP_OUT, 1)] for _ in range(n)]
    k_in = int(np.floor(e_in / step + 1e-6))
    k_out = int(np.floor(e_out / step + 1e-6))
    ks = np.arange(-k_in, k_out + 1)
    sk = ks * step
    s = np.concatenate([sk.astype(np.float32)] * n)
    pos = np.concatenate(
        [
            (x0[c].astype(np.float64) + sk[:, None] * nrm[c].astype(np.float64)).astype(
                np.float32
            )
            for c in range(n)
        ]
    )
    i0 = [0]
    for iv in intervals:
        i0.append(i0[-1] + len(iv))
    flat = [x for iv in intervals for x in iv]
    cls = {-1: R.OFF_GRID, 0: R.OUTSIDE, 1: R.WM, 3: R.CORTEX}
    roots_ = R.Roots(
        edge_key=np.arange(n, dtype=np.int64),
        vertex=np.arange(n, dtype=np.int64),
        owned=np.ones(n, np.bool_),
        root=x0,
        normal=np.asarray(nrm, np.float32),
        normal_length=np.ones(n, np.float32),
    )
    return R.Rays(
        roots=roots_,
        method="normal",
        step=step,
        k_in=np.full(n, k_in, np.int32),
        k_out=np.full(n, k_out, np.int32),
        sample_start=np.arange(n + 1, dtype=np.int64) * len(ks),
        s=s,
        position=pos,
        interval_start=np.asarray(i0, np.int64),
        interval_s0=np.asarray([x[0] for x in flat], np.float32),
        interval_s1=np.asarray([x[1] for x in flat], np.float32),
        interval_label=np.asarray([x[2] for x in flat], np.int32),
        interval_class=np.asarray([cls.get(x[2], R.WM) for x in flat], np.uint8),
        derived_class=np.full(n, R.NEITHER, np.uint8),
        area_coef=np.zeros((0, 3), np.float64),
        focus=np.zeros((0, 2), np.float64),
        area=np.zeros(0, np.float32),
        volume=np.zeros(0, np.float32),
        past_focus=np.zeros(0, np.bool_),
    )


# an oblique, anisotropic grid: voxel sizes 0.16, 0.2, 0.25 mm, a shear, an offset
AFFINE = np.array(
    [
        [0.16, 0.02, 0.0, -3.0],
        [0.0, 0.2, 0.03, 5.0],
        [0.01, 0.0, 0.25, 1.0],
        [0, 0, 0, 1],
    ]
)
SHAPE = (24, 20, 16)


def to_mm(v: npt.ArrayLike) -> npt.NDArray[np.float64]:
    v = np.asarray(v, np.float64)
    out: npt.NDArray[np.float64] = (AFFINE[:3, :3] @ v.T).T + AFFINE[:3, 3]
    return out


def sampler(*vols: npt.ArrayLike) -> column_volume.ColumnVolumeSampler:
    srcs = [ArraySource(np.asarray(v, np.float32)) for v in vols]
    return column_volume.ColumnVolumeSampler(srcs, AFFINE, [AFFINE] * len(srcs), SHAPE)


def test_a_linear_image_is_exact() -> None:
    i, j, k = np.meshgrid(*(np.arange(n) for n in SHAPE), indexing="ij")
    vol = 0.5 * i - 0.25 * j + 2.0 * k + 3.0  # dyadic: exact in float32
    dirs = np.array([[1, 0.5, 0.2], [0, 0, 1]])
    rays = make_rays(to_mm([[8.3, 9.1, 7.7], [12.0, 5.5, 8.0]]), dirs)
    cv = sampler(vol).sample(rays)
    v = (
        np.linalg.inv(AFFINE[:3, :3])
        @ (rays.position.astype(np.float64) - AFFINE[:3, 3]).T
    )
    want = 0.5 * v[0] - 0.25 * v[1] + 2.0 * v[2] + 3.0
    assert cv.valid.all()
    np.testing.assert_allclose(cv.echo[:, 0], want, rtol=0, atol=2e-5)


def test_a_ray_through_a_slab_gives_the_slab_profile() -> None:
    # slab: voxels k in [6, 9] hold 10, the rest 1; ray along +k (a grid axis)
    aff = np.eye(4)
    vol = np.ones((8, 8, 16), np.float32)
    vol[:, :, 6:10] = 10.0
    smp = column_volume.ColumnVolumeSampler([ArraySource(vol)], aff, [aff], (8, 8, 16))
    rays = make_rays([[3, 3, 4]], [[0, 0, 1]], e_in=0.0, e_out=8.0, step=0.25)
    cv = smp.sample(rays)
    s = rays.s.astype(np.float64)
    z = 4.0 + s
    want = np.interp(z, [0, 5, 6, 9, 10, 15], [1, 1, 10, 10, 1, 1])
    np.testing.assert_allclose(cv.echo[:, 0], want, atol=1e-6)


def test_samples_off_the_voxel_centres_are_not_valid() -> None:
    aff = np.eye(4)
    vol = np.ones((4, 4, 4), np.float32)
    smp = column_volume.ColumnVolumeSampler([ArraySource(vol)], aff, [aff], (4, 4, 4))
    rays = make_rays([[1, 1, 0.0]], [[0, 0, 1]], e_in=0.5, e_out=3.5, step=0.5)
    cv = smp.sample(rays)
    z = rays.s.astype(np.float64)
    assert (cv.valid == ((z >= 0) & (z <= 3))).all()
    assert np.isnan(cv.echo[~cv.valid]).all()
    assert (cv.echo[cv.valid] == 1.0).all()


def test_a_longer_extent_gives_the_same_samples_and_more() -> None:
    rng = np.random.default_rng(0)
    vol = rng.random(SHAPE).astype(np.float32)
    root, nrm = to_mm([[10.2, 9.7, 7.1]]), [[0.3, 0.5, 0.8]]
    short = sampler(vol).sample(make_rays(root, nrm, e_in=0.2, e_out=0.4))
    long = sampler(vol).sample(make_rays(root, nrm, e_in=0.4, e_out=1.2))
    sl = {float(s): k for k, s in enumerate(long.rays.s)}
    for k, s in enumerate(short.rays.s):
        m = sl[float(s)]
        assert short.echo[k, 0] == long.echo[m, 0]
    assert len(long.echo) > len(short.echo)


def test_a_column_is_the_same_in_any_batch() -> None:
    rng = np.random.default_rng(1)
    vol = rng.random(SHAPE).astype(np.float32)
    roots = to_mm(rng.uniform([4, 4, 4], [18, 14, 10], (6, 3)))
    nrms = rng.normal(size=(6, 3))
    both = sampler(vol).sample(make_rays(roots, nrms))
    for c in (0, 3, 5):
        alone = sampler(vol).sample(make_rays(roots[c : c + 1], nrms[c : c + 1]))
        a, b = both.rays.sample_start[c], both.rays.sample_start[c + 1]
        assert np.array_equal(both.echo[a:b], alone.echo, equal_nan=True)
        assert np.array_equal(both.valid[a:b], alone.valid)


def test_a_batch_of_columns_gives_the_same_samples() -> None:
    vol, rays = random_case()
    smp = sampler(vol)
    whole = smp.sample(rays, "numpy")
    for lo, hi in ((0, 7), (7, 40)):
        part = smp.sample(column_volume.take_columns(rays, lo, hi), "numpy")
        a, b = rays.sample_start[lo], rays.sample_start[hi]
        assert np.array_equal(part.echo, whole.echo[a:b], equal_nan=True)
        assert np.array_equal(part.valid, whole.valid[a:b])


def test_the_sampler_refuses_a_bad_grid() -> None:
    vol = ArraySource(np.zeros(SHAPE, np.float32))
    cls = column_volume.ColumnVolumeSampler
    with pytest.raises(ValueError, match="shape"):
        cls([ArraySource(np.zeros((3, 3, 3), np.float32))], AFFINE, [AFFINE], SHAPE)
    with pytest.raises(ValueError, match="dtype"):
        cls([ArraySource(np.zeros(SHAPE, np.complex64))], AFFINE, [AFFINE], SHAPE)
    shifted = AFFINE.copy()
    shifted[0, 3] += 0.01
    with pytest.raises(ValueError, match="affine"):
        cls([vol], AFFINE, [shifted], SHAPE)
    # float32 rounding of the affine is accepted
    cls([vol], AFFINE, [AFFINE.astype(np.float32)], SHAPE)


@pytest.fixture(autouse=True)
def numpy_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact tests run on the numpy reference; the device tests name
    their device."""
    monkeypatch.setenv("BAILLARGER_DEVICE", "numpy")


DEVICES = [d for d in backend.available() if d != "numpy"]


def random_case() -> tuple[npt.NDArray[np.float32], R.Rays]:
    rng = np.random.default_rng(7)
    vol = rng.random(SHAPE).astype(np.float32)
    roots = to_mm(rng.uniform([-1, -1, -1], [25, 21, 17], (40, 3)))
    iv = [[(-2.0, 0.3, 41), (0.3, 1.1, 49), (1.1, 8.0, 0)] for _ in range(40)]
    return vol, make_rays(roots, rng.normal(size=(40, 3)), iv, e_in=0.4, e_out=1.2)


@pytest.mark.parametrize("name", DEVICES)
def test_a_device_matches_numpy_to_float32_rounding(name: str) -> None:
    vol, rays = random_case()
    ref = sampler(vol).sample(rays, "numpy")
    got = sampler(vol).sample(rays, name)
    assert ref.valid.any()
    assert not ref.valid.all()
    assert np.array_equal(got.valid, ref.valid)
    assert np.array_equal(np.isnan(got.echo), np.isnan(ref.echo))
    np.testing.assert_allclose(got.echo, ref.echo, rtol=0, atol=4 * 2.0**-24)


@pytest.mark.parametrize("name", DEVICES)
def test_a_device_gives_the_same_result_every_time(name: str) -> None:
    vol, rays = random_case()
    a = sampler(vol).sample(rays, name)
    b = sampler(vol).sample(rays, name)
    assert np.array_equal(a.echo, b.echo, equal_nan=True)


def test_grid_sample_with_align_corners_is_the_convention() -> None:
    """The spec fixes align_corners=True: the centre of voxel i is at the
    normalised coordinate 2 i / (n - 1) - 1. grid_sample then equals the
    numpy reference at the valid samples."""
    torch = pytest.importorskip("torch")
    vol, rays = random_case()
    ref = sampler(vol).sample(rays, "numpy")
    v = sampler(vol).voxel_coordinates(rays.position)
    n = np.asarray(SHAPE, np.float64)
    g = 2.0 * v / (n - 1.0) - 1.0
    # grid_sample takes (x, y, z) = (last axis, ..., first axis)
    grid = torch.as_tensor(g[:, ::-1].copy(), dtype=torch.float32).reshape(
        1, -1, 1, 1, 3
    )
    img = torch.as_tensor(vol).reshape(1, 1, *SHAPE)
    out = torch.nn.functional.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(-1)
    k = ref.valid
    np.testing.assert_allclose(out.numpy()[k], ref.echo[k, 0], rtol=0, atol=1e-5)


def test_reading_through_the_chunked_reader_changes_nothing() -> None:
    vol, rays = random_case()
    plain = sampler(vol).sample(rays, "numpy")
    chunked = ChunkedSource(ArraySource(vol), chunk=5, max_bytes=4000)
    smp = column_volume.ColumnVolumeSampler([chunked], AFFINE, [AFFINE], SHAPE)
    got = smp.sample(rays, "numpy")
    assert np.array_equal(got.echo, plain.echo, equal_nan=True)
    assert chunked.reads > 0


# --- with the real ray sampler ---


def test_the_column_volume_on_the_real_rays(tmp_path: Path) -> None:
    """A cortex slab with a WM layer, through the dseg and mesh commands and
    the real ray sampler: a linear image is exact at every valid sample,
    and the marks on the columns of the top face follow the label changes."""
    from test_rays import PHANTOM_AFF, PHANTOMS, write_phantom

    surf, dseg = write_phantom(tmp_path, PHANTOMS["slab"])
    rs = R.RaySampler(surf, dseg)
    rays = rs.sample((0, 0, 0), 3, e_in=0.4, e_out=1.2)
    shape = rs.shape
    aff = rs.affine
    i, j, k = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    vol = (0.5 * i - 0.25 * j + 2.0 * k + 3.0).astype(np.float32)
    smp = column_volume.ColumnVolumeSampler(
        [ChunkedSource(ArraySource(vol), chunk=8, max_bytes=1 << 20)],
        aff,
        [aff],
        (shape[0], shape[1], shape[2]),
    )
    cv = smp.sample(rays, "numpy")
    v = np.linalg.inv(aff[:3, :3]) @ (rays.position.astype(np.float64) - aff[:3, 3]).T
    want = 0.5 * v[0] - 0.25 * v[1] + 2.0 * v[2] + 3.0
    assert cv.valid.sum() > 10000
    np.testing.assert_allclose(cv.echo[cv.valid, 0], want[cv.valid], rtol=0, atol=2e-4)
    # the columns of the top face of the WM layer: normal along +k, and a
    # label change at the face, which the ray sampler gives
    vs = float(np.linalg.norm(PHANTOM_AFF[:3, 2]))
    top = np.flatnonzero(
        (rays.roots.normal[:, 2] > 0.999)
        & (np.abs(rays.roots.root[:, 2] - (PHANTOM_AFF[2, 3] + 16 * vs)) < vs)
    )
    assert len(top) > 50
    for c in top[:50]:
        assert rays.interval_start[c + 1] - rays.interval_start[c] > 1
