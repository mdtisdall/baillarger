"""orient, Part A: the local cortex level (docs/formats/orient.md)."""

import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import orient_phantoms as op
import pytest
import torch

from baillarger import backend, white_dseg
from baillarger import orient as O

DEVICES = [d for d in ("cpu", "mps") if d in backend.available()]
VOX = (0.16, 0.2, 0.12)


def test_the_taps() -> None:
    for s in (0.5, 1.0, 2.0, 4.0, 6.25):
        k = O.taps(s)
        r = O.radius(s)
        assert len(k) == 2 * r + 1 and r == int(np.ceil(O.TRUNCATE * s - 1e-9))
        assert abs(k.sum() - 1.0) < 1e-15
        assert np.array_equal(k, k[::-1])
    with pytest.raises(ValueError):
        O.taps(0.0)


def shell(shape: tuple[int, int, int] = (40, 40, 40)) -> npt.NDArray[np.bool_]:
    x = np.indices(shape).astype(np.float64)
    r = np.sqrt(((x - 19.7) ** 2).sum(0))
    out: npt.NDArray[np.bool_] = (r >= 8) & (r < 16)
    return out


def test_the_level_of_a_constant_is_the_constant() -> None:
    ctx = shell()
    e = np.stack([np.full(ctx.shape, v) for v in (500.0, 300.0, 120.0)])
    e[:, ~ctx] = 7.0  # outside the cortex: not seen
    lv = O.local_level(e, ctx, 0.64, VOX)
    want = np.log([500.0, 300.0, 120.0])
    for i in range(3):
        np.testing.assert_allclose(lv[i][ctx], want[i], rtol=0, atol=1e-12)
        assert np.all(np.isnan(lv[i][~ctx]))


@pytest.mark.parametrize("dev", DEVICES)
def test_numpy_and_torch_agree(dev: str) -> None:
    """float32 on the device against float64 numpy: within float32 rounding
    of the log values (relative 1e-6, with a few roundings in the sums)."""
    rng = np.random.default_rng(1)
    ctx = shell()
    e = rng.uniform(50.0, 900.0, (3, *ctx.shape))
    lv = O.local_level(e, ctx, 0.64, VOX)
    d = backend.device(dev)
    t = O.local_level(
        torch.as_tensor(e.astype(np.float32), device=d.torch),
        torch.as_tensor(ctx, device=d.torch),
        0.64,
        VOX,
    )
    got = t.cpu().numpy()
    assert np.array_equal(np.isnan(got), np.isnan(lv))
    f = np.isfinite(lv)
    assert np.max(np.abs(got[f] - lv[f]) / np.abs(lv[f])) < 4e-6


def test_the_log_floor() -> None:
    x = np.array([0.0, 0.5, 1.0, np.e])
    np.testing.assert_allclose(O.log_echo(x), [0.0, 0.0, 0.0, 1.0])


def test_gradient_gain_is_the_variance_of_white_noise_gradients() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((1, 64, 64, 64))
    ctx = np.ones((64, 64, 64), bool)
    g = O.gradients(w, ctx)
    core = (slice(None), slice(None)) + (slice(8, -8),) * 3
    assert abs(np.var(g[core]) / O.gradient_gain() - 1.0) < 0.03


# ------------------------------------------------------------ on the files

AFF = np.diag([0.16, 0.2, 0.12, 1.0])
AFF[:3, 3] = [-3.0, 2.0, 5.0]


def write_files(d: Path, period_mm: float = 0.5) -> tuple[list[Path], Path]:
    """A sphere of laminae (op.laminae_sphere) as a right hemisphere: the
    dseg (via the dseg command, 8-cell blocks) and three echo images."""
    shape = (48, 48, 48)
    echoes, labels = op.laminae_sphere(
        shape,
        (0.16, 0.2, 0.12),
        (3.8, 4.7, 2.9),
        1.6,
        3.6,
        period_mm,
        (0.3, 0.2, 0.1),
        (600.0, 350.0, 200.0),
    )
    rng = np.random.default_rng(3)
    echoes = echoes * np.exp(0.05 * rng.standard_normal(echoes.shape))
    src = d / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(parents=True)
    nib.save(nib.nifti1.Nifti1Image(labels, AFF), src)
    out = d / "deriv"
    white_dseg.main([str(src), "--out", str(out)])
    dseg = out / "sub-T01/ses-S1/anat/sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    paths = []
    for i in range(3):
        p = d / f"sub-T01_ses-S1_echo-{i + 1}_part-mag_FLASH.nii"
        nib.save(nib.nifti1.Nifti1Image(echoes[i].astype(np.float32), AFF), p)
        paths.append(p)
    return paths, dseg


@pytest.fixture(scope="module")
def sampler(tmp_path_factory: pytest.TempPathFactory) -> O.OrientSampler:
    paths, dseg = write_files(tmp_path_factory.mktemp("sphere"))
    return O.OrientSampler(paths, dseg, cells=8)


def test_the_level_does_not_depend_on_the_superblock(sampler: O.OrientSampler) -> None:
    """numpy: the same float32 value, bit for bit, from every superblock that
    holds the voxel, and the same as the whole grid."""
    whole = sampler.level((0, 0, 0), 3, sigma_mm=0.64)
    assert whole.level.shape == (3, 48, 48, 48)
    for i, j, k in itertools.product(range(6), repeat=3):
        r = sampler.level((i, j, k), 0, halo_mm=0.5, sigma_mm=0.64)
        want = whole.level[(slice(None), *r.box)]
        assert np.array_equal(r.level, want, equal_nan=True)


@pytest.mark.parametrize("dev", DEVICES)
def test_the_sampler_on_a_device(sampler: O.OrientSampler, dev: str) -> None:
    a = sampler.level((1, 1, 1), 1, halo_mm=0.3, device="numpy")
    b = sampler.level((1, 1, 1), 1, halo_mm=0.3, device=dev)
    assert np.array_equal(np.isnan(a.level), np.isnan(b.level))
    f = np.isfinite(a.level)
    assert np.max(np.abs(a.level[f] - b.level[f]) / np.abs(a.level[f])) < 4e-6


def test_the_sampler_refuses_bad_parameters(sampler: O.OrientSampler) -> None:
    for kw in ({"sigma_mm": 0.0}, {"halo_mm": -1.0}, {"sigma_mm": float("nan")}):
        with pytest.raises(ValueError):
            sampler.level((0, 0, 0), 0, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sampler.level((0, 0, -1), 0)


def test_the_echoes_must_be_on_the_grid(
    sampler: O.OrientSampler, tmp_path: Path
) -> None:
    p = tmp_path / "sub-T01_ses-S1_echo-1_part-mag_FLASH.nii"
    nib.save(nib.nifti1.Nifti1Image(np.zeros((8, 8, 8), np.float32), AFF), p)
    with pytest.raises(ValueError):
        O.OrientSampler([p], sampler.dseg)


# ------------------------------------------------------------------- the spec

DOC = Path(__file__).resolve().parents[1] / "docs" / "formats" / "orient.md"


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


def test_the_spec_gives_the_constants() -> None:
    text = DOC.read_text()
    assert f"SpecVersion `{O.VERSION}`" in text
    assert f"`FLOOR` = {O.FLOOR}" in text
    assert f"`TRUNCATE` = {O.TRUNCATE}" in text
    assert f"`MIN_WEIGHT` ({O.MIN_WEIGHT:.0e})" in text
    assert f"`SIGMA` = {O.SIGMA} mm" in text
    assert f"`SD` = {O.SD:g} voxel" in text and f"`SI` = {O.SI:g} voxels" in text


def test_the_spec_lists_the_parameters_and_fields() -> None:
    import dataclasses
    import inspect

    sig = inspect.signature(O.OrientSampler.level)
    names = [p for p in sig.parameters if p != "self"]
    assert [row[0] for row in doc_rows("### Parameters")] == names
    got = [row[0] for row in doc_rows("### `Level`")]
    assert got == [f.name for f in dataclasses.fields(O.Level)]
