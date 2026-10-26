"""Tests for baillarger.data: on small synthetic NIfTI files, and on the example
data when data/ has it (skipped otherwise)."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

from baillarger import data

GRID = (700, 384, 1148)

# a small grid with an oblique, anisotropic affine
_AFF = np.array(
    [
        [0.0, -0.2, 0.0, 30.5],
        [0.0, 0.0, 0.16, -12.25],
        [-0.18, 0.0, 0.0, 7.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def write_fake(
    d: Path,
    shape: tuple[int, int, int] = (12, 10, 14),
    aseg_dtype: npt.DTypeLike = np.int16,
    aff2: npt.NDArray[np.float64] | None = None,
) -> tuple[list[npt.NDArray[np.int16]], npt.NDArray[np.int16]]:
    """Three echoes (with sidecars) and an aseg in d; returns their arrays."""
    rng = np.random.default_rng(0)
    echoes = [rng.integers(0, 3000, shape).astype(np.int16) for _ in range(3)]
    for k, (e, te) in enumerate(zip(echoes, (0.00937, 0.02071, 0.03205), strict=True)):
        p = data.echo_path(k + 1, d)
        aff = _AFF if aff2 is None or k == 0 else aff2
        nib.save(nib.nifti1.Nifti1Image(e, aff), p)
        side = p.with_name(p.name.removesuffix(".nii.gz") + ".json")
        side.write_text(json.dumps({"EchoTime": te, "EchoNumber": k + 1}))
    aseg = rng.choice(np.array([0, 41, 42, 43, 192, 49]), shape).astype(np.int16)
    nib.save(nib.nifti1.Nifti1Image(aseg.astype(aseg_dtype), _AFF), data.aseg_path(d))
    return echoes, aseg


# ---------------------------------------------------------------- synthetic


def test_rim_labels() -> None:
    aseg = np.array([0, 41, 42, 43, 192, 49, 50, 0, 42], np.int16)
    rim = data.rim_labels(aseg)
    assert rim.dtype == np.int16
    assert rim.tolist() == [1, 2, 3, 0, 0, 0, 0, 1, 3]


def test_patch_boxes_match_the_data_notes() -> None:
    shapes = {"V1V2": (140, 150, 245), "PCC_SPL": (125, 125, 125)}
    shapes["M1S1"] = (151, 151, 151)
    for name, (lo, hi) in data.PATCHES.items():
        assert tuple(b - a for a, b in zip(lo, hi, strict=True)) == shapes[name]
        assert all(0 <= a < b <= n for a, b, n in zip(lo, hi, GRID, strict=True))
    v1v2 = shapes["V1V2"]
    for pts in data.V1V2_REVIEW_POINTS.values():
        for p in pts:
            assert all(0 <= c < n for c, n in zip(p, v1v2, strict=True))


def test_crop_affine_moves_the_origin_to_lo() -> None:
    lo = (3, 5, 7)
    A = data.crop_affine(_AFF, lo)
    ijk = np.array([[0, 0, 0], [2, 1, 4]], float)
    assert np.allclose(
        nib.affines.apply_affine(A, ijk), nib.affines.apply_affine(_AFF, ijk + lo)
    )


def test_data_dir_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BAILLARGER_DATA", str(tmp_path))
    assert data.data_dir() == tmp_path
    assert not data.data_available()
    write_fake(tmp_path)
    assert data.data_available()
    monkeypatch.delenv("BAILLARGER_DATA")
    assert data.data_dir().name == "data"


def test_load_box_reads_the_box(tmp_path: Path) -> None:
    echoes, aseg = write_fake(tmp_path)
    lo, hi = (2, 1, 3), (9, 10, 11)
    p = data.load_box(lo, hi, "test", tmp_path)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    assert p.name == "test" and p.lo == lo and p.hi == hi
    assert p.shape == (7, 9, 8)
    assert p.echoes.dtype == np.float32 and p.echoes.shape == (7, 9, 8, 3)
    for k in range(3):
        assert np.array_equal(p.echoes[..., k], echoes[k][sl])
    assert np.array_equal(p.flash, echoes[1][sl])
    assert p.aseg.dtype == np.int16 and np.array_equal(p.aseg, aseg[sl])
    assert np.array_equal(p.rim, data.rim_labels(aseg[sl]))
    assert np.allclose(p.affine, data.crop_affine(_AFF, lo))
    assert np.allclose(p.voxel_mm, (0.18, 0.2, 0.16))
    assert np.allclose(p.echo_times_ms, (9.37, 20.71, 32.05))


def test_load_box_checks_the_box_and_the_grid(tmp_path: Path) -> None:
    write_fake(tmp_path)
    with pytest.raises(ValueError, match="not inside"):
        data.load_box((0, 0, 0), (13, 10, 14), directory=tmp_path)
    with pytest.raises(ValueError, match="not inside"):
        data.load_box((5, 0, 0), (5, 10, 14), directory=tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    shifted = _AFF.copy()
    shifted[0, 3] += 0.1
    write_fake(other, aff2=shifted)
    with pytest.raises(ValueError, match="not on the grid"):
        data.load_box((0, 0, 0), (2, 2, 2), directory=other)
    flt = tmp_path / "float"
    flt.mkdir()
    write_fake(flt, aseg_dtype=np.float32)
    with pytest.raises(ValueError, match="not integers"):
        data.load_box((0, 0, 0), (2, 2, 2), directory=flt)


# ---------------------------------------------------------------- the example data

need_data = pytest.mark.skipif(not data.data_available(), reason="no example data")


@pytest.fixture(scope="module")
def v1v2() -> data.Patch:
    return data.load_patch("V1V2")


@need_data
def test_v1v2_crop(v1v2: data.Patch) -> None:
    assert v1v2.shape == (140, 150, 245)
    assert v1v2.echoes.shape == (140, 150, 245, 3)
    assert np.allclose(v1v2.voxel_mm, (0.16028, 0.16, 0.16028), atol=1e-5)
    assert np.allclose(v1v2.echo_times_ms, (9.37, 20.71, 32.05))
    assert set(np.unique(v1v2.rim)) == {0, 1, 2, 3}
    # one plane of echo 2, read again from the full file
    img = nib.load(data.echo_path(2))
    assert isinstance(img, nib.nifti1.Nifti1Image)
    lo, hi = data.PATCHES["V1V2"]
    plane = np.asarray(img.dataobj[lo[0] + 70, lo[1] : hi[1], lo[2] : hi[2]])
    assert np.array_equal(v1v2.echoes[70, :, :, 1], plane)
    # the crop's voxels are where the full grid puts them
    ijk = np.array([[0.0, 0.0, 0.0], [139.0, 149.0, 244.0]])
    assert np.allclose(
        nib.affines.apply_affine(v1v2.affine, ijk),
        nib.affines.apply_affine(img.affine, ijk + lo),
    )


@need_data
def test_v1v2_intensities_by_label(v1v2: data.Patch) -> None:
    # the specimen is in Fomblin, which gives no signal: the outside is far
    # darker than the cortex at every echo; WM is darker than cortex
    for k in range(3):
        e = v1v2.echoes[..., k]
        cortex = np.median(e[v1v2.rim == 3])
        assert np.median(e[v1v2.rim == 1]) < 0.2 * cortex
        assert np.median(e[v1v2.rim == 2]) < cortex
