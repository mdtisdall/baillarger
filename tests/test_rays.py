"""The ray sampler (docs/formats/ray-sampler.md): the spec against the code."""

import dataclasses
import inspect
import typing
from pathlib import Path

import pytest

from baillarger import rays

DOC = Path(__file__).resolve().parents[1] / "docs" / "formats" / "ray-sampler.md"


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
    """int64 for npt.NDArray[np.int64], the class name otherwise."""
    args = typing.get_args(t)
    if args:
        return str(args[-1].__name__)
    if t is str:
        return "string"
    return str(getattr(t, "__name__", t))


@pytest.mark.parametrize("cls", [rays.Roots, rays.Rays])
def test_the_spec_lists_the_output_fields(cls: type) -> None:
    hints = typing.get_type_hints(cls)
    want = [[f.name, field_type(hints[f.name])] for f in dataclasses.fields(cls)]
    got = [row[:2] for row in doc_rows(f"### `{cls.__name__}`")]
    assert got == want


@pytest.mark.parametrize(
    ("heading", "names"),
    [
        ("## The label classes", ("OUTSIDE", "WM", "CORTEX", "OTHER", "OFF_GRID")),
        ("## The derived class of a column", ("OPEN", "IN_WM", "NEITHER")),
    ],
)
def test_the_spec_lists_the_class_codes(heading: str, names: tuple[str, ...]) -> None:
    want = [[n, str(getattr(rays, n))] for n in names]
    assert [row[:2] for row in doc_rows(heading)] == want


def test_the_spec_gives_the_constants() -> None:
    text = DOC.read_text()
    assert f"SpecVersion `{rays.VERSION}`" in text
    assert f"`CAP_IN` = {rays.CAP_IN}" in text
    assert f"`CAP_OUT` = {rays.CAP_OUT}" in text
    assert f"`STEP` ({rays.STEP})" in text
    assert rays.METHODS == ("normal",)
    label = str(rays.OFF_GRID_LABEL).replace("-", "\N{MINUS SIGN}")
    assert f"the part of the ray outside the grid. Its label is {label}" in text


def test_the_spec_lists_the_parameters() -> None:
    sig = inspect.signature(rays.RaySampler.sample)
    names = [p for p in sig.parameters if p != "self"]
    assert [row[0] for row in doc_rows("### Parameters")] == names
