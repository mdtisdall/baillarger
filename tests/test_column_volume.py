"""The column-volume sampler (docs/formats/column-volume.md): the spec
against the code."""

import dataclasses
import typing
from pathlib import Path

from baillarger import column_volume

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
