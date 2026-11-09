import re
from pathlib import Path

from baillarger import labels

DOC = Path(__file__).resolve().parents[1] / "docs" / "formats" / "label-set.md"


def doc_filled() -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for line in DOC.read_text().splitlines():
        m = re.match(r"\| ([RL]) \| `([0-9, ]+)` \|$", line)
        if m:
            out[m[1]] = tuple(int(v) for v in m[2].split(","))
    return out


def doc_classes() -> dict[str, tuple[int, int, str]]:
    out: dict[str, tuple[int, int, str]] = {}
    for line in DOC.read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 5 and cells[1].isdigit() and cells[2].isdigit():
            out[cells[0]] = (int(cells[1]), int(cells[2]), cells[3])
    return out


def test_doc_lists_the_filled_labels() -> None:
    assert doc_filled() == labels.FILLED_LABELS


def test_doc_filled_column_matches_the_filled_labels() -> None:
    for hemi, col in (("R", 0), ("L", 1)):
        yes = {
            v[col]
            for v in doc_classes().values()
            if v[2] == "yes" or v[2].startswith("follows")
        }
        assert yes == set(labels.FILLED_LABELS[hemi])


def test_doc_names_the_version() -> None:
    assert f"`{labels.VERSION}`" in DOC.read_text()


def test_every_label_has_a_name_and_the_right_hemisphere() -> None:
    for hemi, prefix in (("R", "Right"), ("L", "Left")):
        for v in labels.FILLED_LABELS[hemi]:
            assert v in labels.LUT_NAMES
            assert v == labels.CC or labels.LUT_NAMES[v].startswith(prefix)
        assert labels.WM_LABEL[hemi] in labels.FILLED_LABELS[hemi]
        assert labels.CORTEX_LABEL[hemi] not in labels.FILLED_LABELS[hemi]
        assert labels.HIPPOCAMPUS_LABEL[hemi] not in labels.FILLED_LABELS[hemi]


def test_the_ventricles_are_outside() -> None:
    for hemi, (lv, ilv, cp) in (("R", (43, 44, 63)), ("L", (4, 5, 31))):
        for v in (lv, ilv, cp):
            assert v in labels.TO_OUTSIDE
            assert v not in labels.FILLED_LABELS[hemi]
            assert v not in labels.LUT_NAMES  # not a class of the set
    assert 24 in labels.TO_OUTSIDE and labels.OUTSIDE == 0


def test_doc_lists_the_labels_that_become_outside() -> None:
    rows = {}
    for line in DOC.read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 3 and cells[1].isdigit() and cells[2].isdigit():
            rows[cells[0]] = (int(cells[1]), int(cells[2]))
    got = {v for pair in rows.values() for v in pair}
    assert got == set(labels.TO_OUTSIDE)
    assert set(rows) == {
        "CSF",
        "Lateral ventricle",
        "Inferior lateral ventricle",
        "Choroid plexus",
    }
