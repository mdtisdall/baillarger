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
    for hemi, (lv, cp) in (("R", (43, 63)), ("L", (4, 31))):
        assert lv not in labels.FILLED_LABELS[hemi]
        assert cp not in labels.FILLED_LABELS[hemi]
