"""Tests for baillarger.bids: entities, the derivatives layout, sidecars
checked against their spec, and the spec documents against the code."""

import json
from pathlib import Path

import pytest

from baillarger import bids, white
from baillarger.bids import Key, SidecarSpec

DOCS = Path(__file__).resolve().parents[1] / "docs" / "formats"


def test_entities_of_a_bids_name() -> None:
    e = bids.entities(
        "/x/sub-INDD121850_ses-7THemix20250516_acq-a1_echo-2_part-mag_FLASH.nii.gz"
    )
    assert e == dict(
        sub="INDD121850", ses="7THemix20250516", acq="a1", echo="2", part="mag"
    )
    assert bids.entities("sub-1_ses-2_aseg.nii.gz") == dict(sub="1", ses="2")
    with pytest.raises(ValueError, match="sub- and ses-"):
        bids.entities("sub-1_aseg.nii.gz")
    with pytest.raises(ValueError, match="not a key-value"):
        bids.entities("sub-1_ses-2_bad_aseg.nii.gz")


def test_derivative_paths_follow_the_bids_order(tmp_path: Path) -> None:
    e = dict(sub="A", ses="B", acq="ignored")
    p = bids.derivative_path(tmp_path, e, "white", ".surf.gii", desc="x", hemi="R")
    assert p == tmp_path / "sub-A/ses-B/anat/sub-A_ses-B_hemi-R_desc-x_white.surf.gii"
    with pytest.raises(ValueError, match="ENTITY_ORDER"):
        bids.derivative_path(tmp_path, e, "dseg", ".nii.gz", foo="1")
    assert bids.sidecar_path(p) == p.with_name("sub-A_ses-B_hemi-R_desc-x_white.json")


def test_dataset_description_is_written_once(tmp_path: Path) -> None:
    p = bids.dataset_description(tmp_path / "deriv")
    d = json.loads(p.read_text())
    assert (
        d["DatasetType"] == "derivative" and d["GeneratedBy"][0]["Name"] == "baillarger"
    )
    p.write_text('{"Name": "kept"}')
    bids.dataset_description(tmp_path / "deriv")
    assert json.loads(p.read_text()) == {"Name": "kept"}


def test_a_sidecar_must_match_its_spec(tmp_path: Path) -> None:
    spec = SidecarSpec(
        "t/1",
        (
            Key("SpecVersion", "string", True, ""),
            Key("N", "integer", True, ""),
            Key("L", "array of integers", False, ""),
        ),
    )
    good = dict(SpecVersion="t/1", N=3, L=[1, 2])
    bids.write_sidecar(tmp_path / "a.json", spec, good)
    assert bids.read_sidecar(tmp_path / "a.json", spec) == good
    spec.check(dict(SpecVersion="t/1", N=3))  # L is optional
    for bad, msg in (
        (dict(SpecVersion="t/1"), "missing key 'N'"),
        (dict(SpecVersion="t/1", N=3, X=1), "unknown key 'X'"),
        (dict(SpecVersion="t/1", N=True), "'N' is not integer"),
        (dict(SpecVersion="t/1", N=3, L=[1, "2"]), "'L' is not array of integers"),
        (dict(SpecVersion="t/2", N=3), "SpecVersion"),
    ):
        with pytest.raises(ValueError, match=msg):
            spec.check(bad)
        with pytest.raises(ValueError):
            bids.write_sidecar(tmp_path / "b.json", spec, bad)
    (tmp_path / "c.json").write_text(json.dumps(dict(SpecVersion="t/2", N=3)))
    with pytest.raises(ValueError, match="SpecVersion"):
        bids.read_sidecar(tmp_path / "c.json", spec)


def doc_keys(doc: Path, version: str) -> set[tuple[str, str, bool]]:
    """(key, type, required) of the table under '## Sidecar keys: <version>'."""
    lines = doc.read_text().splitlines()
    i = lines.index(f"## Sidecar keys: {version}")
    out = set()
    for line in lines[i + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("| `"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            out.add((cells[0].strip("`"), cells[1], cells[2] == "yes"))
    return out


@pytest.mark.parametrize(
    ("doc", "spec"),
    [
        ("wm-dseg.md", white.DSEG_SPEC),
        ("wm-dseg.md", white.FIXES_SPEC),
        ("white-surf.md", white.SURF_SPEC),
    ],
)
def test_spec_documents_list_the_code_keys(doc: str, spec: SidecarSpec) -> None:
    want = {(k.name, k.type, k.required) for k in spec.keys}
    assert doc_keys(DOCS / doc, spec.version) == want
