"""BIDS-style file names, the derivatives layout, and JSON sidecars checked
against a written spec.

Inputs are paths to BIDS-named files: entities reads their key-value pairs
(sub-<label>_ses-<label>_..._<suffix>.<ext>). Outputs go to a BIDS
derivatives root: derivative_path builds
<root>/sub-<sub>/ses-<ses>/anat/sub-<sub>_ses-<ses>[_<key>-<value>...]_<suffix><ext>,
with the entities in the BIDS order, and dataset_description writes the
root's dataset_description.json once.

Each output format has a spec in docs/formats/ and a SidecarSpec here: the
sidecar's keys, each with a type and whether it is required. write_sidecar
and read_sidecar refuse a sidecar with a missing required key, a key that is
not in the spec, a value of the wrong type, or another SpecVersion, so that
the file and its spec cannot drift apart. tests/test_bids.py checks that each
spec document lists the same keys as its SidecarSpec.
"""

import datetime
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baillarger import __version__

#: the BIDS order of the entities that baillarger's outputs use
ENTITY_ORDER = ("sub", "ses", "acq", "run", "echo", "part", "hemi", "space", "desc")
BIDS_VERSION = "1.10.0"
CODE_URL = "https://github.com/mdtisdall/baillarger"

_ENTITY = re.compile(r"^([a-zA-Z0-9]+)-([a-zA-Z0-9]+)$")


def entities(path: Path | str) -> dict[str, str]:
    """The key-value entities of a BIDS file name, in their order. Raises
    ValueError unless sub and ses are among them."""
    name = Path(path).name.split(".", 1)[0]
    out = {}
    for part in name.split("_")[:-1]:
        m = _ENTITY.match(part)
        if m is None:
            raise ValueError(f"{Path(path).name}: {part!r} is not a key-value entity")
        out[m.group(1)] = m.group(2)
    if "sub" not in out or "ses" not in out:
        raise ValueError(f"{Path(path).name}: no sub- and ses- entities")
    return out


def derivative_path(
    root: Path, ents: dict[str, str], suffix: str, ext: str, **extra: str
) -> Path:
    """<root>/sub-<sub>/ses-<ses>/anat/<name>: the name has sub, ses and the
    extra entities (hemi=, desc=, ...), in the BIDS order, then the suffix
    and the extension (with its dot)."""
    e = {"sub": ents["sub"], "ses": ents["ses"], **extra}
    unknown = set(e) - set(ENTITY_ORDER)
    if unknown:
        raise ValueError(f"entities not in ENTITY_ORDER: {sorted(unknown)}")
    name = "_".join(f"{k}-{e[k]}" for k in ENTITY_ORDER if k in e)
    return (
        root / f"sub-{e['sub']}" / f"ses-{e['ses']}" / "anat" / f"{name}_{suffix}{ext}"
    )


def dataset_description(root: Path) -> Path:
    """Write <root>/dataset_description.json (a baillarger derivative) if it
    is not there. Returns its path."""
    p = root / "dataset_description.json"
    if not p.exists():
        root.mkdir(parents=True, exist_ok=True)
        d = dict(
            Name="baillarger",
            BIDSVersion=BIDS_VERSION,
            DatasetType="derivative",
            GeneratedBy=[
                dict(Name="baillarger", Version=__version__, CodeURL=CODE_URL)
            ],
        )
        p.write_text(json.dumps(d, indent=1) + "\n")
    return p


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict[str, Any]:
    """The commit of the checkout this module is in, and whether tracked files
    differ from it; None for both outside a git checkout."""
    repo = Path(__file__).resolve().parent

    def git(*args: str) -> str | None:
        r = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True
        )
        return r.stdout.strip() if r.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return dict(
        commit=git("rev-parse", "HEAD"),
        dirty=None if status is None else bool(status),
    )


def provenance(function: str) -> dict[str, Any]:
    """The Provenance sidecar value: when, by what function, version and
    commit."""
    g = git_state()
    return dict(
        Created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        Function=function,
        Version=__version__,
        Commit=g["commit"],
        Dirty=g["dirty"],
    )


def sources(*paths: Path) -> dict[str, Any]:
    """The Sources and SourcesSHA256 sidecar values of input files."""
    rs = [str(p.resolve()) for p in paths]
    return dict(
        Sources=rs,
        SourcesSHA256={r: file_sha256(p) for r, p in zip(rs, paths, strict=True)},
    )


# ---------------------------------------------------------------- sidecars

#: the type names of a spec, and their check
_TYPES = {
    "string": lambda x: isinstance(x, str),
    "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
    "number": lambda x: isinstance(x, int | float) and not isinstance(x, bool),
    "boolean": lambda x: isinstance(x, bool),
    "object": lambda x: isinstance(x, dict),
    "array of integers": lambda x: (
        isinstance(x, list)
        and all(isinstance(t, int) and not isinstance(t, bool) for t in x)
    ),
    "array of numbers": lambda x: (
        isinstance(x, list)
        and all(isinstance(t, int | float) and not isinstance(t, bool) for t in x)
    ),
    "array of strings": lambda x: (
        isinstance(x, list) and all(isinstance(t, str) for t in x)
    ),
    "array of objects": lambda x: (
        isinstance(x, list) and all(isinstance(t, dict) for t in x)
    ),
}


@dataclass(frozen=True)
class Key:
    """One sidecar key: its name, type (a key of _TYPES), whether it is
    required, and its meaning (the spec document says more)."""

    name: str
    type: str
    required: bool
    meaning: str


@dataclass(frozen=True)
class SidecarSpec:
    """The keys of one format's sidecar, and the SpecVersion it carries."""

    version: str
    keys: tuple[Key, ...]

    def __post_init__(self) -> None:
        bad = [k.type for k in self.keys if k.type not in _TYPES]
        if bad:
            raise ValueError(f"unknown types {bad}")

    def check(self, d: dict[str, Any]) -> None:
        """Raises ValueError unless d is a sidecar of this spec."""
        names = {k.name for k in self.keys}
        problems = [f"unknown key {k!r}" for k in d if k not in names]
        for k in self.keys:
            if k.name not in d:
                if k.required:
                    problems.append(f"missing key {k.name!r}")
            elif not _TYPES[k.type](d[k.name]):
                problems.append(f"{k.name!r} is not {k.type}")
        if d.get("SpecVersion") != self.version:
            problems.append(
                f"SpecVersion {d.get('SpecVersion')!r} is not {self.version!r}"
            )
        if problems:
            raise ValueError("; ".join(problems))


#: the keys that every sidecar of baillarger's formats has
COMMON_KEYS = (
    Key("SpecVersion", "string", True, "format and version of the spec"),
    Key("Description", "string", True, "what the file is"),
    Key("Sources", "array of strings", True, "input files (absolute paths)"),
    Key("SourcesSHA256", "object", True, "SHA-256 of each input, by path"),
    Key("Provenance", "object", True, "Created, Function, Version, Commit, Dirty"),
)


def write_sidecar(path: Path, spec: SidecarSpec, d: dict[str, Any]) -> None:
    """Check d against spec and write it as JSON."""
    spec.check(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=1) + "\n")


def read_sidecar(path: Path, spec: SidecarSpec) -> dict[str, Any]:
    """Read a JSON sidecar and check it against spec."""
    d: dict[str, Any] = json.loads(path.read_text())
    spec.check(d)
    return d


def sidecar_path(path: Path) -> Path:
    """The .json next to a .nii.gz, .surf.gii, ... file."""
    return path.with_name(path.name.split(".", 1)[0] + ".json")
