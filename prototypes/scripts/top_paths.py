"""Input paths for the hemisphere top-stage scripts, from the environment, so
that no script names a data directory:

    TOP_DSEG    the segmentation (dseg) NIfTI
    TOP_SURF    the white surface (.surf.gii)
    TOP_NOISE   orient's noise model (JSON)
    TOP_ECHOES  the echo NIfTI files in echo order, separated by ":"
"""
import os
from pathlib import Path


def _get(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set {name}: see prototypes/scripts/top_paths.py")
    return value


def dseg() -> Path:
    return Path(_get("TOP_DSEG"))


def surf() -> Path:
    return Path(_get("TOP_SURF"))


def noise() -> Path:
    return Path(_get("TOP_NOISE"))


def echoes() -> list[Path]:
    return [Path(p) for p in _get("TOP_ECHOES").split(":")]
