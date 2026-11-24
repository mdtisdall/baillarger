"""Σ at the steps of a run as surface files for viewing (2026-11-23): one
GIFTI surface per step, in world mm, named <prefix>_step-<NNN>.surf.gii, with
a JSON sidecar (the step's record). baillarger.viewer --sigma-dir DIR shows
their outlines on the volume pane.

usage: sigma_steps.py SNAPSHOT_DIR OUT_DIR PREFIX dx,dy,dz|GRID.nii.gz [--watch SECONDS]
  converts a run's snapshots (solve-NNN.npz: V in the grid's mm, F) to step
  surfaces; dx,dy,dz is the world position of voxel (0, 0, 0) (the grid's
  axes are the world's). With --watch it goes on as the run writes them.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def write_step(
    out: Path,
    prefix: str,
    step: int,
    V_world: np.ndarray,
    F: np.ndarray,
    meta: dict[str, Any] | None = None,
) -> Path:
    """One step's surface (pointset in world mm, triangles) and its sidecar."""
    out.mkdir(parents=True, exist_ok=True)
    G: Any = nib.gifti
    img = G.GiftiImage(
        darrays=[
            G.GiftiDataArray(
                np.asarray(V_world, np.float32),
                intent="NIFTI_INTENT_POINTSET",
                datatype="NIFTI_TYPE_FLOAT32",
            ),
            G.GiftiDataArray(
                np.asarray(F, np.int32),
                intent="NIFTI_INTENT_TRIANGLE",
                datatype="NIFTI_TYPE_INT32",
            ),
        ]
    )
    p = out / f"{prefix}_step-{step:03d}.surf.gii"
    tmp = out / f".{p.name}.tmp.gii"
    nib.save(img, tmp)
    tmp.rename(p)  # the viewer never reads half a file
    side = {
        "SpecVersion": "sigma-steps/1",
        "Description": "Σ at one step of a run, world mm",
        "Step": step,
        "Vertices": len(V_world),
        "Faces": len(F),
        **(meta or {}),
    }
    p.with_name(p.name.replace(".surf.gii", ".json")).write_text(
        json.dumps(side, indent=1, default=float)
    )
    return p


def main() -> None:
    snaps, out, prefix = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    a4 = sys.argv[4]  # numbers, or the grid's NIfTI (its affine's translation)
    off = (
        np.asarray(nib.load(a4).affine, float)[:3, 3]
        if a4.endswith((".nii", ".nii.gz"))
        else np.array([float(x) for x in a4.split(",")])
    )
    watch = (
        float(sys.argv[sys.argv.index("--watch") + 1])
        if "--watch" in sys.argv
        else None
    )
    while True:
        new = 0
        for p in sorted(snaps.glob("solve-*.npz")):
            k = int(p.stem.split("-")[1])
            if (out / f"{prefix}_step-{k:03d}.surf.gii").exists():
                continue
            try:
                z = np.load(p, allow_pickle=True)
                V, F = np.asarray(z["V"], float), np.asarray(z["F"])
                meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
            except Exception:  # noqa: BLE001 (a snapshot still being written: next round)
                continue
            keep = {
                a: meta[a] for a in ("step", "energy", "E_stop", "E_tau") if a in meta
            }
            write_step(out, prefix, k, V + off, F, {"Source": str(p), **keep})
            new += 1
        if new:
            print(time.strftime("%H:%M:%S"), new, "new steps", flush=True)
        if watch is None:
            break
        time.sleep(watch)


if __name__ == "__main__":
    main()
