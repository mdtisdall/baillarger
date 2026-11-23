# sigma-steps: the surface Σ at each step of a run

SpecVersion `sigma-steps/1`. Written by `write_step` in
`prototypes/scripts/sigma_steps.py` (which also converts a run's `solve-NNN.npz`
snapshots, once or with `--watch`); read by `baillarger.viewer.sigma_steps.SigmaSteps`
(`python -m baillarger.viewer --sigma-dir DIR`). This is a viewing format: nothing
measured is computed from it. No code enforces the sidecar's keys yet, and
`tests/test_bids.py` does not check this table.

## Files

All the files of a run are in one directory `DIR`. For a run with the prefix
`<prefix>` (for example `sub-<sub>_ses-<ses>_hemi-<h>_desc-sigma`) and step number
`<N>`:

| File | Content |
|---|---|
| `<prefix>_step-<N>.surf.gii` | GIfTI: Σ at step `N`, embedded |
| `<prefix>_step-<N>.json` | Sidecar, `sigma-steps/1` |

The viewer takes every file in `DIR` that matches `*_step-<N>.surf.gii`, with `N` a
non-negative integer, and puts the steps in the order of `N` (the writer pads `N` to
three digits, but the reader does not need that). It reads `DIR` again at each list,
so the steps that a running job writes appear as they are written. The writer saves
each surface under a temporary name and renames it, so a reader never sees half a
file.

| File | Data array | Type | Shape | Intent |
|---|---|---|---|---|
| `.surf.gii` | 1: vertices, world mm (scanner RAS) | float32 | n × 3 | `NIFTI_INTENT_POINTSET` |
| | 2: triangles, vertex indices | int32 | m × 3 | `NIFTI_INTENT_TRIANGLE` |

The coordinates are in the same world mm as the white surface that the viewer shows,
so the viewer needs no transform. (A run that works on a grid in voxel mm adds the
world position of voxel (0, 0, 0) when it writes the step.)

## Sidecar

| Key | Type | Required | Content |
|---|---|---|---|
| `SpecVersion` | string | yes | `sigma-steps/1` |
| `Description` | string | yes | What the surface is |
| `Step` | integer | yes | `N` |
| `Vertices` | integer | yes | n |
| `Faces` | integer | yes | m |
| `Source` | string | no | The snapshot that the step comes from |
| `step`, `energy`, `E_stop`, `E_tau` | number | no | The run's own record of the step, copied from the snapshot when it has them |

The viewer does not read the sidecar. It shows the surface only as outlines on the
volume pane's slices; it shows no per-vertex data and does not draw Σ in the 3D
pane.
