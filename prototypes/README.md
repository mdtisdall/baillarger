# Prototypes

Research scripts that are not part of the `baillarger` package. They are kept as written: ruff
does not check `prototypes/` (see `pyproject.toml`), and only the tests listed in pytest's
`testpaths` run in CI. Run them from the repository root in the devShell. Most write figures and
data next to the paths they are given; those outputs are git-ignored here.

## The column volume and local charts

`scripts/column_volume.py` re-indexes a box of the full grid as a volume of columns: one trace
per vertex of the smoothed white mesh (about a voxel apart), sampled every 0.04 mm from the root,
with echo 2 and echo 3 (trend removed), each trace's class (open, crossing, own bank, to the
contact, failed), the tracer that made it, and horizons (top, far WM, contact estimates).
`scripts/chart.py` cuts a local 3D chart out of that volume around a point, as NIfTI cubes for
ITK-SNAP. Both are described in the plan *Crossing paths and contacts from laminar profiles*.

1. Build the volume of a box (a few minutes; heavy, so under memguard and 4 threads):

   ```bash
   OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nix develop --command \
     scripts/memguard 12 uv run python prototypes/scripts/column_volume.py \
     <figures dir> prototypes/scripts --name M1S1 --box 378 224 448
   ```

   The box is `--box I J K` (its corner in 0-based full-grid voxels) with `--size N` (default
   141), or `--patch NAME` for one of `baillarger.data.PATCHES`. `--points FILE` is optional: a
   JSON object with `"points_full0": [[i, j, k], ...]` (0-based full-grid voxels). With points the
   volume is also scored at them and sections are drawn through them.
   It writes `column_volume_<name>.npz` (and a JSON summary) to the second argument, and figures
   to the first.

2. Pick a point in ITK-SNAP in `prototypes/data/column_volume_<name>_echo2_detrended.nii.gz`
   (written by the first chart of that volume), and cut a chart around it:

   ```bash
   nix develop --command uv run python prototypes/scripts/chart.py \
     prototypes/scripts prototypes/data <name> <i> <j> <k> <radius mm>
   ```

   The point is in ITK-SNAP's 1-based coordinates of that image (`--zero-based` for 0-based box
   voxels); the radius is in mm. It prints the point's tissue, how far the nearest trace is
   (with a warning over 0.3 mm), and writes, per bank, `chart_<name>_<i>_<j>_<k>_r<radius>_bank<n>_`
   `echo2`, `echo3`, `overlay` (horizons and points; labels `chart_labels.txt`) and `class`
   (labels `chart_class_labels.txt`) to `prototypes/data/`.

Tests: `nix develop --command uv run pytest -q prototypes/scripts/test_crossing_lib.py`.
