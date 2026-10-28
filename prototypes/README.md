# Laminar layers from ex vivo FLASH: the prototype pipeline

This pipeline estimates cortical layers in a box of ex vivo multi-echo FLASH (7 T, 160 µm) around a point you choose. For every column of cortex it finds:

- **the first pial surface:** the open top of the cortex, or the contact where two banks are fused;
- **a relative depth τ:** 0 at the white matter, 1 at that pial surface, fitted so that the laminar bands of neighbouring columns line up;
- **the bands themselves** (for example the stria of Gennari in V1), as levels of τ.

The method is seismic flattening adapted to cortex. Each column is a trace; the layer orientation in the image aligns neighbouring traces; and a cut decides, column by column, where its own cortex ends and the facing bank's begins. The design and its results are in `reports/Crossing paths and contacts from laminar profiles.md` (*The joint warp*) and `reports/Seismic flattening for cortical layers.md`.

> **Status.** These are untracked prototype scripts, not part of the `baillarger` package. Only the solver they use, `baillarger.surfcut`, is in the package (with tests). The pipeline has been run on two boxes of one hemisphere (V1/V2 banks, M1/S1). The last step, putting the levels at the bands, has not been validated yet. See *Limitations*.

---

## What you need

### Input files

Three FLASH magnitude echoes and an aseg, all on one grid, in one directory:

| File | Content |
|---|---|
| `<SUBJECT>_acq-channelCOMBx160um_dir-positive_run-02_echo-1_part-mag_FLASH.nii.gz` | echo 1 |
| `…_echo-2_part-mag_FLASH.nii.gz` | echo 2 (the main contrast) |
| `…_echo-3_part-mag_FLASH.nii.gz` | echo 3 |
| `…_echo-{1,2,3}_part-mag_FLASH.json` | BIDS sidecars; `EchoTime` (seconds) is read from them |
| `<SUBJECT>_aseg.nii.gz` | segmentation, integer labels: **41 WM, 42 cortex, 0 outside**; other labels are ignored |

- The file names are fixed in `src/baillarger/data.py` (`SUBJECT = "sub-INDD121850_ses-7THemix20250516"`). For another subject, name or link your files to that pattern, or change `SUBJECT`.
- The directory is `data/` at the top of the checkout, or the one that `$BAILLARGER_DATA` names.
- The example data and its checksums are described in `DATA.md`. Check them with `shasum -a 256 -c data.sha256`.

### A centroid

The voxel you want to process around, in **0-based voxel coordinates of the full grid** (i, j, k). ITK-SNAP numbers voxels from 1, so subtract 1 from each coordinate it shows.

### Optional: marked points

Points you have marked on the contact (fused cortex), in a JSON file:

```json
{"points_full0": [[455, 250, 496], [456, 251, 497]], "source": "where they came from"}
```

The coordinates are 0-based voxels of the full grid. The points are used for two things:
- **scores:** each step reports how far its pial surface is from them;
- **section views:** the sections are drawn through the planes that hold most points.

Without a points file, those section figures are skipped. To get sections around your centroid anyway, give the centroid as the only point and ignore the "at the points" scores, which assume the points lie on a contact.

### Environment

From the top of the checkout, in the project's Nix devShell (loaded by direnv, or `nix develop`):

```bash
nix develop --command uv sync
```

Every command below runs from the top of the checkout. The scripts also need matplotlib, which they get through `uv run --with matplotlib`.

Heavy steps run under `scripts/memguard <GB> <command>`. It kills the run if its memory goes over the cap and prints the peak. Run heavy steps **one at a time**. The peaks below are for one box on a 64 GB machine.

---

## The steps

Throughout, `SITE` is a short name you choose for this box (letters, digits, `_`). Every output file is named after it.

### 1. Turn the centroid into a box

The scripts work on a cube of 141 voxels (about 22.6 mm) given by its corner. The corner is the centroid minus 70 on each axis:

```bash
python3 -c "c=(455, 250, 496); print(*[x - 70 for x in c])"
```

- **The whole box must lie inside the grid** (the example grid is 700 × 384 × 1148). Move the centroid inwards if it does not.
- **Traces whose roots lie within 2 mm of a box face are excluded.** Regions you care about should be at least about 6 mm from every face, so in practice within about 5 mm of the centroid.
- **The box should contain open cortex near any fused stretch you care about.** The pial surface in fused cortex is carried in from nearby open tops and from the facing bank, so a box that cuts a fused stretch off from its open evidence gives a weaker answer there.

`--size N` (column_volume.py) changes the cube's size, or `--size N1 N2 N3` gives a box. Memory grows about linearly with the number of traces.

### 2. The column volume

This step re-indexes the box's cortex as one trace per white-surface vertex (about one per voxel). Each trace runs along the layer orientation from the white matter, in raw millimetres.

```bash
nix develop --command scripts/memguard 12 uv run --with matplotlib python \
    prototypes/scripts/column_volume.py prototypes/figures prototypes/scripts \
    --name SITE --box I J K [--points path/to/points.json]
```

- **Time and memory:** about 3–5 minutes, about 5–6 GB.
- **Writes** `prototypes/scripts/column_volume_SITE.npz`, which holds the traces, their 3D positions, echoes 2 and 3 along each, a class per trace (open, crossing, own bank, to the contact, failed) and the earlier contact estimates. It also writes figures `prototypes/figures/column_volume_SITE_*.png`.
- The second argument must be `prototypes/scripts`: the scripts import each other from there, and they read and write their data there.

### 3. The single-surface start

A first pial surface from one cut. Open tops are soft anchors, a bound on how fast thickness may change is measured from open cortex, and the data cost is darkness. It is the starting point and the curvature trend for step 4.

```bash
nix develop --command scripts/memguard 28 uv run --with matplotlib python \
    prototypes/scripts/pia_horizon.py prototypes/figures prototypes/scripts SITE
```

- **Time and memory:** about 2–6 minutes, about 17–18 GB (the heaviest step).
- **Writes** `pia_SITE.npz` and `.json`, and `pia_SITE_{sections,slopes}.png`.

This surface fails where a contact shows no dark line. At M1/S1 it was 0.8 mm too deep. Step 4 corrects most of that.

### 4. The joint warp: pial surface and depth together

The main step. It solves, for every trace, a few levels of τ (1/3, 2/3 and 1, the pial surface) together:

- **conservation:** bands line up between neighbouring traces;
- **two-sided ties at contacts:** what one bank leaves past its pial surface, the facing bank must include;
- **robust open-top anchors;**
- **smoothness.**

The solve is exact or near-exact by QPBO (`baillarger.surfcut`), coarse to fine. Then the continuous warp is filled in between the levels, using nothing past the pial surface.

```bash
nix develop --command scripts/memguard 32 uv run --with matplotlib python \
    prototypes/scripts/joint_site.py prototypes/figures prototypes/scripts SITE --spacing 0.16
```

- **`--spacing 0.16`** is full density (one trace per 0.16 mm). The default, 0.3, is a quicker subsample: about 3 minutes and 4 GB.
- **Time and memory at full density:** about 10 minutes, about 8–11 GB.
- **`--rounds 3`** re-solves from its own result. At M1/S1 it reached a fixed point after one extra round and changed little.
- **Writes** `joint_SITE_s0.16.npz` and `.json`. The npz holds `sub` (the traces used), `z` (the levels' depths, mm), `cert` (where QPBO certified them), `tg` and `zf` (the warp: depth of each trace at each τ). The figures are:
  - `joint_SITE_s0.16_sections.png`: echo 2 along the traces, with the levels and the pial surface;
  - `_tau_sections.png`: the same planes resampled at constant τ;
  - `_profiles.png`: stacked profiles in raw depth and in τ.
- **What the log reports:** each pass's energy, lower bound, the fraction certified, and the windows' edge hits, which must reach 0. The JSON has the scores at your points, if you gave any, beside the single-surface start on the same traces.

### 5. Find the bands

Each trace is resampled at constant τ and stacked with its neighbours (within 1 mm), weighting each neighbour by how well its profile agrees, so areal borders stay sharp. The bands are the dark and bright extrema of the site's stacked profile that are at least 1.5 voxels wide.

```bash
nix develop --command scripts/memguard 8 uv run --with matplotlib python \
    prototypes/scripts/tau_bands.py prototypes/figures prototypes/scripts SITE SITE_s0.16
```

- **Time and memory:** about 1–2 minutes, about 1 GB. The fourth argument is the joint result's tag: `SITE_s0.16`, or `SITE_s0.16_r3` after `--rounds 3`.
- **Writes** `bands_SITE_s0.16.json`. Per band it gives the kind (dark or bright), its τ, its depth at the median pial surface, the fraction of traces it was found on, and its spread in τ against its spread in raw depth.
- **Also writes** `bands_SITE_s0.16_profile.png` (the stacked profiles and the bands) and `_maps.png` (each band's τ and strength on the roots of the marked planes; needs points).

For example, at the V1/V2 banks: bright at τ 0.18, dark at 0.37 (the stria), bright at 0.81.

### 6. Levels at the bands (not yet validated)

This re-solves with the levels placed at the bands. Each band's stacked evidence becomes a cost on its level, starting from step 4's warp. Copy the bands' τ from `bands_SITE_s0.16.json`:

```bash
nix develop --command scripts/memguard 32 uv run --with matplotlib python \
    prototypes/scripts/joint_site.py prototypes/figures prototypes/scripts SITE --spacing 0.16 \
    --levels 0.177,0.370,0.807 --warp SITE_s0.16 --bands SITE_s0.16
```

- **Time and memory:** about 10 minutes, about 10 GB.
- **Writes** `joint_SITE_s0.16_bands.npz` and its figures, as in step 4. Each band is then a level set of τ.
- The fill between the levels uses distance. Equal volume would be the principled choice, but no estimator of it has been validated yet.

### Viewing in ITK-SNAP

`chart.py` writes a local chart of the column volume around a point as NIfTI cubes for ITK-SNAP. It takes 1-based ITK-SNAP coordinates of `prototypes/data/column_volume_SITE_echo2_detrended.nii.gz`, which is written on first use:

```bash
nix develop --command uv run python prototypes/scripts/chart.py prototypes/scripts prototypes/data SITE I J K 3
```

The joint warp's results (pial surface, τ) are not exported as NIfTI yet. They are in the `joint_*.npz` files, per trace.

---

## Summary

| Step | Command | Main output | Time, memory |
|---|---|---|---|
| 1 | centroid − 70 → `I J K` | — | — |
| 2 | `column_volume.py … --name SITE --box I J K [--points …]` | `column_volume_SITE.npz` | 3–5 min, 6 GB |
| 3 | `pia_horizon.py … SITE` | `pia_SITE.npz` | 2–6 min, 18 GB |
| 4 | `joint_site.py … SITE --spacing 0.16` | `joint_SITE_s0.16.npz` | 10 min, 10 GB |
| 5 | `tau_bands.py … SITE SITE_s0.16` | `bands_SITE_s0.16.json` | 1–2 min, 1 GB |
| 6 | `joint_site.py … --levels … --warp SITE_s0.16 --bands SITE_s0.16` | `joint_SITE_s0.16_bands.npz` | 10 min, 10 GB |

About 30–35 minutes per box in all, on one core.

---

## Limitations

- **Tested on one hemisphere, at 160 µm, with three echoes.** Several scales assume about 0.16 mm voxels: the 0.04 mm sampling, the 1.5-voxel band width and the tapers.
- **Accuracy so far** (median and 90th percentile of the distance from the marked contact points):
  - V1/V2 banks: 0.12 / 0.36 mm;
  - M1/S1: 0.20 / 0.44 mm, about 0.2 mm too deep;
  - shallow excursions where nothing anchors the surface.
  
  The solver is near-exact (96.6–99.8 % certified), so these errors come from the model: its weights, the distance prior and the anchors.
- **Model weights are first-version values,** stated in `joint_levels.Params`, not calibrated. A synthetic test volume with known answers is planned for that.
- **Outputs are per trace (npz) and figures,** not NIfTI volumes.
- **Box faces:** results within about 2 mm of a face are excluded, and those within about 6 mm are less reliable. Large regions need overlapping boxes, which is not automated.
- **Not tracked in git.** These scripts change as the work goes on. The reports describe the current state of each step.

---

## Layout

- `scripts/`: the pipeline above, its libraries (`crossing_lib.py`, `flatten_lib.py`, `joint_levels.py`, `volume_views.py`, …) and their tests (`test_*.py`; run with `nix develop --command uv run pytest -q prototypes/scripts/test_joint_levels.py`, for example). The many other scripts are earlier experiments behind the reports' findings; `reports/Seismic flattening for cortical layers.md` (*Code, data and figures*) says which produced what.
- `data/`: crops for ITK-SNAP, overlays, and the marked points (`M1S1_points_full0.json`, `V1V2_banks_points_full0.json`, and the earlier per-site point files).
- `figures/`: the figures written by the steps and cited in the reports.
