# Example data

The example data is too large for git (about 1.1 GB). It goes in `data/` at
the top of the main checkout, which git ignores. New worktrees get a link to
it (`.worktree-links`). Tests that need it skip when it is missing; the other
tests use synthetic phantoms.

## Files

One ex vivo hemisphere: subject INDD121850, session 7THemix20250516.

| File (in `data/`) | Size | Content |
|---|---|---|
| `sub-INDD121850_ses-7THemix20250516_acq-channelCOMBx160um_dir-positive_run-02_echo-1_part-mag_FLASH.nii.gz` | 357 MB | FLASH magnitude, echo 1 |
| `…_echo-2_part-mag_FLASH.nii.gz` | 354 MB | FLASH magnitude, echo 2 |
| `…_echo-3_part-mag_FLASH.nii.gz` | 351 MB | FLASH magnitude, echo 3 |
| `…_echo-{1,2,3}_part-mag_FLASH.json` | 2 kB each | BIDS sidecars |
| `sub-INDD121850_ses-7THemix20250516_aseg.nii.gz` | 11 MB | PURPLE segmentation |

All images share one grid: 700 × 384 × 1148 voxels of 0.16028 × 0.16 ×
0.16028 mm (int16, RAS). The three echoes are in register.

Check the files (at the top of the checkout):

```bash
shasum -a 256 -c data.sha256
```

## Acquisition

7 T Siemens, 3D gradient echo (`a_gre_exvivo_160um_TR80`): TR 80 ms, flip
angle 25°, echo times 9.37, 20.71 and 32.05 ms, 0.16 mm nominal. The
specimen is in Fomblin, which gives no signal.

## Labels

The aseg labels used: 0 outside, 41 WM, 42 cortex. Other labels (43, 49–53,
192) are other structures. The rim labels of the method are a relabel: 1
outside (aseg 0), 2 WM (41), 3 cortex (42), 0 everything else.

Some blocks of tissue were cut out before the scan. The segmentation imputes
labels there, but the FLASH is black.

## Patches

`baillarger.data.load_patch(name)` reads one box from the three echoes and
the aseg (only the box is kept in memory) and makes the rim labels. The data
directory is `data/` here, or `$BAILLARGER_DATA`.

Voxel boxes (lo inclusive, hi exclusive, in (i, j, k) of the grid above).
Each box spans the boundary points (Dan's areal boundary points) plus a
margin of 10 mm (12 mm for M1S1) on every side.

| Patch | Boundary points | lo | hi | Shape |
|---|---|---|---|---|
| V1V2 | (280, 99, 167), (291, 124, 287), (276, 99, 287) | (214, 37, 105) | (354, 187, 350) | 140 × 150 × 245 |
| PCC_SPL | (514, 180, 422) | (452, 118, 360) | (577, 243, 485) | 125 × 125 × 125 |
| M1S1 | (455, 250, 496) | (380, 175, 421) | (531, 326, 572) | 151 × 151 × 151 |

Review points in V1V2 for the seams in fused (kissing) cortex, in V1V2 crop
coordinates (add (214, 37, 105) for the full grid):

| Point | What is there |
|---|---|
| (69, 65, 69), (68, 60, 71) | A long collapsed sulcus with a faint dark line |
| (81, 49, 162) | A very thin space between kissing cortex; a cortical bridge over a CSF tunnel |
| (61, 94, 78) | An open sulcus wall (an earlier false slit) |
