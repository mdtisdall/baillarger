# wm-distance: the WM distance through tissue

SpecVersion `wm-distance/1`. The code is `baillarger.wm_distance`. This is a
sampler contract (`docs/version-2-plan.md`, §1): the sampler computes its
output on demand, for one superblock and its halo, from published files
only. Nothing is stored. `tests/test_wm_distance.py` checks that the tables
below list the same names and values as the code.

The specs in `docs/formats/` stay at version 1 until a release, and are
edited in place (plan, §8).

## Words

| Word | Meaning |
|---|---|
| WM | The voxels with a label in the sidecar's `FilledLabels` (the filled mask of `wm-dseg/1`). |
| cortex | The voxels with the cortex label of the hemisphere (`label-set/1`): 42 (R) or 3 (L). |
| free voxel | A WM voxel or a cortex voxel. |
| obstacle | Every other voxel: outside (0, which includes the medium, excisions and the ventricles) and every other label (for example the hippocampus, the amygdala). |
| interface | The faces that a WM voxel and a cortex voxel share. |
| D | The WM distance: the signed length (mm) of the shortest path through free voxels from a voxel's centre to the interface. D > 0 in cortex, D < 0 in the WM. |
| cap | The largest \|D\| that the call gives, in mm. `CAP` = 8.0, the column length cap (`ray-sampler/1`). A call can ask for a smaller cap. |
| block, superblock, level | As in `ray-sampler/1`: blocks of `cells` voxels a side from voxel (0, 0, 0); a superblock is 2^m × 2^m × 2^m blocks; m is the level. |

## Data inputs

| Input | Spec | What the sampler reads |
|---|---|---|
| The corrected segmentation | `wm-dseg/1` (`_desc-wmtopo_dseg.nii.gz`) | The labels, the affine of its header, and the sidecar (`FilledLabels`, `Hemisphere`) |

The sampler reads the labels through the chunked reader (`baillarger.blocks`).
It never reads the segmentation whole.

`wm-dseg/1` has no block size. So the block size is a parameter of the
sampler, `cells`, with the default 64 (`baillarger.mesh.BLOCK`, the block
size of `white-surf/1`). A caller that uses the superblocks of a surface
gives that surface's `BlockCells`.

## The call

```python
# dseg: the _desc-wmtopo_dseg.
sampler = WMDistanceSampler(dseg, cells=64)
r = sampler.sample(superblock, level, halo_mm=0.0, cap=CAP)
```

### Parameters

| Parameter | Type | Default | Rule |
|---|---|---|---|
| `superblock` | tuple of 3 integers | (required) | (I, J, K), each ≥ 0 |
| `level` | integer | (required) | m ≥ 0 |
| `halo_mm` | number | 0.0 | 0 ≤ halo_mm, finite, in mm |
| `cap` | number | `CAP` (8.0) | 0 < cap ≤ `CAP`, in mm |

The sampler refuses a value outside its rule (`ValueError`).

### `WMDistance`

| Field | Type | Meaning |
|---|---|---|
| `distance` | float32 array, the shape of `box` | D in mm at each grid point of the box |
| `box` | tuple of 3 slices | The output box, in grid points |
| `cap` | number | The cap of the call, in mm |

## The values

| Voxel | Value |
|---|---|
| A free voxel with \|D\| ≤ cap | D, in mm (float32). Negative in the WM |
| A cortex voxel with no path of length ≤ cap | +inf |
| A WM voxel with no path of length ≤ cap | −inf |
| An obstacle | NaN |

"No path of length ≤ cap" includes tissue that has no path to the interface
at all. A sampler cannot tell the two apart: it sees only its read box, and
a longer path can leave the box. So the value is ±inf, not NaN, and NaN
means an obstacle only.

## The boxes

1. **The output box** is `blocks.superblock_box(superblock, level, cells,
   halo_mm, affine, shape)`: the grid points that the cells of the
   superblock touch, extended by halo_mm on each side, rounded outward to
   whole grid points per axis, and clipped to the grid.
2. **The read box** is the same box with halo_mm + `MARGIN` × cap in place
   of halo_mm, then `REACH` = 2 more grid points on each side (the reach of
   the second-order stencil), clipped to the grid. `MARGIN` = √3 (see
   "Determinism").

The edge of the grid is a boundary: no path leaves the grid.

## The method

The distance is second-order fast marching (scikit-fmm, BSD licence,
`skfmm.distance` with `order=2`) on the read box, in float64:

- φ = −1 on the WM and +1 on cortex. The obstacles are masked. The zero
  level set of φ is the interface, half-way between the centres of a WM
  voxel and a cortex voxel.
- The grid spacing is the voxel size per axis (the norms of the columns of
  the affine's 3 × 3 part).
- The march stops at the cap (`narrow` = cap). On the phantoms and on the
  hemisphere, this gives the same values, bit for bit, as the full march
  where \|D\| ≤ cap.
- The value is then capped as in "The values", cropped to the output box,
  and given as float32.

If the read box has no interface (no WM voxel shares a face with a cortex
voxel), every free voxel is ±inf.

### Accuracy

- An interface on a grid plane: D is exact to float64 rounding (the method
  is exact in one dimension).
- An oblique plane: \|D − d\| ≤ half the largest voxel size, with d the
  distance to the plane. The labels make the interface a staircase within
  half a voxel of the plane.
- Round an obstacle: D is at most 1.5 voxels longer than a brute-force
  shortest path on 3³ sub-voxels per voxel (measured: 1.3 to 1.4 voxels at
  the edge of a wall). The march knows the voxel centres only, so it
  cannot cut the corner of an obstacle's voxel.

## Determinism

Two calls that hold the same voxel in their output boxes give the same
value within `TOLERANCE` = 1e-5 mm, and the same NaN. They give the same ±inf,
except where \|D\| is within `TOLERANCE` of the cap.

**The cause.** The paths are local: a path shorter than the cap stays
within the cap of its start. The fast-marching values are not as local.
The value at a voxel comes from its upwind neighbours on each axis, and
those come from theirs. So the value depends on a cone of voxels. For a
front oblique to all three axes, the cone reaches about √3 × the distance
along one axis. Past that, the cone has very small weights, but not zero.
So the read box adds `MARGIN` × cap, not the cap. The heap's tie order is
not the cause, so a fast marching with a fixed tie rule does not remove the
difference.

**The tolerance is measured, not proven.** The worst case on the phantoms
is a WM ball in a cortex shell (fronts oblique to all three axes), voxels
0.16 × 0.20 × 0.12 mm, with caps small enough that the read box is smaller
than the grid. The largest difference between a call and the whole array,
in float64:

| Read margin (× cap) | Cap 1 mm | Cap 2 mm |
|---|---|---|
| 1.0 | 1.2e-3 mm | 2.9e-4 mm |
| 1.25 | 8.7e-5 mm | 1.2e-6 mm |
| 1.5 | 3.1e-6 mm | 1.5e-9 mm |
| √3 | 5.8e-6 mm (halo 0.5 mm); 0 (no halo) | 0 |
| 2.0 | 3.8e-8 mm (halo 0.5 mm); 0 (no halo) | 0 |
| 2.5 | 0 | 0 |

On the hemisphere, at a read margin of 1.0, 0.31 % of the voxels that two
superblocks share differ, by at most 5.1e-13 mm.

**What a reader may use the values for.** A reader must not decide anything
at the cap: the change from a finite value to ±inf is a discrete jump, and a
difference within `TOLERANCE` can move a voxel across it. `orient` asks for
a cap of 5 mm and uses only the sign of n · ∇D in cortex, at most about
4.5 mm from its WM, so it stays clear of the jump. A difference of 1e-5 mm
cannot change that sign except where ∇D is about 0, where `orient` gives
the sign 0.

## Cost

The march is a heap: O(n log n) for the n voxels of the read box, on the
CPU. The labels go through the chunk cache, and superblocks are read in
Morton order, so each chunk is read from the file about once. The margin
costs compute, not reads: each call marches over its own read box, which is
((B + 2h)/B)³ × the superblock's voxels, with B the superblock's side and
h = halo_mm + `MARGIN` × cap + 2 voxels.
