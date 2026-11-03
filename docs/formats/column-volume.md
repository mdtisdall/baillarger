# column-volume: the echoes along the columns of a superblock

SpecVersion `column-volume/1`. The code is `baillarger.column_volume`. This
is a sampler contract (`docs/version-2-plan.md`, §1 and §5c, step 1): the
sampler computes its output on demand, from the output of the ray sampler
and the echoes. Nothing is stored. `tests/test_column_volume.py` checks that
the tables below list the same names, types and values as the code.

The specs in `docs/formats/` stay at version 1 until a release, and are
edited in place (plan, §8).

## Words

| Word | Meaning |
|---|---|
| column, s, edge key, extents, caps, halo column | As in `docs/formats/ray-sampler.md` |
| sample | One point on a column, at s = k × step. The ray sampler makes the samples. The column volume gives them values |
| echo | One FLASH volume of one echo time, on the grid of `wm-dseg/1` |
| voxel coordinates | A⁻¹ applied to a position in scanner mm. A is the `VoxelToRAS` affine. The centre of voxel (i, j, k) is at (i, j, k) |
| chord | The length of a straight line through the centre of a voxel, along a direction, from one face of the voxel to the opposite face on that line. It is the extent of the voxel along the direction |

## Data inputs

The sampler reads these inputs and no other data:

| Input | Spec | What the sampler reads |
|---|---|---|
| The rays of one call | `ray-sampler/1` (`Rays`) | For each column: the root, the unit normal, the samples (s and position), and the label intervals |
| The echoes | The grid of `wm-dseg/1` | One 3D volume for each echo, on the grid of the segmentation: the same shape, and the same affine (`VoxelToRAS`) |

The sampler refuses an echo (`ValueError`) if:

1. its shape is not the `GridShape` of `white-surf/1`;
2. its dtype is not an integer or a floating-point type; or
3. its header affine is not `voxel_to_ras`, to within float32 rounding
   (an absolute difference of at most 1e-5 × the largest entry of the
   affine on every entry).

The caller gives `voxel_to_ras` and `grid_shape` (the `VoxelToRAS` and
`GridShape` of the sidecar of `white-surf/1`) and the header affine of each
echo (`echo_affines`). So a
`voxel_to_ras` that does not match the echoes is an error.

The sampler reads the echoes through the chunked reader
(`baillarger.blocks`, `Source`). It never reads an echo whole.

## The call

```python
# rays: the output of RaySampler.sample. echoes: one Source for each echo.
sampler = ColumnVolumeSampler(echoes, voxel_to_ras, echo_affines, grid_shape)
volume = sampler.sample(rays)
```

`echoes` is a sequence of sources. `echo_affines` is a sequence of the
header affines, one for each echo. The order of the echoes in `echoes` is
the order of the last axis of `ColumnVolume.echo`.

### Parameters

The call has no parameter that selects the columns, the extents or the
step. The rays of the call fix all three:

| Quantity | Where the caller sets it |
|---|---|
| The columns, with the halo | `superblock`, `level` and `halo_mm` of `RaySampler.sample` |
| The extents e_in and e_out of each column, in mm, within the caps | `e_in` and `e_out` of `RaySampler.sample` |
| The step, in mm (default 0.04) | `step` of `RaySampler.sample` |

So the samples of a column are the samples of its ray, and a longer
extent gives the same samples at each s, and more of them (ray sampler,
"Samples"). The caller sets the extents for each column from the label
intervals. `RaySampler.roots` gives the columns before the call to `sample`.

## The columns

`ColumnVolume` has one entry for each column of `rays`, in the order of
`rays`: the owned columns, then the halo columns. The halo columns hold the
same fields as the owned columns. A reader that needs neighbours of the
owned columns uses them. `rays.roots.owned` tells the two sets apart.

## Interpolation

The sampler gives the value of an echo at a sample by trilinear
interpolation of the voxel values, with the rule below.

1. The voxel coordinates v of the sample are A⁻¹ applied to the sample's
   position (float32, scanner mm), in float64. A is in float64.
2. Let n_a be the grid size on axis a. The sample is **valid** if
   0 ≤ v_a ≤ n_a − 1 on every axis a. Then all the voxels that the rule
   uses are in the grid.
3. For a valid sample, on each axis a: i_a = min(floor(v_a), n_a − 2) and
   t_a = v_a − i_a. So t_a is in [0, 1], and t_a = 1 at v_a = n_a − 1.
4. The value is the weighted sum of the 8 voxels at i + (0 or 1 on each
   axis). The weight of the voxel at i + d is the product over the axes of
   (t_a if d_a = 1, else 1 − t_a).
5. The arithmetic is float64: the voxel values convert to float64, the
   weights and the sum are float64. The terms are added in the order of
   the index 4 d_i + 2 d_j + d_k, from 0 to 7. The result is rounded to
   float32.
6. A sample that is not valid has the value NaN for every echo. The sampler
   does not extend the grid and does not clamp the coordinates.
   The ray sampler gives a label to the half voxel beyond the last voxel
   centre, from n_a − 1 to n_a − 0.5 (the voxel is a cube round its centre).
   There, `valid` is false although a label exists. A reader expects this. The sample keeps its position, its s and its partial-volume
   mark.

The sampler does no filtering other than step 4. At a step smaller than a
voxel, neighbouring samples are not independent.

A device implementation gives the same values to float32 rounding. With
`torch.nn.functional.grid_sample`, the convention is `align_corners=True`,
because the centre of a voxel is at an integer.

## The values are the echoes as acquired

The value is the interpolated voxel value of the echo as acquired. The
voxel values keep their dtype in the source. The sampler converts each
voxel value to float64 before the arithmetic. That conversion is exact for
every float32 and every integer type of 32 bits or fewer. Only the result
is rounded to float32 (step 5 of "Interpolation"). There is no conversion to
float32 before the interpolation. The value has no log, no removal of a
trend, no normalisation and no masking. Those are steps of a reader, not of
the sampler.

## The partial-volume mark

A sample is **partial** if it is near a label change on its column.

1. The label changes of a column are the ends of its label intervals that
   are not an end of the caps. That is, the start s of every interval
   except the first. Two intervals that follow each other have different
   labels (ray sampler, "The rule for an interval boundary"), so each such
   start is a label change. The classes of the two labels do not matter:
   a change from label 41 to label 49 is a label change.
2. A boundary where one label is −1 (off the grid) is a label change.
3. The reach w of a column is the chord of a voxel along the column's unit
   normal n:

       w = min over the axes a with u_a ≠ 0 of 1 / |u_a|,   u = M⁻¹ n

   M is the 3 × 3 part of A. The unit of u is voxels per mm, so w is in mm.
   For a ray along a voxel axis, w is the voxel size on that axis. For
   an oblique ray, w is the length of the ray inside the voxel that the ray
   crosses through its centre. A ray with u = 0 does not exist, because n
   is a unit vector.
4. A sample at s is partial if |s − b| ≤ w for at least one label change b
   of its column. s and b are the float32 values of the ray sampler,
   compared in float64. w is float64.

The mark uses the label intervals over the full caps, so a sample near the
end of its extent is marked if a label change lies just beyond the end. The
mark does not depend on the extents.

The mark does not say that the sample is wrong. It says that its value
mixes tissues. Readers choose what to do with it: the plan leaves the marked
samples out of the data term of the joint warp.

## Outputs

All arrays are numpy arrays on the host. n is the number of columns, N the
number of samples, and E the number of echoes.

### `ColumnVolume`

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `rays` | `Rays` | | The rays of the call. They hold the edge key, the owned flag, the position, s and the label intervals of each sample and column. The column volume does not copy them |
| `echo` | float32 | (N, E) | The value of each echo at each sample. NaN where the sample is not valid |
| `valid` | bool | (N,) | True where the rule of "Interpolation" gives a value |
| `partial` | bool | (N,) | True for a partial-volume sample |
| `reach` | float32 | (n,) | w of each column, mm |

The samples of column c are the rows `rays.sample_start[c]` to
`rays.sample_start[c + 1]` of `echo`, `valid` and `partial`, as for the
fields of `Rays`.

`reach` is w of "The partial-volume mark" (step 3): the chord of a voxel
along the unit normal of the column, in mm, rounded to float32. It is the
reach in s of the mark on each side of a label change. It does not depend
on the extents. `reach` is given so that a reader can find the distance in voxels from a
label change without the affine.

## Determinism

1. A sample depends only on its position, the echoes, the affine and the
   label intervals of its column. A column depends only on its root
   and its extents (ray sampler, "Determinism"). It does not depend on the
   superblock, the level, the halo or the other columns.
2. So the same column gives the same samples, bit for bit, whichever
   superblock computes it: as an owned column of one superblock, or as a
   halo column of another. The sampler must read the voxel values it needs
   from the echoes, and not from a block that depends on the superblock.
3. On the CPU, in numpy, there is one code path. A device implementation
   gives the values to float32 rounding, and the mark and `valid` equal
   those of the CPU, except for a sample whose test is within float32
   rounding of its limit.

## Reads

A call reads the voxels of the echoes in the box of the sample positions,
widened by 1 voxel on each side. That is the halo box of the ray sampler's
call, widened by the largest extent of the columns. It reads each chunk one
time for each call, through the chunk cache. Calls on superblocks in Morton
order read each chunk about one time.

## Not in version 1

- A different interpolation (cubic, or a filter that is wider than a voxel).
- A cache of samples between calls.
- Values at the positions of the equivolume method (phase 2.1). When the
  ray sampler adds A(s) and V(s), the column volume does not change, because
  it reads only positions and label intervals.
