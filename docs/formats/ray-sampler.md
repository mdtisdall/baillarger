# ray-sampler: the columns of a superblock

SpecVersion `ray-sampler/1`. The code is `baillarger.rays`. This is a
sampler contract (`docs/version-2-plan.md`, §1): the sampler computes its
output on demand, for one superblock and its halo, from published files
only. Nothing per column is stored. `tests/test_rays.py` checks that the
tables below list the same names, types and values as the code.

The specs in `docs/formats/` stay at version 1 until a release, and are
edited in place (plan, §8). Phase 2.1 adds the method `equivolume` and the
fields A(s) and V(s) by an edit in place.

## Words

| Word | Meaning |
|---|---|
| root | One vertex of the white surface. Each root has one column. |
| column | The ray of a root and what the ray crosses. A column is keyed by the root's edge key. |
| s | The arc length along the ray, in mm. s = 0 is the root. s < 0 is inward (into the WM below the root). s > 0 is outward. The root is a coordinate, not a depth. |
| caps | The largest extents: 2 mm inward and 8 mm outward. `CAP_IN` = 2.0, `CAP_OUT` = 8.0. |
| extents | e_in and e_out: the part of the ray that the caller asks samples for, from s = −e_in to s = e_out. 0 ≤ e_in ≤ `CAP_IN` and 0 ≤ e_out ≤ `CAP_OUT`. |
| block | A cube of `BlockCells` cells a side of the grid of `white-surf/1`, from voxel (0, 0, 0). Blocks are in Morton order (`docs/formats/white-surf.md`, "Order and the block index"). |
| superblock | An aligned cube of 2^m × 2^m × 2^m blocks. m is the level. Superblock (I, J, K) at level m holds the blocks (i, j, k) with i // 2^m = I, j // 2^m = J, k // 2^m = K. Its blocks are one contiguous range of the Morton order: `morton((I, J, K)) << 3m` to `(morton((I, J, K)) + 1) << 3m`. Level 0 is one block. |

## Data inputs

The sampler reads these files and no other data:

| Input | Spec | What the sampler reads |
|---|---|---|
| The white surface | `white-surf/1` | The vertices (scanner mm), the unit normals, the normal lengths \|n̄\|, the edge keys, the block index, the sidecar (`VoxelToRAS`, `BlockCells`, `GridShape`, `FilledLabels`, `Hemisphere`, `SourcesSHA256`) |
| The corrected segmentation | `wm-dseg/1` (`_desc-wmtopo_dseg.nii.gz`) | The labels, and the affine of its header |

The sampler refuses a segmentation that is not the source of the surface
(its SHA-256 is not in the surface's `SourcesSHA256`). It refuses a
segmentation whose grid is not `GridShape`, or whose affine is not
`VoxelToRAS`.

The sampler reads the labels through the chunked reader
(`baillarger.blocks`). It never reads the segmentation whole.

## The call

```python
# surf: the _white.surf.gii. dseg: the _desc-wmtopo_dseg.
sampler = RaySampler(surf, dseg)
roots = sampler.roots(superblock, level, halo_mm=0.0)
rays = sampler.sample(
    superblock,
    level,
    method="normal",
    e_in=CAP_IN,
    e_out=CAP_OUT,
    step=0.04,
    halo_mm=0.0,
)
```

`roots` gives the columns of a call without their rays. Use it to set
the extents per column before `sample`. `sample` gives the same columns, in
the same order, with their rays (`rays.roots` is equal to `roots`).

### Parameters

| Parameter | Type | Default | Rule |
|---|---|---|---|
| `superblock` | tuple of 3 integers | (required) | (I, J, K), each ≥ 0 |
| `level` | integer | (required) | m ≥ 0 |
| `method` | string | `normal` | One of `METHODS`. Version 1 has `normal` only |
| `e_in` | number, or float array (n,) | `CAP_IN` | 0 ≤ e_in ≤ `CAP_IN`, in mm. An array has one value per column, in the order of `roots` |
| `e_out` | number, or float array (n,) | `CAP_OUT` | 0 ≤ e_out ≤ `CAP_OUT`, in mm. An array as for `e_in` |
| `step` | number | `STEP` (0.04) | step > 0, in mm. One value for the call |
| `halo_mm` | number | 0.0 | halo_mm ≥ 0, in mm |

The sampler refuses a value outside its rule (`ValueError`).

The default step is 0.04 mm: a quarter of a voxel at 160 µm, but 0.8 of a
voxel at 50 µm. A caller on 50 µm data sets the step that it needs. The
label intervals do not depend on the step.

## Which columns a call gives

1. **The owned columns.** The roots that the blocks of the superblock own
   (the rows of the block index whose Morton code is in the superblock's
   range: their vertices `vertex_start` to `vertex_stop`).
2. **The halo columns**, if `halo_mm` > 0. The other roots whose position
   lies in the halo box. The halo box, in voxel coordinates of the grid, is
   the cells of the superblock, from lo_a = I_a × 2^m × `BlockCells` to
   hi_a = (I_a + 1) × 2^m × `BlockCells` on each axis a, with
   halo_mm / vs_a added on each side. vs_a is the voxel size on axis a (the
   norm of column a of the affine's 3 × 3 part). A root at voxel
   coordinates v is in the halo box if lo_a − halo_mm / vs_a ≤ v_a ≤
   hi_a + halo_mm / vs_a on each axis. The box is not a ball: it holds
   every root within halo_mm of the superblock, and more.

The order: the owned columns, then the halo columns, each in the order of
the vertices in `white-surf/1`. Each edge key occurs one time in a call.
`owned` tells the two sets apart.

The root's voxel coordinates v are A⁻¹ applied to the vertex's scanner mm
position, in float64, with A the `VoxelToRAS` affine in float64.

## The ray of a column (method `normal`)

The ray is straight, along the root's unit normal n (the smoothed white
normal of `white-surf/1`, out of the WM), from the root x₀:

    p(s) = x₀ + s n

x₀ and n are in scanner mm, as the files give them (float32). A straight ray
in scanner mm is straight in voxel coordinates: v(s) = v₀ + s u, with
v₀ = A⁻¹ x₀ and u = M⁻¹ n (M is the 3 × 3 part of A). The sampler computes
v₀ and u in float64.

### Samples

The samples are at s_k = k × step for the integers k from −K_in to K_out:

    K_in  = floor(e_in / step + 1e-6)
    K_out = floor(e_out / step + 1e-6)

in float64. So e_in = 2 and step = 0.04 give K_in = 50. Sample k is at the
same s, and has the same position, for every extent that contains it. A
longer extent only adds samples.

The values: s_k is the float64 product k × step, rounded to float32. The
position is p(s_k) computed in float64 from x₀, n and the float64 s_k, then
rounded to float32. A sample beyond the grid has its position like any
other sample. Its label is "off the grid" (see below).

## The label classes

Each voxel label has one class. The map uses the sidecar of `wm-dseg/1`
and the label set (`label-set/1`, `baillarger.labels`):

| Class | Code | Labels |
|---|---|---|
| `OUTSIDE` | 0 | 0 (outside, which includes fluid and excisions) |
| `WM` | 1 | The labels in the sidecar's `FilledLabels` |
| `CORTEX` | 2 | The cortex label of the hemisphere: 42 (R) or 3 (L) |
| `OTHER` | 3 | Every other label, for example the hippocampus (53, 17), the amygdala (54, 18) and the inferior lateral ventricle (44, 5) |
| `OFF_GRID` | 4 | No label: the part of the ray outside the grid. Its label is −1 |

The class of a label follows from `FilledLabels`. If a later phase adds a
label to the filled mask (for example the hippocampus, phase 1.8), that
label becomes `WM` with no change to this spec.

## The label intervals

The label intervals of a column cover the full caps, from s = −`CAP_IN` to
s = `CAP_OUT`, whatever the extents and the step. They are the same for
every call that gives the column.

### The rule for an interval boundary

The sampler finds the voxels that the ray crosses with an exact traversal
(Amanatides and Woo, 1987):

1. Each voxel c (an integer index on each axis) is the cube
   c_a − 0.5 ≤ v_a < c_a + 0.5 round its centre. The voxel that holds
   a point v is floor(v_a + 0.5) on each axis.
2. The ray crosses a face of a voxel where v_a(s) = c_a + 0.5 for an
   integer c_a, on an axis a with u_a ≠ 0. The crossings are at
   s = (c_a + 0.5 − v₀_a) / u_a, in float64.
3. Between two crossings that follow each other, the ray is in one voxel.
   The label along the ray, L(s), is the label of that voxel, or −1 where
   the voxel is outside the grid.
4. An interval is a maximal run of s in [−`CAP_IN`, `CAP_OUT`] with one
   value of L(s). Its start and its end are crossings, or a cap.

The tie rules:

- If the ray crosses faces on two or three axes at the same s (it passes
  through an edge or a corner of a voxel), it steps on all those axes at
  that s. It visits no voxel for a zero length.
- If the ray lies in a face plane (u_a = 0 and v₀_a = c_a + 0.5), it is in
  the voxel c_a + 1 on that axis (rule 1).
- A crossing at a cap (s = −`CAP_IN` or s = `CAP_OUT`) makes no interval of
  zero length.

The intervals of a column tile [−`CAP_IN`, `CAP_OUT`] with no gap and no
overlap. Two intervals that follow each other have different labels, but
they can have the same class (for example 41 and 49, which are both `WM`).

The traversal runs on the CPU, in float64, in one code path. So the
intervals are the same, bit for bit, on every device.

## The derived class of a column

A summary for diagnostics only (plan, §4c, step 3). It decides nothing.
It uses only the intervals, and only their outward part, s in (0,
`CAP_OUT`]:

| Derived class | Code | Rule |
|---|---|---|
| `OPEN` | 1 | An `OUTSIDE` interval starts before the first `WM` interval that comes after a `CORTEX` interval |
| `IN_WM` | 2 | A `WM` interval that comes after a `CORTEX` interval starts before the first `OUTSIDE` interval |
| `NEITHER` | 3 | Neither of the two, within `CAP_OUT` |

"Comes after a `CORTEX` interval" means: its start is at or after the end
of an outward `CORTEX` interval. The WM at the root (the start of the
outward part, before any cortex) does not count. `OTHER` and `OFF_GRID`
intervals do not change the derived class.

A ray that goes from its WM into `OUTSIDE` with no cortex between is
`OPEN`. Such a ray can be at a cut face (plan, §3c, step 2). The diagnostics
count it from the intervals.

There is no bank split, no meeting point and no stop.

## Outputs

All arrays are numpy arrays on the host. n is the number of columns, N the
number of samples, and Q the number of intervals, in the call.

The halo columns hold the same fields as the owned columns, with the same
values that the superblock that owns them gives. So a caller can use them
as neighbours of the owned columns.

### `Roots`

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `edge_key` | int64 | (n,) | The root's edge key (`white-surf/1`). The key of the column |
| `vertex` | int64 | (n,) | The root's row in the vertex array of `white-surf/1` |
| `owned` | bool | (n,) | True for an owned column, false for a halo column |
| `root` | float32 | (n, 3) | x₀: the root's position, scanner mm |
| `normal` | float32 | (n, 3) | n: the root's unit normal, scanner space, out of the WM |
| `normal_length` | float32 | (n,) | \|n̄\|: the length of the smoothed normal before its last division, in [0, 1] |

### `Rays`

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `roots` | `Roots` | | The columns of the call |
| `method` | string | | The method of the call |
| `step` | float | | The step of the call, mm |
| `k_in` | int32 | (n,) | K_in of each column |
| `k_out` | int32 | (n,) | K_out of each column |
| `sample_start` | int64 | (n + 1,) | Column c has the samples `sample_start[c]` to `sample_start[c + 1]`, for k from −K_in to K_out |
| `s` | float32 | (N,) | s_k of each sample, mm |
| `position` | float32 | (N, 3) | p(s_k) of each sample, scanner mm |
| `interval_start` | int64 | (n + 1,) | Column c has the intervals `interval_start[c]` to `interval_start[c + 1]`, in order of s |
| `interval_s0` | float32 | (Q,) | The start s of each interval, mm |
| `interval_s1` | float32 | (Q,) | The end s of each interval, mm |
| `interval_label` | int32 | (Q,) | The label of each interval, −1 off the grid |
| `interval_class` | uint8 | (Q,) | The class of each interval (code) |
| `derived_class` | uint8 | (n,) | The derived class of each column (code) |

The float64 interval boundaries are rounded to float32. The first interval
of a column starts at −`CAP_IN` and the last ends at `CAP_OUT`.

### Not in version 1

These are names for phase 2.1. Version 1 does not give them:

- the method `equivolume`;
- A(s), the cross-section of the column at each sample, mm²;
- V(s), the volume of the column from its inward end, mm³, with V(−e_in) = 0;
- the mark on each sample where A(s) ≤ 0.

## Determinism

1. A column depends only on its root (x₀, n, \|n̄\|, edge key), the affine,
   the labels, and the call's method, step and the column's extents. It does
   not depend on the superblock, the level, the halo or the other columns.
2. So the same column is the same, bit for bit, whichever superblock
   computes it: as an owned column of one superblock, or as a halo column of
   another.
3. The intervals, their labels and classes, and the derived class are the
   same, bit for bit, on every device (one code path on the CPU).
4. The samples are exact float32 roundings of float64 values. A device
   implementation must give them to float32 rounding.

## Reads

A call reads the labels of the halo box widened by `CAP_OUT` + 1 voxel on
each side (a ray can point in any direction, and `CAP_OUT` is larger than
`CAP_IN`). It reads them one time for each call, through the chunk cache.
Calls on superblocks in Morton order read each chunk about one time.
