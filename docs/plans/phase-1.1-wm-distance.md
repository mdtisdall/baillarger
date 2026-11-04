# Phase 1.1: the WM-distance sampler

Stream 1, phase 1 of `docs/version-2-plan.md` (§3d), and shared code (§7).
Read the plan's §1 (*Two kinds of published output*), §2 (*Domain edges*,
*Capped distances are local*) and §3c step 3. The rules in
`docs/plans/round-2.md` apply. The same session does phase 1.2 after this
phase.

## Goal

A sampler of the distance through tissue (WM and cortex) from the WM,
capped at 8 mm, with outside and the other labels as obstacles, for a
superblock and its halo. `orient` (phase 1.2) uses it for the sign of the
data normal.

## Done when

1. `docs/formats/wm-distance.md` gives the sampler's contract, version
   `wm-distance/1`: inputs, parameters, outputs, units, determinism.
2. The exact tests below pass, including: a voxel has the same value,
   bit for bit, whichever superblock computes it.
3. The PR body gives the analysis and the hemisphere check below.
4. `nix develop --command scripts/check` passes.

## Facts from the current code

### `streamlines.tissue_wm_distance` (`streamlines.py` 186–199)

- `tissue_wm_distance(wm, ctx, vox) -> F64`. It uses
  `skfmm.distance(phi, dx=vox, order=2)` (scikit-fmm, BSD licence;
  `pyproject.toml` 13), with `phi = MaskedArray(where(wm, -1, 1), mask=~(wm
  | ctx))`.
- The front starts at the WM/cortex interface. The masked voxels are the
  obstacles. The stencil is the 6-neighbour upwind update, second order,
  with anisotropic spacing.
- Output: signed distance in mm, negative in WM; NaN at obstacles and where
  tissue has no path to the WM.
- No cap and no halo. It works on a whole array. On a box, the box edge is
  a boundary, so values near the edge differ from the whole array.
- `skfmm.distance` has a `narrow` argument that stops the march at a given
  distance.
- Callers: `streamlines.tissue_wm_direction` (201–232) and
  `column_traces.py` (770, 825). Do not change them. Add the sampler as a
  new module.
- Test: only `tests/test_streamlines.py` 315–335 (via the direction).

### The sampler pattern

- `rays.RaySampler` (`rays.py` 125–281) is the pattern: the constructor
  reads the sidecar and wraps the labels in `ChunkedSource`; each call reads
  one box through the cache; the result for a column depends only on that
  column.
- `blocks.superblock_box(sb, m, cells, halo_mm, affine, shape)` (98–118)
  rounds the halo outward to whole voxels and is tested. Use it.
  `blocks.superblock_range` (81), `blocks.voxel_size` (90).
- `wm-dseg/1` has no `BlockCells`. Make the block size a parameter, with
  the default `mesh.BLOCK = 64`.
- The labels: `scratch/v2/derivatives/.../desc-wmtopo_dseg.nii.gz`.
  `FilledLabels` and the hemisphere are in its sidecar.

## Design

1. A new module, `src/baillarger/wm_distance.py`, with a class
   `WMDistanceSampler(dseg, cells=64, cache_bytes=...)` and
   `sample(superblock, level, halo_mm=0.0) -> WMDistance`.
2. **The distance:** through the free voxels (filled WM and cortex), from
   the WM/cortex interface, signed, negative in WM. Cortex is the cortex
   label of `label-set/1`. Outside and every other label are obstacles.
   Excisions are outside in the labels.
3. **The cap:** 8 mm (`CAP = 8.0`, the column length cap). Beyond the cap,
   the value is +inf (or −inf in WM), not NaN. NaN stays for obstacles and
   for tissue with no path. Write this in the spec.
4. **The read box:** the superblock with its halo, plus the cap, plus the
   stencil's reach (2 voxels for second order). A distance shorter than 8
   mm stays within 8 mm of its start (plan §2), so the values inside the
   superblock and halo are exact.
5. **Determinism:** test that the value of each voxel is the same, bit for
   bit, from two different superblocks. If skfmm does not give that (its
   heap order can depend on the box), say so to the coordinator, and give
   the largest difference. Then choose: a tolerance in the contract, or our
   own fast marching in numba with a fixed tie rule. Do not choose without
   the coordinator.
6. **Output:** the distance in mm (float32) on the superblock box with its
   halo, and the box (in voxels). No file is stored.
7. Consider `narrow` in skfmm for the cap. Measure whether it changes the
   values inside the cap.

## Exact tests

- No obstacle: the distance equals the straight-line distance to a plane
  WM boundary, within the second-order error (give the bound).
- A sulcus of outside between two banks: the path goes round it. Compare
  with a brute-force Dijkstra on a fine graph, within a stated bound.
- The cap: values above 8 mm are +inf.
- An obstacle label (not outside, not filled, not cortex) blocks.
- The same value, bit for bit, from two superblocks with the voxel in the
  first's interior and the second's halo.
- Anisotropic voxels.

## Analysis

In the PR body:

1. Cost per voxel and the read factor ((B + 2h)/B)³ with h = 8 mm + 2
   voxels, at each superblock level, at 160 µm and at 50 µm.
2. Fast marching is a heap: O(n log n) per box, on the CPU. Say whether
   this is a bottleneck at 50 µm whole brain. `orient` needs the sign only
   in cortex (at most about 4.5 mm from its WM), so it can ask for a
   smaller cap. Give the cost of a 5 mm cap.

## The hemisphere check

Read `scratch/v2/derivatives/`. Write to `scratch/phase-1.1/`.

1. Sample the superblocks of the V1V2, M1S1 and PCC_SPL sites (their
   centres are in the box prototypes; ask the coordinator if you cannot
   find them). Give the time and the peak per superblock.
2. A view of the distance on one axial slice of each site, with the labels.
3. The fraction of cortex voxels with no path to WM within the cap, by
   site.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | The spec draft | Phase session | 2 |
| 2 | The phantoms and the brute-force reference | Sonnet sub-agent | 1 |
| 3 | The sampler | Phase session | after 1 |
| 4 | The determinism test and the decision on skfmm | Phase session | after 3 |
| 5 | The analysis and the hemisphere check | Phase session | after 4 |

While this PR waits for review, start phase 1.2 (Parts A and B of its plan
do not need this sampler).

## PRs

1. `feature/wm-distance-sampler`. Branch from `origin/main`.

Tell the coordinator when the PR opens. Phase 1.2 stacks on it.

## Decisions for review (made by the coordinator)

- Values beyond the cap are ±inf, not NaN.
- The block size is a parameter (default 64), because `wm-dseg/1` has no
  `BlockCells`.
