# Phase 3.0: the column-volume sampler

Stream 3, phase 0 of `docs/version-2-plan.md` (§5d). The user added this
phase to round 1. The rules in `docs/plans/round-1.md` apply.

## Hold first

Do not write code until phase 2.0's ray-sampler spec PR
(`docs/ray-sampler-spec`) is open. The coordinator tells you when. Until
then:

1. Read this plan, `round-1.md`, and the plan sections in "Read".
2. Read `prototypes/scripts/column_volume.py` and `flatten_lib.py`, to learn
   what the old column volume held and what flattening read from it.
3. Make your worktree with the `start-task` skill. Do not commit.

Then stack every PR of this phase on the spec PR's branch.

## Read

`docs/version-2-plan.md`: §1 (samplers), §2 ("Domain edges"), §5b, §5c step 1
("the column volume"), §5c step 3 ("Extents by censoring"), §5d phase 0, §7.

## Goal

The column-volume sampler: for a superblock and its halo, each column's
echoes sampled along its ray between the extents that the caller gives, each
sample with its 3D position, and the partial-volume samples marked. It is a
sampler with a written contract. Nothing is stored.

## Done when

1. `docs/formats/column-volume.md` exists, with a version `column-volume/1`.
2. The sampler passes its exact tests.
3. A column gives the same samples, bit for bit, whichever superblock
   computes it.
4. The PR body gives the analysis in "Analysis".
5. After phase 2.0's sampler and phase 1.0's outputs exist, the PR body
   shows views of the column volume on the hemisphere (see "The hemisphere
   check").
6. `nix develop --command scripts/check` passes on each PR.

## Design

### Part A: the column-volume spec

Write `docs/formats/column-volume.md`, version `column-volume/1`. It states:

1. **Inputs:** the ray sampler's output for one call (`ray-sampler/1`), and
   the FLASH echoes on the grid of `wm-dseg/1`. No other data.
2. **Parameters:** the echoes to sample; the extents per column (e_in and
   e_out, within the ray sampler's caps); the step, which is the ray
   sampler's step (default 0.04 mm).
3. **Per column, keyed by edge key:** for each sample, the value of each
   echo (float32), the 3D position in scanner mm, the arc length s, and the
   partial-volume mark.
4. **Interpolation:** trilinear, in voxel coordinates from the scanner mm
   position through the inverse affine. State the rule at the grid edge.
5. **The partial-volume mark** (plan §5c, step 1): a sample is marked if it
   lies within 1 voxel of a label change along the ray. The label changes
   come from the ray sampler's label intervals. "1 voxel" along an oblique
   ray on an anisotropic grid needs a rule: state it (for example, the
   voxel's extent along the ray direction).
6. **The values are the echoes as acquired.** No log, no trend removal, no
   normalisation. Those are steps of a reader, not of the sampler.
7. **Determinism:** the same samples, bit for bit, whichever superblock
   computes the column, and on each device up to float32 rounding.

Send the spec to the coordinator before the PR. If the ray-sampler spec
lacks something that the column volume needs, tell the coordinator. Do not
change the ray-sampler spec yourself.

### Part B: the sampler

1. Write a new module, for example `src/baillarger/column_volume.py`. The
   name of the prototype `prototypes/scripts/column_volume.py` is the same.
   That is acceptable, because the prototype is not a package module.
2. Read the echoes through phase 2.0's chunked reader, for the box of the
   superblock and its halo. Read each chunk once for each call. Until the
   chunked-reader PR (`feature/chunked-reader`) is open, write the code
   against the `blocks.Source` protocol, and test it with `ArraySource`.
   Then merge that PR's branch.
3. The input is the ray sampler's output object. Your tests build that
   object directly from known rays, so they do not need the ray sampler's
   code. One integration test calls the real ray sampler when its PR is open.

Exact tests:

- A linear image: every sample is exact.
- A straight ray through a slab: the samples give the slab's profile.
- The partial-volume marks: on a ray with known label changes, the marked
  samples are the ones the rule gives.
- The extents: a longer extent gives the same samples at each s, plus more
  samples.
- A column computed in two superblocks (inside one, in the halo of the other)
  gives the same samples, bit for bit.
- numpy and torch agree to float32 rounding, if the sampler uses torch.
  Check the interpolation convention of `torch.nn.functional.grid_sample`
  (`align_corners`) against the numpy reference in a test.

### Analysis

Write this in the PR body (working rule "judge efficiency by analysis"):

1. The bytes read and the samples made for each column, at 160 µm and at
   50 µm. Cortex is 2–4.5 mm thick, so give the cost at typical extents and
   at the caps.
2. The working set of one superblock and its halo (three echoes, rays,
   samples).
3. Whether the step is bound by bandwidth or by compute, and whether it
   must run on the GPU.

### The hemisphere check

Do this after phase 2.0's sampler PR is open and phase 1.0's outputs exist.
The coordinator tells you when.

1. Sample one superblock that holds a marked site (V1V2 or M1/S1; the
   user's points are in `prototypes/`).
2. Show the column volume as seismic sections: echo 2 along a row of
   neighbouring columns, with the label intervals and the partial-volume
   marks drawn on it. Put the images in `scratch/v2/figures/` and attach them
   to the PR.
3. Give the time and the peak memory for that superblock.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 0 | Read and prepare (the hold) | Phase session | now |
| 1 | Part A: the spec | Phase session | after the ray-sampler spec PR |
| 2 | Part B: the sampler and its exact tests | Sonnet sub-agent writes the tests from the spec; the phase session writes the sampler | after 1 |
| 3 | The analysis | Phase session | after 2 |
| 4 | The hemisphere check | Phase session | after the 2.0 sampler and 1.0's outputs |

## PRs

1. `docs/column-volume-spec`: Part A. Stacked on `docs/ray-sampler-spec`.
2. `feature/column-volume`: Part B and the analysis. Stacked on PR 1. Merge
   `feature/chunked-reader` into it when that PR is open.
3. The hemisphere check goes in the body of PR 2, or in a later PR stacked on
   phase 2.0's `feature/ray-sampler`.

## Coordination

- Phase 2.0 owns the ray-sampler spec and the chunked reader. Send questions
  about them to the coordinator.
- Phase 3.1 (flattening) reads the column volume later. Write the spec so
  that flattening can read neighbouring columns: the halo columns must be
  in the output.
