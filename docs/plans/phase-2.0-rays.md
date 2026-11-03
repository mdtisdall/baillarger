# Phase 2.0: the ray sampler, with normal rays

Stream 2, phase 0 of `docs/version-2-plan.md` (§4d). The user added this
phase to round 1, and added the chunked reader to it (§7, `blocks`). The
rules in `docs/plans/round-1.md` apply.

## Why it can start now

The plan says that 2.0 needs phase 1.0. Only the run on the hemisphere needs
1.0's outputs. The spec, the reader and the sampler can be written and
tested on phantoms now. The `white-surf/1` and `wm-dseg/1` formats exist on
`main`, and the writers in `white_surf` and `white_dseg` (1.0's PR #91) can
make phantom inputs in those formats.

## Goal

1. A written spec for the ray sampler. Stream 3 and the viewer wait on it.
2. A chunked reader that never reads a large volume whole, with a chunk
   cache and superblocks in Morton order.
3. The ray sampler with the **normal** method, on phantoms, then on the
   hemisphere.
4. The local form of `mesh_graph`.

## Done when

1. `docs/formats/ray-sampler.md` exists, with a version `ray-sampler/1`.
2. The reader passes its exact tests, and the mesh outputs on the phantoms do
   not change, byte for byte.
3. The sampler passes its exact tests, including: a column is the same
   whichever superblock computes it.
4. The PR body of the sampler gives the analysis that picks the superblock
   level (see "Analysis").
5. The sampler runs on 1.0's hemisphere outputs, and the PR body gives the
   diagnostics in "The hemisphere run".
6. `nix develop --command scripts/check` passes on each PR.

## Facts from the current code

All paths are relative to the repository root.

### `src/baillarger/blocks.py`

- `morton(ijk)` (lines 41–52): the Z-order code, i before j before k.
- `Source` protocol (63–73): `shape`, `dtype`, `read(box)`.
- `ArraySource`, `EmbeddedSource`, `NiftiSource`, `open_volume` (the single
  entry point).
- `NiftiSource.read` on a `.gz` file reads the whole file one time and keeps
  it (`self._whole`, 120–123). On a `.nii` file it slices the memory map.
- `BlockReader(source, cells, before, after, dev, resident)` (169–238). On a
  torch device it keeps the whole volume resident if it is at most
  `RESIDENT_BYTES = 4 GiB`.
- Importers: `mesh.py` (`BlockReader`, `ArraySource`, `Source`),
  `white_surf.py` (`EmbeddedSource`, `open_volume`), `tests/test_blocks.py`,
  `tests/test_mesh.py`, and two prototypes.

### `src/baillarger/mesh.py`

- `BLOCK = 64`. `marching_tetrahedra_blocks` reads with
  `BlockReader(src, block, 0, 1, device)`.
- `MeshBlocks` gives, for each crossed block, `vertex_start/stop` (the owned
  vertices) and `face_start/stop`.
- `edge_key(low, step) = low × 7 + (4 s_i + 2 s_j + s_k − 1)`, and
  `edge_points` inverts it.
- A vertex belongs to the first block in Morton order that makes it. No
  helper maps a vertex to its owner block. The block index gives the owner
  ranges.

### `WhiteSurface.load` (now in `white_surf.py`)

- It can load the whole surface or one block. For one block it returns the
  vertices that the block's faces use, which includes vertices that earlier
  blocks own. It does not return the owned range.
- Fields: `v` (voxel coordinates), `f`, `normals`, `vertex_ids`,
  `vertex_key`, `affine`, `sidecar`. Phase 1.0 adds the normal length (|n̄|)
  in its third PR.

### The old column code

- `column_traces.py` (926 lines) works on boxes. It makes its own mesh with
  `white.white_mesh_normals`. It has no label intervals and no inward
  extension. Its rays stop at labels. `trace_columns` with "normal" calls
  `streamlines.ray`.
- `streamlines.ray(labels, dface, x0, d0, vox, nmax, step_mm=0.04, ...)`
  (765–804): constant steps outward, the label at the rounded position, with
  stop codes.
- `mesh_graph.py`: `edge_graph`, `geodesic_patches`, `patch_distances`,
  `crossing_vertex`. All four work on the whole mesh. The duplicate-edge fix
  is in a different session (`fix/mesh-graph-duplicate-edges`).
- Importers of `column_traces`: `pial_labels.py`, its tests, and eleven
  prototypes.

### Related work in other phases

- Phase 4.0 brings in #84's `viewer/bricks.py`. Its `BrickSource` protocol
  has `brick`, `dtype`, `levels` and `read(level, b)`. #84 says that a
  NIfTI-Zarr store must be able to take the place of `ArrayPyramid`.
- `docs/scaling.md` lines 32–48 and 162–170: 64³ blocks from voxel 0 in Morton
  order; NIfTI-Zarr comes with the blocked `dseg` (scaling item 3); "until
  then nothing reads a volume block by block".

## Old code

The user's decision: a module that the plan drops stays until the phase
that replaces its callers deletes it. In this phase:

- Do not change `column_traces.py`, `streamlines.py` or the old functions in
  `mesh_graph.py`. `pial_labels` and the prototypes use them.
- Put the sampler in a new module, for example `src/baillarger/rays.py`.

## Design

### Part A: the ray-sampler spec (write this first)

Write `docs/formats/ray-sampler.md`, version `ray-sampler/1`. It is a
sampler contract (plan §1): inputs, parameters, outputs and units. Add a
Python module with the output types and the function signature, so that
phase 3.0 can write code against it. The function can raise
`NotImplementedError` in this PR.

The spec must state:

1. **Data inputs:** `white-surf/1` (vertices, triangles, normals, normal
   lengths, edge keys, block index) and `wm-dseg/1` (`_desc-wmtopo_dseg`).
   No other data.
2. **Which columns a call returns.** A call names a superblock: an aligned
   cube of 2^m × 2^m × 2^m blocks of the 64³ grid, which is one contiguous
   range of the Morton order. It returns the columns whose roots are owned by
   the blocks of the superblock. A parameter can add the columns whose roots
   lie in a halo, given in mm. Stream 3 needs the halo columns as neighbours.
3. **Call parameters:** the method (`normal` in version 1; `equivolume` is
   added in phase 2.1 by an edit in place); the inward extent e_in ≤ 2 mm and
   the outward extent e_out ≤ 8 mm, as one value or one value per column;
   the step in mm (default 0.04).
4. **Per column, keyed by edge key:**
   - the root position (scanner mm), the unit normal and |n̄|;
   - the samples from s = −e_in to s = e_out at the step: the positions in
     scanner mm and the arc length s. s = 0 is the root. The root is a
     coordinate, not a depth;
   - the label intervals over the full caps (−2 mm to 8 mm), whatever the
     extents: for each interval, its start s, its end s and its class;
   - the derived class for diagnostics: open, in WM, or neither within
     8 mm (plan §4c, step 3).
   - A(s) and V(s) are reserved for phase 2.1. Name them in the spec as "not
     in version 1".
5. **The label classes of an interval.** Map `wm-dseg/1` labels to classes:
   filled WM (the sidecar's `FilledLabels`), cortex, outside (0), and
   other. State the map. Phase 1.5's `label-set/1` (PR #89) names the
   labels. Ask the coordinator if a label has no clear class.
6. **The rule for an interval boundary.** Use an exact traversal of the
   voxels that the ray crosses (Amanatides and Woo, 1987), with each voxel
   as a cube round its centre. Then the intervals do not depend on the
   step. Write the rule and its tie rule.
7. **Determinism:** the same column, bit for bit, whichever superblock
   computes it, and on each device up to float32 rounding.

Send the spec to the coordinator before the PR. Phase 3.0 stacks on this
PR, so open it as soon as the coordinator agrees.

### Part B: the chunked reader

Change `blocks.py`. Keep its public names where you can.

1. **No whole reads.** `NiftiSource` never reads a `.gz` file whole. A
   gzip stream cannot seek. So the reader reads an uncompressed `.nii`
   through its memory map. Add a function and a command that decompress a
   `.nii.gz` to a `.nii` one time, as a stream, without holding the volume
   in memory. Record the source SHA in a sidecar beside the `.nii`, and
   refuse a `.nii` whose SHA does not match.
2. **A chunk cache.** Reads go through 64³ chunks, aligned with the block
   grid, with a cache that holds a fixed number of bytes and drops the least
   recently used chunk. A read of a box gathers its chunks.
3. **Superblocks.** Add helpers: the Morton range of a superblock at level m;
   the box of a superblock with a halo in mm (convert mm to voxels on each
   axis with the voxel size; round outward).
4. **The resident path.** Remove the path that keeps a whole volume on the
   device, or make it read through the cache. The mesh outputs on the
   phantoms must not change, byte for byte. Phase 1.0 runs the mesh on the
   hemisphere from its own branch, so this change does not affect that run.
5. **NIfTI-Zarr is not part of this phase** (scaling item 3). Write the
   interface so that a NIfTI-Zarr store can take the place of the `.nii`
   memory map. Compare the interface with #84's `BrickSource`, and write in
   the PR body how a store can serve both. Do not change the viewer.

Exact tests: a box read through chunks equals the same box of the array; a
box that crosses chunk edges and the grid edge; the cache keeps its byte
limit; the superblock range is contiguous in Morton order; the halo box on an
anisotropic voxel; the decompress command gives the same array as nibabel.

### Part C: the local form of `mesh_graph`

Do this after the `mesh_graph` fix PR is open. The coordinator tells you
when. Stack this PR on that fix.

1. Add a function that builds the graph (mesh edges and across-triangle
   edges, in mm) for the roots of a superblock and its halo only, from the
   faces of the blocks that the halo box meets.
2. Exact test: the distances from a root, up to a limit L, are the same in
   the local graph and in the whole graph, if the halo is at least L. Test it
   on a phantom surface with several blocks.
3. Do not change the old whole-mesh functions.

### Part D: the sampler, with normal rays

1. Implement the function of Part A for the method `normal`: straight rays
   along the smoothed white normal, from s = −e_in to s = e_out.
2. Read the labels through the chunked reader, for the superblock box with
   a halo of 8 mm (the outward cap) plus the inward cap where it is larger in
   the other direction. Read only once for each call.
3. The label intervals come from the exact traversal (Part A, item 6), over
   the full caps.
4. Classes come from the intervals only (plan §4c, step 3). There is no bank
   split, no `SAME_MM`, no stop and no meeting point.

Exact tests, on phantom labels written as `wm-dseg/1` and surfaces written as
`white-surf/1` by the 1.0 writers:

- A flat slab: each ray's intervals are WM, then cortex, then outside, at the
  depths that the geometry gives.
- A ball: the rays are radial; the intervals follow from the radii.
- Two fused banks (cortex to cortex with no fluid): the ray runs from its
  WM, through cortex, into the far bank's WM; the class is "in WM".
- An open slot: the class is "open".
- The inward extension: in a thin WM blade, the inward ray enters the far
  cortex at the depth that the geometry gives.
- The extents: a ray with a longer e_out has the same samples at each s as a
  ray with a shorter e_out. The intervals do not change with the extents or
  the step.
- A column is the same, bit for bit, when two different superblocks compute
  it (one has the root inside, the other in its halo).
- numpy and torch give the same samples to float32 rounding, if the
  sampler uses torch.

### Analysis

Write this in the PR body of the sampler (plan §2, "Domain edges"; working
rule "judge efficiency by analysis"):

1. The cost per column: samples, voxels traversed, bytes read.
2. The superblock level m: the working set of one superblock and its halo
   (labels, surface, outputs) at 160 µm on the laptop and at 50 µm on a GPU
   with 24–80 GB. The read factor ((B + 2h)/B)³ for h = 8 mm at each
   candidate level. Pick the level for each case and give the reason.
3. Whether the sampler must run on the GPU: give the arithmetic intensity
   and say whether the step is bound by bandwidth or by compute.

### The hemisphere run

Do this after phase 1.0 writes its outputs in `scratch/v2/derivatives/`.
The coordinator tells you when. A full pass over the hemisphere (12.7 M
columns) can be heavy. Measure one superblock first. Then ask the
coordinator for a heavy-run slot if the full pass needs more than 8 GB.

Diagnostics for the PR body (views, not pass/fail numbers):

1. The counts of each derived class, and their map on the overview mesh.
2. The classes against |n̄| (plan §4d, phase 2 asks for this later; give a
   first table now).
3. The time and the peak memory for one superblock and for the full pass.
4. The fraction of roots where the ray goes from WM straight into outside
   with no cortex between. These are cut faces (plan §3c, step 2).

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Part A: the spec and the output types | Phase session | 2 |
| 2 | Part B: the chunked reader and its tests | Sonnet sub-agent; the phase session reviews the interface | 1 |
| 3 | Part C: the local graph | Sonnet sub-agent | after the `mesh_graph` fix PR |
| 4 | Phantom fixtures in the `wm-dseg/1` and `white-surf/1` formats | Sonnet sub-agent | after 1 |
| 5 | Part D: the sampler | Phase session | after 1, 2 and 4 |
| 6 | The analysis | Phase session | after 5 |
| 7 | The hemisphere run | Phase session | after 5 and 1.0's run |

## PRs

1. `docs/ray-sampler-spec`: Part A. Branch from `origin/main`. If the
   phantom writers need 1.0's PR #91, stack on `refactor/white-split`.
2. `feature/chunked-reader`: Part B. Branch from `origin/main`.
3. `feature/mesh-graph-local`: Part C. Stacked on the `mesh_graph` fix.
4. `feature/ray-sampler`: Part D, the analysis and the hemisphere report.
   Stacked on PRs 1, 2 and 3 (merge their branches).

Tell the coordinator when each PR opens. Phase 3.0 stacks on PR 1, and
merges PR 2 when it needs the reader.

## Coordination

- Phase 3.0 starts its code when your spec PR is open. Answer its questions
  about the spec through the coordinator.
- A change to the spec after PR 1 is open goes through the coordinator,
  because phase 3.0 builds on it.
- Ask before you change `white_surf`, `white_dseg` or `mesh.py`. They belong
  to phase 1.0 in this round.
