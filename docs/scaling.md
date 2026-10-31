# Scaling: from a hemisphere to whole brains

The pipeline moves from boxes to whole hemispheres one stage at a time; each
stage is scaled when we get to it, and only the computational problems met
are fixed. Scaled so far: the WM topology correction, the white surface and
its overview for viewing (`python -m baillarger.white dseg` and `mesh`;
`docs/formats/wm-dseg.md`, `docs/formats/white-surf.md`,
`docs/formats/white-overview.md`).

The target beyond a hemisphere at 160 µm is whole brains at 50 µm: about 65
times the voxels (about 2e10) and on the order of 350 M surface vertices. That
runs on a larger machine with CUDA GPUs; a laptop (64 GB, Apple GPU) is where
the scalable code is built and tested.

## Rules for a scaled stage

- Cost at the large scale decides. Each step is judged by its complexity in
  the voxel size h (volume work grows as h^-3, work on a band round the
  surface as h^-2) and by a realistic bound on its hardware (memory bandwidth
  for stencil work), not by its time on the laptop. The laptop's timings and
  its CPU-to-GPU ratio do not predict a dedicated GPU.
- A GPU implementation wherever it should make a difference at that scale.
  numpy (or numba) is the reference; PyTorch is the GPU backend (CUDA on large
  machines, MPS on the laptop), chosen at run time (`baillarger.backend`).
  Kernels use only operations MPS supports (no float64, no CSR) and no atomic
  additions, so a device gives the same result every run.
- Results are continuous functions of their input: no thresholds that
  rounding can flip, so devices agree to rounding (or, for a solver, to its
  tolerance), and the sidecar records the `Device`. Tests compare each device
  with the reference: integer and boolean results exactly, floating-point
  results within a tolerance relative to the coordinates' size.
- Blocks with a halo: no whole-volume float arrays in the long run. Steps work
  on blocks that fit in memory (and GPU memory) and join exactly. The block
  grid starts at voxel (0, 0, 0) with `BlockCells` (64) cells a side; blocks
  are in Morton (Z) order, so any aligned cube of 2^m blocks is one
  contiguous range (a step with a large halo works on aligned groups of
  blocks, one read each). Halos differ by use: 1 voxel for marching, several
  for intensity filters.
- Global algorithms get a local form when their step is scaled: connected
  components in blocks with a merge; the Euler characteristic as a sum over
  blocks; handle removal as a global search for the handles, then local fixes
  in a box round each.
- One reader for every volume on the grid (`baillarger.blocks`: `open_volume`
  and `BlockReader`), so storage can change under the stages. Now NIfTI; at
  whole-brain scale a conversion step at the start of the pipeline makes
  NIfTI-Zarr stores (NIfTI's header with chunked, compressed arrays and a
  multi-resolution pyramid; chunks of `BlockCells` a side, sharded in Morton
  order; a draft standard, not yet in BIDS, so the conversion keeps the BIDS
  names and sidecars).
- A stable identity for each surface vertex: its grid-edge key, stored with
  the mesh. Vertex numbers change with block order when a local edit is
  remeshed; anything stored by vertex (column roots, cached layers, edits)
  keys on the edge key.
- Each output has a spec in `docs/formats/`, enforced by a
  `bids.SidecarSpec`; until a first release the specs stay at version 1 and
  change in place.

## Precision

Mesh steps run in float32, as on a GPU: marching does its arithmetic in the
dtype asked for, `mesh_smoothing` keeps the vertices' dtype, and surface
readers return float32. Float64 stays where it buys exactness or a reference:
the affine to mm, area sums, `mesh_intersect`'s pair tests. Connectivity-only
sparse graphs use int8. Marching's triangles and orientation follow from the
corners' signs alone (no triangle dropped for zero area), so they are the same
in float32 and float64 on any device.

## GPU roadmap

In order of cost at the large scale:

1. The antialiasing solver (cost model below): multigrid (iterations
   independent of resolution), on the band only in a block-major layout, with
   a fused stencil kernel. Projected quasi-Newton (fewer gradients than FISTA
   in tests, but still growing with resolution) is the simpler step if
   multigrid waits.
2. The neighbour and face tables of the normal smoothing (sorts over billions
   of entries at that scale): numpy now.
3. Connected components of the mesh (the per-piece summary).
4. Connected components of the volume (Betti numbers, cavity fill), in blocks
   with a merge across them.

Not on a GPU: handle removal (a priority queue, sequential; at whole-brain
scale independent handles run in parallel on CPU cores, its first need, since
in one piece it would take about 43 GB), relabelling cuts (small boxes), the
overview's decimation (`baillarger.decimate`: priority queues, compiled, in
parallel by blocks on CPU cores), and reading, writing and hashing files. The self-intersection check
(`mesh_intersect`, numba on the CPU) is a test, not a step of a command.

## The antialiasing solver's cost

`baillarger.antialias` solves for the field whose level set is the white
surface, by FISTA with adaptive restart. With h the voxel size:

- Per iteration it is memory-bound. The separate stencils move about 300 to
  350 bytes per voxel; a fused kernel reading phi once with its 5 × 5 × 5
  neighbourhood from cache would move about 16 to 20, which on a data-centre
  GPU (2 to 3 TB/s) is about 0.01 ns per voxel per iteration.
- Iterations: FISTA's grow with the square root of the condition number, for
  this energy about (L/h)^2 (L the length the band couples over): about 10
  times more at 50 µm than at 160 µm. Projected quasi-Newton needs fewer but
  still more with resolution; multigrid's do not grow.
- Voxels: the whole volume grows as h^-3, the band as h^-2.

So the whole volume with FISTA costs about h^-5, impractical for a whole brain
at 50 µm; the band with multigrid and a fused kernel costs about h^-2 per
cycle, for a fixed number of cycles: the target.

What the prototypes found (`prototypes/scripts/antialias*.py`):

- stencils as shifted slices, not conv3d (on MPS conv3d was 120 times
  slower); a fused 5 × 5 × 5 stencil by strided adds was slower on MPS (a real
  fused kernel is a different thing: it reads from cache once);
- adaptive restart: 3 to 3.5 times fewer iterations; a warm start (blurred
  signed distance) did not help; nor did solving a separate coarse problem
  first (its looser constraints move its surface up to a voxel), which is why
  multigrid must pass the fine problem's error to the coarse grids;
- the stopping test is over a window, since a single step just after a
  restart can be tiny far from the optimum;
- convergence is global, not local: blocks below tolerance stay at 28 of 515
  until iteration 1,300, then nearly all converge by 1,600; retiring
  converged blocks would save under 10 %;
- the band only, gathered by index in 64³ blocks, gives the same answer; its
  speed against the whole volume is not yet measured fairly.

## Measuring

The time per iteration of a solver is constant through a solve, so it is
measured from about 100 iterations, and the iteration count separately; the
antialiasing solver logs both every 100 iterations. Timings are taken on AC
power, with sleep prevented (`caffeinate -i`), one heavy run at a time (under
`scripts/memguard`), nothing else on the GPU. Laptop numbers describe the
laptop only.

The example right hemisphere (grid 700 × 384 × 1148 at 0.16 mm, WM box
632 × 323 × 1006):

| Step | Result | Peak memory |
|---|---|---|
| `dseg` | Betti (1, 111, 182) → (1, 0, 0); 6,307 cavity voxels, 34 fills (570 voxels), 52 cuts (589 voxels); 3.4 min | 21.6 GB |
| `mesh` | one closed genus-0 piece, 72,097 mm² (the binary staircase: 91,972); 12.7 M vertices, 25.3 M faces in 506 blocks; no self-intersections; 1,651 antialiasing iterations, 0.57 s each on the laptop's GPU (217 M voxels, about 160 GB/s); files about 0.7 GB | 20 to 24 GB |
| `mesh`, CPU against MPS | the same triangles and keys, vertex positions bit for bit, normals within 3.5e-7 | |

At 50 µm the whole brain's surface files would be about 20 GB.

## Before a whole brain

No stage runs on a whole brain at 50 µm until these are done, in this order.
The hemisphere `dseg` peaks at about 22 GB; the same whole-array `dseg` on a
whole brain would be on the order of a terabyte.

1. The midline cut. The corpus callosum joins the two hemispheres' filled
   masks, so a whole brain needs a cut (FreeSurfer's cutting planes) before
   `dseg`.
2. `dseg` in blocks. Connected components in blocks with a merge; the Betti
   numbers from the Euler characteristic as a sum over blocks; the cavity fill
   from the blocked components. Handle removal
   (`baillarger.handles.remove_handles`) is the hard part: it grows fronts
   through one global priority queue (in one piece, about 43 GB). Its blocked
   form, "find the handles globally, fix each in a local box", needs a design
   and a prototype first.
3. NIfTI-Zarr volumes, together with item 2 and not before. Until then
   nothing reads a volume block by block; the standard is still a draft
   (nifti-zarr-py 1.0.0rc6); and ITK-SNAP and Freeview cannot open it, so an
   export to `.nii.gz` for hemisphere-size volumes comes with it. The layout
   is in "Rules for a scaled stage" (chunks of `BlockCells` a side, Morton
   order, read through `blocks.open_volume`). The `dseg` writes its store
   directly, with no conversion; the aseg and the FLASH echoes get a
   conversion step.
4. The faster antialiasing solver (GPU roadmap, item 1).

The overview's decimation (under Open) is not on this list: it is not the
first limit.

## Open

- Normals are ill-conditioned where a thin blade's two sides partly cancel in
  the normal smoothing (the averaged normal keeps as little as 0.56 of its
  length, against 0.9998 typically): there a rounding difference can turn the
  normal far. Left for the column stage, whose columns start along them.
- The overview's decimation before whole brains. On the hemisphere (25.3 M
  faces, 4 workers) it peaks at 6.6 GB in the main process and 10.9 GB with
  its workers; what stays whole is the mesh's positions, faces and quadrics
  and its block and lock keys, which grow with the mesh. Two steps:
  1. Int32 for faces and vertex ids, which fit up to about 2.1e9 vertices (a
     whole brain has about 3.5e8); keys and products of indices (edge keys,
     the 1-D edge keys of `boundary_vertices`, block keys) and positions in
     arrays longer than 2^31 stay int64, cast up before they combine, since
     numpy wraps around without a warning. A test builds keys from int32
     indices on a grid past 3e9 points. About half the memory of the largest
     arrays; the same results.
  2. The first pass read block by block from the mesh's files. Its blocks are
     the file's blocks: each worker memory-maps its block's faces, the
     vertices they name and their edge keys, and locks the vertices whose
     edge's cells reach outside the block (from the edge key and `BlockCells`
     alone: a vertex's faces come only from cells that hold its edge), so
     nothing global is needed. The main process sums the locked vertices'
     quadrics from the faces in chunks (the same order of additions, the
     same bits) and runs the second pass in memory on what is left (about 4 %
     of the faces), on the grid shifted by half a block in voxels. The
     overview's `BlockMM` becomes the block size in cells. The main process
     then holds the reduced mesh only (for a whole brain about 3e7 faces and
     one carried id per true vertex: a few GB).
  Neither is the first limit at whole-brain scale: handle removal and the
  whole-volume arrays of `dseg` and `mesh` are.
- The overview's self-intersections are repaired (`baillarger.decimate`:
  rounds that pin what crossing faces stand for, rerunning only the blocks
  that change). If near-contacts of the white surface are far more common at
  50 µm, simplification envelopes (Cohen, Varshney, Manocha et al. 1996:
  offset surfaces that narrow where the surface comes close to itself, which
  the result must stay between) would prevent them instead, at the cost of
  building non-overlapping envelopes in blocks.
