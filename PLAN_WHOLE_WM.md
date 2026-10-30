# Plan: the white mesh of a whole hemisphere

Planning file for the first stage of the move from boxes to whole hemispheres:
the white (WM) mesh. Later stages stay on boxes until their turn; each is
scaled when we get to it, and only the computational problems that we meet
are fixed. When this stage is done, the lasting content goes into the module
docstrings and the format specs (`docs/formats/`), and this file is deleted.

## Scaling principles

The target beyond this hemisphere is whole brains at 50 µm: about 65 times
the voxels (about 2e10) and on the order of 350 M vertices. That runs on a
larger machine with CUDA GPUs; the laptop (64 GB, Apple GPU) is where the
scalable code is built and tested. Each step that this work touches follows
these rules; steps not yet touched keep their current form.

- Cost at the large scale decides: each step is judged by its complexity in
  the voxel size h (volume work grows as h^-3, work on a band round the
  surface as h^-2) and by a realistic bound on its hardware (memory
  bandwidth for stencil work), not by its time on the laptop. The laptop's
  timings and its CPU-to-GPU ratio do not predict a dedicated GPU.
- A GPU implementation wherever it should make a difference at that scale.
  numpy (or numba) is the reference; PyTorch is the GPU backend (CUDA on
  large machines, MPS on the laptop), chosen at run time
  (`baillarger.backend`: `--device`, `$BAILLARGER_DEVICE`, default cuda,
  then mps, then cpu), using only operations MPS supports (no float64, no
  CSR) and no atomic additions, so a device gives the same result every
  run. Tests compare each device with the reference: integer and boolean
  results exactly, floating-point results within a tolerance stated
  relative to the coordinates' size.
- Results are continuous functions of their input: no thresholds that
  rounding can flip, so devices agree to rounding (or, for a solver, to its
  tolerance), and the sidecar records the `Device`.
- Blocks with a halo: no whole-volume float arrays in the long run. Steps
  work on blocks that fit in memory (and in GPU memory) and join exactly.
  The block grid starts at voxel (0, 0, 0) with `BlockCells` (64) cells a
  side; blocks are in Morton (Z) order, so any aligned cube of 2^m blocks
  is one contiguous range (a step with a large halo works on aligned groups
  of blocks, one read each).
- Global algorithms get a local form when their step is scaled: connected
  components in blocks with a merge; the Euler characteristic as a sum over
  blocks; handle removal as a global search for the handles, then local
  fixes in a box round each.
- One reader for every volume on the grid (segmentations and FLASH echoes
  alike): `baillarger.blocks.open_volume` (a box of voxels) and
  `BlockReader` (block (i, j, k) with a halo, as an array on a device).
  Stages and the viewer read volumes only through them, so storage can
  change under them: now NIfTI (memory-mapped `.nii`, or a `.nii.gz` read
  once); at whole-brain scale a conversion step at the start of the
  pipeline makes NIfTI-Zarr stores (NIfTI's header with chunked, compressed
  arrays and a multi-resolution pyramid; chunks of `BlockCells` a side,
  sharded in Morton order; a draft standard, not yet in BIDS, so the
  conversion keeps the BIDS names and sidecars), read chunk by chunk.
  Halos differ by use: 1 voxel for marching, several for intensity filters.
- A stable identity for each vertex: its grid-edge key, stored with the
  mesh (`desc-edgekeys`). Vertex numbers change with block order when a
  local edit is remeshed; anything stored by vertex (column roots, cached
  layers, edits) keys on the edge key.
- Manual edits enter the segmentation as a second `dseg` source (a sparse
  edits file), recorded as `manual` moves; noted in `wm-dseg.md`, built
  when the editor needs it.

GPU roadmap, in order of cost at the large scale:
1. The antialiasing solver (its cost model is below): multigrid
   (iterations independent of resolution), working on the band only in a
   block-major layout, and a fused stencil kernel. Projected quasi-Newton
   (fewer gradients than FISTA in tests, but still growing with resolution)
   is the simpler step if multigrid waits.
2. The neighbour and face tables of the normal smoothing (sorts over
   billions of entries at that scale): numpy now.
3. Connected components of the mesh (the per-piece summary).
4. Connected components of the volume (Betti numbers, cavity fill), in
   blocks with a merge across them.
Not on a GPU: handle removal (a priority queue, sequential; at whole-brain
scale independent handles run in parallel on CPU cores, its first need,
since in one piece it would take about 43 GB), relabelling cuts (small
boxes), and reading, writing and hashing files. The self-intersection check
(`mesh_intersect`, numba on the CPU) is a test, not a step of the command.

## The two commands

```
python -m baillarger.white dseg ASEG --out DERIV
python -m baillarger.white mesh DSEG --out DERIV [--block-cells B] [--device D]
```

1. `dseg` reads the aseg and writes a corrected segmentation: the aseg with
   the filled WM mask made handle-free and cavity-free. The aseg file itself
   is never changed.
2. `mesh` reads that segmentation and writes the white mesh of the filled WM
   mask, as GIfTI.

Every later stage reads the corrected segmentation, not the aseg plus a list
of changes, so the mesh and the labels cannot disagree. Neither command takes
a box: they work on the whole grid. Both log the time and peak memory of each
step (`baillarger.timing`).

## BIDS conventions

- Inputs are paths to BIDS-named files. The `sub` and `ses` entities are read
  from the input's name and carried to the outputs. The aseg's name
  (`sub-INDD121850_ses-7THemix20250516_aseg.nii.gz`) has a nonstandard
  suffix; any NIfTI with `sub-` and `ses-` entities is accepted.
- `--out` is a BIDS derivatives root. Files go to
  `DERIV/sub-<sub>/ses-<ses>/anat/`. The command writes
  `DERIV/dataset_description.json` (`DatasetType: derivative`, `GeneratedBy`:
  baillarger, its version and commit) if it is not there.
- Every output has a JSON sidecar with the BIDS keys `Description` and
  `Sources` (input paths), and our own keys, all listed in the spec:
  `SourcesSHA256`, `Provenance` (created, function, version, commit, dirty)
  and `SpecVersion`.
- BIDS has no settled rule for surface derivatives (the structural
  derivatives extension is a draft). Where it says nothing, the spec records
  our choice as BIDS-like.

## Format specs

Each output has a written spec in `docs/formats/` (`wm-dseg.md`,
`white-surf.md`): every file, array and sidecar key, with type, units,
required or not, and meaning. One writer and one reader per format enforce it
through a `bids.SidecarSpec`: the writer and the reader refuse a missing key,
a key not in the spec, a value of the wrong type or another `SpecVersion`.
`tests/test_bids.py` checks that each spec document's key table matches its
`SidecarSpec`. A field changes only through a PR that edits the spec. Until a
first release, the specs stay at version 1 and change in place (no users, so
no compatibility to keep); version bumps start with that release.

## Step 1: `dseg`

Labels (FreeSurfer LUT). The filled WM mask of the right hemisphere is
41 (WM), 43 (lateral ventricle), 49–52 (thalamus, caudate, putamen,
pallidum) and 192 (corpus callosum); of the left, 2, 4, 10–13 and 192. The
hemisphere is the one whose WM label is in the aseg. The hippocampus (53 /
17) is not filled: it is archicortex, whose laminar axis does not run out
from the WM below it.

1. Read the aseg. Mask = its voxels with a filled label, on the mask's
   bounding box plus 2 voxels.
2. Fill the cavities: background components (Kuhn adjacency, as
   `handles.kuhn_betti` counts them) that do not reach the box's border.
3. Remove the handles (`handles.remove_handles`), which cuts or fills whole
   voxels with the Kuhn adjacency that marching tetrahedra matches, so the
   mesh of the result is genus 0.
4. The corrected segmentation: the aseg, with
   - cavity voxels and handle fills set to the WM label;
   - each handle cut given the label of the nearest voxel that is outside the
     mask and not a cut (Euclidean distance transform with indices, on the
     cuts' bounding box plus a margin).

   Check: its voxels with a filled label are exactly the mask of step 3.
5. Write:
   - `sub-<sub>_ses-<ses>_desc-wmtopo_dseg.nii.gz`: the corrected
     segmentation (int16, the aseg's grid and header), with `_dseg.tsv` (the
     label names present, from the FreeSurfer LUT) and `_dseg.json`:
     the filled labels, hemisphere, the Betti numbers (b0, b1, b2) before and
     after, voxels set to WM by cavity fill and by handle fill, voxels cut and
     their new labels, and per move its kind, voxels and centre.
   - `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.nii.gz` (diagnostic, uint8):
     1 cut, 2 handle fill, 3 cavity fill; with `.tsv` and `.json`.

## Step 2: `mesh`

1. Read the corrected segmentation; mask = its voxels with a filled label
   (the label set from its sidecar). Check `kuhn_betti(mask)` has b1 = 0 and
   b2 = 0, and refuse otherwise.
2. The antialiased field of the mask (below), on the mask's box plus a
   margin, on the device.
3. Marching tetrahedra of the field at 0, in Morton-ordered blocks on the
   whole grid (the field embedded in it, `blocks.EmbeddedSource`), faces
   oriented out of the WM: the binary mask's triangles, topology and edge
   keys, each vertex moved along its own grid edge.
4. Normals: area-weighted vertex normals smoothed along the mesh (10 steps).
5. Write the mesh as GIfTI with its data outside the XML
   (`Encoding="ExternalFileBinary"`): one copy of the mesh, in a standard
   format, readable by block.

| File (in `sub-<sub>/ses-<ses>/anat/`) | Content |
|---|---|
| `sub-<sub>_ses-<ses>_hemi-<R\|L>_white.surf.gii` and `.dat` | Vertices (float32, scanner RAS mm) and triangles (int32), external binary data; metadata `AnatomicalStructurePrimary`, `AnatomicalStructureSecondary`, `GeometricType` |
| `…_desc-normals_white.func.gii` and `.dat` | The unit normals (float32, out of the WM), in a file of their own so that the `.surf.gii` holds only what every surface reader expects |
| `…_desc-edgekeys_white.func.gii` and `.dat` | Each vertex's grid-edge key (int64 as two int32 words): lin(lower end) × 7 + the edge's direction, about 7 times the voxel count at most, so int64 holds it for any grid |
| `…_desc-blocks_white.tsv` | The block index: per block, its index (i, j, k), its vertex and face ranges, and its vertices' bounding box (as a reader gets them from the file, rounded outward) |
| `…_white.json` | Sidecar: `Sources` and their SHA-256, the SHA-256 of each `.dat`, the filled labels, `BlockCells`, `GridShape`, the voxel-to-RAS affine, `Antialias` (how the field was solved), `Device`, and the summary (vertices, faces, per piece its genus, boundary loops and area) |

Order: blocks in Morton order; the faces block by block; the vertices by the
first block that makes them, then by edge key. A block's faces use its own
vertices and some boundary vertices of earlier neighbouring blocks.
`ExternalFileName` is the `.dat`'s name only, so the files move together.

Readers (checked in their source): nibabel (memory-mapped; nilearn through
it), Connectome Workbench, gifticlib and so FreeSurfer and AFNI/SUMA (run
them in the files' directory: gifticlib opens a relative `ExternalFileName`
from the working directory), the MATLAB `gifti` toolbox and R's `gifti`.
They read the arrays whole; reading by block is ours: `WhiteSurface.load`
(path, block=None) memory-maps the `.dat` files and returns the whole mesh or
one block (faces re-indexed, with the boundary vertices neighbouring blocks
own), vertices in voxel coordinates, normals and edge keys; the viewer reads
blocks through it. NiiVue does not read external data. nibabel cannot write
this encoding, so `baillarger.gifti` writes the XML and the tests read it
back with nibabel; a feature PR to nibabel could add the writer.

Size: uncompressed, about 0.7 GB for this hemisphere (12.7 M vertices,
25.3 M faces, with normals and keys), about 20 GB for a whole brain at 50 µm.

No per-vertex cortex label: every vertex is a column root. Columns run out
from the WM until they reach background (label 0) or a maximum length, and
what they pass through (cortex, hippocampus, water mislabelled as cortex) is
left to later stages.

## Antialiasing

The binary mask fixes only which voxel centres are inside; its surface is a
staircase. The field phi keeps every voxel's sign (phi >= 0.02 on the mask,
<= -0.02 off it) and minimises the thin-plate energy (its squared second
derivatives) in a band of 5 voxels of signed distance round the boundary;
outside the band it is the signed distance (Lempitsky, CVPR 2010; Whitaker,
2000; `baillarger.antialias`). Marching tetrahedra of phi then gives:
- the binary mask's triangles, orientation, topology and edge keys (they
  follow from the signs alone);
- each vertex on its own grid edge between an inside and an outside voxel
  centre, where the true boundary also crosses: the staircase goes without
  the surface moving past the data;
- no self-intersections, by construction (one piecewise-linear level set;
  sheets keep about 2 x 0.02 voxel apart, far above float32 rounding);
- one solution of a convex problem, so devices agree to the tolerance (on
  the hemisphere the CPU and MPS meshes are identical bit for bit).

Smoothing the mesh instead (relaxation and Gaussian smoothing) can push
close sheets through each other, and every repair tried either rested on a
threshold rounding could flip (devices disagreed by up to half a voxel) or
held back much of the smoothing; the field avoids both.

Quality. Toys: a tilted plane's normals within 0.47 deg (median; the raw
staircase 31 deg) and points within 0.017 voxel; a sphere's normals within
0.41 deg from the field's gradient. Hemisphere: area 72,097 mm^2 (raw
91,972); neighbouring normals after 10 steps of normal smoothing 1.3 / 3.9 /
11.4 deg (median / 90th / 99th percentile), as smooth as Gaussian mesh
smoothing gave; the field's gradient gives worse normals in the tail (17.6
deg at the 99th percentile, at thin blades), so the mesh's are used.

The solver: accelerated projected gradient (FISTA) with adaptive restart
(O'Donoghue and Candes 2015), in PyTorch on any device. The six stencils are
shifted slices of a padded copy, each applied, then applied again as its own
transpose. It stops when the largest mean change per iteration over a window
of 25 iterations, at the voxels next to the boundary (they set the
vertices), is under 3e-6: every vertex within about 1e-3 voxel of the fully
converged field. At a tight tolerance the field matches SciPy's L-BFGS-B and
CVXPY with Clarabel (an interior-point method) to 1e-5 (tests).

Cost model (h the voxel size):
- Per iteration the solver is memory-bound. The separate stencils move
  about 300 to 350 bytes per voxel; a fused kernel reading phi once with its
  5 x 5 x 5 neighbourhood from cache would move about 16 to 20, which on a
  data-centre GPU (2 to 3 TB/s) is about 0.01 ns per voxel per iteration.
- Iterations: FISTA's grow with the square root of the condition number,
  for this energy about (L/h)^2 (L the length the band couples over): about
  10 times more at 50 µm than at 160 µm. Projected quasi-Newton needs fewer
  but still more with resolution; multigrid's do not grow.
- Voxels: the whole volume grows as h^-3, the band as h^-2.
So the whole volume with FISTA costs about h^-5, impractical for a whole
brain at 50 µm; the band with multigrid and a fused kernel costs about h^-2
per cycle, a fixed number of cycles: the target (roadmap item 1).

Findings that shaped it (prototypes/scripts/antialias*.py):
- stencils as shifted slices, not conv3d (on MPS conv3d was 120 times
  slower); a fused 5 x 5 x 5 stencil by strided adds was slower on MPS (a
  real fused kernel is a different thing: it reads from cache once);
- adaptive restart: 3 to 3.5 times fewer iterations; a warm start (blurred
  signed distance) did not help; nor did solving a separate coarse problem
  first (its looser constraints move its surface up to a voxel), which is
  why multigrid must pass the fine problem's error to the coarse grids;
- the stopping test is over a window, since a single step just after a
  restart can be tiny far from the optimum;
- convergence is global, not local: blocks below tolerance stay at 28 of
  515 until iteration 1,300, then nearly all converge by 1,600; retiring
  converged blocks would save under 10 %;
- the band only, gathered by index in 64^3 blocks: the same answer; its
  speed against the whole volume is not yet measured fairly (see Timings).

## Blocked marching tetrahedra

- Cells are taken in blocks; a block whose grid points are all on one side
  of the level is skipped. In one piece, `marching_tetrahedra` kept about
  6.5 kB of arrays per surface cell (about 40 GB for the hemisphere).
- A vertex is keyed by its grid edge and computed from the edge's lower end,
  so every block gives the same bits for a shared vertex; the blocks join
  by one `np.unique` over the keys, and the surface does not depend on the
  block size (tests: blocks of 1, 3, 7, 16 cells give the one-shot mesh).
- Triangles and their orientation follow from the corners' signs alone (a
  table, `_FLIP`), and no triangle is dropped for zero area, so they are the
  same in float32 and float64, on any device: in float32 a field with grid
  points exactly on the level had come apart when orientation and dropping
  depended on geometry.
- It reads its volume through a `Source` (an array, a file, or a field
  embedded in the whole grid), so a grid larger than memory works (test: a
  4.3e9-point grid).

## Precision and devices

The mesh steps run in float32, as on a GPU (MPS has no float64): marching
does its arithmetic in the dtype asked for, `mesh_smoothing` keeps the
vertices' dtype, and `WhiteSurface.load` returns float32. Float64 stays
where it buys exactness or a reference: the affine to mm, area sums,
`mesh_intersect`'s pair tests. Connectivity-only sparse graphs use int8.
Against the float64 reference: the same triangles, vertices within float32
rounding of the coordinates (a tolerance relative to their size, since
rounding grows with it).

Normals are ill-conditioned where a thin blade's two sides partly cancel in
the normal smoothing (the averaged normal keeps as little as 0.56 of its
length, against 0.9998 typically): there a rounding difference can turn the
normal far. Left for the column stage, whose columns start along them.

## What goes (a separate PR, after this one)

- From `baillarger.white`: `WMChanges`, `wm_changes` and the `changes`
  subcommand (replaced by `dseg`); `white_matter_box`; the `box` subcommand,
  `box_report`, `write_box` and `overlay`; `box_mesh`, the box fields of
  `WhiteMesh` and its `.npz`.
- The box stages that took the WM changes (`column_traces.load_tissue`,
  `changes=`; `pial_labels`, `--wm-changes`) read their box's labels from the
  corrected segmentation instead, as they read the aseg now. Their `mesh=` /
  `--mesh` option goes: they make their own box's mesh until their stage is
  scaled.
- `prototypes/scripts/wm_handles*.py` stay as records and are not updated.

## Tests

- Marching: blocks give the one-shot surface exactly (every block size,
  float32 and float64, fields with grid points on the level); the block
  index; Morton order; the orientation table for any values; edge keys name
  their edge, do not depend on the block size, survive a local edit, and
  work on a grid past 3e9 points.
- `dseg` on a synthetic aseg (a tunnel, an enclosed ventricle, a cavity, a
  rod): Betti numbers after (b0, 0, 0), the filled-label voxels equal the
  corrected mask, cuts relabelled by the nearest rule, other voxels
  unchanged.
- BIDS names and layout; sidecars refuse a missing or unknown key, a wrong
  type or another `SpecVersion`; each spec document lists its code's keys.
- `mesh` and its reader: the surface and sidecar, nibabel reads the external
  data, blocks cover the mesh, files move together, a handle is refused,
  float32 throughout.
- Antialiasing: a tilted plane comes out flat and a ball keeps its radius
  (every device); signs kept by at least eps, signed distance outside the
  band; the binary mesh's triangles and keys; no self-intersections; the
  same optimum as L-BFGS-B and as CVXPY with Clarabel; devices agree.
- Devices: marching and the normal smoothing on each PyTorch device against
  numpy; the block reader on each device; the device names and refusals.

## Measurements and timings

Results in `scratch/whole-wm/` (right hemisphere, grid 700 × 384 × 1148, mask
box 632 × 323 × 1006):

| Step | Result | Peak |
|---|---|---|
| `dseg` | Betti (1, 111, 182) → (1, 0, 0); 6,307 cavity voxels, 34 fills (570 voxels), 52 cuts (589 voxels: 529 to cortex, 60 to background); 3.4 min | 21.6 GB |
| `mesh` | one closed genus-0 piece, 72,097 mm²; 12.7 M vertices, 25.3 M faces in 506 blocks; 0 self-intersecting faces in the file; the antialiasing 1,651 iterations on every device | 20 to 24 GB |
| `mesh`, CPU against MPS | the same triangles and keys, vertex positions bit for bit, normals within 3.5e-7 | |

For comparison, label 41 alone has Betti (1, 243, 31), and 14 % of the
surface of its outer voxels faces something other than cortex (8 % with the
filled labels, mostly the medial cut face against the Fomblin).

Timings: the time per iteration is constant through a solve (the same work
every iteration), so it is measured from about 100 iterations and the
iteration count separately; the solver logs both every 100 iterations.
Laptop numbers are for this laptop only: 0.57 s per antialiasing iteration
on MPS on AC power (217 M voxels, 2.1 ns per voxel, about 160 GB/s), about
1.2 s on battery. Timings are taken on AC, with sleep prevented
(`caffeinate -i`), one run at a time, nothing else on the GPU. To measure
fairly next: the band layout against the whole volume, per iteration.

Left until a measurement asks for it: decimating the mesh (a question for
the column stage).

## Open

- Whole brains: the corpus callosum joins the two hemispheres' filled masks,
  so a whole brain needs a midline cut (FreeSurfer's cutting planes) before
  step 1. Not needed for the example hemisphere.
