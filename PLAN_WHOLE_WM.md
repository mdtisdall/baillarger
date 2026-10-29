# Plan: the white mesh of a whole hemisphere

Planning file for the first stage of the move from boxes to whole hemispheres:
the white (WM) mesh. Later stages stay on boxes until their turn; each is
scaled when we get to it, and only the computational problems that we meet
are fixed. When this stage is done, the lasting content goes into the module
docstrings and the format specs (`docs/formats/`), and this file is deleted.

## Scaling principles

The target beyond this hemisphere is whole brains at 50 µm: about 65 times
the voxels (about 20 billion) and on the order of 350 M vertices. That runs
on a larger machine with CUDA GPUs; the laptop (64 GB, Apple GPU) is where
the scalable code is built and tested. Each step that this work touches
follows these rules; steps not yet touched keep their current form.

- Blocks with a halo: no whole-volume float arrays. Whole-volume steps work
  on blocks that fit in memory (and in GPU memory), and join exactly
  (marching tetrahedra: by global edge keys).
- A backend per step: numpy is the reference; PyTorch is the GPU backend
  (CUDA on large machines, MPS on the laptop), chosen at run time, CPU
  when no GPU is there. Tests on the laptop compare the PyTorch backend on
  MPS with numpy: integer and boolean results (vertex keys, labels,
  topology) exactly, floating-point results within a stated tolerance
  (GPUs work in float32).
- Global algorithms get a local form when their step is scaled: connected
  components in blocks; the Euler characteristic as a sum over blocks;
  handle removal as a global search for the handles, then local fixes in a
  box round each.
- Chunked storage when the data needs it: readers take a block, not the
  whole array, so that OME-Zarr (accepted by BIDS for microscopy) or
  memory-mapped uncompressed NIfTI can replace `.nii.gz`. Not needed for
  the example hemisphere.

For this stage: marching tetrahedra in blocks (done) gets a PyTorch backend
first; the smoothing (sparse matrix-vector products) next. Handle removal
stays sequential on the CPU for one hemisphere (3 min, 21 GB); its local
form is the first need at whole-brain scale.

## The two commands

```
python -m baillarger.white dseg ASEG --out DERIV
python -m baillarger.white mesh DSEG --out DERIV [--sigma S]
```

1. `dseg` reads the aseg and writes a corrected segmentation: the aseg with
   the filled WM mask made handle-free and cavity-free. The aseg file itself
   is never changed.
2. `mesh` reads that segmentation and writes the white mesh of the filled WM
   mask, as GIfTI.

Every later stage reads the corrected segmentation, not the aseg plus a list
of changes, so the mesh and the labels cannot disagree. Neither command takes
a box: they work on the whole grid.

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

Each output has a written, versioned spec in `docs/formats/`
(`wm-dseg.md`, `white-surf.md`): every file, array and sidecar key, with
type, units, required or not, and meaning. One writer and one reader per
format enforce it through a `bids.SidecarSpec` (each key's type and whether
it is required): the writer and the reader refuse a missing key, a key not
in the spec, a value of the wrong type or another `SpecVersion`.
`tests/test_bids.py` checks that each spec document's key table matches its
`SidecarSpec`. A field changes only through a PR that edits the spec and
bumps the version. (`wm-dseg.md`: done.)

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
2. Marching tetrahedra at 0.5 of the mask, in blocks (below), faces oriented
   out of the WM.
3. Tangential relaxation, twicing Gaussian smoothing (sigma in voxels,
   default 2) and smoothed normals (`mesh_smoothing`), as now.
4. Write the mesh as GIfTI with its data outside the XML
   (`Encoding="ExternalFileBinary"`), so the arrays are raw binary at known
   offsets: one copy of the mesh, in a standard format, readable by block.

| File (in `sub-<sub>/ses-<ses>/anat/`) | Content |
|---|---|
| `sub-<sub>_ses-<ses>_hemi-<R\|L>_white.surf.gii` | GIfTI XML (small): a vertex array (float32, scanner RAS mm, `NIFTI_INTENT_POINTSET`, coordinate system `NIFTI_XFORM_SCANNER_ANAT`) and a triangle array (int32, `NIFTI_INTENT_TRIANGLE`), each `ExternalFileBinary` into the `.dat` with its offset; metadata `AnatomicalStructurePrimary` (`CortexRight` / `CortexLeft`) and `GeometricType` (`Anatomical`) |
| `sub-<sub>_ses-<ses>_hemi-<R\|L>_white.dat` | The vertices (n × 3) then the triangles (m × 3), little-endian, row-major |
| `sub-<sub>_ses-<ses>_hemi-<R\|L>_desc-normals_white.func.gii` and `.dat` | The smoothed unit normals (float32, out of the WM, `NIFTI_INTENT_VECTOR`), in the same way, in a file of their own so that the `.surf.gii` holds only what every surface reader expects |
| `sub-<sub>_ses-<ses>_hemi-<R\|L>_desc-blocks_white.tsv` | The block index: one row per grid block, its position (`i`, `j`, `k`), its vertex and face ranges (`vertex_start`, `vertex_stop`, `face_start`, `face_stop`) and its vertices' bounding box in voxels |
| `sub-<sub>_ses-<ses>_hemi-<R\|L>_white.json` | Sidecar: `Sources` (the corrected segmentation) and its SHA-256, the SHA-256 of each `.dat`, the filled labels, sigma, `BlockCells`, the voxel-to-RAS affine of the grid (so a reader gets voxel coordinates without the segmentation), and the summary: vertices, faces and per piece its genus, boundary loops and area |

Order: the faces block by block (each face comes from the cells of one
block); the vertices by the first block that makes them, then by edge key.
A block's faces use its own vertices and some boundary vertices of
neighbouring blocks, so one block is read as its face range, then the
vertices those faces name. Smoothing keeps the order. `ExternalFileName` is
the `.dat`'s name only (relative), so the files can be moved together.

Who reads it (checked in their source): nibabel (memory-mapped; nilearn
through it), Connectome Workbench, gifticlib and so FreeSurfer
(`freeview`, `mris_convert`) and AFNI/SUMA, the MATLAB `gifti` toolbox
(SPM) and R's `gifti`. They read the arrays whole; reading by block is our
reader's. gifticlib opens a relative `ExternalFileName` from the working
directory, not the `.surf.gii`'s, so FreeSurfer and AFNI tools are run in
the files' directory (nibabel and MATLAB resolve it next to the file).
NiiVue does not read external data; our browser panes read a block's byte
ranges of the `.dat` directly (HTTP Range requests), using the block index.
nibabel cannot write this encoding (its writer raises
`NotImplementedError`; no issue or PR asks for it), so `baillarger.gifti`
writes the XML itself, and the tests read it back with nibabel. A feature
PR to nibabel could add the writer; ours stays until a release has it.

Size: uncompressed, about 0.6 GB for this hemisphere (12.7 M vertices,
25.3 M faces, with the normals), about 17 GB for a whole brain at 50 µm.

The reader (`WhiteSurface.load(path, block=None)`) memory-maps the `.dat`
files and returns, for the whole mesh or one block, vertices in voxel
coordinates of the grid, faces, normals and the affine; `check(dseg_path)`
compares the segmentation's SHA-256 with the sidecar's.

No per-vertex cortex label: every vertex is a column root. Columns run out
from the WM until they reach background (label 0) or a maximum length, and
what they pass through (cortex, hippocampus, water mislabelled as cortex) is
left to later stages.

## Blocked marching tetrahedra (done)

In one piece, `marching_tetrahedra` kept about 6.5 kB of arrays per surface
cell, about 40 GB for the hemisphere's 5.85 M surface cells.

- Cells are taken in blocks (default 64³ cells). A block whose grid points
  are all on one side of the level is skipped: no whole-grid float copy.
- Each block's vertices are keyed by their grid edge in the whole grid and
  computed from the edge's ends in a fixed order, so every block gives the
  same bits for a shared vertex; one `np.unique` over the keys joins them.
  The result does not depend on the block size (tests: blocks of 1, 3, 7
  and 16 cells give the one-shot mesh exactly).
- Block order (`marching_tetrahedra_blocks`): faces by block, vertices by
  the first block that makes them, then by edge key, with the block index
  (`MeshBlocks`). Done; tests check the ranges and that the surface is the
  same for every block size.
- Next: the per-block work in PyTorch, tested on MPS against numpy (exact
  keys and faces; vertices within float32 tolerance).

## What goes

- From `baillarger.white`: `WMChanges`, `wm_changes` and the `changes`
  subcommand (replaced by `dseg`); `white_matter_box`; the `box` subcommand,
  `box_report`, `write_box` and `overlay`; `box_mesh`, the box fields of
  `WhiteMesh` and its `.npz`. (Done: the box `mesh` subcommand, with its
  `--patch`, `--lo`, `--hi`, `.npz` and `.vtk` output, gave way to the new
  `mesh`; `mesh.write_vtk` stays for other tools.)
- The box stages that took the WM changes (`column_traces.load_tissue`,
  `changes=`; `pial_labels`, `--wm-changes`) read their box's labels from the
  corrected segmentation instead, as they read the aseg now. Their `mesh=` /
  `--mesh` option goes: they make their own box's mesh until their stage is
  scaled.
- `prototypes/scripts/wm_handles*.py` stay as records and are not updated.

## Tests

- Blocked marching tetrahedra equal the one-shot result on phantoms
  (vertices, faces, orientation), with blocks smaller than the features, and
  skipped blocks change nothing.
- `dseg` on a synthetic aseg with a torus-shaped WM, an enclosed ventricle
  and a cavity: Betti numbers after (b0, 0, 0), the filled-label voxels equal
  the corrected mask, cavity voxels set to WM, cuts relabelled by the nearest
  rule, other voxels unchanged.
- Names: entities read from the input and carried; derivatives layout;
  `dataset_description.json` written once.
- GIfTI round trip: coordinates (voxel → mm → voxel), faces, orientation,
  normals; the reader refuses a missing key or a wrong `SpecVersion`; each
  sidecar validates against its JSON Schema.
- `mesh` refuses a segmentation whose mask has a handle or a cavity.

## The full run

Both commands on the example hemisphere under `scripts/memguard`, logging the
time and peak memory of each step; results in `scratch/whole-wm/`.

Measured so far (right hemisphere, grid 700 × 384 × 1148, mask box
632 × 323 × 1006):

| Step | Time | Peak | Result |
|---|---|---|---|
| `dseg` (whole command) | 3.4 min | 21.6 GB | Betti (1, 111, 182) → (1, 0, 0); 6,307 cavity voxels, 34 fills (570 voxels), 52 cuts (589 voxels: 529 to cortex, 60 to background) |
| Blocked marching tetrahedra of its mask | 31 s | 4.3 GB (3.2 GB of it the mask) | 12.7 M vertices, 25.3 M faces; one closed genus-0 surface, no non-manifold edges |
| `mesh` (whole command, float32: Betti check, marching, relax, smoothing, untangle, normals, pieces, files) | 5.7 min, about 1.1 cores | 17.1 GB | one closed genus-0 piece, 72,074 mm²; 12.7 M vertices, 25.3 M faces in 506 blocks of 64 cells; 0.61 GB of files |
| Self-intersections | 16–20 s per check | 11–12 GB | smoothing made 13 intersecting faces at two places (a thin fold at voxel (147, 153, 656), and two faces at (276, 216, 139) that crossed only once rounded to float32); untangle moved 165 vertices back to half their smoothing; the written file has none |

For comparison, label 41 alone has Betti (1, 243, 31), and 14 % of the
surface of its outer voxels faces something other than cortex (8 % with the
filled labels, mostly the medial cut face against the Fomblin).

Self-intersections: the mesh before smoothing has none (it is the level set
of one piecewise-linear function), smoothing can make some, so `mesh`
untangles: it blends the vertices round intersecting faces back toward
their positions before smoothing until the mesh as written (mm, float32)
has none, records a summary in the sidecar and prints the places. The
check is whole-mesh (12 GB here); at whole-brain scale it runs in blocks
with a halo, since intersections are local.

Precision: the white mesh steps run in float32, as they will on a GPU (MPS
has no float64). `mesh_smoothing` keeps the vertices' dtype, and
`marching_tetrahedra` does its arithmetic in the dtype asked for. Its
triangles and their orientation follow from the corners' signs alone (a
table, `_FLIP`), and no triangle is dropped for zero area, so they are the
same in either precision: in float32 a field with grid points exactly on
the level had come apart (the 1e-12 nudge rounds away, the crossings land
on the grid point) until orientation and dropping stopped depending on
geometry. Against the float64 reference: the same triangles; vertices
within 2e-5 voxel, normals within 6e-6; on the hemisphere the same
self-intersections and area. Float64 stays where it buys exactness or a
reference: the affine to mm, area sums, `mesh_intersect`'s pair tests.
The sparse graphs that only give connectivity use int8.

Block bounding boxes are of the vertices as a reader gets them from the
file (mm back through the inverse affine), rounded outward: on the
hemisphere none of the 506 misses a vertex.

Parallelism: only `mesh_intersect` uses all cores (numba). Marching's
blocks are independent and the smoothing is sparse products; both go
parallel with the PyTorch backend (multi-threaded on the CPU, batched on a
GPU), the next piece of work.

Left until a measurement asks for it: handle removal in blocks round each handle (whole-brain
scale would be about 43 GB); decimating the mesh (a question for the column
stage).

## Open

- Whole brains: the corpus callosum joins the two hemispheres' filled masks,
  so a whole brain needs a midline cut (FreeSurfer's cutting planes) before
  step 1. Not needed for the example hemisphere.
