# Rust and GPU streaming

**Question.** Would Rust be a better language for parts of the code? Two
parts were asked about: the viewer's server, and the core of the processing,
where torch moves a lot of data between host memory and the GPU.

**Status.** These notes come from reading the code at `e4e07e2`
(2026-11-10). Nothing was measured, and nothing was changed. We will come
back to them. The first step then is to measure (*Next step*).

**Short answer.**

- No Rust rewrite of the server or of the core.
- The server does little work of its own. The work is in the modules that
  it calls, and the pipeline uses the same modules.
- The cost of moving data to and from the GPU comes from the order of the
  steps and from the waits between them, not from Python. A Rust core
  would still copy the data to the GPU. It would also need its own GPU
  layer for CUDA and MPS, and that layer would have to be tested against
  numpy as the torch kernels are now.
- The fixes are in the torch code: send each superblock to the device
  once, overlap the copies with the compute, send narrow types, and remove
  the waits on each block.
- Rust can still be useful for single CPU functions that numba does badly,
  as a PyO3 extension. It is not useful as a rewrite.

## The viewer's server

`baillarger.viewer.server` is a stdlib `ThreadingHTTPServer` on 127.0.0.1.
One person uses it. It sends JSON, sends Server-Sent Events, and runs one
computation at a time under a lock. Its requests are not a bottleneck, so
a Rust HTTP layer would give no gain that a person sees.

Do not move its computations to Rust:

- **One implementation.** The columns pane gives each column "bit for bit"
  as the published ray sampler gives it (`viewer/columns.py`). The review
  previews call `wm_choices` directly. A Rust copy would be a second
  implementation, which must be kept the same as the first.
- **The kernels are native already.** numba compiles the inner loops.
  scipy, zlib and numpy release the GIL for their large operations.
- **The GPU goal.** It needs torch or CUDA kernels, not CPU Rust.

Two places need attention, but not Rust first:

1. **`mesh_graph.local_graph`.** It takes 2.5–4 s for each superblock on
   the hemisphere (`viewer/columns.py`, step 2). It uses numpy and scipy,
   not numba. Profile it. If the time is in Python work on graph
   structures (hash maps, building adjacency lists), then try numba or
   vectorised code first. A PyO3 function, shared with the pipeline, is
   the right form if those fail.
2. **Brick serving at 50 µm.** `viewer/bricks.ArrayPyramid` holds every
   level in memory and compresses each response with gzip. The hemisphere
   at 160 µm needs about 0.7 GB as int16. The whole brain at 50 µm needs
   about 45 GB (×(160/50)³, ×2 hemispheres). The fix is a change of
   storage, which the module already names: a multi-resolution NIfTI-Zarr
   store whose bricks are compressed once, when it is written. The server
   then sends the stored bytes. That is not a question of language.

## The core: how data goes to the GPU now

Each stage on a torch device uses the same pattern (`blocks.BlockReader.read`,
`orient.OrientSampler`, `column_volume._weighted_sum_torch`, `mesh` marching):

1. Read a block or superblock from the chunk cache into host memory.
2. Convert it on the host (`e.astype(np.float32)`: a full copy).
3. `torch.as_tensor(..., device=...)`: a copy that waits until it is done,
   from memory that is not pinned.
4. Compute on the device.
5. `.cpu().numpy()`: a copy that waits again. The next read starts only
   after this.

This keeps the volume off the device, which is correct (`blocks.py`: "the
volume is never held on the device"). But the copies and the compute never
overlap, and the GPU is idle during each read.

Sites that add to the cost:

| Site | What it does |
|---|---|
| `mesh_torch.MeshTorch.smooth` | Copies vertices and faces to the host for `gauss_steps`, then sends them to the device again, in the middle of the smoothing. |
| `orient.leading_any` | Runs `eigh` on the device, then copies vectors and eigenvalues to numpy for the coherence. |
| `orient.OrientSampler.orient` | Reads the WM distance for the sign on the host, after the tensor is computed on the device. |
| `mesh` marching (`bool(post.all())`, `bool(post.any())`) | The host waits for the device on each block to decide whether to skip it. |
| `orient`, `column_volume` | int16 echoes become float32 on the host, so twice the bytes cross to the device. |
| `orient.level` and `orient.orient` | Each stage reads and sends its own copy of the same superblock. |

On the laptop (MPS) host and GPU share memory, so these copies cost less
than on CUDA. The cluster numbers are the ones that matter.

## Why Rust does not fix this

- The copies come from the data layout and the order of the steps. Each
  Python call works on blocks of 64³ voxels or more, so the interpreter's
  cost is small.
- A Rust core needs its own GPU layer (cudarc, wgpu, candle or Burn).
  That layer would have to support CUDA and MPS, keep the kernels
  deterministic (no atomic additions), and be tested against the numpy
  path. torch gives all of this now.
- The numpy path is the reference for every device test. A Rust core
  would replace it or need it kept beside it.

## What would fix it

In order of expected gain:

1. **Measure.** `timing.py` already waits for the device before it reads
   the clock. Add a split between copy time and compute time for one
   superblock of `orient`. Measure on the CUDA cluster (torch profiler or
   `nsys`); the laptop gives a lower bound only.
2. **Send each superblock once.** Run level, orient, coherence and sign
   on the same tensors. Move `gauss_steps` and the numpy part of
   `leading_any` into torch.
3. **Overlap the copies with the compute.** Pinned staging buffers,
   `non_blocking=True`, a second CUDA stream, and a thread that reads and
   decompresses block n + 1 while block n computes. zlib and the Zarr
   codecs release the GIL, so the thread runs in parallel.
4. **Send narrow types.** Send int16 and convert to float32 on the device.
   This halves the copy and removes the host copy.
5. **Remove the waits on each block.** Test for empty blocks in a batch,
   or let the kernel handle an empty block.

## Where Rust can help later

- Single CPU functions that numba does badly: mesh topology, graph
  building, `local_graph`. Write them as PyO3 functions that the pipeline
  and the viewer share.
- Reading chunks at whole-brain scale. Try the existing libraries first:
  `zarrs` (Rust, MIT/Apache-2.0) and `tensorstore` (C++, Apache-2.0) both
  have Python bindings. Neither is GPL.

## Next step

Profile one superblock of `orient` on a torch device and record the share
of time in host-to-device copies, device-to-host copies, waits and
compute. Decide on items 2–5 from those numbers.
