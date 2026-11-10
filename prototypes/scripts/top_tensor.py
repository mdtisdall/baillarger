"""orient's tensor J with a dilated cortex mask, for the whole hemisphere, one
level-1 superblock at a time, into a float32 memmap (6, X, Y, Z).
The local level is measured from cortex voxels only and evaluated across the
dilated band; detrending, gradients and the tensor use the dilated mask
(radius 0 reproduces orient; checked on superblock 2_2_4 to 1e-6).

args: OUT RADIUS_MM   env: DEVICE (mps); inputs: top_paths.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from baillarger import orient as O
from baillarger.backend import device as get_device
from baillarger.orient_noise import read_noise, superblocks
from baillarger.timing import peak_rss_gb

sys.path.insert(0, str(Path(__file__).parent))
import top_paths as TP  # noqa: E402

DSEG, ECHOES, NOISE = TP.dseg(), TP.echoes(), TP.noise()

OUT, RADIUS = Path(sys.argv[1]), float(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
DEV = get_device(os.environ.get("DEVICE", "mps"))
s = O.OrientSampler(ECHOES, DSEG, cache_bytes=256 << 20, noise=NOISE)
noise = read_noise(NOISE)
vs = np.asarray(s.voxel_mm, np.float64)
sigma = O.SIGMA
shape = (6, *s.shape)
mm_path = OUT / "tensor.f32"
done_path = OUT / "tensor_done.json"
done = json.loads(done_path.read_text()) if done_path.exists() else []
mode = "r+" if mm_path.exists() else "w+"
J = np.memmap(mm_path, np.float32, mode, shape=shape)
if mode == "w+":
    J[:] = np.nan
extra = int(np.ceil(RADIUS / vs.min())) + 1


def tensor(e, ctx, dil, crop):
    te = torch.as_tensor(e.astype(np.float32), device=DEV.torch)
    tc = torch.as_tensor(ctx, device=DEV.torch)
    td = torch.as_tensor(dil, device=DEV.torch)
    le = O.log_echo(te)
    del te
    lv0 = torch.nan_to_num(O.masked_smooth(le, tc, O._kernels(sigma, vs)))
    det = torch.where(td, le - lv0, 0.0)
    del le
    sm = torch.where(td, O.masked_smooth(det, td, [O.taps(O.SD)] * 3), 0.0)
    del det
    g = torch.where(td, torch.stack([O.central(sm, a) / float(vs[a]) for a in range(3)], 1), 0.0)
    del sm
    w = O._inv(O._noise_c(lv0, noise))
    del lv0
    h = torch.einsum("ef...,fa...->ea...", w, g)
    del w
    comp = [O.masked_smooth((g[:, i] * h[:, j]).sum(0), td, [O.taps(O.SI)] * 3) for i, j in O.PAIRS]
    del g, h
    Jt = torch.where(td, torch.stack(comp, 0), torch.nan)
    return Jt[(slice(None), *crop)].float().cpu().numpy()


sbs = list(superblocks(s, 1))
print(len(sbs), "superblocks", flush=True)
t0 = time.perf_counter()
for i, sb in enumerate(sbs):
    if list(sb) in done:
        continue
    t = time.perf_counter()
    out = s.output_box(sb, 1, 0.0)
    reach = [r + extra for r in O.orient_reach(vs, sigma)]
    read = s.grow(out, reach)
    e, ctx = s.read(read)
    dil = distance_transform_edt(~ctx, sampling=vs) <= RADIUS + 1e-6 if RADIUS > 0 else ctx
    crop = tuple(slice(o.start - r.start, o.stop - r.start) for o, r in zip(out, read))
    if dil.any():
        J[(slice(None), *out)] = tensor(e, ctx, dil, crop)
        J.flush()
    done.append(list(sb))
    done_path.write_text(json.dumps(done))
    print(f"{i + 1}/{len(sbs)} {sb} {time.perf_counter() - t:.1f} s, peak {peak_rss_gb():.1f} GB", flush=True)
(OUT / "tensor.json").write_text(json.dumps(dict(shape=shape, dtype="float32", affine=s.affine.tolist(),
                                                 seconds=time.perf_counter() - t0, sigma=sigma, dilate_mm=RADIUS,
                                                 device=DEV.name)))
print("done", time.perf_counter() - t0, flush=True)
