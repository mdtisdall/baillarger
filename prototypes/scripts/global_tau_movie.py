"""Frames of τ as a global-τ run evolves, and a page to step through them
(after tube_movie.py, branch feature/tube-solver).

usage: global_tau_movie.py SITE_NPZ RUN_DIR OUT_DIR [--watch SECONDS]
Reads RUN_DIR/snapshots/*.npz (global_tau_run.py snapshots=true: one before
and one after each LM iteration of the global solves, and one after each
ownership update). For each snapshot without a frame, writes
OUT_DIR/frames/<name>.png: three slices through the site centre with echo 2,
τ (colour, mm), the top τ = 0 (red), the owner-patch borders (thin white)
and the contact faces (cyan). τ is stored in the domain only. Outside it,
τ is filled from the side (inside the white surface: below the top, -9; the
rest: above it, +1), so the red line is only the top. Writes
OUT_DIR/frames.json and OUT_DIR/index.html (a slider, play, and a reload of
frames.json every 10 s). With --watch, repeats every SECONDS until
RUN_DIR/log.json exists (the run has ended), then once more.
"""

import json
import sys
import time
from pathlib import Path

import global_tau_prep as TP
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

PAGE = """<!doctype html><meta charset="utf-8"><title>Global τ evolution</title>
<style>body{font:14px system-ui;margin:16px;background:#111;color:#ddd}img{max-width:100%}
#bar{display:flex;gap:8px;align-items:center;margin:8px 0}input[type=range]{flex:1}</style>
<div id="bar"><button id="play">play</button><input id="s" type="range" min="0" max="0" value="0">
<span id="name"></span><label><input id="follow" type="checkbox" checked> follow the newest</label></div>
<img id="img"><p>One frame before and after each LM iteration of each outer iteration
(outerK-lmNN), and one after each ownership update (outerK-owner).
τ in colour (mm), the top τ = 0 red, owner-patch borders thin white, contact faces cyan.</p>
<script>
let F=[],t=null;const s=document.getElementById("s"),img=document.getElementById("img"),nm=document.getElementById("name");
function show(i){s.value=i;nm.textContent=(+i+1)+" / "+F.length+"  "+F[i];img.src="frames/"+F[i]+".png";}
async function load(){const r=await fetch("frames.json?"+Date.now());F=await r.json();s.max=F.length-1;
 if(document.getElementById("follow").checked&&!t)show(F.length-1);}
s.oninput=()=>show(s.value);
document.getElementById("play").onclick=e=>{if(t){clearInterval(t);t=null;e.target.textContent="play";return;}
 e.target.textContent="stop";t=setInterval(()=>show((+s.value+1)%F.length),400);};
load();setInterval(load,10000);
</script>"""


def load_snap(p: Path, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    z = np.load(p)
    tau = np.full(int(np.prod(shape)), np.nan, np.float32)
    own = np.full(int(np.prod(shape)), -2, np.int32)
    tau[z["index"]] = z["tau"]
    own[z["index"]] = z["owner"]
    return tau.reshape(shape), own.reshape(shape), z["faces"], json.loads(str(z["meta"]))


def face_segs(faces: np.ndarray, axis: int, k: int) -> list:
    """Contact faces cut by the slice (axis = k): segments in (col, row)."""
    rest = [a for a in range(3) if a != axis]
    segs = []
    for x0, y0, z0, a in faces:
        p = (x0, y0, z0)
        if p[axis] != k or a == axis:
            continue
        i, j = p[rest[0]], p[rest[1]]
        if a == rest[0]:
            segs.append(((j - 0.5, i + 0.5), (j + 0.5, i + 0.5)))
        else:
            segs.append(((j + 0.5, i - 0.5), (j + 0.5, i + 0.5)))
    return segs


def render(z: dict, inside: np.ndarray, snap: Path, out: Path, cv: np.ndarray) -> None:
    e2 = z["echoes"][1].astype(float)
    shape = e2.shape
    tau, own, faces, meta = load_snap(snap, shape)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.3))
    im = None
    for ax in range(3):
        a = axs[ax]
        k = int(cv[ax])
        a.imshow(np.take(e2, k, axis=ax), cmap="gray", origin="lower", interpolation="nearest")
        t = np.take(tau, k, axis=ax)
        im = a.imshow(np.ma.masked_invalid(t), cmap="viridis", vmin=-3, vmax=0.2, origin="lower", alpha=0.55,
                      interpolation="nearest")
        ins = np.take(inside, k, axis=ax)
        a.contour(np.where(np.isnan(t), np.where(ins, -9.0, 1.0), t), [0.0], colors="red", linewidths=1.0)
        o = np.take(own, k, axis=ax)
        b = ((np.diff(o, axis=0, prepend=o[:1]) != 0) | (np.diff(o, axis=1, prepend=o[:, :1]) != 0)) & (o >= 0)
        a.imshow(np.ma.masked_where(~b, b.astype(float)), cmap="Greys_r", vmin=0, vmax=1, origin="lower",
                 alpha=0.25, interpolation="nearest")
        a.add_collection(LineCollection(face_segs(faces, ax, k), colors="cyan", linewidths=1.4))
        a.set_title(f"{'xyz'[ax]} = {k}")
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.8, label="τ (mm)")
    e = meta.get("energy")
    fig.suptitle(f"{snap.stem}" + (f"   E = {e:.4g}" if e is not None else "") + f"   contact faces {len(faces)}")
    fig.savefig(out, dpi=80)
    plt.close(fig)


def once(z: dict, inside: np.ndarray, run: Path, out: Path, cv: np.ndarray) -> int:
    (out / "frames").mkdir(parents=True, exist_ok=True)
    snaps = sorted((run / "snapshots").glob("*.npz"))
    new = 0
    for p in snaps:
        f = out / "frames" / f"{p.stem}.png"
        if not f.exists():
            render(z, inside, p, f, cv)
            new += 1
    (out / "frames.json").write_text(json.dumps([p.stem for p in snaps]))
    (out / "index.html").write_text(PAGE)
    return new


def main() -> None:
    site_p, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    watch = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[4] == "--watch" else None
    z = dict(np.load(site_p))
    inside = TP.load_site(site_p).inside
    cv = np.round((np.linalg.inv(z["affine"]) @ np.r_[z["site_mm"], 1])[:3]).astype(int)
    while True:
        done = (run / "log.json").exists()
        n = once(z, inside, run, out, cv)
        if n:
            print(f"{time.strftime('%H:%M:%S')} {n} new frames", flush=True)
        if watch is None or done:
            break
        time.sleep(watch)


if __name__ == "__main__":
    main()
