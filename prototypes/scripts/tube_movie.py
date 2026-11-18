"""Frames of the front as a tube-solver run evolves, and a page to step
through them.

usage: tube_movie.py SITE_NPZ RUN_DIR OUT_DIR [--watch SECONDS]
Reads RUN_DIR/snapshots/*.npz (one after each sweep and each step). For each
snapshot without a frame, writes OUT_DIR/frames/<name>.png: τ (colour) and the
front τ = 0 (red) over echo 2 on the three slices through the site, the WM
label edge (cyan dotted) and label 0 (blue). Outside the band, τ is filled from the label (WM: below the
front, else above it), so the red line is only the front. Writes OUT_DIR/frames.json (the
frame names in order) and OUT_DIR/index.html (a slider, play, and a reload of
frames.json every 10 s). With --watch, repeats every SECONDS until
RUN_DIR/log.json exists (the run has ended), then once more.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tube_figures import base, load_snap, sl  # noqa: E402

PAGE = """<!doctype html><meta charset="utf-8"><title>Front evolution</title>
<style>body{font:14px system-ui;margin:16px;background:#111;color:#ddd}img{max-width:100%}
#bar{display:flex;gap:8px;align-items:center;margin:8px 0}input[type=range]{flex:1}</style>
<div id="bar"><button id="play">play</button><input id="s" type="range" min="0" max="0" value="0">
<span id="name"></span><label><input id="follow" type="checkbox" checked> follow the newest</label></div>
<img id="img"><p>Each frame is one sweep (win: window sweep, full: full-depth sweep) or the end of a step.
τ in colour (mm), the front τ = 0 red, WM label edge cyan dotted, label 0 blue.</p>
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


def render(z: dict, snap: Path, out: Path, cv: np.ndarray) -> None:
    labels, e2 = z["labels"], z["echoes"][1].astype(float)
    tau, _ = load_snap(snap, labels.shape)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.3))
    for ax in range(3):
        a = axs[ax]
        base(a, sl(e2, ax, cv[ax]), sl(labels, ax, cv[ax]))
        t = sl(tau, ax, cv[ax])
        im = a.imshow(np.ma.masked_invalid(t), cmap="viridis", vmin=-3, vmax=0.2, origin="lower", alpha=0.55)
        # outside the band τ is not stored: deep WM is below the band (fill -9), the rest above it (fill +1),
        # so that the only τ = 0 contour is the front
        lab = sl(labels, ax, cv[ax])
        a.contour(np.where(np.isnan(t), np.where(lab == 41, -9.0, 1.0), t), [0.0], colors="red", linewidths=1.0)
        a.set_title(f"{'xyz'[ax]} = {cv[ax]}")
    fig.colorbar(im, ax=axs, shrink=0.8, label="τ (mm)")
    fig.suptitle(snap.stem)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def once(z: dict, run: Path, out: Path, cv: np.ndarray) -> int:
    (out / "frames").mkdir(parents=True, exist_ok=True)
    snaps = sorted(p for p in (run / "snapshots").glob("step-*.npz"))
    new = 0
    for p in snaps:
        f = out / "frames" / f"{p.stem}.png"
        if not f.exists():
            render(z, p, f, cv)
            new += 1
    (out / "frames.json").write_text(json.dumps([p.stem for p in snaps]))
    (out / "index.html").write_text(PAGE)
    return new


def main() -> None:
    site_p, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    watch = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[4] == "--watch" else None
    z = dict(np.load(site_p))
    cv = np.round((np.linalg.inv(z["affine"]) @ np.r_[z["site_mm"], 1])[:3]).astype(int)
    while True:
        done = (run / "log.json").exists()
        n = once(z, run, out, cv)
        if n:
            print(f"{time.strftime('%H:%M:%S')} {n} new frames", flush=True)
        if watch is None or done:
            break
        time.sleep(watch)


if __name__ == "__main__":
    main()
