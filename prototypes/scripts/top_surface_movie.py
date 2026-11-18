"""Frames of Σ (and later τ) as a top-surface run evolves, and a page to step
through them: three slices and a 3D view (docs/plans/top-surface.md,
"Watching the optimization").

usage: top_surface_movie.py SITE_NPZ RUN_DIR OUT_DIR [--watch SECONDS]
Reads RUN_DIR/snapshots/*.npz (top_surface_run.py snapshots=true). For each
snapshot without a frame, writes
  OUT_DIR/frames/<name>.png: three slices through the site centre with echo
    2, τ in colour where the snapshot has it, the white surface (blue) and
    Σ (red) as cut lines, and the BLOCKED vertices (first contacts, cyan)
    within half a voxel of the slice;
  OUT_DIR/mesh/<name>.bin: Σ for the 3D view (uint32 nV, nF; float32 V
    (mm, centred on the site); uint32 F; float32 displacement (mm); uint8
    state).
Writes OUT_DIR/frames.json and OUT_DIR/index.html (slider, play, follow the
newest; the 3D view keeps its angle between frames). With --watch, repeats
every SECONDS until RUN_DIR/log.json exists (the run has ended), then once
more. three.js 0.160.0 (MIT) from jsDelivr.
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
import top_surface as TS  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

PAGE = """<!doctype html><meta charset="utf-8"><title>Top surface evolution</title>
<style>body{font:14px system-ui;margin:16px;background:#111;color:#ddd}
#bar{display:flex;gap:8px;align-items:center;margin:8px 0}input[type=range]{flex:1}
#row{display:flex;flex-direction:column;gap:12px}#img{width:100%;height:auto}
#v3{width:100%;height:560px;background:#000;position:relative}
#key{position:absolute;left:8px;bottom:8px;font-size:12px;background:#0008;padding:4px 6px}
#meta{font:12px ui-monospace,monospace;white-space:pre-wrap;color:#aaa}</style>
<div id="bar"><button id="play">play</button><input id="s" type="range" min="0" max="0" value="0">
<span id="name"></span><label><input id="follow" type="checkbox" checked> follow the newest</label></div>
<div id="row"><img id="img"><div id="v3"><div id="key">colour: displacement in this step,
0 (dark) to h/4 (yellow); grey = stopped at label 0; orange = stopped in WM;
cyan = blocked (first contact). Drag to rotate, wheel to zoom.</div></div></div>
<p>One frame after each start step (start-NN). Slices: echo 2, the white surface (blue),
Σ (red), the blocked vertices near the slice (cyan).</p><div id="meta"></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const H=__H__;
let F=[],t=null,cur=-1;const s=document.getElementById("s"),img=document.getElementById("img"),
nm=document.getElementById("name"),box=document.getElementById("v3"),meta=document.getElementById("meta");
const ren=new THREE.WebGLRenderer({antialias:true});box.appendChild(ren.domElement);
const scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(40,1,0.01,100);cam.position.set(0,-16,8);
scene.add(new THREE.HemisphereLight(0xffffff,0x333333,2.2));const dl=new THREE.DirectionalLight(0xffffff,1.2);
cam.add(dl);scene.add(cam);const ctl=new OrbitControls(cam,ren.domElement);let mesh=null;
function size(){const w=box.clientWidth,h=box.clientHeight;ren.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix();}
window.onresize=size;size();(function loop(){requestAnimationFrame(loop);ctl.update();ren.render(scene,cam);})();
function ramp(x){x=Math.max(0,Math.min(1,x));return [0.27+x*0.7,0.0+x*0.9,0.33-x*0.2];}
async function load3d(name){const b=await (await fetch("mesh/"+name+".bin")).arrayBuffer();
 const hd=new Uint32Array(b,0,2),nv=hd[0],nf=hd[1];let o=8;
 const V=new Float32Array(b,o,nv*3);o+=nv*12;const Fi=new Uint32Array(b,o,nf*3);o+=nf*12;
 const D=new Float32Array(b,o,nv);o+=nv*4;const S=new Uint8Array(b,o,nv);
 const C=new Float32Array(nv*3);for(let i=0;i<nv;i++){let c;
  if(S[i]==4)c=[0,0.9,0.9];else if(D[i]>0)c=ramp(D[i]/(H/4));else if(S[i]==1)c=[0.55,0.55,0.55];
  else if(S[i]==2)c=[0.9,0.5,0.1];else c=ramp(0);C.set(c,3*i);}
 const g=new THREE.BufferGeometry();g.setAttribute("position",new THREE.BufferAttribute(V,3));
 g.setAttribute("color",new THREE.BufferAttribute(C,3));g.setIndex(new THREE.BufferAttribute(Fi,1));g.computeVertexNormals();
 if(mesh){scene.remove(mesh);mesh.geometry.dispose();}
 mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:0.8}));scene.add(mesh);}
async function show(i){s.value=i;cur=+i;const n=F[i];nm.textContent=(+i+1)+" / "+F.length+"  "+n;
 img.src="frames/"+n+".png";load3d(n);
 try{meta.textContent=JSON.stringify(await (await fetch("meta/"+n+".json")).json());}catch(e){meta.textContent="";}}
async function load(){const r=await fetch("frames.json?"+Date.now());F=await r.json();s.max=F.length-1;
 if(document.getElementById("follow").checked&&!t&&cur!=F.length-1)show(F.length-1);}
s.oninput=()=>show(s.value);
document.getElementById("play").onclick=e=>{if(t){clearInterval(t);t=null;e.target.textContent="play";return;}
 e.target.textContent="stop";t=setInterval(()=>show((+s.value+1)%F.length),700);};
load();setInterval(load,10000);
</script>"""


def cut_segments(V: np.ndarray, F: np.ndarray, axis: int, k: float) -> np.ndarray:
    """Segments where the plane x[axis] = k (voxel units) cuts the mesh, in
    the slice's plot coordinates (column = second remaining axis, row = the
    first)."""
    rest = [a for a in range(3) if a != axis]
    d = V[:, axis] - k
    D = d[F]
    side = D > 0
    cut = side.any(1) & ~side.all(1)
    segs = []
    for f, dd in zip(F[cut], D[cut]):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (dd[i] > 0) != (dd[j] > 0):
                t = dd[i] / (dd[i] - dd[j])
                p = V[f[i]] + t * (V[f[j]] - V[f[i]])
                pts.append((p[rest[1]], p[rest[0]]))
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs).reshape(-1, 2, 2)


def render(z: dict, wv: np.ndarray, wf: np.ndarray, snap: Path, out: Path, cv: np.ndarray, vox: np.ndarray) -> dict:
    sn = np.load(snap)
    V = sn["V"].astype(float) / vox
    Fs = sn["F"].astype(np.int64)
    st = sn["state"]
    meta = json.loads(str(sn["meta"]))
    e2 = z["echoes"][1].astype(float)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.3), layout="constrained")
    for ax in range(3):
        a = axs[ax]
        k = int(cv[ax])
        a.imshow(np.take(e2, k, axis=ax), cmap="gray", origin="lower", interpolation="nearest")
        rest = [b for b in range(3) if b != ax]
        a.add_collection(LineCollection(cut_segments(wv, wf, ax, k), colors="deepskyblue", linewidths=0.8))
        a.add_collection(LineCollection(cut_segments(V, Fs, ax, k), colors="red", linewidths=1.0))
        bl = (st == TS.BLOCKED) & (np.abs(V[:, ax] - k) < 0.5)
        a.plot(V[bl, rest[1]], V[bl, rest[0]], ".", color="cyan", ms=2.5)
        a.set_title(f"{'xyz'[ax]} = {k}")
        a.set_xticks([])
        a.set_yticks([])
    states = meta.get("states", {})
    fig.suptitle(
        f"{snap.stem}   moving {states.get('moving', '-')}   top {states.get('top', '-')}   "
        f"wm {states.get('wm', '-')}   edge {states.get('edge', '-')}   blocked {states.get('blocked', '-')}"
    )
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return {"V": sn["V"], "F": Fs, "disp": sn["disp"], "state": st, "meta": meta}


def write_bin(p: Path, V: np.ndarray, F: np.ndarray, disp: np.ndarray, st: np.ndarray, c_mm: np.ndarray) -> None:
    with open(p, "wb") as fh:
        fh.write(np.array([len(V), len(F)], np.uint32).tobytes())
        fh.write((V - c_mm).astype(np.float32).tobytes())
        fh.write(F.astype(np.uint32).tobytes())
        fh.write(disp.astype(np.float32).tobytes())
        fh.write(st.astype(np.uint8).tobytes())


def once(z: dict, wv: np.ndarray, wf: np.ndarray, run: Path, out: Path, cv: np.ndarray, vox: np.ndarray) -> int:
    for d in ("frames", "mesh", "meta"):
        (out / d).mkdir(parents=True, exist_ok=True)
    snaps = sorted((run / "snapshots").glob("*.npz"))
    c_mm = cv * vox
    new = 0
    for p in snaps:
        f = out / "frames" / f"{p.stem}.png"
        if f.exists():
            continue
        r = render(z, wv, wf, p, f, cv, vox)
        write_bin(out / "mesh" / f"{p.stem}.bin", r["V"], r["F"], r["disp"], r["state"], c_mm)
        (out / "meta" / f"{p.stem}.json").write_text(json.dumps(r["meta"]))
        new += 1
    (out / "frames.json").write_text(json.dumps([p.stem for p in snaps]))
    (out / "index.html").write_text(PAGE.replace("__H__", f"{float(vox.mean()):.6f}"))
    return new


def main() -> None:
    site_p, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    watch = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[4] == "--watch" else None
    z = dict(np.load(site_p))
    A = z["affine"].astype(float)
    vox = np.abs(np.diag(A)[:3])
    wv = TP.mm_to_vox(A, z["white_v"].astype(float))
    wf = z["white_f"].astype(np.int64)
    cv = np.round((np.linalg.inv(A) @ np.r_[z["site_mm"], 1])[:3]).astype(int)
    while True:
        done = (run / "log.json").exists()
        n = once(z, wv, wf, run, out, cv, vox)
        if n:
            print(f"{time.strftime('%H:%M:%S')} {n} new frames", flush=True)
        if watch is None or done:
            break
        time.sleep(watch)


if __name__ == "__main__":
    main()
