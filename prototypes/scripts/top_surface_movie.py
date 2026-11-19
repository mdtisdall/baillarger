"""Frames of Σ (and later τ) as a top-surface run evolves, and a page to step
through them: three slices and a 3D view (docs/plans/top-surface.md,
"Watching the optimization").

usage: top_surface_movie.py SITE_NPZ RUN_DIR OUT_DIR [--watch SECONDS] [--at I,J,K]
Reads RUN_DIR/snapshots/*.npz (top_surface_run.py snapshots=true). For each
snapshot without a frame, writes
  OUT_DIR/frames/<name>.png: three slices through the site centre with echo
    2, τ in colour where the snapshot has it, the white surface (blue) and
    Σ (red) as cut lines, and the contact vertices (top_surface.contacts,
    cyan) within half a voxel of the slice;
  OUT_DIR/mesh/<name>.bin: Σ for the 3D view (uint32 nV, nF; float32 V
    (mm, centred on the site); uint32 F; float32 displacement (mm); uint8
    state, with CONTACT_CODE for a contact vertex: display only).
Writes OUT_DIR/frames.json and OUT_DIR/index.html (slider, play, follow the
newest; the 3D view keeps its angle between frames). With --watch, repeats
every SECONDS until RUN_DIR/log.json exists (the run has ended), then once
more. With --at, the slices go through voxel I,J,K. A snapshot with dtau
(the change of τ since the previous snapshot) gets a second row of slices
with it; meta "title" replaces the start-step title. three.js 0.160.0 (MIT) from jsDelivr.
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

CONTACT_CODE = 4  # the state code the 3D view shows in cyan (not a top_surface state)
TAU_RANGE_MM = (-2.0, 0.0)  # the fixed colour scale of τ in the slices
DTAU_LIN_MM, DTAU_MAX_MM = 1e-4, 0.05  # the Δτ row (symmetric log scale): linear below, clipped above

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
cyan = contact. Drag to rotate, wheel to zoom.</div></div></div>
<p>One frame after each start step (start-NN). Slices: echo 2, the white surface (blue),
Σ (red), the contact vertices near the slice (cyan).</p><div id="meta"></div>
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
    # a snapshot from before the contacts were measured: its old code 4 (BLOCKED)
    contact = sn["contact"] if "contact" in sn else st == CONTACT_CODE
    meta = json.loads(str(sn["meta"]))
    e2 = z["echoes"][1].astype(float)
    has_d = "dtau" in sn
    fig, axg = plt.subplots(2 if has_d else 1, 3, figsize=(15, 10.6 if has_d else 5.3), layout="constrained", squeeze=False)
    axs = axg[0]
    if has_d:
        # Δτ since the previous snapshot: where τ still moves
        from matplotlib.colors import SymLogNorm

        nrm = SymLogNorm(DTAU_LIN_MM, vmin=-DTAU_MAX_MM, vmax=DTAU_MAX_MM)
        for ax in range(3):
            a = axg[1, ax]
            k = int(cv[ax])
            a.imshow(np.take(e2, k, axis=ax), cmap="gray", origin="lower", interpolation="nearest")
            di = a.imshow(np.take(sn["dtau"], k, axis=ax), cmap="RdBu_r", norm=nrm, origin="lower", interpolation="nearest")
            V0 = sn["V"].astype(float) / vox
            a.add_collection(LineCollection(cut_segments(V0, sn["F"].astype(np.int64), ax, k), colors="black", linewidths=0.8))
            a.set_title(f"Δτ since the previous frame, {'xyz'[ax]} = {k}")
            a.set_xticks([])
            a.set_yticks([])
        fig.colorbar(di, ax=axg[1], shrink=0.8, label="Δτ (mm, symmetric log)")
    for ax in range(3):
        a = axs[ax]
        k = int(cv[ax])
        a.imshow(np.take(e2, k, axis=ax), cmap="gray", origin="lower", interpolation="nearest")
        if "tau" in sn:
            # one fixed scale for every frame: τ from TAU_RANGE_MM[0] (dark) to 0 (yellow);
            # values outside it are clipped and marked (red: τ > 0, white: below the range)
            tk = np.take(sn["tau"], k, axis=ax)
            ti = a.imshow(tk, cmap="viridis", origin="lower", interpolation="nearest", alpha=0.6,
                          vmin=TAU_RANGE_MM[0], vmax=TAU_RANGE_MM[1])
            out_hi = np.where(tk > TAU_RANGE_MM[1], 1.0, np.nan)
            out_lo = np.where(tk < TAU_RANGE_MM[0], 1.0, np.nan)
            a.imshow(out_hi, cmap="autumn", origin="lower", interpolation="nearest", vmin=0, vmax=1)
            a.imshow(out_lo, cmap="gray", origin="lower", interpolation="nearest", vmin=0, vmax=1)
        rest = [b for b in range(3) if b != ax]
        a.add_collection(LineCollection(cut_segments(wv, wf, ax, k), colors="deepskyblue", linewidths=0.8))
        a.add_collection(LineCollection(cut_segments(V, Fs, ax, k), colors="red", linewidths=1.0))
        bl = contact & (np.abs(V[:, ax] - k) < 0.5)
        a.plot(V[bl, rest[1]], V[bl, rest[0]], ".", color="cyan", ms=2.5)
        a.set_title(f"{'xyz'[ax]} = {k}")
        a.set_xticks([])
        a.set_yticks([])
    if "tau" in sn:
        fig.colorbar(ti, ax=axs, shrink=0.8, label="τ (mm); red = τ > 0, white = below −2 mm")
    states = meta.get("states", {})
    if "title" in meta:
        fig.suptitle(f"{snap.stem}   {meta['title']}")
    else:
        fig.suptitle(
            f"{snap.stem}   moving {states.get('moving', '-')}   top {states.get('top', '-')}   "
            f"wm {states.get('wm', '-')}   edge {states.get('edge', '-')}   contact {int(contact.sum())}"
        )
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    shown = np.where(contact, CONTACT_CODE, st)
    return {"V": sn["V"], "F": Fs, "disp": sn["disp"], "state": shown, "meta": meta}


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
    watch = float(sys.argv[sys.argv.index("--watch") + 1]) if "--watch" in sys.argv else None
    z = dict(np.load(site_p))
    A = z["affine"].astype(float)
    vox = np.abs(np.diag(A)[:3])
    wv = TP.mm_to_vox(A, z["white_v"].astype(float))
    wf = z["white_f"].astype(np.int64)
    cv = np.round((np.linalg.inv(A) @ np.r_[z["site_mm"], 1])[:3]).astype(int)
    if "--at" in sys.argv:  # the slices through this voxel, not the site centre
        cv = np.array([int(c) for c in sys.argv[sys.argv.index("--at") + 1].split(",")])
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
