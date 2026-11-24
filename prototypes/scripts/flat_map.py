"""One conformal flat map of a box's white surface: the command line and the figures. The
algorithms are in baillarger.viewer.flatmap (topology, handle cuts, LSCM, distortion).

Usage: flat_map.py <figures> <work> <site> [--to-disk]   (writes <work>/flat_map_<site>.npz, .json and
       the views <figures>/flat_map_<site>_distortion.png, _hist_handles.png)
"""

import json
import sys
import time

import numpy as np

from baillarger.viewer.flatmap import (  # noqa: F401  (re-exported for the prototypes)
    CONE_W,
    boundary_loops,
    cot_laplacian,
    cut,
    cut_handles,
    distortion,
    edges_of,
    face_areas,
    flatten,
    greedy_loops,
    join_boundaries,
    loop_edges,
    lscm,
    orientable_check,
    pieces,
    shortest_handle,
    tangent_maps,
    topology,
    wpct,
)


def main(fig, work, site, to_disk=False):
    import matplotlib

    matplotlib.use("Agg")

    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    Z = np.load(f"{work}/column_volume_{site}.npz")
    vox, lo = Z["vox"], Z["lo"]
    vb = Z["v"].astype(float)
    v, f = vb * vox, Z["f"].astype(np.int64)
    assert orientable_check(f)
    out = flatten(v, f, to_disk=to_disk, log=say)
    say(f"flattened {len(out['pieces'])} pieces")
    pc_f = out["piece"][out["f_cut"][:, 0]]
    big = int(np.argmax([d["area_mm2"] for d in out["pieces"]]))
    tv = Z["tv"]
    loops = []
    for i, (lp, L) in enumerate(out["loops"]):
        c = vb[lp].mean(0)
        loops.append(dict(handle=i + 1, vertices=int(lp.size), length_mm=round(L, 3),
                          centre_itksnap=[int(x) for x in np.round(c) + 1],
                          centre_full_grid0=[int(x) for x in np.round(c) + lo],
                          extent_mm=[round(float(x), 2) for x in np.ptp(v[lp], 0)],
                          to_box_face_mm=round(float(np.min(np.r_[c, np.array(Z["disp"].shape) - 1 - c]) * vox.min()), 2),
                          traces_rooted_within_1mm=int(np.sum(np.min(np.linalg.norm(
                              v[tv][:, None] - v[lp][None, ::max(1, lp.size // 50)], axis=2), 1) < 1.0))))
    summ = dict(site=site, to_disk=to_disk, pieces=out["pieces"], handles=loops,
                distortion_all=distortion(out), distortion_largest=distortion(out, pc_f == big),
                seconds=round(time.time() - t0, 1))
    tag = "" if not to_disk else "_disk"
    lens = np.array([lp.size for lp, _ in out["loops"]], np.int64)
    np.savez_compressed(f"{work}/flat_map_{site}{tag}.npz", uv_v=out["uv_v"], T_v=out["T_v"], f_cut=out["f_cut"],
                        orig=out["orig"], uv=out["uv"], piece=out["piece"], s1=out["s1"], s2=out["s2"],
                        area3=out["area3"], signed_area=out["signed_area"],
                        loop_vertices=np.concatenate([lp for lp, _ in out["loops"]]) if lens.size else np.zeros(0, int),
                        loop_sizes=lens, loop_mm=np.array([L for _, L in out["loops"]]))
    json.dump(summ, open(f"{work}/flat_map_{site}{tag}.json", "w"), indent=1)
    print(json.dumps({k: v_ for k, v_ in summ.items() if k != "pieces"}, indent=1))
    view(fig, site + tag, out, v, vb)
    say("done")


def view(fig, name, out, v, vb):
    """The map coloured by log2 area factor and by conformal distortion, with the handle cuts;
    histograms; a 3D view of the mesh round each handle loop."""
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    uv, fc = out["uv"], out["f_cut"]
    la = -np.log2(out["s1"] * out["s2"])
    qc = out["s1"] / out["s2"] - 1
    tri = Triangulation(uv[:, 0], uv[:, 1], fc)
    nl = len(out["loops"])
    W = np.ptp(uv[:, 0])
    H = np.ptp(uv[:, 1])
    fw = 9 * W / max(H, 1e-9)
    figm, ax = plt.subplots(2, 1, figsize=(min(max(fw, 8), 22), 18))
    for a, val, cm, lim, ttl in [(ax[0], la, "RdBu_r", (-3, 3), "log2 area factor (map / surface)"),
                                 (ax[1], qc, "magma", (0, 0.2), "conformal distortion s1/s2 - 1")]:
        im = a.tripcolor(tri, facecolors=np.nan_to_num(val), cmap=cm, vmin=lim[0], vmax=lim[1])
        plt.colorbar(im, ax=a, shrink=0.6)
        for i, (lp, L) in enumerate(out["loops"]):
            cp = np.isin(out["orig"], lp)
            a.plot(uv[cp, 0], uv[cp, 1], ".", c="#00e000", ms=2)
            a.annotate(f"handle {i + 1} ({L:.1f} mm)", uv[cp].mean(0), color="#00a000", fontsize=9)
        a.set_aspect("equal")
        a.set_title(f"{name}: {ttl}; green: handle cuts", fontsize=11)
        a.set_xlabel("map mm")
    figm.tight_layout()
    figm.savefig(f"{fig}/flat_map_{name}_distortion.png", dpi=70)
    plt.close(figm)
    figh, ax = plt.subplots(1, 2 + nl, figsize=(5 * (2 + nl), 4.5), squeeze=False)
    ax = ax[0]
    ok = np.isfinite(la)
    ax[0].hist(la[ok], bins=120, weights=out["area3"][ok], range=(-6, 6))
    ax[0].set_title("log2 area factor (area-weighted)")
    ax[1].hist(qc[ok], bins=120, weights=out["area3"][ok], range=(0, 0.5))
    ax[1].set_title("s1/s2 - 1 (area-weighted)")
    for i, (lp, L) in enumerate(out["loops"]):
        a = figh.add_subplot(1, 2 + nl, 3 + i, projection="3d")
        ax[2 + i].axis("off")
        c = v[lp].mean(0)
        r = max(1.5, 0.75 * np.ptp(v[lp], 0).max())
        near = np.linalg.norm(v - c, axis=1) < r
        fk = out["orig"][fc[near[out["orig"][fc]].all(1)]]
        u, fl = np.unique(fk, return_inverse=True)
        p = v[u]
        a.plot_trisurf(p[:, 0], p[:, 1], p[:, 2], triangles=fl.reshape(-1, 3), color="#c8c8c8", alpha=0.5, lw=0.1,
                       edgecolor="#999")
        a.set_box_aspect(np.ptp(p, 0))
        q = v[np.r_[lp, lp[:1]]]
        a.plot(q[:, 0], q[:, 1], q[:, 2], "-", c="#e00000", lw=2)
        a.set_title(f"handle {i + 1}: {L:.2f} mm round; centre ITK-SNAP {np.round(vb[lp].mean(0)).astype(int) + 1}",
                    fontsize=8)
    figh.tight_layout()
    figh.savefig(f"{fig}/flat_map_{name}_hist_handles.png", dpi=70)
    plt.close(figh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--to-disk" in sys.argv)
