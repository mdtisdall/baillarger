"""How bad is the base pial surface, and why (plan: reports/Pial editing and edit
priority.md, step 4 follow-up). M1S1 session file; all 3D.

For each stand-in base (label: fluid else ridge; one pass, equal weights):
- roughness: |t - median of t over the vertex's 1-ring| (mm), overall and by how the top
  was placed;
- self-intersecting face pairs of the pial mesh (baillarger.mesh_intersect), overall and
  >= 2 mm from the box faces;
- the same after a robust smoothing of t over the white mesh: 1-ring medians (3 passes),
  then a Gaussian along the mesh at 0.32 mm with twicing, as the white surface. Cortex has
  no faults, so a top that jumps from its neighbours is an error, not structure.
Views: prototypes/figures/pial_base_<base>.png, three planes through the patch centre:
echo 2 trend removed, the white surface, the base top and the smoothed top.

Usage: pial_base_quality.py <session.npz> <volume.nii.gz> <figures dir> <out json>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "pial_edit"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import sparse

import mesh_smoothing as M
import pial_height as P
from baillarger.mesh_intersect import intersecting_faces
from editor_server import load_session


def rings(f, n):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).tocsr()
    return A


def ring_median(t, A):
    out = t.copy()
    ip, ix = A.indptr, A.indices
    for v in range(len(t)):
        nb = ix[ip[v] : ip[v + 1]]
        vals = np.concatenate([[t[v]], t[nb]])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[v] = np.median(vals)
    return out


def smooth_scalar(t, v, f, sigma_vox):
    """Gaussian smoothing of a scalar along the mesh with twicing: the operator of
    mesh_smoothing.twice (umbrella steps, their number from the mesh's own edge lengths),
    applied to t. Box-face boundary vertices keep their values."""
    A, deg, _ = M._umbrella(v, f)
    fixed = M.boundary_vertices(f, len(v))
    n = M.gauss_steps(v, f, sigma_vox)

    def G(x):
        for _ in range(n):
            d = (A @ x) / deg - x
            d[fixed] = 0
            x = x + 0.5 * d
        return x

    g = G(t.copy())
    return 2 * g - G(g)


def roughness(t, A, kinds=None):
    med = ring_median(t, A)
    return np.abs(t - med)


def pct(x, q=(50, 90, 99)):
    x = x[np.isfinite(x)]
    return [round(float(v), 3) for v in np.percentile(x, q)] if x.size else None


def main() -> None:
    sess, vol, figdir, out = sys.argv[1:5]
    s, extra = load_session(sess, "label")
    z = np.load(sess)
    h = s.h
    img = np.asanyarray(nib.load(vol).dataobj)
    keep = ~extra["near_face"]
    A = rings(s.f, len(s.v))
    res = {}
    for base in ("label", "onepass_equal"):
        t = np.asarray(z[f"t_{base}"], np.float64)
        kind = z[f"kind_{base}"]
        r0 = roughness(t, A) * h
        t1 = t
        for _ in range(3):
            t1 = ring_median(t1, A)
        t2 = smooth_scalar(t1, s.v, s.f, 2.0)
        r2 = roughness(t2, A) * h
        p0 = P.point_at(s.cols, t)
        p2 = P.point_at(s.cols, t2)
        for p in (p0, p2):
            bad = ~np.isfinite(p).all(1)
            p[bad] = s.v[bad]
        fk = keep[s.f].all(1)
        x0 = intersecting_faces(p0, s.f)[0]
        x2 = intersecting_faces(p2, s.f)[0]
        r = dict(
            roughness_mm=pct(r0[keep]),
            roughness_by_kind_mm={
                P.TOP_NAMES[k]: pct(r0[keep & (kind == k)]) for k in P.TOP_NAMES
            },
            share_rough_over_0p16=round(float(np.mean(r0[keep] > 0.16)), 3),
            share_rough_over_0p16_by_kind={
                P.TOP_NAMES[k]: round(float(np.mean(r0[keep & (kind == k)] > 0.16)), 3)
                if (keep & (kind == k)).any()
                else None
                for k in P.TOP_NAMES
            },
            intersecting_faces=int(x0.sum()),
            intersecting_faces_inside=int((x0 & fk).sum()),
            smoothed=dict(
                roughness_mm=pct(r2[keep]),
                share_rough_over_0p16=round(float(np.mean(r2[keep] > 0.16)), 3),
                intersecting_faces=int(x2.sum()),
                intersecting_faces_inside=int((x2 & fk).sum()),
                change_mm=pct(np.abs(t2 - t)[keep] * h, (50, 90, 99)),
            ),
        )
        res[base] = r
        print(base, json.dumps(r, indent=1), flush=True)
        # view
        c = np.array(img.shape) // 2
        fig, axs = plt.subplots(1, 3, figsize=(18, 6.4))
        for a, ax in enumerate(axs):
            sl = [slice(None)] * 3
            sl[a] = c[a]
            o = [i for i in range(3) if i != a]
            ax.imshow(
                img[tuple(sl)].T,
                cmap="gray",
                origin="lower",
                vmin=-0.25,
                vmax=0.25,
                interpolation="nearest",
            )
            for pts, col, lw in (
                (s.v, "#1baf7a", 0.8),
                (p0, "#eb6834", 0.7),
                (p2, "#2a78d6", 1.0),
            ):
                for sg in P.plane_section(pts, s.f, a, float(c[a])):
                    ax.plot(sg[:, o[0]], sg[:, o[1]], color=col, lw=lw)
            ax.set_title(f"plane {'ijk'[a]} = {c[a]}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"M1S1, base '{base}': white surface (green), base top (orange), robustly smoothed top (blue); "
            "echo 2, trend removed",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(f"{figdir}/pial_base_{base}.png", dpi=110)
        plt.close(fig)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
