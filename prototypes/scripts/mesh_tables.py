"""Markdown tables for finding 28 from mesh_phantoms.json (mm; H = 0.16 mm per voxel).

Usage: mesh_tables.py <mesh_phantoms.json>
"""

import json
import sys

H = 0.16
R = json.load(open(sys.argv[1]))
S, MA, FIN = R["surface"], R["march"], R["fin"]
mm = lambda x: f"{x * H:+.3f}" if x == x else "–"  # noqa: E731
mma = lambda x: f"{x * H:.3f}" if x == x else "–"  # noqa: E731

print("### Surface\n")
print("| Start surface | Ball R 1.28 mm: bias, \\|d\\| p95 | Wavy: \\|d\\| med, p95 | Plane 4°: \\|d\\| med | Normal error, wavy (°) | H error med / p95 at 0.48 mm, wavy; ball (1/mm) | Blade 1.5 / 2 / 3 voxels: sides covered | Fin w 2: tip retraction |")
print("|---|---|---|---|---|---|---|---|")
names = ["voxel faces (quadric)", "binary contour", "white_mesh s0.5", "white_mesh s1", "white_mesh s1.5", "white_mesh s2",
         "twice 0.08mm", "twice 0.16mm", "twice 0.32mm", "twice 0.48mm", "taubin x50", "s0.5+twice 0.16mm", "s0.5+twice 0.32mm"]
for n in names:
    b, w, p = S["ball_R8"][n], S["wavy"][n], S["plane_4deg"][n]
    cov = []
    for bl in ("blade_w1.5", "blade_w2", "blade_w3"):
        r = S[bl][n]
        cov.append(f"{min(r['cover_+'], r['cover_-']):.2f}" if "cover_+" in r else "–")
    fin = FIN["fin_w2"].get(n, {}).get("tip_retraction_med", float("nan"))
    print(f"| {n} | {mm(b['pos_bias'])}, {mma(b['pos_abs_p95'])} | {mma(w['pos_abs_med'])}, {mma(w['pos_abs_p95'])} | "
          f"{mma(p['pos_abs_med'])} | {w['normal_med']:.1f} | {w['H_err_med_0.48']:.3f} / {w['H_err_p95_0.48']:.3f}; "
          f"{b['H_err_med_0.48']:.3f} / {b['H_err_p95_0.48']:.3f} | {' / '.join(cov)} | {mma(fin)} |")

print("\n### March, at the top (median signed error, IQR; mm)\n")
for ph in ("ball_R8", "wavy", "plane_4deg", "blade_w2"):
    print(f"\n{ph}\n")
    print("| Start | Ratio | Grid | Arrival: bias (IQR) | Arc length: bias (IQR) | Root, tangential: med / p95 | div of unit grad, 0.24–0.48 mm above: bias (IQR), 1/mm |")
    print("|---|---|---|---|---|---|---|")
    for k, r in MA[ph].items():
        st, ra, gr = [x.strip() for x in k.split("|")]
        print(f"| {st} | {ra.split()[1]} | {gr} | {mm(r['arr_bias'])} ({mma(r['arr_iqr'])}) | {mm(r['len_bias'])} ({mma(r['len_iqr'])}) | "
              f"{mma(r['root_tan_med'])} / {mma(r['root_tan_p95'])} | {r['div_1.5_3_bias']:+.2f} ({r['div_1.5_3_iqr']:.2f}) |")

if "band" in R:
    B = R["band"]
    print("\n### Narrow band (|d| <= 2 voxels), wavy body: 1x / 2x grid; blade 1.5 and 2 voxels at 2x; triangles (wavy)\n")
    print("| Start | Signed distance error, IQR (p95), mm | Foot error median / tangential p95, mm | Wrong sign, 2x (wavy; blade 2) | Foot jump per step p95; share > 2, 2x | Blade 1.5 / 2, 2x: foot on the wrong side; inside read as outside | Triangles: min angle p5, aspect p95, dual-area CV |")
    print("|---|---|---|---|---|---|---|")
    for n in ["voxel faces", "binary contour", "white_mesh s0.5", "white_mesh s1", "white_mesh s1.5", "white_mesh s2",
              "twice 0.16mm", "twice 0.32mm", "twice 0.48mm", "implicit s0.5", "implicit s1"]:
        w1, w2 = B["wavy"][f"{n} | 1x"], B["wavy"][f"{n} | 2x"]
        b2 = B["blade_w2"][f"{n} | 2x"]
        b15 = B["blade_w1.5"][f"{n} | 2x"]
        side = []
        for b in (b15, b2):
            side.append("gone" if b.get("missing") else f"{b['wrong_side']:.3f}; {b['inside_sign_err']:.3f}")
        tri = (f"{w1['minangle_p5']:.0f}°, {w1['aspect_p95']:.1f}, {w1['dual_cv']:.2f}" if "aspect_p95" in w1 else "—")
        bs = "gone" if b2.get("missing") else f"{b2['sign_err']:.3f}"
        print(f"| {n} | {w1['sd_iqr'] * H:.3f} ({w1['sd_p95'] * H:.3f}) / {w2['sd_iqr'] * H:.3f} ({w2['sd_p95'] * H:.3f}) | "
              f"{w1['foot_med'] * H:.3f} / {w1['foot_tan_p95'] * H:.3f} | {w2['sign_err']:.3f}; {bs} | "
              f"{w2['jump_p95']:.2f}; {w2['jump_gt2']:.3f} | {' / '.join(side)} | {tri} |")
