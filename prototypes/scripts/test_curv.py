import sys

sys.path.insert(0, sys.argv[1])
import numpy as np
from scipy import ndimage as ndi
import curv_lib as C

vox = np.array([0.16, 0.16, 0.16])
g = np.indices((90, 90, 90)) * 0.16
c = 45 * 0.16
for name, wm in {
    "sphere R=3 (k=+0.33,+0.33)": ((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2)
    < 9,
    "hole R=3 in WM (k=-0.33,-0.33)": (
        (g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2
    )
    >= 9,
    "cylinder R=2 (k=+0.5, 0)": ((g[0] - c) ** 2 + (g[1] - c) ** 2) < 4,
    "blade 0.32 mm thick (0, 0)": np.abs(g[0] - c) < 0.17,
}.items():
    B = dict(wm=wm, vox=vox, shell=wm & ~ndi.binary_erosion(wm, border_value=1))
    for sg in (0.32, 0.48):
        K = C.curvature(B, sg)
        inner = np.all((K["pts"] > 20) & (K["pts"] < 70), 1)
        print(
            f"{name:32s} sig {sg}: k1 median {np.median(K['k1'][inner]):+.3f} (p5-95 {np.percentile(K['k1'][inner], [5, 95]).round(2)}), "
            f"k2 {np.median(K['k2'][inner]):+.3f} (p5-95 {np.percentile(K['k2'][inner], [5, 95]).round(2)})"
        )
