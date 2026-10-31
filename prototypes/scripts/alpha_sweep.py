import pickle, sys
import numpy as np
from scipy.stats import spearmanr

S = sys.argv[1]
R = pickle.load(open(f"{S}/ev_closed_results.pkl", "rb"))
cz = np.load(f"{S}/curv_V1V2.npz")
idx = {tuple(p): n for n, p in enumerate(cz["pts"])}
for sc in ("0.32", "0.48", "0.8"):
    for lab in ("open", "A", "B"):
        rs = R[lab]
        out = []
        for al in (0.0, 0.25, 0.5, 0.75, 1.0):
            f, h, nf = [], [], 0
            for r in rs:
                n = idx[tuple(r["root"] + cz["lo"] - cz["lo"])]
                k1, k2 = al * cz[f"k1_{sc}"][n], al * cz[f"k2_{sc}"][n]
                s = r["s"][: r["kt"] + 1]
                a = (1 + k1 * s) * (1 + k2 * s)
                if a.min() < 0.1:
                    nf += 1
                    continue
                V = s + 0.5 * (k1 + k2) * s**2 + k1 * k2 * s**3 / 3
                f.append(V[r["ks"]] / V[-1])
                h.append(r["Hm"])
            f = np.array(f)
            out.append(
                f"a={al:.2f}: med {np.median(f):.3f} IQR {np.subtract(*np.percentile(f, [75, 25])):.3f} rho {spearmanr(h, f).statistic:+.2f} focal {nf}"
            )
        print(f"scale {sc} {lab:5s} | " + " | ".join(out))
