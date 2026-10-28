import pickle, sys
import numpy as np
from scipy.stats import spearmanr
R = pickle.load(open(f"{sys.argv[1]}/ev_closed_results.pkl", "rb"))
rs = R["open"]
h = np.array([r["Hm"] for r in rs])
top = np.array([r["s"][r["kt"]] for r in rs]); st = np.array([r["s"][r["ks"]] for r in rs])
for lab, y in (("total thickness", top), ("WM to stria (deep part)", st), ("stria to pia (superficial part)", top - st)):
    core = (h > np.percentile(h, 5)) & (h < np.percentile(h, 95))
    sl = np.polyfit(h[core], y[core], 1)[0]
    print(f"{lab:32s} median {np.median(y):.2f} mm  IQR {np.subtract(*np.percentile(y,[75,25])):.2f}  slope {sl:+.2f} mm per /mm  rho {spearmanr(h,y).statistic:+.2f}  (n {len(y)})")
