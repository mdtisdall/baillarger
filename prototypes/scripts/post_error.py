"""Where the posterior misses: at each of the user's points, each front's predicted thickness minus
its arrival at the point (mu - T), with +-1 SD, from collide2.py's per-point rows."""

import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S, OUT = sys.argv[1], sys.argv[2]
rows = np.load(f"{S}/collide2_rows.npy", allow_pickle=True)
k = np.arange(1, len(rows) + 1)
e1 = np.array([r["mu1"] - r["T1"] for r in rows])
e2 = np.array([r["mu2"] - r["T2"] for r in rows])
s1 = np.array([r["s1"] for r in rows])
s2 = np.array([r["s2"] for r in rows])
post = np.array([r["post"] for r in rows])
fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
a = axs[0]
a.errorbar(
    k - 0.1,
    e1,
    yerr=s1,
    fmt="o",
    c="#2b7bd9",
    label="front 1: predicted thickness - arrival at your point",
)
a.errorbar(
    k + 0.1,
    e2,
    yerr=s2,
    fmt="s",
    c="#e8913a",
    label="front 2: predicted thickness - arrival at your point",
)
a.axhline(0, c="k", lw=0.8)
a.set_ylabel("mu - T at the point (mm), +-1 SD")
a.set_title(
    "If your points are the contact, each front's thickness up to them is T; the field predicted mu +- SD"
)
a.legend(fontsize=9)
a = axs[1]
a.bar(k, np.abs(post), color=["#d62728" if abs(p) > 0.3 else "#7f7f7f" for p in post])
a.set_ylabel("|posterior offset| (mm)")
a.set_xlabel("your contact point")
a.set_xticks(k)
a.set_title("posterior offset at each point (red: more than 0.3 mm)")
fig.tight_layout()
fig.savefig(f"{OUT}/collide2_error.png", dpi=80)
for i in range(len(rows)):
    z1, z2 = e1[i] / s1[i], e2[i] / s2[i]
    print(
        f"{i + 1:2d}: front 1 mu-T {e1[i]:+.2f} (z {z1:+.1f})  front 2 {e2[i]:+.2f} (z {z2:+.1f})  |post| {abs(post[i]):.2f}"
    )
