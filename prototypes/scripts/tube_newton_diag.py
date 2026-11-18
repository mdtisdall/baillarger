"""Newton against Gauss-Newton diagnostics on sampled tube problems.

usage: tube_newton_diag.py SITE_NPZ SNAPSHOT_NPZ OUT_DIR [n_window] [n_full] [seed]

Loads a solver snapshot (τ, owner, stopped) into a TubeSolver on the site,
samples tubes, builds their local problems exactly as the solver does
(window τ >= -D, and full depth), and runs two instrumented solves from the
same start:

- Newton as in tube_energy.newton_batch (|λ| with floor, Armijo);
- Gauss-Newton / Levenberg-Marquardt on the least-squares residuals with
  frozen weights (tube_energy.ls_residuals), gradient exact.

Per problem and iteration it records (diagnostics 1-4, 6):
1. ||H - H_GN||_F / ||H_GN||_F (H exact, H_GN = 2 JᵀJ);
2. negative eigenvalues of H: count, λ_min / max |λ|; condition of |H|;
3. Newton: accepted α, ρ = actual / model decrease (modified model);
   GN: ρ and μ;
4. kink crossings: unknowns whose f regime (τ <= -h/2, |τ| < h/2, τ >= h/2)
   changed in the iteration;
6. for the eigenvectors of λ_min and λ_max of H (first iteration): the share
   of their gradient along ĝ₀ (1 = along the column, 0 = across).
And (diagnostic 5) the head-to-head: final energies, iterations, wall time,
max |u_Newton - u_GN|.
Writes OUT_DIR/newton_diag.json and newton_diag.png.
"""

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tube_energy as TE  # noqa: E402
import tube_prep as TP  # noqa: E402
import tube_solver as TS  # noqa: E402

ITERS = 20
TOL = 1e-6
ALPHAS = 0.5 ** np.arange(20.0)


def load_snapshot(S: TS.TubeSolver, path: Path) -> None:
    s = np.load(path)
    inner = tuple(S.shape_p - 2 * TS.PAD)
    xyz = np.stack(np.unravel_index(s["index"], inner), 1) + TS.PAD
    flat = np.ravel_multi_index(xyz.T, tuple(S.shape_p))
    S.tau[:] = np.nan
    S.owner[:] = -1
    S.tau[flat] = s["tau"]
    S.owner[flat] = s["owner"]
    S.stopped = s["stopped"].astype(bool)
    S._sync_obj()
    S._colour()
    S._resid = S.residual()


def make_fns():
    def E(u, P, g0):
        return TE.energy(TE._embed(u, P), P, g0)

    def gn(u, P, g0, fw, tw):
        r = lambda uu: TE.ls_residuals(TE._embed(uu, P), P, g0, fw, tw)  # noqa: E731
        Jm = jax.jacfwd(r)(u)
        return 2 * Jm.T @ Jm

    def weights(u, P):
        return TE.irls_weights(TE._embed(u, P), P)

    def along_share(e, P, g0):
        x = jnp.zeros_like(P["x0"]).at[P["unk"]].set(e)
        g = TE.gradient(x, P)
        w = P["evalw"]
        al = jnp.sum(w * jnp.sum(g0 * g, -1) ** 2)
        return al / jnp.maximum(jnp.sum(w * jnp.sum(g * g, -1)), 1e-30)

    return {
        "E": jax.jit(E), "g": jax.jit(jax.grad(E)), "H": jax.jit(jax.hessian(E)), "GN": jax.jit(gn),
        "W": jax.jit(weights), "g0": jax.jit(TE.g0_of), "share": jax.jit(along_share),
        "Ea": jax.jit(jax.vmap(E, in_axes=(0, None, None))),
    }


def regime(u, h):
    return np.where(u <= -h / 2, 0, np.where(u >= h / 2, 2, 1))


def newton(F, P, g0, u0, um, inner_u, nev, h):
    u, e = u0.copy(), float(F["E"](u0, P, g0))
    recs, mask = [], um[:, None] * um[None, :]
    for it in range(ITERS):
        gr = np.asarray(F["g"](u, P, g0)) * um
        H = np.asarray(F["H"](u, P, g0)) * mask + np.diag(1 - um)
        fw, tw = F["W"](u, P)
        Hg = np.asarray(F["GN"](u, P, g0, fw, tw)) * mask
        real = um > 0
        Hr, Hgr = H[np.ix_(real, real)], Hg[np.ix_(real, real)]
        w, V = np.linalg.eigh(H)
        wr = np.linalg.eigvalsh(Hr)
        aw = np.maximum(np.abs(w), 1e-6 * np.abs(w).max() + 1e-12)
        d = -(V @ ((V.T @ gr) / aw)) * um
        rec = {
            "R_ratio": float(np.linalg.norm(Hr - Hgr) / max(np.linalg.norm(Hgr), 1e-30)),
            "n_neg": int(np.sum(wr < -1e-9 * np.abs(wr).max())), "n": int(real.sum()),
            "neg_ratio": float(wr.min() / np.abs(wr).max()),
            "cond_abs": float(np.abs(wr).max() / max(np.abs(wr).min(), 1e-300)),
            "gnorm": float(np.linalg.norm(gr)),
        }
        if it == 0:
            Vr = np.linalg.eigh(Hr)[1]
            for nm, col in (("share_min", 0), ("share_max", -1)):
                e_full = np.zeros_like(u)
                e_full[real] = Vr[:, col]
                rec[nm] = float(F["share"](e_full, P, g0))
        es = np.asarray(F["Ea"](u[None] + ALPHAS[:, None] * d[None], P, g0))
        gd = float(gr @ d)
        ok = es <= e + 1e-4 * ALPHAS * gd
        if not ok.any():
            rec["alpha"] = 0.0
            recs.append(rec)
            break
        k = int(np.argmax(ok))
        a = ALPHAS[k]
        dHd = float(np.sum(aw * (V.T @ d) ** 2))
        model = -(a * gd + 0.5 * a * a * dHd)
        rec.update(alpha=float(a), rho=float((e - es[k]) / max(model, 1e-300)))
        un = u + a * d
        ch = (regime(un, h) != regime(u, h)) & real & ~inner_u
        rec["kinks"] = int(ch.sum())
        recs.append(rec)
        de = e - float(es[k])
        u, e = un, float(es[k])
        if de < TOL * nev:
            break
    return u, e, recs


def gauss_newton(F, P, g0, u0, um, inner_u, nev, h):
    u, e = u0.copy(), float(F["E"](u0, P, g0))
    mask = um[:, None] * um[None, :]
    recs, mu = [], None
    for it in range(ITERS):
        gr = np.asarray(F["g"](u, P, g0)) * um
        fw, tw = F["W"](u, P)
        Hg = np.asarray(F["GN"](u, P, g0, fw, tw)) * mask
        D = np.maximum(np.diag(Hg), 1e-12 * max(np.diag(Hg).max(), 1e-12)) * um + (1 - um)
        if mu is None:
            mu = 1e-3
        acc = False
        for _ in range(30):
            d = -np.linalg.solve(Hg + mu * np.diag(D) + np.diag(1 - um), gr) * um
            en = float(F["E"](u + d, P, g0))
            model = -(gr @ d + 0.5 * d @ Hg @ d)
            rho = (e - en) / max(model, 1e-300)
            if en < e:
                acc = True
                break
            mu *= 4.0
        if not acc:
            recs.append({"rho": float(rho), "mu": float(mu), "accepted": False})
            break
        un = u + d
        ch = (regime(un, h) != regime(u, h)) & (um > 0) & ~inner_u
        recs.append({"rho": float(rho), "mu": float(mu), "accepted": True, "kinks": int(ch.sum()),
                     "gnorm": float(np.linalg.norm(gr))})
        mu = mu / 3 if rho > 0.75 else (mu * 2 if rho < 0.25 else mu)
        de = e - en
        u, e = un, en
        if de < TOL * nev:
            break
    return u, e, recs


def main() -> None:
    site_p, snap_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    nwin = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    nfull = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    out.mkdir(parents=True, exist_ok=True)
    site = TP.load_site(site_p)
    z = np.load(site_p)
    cv = TP.mm_to_vox(site.affine, z["site_mm"][None].astype(float))[0]
    pt = TP.make_patches(site)
    s = TS.Settings()
    S = TS.TubeSolver(site, pt, s, cv)
    load_snapshot(S, snap_p)
    S._index()
    live = np.flatnonzero(np.diff(S.tvptr) > 0)
    rng = np.random.default_rng(seed)
    F = make_fns()
    rows = []
    for depth, n in ((s.depth_mm, nwin), (np.inf, nfull)):
        for t in rng.choice(live, min(n, len(live)), replace=False):
            p = S.build(np.array([t]), depth)
            if p is None:
                continue
            Sz, U = TS._pow2(len(p["slots"]) + 1), TS._pow2(len(p["unk"]))
            q = S.pad(p, Sz, U, 1)
            P = {k: jnp.asarray(v) for k, v in q.items()}
            g0 = F["g0"](P["x0"], P)
            u0 = np.asarray(q["x0"][q["unk"]])
            um = q["umask"]
            inner_u = S.inside[p["slots"][np.minimum(q["unk"], len(p["slots"]) - 1)]] & (um > 0)
            nev = max(float(q["evalw"].sum()), 1.0)
            t0 = time.time()
            uN, eN, rN = newton(F, P, g0, u0, um, inner_u, nev, S.htau)
            t1 = time.time()
            uG, eG, rG = gauss_newton(F, P, g0, u0, um, inner_u, nev, S.htau)
            t2 = time.time()
            e0 = float(F["E"](u0, P, g0))
            rows.append({
                "tube": int(t), "full": bool(np.isinf(depth)), "U": int(um.sum()), "S": len(p["slots"]),
                "tissue_depth_mm": float(-np.nanmin(S.tau[S.tvidx[S.tvptr[t]:S.tvptr[t + 1]]])),
                "E0": e0, "E_newton": eN, "E_gn": eG, "it_newton": len(rN), "it_gn": len(rG),
                "s_newton": t1 - t0, "s_gn": t2 - t1, "du_max": float(np.max(np.abs(uN - uG) * um)),
                "newton": rN, "gn": rG,
            })
        print(f"depth {depth}: {sum(r['full'] == np.isinf(depth) for r in rows)} problems", flush=True)
    (out / "newton_diag.json").write_text(json.dumps({"snapshot": str(snap_p), "rows": rows}))
    summarise(rows, out)


def summarise(rows, out):
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    for full, colr, lab in ((False, "C0", "window (D = 2 mm)"), (True, "C3", "full depth")):
        R = [r for r in rows if r["full"] == full]
        if not R:
            continue
        first = [r["newton"][0] for r in R if r["newton"]]
        last = [r["newton"][-1] for r in R if r["newton"]]
        kw = dict(alpha=0.5, color=colr, label=lab)
        axs[0, 0].hist(np.log10([f["R_ratio"] for f in first]), bins=30, histtype="step", lw=1.5, **kw)
        axs[0, 0].hist(np.log10([f["R_ratio"] for f in last]), bins=30, histtype="step", ls="--", color=colr)
        axs[0, 1].scatter([f["n_neg"] / f["n"] for f in first], [-f["neg_ratio"] for f in first], s=8, **kw)
        axs[0, 2].hist(np.log10([f["cond_abs"] for f in first]), bins=30, **kw)
        al = [x.get("alpha", 0) for r in R for x in r["newton"]]
        axs[0, 3].hist(np.log2(np.maximum(al, 2.0**-21)), bins=22, **kw)
        rho = [x["rho"] for r in R for x in r["newton"] if "rho" in x]
        rhog = [x["rho"] for r in R for x in r["gn"] if x.get("accepted")]
        axs[1, 0].hist(np.clip(rho, -1, 2), bins=40, histtype="step", lw=1.5, color=colr, label=lab + " Newton")
        axs[1, 0].hist(np.clip(rhog, -1, 2), bins=40, histtype="step", ls="--", color=colr, label=lab + " GN")
        kn = [sum(x.get("kinks", 0) for x in r["newton"]) for r in R]
        axs[1, 1].scatter(kn, [r["it_newton"] for r in R], s=8, **kw)
        axs[1, 2].scatter([f["share_min"] for f in first], [f["share_max"] for f in first], s=8, **kw)
        eb = np.minimum([r["E_newton"] for r in R], [r["E_gn"] for r in R])
        dn = np.array([r["E_newton"] for r in R]) - eb
        dg = np.array([r["E_gn"] for r in R]) - eb
        axs[1, 3].scatter(np.log10(dn + 1e-6), np.log10(dg + 1e-6), s=8, **kw)
    t = [("log10 ||H - H_GN|| / ||H_GN||: first (solid), last (dashed)", ""),
         ("fraction of negative eigenvalues", "|λ_min| / max|λ|"),
         ("log10 condition of |H| (first iteration)", ""),
         ("log2 accepted α (Newton; -21 = none)", ""),
         ("ρ = actual / model decrease", ""),
         ("kink crossings (Newton, whole solve)", "Newton iterations"),
         ("along-column share, λ_min eigenvector", "along-column share, λ_max eigenvector"),
         ("log10 (E_Newton - E_best + 1e-6)", "log10 (E_GN - E_best + 1e-6)")]
    for a, (xl, yl) in zip(axs.ravel(), t):
        a.set_xlabel(xl, fontsize=8)
        a.set_ylabel(yl, fontsize=8)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "newton_diag.png", dpi=95)
    summ = {}
    for full in (False, True):
        R = [r for r in rows if r["full"] == full]
        if not R:
            continue
        first = [r["newton"][0] for r in R if r["newton"]]
        eb = np.minimum([r["E_newton"] for r in R], [r["E_gn"] for r in R])
        summ["full" if full else "window"] = {
            "problems": len(R), "U_median": float(np.median([r["U"] for r in R])),
            "R_ratio_first_median": float(np.median([f["R_ratio"] for f in first])),
            "R_ratio_last_median": float(np.median([r["newton"][-1]["R_ratio"] for r in R if r["newton"]])),
            "frac_indefinite_first": float(np.mean([f["n_neg"] > 0 for f in first])),
            "neg_ratio_median": float(np.median([f["neg_ratio"] for f in first])),
            "it_newton_median": float(np.median([r["it_newton"] for r in R])),
            "it_newton_at_limit": int(sum(r["it_newton"] >= ITERS for r in R)),
            "it_gn_median": float(np.median([r["it_gn"] for r in R])),
            "it_gn_at_limit": int(sum(r["it_gn"] >= ITERS for r in R)),
            "newton_lower_E": int(sum(r["E_newton"] < r["E_gn"] - 1e-6 for r in R)),
            "gn_lower_E": int(sum(r["E_gn"] < r["E_newton"] - 1e-6 for r in R)),
            "dE_newton_median": float(np.median(np.array([r["E_newton"] for r in R]) - eb)),
            "dE_gn_median": float(np.median(np.array([r["E_gn"] for r in R]) - eb)),
            "du_max_median_mm": float(np.median([r["du_max"] for r in R])),
            "kinks_median": float(np.median([sum(x.get("kinks", 0) for x in r["newton"]) for r in R])),
            "share_min_median": float(np.median([f["share_min"] for f in first])),
            "share_max_median": float(np.median([f["share_max"] for f in first])),
        }
    (out / "newton_diag_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
