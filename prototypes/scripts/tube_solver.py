"""Tube front-growth τ solver (prototype), after the design notes
(scratch/.../plan-draft/notes.md) and the user's choices A-Q (2026-11-17).

τ is stored on the voxel grid, in the band only: the voxels inside the white
surface within INNER_VOX of it (inner, for the roots), the tissue voxels
(τ < 0) and a shell of voxels beyond the front (τ > 0, one voxel). Every band
voxel belongs to one tube (a white-surface patch). The front τ = 0 grows by
one voxel per step:

 1. prediction: τ -= δ in the tubes that are not stopped (sign changes
    checked with the simple-point test; a blocked voxel keeps τ = +ε);
    the shell is rebuilt;
 2. colour sweeps: for each colour, the tubes of that colour in parallel
    (batched): Newton on the window τ >= -D with every other τ frozen; then
    a line search on the accumulated step for the energy and the topology;
 3. repeat the sweeps until max |Δτ| < SWEEP_TOL or SWEEP_MAX sweeps;
 4. events: stops, retreats (with the per-term energy change of the solve),
    advances, contacts, topology blocks, the τ change beyond the shift;
 5. every M steps: full-depth sweeps; every R steps: re-trace the tubes
    (flow lines of -∇τ to the white surface) and rebuild the neighbours and
    colours; every C steps: the coarse correction (one τ shift per coarse
    patch over its fine tubes' windows, levels 4, 2, 1 mm);
 6. stop when a step changes no sign; then full-depth sweeps.

Contacts: a stencil does not cross between two tubes whose patches are not
in the same bank (> SAME_MM apart along the patch graph); there the
difference is one-sided. Tubes across a contact are otherwise ordinary
neighbours (colouring, ownership, topology).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tube_energy as TE
import tube_kernels as K
import tube_prep as TP
from scipy.spatial import cKDTree

PAD = K.PAD


@dataclass
class Settings:
    delta_vox: float = 1.0  # δ
    depth_mm: float = 2.0  # D
    m_full: int = 5  # m
    r_retrace: int = 5  # R
    c_coarse: int = 10  # C
    patch_mm: float = TP.PATCH_MM
    newton_iters: int = 20  # L
    newton_tol: float = 1e-6  # L: nats per eval voxel
    sweep_tol_mm: float = 0.01  # L
    sweep_max: int = 10  # L
    max_steps: int = 60  # guard: 9.6 mm of growth
    inner_vox: float = 1.5  # inner band depth inside the white surface
    block_eps: float = 1e-3  # τ of a voxel whose sign change is blocked
    thin_vox: int = 4  # M: thinner tubes merge
    median_mm: float = 2.35  # F
    curv_ratio: float = 1.23  # F
    corr_mm: float = 10.0  # F
    trace_step: float = 0.5  # voxels
    trace_max: int = 200
    batch_bytes: float = 6e8
    region_mm: float = 12.0  # A: the reported centre
    w_eq: float = TE.W_EQ


@dataclass
class Log:
    steps: list[dict[str, Any]] = field(default_factory=list)
    events: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "retreat": [], "advance": [], "stop": [], "contact": [], "block": [],
            "trace_fail": [], "merge": [], "topology_reject": [],
        }
    )


def _pow2(n: int, lo: int = 16) -> int:
    p = lo
    while p < n:
        p *= 2
    return p


class TubeSolver:
    def __init__(self, site: TP.Site, pt: TP.Patches, s: Settings, centre_vox: np.ndarray):
        self.site, self.pt, self.s = site, pt, s
        self.log = Log()
        shape = np.array(site.labels.shape)
        self.shape_p = shape + 2 * PAD
        sp = self.shape_p
        self.strides = np.array([sp[1] * sp[2], sp[2], 1], np.int64)
        N = int(np.prod(sp))
        self.N = N

        def padf(a: np.ndarray, fill: Any) -> np.ndarray:
            return np.pad(a, PAD, constant_values=fill).ravel()

        self.labels = padf(site.labels.astype(np.int64), -1)
        self.inside = padf(site.inside, False)
        self.allowed = ((self.labels == TP.CORTEX) | (self.labels == TP.OUTSIDE)) & ~self.inside
        self.c = np.where(self.labels == TP.OUTSIDE, TE.LOG_ODDS, 0.0)
        self.c = np.where(self.labels == TP.CORTEX, -TE.LOG_ODDS, self.c)
        self.coh = padf(site.coh.astype(np.float64), 0.0)
        self.nhat = np.stack([padf(site.nhat[..., a].astype(np.float64), 0.0) for a in range(3)], -1)
        self.h = site.vox.astype(np.float64)
        self.htau = float(self.h.mean())
        self.delta = s.delta_vox * self.htau
        dist = padf(site.dist, np.nan)

        # patches, alias (merges), areas, roots, median
        self.ntube = len(pt.centre)
        self.alias = np.arange(self.ntube)
        self.area = pt.area.copy()
        vmm = site.v * site.vox
        cen_vox = site.v[pt.centre] + PAD
        self.cen_mm = vmm[pt.centre]
        lo = np.floor(cen_vox).astype(np.int64)
        fr = cen_vox - lo
        self.roots = np.zeros((self.ntube, 8), np.int64)
        self.rootw = np.zeros((self.ntube, 8))
        for c in range(8):
            b = np.array([(c >> 2) & 1, (c >> 1) & 1, c & 1])
            q = lo + b
            self.roots[:, c] = q @ self.strides
            self.rootw[:, c] = np.prod(np.where(b == 1, fr, 1 - fr), 1)
        # region (A): patch centres inside the centre cube
        half = s.region_mm / 2
        self.in_region = np.all(np.abs((site.v[pt.centre] - centre_vox) * site.vox) <= half, axis=1)
        Hc = site.H[pt.centre]
        h10, h90 = np.percentile(Hc[self.in_region], [10, 90])
        self.beta = float(np.log(s.curv_ratio) / max(h90 - h10, 1e-9))
        self.logmed = np.log(s.median_mm) + self.beta * Hc
        # thickness-smoothness pairs: mesh-adjacent patches
        adj = np.concatenate([pt.adj, pt.adj[:, ::-1]])
        o = np.argsort(adj[:, 0], kind="stable")
        adj = adj[o]
        self.tptr = np.searchsorted(adj[:, 0], np.arange(self.ntube + 1))
        self.tnb = adj[:, 1]
        d = np.linalg.norm(self.cen_mm[adj[:, 0]] - self.cen_mm[adj[:, 1]], axis=1)
        self.tsd = TE.THICK_SD * np.sqrt(2 * (1 - np.exp(-d / s.corr_mm)))
        self.kmax = _pow2(int(np.max(np.diff(self.tptr))), 4)

        # facing vertices for the re-trace
        fv = np.flatnonzero(pt.vpatch >= 0)
        self.fv = fv
        self.ftree = cKDTree(site.v[fv] * site.vox)

        # initial band (Q): τ = d - h; tissue 0 <= d < h, inner -inner*h <= d < 0
        self.tau = np.full(N, np.nan)
        self.owner = -np.ones(N, np.int64)
        hm = self.htau
        cand = np.flatnonzero(
            np.isfinite(dist)
            & ((self.allowed & (dist >= 0) & (dist < hm)) | (self.inside & (dist >= -s.inner_vox * hm)))
        )
        xyz = np.stack(np.unravel_index(cand, sp), 1) - PAD
        vt = cKDTree(site.v * site.vox)
        _, j = vt.query(xyz * site.vox)
        own = pt.vpatch[j]
        keep = own >= 0
        cand, own = cand[keep], own[keep]
        self.tau[cand] = dist[cand] - hm
        self.owner[cand] = own
        self.inner = self.inside & (self.owner >= 0)
        self.inner_list = np.flatnonzero(self.inner)
        self.obj3 = np.zeros(tuple(sp), np.uint8)
        self._sync_obj()
        self.off2 = K.offsets_l1(2, self.strides)
        o26 = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)])
        self.off26 = o26 @ self.strides
        self.dist26 = np.linalg.norm(o26 * self.h, axis=1)
        ol1 = K.offsets_l1(4, self.strides)
        oinf = np.array([(a, b, c) for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)]) @ self.strides
        offs = np.unique(np.concatenate([ol1, oinf]))
        self.offs_nb = offs[offs != 0]
        self.localid = -np.ones(N, np.int64)
        self.wq_tube = s.w_eq / (2 * self.area)
        self.stopped = np.zeros(self.ntube, bool)
        self.step = 0
        self._merge_thin()
        self._rebuild_shell()
        self._index()
        self._colour()

    # ------------------------------------------------------------ state

    def _sync_obj(self) -> None:
        o = self.inside | ((self.owner >= 0) & (self.tau < 0))
        self.obj3[...] = o.reshape(tuple(self.shape_p))

    def band(self) -> np.ndarray:
        return np.flatnonzero(self.owner >= 0)

    def _index(self) -> None:
        b = self.band()
        o = np.argsort(self.owner[b], kind="stable")
        self.tvidx = b[o]
        self.tvptr = np.searchsorted(self.owner[self.tvidx], np.arange(self.ntube + 1))

    def tissue_count(self) -> np.ndarray:
        b = self.band()
        t = b[(self.tau[b] < 0) & ~self.inside[b]]
        return np.bincount(self.owner[t], minlength=self.ntube)

    def thickness(self) -> np.ndarray:
        tv = self.tau[self.roots]
        ok = self.owner[self.roots] >= 0
        w = np.where(ok, self.rootw, 0.0)
        tot = w.sum(1)
        th = -np.nansum(np.where(ok, w * tv, 0.0), 1) / np.maximum(tot, 1e-12)
        return np.where(tot > 0, th, np.nan)

    def residual(self) -> np.ndarray:
        t = np.maximum(self.thickness(), 0.02)
        return np.log(t) - self.logmed

    def _rebuild_shell(self) -> int:
        n = K.rebuild_shell(
            self.tau, self.owner, self.inside, self.allowed, self.band(),
            self.off26, self.dist26, self.s.block_eps,
        )
        return int(n)

    def _merge_thin(self) -> None:
        cnt = np.bincount(self.owner[self.owner >= 0], minlength=self.ntube)
        live = cnt > 0
        for t in np.flatnonzero(live & (cnt < self.s.thin_vox)):
            nbs = self.tnb[self.tptr[t] : self.tptr[t + 1]]
            nbs = nbs[cnt[nbs] >= self.s.thin_vox] if np.any(cnt[nbs] >= self.s.thin_vox) else nbs[cnt[nbs] > 0]
            if len(nbs) == 0:
                continue
            q = nbs[np.argmin(np.linalg.norm(self.cen_mm[nbs] - self.cen_mm[t], axis=1))]
            self.owner[self.owner == t] = q
            self.alias[self.alias == t] = q
            self.area[q] += self.area[t]
            self.wq_tube[q] = self.s.w_eq / (2 * self.area[q])
            self.log.events["merge"].append({"step": self.step, "tube": int(t), "into": int(q), "voxels": int(cnt[t])})
            cnt[q] += cnt[t]
            cnt[t] = 0

    def _colour(self) -> None:
        self._index()
        pairs = K.tube_pairs(self.tvptr, self.tvidx, self.owner, self.offs_nb, self.ntube)
        o = np.argsort(pairs[:, 0], kind="stable")
        pairs = pairs[o]
        ptr = np.searchsorted(pairs[:, 0], np.arange(self.ntube + 1))
        deg = np.diff(ptr)
        order = np.argsort(-deg, kind="stable")
        self.colour = K.greedy_colour(ptr, pairs[:, 1].copy(), order)
        self.nb_ptr, self.nb_idx = ptr, pairs[:, 1]
        self.ncolour = int(self.colour.max()) + 1

    # ------------------------------------------------------------ problems

    def build(self, members: np.ndarray, depth: float) -> dict[str, np.ndarray] | None:
        sl, un, ev, nb, nbv, valid, rslot, rw, hasroot = K.build_problem(
            members.astype(np.int64), self.tvptr, self.tvidx, self.tau, self.owner, self.inside,
            depth, self.pt.same_ptr, self.pt.same_idx, self.strides, self.off2, self.localid,
            self.roots, self.rootw,
        )
        if len(un) == 0:
            return None
        r = self._resid
        T = len(members)
        rj = np.zeros((T, self.kmax))
        sj = np.ones((T, self.kmax))
        mj = np.zeros((T, self.kmax))
        for a, t in enumerate(members):
            k0, k1 = self.tptr[t], self.tptr[t + 1]
            js = self.tnb[k0:k1]
            ok = np.isfinite(r[js]) & (self.alias[js] == js)
            n = len(js)
            rj[a, :n] = np.where(ok, r[js], 0.0)
            sj[a, :n] = self.tsd[k0:k1]
            mj[a, :n] = ok
        return {
            "slots": sl, "unk": un, "evalw": ev.astype(np.float64), "nb": nb, "nbv": nbv,
            "valid": valid.astype(np.float64), "rslot": rslot, "rw": rw, "hasroot": hasroot,
            "logmed": self.logmed[members], "rj": rj, "sj": sj, "mj": mj,
        }

    def pad(self, p: dict[str, np.ndarray], S: int, U: int, T: int) -> dict[str, np.ndarray]:
        sl = p["slots"]
        n = len(sl)
        dummy = S - 1
        x0 = np.zeros(S)
        x0[:n] = self.tau[sl]
        nb = np.tile(np.arange(S)[:, None, None], (1, 3, 2))
        nb[:n] = p["nb"]
        nbv = np.zeros((S, 3, 2), bool)
        nbv[:n] = p["nbv"]

        def z(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
            out = np.full((S,) + a.shape[1:], fill)
            out[:n] = a
            return out

        own = self.owner[sl]
        unk = np.full(U, dummy, np.int64)
        unk[: len(p["unk"])] = p["unk"]
        um = np.zeros(U)
        um[: len(p["unk"])] = 1.0

        def zt(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
            out = np.full((T,) + a.shape[1:], fill)
            out[: len(a)] = a
            return out

        rslot = np.full((T, 8), dummy, np.int64)
        rslot[: len(p["rslot"])] = p["rslot"]
        return {
            "x0": x0, "unk": unk, "umask": um, "evalw": z(p["evalw"]), "nb": nb, "nbv": nbv,
            "valid": z(p["valid"]), "inner": z(self.inside[sl].astype(np.float64)), "c": z(self.c[sl]), "coh": z(self.coh[sl]),
            "nhat": z(self.nhat[sl]), "wq": z(self.wq_tube[own]),
            "h": self.h, "htau": np.float64(self.htau),
            "rslot": rslot, "rw": zt(p["rw"]), "hasroot": zt(p["hasroot"]),
            "logmed": zt(p["logmed"]), "rj": zt(p["rj"]), "sj": zt(p["sj"], 1.0), "mj": zt(p["mj"]),
        }

    # ------------------------------------------------------------ solves

    def solve_groups(self, groups: list[np.ndarray], depth: float, coarse: bool = False) -> dict[str, Any]:
        """Solve each group (a tube, or a coarse patch's member tubes) with
        every other τ frozen. Groups must not interact (same colour).
        Returns statistics."""
        self._resid = self.residual()
        probs = []
        for g in groups:
            p = self.build(g, depth)
            if p is not None:
                probs.append((g, p))
        stats = {"problems": len(probs), "max_change": 0.0, "newton_iters": [], "rejected": 0,
                 "advance": 0, "retreat": 0}
        buckets: dict[tuple[int, int, int], list[Any]] = {}
        for g, p in probs:
            key = (_pow2(len(p["slots"]) + 1), _pow2(len(p["unk"])), _pow2(len(g), 1))
            buckets.setdefault(key, []).append((g, p))
        for (S, U, T), items in sorted(buckets.items()):
            per = (U * S * 500.0) if not coarse else S * 500.0
            B = int(max(1, min(64, self.s.batch_bytes // per)))
            for a in range(0, len(items), B):
                chunk = items[a : a + B]
                padded = [self.pad(p, S, U, T) for _, p in chunk]
                while len(padded) < B:
                    padded.append(padded[0])
                P = {k: np.stack([q[k] for q in padded]) for k in padded[0] if k not in ("h", "htau")}
                P["h"] = np.broadcast_to(self.h, (B, 3)).copy()
                P["htau"] = np.full(B, self.htau)
                if coarse:
                    sh, e0, e1, p0, p1 = (np.asarray(x) for x in TE.shift_batch(P, iters=self.s.newton_iters, tol=self.s.newton_tol))
                    u1 = P["x0"][np.arange(B)[:, None], P["unk"]] + sh[:, None] * P["umask"]
                    its = np.zeros(B)
                else:
                    u1, e0, e1, its, p0, p1 = (np.asarray(x) for x in TE.newton_batch(P, iters=self.s.newton_iters, tol=self.s.newton_tol))
                for b, (g, p) in enumerate(chunk):
                    nu = len(p["unk"])
                    gidx = p["slots"][p["unk"]]
                    old = self.tau[gidx].copy()
                    new = u1[b, :nu]
                    acc = self._accept(P, b, gidx, old, new, e0[b], S, U, T, padded)
                    stats["newton_iters"].append(int(its[b]))
                    if acc is None:
                        stats["rejected"] += 1
                        self.log.events["topology_reject"].append({"step": self.step, "tubes": [int(x) for x in g]})
                        continue
                    alpha, newv = acc
                    ch = np.abs(newv - old)
                    stats["max_change"] = max(stats["max_change"], float(ch.max()) if len(ch) else 0.0)
                    self.tau[gidx] = newv
                    ins = self.inside[gidx]
                    ret = (~ins) & (old < 0) & (newv >= 0)
                    adv = (~ins) & (old >= 0) & (newv < 0)
                    stats["advance"] += int(adv.sum())
                    stats["retreat"] += int(ret.sum())
                    if ret.any():
                        dterm = (p1[b] - p0[b]).tolist() if alpha == 1.0 else None
                        for v in gidx[ret]:
                            self.log.events["retreat"].append({
                                "step": self.step, "tube": int(self.owner[v]), "voxel": int(v),
                                "alpha": alpha, "dE_terms": dterm, "coarse": coarse,
                            })
                    for v in gidx[adv]:
                        self.log.events["advance"].append({"step": self.step, "tube": int(self.owner[v]), "voxel": int(v)})
                    for t in g:
                        self._resid[t] = np.log(max(self.thickness_one(t), 0.02)) - self.logmed[t]
        return stats

    def thickness_one(self, t: int) -> float:
        r = self.roots[t]
        ok = self.owner[r] >= 0
        w = np.where(ok, self.rootw[t], 0.0)
        if w.sum() <= 0:
            return np.nan
        return float(-np.sum(w * np.where(ok, self.tau[r], 0.0)) / w.sum())

    def _accept(self, P: dict[str, np.ndarray], b: int, gidx: np.ndarray, old: np.ndarray,
                new: np.ndarray, e0: float, S: int, U: int, T: int,
                padded: list[dict[str, np.ndarray]]) -> tuple[float, np.ndarray] | None:
        """Topology and energy line search on the accumulated step."""
        if K.apply_signs(self.obj3, self.shape_p, gidx, old, new, self.inside):
            return 1.0, new
        alphas = 0.5 ** np.arange(1, 11)
        one = {k: v[b : b + 1] for k, v in P.items()}
        u1 = np.zeros((1, U))
        u1[0, : len(new)] = new
        u1[0, len(new):] = P["x0"][b][P["unk"][b][len(new):]]
        es = np.asarray(TE.energy_along(one, u1, alphas))[0]
        for a, e in zip(alphas, es):
            if not e < e0:
                continue
            cand = old + a * (new - old)
            if K.apply_signs(self.obj3, self.shape_p, gidx, old, cand, self.inside):
                return float(a), cand
        return None

    def sweep(self, depth: float) -> dict[str, Any]:
        self._index()
        st = {"max_change": 0.0, "problems": 0, "newton_iters": [], "rejected": 0, "advance": 0, "retreat": 0}
        live = np.flatnonzero(np.diff(self.tvptr) > 0)
        for c in range(self.ncolour):
            tubes = live[self.colour[live] == c]
            if len(tubes) == 0:
                continue
            r = self.solve_groups([np.array([t]) for t in tubes], depth)
            for k in ("problems", "rejected", "advance", "retreat"):
                st[k] += r[k]
            st["newton_iters"] += r["newton_iters"]
            st["max_change"] = max(st["max_change"], r["max_change"])
        return st

    def sweeps(self, depth: float) -> list[dict[str, Any]]:
        out = []
        for _ in range(self.s.sweep_max):
            st = self.sweep(depth)
            clamped = self._rebuild_shell()
            self._colour()
            st["shell_clamped"] = clamped
            its = st.pop("newton_iters")
            st["newton_mean"] = float(np.mean(its)) if its else 0.0
            st["newton_max"] = int(np.max(its)) if its else 0
            out.append(st)
            if st["max_change"] < self.s.sweep_tol_mm:
                break
        return out

    def coarse(self) -> list[dict[str, Any]]:
        out = []
        self._index()
        live = np.flatnonzero(np.diff(self.tvptr) > 0)
        for lev in range(len(self.pt.coarse) - 1, -1, -1):
            cmap = self.pt.coarse[lev]
            cl = cmap[live]
            groups = {int(c): live[cl == c] for c in np.unique(cl)}
            # coarse neighbour graph from the fine one
            keys = list(groups)
            kid = {c: i for i, c in enumerate(keys)}
            nbr: list[set[int]] = [set() for _ in keys]
            for c, mem in groups.items():
                for t in mem:
                    for j in self.nb_idx[self.nb_ptr[t] : self.nb_ptr[t + 1]]:
                        cj = int(cmap[j])
                        if cj != c and cj in kid:
                            nbr[kid[c]].add(kid[cj])
            col = -np.ones(len(keys), np.int64)
            for i in np.argsort([-len(x) for x in nbr], kind="stable"):
                used = {col[j] for j in nbr[i] if col[j] >= 0}
                k = 0
                while k in used:
                    k += 1
                col[i] = k
            for k in range(int(col.max()) + 1 if len(col) else 0):
                gs = [groups[keys[i]] for i in np.flatnonzero(col == k)]
                r = self.solve_groups(gs, self.s.depth_mm, coarse=True)
                r.pop("newton_iters")
                r["level_mm"] = TP.COARSE_MM[lev]
                r["colour"] = k
                out.append(r)
            self._rebuild_shell()
            self._index()
        return out

    def retrace(self) -> dict[str, Any]:
        b = self.band()
        start = b[~self.inside[b]]
        G = K.voxel_gradient(b, self.tau, self.owner, self.strides, self.h, self.pt.same_ptr, self.pt.same_idx)
        gidx_of = -np.ones(self.N, np.int64)
        gidx_of[b] = np.arange(len(b))
        endp, status = K.trace_down(start, self.shape_p, gidx_of, G, self.owner, self.inside,
                                    self.pt.same_ptr, self.pt.same_idx, self.s.trace_step, self.s.trace_max)
        ok = status == 0
        d, j = self.ftree.query((endp[ok] - PAD) * self.site.vox)
        newp = self.alias[self.pt.vpatch[self.fv[j]]]
        far = d > 2 * self.htau
        idx_ok = np.flatnonzero(ok)
        fail = np.zeros(len(start), np.int64)
        fail[~ok] = status[~ok]
        fail[idx_ok[far]] = 5
        good = idx_ok[~far]
        changed = int(np.sum(self.owner[start[good]] != newp[~far]))
        self.owner[start[good]] = newp[~far]
        for k in np.flatnonzero(fail > 0):
            self.log.events["trace_fail"].append({"step": self.step, "voxel": int(start[k]), "tube": int(self.owner[start[k]]), "status": int(fail[k])})
        self._merge_thin()
        self._rebuild_shell()
        self._sync_obj()
        self._colour()
        return {"traced": int(len(start)), "failed": int((fail > 0).sum()), "reassigned": changed,
                "fail_by_status": {int(s): int((fail == s).sum()) for s in np.unique(fail[fail > 0])}}

    # ------------------------------------------------------------ energy

    def global_energy(self) -> dict[str, float]:
        """Total energy by term (all band voxels, all tubes)."""
        live = np.flatnonzero(np.diff(self.tvptr) > 0) if hasattr(self, "tvptr") else np.arange(self.ntube)
        self._index()
        self._resid = self.residual()
        tot = dict.fromkeys(TE.TERMS, 0.0)
        # voxel terms by chunks of tubes (each eval voxel once: eval = own voxels)
        b = self.band()
        evmask = np.zeros(self.N, bool)
        evmask[b] = True
        chunks = np.array_split(live, max(1, len(live) // 400))
        for ch in chunks:
            p = self.build(ch, np.inf)
            if p is None:
                continue
            p["unk"] = p["unk"][:1]
            own = np.isin(self.owner[p["slots"]], ch) & evmask[p["slots"]]
            p["evalw"] = own.astype(np.float64)
            S = _pow2(len(p["slots"]) + 1)
            q = self.pad(p, S, 16, _pow2(len(ch), 1))
            parts = np.asarray(TE.global_parts(q))
            for i, k in enumerate(TE.TERMS[:5]):
                tot[k] += float(parts[i].sum())
        th = np.maximum(self.thickness(), 0.02)
        r = np.log(th) - self.logmed
        ok = np.isfinite(r) & (self.alias == np.arange(self.ntube))
        tot["thick"] = float(np.sum((np.log(th) + r**2 / (2 * TE.THICK_SD**2))[ok]))
        i = np.repeat(np.arange(self.ntube), np.diff(self.tptr))
        j = self.tnb
        m = ok[i] & ok[j] & (i < j)
        dd = r[i[m]] - r[j[m]]
        tot["tsmooth"] = float(np.sum((TE.NU + 1) / 2 * np.log1p(dd**2 / (TE.NU * self.tsd[m] ** 2))))
        tot["total"] = sum(tot[k] for k in TE.TERMS)
        return tot

    # ------------------------------------------------------------ steps

    def contacts_now(self) -> np.ndarray:
        return K.contacts(self.band(), self.tau, self.owner, self.inside, self.strides,
                          self.pt.same_ptr, self.pt.same_idx)

    def snapshot(self, path: Any) -> None:
        """Band-only τ and owner (flat indices of the unpadded crop)."""
        if path is None:
            return
        b = self.band()
        xyz = np.stack(np.unravel_index(b, tuple(self.shape_p)), 1) - PAD
        flat = np.ravel_multi_index(xyz.T, tuple(self.shape_p - 2 * PAD))
        np.savez_compressed(path, index=flat.astype(np.int32), tau=self.tau[b].astype(np.float32),
                            owner=self.owner[b].astype(np.int32), stopped=self.stopped)

    def run(self, progress: Any = print, snap_dir: Any = None) -> None:
        def snap(name: str) -> None:
            if snap_dir is not None:
                self.snapshot(snap_dir / f"{name}.npz")

        s = self.s
        seen: set[tuple[int, int]] = set()
        self._index()
        rec0 = {"step": 0, "energy": self.global_energy(), "tissue": int(self.tissue_count().sum())}
        self.log.steps.append(rec0)
        snap("step-000")
        while self.step < s.max_steps:
            self.step += 1
            t0 = time.time()
            self._index()
            n0 = self.tissue_count()
            tau_before = self.tau.copy()
            go = ~self.stopped
            blocked = K.predict(self.obj3, self.shape_p, self.tvptr, self.tvidx, self.tau, self.inside,
                                go, self.delta, s.block_eps)
            for t in np.flatnonzero(blocked):
                self.log.events["block"].append({"step": self.step, "tube": int(t), "voxels": int(blocked[t])})
            clamped = self._rebuild_shell()
            self._colour()
            rec: dict[str, Any] = {"step": self.step, "predict_blocked": int(blocked.sum()), "shell_clamped": clamped}
            rec["sweeps"] = self.sweeps(s.depth_mm)
            if self.step % s.m_full == 0:
                rec["full_sweeps"] = self.sweeps(np.inf)
            if self.step % s.c_coarse == 0:
                rec["coarse"] = self.coarse()
            if self.step % s.r_retrace == 0:
                rec["retrace"] = self.retrace()
            self._index()
            n1 = self.tissue_count()
            live = np.diff(self.tvptr) > 0
            newly = live & (n1 <= n0) & ~self.stopped
            restart = live & (n1 > n0) & self.stopped
            for t in np.flatnonzero(newly):
                self.log.events["stop"].append({"step": self.step, "tube": int(t), "tissue": int(n1[t])})
            self.stopped = live & (n1 <= n0)
            rec["stopped"] = int(self.stopped.sum())
            rec["restarted"] = int(restart.sum())
            rec["live"] = int(live.sum())
            # change of τ beyond the shift
            both = np.isfinite(tau_before) & np.isfinite(self.tau)
            sh = np.where(go[np.maximum(self.owner, 0)], self.delta, 0.0)
            dchg = np.abs(self.tau - (tau_before - sh))[both]
            rec["beyond_shift_max"] = float(dchg.max()) if dchg.size else 0.0
            rec["beyond_shift_mean"] = float(dchg.mean()) if dchg.size else 0.0
            cs = self.contacts_now()
            newc = 0
            for i, j, v in cs:
                key = (int(min(i, j)), int(max(i, j)))
                if key not in seen:
                    seen.add(key)
                    newc += 1
                    self.log.events["contact"].append({"step": self.step, "a": key[0], "b": key[1], "voxel": int(v)})
            rec["contacts_new"] = newc
            rec["tissue"] = int(n1.sum())
            rec["energy"] = self.global_energy()
            rec["seconds"] = time.time() - t0
            moved = int(np.sum((np.sign(tau_before[both]) != np.sign(self.tau[both]))))
            rec["sign_changes"] = moved
            self.log.steps.append(rec)
            snap(f"step-{self.step:03d}")
            progress(json.dumps({k: rec[k] for k in ("step", "tissue", "stopped", "live", "contacts_new", "sign_changes", "seconds")}))
            progress("  sweeps: " + json.dumps([(x["max_change"], x["advance"], x["retreat"], x["rejected"]) for x in rec["sweeps"]]))
            if self.stopped[live].all() and moved == 0:
                break
        final = self.sweeps(np.inf)
        snap("final")
        self.log.steps.append({"step": "final", "full_sweeps": final, "energy": self.global_energy()})

    # ------------------------------------------------------------ output

    def grid(self, a: np.ndarray) -> np.ndarray:
        g = a.reshape(tuple(self.shape_p))
        return g[PAD:-PAD, PAD:-PAD, PAD:-PAD]
