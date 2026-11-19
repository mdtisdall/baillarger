"""Global sparse τ solve (first prototype), after the design accepted on
2026-11-17 and the user's choices G1-G11 (2026-11-17).

1. Domain (G1): label 42 dilated by 2 voxels (6-connected) into label 0, not
   inside the white surface; plus the inner voxels within 1.5 voxels inside
   the white surface (they carry the pull term only).
2. First ownership (G2): multi-source Dijkstra (scipy.sparse.csgraph) from
   the white-surface patches,
   26 neighbours, step cost sqrt(dᵀ M d), M = n̂n̂ᵀ + (1 + α coh)(I - n̂n̂ᵀ),
   α = 8. A 6-face between owners not in the same bank (> SAME_MM on the
   patch graph) is a contact face.
3. τ solve: one global problem; Levenberg-Marquardt (G4) with JᵀJ from the
   least-squares residuals (f and the Student-t weights frozen) and the full
   energy gradient; PCG (scipy.sparse.linalg.cg) with the column-line preconditioner (G6). The
   Jacobian is built by colouring (G5): one jvp pass per colour, each linear
   in the number of voxels. Contact faces carry the ghost τ = 0 (G7).
4. Upwind ownership (G8) along -∇τ, then the contact faces again.
5. Face position (G9): each face unit (the contact faces between one pair of
   1 mm coarse patches; user decision 2026-11-17) is tried at offsets
   -3..+3 voxels by a local LM solve; the move of -1/0/+1 with the smallest
   sum of both sides' energies is kept; the signed bank disagreement is
   (B's best offset - A's best offset) along the normal from A to B, in mm:
   positive = a gap, negative = the two sides' evidence disagrees (never
   reported as overlap).
6. Topology (G10): after each solve, the Euler number of each component of
   the object {τ < 0} cut at the contact faces against that of its inner
   voxels; a difference is an event. No repair.
The outer loop (G11) stops when no owner changes and no face moves, after
10 outer iterations, or when a state repeats (an oscillation, reported, not
damped).

Ridge form (Settings.contacts = "ridge"; the user's decisions 2026-11-18,
after the first site-1 run): a contact is not decided before the solve.
- No contact faces, no ghost values, no 5 mm relation, no face test.
- The first derivatives are the continuous upwind split
  max(D-, 0) + min(D+, 0) on each axis (global_tau_energy). (The first
  ridge version switched between D- and D+ by the lower neighbour, fixed
  per LM iteration. That energy jumps at every switch; on site 1 it
  diverged, and with the acceptance on the true energy LM could not take
  one step. User decision 2026-11-18: the continuous split.)
- The rim voxels (no neighbour on some axis, valid = 0) are fixed at τ0
  (user decision 2026-11-18: in the first ridge version a rim voxel above
  its neighbours was in no term and jumped 1.8e5 mm).
- div v counts each neighbour's v weighted by its tissue fraction f, so a
  neighbour beyond the top adds nothing.
- The stop term, the pull (convex beyond the top) and the other terms are
  unchanged.
- τ0 = -(distance to label 0), with no contact seeds; or, with
  Settings.tau0_faces, also -(distance to given faces), as a start only.
- Upwind ownership along -∇τ with no 5 mm restriction; the owners set only
  the equivolume weight (the owner patch's area) and the diagnostics.
- After each solve the ridge faces are found from τ (ridge_faces): 6-faces
  where τ falls away on both sides. A ridge face whose two voxels are in
  tissue (τ < 0) and whose two one-sided extrapolations to the face reach
  τ >= 0 is a contact face (the top reached from both sides). The other
  ridge faces are reported: "below" (both voxels in tissue, the top not
  reached at the face: columns meet in tissue) and "beyond" (a voxel at
  τ >= 0).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import global_tau_energy as GE
import global_tau_kernels as K
import global_tau_prep as TP
import jax.numpy as jnp
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as spla
from skimage.measure import euler_number

PAD = K.PAD
COLOURING = K.find_colouring(4)


@dataclass
class Settings:
    margin_vox: int = 2  # G1
    inner_vox: float = 1.5  # brief, item 1
    alpha: float = 8.0  # G2
    lm_iters: int = 30  # G4
    lm_tol: float = 1e-6  # G4: nats per unknown
    # the damping update: "x10" (÷10 on accept, ×10 on reject, G4) or
    # "nielsen" (by the gain ratio; Madsen, Nielsen and Tingleff 2004)
    lm_update: str = "x10"
    lm_lambda0: float = 1e-3  # Marquardt start (relative to diag(JᵀJ))
    lm_retries: int = 10
    #: "lm" (the original energy, LM) or "lagged" (the lagged quadratic
    #: form, global_tau_energy; a fixed Σ only): one PCG solve per outer step
    estimator: str = "lm"
    lag_iters: int = 30
    lag_tol: float = 1e-6  # max |Δτ| (mm) of an outer step
    #: a fixed top surface Σ only: the rim voxels (no neighbour and no ghost
    #: on some axis: the crop edge) stay at τ0, as in the ridge form
    rim_fixed: bool = False
    cg_tol: float = 1e-3  # G4
    cg_iters: int = 200  # G4
    line_max: int = 32  # G6: lines longer than this are split
    face_range: int = 3  # G9
    face_unk_vox: int = 4  # G9: the window's unknowns: the patch's box + 4 (+ the move)
    outer_iters: int = 10  # G11
    patch_mm: float = TP.PATCH_MM
    median_mm: float = 2.35  # F
    curv_ratio: float = 1.23  # F
    corr_mm: float = 10.0  # F
    region_mm: float = 12.0  # A
    face_test: bool = True
    contacts: str = "faces"  # "faces" (first prototype) or "ridge" (2026-11-18)
    #: ridge form only: an npz whose "faces" (minus voxel x, y, z and the axis,
    #: as result.npz) are τ = 0 seeds of τ0, a start only (user, 2026-11-18)
    tau0_faces: str = ""
    snapshots: bool = False  # global_tau_run.py: write RUN_DIR/snapshots


@dataclass
class Log:
    outer: list[dict[str, Any]] = field(default_factory=list)
    solves: list[dict[str, Any]] = field(default_factory=list)
    events: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "topology": [],
            "oscillation": [],
            "face_move": [],
            "lm_stall": [],
        }
    )


def _pow2(n: int, lo: int = 16) -> int:
    p = lo
    while p < n:
        p *= 2
    return p


class Grid:
    """Padded grid and the slot (domain voxel) index."""

    def __init__(self, domain: np.ndarray):
        self.shape = np.array(domain.shape)
        sp = self.shape + 2 * PAD
        self.shape_p = sp
        self.strides = np.array([sp[1] * sp[2], sp[2], 1], np.int64)
        dp = np.pad(domain, PAD).ravel()
        self.slot_flat = np.flatnonzero(dp)
        self.S = len(self.slot_flat)
        self.slot_of = -np.ones(int(np.prod(sp)), np.int64)
        self.slot_of[self.slot_flat] = np.arange(self.S)
        self.coords = np.stack(np.unravel_index(self.slot_flat, tuple(sp)), 1).astype(
            np.int64
        )  # padded
        m, a, b = COLOURING
        c = self.coords
        self.colour = (c[:, 0] + a * c[:, 1] + b * c[:, 2]) % m
        self.ncolour = m

    def grid(self, v: np.ndarray, fill: Any = np.nan) -> np.ndarray:
        out = np.full(tuple(self.shape_p), fill, dtype=np.result_type(v, type(fill)))
        out.reshape(-1)[self.slot_flat] = v[: self.S]
        return out[PAD:-PAD, PAD:-PAD, PAD:-PAD]

    def sub_index(self, sub: np.ndarray) -> np.ndarray:
        """slot_of for a subset of slots (local index)."""
        so = -np.ones_like(self.slot_of)
        so[self.slot_flat[sub]] = np.arange(len(sub))
        return so


def march_graph(
    G: Grid,
    h: np.ndarray,
    nhat: np.ndarray,
    coh: np.ndarray,
    alpha: float,
    seed_slot: np.ndarray,
    seed_cost: np.ndarray,
) -> sparse.csr_matrix:
    """The 26-neighbour graph of the slots with the step cost sqrt(dᵀ M d),
    M = n̂n̂ᵀ + (1 + α coh)(I - n̂n̂ᵀ), the mean of the two ends' costs (G2);
    plus one virtual node per seed (index S + k) with an edge of its cost to
    its slot."""
    S = G.S
    rows, cols, w = [], [], []
    for o in K.O26[: len(K.O26) // 2]:  # each undirected edge once
        u = G.slot_of[G.slot_flat + o @ G.strides]
        m = u >= 0
        s_, u_ = np.flatnonzero(m), u[m]
        d = o * h
        L2 = float(d @ d)

        def cost(q: np.ndarray) -> np.ndarray:
            pn = nhat[q] @ d
            return np.sqrt(L2 + alpha * coh[q] * np.maximum(L2 - pn * pn, 0.0))

        rows.append(s_)
        cols.append(u_)
        w.append(0.5 * (cost(s_) + cost(u_)))
    n = S + len(seed_slot)
    rows.append(S + np.arange(len(seed_slot)))
    cols.append(seed_slot)
    w.append(np.maximum(seed_cost, 1e-9))
    R, C, W = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    A = sparse.coo_matrix((W, (R, C)), shape=(n, n)).tocsr()
    return A.maximum(A.T).tocsr()


def stencil_jacobian(
    x: Any,
    P: dict[str, Any],
    g0: Any,
    fw: Any,
    colour: np.ndarray,
    ncolour: int,
    col: np.ndarray,
    unk: np.ndarray,
    S: int,
    jvp: Any = None,
) -> np.ndarray:
    """The Jacobian of the voxel residuals in stencil storage (S, NOFF, NRES)
    by colouring (G5): one jvp per colour; two unknowns of one colour are more
    than L1 4 apart, so they never share a residual. jvp(t) -> J t (S1, k)
    replaces the residuals of the original energy (the lagged form)."""
    if jvp is None:
        jvp = lambda t: GE.j_residual_jvp(x, P, g0, fw, t)  # noqa: E731
    Jst: np.ndarray | None = None
    S1 = len(unk)
    for c in range(ncolour):
        t = np.zeros(S1)
        t[:S] = (colour == c) & unk[:S]
        if not t.any():
            continue
        jv = np.asarray(jvp(jnp.asarray(t)))[:S]
        if Jst is None:
            Jst = np.zeros((S, K.NOFF, jv.shape[1]))
        K.scatter_colour(Jst, jv, col, colour, c, unk[:S])
    if Jst is None:
        Jst = np.zeros((S, K.NOFF, GE.NRES))
    return Jst


def upwind_stencil(x: np.ndarray, nbr: np.ndarray, nbv: np.ndarray) -> np.ndarray:
    """Which side τ falls to, from τ = x (S1,), for the ridge test only:
    up (S1, 3), -1 / +1 = the minus / plus neighbour is the lower one and is
    below τ (both neighbours real; ties: minus), 0 otherwise."""
    up = np.zeros((len(x), 3), np.int32)
    for a in range(3):
        m, p = nbr[:, a, 0], nbr[:, a, 1]
        both = nbv[:, a, 0] & nbv[:, a, 1]
        tm, tp = x[m], x[p]
        lm = tm <= tp
        up[:, a] = np.where(
            both & lm & (tm < x), -1, np.where(both & ~lm & (tp < x), 1, 0)
        )
    return up


RIDGE_CLASSES = ("contact", "below", "beyond")


def ridge_faces(
    x: np.ndarray,
    nbr: np.ndarray,
    nbv: np.ndarray,
    up: np.ndarray,
    S: int,
    grad: np.ndarray,
) -> dict[str, np.ndarray]:
    """6-faces between slots s (minus side) and u (plus side) where τ falls
    away on both sides: s uses its minus neighbour (up = -1) and u its plus
    neighbour (up = +1), and the face's axis is the largest component of
    the upwind gradient grad (S, 3) at both s and u (so a face between two
    voxels of one level set, where τ differs by rounding only, is not a
    ridge). Each side's τ is extrapolated half a voxel to the
    face with its own one-sided slope. Classes: "contact" (both τ < 0, both
    extrapolations >= 0), "below" (both τ < 0, an extrapolation < 0),
    "beyond" (a voxel at τ >= 0). Each class is (n, 3): s, u, axis."""
    out: dict[str, list[np.ndarray]] = {k: [] for k in RIDGE_CLASSES}
    ss = np.arange(S)
    for a in range(3):
        u = nbr[:S, a, 1]
        dom = np.argmax(np.abs(grad[:S]), 1) == a
        r = nbv[:S, a, 1] & (up[:S, a] == -1) & (up[u, a] == 1) & dom
        r &= dom[np.minimum(u, S - 1)]
        s_, u_ = ss[r], u[r]
        ts, tu = x[s_], x[u_]
        es = ts + 0.5 * (ts - x[nbr[s_, a, 0]])
        eu = tu + 0.5 * (tu - x[nbr[u_, a, 1]])
        tis = (ts < 0) & (tu < 0)
        reach = (es >= 0) & (eu >= 0)
        f = np.c_[s_, u_, np.full(len(s_), a)]
        out["contact"].append(f[tis & reach])
        out["below"].append(f[tis & ~reach])
        out["beyond"].append(f[~tis])
    return {k: np.concatenate(v).astype(np.int64) for k, v in out.items()}


def make_domain(site: TP.Site, s: Settings) -> tuple[np.ndarray, np.ndarray]:
    lab = site.labels
    ctx = lab == TP.CORTEX
    margin = ndi.binary_dilation(ctx, iterations=s.margin_vox) & (lab == TP.OUTSIDE)
    tissue = (ctx | margin) & ~site.inside
    hm = float(site.vox.mean())
    inner = site.inside & np.isfinite(site.dist) & (site.dist >= -s.inner_vox * hm)
    return tissue | inner, inner


@dataclass
class Problem:
    """A slot set (global ids sub), its local arrays and the JAX dict P."""

    sub: np.ndarray
    slot_of: np.ndarray
    S: int
    S1: int
    unk: np.ndarray  # (S1,) bool
    P: dict[str, Any]
    col: np.ndarray
    colour: np.ndarray
    coords: np.ndarray
    slot_flat: np.ndarray
    faces: np.ndarray  # local
    patch_ids: np.ndarray  # global patch id of each patch row
    pair_ids: np.ndarray
    nbr: np.ndarray | None = None  # (S1, 3, 2), numpy copies for the ridge stencil
    nbv: np.ndarray | None = None


@dataclass
class SigmaTop:
    """A fixed top surface Σ for the τ solve (T3, docs/plans/top-surface.md),
    on the site's grid: the domain (voxel centres between the white surface
    and Σ, plus the inner voxels), the inner voxels, the axis sides that
    cross Σ before the neighbour's centre (ghost, (X, Y, Z, 3, 2)) with the
    crossing's distance as a fraction of h (gdist, 0 < θ ≤ 1), and τ0."""

    domain: np.ndarray
    inner: np.ndarray
    ghost: np.ndarray
    gdist: np.ndarray
    tau0: np.ndarray


class GlobalTau:
    def __init__(
        self,
        site: TP.Site,
        pt: TP.Patches,
        s: Settings,
        centre_vox: np.ndarray,
        sigma: SigmaTop | None = None,
    ):
        """With sigma (T3): the domain, τ = 0 at the crossings of Σ (ghost
        values at θh), no stop term (Σ is fixed), no contact faces from the
        owners, and τ0 from sigma."""
        self.site, self.pt, self.s = site, pt, s
        self.sigma = sigma
        self.log = Log()
        self.progress: Any = lambda m: None
        #: when set, a snapshot after each LM iteration of the global solves
        #: and after each ownership update (snapshot())
        self.snap_dir: Path | None = None
        self._snap_n = 0
        t0 = time.time()
        if sigma is None:
            self.domain, inner = make_domain(site, s)
        else:
            self.domain, inner = sigma.domain, sigma.inner
        self.G = G = Grid(self.domain)
        self.h = site.vox.astype(np.float64)
        self.htau = float(self.h.mean())
        X = tuple(G.coords.T - PAD)
        self.labels = site.labels[X].astype(np.int64)
        self.inner = inner[X]
        c = np.zeros(G.S)
        c[self.labels == TP.OUTSIDE] = GE.LOG_ODDS
        c[self.labels == TP.CORTEX] = -GE.LOG_ODDS
        c[self.inner] = 0.0
        X0 = tuple(G.coords.T - PAD)
        if sigma is not None:
            c[:] = 0.0  # Σ is fixed: the stop term does not depend on τ
            self.sigma_ghost: np.ndarray | None = sigma.ghost[X0]
            self.sigma_gdist: np.ndarray | None = sigma.gdist[X0]
        else:
            self.sigma_ghost = self.sigma_gdist = None
        self.c = c
        self.coh = site.coh[X].astype(np.float64)
        self.nhat = site.nhat[X].astype(np.float64)
        self.ntube = len(pt.centre)
        self.area = pt.area.copy()
        # roots (thickness) of the patches: 8 corners of the centre vertex
        cen_vox = site.v[pt.centre] + PAD
        lo = np.floor(cen_vox).astype(np.int64)
        fr = cen_vox - lo
        self.rslot = np.zeros((self.ntube, 8), np.int64)
        self.rw = np.zeros((self.ntube, 8))
        for k in range(8):
            b = np.array([(k >> 2) & 1, (k >> 1) & 1, k & 1])
            q = (lo + b) @ G.strides
            self.rslot[:, k] = G.slot_of[q]
            self.rw[:, k] = np.prod(np.where(b == 1, fr, 1 - fr), 1)
        self.rw[self.rslot < 0] = 0.0
        tot = self.rw.sum(1)
        self.hasroot = tot > 1e-9
        self.rw[self.hasroot] /= tot[self.hasroot, None]
        vmm = site.v * site.vox
        self.cen_mm = vmm[pt.centre]
        half = s.region_mm / 2
        self.in_region = np.all(
            np.abs((site.v[pt.centre] - centre_vox) * site.vox) <= half, axis=1
        )
        Hc = site.H[pt.centre]
        h10, h90 = np.percentile(Hc[self.in_region], [10, 90])
        self.beta = float(np.log(s.curv_ratio) / max(h90 - h10, 1e-9))
        self.logmed = np.log(s.median_mm) + self.beta * Hc
        adj = pt.adj
        d = np.linalg.norm(self.cen_mm[adj[:, 0]] - self.cen_mm[adj[:, 1]], axis=1)
        self.pairs = adj
        self.psd = GE.THICK_SD * np.sqrt(2 * (1 - np.exp(-d / s.corr_mm)))
        self.same_ptr, self.same_idx = pt.same_ptr, pt.same_idx.astype(np.int64)
        self.timing = {"setup_s": time.time() - t0}
        # first ownership (G2)
        t1 = time.time()
        self.march_cost, self.owner = self.first_ownership()
        self.owner_march = self.owner.copy()
        self.timing["march_s"] = time.time() - t1
        self.ridge = s.contacts == "ridge"
        self.ridges: dict[str, np.ndarray] = {
            k: np.zeros((0, 3), np.int64) for k in RIDGE_CLASSES
        }
        if sigma is not None:
            self.faces = np.zeros((0, 3), np.int64)
            self.tau = sigma.tau0[X0].astype(np.float64)
        elif self.ridge:
            self.faces = np.zeros((0, 3), np.int64)
            seeds = self.seed_faces(s.tau0_faces) if s.tau0_faces else self.faces
            self.tau = self.tau_init(seeds)
        else:
            self.faces = self.contact_faces(self.owner)
            self.tau = self.tau_init(self.faces)
        self.n_seed_faces = len(seeds) if self.ridge else 0
        self.tau0 = self.tau.copy()

    # ------------------------------------------------------------ ownership

    def first_ownership(self) -> tuple[np.ndarray, np.ndarray]:
        """Multi-source Dijkstra (scipy.sparse.csgraph) from the facing
        vertices of the patches; returns the arrival cost (mm) and the owner
        patch of each slot (-1 where not reached)."""
        G, site, pt = self.G, self.site, self.pt
        fv = np.flatnonzero(pt.vpatch >= 0)
        v = site.v[fv] + PAD
        lo = np.floor(v).astype(np.int64)
        ss, cs, os_ = [], [], []
        for k in range(8):
            b = np.array([(k >> 2) & 1, (k >> 1) & 1, k & 1])
            q = lo + b
            ok = np.all((q >= 0) & (q < G.shape_p), 1)
            sl = np.full(len(fv), -1, np.int64)
            sl[ok] = G.slot_of[q[ok] @ G.strides]
            m = sl >= 0
            ss.append(sl[m])
            cs.append(np.linalg.norm((q[m] - v[m]) * self.h, axis=1))
            os_.append(pt.vpatch[fv[m]])
        seed_slot, seed_cost, seed_owner = (
            np.concatenate(ss),
            np.concatenate(cs),
            np.concatenate(os_),
        )
        A = march_graph(
            G, self.h, self.nhat, self.coh, self.s.alpha, seed_slot, seed_cost
        )
        src = G.S + np.arange(len(seed_slot))
        dist, _, source = csgraph.dijkstra(
            A, directed=False, indices=src, min_only=True, return_predecessors=True
        )
        owner = np.where(
            source[: G.S] >= 0, seed_owner[np.maximum(source[: G.S] - G.S, 0)], -1
        )
        return dist[: G.S], owner.astype(np.int64)

    def contact_faces(self, owner: np.ndarray) -> np.ndarray:
        G = self.G
        return K.contact_faces(
            G.slot_flat, G.slot_of, G.strides, owner, self.same_ptr, self.same_idx
        )

    def seed_faces(self, path: str) -> np.ndarray:
        """Faces (minus voxel, axis) from an npz as (minus slot, plus slot,
        axis); faces with a voxel outside the domain are dropped."""
        G = self.G
        f = np.load(path)["faces"].astype(np.int64)
        cm = f[:, :3] + PAD
        cp = cm + np.eye(3, dtype=np.int64)[f[:, 3]]
        sm, sp = G.slot_of[cm @ G.strides], G.slot_of[cp @ G.strides]
        ok = (sm >= 0) & (sp >= 0)
        return np.c_[sm[ok], sp[ok], f[ok, 3]]

    def same(self, a: int, b: int) -> bool:
        return bool(K.same_bank(a, b, self.same_ptr, self.same_idx))

    def tau_init(self, faces: np.ndarray) -> np.ndarray:
        """G3: τ0 = -(distance to the nearest top seed) - with label-0 voxels
        and the contact faces (τ = 0 on the face) as seeds; > 0 in label 0."""
        G, h = self.G, self.h
        shape = tuple(G.shape)
        l0 = self.site.labels == TP.OUTSIDE
        d_l0 = (
            ndi.distance_transform_edt(~l0, sampling=h)
            if l0.any()
            else np.full(shape, 1e3)
        )
        notl0 = self.domain & ~l0
        d_in = (
            ndi.distance_transform_edt(~notl0, sampling=h)
            if notl0.any()
            else np.full(shape, 1e3)
        )
        hm = 0.5 * self.htau
        X = tuple(G.coords.T - PAD)
        top = d_l0[X] - hm
        if len(faces):
            fine = np.ones(tuple(2 * G.shape - 1), bool)
            cm = G.coords[faces[:, 0]] - PAD
            cp = G.coords[faces[:, 1]] - PAD
            fine[tuple((cm + cp).T)] = False
            d_f = ndi.distance_transform_edt(fine, sampling=h / 2)
            top = np.minimum(top, d_f[tuple(2 * (G.coords.T - PAD))])
        tau = -top
        is0 = self.labels == TP.OUTSIDE
        tau[is0] = d_in[X][is0] - hm
        return tau

    # ------------------------------------------------------------ problems

    def build(
        self,
        sub: np.ndarray,
        unk_g: np.ndarray,
        eval_g: np.ndarray,
        owner: np.ndarray,
        faces_g: np.ndarray | None = None,
    ) -> Problem:
        """Problem on the slots sub (sorted global ids); unk_g / eval_g are
        global masks; faces from the owners of the sub (local)."""
        G = self.G
        so = G.sub_index(sub) if len(sub) < G.S else G.slot_of
        sf = G.slot_flat[sub]
        own = owner[sub]
        if self.ridge or self.sigma is not None:
            faces = np.zeros((0, 3), np.int64)
        else:
            faces = K.contact_faces(
                sf, so, G.strides, own, self.same_ptr, self.same_idx
            )
        S = len(sub)
        S1 = _pow2(S + 1)
        nbr, nbv, gh = K.neighbours(sf, so, G.strides, faces)

        def padn(a: np.ndarray, fill: Any) -> np.ndarray:
            out = np.full((S1, *a.shape[1:]), fill, dtype=a.dtype)
            out[:S] = a
            return out

        nbr_p = padn(nbr[:S], S)
        nbr_p[S:] = S
        nbv_p = padn(nbv[:S], False)
        gh_p = padn(gh[:S], False)
        gd_p = np.full((S1, 3, 2), 0.5)
        if self.sigma_ghost is not None:  # fixed top surface Σ (T3)
            sg, sd = self.sigma_ghost[sub], self.sigma_gdist[sub]
            gh_p[:S] |= sg
            nbv_p[:S] &= ~sg  # a neighbour across Σ is not a neighbour
            gd_p[:S] = np.where(sg, sd, gd_p[:S])
        valid = np.all((nbv_p | gh_p).any(2), 1)
        area_o = np.where(own >= 0, self.area[np.maximum(own, 0)], np.inf)
        wq = GE.W_EQ / (2 * area_o)
        # patches whose roots are all in the sub
        loc = -np.ones(G.S + 1, np.int64)
        loc[sub] = np.arange(S)
        rs = np.where(self.rslot >= 0, loc[np.maximum(self.rslot, 0)], -1)
        rs = np.where(self.rslot >= 0, rs, S)
        okp = self.hasroot & np.all((rs >= 0) | (self.rw == 0), 1)
        pid = np.flatnonzero(okp)
        T = _pow2(len(pid) + 1)
        rslot = np.full((T, 8), S, np.int64)
        rw = np.zeros((T, 8))
        rslot[: len(pid)] = np.where(rs[pid] >= 0, rs[pid], S)
        rw[: len(pid)] = self.rw[pid]
        hasroot = np.zeros(T)
        hasroot[: len(pid)] = 1.0
        logmed = np.zeros(T)
        logmed[: len(pid)] = self.logmed[pid]
        prow = -np.ones(self.ntube, np.int64)
        prow[pid] = np.arange(len(pid))
        pr = prow[self.pairs]
        qok = np.all(pr >= 0, 1)
        qid = np.flatnonzero(qok)
        Q = _pow2(len(qid) + 1)
        pi = np.zeros(Q, np.int64)
        pj = np.zeros(Q, np.int64)
        psd = np.ones(Q)
        pm = np.zeros(Q)
        pi[: len(qid)], pj[: len(qid)] = pr[qid, 0], pr[qid, 1]
        psd[: len(qid)] = self.psd[qid]
        pm[: len(qid)] = 1.0
        P = {
            "nb": nbr_p,
            "nbv": nbv_p,
            "ridge": np.float64(1.0 if self.ridge else 0.0),
            "ghost": gh_p,
            "gdist": gd_p,
            "h": self.h,
            "htau": self.htau,
            "valid": valid.astype(np.float64),
            "c": padn(self.c[sub], 0.0),
            "coh": padn(self.coh[sub], 0.0),
            "nhat": padn(self.nhat[sub], 0.0),
            "inner": padn(self.inner[sub].astype(np.float64), 1.0),
            "wq": padn(wq, 0.0),
            "evalw": padn(eval_g[sub].astype(np.float64), 0.0),
            "rslot": rslot,
            "rw": rw,
            "hasroot": hasroot,
            "logmed": logmed,
            "pi": pi,
            "pj": pj,
            "psd": psd,
            "pm": pm,
        }
        P = {k: jnp.asarray(v) for k, v in P.items()}
        unk = np.zeros(S1, bool)
        unk[:S] = unk_g[sub]
        col = K.stencil_cols(sf, so, G.strides, K.OFFS)
        return Problem(
            sub,
            so,
            S,
            S1,
            unk,
            P,
            col,
            G.colour[sub],
            G.coords[sub],
            sf,
            faces,
            pid,
            qid,
            nbr_p,
            nbv_p,
        )

    def find_ridges(self, pb: Problem, x: np.ndarray) -> dict[str, np.ndarray]:
        assert pb.nbr is not None and pb.nbv is not None
        up = upwind_stencil(x, pb.nbr, pb.nbv)
        g = np.asarray(GE.gradient(jnp.asarray(x), pb.P))
        return ridge_faces(x, pb.nbr, pb.nbv, up, pb.S, g)

    def _snap_faces(self, pb: Problem, x: np.ndarray) -> np.ndarray:
        return self.find_ridges(pb, x)["contact"] if self.ridge else pb.faces

    # ------------------------------------------------------------ LM + PCG

    def _patch_jac(
        self, pb: Problem, x: np.ndarray, tw: np.ndarray, t0: np.ndarray | None = None
    ) -> sparse.csr_matrix:
        """With t0 (the lagged form): d r / d τ = -rw / t0 for every patch."""
        P = pb.P
        rslot = np.asarray(P["rslot"])
        rw = np.asarray(P["rw"])
        hr = np.asarray(P["hasroot"])
        t = -np.sum(rw * x[rslot], 1) if t0 is None else t0
        live = t > 0.02 if t0 is None else np.ones(len(t), bool)
        dr = np.where(live[:, None], -rw / np.maximum(t, 0.02)[:, None], 0.0)  # (T, 8)
        T = len(t)
        w_th = hr / (np.sqrt(2.0) * GE.THICK_SD)
        rows = [np.repeat(np.arange(T), 8)]
        cols = [rslot.ravel()]
        vals = [(w_th[:, None] * dr).ravel()]
        pi, pj, pm = np.asarray(P["pi"]), np.asarray(P["pj"]), np.asarray(P["pm"])
        wq = np.sqrt(pm * tw / 2)
        Q = len(pi)
        r0 = T + np.arange(Q)
        rows += [np.repeat(r0, 8), np.repeat(r0, 8)]
        cols += [rslot[pi].ravel(), rslot[pj].ravel()]
        vals += [(wq[:, None] * dr[pi]).ravel(), (-wq[:, None] * dr[pj]).ravel()]
        R, C, V = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        ok = pb.unk[C] & (V != 0)
        return sparse.csr_matrix((V[ok], (R[ok], C[ok])), shape=(T + Q, pb.S1))

    def lm_solve(
        self, pb: Problem, x: np.ndarray, g0: Any, tag: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        s = self.s
        unk = pb.unk
        nunk = max(int(unk.sum()), 1)
        P = pb.P
        lam = s.lm_lambda0
        nu = 2.0  # Nielsen's damping update (Madsen, Nielsen and Tingleff 2004, eq. 3.16)
        x = x.copy()
        x[~unk] = np.where(np.arange(pb.S1)[~unk] < pb.S, x[~unk], 0.0)
        # with a fixed Σ (T3) every domain voxel is inside Σ, so τ ≤ 0 is a
        # bound: a projected LM (the voxels at the bound whose gradient points
        # beyond it are held for that iteration; each step is clipped to it)
        bounded = self.sigma is not None
        if bounded:
            x = np.where(unk, np.minimum(x, 0.0), x)
        olut = -np.ones((5, 5, 5), np.int64)
        for o, (a, b, c) in enumerate(K.OFFS):
            olut[a + 2, b + 2, c + 2] = o
        its: list[dict[str, Any]] = []
        e_first = None
        t_solve = time.time()
        snap = self.snap_dir is not None and "/" not in tag and pb.S == self.G.S
        for it in range(s.lm_iters):
            ti = time.time()
            xj = jnp.asarray(x)
            e, grad = GE.j_energy_grad(xj, P, g0)
            e = float(e)
            if e_first is None:
                e_first = e
                if snap:
                    self.snapshot(f"{tag}-lm00", x, self.owner, self._snap_faces(pb, x),
                                  {"tag": tag, "lm_iter": 0, "energy": e})
            grad = np.where(unk, np.asarray(grad), 0.0)
            free = unk & ~((x >= 0.0) & (grad < 0.0)) if bounded else unk
            fw, tw = GE.j_weights(xj, P)
            Jst = stencil_jacobian(
                xj, P, g0, fw, pb.colour, self.G.ncolour, pb.col, free, pb.S
            )
            Jp = self._patch_jac(pb, x, np.asarray(tw))
            t_jac = time.time() - ti
            dvox = np.zeros(pb.S1)
            dvox[: pb.S] = K.jst_diag(Jst, pb.col, pb.S)
            dp = np.asarray(Jp.multiply(Jp).sum(0)).ravel()
            D = np.where(free, dvox + dp, 0.0)
            D = np.maximum(D, 1e-12 * max(D.max(), 1e-12))
            ghat = np.asarray(GE.j_g0(xj, P))[: pb.S]
            nbr = np.asarray(P["nb"])[: pb.S + 1].copy()
            nbr[nbr >= pb.S] = pb.S
            nbv = np.asarray(P["nbv"])[: pb.S + 1].copy()
            nbv[pb.S] = False
            nbv[: pb.S] &= free[nbr[: pb.S]]
            lptr, lsl = K.column_lines(nbr, nbv, ghat, s.line_max)
            keep = free[lsl]
            # keep only freenown slots in the lines (fixed slots: own singleton lines dropped)
            lens = np.diff(lptr)
            cid = np.repeat(np.arange(len(lens)), lens)
            lsl2 = lsl[keep]
            cid2 = cid[keep]
            cnt = np.bincount(cid2, minlength=len(lens))
            lptr2 = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
            Jst_s = Jst

            def Av(v: np.ndarray, lam_: float) -> np.ndarray:
                vv = np.where(free, v, 0.0)
                r = K.jst_apply(Jst_s, pb.col, vv[: pb.S])
                out = np.zeros(pb.S1)
                out[: pb.S] = K.jst_apply_t(Jst_s, pb.col, r, pb.S)
                out += Jp.T @ (Jp @ vv)
                out += lam_ * D * vv
                return np.where(free, out, 0.0)

            accepted = False
            tries = []
            for _ in range(s.lm_retries):
                tf = time.time()
                diag_add = np.zeros(pb.S)
                diag_add[:] = (lam * D + dp)[: pb.S]
                B, bptr = K.line_blocks(
                    lptr2,
                    lsl2,
                    Jst_s,
                    pb.col,
                    pb.slot_flat,
                    pb.slot_of,
                    self.G.strides,
                    K.OFFS,
                    olut,
                    pb.coords,
                    diag_add,
                )
                t_fac = time.time() - tf
                tc = time.time()
                # Gauss-Newton: E = Σ r² (+ the log t term), ∇E = 2 Jᵀr and
                # the Hessian ≈ 2 JᵀJ, so (JᵀJ + λD) dx = -∇E / 2
                b = -0.5 * grad
                n_it = [0]
                A_op = spla.LinearOperator(
                    (pb.S1, pb.S1), matvec=lambda v, lam_=lam: Av(v, lam_), dtype=float
                )

                def prec(
                    r: np.ndarray, B_: np.ndarray = B, bp_: np.ndarray = bptr
                ) -> np.ndarray:
                    z = np.zeros(pb.S1)
                    z[: pb.S] = K.line_solve(
                        lptr2, lsl2, B_, bp_, np.where(free, r, 0.0)[: pb.S]
                    )
                    return np.where(free, z, 0.0)

                M_op = spla.LinearOperator((pb.S1, pb.S1), matvec=prec, dtype=float)
                dx, _ = spla.cg(
                    A_op,
                    b,
                    rtol=s.cg_tol,
                    atol=0.0,
                    maxiter=s.cg_iters,
                    M=M_op,
                    callback=lambda xk: n_it.__setitem__(0, n_it[0] + 1),
                )
                ncg = n_it[0]
                r = b - Av(dx, lam)
                bn = float(np.linalg.norm(b))
                t_cg = time.time() - tc
                rel = float(np.linalg.norm(r) / max(bn, 1e-300))
                xn = x + np.where(free, dx, 0.0)
                if bounded:  # projected step: τ ≤ 0 inside a fixed Σ
                    xn = np.where(unk, np.minimum(xn, 0.0), xn)
                e1 = float(GE.j_energy(jnp.asarray(xn), P, g0))
                tries.append(
                    {
                        "lambda": lam,
                        "cg_iters": ncg,
                        "cg_rel": rel,
                        "cg_s": t_cg,
                        "factor_s": t_fac,
                        "e_trial": e1,
                    }
                )
                self.log.solves.append(
                    {
                        "tag": tag,
                        "lm_iter": it,
                        "lambda": lam,
                        "cg_iters": ncg,
                        "cg_rel": rel,
                        "cg_s": t_cg,
                        "factor_s": t_fac,
                        "n_unk": nunk,
                        "accepted": e1 < e,
                    }
                )
                # gain ratio: actual / predicted decrease of the model
                # the model L(dx) = E + ∇E·dx + dxᵀJᵀJdx, so with
                # (JᵀJ + λD) dx = b = -∇E / 2: L(0) - L(dx) = dx·(λ D dx + b)
                dxf = np.where(free, dx, 0.0)
                pred = float(dxf @ (lam * D * dxf + b))
                rho = (e - e1) / pred if pred > 0 else -1.0
                tries[-1]["rho"] = rho
                if e1 < e:
                    accepted = True
                    x = xn
                    if s.lm_update == "nielsen":
                        lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), 1e-12)
                        nu = 2.0
                    else:
                        lam = max(lam / 10, 1e-12)
                    break
                if s.lm_update == "nielsen":
                    lam *= nu
                    nu *= 2.0
                else:
                    lam *= 10
            its.append(
                {
                    "iter": it,
                    "e": e,
                    "e_new": tries[-1]["e_trial"] if accepted else e,
                    "jac_s": t_jac,
                    "lines": int(len(lptr2) - 1),
                    "tries": tries,
                    "s": time.time() - ti,
                }
            )
            if not accepted:
                self.log.events["lm_stall"].append({"tag": tag, "iter": it, "e": e})
                break
            if snap:
                self.snapshot(f"{tag}-lm{it + 1:02d}", x, self.owner, self._snap_faces(pb, x),
                              {"tag": tag, "lm_iter": it + 1, "energy": tries[-1]["e_trial"], "lambda": lam})
            if e - tries[-1]["e_trial"] < s.lm_tol * nunk:
                break
        e_end = float(GE.j_energy(jnp.asarray(x), P, g0))
        return x, {
            "tag": tag,
            "iters": its,
            "e0": e_first,
            "e1": e_end,
            "n_unk": nunk,
            "s": time.time() - t_solve,
        }

    def lagged_solve(
        self, pb: Problem, x: np.ndarray, g0: Any, tag: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """The lagged quadratic form: each outer step freezes ĝ0, s0, P0, t0
        and tw0 at x (GE.lag_state), and minimises the quadratic model Q by
        one PCG solve of (JᵀJ) dx = -∇Q / 2 (J is exact: the residuals are
        affine in x) with the column-line preconditioner; then τ ≤ 0 by
        projection. Each step logs the original energy (g0 as in lm_solve)
        and Q before and after."""
        s = self.s
        unk = pb.unk
        P = pb.P
        x = x.copy()
        x[~unk] = np.where(np.arange(pb.S1)[~unk] < pb.S, x[~unk], 0.0)
        x = np.where(unk, np.minimum(x, 0.0), x)
        olut = -np.ones((5, 5, 5), np.int64)
        for o, (a, b, c) in enumerate(K.OFFS):
            olut[a + 2, b + 2, c + 2] = o
        its: list[dict[str, Any]] = []
        t_solve = time.time()
        e_first = float(GE.j_energy(jnp.asarray(x), P, g0))
        for it in range(s.lag_iters):
            ti = time.time()
            xj = jnp.asarray(x)
            L = GE.j_lag_state(xj, P)
            q0, gq = GE.j_lagged_energy_grad(xj, P, L)
            gq = np.where(unk, np.asarray(gq), 0.0)
            free = unk & ~((x >= 0.0) & (gq < 0.0))
            Jst = stencil_jacobian(
                xj, P, g0, None, pb.colour, self.G.ncolour, pb.col, free, pb.S,
                jvp=lambda t, xj=xj, L=L: GE.j_lagged_jvp(xj, P, L, t),
            )
            Jp = self._patch_jac(pb, x, np.asarray(L["tw"]), np.asarray(L["t"]))
            t_jac = time.time() - ti
            dvox = np.zeros(pb.S1)
            dvox[: pb.S] = K.jst_diag(Jst, pb.col, pb.S)
            dp = np.asarray(Jp.multiply(Jp).sum(0)).ravel()
            D = np.where(free, dvox + dp, 0.0)
            reg = 1e-12 * max(D.max(), 1e-12)  # only against an exact null space
            ghat = np.asarray(L["gh"])[: pb.S]
            nbr = np.asarray(P["nb"])[: pb.S + 1].copy()
            nbr[nbr >= pb.S] = pb.S
            nbv = np.asarray(P["nbv"])[: pb.S + 1].copy()
            nbv[pb.S] = False
            nbv[: pb.S] &= free[nbr[: pb.S]]
            lptr, lsl = K.column_lines(nbr, nbv, ghat, s.line_max)
            keep = free[lsl]
            lens = np.diff(lptr)
            cid = np.repeat(np.arange(len(lens)), lens)
            lsl2 = lsl[keep]
            cnt = np.bincount(cid[keep], minlength=len(lens))
            lptr2 = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)

            def Av(v: np.ndarray, Jst: np.ndarray = Jst, Jp: Any = Jp,
                   free: np.ndarray = free) -> np.ndarray:
                vv = np.where(free, v, 0.0)
                out = np.zeros(pb.S1)
                out[: pb.S] = K.jst_apply_t(Jst, pb.col, K.jst_apply(Jst, pb.col, vv[: pb.S]), pb.S)
                out += Jp.T @ (Jp @ vv) + reg * vv
                return np.where(free, out, 0.0)

            tf = time.time()
            B, bptr = K.line_blocks(
                lptr2, lsl2, Jst, pb.col, pb.slot_flat, pb.slot_of, self.G.strides,
                K.OFFS, olut, pb.coords, (dp + reg)[: pb.S],
            )
            t_fac = time.time() - tf

            def prec(r: np.ndarray, B: np.ndarray = B, bptr: np.ndarray = bptr,
                     free: np.ndarray = free) -> np.ndarray:
                z = np.zeros(pb.S1)
                z[: pb.S] = K.line_solve(lptr2, lsl2, B, bptr, np.where(free, r, 0.0)[: pb.S])
                return np.where(free, z, 0.0)

            b = -0.5 * gq
            n_it = [0]
            tc = time.time()
            dx, _ = spla.cg(
                spla.LinearOperator((pb.S1, pb.S1), matvec=Av, dtype=float),
                b, rtol=s.cg_tol, atol=0.0, maxiter=s.cg_iters,
                M=spla.LinearOperator((pb.S1, pb.S1), matvec=prec, dtype=float),
                callback=lambda xk: n_it.__setitem__(0, n_it[0] + 1),
            )
            t_cg = time.time() - tc
            rel = float(np.linalg.norm(b - Av(dx)) / max(np.linalg.norm(b), 1e-300))
            xn = np.where(unk, np.minimum(x + np.where(free, dx, 0.0), 0.0), x)
            q1 = float(GE.lagged_energy(jnp.asarray(xn), P, L))
            e1 = float(GE.j_energy(jnp.asarray(xn), P, g0))
            step = float(np.abs(xn - x).max())
            its.append({
                "iter": it, "q0": float(q0), "q1": q1, "e": e1, "step_mm": step,
                "cg_iters": n_it[0], "cg_rel": rel, "jac_s": t_jac, "factor_s": t_fac,
                "cg_s": t_cg, "s": time.time() - ti,
                "s0_below_0.1": int(np.sum(free[: pb.S] & (np.asarray(L["s"])[: pb.S] < 0.1))),
            })
            self.log.solves.append({"tag": tag, "lag_iter": it, **its[-1]})
            x = xn
            if step < s.lag_tol:
                break
        return x, {
            "tag": tag,
            "iters": its,
            "e0": e_first,
            "e1": float(GE.j_energy(jnp.asarray(x), P, g0)),
            "n_unk": int(unk.sum()),
            "s": time.time() - t_solve,
        }

    # ------------------------------------------------------------ global step

    def global_problem(self, owner: np.ndarray) -> Problem:
        allm = np.ones(self.G.S, bool)
        pb = self.build(np.arange(self.G.S), allm, allm, owner)
        if self.ridge or (self.sigma is not None and self.s.rim_fixed):
            # the rim voxels stay at τ0
            pb.unk[: pb.S] &= np.asarray(pb.P["valid"])[: pb.S] > 0
        return pb

    def solve_global(
        self, tag: str, g0: Any | None = None
    ) -> tuple[dict[str, Any], Any]:
        pb = self.global_problem(self.owner)
        x = np.zeros(pb.S1)
        x[: pb.S] = self.tau
        if g0 is None:
            g0 = GE.j_g0(jnp.asarray(x), pb.P)
        if self.s.estimator == "lagged":
            if self.sigma is None:
                raise ValueError("the lagged estimator needs a fixed top surface Σ")
            x, info = self.lagged_solve(pb, x, g0, tag)
        else:
            x, info = self.lm_solve(pb, x, g0, tag)
        self.tau = x[: pb.S].copy()
        if self.ridge:
            self.ridges = self.find_ridges(pb, x)
            self.faces = self.ridges["contact"]
        else:
            self.faces = pb.faces
        self._pb, self._g0 = pb, g0
        return info, g0

    def slot_energies(
        self, pb: Problem, x: np.ndarray, g0: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vox, thick, ts = GE.j_slot_parts(jnp.asarray(x), pb.P, g0)
        return np.asarray(vox)[:, : pb.S], np.asarray(thick), np.asarray(ts)

    # ------------------------------------------------------------ ownership update

    def upwind(self) -> np.ndarray:
        pb = self._pb
        x = np.zeros(pb.S1)
        x[: pb.S] = self.tau
        ghat = np.asarray(GE.j_g0(jnp.asarray(x), pb.P))[: pb.S]
        order = np.argsort(self.tau, kind="stable")
        G = self.G
        return K.upwind_owner(
            order,
            G.slot_flat,
            G.slot_of,
            G.strides,
            K.O26,
            self.h,
            self.tau,
            ghat,
            self.owner,
            self.inner,
            self.same_ptr,
            self.same_idx,
            not self.ridge,
        )

    # ------------------------------------------------------------ topology (G10)

    def topology(self, tag: str) -> dict[str, Any]:
        pb = self._pb
        obj = np.zeros(pb.S1, bool)
        obj[: pb.S] = self.tau < 0
        nbr = np.asarray(pb.P["nb"])[: pb.S]
        nbv = np.asarray(pb.P["nbv"])[: pb.S].copy()
        if self.ridge:  # cut at the contact faces of the ridge form
            for s_, u_, a_ in self.faces:
                nbv[s_, a_, 1] = False
                nbv[u_, a_, 0] = False
        o = obj[: pb.S]
        ss, aa, kk = np.nonzero(nbv & o[:, None, None])
        uu = nbr[ss, aa, kk]
        e = o[uu]
        A = sparse.coo_matrix(
            (np.ones(int(e.sum())), (ss[e], uu[e])), shape=(pb.S, pb.S)
        )
        _, lab0 = csgraph.connected_components(A, directed=False)
        keep = np.unique(lab0[o])
        remap = -np.ones(lab0.max() + 1, np.int64)
        remap[keep] = np.arange(len(keep))
        lab = np.where(o, remap[lab0], -1)
        n = len(keep)
        G = self.G
        cc = G.coords - PAD
        events = []
        for c in range(n):
            m = lab == c
            pts = cc[m]
            lo, hi = pts.min(0), pts.max(0) + 1
            box = np.zeros(tuple(hi - lo + 2), bool)
            q = pts - lo + 1
            box[tuple(q.T)] = True
            inn = np.zeros_like(box)
            qi = cc[m & self.inner] - lo + 1
            inn[tuple(qi.T)] = True
            chi_o = int(euler_number(box, connectivity=3))
            chi_i = int(euler_number(inn, connectivity=3)) if inn.any() else 0
            if chi_o != chi_i:
                ev = {
                    "tag": tag,
                    "component": c,
                    "voxels": int(m.sum()),
                    "inner_voxels": int(qi.shape[0]),
                    "chi_object": chi_o,
                    "chi_inner": chi_i,
                    "centroid_mm": (pts.mean(0) * self.h).tolist(),
                }
                events.append(ev)
        self.log.events["topology"].extend(events)
        return {"components": n, "events": len(events)}

    # ------------------------------------------------------------ face patches (G9)

    def face_units(
        self, owner: np.ndarray, faces: np.ndarray
    ) -> tuple[list[dict[str, np.ndarray]], int]:
        """Face units (user decision 2026-11-17, replaces G9's one-axis
        patches): all contact faces whose two owners map to the same pair of
        1 mm coarse patches (tube_prep's first coarse level). A face whose two
        owners map to one coarse patch uses the pair of fine owners. Each unit
        has a side A (the smaller key) and B; per face the A slot, the B slot,
        the axis and the step from A to B (+1 or -1)."""
        if len(faces) == 0:
            return [], 0
        crs = self.pt.coarse[0]
        om, op = owner[faces[:, 0]], owner[faces[:, 1]]
        cm, cp = crs[np.maximum(om, 0)], crs[np.maximum(op, 0)]
        fb = cm == cp
        km = np.where(fb, om, cm) + np.where(fb, 1, 0) * (1 << 40)
        kp = np.where(fb, op, cp) + np.where(fb, 1, 0) * (1 << 40)
        lo, hi = np.minimum(km, kp), np.maximum(km, kp)
        a_is_minus = km <= kp
        keys = np.c_[lo, hi]
        uk, inv = np.unique(keys, axis=0, return_inverse=True)
        units = []
        for u in range(len(uk)):
            m = inv.ravel() == u
            f = faces[m]
            am = a_is_minus[m]
            units.append(
                {
                    "a": np.where(am, f[:, 0], f[:, 1]),
                    "b": np.where(am, f[:, 1], f[:, 0]),
                    "axis": f[:, 2],
                    "step": np.where(am, 1, -1),
                    "fallback": bool(fb[m].any()),
                }
            )
        return units, int(fb.sum())

    def _moved_owner(
        self,
        owner: np.ndarray,
        un: dict[str, np.ndarray],
        k: int,
        ref_a: int,
        ref_b: int,
    ) -> np.ndarray:
        """Owners after a move of the unit by k voxels from A towards B
        (k > 0: A gains k layers of B's voxels; k < 0: B gains A's)."""
        G = self.G
        own = owner.copy()
        if k == 0:
            return own
        for sa, sb, ax, st in zip(un["a"], un["b"], un["axis"], un["step"]):
            if k > 0:
                src, cur, sg, ref = owner[sa], int(sb), int(st), ref_b
            else:
                src, cur, sg, ref = owner[sb], int(sa), -int(st), ref_a
            for _ in range(abs(k)):
                if cur < 0 or not self.same(int(owner[cur]), int(ref)):
                    break
                own[cur] = src
                cur = int(G.slot_of[G.slot_flat[cur] + sg * G.strides[ax]])
        return own

    def face_test(
        self,
        owner: np.ndarray,
        g0_global: Any,
        tag: str,
        select: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """The face test (G9) on every face unit, or on the units select."""
        faces = self.contact_faces(owner)
        units, nfb = self.face_units(owner, faces)
        G, s = self.G, self.s
        cc = G.coords - PAD
        out = []
        new_owner = owner.copy()
        ghat_all = np.asarray(g0_global)[: G.S]
        self.progress(
            f"  face units: {len(units)} ({nfb} faces use the fine-owner fallback)"
        )
        for c, un in enumerate(units):
            if select is not None and c not in select:
                continue
            ref_a = int(np.bincount(owner[un["a"]]).argmax())
            ref_b = int(np.bincount(owner[un["b"]]).argmax())
            pts = np.concatenate([cc[un["a"]], cc[un["b"]]])
            lo, hi = pts.min(0), pts.max(0)
            r_u = s.face_unk_vox + s.face_range
            inb = lambda r: np.all((cc >= lo - r) & (cc <= hi + r), 1)  # noqa: E731
            unk_g, eval_g, sub_m = inb(r_u), inb(r_u + 2), inb(r_u + 4)
            sub = np.flatnonzero(sub_m)
            res = []
            t0 = time.time()
            for k in range(-s.face_range, s.face_range + 1):
                own_k = self._moved_owner(owner, un, k, ref_a, ref_b)
                pb = self.build(sub, unk_g, eval_g, own_k)
                x = np.zeros(pb.S1)
                x[: pb.S] = self.tau[sub]
                g0 = np.zeros((pb.S1, 3))
                g0[: pb.S] = ghat_all[sub]
                x, info = self.lm_solve(pb, x, jnp.asarray(g0), f"{tag}/unit{c}/k{k}")
                vox, thick, ts = self.slot_energies(pb, x, jnp.asarray(g0))
                ev = vox.sum(0)
                ow = own_k[sub]
                side_a = np.array([self.same(int(o), ref_a) for o in ow])
                side_b = np.array([self.same(int(o), ref_b) for o in ow]) & ~side_a
                pids = pb.patch_ids
                pa_ = np.array([self.same(int(p), ref_a) for p in pids], bool)
                pb_ = np.array([self.same(int(p), ref_b) for p in pids], bool) & ~pa_
                th = thick[: len(pids)]
                e_a = float(ev[side_a].sum() + th[pa_].sum())
                e_b = float(ev[side_b].sum() + th[pb_].sum())
                res.append(
                    {
                        "k": k,
                        "e_a": e_a,
                        "e_b": e_b,
                        "e_total": float(ev.sum() + thick.sum() + ts.sum()),
                        "lm_iters": len(info["iters"]),
                        "lm_limit": len(info["iters"]) >= s.lm_iters,
                        "n_unk": info["n_unk"],
                        "s": info["s"],
                    }
                )
            ea = np.array([r["e_a"] for r in res])
            eb = np.array([r["e_b"] for r in res])
            ks = np.arange(-s.face_range, s.face_range + 1)
            mid = s.face_range
            near = slice(mid - 1, mid + 2)
            best = int(ks[near][np.argmin((ea + eb)[near])])
            # A's preferred face position: argmin of A's energy (A extends by k);
            # B's: argmin of B's energy, in the same coordinate (B retreats by k).
            pa_best, pb_best = int(ks[np.argmin(ea)]), int(ks[np.argmin(eb)])
            rec = {
                "tag": tag,
                "unit": c,
                "faces": len(un["a"]),
                "area_mm2": float(len(un["a"]) * self.htau**2),
                "fallback": un["fallback"],
                "owners": [ref_a, ref_b],
                "axes": np.bincount(un["axis"], minlength=3).tolist(),
                "centroid_mm": (pts.mean(0) * self.h).tolist(),
                "energies": res,
                "move": best,
                "best_a": pa_best,
                "best_b": pb_best,
                "disagreement_mm": float((pb_best - pa_best) * self.htau),
                "censored": bool(
                    abs(pa_best) == s.face_range or abs(pb_best) == s.face_range
                ),
                "s": time.time() - t0,
            }
            out.append(rec)
            if best != 0:
                self.log.events["face_move"].append(
                    {k: rec[k] for k in ("tag", "unit", "faces", "move")}
                )
                moved = self._moved_owner(owner, un, best, ref_a, ref_b)
                ch = moved != owner
                new_owner[ch] = moved[ch]
            self.progress(
                f"  unit {c}: {len(un['a'])} faces, move {best:+d}, disagreement "
                f"{rec['disagreement_mm']:+.2f} mm{' (censored)' if rec['censored'] else ''}, {rec['s']:.1f} s"
            )
        return new_owner, out

    # ------------------------------------------------------------ snapshots

    def snapshot(self, name: str, tau: np.ndarray, owner: np.ndarray, faces: np.ndarray,
                 meta: dict[str, Any]) -> None:
        """Write snap_dir/<nnnn>-<name>.npz: index (flat index of each domain
        voxel in the unpadded grid), tau, owner, faces (minus voxel x, y, z and
        the axis) and meta (JSON)."""
        if self.snap_dir is None:
            return
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        G = self.G
        c = G.coords - PAD
        index = np.ravel_multi_index(tuple(c.T), tuple(G.shape)).astype(np.int32)
        fc = np.c_[c[faces[:, 0]], faces[:, 2]].astype(np.int16) if len(faces) else np.zeros((0, 4), np.int16)
        np.savez_compressed(
            self.snap_dir / f"{self._snap_n:04d}-{name}.npz", index=index, tau=tau[: G.S].astype(np.float32),
            owner=owner[: G.S].astype(np.int32), faces=fc, shape=G.shape, meta=json.dumps(meta, default=float),
        )
        self._snap_n += 1

    # ------------------------------------------------------------ outer loop

    def state_hash(self, owner: np.ndarray, faces: np.ndarray | None = None) -> str:
        hs = hashlib.sha1(owner.astype(np.int64).tobytes())
        if faces is not None and self.ridge:
            hs.update(np.ascontiguousarray(faces, np.int64).tobytes())
        return hs.hexdigest()

    def run(self) -> list[dict[str, Any]]:
        s = self.s
        seen = {self.state_hash(self.owner, self.faces): 0}
        self.face_records: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, np.ndarray]] = []
        for k in range(s.outer_iters):
            t0 = time.time()
            nf0 = len(self.faces)
            info, g0 = self.solve_global(f"outer{k}")
            t_solve = time.time() - t0
            topo = self.topology(f"outer{k}")
            t1 = time.time()
            own_u = self.upwind()
            n_up = int((own_u != self.owner).sum())
            bank_up = (
                None
                if self.ridge
                else int(
                    sum(
                        not self.same(int(a), int(b))
                        for a, b in zip(
                            self.owner[own_u != self.owner],
                            own_u[own_u != self.owner],
                        )
                    )
                )
            )
            t_up = time.time() - t1
            t2 = time.time()
            if s.face_test and not self.ridge:
                own_f, recs = self.face_test(own_u, g0, f"outer{k}")
            else:
                own_f, recs = own_u, []
            self.face_records.extend(recs)
            t_face = time.time() - t2
            n_face = int((own_f != own_u).sum())
            faces_new = self.faces if self.ridge else self.contact_faces(own_f)
            self.snapshot(f"outer{k}-owner", self.tau, own_f, faces_new,
                          {"tag": f"outer{k}", "owner_changes_upwind": n_up, "owner_changes_face": n_face})
            self.snapshots.append(
                {
                    "tau": self.tau.copy(),
                    "owner": self.owner.copy(),
                    "faces": self.faces.copy(),
                }
            )
            prev = self.owner
            self.owner = own_f
            self.faces = faces_new
            h = self.state_hash(own_f, faces_new)
            rec = {
                "outer": k,
                "solve_s": t_solve,
                "upwind_s": t_up,
                "face_s": t_face,
                "s": time.time() - t0,
                "lm_iters": len(info["iters"]),
                "e0": info["e0"],
                "e1": info["e1"],
                "cg_iters": [
                    t["cg_iters"] for it in info["iters"] for t in it["tries"]
                ],
                "cg_s": [t["cg_s"] for it in info["iters"] for t in it["tries"]],
                "jac_s": [it["jac_s"] for it in info["iters"]],
                "faces_before": nf0,
                "faces_after": len(faces_new),
                "owner_changes_upwind": n_up,
                "owner_changes_upwind_bank": bank_up,
                "owner_changes_face": n_face,
                "face_patches": len(recs),
                "face_moves": int(sum(r["move"] != 0 for r in recs)),
                "face_test_disabled": not s.face_test,
                "owner_changes_total": int((own_f != prev).sum()),
                "topology": topo,
                "ridge_faces": {k: len(v) for k, v in self.ridges.items()}
                if self.ridge
                else None,

                "tau_mm_pct": np.percentile(self.tau, [1, 50, 99]).tolist(),
            }
            self.log.outer.append(rec)
            self.progress(
                f"outer {k}: solve {t_solve:.1f} s ({rec['lm_iters']} LM, E {info['e0']:.1f} -> {info['e1']:.1f}), "
                f"upwind {n_up} changes ({bank_up} > 5 mm), faces {nf0} -> {len(faces_new)}, "
                f"face moves {rec['face_moves']}/{len(recs)} ({n_face} voxels), topology events {topo['events']}, "
                f"{rec['s']:.1f} s"
            )
            if rec["owner_changes_total"] == 0:
                rec["stop"] = "fixed point"
                break
            if h in seen:
                ev = {"outer": k, "same_as": seen[h], "period": k + 1 - seen[h]}
                self.log.events["oscillation"].append(ev)
                rec["stop"] = "oscillation"
                self.progress(
                    f"oscillation: the state after outer {k} equals the state after outer {seen[h] - 1}"
                )
                break
            seen[h] = k + 1
        return self.log.outer
