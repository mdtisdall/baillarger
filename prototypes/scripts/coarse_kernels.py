"""The coarse scan's array stages, on numpy (the reference) and on PyTorch
(cpu, cuda or mps): trilinear sampling, the leading eigenvector of orient's
tensor, eigenvector tracing, detrended log profiles and neighbour
correlation curves.

Coordinates: points x in mm; box voxel coordinates q = x @ Ainv.T + b.
D maps a box-voxel direction to a mm direction (A[:3, :3] / voxel_mm).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, map_coordinates

PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
L1_MIN_C, L1_MAX_TURN = 0.3, 45.0


# ---------------------------------------------------------------- numpy


def sample_np(vol, q):
    """Trilinear, edges extended. vol (X, Y, Z) -> (n,), or (C, X, Y, Z) -> (C, n)."""
    if vol.ndim == 3:
        return map_coordinates(vol, q.T, order=1, mode="nearest")
    return np.stack([map_coordinates(v, q.T, order=1, mode="nearest") for v in vol])


def field_np(T, q, D):
    """Leading eigenvector (as a mm direction), coherence and finite mask at q."""
    t = sample_np(T, q)
    ok = np.all(np.isfinite(t), 0)
    M = np.zeros((len(q), 3, 3))
    for k, (i, j) in enumerate(PAIRS):
        M[:, i, j] = M[:, j, i] = np.where(ok, t[k], 0)
    w, V = np.linalg.eigh(M)
    v = V[:, :, 2] @ D.T
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.clip((w[:, 2] - w[:, 1]) / (w[:, 2] + w[:, 1]), 0, 1)
    return v, np.where(ok, c, 0.0), ok


def heading_np(v, c, ok, h):
    dot = np.einsum("ij,ij->i", v, h)
    vv = v * np.where(dot < 0, -1.0, 1.0)[:, None]
    use = ok & (c >= L1_MIN_C) & (np.abs(dot) >= np.cos(np.radians(L1_MAX_TURN)))
    return np.where(use[:, None], vv, h)


def trace_np(T, out0, R, N, Ainv, b, D, s_max, tstep, early_mm, s_grid):
    """Midpoint eigenvector tracing from roots R along normals N. Returns the
    paths resampled on s_grid (m, len(s_grid), 3) and the share of steps traced."""
    m = len(R)
    nt = int(round(s_max / tstep)) + 1
    Pt = np.empty((m, nt, 3))
    x = R.astype(np.float64).copy()
    h = N / np.linalg.norm(N, axis=1, keepdims=True)
    Pt[:, 0] = x
    act = np.ones(m, bool)
    hit = np.full(m, -1)

    def qb(z):
        return z @ Ainv.T + b

    for k in range(1, nt):
        ia = np.flatnonzero(act)
        if not len(ia):
            Pt[:, k] = x
            continue
        xa, ha = x[ia], h[ia]
        h1 = heading_np(*field_np(T, qb(xa), D), ha)
        h2 = heading_np(*field_np(T, qb(xa + 0.5 * tstep * h1), D), h1)
        x[ia] = xa + tstep * h2
        h[ia] = h2
        Pt[:, k] = x
        if early_mm > 0:
            o = sample_np(out0, qb(x[ia])) > 0.5
            new = o & (hit[ia] < 0)
            hit[ia[new]] = k
            done = (hit[ia] >= 0) & ((k - hit[ia]) * tstep >= early_mm)
            act[ia[done]] = False
    share = (
        float(np.mean(np.where(hit >= 0, np.minimum(hit + early_mm / tstep, nt), nt)) / nt)
        if early_mm > 0
        else 1.0
    )
    return resample_np(Pt, tstep, s_grid), share


def resample_np(Pt, tstep, s_grid):
    nt = Pt.shape[1]
    if nt == len(s_grid) and np.allclose(np.arange(nt) * tstep, s_grid):
        return Pt
    st = np.arange(nt) * tstep
    j = np.clip(np.searchsorted(st, s_grid, side="right") - 1, 0, nt - 2)
    w = ((s_grid - st[j]) / tstep)[None, :, None]
    return Pt[:, j] * (1 - w) + Pt[:, j + 1] * w


def profile_np(echo, Pq, trend_samples):
    """Detrended log profile: echo (X, Y, Z), Pq (m, ns, 3) box coordinates."""
    m, ns, _ = Pq.shape
    v = np.log(np.maximum(sample_np(echo, Pq.reshape(-1, 3)).reshape(m, ns), 1.0))
    return v - gaussian_filter1d(v, trend_samples, axis=1, mode="nearest")


def curves_np(fa, gb, pairs, top, sp, trim, shifts, min_n, kp):
    """Correlation of fa[a] with gb[b] shifted by each of shifts, over
    trim <= s <= min(top[a], top[b]) - trim; NaN where the overlap, counted
    in 0.04 mm samples (count x kp), is under min_n."""
    a, b = pairs[:, 0], pairs[:, 1]
    lim = np.fmin(top[a], top[b]) - trim
    mk = (sp[None] >= trim) & (sp[None] <= np.where(np.isfinite(lim), lim, -1)[:, None])
    out = np.full((len(pairs), len(shifts)), np.nan)
    x0, y0 = fa[a], gb[b]
    for q, kk in enumerate(shifts):
        y = np.roll(y0, kk, axis=1)
        mm = mk.copy()
        if kk > 0:
            mm[:, :kk] = False
        elif kk < 0:
            mm[:, kk:] = False
        cnt = mm.sum(1)
        x_, y_ = np.where(mm, x0, 0.0), np.where(mm, y, 0.0)
        c1 = np.maximum(cnt, 1)
        mx, my = x_.sum(1) / c1, y_.sum(1) / c1
        cov = (x_ * y_).sum(1) / c1 - mx * my
        vx = (x_ * x_).sum(1) / c1 - mx**2
        vy = (y_ * y_).sum(1) / c1 - my**2
        r = cov / np.sqrt(np.maximum(vx * vy, 1e-12))
        good = cnt * kp >= min_n
        out[good, q] = r[good]
    return out


# ---------------------------------------------------------------- torch


def sample_t(vol, q):
    """Trilinear, edges extended (as sample_np). vol (C, X, Y, Z), q (n, 3) -> (C, n)."""
    C = vol.shape[0]
    S = torch.tensor(vol.shape[1:], device=q.device, dtype=q.dtype)
    q = torch.minimum(torch.clamp(q, min=0), S - 1)
    i0 = torch.floor(q).long()
    i0 = torch.minimum(i0, (S - 2).clamp(min=0).long())
    f = q - i0.to(q.dtype)
    X, Y, Z = vol.shape[1:]
    flat = vol.reshape(C, -1)
    out = torch.zeros((C, len(q)), dtype=vol.dtype, device=vol.device)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                idx = (i0[:, 0] + dx).clamp(max=X - 1) * (Y * Z) + (i0[:, 1] + dy).clamp(
                    max=Y - 1
                ) * Z + (i0[:, 2] + dz).clamp(max=Z - 1)
                # as scipy: a NaN corner gives NaN even at zero weight
                out = out + flat[:, idx] * (wx * wy * wz).to(vol.dtype)
    return out


def leading_t(M):
    """Closed-form eigen-analysis of symmetric 3x3 matrices (n, 3, 3): the
    leading eigenvector (unit, any sign) and the two largest eigenvalues.
    The vector is accurate where the leading eigenvalue is well separated;
    where it is not, the coherence is low and tracing keeps its heading."""
    n = M.shape[0]
    scale = M.abs().reshape(n, -1).amax(1).clamp(min=1e-30)
    A = M / scale[:, None, None]
    a00, a11, a22 = A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]
    a01, a02, a12 = A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]
    q = (a00 + a11 + a22) / 3
    p1 = a01**2 + a02**2 + a12**2
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2 * p1
    p = torch.sqrt(p2 / 6)
    ps = torch.where(p > 1e-12, p, torch.ones_like(p))
    b00, b11, b22 = (a00 - q) / ps, (a11 - q) / ps, (a22 - q) / ps
    b01, b02, b12 = a01 / ps, a02 / ps, a12 / ps
    det = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)
    r = torch.clamp(det / 2, -1, 1)
    phi = torch.acos(r) / 3
    e1 = q + 2 * p * torch.cos(phi)
    e3 = q + 2 * p * torch.cos(phi + 2 * math.pi / 3)
    e2 = 3 * q - e1 - e3
    I3 = torch.eye(3, dtype=M.dtype, device=M.device)
    Rm = A - e1[:, None, None] * I3
    r0, r1, r2 = Rm[:, 0], Rm[:, 1], Rm[:, 2]
    cs = torch.stack([torch.linalg.cross(r0, r1), torch.linalg.cross(r0, r2), torch.linalg.cross(r1, r2)], 1)
    nr = torch.linalg.norm(cs, dim=2)
    best = nr.argmax(1)
    v = cs[torch.arange(n, device=M.device), best]
    nv = nr[torch.arange(n, device=M.device), best]
    fallback = torch.zeros_like(v)
    fallback[:, 0] = 1
    v = torch.where((nv > 0)[:, None], v / nv.clamp(min=1e-30)[:, None], fallback)
    return v, e1 * scale, e2 * scale


def field_t(T, q, D):
    t = sample_t(T, q)
    ok = torch.all(torch.isfinite(t), 0)
    t = torch.where(ok, t, torch.zeros((), dtype=t.dtype, device=t.device))
    M = torch.zeros((q.shape[0], 3, 3), dtype=t.dtype, device=t.device)
    for k, (i, j) in enumerate(PAIRS):
        M[:, i, j] = t[k]
        M[:, j, i] = t[k]
    v, w1, w2 = leading_t(M)
    v = v @ D.T
    v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp(min=1e-12)
    c = torch.nan_to_num(torch.clamp((w1 - w2) / (w1 + w2), 0, 1), nan=0.0)
    return v, torch.where(ok, c, torch.zeros_like(c)), ok


def heading_t(v, c, ok, h):
    dot = (v * h).sum(1)
    vv = v * torch.where(dot < 0, -1.0, 1.0).to(v.dtype)[:, None]
    use = ok & (c >= L1_MIN_C) & (dot.abs() >= math.cos(math.radians(L1_MAX_TURN)))
    return torch.where(use[:, None], vv, h)


def trace_t(T, out0, R, N, Ainv, b, D, s_max, tstep, early_mm, s_grid):
    """trace_np on a torch device. T (6, X, Y, Z) and out0 (1, X, Y, Z) are
    tensors; the rest numpy. Returns numpy float64 paths and the share."""
    dev, dt = T.device, T.dtype
    m = len(R)
    nt = int(round(s_max / tstep)) + 1
    Ai = torch.as_tensor(Ainv, dtype=dt, device=dev)
    bb = torch.as_tensor(b, dtype=dt, device=dev)
    Dt = torch.as_tensor(D, dtype=dt, device=dev)
    x = torch.as_tensor(R, dtype=dt, device=dev).clone()
    Nn = N / np.linalg.norm(N, axis=1, keepdims=True)
    h = torch.as_tensor(Nn, dtype=dt, device=dev).clone()
    Pt = torch.empty((m, nt, 3), dtype=dt, device=dev)
    Pt[:, 0] = x
    act = torch.ones(m, dtype=torch.bool, device=dev)
    hit = torch.full((m,), -1, dtype=torch.long, device=dev)
    for k in range(1, nt):
        ia = torch.nonzero(act).flatten()
        if ia.numel() == 0:
            Pt[:, k:] = x[:, None]
            break
        xa, ha = x[ia], h[ia]
        h1 = heading_t(*field_t(T, xa @ Ai.T + bb, Dt), ha)
        h2 = heading_t(*field_t(T, (xa + 0.5 * tstep * h1) @ Ai.T + bb, Dt), h1)
        x[ia] = xa + tstep * h2
        h[ia] = h2
        Pt[:, k] = x
        if early_mm > 0:
            o = sample_t(out0, x[ia] @ Ai.T + bb)[0] > 0.5
            hi = hit[ia]
            new = o & (hi < 0)
            hit[ia[new]] = k
            hi = hit[ia]
            done = (hi >= 0) & ((k - hi).to(dt) * tstep >= early_mm)
            act[ia[done]] = False
    hit_n = hit.cpu().numpy()
    share = (
        float(np.mean(np.where(hit_n >= 0, np.minimum(hit_n + early_mm / tstep, nt), nt)) / nt)
        if early_mm > 0
        else 1.0
    )
    return resample_np(Pt.cpu().numpy().astype(np.float64), tstep, s_grid), share


def _gauss_kernel(sigma, truncate=4.0):
    r = int(truncate * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


def profile_t(echo, Pq, trend_samples):
    """profile_np on a torch device: echo (1, X, Y, Z) tensor, Pq (m, ns, 3) tensor."""
    m, ns, _ = Pq.shape
    v = torch.log(torch.clamp(sample_t(echo, Pq.reshape(-1, 3))[0].reshape(m, ns), min=1.0))
    k, r = _gauss_kernel(trend_samples)
    kt = torch.as_tensor(k, dtype=v.dtype, device=v.device).view(1, 1, -1)
    # scipy mode "nearest": the edge value repeated
    vp = torch.cat([v[:, :1].expand(m, r), v, v[:, -1:].expand(m, r)], 1)
    trend = torch.nn.functional.conv1d(vp[:, None], kt)[:, 0]
    return v - trend


def curves_t(fa, gb, pairs, top, sp, trim, shifts, min_n, kp):
    """curves_np on a torch device; fa, gb tensors (m, ns); the rest numpy."""
    dev, dt = fa.device, fa.dtype
    a = torch.as_tensor(pairs[:, 0], device=dev)
    b = torch.as_tensor(pairs[:, 1], device=dev)
    lim = np.fmin(top[pairs[:, 0]], top[pairs[:, 1]]) - trim
    mk_n = (sp[None] >= trim) & (sp[None] <= np.where(np.isfinite(lim), lim, -1)[:, None])
    mk = torch.as_tensor(mk_n, device=dev)
    x0, y0 = fa[a], gb[b]
    out = torch.full((len(pairs), len(shifts)), float("nan"), dtype=dt, device=dev)
    zero = torch.zeros((), dtype=dt, device=dev)
    for q, kk in enumerate(shifts):
        kk = int(kk)
        y = torch.roll(y0, kk, dims=1)
        mm = mk.clone()
        if kk > 0:
            mm[:, :kk] = False
        elif kk < 0:
            mm[:, kk:] = False
        cnt = mm.sum(1)
        x_, y_ = torch.where(mm, x0, zero), torch.where(mm, y, zero)
        c1 = cnt.clamp(min=1).to(dt)
        mx, my = x_.sum(1) / c1, y_.sum(1) / c1
        cov = (x_ * y_).sum(1) / c1 - mx * my
        vx = (x_ * x_).sum(1) / c1 - mx**2
        vy = (y_ * y_).sum(1) / c1 - my**2
        r = cov / torch.sqrt(torch.clamp(vx * vy, min=1e-12))
        good = cnt * kp >= min_n
        out[:, q] = torch.where(good, r, out[:, q])
    return out.cpu().numpy().astype(np.float64)
