"""Tests for baillarger.antialias: the thin-plate field with a mask's signs, on every
PyTorch device here, against known surfaces, its constraints, an independent solver
(SciPy's L-BFGS-B) and the other devices."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from scipy import sparse
from scipy.optimize import minimize

from baillarger import antialias as AA
from baillarger import backend
from baillarger import mesh_intersect as mi
from baillarger import mesh_smoothing as S
from baillarger.mesh import marching_tetrahedra_blocks

DEVICES = [d for d in backend.available() if d != "numpy"]
N = 40
C = np.array([19.3, 20.1, 19.7])


def plane() -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    n = np.array([0.42, 0.29, 0.86])
    n /= np.linalg.norm(n)
    x = np.indices((N, N, N)).astype(float)
    return np.einsum("i...,i->...", x - C[:, None, None, None], n) < 0, n


def ball(r: float = 10.0) -> npt.NDArray[np.bool_]:
    x = np.indices((N, N, N)).astype(float)
    out: npt.NDArray[np.bool_] = np.sqrt(((x - C[:, None, None, None]) ** 2).sum(0)) < r
    return out


def mesh(
    phi: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    v, f, _ = marching_tetrahedra_blocks(phi, 0.0, 16, np.float32)
    return v.astype(np.float64), f[:, [0, 2, 1]]


def away_from_border(v: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    out: npt.NDArray[np.bool_] = np.all((v > 9) & (v < N - 10), axis=1)
    return out


def angle(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    c = np.clip(np.einsum("ij,ij->i", a, b), -1, 1)
    out: npt.NDArray[np.float64] = np.degrees(np.arccos(c))
    return out


@pytest.mark.parametrize("name", DEVICES)
def test_a_tilted_plane_comes_out_flat(name: str) -> None:
    mask, n = plane()
    phi, solve = AA.thin_plate(mask, backend.device(name))
    assert solve.converged
    v, f = mesh(phi)
    vr, fr = mesh(mask.view(np.uint8) - 0.5)
    keep = away_from_border(v)
    # the staircase is gone: normals within 2 deg, points within 0.1 voxel
    err = angle(S.vertex_normals(v, f), np.tile(n, (len(v), 1)))[keep]
    raw = angle(S.vertex_normals(vr, fr), np.tile(n, (len(vr), 1)))[
        away_from_border(vr)
    ]
    assert np.median(err) < 1.0 and err.max() < 2.5 and np.median(raw) > 20
    assert np.abs((v - C) @ n)[keep].max() < 0.1


@pytest.mark.parametrize("name", DEVICES)
def test_a_ball_keeps_its_radius_and_its_mesh(name: str) -> None:
    mask = ball()
    phi, _ = AA.thin_plate(mask, backend.device(name))
    v, f = mesh(phi)
    rad = np.linalg.norm(v - C, axis=1)
    assert np.abs(rad - 10.0).max() < 0.2
    nrm = S.vertex_normals(v, f)
    assert np.median(angle(nrm, (v - C) / rad[:, None])) < 2.0
    # the binary mask's triangles and keys; no self-intersections
    v0, f0, b0 = marching_tetrahedra_blocks(mask.view(np.uint8), 0.5, 16, np.float32)
    _, f1, b1 = marching_tetrahedra_blocks(phi, 0.0, 16, np.float32)
    assert np.array_equal(f1, f0) and np.array_equal(b1.vertex_key, b0.vertex_key)
    assert mi.intersecting_faces(v, f)[1] == 0


@pytest.mark.parametrize("name", DEVICES)
def test_the_signs_are_kept(name: str) -> None:
    """Two balls almost touching and a one-voxel blade: every voxel keeps its sign
    by at least eps, and outside the band the field is the signed distance."""
    x = np.indices((N, N, N)).astype(float)
    mask = (np.sqrt(((x - 12.4) ** 2).sum(0)) < 7) | (
        np.sqrt(((x - 27.1) ** 2).sum(0)) < 7
    )
    mask[19:21, 8:32, 19] = True  # a blade one voxel thick
    phi, _ = AA.thin_plate(mask, backend.device(name))
    assert phi[mask].min() >= AA.EPS - 1e-6 and phi[~mask].max() <= -AA.EPS + 1e-6
    sd = AA.signed_distance(mask, AA.BAND + 3)
    out = np.abs(sd) > AA.BAND
    assert np.array_equal(phi[out], sd[out])
    v, f = mesh(phi)
    assert mi.intersecting_faces(v, f)[1] == 0


def test_lbfgsb_finds_the_same_field() -> None:
    """The same problem solved by SciPy's L-BFGS-B (float64, CPU): the same vertices."""
    n = 20
    x = np.indices((n, n, n)).astype(float)
    mask = np.sqrt(((x - 9.6) ** 2).sum(0)) < 6.2
    phi, _ = AA.thin_plate(mask, backend.device("cpu"), tol=1e-7)
    sd = AA.signed_distance(mask, AA.BAND + 3)
    free = np.abs(sd) <= AA.BAND
    bounds: list[tuple[float | None, float | None]] = []
    for s, fr, m in zip(sd.ravel(), free.ravel(), mask.ravel(), strict=True):
        if not fr:
            bounds.append((float(s), float(s)))
        elif m:
            bounds.append((AA.EPS, None))
        else:
            bounds.append((None, -AA.EPS))

    def fg(z: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
        t = torch.as_tensor(z.reshape(sd.shape))
        return AA.energy(t), AA.energy_gradient(t).numpy().ravel()

    r = minimize(
        fg,
        sd.astype(np.float64).ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10),
    )
    ref = r.x.reshape(sd.shape)
    va, _ = mesh(phi)
    vb, _ = mesh(ref)
    assert np.abs(va - vb).max() < 1e-3
    assert (
        abs(AA.energy(torch.as_tensor(phi, dtype=torch.float64)) - r.fun) < 1e-3 * r.fun
    )


@pytest.mark.skipif(len(DEVICES) < 2, reason="one PyTorch device here")
def test_devices_agree() -> None:
    mask = ball(9.3)
    v = [mesh(AA.thin_plate(mask, backend.device(d))[0])[0] for d in DEVICES]
    for other in v[1:]:
        assert np.abs(other - v[0]).max() < 2e-3


def stencil_matrices(
    shape: tuple[int, int, int],
) -> list[tuple[sparse.csr_matrix, float]]:
    """The six stencils of baillarger.antialias as sparse matrices on a grid of shape
    (zero outside it, as the solver's padding), with their weights; read from the
    solver's own stencils applied to a point."""
    n = int(np.prod(shape))
    idx = np.arange(n).reshape(shape)
    out = []
    for i, w in enumerate((1.0, 1.0, 1.0, 2.0, 2.0, 2.0)):
        delta = torch.zeros((5, 5, 5), dtype=torch.float64)
        delta[2, 2, 2] = 1.0
        k = AA._stencil(delta, i).numpy()
        rows, cols, vals = [], [], []
        for o in np.argwhere(np.abs(k) > 0):
            d = 2 - o  # (D x)[p] = sum_d c_d x[p + d], c_d = k[centre - d]
            c = k[tuple(o)]
            lo = np.maximum(-d, 0)
            hi = np.array(shape) - np.maximum(d, 0)
            p = idx[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].ravel()
            q = idx[
                lo[0] + d[0] : hi[0] + d[0],
                lo[1] + d[1] : hi[1] + d[1],
                lo[2] + d[2] : hi[2] + d[2],
            ].ravel()
            rows.append(p)
            cols.append(q)
            vals.append(np.full(len(p), c))
        D = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )
        out.append((D, w))
    return out


REFERENCE = Path(__file__).parent / "reference"


def test_cvxpy_finds_the_same_optimum(tmp_path: Path) -> None:
    """The same problem, built as an explicit quadratic program and solved to high
    precision by an interior-point method (CVXPY with Clarabel), in the reference
    environment of its own (tests/reference: CVXPY's HiGHS must not share a process
    with OR-Tools): the same field, energy and vertices as our solver."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on the path (run in the Nix devShell)")
    n = 16
    x = np.indices((n, n, n)).astype(float)
    mask = np.sqrt(((x - 7.6) ** 2).sum(0)) < 4.7
    sd = AA.signed_distance(mask, AA.BAND + 3)
    mats = stencil_matrices((n, n, n))
    # the matrices are the solver's energy
    z = np.random.default_rng(9).standard_normal(sd.shape)
    e_mat = sum(w * float(np.sum((D @ z.ravel()) ** 2)) for D, w in mats)
    assert np.isclose(e_mat, AA.energy(torch.as_tensor(z)), rtol=1e-12)

    free = (np.abs(sd) <= AA.BAND).ravel()
    m = mask.ravel()
    arrays: dict[str, npt.ArrayLike] = dict(
        n=n**3,
        eps=AA.EPS,
        count=len(mats),
        fixed_idx=np.flatnonzero(~free),
        fixed_val=sd.ravel()[~free],
        pos_idx=np.flatnonzero(free & m),
        neg_idx=np.flatnonzero(free & ~m),
    )
    for i, (D, w) in enumerate(mats):
        arrays.update(
            {f"D{i}_data": D.data, f"D{i}_indices": D.indices, f"D{i}_indptr": D.indptr}
        )
        arrays[f"w{i}"] = w
    src, dst = tmp_path / "problem.npz", tmp_path / "solution.npz"
    np.savez(src, **arrays)  # type: ignore[arg-type]
    subprocess.run(
        [
            uv,
            "run",
            "--frozen",
            "--project",
            str(REFERENCE),
            "python",
            str(REFERENCE / "solve.py"),
            str(src),
            str(dst),
        ],
        check=True,
    )
    out = np.load(dst)
    assert str(out["status"]) == "optimal"
    ref = out["phi"].reshape(sd.shape).astype(np.float64)
    value = float(out["value"])

    ours, _ = AA.thin_plate(mask, backend.device("cpu"), tol=1e-8, max_iter=100000)
    e_ours = AA.energy(torch.as_tensor(ours, dtype=torch.float64))
    assert abs(e_ours - value) <= 1e-8 * value
    near = np.abs(ref) < 1.0
    assert np.abs(ours - ref)[near].max() < 1e-5
    va, _ = mesh(ours)
    vb, _ = mesh(ref)
    assert np.abs(va - vb).max() < 1e-5
