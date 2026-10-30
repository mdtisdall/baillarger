"""The white surface as the viewer shows it, without the columns: its coordinates,
mesh graph, flat map and inflation (cached), picking, and the bytes the panes load.
baillarger.viewer.data.Site holds one and adds the traces.

Made from what a mesh reader gives: vertices in voxel coordinates of a grid, faces,
the voxel size and the grid's voxel-to-world affine. Box mm: voxel index times the
voxel size (the sections and the 3D surface pane); world mm: through the affine (the
volume pane). The flat map and the inflation are cached in the cache directory under
the surface's name, keyed by `key` (default: a hash of the mesh), so that a rebuilt
mesh gets new ones. With flat=False there is no flat map (uv_v and T_v are NaN): the
flat map needs a boundary on every piece, which a closed surface does not have.
"""

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

from baillarger.mesh_smoothing import edges
from baillarger.viewer import flatmap as FM
from baillarger.viewer import inflation as IN

type Float = npt.NDArray[np.float64]
type Arr = npt.NDArray[Any]


def mesh_graph(v_mm: Float, f: Arr) -> csr_matrix:
    E = edges(f)
    w = np.linalg.norm(v_mm[E[:, 0]] - v_mm[E[:, 1]], axis=1)
    G = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(v_mm),) * 2).tocsr()
    return G.maximum(G.T).tocsr()


def mesh_key(v: Arr, f: Arr) -> str:
    h = hashlib.sha1(np.ascontiguousarray(v, np.float32).tobytes())
    h.update(np.ascontiguousarray(f, np.int64).tobytes())
    return h.hexdigest()[:16]


def cached(path: Path, key: str, make: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """The arrays in path if its key matches, else make() saved there with the key."""
    if path.exists():
        with np.load(path, allow_pickle=False) as Q:
            if str(Q["key"]) == key:
                return {k: Q[k] for k in Q.files if k != "key"}
    out = make()
    arrays = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    result = dict(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays["key"] = np.array(key)
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    summ = {k: v for k, v in out.items() if not isinstance(v, np.ndarray)}
    path.with_suffix(".json").write_text(json.dumps(summ, indent=1, default=float))
    return result


def gifti_bytes(v_world: Float, f: Arr) -> bytes:
    g = GiftiImage()
    g.add_gifti_data_array(
        GiftiDataArray(
            np.asarray(v_world, np.float32),
            intent="NIFTI_INTENT_POINTSET",
            datatype="NIFTI_TYPE_FLOAT32",
        )
    )
    g.add_gifti_data_array(
        GiftiDataArray(
            np.asarray(f, np.int32),
            intent="NIFTI_INTENT_TRIANGLE",
            datatype="NIFTI_TYPE_INT32",
        )
    )
    out: bytes = g.to_bytes()
    return out


class Surface:
    """A white surface for viewing; see the module docstring. v_vox: vertices (voxel
    coordinates); f: faces; vox: voxel size (mm); affine: voxel to world mm; flat:
    make the flat map."""

    def __init__(
        self,
        v_vox: Float,
        f: Arr,
        vox: Float,
        affine: Float,
        cache: Path,
        name: str,
        key: str | None = None,
        log: Callable[[str], object] = print,
        flat: bool = True,
    ) -> None:
        t0 = time.time()
        self.v_vox = np.asarray(v_vox, float)
        self.f: Arr = np.asarray(f).astype(np.int64)
        self.vox = np.asarray(vox, float)
        self.affine = np.asarray(affine, float)
        self.vmm: Float = self.v_vox * self.vox
        self.key = mesh_key(self.v_vox, self.f) if key is None else key
        self.G = mesh_graph(self.vmm, self.f)
        cache = Path(cache)
        if flat:
            F = cached(
                cache / f"flat_map_{name}.npz",
                self.key,
                lambda: FM.site_flat_map(self.vmm, self.f, log=log),
            )
            self.uv_v: Float = F["uv_v"]
            self.T_v: Float = F["T_v"]
        else:
            self.uv_v = np.full((len(self.vmm), 2), np.nan)
            self.T_v = np.full((len(self.vmm), 3, 2), np.nan)
        Ip = cached(
            cache / f"inflate_{name}.npz",
            f"{self.key}-v{IN.VERSION}",
            lambda: IN.site_inflation(self.vmm, self.f, log=log),
        )
        self.infl: list[Float] = [self.vmm] + [x.astype(float) for x in Ip["x"]]
        self.infl_names = ["true surface"] + [
            f"inflated, lam {lv:g}" for lv in Ip["levels"]
        ]
        self.vf = IN.vertex_faces(self.f, len(self.vmm))
        self.sulc = IN.sulcal_depth(self.vmm, self.infl[-1], self.f)
        self.vertex_tree = cKDTree(self.world(self.v_vox))
        self._white: bytes | None = None
        log(
            f"surface {name}: {len(self.vmm):,} vertices, {len(self.f):,} faces "
            f"({time.time() - t0:.0f} s)"
        )

    def world(self, p_vox: npt.ArrayLike) -> Float:
        p = np.atleast_2d(np.asarray(p_vox, float))
        out: Float = (np.c_[p, np.ones(len(p))] @ self.affine.T)[:, :3]
        return out

    def nearest(self, mm: npt.ArrayLike) -> int:
        """The vertex nearest world point mm."""
        _, k = self.vertex_tree.query(np.asarray(mm, float))
        return int(k)

    def mesh_payload(self, level: int, vtrace: Arr | None = None) -> bytes:
        """uint32 header length, JSON header padded to 4 bytes, then float32 positions
        (box mm), normals, sulcal depth (inflated level `level`), and with vtrace
        (int32 per vertex): vtrace and uint32 faces."""
        x = self.infl[int(level)]
        parts: list[Arr] = [
            x.astype(np.float32),
            IN.vertex_normals(x, self.f).astype(np.float32),
            self.sulc.astype(np.float32),
        ]
        faces = vtrace is not None
        if vtrace is not None:
            parts += [vtrace, self.f.astype(np.uint32)]
        c = x.mean(0)
        hdr = dict(
            nv=len(x),
            nf=len(self.f),
            level=int(level),
            faces=bool(faces),
            centre=c.tolist(),
            radius=float(np.linalg.norm(x - c, axis=1).max()),
        )
        h = json.dumps(hdr).encode()
        h += b" " * (-len(h) % 4)
        return (
            np.uint32(len(h)).tobytes()
            + h
            + b"".join(np.ascontiguousarray(p).tobytes() for p in parts)
        )

    def white_bytes(self) -> bytes:
        """The surface as GIfTI bytes (world mm), for the volume pane."""
        if self._white is None:
            self._white = gifti_bytes(self.world(self.v_vox), self.f)
        return self._white
