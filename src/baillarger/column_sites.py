"""The review sites of the column top stage (format:
docs/formats/column-sites.md; plan: docs/plans/column-sites.md).

The command line:

    python -m baillarger.column_sites SURF CONF FEAT... [--out DIR] [--radius MM]

SURF is the white surface that the column stage cast its rays from (its
sidecar names the corrected segmentation, whose grid the sites use). CONF is
the stage's top confidence (.npz, keys in CONF_KEYS). FEAT are the stage's
feature files (.npz, each with edges_v: the column graph, as pairs of
vertices). It writes the sites in DIR, or next to the surface, and prints
"role: path" for each file.

The stage's review queue orders the columns: no top first, then the lowest
confidence first. Confidence orders the queue. It does not gate a column. So
the sites use no confidence threshold (group_columns):

1. Take the columns in queue order.
2. A column joins the site of a neighbour (on the column graph) that came
   before it, if the column is within radius mm of that site's first column
   (its seed). With more than one such site, it joins the site with the
   earliest seed.
3. Otherwise, the column starts a new site and is its seed.

Each column is in exactly one site. The sites are in the order of their
seeds. A site's score is its seed's confidence, and its voxel is its seed's
root on the segmentation's grid.

The answers at these sites go to the answer log of the WM review
(wm-edits/1), with the site's kind as SiteKind. dseg replays them as it
replays any site answer.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix

from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    provenance,
    read_sidecar,
    sidecar_path,
    sources,
    write_sidecar,
)
from baillarger.timing import log_to_stderr
from baillarger.white_surf import SURF_SPEC

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]

#: mm: a column joins a site only within this distance of the site's seed
RADIUS = 2.0
#: the site kinds: the seed has no top, or the seed has a top
KINDS = ("column-no-top", "column-top")
#: the keys of the stage's top confidence (.npz) that the sites read
CONF_KEYS = (
    "vert",
    "root",
    "rank",
    "conf",
    "top_mm",
    "top_k",
    "terms",
    "term_names",
    "gm_behind",
    "fl_behind",
)
#: the table's columns
COLUMNS = (
    "rank",
    "kind",
    "i",
    "j",
    "k",
    "score",
    "vertex",
    "columns",
    "no_top",
    "median_score",
    "extent_mm",
    "top_mm",
    "gm_behind_mm",
    "fluid_behind_mm",
    "lowest_term",
)
#: the columns of the table that are integers, and those that are numbers
#: (n/a: no value, read as None)
_INTS = ("rank", "i", "j", "k", "vertex", "columns", "no_top")
_NUMS = (
    "score",
    "median_score",
    "extent_mm",
    "top_mm",
    "gm_behind_mm",
    "fluid_behind_mm",
)
#: the members table's columns
MEMBER_COLUMNS = ("vertex", "site")

#: docs/formats/column-sites.md
COLUMN_SITES_SPEC = SidecarSpec(
    "column-sites/1",
    COMMON_KEYS
    + (
        Key("Hemisphere", "string", True, "R or L"),
        Key("RadiusMm", "number", True, "largest distance of a column to its seed"),
        Key("Columns", "integer", True, "columns in the sites"),
        Key("Edges", "integer", True, "edges of the column graph that were used"),
        Key("Sites", "integer", True, "rows of the table"),
        Key("SitesByKind", "object", True, "rows of the table, by kind"),
        Key("TermNames", "array of strings", True, "the stage's confidence terms"),
    ),
)


def queue_order(rank: npt.ArrayLike) -> Int:
    """The column indices in queue order, from the stage's rank (0 is the
    front of the queue). Raises ValueError unless rank is a permutation of
    0 .. n - 1."""
    r = np.asarray(rank, np.int64)
    if not np.array_equal(np.sort(r), np.arange(len(r))):
        raise ValueError("rank is not a permutation of 0 .. n - 1")
    out = np.empty(len(r), np.int64)
    out[r] = np.arange(len(r))
    return out


def column_edges(vert: npt.ArrayLike, edges_v: npt.ArrayLike) -> Int:
    """The edges of the column graph as pairs of column indices (rows of
    vert), from pairs of vertices. Edges with a vertex that is not a column,
    and loops, are dropped. Each edge is once, as (low, high), sorted."""
    v = np.asarray(vert, np.int64)
    e = np.asarray(edges_v, np.int64).reshape(-1, 2)
    s = np.argsort(v, kind="stable")
    sv = v[s]
    pos = np.searchsorted(sv, e)
    pos_c = np.minimum(pos, len(sv) - 1)
    ok = (sv[pos_c] == e).all(1) if len(sv) else np.zeros(len(e), bool)
    c = s[pos_c[ok]]
    c = c[c[:, 0] != c[:, 1]]
    c = np.sort(c, axis=1)
    out: Int = np.unique(c, axis=0) if len(c) else np.zeros((0, 2), np.int64)
    return out


@dataclass(frozen=True)
class Grouping:
    """site: the site of each column (0, 1, ..., in the order of the seeds);
    seeds: the seed column of each site."""

    site: Int
    seeds: Int


def group_columns(order: Int, edges: Int, pos: Float, radius: float) -> Grouping:
    """The sites of the columns (module docstring): order holds the column
    indices in queue order, edges the column graph (pairs of column
    indices), pos the columns' positions (mm)."""
    n = len(pos)
    A = coo_matrix(
        (np.ones(2 * len(edges)), (edges.ravel(), edges[:, ::-1].ravel())),
        shape=(n, n),
    ).tocsr()
    ptr, nbr = A.indptr, A.indices
    site = np.full(n, -1, np.int64)
    seeds: list[int] = []
    r2 = radius * radius
    for c in order.tolist():
        best = -1
        for q in nbr[ptr[c] : ptr[c + 1]].tolist():
            s = int(site[q])
            if s < 0 or (best >= 0 and s >= best):
                continue
            d = pos[c] - pos[seeds[s]]
            if float(d @ d) <= r2:
                best = s
        if best < 0:
            best = len(seeds)
            seeds.append(c)
        site[c] = best
    return Grouping(site, np.array(seeds, np.int64))


def _num(x: float) -> float | None:
    return round(float(x), 4) if math.isfinite(x) else None


def site_rows(
    g: Grouping,
    conf: dict[str, Any],
    pos: Float,
    vox: Int,
) -> list[dict[str, Any]]:
    """One row per site (COLUMNS), in the order of the seeds: conf holds the
    arrays of CONF_KEYS, pos the columns' roots (mm), vox the same roots on
    the segmentation's grid. A value that is not finite is None (n/a)."""
    score = np.asarray(conf["conf"], np.float64)
    no_top = np.asarray(conf["top_k"]) < 0
    terms = np.asarray(conf["terms"], np.float64)
    names = [str(t) for t in conf["term_names"]]
    by = np.argsort(g.site, kind="stable")
    cuts = np.searchsorted(g.site[by], np.arange(len(g.seeds) + 1))
    rows = []
    for s, seed in enumerate(g.seeds.tolist()):
        m = by[cuts[s] : cuts[s + 1]]
        t = terms[seed]
        lowest = names[int(np.nanargmin(t))] if np.isfinite(t).any() else "n/a"
        ext = np.sqrt(((pos[m] - pos[seed]) ** 2).sum(1)).max()
        rows.append(
            dict(
                rank=s + 1,
                kind=KINDS[0] if no_top[seed] else KINDS[1],
                i=int(vox[seed, 0]),
                j=int(vox[seed, 1]),
                k=int(vox[seed, 2]),
                score=_num(score[seed]),
                vertex=int(conf["vert"][seed]),
                columns=len(m),
                no_top=int(no_top[m].sum()),
                median_score=_num(float(np.median(score[m]))),
                extent_mm=_num(ext),
                top_mm=_num(float(conf["top_mm"][seed])),
                gm_behind_mm=_num(float(conf["gm_behind"][seed])),
                fluid_behind_mm=_num(float(conf["fl_behind"][seed])),
                lowest_term=lowest,
            )
        )
    return rows


def sites_paths(surf: Path, out: Path | None = None) -> dict[str, Path]:
    """The sites' files of a white surface's .surf.gii: in the directory out,
    or next to the surface if out is None."""
    stem = surf.name.removesuffix("_white.surf.gii")
    d = surf.parent if out is None else out
    return dict(
        table=d / f"{stem}_desc-columnsites_sites.tsv",
        sidecar=d / f"{stem}_desc-columnsites_sites.json",
        members=d / f"{stem}_desc-columnsites_members.tsv",
    )


def _cell(x: Any) -> str:
    return "n/a" if x is None else str(x)


def write_column_sites(
    surf: Path,
    conf_path: Path,
    feat_paths: list[Path],
    out: Path | None = None,
    radius: float = RADIUS,
) -> dict[str, Path]:
    """Group the columns of the stage's top confidence (conf_path) into
    sites, with the column graph of the feature files, and write them for
    the white surface surf (docs/formats/column-sites.md). Returns the paths
    written."""
    side = read_sidecar(sidecar_path(surf), SURF_SPEC)
    dseg = Path(side["Sources"][0])
    z = np.load(conf_path)
    conf = {k: z[k] for k in CONF_KEYS}
    edges_v = [np.load(p)["edges_v"] for p in feat_paths]
    E = column_edges(conf["vert"], np.concatenate(edges_v) if edges_v else [])
    pos = np.asarray(conf["root"], np.float64)
    img = nib.load(dseg)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{dseg.name}: not a NIfTI-1 image")
    hi = np.array(img.shape[:3], np.int64) - 1
    ijk = nib.affines.apply_affine(np.linalg.inv(img.affine), pos)
    vox = np.clip(np.rint(ijk).astype(np.int64), 0, hi)
    g = group_columns(queue_order(conf["rank"]), E, pos, radius)
    rows = site_rows(g, conf, pos, vox)
    p = sites_paths(surf, out)
    p["table"].parent.mkdir(parents=True, exist_ok=True)
    with p["table"].open("w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join(_cell(r[c]) for c in COLUMNS) + "\n")
    order = np.lexsort((conf["vert"], g.site))
    with p["members"].open("w") as fh:
        fh.write("\t".join(MEMBER_COLUMNS) + "\n")
        for v, s in zip(
            conf["vert"][order].tolist(), (g.site[order] + 1).tolist(), strict=True
        ):
            fh.write(f"{v}\t{s}\n")
    dat = surf.with_name(surf.name.removesuffix(".surf.gii") + ".dat")
    write_sidecar(
        p["sidecar"],
        COLUMN_SITES_SPEC,
        dict(
            SpecVersion=COLUMN_SITES_SPEC.version,
            Description="The review queue of the column top stage: every "
            "column in one site, the sites in the order of their seeds.",
            **sources(surf, dat, dseg, conf_path, *feat_paths),
            Provenance=provenance("baillarger.column_sites.write_column_sites"),
            Hemisphere=side["Hemisphere"],
            RadiusMm=float(radius),
            Columns=len(pos),
            Edges=len(E),
            Sites=len(rows),
            SitesByKind={k: sum(r["kind"] == k for r in rows) for k in KINDS},
            TermNames=[str(t) for t in conf["term_names"]],
        ),
    )
    return p


def read_column_sites(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The sites' table at path (a list of rows, typed; n/a is None) and its
    sidecar."""
    side = read_sidecar(sidecar_path(path), COLUMN_SITES_SPEC)
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    if tuple(head) != COLUMNS:
        raise ValueError(f"{path.name}: columns {head} are not {list(COLUMNS)}")
    rows = []
    for line in lines[1:]:
        d: dict[str, Any] = dict(zip(head, line.split("\t"), strict=True))
        for k in COLUMNS:
            if d[k] == "n/a":
                d[k] = None
            elif k in _INTS:
                d[k] = int(d[k])
            elif k in _NUMS:
                d[k] = float(d[k])
        rows.append(d)
    return rows, side


def read_members(path: Path) -> tuple[Int, Int]:
    """The members table at path: the vertex of each column and its site's
    rank."""
    with path.open() as fh:
        head = fh.readline().rstrip("\n").split("\t")
    if tuple(head) != MEMBER_COLUMNS:
        raise ValueError(f"{path.name}: columns are not {list(MEMBER_COLUMNS)}")
    a = np.loadtxt(path, np.int64, skiprows=1, ndmin=2).reshape(-1, 2)
    return a[:, 0], a[:, 1]


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    log_to_stderr()
    ap = argparse.ArgumentParser(description="the review sites of the column stage")
    ap.add_argument("surf", type=Path, help="a hemi-<h>_white.surf.gii")
    ap.add_argument("conf", type=Path, help="the stage's top confidence (.npz)")
    ap.add_argument("feat", type=Path, nargs="+", help="the stage's feature files")
    ap.add_argument(
        "--out", type=Path, help="the directory to write to (default: the surface's)"
    )
    ap.add_argument("--radius", type=float, default=RADIUS, help="mm (default 2)")
    a = ap.parse_args(argv)
    for role, path in write_column_sites(
        a.surf, a.conf, a.feat, a.out, a.radius
    ).items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
