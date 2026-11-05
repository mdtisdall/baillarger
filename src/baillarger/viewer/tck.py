"""The MRtrix .tck track format, which NiiVue reads: the viewer's writer of
polylines (columns, section curves) for the 3D views and for export, and its
reader for tests. It reads no data files."""

import numpy as np
import numpy.typing as npt

type Float = npt.NDArray[np.float64]


def tck(lines: list[Float]) -> bytes:
    """MRtrix .tck bytes (float32 little-endian) of polylines in mm: a row of nan, then
    each line's points and a row of nan; an inf row ends the file. NiiVue (0.69) takes
    each nan row as the end of a line and the first offset as 0 only after a nan row:
    without the leading one it draws every line but the first (none of one line)."""
    body = [np.full((1, 3), np.nan, np.float32)]
    for p in lines:
        body.append(np.asarray(p, np.float32).reshape(-1, 3))
        body.append(np.full((1, 3), np.nan, np.float32))
    body.append(np.full((1, 3), np.inf, np.float32))
    head = f"mrtrix tracks\ndatatype: Float32LE\ncount: {len(lines)}\nfile: . "
    off = len(head) + 20
    hdr = f"{head}{off}\nEND\n".ljust(off).encode()
    return hdr + np.vstack(body).astype("<f4").tobytes()


def read_tck(data: bytes) -> list[Float]:
    """The polylines of .tck bytes as tck writes them (a leading nan row starts no
    line)."""
    head = data[: data.index(b"END\n")].decode()
    off = int(next(x for x in head.splitlines() if x.startswith("file:")).split()[-1])
    a = np.frombuffer(data[off:], "<f4").reshape(-1, 3).astype(float)
    if len(a) and np.isnan(a[0]).any():
        a = a[1:]
    out: list[Float] = []
    cur: list[Float] = []
    for row in a:
        if np.isinf(row).any():
            break
        if np.isnan(row).any():
            out.append(np.array(cur).reshape(-1, 3))
            cur = []
        else:
            cur.append(row)
    return out
