"""Tests of baillarger.viewer.tck, with exact answers."""

import numpy as np

from baillarger.viewer import tck as SV


def test_tck_round_trip() -> None:
    lines = [np.array([[0.0, 1, 2], [3, 4, 5]]), np.array([[-1.5, 0, 0.25]])]
    back = SV.read_tck(SV.tck(lines))
    assert len(back) == 2
    assert all(np.array_equal(a, b) for a, b in zip(lines, back, strict=True))
    assert SV.tck([]).startswith(b"mrtrix tracks\n")
    assert SV.read_tck(SV.tck([])) == []


def header_start_offsets(data: bytes) -> list[int]:
    """The line ends that a reader finds that starts the first line at the header
    (offset 0), then writes the point count at every non-finite row from the first
    slot, and stops at a row that is not nan; it draws the points between
    consecutive offsets. (NiiVue 0.69's readTCK loop, transcribed.)"""
    head = data[: data.index(b"END\n")].decode()
    off = int(next(x for x in head.splitlines() if x.startswith("file:")).split()[-1])
    a = np.frombuffer(data[off:], "<f4").reshape(-1, 3)
    h, u, n = [0], 0, 0
    for x in a[:, 0]:
        if np.isfinite(x):
            n += 1
            continue
        if u < len(h):
            h[u] = n
        else:
            h.append(n)
        u += 1
        if not np.isnan(x):
            break
    return h[:u]


def test_a_reader_that_starts_at_the_header_gets_every_line() -> None:
    lines = [np.zeros((4, 3)), np.ones((3, 3)), np.ones((2, 3))]
    h = header_start_offsets(SV.tck(lines))
    drawn = [b - a for a, b in zip(h[:-1], h[1:], strict=True) if b > a]
    assert drawn == [4, 3, 2]
    assert header_start_offsets(SV.tck(lines[:1]))[:2] == [0, 4]
