"""Full-band presence along a window: where each record is audible, and where it stands still.

Per `CHUNK_S` chunk (hop `HOP_S`) of the window the fetcher's fingerprint is matched against each
record two ways. The free best offset, with the matched record time, says where in the record the
chunk sounds like. The anchored count, only offsets within `TOL_S` of where the record should be, says
whether the record is playing here at the known alignment. Presence is the anchored count clearing a
floor two chunks in a row. A loop is a run of chunks where the record is present but its matched time
does not advance with the mix.

The incoming record's first present chunk and the outgoing record's last present chunk give the seam's
overlap in seconds, full band. Validated on 60 Raveform seams against the alignment: both records found
on 56, anchor lag median 0.06 s.
"""

import numpy as np
import pandas as pd

from .fingerprint import FRAME_S, SR, fingerprint

CHUNK_S, HOP_S = 10.0, 2.5
PERCENTILE = 85.0
TOL_S = 0.5


def free_votes(chunk_fp: dict, track_fp: dict) -> tuple[int, float]:
    """(votes at the best offset, the record time that offset puts the chunk's start at)."""
    offsets = [
        tf - cf
        for h, chunk_frames in chunk_fp.items()
        if h in track_fp
        for cf in chunk_frames
        for tf in track_fp[h]
    ]
    if not offsets:
        return 0, 0.0
    values, counts = np.unique(np.array(offsets), return_counts=True)
    best = int(np.argmax(counts))
    return int(counts[best]), float(values[best] * FRAME_S)


def anchored_votes(
    chunk_fp: dict, track_fp: dict, expected_t: float, tol_s: float = TOL_S
) -> tuple[int, float]:
    earliest, latest = expected_t - tol_s, expected_t + tol_s
    offsets = [
        tf - cf
        for h, chunk_frames in chunk_fp.items()
        if h in track_fp
        for cf in chunk_frames
        for tf in track_fp[h]
        if earliest <= (tf - cf) * FRAME_S <= latest
    ]
    if not offsets:
        return 0, expected_t
    values, counts = np.unique(np.array(offsets), return_counts=True)
    best = int(np.argmax(counts))
    return int(counts[best]), float(values[best] * FRAME_S)


def curves(mix: np.ndarray, track_fps: dict, anchors: dict) -> pd.DataFrame:
    """Per chunk: t, and per record name its free votes, matched time, anchored votes, anchored time
    and expected time. anchors[name] = (track time at anchor, mix time at anchor), records at mix speed."""
    rows = []
    chunk, hop = int(CHUNK_S * SR), int(HOP_S * SR)
    for start in range(0, len(mix) - chunk + 1, hop):
        chunk_fp = fingerprint(mix[start : start + chunk], PERCENTILE)
        t = start / SR
        row = {"t": t}
        for name, track_fp in track_fps.items():
            track_t, mix_t = anchors[name]
            expected = track_t + (t - mix_t)
            row[name], row[f"{name}_t"] = free_votes(chunk_fp, track_fp)
            row[f"{name}_anch"], row[f"{name}_anch_t"] = anchored_votes(
                chunk_fp, track_fp, expected
            )
            row[f"{name}_exp_t"] = expected
        rows.append(row)
    return pd.DataFrame(rows)


def present(curve: pd.DataFrame, name: str, floor: int) -> np.ndarray:
    """Chunks where the record's anchored votes clear the floor with a neighbour that does too."""
    above = (curve[f"{name}_anch"] >= floor).to_numpy()
    left = np.concatenate([[False], above[:-1]])
    right = np.concatenate([above[1:], [False]])
    return above & (left | right)


def first_present(curve: pd.DataFrame, name: str, floor: int) -> float | None:
    idx = np.nonzero(present(curve, name, floor))[0]
    return float(curve.t.iloc[idx[0]]) if len(idx) else None


def last_present(curve: pd.DataFrame, name: str, floor: int) -> float | None:
    """The centre of the last present chunk. A chunk half covered by the record still clears the
    floor, so its end overshoots: on seven mixer renders with the overlap end known (2026-10-09) the
    chunk end read 4 s (2.3 bars) late on every one, the centre 1 s early."""
    idx = np.nonzero(present(curve, name, floor))[0]
    return float(curve.t.iloc[idx[-1]]) + CHUNK_S / 2 if len(idx) else None


def here_at(curve: pd.DataFrame, name: str, floor: int, times) -> np.ndarray:
    """Whether the record is present at each of `times` (another grid), nearest chunk."""
    flags = present(curve, name, floor)
    near = np.searchsorted(curve.t.to_numpy(), np.asarray(times), side="right") - 1
    return flags[np.clip(near, 0, len(flags) - 1)]
