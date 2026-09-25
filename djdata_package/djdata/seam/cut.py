"""Cut one window per seam out of the mix, and check by fingerprint that it holds what it should.

The cut is an ffmpeg stream copy between the pair's window bounds: no re-encode, exact to the frame
(149 of 149 within 0.026 s on the earlier bounded cut). The audit then asks the one question that
matters about a window: does it start with A alone and end with B alone. The first and last
`AUDIT_S` seconds are fingerprinted, and each record's slice that should be playing there is looked
up in them. A control record goes through the same path so the yes or no is against a measured floor.
Ten seconds is enough to say whether one record is there, and a longer window would fail on a DJ who
plays records back to back.
"""

from pathlib import Path
import subprocess

import numpy as np

from . import floors
from .fingerprint import SR, as_arrays, at_mix_speed, fingerprint, load, match

AUDIT_S = 10.0
SLICE_PAD_S = (
    5.0  # the record slice is this much longer either side, so a small drift still matches
)


def existing_window(folder: Path, seam_id: str) -> Path | None:
    """A window already on disk for this seam, whatever its extension."""
    hits = sorted(p for p in Path(folder).glob(f"{seam_id}.*") if p.suffix != ".part")
    return hits[0] if hits else None


def duration_s(path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


def cut_window(src, t0: float, t1: float, dest_stem: Path) -> Path:
    """Stream copy [t0, t1) of `src` to `dest_stem` with the source's suffix. Returns the file."""
    dest = Path(str(dest_stem) + Path(src).suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ss",
            f"{t0:.3f}",
            "-to",
            f"{t1:.3f}",
            "-c",
            "copy",
            str(dest),
        ],
        check=True,
    )
    return dest


def record_slice(track: np.ndarray, rate: float, from_s: float, seconds: float) -> np.ndarray:
    """The part of a record that plays over `seconds` of mix time starting at record time `from_s`,
    at mix speed, with SLICE_PAD_S either side. Empty when that lies outside the record."""
    start = max(0.0, from_s - SLICE_PAD_S)
    end = from_s + seconds * rate + SLICE_PAD_S
    if end <= start + 1.0:
        return np.zeros(0, dtype=np.float32)
    piece = track[int(start * SR) : int(end * SR)]
    if len(piece) < SR:
        return np.zeros(0, dtype=np.float32)
    return at_mix_speed(piece, rate)


def votes_in(window_audio: np.ndarray, piece: np.ndarray) -> int:
    if not len(piece) or not len(window_audio):
        return 0
    votes, _ = match(fingerprint(piece), as_arrays(fingerprint(window_audio)))
    return votes


def audit(
    window_path,
    window_t0: float,
    a: dict,
    b: dict,
    a_track: np.ndarray,
    b_track: np.ndarray,
    controls: list[np.ndarray],
) -> dict:
    """a, b: {time_zero_s, rate}. Returns votes for A and B at the start and the end of the window,
    the control floor, and whether the start is A alone and the end B alone."""
    length = duration_s(window_path)
    head = load(window_path, offset=0.0, duration=AUDIT_S)
    tail = load(window_path, offset=max(0.0, length - AUDIT_S), duration=AUDIT_S)
    t_head, t_tail = window_t0, window_t0 + max(0.0, length - AUDIT_S)

    def at(rec: dict, track: np.ndarray, mix_t: float) -> np.ndarray:
        return record_slice(
            track, rec["rate"], (mix_t - rec["time_zero_s"]) * rec["rate"], AUDIT_S
        )

    out = {
        "audit_start_a": votes_in(head, at(a, a_track, t_head)),
        "audit_start_b": votes_in(head, at(b, b_track, t_head)),
        "audit_end_a": votes_in(tail, at(a, a_track, t_tail)),
        "audit_end_b": votes_in(tail, at(b, b_track, t_tail)),
    }
    ctrl = []
    for c in controls:
        piece = c[: int((AUDIT_S + 2 * SLICE_PAD_S) * SR)]
        ctrl.append(max(votes_in(head, piece), votes_in(tail, piece)))
    floor = floors.from_controls(ctrl)
    out["audit_floor"] = floor.floor
    out["audit_control_max"] = floor.control_max
    out["audit_start_ok"] = int(
        floor.clears(out["audit_start_a"]) and not floor.clears(out["audit_start_b"])
    )
    out["audit_end_ok"] = int(
        floor.clears(out["audit_end_b"]) and not floor.clears(out["audit_end_a"])
    )
    return out
