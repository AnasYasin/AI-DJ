"""Clips for Anas's ear, built from the tables so what he hears is what the numbers say.

One stereo file per seam. Left ear: the cut window as the mix has it. Right ear: record A and record
B from our own files, each placed at the time zero and speed the locate found, summed. In sync the
right ear is the mix without the DJ's EQ and effects; out of sync one record fights the other.
`MARKS.csv` beside the clips carries the measured entry, exit, bass swap and label per clip and empty
verdict columns for him to fill. His ear is the ground truth; nothing here counts as proof.
"""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from .fingerprint import SR, at_mix_speed, load

MARKS = [
    "n",
    "file",
    "seam_id",
    "dj",
    "track_a",
    "track_b",
    "window_s",
    "in_s",
    "out_s",
    "bass_swap_s",
    "label",
    "a_confidence",
    "b_confidence",
    "audit_start_ok",
    "audit_end_ok",
    "your_verdict_a_aligned",
    "your_verdict_b_aligned",
    "your_verdict_transition_inside",
    "your_verdict_label_right",
    "notes",
]


def record_on_window(
    track: np.ndarray, time_zero_s: float, rate: float, window_t0: float, n: int
) -> np.ndarray:
    """The record as it sounds over the window: at mix speed, its own 0:00 at `time_zero_s`."""
    played = at_mix_speed(track, rate)
    out = np.zeros(n, dtype=np.float32)
    start = int(round((time_zero_s - window_t0) * SR))
    if start >= n or start + len(played) <= 0:
        return out
    src0 = max(0, -start)
    dst0 = max(0, start)
    length = min(len(played) - src0, n - dst0)
    out[dst0 : dst0 + length] = played[src0 : src0 + length]
    return out


def clip(window_path, window_t0: float, a: dict, b: dict, out_path: Path) -> float:
    """Write the stereo clip. a, b: {path, time_zero_s, rate}. Returns its length in seconds."""
    left = load(window_path)
    n = len(left)
    right = record_on_window(load(a["path"]), a["time_zero_s"], a["rate"], window_t0, n)
    right += record_on_window(load(b["path"]), b["time_zero_s"], b["rate"], window_t0, n)
    peak = max(np.abs(left).max(), np.abs(right).max(), 1e-9)
    stereo = np.stack([left, right], axis=1) * (0.9 / peak)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, stereo, SR)
    return n / SR


def write_marks(out_dir: Path, rows: list[dict]) -> Path:
    path = Path(out_dir) / "MARKS.csv"
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=MARKS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in MARKS})
    readme = Path(out_dir) / "README.txt"
    readme.write_text(
        "Left ear: the cut window, as the mix has it.\n"
        "Right ear: record A and record B from our files, placed at the time zero and speed the locate found.\n"
        "Right: the right ear is the mix without the DJ's EQ and effects, in time. Wrong: one record fights the other.\n"
        "Mark per clip in MARKS.csv: A aligned yes/no/drifts, B aligned yes/no/drifts, transition inside yes/no,\n"
        "label right yes/no, notes. in_s, out_s and bass_swap_s are seconds into the clip.\n"
    )
    return path
