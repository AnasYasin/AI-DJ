"""Whose lows, mids and highs are in the mix, step by step, from a fingerprint made inside one band.

Per band a fingerprint is built from peaks inside that band only (low on a fine 2.7 Hz grid, mid and
high on the normal grid), and votes are counted at the known alignment: for every 4 s step of the
window, the record's expected position is known from its time zero and rate, and only offsets within
`TOL_S` of it count. A wrong record scores a handful, the right one scores tens to hundreds when its
band is open and nothing when the DJ has it cut. So per band the first step the incoming record's band
is seen and the last step the outgoing record's is seen give the EQ order at the change.

Measured on 30 Raveform seams with three wrong-record controls each: separations 15 (low), 67 (mid),
24 (high) times the control floor. Mids enter before bass on 13 of 16, bass leaves before highs on
13 of 18. The floor is never a constant here; the caller derives it from controls per seam per band.
"""

from collections import defaultdict

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.signal import butter, sosfiltfilt

from .fingerprint import FRAME_S, HOP, SR

BANDS = {  # name: (low Hz, high Hz, n_fft)
    "low": (20.0, 200.0, 8192),
    "mid": (200.0, 3000.0, 2048),
    "high": (3000.0, 14000.0, 2048),
}
PEAK_NEIGHBOURHOOD = (7, 9)
PEAK_PERCENTILE = 80.0
FAN_OUT, GAP_MIN, GAP_MAX = 10, 1, 60
STEP_S, HOP_S, TOL_S = 4.0, 1.0, 0.25


def band_fp(audio: np.ndarray, band: str, percentile: float = PEAK_PERCENTILE) -> dict:
    """hash -> frames, from peaks inside the band only."""
    low_hz, high_hz, n_fft = BANDS[band]
    spec_db = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=HOP)))
    bin_hz = SR / n_fft
    part = spec_db[int(low_hz / bin_hz) : int(high_hz / bin_hz)]
    return peak_pairs(part, percentile)


def peak_pairs(spec_db: np.ndarray, percentile: float) -> dict:
    """The landmark table of one spectrogram slab: hash (bin, bin, gap) -> frames."""
    local_max = maximum_filter(spec_db, size=PEAK_NEIGHBOURHOOD, mode="nearest")
    peaks = (spec_db == local_max) & (spec_db > np.percentile(spec_db, percentile))
    freqs, frames = np.nonzero(peaks)
    order = np.argsort(frames)
    freqs, frames = freqs[order], frames[order]
    table = defaultdict(list)
    n = len(frames)
    for i in range(n):
        for j in range(i + 1, min(i + 1 + FAN_OUT, n)):
            gap = int(frames[j] - frames[i])
            if gap < GAP_MIN:
                continue
            if gap > GAP_MAX:
                break
            table[(int(freqs[i]), int(freqs[j]), gap)].append(int(frames[i]))
    return table


def anchored_votes(
    chunk_fp: dict, track_fp: dict, expected_track_time: float, tol_s: float = TOL_S
) -> int:
    """Votes in the single best offset bin within `tol_s` of where the record should be."""
    earliest, latest = expected_track_time - tol_s, expected_track_time + tol_s
    offsets = [
        tf - cf
        for h, chunk_frames in chunk_fp.items()
        if h in track_fp
        for cf in chunk_frames
        for tf in track_fp[h]
        if earliest <= (tf - cf) * FRAME_S <= latest
    ]
    if not offsets:
        return 0
    _, counts = np.unique(np.array(offsets), return_counts=True)
    return int(counts.max())


def votes_along(mix: np.ndarray, track_fps: dict, anchors: dict, fp_of) -> pd.DataFrame:
    """Per STEP_S step (hop HOP_S) of `mix`: anchored votes for every named record.
    anchors[name] = (track time at anchor, mix time at anchor), records already at mix speed.
    `fp_of(chunk)` builds the chunk's table in the same band as the records' tables."""
    rows = []
    step, hop = int(STEP_S * SR), int(HOP_S * SR)
    for start in range(0, len(mix) - step + 1, hop):
        chunk_fp = fp_of(mix[start : start + step])
        t = start / SR
        row = {"t": t}
        for name, track_fp in track_fps.items():
            track_t, mix_t = anchors[name]
            row[name] = anchored_votes(chunk_fp, track_fp, track_t + (t - mix_t))
        rows.append(row)
    return pd.DataFrame(rows)


def band_votes(mix: np.ndarray, band: str, track_fps: dict, anchors: dict) -> pd.DataFrame:
    return votes_along(mix, track_fps, anchors, lambda chunk: band_fp(chunk, band))


def first_last(curves: pd.DataFrame, name: str, floor: int) -> tuple[float | None, float | None]:
    """First step where the record is seen two steps in a row, and the end of the last such step."""
    seen = (curves[name] >= floor).to_numpy()
    two = seen[:-1] & seen[1:]
    if not two.any():
        return None, None
    idx = np.nonzero(two)[0]
    return float(curves.t.iloc[idx[0]]), float(curves.t.iloc[idx[-1] + 1]) + STEP_S


def band_filter(band: str):
    low_hz, high_hz, _ = BANDS[band]
    if band == "low":
        return butter(4, high_hz, btype="low", fs=SR, output="sos")
    if band == "high":
        return butter(4, low_hz, btype="high", fs=SR, output="sos")
    return butter(4, [low_hz, high_hz], btype="band", fs=SR, output="sos")


def band_level_db(audio: np.ndarray, band: str, start_s: float, n_steps: int) -> np.ndarray:
    """RMS in dB of the band per STEP_S step from `start_s`, nan where the audio runs out."""
    filtered = sosfiltfilt(band_filter(band), audio)
    step = int(STEP_S * SR)
    out = []
    for k in range(n_steps):
        first = int(round((start_s + k * HOP_S) * SR))
        piece = filtered[max(first, 0) : first + step] if first + step > 0 else np.zeros(0)
        ok = len(piece) >= step // 2
        out.append(20 * np.log10(np.sqrt(np.mean(piece**2)) + 1e-9) if ok else np.nan)
    return np.array(out)
