"""The landmark fingerprint every stage shares, and the vote arithmetic on top of it.

One definition. The fetcher, the locator, the presence sweep and the cut audit all count the same thing:
peak pairs shared between a record and a stretch of mix audio, each voting for one time offset. The right
recording piles hundreds to thousands of votes on one offset, a wrong one scatters a handful. The
constants are imported from the fetcher, not copied, so the two cannot drift apart.

The fingerprint is not tempo invariant. A record played 2 % fast scores at the noise floor, so a record is
resampled to the speed the mix plays it at (`at_mix_speed`) before it is matched.

Frames. Everything is in frames of `FRAME_S` seconds (256 samples at 22,050 Hz, 11.6 ms). A mix is
fingerprinted once, ten minutes at a time, into one table (`mix_fingerprint`), and every record is looked
up in that one table (`match`).
"""

from collections import defaultdict

import librosa
import numpy as np
from scipy.ndimage import maximum_filter

from ..legacy.track_fetcher import (
    DT_MAX_FRAMES,
    DT_MIN_FRAMES,
    FAN_OUT,
    PEAK_NEIGHBOURHOOD,
    PEAK_PERCENTILE,
    VERIFY_HOP,
    VERIFY_NFFT,
    VERIFY_SR,
)

SR = VERIFY_SR
HOP = VERIFY_HOP
FRAME_S = HOP / SR
CHUNK_S = (
    600.0  # a mix is fingerprinted ten minutes at a time to keep one worker under about 1.5 GB
)


def load(path, offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Mono float32 audio at the fingerprint rate."""
    audio, _ = librosa.load(str(path), sr=SR, mono=True, offset=offset, duration=duration)
    return audio.astype(np.float32)


def at_mix_speed(track: np.ndarray, rate: float) -> np.ndarray:
    """The record as the mix plays it: `rate` times faster, every frequency `rate` times higher."""
    if abs(rate - 1.0) < 1e-6:
        return track
    return librosa.resample(track, orig_sr=int(round(SR * rate)), target_sr=SR)


def fingerprint(audio: np.ndarray, percentile: float = PEAK_PERCENTILE) -> dict:
    """Hash -> list of frames where a peak pair with that hash starts. The fetcher's maths, unchanged.
    `percentile` is the fetcher's by default; the presence curves use a lower one on short chunks."""
    spec_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=VERIFY_NFFT, hop_length=HOP))
    )
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
            if gap < DT_MIN_FRAMES:
                continue
            if gap > DT_MAX_FRAMES:
                break
            table[(freqs[i] // 2, freqs[j] // 2, gap)].append(int(frames[i]))
    return table


def as_arrays(table: dict) -> dict:
    """The same table with numpy frame arrays, the form `pairs` wants."""
    return {k: np.asarray(v, dtype=np.int64) for k, v in table.items()}


def mix_fingerprint(path) -> tuple[dict, float]:
    """The whole mix as one hash table of frame arrays, built in CHUNK_S pieces. Also its length in s."""
    table = defaultdict(list)
    start = 0.0
    while True:
        chunk = load(path, offset=start, duration=CHUNK_S)
        if len(chunk) < SR:
            break
        frame0 = int(round(start / FRAME_S))
        for key, frames in fingerprint(chunk).items():
            table[key].extend(f + frame0 for f in frames)
        if len(chunk) < CHUNK_S * SR - SR:
            start += len(chunk) / SR
            break
        start += CHUNK_S
    return as_arrays(table), start


def pairs(track_fp: dict, mix_fp: dict) -> tuple[np.ndarray, np.ndarray]:
    """Every shared hash as (mix frame, offset). The offset is the mix frame the record's frame zero
    falls on, so all true pairs of one playing agree on it."""
    mf, off = [], []
    for key, track_frames in track_fp.items():
        mix_frames = mix_fp.get(key)
        if mix_frames is None:
            continue
        tf = np.asarray(track_frames, dtype=np.int64)
        grid_m = np.broadcast_to(mix_frames[None, :], (len(tf), len(mix_frames))).ravel()
        grid_o = (mix_frames[None, :] - tf[:, None]).ravel()
        mf.append(grid_m)
        off.append(grid_o)
    if not mf:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(mf), np.concatenate(off)


def match(track_fp: dict, mix_fp: dict) -> tuple[int, int]:
    """(votes at the best offset, that offset in mix frames). One vote per shared pair on one offset."""
    _, offsets = pairs(track_fp, mix_fp)
    if not len(offsets):
        return 0, 0
    values, counts = np.unique(offsets, return_counts=True)
    best = int(np.argmax(counts))
    return int(counts[best]), int(values[best])


def votes_near(track_fp: dict, mix_fp: dict, offset: int, tolerance: int = 2) -> int:
    """Votes that land within `tolerance` frames of a known offset. The audit's question: is this record
    here, at the place the locator said, and nowhere else is asked."""
    _, offsets = pairs(track_fp, mix_fp)
    if not len(offsets):
        return 0
    return int(np.count_nonzero(np.abs(offsets - offset) <= tolerance))
