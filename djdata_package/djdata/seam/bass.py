"""Whose bass is in the mix, where it swaps, and where the DJ cut it or added it.

The swap. A fingerprint of kick fundamentals and bass notes only (peaks below `LOW_HZ` on a 2.7 Hz grid)
is counted at the known alignment per 4 s step. Two records almost never share bass pitches, so the
piles separate where low-band energy could not. The bass swap is the first step where the incoming
record's low votes stay above the outgoing record's for two steps, after the outgoing record's bass had
been seen. Found on 24 of 25 Raveform seams with a control of 3 or less.

The words (Anas's framing, 2026-09-17). DJs mostly leave the bass alone at a seam, sometimes cut it on
purpose to build tension and bring it back on the drop, and some keep two records running with bands cut
for minutes. Per step the mix's low level, each record's low level at the spot it is playing, and the
low-band votes give one word:

    outgoing | both | incoming   bass in the mix, attributed
    cut       no bass in the mix while a present record has bass there (tension)
    extra     bass in the mix while neither record has bass there (added bass)
    break     no bass anywhere (both records in a breakdown)
    unsure    bass in the mix, a record has bass there, the fingerprint could not say whose
"""

import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .bands import HOP_S, STEP_S, peak_pairs, votes_along
from .fingerprint import HOP, SR

LOW_HZ = 250.0
NFFT = 8192
PEAK_PERCENTILE = 75.0
LEVEL_HZ = 200.0  # the low band whose level says whether there is bass at all
BASS_DROP_DB = 8.0  # this far under the reference counts as no bass
WORDS = ("outgoing", "both", "incoming", "cut", "extra", "break", "unsure")
_LOW_SOS = butter(4, LEVEL_HZ, btype="low", fs=SR, output="sos")


def low_band_fp(audio: np.ndarray) -> dict:
    """hash -> frames from peaks below LOW_HZ only."""
    spec_db = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=NFFT, hop_length=HOP)))
    return peak_pairs(spec_db[: int(LOW_HZ / (SR / NFFT))], PEAK_PERCENTILE)


def low_band_votes(mix: np.ndarray, track_fps: dict, anchors: dict) -> pd.DataFrame:
    return votes_along(mix, track_fps, anchors, low_band_fp)


def bass_swap(curves: pd.DataFrame, floor: int) -> float | None:
    """The second (step centre) where B's low votes first stay above A's for two steps, after A's bass
    had been seen. None when either record's bass never clears the floor."""
    if not ((curves.A >= floor).any() and (curves.B >= floor).any()):
        return None
    b_leads = ((curves.B > curves.A) & (curves.B >= floor)).to_numpy()
    a_seen = (curves.A >= floor).to_numpy()
    for i in range(len(curves) - 1):
        if b_leads[i] and b_leads[i + 1] and a_seen[:i].any():
            return float(curves.t.iloc[i]) + STEP_S / 2
    return None


def low_level_db(audio: np.ndarray, start_s: float, n_steps: int) -> np.ndarray:
    """Low-band RMS in dB per step from `start_s`, nan where the audio runs out."""
    low = sosfiltfilt(_LOW_SOS, audio)
    step = int(STEP_S * SR)
    out = []
    for k in range(n_steps):
        first = int(round((start_s + k * HOP_S) * SR))
        piece = low[max(first, 0) : first + step] if first + step > 0 else np.zeros(0)
        ok = len(piece) >= step // 2
        out.append(20 * np.log10(np.sqrt(np.mean(piece**2)) + 1e-9) if ok else np.nan)
    return np.array(out)


def has_bass(level_db: np.ndarray, reference_db: float) -> np.ndarray:
    return level_db >= reference_db - BASS_DROP_DB


def words(
    mix_has_bass, a_here, b_here, a_has_bass, b_has_bass, a_votes_ok, b_votes_ok
) -> list[str]:
    """One word per step. All inputs are boolean arrays of one length: whether the mix has bass, whether
    each record is playing here, whether each record has bass at the spot it is playing, and whether
    each record's low-band votes clear the floor."""
    out = []
    for i in range(len(mix_has_bass)):
        a_expected = bool(a_here[i] and a_has_bass[i])
        b_expected = bool(b_here[i] and b_has_bass[i])
        if not mix_has_bass[i]:
            out.append("cut" if (a_expected or b_expected) else "break")
        elif a_votes_ok[i] and b_votes_ok[i]:
            out.append("both")
        elif a_votes_ok[i]:
            out.append("outgoing")
        elif b_votes_ok[i]:
            out.append("incoming")
        elif not a_expected and not b_expected:
            out.append("extra")
        else:
            out.append("unsure")
    return out


def segments(times, word_list) -> list[tuple[float, float, str]]:
    """Runs of one word as (start, end, word), the end being the last step's end."""
    out, start = [], 0
    for i in range(1, len(word_list) + 1):
        if i == len(word_list) or word_list[i] != word_list[start]:
            out.append((float(times[start]), float(times[i - 1]) + STEP_S, word_list[start]))
            start = i
    return out


def seconds_per_word(word_list) -> dict:
    return {w: round(sum(1 for x in word_list if x == w) * HOP_S, 1) for w in WORDS}


def longest_run(times, word_list, word: str) -> float:
    runs = [end - start for start, end, w in segments(times, word_list) if w == word]
    return round(max(runs), 1) if runs else 0.0
