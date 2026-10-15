"""Seconds per bar of a record, from its own audio.

Two steps. A coarse tempo from librosa's onset autocorrelation over the middle of the record gives a
hint. The mixer's tempo measurement (`src.audio.audio_mixer._measure_tempo`, the kick-envelope
autocorrelation at the one-bar lag with a parabolic peak) then refines it within its search width.
That function is the one tempo instrument in the repo that has been checked by ear, on real renders;
the coarse hint only picks the octave. When the refinement finds no periodic peak there is no tempo and
no bars, never a guess.

Tempo is measured on the record at its own speed. The bar length in the mix is the record's bar
divided by the playback rate.
"""

from pathlib import Path
import sys

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio import audio_mixer as mixer  # noqa: E402

MIN_BPM, MAX_BPM = 60.0, 200.0
HINT_WINDOW_S = 60.0


def coarse_bpm(audio: np.ndarray, sr: int) -> float | None:
    """librosa's tempo over the middle HINT_WINDOW_S, folded into [MIN_BPM, MAX_BPM]."""
    n = int(HINT_WINDOW_S * sr)
    if len(audio) > n:
        start = (len(audio) - n) // 2
        audio = audio[start : start + n]
    if len(audio) < sr * 10:
        return None
    onset = librosa.onset.onset_strength(y=audio, sr=sr)
    bpm = float(librosa.feature.tempo(onset_envelope=onset, sr=sr, start_bpm=128.0)[0])
    if not np.isfinite(bpm) or bpm <= 0:
        return None
    while bpm < MIN_BPM:
        bpm *= 2
    while bpm > MAX_BPM:
        bpm /= 2
    return bpm


def bpm_of(path) -> float | None:
    """The record's BPM at its own speed, or None when no periodic kick was found."""
    audio, _ = librosa.load(str(path), sr=mixer.SR, mono=True)
    hint = coarse_bpm(audio, mixer.SR)
    if hint is None:
        return None
    measured = mixer._measure_tempo(audio, hint)
    if measured is None:
        return None
    return round(float(measured), 3)


def bar_seconds(bpm: float | None, rate: float) -> float | None:
    """Seconds per 4/4 bar in the mix, for a record with `bpm` played `rate` times faster."""
    if bpm is None:
        return None
    return round(4 * 60.0 / bpm / rate, 4)
