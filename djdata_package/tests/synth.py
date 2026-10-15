"""Synthetic audio with a known answer, for every test that needs a record and a mix.

A record is a run of random chords: every `note_s` seconds six sinusoids at random frequencies between
80 and 8,000 Hz with random amplitudes, with a short fade at each change so the spectrum has clear
peaks in time and frequency, which is what the landmark fingerprint keys on. Two seeds give two records
that share nothing. A mix is records placed at known mix times and speeds, summed, so a test knows the
time zero, the rate and the played part before the locator runs.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from djdata.seam.fingerprint import SR, at_mix_speed

NOTE_S = 0.25
VOICES = 6


def record(seconds: float, seed: int, sr: int = SR, bass_voices: int = 0) -> np.ndarray:
    """With `bass_voices`, that many extra tones between 40 and 150 Hz per note, so the low band has
    peaks of its own for the bass fingerprint and the level reading."""
    rng = np.random.default_rng(seed)
    n_notes = int(np.ceil(seconds / NOTE_S))
    note_n = int(NOTE_S * sr)
    fade = np.linspace(0.0, 1.0, int(0.01 * sr))
    env = np.ones(note_n)
    env[: len(fade)] = fade
    env[-len(fade) :] = fade[::-1]
    t = np.arange(note_n) / sr
    out = np.zeros(n_notes * note_n, dtype=np.float32)
    for k in range(n_notes):
        freqs = np.concatenate(
            [rng.uniform(80.0, 8000.0, VOICES), rng.uniform(40.0, 150.0, bass_voices)]
        )
        amps = np.concatenate([rng.uniform(0.2, 1.0, VOICES), rng.uniform(0.6, 1.0, bass_voices)])
        chord = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
        out[k * note_n : (k + 1) * note_n] = chord * env
    out = out[: int(seconds * sr)]
    return (0.3 * out / np.abs(out).max()).astype(np.float32)


def mix(placements: list[tuple], seconds: float, sr: int = SR) -> np.ndarray:
    """placements: [(record, time_zero_s, rate)] or [(record, time_zero_s, rate, heard_from_record_s)].
    A record's own 0:00 falls at time_zero_s. With heard_from_record_s the DJ cued in deep: nothing of
    the record before that record time is in the mix. Everything past `seconds` is dropped."""
    out = np.zeros(int(seconds * sr), dtype=np.float32)
    for placement in placements:
        rec, zero_s, rate = placement[:3]
        heard_from = placement[3] if len(placement) > 3 else 0.0
        played = at_mix_speed(rec, rate)
        skip = int(round(heard_from / rate * sr))
        played = played[skip:]
        start = int(round(zero_s * sr)) + skip
        if start < 0:
            played = played[-start:]
            start = 0
        end = min(start + len(played), len(out))
        out[start:end] += played[: end - start]
    return out


def split_low(audio: np.ndarray, cutoff_hz: float = 200.0, sr: int = SR):
    """(low band, the rest) so a test can cut or open one record's bass at a known second."""
    low = butter(4, cutoff_hz, btype="low", fs=sr, output="sos")
    high = butter(4, cutoff_hz, btype="high", fs=sr, output="sos")
    return sosfiltfilt(low, audio).astype(np.float32), sosfiltfilt(high, audio).astype(np.float32)


def ramp(n: int, at_s: float, seconds: float = 0.5, sr: int = SR) -> np.ndarray:
    """0 before `at_s`, 1 after it, linear over `seconds`."""
    t = np.arange(n) / sr
    return np.clip((t - at_s) / seconds, 0.0, 1.0).astype(np.float32)
