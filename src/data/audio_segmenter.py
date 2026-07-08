"""
Phase 7 — structural segmentation of full tracks (inference-time).

Segments a full track into DJ-relevant sections and returns a bar-aligned map:

    {"bpm": 126.0,
     "beats": [0.0, 0.476, ...],          # seconds
     "bars": [0.0, 1.905, ...],           # downbeat times, seconds
     "sections": [
        {"label": "intro",     "start": 0.0,   "end": 30.5,  "bars": [0, 16]},
        {"label": "buildup",   "start": 30.5,  "end": 45.7,  "bars": [16, 24]},
        {"label": "drop",      "start": 45.7,  "end": 106.7, "bars": [24, 56]},
        {"label": "breakdown", ...},
        {"label": "outro",     ...}]}

Method:
  1. BPM: DeepRhythm CNN (same engine as the feature pipeline).
  2. Beat grid: librosa beat tracker locked to the DeepRhythm tempo;
     downbeats = 4-beat phase with maximum kick-band (20-150 Hz) energy.
  3. Per-bar features: RMS, onset density, spectral flux, kick energy,
     harmonic ratio, spectral centroid.
  4. Boundaries: checkerboard-novelty on the bar-level self-similarity matrix,
     snapped to 4-bar lines (phrases).
  5. Labels: heuristics over section-mean features (see _label_sections).

Only detected sections are included; `bpm`, `beats`, `bars` are always present.
Designed for 4/4 electronic music — the inference pool (Jamendo) and all six
training genres are 4/4.
"""

import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

log = logging.getLogger(__name__)

SR = 22_050
PHRASE_BARS = 4          # boundaries snap to multiples of this
MIN_SECTION_BARS = 8     # merge sections shorter than this
KICK_BAND = (20, 150)    # Hz


# ── Beat / bar grid ────────────────────────────────────────────────────────────


def _bpm(y: np.ndarray, sr: int) -> float:
    """DeepRhythm BPM with librosa fallback."""
    try:
        from deeprhythm import DeepRhythmPredictor

        if not hasattr(_bpm, "_model"):
            _bpm._model = DeepRhythmPredictor()
        bpm = float(_bpm._model.predict_from_audio(y, sr))
        if 60 <= bpm <= 200:
            return bpm
    except Exception as e:  # model missing/failed — librosa is good enough for 4/4
        log.warning("DeepRhythm unavailable (%s) — falling back to librosa", e)
    return float(
        librosa.feature.tempo(y=y, sr=sr, aggregate=np.median, start_bpm=128)[0]
    )


def _beat_grid(y: np.ndarray, sr: int, bpm: float) -> tuple[np.ndarray, np.ndarray]:
    """Beat times locked to the known tempo, downbeats by kick-band phase."""
    _, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, start_bpm=bpm, tightness=400, trim=False, units="frames"
    )
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beats) < 8:
        raise ValueError("too few beats detected")

    # kick-band onset energy at each beat → best of 4 downbeat phases
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= KICK_BAND[0]) & (freqs < KICK_BAND[1])
    kick = S[band].sum(axis=0)
    kick_at_beat = kick[np.clip(beat_frames, 0, len(kick) - 1)]
    phase = max(range(4), key=lambda p: kick_at_beat[p::4].sum())
    bars = beats[phase::4]
    return beats, bars


# ── Bar-level features ─────────────────────────────────────────────────────────


def _bar_features(y: np.ndarray, sr: int, bars: np.ndarray) -> np.ndarray:
    """(n_bars, 6) matrix: rms, onset, flux, kick, harmonic_ratio, centroid."""
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    kick_band = (freqs >= KICK_BAND[0]) & (freqs < KICK_BAND[1])

    rms = librosa.feature.rms(S=S)[0]
    onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
    flux = np.r_[0, np.sqrt(((np.diff(S, axis=1).clip(min=0)) ** 2).sum(axis=0))]
    kick = S[kick_band].sum(axis=0)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    H, P = librosa.decompose.hpss(S)
    harm_ratio = H.sum(axis=0) / (H.sum(axis=0) + P.sum(axis=0) + 1e-9)

    frames = librosa.time_to_frames(bars, sr=sr, hop_length=hop)
    frames = np.r_[frames, S.shape[1]]
    feats = []
    for a, b in zip(frames[:-1], frames[1:]):
        b = max(b, a + 1)
        feats.append(
            [
                rms[a:b].mean(),
                onset[a : min(b, len(onset))].mean(),
                flux[a:b].mean(),
                kick[a:b].mean(),
                harm_ratio[a:b].mean(),
                centroid[a:b].mean(),
            ]
        )
    F = np.array(feats, dtype=np.float32)
    return (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)


# ── Boundary detection ─────────────────────────────────────────────────────────


def _boundaries(F: np.ndarray) -> list[int]:
    """
    Checkerboard novelty on the bar SSM → phrase-snapped boundary bar indices.
    Adaptive count: aim for one section per ~16 bars, taking the strongest
    novelty peaks (a fixed threshold under-segments short/flat tracks).
    """
    n = len(F)
    ssm = F @ F.T / F.shape[1]
    novelty = np.zeros(n)
    for L in (4, 8):  # multi-scale kernels catch both phrase and section changes
        kernel = np.outer(
            np.r_[-np.ones(L), np.ones(L)], np.r_[-np.ones(L), np.ones(L)]
        ) / (4 * L * L)
        pad = np.pad(ssm, L, mode="edge")
        for i in range(n):
            novelty[i] += (pad[i : i + 2 * L, i : i + 2 * L] * kernel).sum()
    novelty = gaussian_filter1d(np.maximum(novelty, 0), 1.5)

    candidates = [
        i
        for i in range(2, n - 2)
        if novelty[i] == novelty[max(0, i - 2) : i + 3].max() and novelty[i] > 0
    ]
    target = max(2, n // 16)
    picked: list[int] = []
    for p in sorted(candidates, key=lambda i: -novelty[i]):
        b = int(round(p / PHRASE_BARS)) * PHRASE_BARS
        if not (MIN_SECTION_BARS <= b <= n - MIN_SECTION_BARS):
            continue
        if all(abs(b - q) >= MIN_SECTION_BARS for q in picked):
            picked.append(b)
        if len(picked) >= target:
            break
    return sorted(picked)


# ── Section labelling ──────────────────────────────────────────────────────────


def _label_sections(F: np.ndarray, bounds: list[int]) -> list[dict]:
    """
    Heuristics over section means (features are z-scored per track):
      drop      — highest energy: rms + kick both high
      intro     — first section, low rms/kick
      outro     — last section, low rms/kick
      breakdown — low kick + high harmonic ratio, mid-track
      buildup   — rising rms/onset leading into a higher-energy section
      groove    — everything else (steady mid-energy body)
    """
    edges = [0] + bounds + [len(F)]
    secs = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = F[a:b].mean(axis=0)
        secs.append({"a": a, "b": b, "rms": m[0], "onset": m[1], "kick": m[3], "harm": m[4]})

    # features are z-scored per track, so absolute thresholds work regardless
    # of how many sections exist (percentiles degenerate with 1-2 sections)
    energy = np.array([s["rms"] + s["kick"] for s in secs])

    labels = []
    for i, s in enumerate(secs):
        e = energy[i]
        nxt = energy[i + 1] if i + 1 < len(secs) else e
        if i == 0 and e < 0.2:
            lab = "intro"
        elif i == len(secs) - 1 and e < 0.2:
            lab = "outro"
        elif s["kick"] < -0.2 and s["harm"] > 0.1 and 0 < i < len(secs) - 1:
            lab = "breakdown"
        elif nxt > e + 0.3 and i < len(secs) - 1:
            lab = "buildup"
        elif e > 0.3:
            lab = "drop"
        else:
            lab = "groove"
        labels.append(lab)
    out = []
    for lab, s in zip(labels, secs):
        if out and out[-1]["label"] == lab:  # merge consecutive same-label sections
            out[-1]["bar_end"] = s["b"]
        else:
            out.append({"label": lab, "bar_start": s["a"], "bar_end": s["b"]})
    return out


# ── Public API ─────────────────────────────────────────────────────────────────


def segment(audio_path: str | Path) -> dict:
    """Full analysis of one track → section map (see module docstring)."""
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    bpm = _bpm(y, sr)
    beats, bars = _beat_grid(y, sr, bpm)
    F = _bar_features(y, sr, bars)
    bounds = _boundaries(F)
    sections = _label_sections(F, bounds)

    dur = len(y) / sr
    bar_times = np.r_[bars, dur]
    for s in sections:
        s["start"] = round(float(bar_times[s["bar_start"]]), 3)
        s["end"] = round(float(bar_times[min(s["bar_end"], len(bar_times) - 1)]), 3)
        s["bars"] = [int(s.pop("bar_start")), int(s.pop("bar_end"))]

    return {
        "bpm": round(bpm, 1),
        "duration": round(dur, 2),
        "n_bars": len(bars),
        "beats": [round(float(t), 3) for t in beats],
        "bars": [round(float(t), 3) for t in bars],
        "sections": sections,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for path in sys.argv[1:]:
        result = segment(path)
        compact = {k: v for k, v in result.items() if k not in ("beats", "bars")}
        print(f"\n=== {Path(path).name}")
        print(f"bpm={compact['bpm']}  dur={compact['duration']}s  bars={compact['n_bars']}")
        for s in compact["sections"]:
            print(
                f"  {s['label']:<10} {s['start']:>7.1f}–{s['end']:<7.1f}s  bars {s['bars'][0]}–{s['bars'][1]}"
            )
        out = Path(path).with_suffix(".segments.json")
        out.write_text(json.dumps(result, indent=1))
        print(f"  → {out.name}")
