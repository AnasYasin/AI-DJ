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
  2. Beat grid: Beat This! (learned joint beat/downbeat tracker); its downbeats
     vote for one of the four beat phases so bars stay 4 beats wide. Fallback
     when the model is unavailable: librosa beats + kick-heaviest phase, which
     on four-on-the-floor music is a guess (`downbeat_source` says which ran).
  3. Per-bar features: RMS, onset density, spectral flux, kick energy,
     harmonic ratio, spectral centroid.
  4. Phrase offset: which residue mod 8 the novelty peaks land on. Boundaries:
     checkerboard-novelty on the bar-level self-similarity matrix, snapped to
     half-phrase (4-bar) lines of that grid.
  5. Labels: heuristics over section-mean features (see _label_sections).

Only detected sections are included; `bpm`, `beats`, `bars`, `phrase_offset`
are always present.
Designed for 4/4 electronic music — the inference pool (Jamendo) and all six
training genres are 4/4.
"""

import hashlib
import json
import logging
from pathlib import Path
import sys

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

log = logging.getLogger(__name__)

SR = 22_050
PHRASE_LEN_BARS = 8  # a phrase in 4/4 dance music; the mixer's grid
PHRASE_BARS = 4  # boundaries snap to half-phrase lines, relative to the phrase offset
MIN_SECTION_BARS = 8  # merge sections shorter than this
KICK_BAND = (20, 150)  # Hz
# The learned downbeat tracker downloads a checkpoint on first use. Tests turn
# it off (conftest) so they never reach for the network.
BEAT_THIS_ENABLED = True


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
    return float(librosa.feature.tempo(y=y, sr=sr, aggregate=np.median, start_bpm=128)[0])


def _tracked_grid(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Beats and downbeats from the Beat This! tracker (Foscarin et al., ISMIR 2024).

    The kick-phase heuristic below cannot find a downbeat in four-on-the-floor
    music: every beat has a kick, and on six real tracks the winning phase beat
    the runner-up by 0.5-3%, which is noise. A learned joint beat/downbeat model
    is the standard answer. The tracker's downbeats are used as votes for one of
    the four beat phases, so the bar grid stays exactly four beats wide, which
    the rest of the pipeline assumes. Returns (beats, bars, confidence) where
    confidence is the share of tracker downbeats that agree with the chosen
    phase, or None when the tracker is unavailable.
    """
    if not BEAT_THIS_ENABLED:
        return None
    try:
        from beat_this.inference import Audio2Beats

        if not hasattr(_tracked_grid, "_model"):
            _tracked_grid._model = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        beats, downbeats = _tracked_grid._model(y, sr)
    except Exception as e:  # not installed, checkpoint missing, or inference failed
        log.warning("beat_this unavailable (%s) — downbeats fall back to kick phase", e)
        return None
    beats = np.asarray(beats, dtype=np.float64)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    if len(beats) < 8 or len(downbeats) < 4:
        return None
    return regular_bars(beats, downbeats)


BAR_TOLERANCE = 0.15  # a tracker downbeat within this fraction of a bar confirms a bar line


def regular_bars(
    beats: np.ndarray, downbeats: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    A regular 4-beat bar grid from a tracker's beats and downbeats.

    Without its DBN post-processor the tracker's downbeats are raw peaks: on one
    real track it fired on every beat for the first five seconds, and beat slips
    (a 3- or 5-beat gap) shift every later downbeat's beat index, so counting
    "beat index mod 4" votes gave a 30% majority on a track whose bars were
    perfectly clear to the ear.

    So the grid is built from downbeat TIMES. The longest run of consecutive
    downbeats spaced one bar apart is the anchor. From there the grid extends
    both ways one bar at a time, using the local beat spacing for the bar
    length; a tracker downbeat within BAR_TOLERANCE of the expected line snaps
    the line onto it, otherwise the line is placed where the tempo says. Beats
    are then four even subdivisions of each bar, so bars are always 4 beats.

    Returns (beats, bars, confidence): confidence is the share of tracker
    downbeats that lie on the final grid. None when no clean run exists.
    """
    ibi = np.median(np.diff(beats))
    bar_len = 4 * ibi
    gaps = np.diff(downbeats)
    clean = np.abs(gaps / bar_len - 1) < BAR_TOLERANCE

    # longest run of clean gaps → anchor
    best_len, best_start, run_start = 0, -1, 0
    for i in range(len(clean) + 1):
        if i == len(clean) or not clean[i]:
            if i - run_start > best_len:
                best_len, best_start = i - run_start, run_start
            run_start = i + 1
    if best_len < 2:
        return None
    anchor = list(downbeats[best_start : best_start + best_len + 1])

    def local_bar(t: float) -> float:
        """Bar length from the beats around t (tempo may breathe a little)."""
        i = int(np.searchsorted(beats, t))
        lo, hi = max(i - 8, 0), min(i + 8, len(beats) - 1)
        if hi - lo < 2:
            return bar_len
        return 4 * float(np.median(np.diff(beats[lo : hi + 1])))

    def extend(start: float, direction: int, limit: float) -> list[float]:
        out = []
        t = start
        while True:
            t_next = t + direction * local_bar(t)
            if (direction > 0 and t_next > limit) or (direction < 0 and t_next < limit):
                break
            near = downbeats[np.abs(downbeats - t_next) < BAR_TOLERANCE * bar_len]
            t = float(near[np.argmin(np.abs(near - t_next))]) if len(near) else t_next
            out.append(t)
        return out

    end = beats[-1] + 0.5 * ibi
    before = extend(anchor[0], -1, beats[0] - 0.5 * ibi)
    after = extend(anchor[-1], +1, end)
    bars = np.array(sorted(before) + anchor + after, dtype=np.float64)
    bars = bars[bars >= 0]

    # beats as four even subdivisions of each bar (last bar extrapolated)
    grid = []
    for a, b in zip(bars[:-1], bars[1:]):
        grid.extend(np.linspace(a, b, 4, endpoint=False))
    grid.extend(bars[-1] + np.arange(4) * local_bar(bars[-1]) / 4)
    grid = np.array(grid)
    grid = grid[grid <= end]

    on_grid = np.min(np.abs(downbeats[:, None] - bars[None, :]), axis=1) < BAR_TOLERANCE * bar_len
    return grid, bars, float(on_grid.mean())


def _kick_phase_grid(y: np.ndarray, sr: int, bpm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Fallback: librosa beats locked to the tempo, downbeat = kick-heaviest phase."""
    _, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, start_bpm=bpm, tightness=400, trim=False, units="frames"
    )
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beats) < 8:
        raise ValueError("too few beats detected")

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= KICK_BAND[0]) & (freqs < KICK_BAND[1])
    kick = S[band].sum(axis=0)
    kick_at_beat = kick[np.clip(beat_frames, 0, len(kick) - 1)]
    per_phase = np.array([kick_at_beat[p::4].sum() for p in range(4)])
    phase = int(np.argmax(per_phase))
    # how clearly the winner won: 0 when tied with the runner-up, 1 when the
    # runner-up has nothing. On real 4/4 tracks this is ~0.01-0.03.
    ranked = np.sort(per_phase)[::-1]
    confidence = float(1.0 - ranked[1] / max(ranked[0], 1e-9))
    return beats, beats[phase::4], confidence


def _beat_grid(y: np.ndarray, sr: int, bpm: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Beats and bar (downbeat) times, plus how the downbeat was found."""
    tracked = _tracked_grid(y, sr)
    if tracked is not None:
        beats, bars, conf = tracked
        return beats, bars, {"downbeat_source": "beat_this", "downbeat_confidence": conf}
    beats, bars, conf = _kick_phase_grid(y, sr, bpm)
    return beats, bars, {"downbeat_source": "kick-phase", "downbeat_confidence": conf}


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


def _novelty(F: np.ndarray) -> np.ndarray:
    """Checkerboard novelty on the bar-level self-similarity matrix."""
    n = len(F)
    ssm = F @ F.T / F.shape[1]
    novelty = np.zeros(n)
    for L in (4, 8):  # multi-scale kernels catch both phrase and section changes
        kernel = np.outer(np.r_[-np.ones(L), np.ones(L)], np.r_[-np.ones(L), np.ones(L)]) / (
            4 * L * L
        )
        pad = np.pad(ssm, L, mode="edge")
        for i in range(n):
            novelty[i] += (pad[i : i + 2 * L, i : i + 2 * L] * kernel).sum()
    return gaussian_filter1d(np.maximum(novelty, 0), 1.5)


def _novelty_peaks(novelty: np.ndarray) -> list[int]:
    n = len(novelty)
    return [
        i
        for i in range(2, n - 2)
        if novelty[i] == novelty[max(0, i - 2) : i + 3].max() and novelty[i] > 0
    ]


def phrase_offset(peaks: list[int], novelty: np.ndarray, phrase_bars: int = 8) -> int:
    """
    Which bar the track's phrases start on, 0..phrase_bars-1.

    Dance music changes on phrase lines: a new element, a drop, a breakdown all
    arrive on bar 1 of a phrase. So the phrase grid is whichever of the
    `phrase_bars` residues the novelty peaks fall on most, weighted by how big
    the change was. A peak one bar off still votes at half weight, because the
    novelty kernel blurs a change across its neighbours. Bar 0 is only the
    first tracked downbeat, not a phrase start, which is why this exists.
    """
    if not peaks:
        return 0
    votes = np.zeros(phrase_bars)
    for p in peaks:
        votes[p % phrase_bars] += novelty[p]
        votes[(p - 1) % phrase_bars] += 0.5 * novelty[p]
        votes[(p + 1) % phrase_bars] += 0.5 * novelty[p]
    return int(np.argmax(votes))


def _boundaries(F: np.ndarray, phrase_bars: int = 8) -> tuple[list[int], int]:
    """
    Boundary bar indices snapped to the track's own half-phrase grid, and the
    phrase offset they were snapped to. Adaptive count: aim for one section per
    ~16 bars, taking the strongest novelty peaks (a fixed threshold
    under-segments short/flat tracks).
    """
    n = len(F)
    novelty = _novelty(F)
    candidates = _novelty_peaks(novelty)
    offset = phrase_offset(candidates, novelty, phrase_bars)
    target = max(2, n // 16)
    picked: list[int] = []
    for p in sorted(candidates, key=lambda i: -novelty[i]):
        b = int(round((p - offset) / PHRASE_BARS)) * PHRASE_BARS + offset
        if not (MIN_SECTION_BARS <= b <= n - MIN_SECTION_BARS):
            continue
        if all(abs(b - q) >= MIN_SECTION_BARS for q in picked):
            picked.append(b)
        if len(picked) >= target:
            break
    return sorted(picked), offset


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


CACHE_DIR = Path("data/interim/segments")
# Bump when the analysis changes, so stale caches re-analyse. v2: learned
# downbeats, phrase offset, phrase-relative boundary snapping. v3: bar grid
# regularised from downbeat times (regular_bars) instead of beat-index votes.
SEGMENT_VERSION = 3


def _cache_key(path: Path) -> str:
    """Identify a file by path, size, mtime and analysis version."""
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|v{SEGMENT_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def segment(audio_path: str | Path, use_cache: bool = True) -> dict:
    """
    Full analysis of one track → section map (see module docstring).

    Results are cached on disk. Beat tracking and HPSS take tens of seconds per
    track, and the mixer analyses the same files on every render.
    """
    audio_path = Path(audio_path)
    cache_path = None
    if use_cache and audio_path.exists():
        cache_path = CACHE_DIR / f"{_cache_key(audio_path)}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                log.warning("discarding unreadable segment cache %s", cache_path.name)

    result = _segment_uncached(audio_path)
    if cache_path is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result))
    return result


def _segment_uncached(audio_path: str | Path) -> dict:
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    bpm = _bpm(y, sr)
    beats, bars, grid_meta = _beat_grid(y, sr, bpm)
    F = _bar_features(y, sr, bars)
    bounds, offset = _boundaries(F, PHRASE_LEN_BARS)
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
        # bar index of the first phrase start, and the phrase length it was
        # estimated for; the mixer cuts, swaps bass and hands over on this grid
        "phrase_bars": PHRASE_LEN_BARS,
        "phrase_offset": int(offset),
        **grid_meta,
        "sections": sections,
        # per-bar energy (z-scored rms+kick) — used by the mixer to pick which
        # window of the track to play against the set's energy curve
        "bar_energy": [round(float(v), 3) for v in (F[:, 0] + F[:, 3])],
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
