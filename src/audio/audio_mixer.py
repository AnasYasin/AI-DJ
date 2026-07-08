"""
Phase 8b — audio mixer: ordered tracks → one continuous mixed .mp3.

v2: transition-recipe engine. Each transition is classified (slam / rise /
fade / melt / wave / blend) by the Phase-4 labeler rules computed from the
two tracks' audio (energy, onset, loudness, key, BPM), then rendered by its
recipe: per-band (low/mid/high) gain envelopes + a full-band volume envelope
over a bar-aligned overlap. Staggered band envelopes approximate filter
sweeps (HPF-in = highs first, mids later, lows at the bass swap, etc.).

Recipe table (agreed 2026-07-08 — edit RECIPES to taste):
  slam  4 bars   center cut               no EQ, no filters
  rise  32 bars  overlap                  end bass swap, HPF-in B / HPF-out A
  fade  32 bars  smooth crossfade         center bass swap
  melt  64 bars  equal-power crossfade    slow center bass swap, no filters
  wave  16 bars  overlap                  center bass swap, LPF-in/out staggered
  blend 16 bars  overlap                  3-band fade

Run:
  python -m src.audio.audio_mixer out.mp3 track1.mp3 track2.mp3 ... [--genre techno]
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from src.data.audio_segmenter import segment

log = logging.getLogger(__name__)

SR = 44_100
MIN_BODY_BARS = 24
BAND_LOW_HZ = 180
BAND_HIGH_HZ = 3_000

# ── Transition classification (Phase-4 labeler rules) ─────────────────────────

BPM_TIGHT, BPM_LOOSE = 0.03, 0.07
ENERGY_RISE, ENERGY_FALL, ENERGY_SLAM, ENERGY_MELT = 0.08, -0.08, 0.15, 0.05
HARM_PERFECT, HARM_COMPATIBLE, HARM_CLASH = 1, 2, 5
LOUD_MELT, ONSET_HIGH = 3.0, 0.35

_CAMELOT = {
    "C": 8, "Cm": 5, "C#": 3, "C#m": 12, "D": 10, "Dm": 7, "D#": 5, "D#m": 2,
    "E": 12, "Em": 9, "F": 7, "Fm": 4, "F#": 2, "F#m": 11, "G": 9, "Gm": 6,
    "G#": 4, "G#m": 1, "A": 11, "Am": 8, "A#": 6, "A#m": 3, "B": 1, "Bm": 10,
}


def _camelot_dist(ka: str, kb: str) -> float:
    a, b = _CAMELOT.get(ka), _CAMELOT.get(kb)
    if a is None or b is None:
        return 2.5
    d = abs(a - b)
    return min(d, 12 - d) + (0.5 if ka.endswith("m") != kb.endswith("m") else 0.0)


def classify_transition(ta: dict, tb: dict) -> str:
    """First-match-wins cascade on pair features (same rules as transition_labeler)."""
    bpm_ratio = abs(np.log(ta["bpm"] / tb["bpm"]))
    d_energy = tb["energy"] - ta["energy"]
    harm = _camelot_dist(ta["key"], tb["key"])
    d_loud = abs(tb["lufs"] - ta["lufs"])
    if (bpm_ratio <= BPM_TIGHT and d_energy > ENERGY_SLAM) or (
        harm >= HARM_CLASH and d_energy > ENERGY_RISE
    ):
        return "slam"
    if d_energy > ENERGY_RISE and bpm_ratio <= BPM_LOOSE and harm <= HARM_COMPATIBLE:
        return "rise"
    if d_energy < ENERGY_FALL:
        return "fade"
    if (
        bpm_ratio <= BPM_TIGHT
        and harm <= HARM_PERFECT
        and abs(d_energy) < ENERGY_MELT
        and d_loud < LOUD_MELT
    ):
        return "melt"
    if bpm_ratio <= BPM_TIGHT and ta["onset"] > ONSET_HIGH and tb["onset"] > ONSET_HIGH:
        return "wave"
    return "blend"


# ── Recipes ────────────────────────────────────────────────────────────────────
# Envelopes are (time_fraction, gain) breakpoints, linearly interpolated then
# lightly smoothed. Bands: A/B_bands = {low, mid, high}; None = full-band only.

RECIPES = {
    "slam": {  # center cut on a bar line; B lands on its drop if it has one
        "A_vol": [(0.0, 1), (0.49, 1), (0.51, 0)],
        "B_vol": [(0.0, 0), (0.49, 0), (0.51, 1)],
        "A_bands": None,
        "B_bands": None,
        "duck": 1.0,
    },
    "rise": {  # overlap; end bass swap; HPF-in on B, HPF-out on A
        "A_vol": [(0.0, 1), (0.92, 1), (1.0, 0)],
        "B_vol": [(0.0, 0), (0.08, 1), (1.0, 1)],
        "A_bands": {
            "low": [(0.0, 1), (0.82, 1), (0.88, 0)],           # keeps bass until end swap
            "mid": [(0.0, 1), (0.6, 1), (1.0, 0.2)],           # thins upward (HPF-out)
            "high": [(0.0, 1), (1.0, 0.5)],
        },
        "B_bands": {
            "low": [(0.0, 0), (0.82, 0), (0.88, 1)],           # end bass swap
            "mid": [(0.0, 0), (0.3, 0), (0.75, 1)],            # HPF-in opens downward
            "high": [(0.0, 0), (0.18, 1)],                     # airy entry
        },
        "duck": 0.85,
    },
    "fade": {  # smooth crossfade + center bass swap
        "A_vol": "xfade_out",
        "B_vol": "xfade_in",
        "A_bands": {"low": [(0.0, 1), (0.46, 1), (0.54, 0)], "mid": None, "high": None},
        "B_bands": {"low": [(0.0, 0), (0.46, 0), (0.54, 1)], "mid": None, "high": None},
        "duck": 1.0,
    },
    "melt": {  # invisible: equal-power xfade, slow center bass swap, no filters
        "A_vol": "xfade_out",
        "B_vol": "xfade_in",
        "A_bands": {"low": [(0.0, 1), (0.35, 1), (0.65, 0)], "mid": None, "high": None},
        "B_bands": {"low": [(0.0, 0), (0.35, 0), (0.65, 1)], "mid": None, "high": None},
        "duck": 1.0,
    },
    "wave": {  # grooves ride together; LPF-in B / LPF-out A, staggered
        "A_vol": [(0.0, 1), (0.9, 1), (1.0, 0)],
        "B_vol": [(0.0, 0), (0.1, 1), (1.0, 1)],
        "A_bands": {
            "low": [(0.0, 1), (0.46, 1), (0.54, 0)],           # center bass swap
            "mid": [(0.0, 1), (0.55, 1), (1.0, 0.3)],          # LPF-out (darkens late)
            "high": [(0.0, 1), (0.5, 1), (0.95, 0.1)],
        },
        "B_bands": {
            "low": [(0.0, 0), (0.46, 0), (0.54, 1)],
            "mid": [(0.0, 0.3), (0.45, 1)],                    # LPF-in (opens early)
            "high": [(0.0, 0.1), (0.05, 0.1), (0.5, 1)],
        },
        "duck": 0.85,
    },
    "blend": {  # overlap + 3-band fade: highs first, mids center, lows last
        "A_vol": [(0.0, 1), (0.9, 1), (1.0, 0)],
        "B_vol": [(0.0, 0), (0.1, 1), (1.0, 1)],
        "A_bands": {
            "low": [(0.0, 1), (0.7, 1), (0.8, 0)],
            "mid": [(0.0, 1), (0.45, 1), (0.6, 0.15)],
            "high": [(0.0, 1), (0.2, 1), (0.4, 0.15)],
        },
        "B_bands": {
            "low": [(0.0, 0), (0.7, 0), (0.8, 1)],
            "mid": [(0.0, 0.15), (0.45, 0.15), (0.6, 1)],
            "high": [(0.0, 0.15), (0.2, 0.15), (0.4, 1)],
        },
        "duck": 0.85,
    },
}

DEFAULT_BARS = {"slam": 4, "rise": 32, "fade": 32, "melt": 64, "wave": 16, "blend": 16}
GENRE_BARS = {
    "techno": DEFAULT_BARS,
    "tech house": {"slam": 4, "rise": 24, "fade": 16, "melt": 64, "wave": 16, "blend": 32},
    "melodic house": {"slam": 4, "rise": 24, "fade": 16, "melt": 64, "wave": 16, "blend": 32},
    "afro house": {"slam": 4, "rise": 24, "fade": 16, "melt": 64, "wave": 16, "blend": 32},
    "trance": {"slam": 4, "rise": 32, "fade": 32, "melt": 64, "wave": 16, "blend": 24},
    # dnb BPM is detected half-time (~86), so bars are twice as long — halve counts
    "drum and base": {"slam": 4, "rise": 8, "fade": 8, "melt": 16, "wave": 8, "blend": 8},
}


# ── DSP helpers ────────────────────────────────────────────────────────────────


def _stretch(y: np.ndarray, rate: float) -> np.ndarray:
    if abs(rate - 1.0) < 1e-4:
        return y
    if shutil.which("rubberband"):
        import pyrubberband

        return pyrubberband.time_stretch(y, SR, rate)
    return librosa.effects.time_stretch(y, rate=rate)


def _bands(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into low/mid/high; low + mid + high reconstructs y exactly."""
    low = sosfilt(butter(4, BAND_LOW_HZ, "lowpass", fs=SR, output="sos"), y)
    high = sosfilt(butter(4, BAND_HIGH_HZ, "highpass", fs=SR, output="sos"), y)
    return low, y - low - high, high


def _env(spec, n: int, kind_default: str) -> np.ndarray:
    """Breakpoint spec → smooth gain curve of length n."""
    if spec == "xfade_out" or (spec is None and kind_default == "out"):
        return np.cos(np.linspace(0, 1, n) * np.pi / 2).astype(np.float32)
    if spec == "xfade_in" or (spec is None and kind_default == "in"):
        return np.sin(np.linspace(0, 1, n) * np.pi / 2).astype(np.float32)
    t = np.linspace(0, 1, n)
    ts, gs = zip(*spec)
    g = np.interp(t, ts, gs).astype(np.float32)
    w = max(int(0.05 * SR), 1)  # 50ms smoothing to soften envelope corners
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(np.pad(g, (w, w), mode="edge"), kernel, mode="same")[w:-w]


def _apply_recipe(part: np.ndarray, vol_spec, band_spec, kind: str, duck: float) -> np.ndarray:
    n = len(part)
    out_full = kind == "out"
    if band_spec is None:
        shaped = part
    else:
        low, mid, high = _bands(part)
        shaped = np.zeros_like(part)
        for band, sig in (("low", low), ("mid", mid), ("high", high)):
            spec = band_spec.get(band)
            if spec is None:
                shaped += sig  # untouched band, volume env handles it
            else:
                shaped += sig * _env(spec, n, "out" if out_full else "in")
    return shaped * _env(vol_spec, n, kind) * duck


# ── Track preparation ──────────────────────────────────────────────────────────


def _analyse_body(y: np.ndarray) -> dict:
    """energy / onset / LUFS / key of a track body — same conventions as Phase 2."""
    y22 = librosa.resample(y, orig_sr=SR, target_sr=22_050)
    energy = float(librosa.feature.rms(y=y22)[0].mean())
    onset = float(librosa.onset.onset_strength(y=y22, sr=22_050).mean() / 5.0)
    try:
        import pyloudnorm as pyln

        lufs = float(pyln.Meter(22_050).integrated_loudness(np.stack([y22, y22], axis=1)))
        if not np.isfinite(lufs):
            lufs = -70.0
    except Exception:
        lufs = -70.0
    try:
        import essentia.standard as es

        note, scale, _ = es.KeyExtractor(sampleRate=22_050.0, profileType="edma")(
            y22.astype(np.float32)
        )
        key = f"{note}{'m' if scale == 'minor' else ''}"
    except Exception:
        key = "?"
    return {"energy": energy, "onset": onset, "lufs": lufs, "key": key}


def _choose_window(
    info: dict, play_bars: int, tail_bars: int, e_target01: float | None
) -> tuple[int, int]:
    """
    Pick which window of the track to play (real DJs play ~3-5 min of a track,
    not intro-to-outro — median 4.05 min across 1,387 timestamped sets).

    Scores every phrase-aligned window of `play_bars` bars:
      - energy match: mean |bar_energy − target| where target is the set's
        energy-curve value mapped onto this track's own bar_energy range
        (low target → windows that avoid drops; high target → contain the drop)
      - boundary bonus for starting/ending on detected section boundaries
    Returns (start_bar, end_bar); a tail of `tail_bars` after end must exist.
    """
    n = len(info["bars"])
    be = np.array(info["bar_energy"], dtype=np.float32)
    bounds = {s["bars"][0] for s in info["sections"]} | {s["bars"][1] for s in info["sections"]}

    labels = {s["label"]: s for s in info["sections"]}
    first_ok = labels["intro"]["bars"][1] if "intro" in labels else 0
    last_ok = n - 1 - tail_bars
    play_bars = min(play_bars, last_ok - max(first_ok - 8, 0))
    if play_bars < MIN_BODY_BARS:
        return max(first_ok - 8, 0), max(last_ok, max(first_ok - 8, 0) + 1)

    if e_target01 is None:
        target = np.quantile(be, 0.65)  # default: energetic but not only-the-drop
    else:
        target = np.quantile(be, 0.25 + 0.55 * float(e_target01))

    best, best_score = None, -np.inf
    for start in range(max(first_ok - 8, 0), last_ok - play_bars + 1, PHRASE_BARS):
        end = start + play_bars
        score = -float(np.abs(be[start:end] - target).mean())
        score += 0.15 * (start in bounds) + 0.15 * (end in bounds)
        if score > best_score:
            best, best_score = (start, end), score
    return best if best else (max(first_ok - 8, 0), last_ok)


PHRASE_BARS = 8  # windows snap to 8-bar phrases


def _prepare_track(
    path: str | Path,
    target_bpm: float,
    max_tail_bars: int,
    play_bars: int,
    e_target01: float | None = None,
) -> dict:
    info = segment(path)
    rate = target_bpm / info["bpm"]
    if not 0.8 <= rate <= 1.25:
        log.warning("%s: stretch rate %.2f is extreme", Path(path).name, rate)
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    y = _stretch(y, rate).astype(np.float32)
    bars = np.array(info["bars"], dtype=np.float64) / rate

    cue_in_bar, cue_out_bar = _choose_window(info, play_bars, max_tail_bars, e_target01)

    drop_bar = None  # slam entry point: first drop inside the window, if any
    for s in info["sections"]:
        if s["label"] == "drop" and s["bars"][0] >= cue_in_bar:
            drop_bar = s["bars"][0]
            break

    body = y[int(bars[cue_in_bar] * SR) : int(bars[cue_out_bar] * SR)]
    feats = _analyse_body(body if len(body) > SR * 20 else y)

    return {
        "name": Path(path).stem,
        "audio": y,
        "bars": bars,
        "bpm": info["bpm"],
        "rate": rate,
        "cue_in_bar": int(cue_in_bar),
        "cue_out_bar": int(cue_out_bar),
        "drop_bar": drop_bar,
        **feats,
    }


# ── Mix rendering ──────────────────────────────────────────────────────────────


def render_mix(
    track_paths: list[str | Path],
    out_path: str | Path,
    target_bpm: float | None = None,
    genre: str | None = None,
    play_minutes: float = 3.2,
    energy_targets: list[float] | None = None,
) -> dict:
    """
    play_minutes: how long each track plays (real-DJ median: 4.05 min; default
    slightly shorter so overlaps make up a bigger share of airtime).
    energy_targets: per-track 0-1 values from the planner's curve — steers
    WHICH window of each track is played (low → avoid drops, high → include).
    """
    bars_table = GENRE_BARS.get(genre or "", DEFAULT_BARS)
    max_tail = max(bars_table.values())

    if target_bpm is None:
        bpms = [segment(p)["bpm"] for p in track_paths]
        target_bpm = float(np.median(bpms))
        log.info("target BPM (median): %.1f  (tracks: %s)", target_bpm, bpms)

    bar_dur_t = 4 * 60.0 / target_bpm
    play_bars = max(int(round(play_minutes * 60 / bar_dur_t / PHRASE_BARS)) * PHRASE_BARS,
                    MIN_BODY_BARS)

    tracks = []
    for i, p in enumerate(track_paths):
        et = energy_targets[i] if energy_targets else None
        t = _prepare_track(p, target_bpm, max_tail, play_bars, et)
        log.info(
            "  %s: %.0f→%.0f bpm, window bars %d–%d (%.1f min), e=%.2f key=%s",
            t["name"][:40], t["bpm"], target_bpm, t["cue_in_bar"], t["cue_out_bar"],
            (t["cue_out_bar"] - t["cue_in_bar"]) * bar_dur_t / 60, t["energy"], t["key"],
        )
        tracks.append(t)

    bar_dur = 4 * 60.0 / target_bpm
    ref_rms = np.sqrt(np.mean(tracks[0]["audio"] ** 2)) + 1e-9
    for t in tracks:
        g = ref_rms / (np.sqrt(np.mean(t["audio"] ** 2)) + 1e-9)
        t["audio"] = t["audio"] * min(g, 2.0)

    mix = tracks[0]["audio"][
        int(tracks[0]["bars"][tracks[0]["cue_in_bar"]] * SR)
        : int(tracks[0]["bars"][tracks[0]["cue_out_bar"]] * SR)
    ].copy()

    trans_report = []
    for i in range(1, len(tracks)):
        prev, cur = tracks[i - 1], tracks[i]
        ttype = classify_transition(prev, cur)
        recipe = RECIPES[ttype]
        n_bars = bars_table[ttype]

        # incoming entry point: slam lands on the drop, everything else on cue-in
        in_bar = cur["cue_in_bar"]
        if ttype == "slam" and cur["drop_bar"] is not None:
            in_bar = max(cur["drop_bar"] - bars_table["slam"] // 2, 0)

        # clamp overlap to available audio on both sides (whole bars)
        avail_prev = len(prev["audio"]) / SR - prev["bars"][prev["cue_out_bar"]]
        avail_cur = (len(cur["audio"]) / SR - cur["bars"][in_bar]) - MIN_BODY_BARS * bar_dur
        n_bars = max(min(n_bars, int(avail_prev / bar_dur), int(avail_cur / bar_dur)), 2)
        O = int(n_bars * bar_dur * SR)

        a0 = int(prev["bars"][prev["cue_out_bar"]] * SR)
        b0 = int(cur["bars"][in_bar] * SR)
        tail, head = prev["audio"][a0 : a0 + O], cur["audio"][b0 : b0 + O]
        O = min(len(tail), len(head))
        tail, head = tail[:O], head[:O]

        duck = recipe["duck"]
        overlap = _apply_recipe(tail, recipe["A_vol"], recipe["A_bands"], "out", duck) + \
            _apply_recipe(head, recipe["B_vol"], recipe["B_bands"], "in", duck)

        body = cur["audio"][b0 + O : int(cur["bars"][cur["cue_out_bar"]] * SR)]
        mix = np.concatenate([mix, overlap, body])
        trans_report.append(
            {"from": prev["name"][:30], "to": cur["name"][:30], "type": ttype, "bars": n_bars}
        )
        log.info("  transition %d: %s (%d bars)", i, ttype, n_bars)

    peak = np.abs(mix).max()
    if peak > 0.98:
        mix = mix * (0.98 / peak)

    out_path = Path(out_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, mix, SR)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
             "-codec:a", "libmp3lame", "-b:a", "192k", str(out_path)],
            check=True,
        )
        Path(tmp.name).unlink()

    report = {
        "out": str(out_path),
        "duration_s": round(len(mix) / SR, 1),
        "target_bpm": target_bpm,
        "n_tracks": len(tracks),
        "transitions": trans_report,
    }
    log.info("mix rendered: %.1f min → %s", report["duration_s"] / 60, out_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Render ordered tracks into one mix.")
    p.add_argument("out")
    p.add_argument("tracks", nargs="+")
    p.add_argument("--genre", default=None)
    p.add_argument("--bpm", type=float, default=None)
    p.add_argument("--play-minutes", type=float, default=3.2)
    args = p.parse_args()
    rep = render_mix(args.tracks, args.out, target_bpm=args.bpm, genre=args.genre,
                     play_minutes=args.play_minutes)
    print(f"\n{rep['out']}  {rep['duration_s']/60:.1f} min @ {rep['target_bpm']:.0f} bpm")
    for tr in rep["transitions"]:
        print(f"  {tr['from']}  →  {tr['to']}   {tr['type'].upper()} {tr['bars']} bars")
