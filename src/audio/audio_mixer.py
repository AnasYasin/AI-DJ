"""
Phase 8b — audio mixer: ordered tracks → one continuous mixed .mp3/.flac.

Audio is carried as (samples, channels) in stereo from decode to write. Every
measurement — beat grid, sections, key, loudness, the seam correlation — runs on
the mono sum, because those are questions about timing and level, not about
width. Rendering in mono collapsed the side channel of every record (measured
−10 to −17 dB relative to mid across real tracks), which is most of what makes a
dance mix sound wide, and it was the single largest quality defect in the output.

v2: transition-recipe engine. Each transition is classified by the Phase-4
labeler rules computed from the two tracks' audio (energy, onset, loudness,
key, BPM), gated by the set's energy curve, then rendered by its recipe:
per-band (low/mid/high) gain envelopes + a full-band volume envelope over a
bar-aligned overlap. Staggered band envelopes approximate filter sweeps
(HPF-in = highs first, mids later, lows at the bass swap, etc.).

Recipe table (edit RECIPES to taste):
  slam  4 bars   center cut               no EQ, no filters
  rise  32 bars  overlap                  end bass swap, HPF-in B / HPF-out A
  fade  32 bars  smooth crossfade         center bass swap
  melt  64 bars  equal-power crossfade    slow center bass swap, no filters
  wave  16 bars  overlap                  center bass swap, LPF-in/out staggered
  blend 16 bars  overlap                  3-band fade
  drop  16 bars  ends ON the incoming drop B bass cut + filter opens over its
                                          buildup, A holds the low end

The pair rules cannot see the set, so `gate_transition` demotes their choice to
what the curve allows (a chill set never slams). Beat alignment is measured at
every seam and corrected, not assumed — see `measure_seam_offset`.

Run:
  python -m src.audio.audio_mixer out.flac t1.mp3 t2.mp3 ... [--genre techno]
      [--curve build] [--force-type drop] [--no-drop-align]
"""

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt
import soundfile as sf

from src.data.audio_segmenter import segment
from src.features.compatibility import compatibility, max_overlap_seconds
from src.features.transition_labeler import (
    BPM_RATIO_LOOSE as BPM_LOOSE,
)
from src.features.transition_labeler import (
    BPM_RATIO_TIGHT as BPM_TIGHT,
)
from src.features.transition_labeler import (
    ENERGY_FALL_MIN as ENERGY_FALL,
)
from src.features.transition_labeler import (
    ENERGY_MELT_MAX as ENERGY_MELT,
)
from src.features.transition_labeler import (
    ENERGY_RISE_MIN as ENERGY_RISE,
)
from src.features.transition_labeler import (
    ENERGY_SLAM_MIN as ENERGY_SLAM,
)
from src.features.transition_labeler import (
    HARM_CLASH,
    HARM_COMPATIBLE,
    HARM_PERFECT,
    normalise_onset,
)
from src.features.transition_labeler import (
    LOUD_MELT_MAX as LOUD_MELT,
)
from src.features.transition_labeler import (
    ONSET_HIGH_MIN as ONSET_HIGH,
)

log = logging.getLogger(__name__)

SR = 44_100
MIN_BODY_BARS = 24
SEAM_TOLERANCE_S = 0.004  # 4 ms — below this a beat offset is inaudible
SEAM_PASSES = 4  # re-measure and re-correct until the seam settles
BAND_LOW_HZ = 180
BAND_HIGH_HZ = 3_000

# ── Transition classification (Phase-4 labeler rules) ─────────────────────────

# The rule thresholds live in transition_labeler and are imported, not copied.
# They were duplicated here before, which is how the onset scale drifted: the
# labeler compared a raw value against a threshold meant for the normalised one.

_CAMELOT = {
    "C": 8,
    "Cm": 5,
    "C#": 3,
    "C#m": 12,
    "D": 10,
    "Dm": 7,
    "D#": 5,
    "D#m": 2,
    "E": 12,
    "Em": 9,
    "F": 7,
    "Fm": 4,
    "F#": 2,
    "F#m": 11,
    "G": 9,
    "Gm": 6,
    "G#": 4,
    "G#m": 1,
    "A": 11,
    "Am": 8,
    "A#": 6,
    "A#m": 3,
    "B": 1,
    "Bm": 10,
}


def _camelot_dist(ka: str, kb: str) -> float:
    a, b = _CAMELOT.get(ka), _CAMELOT.get(kb)
    if a is None or b is None:
        return 2.5
    d = abs(a - b)
    return min(d, 12 - d) + (0.5 if ka.endswith("m") != kb.endswith("m") else 0.0)


def measured_transition(ta: dict, tb: dict) -> str:
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


# ── Set intent gate ────────────────────────────────────────────────────────────
# The rules above see one pair of tracks and nothing else, so a chill set fires a
# SLAM whenever two neighbouring windows happen to differ by 0.15 energy. The set
# has an intent, and the intent decides which transitions are allowed to exist.
# CURVE_GATES maps a measured type to what the curve permits.

CURVE_GATES = {
    "chill": {"slam": "melt", "rise": "blend", "wave": "blend"},
    "arc": {"slam": "rise"},
    "wave": {"slam": "blend"},
    "build": {},
    "peak": {},
}

# A slam is a peak-time move. In a set that is still climbing it lands early and
# reads as a mistake, so it is only allowed past this fraction of the set.
SLAM_EARLIEST = {"build": 0.5, "arc": 0.4, "wave": 0.35, "peak": 0.15}


def gate_transition(ttype: str, curve: str | None, position: float | None = None) -> str:
    """Demote a measured transition type to what the set's energy curve allows."""
    if curve is None:
        return ttype
    gated = CURVE_GATES.get(curve, {}).get(ttype, ttype)
    if gated == "slam" and position is not None:
        earliest = SLAM_EARLIEST.get(curve)
        if earliest is not None and position < earliest:
            return "rise"
    return gated


def classify_transition(
    ta: dict, tb: dict, curve: str | None = None, position: float | None = None
) -> str:
    """
    Measured pair type, then gated by the set's intent.

    curve: the planner's energy curve for this set ("build", "peak", "wave",
    "chill", "arc"). position: where this transition sits in the set, 0-1.
    """
    return gate_transition(measured_transition(ta, tb), curve, position)


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
        # low band is an EQ kill (the bass swap); the thinning is a real filter
        "A_bands": {"low": [(0.0, 1), (0.82, 1), (0.88, 0)]},
        "B_bands": {"low": [(0.0, 0), (0.82, 0), (0.88, 1)]},
        "A_sweep": {"kind": "highpass", "hz": [(0.0, 20), (0.55, 20), (1.0, 800)]},
        "B_sweep": {"kind": "highpass", "hz": [(0.0, 2500), (0.75, 25), (1.0, 25)]},
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
        "A_bands": {"low": [(0.0, 1), (0.46, 1), (0.54, 0)]},  # center bass swap
        "B_bands": {"low": [(0.0, 0), (0.46, 0), (0.54, 1)]},
        "A_sweep": {"kind": "lowpass", "hz": [(0.0, 20000), (0.5, 18000), (1.0, 900)]},
        "B_sweep": {"kind": "lowpass", "hz": [(0.0, 1200), (0.55, 18000), (1.0, 20000)]},
        "duck": 0.85,
    },
    "drop": {  # filter blend into the incoming drop, see DROP-ALIGNED below
        # B enters over its own buildup with the bass cut and the filter
        # closed, opening as the buildup climbs. A holds the low end so the
        # floor never loses its bottom, then clears out. The overlap ends on
        # B's downbeat, so B's full spectrum arrives with the drop.
        "A_vol": [(0.0, 1), (0.94, 1), (1.0, 0)],
        "B_vol": [(0.0, 0), (0.10, 1), (1.0, 1)],
        "A_bands": {
            "low": [(0.0, 1), (0.88, 1), (0.97, 0)],  # bass makes way just before
            "mid": [(0.0, 1), (0.70, 1), (1.0, 0.2)],
            "high": [(0.0, 1), (0.75, 1), (1.0, 0.1)],
        },
        "B_bands": {"low": [(0.0, 0), (1.0, 0)]},  # bass cut for the whole buildup
        # the filter opens across the buildup and is wide open at the drop
        "B_sweep": {"kind": "highpass", "hz": [(0.0, 1600), (0.85, 60), (1.0, 25)]},
        "duck": 0.9,
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

# ── Who is the lead record ─────────────────────────────────────────────────────
# A long overlap is not a long crossfade. Sitting both records at the same level
# for a minute gives the ear nothing to hold onto; it hears two tracks, not a
# mix. A DJ keeps one record clearly in front, brings the other in underneath as
# a bed, and hands the lead over quickly at a phrase line.
#
# So the ambiguous zone is measured in BARS, not in a fraction of the overlap.
# A 64-bar melt and a 16-bar blend both hand over in about 8 bars; the melt just
# spends longer with A in front and B underneath first.
#
#   support_db  how far under the lead the other record sits during the bed
#   handover    where the lead changes hands, as a fraction of the overlap
#   swap_bars   how long the actual level swap takes (absolute)
#   entry_bars  how long the incoming record takes to reach its bed level
#   mid_dip_db  extra midrange cut on whichever record is supporting, so the
#               two do not fight over the same band
LEAD = {
    "melt": {
        "support_db": -8.0,
        "handover": 0.70,
        "swap_bars": 8,
        "entry_bars": 8,
        "mid_dip_db": -4.0,
    },
    "fade": {
        "support_db": -7.0,
        "handover": 0.60,
        "swap_bars": 8,
        "entry_bars": 6,
        "mid_dip_db": -3.0,
    },
    "rise": {
        "support_db": -6.0,
        "handover": 0.75,
        "swap_bars": 8,
        "entry_bars": 6,
        "mid_dip_db": -3.0,
    },
    "blend": {
        "support_db": -6.0,
        "handover": 0.60,
        "swap_bars": 6,
        "entry_bars": 4,
        "mid_dip_db": -3.0,
    },
    # a wave is meant to have both grooves riding, so the bed sits higher than
    # elsewhere — still far enough down to say which record is in front
    "wave": {
        "support_db": -5.5,
        "handover": 0.60,
        "swap_bars": 6,
        "entry_bars": 4,
        "mid_dip_db": -2.0,
    },
    # the drop transition hands over AT the drop, so the lead changes at the very end
    "drop": {
        "support_db": -5.0,
        "handover": 0.90,
        "swap_bars": 4,
        "entry_bars": 4,
        "mid_dip_db": 0.0,
    },
    # slam is a cut on a bar line — there is no shared time to shape
}
MIN_LEAD_BARS = 4  # below this an overlap is too short to hand over inside

# Types whose length is a preference rather than a constraint. `slam` is a cut on
# a bar line and `drop` is pinned to the incoming downbeat, so neither stretches.
STRETCHABLE = frozenset({"blend", "wave", "rise", "fade", "melt"})
LONG_STRETCH = 2.0  # a well-matched pair may run to twice its type's default


# ── How long two records may be held together ──────────────────────────────────
# The score, weights and tiers live in src/features/compatibility so the planner
# can optimise for the same thing the mixer measures. Here the distances come
# from the windows that actually play.


def pair_compatibility(ta: dict, tb: dict) -> float:
    """0-1 measure of how well two records will sit on top of each other."""
    return float(
        compatibility(
            _camelot_dist(ta["key"], tb["key"]),
            abs(ta["energy"] - tb["energy"]),
            abs(ta["lufs"] - tb["lufs"]),
            abs(np.log(ta["bpm"] / tb["bpm"])) * 100,  # percent, before stretching
        )
    )


def _lead_envelopes(n_bars: int, cfg: dict) -> tuple[list, list, list, list]:
    """
    Breakpoint specs for (A volume, B volume, A mid, B mid).

    A leads, B comes in underneath at `support_db`, the lead swaps over
    `swap_bars`, then A falls away. The supporting record also loses midrange so
    it sits behind rather than alongside.
    """
    support = 10 ** (cfg["support_db"] / 20)
    dip = 10 ** (cfg["mid_dip_db"] / 20)

    entry = min(cfg["entry_bars"] / max(n_bars, 1), 0.35)
    swap = min(cfg["swap_bars"] / max(n_bars, 1), 0.5)
    h0 = min(max(cfg["handover"] - swap / 2, entry + 0.02), 1.0 - swap - 0.02)
    h1 = h0 + swap

    a_vol = [(0.0, 1.0), (h0, 1.0), (h1, support), (1.0, 0.0)]
    b_vol = [(0.0, 0.0), (entry, support), (h0, support), (h1, 1.0), (1.0, 1.0)]
    a_mid = [(0.0, 1.0), (h0, 1.0), (h1, dip), (1.0, dip)]
    b_mid = [(0.0, dip), (h0, dip), (h1, 1.0), (1.0, 1.0)]
    return a_vol, b_vol, a_mid, b_mid


DEFAULT_BARS = {"slam": 4, "rise": 32, "fade": 32, "melt": 64, "wave": 16, "blend": 16, "drop": 16}

# How long a track plays, measured from 24,831 timestamped track changes across
# 1,665 real sets (consecutive `starting_time` gaps in tracklist_clean.csv).
#
# What the measurement actually said:
#   genre explains 12.5% of the variance in play length, the DJ 22%, the
#   individual set 42%. Position in the set explains nothing (Spearman +0.026,
#   median 4.00 min at the start against 4.30 at the end).
#   Track features explain nothing either. Regressing log duration on
#   within-genre energy and loudness z-scores gives R² = 0.0022: +1 sd of energy
#   moves the length by +0.2%, +1 sd of loudness by −2.3%. Both are smaller than
#   the 8-bar phrase this gets rounded to, so neither is modelled here.
# Genre is therefore the only track-independent driver worth using until DJ
# profiles exist (Phase 9), which would roughly double the explained variance.
GENRE_PLAY_MINUTES = {
    "afro house": 5.17,
    "melodic house": 4.63,
    "trance": 4.42,
    "techno": 4.08,
    "tech house": 4.00,
    "drum and base": 2.58,
}
DEFAULT_PLAY_MINUTES = 4.17  # median across all genres

# A window may stretch or shrink by this much to end on a real section boundary,
# which is where the per-track variation legitimately comes from.
LENGTH_FLEX_BARS = 16
LENGTH_PENALTY = 0.5  # cost per unit of relative deviation from the genre target
CUE_IN_BONUS, CUE_OUT_BONUS = 0.10, 0.25  # for starting / ending on a section change
GENRE_BARS = {
    "techno": DEFAULT_BARS,
    "tech house": {
        "slam": 4,
        "rise": 24,
        "fade": 16,
        "melt": 64,
        "wave": 16,
        "blend": 32,
        "drop": 16,
    },
    "melodic house": {
        "slam": 4,
        "rise": 24,
        "fade": 16,
        "melt": 64,
        "wave": 16,
        "blend": 32,
        "drop": 16,
    },
    "afro house": {
        "slam": 4,
        "rise": 24,
        "fade": 16,
        "melt": 64,
        "wave": 16,
        "blend": 32,
        "drop": 16,
    },
    "trance": {"slam": 4, "rise": 32, "fade": 32, "melt": 64, "wave": 16, "blend": 24, "drop": 16},
    # dnb BPM is detected half-time (~86), so bars are twice as long — halve counts
    "drum and base": {
        "slam": 4,
        "rise": 8,
        "fade": 8,
        "melt": 16,
        "wave": 8,
        "blend": 8,
        "drop": 8,
    },
}


# ── DSP helpers ────────────────────────────────────────────────────────────────


def _to_mono(y: np.ndarray) -> np.ndarray:
    """Mono sum for analysis. Every measurement — beats, sections, key, the seam
    correlation — wants one signal; only the audio path stays stereo."""
    return y if y.ndim == 1 else y.mean(axis=1)


def load_audio(path: str | Path) -> np.ndarray:
    """
    Decode to (samples, channels) at SR, always 2 channels.

    Audio is carried as (samples, channels) throughout so that every slice in the
    renderer indexes time, exactly as it did when this was mono. A genuinely mono
    source is duplicated to two channels so tracks can be concatenated.
    """
    y, _ = librosa.load(str(path), sr=SR, mono=False)
    y = np.atleast_2d(y)
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    return np.ascontiguousarray(y.T, dtype=np.float32)


def _stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """
    Time-stretch to the mix tempo. rubberband preserves transients; the librosa
    phase vocoder smears every kick, which is audible on a 4/4 mix, so the
    fallback warns loudly instead of degrading quietly.
    """
    if abs(rate - 1.0) < 1e-4:
        return y
    if shutil.which("rubberband"):
        import pyrubberband

        # rubberband stretches the channels together, so the stereo image stays
        # phase-locked instead of drifting apart.
        return pyrubberband.time_stretch(y, SR, rate).astype(np.float32)
    log.warning(
        "rubberband CLI not found — falling back to the librosa phase vocoder, "
        "which smears transients. Install it: apt install rubberband-cli"
    )
    if y.ndim == 1:
        return librosa.effects.time_stretch(y, rate=rate)
    stretched = librosa.effects.time_stretch(np.ascontiguousarray(y.T), rate=rate)
    return np.ascontiguousarray(stretched.T, dtype=np.float32)


def _bands(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into low/mid/high; low + mid + high reconstructs y exactly.

    sosfiltfilt runs the filter forwards and backwards, so the band split is
    zero-phase. A one-pass butterworth delays the low band by a few ms more
    than the highs, which smears the kick against its own hats during a
    transition. Offline rendering makes the non-causal pass free.
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) <= 24:  # filtfilt needs padlen headroom; too short to matter
        return np.zeros_like(y), y.copy(), np.zeros_like(y)
    # axis 0 is time for both (samples,) and (samples, channels)
    low = sosfiltfilt(butter(4, BAND_LOW_HZ, "lowpass", fs=SR, output="sos"), y, axis=0)
    high = sosfiltfilt(butter(4, BAND_HIGH_HZ, "highpass", fs=SR, output="sos"), y, axis=0)
    return low, y - low - high, high


SWEEP_BLOCK = 8192  # ~186 ms; cutoff is refreshed every half-block
SWEEP_ORDER = 4


def _swept_filter(y: np.ndarray, cutoff_hz: np.ndarray, kind: str) -> np.ndarray:
    """
    Real high- or low-pass with a moving corner frequency.

    The band-gain approximation this replaces could only raise and lower three
    fixed bands, which gives a shelf at 180 Hz or 3 kHz rather than a corner you
    hear travel. Here the filter is redesigned every half block and the blocks
    are Hann overlap-added, so the cutoff glides.

    `cutoff_hz` is one value per sample; only its median per block is used.
    """
    n = len(y)
    if n < SWEEP_BLOCK * 2:  # too short to sweep — use the mean cutoff flat
        return _fixed_filter(y, float(np.median(cutoff_hz)), kind)

    frame = SWEEP_BLOCK * 2
    hop = SWEEP_BLOCK
    window = np.hanning(frame + 1)[:-1]  # periodic, sums to 1 at 50% overlap
    if y.ndim == 2:
        window = window[:, None]

    out = np.zeros_like(y, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)
    flat_win = np.hanning(frame + 1)[:-1]
    for start in range(0, n, hop):
        stop = min(start + frame, n)
        seg = y[start:stop]
        if len(seg) < 32:
            break
        w = window[: len(seg)]
        hz = float(np.median(cutoff_hz[start:stop]))
        out[start:stop] += _fixed_filter(seg, hz, kind) * w
        norm[start:stop] += flat_win[: len(seg)]

    norm = np.maximum(norm, 1e-6)
    return out / (norm[:, None] if y.ndim == 2 else norm)


def _fixed_filter(y: np.ndarray, hz: float, kind: str) -> np.ndarray:
    """One zero-phase pass at a fixed cutoff. Cutoffs at the edges are no-ops."""
    if kind == "highpass" and hz <= 25.0:
        return np.asarray(y, dtype=np.float64)
    if kind == "lowpass" and hz >= SR / 2 * 0.95:
        return np.asarray(y, dtype=np.float64)
    hz = float(np.clip(hz, 20.0, SR / 2 * 0.95))
    sos = butter(SWEEP_ORDER, hz, kind, fs=SR, output="sos")
    return sosfiltfilt(sos, np.asarray(y, dtype=np.float64), axis=0)


def _sweep_cutoffs(spec: list, n: int) -> np.ndarray:
    """Breakpoints of (time fraction, Hz) → a per-sample cutoff, swept in octaves."""
    ts, hz = zip(*spec)
    return np.exp(np.interp(np.linspace(0, 1, n), ts, np.log(hz)))


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


def _apply_recipe(
    part: np.ndarray, vol_spec, band_spec, kind: str, duck: float, mid_spec=None, sweep=None
) -> np.ndarray:
    n = len(part)
    out_full = kind == "out"

    def env(spec, default):
        """Gain curve shaped to broadcast over however many channels there are."""
        e = _env(spec, n, default)
        return e if part.ndim == 1 else e[:, None]

    if band_spec is None:
        shaped = np.asarray(part, dtype=np.float64)
    else:
        low, mid, high = _bands(part)  # float64, zero-phase
        shaped = np.zeros_like(low)
        default = "out" if out_full else "in"
        for band, sig in (("low", low), ("mid", mid), ("high", high)):
            spec = band_spec.get(band)
            gain = 1.0 if spec is None else env(spec, default)
            # the supporting record gives up midrange on top of whatever the
            # recipe already does to that band
            if band == "mid" and mid_spec is not None:
                gain = gain * env(mid_spec, default)
            shaped += sig * gain

    # Signal path order follows a real mixer channel: EQ, then the filter knob,
    # then the fader.
    if sweep is not None:
        shaped = _swept_filter(shaped, _sweep_cutoffs(sweep["hz"], n), sweep["kind"])
    return (shaped * env(vol_spec, kind) * duck).astype(np.float32)


# ── Loudness ───────────────────────────────────────────────────────────────────

MIX_TARGET_LUFS = -14.0  # streaming-normal; leaves headroom for the true peak
MAX_TRACK_GAIN_DB = 6.0  # beyond this a track is simply mastered differently


def _loudness(y: np.ndarray, sr: int = SR) -> float:
    """Integrated loudness (LUFS). Returns nan when the signal is too short/quiet."""
    try:
        import pyloudnorm as pyln

        if len(y) < sr:  # meter needs ≥ 400 ms blocks
            return float("nan")
        # pyloudnorm takes (samples, channels) and applies the ITU channel
        # weighting itself, so stereo is measured properly rather than summed.
        val = float(pyln.Meter(sr).integrated_loudness(np.asarray(y, dtype=np.float64)))
        return val if np.isfinite(val) else float("nan")
    except Exception as e:
        log.warning("loudness measurement failed (%s) — falling back to RMS", e)
        return float("nan")


def _rms_db(y: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(np.square(y, dtype=np.float64))) + 1e-12)


def _level_db(y: np.ndarray) -> float:
    """LUFS where measurable, RMS dB otherwise. Both are dB, so deltas compose."""
    lufs = _loudness(y)
    return lufs if np.isfinite(lufs) else _rms_db(y)


# ── Track preparation ──────────────────────────────────────────────────────────


def _analyse_body(y: np.ndarray) -> dict:
    """energy / onset / LUFS / key of a track body — same conventions as Phase 2."""
    y22 = librosa.resample(_to_mono(y), orig_sr=SR, target_sr=22_050)
    energy = float(librosa.feature.rms(y=y22)[0].mean())
    onset = float(normalise_onset(librosa.onset.onset_strength(y=y22, sr=22_050).mean()))
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
    info: dict, target_bars: int, tail_bars: int, e_target01: float | None
) -> tuple[int, int]:
    """
    Pick which window of the track to play, and how long it runs.

    `target_bars` is the genre's measured play length. The window may stretch or
    shrink by up to LENGTH_FLEX_BARS around it so that it can start and end on
    real section boundaries, which is where the per-track variation comes from.
    Ending a record mid-phrase is what makes a set sound arbitrary.

    Scores every phrase-aligned (start, length) pair:
      - energy match: mean |bar_energy − target| where target is the set's
        energy-curve value mapped onto this track's own bar_energy range
        (low target → windows that avoid drops; high target → contain the drop)
      - boundary bonus for starting/ending on detected section boundaries
      - penalty for drifting from the genre's play length
    Returns (start_bar, end_bar); a tail of `tail_bars` after end must exist.
    """
    n = len(info["bars"])
    be = np.array(info["bar_energy"], dtype=np.float32)
    # Real section changes only. Bar 0 and the final bar are the ends of the
    # file, not musical boundaries, and rewarding them pins every window to the
    # top of the track.
    bounds = {s["bars"][0] for s in info["sections"]} | {s["bars"][1] for s in info["sections"]}
    bounds -= {0, n, n - 1}

    labels = {s["label"]: s for s in info["sections"]}
    first_ok = labels["intro"]["bars"][1] if "intro" in labels else 0
    lo_start = max(first_ok - 8, 0)
    last_ok = n - 1 - tail_bars
    room = last_ok - lo_start
    target_bars = min(target_bars, room)
    if target_bars < MIN_BODY_BARS:  # short track: play whatever it has
        return lo_start, max(last_ok, lo_start + 1)

    if e_target01 is None:
        target = np.quantile(be, 0.65)  # default: energetic but not only-the-drop
    else:
        target = np.quantile(be, 0.25 + 0.55 * float(e_target01))

    lengths = [
        target_bars + d
        for d in range(-LENGTH_FLEX_BARS, LENGTH_FLEX_BARS + 1, PHRASE_BARS)
        if MIN_BODY_BARS <= target_bars + d <= room
    ]
    best, best_score = None, -np.inf
    for length in lengths:
        drift = LENGTH_PENALTY * abs(length - target_bars) / target_bars
        for start in range(lo_start, last_ok - length + 1, PHRASE_BARS):
            end = start + length
            score = -float(np.abs(be[start:end] - target).mean())
            # The cue-out earns more than the cue-in: it is where the record
            # hands over, so a ragged end is the one the ear catches.
            score += CUE_IN_BONUS * (start in bounds) + CUE_OUT_BONUS * (end in bounds) - drift
            if score > best_score:
                best, best_score = (start, end), score
    return best if best else (lo_start, last_ok)


PHRASE_BARS = 8  # windows snap to 8-bar phrases


def _precise_bpm(info: dict) -> float:
    """
    Refine the (integer-quantised) DeepRhythm BPM from the measured beat grid.
    A 0.3 BPM error in the stretch ratio drifts ~a beat over a 64-bar overlap;
    the median inter-beat interval pins the true tempo to ~0.01 BPM.
    """
    beats = np.asarray(info["beats"], dtype=np.float64)
    if len(beats) < 16:
        return float(info["bpm"])
    measured = 60.0 / float(np.median(np.diff(beats)))
    if abs(measured - info["bpm"]) / info["bpm"] > 0.08:  # octave/tracking failure
        return float(info["bpm"])
    return measured


def _calibrate_grid_phase(y: np.ndarray, beat_times: np.ndarray, bpm: float) -> float:
    """
    Measure the track's grid phase error ε: median offset between ESTIMATED
    beat times and the track's ACTUAL kick-band transients. Cutting two tracks
    on their own estimated grids leaves an audible offset of ε_b − ε_a at the
    seam — correcting each grid by its own ε makes every bar time point at a
    real kick, so seams align by construction. (Cut-point "grid alignment" is
    a tautology: bar cuts are always on their own grid; the grid itself is
    what's offset from the audio.)
    """
    hop = 256
    beat = 60.0 / bpm
    half_win = int(beat / 4 * SR / hop)  # ±¼ beat search per beat
    S = np.abs(librosa.stft(_to_mono(y), n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    kick = S[(f >= 30) & (f < 130)].sum(axis=0)
    env = np.maximum(np.diff(kick, prepend=kick[:1]), 0.0)

    mid = beat_times[len(beat_times) // 4 : -len(beat_times) // 4]  # skip intro/outro
    deltas = []
    for t in mid[:: max(len(mid) // 200, 1)]:
        c = int(t * SR / hop)
        a, b = max(c - half_win, 0), min(c + half_win + 1, len(env))
        if b - a < 3:
            continue
        w = env[a:b]
        if w.max() < np.median(env) * 2:  # no clear transient near this beat
            continue
        deltas.append((a + int(np.argmax(w)) - c) * hop / SR)
    if len(deltas) < 20:
        return 0.0
    return float(np.median(deltas))


ONSET_HOP = 128  # 2.9 ms per frame; a flam starts being audible around 10 ms


def _kick_envelope(y: np.ndarray) -> np.ndarray:
    """Rising kick-band energy — where the beat actually lands, frame by frame."""
    S = np.abs(librosa.stft(_to_mono(y), n_fft=1024, hop_length=ONSET_HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    kick = S[(f >= 30) & (f < 130)].sum(axis=0)
    return np.maximum(np.diff(kick, prepend=kick[:1]), 0.0)


def measure_seam_offset(tail: np.ndarray, head: np.ndarray, bpm: float) -> float:
    """
    How far the incoming track's beat sits from the outgoing one's, in seconds.

    Cross-correlates the two kick envelopes over ±½ beat. A positive result
    means the head lands late and has to start earlier. This is the number the
    ear hears as a flam, so it is what gets measured and reported per seam.
    """
    ea, eb = _kick_envelope(tail), _kick_envelope(head)
    n = min(len(ea), len(eb))
    if n < 32:
        return 0.0
    ea, eb = ea[:n], eb[:n]
    ea = (ea - ea.mean()) / (ea.std() + 1e-9)
    eb = (eb - eb.mean()) / (eb.std() + 1e-9)
    max_lag = max(int(0.5 * (60.0 / bpm) * SR / ONSET_HOP), 1)
    corr = correlate(ea, eb, mode="full") / n
    centre = n - 1
    window = corr[centre - max_lag : centre + max_lag + 1]
    k = int(np.argmax(window))

    # Parabolic interpolation through the peak and its neighbours: the true
    # offset rarely lands exactly on a frame, and rounding to one would leave
    # up to half a frame of error in the correction.
    if 0 < k < len(window) - 1:
        a, b, c = window[k - 1], window[k], window[k + 1]
        denom = a - 2 * b + c
        k = k + (0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0)
    return -float((k - max_lag) * ONSET_HOP / SR)


def _prepare_track(
    path: str | Path,
    target_bpm: float,
    max_tail_bars: int,
    target_bars: int,
    e_target01: float | None = None,
) -> dict:
    info = segment(path)
    bpm = _precise_bpm(info)
    rate = target_bpm / bpm
    if not 0.8 <= rate <= 1.25:
        log.warning("%s: stretch rate %.2f is extreme", Path(path).name, rate)
    y = load_audio(path)  # (samples, channels)
    y = _stretch(y, rate).astype(np.float32)
    bars = np.array(info["bars"], dtype=np.float64) / rate

    cue_in_bar, cue_out_bar = _choose_window(info, target_bars, max_tail_bars, e_target01)

    drop_bar = None  # slam entry point: first drop inside the window, if any
    for s in info["sections"]:
        if s["label"] == "drop" and s["bars"][0] >= cue_in_bar:
            drop_bar = s["bars"][0]
            break

    body = y[int(bars[cue_in_bar] * SR) : int(bars[cue_out_bar] * SR)]
    feats = _analyse_body(body if len(body) > SR * 20 else y)

    beat_times = np.asarray(info["beats"], dtype=np.float64) / rate  # stretched
    eps = _calibrate_grid_phase(y, beat_times, target_bpm)
    if abs(eps) > 0.003:
        log.info("  grid phase calibration: %+.0f ms (%s)", eps * 1000, Path(path).stem[:30])
        bars = bars + eps
        beat_times = beat_times + eps

    return {
        "name": Path(path).stem,
        "audio": y,
        "bars": bars,
        "beat_times": beat_times,
        "target_bpm": target_bpm,
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
    play_minutes: float | None = None,
    energy_targets: list[float] | None = None,
    curve: str | None = None,
    force_type: str | None = None,
    drop_align: bool = True,
) -> dict:
    """
    play_minutes: target play length per track. None uses the genre's measured
    median from GENRE_PLAY_MINUTES. The actual length is decided per track,
    flexing around this to land on section boundaries.
    energy_targets: per-track 0-1 values straight off the planner's energy
    curve — steers WHICH window of each track is played (low → avoid drops,
    high → include). These must be the raw curve values, NOT the plan's
    `target_energy`, which is in absolute energy units.
    curve: the planner's energy curve, which gates transition types to the
    set's intent (a chill set never fires a slam).
    force_type: override the classifier and render every transition as this
    type. For A/B testing recipes, not for real sets.
    drop_align: let a rise become a drop-aligned filter blend when the incoming
    track has a drop to aim the bass swap at.
    """
    if force_type is not None and force_type not in RECIPES:
        raise ValueError(f"unknown transition type {force_type!r}")
    if energy_targets is not None:
        if len(energy_targets) != len(track_paths):
            raise ValueError(
                f"energy_targets has {len(energy_targets)} values for {len(track_paths)} tracks"
            )
        if not all(0.0 <= float(e) <= 1.0 for e in energy_targets):
            raise ValueError(
                "energy_targets must be raw 0-1 curve values. Absolute track "
                "energies (the plan's `target_energy`) silently flatten the curve; "
                "use `energy_target01` instead."
            )
    bars_table = GENRE_BARS.get(genre or "", DEFAULT_BARS)
    # Tail reserved after each window so the next track has something to mix
    # over. The transition type is not known yet, because it depends on features
    # measured from the prepared tracks, so this has to be an estimate.
    # Reserving the longest possible transition (a 64-bar melt) costs every
    # track two minutes of playable window, which caps a 137-bar track at half
    # the genre's play length. The typical transition is reserved instead, and
    # the overlap clamp below shortens a long one when the tail runs out.
    max_tail = int(np.median(list(bars_table.values())))

    if target_bpm is None:
        bpms = [segment(p)["bpm"] for p in track_paths]
        target_bpm = float(np.median(bpms))
        log.info("target BPM (median): %.1f  (tracks: %s)", target_bpm, bpms)

    bar_dur_t = 4 * 60.0 / target_bpm
    if play_minutes is None:
        play_minutes = GENRE_PLAY_MINUTES.get(genre or "", DEFAULT_PLAY_MINUTES)
        log.info("play length: %.2f min from genre %r", play_minutes, genre or "(none)")
    target_bars = max(
        int(round(play_minutes * 60 / bar_dur_t / PHRASE_BARS)) * PHRASE_BARS, MIN_BODY_BARS
    )

    tracks = []
    for i, p in enumerate(track_paths):
        et = energy_targets[i] if energy_targets else None
        t = _prepare_track(p, target_bpm, max_tail, target_bars, et)
        played_bars = t["cue_out_bar"] - t["cue_in_bar"]
        log.info(
            "  %s: %.0f→%.0f bpm, window bars %d–%d (%d bars, %.2f min vs %.2f target),"
            " e_target=%s key=%s",
            t["name"][:40],
            t["bpm"],
            target_bpm,
            t["cue_in_bar"],
            t["cue_out_bar"],
            played_bars,
            played_bars * bar_dur_t / 60,
            play_minutes,
            f"{et:.2f}" if et is not None else "none",
            t["key"],
        )
        tracks.append(t)

    bar_dur = 4 * 60.0 / target_bpm

    # Loudness match on the window that actually plays, in LUFS. RMS matching
    # chases peak density, so a busy track and a sparse track at the same RMS
    # are perceived a long way apart, and the sparse one pumps when it enters.
    # The target is the set median, which keeps every gain small.
    for t in tracks:
        body = t["audio"][
            int(t["bars"][t["cue_in_bar"]] * SR) : int(t["bars"][t["cue_out_bar"]] * SR)
        ]
        t["level_db"] = _level_db(body if len(body) > SR else t["audio"])
    target_db = float(np.median([t["level_db"] for t in tracks]))
    for t in tracks:
        gain_db = float(np.clip(target_db - t["level_db"], -MAX_TRACK_GAIN_DB, MAX_TRACK_GAIN_DB))
        t["gain_db"] = round(gain_db, 2)
        t["audio"] = (t["audio"] * (10 ** (gain_db / 20))).astype(np.float32)
        log.info("  %s: %.1f LUFS → %+.1f dB", t["name"][:40], t["level_db"], gain_db)

    mix = tracks[0]["audio"][
        int(tracks[0]["bars"][tracks[0]["cue_in_bar"]] * SR) : int(
            tracks[0]["bars"][tracks[0]["cue_out_bar"]] * SR
        )
    ].copy()

    trans_report = []
    for i in range(1, len(tracks)):
        prev, cur = tracks[i - 1], tracks[i]
        position = i / max(len(tracks) - 1, 1)
        measured = measured_transition(prev, cur)
        ttype = force_type or gate_transition(measured, curve, position)
        if ttype != measured:
            why = "forced" if force_type else f"curve={curve}"
            log.info("  transition %d: %s → %s (%s)", i, measured, ttype, why)
        recipe = RECIPES[ttype]
        n_bars = bars_table[ttype]

        # DROP-ALIGNED ENTRY. When the incoming track has a drop, the strongest
        # move is to bring it in over its own buildup with the bass cut and the
        # filter closed, and let the low end swap back at the drop itself. That
        # needs the overlap to END on the drop, so the entry bar is measured
        # backwards from it. A rise already wants to hand over on a climb, so
        # it is promoted when the track gives us a drop to aim at.
        if (
            drop_align
            and ttype == "rise"
            and cur["drop_bar"] is not None
            and cur["drop_bar"] - bars_table["drop"] >= 0
        ):
            log.info("  transition %d: rise → drop (aligning to the incoming drop)", i)
            ttype = "drop"
            recipe, n_bars = RECIPES[ttype], bars_table[ttype]

        # The transition type says HOW the two records are blended. How LONG they
        # are held together is the pair's business, not the type's. The type's
        # bar count is a genre default: a pair that has earned a long overlap may
        # run past it, up to LONG_STRETCH times, so a blend between two records
        # that genuinely fit becomes a slow blend rather than a quick one. A slam
        # is a cut and a drop is pinned to the incoming downbeat, so neither
        # stretches.
        compat = pair_compatibility(prev, cur)
        ceiling_bars = max(int(max_overlap_seconds(compat) / bar_dur), 2)
        want = (
            min(ceiling_bars, int(n_bars * LONG_STRETCH))
            if ttype in STRETCHABLE
            else min(ceiling_bars, n_bars)
        )
        if want != n_bars:
            log.info(
                "  transition %d: %d → %d bars (compatibility %.2f allows %.0fs)",
                i,
                n_bars,
                want,
                compat,
                max_overlap_seconds(compat),
            )
            n_bars = max(want, 2)

        # incoming entry point: slam lands on the drop, a drop-aligned blend
        # ends on it, everything else starts at cue-in
        in_bar = cur["cue_in_bar"]
        if ttype == "slam" and cur["drop_bar"] is not None:
            in_bar = max(cur["drop_bar"] - bars_table["slam"] // 2, 0)
        elif ttype == "drop" and cur["drop_bar"] is not None:
            in_bar = max(cur["drop_bar"] - n_bars, 0)

        # clamp overlap to available audio on both sides (whole bars)
        avail_prev = len(prev["audio"]) / SR - prev["bars"][prev["cue_out_bar"]]
        avail_cur = (len(cur["audio"]) / SR - cur["bars"][in_bar]) - MIN_BODY_BARS * bar_dur
        n_bars = max(min(n_bars, int(avail_prev / bar_dur), int(avail_cur / bar_dur)), 2)
        # a shortened overlap moves the entry with it, or the drop stops
        # landing on the seam and the whole point is lost
        if ttype == "drop" and cur["drop_bar"] is not None:
            in_bar = max(cur["drop_bar"] - n_bars, 0)
        n_ov = int(n_bars * bar_dur * SR)

        # Both grids are phase-calibrated to real kicks (the ε correction at
        # prepare time), so cutting on bar times already lands close. What is
        # left is per-seam: beat trackers drift, and a track's tempo is not
        # perfectly constant, so the residual at this particular cut is
        # measured against the audio and removed.
        a0 = max(int(prev["bars"][prev["cue_out_bar"]] * SR), 0)
        b0 = max(int(cur["bars"][in_bar] * SR), 0)
        tail = prev["audio"][a0 : a0 + n_ov]
        head = cur["audio"][b0 : b0 + n_ov]
        n_ov = min(len(tail), len(head))
        tail, head = tail[:n_ov], head[:n_ov]

        before = after = measure_seam_offset(tail, head, target_bpm)
        max_shift = int(0.5 * (60.0 / target_bpm) * SR)
        total_shift = 0
        # Shifting the cut changes which audio is in the head, so the offset is
        # re-measured and re-corrected until it settles. One pass typically
        # leaves a few ms behind.
        for _ in range(SEAM_PASSES):
            if abs(after) <= SEAM_TOLERANCE_S:
                break
            step = int(np.clip(round(after * SR), -max_shift, max_shift))
            if b0 + step < 0 or step == 0:
                break
            b0 += step
            total_shift += step
            tail = prev["audio"][a0 : a0 + n_ov]
            head = cur["audio"][b0 : b0 + n_ov]
            n_ov = min(len(tail), len(head))
            tail, head = tail[:n_ov], head[:n_ov]
            after = measure_seam_offset(tail, head, target_bpm)
        if total_shift:
            log.info(
                "  seam %d: beat offset %+.1f ms → %+.1f ms (shifted %+.1f ms)",
                i,
                before * 1000,
                after * 1000,
                total_shift / SR * 1000,
            )

        at = len(mix) / SR
        duck = recipe["duck"]
        lead_cfg = LEAD.get(ttype)
        if lead_cfg and n_bars >= MIN_LEAD_BARS:
            a_vol, b_vol, a_mid, b_mid = _lead_envelopes(n_bars, lead_cfg)
            swap_at = at + n_ov / SR * lead_cfg["handover"]
            lead_swap = f"{int(swap_at // 60)}:{int(swap_at % 60):02d}"
        else:  # too short to hand over inside — straight crossfade
            a_vol, b_vol, a_mid, b_mid = recipe["A_vol"], recipe["B_vol"], None, None
            lead_swap = None
        overlap = _apply_recipe(
            tail, a_vol, recipe["A_bands"], "out", duck, a_mid, recipe.get("A_sweep")
        ) + _apply_recipe(head, b_vol, recipe["B_bands"], "in", duck, b_mid, recipe.get("B_sweep"))

        # Constant-loudness transition. Two tracks summed must not be louder
        # than one, or every overlap jumps out and the bodies sound weak after
        # the whole mix is normalised. Measured in the same units as the track
        # gains above, so the overlap sits level with its loudest neighbour.
        loud_side = max(_level_db(tail), _level_db(head))
        ov_db = _level_db(overlap)
        if ov_db > loud_side:
            overlap = (overlap * 10 ** ((loud_side - ov_db) / 20)).astype(np.float32)

        body = cur["audio"][b0 + n_ov : int(cur["bars"][cur["cue_out_bar"]] * SR)]
        mix = np.concatenate([mix, overlap, body])
        trans_report.append(
            {
                "from": prev["name"][:30],
                "to": cur["name"][:30],
                "type": ttype,
                "measured": measured,
                "bars": n_bars,
                "compatibility": round(compat, 3),
                "max_overlap_s": max_overlap_seconds(compat),
                "lead_swap": lead_swap,
                "seam_offset_ms": round(after * 1000, 1),
                "seam_offset_before_ms": round(before * 1000, 1),
                "at": f"{int(at // 60)}:{int(at % 60):02d}",
                "end": f"{int((at + n_ov / SR) // 60)}:{int((at + n_ov / SR) % 60):02d}",
            }
        )
        log.info("  transition %d: %s (%d bars) at %s", i, ttype, n_bars, trans_report[-1]["at"])

    # Normalise the finished mix to a fixed loudness, then back off if the
    # peak would clip. Peak-only normalisation let a dense mix and a sparse one
    # land at very different perceived volumes.
    mix_db = _level_db(mix)
    if np.isfinite(mix_db):
        mix = mix * 10 ** ((MIX_TARGET_LUFS - mix_db) / 20)
    peak = float(np.abs(mix).max())
    if peak > 0.98:
        mix = mix * (0.98 / peak)
        log.info(
            "peak-limited by %.1f dB after loudness normalisation", 20 * np.log10(peak / 0.98)
        )
    mix = mix.astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in (".wav", ".flac"):
        # Lossless out. The sources are already lossy YouTube audio, so an MP3
        # render is a second generation of loss on top of the first.
        sf.write(str(out_path), mix, SR)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, mix, SR)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    tmp.name,
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "320k",
                    str(out_path),
                ],
                check=True,
            )
            Path(tmp.name).unlink()

    report = {
        "out": str(out_path),
        "duration_s": round(len(mix) / SR, 1),
        "target_bpm": target_bpm,
        "n_tracks": len(tracks),
        "channels": 1 if mix.ndim == 1 else mix.shape[1],
        "lufs": round(_level_db(mix), 1),
        "peak": round(float(np.abs(mix).max()), 3),
        "curve": curve,
        "transitions": trans_report,
    }
    log.info("mix rendered: %.1f min → %s", report["duration_s"] / 60, out_path)
    return report


def render_plan(
    plan_path: str | Path,
    tracks_dir: str | Path,
    out_path: str | Path,
    **kwargs,
) -> dict:
    """
    Render a plan from `predict_model` using audio fetched by `track_fetcher`.

    This is the join between planning and rendering. The plan's genre and energy
    curve steer the transition gate and the play-length default, and its per-track
    curve values steer which window of each track is played. Files are matched by
    track_id, and any the fetcher could not verify are dropped from the set.
    """
    plan = json.loads(Path(plan_path).read_text())
    tracks_dir = Path(tracks_dir)

    paths, targets, missing = [], [], []
    for t in plan["tracks"]:
        hits = sorted(p for p in tracks_dir.glob(f"{t['track_id']}.*") if p.is_file())
        if not hits:
            missing.append(f"{t['artist']} – {t['title']}")
            continue
        paths.append(hits[0])
        targets.append(t.get("energy_target01"))

    if missing:
        log.warning(
            "%d planned tracks have no audio, dropped: %s", len(missing), "; ".join(missing)
        )
    if len(paths) < 2:
        raise ValueError(f"only {len(paths)} of {len(plan['tracks'])} planned tracks have audio")

    if any(t is None for t in targets):
        # Plans written before energy_target01 existed. Better to lose the
        # window steering than to guess a scale.
        log.warning("plan has no energy_target01 values — window steering disabled")
        targets = None

    return render_mix(
        paths,
        out_path,
        genre=plan.get("genre"),
        curve=plan.get("curve"),
        energy_targets=targets,
        **kwargs,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Render ordered tracks into one mix.")
    p.add_argument("out")
    p.add_argument("tracks", nargs="*", help="track files, in order (omit when using --plan)")
    p.add_argument("--plan", default=None, help="plan JSON from predict_model")
    p.add_argument("--tracks-dir", default=None, help="directory of fetched audio, with --plan")
    p.add_argument("--genre", default=None)
    p.add_argument("--bpm", type=float, default=None)
    p.add_argument(
        "--play-minutes", type=float, default=None, help="default: the genre's measured median"
    )
    p.add_argument("--curve", default=None, choices=list(CURVE_GATES) + [None])
    p.add_argument("--force-type", default=None, choices=list(RECIPES))
    p.add_argument("--no-drop-align", action="store_true")
    args = p.parse_args()

    common = dict(
        target_bpm=args.bpm,
        play_minutes=args.play_minutes,
        force_type=args.force_type,
        drop_align=not args.no_drop_align,
    )
    if args.plan:
        if not args.tracks_dir:
            p.error("--plan needs --tracks-dir")
        rep = render_plan(args.plan, args.tracks_dir, args.out, **common)
    elif args.tracks:
        rep = render_mix(args.tracks, args.out, genre=args.genre, curve=args.curve, **common)
    else:
        p.error("give track files, or --plan with --tracks-dir")
    print(f"\n{rep['out']}  {rep['duration_s'] / 60:.1f} min @ {rep['target_bpm']:.0f} bpm")
    for tr in rep["transitions"]:
        print(f"  {tr['from']}  →  {tr['to']}   {tr['type'].upper()} {tr['bars']} bars")
