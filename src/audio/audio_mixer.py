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
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt
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
    normalise_key,
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
SEAM_WINDOW_BARS = 4  # drift is measured per this many bars across the overlap
SEAM_DRIFT_WARN_S = 0.020  # a 20 ms flam is clearly audible
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
    },
    "rise": {  # overlap; end bass swap; HPF-in on B, HPF-out on A
        "A_vol": [(0.0, 1), (0.92, 1), (1.0, 0)],
        "B_vol": [(0.0, 0), (0.08, 1), (1.0, 1)],
        # low band is an EQ kill (the bass swap); the thinning is a real filter
        "A_bands": {"low": [(0.0, 1), (0.82, 1), (0.88, 0)]},
        "B_bands": {"low": [(0.0, 0), (0.82, 0), (0.88, 1)]},
        "A_sweep": {"kind": "highpass", "hz": [(0.0, 20), (0.55, 20), (1.0, 800)]},
        "B_sweep": {"kind": "highpass", "hz": [(0.0, 2500), (0.75, 25), (1.0, 25)]},
    },
    "fade": {  # smooth crossfade + center bass swap
        "A_vol": "xfade_out",
        "B_vol": "xfade_in",
        "A_bands": {"low": [(0.0, 1), (0.46, 1), (0.54, 0)], "mid": None, "high": None},
        "B_bands": {"low": [(0.0, 0), (0.46, 0), (0.54, 1)], "mid": None, "high": None},
    },
    "melt": {  # invisible: equal-power xfade, slow center bass swap, no filters
        "A_vol": "xfade_out",
        "B_vol": "xfade_in",
        "A_bands": {"low": [(0.0, 1), (0.35, 1), (0.65, 0)], "mid": None, "high": None},
        "B_bands": {"low": [(0.0, 0), (0.35, 0), (0.65, 1)], "mid": None, "high": None},
    },
    "wave": {  # grooves ride together; LPF-in B / LPF-out A, staggered
        "A_vol": [(0.0, 1), (0.9, 1), (1.0, 0)],
        "B_vol": [(0.0, 0), (0.1, 1), (1.0, 1)],
        "A_bands": {"low": [(0.0, 1), (0.46, 1), (0.54, 0)]},  # center bass swap
        "B_bands": {"low": [(0.0, 0), (0.46, 0), (0.54, 1)]},
        "A_sweep": {"kind": "lowpass", "hz": [(0.0, 20000), (0.5, 18000), (1.0, 900)]},
        "B_sweep": {"kind": "lowpass", "hz": [(0.0, 1200), (0.55, 18000), (1.0, 20000)]},
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


def _lead_envelopes(n_bars: int, cfg: dict, grid_bars: int = 1) -> tuple[list, list, list, list]:
    """
    Breakpoint specs for (A volume, B volume, A mid, B mid).

    A leads, B comes in underneath at `support_db`, the lead swaps over
    `swap_bars`, then A falls away. The supporting record also loses midrange so
    it sits behind rather than alongside.

    The handover starts on a bar line of the `grid_bars` grid (a half phrase in
    render_mix), because the moment the lead changes hands is a move the ear
    marks, and a DJ makes it on the "1".
    """
    support = 10 ** (cfg["support_db"] / 20)
    dip = 10 ** (cfg["mid_dip_db"] / 20)

    n = max(n_bars, 1)
    entry = min(cfg["entry_bars"] / n, 0.35)
    swap = min(cfg["swap_bars"] / n, 0.5)
    h0 = min(max(cfg["handover"] - swap / 2, entry + 0.02), 1.0 - swap - 0.02)
    if grid_bars > 1 and n_bars > grid_bars:
        # nearest grid line, but not before the incoming record has reached its
        # bed level; if the swap would then run past the end, the swap shortens
        first_ok = int(np.ceil(np.ceil((entry + 0.02) * n_bars) / grid_bars)) * grid_bars
        k = max(int(round(h0 * n_bars / grid_bars)) * grid_bars, first_ok)
        if k <= n_bars - 2:
            h0 = k / n_bars
            swap = min(swap, (n_bars - 1 - k) / n_bars)
    h1 = h0 + swap

    a_vol = [(0.0, 1.0), (h0, 1.0), (h1, support), (1.0, 0.0)]
    b_vol = [(0.0, 0.0), (entry, support), (h0, support), (h1, 1.0), (1.0, 1.0)]
    a_mid = [(0.0, 1.0), (h0, 1.0), (h1, dip), (1.0, dip)]
    b_mid = [(0.0, dip), (h0, dip), (h1, 1.0), (1.0, 1.0)]
    return a_vol, b_vol, a_mid, b_mid


DEFAULT_BARS = {"slam": 4, "rise": 32, "fade": 32, "melt": 64, "wave": 16, "blend": 16, "drop": 16}

# ── Phrase grid ────────────────────────────────────────────────────────────────
# Dance music is built in 8-bar phrases (16 and 32 at the section level), and a
# DJ puts every move that the ear marks on a phrase line: the cue-in, the
# cue-out, the bass swap, the handover. The segmenter reports where each
# track's phrases start (`phrase_offset`); everything below is placed on that
# grid. dnb is tracked half-time so its bars are twice as long: 4 of them make
# the same phrase.
GENRE_PHRASE_BARS = {"drum and base": 4}
DEFAULT_PHRASE_BARS = 8
SWAP_BARS = 1  # a bass swap is one decisive move, not a slow crossfade


def _whole_phrases(n_bars: int, phrase: int) -> int:
    """Round an overlap down to whole phrases; below one phrase, keep whole bars."""
    if n_bars < phrase:
        return max(n_bars, 1)
    return (n_bars // phrase) * phrase


def _snap_low_swap(spec, n_bars: int, grid: int):
    """
    Move a low-band swap onto a bar line and make it SWAP_BARS long.

    Recipe low-band specs are written as fractions of the overlap, e.g.
    (0.46, 1) → (0.54, 0), which on a 45-bar overlap is a 3.6-bar swap starting
    at bar 20.7. This finds the swap, snaps its start to the nearest `grid` bar
    line and gives it a fixed length in bars. Specs with no single swap (None,
    constant, or more than one move) are returned unchanged.
    """
    if spec is None or n_bars < 2:
        return spec
    ts, gs = zip(*spec)
    changes = [i for i in range(1, len(gs)) if gs[i] != gs[i - 1]]
    if len(changes) != 1:
        return spec
    i = changes[0]
    g0, g1 = gs[i - 1], gs[i]
    centre_bar = 0.5 * (ts[i - 1] + ts[i]) * n_bars
    grid = max(min(grid, n_bars // 2), 1)
    k = int(round(centre_bar / grid)) * grid
    k = int(np.clip(k, grid if n_bars > grid else 1, n_bars - SWAP_BARS))
    k = max(k, 1)
    t0, t1 = k / n_bars, min((k + SWAP_BARS) / n_bars, 1.0)
    return [(0.0, g0), (t0, g0), (t1, g1), (1.0, g1)]


def _snap_bands(band_spec, n_bars: int, grid: int):
    if band_spec is None:
        return None
    out = dict(band_spec)
    if "low" in out:
        out["low"] = _snap_low_swap(out["low"], n_bars, grid)
    return out


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


# Rubber Band's R3 engine. Measured on 45 s of real stereo: kick transients
# sharper than R2 (peak/mean 13.3 vs 11.3), identical high-frequency content,
# about 40% more stretch time. The tool's own manual says R3 "almost always
# produces better results".
STRETCH_ENGINE_ARGS = {"-3": ""}

# How far a record may be stretched. DJs pitch ±6% at the outside; a planned
# set stays near 3% because the planner's BPM range is what the user asked for.
# Past STRETCH_MAX the kick is audibly smeared, so the render stops rather than
# quietly degrading.
STRETCH_WARN = 0.04
STRETCH_MAX = 0.08


def _check_stretch_rate(rate: float, name: str) -> None:
    """Warn at STRETCH_WARN, refuse at STRETCH_MAX. `rate` = target_bpm / track_bpm."""
    amount = abs(float(np.log(rate)))
    if amount > STRETCH_MAX:
        raise ValueError(
            f"{name}: stretch rate {rate:.3f} ({amount * 100:.1f}%) exceeds "
            f"{STRETCH_MAX * 100:.0f}% — the track is too far from the set tempo"
        )
    if amount > STRETCH_WARN:
        log.warning(
            "%s: stretch rate %.3f (%.1f%%) is past what a DJ would pitch",
            name,
            rate,
            amount * 100,
        )


# Stretching a 6-minute record with rubberband R3 takes ~40 s, nine tenths of
# preparing a track, and every render of the same pair repeats it. The result
# is cached by the audio's content hash, the rate and the engine settings, so a
# changed tempo measurement or engine flag misses the cache by construction.
STRETCH_CACHE_DIR = Path("data/interim/stretch")
STRETCH_CACHE_VERSION = 1


def _stretch_cache_path(y: np.ndarray, rate: float) -> Path:
    import hashlib

    h = hashlib.sha1()
    h.update(np.ascontiguousarray(y, dtype=np.float32).tobytes())
    h.update(
        f"|{y.shape}|{SR}|{rate:.7f}|{sorted(STRETCH_ENGINE_ARGS.items())}|v{STRETCH_CACHE_VERSION}".encode()
    )
    return STRETCH_CACHE_DIR / f"{h.hexdigest()[:20]}.npy"


def _stretch(y: np.ndarray, rate: float, use_cache: bool = True) -> np.ndarray:
    """
    Time-stretch to the mix tempo. rubberband preserves transients; the librosa
    phase vocoder smears every kick, which is audible on a 4/4 mix, so the
    fallback warns loudly instead of degrading quietly.
    """
    if abs(rate - 1.0) < 1e-4:
        return y
    if shutil.which("rubberband"):
        import pyrubberband

        cache = _stretch_cache_path(y, rate) if use_cache else None
        if cache is not None and cache.exists():
            try:
                return np.load(cache)
            except (OSError, ValueError):
                log.warning("discarding unreadable stretch cache %s", cache.name)
        # rubberband stretches the channels together, so the stereo image stays
        # phase-locked instead of drifting apart.
        out = pyrubberband.time_stretch(y, SR, rate, rbargs=dict(STRETCH_ENGINE_ARGS)).astype(
            np.float32
        )
        if cache is not None:
            STRETCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npy")
            np.save(tmp, out)
            tmp.replace(cache)
        return out
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


SWEEP_ORDER = 4
SWEEP_STEPS_PER_OCTAVE = 3  # grid of fixed cutoffs the sweep crossfades between
SWEEP_PAD = SR // 2  # settle room around each grid step's active span


def _swept_filter(y: np.ndarray, cutoff_hz: np.ndarray, kind: str) -> np.ndarray:
    """
    Real high- or low-pass with a moving corner frequency.

    The corner is swept by crossfading, per sample, between fixed zero-phase
    filters on a 1/3-octave grid that spans the cutoff curve. Each grid step is
    only computed over the span where its weight is non-zero (plus settle
    room), so the cost is about two passes over the signal however far the
    sweep travels.

    This replaces a block-wise design: the filter was redesigned every 8192
    samples and the blocks Hann overlap-added. Every block ran filtfilt from a
    cold start, so each block edge carried a transient, measured at 20% of
    signal peak with a constant cutoff, which is a click every 186 ms under a
    rise or a wave. A crossfade of whole-signal filters has no block edges.

    `cutoff_hz` is one value per sample.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    cut = np.clip(np.asarray(cutoff_hz, dtype=np.float64), 20.0, SR / 2 * 0.95)
    lo, hi = float(cut.min()), float(cut.max())
    step = 2 ** (1 / SWEEP_STEPS_PER_OCTAVE)
    if n < 64 or hi / lo < step**0.5:  # too short, or the corner barely moves
        return _fixed_filter(y, float(np.exp(np.log(cut).mean())), kind)

    n_steps = int(np.ceil(np.log2(hi / lo) * SWEEP_STEPS_PER_OCTAVE)) + 1
    grid = lo * step ** np.arange(n_steps)
    pos = np.clip(np.log2(cut / lo) * SWEEP_STEPS_PER_OCTAVE, 0.0, n_steps - 1 - 1e-9)
    i0 = np.floor(pos).astype(int)
    frac = pos - i0

    out = np.zeros_like(y)
    for k, hz in enumerate(grid):
        w = np.where(i0 == k, 1.0 - frac, 0.0) + np.where(i0 + 1 == k, frac, 0.0)
        active = np.flatnonzero(w > 0)
        if len(active) == 0:
            continue
        a, b = max(int(active[0]) - SWEEP_PAD, 0), min(int(active[-1]) + SWEEP_PAD + 1, n)
        wk = w[a:b] if y.ndim == 1 else w[a:b, None]
        out[a:b] += _fixed_filter(y[a:b], float(hz), kind) * wk
    return out


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
    part: np.ndarray, vol_spec, band_spec, kind: str, mid_spec=None, sweep=None
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
    return (shaped * env(vol_spec, kind)).astype(np.float32)


# ── Overlap level ───────────────────────────────────────────────────────────────
# Measured on a phrase-aligned render: A's body ended at -13.9 dB and the first
# overlap bars sat at -18.5 dB, a 5 dB hole, because the cue-out landed where A's
# breakdown starts (A's own level drops 7.5 dB there), a fixed 0.85 duck took
# another 1.4 dB, and the "not louder than either side" guard was one scalar over
# the whole overlap, so the summed middle turned the END down too and B's body
# then jumped up 6 dB. A DJ rides the trims so the room never hears a step. Here
# the overlap's gain is anchored at both ends: A's level going in, B's level
# coming out, linear in dB between. The interior is left alone so a drop that
# lands inside the overlap still hits.
# Start anchor: the level of A's last real bars, i.e. what the ear just heard, so
# the overlap never opens with a step. Measured on Farrago → Heil (regression p04):
# anchoring the start on A's BODY level while A had gone into its breakdown two bars
# before the seam lifted the first overlap bar +9 dB, a 4.6 dB step exactly at the seam.
# End: unity. Every recipe has A at zero volume by the last bar, so the overlap's end
# IS B at its matched level and the bar after it is the same B at the same gain; there
# is nothing to anchor to. Anchoring the end on B's bars after the overlap tilted the
# techno_build overlap 8 dB down into Mosaic's 8-bar breakdown (2026-09-08).
OVERLAP_ANCHOR_BARS = 4  # bars measured on each side of each seam edge
OVERLAP_ANCHOR_BARS_WIDE = 8  # look this far back so drop-out bars can be skipped
DROPOUT_BELOW_DB = 6.0  # a reference bar this far under the loudest one is a break, not the level
EDGE_BARS = 2  # the ear compares the last couple of bars with the next couple; anchor on those


def _anchor_level_db(levels: np.ndarray) -> float:
    """
    The level the ear holds as "the track's level" over these bars: the median
    of the bars that are within DROPOUT_BELOW_DB of the loudest one. Measured on
    Alex Niggemann – Materium: the last four body bars read -22, -42, -25, -17;
    two are one-bar drop-outs, the plain median said -23 and the overlap was
    matched 4-6 dB under the -17 the listener had just heard.
    """
    levels = np.asarray(levels, dtype=np.float64)
    keep = levels[levels >= levels.max() - DROPOUT_BELOW_DB]
    return float(np.median(keep))


def _edge_level_db(levels: np.ndarray, edge: str) -> float:
    """
    Level at the seam edge: the median of the last (edge="end") or first
    (edge="start") EDGE_BARS bars that are not drop-outs. Measured on the house
    clip: an 8-bar median still left a 2-3 dB step at the seam because the ear
    compares the last two seconds with the next two, not eight bars with eight.
    """
    levels = np.asarray(levels, dtype=np.float64)
    keep = levels[levels >= levels.max() - DROPOUT_BELOW_DB]
    # the louder of the edge bars: a section rarely ends below the level the ear
    # just held, and the overlap must not start under it
    picked = keep[-EDGE_BARS:] if edge == "end" else keep[:EDGE_BARS]
    return float(np.max(picked))


OVERLAP_GAIN_MAX_DB = 9.0
# The first overlap bar may not sit more than this above the bar the ear just heard.
# `_edge_level_db` skips one-bar drop-outs on purpose (Materium), but a record that
# goes into its break two bars before the seam (Farrago → Heil) must not be lifted back
# to its pre-break level at the seam: that is heard as a step, not a ride.
SEAM_STEP_MAX_DB = 1.5


def _bar_levels_db(y: np.ndarray, bar_n: int) -> np.ndarray:
    n_bars = max(len(y) // bar_n, 1)
    return np.array([_rms_db(y[k * bar_n : (k + 1) * bar_n]) for k in range(n_bars)])


def _overlap_gain_ride(
    overlap: np.ndarray,
    a_body: np.ndarray,
    b_body: np.ndarray,
    bar_n: int,
    anchor_end: bool = True,
) -> tuple[np.ndarray, float, float]:
    """
    Gain the overlap so it opens at the level of A's last real bars and ends at unity.

    a_body: A's played window (already track-gained); the start anchor is the
    louder of its last two non-break bars (`_edge_level_db`). b_body and
    anchor_end are kept for the call signature; the end gain is always 0 dB
    (see the note above). Returns (overlap, g0_db, g1_db).
    """
    k = OVERLAP_ANCHOR_BARS_WIDE
    ov = _bar_levels_db(overlap, bar_n)
    if len(a_body) < bar_n:
        return overlap, 0.0, 0.0
    # bars counted back from the seam, so the last chunk IS the last bar heard
    a_levels = _bar_levels_db(a_body[len(a_body) % bar_n :], bar_n)
    la = _edge_level_db(a_levels[-k:], "end")
    # The overlap's own edges are judged within their end of the overlap only. An
    # overlap legitimately ramps (A's tail out, B in), so against the whole
    # overlap its quiet opening bars all look like drop-outs and the start
    # anchor lands mid-overlap: measured on Farrago – Sinner, the overlap opened
    # 7 dB under A's last bar and was gained -1.4 dB instead of +7.
    g0 = float(
        np.clip(la - _edge_level_db(ov[:k], "start"), -OVERLAP_GAIN_MAX_DB, OVERLAP_GAIN_MAX_DB)
    )
    if g0 > 0:  # continuity at the seam: no step up beyond what the ear just heard
        g0 = min(g0, max(0.0, float(a_levels[-1] + SEAM_STEP_MAX_DB - ov[0])))
    g1 = 0.0  # unity: the overlap ends as B alone at its matched level
    gain_db = np.linspace(g0, g1, len(overlap), dtype=np.float32)
    gain = (10 ** (gain_db / 20)).astype(np.float32)
    if overlap.ndim == 2:
        gain = gain[:, None]
    return (overlap * gain).astype(np.float32), g0, g1


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


# ── True-peak limiter ───────────────────────────────────────────────────────────
# The mix used to be scaled to -14 LUFS and then scaled DOWN AGAIN as a whole if
# any sample peaked above 0.98, so one loud moment turned the entire set down. A
# lookahead limiter touches only the loud moments.
LIMIT_CEILING_DB = -1.0  # true peak; headroom for lossy playback codecs
LIMIT_ATTACK_S = 0.005
LIMIT_RELEASE_S = 0.100
LIMIT_OVERSAMPLE = 4
_LIMIT_CHUNK = 1 << 20


def _true_peak_per_sample(x: np.ndarray) -> np.ndarray:
    """Max |value| of the 4x-oversampled signal within each original sample, across channels."""
    n = len(x)
    xs = x if x.ndim == 2 else x[:, None]
    out = np.empty(n, dtype=np.float32)
    pad = 64
    for a in range(0, n, _LIMIT_CHUNK):
        b = min(a + _LIMIT_CHUNK, n)
        lo, hi = max(a - pad, 0), min(b + pad, n)
        seg = resample_poly(xs[lo:hi].astype(np.float64), LIMIT_OVERSAMPLE, 1, axis=0)
        seg = np.abs(seg).max(axis=1)
        seg = seg[(a - lo) * LIMIT_OVERSAMPLE : (b - lo) * LIMIT_OVERSAMPLE]
        out[a:b] = seg.reshape(b - a, LIMIT_OVERSAMPLE).max(axis=1)
    return out


def _true_peak_limiter(
    x: np.ndarray,
    ceiling_db: float = LIMIT_CEILING_DB,
    attack_s: float = LIMIT_ATTACK_S,
    release_s: float = LIMIT_RELEASE_S,
) -> tuple[np.ndarray, dict]:
    """
    Lookahead true-peak limiter.

    Required gain per sample from the 4x-oversampled peak; a moving minimum
    looking `attack_s` ahead, then a causal average over the same length, so
    the gain has fully arrived when the peak does and moves smoothly getting
    there. Release is an exponential recovery of the gain reduction in dB,
    computed as a max over lagged copies (vectorised, no per-sample loop).
    """
    ceiling = 10 ** (ceiling_db / 20)
    pk = _true_peak_per_sample(x)
    g_req = np.minimum(1.0, ceiling / np.maximum(pk, 1e-9)).astype(np.float64)
    if g_req.min() >= 1.0 - 1e-6:
        return x, {"max_reduction_db": 0.0, "reduced_pct": 0.0}

    a = max(int(attack_s * SR), 2)
    # window [n, n+a): a negative origin shifts the filter window to later samples
    g_min = minimum_filter1d(g_req, size=a, origin=-(a // 2), mode="nearest")
    # causal average over [n-a, n]: a positive origin shifts the window to earlier samples
    g_att = uniform_filter1d(g_min, size=a, origin=(a - 1) // 2, mode="nearest")
    red = -20 * np.log10(np.maximum(g_att, 1e-6))

    r = max(int(release_s * SR), 1)
    lags = np.unique(np.round(np.geomspace(1, 6 * r, 48)).astype(int))
    out = red.copy()
    for k in lags:
        if k >= len(red):
            break
        np.maximum(out[k:], red[:-k] * np.exp(-k / r), out=out[k:])
    gain = (10 ** (-out / 20)).astype(np.float32)
    y = (x * (gain[:, None] if x.ndim == 2 else gain)).astype(np.float32)
    stats = {
        "max_reduction_db": round(float(out.max()), 2),
        "reduced_pct": round(float(100 * np.mean(out > 0.05)), 2),
    }
    return y, stats


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
        key = normalise_key(note, scale)
    except Exception:
        key = "?"
    return {"energy": energy, "onset": onset, "lufs": lufs, "key": key}


# ── Structure by intent ─────────────────────────────────────────────────────────
# The slot's curve value (energy_target01 from the plan) decides how the two
# records meet. Not every cue-in is on the drop: a minimal set wants long
# buildups, entries over breakdowns and a few well-placed drops; a peak set wants
# the incoming drop to be the moment. Loose files without a plan keep the middle
# behaviour.
STRUCTURE_LOW = 0.35  # below: never land on a drop; enter over the outgoing breakdown
STRUCTURE_HIGH = 0.65  # at or above: end the overlap on the incoming drop when reachable
TAIL_QUIET_BONUS = 0.15  # window score bonus when the outgoing tail is quiet material
QUIET_SECTIONS = frozenset({"breakdown", "outro", "intro"})


def structure_regime(e_target01: float | None) -> str:
    if e_target01 is None:
        return "mid"
    if e_target01 < STRUCTURE_LOW:
        return "low"
    if e_target01 >= STRUCTURE_HIGH:
        return "high"
    return "mid"


def _section_at(sections: list, bar: int) -> str | None:
    for sec in sections:
        if sec["bars"][0] <= bar < sec["bars"][1]:
            return sec["label"]
    return None


def _land_on_drop(
    cue_in: int, drop_bar: int, phrase: int, max_bars: int
) -> tuple[int, int] | None:
    """
    Entry bar and length so the overlap ENDS on `drop_bar`, in whole phrases.
    None when the drop is less than a phrase past the cue-in.
    """
    span = drop_bar - cue_in
    if span < phrase:
        return None
    n = min(_whole_phrases(span, phrase), _whole_phrases(max_bars, phrase))
    if n < phrase:
        return None
    return drop_bar - n, n


def _next_drop_after(sections: list, bar: int, min_gap: int) -> int | None:
    """First drop section starting at least `min_gap` bars after `bar`, else None."""
    for sec in sections:
        if sec["label"] == "drop" and sec["bars"][0] >= bar + min_gap:
            return int(sec["bars"][0])
    return None


def _end_anchored_swap(n_bars: int) -> tuple[list, list]:
    """Bass swap in the last bar: A's low leaves as B's arrives with the drop."""
    t0 = max(n_bars - SWAP_BARS, 1) / n_bars
    return [(0.0, 1), (t0, 1), (1.0, 0)], [(0.0, 0), (t0, 0), (1.0, 1)]


def _choose_window(
    info: dict,
    target_bars: int,
    tail_bars: int,
    e_target01: float | None,
    phrase_bars: int | None = None,
    quiet_tail: bool = False,
) -> tuple[int, int]:
    """
    Pick which window of the track to play, and how long it runs.

    `quiet_tail`: the NEXT slot is low-energy, so prefer a cue-out whose tail
    (what plays under the incoming record) is a breakdown, outro or intro.

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
    Both land on the track's phrase grid: bar index ≡ `phrase_offset` (mod phrase).
    """
    n = len(info["bars"])
    be = np.array(info["bar_energy"], dtype=np.float32)
    phrase = phrase_bars or info.get("phrase_bars", PHRASE_BARS)
    offset = int(info.get("phrase_offset", 0)) % phrase
    # Real section changes only. Bar 0 and the final bar are the ends of the
    # file, not musical boundaries, and rewarding them pins every window to the
    # top of the track.
    bounds = {s["bars"][0] for s in info["sections"]} | {s["bars"][1] for s in info["sections"]}
    bounds -= {0, n, n - 1}

    labels = {s["label"]: s for s in info["sections"]}
    first_ok = labels["intro"]["bars"][1] if "intro" in labels else 0
    lo_start = max(first_ok - 8, 0)
    lo_start += (offset - lo_start) % phrase  # first phrase start at or after lo_start
    last_ok = n - 1 - tail_bars
    room = last_ok - lo_start
    target_bars = min(target_bars, room)
    if target_bars < MIN_BODY_BARS:  # short track: play whatever it has
        return lo_start, max(last_ok, lo_start + 1)

    if e_target01 is None:
        target = np.quantile(be, 0.65)  # default: energetic but not only-the-drop
    else:
        target = np.quantile(be, 0.25 + 0.55 * float(e_target01))

    # whole phrases only, so the cue-out is a phrase line whenever the cue-in is
    target_bars = max(_whole_phrases(target_bars, phrase), MIN_BODY_BARS)
    lengths = [
        target_bars + d
        for d in range(-LENGTH_FLEX_BARS, LENGTH_FLEX_BARS + 1, phrase)
        if MIN_BODY_BARS <= target_bars + d <= room
    ]
    best, best_score = None, -np.inf
    for length in lengths:
        drift = LENGTH_PENALTY * abs(length - target_bars) / target_bars
        for start in range(lo_start, last_ok - length + 1, phrase):
            end = start + length
            score = -float(np.abs(be[start:end] - target).mean())
            # The cue-out earns more than the cue-in: it is where the record
            # hands over, so a ragged end is the one the ear catches.
            score += CUE_IN_BONUS * (start in bounds) + CUE_OUT_BONUS * (end in bounds) - drift
            if quiet_tail and _section_at(info["sections"], end) in QUIET_SECTIONS:
                score += TAIL_QUIET_BONUS
            if score > best_score:
                best, best_score = (start, end), score
    return best if best else (lo_start, last_ok)


PHRASE_BARS = 8  # windows snap to 8-bar phrases


TEMPO_WINDOW_S = 180.0  # middle of the track: enough bars for 0.01 BPM, skips intro/outro
TEMPO_SEARCH = 0.06  # search ±6% around the DeepRhythm hint; the next bar-multiple is 25% away
TEMPO_MAX_DEVIATION = 0.08  # further than this from the hint is a tracking failure


def _measure_tempo(y: np.ndarray, bpm_hint: float) -> float | None:
    """
    Tempo from the audio itself: autocorrelation of the kick-onset envelope at
    the one-bar lag, with a parabolic peak. Frames are ONSET_HOP (2.9 ms), so
    one frame is 0.16% of a bar and the interpolated peak lands near 0.01%.

    This exists because the beat grid cannot give a precise tempo. The tracker
    runs at hop 512 / 22,050 Hz, so every beat time is rounded to 23.2 ms and
    the median inter-beat interval snaps to a few values: tracks at 134, 135
    and 137 BPM all measured 136.05, a 1.5% error that drifted the beats half a
    beat inside one overlap. Returns None when no periodic peak is found.
    """
    mono = _to_mono(y)
    n_win = int(TEMPO_WINDOW_S * SR)
    if len(mono) > n_win:
        start = (len(mono) - n_win) // 2
        mono = mono[start : start + n_win]
    env = _kick_envelope(mono)
    env = env - env.mean()
    n = len(env)
    bar_frames = 4 * 60.0 / bpm_hint * SR / ONSET_HOP
    lo = int(bar_frames * (1 - TEMPO_SEARCH))
    hi = int(bar_frames * (1 + TEMPO_SEARCH)) + 2
    if hi + 1 >= n or lo < 2:
        return None
    size = 1 << (2 * n - 1).bit_length()
    ac = np.fft.irfft(np.abs(np.fft.rfft(env, size)) ** 2)[:n]
    k = lo + int(np.argmax(ac[lo:hi]))
    if ac[k] <= 0:
        return None
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    denom = a - 2 * b + c
    if abs(denom) > 1e-12:
        k = k + 0.5 * (a - c) / denom
    return float(4 * 60.0 / (k * ONSET_HOP / SR))


def _precise_bpm(info: dict, y: np.ndarray | None = None) -> float:
    """
    Refine the (integer-quantised) DeepRhythm BPM.

    With audio, the tempo is measured from the kick envelope (`_measure_tempo`).
    Without it, fall back to the median inter-beat interval of the grid, which
    is only good to one 23 ms frame and is kept for callers that have no audio.
    """
    hint = float(info["bpm"])
    if y is not None:
        measured = _measure_tempo(y, hint)
        if measured is not None and abs(measured - hint) / hint <= TEMPO_MAX_DEVIATION:
            return measured
        log.warning("tempo measurement failed for hint %.1f BPM — using the beat grid", hint)
    beats = np.asarray(info["beats"], dtype=np.float64)
    if len(beats) < 16:
        return hint
    measured = 60.0 / float(np.median(np.diff(beats)))
    if abs(measured - hint) / hint > TEMPO_MAX_DEVIATION:  # octave/tracking failure
        return hint
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


KICK_PERIODIC_MIN = 0.12  # below this the envelope has no beat-regular pulse to align on
# Periodicity is scale-invariant, so a kick band that is nearly silent can still
# look periodic from leakage of the other bands. Measured: real sections carry
# 0.05-0.45 of their onset energy in the kick band (an intro with a bare kick
# 0.049), a 3 kHz stab pattern with no kick 0.0008. The floor sits between.
KICK_ENERGY_MIN_SHARE = 0.02


def _onset_envelope(y: np.ndarray) -> np.ndarray:
    """Full-band rising spectral energy, frame by frame (same hop as the kick envelope)."""
    S = np.abs(librosa.stft(_to_mono(y), n_fft=1024, hop_length=ONSET_HOP))
    return np.maximum(np.diff(S.sum(axis=0), prepend=S[:, :1].sum()), 0.0)


def _envelope_periodicity(env: np.ndarray, bpm: float) -> float:
    """Normalised autocorrelation of an onset envelope at the one-beat lag."""
    e = env - env.mean()
    lag = int(60.0 / bpm * SR / ONSET_HOP)
    if len(e) <= lag + 8:
        return 0.0
    return float(np.dot(e[: len(e) - lag], e[lag:]) / (np.dot(e, e) + 1e-9))


def _kick_periodicity(y: np.ndarray, bpm: float) -> float:
    """
    How beat-regular the kick is in this stretch of audio. Measured on a real
    pair: 0.17-0.38 where the kick runs, -0.04 to 0.10 in a breakdown.
    """
    return _envelope_periodicity(_kick_envelope(y), bpm)


def _seam_envelopes(
    tail: np.ndarray, head: np.ndarray, bpm: float
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    The pair of onset envelopes a seam can be measured on, and which band.

    Kick band when both records have a beat-regular kick. Otherwise the
    full-band onset envelope. Measured on a real breakdown-to-drop seam: the
    kick correlation read -72 ms (corr 0.03, the outgoing kick was absent) while
    low-mid, mid and full band all agreed at -15 to -17 ms; hats alone read
    +220 ms because the breakdown's hats sit off the beat. None when neither
    band is periodic in both records: then the calibrated grids are trusted
    and nothing is shifted.
    """
    ka, kb = _kick_envelope(tail), _kick_envelope(head)
    fa, fb = _onset_envelope(tail), _onset_envelope(head)
    kick_present = (
        ka.sum() >= KICK_ENERGY_MIN_SHARE * fa.sum()
        and kb.sum() >= KICK_ENERGY_MIN_SHARE * fb.sum()
    )
    if kick_present and min(_envelope_periodicity(ka, bpm), _envelope_periodicity(kb, bpm)) >= (
        KICK_PERIODIC_MIN
    ):
        return ka, kb, "kick"
    if min(_envelope_periodicity(fa, bpm), _envelope_periodicity(fb, bpm)) >= KICK_PERIODIC_MIN:
        return fa, fb, "full"
    return None


# ── Alignment against the record's OWN grid ─────────────────────────────────────
# Cross-correlating A's audio against B's audio conflates alignment with content:
# on a techno pair the kick band read +103 ms (corr 0.12, A's kick weak and
# syncopated) while lowmid/mid/hats/full agreed at -116, and A's off-beat open
# hats put every other band exactly half a beat out. The beat tracker is trained
# to find the beat under exactly that, so its grid is the arbiter: measure how
# far each record's sounds sit from its OWN grid, within ±GRID_PHASE_MAX_S, on
# whichever band matches the grid best, and shift by the difference. A 40 ms
# window can never produce a half-beat jump.
# Two stages. RHYTHM_BANDS (kick, bass) are read over ±half a beat: they are what
# a DJ aligns, and a tracker grid can be a quarter beat off (measured on Ben
# Böhmer – Vale: kick, bass and mids all 122 ms before the grid, corr 0.60; a
# 40 ms window read the hats at 0 and called it fine). Hats and mids are kept
# OUT of that decision: on Sin Sin – Break Down the off-beat open hats put
# every upper band exactly half a beat out. They are only consulted within
# ±40 ms when neither rhythm band is readable.
GRID_PHASE_MAX_S = 0.040
GRID_PHASE_MIN_CORR = 0.04  # floor; the working threshold scales with 1/sqrt(frames)
KICK_BASS_AGREE_S = 0.060  # kick and bass within this: a clean kick, and it decides
RHYTHM_BANDS = ((30, 130, "kick"), (130, 1000, "lowmid"))
UPPER_BANDS = ((1000, 5000, "mid"), (5000, 12000, "high"))
GRID_BANDS = RHYTHM_BANDS + UPPER_BANDS
SEAM_METHOD = "xcorr"  # what the approved ear-test clips used; "grid" kept for experiments, sounded worse to Anas on techno pair 2


def _band_envelopes(y: np.ndarray) -> dict:
    """Rising spectral energy per band plus full band, all at ONSET_HOP."""
    S = np.abs(librosa.stft(_to_mono(y), n_fft=1024, hop_length=ONSET_HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    out = {}
    for lo, hi, name in GRID_BANDS:
        band = S[(f >= lo) & (f < hi)].sum(axis=0)
        out[name] = np.maximum(np.diff(band, prepend=band[:1]), 0.0)
    full = S.sum(axis=0)
    out["full"] = np.maximum(np.diff(full, prepend=full[:1]), 0.0)
    return out


def _phase_on_bands(
    y: np.ndarray, beat_times: np.ndarray, bands: tuple, max_shift_s: float, floor_k: float
) -> tuple[float, float, str] | None:
    n_frames = int(len(y) / ONSET_HOP) + 1
    beats = beat_times[(beat_times >= 0) & (beat_times < len(y) / SR)]
    if len(beats) < 4 or n_frames < 64:
        return None
    train = np.zeros(n_frames)
    train[np.clip(np.round(beats * SR / ONSET_HOP).astype(int), 0, n_frames - 1)] = 1.0
    train = (train - train.mean()) / (train.std() + 1e-9)
    max_lag = max(int(max_shift_s * SR / ONSET_HOP), 1)
    envs = _band_envelopes(y)
    results = {}
    for _, _, band in bands:
        env = envs[band]
        n = min(len(env), len(train))
        e = env[:n]
        if e.std() < 1e-9:
            continue
        e = (e - e.mean()) / e.std()
        corr = correlate(e, train[:n], mode="full") / n
        centre = n - 1
        window = corr[centre - max_lag : centre + max_lag + 1]
        k = int(np.argmax(window))
        peak = float(window[k])
        if 0 < k < len(window) - 1:
            a_, b_, c_ = window[k - 1], window[k], window[k + 1]
            denom = a_ - 2 * b_ + c_
            k = k + (0.5 * (a_ - c_) / denom if abs(denom) > 1e-12 else 0.0)
        results[band] = (float((k - max_lag) * ONSET_HOP / SR), peak, band)
    if not results:
        return None
    # random peaks over 2·max_lag+1 lags scale like sqrt(2 ln L / frames); stay well above
    threshold = max(GRID_PHASE_MIN_CORR, floor_k / np.sqrt(n_frames))
    best = max(results.values(), key=lambda r: r[1])
    if best[1] < threshold:
        return None
    kick, bass = results.get("kick"), results.get("lowmid")
    if kick is not None and kick[1] >= threshold:
        if bass is None or bass[1] < threshold:
            return kick
        # In four-on-the-floor music the kick and the bass sit together. When
        # the two bands disagree by more than a 16th the kick band is not
        # reading a clean kick (a syncopated sub, a weak kick under a bass
        # line), so the better-correlating band decides; otherwise the kick.
        if abs(kick[0] - bass[0]) <= KICK_BASS_AGREE_S:
            return kick
        return kick if kick[1] >= bass[1] else bass
    return best


def grid_phase(
    y: np.ndarray, beat_times: np.ndarray, bpm: float
) -> tuple[float, float, str] | None:
    """
    How far this record's sounds sit from its own beat grid, in seconds
    (positive = the sounds land after the grid beats). `beat_times` are the
    grid beats inside `y`, relative to its start.

    Stage 1: kick and bass over ±half a beat, kick preferred. Stage 2, only if
    neither is readable: mids and highs within ±40 ms, where off-beat hats
    cannot reach. Returns (offset_s, corr, band) or None.
    """
    got = _phase_on_bands(y, beat_times, RHYTHM_BANDS, 0.5 * 60.0 / bpm, floor_k=6.0)
    if got is not None:
        return got
    return _phase_on_bands(y, beat_times, UPPER_BANDS, GRID_PHASE_MAX_S, floor_k=5.0)


def calibrate_grid_phase_own(y: np.ndarray, beat_times: np.ndarray, bpm: float) -> float:
    """
    Whole-track grid phase error from `grid_phase`, over the middle half of the
    record (intro and outro often have no pulse). Replaces the kick-only
    ±¼-beat search for tracker grids: on one real track that search moved a
    correct grid 64 ms late because the kick band carried a syncopated bass.
    """
    n = len(y)
    a, b = n // 4, 3 * n // 4
    seg = y[a:b]
    beats = beat_times - a / SR
    got = grid_phase(seg, beats, bpm)
    return 0.0 if got is None else got[0]


def seam_offset_from_grids(
    tail: np.ndarray,
    head: np.ndarray,
    beats_tail: np.ndarray,
    beats_head: np.ndarray,
    bpm: float,
) -> tuple[float, dict]:
    """
    Seam offset as the difference of the two records' own-grid residuals.
    Positive means the head's sounds land late and it has to start earlier
    (same sign as `measure_seam_offset`). 0.0 when a side has no readable
    pulse: the grids are then trusted as they are.
    """
    pa, pb = grid_phase(tail, beats_tail, bpm), grid_phase(head, beats_head, bpm)
    detail = {
        "tail": None if pa is None else (round(pa[0] * 1000, 1), round(pa[1], 2), pa[2]),
        "head": None if pb is None else (round(pb[0] * 1000, 1), round(pb[1], 2), pb[2]),
    }
    if pa is None or pb is None:
        return 0.0, detail
    return float(pb[0] - pa[0]), detail


SEAM_AGREE_S = 0.060  # kick and the other layers within this: the kick decides
SEAM_HALF_BEAT_TOL = 0.20  # a disagreement within 20% of half a beat is off-beat layering


def _lag_offset(ea: np.ndarray, eb: np.ndarray, bpm: float) -> tuple[float, float]:
    """Cross-correlation lag between two envelopes over ±½ beat → (offset_s, peak corr)."""
    n = min(len(ea), len(eb))
    if n < 32:
        return 0.0, 0.0
    ea, eb = ea[:n], eb[:n]
    ea = (ea - ea.mean()) / (ea.std() + 1e-9)
    eb = (eb - eb.mean()) / (eb.std() + 1e-9)
    max_lag = max(int(0.5 * (60.0 / bpm) * SR / ONSET_HOP), 1)
    corr = correlate(ea, eb, mode="full") / n
    centre = n - 1
    window = corr[centre - max_lag : centre + max_lag + 1]
    k = int(np.argmax(window))
    peak = float(window[k])
    if 0 < k < len(window) - 1:
        a, b, c = window[k - 1], window[k], window[k + 1]
        denom = a - 2 * b + c
        k = k + (0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0)
    return -float((k - max_lag) * ONSET_HOP / SR), peak


# ── Kick-hit verifier ──────────────────────────────────────────────────────────
# The band correlations above match PATTERNS: everything in 30–130 Hz, kick and
# bass note alike. On Amelie Lens → Mosaic the best pattern match put her kick on
# Mosaic's off-beat bass note and shifted an aligned pair +121 ms (a 16th). This
# check compares the drums themselves: the single loudest low hit per beat in each
# record, matched to its nearest neighbour in the other. It is only trusted when
# both records give a clear kick on most beats and the 4-bar windows agree; then a
# residual beyond KICK_VERIFY_MIN_CORRECTION_S corrects the correlation's shift.
# Measured on the 15-pair regression set: decisive on 5 pairs (the false +121
# among them), spread over 100 ms on the other 10 (sparse kicks, syncopated low
# end, drum and bass), where it must stay silent.
KICK_VERIFY_MIN_PER_BEAT = 0.8  # clear kick on at least this share of beats, both sides
KICK_VERIFY_AGREE_S = 0.040  # 4-bar window medians within this of each other
KICK_VERIFY_MIN_CORRECTION_S = 0.030  # below this the correlation was right
KICK_VERIFY_MAX_BEAT_FRACTION = 1 / 3  # never flip to a neighbouring beat, only nudge
KICK_VERIFY_WINDOW_BARS = 4


def _strong_kicks(env: np.ndarray, beat_s: float) -> np.ndarray:
    """One kick per beat: the loudest rising kick-band frame within ±0.6 beat, > 6× median."""
    size = 2 * int(0.6 * beat_s * SR / ONSET_HOP) + 1
    is_max = (env == maximum_filter1d(env, size=size, mode="nearest")) & (env > np.median(env) * 6)
    t = np.flatnonzero(is_max) * ONSET_HOP / SR
    keep, last = [], -1e9
    for x in t:
        if x - last >= 0.75 * beat_s:
            keep.append(x)
            last = x
    return np.asarray(keep)


def kick_hit_residual(tail: np.ndarray, head: np.ndarray, bpm: float) -> tuple[float, dict] | None:
    """
    Seconds the head must still move earlier so its kicks land on the tail's, or
    None when the reading cannot be trusted (gate). detail carries the numbers.
    """
    beat = 60.0 / bpm
    n = min(len(tail), len(head))
    n_beats = n / SR / beat
    if n_beats < 2 * KICK_VERIFY_WINDOW_BARS * 4:
        return None
    ka = _strong_kicks(_kick_envelope(tail[:n]), beat)
    kb = _strong_kicks(_kick_envelope(head[:n]), beat)
    detail = {"a_per_beat": round(len(ka) / n_beats, 2), "b_per_beat": round(len(kb) / n_beats, 2)}
    if min(detail["a_per_beat"], detail["b_per_beat"]) < KICK_VERIFY_MIN_PER_BEAT:
        return None
    win = KICK_VERIFY_WINDOW_BARS * 4 * beat
    medians, all_d = [], []
    for w in range(int(n / SR // win)):
        a = ka[(ka >= w * win) & (ka < (w + 1) * win)]
        b = kb[(kb >= w * win) & (kb < (w + 1) * win)]
        if len(a) < 4 or len(b) < 4:
            return None
        d = np.array([b[np.argmin(np.abs(b - x))] - x for x in a])
        d = d[np.abs(d) < beat / 2]
        if len(d) < 4:
            return None
        medians.append(float(np.median(d)))
        all_d.extend(d.tolist())
    detail["windows_ms"] = [round(m * 1000, 1) for m in medians]
    if len(medians) < 2 or max(medians) - min(medians) > KICK_VERIFY_AGREE_S:
        return None
    residual = float(np.median(all_d))  # + : head's kicks land late → move the head earlier
    if abs(residual) > KICK_VERIFY_MAX_BEAT_FRACTION * beat:
        return None
    return residual, detail


def seam_decision(tail: np.ndarray, head: np.ndarray, bpm: float) -> tuple[float, str, dict]:
    """
    Seam offset (seconds the head must move earlier) and which reading decided.

    The kick is what a DJ aligns, but a weak kick reads noise. Two real seams,
    both with kick correlation 0.12, went opposite ways by ear:
      Sin Sin → Nova:      kick +103 ms, bass/mids/hats/full −116 ms. Disagreement
                           219 ms = half a beat at 127.5: off-beat hats. Kick right.
      Wigbert → Joyhauser: kick +75 ms,  bass/mids/hats/full +194 ms. Disagreement
                           119 ms, a quarter beat, which no pattern makes. Kick wrong.
    So: agree → kick; disagree by half a beat → kick; disagree otherwise → the
    consensus of the other layers; no kick → consensus; nothing → 0.
    """
    ea, eb = _band_envelopes(tail), _band_envelopes(head)

    def ok(a, b):
        return (
            min(_envelope_periodicity(a, bpm), _envelope_periodicity(b, bpm)) >= KICK_PERIODIC_MIN
        )

    kick_present = (
        ea["kick"].sum() >= KICK_ENERGY_MIN_SHARE * ea["full"].sum()
        and eb["kick"].sum() >= KICK_ENERGY_MIN_SHARE * eb["full"].sum()
    )
    kick = (
        _lag_offset(ea["kick"], eb["kick"], bpm)
        if kick_present and ok(ea["kick"], eb["kick"])
        else None
    )
    others = {
        b: _lag_offset(ea[b], eb[b], bpm)
        for b in ("lowmid", "mid", "high", "full")
        if ok(ea[b], eb[b])
    }
    detail = {
        "kick": kick,
        "others": {b: (round(o * 1000, 1), round(c, 2)) for b, (o, c) in others.items()},
    }
    consensus = float(np.median([o for o, _ in others.values()])) if len(others) >= 2 else None
    if kick is None and consensus is None:
        return 0.0, "none", detail
    if kick is None:
        return consensus, "consensus", detail
    if consensus is None:
        return kick[0], "kick", detail
    diff = abs(kick[0] - consensus)
    half = 0.5 * 60.0 / bpm
    if diff <= SEAM_AGREE_S:
        return kick[0], "kick", detail
    if abs(diff - half) <= SEAM_HALF_BEAT_TOL * half:
        return kick[0], "kick(offbeat layers)", detail
    return consensus, "consensus", detail


def measure_seam_offset(tail: np.ndarray, head: np.ndarray, bpm: float) -> float:
    """
    How far the incoming track's beat sits from the outgoing one's, in seconds.
    Positive means the head lands late and has to start earlier. See
    `seam_decision` for which layer is trusted.
    """
    return seam_decision(tail, head, bpm)[0]


def _prepare_track(
    path: str | Path,
    target_bpm: float,
    max_tail_bars: int,
    target_bars: int,
    e_target01: float | None = None,
    phrase_bars: int = DEFAULT_PHRASE_BARS,
    quiet_tail: bool = False,
    cue: tuple[int | None, int | None] | None = None,
) -> dict:
    info = segment(path)
    y = load_audio(path)  # (samples, channels)
    bpm = _precise_bpm(info, y)
    log.info("  %s: tempo %.2f BPM (DeepRhythm %.0f)", Path(path).stem[:30], bpm, info["bpm"])
    rate = target_bpm / bpm
    _check_stretch_rate(rate, Path(path).name)
    y = _stretch(y, rate).astype(np.float32)
    bars = np.array(info["bars"], dtype=np.float64) / rate

    cue_in_bar, cue_out_bar = _choose_window(
        info, target_bars, max_tail_bars, e_target01, phrase_bars, quiet_tail
    )
    if cue is not None:  # test hook: the regression set pins the window on a chosen part
        cue_in_bar = cue_in_bar if cue[0] is None else int(cue[0])
        cue_out_bar = cue_out_bar if cue[1] is None else int(cue[1])
        if not 0 <= cue_in_bar < cue_out_bar <= len(bars) - 1:
            raise ValueError(
                f"{Path(path).stem}: cue override {cue} outside the track's {len(bars)} bars"
            )
        log.info(
            "  %s: window pinned to bars %d–%d", Path(path).stem[:30], cue_in_bar, cue_out_bar
        )
    if info.get("downbeat_source", "kick-phase") != "beat_this":
        log.warning(
            "  %s: downbeats from kick phase (confidence %.2f) — bar alignment is a guess",
            Path(path).stem[:30],
            info.get("downbeat_confidence", 0.0),
        )

    drop_bar = None  # slam entry point: first drop inside the window, if any
    for s in info["sections"]:
        if s["label"] == "drop" and s["bars"][0] >= cue_in_bar:
            drop_bar = s["bars"][0]
            break

    body = y[int(bars[cue_in_bar] * SR) : int(bars[cue_out_bar] * SR)]
    feats = _analyse_body(body if len(body) > SR * 20 else y)

    beat_times = np.asarray(info["beats"], dtype=np.float64) / rate  # stretched
    if info.get("downbeat_source") == "beat_this" and SEAM_METHOD == "grid":
        eps = calibrate_grid_phase_own(y, beat_times, target_bpm)
    else:  # a librosa grid can be far off; keep the wide kick search for it
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
        "sections": info["sections"],
        "phrase_offset": int(info.get("phrase_offset", 0)),
        "downbeat_source": info.get("downbeat_source", "kick-phase"),
        "downbeat_confidence": float(info.get("downbeat_confidence", 0.0)),
        **feats,
    }


def _last_track_end_bar(
    track: dict,
    cue_out: int,
    body_start: int,
    phrase: int,
    bar_dur: float,
    elapsed_s: float,
    total_s: float,
) -> int:
    """
    Bar where the last track stops so the mix lands on `total_s`. `elapsed_s`
    is the mix length up to the end of the final overlap and `body_start` the
    bar of the track where its solo body begins. The end sits on the track's
    phrase grid (counted from its cue-in), keeps at least one phrase of body,
    and never passes the track's last bar.
    """
    needed = (total_s - elapsed_s) / bar_dur
    want = max(int(round(needed / phrase)) * phrase, phrase)
    end = body_start + want
    end -= (end - track["cue_in_bar"]) % phrase  # onto the track's own phrase grid
    end = max(end, body_start + phrase)
    return int(min(end, len(track["bars"]) - 1))


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
    total_minutes: float | None = None,
    force_bars: int | None = None,
    cue_overrides: list | None = None,
) -> dict:
    """
    total_minutes: when given, the LAST track keeps playing past its window until
    the mix reaches this length (whole phrases; it stops early only if the track
    runs out), or is cut short if the mix would overshoot by more than a phrase.
    The earlier tracks keep their genre-typical windows, so this is how a set
    lands on the length that was asked for.
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
    force_bars: override the overlap length (bars) of every transition; still
    clamped to the audio available and to whole phrases. Test hook.
    cue_overrides: per track, None or (cue_in_bar, cue_out_bar) with None for
    "keep the mixer's choice". Pins a track's window on a chosen part. Test hook.
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
    phrase = GENRE_PHRASE_BARS.get(genre or "", DEFAULT_PHRASE_BARS)
    if play_minutes is None:
        play_minutes = GENRE_PLAY_MINUTES.get(genre or "", DEFAULT_PLAY_MINUTES)
        log.info("play length: %.2f min from genre %r", play_minutes, genre or "(none)")
    target_bars = max(int(round(play_minutes * 60 / bar_dur_t / phrase)) * phrase, MIN_BODY_BARS)

    tracks = []
    for i, p in enumerate(track_paths):
        et = energy_targets[i] if energy_targets else None
        next_et = energy_targets[i + 1] if energy_targets and i + 1 < len(track_paths) else None
        quiet_tail = structure_regime(next_et) == "low"
        cue = cue_overrides[i] if cue_overrides and i < len(cue_overrides) else None
        t = _prepare_track(p, target_bpm, max_tail, target_bars, et, phrase, quiet_tail, cue)
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
        regime = structure_regime(energy_targets[i] if energy_targets else None)
        lands_on_drop = False

        # DROP-ALIGNED ENTRY. When the incoming track has a drop, the strongest
        # move is to bring it in over its own buildup with the bass cut and the
        # filter closed, and let the low end swap back at the drop itself. That
        # needs the overlap to END on the drop, so the entry bar is measured
        # backwards from it. A rise already wants to hand over on a climb, so
        # it is promoted when the track gives us a drop to aim at.
        if (
            drop_align
            and regime != "low"
            and ttype == "rise"
            and cur["drop_bar"] is not None
            and cur["drop_bar"] - bars_table["drop"] >= 0
        ):
            log.info("  transition %d: rise → drop (aligning to the incoming drop)", i)
            ttype = "drop"
            recipe, n_bars = RECIPES[ttype], bars_table[ttype]
            lands_on_drop = True

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

        # incoming entry point. The cue-in is a phrase line. A slam is a cut at
        # the centre of its bars, so it starts half its length early and the cut
        # lands on the phrase line (or on the drop when there is one). A
        # drop-aligned blend ends on the drop. Everything else starts at cue-in.
        in_bar = cur["cue_in_bar"]
        if ttype == "slam":
            land_on = cur["drop_bar"] if cur["drop_bar"] is not None else cur["cue_in_bar"]
            in_bar = max(land_on - bars_table["slam"] // 2, 0)
        elif ttype == "drop" and cur["drop_bar"] is not None:
            in_bar = max(cur["drop_bar"] - n_bars, 0)
            lands_on_drop = True
        elif regime == "high":
            # a peak slot: whatever the type's EQ character, the overlap ends on
            # the incoming drop, so the bass swap and the handover ARE the drop.
            # The drop has to be at least a phrase past the cue-in, or there is
            # nothing to build under; a drop AT the cue-in is already the entry.
            target_drop = _next_drop_after(cur["sections"], cur["cue_in_bar"], phrase)
            landing = (
                _land_on_drop(
                    cur["cue_in_bar"],
                    target_drop,
                    phrase,
                    max(n_bars, int(bars_table[ttype] * LONG_STRETCH)),
                )
                if target_drop is not None
                else None
            )
            if landing is not None:
                in_bar, n_bars = landing
                cur["drop_bar"] = target_drop
                lands_on_drop = True
                log.info(
                    "  transition %d: %s lands on the incoming drop (%d bars)", i, ttype, n_bars
                )

        # clamp overlap to available audio on both sides (whole bars), then to
        # whole phrases so it ends on a phrase line of both records
        avail_prev = len(prev["audio"]) / SR - prev["bars"][prev["cue_out_bar"]]
        avail_cur = (len(cur["audio"]) / SR - cur["bars"][in_bar]) - MIN_BODY_BARS * bar_dur
        n_bars = max(min(n_bars, int(avail_prev / bar_dur), int(avail_cur / bar_dur)), 2)
        if ttype != "slam":
            n_bars = max(_whole_phrases(n_bars, phrase), 2)
        if (
            force_bars is not None
        ):  # test hook: final say on the length, within the audio available
            n_bars = max(
                min(int(force_bars), int(avail_prev / bar_dur), int(avail_cur / bar_dur)), 2
            )
        # a shortened overlap moves the entry with it, or the drop stops
        # landing on the seam and the whole point is lost
        if lands_on_drop and cur["drop_bar"] is not None:
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

        def seam_measure(tail_, head_, b0_):
            """Offset the head must move by, and the band it was read on."""
            if SEAM_METHOD == "grid":
                bt = prev["beat_times"] - a0 / SR
                bh = cur["beat_times"] - b0_ / SR
                off, det = seam_offset_from_grids(tail_, head_, bt, bh, target_bpm)
                band = "grid:" + "/".join(
                    str(v[2]) if v else "none" for v in (det["tail"], det["head"])
                )
                return off, band
            off, label, _ = seam_decision(tail_, head_, target_bpm)
            return off, label

        before, seam_band = seam_measure(tail, head, b0)
        after = before
        if seam_band.endswith("none") or seam_band == "none":
            log.info("  seam %d: no readable pulse on one side — grids trusted, no shift", i)
        # the grid method is bounded by GRID_PHASE_MAX_S per side; xcorr by half a beat
        max_shift = int(
            (1.0 if SEAM_METHOD == "grid" else 0.5)
            * 60.0
            / target_bpm
            * SR  # grid: ½ beat per side
        )
        total_shift = 0
        # Shifting the cut changes which audio is in the head, so the offset is
        # re-measured and re-corrected until it settles. One pass typically
        # leaves a few ms behind. The grid method is different: each record's
        # residual against its own grid does not move with the cut, so its
        # difference is applied exactly once and is then zero by construction.
        passes = 1 if SEAM_METHOD == "grid" else SEAM_PASSES
        for _ in range(passes):
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
            if SEAM_METHOD == "grid":
                after = before - total_shift / SR  # what the one shift could not remove (rounding)
            else:
                after, _ = seam_measure(tail, head, b0)
        if total_shift:
            log.info(
                "  seam %d: beat offset %+.1f ms → %+.1f ms (shifted %+.1f ms)",
                i,
                before * 1000,
                after * 1000,
                total_shift / SR * 1000,
            )

        # Verify the correlation's shift against the kick hits themselves (gated).
        kick_verify_ms = None
        verify = kick_hit_residual(tail, head, target_bpm)
        if verify is not None:
            residual, vdetail = verify
            step = int(round(residual * SR))
            if abs(residual) > KICK_VERIFY_MIN_CORRECTION_S and b0 + step >= 0:
                b0 += step
                total_shift += step
                tail = prev["audio"][a0 : a0 + n_ov]
                head = cur["audio"][b0 : b0 + n_ov]
                n_ov = min(len(tail), len(head))
                tail, head = tail[:n_ov], head[:n_ov]
                kick_verify_ms = round(residual * 1000, 1)
                seam_band += "+kick-hits"
                log.info(
                    "  seam %d: kick hits %+.0f ms off after the shift (windows %s) — corrected, total %+.0f ms",
                    i,
                    residual * 1000,
                    vdetail["windows_ms"],
                    total_shift / SR * 1000,
                )
            else:
                log.info("  seam %d: kick hits confirm the shift (%+.0f ms)", i, residual * 1000)

        # The single correlation above is the AVERAGE lag over the overlap. Two
        # records at slightly different tempos read as aligned on average while
        # drifting apart at both ends, so the offset is also measured per
        # window across the overlap. Drift moves MOST windows, so the median is
        # the drift statistic; a lone half-beat reading in a breakdown, where
        # one record's kick pattern is off the beat, is counted, not averaged.
        win = int(SEAM_WINDOW_BARS * bar_dur * SR)
        window_offsets, periodic = [], []
        for k in range(0, n_ov - win + 1, win):
            if SEAM_METHOD == "grid":
                bt = prev["beat_times"] - (a0 + k) / SR
                bh = cur["beat_times"] - (b0 + k) / SR
                o, det = seam_offset_from_grids(
                    tail[k : k + win], head[k : k + win], bt, bh, target_bpm
                )
                o -= (
                    total_shift / SR
                )  # residuals ignore the cut, so remove the shift already applied
                ok = det["tail"] is not None and det["head"] is not None
            else:
                o, label_k, _ = seam_decision(tail[k : k + win], head[k : k + win], target_bpm)
                ok = label_k != "none"
            window_offsets.append(o)
            periodic.append(ok)
        window_offsets = window_offsets or [after]
        judged = [o for o, ok in zip(window_offsets, periodic) if ok]
        # a median of one window is that window; two is the least that can
        # say anything about a trend
        seam_drift = float(np.median(np.abs(judged))) if len(judged) >= 2 else None
        windows_off = int(sum(abs(o) > SEAM_DRIFT_WARN_S for o in judged))
        if seam_drift is not None and seam_drift > SEAM_DRIFT_WARN_S:
            log.warning(
                "  seam %d: beats drift — median window offset %.0f ms (avg %.1f ms)",
                i,
                seam_drift * 1000,
                after * 1000,
            )

        at = len(mix) / SR
        half_phrase = max(phrase // 2, 1)
        a_bands = _snap_bands(recipe["A_bands"], n_bars, half_phrase)
        b_bands = _snap_bands(recipe["B_bands"], n_bars, half_phrase)
        lead_cfg = LEAD.get(ttype)
        if lands_on_drop and ttype != "drop":
            a_low, b_low = _end_anchored_swap(n_bars)
            a_bands = {**(a_bands or {}), "low": a_low}
            b_bands = {**(b_bands or {}), "low": b_low}
            lead_cfg = LEAD["drop"]
        if lead_cfg and n_bars >= MIN_LEAD_BARS:
            a_vol, b_vol, a_mid, b_mid = _lead_envelopes(n_bars, lead_cfg, half_phrase)
            swap_at = at + n_ov / SR * a_vol[1][0]  # the handover starts where A begins to fall
            lead_swap = f"{int(swap_at // 60)}:{int(swap_at % 60):02d}"
        else:  # too short to hand over inside — straight crossfade
            a_vol, b_vol, a_mid, b_mid = recipe["A_vol"], recipe["B_vol"], None, None
            lead_swap = None
        overlap = _apply_recipe(
            tail, a_vol, a_bands, "out", a_mid, recipe.get("A_sweep")
        ) + _apply_recipe(head, b_vol, b_bands, "in", b_mid, recipe.get("B_sweep"))

        end_bar = cur["cue_out_bar"]
        if total_minutes is not None and i == len(tracks) - 1:
            end_bar = _last_track_end_bar(
                cur,
                end_bar,
                in_bar + n_bars,
                phrase,
                bar_dur,
                (len(mix) + n_ov) / SR,
                total_minutes * 60.0,
            )
            log.info(
                "  last track plays to bar %d of %d (%+d bars) to land on %.1f min",
                end_bar,
                len(cur["bars"]),
                end_bar - cur["cue_out_bar"],
                total_minutes,
            )
        body = cur["audio"][b0 + n_ov : int(cur["bars"][end_bar] * SR)]
        bar_n = int(bar_dur * SR)
        # anchors: each record's body level over the window it plays in this mix
        a_played = prev["audio"][
            int(prev["bars"][prev["cue_in_bar"]] * SR) : int(
                prev["bars"][prev["cue_out_bar"]] * SR
            )
        ]
        overlap, gain_in_db, gain_out_db = _overlap_gain_ride(
            overlap,
            a_played,
            body,
            bar_n,
            anchor_end=not lands_on_drop,
        )
        log.info(
            "  transition %d: overlap gain %+.1f dB in, %+.1f dB out", i, gain_in_db, gain_out_db
        )
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
                "seam_drift_ms": None if seam_drift is None else round(seam_drift * 1000, 1),
                "seam_band": seam_band,
                "kick_verify_ms": kick_verify_ms,
                "seam_shift_total_ms": round(total_shift / SR * 1000, 1),
                "overlap_gain_db": [round(gain_in_db, 1), round(gain_out_db, 1)],
                "seam_windows_off": [windows_off, len(judged), len(window_offsets)],
                "regime": regime,
                "lands_on_drop": lands_on_drop,
                "tail_section": _section_at(prev["sections"], prev["cue_out_bar"]),
                "head_section": _section_at(cur["sections"], in_bar),
                "in_bar": int(in_bar),
                "out_bar": int(prev["cue_out_bar"]),
                "phrase_offsets": [prev["phrase_offset"], cur["phrase_offset"]],
                "downbeats": [prev["downbeat_source"], cur["downbeat_source"]],
                "at": f"{int(at // 60)}:{int(at % 60):02d}",
                "end": f"{int((at + n_ov / SR) // 60)}:{int((at + n_ov / SR) % 60):02d}",
                "at_s": round(at, 3),
                "end_s": round(at + n_ov / SR, 3),
            }
        )
        log.info("  transition %d: %s (%d bars) at %s", i, ttype, n_bars, trans_report[-1]["at"])

    # Normalise the finished mix to a fixed loudness, then back off if the
    # peak would clip. Peak-only normalisation let a dense mix and a sparse one
    # land at very different perceived volumes.
    mix_db = _level_db(mix)
    if np.isfinite(mix_db):
        mix = mix * 10 ** ((MIX_TARGET_LUFS - mix_db) / 20)
    mix, limiter = _true_peak_limiter(mix.astype(np.float32))
    if limiter["max_reduction_db"] > 0:
        log.info(
            "limiter: max %.1f dB reduction on %.1f%% of samples (ceiling %.0f dBTP)",
            limiter["max_reduction_db"],
            limiter["reduced_pct"],
            LIMIT_CEILING_DB,
        )

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
        "true_peak_dbtp": round(float(20 * np.log10(_true_peak_per_sample(mix).max() + 1e-12)), 2),
        "limiter": limiter,
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
