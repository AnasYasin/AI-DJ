"""
Phase 4 — Heuristic transition labeller.

Reads mix CSVs + features.parquet, computes per-pair audio deltas, applies
rule-based heuristics to assign one of 6 transition types, and writes
data/processed/transition_labels.csv.

Transition classes (priority order applied during classification)
─────────────────────────────────────────────────────────────────
  slam   Hard cut — abrupt energy spike or intentional key clash
  rise   Energy builds into the next track — gradual momentum increase
  fade   Energy drops going into the next track — cool-down or breakdown
  melt   Ultra-smooth blend — tight BPM, compatible key, minimal energy shift
  wave   Rhythmic punchiness — high onset strength on both tracks, tight BPM
  blend  Standard smooth mix — catch-all for moderate transitions

Decision factors (in priority order, per MIR research)
───────────────────────────────────────────────────────
  1. BPM ratio          bpm_b / bpm_a
  2. Harmonic distance  Camelot wheel steps (0 = same key, 6 = max clash)
  3. Energy delta       energy_mean_b - energy_mean_a
  4. Loudness delta     loudness_lufs_b - loudness_lufs_a (dBLUFS)
  5. Onset strength     rhythmic punchiness proxy (librosa mean onset_strength)

Output CSV columns
──────────────────
  from_track_id, to_track_id, label, confidence,
  bpm_ratio, energy_delta, harm_dist, time_gap_norm

Run:
  conda activate djtest
  python src/features/transition_labeler.py
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

FEATURES_PATH = Path("data/processed/features.parquet")
LABELS_PATH = Path("data/processed/transition_labels.csv")
TRACKLIST_PATH = Path("data/interim/tracklist.csv")
MIX_METADATA_PATH = Path("data/processed/mix_metadata.csv")

# ── Classification thresholds ─────────────────────────────────────────────────
# All constants are tunable — rationale in CLAUDE.md and module docstring.

# BPM ratio = bpm_b / bpm_a
BPM_RATIO_TIGHT = 0.03  # |1 - ratio| < this → tight  (~3-4 BPM at 120)
BPM_RATIO_LOOSE = 0.07  # |1 - ratio| < this → loose  (~8 BPM at 120)

# Energy delta = energy_mean_b - energy_mean_a  (normalised RMS, ~0–1)
ENERGY_RISE_MIN = 0.08  # positive delta above this  → rising energy
ENERGY_FALL_MIN = -0.08  # negative delta below this  → falling energy
ENERGY_SLAM_MIN = 0.15  # spike above this + tight BPM → slam candidate
ENERGY_MELT_MAX = 0.05  # |delta| below this         → melt candidate

# Camelot wheel harmonic distance  (0 = same key, 6 = maximally incompatible)
HARM_PERFECT = 1  # ≤ 1 step   → very compatible
HARM_COMPATIBLE = 2  # ≤ 2 steps  → workable
HARM_CLASH = 5  # ≥ 5 steps  → harsh clash

# Loudness delta = loudness_lufs_b - loudness_lufs_a  (dBLUFS)
LOUD_MELT_MAX = 3.0  # |delta| below this → melt candidate

# Onset strength: proxy for rhythmic punchiness / danceability.
#
# build_features.py stores librosa's RAW mean onset_strength, which runs about
# 1.1 to 2.5 across the catalog. Every threshold in this project is written
# against the NORMALISED scale (raw / ONSET_SCALE), where the same catalog runs
# 0.22 to 0.51 and 0.35 sits at the median. Normalise before comparing, always.
#
# This was a live bug: the rules below read the raw column and compared it to
# 0.35, which 100% of tracks exceed, so the `wave` rule collapsed into
# "tight BPM and not rise and not fade" and the confidence bump always applied.
ONSET_SCALE = 5.0
ONSET_HIGH_MIN = 0.35  # both tracks need normalised onset > this for wave


def normalise_onset(raw):
    """Raw librosa onset_strength → the 0-1 scale every threshold is written for."""
    return np.minimum(np.asarray(raw, dtype=np.float32) / ONSET_SCALE, 1.0)


# Essentia's KeyExtractor names five keys with flats (Db Eb Gb Ab Bb). The
# catalog and the Camelot table use sharps. build_features.py normalised the
# catalog; the mixer did not, so ~20% of played windows came back as an unknown
# key (Camelot distance 2.5), which blocked melt and rise for those pairs.
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def normalise_key(note: str, scale: str) -> str:
    """Essentia (note, scale) → the catalog's key string, e.g. ("Eb", "minor") → "D#m"."""
    note = _FLAT_TO_SHARP.get(note, note)
    return f"{note}{'m' if scale == 'minor' else ''}"


# Time gap between consecutive mix entries (minutes)
# starting_time rolls over each hour; pairs with gap > this are skipped.
TIME_GAP_MAX_MIN = 15.0

# ── Camelot wheel ─────────────────────────────────────────────────────────────
# Mirrors train_model.py — duplicated to avoid coupling features ↔ models.
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


def _harmonic_dist(key_a: str, key_b: str) -> float:
    """
    Camelot wheel distance: 0 (same key) to 6 (maximally incompatible).
    Wraps around the 12-position wheel. Unknown keys default to 6.
    """
    ca = _CAMELOT.get(key_a, -1)
    cb = _CAMELOT.get(key_b, -1)
    if ca < 0 or cb < 0:
        return 6.0
    diff = abs(ca - cb)
    return float(min(diff, 12 - diff))


def _track_id(artist: str, track: str) -> str:
    """Deterministic 12-char ID — identical to preview_fetcher.py and train_model.py."""
    return hashlib.md5(f"{artist}|{track}".lower().encode()).hexdigest()[:12]


# ── Feature computation ───────────────────────────────────────────────────────


def _pair_features(row_a: pd.Series, row_b: pd.Series, time_gap: float) -> dict:
    """
    Compute the delta features for a consecutive track pair.

    Args:
        row_a:    features row for the outgoing track (anchor).
        row_b:    features row for the incoming track.
        time_gap: minutes between starting_time entries (post-rollover).

    Returns:
        Dict with keys: bpm_ratio, energy_delta, loudness_delta,
                        harm_dist, onset_a, onset_b, time_gap_norm.
    """
    bpm_ratio = float(row_b["bpm"]) / max(float(row_a["bpm"]), 1.0)
    energy_delta = float(row_b["energy_mean"]) - float(row_a["energy_mean"])
    loudness_delta = float(row_b["loudness_lufs"]) - float(row_a["loudness_lufs"])
    harm_dist = _harmonic_dist(str(row_a["key"]), str(row_b["key"]))
    onset_a = float(normalise_onset(row_a.get("onset_strength", 0.0)))
    onset_b = float(normalise_onset(row_b.get("onset_strength", 0.0)))
    time_gap_norm = float(np.clip(time_gap / TIME_GAP_MAX_MIN, 0.0, 1.0))

    return {
        "bpm_ratio": bpm_ratio,
        "energy_delta": energy_delta,
        "loudness_delta": loudness_delta,
        "harm_dist": harm_dist,
        "onset_a": onset_a,
        "onset_b": onset_b,
        "time_gap_norm": time_gap_norm,
    }


# ── Classification ────────────────────────────────────────────────────────────


def _classify(feats: dict) -> tuple[str, float]:
    """
    Apply heuristic rules to assign a transition label and confidence.

    Priority: slam → rise → fade → melt → wave → blend
    Returns (label, confidence) where confidence ∈ (0, 1].
    """
    ed = feats["energy_delta"]
    hd = feats["harm_dist"]
    br = feats["bpm_ratio"]
    ld = feats["loudness_delta"]
    oa = feats["onset_a"]
    ob = feats["onset_b"]

    bpm_dev = abs(1.0 - br)
    bpm_tight = bpm_dev < BPM_RATIO_TIGHT
    bpm_loose = bpm_dev < BPM_RATIO_LOOSE

    # ── slam ──
    # Sharp energy spike with tight BPM, OR key clash with energy jump.
    # Research: hard cuts work best within ±3 BPM for impact; key clashes
    # combined with energy spikes create intentional genre/mood breaks.
    is_slam = (bpm_tight and ed > ENERGY_SLAM_MIN) or (hd >= HARM_CLASH and ed > ENERGY_RISE_MIN)

    # ── rise ──
    # Clear positive energy trend, harmonically compatible, BPM in range.
    # Research: +1 semitone on Camelot (energy boost) favours rise; exclude slams.
    is_rise = (not is_slam) and ed > ENERGY_RISE_MIN and bpm_loose and hd <= HARM_COMPATIBLE

    # ── fade ──
    # Clear negative energy trend regardless of BPM/key.
    # Research: fades tolerate larger BPM and key jumps since the energy drop
    # itself signals the transition clearly to the listener.
    is_fade = (not is_slam) and ed < ENERGY_FALL_MIN

    # ── melt ──
    # Ultra-smooth: tight BPM, very compatible keys, minimal energy and loudness shift.
    # Research: requires BPM within ~3%, same or adjacent Camelot position,
    # energy delta < 0.05 and loudness delta < 3 dBLUFS.
    is_melt = (
        bpm_tight and hd <= HARM_PERFECT and abs(ed) < ENERGY_MELT_MAX and abs(ld) < LOUD_MELT_MAX
    )

    # ── wave ──
    # Rhythmically punchy on both sides, tight BPM, not a clear energy shift.
    # Research: high danceability (onset_strength proxy > 0.35) on both tracks
    # enables short rhythmic in/out mixing.
    is_wave = (
        bpm_tight and oa > ONSET_HIGH_MIN and ob > ONSET_HIGH_MIN and not is_rise and not is_fade
    )

    # Priority assignment
    if is_slam:
        label = "slam"
    elif is_rise:
        label = "rise"
    elif is_fade:
        label = "fade"
    elif is_melt:
        label = "melt"
    elif is_wave:
        label = "wave"
    else:
        label = "blend"

    confidence = _confidence(label, feats)
    return label, confidence


def _confidence(label: str, feats: dict) -> float:
    """
    Rule-strength confidence score in (0, 1].

    Scores are additive bonuses on top of a per-class base score.
    Higher score = the features match the rule more clearly.
    """
    ed = feats["energy_delta"]
    hd = feats["harm_dist"]
    br = feats["bpm_ratio"]
    ld = abs(feats["loudness_delta"])
    bpm_dev = abs(1.0 - br)

    if label == "slam":
        score = 0.65
        if ed > 0.20:
            score += 0.15  # very large spike
        if hd >= HARM_CLASH:
            score += 0.10  # confirmed key clash
        if bpm_dev < BPM_RATIO_TIGHT:
            score += 0.05

    elif label == "rise":
        score = 0.60
        # Proportional bonus: the further above the threshold, the clearer the rise
        overshoot = min((ed - ENERGY_RISE_MIN) / ENERGY_RISE_MIN, 1.0)
        score += 0.20 * overshoot
        if hd <= HARM_PERFECT:
            score += 0.10
        if bpm_dev < BPM_RATIO_TIGHT:
            score += 0.05

    elif label == "fade":
        score = 0.60
        overshoot = min((abs(ed) - abs(ENERGY_FALL_MIN)) / abs(ENERGY_FALL_MIN), 1.0)
        score += 0.20 * overshoot
        if hd <= HARM_PERFECT:
            score += 0.05

    elif label == "melt":
        score = 0.65
        if hd == 0:
            score += 0.15  # same key
        elif hd <= 1:
            score += 0.08  # adjacent key
        if bpm_dev < 0.01:
            score += 0.10  # essentially same BPM
        if abs(ed) < 0.02:
            score += 0.05
        if ld < 1.5:
            score += 0.05

    elif label == "wave":
        score = 0.60
        if bpm_dev < 0.015:
            score += 0.10
        if feats["onset_a"] > 0.50 and feats["onset_b"] > 0.50:
            score += 0.10

    else:  # blend
        score = 0.50

    return round(min(score, 0.95), 3)


# ── Pair building from mix CSVs ───────────────────────────────────────────────


def _time_gap(t_a: float, t_b: float) -> float:
    """
    Minutes between two starting_time values (0–59, rolling over each hour).
    If t_b < t_a the clock rolled over: gap = (60 - t_a) + t_b.
    """
    if t_b >= t_a:
        return t_b - t_a
    return (60.0 - t_a) + t_b


def label_transitions(
    features: pd.DataFrame,
    mix_csvs: list[Path],
    mix_metadata_path: Path = Path("data/processed/mix_metadata.csv"),
) -> pd.DataFrame:
    """
    Build consecutive-pair transition labels from mix CSVs + features.

    For each mix, consecutive tracks (by starting_time) form a pair.
    Pairs where either track has no feature row, or where the time gap
    exceeds TIME_GAP_MAX_MIN, are skipped.

    Args:
        features:          DataFrame from features.parquet.
                           Must contain: track_id, bpm, key, loudness_lufs,
                                         energy_mean, onset_strength.
        mix_csvs:          List of CSV paths, each with columns
                           mix_id, artist_name, track_name, starting_time.
        mix_metadata_path: Optional path to mix_metadata.csv for energy curve shapes.

    Returns:
        DataFrame with columns:
          mix_id, from_track_id, to_track_id, from_position, to_position,
          n_tracks_in_mix, mix_energy_curve_shape,
          label, confidence, bpm_ratio, energy_delta, harm_dist, time_gap_norm
    """
    feat_idx = features.set_index("track_id")
    feat_ids = set(feat_idx.index)

    # Load energy curve shapes if available
    curve_shapes: dict[str, str] = {}
    if mix_metadata_path.exists():
        meta = pd.read_csv(mix_metadata_path)
        curve_shapes = dict(zip(meta["mix_id"], meta["energy_curve_shape"]))
        log.info("Loaded energy curve shapes for %d mixes", len(curve_shapes))
    else:
        log.info("mix_metadata.csv not found — mix_energy_curve_shape will be 'unknown'")

    records = []

    for csv_path in mix_csvs:
        df = pd.read_csv(csv_path)

        # Only sequential tracks form training pairs
        if "play_type" in df.columns:
            df = df[df["play_type"] == "sequential"]

        for mix_id, group in df.groupby("mix_id"):
            # Sort by time if available, else preserve CSV order
            if group["starting_time"].isna().all():
                group = group.reset_index(drop=True)
            else:
                group = group.sort_values("starting_time").reset_index(drop=True)

            # Use track_id column if present, else compute from artist/track
            if "track_id" in group.columns:
                entries = [
                    (
                        row["track_id"],
                        float(row["starting_time"])
                        if pd.notna(row["starting_time"])
                        else float("nan"),
                    )
                    for _, row in group.iterrows()
                    if row["track_id"] in feat_ids
                ]
            else:
                entries = []
                for _, row in group.iterrows():
                    tid = _track_id(str(row["artist_name"]), str(row["track_name"]))
                    if tid in feat_ids:
                        entries.append(
                            (
                                tid,
                                float(row["starting_time"])
                                if pd.notna(row["starting_time"])
                                else float("nan"),
                            )
                        )

            n_tracks = len(entries)
            curve_shape = curve_shapes.get(str(mix_id), "unknown")

            for i in range(n_tracks - 1):
                tid_a, t_a = entries[i]
                tid_b, t_b = entries[i + 1]

                # Handle NaN timestamps — use default gap
                if np.isnan(t_a) or np.isnan(t_b):
                    gap = TIME_GAP_MAX_MIN / 2
                else:
                    gap = _time_gap(t_a, t_b)
                    if gap > TIME_GAP_MAX_MIN:
                        log.debug(
                            "Skipping pair %s→%s in mix %s: gap %.1f min > %.0f min",
                            tid_a,
                            tid_b,
                            mix_id,
                            gap,
                            TIME_GAP_MAX_MIN,
                        )
                        continue

                feats = _pair_features(feat_idx.loc[tid_a], feat_idx.loc[tid_b], gap)
                label, confidence = _classify(feats)

                records.append(
                    {
                        "mix_id": mix_id,
                        "from_track_id": tid_a,
                        "to_track_id": tid_b,
                        "from_position": i,
                        "to_position": i + 1,
                        "n_tracks_in_mix": n_tracks,
                        "mix_energy_curve_shape": curve_shape,
                        "label": label,
                        "confidence": confidence,
                        "bpm_ratio": round(feats["bpm_ratio"], 4),
                        "energy_delta": round(feats["energy_delta"], 4),
                        "harm_dist": feats["harm_dist"],
                        "time_gap_norm": round(feats["time_gap_norm"], 4),
                    }
                )

    result = pd.DataFrame(records)
    if not result.empty:
        log.info(
            "Labelled %d transitions — class distribution:\n%s",
            len(result),
            result["label"].value_counts().to_string(),
        )
    return result


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not FEATURES_PATH.exists():
        log.error("features.parquet not found at %s — run Phase 2 first.", FEATURES_PATH)
        return

    if not TRACKLIST_PATH.exists():
        log.error("tracklist.csv not found at %s — run scraper first.", TRACKLIST_PATH)
        return

    features = pd.read_parquet(FEATURES_PATH)
    log.info("Loading features: %d tracks from %s", len(features), FEATURES_PATH)
    log.info("Processing %s", TRACKLIST_PATH)

    labels = label_transitions(features, [TRACKLIST_PATH], MIX_METADATA_PATH)

    if labels.empty:
        log.warning("No transition pairs produced — check features.parquet coverage.")
        return

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABELS_PATH, index=False)
    log.info("Wrote %d rows → %s", len(labels), LABELS_PATH)


if __name__ == "__main__":
    main()
