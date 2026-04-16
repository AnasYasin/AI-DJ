"""
Tests for src/features/transition_labeler.py

Pure logic tests — no audio files, no model loading, no disk I/O.
Synthetic features DataFrames drive all assertions.

Coverage
────────
  _harmonic_dist      Camelot wheel distance helper
  _time_gap           60-minute rollover arithmetic
  _pair_features      delta computation
  _classify           each of the 6 transition classes
  _confidence         output range and label consistency
  label_transitions   end-to-end with a CSV fixture (uses conftest sample_tracklist)
"""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.features.transition_labeler import (
    ENERGY_FALL_MIN,
    ENERGY_MELT_MAX,
    ENERGY_RISE_MIN,
    ENERGY_SLAM_MIN,
    HARM_CLASH,
    HARM_COMPATIBLE,
    HARM_PERFECT,
    LOUD_MELT_MAX,
    ONSET_HIGH_MIN,
    TIME_GAP_MAX_MIN,
    _classify,
    _confidence,
    _harmonic_dist,
    _pair_features,
    _time_gap,
    label_transitions,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _track_id(artist: str, track: str) -> str:
    return hashlib.md5(f"{artist}|{track}".lower().encode()).hexdigest()[:12]


def _make_row(
    bpm: float = 120.0,
    key: str = "Am",
    loudness_lufs: float = -8.0,
    energy_mean: float = 0.30,
    onset_strength: float = 0.40,
) -> pd.Series:
    """Build a minimal feature row for one track."""
    return pd.Series({
        "bpm":            bpm,
        "key":            key,
        "loudness_lufs":  loudness_lufs,
        "energy_mean":    energy_mean,
        "onset_strength": onset_strength,
    })


def _feats(
    bpm_ratio: float = 1.0,
    energy_delta: float = 0.0,
    harm_dist: float = 0.0,
    loudness_delta: float = 0.0,
    onset_a: float = 0.40,
    onset_b: float = 0.40,
    time_gap_norm: float = 0.1,
) -> dict:
    """Build a pre-computed features dict for direct classify/confidence tests."""
    return {
        "bpm_ratio":      bpm_ratio,
        "energy_delta":   energy_delta,
        "harm_dist":      harm_dist,
        "loudness_delta": loudness_delta,
        "onset_a":        onset_a,
        "onset_b":        onset_b,
        "time_gap_norm":  time_gap_norm,
    }


def _make_features_df(track_specs: list[dict]) -> pd.DataFrame:
    """
    Build a features DataFrame from a list of track dicts.
    Each dict must have: track_id, bpm, key, loudness_lufs, energy_mean, onset_strength.
    """
    return pd.DataFrame(track_specs)


# ── _harmonic_dist ─────────────────────────────────────────────────────────────

def test_harmonic_dist_same_key():
    assert _harmonic_dist("Am", "Am") == 0.0


def test_harmonic_dist_adjacent():
    # Am=8, Em=9 → distance 1
    assert _harmonic_dist("Am", "Em") == 1.0


def test_harmonic_dist_wraps_around():
    # Distance wraps: max is 6
    assert _harmonic_dist("C", "F#") <= 6.0


def test_harmonic_dist_unknown_key_defaults_to_worst():
    assert _harmonic_dist("X", "Am") == 6.0
    assert _harmonic_dist("Am", "??") == 6.0


def test_harmonic_dist_max_is_six():
    # Exhaustive check: no pair should exceed 6
    keys = ["C", "Cm", "C#", "C#m", "D", "Dm", "G", "Gm", "Am", "A", "Bm", "B"]
    for ka in keys:
        for kb in keys:
            assert _harmonic_dist(ka, kb) <= 6.0


# ── _time_gap ──────────────────────────────────────────────────────────────────

def test_time_gap_normal():
    assert _time_gap(10.0, 20.0) == pytest.approx(10.0)


def test_time_gap_rollover():
    # t_b < t_a: clock rolled over the hour
    assert _time_gap(55.0, 5.0) == pytest.approx(10.0)


def test_time_gap_same_time():
    assert _time_gap(30.0, 30.0) == pytest.approx(0.0)


def test_time_gap_full_hour_rollover():
    assert _time_gap(59.0, 1.0) == pytest.approx(2.0)


# ── _pair_features ─────────────────────────────────────────────────────────────

def test_pair_features_bpm_ratio():
    ra = _make_row(bpm=120.0)
    rb = _make_row(bpm=126.0)
    f = _pair_features(ra, rb, time_gap=5.0)
    assert f["bpm_ratio"] == pytest.approx(126.0 / 120.0, rel=1e-4)


def test_pair_features_energy_delta():
    ra = _make_row(energy_mean=0.20)
    rb = _make_row(energy_mean=0.35)
    f = _pair_features(ra, rb, time_gap=5.0)
    assert f["energy_delta"] == pytest.approx(0.15, abs=1e-4)


def test_pair_features_harm_dist():
    ra = _make_row(key="Am")
    rb = _make_row(key="Am")
    f = _pair_features(ra, rb, time_gap=5.0)
    assert f["harm_dist"] == 0.0


def test_pair_features_time_gap_norm_clipped():
    ra = _make_row()
    rb = _make_row()
    # Gap beyond max should be clipped to 1.0
    f = _pair_features(ra, rb, time_gap=TIME_GAP_MAX_MIN * 2)
    assert f["time_gap_norm"] == pytest.approx(1.0)


def test_pair_features_all_keys_present():
    ra = _make_row()
    rb = _make_row()
    f = _pair_features(ra, rb, time_gap=3.0)
    for key in ("bpm_ratio", "energy_delta", "loudness_delta", "harm_dist",
                "onset_a", "onset_b", "time_gap_norm"):
        assert key in f


# ── _classify — each transition class ─────────────────────────────────────────

def test_classify_slam_energy_spike():
    # Tight BPM + large energy spike → slam
    f = _feats(bpm_ratio=1.01, energy_delta=ENERGY_SLAM_MIN + 0.05, harm_dist=0.0)
    label, _ = _classify(f)
    assert label == "slam"


def test_classify_slam_key_clash():
    # Key clash + moderate energy rise → slam even without BPM match
    f = _feats(bpm_ratio=1.10, energy_delta=ENERGY_RISE_MIN + 0.02, harm_dist=float(HARM_CLASH))
    label, _ = _classify(f)
    assert label == "slam"


def test_classify_rise():
    # Clear positive energy delta, compatible key, loose BPM → rise
    f = _feats(bpm_ratio=1.03, energy_delta=ENERGY_RISE_MIN + 0.05, harm_dist=float(HARM_COMPATIBLE))
    label, _ = _classify(f)
    assert label == "rise"


def test_classify_fade():
    # Clear negative energy delta → fade
    f = _feats(bpm_ratio=1.05, energy_delta=ENERGY_FALL_MIN - 0.05, harm_dist=2.0)
    label, _ = _classify(f)
    assert label == "fade"


def test_classify_melt():
    # Tight BPM, same key, tiny energy delta, small loudness delta → melt
    f = _feats(
        bpm_ratio=1.005,
        energy_delta=0.01,
        harm_dist=0.0,
        loudness_delta=1.0,
    )
    label, _ = _classify(f)
    assert label == "melt"


def test_classify_wave():
    # Tight BPM, high onset on both, harm_dist=2 (fails melt check) → wave
    # harm_dist must be > HARM_PERFECT so melt is not triggered first.
    f = _feats(
        bpm_ratio=1.01,
        energy_delta=0.02,
        harm_dist=float(HARM_PERFECT + 1),
        onset_a=ONSET_HIGH_MIN + 0.10,
        onset_b=ONSET_HIGH_MIN + 0.10,
    )
    label, _ = _classify(f)
    assert label == "wave"


def test_classify_blend_default():
    # Nothing stands out → blend
    f = _feats(bpm_ratio=1.05, energy_delta=0.03, harm_dist=3.0, onset_a=0.20, onset_b=0.20)
    label, _ = _classify(f)
    assert label == "blend"


def test_classify_slam_takes_priority_over_rise():
    # Both slam and rise conditions met → slam wins
    f = _feats(
        bpm_ratio=1.01,
        energy_delta=ENERGY_SLAM_MIN + 0.05,   # triggers both slam and rise
        harm_dist=0.0,
    )
    label, _ = _classify(f)
    assert label == "slam"


def test_classify_rise_not_triggered_without_bpm_range():
    # Energy rise but BPM too far apart → not rise
    f = _feats(bpm_ratio=1.15, energy_delta=ENERGY_RISE_MIN + 0.05, harm_dist=1.0)
    label, _ = _classify(f)
    assert label != "rise"


def test_classify_melt_blocked_by_energy_delta():
    # BPM tight, key compatible but energy delta too large → not melt
    f = _feats(bpm_ratio=1.005, energy_delta=ENERGY_MELT_MAX + 0.05, harm_dist=0.0)
    label, _ = _classify(f)
    assert label != "melt"


def test_classify_melt_blocked_by_loudness():
    # BPM tight, key same, energy tiny, but loudness mismatch → not melt
    f = _feats(bpm_ratio=1.005, energy_delta=0.01, harm_dist=0.0, loudness_delta=LOUD_MELT_MAX + 1.0)
    label, _ = _classify(f)
    assert label != "melt"


def test_classify_wave_blocked_by_low_onset():
    # Low onset on one track → not wave
    f = _feats(bpm_ratio=1.01, energy_delta=0.02, onset_a=ONSET_HIGH_MIN - 0.10, onset_b=0.60)
    label, _ = _classify(f)
    assert label != "wave"


# ── _confidence ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("label", ["slam", "rise", "fade", "melt", "wave", "blend"])
def test_confidence_in_valid_range(label):
    f = _feats()
    score = _confidence(label, f)
    assert 0.0 < score <= 1.0


def test_confidence_slam_increases_with_energy_spike():
    base = _confidence("slam", _feats(energy_delta=ENERGY_SLAM_MIN))
    high = _confidence("slam", _feats(energy_delta=0.25))
    assert high > base


def test_confidence_rise_increases_with_larger_delta():
    low  = _confidence("rise", _feats(energy_delta=ENERGY_RISE_MIN + 0.01))
    high = _confidence("rise", _feats(energy_delta=ENERGY_RISE_MIN + 0.10))
    assert high > low


def test_confidence_melt_highest_when_same_key_tight_bpm():
    loose = _confidence("melt", _feats(bpm_ratio=1.02, harm_dist=1.0))
    tight = _confidence("melt", _feats(bpm_ratio=1.001, harm_dist=0.0))
    assert tight > loose


def test_confidence_blend_is_lowest_base():
    # Each label with features that strongly match its own rule should outscore blend.
    blend_score = _confidence("blend", _feats())
    strong = {
        "slam": _feats(bpm_ratio=1.01, energy_delta=0.25, harm_dist=0.0),
        "rise": _feats(bpm_ratio=1.02, energy_delta=ENERGY_RISE_MIN + 0.10, harm_dist=1.0),
        "fade": _feats(bpm_ratio=1.04, energy_delta=ENERGY_FALL_MIN - 0.10, harm_dist=1.0),
        "melt": _feats(bpm_ratio=1.001, energy_delta=0.01, harm_dist=0.0, loudness_delta=0.5),
        "wave": _feats(bpm_ratio=1.01, energy_delta=0.02, onset_a=0.60, onset_b=0.60),
    }
    for label, feats in strong.items():
        assert _confidence(label, feats) > blend_score, f"{label} strong score should beat blend"


# ── label_transitions (end-to-end) ────────────────────────────────────────────

def _make_features_for_csv(csv_path: str) -> pd.DataFrame:
    """
    Build a features DataFrame whose track_ids match the conftest sample_tracklist fixture.
    Tracks: Artist 1/Track A, Artist 2/Track B, Artist 3/Track C, Artist 4/Track D.
    """
    specs = [
        {"artist_name": "Artist 1", "track_name": "Track A"},
        {"artist_name": "Artist 2", "track_name": "Track B"},
        {"artist_name": "Artist 3", "track_name": "Track C"},
        {"artist_name": "Artist 4", "track_name": "Track D"},
    ]
    rows = []
    for s in specs:
        rows.append({
            "track_id":       _track_id(s["artist_name"], s["track_name"]),
            "bpm":            120.0,
            "key":            "Am",
            "loudness_lufs":  -8.0,
            "energy_mean":    0.30,
            "energy_std":     0.05,
            "onset_strength": 0.40,
        })
    return pd.DataFrame(rows)


def test_label_transitions_returns_dataframe(sample_tracklist):
    features = _make_features_for_csv(sample_tracklist)
    result = label_transitions(features, [Path(sample_tracklist)])
    assert isinstance(result, pd.DataFrame)


def test_label_transitions_expected_columns(sample_tracklist):
    features = _make_features_for_csv(sample_tracklist)
    result = label_transitions(features, [Path(sample_tracklist)])
    for col in ("from_track_id", "to_track_id", "label", "confidence",
                "bpm_ratio", "energy_delta", "harm_dist", "time_gap_norm"):
        assert col in result.columns, f"Missing column: {col}"


def test_label_transitions_labels_are_valid(sample_tracklist):
    features = _make_features_for_csv(sample_tracklist)
    result = label_transitions(features, [Path(sample_tracklist)])
    valid = {"slam", "melt", "blend", "rise", "fade", "wave"}
    assert set(result["label"]).issubset(valid)


def test_label_transitions_confidence_in_range(sample_tracklist):
    features = _make_features_for_csv(sample_tracklist)
    result = label_transitions(features, [Path(sample_tracklist)])
    assert (result["confidence"] > 0).all()
    assert (result["confidence"] <= 1.0).all()


def test_label_transitions_skips_missing_tracks(sample_tracklist):
    # Features for only 2 of the 4 tracks → pairs involving missing tracks skipped
    features = _make_features_for_csv(sample_tracklist).head(2)
    result = label_transitions(features, [Path(sample_tracklist)])
    # Should not crash and should produce ≤ 2 rows
    assert len(result) <= 2


def test_label_transitions_skips_large_time_gap(tmp_path):
    # Create a CSV where the gap between tracks exceeds TIME_GAP_MAX_MIN
    data = {
        "mix_id": [1, 1],
        "url": ["http://x.com"] * 2,
        "starting_time": [0.0, TIME_GAP_MAX_MIN + 5.0],   # gap > threshold
        "track_name": ["Track A", "Track B"],
        "artist_name": ["Artist 1", "Artist 2"],
    }
    csv_path = tmp_path / "gap_test.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    features = pd.DataFrame([
        {"track_id": _track_id("Artist 1", "Track A"), "bpm": 120.0, "key": "Am",
         "loudness_lufs": -8.0, "energy_mean": 0.30, "energy_std": 0.05, "onset_strength": 0.40},
        {"track_id": _track_id("Artist 2", "Track B"), "bpm": 122.0, "key": "Em",
         "loudness_lufs": -7.0, "energy_mean": 0.32, "energy_std": 0.05, "onset_strength": 0.42},
    ])
    result = label_transitions(features, [csv_path])
    assert len(result) == 0


def test_label_transitions_empty_when_no_features_match(sample_tracklist):
    # Empty features → no pairs
    features = pd.DataFrame(columns=["track_id", "bpm", "key", "loudness_lufs",
                                      "energy_mean", "energy_std", "onset_strength"])
    result = label_transitions(features, [Path(sample_tracklist)])
    assert result.empty


def test_label_transitions_multiple_csvs(sample_tracklist, tmp_path):
    # Two CSVs → results combined
    features = _make_features_for_csv(sample_tracklist)
    result_single = label_transitions(features, [Path(sample_tracklist)])
    result_double = label_transitions(features, [Path(sample_tracklist), Path(sample_tracklist)])
    assert len(result_double) == 2 * len(result_single)
