"""Tests for the fetch gates: duration, title form, and recording identity."""

import numpy as np
import pytest

from src.data.preprocess_tracklist import is_unidentified
from src.data.track_fetcher import (
    MAX_SECONDS,
    MIN_SECONDS,
    _rank,
    _title_overlap,
    verify_match,
)

# ── Issue 3: short YouTube edits ───────────────────────────────────────────────


def _cand(title, duration, vid="x"):
    return {"id": vid, "title": title, "duration": duration}


def test_radio_edits_and_snippets_are_rejected():
    for bad in (
        "Artist - Track (Radio Edit)",
        "Artist - Track snippet",
        "Artist - Track [sped up]",
        "Artist Track - teaser",
    ):
        assert _rank("Artist Track", [_cand(bad, 300)]) == []


def test_tracks_shorter_than_four_minutes_are_rejected():
    """The old 120 s floor let one-minute edits into the plan."""
    assert _rank("Artist Track", [_cand("Artist - Track", 180)]) == []
    assert _rank("Artist Track", [_cand("Artist - Track", MIN_SECONDS - 1)]) == []
    assert len(_rank("Artist Track", [_cand("Artist - Track", MIN_SECONDS)])) == 1


def test_dj_sets_and_album_rips_are_rejected():
    assert _rank("Artist Track", [_cand("Artist - Track", MAX_SECONDS + 1)]) == []
    assert _rank("Artist Track", [_cand("Artist Track live set 2024", 600)]) == []


def test_missing_duration_is_rejected():
    assert _rank("Artist Track", [{"id": "x", "title": "Artist - Track", "duration": None}]) == []


def test_extended_mixes_survive_and_rank_by_title_match():
    ranked = _rank(
        "Artist Track",
        [
            _cand("Completely Different Song", 300, "a"),
            _cand("Artist - Track (Extended Mix)", 420, "b"),
        ],
    )
    assert [c["id"] for c in ranked] == ["b", "a"]


def test_title_overlap_ignores_bracketed_extras():
    assert _title_overlap("Artist Track", "Artist - Track (Original Mix)") == 1.0


# ── Issue 4: unidentified tracks ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "artist,title",
    [
        ("Charlotte de Witte", "ID"),
        ("Massano", "ID (Working Title)"),
        ("Some DJ", "ID (Sara Landry Remix)"),
        ("Some DJ", "ID 02 (Working Title)"),
        ("Unknown", "Dark"),
        ("Some DJ", "Untitled"),
        ("ID", "Real Title"),
    ],
)
def test_placeholder_names_are_unidentified(artist, title):
    assert is_unidentified(artist, title)


@pytest.mark.parametrize(
    "artist,title",
    [
        ("Monolink", "Return To Oz (ARTBAT Remix)"),
        ("Adam Port", "Your Love"),
        ("Some DJ", "Identity"),
        ("Some DJ", "Secret ID"),
    ],
)
def test_real_names_survive(artist, title):
    assert not is_unidentified(artist, title)


# ── Identity check ─────────────────────────────────────────────────────────────


def test_fingerprint_matches_an_excerpt_of_the_same_recording(tmp_path):
    """A 30 s excerpt must be located inside its own track, at the right offset."""
    import soundfile as sf

    sr = 22_050
    rng = np.random.default_rng(0)
    full = rng.standard_normal(sr * 120).astype(np.float32) * 0.2
    offset_s = 40
    excerpt = full[offset_s * sr : (offset_s + 30) * sr]

    fp, ep = tmp_path / "full.wav", tmp_path / "excerpt.wav"
    sf.write(fp, full, sr)
    sf.write(ep, excerpt, sr)

    votes, offset = verify_match(ep, fp)
    assert votes > 50
    assert offset == pytest.approx(offset_s, abs=0.5)


def test_fingerprint_rejects_a_different_recording(tmp_path):
    import soundfile as sf

    sr = 22_050
    rng = np.random.default_rng(0)
    fp, ep = tmp_path / "full.wav", tmp_path / "other.wav"
    sf.write(fp, rng.standard_normal(sr * 120).astype(np.float32) * 0.2, sr)
    sf.write(ep, rng.standard_normal(sr * 30).astype(np.float32) * 0.2, sr)

    votes, _ = verify_match(ep, fp)
    assert votes < 50
