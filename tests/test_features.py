"""Tests for feature extraction: CLAPEmbedder and LibrosaExtractor."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import CLAPEmbedder, LibrosaExtractor, build_features, _CHROMA_TO_NOTE


# ── LibrosaExtractor ───────────────────────────────────────────────────────────
# Uses tmp_audio_file fixture from conftest.py (5s sine wave WAV).
# No network, no model download.

def test_librosa_returns_all_expected_keys(tmp_audio_file):
    ext = LibrosaExtractor()
    result = ext.extract(tmp_audio_file)
    assert result is not None
    for k in ["bpm", "key", "energy_mean", "energy_std", "spectral_centroid", "onset_strength"]:
        assert k in result, f"Missing key: {k}"
    for i in range(13):
        assert f"mfcc_{i}" in result, f"Missing mfcc_{i}"


def test_librosa_bpm_is_positive(tmp_audio_file):
    assert LibrosaExtractor().extract(tmp_audio_file)["bpm"] > 0


def test_librosa_key_is_valid_note(tmp_audio_file):
    result = LibrosaExtractor().extract(tmp_audio_file)
    assert result["key"] in _CHROMA_TO_NOTE, f"Unknown key: {result['key']}"


def test_librosa_energy_mean_is_positive(tmp_audio_file):
    assert LibrosaExtractor().extract(tmp_audio_file)["energy_mean"] > 0


def test_librosa_returns_none_for_missing_file():
    assert LibrosaExtractor().extract("/nonexistent/path/track.mp3") is None


# ── CLAPEmbedder ───────────────────────────────────────────────────────────────
# CLAP downloads ~860MB on first run — marked slow, skip with: pytest -m "not slow"
# Run explicitly with: pytest tests/test_features.py -v -m slow

@pytest.mark.slow
def test_clap_embedding_shape(tmp_audio_file):
    """CLAP must return exactly 512 dimensions."""
    result = CLAPEmbedder().embed(tmp_audio_file)
    assert result is not None
    assert result.shape == (512,), f"Expected (512,), got {result.shape}"


@pytest.mark.slow
def test_clap_embedding_is_finite(tmp_audio_file):
    """Embedding must not contain NaN or Inf — corrupts downstream training."""
    result = CLAPEmbedder().embed(tmp_audio_file)
    assert np.all(np.isfinite(result)), "Embedding contains NaN or Inf"


@pytest.mark.slow
def test_clap_same_audio_gives_same_embedding(tmp_audio_file):
    """CLAP is deterministic — same audio must produce identical embeddings."""
    embedder = CLAPEmbedder()
    np.testing.assert_array_equal(embedder.embed(tmp_audio_file), embedder.embed(tmp_audio_file))


@pytest.mark.slow
def test_clap_returns_none_for_missing_file():
    assert CLAPEmbedder().embed("/nonexistent/path/track.mp3") is None


# ── build_features (mocked — no model load, no real audio) ────────────────────

def test_build_features_creates_parquet(tmp_path, monkeypatch):
    """
    build_features() writes features.parquet with correct columns.
    Extractors are monkeypatched — no model or real audio needed.
    """
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "track_id": ["abc123"], "artist": ["Test Artist"],
        "track_name": ["Test Track"], "source": ["itunes"],
        "local_path": ["/fake/path.mp3"],
    }).to_csv(manifest_path, index=False)

    features_path = tmp_path / "features.parquet"
    _patch_extractors(monkeypatch, features_path)

    result = build_features(str(manifest_path))

    assert features_path.exists(), "features.parquet was not created"
    assert len(result) == 1
    assert "clap_0" in result.columns
    assert "clap_511" in result.columns
    assert "bpm" in result.columns
    assert result.iloc[0]["bpm"] == 128.0
    assert result.iloc[0]["key"] == "Am"


def test_build_features_skips_not_found_tracks(tmp_path, monkeypatch):
    """Tracks with source='not_found' must be ignored."""
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "track_id": ["abc123", "def456"],
        "artist": ["Artist A", "Artist B"],
        "track_name": ["Track A", "Track B"],
        "source": ["itunes", "not_found"],
        "local_path": ["/fake/path.mp3", None],
    }).to_csv(manifest_path, index=False)

    features_path = tmp_path / "features.parquet"
    _patch_extractors(monkeypatch, features_path)

    result = build_features(str(manifest_path))

    assert len(result) == 1
    assert result.iloc[0]["track_id"] == "abc123"


def test_build_features_is_idempotent(tmp_path, monkeypatch):
    """Running build_features twice must not duplicate rows."""
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "track_id": ["abc123"], "artist": ["A"], "track_name": ["T"],
        "source": ["itunes"], "local_path": ["/fake/path.mp3"],
    }).to_csv(manifest_path, index=False)

    features_path = tmp_path / "features.parquet"
    _patch_extractors(monkeypatch, features_path)

    build_features(str(manifest_path))          # first run
    result = build_features(str(manifest_path)) # second run — must skip existing

    assert len(result) == 1, f"Expected 1 row, got {len(result)} — idempotency broken"


# ── Shared test helper ─────────────────────────────────────────────────────────

def _patch_extractors(monkeypatch, features_path):
    """Patch both extractors and FEATURES_PATH for unit tests."""
    fake_librosa = {
        "bpm": 128.0, "key": "Am", "energy_mean": 0.1, "energy_std": 0.02,
        "spectral_centroid": 3000.0, "onset_strength": 0.5,
        **{f"mfcc_{i}": float(i) for i in range(13)},
    }
    monkeypatch.setattr("src.features.build_features.CLAPEmbedder.embed",
                        lambda self, p: np.random.rand(512).astype(np.float32))
    monkeypatch.setattr("src.features.build_features.LibrosaExtractor.extract",
                        lambda self, p: fake_librosa)
    monkeypatch.setattr("src.features.build_features.FEATURES_PATH", features_path)
