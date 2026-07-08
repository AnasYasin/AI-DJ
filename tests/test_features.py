"""Tests for feature extraction: DiscogsEmbedder and LibrosaExtractor."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    VALID_KEYS,
    DiscogsEmbedder,
    LibrosaExtractor,
    build_features,
)

_FAKE_LIBROSA_FEATS = {
    "bpm": 128.0,
    "key": "Am",
    "loudness_lufs": -14.0,
    "energy_mean": 0.1,
    "energy_std": 0.02,
    "spectral_centroid": 3000.0,
    "onset_strength": 0.5,
    **{f"mfcc_{i}": float(i) for i in range(13)},
}

_FAKE_EMBEDDING = np.ones(1280, dtype=np.float32)


# ── LibrosaExtractor ───────────────────────────────────────────────────────────
# Uses tmp_audio_file fixture from conftest.py (5s sine wave WAV).
# No network, no model download.


def test_librosa_returns_all_expected_keys(tmp_audio_file, monkeypatch):
    monkeypatch.setattr(
        "deeprhythm.DeepRhythmPredictor.predict_from_audio", lambda self, a, sr: 128.0
    )
    ext = LibrosaExtractor()
    result = ext.extract(tmp_audio_file)
    assert result is not None
    for k in [
        "bpm",
        "key",
        "loudness_lufs",
        "energy_mean",
        "energy_std",
        "spectral_centroid",
        "onset_strength",
    ]:
        assert k in result, f"Missing key: {k}"
    for i in range(13):
        assert f"mfcc_{i}" in result, f"Missing mfcc_{i}"


def test_librosa_bpm_is_positive(tmp_audio_file, monkeypatch):
    monkeypatch.setattr(
        "deeprhythm.DeepRhythmPredictor.predict_from_audio", lambda self, a, sr: 128.0
    )
    assert LibrosaExtractor().extract(tmp_audio_file)["bpm"] > 0


def test_librosa_key_is_valid_note(tmp_audio_file, monkeypatch):
    monkeypatch.setattr(
        "deeprhythm.DeepRhythmPredictor.predict_from_audio", lambda self, a, sr: 128.0
    )
    result = LibrosaExtractor().extract(tmp_audio_file)
    assert result["key"] in VALID_KEYS, f"Unknown key: {result['key']}"


def test_librosa_energy_mean_is_positive(tmp_audio_file, monkeypatch):
    monkeypatch.setattr(
        "deeprhythm.DeepRhythmPredictor.predict_from_audio", lambda self, a, sr: 128.0
    )
    assert LibrosaExtractor().extract(tmp_audio_file)["energy_mean"] > 0


def test_librosa_returns_none_for_missing_file():
    assert LibrosaExtractor().extract("/nonexistent/path/track.mp3") is None


# ── DiscogsEmbedder ────────────────────────────────────────────────────────────
# Requires discogs-effnet model file (~100MB) — marked slow.
# Skip with: pytest -m "not slow"
# Run explicitly: pytest tests/test_features.py -v -m slow


@pytest.mark.dl_model
def test_discogs_embedding_shape(tmp_audio_file):
    """discogs-effnet must return exactly 1280 dimensions."""
    result = DiscogsEmbedder().embed(tmp_audio_file)
    assert result is not None
    assert result.shape == (1280,), f"Expected (1280,), got {result.shape}"


@pytest.mark.dl_model
def test_discogs_embedding_is_finite(tmp_audio_file):
    """Embedding must not contain NaN or Inf — corrupts downstream training."""
    result = DiscogsEmbedder().embed(tmp_audio_file)
    assert np.all(np.isfinite(result)), "Embedding contains NaN or Inf"


@pytest.mark.dl_model
def test_discogs_same_audio_gives_same_embedding(tmp_audio_file):
    """discogs-effnet is deterministic — same audio must produce identical embeddings."""
    embedder = DiscogsEmbedder()
    np.testing.assert_array_equal(embedder.embed(tmp_audio_file), embedder.embed(tmp_audio_file))


@pytest.mark.dl_model
def test_discogs_returns_none_for_missing_file():
    assert DiscogsEmbedder().embed("/nonexistent/path/track.mp3") is None


# ── build_features --mode both (default) ──────────────────────────────────────


def test_build_features_creates_parquet(tmp_path, monkeypatch):
    """build_features() writes features.parquet with embedding + audio feature columns."""
    manifest_path = _write_manifest(tmp_path, ["abc123"])
    features_path = tmp_path / "features.parquet"
    _patch_both(monkeypatch, features_path)

    result = build_features(str(manifest_path))

    assert features_path.exists()
    assert len(result) == 1
    assert "embedding" in result.columns
    assert "bpm" in result.columns
    assert result.iloc[0]["bpm"] == 128.0
    assert result.iloc[0]["key"] == "Am"


def test_build_features_skips_not_found_tracks(tmp_path, monkeypatch):
    """Tracks with source='not_found' must be ignored."""
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "track_id": ["abc123", "def456"],
            "artist": ["Artist A", "Artist B"],
            "track_name": ["Track A", "Track B"],
            "source": ["itunes", "not_found"],
            "local_path": ["/fake/path.mp3", None],
        }
    ).to_csv(manifest_path, index=False)
    features_path = tmp_path / "features.parquet"
    _patch_both(monkeypatch, features_path)

    result = build_features(str(manifest_path))

    assert len(result) == 1
    assert result.iloc[0]["track_id"] == "abc123"


def test_build_features_is_idempotent(tmp_path, monkeypatch):
    """Running build_features twice must not duplicate rows."""
    manifest_path = _write_manifest(tmp_path, ["abc123"])
    features_path = tmp_path / "features.parquet"
    _patch_both(monkeypatch, features_path)

    build_features(str(manifest_path))
    result = build_features(str(manifest_path))

    assert len(result) == 1, f"Expected 1 row, got {len(result)} — idempotency broken"


# discogs-only tests require GPU + TensorFlow — run on EC2 instance only


# ── build_features --mode librosa-only ────────────────────────────────────────


def test_librosa_only_does_not_require_embeddings_parquet(tmp_path, monkeypatch):
    """librosa-only must run without embeddings.parquet — core of the split-machine workflow."""
    manifest_path = _write_manifest(tmp_path, ["abc123"])
    librosa_path = tmp_path / "librosa_features.parquet"
    _patch_librosa_only(monkeypatch, librosa_path)
    # Point EMBEDDINGS_PATH at a nonexistent file to prove it is never read
    monkeypatch.setattr(
        "src.features.build_features.EMBEDDINGS_PATH", tmp_path / "nonexistent.parquet"
    )

    result = build_features(str(manifest_path), mode="librosa-only")

    assert librosa_path.exists()
    assert len(result) == 1
    assert "bpm" in result.columns
    assert "embedding" not in result.columns


def test_librosa_only_is_resumable(tmp_path, monkeypatch):
    """Re-running librosa-only must skip already-processed track_ids."""
    manifest_path = _write_manifest(tmp_path, ["abc123"])
    librosa_path = tmp_path / "librosa_features.parquet"
    _patch_librosa_only(monkeypatch, librosa_path)

    build_features(str(manifest_path), mode="librosa-only")
    result = build_features(str(manifest_path), mode="librosa-only")

    assert len(result) == 1, f"Expected 1 row, got {len(result)} — resumability broken"


# ── preprocess_features (merge step, moved out of build_features) ─────────────


def test_merge_joins_on_track_id(tmp_path, monkeypatch):
    """merge must inner-join embeddings + librosa features on track_id → features.parquet."""
    from src.data import preprocess_features as pf

    embeddings_path = tmp_path / "embeddings.parquet"
    librosa_path = tmp_path / "librosa_features.parquet"
    features_path = tmp_path / "features.parquet"

    pd.DataFrame(
        {"track_id": ["t1", "t2"], "embedding": [_FAKE_EMBEDDING.tolist()] * 2}
    ).to_parquet(embeddings_path, index=False)
    pd.DataFrame(
        {"track_id": ["t1", "t2"], **{k: [v, v] for k, v in _FAKE_LIBROSA_FEATS.items()}}
    ).to_parquet(librosa_path, index=False)

    monkeypatch.setattr(pf, "EMBEDDINGS_PATH", embeddings_path)
    monkeypatch.setattr(pf, "LIBROSA_FEATURES_PATH", librosa_path)
    monkeypatch.setattr(pf, "FEATURES_PATH", features_path)

    result = pf.run()

    assert features_path.exists()
    assert len(result) == 2
    assert "embedding" in result.columns
    assert "bpm" in result.columns
    assert set(result["track_id"]) == {"t1", "t2"}


def test_merge_inner_joins_partial_librosa(tmp_path, monkeypatch):
    """merge must drop tracks missing from librosa_features (inner join)."""
    from src.data import preprocess_features as pf

    embeddings_path = tmp_path / "embeddings.parquet"
    librosa_path = tmp_path / "librosa_features.parquet"
    features_path = tmp_path / "features.parquet"

    pd.DataFrame(
        {"track_id": ["t1", "t2"], "embedding": [_FAKE_EMBEDDING.tolist()] * 2}
    ).to_parquet(embeddings_path, index=False)
    pd.DataFrame(
        {"track_id": ["t1"], **{k: [v] for k, v in _FAKE_LIBROSA_FEATS.items()}}
    ).to_parquet(librosa_path, index=False)

    monkeypatch.setattr(pf, "EMBEDDINGS_PATH", embeddings_path)
    monkeypatch.setattr(pf, "LIBROSA_FEATURES_PATH", librosa_path)
    monkeypatch.setattr(pf, "FEATURES_PATH", features_path)

    result = pf.run()

    assert len(result) == 1
    assert result.iloc[0]["track_id"] == "t1"


def test_merge_raises_if_embeddings_missing(tmp_path, monkeypatch):
    from src.data import preprocess_features as pf

    monkeypatch.setattr(pf, "EMBEDDINGS_PATH", tmp_path / "missing_emb.parquet")
    monkeypatch.setattr(pf, "LIBROSA_FEATURES_PATH", tmp_path / "missing_lib.parquet")
    with pytest.raises(FileNotFoundError):
        pf.run()


def test_merge_raises_if_librosa_features_missing(tmp_path, monkeypatch):
    from src.data import preprocess_features as pf

    embeddings_path = tmp_path / "embeddings.parquet"
    pd.DataFrame({"track_id": ["t1"], "embedding": [_FAKE_EMBEDDING.tolist()]}).to_parquet(
        embeddings_path, index=False
    )
    monkeypatch.setattr(pf, "EMBEDDINGS_PATH", embeddings_path)
    monkeypatch.setattr(pf, "LIBROSA_FEATURES_PATH", tmp_path / "missing.parquet")
    with pytest.raises(FileNotFoundError):
        pf.run()


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _write_manifest(tmp_path, track_ids: list[str]):
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "track_id": track_ids,
            "artist": ["Artist"] * len(track_ids),
            "track_name": ["Track"] * len(track_ids),
            "source": ["itunes"] * len(track_ids),
            "local_path": ["/fake/path.mp3"] * len(track_ids),
        }
    ).to_csv(path, index=False)
    return path


def _patch_both(monkeypatch, features_path):
    """Patch all extractors + FEATURES_PATH for --mode both tests."""
    fake_muta = MagicMock()
    fake_muta.info.length = 5.0
    monkeypatch.setattr("src.features.build_features.MutaFile", lambda p: fake_muta)
    monkeypatch.setattr(
        "src.features.build_features.DiscogsEmbedder.__init__",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.features.build_features.DiscogsEmbedder.embed",
        lambda self, p: _FAKE_EMBEDDING.copy(),
    )
    monkeypatch.setattr(
        "src.features.build_features.LibrosaExtractor.extract",
        lambda self, p: _FAKE_LIBROSA_FEATS.copy(),
    )
    monkeypatch.setattr("src.features.build_features.FEATURES_PATH", features_path)


def _patch_librosa_only(monkeypatch, librosa_path):
    """Patch librosa extractor + LIBROSA_FEATURES_PATH for --mode librosa-only tests."""
    monkeypatch.setattr(
        "src.features.build_features.LibrosaExtractor.extract",
        lambda self, p: _FAKE_LIBROSA_FEATS.copy(),
    )
    monkeypatch.setattr("src.features.build_features.LIBROSA_FEATURES_PATH", librosa_path)
