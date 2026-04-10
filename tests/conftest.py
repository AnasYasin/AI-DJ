"""
Shared pytest fixtures used across all test files.
conftest.py is automatically loaded by pytest — no import needed in test files.
"""
import numpy as np
import pytest


@pytest.fixture
def tmp_audio_file(tmp_path):
    """
    Creates a 5-second 440Hz sine wave WAV file in a temp directory.
    Used as a stand-in for a real MP3 preview in unit tests.
    This avoids needing real audio files in the test suite.
    """
    import soundfile as sf

    sr = 22050
    duration = 5  # seconds
    t = np.linspace(0, duration, sr * duration)
    # 440Hz = concert A. Using sine wave = simplest possible audio signal.
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    path = tmp_path / "test_audio.wav"
    sf.write(str(path), audio, sr)
    return str(path)


@pytest.fixture
def sample_tracklist(tmp_path):
    """
    Creates a minimal tracklist CSV with 2 mixes and 4 tracks total.
    Used for testing transition labeler and vector store without real data.
    """
    import pandas as pd

    data = {
        "mix_id": [1, 1, 2, 2],
        "url": ["http://example.com/mix1"] * 2 + ["http://example.com/mix2"] * 2,
        "starting_time": [10.0, 20.0, 5.0, 15.0],
        "track_name": ["Track A", "Track B", "Track C", "Track D"],
        "artist_name": ["Artist 1", "Artist 2", "Artist 3", "Artist 4"],
    }
    path = tmp_path / "tracklist.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)
