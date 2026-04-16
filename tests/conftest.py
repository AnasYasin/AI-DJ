"""
Shared pytest fixtures used across all test files.
conftest.py is automatically loaded by pytest — no import needed in test files.
"""

import numpy as np
import pytest


@pytest.fixture
def tmp_audio_file(tmp_path):
    """
    Creates a 5-second WAV with a 120 BPM kick-drum pulse + 440Hz tone.

    Why not a pure sine wave: librosa.beat_track detects rhythm from amplitude
    changes. A flat sine wave has no amplitude variation so beat_track returns
    0.0 BPM — correct behaviour but useless for testing the extractor.

    This fixture adds a sharp amplitude spike every 0.5 seconds (120 BPM),
    which gives librosa enough rhythmic structure to detect a tempo > 0.
    """
    import soundfile as sf

    sr = 22050
    duration = 5
    t = np.linspace(0, duration, sr * duration)

    # Base tone: 440Hz sine wave
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    # Add a sharp kick pulse every 0.5 seconds (= 120 BPM)
    # Each pulse is a 10ms burst of high amplitude — simulates a kick drum hit
    beat_interval = int(sr * 0.5)  # samples between beats
    pulse_len = int(sr * 0.01)  # 10ms pulse width
    for onset in range(0, len(audio) - pulse_len, beat_interval):
        audio[onset : onset + pulse_len] += 0.7

    audio = np.clip(audio, -1.0, 1.0)  # prevent clipping above ±1
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
