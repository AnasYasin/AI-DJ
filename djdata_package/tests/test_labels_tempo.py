"""The label rules as written, and the tempo of a synthetic record with a known kick."""

import numpy as np
import pytest
import soundfile as sf

from djdata.seam import labels, tempo
from tests import synth

BAR = 4 * 60.0 / 128.0  # 1.875 s at 128 BPM


def measures(**over):
    base = {
        "overlap_s": 20 * BAR,
        "bass_cut_longest_s": 0.0,
        "bass_both_longest_s": 0.0,
        "loop_steps": 0,
        "low_out_s": 100.0,
        "mid_out_s": 100.0,
        "high_out_s": 100.0,
        "low_in_s": 40.0,
        "mid_in_s": 40.0,
        "high_in_s": 40.0,
    }
    base.update(over)
    return base


def test_long_and_short_blend_and_cut_by_overlap_bars():
    assert labels.label("s", measures(overlap_s=20 * BAR), BAR).label == "long_blend"
    assert labels.label("s", measures(overlap_s=5 * BAR), BAR).label == "short_blend"
    assert labels.label("s", measures(overlap_s=0.5 * BAR), BAR).label == "cut"
    assert labels.label("s", measures(overlap_s=-1.5 * BAR), BAR).label == "cut"
    assert labels.label("s", measures(overlap_s=-3 * BAR), BAR).label == "edit_or_talk"


def test_priority_order_loop_first_then_tension_sweeps_layer():
    got = labels.label("s", measures(loop_steps=4, bass_cut_longest_s=10 * BAR), BAR)
    assert (
        got.label == "loop"
        and "tension" in got.labels.split()
        and "long_blend" in got.labels.split()
    )
    assert labels.label("s", measures(bass_cut_longest_s=8 * BAR), BAR).label == "tension"
    assert labels.label("s", measures(bass_cut_longest_s=7.9 * BAR), BAR).label == "long_blend"
    assert labels.label("s", measures(bass_both_longest_s=16 * BAR), BAR).label == "layer"


def test_sweeps_need_the_band_order_over_eight_bars():
    out = measures(low_out_s=80.0, mid_out_s=90.0, high_out_s=80.0 + 8 * BAR)
    got = labels.label("s", out, BAR)
    assert got.label == "sweep_out" and got.sweep_out_bars == 8.0
    wrong_order = measures(low_out_s=100.0, mid_out_s=90.0, high_out_s=120.0)
    assert "sweep_out" not in labels.label("s", wrong_order, BAR).labels
    short = measures(low_out_s=80.0, mid_out_s=82.0, high_out_s=84.0)
    assert "sweep_out" not in labels.label("s", short, BAR).labels
    sweep_in = measures(high_in_s=40.0, mid_in_s=45.0, low_in_s=40.0 + 9 * BAR)
    assert labels.label("s", sweep_in, BAR).label == "sweep_in"
    missing = measures(high_in_s=None)
    assert "sweep_in" not in labels.label("s", missing, BAR).labels


def test_no_tempo_or_no_overlap_is_unmeasured_not_guessed():
    assert labels.label("s", measures(), None).label == "unmeasured"
    assert labels.label("s", measures(overlap_s=None), BAR).label == "unmeasured"


def test_bar_seconds_follows_the_playback_rate():
    assert tempo.bar_seconds(128.0, 1.0) == pytest.approx(BAR, abs=1e-3)
    assert tempo.bar_seconds(128.0, 1.02) == pytest.approx(BAR / 1.02, abs=1e-3)
    assert tempo.bar_seconds(None, 1.0) is None


def test_tempo_of_a_record_with_a_kick_every_beat(tmp_path):
    sr = tempo.mixer.SR
    bpm = 127.5
    seconds = 150.0
    n = int(seconds * sr)
    t = np.arange(n) / sr
    music = synth.record(seconds, seed=9, sr=sr)
    kicks = np.zeros(n, dtype=np.float32)
    beat = 60.0 / bpm
    for k in range(int(seconds / beat)):
        start = int(k * beat * sr)
        length = int(0.08 * sr)
        env = np.exp(-np.arange(length) / (0.02 * sr))
        kicks[start : start + length] += (np.sin(2 * np.pi * 55.0 * t[:length]) * env).astype(
            np.float32
        )
    audio = 0.5 * music[:n] + 0.9 * kicks
    path = tmp_path / "kick.wav"
    sf.write(path, audio / np.abs(audio).max(), sr)
    got = tempo.bpm_of(path)
    assert got == pytest.approx(bpm, abs=0.2)
