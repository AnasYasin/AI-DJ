"""The label rules as written, and the tempo of a synthetic record with a known kick."""

import numpy as np
import pytest
import soundfile as sf

from djdata.seam import labels, tempo
from tests import synth

BAR = 4 * 60.0 / 128.0  # 1.875 s at 128 BPM


def measures(**over):
    base = {
        "in_s": 40.0,
        "out_s": 100.0,
        "bass_swap_s": 70.0,
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


def test_moves_in_bars_and_as_fractions_of_the_overlap():
    got = labels.label(
        "s", measures(in_s=40.0, out_s=100.0, bass_swap_s=70.0, overlap_s=60.0), BAR
    )
    assert got.measured == 1 and got.overlap_bars == 32.0 and got.swap_pos == 0.5
    assert got.b_high_pos == 0.0 and got.a_low_pos == 1.0  # bands at the entry and the exit
    assert got.tension == 0 and got.loop == 0


def test_tension_and_loop_flags_and_sweeps():
    assert labels.label("s", measures(bass_cut_longest_s=8 * BAR), BAR).tension == 1
    assert labels.label("s", measures(bass_cut_longest_s=7.9 * BAR), BAR).tension == 0
    assert labels.label("s", measures(loop_steps=4), BAR).loop == 1
    out = measures(low_out_s=80.0, mid_out_s=90.0, high_out_s=80.0 + 8 * BAR)
    assert labels.label("s", out, BAR).sweep_out_bars == 8.0
    wrong_order = measures(low_out_s=100.0, mid_out_s=90.0, high_out_s=120.0)
    assert labels.label("s", wrong_order, BAR).sweep_out_bars is None
    sweep_in = measures(high_in_s=40.0, mid_in_s=45.0, low_in_s=40.0 + 9 * BAR)
    assert labels.label("s", sweep_in, BAR).sweep_in_bars == 9.0
    assert labels.label("s", measures(high_in_s=None), BAR).sweep_in_bars is None


def test_no_tempo_or_no_overlap_is_unmeasured_not_guessed():
    assert labels.label("s", measures(), None).measured == 0
    assert labels.label("s", measures(overlap_s=None), BAR).measured == 0
    assert labels.label("s", measures(overlap_s=None), BAR).swap_pos is None


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
