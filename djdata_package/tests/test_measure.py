"""One synthetic seam with a known bass swap, entry and exit, measured with every floor from controls.

The window is 150 s. A plays throughout until it leaves at 110 s. B enters at 40 s with its low band
cut; B's bass opens and A's bass is cut at 80 s (the swap). A's mids and highs stay until 110 s.
So per band: mid and high of B first seen near 40 s, low of B first seen near 80 s, low of A last seen
near 80 s, mid and high of A last seen near 110 s, bass swap near 80 s, overlap 40 to 110 s.
"""

import numpy as np
import pytest
import soundfile as sf

from djdata.seam import bands, bass, loops, measure, presence
from djdata.seam.fingerprint import SR, as_arrays, fingerprint, match
from tests import synth

A_OUT, B_IN, SWAP, LEN = 110.0, 40.0, 80.0, 150.0


@pytest.fixture(scope="module")
def seam(tmp_path_factory):
    root = tmp_path_factory.mktemp("seam")
    a = synth.record(200.0, seed=61, bass_voices=2)
    b = synth.record(200.0, seed=62, bass_voices=2)
    n = int(LEN * SR)
    a_low, a_rest = synth.split_low(a[:n])
    b_low, b_rest = synth.split_low(b[:n])
    a_gate = 1.0 - synth.ramp(n, A_OUT)
    a_bass_gate = 1.0 - synth.ramp(n, SWAP)
    b_gate = synth.ramp(n, B_IN)
    b_bass_gate = synth.ramp(n, SWAP)
    window = a_rest * a_gate + a_low * a_bass_gate + b_rest * b_gate + b_low * b_bass_gate
    window = (0.9 * window / np.abs(window).max()).astype(np.float32)
    paths = {}
    for name, audio in (("window", window), ("a", a), ("b", b)):
        paths[name] = root / f"{name}.wav"
        sf.write(paths[name], audio, SR)
    controls = []
    for s in (71, 72, 73):
        p = root / f"c{s}.wav"
        sf.write(p, synth.record(200.0, seed=s, bass_voices=2), SR)
        controls.append(p)
    # the window starts at mix 600 s; A's time zero is 600 (window = record from its start), B's is
    # 600 too (B's record time 0 falls on window start; it is simply silent until B_IN)
    job = {
        "seam_id": "m_a_b",
        "window_path": paths["window"],
        "window_t0": 600.0,
        "a": {"path": paths["a"], "time_zero_s": 600.0, "rate": 1.0},
        "b": {"path": paths["b"], "time_zero_s": 600.0, "rate": 1.0},
        "controls": controls,
    }
    row, curves = measure.measure_one(job)
    return {"row": row, "curves": curves, "a": a, "b": b, "window": window}


def test_both_records_measured_far_above_their_floors(seam):
    row = seam["row"]
    assert row["measured"] == 1 and row["control_trusted"] == 1
    assert row["a_best_separation"] > 5 and row["b_best_separation"] > 5
    for band in bands.BANDS:
        assert row[f"{band}_floor_trusted"] is True
        assert row[f"{band}_control_max"] < row[f"{band}_a_votes"]


def test_band_entries_and_exits_follow_the_gates(seam):
    row = seam["row"]
    tol = bands.STEP_S + bands.HOP_S
    assert row["mid_in_s"] == pytest.approx(B_IN, abs=tol)
    assert row["high_in_s"] == pytest.approx(B_IN, abs=tol)
    assert row["low_in_s"] == pytest.approx(SWAP, abs=tol)
    assert row["low_out_s"] == pytest.approx(SWAP, abs=tol + bands.STEP_S)
    assert row["mid_out_s"] == pytest.approx(A_OUT, abs=tol + bands.STEP_S)
    assert row["high_out_s"] == pytest.approx(A_OUT, abs=tol + bands.STEP_S)


def test_bass_swap_lands_on_the_swap(seam):
    row = seam["row"]
    assert row["bass_swap_s"] == pytest.approx(SWAP, abs=bands.STEP_S + bands.HOP_S)
    assert row["bass_a_separation"] > 2 and row["bass_b_separation"] > 2


def test_presence_entry_exit_and_overlap(seam):
    row = seam["row"]
    assert row["in_s"] == pytest.approx(B_IN, abs=presence.CHUNK_S)
    assert row["out_s"] == pytest.approx(A_OUT, abs=presence.CHUNK_S)
    assert row["overlap_s"] == pytest.approx(A_OUT - B_IN, abs=2 * presence.CHUNK_S)
    assert row["loop_steps"] == 0


def test_bass_words_say_outgoing_then_incoming(seam):
    row = seam["row"]
    assert row["bass_outgoing_s"] > 20
    assert row["bass_incoming_s"] > 20
    assert row["bass_cut_s"] < 10 and row["bass_extra_s"] < 10


def test_words_rule():
    mix_has = np.array([True, True, False, False, True, True])
    a_here = np.array([True, True, True, False, False, False])
    b_here = np.array([False, True, True, True, True, True])
    a_bass = np.ones(6, bool)
    b_bass = np.ones(6, bool)
    a_ok = np.array([True, True, False, False, False, False])
    b_ok = np.array([False, True, False, False, True, False])
    got = bass.words(mix_has, a_here, b_here, a_bass, b_bass, a_ok, b_ok)
    assert got == ["outgoing", "both", "cut", "cut", "incoming", "unsure"]
    assert bass.segments([0, 1, 2, 3, 4, 5], got)[2] == (2, 3 + bands.STEP_S, "cut")
    assert bass.seconds_per_word(got)["cut"] == 2 * bands.HOP_S
    assert bass.longest_run([0, 1, 2, 3, 4, 5], got, "cut") == 1 + bands.STEP_S


def test_loop_is_a_record_standing_still():
    t = np.arange(0, 60, presence.HOP_S)
    votes = np.full(len(t), 40)
    advancing = 100.0 + t  # matched time moves with the mix
    stuck = np.where(t < 30, 100.0 + (t % 8.0), 100.0 + t)  # first 30 s: a 8 s loop
    free = {"t": t, "A": votes, "A_t": advancing}
    looped = {"t": t, "A": votes, "A_t": stuck}
    import pandas as pd

    assert loops.loop_steps(pd.DataFrame(free), "A", floor=15, before_s=None) == 0
    steps = loops.loop_steps(pd.DataFrame(looped), "A", floor=15, before_s=None)
    assert loops.is_loop(steps) and steps >= 4
    assert (
        loops.loop_steps(pd.DataFrame(looped), "A", floor=50, before_s=None) == 0
    )  # under the floor


def test_low_band_fingerprint_separates_two_bass_lines(seam):
    a, b = seam["a"], seam["b"]
    table = as_arrays(bass.low_band_fp(a))
    right, _ = match(bass.low_band_fp(a[: 60 * SR]), table)
    wrong, _ = match(bass.low_band_fp(b[: 60 * SR]), table)
    assert right > 10 * max(wrong, 1)


def test_band_fingerprint_is_blind_outside_its_band(seam):
    a = seam["a"]
    low_only, _ = synth.split_low(a[: 60 * SR])
    high_table = as_arrays(bands.band_fp(a[: 60 * SR], "high"))
    votes, _ = match(bands.band_fp(low_only, "high"), high_table)
    full, _ = match(bands.band_fp(a[: 60 * SR], "high"), high_table)
    assert full > 10 * max(votes, 1)


def test_fingerprint_percentile_knob_only_changes_density():
    a = synth.record(20.0, seed=5)
    dense = sum(len(v) for v in fingerprint(a, 80.0).values())
    sparse = sum(len(v) for v in fingerprint(a).values())
    assert dense > sparse
