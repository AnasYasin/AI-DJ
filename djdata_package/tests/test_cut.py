"""The cut is exact and the audit says whether the window starts with A alone and ends with B alone."""

import pytest
import soundfile as sf

from djdata.seam import cut
from djdata.seam.fingerprint import SR
from tests import synth


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    a = synth.record(120.0, seed=41)
    b = synth.record(120.0, seed=42)
    c = synth.record(60.0, seed=43)
    # A from 0 for 120 s; B from 90 s at speed 1.01, cued 20 s deep: overlap 90 to 120
    b_zero = 90.0 - 20.0 / 1.01
    audio = synth.mix([(a, 0.0, 1.0), (b, b_zero, 1.01, 20.0)], seconds=200.0)
    root = tmp_path_factory.mktemp("cut")
    mix_path = root / "mix.wav"
    sf.write(mix_path, audio, SR)
    return {"root": root, "mix": mix_path, "a": a, "b": b, "c": c, "b_zero": b_zero}


def test_cut_is_exact(mixed):
    out = cut.cut_window(mixed["mix"], 60.0, 150.0, mixed["root"] / "w1")
    assert out.suffix == ".wav"
    assert cut.duration_s(out) == pytest.approx(90.0, abs=0.03)


def test_audit_passes_a_good_window(mixed):
    out = cut.cut_window(mixed["mix"], 60.0, 150.0, mixed["root"] / "good")
    got = cut.audit(
        out,
        60.0,
        {"time_zero_s": 0.0, "rate": 1.0},
        {"time_zero_s": mixed["b_zero"], "rate": 1.01},
        mixed["a"],
        mixed["b"],
        controls=[mixed["c"], synth.record(30, 44), synth.record(30, 45)],
    )
    assert got["audit_start_ok"] == 1 and got["audit_end_ok"] == 1
    assert got["audit_start_a"] > got["audit_floor"] > got["audit_start_b"]
    assert got["audit_end_b"] > got["audit_floor"] > got["audit_end_a"]


def test_audit_fails_a_window_that_starts_inside_the_overlap(mixed):
    out = cut.cut_window(mixed["mix"], 100.0, 150.0, mixed["root"] / "late")
    got = cut.audit(
        out,
        100.0,
        {"time_zero_s": 0.0, "rate": 1.0},
        {"time_zero_s": mixed["b_zero"], "rate": 1.01},
        mixed["a"],
        mixed["b"],
        controls=[mixed["c"], synth.record(30, 44), synth.record(30, 45)],
    )
    assert got["audit_start_ok"] == 0  # B is already there
    assert got["audit_end_ok"] == 1


def test_audit_fails_a_window_that_ends_before_a_leaves(mixed):
    out = cut.cut_window(mixed["mix"], 60.0, 110.0, mixed["root"] / "early")
    got = cut.audit(
        out,
        60.0,
        {"time_zero_s": 0.0, "rate": 1.0},
        {"time_zero_s": mixed["b_zero"], "rate": 1.01},
        mixed["a"],
        mixed["b"],
        controls=[mixed["c"], synth.record(30, 44), synth.record(30, 45)],
    )
    assert got["audit_start_ok"] == 1
    assert got["audit_end_ok"] == 0  # A is still there


def test_record_slice_outside_the_record_is_empty(mixed):
    assert len(cut.record_slice(mixed["a"], 1.0, from_s=500.0, seconds=10.0)) == 0
    assert len(cut.record_slice(mixed["a"], 1.0, from_s=-30.0, seconds=10.0)) == 0
    piece = cut.record_slice(mixed["a"], 1.02, from_s=30.0, seconds=10.0)
    assert abs(len(piece) / SR - (10.0 * 1.02 + 2 * cut.SLICE_PAD_S + 0) / 1.02) < 0.5
