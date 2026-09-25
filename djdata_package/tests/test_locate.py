"""The locator on a synthetic mix with a known answer: time zero, speed, played part, and the floor."""

import numpy as np
import pytest
import soundfile as sf

from djdata.seam import fingerprint as fp
from djdata.seam import locate
from tests import synth


@pytest.fixture(scope="module")
def known_mix(tmp_path_factory):
    """B from 0.0 at speed 1.0 for its whole 100 s; A cued in 30 s deep at 3 % fast with its time zero
    at 90.0 s, so its first 30 s of record time are never heard; C is not in the mix at all."""
    a = synth.record(150.0, seed=11)
    b = synth.record(100.0, seed=12)
    c = synth.record(120.0, seed=13)
    a_zero, a_rate = 120.0 - 30.0 / 1.03, 1.03  # heard from mix 120 s, record time 30 s
    audio = synth.mix([(b, 0.0, 1.0), (a, a_zero, a_rate, 30.0)], seconds=260.0)
    path = tmp_path_factory.mktemp("mix") / "mix.wav"
    sf.write(path, audio, fp.SR)
    table, length = fp.mix_fingerprint(path)
    # the presence floor comes from three control records, as the pipeline does it
    controls = [
        locate.locate_track(f"c{s}", synth.record(90.0, seed=s), table) for s in (21, 22, 23)
    ]
    floor = locate.sweep_floor_from(controls)
    return {
        "table": table,
        "length": length,
        "a": a,
        "b": b,
        "c": c,
        "a_zero": a_zero,
        "a_rate": a_rate,
        "floor": floor,
    }


def test_speed_and_time_zero_of_a_deep_cued_fast_record(known_mix):
    got = locate.locate_track("a", known_mix["a"], known_mix["table"], known_mix["floor"])
    assert got.rate == pytest.approx(known_mix["a_rate"], abs=0.0006)
    assert got.time_zero_s == pytest.approx(known_mix["a_zero"], abs=0.05)
    assert got.votes > 300
    assert got.sections_agree >= locate.MIN_SECTIONS_AGREE


def test_played_part_is_in_record_time(known_mix):
    got = locate.locate_track("a", known_mix["a"], known_mix["table"], known_mix["floor"])
    # heard from mix 120 s: record time 30 s. The mix ends at 260 s: record time (260 - zero) * rate.
    assert got.first_heard_s == pytest.approx(120.0, abs=5.0)
    assert got.played_from_s == pytest.approx(30.0, abs=5.0)
    expected_to = min((260.0 - known_mix["a_zero"]) * 1.03, 150.0)
    assert got.played_to_s == pytest.approx(expected_to, abs=5.0)


def test_record_at_the_start_at_speed_one(known_mix):
    got = locate.locate_track("b", known_mix["b"], known_mix["table"], known_mix["floor"])
    assert got.rate == pytest.approx(1.0, abs=0.0006)
    assert got.time_zero_s == pytest.approx(0.0, abs=0.05)
    assert got.first_heard_s == pytest.approx(0.0, abs=5.0)
    assert got.last_heard_s == pytest.approx(100.0, abs=5.0)


def test_absent_record_scores_under_the_floor_set_by_controls(known_mix):
    controls = [synth.record(90.0, seed=s) for s in (21, 22, 23)]
    ctrl = [
        locate.locate_track(f"c{s}", c, known_mix["table"]) for s, c in zip((21, 22, 23), controls)
    ]
    floor = locate.floors.from_controls([c.votes for c in ctrl])
    absent = locate.locate_track("c", known_mix["c"], known_mix["table"])
    present = locate.locate_track("a", known_mix["a"], known_mix["table"])
    assert floor.trusted
    assert locate.confidence(absent.votes, floor, absent.sections_agree) == "not found"
    assert locate.confidence(present.votes, floor, present.sections_agree) == "confident"
    assert locate.floors.separation(present.votes, floor) > 5


def test_consensus_moves_the_pick_to_where_most_sections_agree():
    near = 1000
    sections = [(20, near), (30, near + 2000), (25, near + 2001), (28, near + 1999), (0, near)]
    agree, refined = locate.consensus(sections, near)
    assert refined == near + 2000
    assert agree == 3


def test_consensus_with_no_live_sections_keeps_the_whole_record_pick():
    assert locate.consensus([(0, 5), (2, 9)], near=5) == (0, 5)


def test_confidence_rules():
    floor = locate.floors.from_controls([50, 60, 70])  # 140
    assert locate.confidence(200, floor, 5) == "confident"
    assert locate.confidence(200, floor, 1) == "by votes"
    assert locate.confidence(50, floor, 3) == "by sections"
    assert locate.confidence(50, floor, 2) == "not found"


def test_presence_is_empty_for_an_absent_record(known_mix):
    first, last, best, _, _ = locate.presence(
        fp.fingerprint(known_mix["c"]), known_mix["table"], 200
    )
    assert first is None and last is None and best < 200


def test_presence_follows_a_looped_record(tmp_path):
    """A's first 40 s, then its 20 to 40 s looped three times, then its 40 to 100 s: one record, four
    offsets. Presence must span the whole 160 s and the section consensus still gives one time zero."""
    a = synth.record(100.0, seed=71)
    n = lambda sec: int(sec * fp.SR)  # noqa: E731
    looped = np.concatenate(
        [a[: n(40)], a[n(20) : n(40)], a[n(20) : n(40)], a[n(20) : n(40)], a[n(40) : n(100)]]
    )
    audio = synth.mix([(synth.record(30.0, seed=72), 0.0, 1.0)], 30.0)
    audio = np.concatenate([audio, looped, np.zeros(n(10), dtype=np.float32)])
    path = tmp_path / "looped.wav"
    sf.write(path, audio, fp.SR)
    table, _ = fp.mix_fingerprint(path)
    controls = [
        locate.locate_track(f"c{s}", synth.record(90.0, seed=s), table) for s in (73, 74, 75)
    ]
    got = locate.locate_track("a", a, table, sweep_floor=locate.sweep_floor_from(controls))
    assert got.first_heard_s == pytest.approx(30.0, abs=5.0)
    assert got.last_heard_s == pytest.approx(30.0 + 160.0, abs=5.0)
    assert got.played_from_s == pytest.approx(0.0, abs=5.0)
    assert got.played_to_s == pytest.approx(100.0, abs=5.0)
    assert got.time_zero_s == pytest.approx(30.0, abs=0.05)  # one time zero despite four offsets


def test_sweep_floor_comes_from_the_controls_best_windows(known_mix):
    controls = [
        locate.locate_track(f"c{s}", synth.record(90.0, seed=s), known_mix["table"])
        for s in (24, 25)
    ]
    floor = locate.sweep_floor_from(controls)
    assert floor >= locate.SWEEP_MIN_HASHES
    assert floor >= 2 * max(c.sweep_votes for c in controls) or floor == locate.SWEEP_MIN_HASHES
    # a present record's best window is far above the floor and its extent is the whole playing
    got = locate.locate_track("b", known_mix["b"], known_mix["table"], sweep_floor=floor)
    assert got.sweep_votes > 5 * floor
    assert got.first_heard_s == pytest.approx(0.0, abs=5.0)
    assert got.last_heard_s == pytest.approx(100.0, abs=5.0)


def test_locate_mix_end_to_end(known_mix, tmp_path):
    a_path, c_path = tmp_path / "a.wav", tmp_path / "c.wav"
    sf.write(a_path, known_mix["a"], fp.SR)
    sf.write(c_path, known_mix["c"], fp.SR)
    ctrl_paths = []
    for s in (31, 32, 33):
        p = tmp_path / f"ctrl{s}.wav"
        sf.write(p, synth.record(60.0, seed=s), fp.SR)
        ctrl_paths.append({"track_id": f"ctrl{s}", "path": p})
    mix_path = tmp_path / "mix.wav"
    sf.write(
        mix_path,
        synth.mix(
            [(known_mix["b"], 0.0, 1.0), (known_mix["a"], known_mix["a_zero"], 1.03, 30.0)], 260.0
        ),
        fp.SR,
    )
    got = locate.locate_mix(
        mix_path,
        [{"track_id": "a", "path": a_path}, {"track_id": "c", "path": c_path}],
        ctrl_paths,
    )
    assert got["mix_len_s"] == pytest.approx(260.0, abs=1.0)
    by_id = {r.track_id: r for r in got["records"]}
    assert (
        locate.confidence(by_id["a"].votes, got["floor"], by_id["a"].sections_agree) == "confident"
    )
    assert (
        locate.confidence(by_id["c"].votes, got["floor"], by_id["c"].sections_agree) == "not found"
    )
    assert np.isfinite(by_id["a"].seconds)
