"""Stacked records, windowed lookup at exact times, the own-DJ control rule and the source table.

A synthetic corpus with a known answer. Mix m1: r1 from 0 s, r2 and r3 entering at 110 s, so from
110 s to 150 s three records sound at once; there r1 plays its bass only and r2 everything but its
bass. Mix m2 carries an exact-time list: r5 is listed at the time it plays, r6 at a time far from
where it plays."""

import csv

import numpy as np
import pytest
import soundfile as sf
import yaml

from djdata import config as config_mod
from djdata import pipeline
from djdata.seam.fingerprint import SR
from djdata.sources import tracklists
from djdata.store import tables
from tests import synth
from tests.test_sources import TRACKLIST_COLUMNS


def _place(out, audio, at_s):
    start = int(at_s * SR)
    end = min(start + len(audio), len(out))
    out[start:end] += audio[: end - start]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("layers")
    (root / "mixes_tmp").mkdir()
    (root / "tracks").mkdir()
    recs = {f"r{i}": synth.record(150.0, seed=500 + i, bass_voices=2) for i in range(1, 7)}
    ctrls = {f"c{i}": synth.record(150.0, seed=600 + i, bass_voices=2) for i in range(1, 4)}
    for tid, rec in {**recs, **ctrls}.items():
        sf.write(root / "tracks" / f"{tid}.wav", rec, SR)

    # m1: r1 full to 100 s, its bass only from 100 s to 150 s; r2 from 110 s without its bass until
    # 150 s, then full; r3 full from 110 s
    m1 = np.zeros(int(300 * SR), dtype=np.float32)
    r1_low, r1_rest = synth.split_low(recs["r1"])
    cut = int(100 * SR)
    _place(m1, np.concatenate([recs["r1"][:cut], r1_low[cut:]]), 0.0)
    r2_low, r2_rest = synth.split_low(recs["r2"])
    edge = int(40 * SR)
    _place(m1, np.concatenate([r2_rest[:edge], recs["r2"][edge:]]), 110.0)
    _place(m1, recs["r3"], 110.0)
    sf.write(root / "mixes_tmp" / "m1.wav", m1, SR)

    # m2: r5 at 200 s, r6 at 60 s; the exact list says r5 at 200 s (right) and r6 at 520 s (wrong)
    m2 = np.zeros(int(600 * SR), dtype=np.float32)
    _place(m2, recs["r5"], 200.0)
    _place(m2, recs["r6"], 60.0)
    sf.write(root / "mixes_tmp" / "m2.wav", m2, SR)

    def row(mix_id, dj, tid, minute, **extra):
        return {
            "mix_id": mix_id,
            "mix_title": mix_id,
            "dj_name": dj,
            "genre": "house",
            "track_id": tid,
            "starting_time": minute,
            "track_name": tid,
            "artist_name": "A",
            "play_type": "sequential",
            **extra,
        }

    with open(root / "tracklist.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=TRACKLIST_COLUMNS)
        w.writeheader()
        for tid, minute in (("r1", "0.0"), ("r2", "1.8"), ("r3", "1.9")):
            w.writerow(row("m1", "Dj A", tid, minute))
        for tid in ctrls:
            w.writerow(row("m9", "Dj C", tid, "0.0"))
        w.writerow(row("m8", "Dj B", "r1", "0.0"))  # Dj B also lists r1, in a mix with no audio
    with open(root / "exact.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=TRACKLIST_COLUMNS + ["exact_time"])
        w.writeheader()
        w.writerow(row("m2", "Dj B", "r5", str(200 / 60), exact_time=1))
        w.writerow(row("m2", "Dj B", "r6", str(520 / 60), exact_time=1))
    cfg_yaml = {
        "source": "tracklists",
        "root": str(root),
        "raveform_dir": str(root),
        "tracklists_csv": str(root / "tracklist.csv"),
        "extra_tracklists": [str(root / "exact.csv")],
        "control_djs": ["Dj C"],
        "controls_exclude_own_dj": ["Dj B"],
        "run_tiers": ["tracklists"],
        "tiers": {},
        "usable": {},
        "workers": {},
        "download": {},
        "window": {},
        "analysis": {},
        "logging": {"level": "INFO"},
    }
    (root / "cfg.yaml").write_text(yaml.safe_dump(cfg_yaml))
    cfg = config_mod.load(root / "cfg.yaml")
    pipeline.locate(cfg, workers=2)
    pipeline.layers(cfg)
    pipeline.layer_bands(cfg, workers=2)
    return cfg


def _plays(cfg):
    return {
        (r["mix_id"], r["track_id"]): r
        for r in tables.plays(cfg.dirs["out"]).rows()
        if r["is_control"] == "0"
    }


def test_exact_time_list_is_searched_at_its_time_and_presence_still_reads_the_whole_mix(corpus):
    plays = _plays(corpus)
    r5, r6 = plays[("m2", "r5")], plays[("m2", "r6")]
    assert r5["search"] == "window" and r6["search"] == "window"
    assert r5["found"] == "1"
    assert tables.num(r5["time_zero_s"]) == pytest.approx(200.0, abs=0.1)
    assert tables.num(r5["window_votes"]) >= tables.num(r5["window_floor"])
    # r6 plays at 60 s, far outside a window around its listed 520 s: not found there
    assert r6["found"] == "0"
    # but the whole-mix presence still hears it where it really plays
    assert tables.num(r6["first_heard_s"]) == pytest.approx(60.0, abs=5.0)
    # a 1001-style list keeps the whole-mix search and leaves the window columns empty
    assert plays[("m1", "r1")]["search"] == "whole" and plays[("m1", "r1")]["window_votes"] == ""


def test_presence_rows_follow_each_record_and_three_are_present_where_they_stack(corpus):
    pres = tables.presence(corpus.dirs["out"]).rows()
    r3 = [
        tables.num(r["window_start_s"])
        for r in pres
        if r["mix_id"] == "m1" and r["track_id"] == "r3" and r["state"] == "present"
    ]
    # r3 plays 110 s to 260 s: the 30 s windows that overlap it start between 80 s and 260 s
    assert 80.0 <= min(r3) <= 110.0 and 230.0 <= max(r3) <= 260.0
    assert {r["state"] for r in pres} <= {"present", "weak"}
    # weak rows are chance-level evidence by design: above the loudest control, under twice it
    for r in pres:
        if r["state"] == "weak":
            assert (
                tables.num(r["weak_level"]) < tables.num(r["votes"]) < tables.num(r["sweep_floor"])
            )
    lay = [r for r in tables.layers(corpus.dirs["out"]).rows() if r["mix_id"] == "m1"]
    three = [tables.num(r["window_start_s"]) for r in lay if r["n_present"] == "3"]
    assert three, "no window with three records present"
    assert min(three) >= 80.0 and max(three) <= 150.0
    alone = [r for r in lay if tables.num(r["window_start_s"]) <= 40.0]
    assert alone and all(r["present"] == "r1" and r["n_present"] == "1" for r in alone)


def test_layer_bands_say_which_record_carries_which_band_in_the_stack(corpus):
    rows = [r for r in tables.layer_bands(corpus.dirs["out"]).rows() if r["mix_id"] == "m1"]
    assert rows, "the three-record stack gave no span"
    seen = {}
    for r in rows:
        key = (r["track_id"], r["band"])
        seen[key] = max(seen.get(key, 0.0), tables.num(r["seen_s"]))
    # in the stack r1 plays its bass only, r2 everything but its bass (until 150 s), r3 everything
    assert seen[("r3", "low")] > 0 and seen[("r3", "mid")] > 0
    assert seen[("r1", "low")] > 0
    assert seen[("r2", "mid")] > 0
    stack_rows = [r for r in rows if tables.num(r["span_start_s"]) <= 120.0]
    r1_mid = [
        tables.num(r["seen_s"]) for r in stack_rows if r["track_id"] == "r1" and r["band"] == "mid"
    ]
    r1_low = [
        tables.num(r["seen_s"]) for r in stack_rows if r["track_id"] == "r1" and r["band"] == "low"
    ]
    assert sum(r1_low) > sum(r1_mid)


def test_mixes_table_carries_the_source_quality(corpus):
    mixes = {r["mix_id"]: r for r in tables.mixes(corpus.dirs["out"]).rows()}
    assert set(mixes) == {"m1", "m2"}
    assert mixes["m1"]["codec"].startswith("pcm") and mixes["m1"]["sample_rate"] == str(SR)
    assert tables.num(mixes["m2"]["duration_s"]) == pytest.approx(600.0, abs=0.5)


def test_own_dj_records_are_never_a_control_for_that_dj(corpus):
    own = tracklists.dj_records(corpus.tracklists_csv, ["Dj B"], corpus.raw["extra_tracklists"])
    assert own == {"r1", "r5", "r6"}
    pool = [{"track_id": t, "path": None} for t in ("r1", "c1", "c2", "c3")]
    picked = tracklists.controls_for({"mix_id": "m2", "tracks": []}, pool, 3, own)
    assert {c["track_id"] for c in picked} == {"c1", "c2", "c3"}
    ctrl = [r for r in tables.plays(corpus.dirs["out"]).rows() if r["is_control"] == "1"]
    assert {r["track_id"] for r in ctrl} <= {"c1", "c2", "c3"}


def test_layer_spans_skip_an_ordinary_seam_and_keep_a_stack_or_a_far_pair():
    stack = {
        0.0: {"a": "present"},
        10.0: {"a": "present", "b": "present"},  # a -> b, the ordinary seam
        20.0: {
            "a": "present",
            "b": "present",
            "x": "weak",
            "y": "weak",
        },  # weak never makes a span
        30.0: {"b": "present"},
        40.0: {"b": "present", "d": "present"},  # b with d, not adjacent in audio order
        50.0: {"b": "present", "c": "present", "d": "present", "z": "weak"},  # three at once
        60.0: {"c": "present"},
    }
    spans = pipeline.layer_spans(stack, {frozenset({"a", "b"}), frozenset({"b", "c"})})
    assert len(spans) == 1
    sp = spans[0]
    assert sp["start_s"] == 40.0 and sp["end_s"] == 50.0 + 30.0
    assert sp["records"] == {"b": "present", "c": "present", "d": "present"}


def test_one_recording_under_two_ids_is_counted_once():
    plays_rows = [
        {"track_id": "v1", "is_control": "0", "time_zero_s": "425.31"},
        {"track_id": "v2", "is_control": "0", "time_zero_s": "425.27"},  # the same audio, other id
        {"track_id": "w", "is_control": "0", "time_zero_s": "300.0"},  # a real second record
    ]
    pres = [
        {"track_id": t, "window_start_s": str(s), "state": "present"}
        for t in ("v1", "v2")
        for s in (430, 440, 450, 460)
    ] + [{"track_id": "w", "window_start_s": str(s), "state": "present"} for s in (440, 450)]
    # without an audio check nothing is merged: two records started together look the same here
    assert pipeline.same_audio(plays_rows, pres)["v2"] == "v2"
    canon = pipeline.same_audio(plays_rows, pres, lambda a, b: {a, b} == {"v1", "v2"})
    assert canon["v2"] == "v1" and canon["w"] == "w"
    stack = pipeline._stack(pres, canon)
    assert stack[450.0] == {"v1": "present", "w": "present"}
    assert pipeline._aliases(canon) == {"v1": "v1=v2"}


def test_same_recording_is_decided_on_the_audio_against_controls(tmp_path):
    rec = synth.record(90.0, seed=800)
    other = synth.record(90.0, seed=801)
    ctrls = [synth.record(90.0, seed=810 + i) for i in range(3)]
    for name, audio in {
        "a": rec,
        "a_copy": rec,
        "b": other,
        **{f"c{i}": c for i, c in enumerate(ctrls)},
    }.items():
        sf.write(tmp_path / f"{name}.wav", audio, SR)
    c = [tmp_path / f"c{i}.wav" for i in range(3)]
    assert pipeline.same_recording(tmp_path / "a.wav", tmp_path / "a_copy.wav", c)
    assert not pipeline.same_recording(tmp_path / "a.wav", tmp_path / "b.wav", c)


def test_matching_the_cut_table_gives_the_votes_of_the_span_filter():
    from djdata.seam import locate
    from djdata.seam.fingerprint import FRAME_S, as_arrays, fingerprint, match

    rec = synth.record(60.0, seed=700)
    mix = synth.mix([(rec, 40.0, 1.0), (synth.record(60.0, seed=701), 120.0, 1.0)], 200.0)
    table = as_arrays(fingerprint(mix))
    fp = fingerprint(rec)
    for span in (
        (0, int(200 / FRAME_S)),
        (int(30 / FRAME_S), int(90 / FRAME_S)),
        (int(110 / FRAME_S), int(190 / FRAME_S)),
    ):
        assert match(fp, locate.restrict(table, span)) == match(fp, table, span)
