"""A synthetic Raveform corpus: alignment in, windows on disk, the same stages through to a label."""

import json

import numpy as np
import pytest
import soundfile as sf
import yaml

from djdata import config as config_mod
from djdata import pipeline
from djdata.seam import cut, measure
from djdata.seam.fingerprint import SR
from djdata.state import State
from djdata.store import tables
from tests import synth

A_ZERO, B_ZERO, B_RATE = 0.0, 100.0, 1.0
A_OUT, B_IN = 130.0, 100.0  # A leaves at 130, B enters at 100 with its start


@pytest.fixture(scope="module")
def rave(tmp_path_factory):
    root = tmp_path_factory.mktemp("rave")
    R = root / "raw"
    (R / "alignments").mkdir(parents=True)
    for d in ("windows", "tracks", "out", "logs"):
        (root / d).mkdir()
    a = synth.record(150.0, seed=81, bass_voices=2)
    b = synth.record(150.0, seed=82, bass_voices=2)
    c = synth.record(150.0, seed=83, bass_voices=2)
    for tid, rec in (("vidA", a), ("vidB", b), ("vidC", c)):
        sf.write(root / "tracks" / f"{tid}.wav", rec, SR)
    # a second mix of three other records, so the first mix has three controls to draw from
    others = {f"vid{k}": synth.record(60.0, seed=90 + i) for i, k in enumerate("DEF")}
    for tid, rec in others.items():
        sf.write(root / "tracks" / f"{tid}.wav", rec, SR)
    sf.write(
        root / "windows" / "mixY_vidD_vidE.wav", synth.mix([(others["vidD"], 0.0, 1.0)], 30.0), SR
    )
    sf.write(
        root / "windows" / "mixY_vidE_vidF.wav", synth.mix([(others["vidE"], 0.0, 1.0)], 30.0), SR
    )
    # the mix: A alone, then A and B together 100 to 130, then B alone; the window is 60 to 200
    n = int(200.0 * SR)
    mix = synth.mix([(a, A_ZERO, 1.0), (b, B_ZERO, B_RATE)], 200.0)
    gate_a = 1.0 - synth.ramp(n, A_OUT)
    a_part = synth.mix([(a, A_ZERO, 1.0)], 200.0)
    mix = (mix - a_part) + a_part * gate_a
    t0, t1 = 60.0, 200.0
    sf.write(root / "windows" / "mixX_vidA_vidB.wav", mix[int(t0 * SR) : int(t1 * SR)], SR)
    (R / "mixes.jsonl").write_text(
        json.dumps(
            {
                "id": "mixX",
                "title": "Some DJ - Some Set 2020",
                "tracklist": [{"id": "vidA"}, {"id": "vidB"}, {"id": "vidC"}],
                "audio_url": "",
                "audio_source": "",
                "genres": ["techno"],
            }
        )
        + "\n"
    )
    (R / "tracks.jsonl").write_text(
        "".join(
            json.dumps({"id": t, "title": t, "duration": 150.0}) + "\n"
            for t in ("vidA", "vidB", "vidC", "vidD", "vidE", "vidF")
        )
    )
    als = [
        {
            "mix_id": "mixX",
            "track_id": "vidA",
            "mixin_time_mix": 0.0,
            "mixin_time_track": 0.0,
            "mixout_time_mix": A_OUT,
            "mixout_time_track": A_OUT,
            "matched_time_track": 100.0,
            "matched_time_mix": 100.0,
            "match_rate": 0.9,
        },
        {
            "mix_id": "mixX",
            "track_id": "vidB",
            "mixin_time_mix": B_IN,
            "mixin_time_track": 0.0,
            "mixout_time_mix": 200.0,
            "mixout_time_track": 100.0,
            "matched_time_track": 50.0,
            "matched_time_mix": 50.0,
            "match_rate": 0.9,
        },
        {
            "mix_id": "mixX",
            "track_id": "vidC",
            "mixin_time_mix": 200.0,
            "mixin_time_track": 0.0,
            "mixout_time_mix": 300.0,
            "mixout_time_track": 100.0,
            "matched_time_track": 50.0,
            "matched_time_mix": 50.0,
            "match_rate": 0.9,
        },
    ]
    (R / "alignments" / "mixX.align.jsonl").write_text("".join(json.dumps(x) + "\n" for x in als))
    with open(R / "mixes.jsonl", "a") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "mixY",
                    "title": "Other DJ - Other Set 2021",
                    "audio_url": "",
                    "audio_source": "",
                    "genres": ["house"],
                    "tracklist": [{"id": "vidD"}, {"id": "vidE"}, {"id": "vidF"}],
                }
            )
            + "\n"
        )
    y_als = []
    for i, tid in enumerate(("vidD", "vidE", "vidF")):
        y_als.append(
            {
                "mix_id": "mixY",
                "track_id": tid,
                "mixin_time_mix": 50.0 * i,
                "mixin_time_track": 0.0,
                "mixout_time_mix": 50.0 * i + 55.0,
                "mixout_time_track": 55.0,
                "matched_time_track": 20.0,
                "matched_time_mix": 20.0,
                "match_rate": 0.9,
            }
        )
    (R / "alignments" / "mixY.align.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in y_als)
    )
    state = State(root / "state.sqlite")
    state.add_seam(
        "mixX_vidA_vidB",
        "mixX",
        "vidA",
        "vidB",
        "t1",
        "raveform",
        {
            "A": {"orig_t": 0.0, "mix_t": 0.0, "rate": 1.0},
            "B": {"orig_t": 0.0, "mix_t": B_IN, "rate": 1.0},
            "window": [t0, t1],
        },
    )
    for sid, aa, bb in (("mixY_vidD_vidE", "vidD", "vidE"), ("mixY_vidE_vidF", "vidE", "vidF")):
        state.add_seam(
            sid, "mixY", aa, bb, "t1", "raveform", {"A": {}, "B": {}, "window": [0.0, 30.0]}
        )
    state.add_seam(
        "mixX_vidB_vidC",
        "mixX",
        "vidB",
        "vidC",
        "t1",
        "raveform",
        {"A": {}, "B": {}, "window": [150.0, 300.0]},
    )  # no window file on disk for this one
    cfg_yaml = {
        "source": "raveform",
        "root": str(root),
        "raveform_dir": str(R),
        "run_tiers": ["t1"],
        "tiers": {},
        "usable": {},
        "workers": {},
        "download": {},
        "window": {},
        "analysis": {},
        "logging": {"level": "INFO"},
    }
    (root / "cfg.yaml").write_text(yaml.safe_dump(cfg_yaml))
    return config_mod.load(root / "cfg.yaml")


def test_alignment_stands_in_for_locate(rave):
    counts = pipeline.locate(rave)
    assert counts["records"] == 6 and counts["ran"] == 0
    rows = {r["track_id"]: r for r in tables.plays(rave.dirs["out"]).rows()}
    assert rows["vidA"]["confidence"] == "alignment" and rows["vidB"]["found"] == "1"
    assert rows["vidC"]["confidence"] == "no window" and rows["vidC"]["found"] == "0"
    assert tables.num(rows["vidB"]["time_zero_s"]) == B_ZERO
    assert tables.num(rows["vidA"]["last_heard_s"]) == A_OUT
    assert pipeline.locate(rave)["records"] == 0


def test_pairs_adopt_the_manifest_window(rave):
    pipeline.locate(rave)
    counts = pipeline.pairs(rave)
    assert counts["seams"] == 3 and counts["usable"] == 3
    s = {r["seam_id"]: r for r in tables.seams(rave.dirs["out"]).rows()}["mixX_vidA_vidB"]
    assert tables.num(s["window_t0"]) == 60.0 and tables.num(s["window_t1"]) == 200.0


def test_cut_adopts_and_audits_then_measures_and_labels(rave):
    pipeline.locate(rave)
    pipeline.pairs(rave)
    counts = pipeline.cut(rave, workers=1)
    assert counts["cut"] == 3 and counts["failed"] == 0
    c = {r["seam_id"]: r for r in tables.cuts(rave.dirs["out"]).rows()}["mixX_vidA_vidB"]
    assert c["status"] == "adopted" and c["window_file"] == "mixX_vidA_vidB.wav"
    assert c["audit_start_ok"] == "1" and c["audit_end_ok"] == "1"
    m = pipeline.measure(rave, workers=1)
    assert m["measured"] == 3
    row = {r["seam_id"]: r for r in tables.measures(rave.dirs["out"]).rows()}["mixX_vidA_vidB"]
    assert row["measured"] == "1" and row["control_trusted"] == "1"
    assert tables.num(row["in_s"]) == pytest.approx(B_IN - 60.0, abs=10.0)
    assert tables.num(row["out_s"]) == pytest.approx(A_OUT - 60.0, abs=10.0)
    pipeline.tempo(rave, workers=1)
    assert pipeline.label(rave)["labelled"] == 3
    assert pipeline.export(rave)["seams"] == 3


EXCESS_S = 8.0


def test_a_window_that_starts_early_is_read_from_its_real_start(rave):
    """The manifest's webm windows start at the keyframe before the asked start, so the file holds
    extra audio at the front. Read from the asked start every record sits EXCESS_S off its anchor;
    read from file_start it is where it belongs."""
    t0, asked = 60.0, 140.0
    a, _ = sf.read(rave.dirs["tracks"] / "vidA.wav", dtype="float32")
    clean_audio, _ = sf.read(rave.dirs["windows"] / "mixX_vidA_vidB.wav", dtype="float32")
    path = rave.root / "early.wav"
    sf.write(path, np.concatenate([a[int((t0 - EXCESS_S) * SR) : int(t0 * SR)], clean_audio]), SR)
    actual = len(clean_audio) / SR + EXCESS_S
    start = cut.file_start(t0, asked, actual, adopted=True)
    assert start == pytest.approx(t0 - EXCESS_S, abs=0.01)
    assert cut.file_start(t0, asked, actual, adopted=False) == t0
    assert cut.file_start(t0, asked, asked + 0.1, adopted=True) == t0

    tracks = rave.dirs["tracks"]

    def job(window_t0):
        return {
            "seam_id": "early",
            "window_path": path,
            "window_t0": window_t0,
            "a": {"path": tracks / "vidA.wav", "time_zero_s": A_ZERO, "rate": 1.0},
            "b": {"path": tracks / "vidB.wav", "time_zero_s": B_ZERO, "rate": B_RATE},
            "controls": [tracks / f"vid{k}.wav" for k in "DEF"],
        }

    right, _ = measure.measure_one(job(start))
    wrong, _ = measure.measure_one(job(t0))
    clean, _ = measure.measure_one(
        dict(job(t0), window_path=rave.dirs["windows"] / "mixX_vidA_vidB.wav")
    )
    # the early file holds the clean window EXCESS_S later, so every band time moves by exactly that
    for key in ("low_in_s", "mid_in_s", "bass_swap_s", "low_out_s", "mid_out_s"):
        assert right[key] == pytest.approx(clean[key] + EXCESS_S, abs=1.0), key
    assert right["mid_a_separation"] > 20 and right["mid_b_separation"] > 20
    # read from the asked start, both records sit at their floor and presence finds neither
    assert wrong["mid_a_separation"] <= 1.5 and wrong["mid_b_separation"] <= 1.5
    assert wrong["in_s"] is None and wrong["out_s"] is None
    assert right["presence_a_votes"] > 50 * max(1, wrong["presence_a_votes"])
