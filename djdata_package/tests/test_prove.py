"""Which file is which show, by play order: the rule on its own, then the stage end to end."""

import csv

import pytest
import soundfile as sf
import yaml

from djdata import config as config_mod
from djdata import pipeline
from djdata.seam import prove
from djdata.seam.fingerprint import SR
from djdata.store import tables
from tests import synth
from tests.test_sources import TRACKLIST_COLUMNS


def test_longest_rising_finds_the_chain_in_list_order():
    times = [10.0, 50.0, 20.0, 30.0, 5.0, 40.0]
    assert [times[i] for i in prove.longest_rising(times)] == [10.0, 20.0, 30.0, 40.0]
    assert prove.longest_rising([]) == []


def test_order_chance_is_small_for_a_long_chain_and_large_for_scatter():
    _, p_sorted = prove.order_chance([float(t) for t in range(8)], 8, seed="a")
    assert p_sorted < 0.001  # one ordering in 40,320
    scatter = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
    observed = len(prove.longest_rising(scatter))
    median, p_scatter = prove.order_chance(scatter, observed, seed="a")
    assert p_scatter > 0.1 and median >= 3


def test_four_in_order_is_not_enough_on_its_own():
    # four records in order is one ordering in 24: the shuffle says so and the rule does not pass it
    rows = [
        {
            "order_listed": k,
            "time_zero_s": 60.0 * k,
            "found": True,
            "confidence": "by votes",
            "floor": 90,
            "control_max": 45,
        }
        for k in range(1, 5)
    ]
    s = prove.summarise("L", "F", rows, listed=10)
    assert s["in_order"] == 4 and s["p_order"] == pytest.approx(1 / 24, abs=0.015)
    assert s["proved"] == 0


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("prove")
    (root / "tracks").mkdir()
    (root / "files").mkdir()
    recs = {f"r{i}": synth.record(40.0, seed=300 + i) for i in range(1, 6)}
    ctrls = {f"c{i}": synth.record(40.0, seed=400 + i) for i in range(1, 4)}
    for tid, rec in {**recs, **ctrls}.items():
        sf.write(root / "tracks" / f"{tid}.wav", rec, SR)
    in_order = ["r1", "r2", "r3", "r4", "r5"]
    scrambled = ["r3", "r5", "r1", "r4", "r2"]  # the tour's same records, another night's order
    for name, order in (("f1", in_order), ("f2", scrambled)):
        sf.write(
            root / "files" / f"{name}.wav",
            synth.mix([(recs[t], 40.0 * k, 1.0) for k, t in enumerate(order)], 200.0),
            SR,
        )
    rows = [("L1", "Dj A", t, str(k)) for k, t in enumerate(in_order)]
    rows += [("M9", "Dj C", t, str(k)) for k, t in enumerate(ctrls)]
    with open(root / "tracklist.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=TRACKLIST_COLUMNS)
        w.writeheader()
        for mix_id, dj, tid, minute in rows:
            w.writerow(
                {
                    "mix_id": mix_id,
                    "mix_title": mix_id,
                    "dj_name": dj,
                    "genre": "house",
                    "track_id": tid,
                    "starting_time": minute,
                    "track_name": tid,
                    "artist_name": "A",
                }
            )
    cfg_yaml = {
        "source": "tracklists",
        "root": str(root),
        "raveform_dir": str(root),
        "tracklists_csv": str(root / "tracklist.csv"),
        "control_djs": ["Dj C"],
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
    with open(root / "candidates.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["file_id", "path"])
        w.writeheader()
        for name in ("f1", "f2"):
            w.writerow({"file_id": name, "path": root / "files" / f"{name}.wav"})
    with open(root / "lists.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=tables.PROOF_LISTS)
        w.writeheader()
        w.writerows(pipeline.lists_1001(cfg, ["Dj A"]))
    return cfg


def test_prove_stage_proves_the_right_file_and_not_the_one_with_the_same_records(corpus):
    root = corpus.root
    counts = pipeline.prove(corpus, root / "candidates.csv", root / "lists.csv", workers=2)
    assert counts["ran"] == 2 and counts["failed"] == 0 and counts["proved"] == 1
    summary = {r["file_id"]: r for r in tables.proof_summary(corpus.dirs["out"]).rows()}
    right, wrong = summary["f1"], summary["f2"]
    # the wrong file holds every record: votes alone would take it
    assert right["found"] == "5" and wrong["found"] == "5"
    assert right["in_order"] == "5" and right["proved"] == "1"
    assert right["chain_orders"] == "1 2 3 4 5"
    assert int(wrong["in_order"]) <= 2 and wrong["proved"] == "0"
    rows = tables.proofs(corpus.dirs["out"]).rows()
    ctrl = [r for r in rows if r["is_control"] == "1"]
    assert len(ctrl) == 6 and {r["track_id"] for r in ctrl} == {"c1", "c2", "c3"}
    # resume: nothing left to run, the summary is rebuilt from the table
    again = pipeline.prove(corpus, root / "candidates.csv", root / "lists.csv", workers=1)
    assert again["ran"] == 0 and again["done_before"] == 2 and again["proved"] == 1


def test_time_columns_report_how_far_found_records_sit_from_the_listed_time():
    rows = [
        {
            "order_listed": 1,
            "time_zero_s": 10.0,
            "first_heard_s": 70.0,
            "listed_min": 1.0,
            "found": True,
            "confidence": "confident",
            "floor": 90,
            "control_max": 45,
        },
        {
            "order_listed": 2,
            "time_zero_s": 200.0,
            "first_heard_s": 400.0,
            "listed_min": 5.0,
            "found": True,
            "confidence": "confident",
            "floor": 90,
            "control_max": 45,
        },
        {
            "order_listed": 3,
            "time_zero_s": 300.0,
            "first_heard_s": None,
            "listed_min": 9.0,
            "found": False,
            "confidence": "not found",
            "floor": 90,
            "control_max": 45,
        },
    ]
    s = prove.summarise("L", "F", rows, listed=3)
    assert s["time_checked"] == 2 and s["time_within_60s"] == 1
    assert s["time_median_off_s"] == 100.0  # 0 s and 100 s off, the upper median of two


def test_usb002_solo_lists_place_each_track_inside_its_own_segment(tmp_path):
    import json

    from djdata.sources import usb002

    app = {
        "video_id": "v",
        "shows": [
            {
                "name": "Madrid",
                "sets": [
                    {
                        "name": "Fred",
                        "global_start": 1000,
                        "global_end": 2000,
                        "tracks": [
                            {"kind": "track", "t": 1300, "title": "Two", "artist": "B"},
                            {"kind": "track", "t": 1060, "title": "One", "artist": "A & C"},
                        ],
                    },
                    {
                        "name": "Fred b2b X",
                        "global_start": 2000,
                        "global_end": 3000,
                        "tracks": [{"kind": "track", "t": 2100, "title": "Three", "artist": "D"}],
                    },
                ],
            }
        ],
    }
    (tmp_path / "app.json").write_text(json.dumps(app))
    with open(tmp_path / "tracks.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["track_id", "source", "artist", "title"])
        w.writeheader()
        w.writerow({"track_id": "t1", "source": "1001 on disk", "artist": "A & C", "title": "One"})
        w.writerow({"track_id": "usb_2", "source": "to fetch", "artist": "B", "title": "Two"})
    with open(tmp_path / "segments.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["file", "marathon_start_s", "is_b2b"])
        w.writeheader()
        w.writerow({"file": "fred_07_madrid_fred.m4a", "marathon_start_s": "1000", "is_b2b": "0"})
        w.writerow({"file": "fred_08_x.m4a", "marathon_start_s": "2000", "is_b2b": "1"})
    lists, candidates = usb002.solo_lists(
        tmp_path / "app.json", tmp_path / "tracks.csv", tmp_path / "segments.csv"
    )
    assert [(r["order_listed"], r["track_id"], r["listed_min"]) for r in lists] == [
        (1, "t1", 1.0),
        (2, "usb_2", 5.0),
    ]
    assert {r["list_id"] for r in lists} == {"usb_fred_07_madrid_fred"}
    assert candidates == [
        {
            "file_id": "fred_07_madrid_fred",
            "path": "fred_07_madrid_fred.m4a",
            "list_id": "usb_fred_07_madrid_fred",
        }
    ]
    rows = usb002.tracklist_rows(
        tmp_path / "app.json", tmp_path / "tracks.csv", tmp_path / "segments.csv"
    )
    assert [(r["mix_id"], r["track_id"], r["starting_time"], r["exact_time"]) for r in rows] == [
        ("usb_fred_07_madrid_fred", "t1", 1.0, 1),
        ("usb_fred_07_madrid_fred", "usb_2", 5.0, 1),
    ]
    assert rows[0]["artist_name"] == "A & C" and rows[0]["track_name"] == "One"
    assert rows[0]["dj_name"] == "Fredagain.."
