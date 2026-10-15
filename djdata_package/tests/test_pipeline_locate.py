"""The locate stage end to end on a synthetic corpus: two mixes, known placements, resume."""

import csv

import pytest
import soundfile as sf
import yaml

from djdata import config as config_mod
from djdata import pipeline
from djdata.seam.fingerprint import SR
from djdata.store import tables
from tests import synth
from tests.test_sources import TRACKLIST_COLUMNS


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    (root / "mixes_tmp").mkdir()
    (root / "tracks").mkdir()
    recs = {f"t{i}": synth.record(150.0, seed=100 + i) for i in range(1, 7)}
    for tid, rec in recs.items():
        sf.write(root / "tracks" / f"{tid}.wav", rec, SR)
    # records are 150 s so three 45 s sections can agree; m1: t1 from 0, t2 from 60 s cued 10 s deep
    # at 1.02; t3 listed, on disk, not in the mix
    sf.write(
        root / "mixes_tmp" / "m1.wav",
        synth.mix([(recs["t1"], 0.0, 1.0), (recs["t2"], 60.0 - 10.0 / 1.02, 1.02, 10.0)], 220.0),
        SR,
    )
    # m2: t4 from 0, t5 from 50
    sf.write(
        root / "mixes_tmp" / "m2.wav",
        synth.mix([(recs["t4"], 0.0, 1.0), (recs["t5"], 50.0, 1.0)], 210.0),
        SR,
    )
    rows = [
        ("m1", "Set one", "Dj A", "techno", "t1", "0.0"),
        ("m1", "Set one", "Dj A", "techno", "t2", "1.0"),
        ("m1", "Set one", "Dj A", "techno", "t3", "1.5"),
        ("m2", "Set two", "Dj B", "house", "t4", "0.0"),
        ("m2", "Set two", "Dj B", "house", "t5", "0.8"),
    ]
    with open(root / "tracklist.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=TRACKLIST_COLUMNS)
        w.writeheader()
        for mix_id, title, dj, genre, tid, minute in rows:
            w.writerow(
                {
                    "mix_id": mix_id,
                    "mix_title": title,
                    "dj_name": dj,
                    "genre": genre,
                    "url": "",
                    "track_id": tid,
                    "starting_time": minute,
                    "track_name": tid,
                    "artist_name": "A",
                    "play_type": "sequential",
                    "overlay_parent": "",
                }
            )
    cfg_yaml = {
        "source": "tracklists",
        "root": str(root),
        "raveform_dir": str(root),
        "tracklists_csv": str(root / "tracklist.csv"),
        "run_tiers": ["tracklists"],
        "tiers": {},
        "usable": {},
        "workers": {},
        "download": {},
        "window": {},
        "analysis": {},
        "logging": {"level": "INFO"},
    }
    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_yaml))
    return config_mod.load(cfg_path)


def test_locate_stage_writes_plays_with_floors_and_resumes(corpus):
    counts = pipeline.locate(corpus, workers=2)
    assert counts["ran"] == 2 and counts["failed"] == 0 and counts["records"] == 5
    rows = tables.plays(corpus.dirs["out"]).rows()
    own = {(r["mix_id"], r["track_id"]): r for r in rows if r["is_control"] == "0"}
    ctrl = [r for r in rows if r["is_control"] == "1"]
    # three controls per mix from the other mix's records: m2 has only two, so m1's floor is thin
    assert len(ctrl) == 5
    m1_ctrl = [r for r in ctrl if r["mix_id"] == "m1"]
    assert len(m1_ctrl) == 2 and m1_ctrl[0]["control_n"] == "2"
    assert all(r["confidence"] == "control" for r in ctrl)
    t2 = own[("m1", "t2")]
    assert t2["confidence"] == "confident" and t2["found"] == "1"
    assert tables.num(t2["rate"]) == pytest.approx(1.02, abs=0.0006)
    assert tables.num(t2["time_zero_s"]) == pytest.approx(60.0 - 10.0 / 1.02, abs=0.05)
    assert tables.num(t2["first_heard_s"]) == pytest.approx(60.0, abs=5.0)
    assert tables.num(t2["played_from_s"]) == pytest.approx(10.0, abs=5.0)
    assert tables.num(t2["drift_min"]) == pytest.approx((60.0 - 10.0 / 1.02) / 60 - 1.0, abs=0.01)
    assert (
        tables.num(t2["floor"]) == tables.num(ctrl[0]["floor"])
        if ctrl[0]["mix_id"] == "m1"
        else True
    )
    assert own[("m2", "t5")]["confidence"] == "confident"
    assert own[("m1", "t3")]["confidence"] == "not found" and own[("m1", "t3")]["found"] == "0"
    # resume: nothing left to run
    again = pipeline.locate(corpus, workers=1)
    assert again["ran"] == 0 and again["done_before"] == 2
