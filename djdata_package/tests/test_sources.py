"""The corpus is what is on disk: a mix with audio and a tracklist, records with audio."""

import csv

from djdata.sources import tracklists

TRACKLIST_COLUMNS = [
    "mix_id",
    "mix_title",
    "dj_name",
    "genre",
    "url",
    "track_id",
    "starting_time",
    "track_name",
    "artist_name",
    "play_type",
    "overlay_parent",
]


def _tracklist(path, rows):
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=TRACKLIST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in TRACKLIST_COLUMNS})


def _corpus(tmp_path):
    (tmp_path / "mixes_tmp").mkdir()
    (tmp_path / "tracks").mkdir()
    (tmp_path / "mixes_tmp" / "m1.mp3").write_bytes(b"x")
    (tmp_path / "mixes_tmp" / "m1.mp3.part").write_bytes(b"x")
    for t in ("t1", "t2", "t9"):
        (tmp_path / "tracks" / f"{t}.m4a").write_bytes(b"x")
    rows = [
        {
            "mix_id": "m1",
            "mix_title": "Set one",
            "dj_name": "Dj A",
            "genre": "techno",
            "track_id": "t1",
            "starting_time": "0.0",
            "track_name": "One",
            "artist_name": "X",
            "play_type": "sequential",
        },
        {
            "mix_id": "m1",
            "mix_title": "Set one",
            "dj_name": "Dj A",
            "genre": "techno",
            "track_id": "t2",
            "starting_time": "5.5",
            "track_name": "Two",
            "artist_name": "Y",
            "play_type": "sequential",
        },
        {
            "mix_id": "m1",
            "mix_title": "Set one",
            "dj_name": "Dj A",
            "genre": "techno",
            "track_id": "t3",
            "starting_time": "",
            "track_name": "Three",
            "artist_name": "Z",
            "play_type": "simultaneous",
            "overlay_parent": "t2",
        },
        {
            "mix_id": "m2",
            "mix_title": "Set two",
            "dj_name": "Dj B",
            "genre": "house",
            "track_id": "t9",
            "starting_time": "1.0",
            "track_name": "Nine",
            "artist_name": "W",
            "play_type": "sequential",
        },
    ]
    _tracklist(tmp_path / "tracklist.csv", rows)
    return tmp_path


def test_audio_file_by_id_ignores_part_files(tmp_path):
    _corpus(tmp_path)
    assert tracklists.audio_file(tmp_path / "mixes_tmp", "m1").name == "m1.mp3"
    assert tracklists.audio_file(tmp_path / "mixes_tmp", "m2") is None


def test_listed_keeps_file_order_and_parses_minutes(tmp_path):
    _corpus(tmp_path)
    by_mix = tracklists.listed(tmp_path / "tracklist.csv")
    m1 = by_mix["m1"]
    assert [r["order_listed"] for r in m1] == [1, 2, 3]
    assert m1[1]["listed_min"] == 5.5
    assert m1[2]["listed_min"] is None
    assert m1[2]["play_type"] == "simultaneous" and m1[2]["overlay_parent"] == "t2"
    assert m1[0]["title"] == "X - One"


def test_mixes_on_disk_needs_mix_audio_and_reports_missing_tracks(tmp_path):
    _corpus(tmp_path)
    mixes = tracklists.mixes_on_disk(tmp_path, tmp_path / "tracklist.csv")
    assert [m["mix_id"] for m in mixes] == ["m1"]  # m2 has no mix audio
    m1 = mixes[0]
    assert [t["track_id"] for t in m1["tracks"]] == ["t1", "t2"]
    assert m1["missing"] == ["t3"]
    assert m1["dj"] == "Dj A" and m1["genre"] == "techno"
    assert tracklists.mixes_on_disk(tmp_path, tmp_path / "tracklist.csv", only=["m2"]) == []


def test_controls_come_from_other_mixes_and_are_stable_per_mix(tmp_path):
    _corpus(tmp_path)
    mixes = tracklists.mixes_on_disk(tmp_path, tmp_path / "tracklist.csv")
    pool = tracklists.track_pool(mixes) + [
        {"track_id": "t9", "path": tmp_path / "tracks" / "t9.m4a"},
        {"track_id": "t8", "path": tmp_path / "tracks" / "t8.m4a"},
    ]
    first = tracklists.controls_for(mixes[0], pool, 3)
    again = tracklists.controls_for(mixes[0], pool, 3)
    ids = [c["track_id"] for c in first]
    assert ids == [c["track_id"] for c in again]
    assert set(ids) == {"t9", "t8"}  # never the mix's own records, and no more than the pool has


def test_other_dj_pool_takes_other_djs_records_never_one_the_own_dj_plays(tmp_path):
    _corpus(tmp_path)
    (tmp_path / "tracks" / "t8.m4a").write_bytes(b"x")
    rows = list(csv.DictReader(open(tmp_path / "tracklist.csv")))
    shared = dict(rows[3], mix_id="m3", track_id="t1")  # Dj B also plays Dj A's t1
    extra = dict(rows[3], mix_id="m3", track_id="t8")
    no_audio = dict(rows[3], mix_id="m3", track_id="t7")
    _tracklist(tmp_path / "tracklist.csv", rows + [shared, extra, no_audio])
    pool = tracklists.other_dj_pool(tmp_path, tmp_path / "tracklist.csv", ["Dj B"], {"Dj A"})
    assert [c["track_id"] for c in pool] == ["t8", "t9"]


def test_excluded_mixes_are_read_from_the_list(tmp_path):
    with open(tmp_path / "excluded.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["mix_id", "reason"])
        w.writeheader()
        w.writerow({"mix_id": "m1", "reason": "radio show"})
    assert tracklists.excluded_mixes(tmp_path / "excluded.csv") == {"m1"}
    assert tracklists.excluded_mixes(None) == set()
