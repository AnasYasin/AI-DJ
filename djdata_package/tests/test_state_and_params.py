import json
from pathlib import Path

import numpy as np

from djdata.manifest.raveform import _coarse, _tier_of
from djdata.seam.params import crossing
from djdata.state import State


def test_state_claims_and_archivable(tmp_path):
    st = State(tmp_path / "s.sqlite")
    st.add_mix("m1", "2019 - DJ X @ Club", "https://soundcloud.com/x/y", "soundcloud", 2019, ["Techno"])
    st.add_track("ta", "A - a", "https://youtu.be/ta", 400)
    st.add_track("tb", "B - b", "https://youtu.be/tb", 380)
    st.add_seam("m1_ta_tb", "m1", "ta", "tb", "tier1", "raveform", {"window": [0, 100]})
    assert st.claim_mix(["tier1"], "w")["mix_id"] == "m1"
    assert st.claim_mix(["tier1"], "w") is None            # already claimed
    assert st.claim_track(["tier1"], "w") is None          # mix windows not ready yet
    st.set_seam("m1_ta_tb", "ready", window_path="w.mp3")
    st.set_mix("m1", "windows_ready")
    t = st.claim_track(["tier1"], "w")
    assert t["track_id"] in ("ta", "tb")
    assert st.claim_seam(["tier1"], "w") is None           # tracks not ready
    st.set_track("ta", "ready", path="ta.m4a")
    st.set_track("tb", "ready", path="tb.m4a")
    st._conn().execute("UPDATE tracks SET status='ready'")
    s = st.claim_seam(["tier1"], "w")
    assert s["seam_id"] == "m1_ta_tb" and s["a_path"] == "ta.m4a"
    assert st.archivable(["tier1"]) == {"tracks": [], "windows": []}
    st.set_seam("m1_ta_tb", "done")
    arch = st.archivable(["tier1"])
    assert {r["track_id"] for r in arch["tracks"]} == {"ta", "tb"}
    assert arch["windows"][0]["seam_id"] == "m1_ta_tb"
    st.mark_archived(["ta"], ["m1_ta_tb"])
    assert {r["track_id"] for r in st.archivable(["tier1"])["tracks"]} == {"tb"}
    assert not st.work_left(["tier1"])


def test_coarse_window_from_tracks():
    a = {"mixin_time_mix": 1000.0, "mixout_time_mix": 1300.0, "mixin_time_track": 60.0, "mixout_time_track": 360.0,
         "matched_time_track": 300.0, "matched_time_mix": 300.0, "match_rate": 0.8}
    b = {"mixin_time_mix": 1240.0, "mixout_time_mix": 1600.0, "mixin_time_track": 40.0, "mixout_time_track": 400.0,
         "matched_time_track": 360.0, "matched_time_mix": 360.0, "match_rate": 0.7}
    c = _coarse(a, b, a_dur=400.0, b_dur=420.0, pad_s=20, alone_s=45)
    assert c["overlap_start_mix_t"] == 1240.0 and c["overlap_end_mix_t"] == 1300.0
    assert c["b_track_start_mix_t"] == 1200.0            # B's first beat falls 40 s before Raveform's mix-in
    assert c["a_track_end_mix_t"] == 1340.0
    assert c["window"][0] == 1180.0                       # 20 s before B's first beat
    assert c["window"][1] == 1360.0                       # 20 s after A's last beat
    assert c["window"][0] >= a["mixin_time_mix"] and c["window"][1] <= b["mixout_time_mix"]


def test_tier_matching():
    tiers = {"tier1": {"djs": ["adam beyer"], "min_year": 2014}, "tier2": {"genres": ["Techno"], "min_year": 2016}}
    assert _tier_of("2016 - Adam Beyer @ Awakenings", ["Techno"], 2016, tiers, ["tier1", "tier2"]) == "tier1"
    assert _tier_of("2017 - Someone @ Club", ["Techno"], 2017, tiers, ["tier1", "tier2"]) == "tier2"
    assert _tier_of("2015 - Someone @ Club", ["Techno"], 2015, tiers, ["tier1", "tier2"]) is None
    assert _tier_of("2012 - Adam Beyer @ Club", ["Techno"], 2012, tiers, ["tier1", "tier2"]) is None


def test_crossing():
    x = np.array([0, 0, -3, -7, -20])
    assert crossing(x, -6, "down") == 3
    assert crossing(x[::-1], -6, "up") == 2
    assert crossing(x, -30, "down") is None


def test_download_only_work_left_ends_when_windows_and_tracks_are_on_disk(tmp_path):
    """workers.seams: 0 — the run must stop once every mix is cut and every needed track fetched,
    even though the seams stay `ready` (nothing analyses them)."""
    st = State(tmp_path / "s.sqlite")
    st.add_mix("m1", "2019 - DJ X @ Club", "https://soundcloud.com/x/y", "soundcloud", 2019, ["Techno"])
    st.add_track("ta", "A - a", "https://youtu.be/ta", 400)
    st.add_track("tb", "B - b", "https://youtu.be/tb", 380)
    st.add_track("tz", "Z - other tier", "https://youtu.be/tz", 380)
    st.add_seam("m1_ta_tb", "m1", "ta", "tb", "tier1", "raveform", {"window": [0, 100]})
    st.add_seam("m1_tb_tz", "m1", "tb", "tz", "tier2", "raveform", {"window": [100, 200]})
    assert st.download_work_left(["tier1"])                 # mix not fetched
    st.set_mix("m1", "windows_ready")
    st.set_seam("m1_ta_tb", "ready", window_path="w.mp3")
    assert st.download_work_left(["tier1"])                 # tracks not fetched
    st.set_track("ta", "ready", path="ta.m4a")
    st.set_track("tb", "failed", error="Video unavailable")
    assert not st.download_work_left(["tier1"])             # tz belongs to tier2 only
    assert st.work_left(["tier1"])                          # the analysing run would still have the seam to do
    assert st.download_work_left(["tier1", "tier2"])        # tier2 still needs tz
