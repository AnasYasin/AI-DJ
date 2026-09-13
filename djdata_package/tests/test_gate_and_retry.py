import json
import threading
import time

from djdata.fetch.yt import BLOCK_MARKERS, is_block, uses_cookies
from djdata.gate import Gate
from djdata.state import State


def test_is_block_reads_warnings_not_only_the_error():
    # 2026-09-13: the 403 came as a yt-dlp WARNING, the ERROR said "Requested format is not available"
    assert is_block("Requested format is not available", ["Unable to download API page: HTTP Error 403: Forbidden"]) == "HTTP Error 403"
    assert is_block("Sign in to confirm you're not a bot", []) == "not a bot"
    assert is_block("Video unavailable", []) is None
    assert is_block("Sign in to confirm your age. This video may be inappropriate for some users.", []) is None
    assert is_block("Requested format is not available", ["some other warning"]) is None
    assert all(m in BLOCK_MARKERS for m in ("HTTP Error 403", "HTTP Error 429"))


def test_cookies_go_to_youtube_only():
    class Cfg:
        download = {"cookies_file": "/tmp/c.txt"}

    assert uses_cookies(Cfg, "https://www.youtube.com/watch?v=abc")
    assert uses_cookies(Cfg, "https://youtu.be/abc")
    assert not uses_cookies(Cfg, "https://soundcloud.com/x/y")
    assert not uses_cookies(Cfg, "https://www.mixcloud.com/x/y/")
    Cfg.download = {"cookies_file": None}
    assert not uses_cookies(Cfg, "https://www.youtube.com/watch?v=abc")


def test_gate_paces_downloads_across_workers():
    now = [100.0]
    slept = []

    def clock():
        return now[0]

    def sleep(s):
        slept.append(s)
        now[0] += s

    g = Gate(20, 900, lambda: True, threading.Event(), clock=clock, sleep=sleep)
    g.pace()                       # first start: no wait
    g.pace()                       # second: must wait the full interval
    g.pace()
    assert slept == [20.0, 20.0]
    now[0] += 50                   # idle for a while: the next start is immediate
    g.pace()
    assert slept == [20.0, 20.0]


def test_gate_closes_on_block_and_reopens_when_probe_passes(tmp_path):
    answers = [False, False, True]   # probe at report time, first recovery probe, second recovery probe
    probed = []

    def probe():
        r = answers.pop(0)
        probed.append(r)
        return r

    stop = threading.Event()
    status = tmp_path / "gate.json"
    g = Gate(0, 0.01, probe, stop, status_path=status)
    assert g.is_open() and json.loads(status.read_text())["state"] == "open"
    assert g.blocked("HTTP Error 403") is True
    assert g.blocked("HTTP Error 403") is True   # second report during the same block: no extra probe
    assert g.blocks == 1 and probed == [False]
    assert json.loads(status.read_text())["state"] == "blocked"
    g.wait_open()
    assert g.is_open() and probed == [False, False, True]
    row = json.loads(status.read_text())
    assert row["state"] == "open" and row["blocks"] == 1 and row["reason"] == "probe passed"
    stop.set()


def test_gate_stays_open_when_only_one_item_is_refused():
    g = Gate(0, 0.01, lambda: True, threading.Event())
    assert g.blocked("HTTP Error 403") is False    # probe passed: the item is the problem
    assert g.is_open() and g.blocks == 0


def test_gate_rotates_the_ip_instead_of_waiting_out_a_block(tmp_path):
    """A blocked address was measured not to recover, so a new one is taken before any wait."""
    probes = [False, True]          # at report time, then after the first rotation
    rotated = []

    def rotate():
        rotated.append(f"1.2.3.{len(rotated)}")
        return rotated[-1]

    stop = threading.Event()
    status = tmp_path / "gate.json"
    g = Gate(0, 300, lambda: probes.pop(0), stop, status_path=status, rotate=rotate, max_rotations=3, settle_s=0.01)
    assert g.blocked("HTTP Error 403") is True
    g.wait_open()                    # would hang for 300 s if it waited instead of rotating
    assert g.is_open() and rotated == ["1.2.3.0"] and g.rotations == 1
    assert json.loads(status.read_text())["rotations"] == 1
    stop.set()


def test_gate_stops_rotating_at_the_cap():
    """A fault that looks like a block must not burn addresses all night."""
    rotated = []
    stop = threading.Event()
    g = Gate(0, 0.01, lambda: False, stop, rotate=lambda: rotated.append(1), max_rotations=2, settle_s=0.01)
    g.blocked("HTTP Error 403")
    for _ in range(200):
        if len(rotated) >= 2 and g.rotations >= 2:
            break
        time.sleep(0.01)
    time.sleep(0.1)
    stop.set()
    assert len(rotated) == 2         # capped: the gate keeps probing but takes no more addresses


def test_retry_tracks_requeues_tracks_and_their_seams(tmp_path):
    st = State(tmp_path / "s.sqlite")
    st.add_mix("m1", "2019 - DJ X @ Club", "https://soundcloud.com/x/y", "soundcloud", 2019, ["Techno"])
    for t in ("ta", "tb", "tc", "td"):
        st.add_track(t, f"{t} - title", f"https://youtu.be/{t}", 300)
    st.add_seam("m1_ta_tb", "m1", "ta", "tb", "tier1", "raveform", {"window": [0, 100]})
    st.add_seam("m1_tb_tc", "m1", "tb", "tc", "tier1", "raveform", {"window": [100, 200]})
    st.add_seam("m1_tc_td", "m1", "tc", "td", "tier1", "raveform", {"window": [200, 300]})
    st.set_mix("m1", "windows_ready")
    st.set_seam("m1_ta_tb", "ready", window_path="w1.mp3")
    st.set_seam("m1_tb_tc", "ready", window_path="w2.mp3")
    # tb blocked (403), tc genuinely gone, ta and td fine
    st.set_track("ta", "ready", path="ta.m4a")
    st.set_track("td", "ready", path="td.m4a")
    st.set_track("tb", "failed", error="HTTP Error 403: Requested format is not available")
    st.set_track("tc", "failed", error="ERROR: [youtube] tc: Video unavailable")
    st.set_seam("m1_ta_tb", "failed", error="track: HTTP Error 403: Requested format is not available")
    st.set_seam("m1_tb_tc", "failed", error="track: HTTP Error 403: Requested format is not available")
    st.set_seam("m1_tc_td", "failed", error="track: ERROR: [youtube] tc: Video unavailable")

    assert st.retry_tracks("HTTP Error 403") == {"tracks": 1, "seams": 1}
    rows = {r["track_id"]: r["status"] for r in st._conn().execute("SELECT track_id, status FROM tracks")}
    assert rows == {"ta": "ready", "tb": "pending", "tc": "failed", "td": "ready"}
    seams = {r["seam_id"]: r["status"] for r in st._conn().execute("SELECT seam_id, status FROM seams")}
    assert seams["m1_ta_tb"] == "ready"          # window exists, both tracks retryable
    assert seams["m1_tb_tc"] == "failed"         # tc is a real failure, so the seam stays failed
    assert seams["m1_tc_td"] == "failed"
    assert st.claim_track(["tier1"], "w")["track_id"] == "tb"
    summary = st.failure_summary(["tier1"])
    assert summary["tracks_failed"] == {"ERROR: [youtube] Video unavailable": 1}
    assert summary["tracks_waiting"] == {}
    assert sum(summary["seams_failed"].values()) == 2 and len(summary["seams_failed"]) == 2   # a 403 text and a "Video unavailable" text
