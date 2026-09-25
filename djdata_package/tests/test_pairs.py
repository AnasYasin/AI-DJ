"""Pairs follow the audio, flag the tracklist, and never drop a row silently."""

from djdata.seam import pairs


def play(
    mix_id,
    track_id,
    order,
    first,
    last,
    zero=None,
    rate=1.0,
    length=300.0,
    found=1,
    play_type="sequential",
    confidence="confident",
):
    return {
        "mix_id": mix_id,
        "dj": "Dj",
        "genre": "techno",
        "track_id": track_id,
        "is_control": 0,
        "order_listed": order,
        "play_type": play_type,
        "found": found,
        "confidence": confidence,
        "time_zero_s": first if zero is None else zero,
        "rate": rate,
        "track_len_s": length,
        "first_heard_s": first if found else "",
        "last_heard_s": last if found else "",
    }


def test_consecutive_overlapping_records_make_one_usable_seam():
    rows = [play("m", "a", 1, 0.0, 300.0), play("m", "b", 2, 280.0, 580.0)]
    got = pairs.pairs_for_mix(rows, mix_len_s=600.0)
    assert len(got) == 1
    p = got[0]
    assert p.seam_id == "m_a_b" and p.order_ok == 1 and p.usable == 1 and p.reason == ""
    assert p.gap_s == -20.0
    assert p.listed_between == 0 and p.unlocated_between == 0 and p.third_records == ""
    # window: from B's first heard (its time zero is the same here) less the pad, to A's end plus pad
    assert p.window_t0 == 280.0 - pairs.PAD_S
    assert p.window_t1 == 300.0 + pairs.PAD_S
    assert p.window_s == p.window_t1 - p.window_t0


def test_audio_order_wins_over_the_tracklist_and_says_so():
    rows = [
        play("m", "a", 1, 0.0, 300.0),
        play("m", "c", 3, 280.0, 580.0),
        play("m", "b", 2, 560.0, 860.0),
    ]
    got = pairs.pairs_for_mix(rows, mix_len_s=900.0)
    assert [(p.track_a, p.track_b) for p in got] == [("a", "c"), ("c", "b")]
    assert got[0].order_ok == 0 and got[0].listed_between == 1 and got[0].unlocated_between == 0
    assert got[1].order_ok == 0


def test_unlocated_record_between_is_flagged_not_dropped():
    rows = [
        play("m", "a", 1, 0.0, 300.0),
        play("m", "x", 2, 0, 0, found=0, confidence="not found"),
        play("m", "b", 3, 280.0, 580.0),
    ]
    got = pairs.pairs_for_mix(rows, mix_len_s=600.0)
    assert len(got) == 1
    assert got[0].order_ok == 0 and got[0].listed_between == 1 and got[0].unlocated_between == 1
    assert got[0].usable == 1


def test_a_long_gap_is_a_row_marked_not_usable():
    rows = [play("m", "a", 1, 0.0, 300.0), play("m", "b", 2, 1800.0, 2100.0)]
    got = pairs.pairs_for_mix(rows, mix_len_s=2400.0)
    assert len(got) == 1 and got[0].usable == 0
    assert "gap 1500 s" in got[0].reason


def test_a_cut_with_a_short_gap_spans_both_sides():
    rows = [play("m", "a", 1, 0.0, 300.0), play("m", "b", 2, 310.0, 610.0)]
    p = pairs.pairs_for_mix(rows, mix_len_s=700.0)[0]
    assert p.usable == 1 and p.gap_s == 10.0
    assert p.window_t0 == 300.0 - pairs.PAD_S
    assert p.window_t1 == 310.0 + pairs.PAD_S


def test_deep_cue_bounds_the_window_by_reach_not_by_time_zero():
    # B cued in 5 minutes deep: its time zero is 300 s before it is first heard
    rows = [play("m", "a", 1, 0.0, 300.0), play("m", "b", 2, 280.0, 580.0, zero=-20.0)]
    p = pairs.pairs_for_mix(rows, mix_len_s=600.0)[0]
    assert p.window_t0 == 280.0 - pairs.REACH_S - pairs.PAD_S
    # and A last heard long before its record ends: the end is bounded by reach too
    rows = [play("m", "a", 1, 0.0, 300.0, length=1200.0), play("m", "b", 2, 280.0, 580.0)]
    p = pairs.pairs_for_mix(rows, mix_len_s=900.0)[0]
    assert p.window_t1 == 300.0 + pairs.REACH_S + pairs.PAD_S


def test_window_is_clipped_to_the_mix():
    rows = [play("m", "a", 1, 0.0, 30.0), play("m", "b", 2, 10.0, 100.0)]
    p = pairs.pairs_for_mix(rows, mix_len_s=100.0)[0]
    assert p.window_t0 == 0.0 and p.window_t1 == 100.0


def test_third_record_and_overlay_are_reported_not_chained():
    rows = [
        play("m", "a", 1, 0.0, 300.0),
        play("m", "b", 2, 280.0, 580.0),
        play("m", "w", 3, 290.0, 320.0, play_type="simultaneous"),
        play("m", "c", 4, 560.0, 860.0),
    ]
    got = pairs.pairs_for_mix(rows, mix_len_s=900.0)
    assert [(p.track_a, p.track_b) for p in got] == [("a", "b"), ("b", "c")]
    assert got[0].third_records == "w"
    assert got[1].third_records == ""


def test_controls_and_unheard_records_are_ignored():
    rows = [
        play("m", "a", 1, 0.0, 300.0),
        play("m", "b", 2, 280.0, 580.0),
        {**play("m", "k", "", 0.0, 100.0), "is_control": 1},
    ]
    got = pairs.pairs_for_mix(rows, mix_len_s=600.0)
    assert len(got) == 1 and got[0].third_records == ""


def test_a_record_looped_past_its_end_bounds_the_window_by_last_heard():
    # A is a 150 s record heard until 400 s: looped. The window must run to where it was last heard.
    rows = [play("m", "a", 1, 0.0, 400.0, length=150.0), play("m", "b", 2, 380.0, 680.0)]
    p = pairs.pairs_for_mix(rows, mix_len_s=900.0)[0]
    assert p.window_t1 == 400.0 + pairs.PAD_S
    assert p.usable == 1
