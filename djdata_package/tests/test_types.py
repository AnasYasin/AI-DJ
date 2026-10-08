"""The signature and nearest recipe as written in seam/types.py."""

import pytest

from djdata.seam import types

BAR = 4 * 60.0 / 128.0


def feats(**over):
    base = {
        "measured": 1,
        "overlap_bars": 16.0,
        "tension": 0,
        "loop": 0,
        "swap_pos": 0.56,
        "b_high_pos": 0.18,
        "b_mid_pos": 0.04,
        "b_low_pos": 0.49,
        "a_high_pos": 0.95,
        "a_mid_pos": 0.74,
        "a_low_pos": 0.64,
    }
    base.update(over)
    return base


def got(f=None, overlap_s=None, shorter=None, drop=None):
    return types.classify(f or feats(), overlap_s, shorter, drop)


def test_signatures_in_order():
    assert got(feats(overlap_bars=-3))["signature"] == "gap"
    assert got(feats(loop=1, tension=1))["signature"] == "loop"
    assert got(feats(tension=1))["signature"] == "tension"
    assert got(feats(overlap_bars=6))["signature"] == "cut"
    assert got()["signature"] == "centre swap"
    assert got(feats(swap_pos=0.78))["signature"] == "late swap"
    assert got(feats(swap_pos=0.99))["signature"] == "end swap"
    assert got(feats(swap_pos=None))["signature"] is None


def test_nearest_recipe_and_distance():
    r = got()
    assert r["type"] == "fade" and r["distance"] == 0.0
    r = got(dict(feats(), **dict(zip(types.FEATURES, types.TEMPLATES["blend"]))))
    assert r["type"] == "blend" and r["distance"] == 0.0
    r = got(dict(feats(), **dict(zip(types.FEATURES, types.TEMPLATES["rise"]))))
    assert r["type"] == "rise"  # drop needs the drop position
    assert (
        got(dict(feats(), **dict(zip(types.FEATURES, types.TEMPLATES["rise"]))), drop=0.5)["type"]
        == "drop"
    )
    few = feats(b_high_pos=None, b_mid_pos=None, b_low_pos=None, a_high_pos=None, a_mid_pos=None)
    assert got(few)["type"] is None  # two moves are not enough to match a template


def test_unmeasured_and_unusable():
    assert got(feats(measured=0))["signature"] == "unmeasured"
    assert got(overlap_s=500.0, shorter=400.0)["signature"] == "unusable"


def test_swap_to_drop_in_the_records_own_clock():
    # record B starts at mix 100 s, played 2 % fast; swap at file 30 s of a file starting at mix 200 s
    # -> B's own time (200 + 30 - 100) x 1.02 = 132.6 s; drop at 130.725 s is 1 bar away at BAR
    assert types.swap_to_drop_bars(
        200.0, 100.0, 1.02, 30.0, BAR, [60.0, 130.725]
    ) == pytest.approx(1.0, abs=0.01)
    assert types.swap_to_drop_bars(200.0, 100.0, 1.02, None, BAR, [60.0]) is None
    assert types.swap_to_drop_bars(200.0, 100.0, 1.02, 30.0, BAR, []) is None
