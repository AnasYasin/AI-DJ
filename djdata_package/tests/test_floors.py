"""The floor is a measurement of the noise, and every row carries the evidence for it."""

import math

from djdata.seam import floors


def test_floor_is_twice_the_loudest_control_and_trusted_with_three():
    f = floors.from_controls([43, 47, 58])
    assert f.floor == math.ceil(2.0 * 58) == 116
    assert f.control_max == 58
    assert f.control_n == 3
    assert f.trusted
    assert f.clears(116) and not f.clears(115)


def test_fewer_than_three_controls_is_usable_but_untrusted():
    f = floors.from_controls([13])
    assert f.floor == 26
    assert not f.trusted
    assert "1 controls" in f.reason


def test_no_control_gives_the_absolute_minimum_and_says_so():
    f = floors.from_controls([])
    assert f.floor == floors.ABSOLUTE_MIN
    assert not f.trusted
    assert f.reason == "no control ran"


def test_absolute_minimum_holds_when_controls_are_silent():
    assert floors.from_controls([0, 1, 1]).floor == floors.ABSOLUTE_MIN


def test_separation_is_votes_over_floor():
    f = floors.from_controls([10, 12, 14])  # floor 28
    assert floors.separation(900, f) == round(900 / 28, 2)
    assert floors.separation([5, 30, 12], f) == round(30 / 28, 2)
    assert floors.separation(27, f) < 1.0


def test_columns_carry_the_evidence():
    cols = floors.from_controls([3, 4, 5]).as_columns(prefix="low_")
    assert cols["low_floor"] == 10
    assert cols["low_control_votes"] == "3 4 5"
    assert cols["low_trusted"] is True
