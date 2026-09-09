"""Planner key rules: soft term, clash cap, no two clashes in a row."""

import numpy as np

from src.models import key_rules
from src.models.key_rules import clash_allowed, is_clash, key_term, max_clashes, sequence_allowed


def test_clash_is_beyond_two_steps():
    assert list(is_clash([0, 1, 2, 2.5, 3, 6])) == [False, False, False, True, True, True]


def test_key_term_only_acts_beyond_two_steps_and_follows_the_genre_weight(monkeypatch):
    monkeypatch.setitem(key_rules.KEY_WEIGHT, "techno", 0.5)
    np.testing.assert_allclose(key_term("techno", [0, 2, 3, 6]), [0, 0, -0.5, -2.0])
    np.testing.assert_allclose(key_term("no such genre", [6]), [0.0])
    monkeypatch.setitem(key_rules.KEY_WEIGHT, "afro house", -0.3)  # a bonus where DJs clash more
    np.testing.assert_allclose(key_term("afro house", [4]), [0.6])


def test_cap_is_half_the_joins_and_never_two_in_a_row():
    assert max_clashes(9) == 4 and max_clashes(3) == 1
    assert clash_allowed([], 9)
    assert not clash_allowed([5.0], 9)  # the join before was a clash
    assert clash_allowed([5.0, 1.0], 9)
    assert not clash_allowed([5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0], 9)  # 4 of 4 used
    assert sequence_allowed([5, 1, 5, 1], 4) is True  # 2 of 2 allowed
    assert sequence_allowed([5, 1, 5, 1, 5], 5) is False  # 3 clashes, cap is 2
    assert sequence_allowed([5, 5, 1, 1], 5) is False  # two in a row
