"""Tests for planner rules that are easy to evade by accident."""

import pytest

from src.models.predict_model import CURVES, split_artists


@pytest.mark.parametrize(
    "credit,expected",
    [
        ("Reinier Zonneveld & Miro", {"reinier zonneveld", "miro"}),
        ("Odd Mob & OMNOM pres. HYPERBEAM", {"odd mob", "omnom", "hyperbeam"}),
        ("Eric Prydz vs. Pink Floyd", {"eric prydz", "pink floyd"}),
        ("Simon Doty feat. Forrest", {"simon doty", "forrest"}),
        ("Solardo x JOSHWA", {"solardo", "joshwa"}),
        ("Charlotte de Witte", {"charlotte de witte"}),
        ("Mind Against", {"mind against"}),
        ("", set()),
    ],
)
def test_split_artists(credit, expected):
    assert split_artists(credit) == expected


def test_a_collaboration_blocks_the_solo_artist():
    """The bug this guards: comparing whole credit strings let Reinier Zonneveld
    open and close the same three-track set."""
    assert split_artists("Reinier Zonneveld & Miro") & split_artists("Reinier Zonneveld")


def test_unrelated_artists_do_not_collide():
    assert not (split_artists("Mind Against") & split_artists("Charlotte de Witte"))


def test_trailing_period_is_not_left_behind():
    """`pres.` and `ft.` must take their period with them."""
    for credit in ("A pres. B", "A ft. B", "A feat. B", "A vs. B"):
        assert all(not n.startswith(".") for n in split_artists(credit)), credit


# ── Energy curves ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(CURVES))
def test_curves_stay_in_range(name):
    import numpy as np

    t = np.linspace(0, 1, 32)
    v = CURVES[name](t)
    assert v.min() >= -1e-9 and v.max() <= 1 + 1e-9, name


def test_chill_curve_stays_low():
    import numpy as np

    assert CURVES["chill"](np.linspace(0, 1, 32)).max() < 0.3


def test_build_curve_rises_monotonically():
    import numpy as np

    v = CURVES["build"](np.linspace(0, 1, 32))
    assert (np.diff(v) >= 0).all() and v[0] < v[-1]
