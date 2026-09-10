"""Camelot distance between key names."""

from src.audio.key_shift import camelot_distance


def test_camelot_distance_counts_steps_the_short_way_and_half_for_mode():
    assert camelot_distance("C", "C") == 0
    assert camelot_distance("C", "G") == 1  # 8B → 9B
    assert camelot_distance("Cm", "F#m") == 6  # 5A → 11A, opposite sides
    assert camelot_distance("C", "Am") == 0.5  # relative keys share a position
    assert camelot_distance("C", "Cm") == 3.5  # 8B vs 5A
    assert camelot_distance("?", "C") == 2.5  # unknown key is neutral
