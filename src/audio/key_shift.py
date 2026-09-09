"""
Camelot distance between key names. The single definition of "how far apart"
two keys are, shared by the mixer's compatibility measure and the planner's
key rules.

A semitone-shift policy and an audio-based harmonic fit were built and tried
on 2026-09-09 and removed: no detector reads mode reliably on electronic music
(essentia 14/21, two CNNs 8/21 against published keys), no audio fit measure
was stable across segments, and a ±1 semitone shift of the incoming record was
inaudible to Anas on the test pair. Key stays a planner preference and a report
field; the mixer does not act on it.
"""

from src.features.transition_labeler import _CAMELOT

UNKNOWN_KEY_DISTANCE = 2.5  # neutral: neither a match nor a clash
CLASH_DISTANCE = 2.0  # beyond two steps a join is a clash


def camelot_distance(key_a: str, key_b: str) -> float:
    """Steps round the wheel the short way, plus 0.5 when the modes differ."""
    a, b = _CAMELOT.get(key_a), _CAMELOT.get(key_b)
    if a is None or b is None:
        return UNKNOWN_KEY_DISTANCE
    d = abs(a - b)
    return min(d, 12 - d) + (0.5 if key_a.endswith("m") != key_b.endswith("m") else 0.0)
