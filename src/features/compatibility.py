"""
How well two tracks sit on top of each other, and for how long.

A long overlap is a property of the PAIR, not of the transition type. Two records
that share key, density and tempo can ride together for a minute; two that do not
turn to mud in fifteen seconds.

Weights and thresholds are calibrated against 43,073 real consecutive pairs from
the scraped sets. The score separates real pairs from random same-genre pairs at
AUC 0.643, and the tier thresholds are the 40th, 75th and 90th percentiles of the
real-pair distribution.

This module takes DISTANCES rather than tracks, so the planner (which holds
catalog arrays) and the mixer (which measures the played windows) can share one
definition without sharing a representation. It is deliberately numpy-only, so
importing it costs nothing.

Onset strength is deliberately absent: dropping it scored very slightly better,
and it is the one feature whose scale differs between the catalog and the mixer.
"""

import numpy as np

COMPAT_WEIGHTS = {"key": 0.35, "energy": 0.30, "loudness": 0.15, "bpm": 0.20}

# Each distance is clipped at the point where real DJ pairs stop caring.
COMPAT_LIMITS = {"key": 4.0, "energy": 0.15, "loudness": 5.0, "bpm": 5.0}

# score threshold → longest the two records may overlap, in seconds
COMPAT_TIERS = [
    (0.73, 90.0),  # p90 — exceptional pair, the full melt
    (0.59, 60.0),  # p75 — a genuinely long blend
    (0.38, 32.0),  # p40 — ordinary
    (0.00, 16.0),  # everything else gets out of the way quickly
]


def compatibility(cam_dist, d_energy, d_loud, d_bpm_pct):
    """
    0-1 measure of how well two records will sit on top of each other.

    Works on scalars or arrays. 1.0 means same key, same density, same loudness
    and no stretching needed. `d_bpm_pct` is |log(bpm_a / bpm_b)| × 100, so it
    measures how far apart the two were BEFORE either was stretched.
    """
    w, lim = COMPAT_WEIGHTS, COMPAT_LIMITS
    return (
        np.clip(1 - np.asarray(cam_dist) / lim["key"], 0, 1) * w["key"]
        + np.clip(1 - np.asarray(d_energy) / lim["energy"], 0, 1) * w["energy"]
        + np.clip(1 - np.asarray(d_loud) / lim["loudness"], 0, 1) * w["loudness"]
        + np.clip(1 - np.asarray(d_bpm_pct) / lim["bpm"], 0, 1) * w["bpm"]
    )


def max_overlap_seconds(compat: float) -> float:
    """Longest overlap this pair has earned."""
    for threshold, seconds in COMPAT_TIERS:
        if compat >= threshold:
            return seconds
    return COMPAT_TIERS[-1][1]
