"""The transition type of a seam: the signature its measured moves match, not a rule on its length.

Two answers per seam, from the features in `labels.py`:

    signature   what the seam reader can tell apart, trusted for training:
                  gap          more than GAP_BARS of nothing between the records (talk, an edit, a
                               record missing from the list); not a transition
                  loop         the outgoing record stood still before the change
                  tension      no bass at all for TENSION_MIN_BARS or more
                  cut          the records overlapped SLAM_MAX_BARS or less
                  centre swap  the bass swapped in the middle of the overlap (fade, melt or wave)
                  late swap    the bass swapped late, the highs handed over early (blend)
                  end swap     the bass swapped at the very end (rise or drop)
    type        the nearest of the mixer's seven recipes (`src/audio/audio_mixer.py` RECIPES) by the
                distance of the seam's moves to each recipe's measured template, with that distance
                beside it. Inside a signature group the reader cannot separate the members (fade,
                melt and wave differ by how loud a band is, and fingerprints only say whether it is
                there), so `type` is a best match and `distance` says how close.

Everything here was measured on 2026-10-09 on seven renders of one steady techno pair by the current
mixer, one per type, 16-bar overlaps (slam 4), the overlap's true start and end known from the mixer:
the bass swap read within a bar of the recipe, the incoming record on time, the outgoing one 2.3 bars
late (fixed since in `presence.last_present`). The templates are those readings as fractions of the
overlap; the group boundaries are the midpoints between the groups' swap positions (0.56, 0.78, 0.99).
SLAM_MAX_BARS: a 4-bar cut read as 7 bars, because a presence chunk half covered still clears the floor.
One pair, so these numbers are a first template, to be re-measured when more renders or ear-checked
seams exist.
"""

import math

GAP_BARS = 2.0
SLAM_MAX_BARS = 8.0
CENTRE_MAX, LATE_MAX = 0.67, 0.88  # swap position boundaries: centre | late | end
FEATURES = (
    "swap_pos",
    "b_high_pos",
    "b_mid_pos",
    "b_low_pos",
    "a_high_pos",
    "a_mid_pos",
    "a_low_pos",
)
TEMPLATES = {
    "fade": (0.56, 0.18, 0.04, 0.49, 0.95, 0.74, 0.64),
    "melt": (0.56, 0.32, 0.11, 0.49, 0.99, 0.74, 0.64),
    "wave": (0.56, 0.18, 0.00, 0.46, 0.74, 0.74, 0.64),
    "blend": (0.78, 0.18, 0.35, 0.67, 0.49, 0.67, 0.88),
    "rise": (0.99, 0.67, 0.00, 0.92, 1.03, 0.85, 1.06),
    "drop": (0.99, 0.67, 0.00, 0.92, 1.03, 0.85, 1.06),
}
MIN_FEATURES = 3  # fewer readable moves than this and no template can be matched
DROP_MAX_BARS = 1.0  # the bass swap this close to the incoming record's annotated drop start


def nearest(features: dict) -> tuple[str | None, float | None]:
    """(type, distance): the template closest to the seam's moves, RMS over the moves that were read."""
    best, best_d = None, None
    for name, template in TEMPLATES.items():
        pairs = [
            (features[f], t) for f, t in zip(FEATURES, template) if features.get(f) is not None
        ]
        if len(pairs) < MIN_FEATURES:
            continue
        d = math.sqrt(sum((x - t) ** 2 for x, t in pairs) / len(pairs))
        if best_d is None or d < best_d:
            best, best_d = name, round(d, 3)
    return best, best_d


def classify(
    features: dict,
    overlap_s: float | None,
    shorter_record_s: float | None,
    swap_to_drop_bars: float | None,
) -> dict:
    """`features` is the labels row as numbers. Returns {"signature", "type", "distance"}."""
    none = {"type": None, "distance": None}
    if not features.get("measured") or features.get("overlap_bars") is None:
        return {"signature": "unmeasured", **none}
    if overlap_s is not None and shorter_record_s is not None and overlap_s > shorter_record_s:
        return {"signature": "unusable", **none}
    ov = features["overlap_bars"]
    if ov < -GAP_BARS:
        return {"signature": "gap", **none}
    if features.get("loop"):
        return {"signature": "loop", **none}
    if features.get("tension"):
        return {"signature": "tension", **none}
    if ov <= SLAM_MAX_BARS:
        return {"signature": "cut", "type": "slam", "distance": 0.0}
    swap = features.get("swap_pos")
    if swap is None:
        signature = None
    elif swap <= CENTRE_MAX:
        signature = "centre swap"
    elif swap <= LATE_MAX:
        signature = "late swap"
    else:
        signature = "end swap"
    name, distance = nearest(features)
    if signature == "end swap" and swap_to_drop_bars is not None:
        name = "drop" if swap_to_drop_bars <= DROP_MAX_BARS else "rise"
    elif name == "drop":
        name = "rise"  # without the drop position the two cannot be told apart; rise is the safer name
    return {"signature": signature, "type": name, "distance": distance}


def swap_to_drop_bars(
    file_t0, b_time_zero_s, b_rate, bass_swap_s, bar_s, drop_starts
) -> float | None:
    """Bars from the bass swap to the nearest drop start of the incoming record, in its own clock.
    `bass_swap_s` is window-file time and `file_t0` the mix time the file starts at (what measure
    used); the record's time at mix time T is (T - time zero) x rate."""
    if not drop_starts or None in (file_t0, b_time_zero_s, b_rate, bass_swap_s, bar_s):
        return None
    at = (file_t0 + bass_swap_s - b_time_zero_s) * b_rate
    return round(min(abs(at - d) for d in drop_starts) / bar_s, 2)
