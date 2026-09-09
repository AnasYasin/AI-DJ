"""
Key rules for the planner: no veto, a per-genre soft term, and a clash cap.

Real DJs treat key as a weak preference. Over 43,073 consecutive pairs from real
sets, 14 % were the same key, 36 % within two Camelot steps and 50 % further
apart (a clash). The planner used to veto anything beyond two steps, which made
62 % of its joins the same key and every set harmonically flat. The learned edge
model already sees the Camelot distance, so with the veto gone the shares move
most of the way to the DJ numbers on their own; KEY_WEIGHT nudges each genre the
rest of the way, and the cap keeps a short set from being clashes end to end.
"""

import numpy as np

from src.audio.key_shift import CLASH_DISTANCE as CLASH_STEPS  # one definition of "clash"

# Share of clashing joins in real sets, per genre (measured 2026-09-08).
REAL_CLASH_SHARE = {
    "melodic house": 0.42,
    "trance": 0.43,
    "techno": 0.53,
    "tech house": 0.55,
    "drum and base": 0.54,
    "afro house": 0.51,
}

# Soft term per extra step beyond CLASH_STEPS, added to the beam score. Positive
# discourages clashes, negative encourages them. Swept per genre over
# -0.8 … +0.8 (scripts/diag/key_weight_sweep.py, 3 curves x 10 tracks): at 0 the
# planned clash share was 22-41 %, any negative weight drives every genre to the
# cap (4 of 9 joins, 44 %), positive weights fall to 0-15 %. -0.2 is the smallest
# nudge that reaches the cap; melodic house and trance then sit within 2 points
# of the real share, the other four genres 7-11 points under because the cap
# binds. Raising the cap for those genres is a by-ear decision left open.
KEY_WEIGHT: dict[str, float] = {genre: -0.2 for genre in REAL_CLASH_SHARE}


def is_clash(cam_dist) -> np.ndarray:
    return np.asarray(cam_dist) > CLASH_STEPS


def key_term(genre: str, cam_dist) -> np.ndarray:
    """Score adjustment for a candidate's key distance to the record before it."""
    return -KEY_WEIGHT.get(genre, 0.0) * np.maximum(
        np.asarray(cam_dist, dtype=float) - CLASH_STEPS, 0.0
    )


def max_clashes(n_joins: int) -> int:
    return n_joins // 2


def clash_allowed(distances_so_far: list[float], n_joins: int) -> bool:
    """May the NEXT join be a clash, given the joins already in the set? Never two in a row, never past the cap."""
    clashes = [bool(c) for c in is_clash(distances_so_far)]
    return not (clashes and clashes[-1]) and sum(clashes) < max_clashes(n_joins)


def sequence_allowed(distances: list[float], n_joins: int) -> bool:
    """Does a whole sequence of joins respect the cap and the no-two-in-a-row rule?"""
    clashes = [bool(c) for c in is_clash(distances)]
    return sum(clashes) <= max_clashes(n_joins) and not any(
        a and b for a, b in zip(clashes, clashes[1:])
    )
