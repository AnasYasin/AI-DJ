"""A loop: the record is present but its matched time stands still while the mix moves on.

From the free best-offset presence curve of one record (`presence.curves`, columns `<name>` votes and
`<name>_t` matched record time, hop `presence.HOP_S`). A chunk is stuck when, over the last `SPAN`
chunks, every chunk clears the vote floor, the matched time advanced less than `ADVANCE_SHARE` of the
mix's advance, and the matched times all sit within `MAX_SPREAD_S`. The result is the longest run of
stuck chunks before `before_s`, the seam's entry, so it reads the outgoing record's tail.

Ported from the labeller's rule. Anas heard it hold on Raveform. The vote floor there was the constant
15 (wrong records stayed at 14 or under on 25 seams); here the caller passes a floor from controls.
"""

import numpy as np
import pandas as pd

from .presence import HOP_S

SPAN = 5  # chunks, 12.5 s at the presence hop
ADVANCE_SHARE = 0.4  # advanced less than this share of the mix's advance over the span
MAX_SPREAD_S = 12.0
MIN_STEPS = 3  # this many stuck chunks in a row is a loop


def stuck(curve: pd.DataFrame, name: str, floor: int) -> np.ndarray:
    votes = curve[name].to_numpy()
    matched = curve[f"{name}_t"].to_numpy()
    out = np.zeros(len(curve), dtype=bool)
    for i in range(SPAN, len(curve)):
        window_votes = votes[i - SPAN : i + 1]
        window_t = matched[i - SPAN : i + 1]
        if (window_votes < floor).any():
            continue
        advanced = matched[i] - matched[i - SPAN]
        if (
            advanced < ADVANCE_SHARE * SPAN * HOP_S
            and (window_t.max() - window_t.min()) < MAX_SPREAD_S
        ):
            out[i] = True
    return out


def loop_steps(curve: pd.DataFrame, name: str, floor: int, before_s: float | None) -> int:
    """The longest run of stuck chunks before `before_s` (the whole curve when None)."""
    flags = stuck(curve, name, floor)
    if before_s is not None:
        flags &= curve.t.to_numpy() < before_s
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return int(best)


def is_loop(steps: int) -> bool:
    return steps >= MIN_STEPS
