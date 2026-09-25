"""Which record follows which, from what was heard. One row per consecutive pair, nothing dropped.

The rule (agreed 2026-09-25). Records are put in audio order by when they were first heard. A seam is
each found record and the next one heard after it. B's first-heard must sit before A's last-heard, or
within `MAX_GAP_S` after it for a cut. If the tracklist order and the audio order disagree, the audio
order wins and the row says so. If a listed record between them was not located, the seam still forms
and the row is flagged so the gap is visible. A pair too far apart is still a row, marked not usable,
with the reason.

Time zero is not used to pair records. It is used to bound the window: B cannot sound before its own
time zero and A cannot sound after its own end, so a window from `B_start` to `A_end` holds the whole
transition by construction. Each bound is also kept within `REACH_S` of the record's first or last
heard so a deep cue does not make a ten minute window, and `PAD_S` is added either side.

Overlay records (play_type simultaneous, a "w/" on the tracklist) are not links in the chain. They are
reported as third records on the seams whose windows they fall in.
"""

from dataclasses import asdict, dataclass

from ..store.tables import num

MAX_GAP_S = 120.0  # B first heard this long after A last heard is still a seam (a cut, a break)
REACH_S = (
    120.0  # a window bound may run this far before first heard or after last heard. The presence
)
# sweep loses a record early once the other one dominates the peaks (measured on a synthetic seam,
# 2026-09-25: last heard 80 s before the record stopped), so the reach is generous and the measure
# stage's anchored per-band exit is the precise number. The cut audit catches a window that still ends
# too soon.
PAD_S = 25.0  # added either side of the bounded span


@dataclass
class Pair:
    seam_id: str
    mix_id: str
    dj: str
    genre: str
    track_a: str
    track_b: str
    order_a: int
    order_b: int
    order_ok: int
    listed_between: int
    unlocated_between: int
    gap_s: float
    usable: int
    reason: str
    a_confidence: str
    b_confidence: str
    a_time_zero_s: float
    a_rate: float
    a_first_heard_s: float
    a_last_heard_s: float
    a_end_s: float
    b_time_zero_s: float
    b_rate: float
    b_first_heard_s: float
    b_last_heard_s: float
    third_records: str
    window_t0: float
    window_t1: float
    window_s: float

    def as_row(self) -> dict:
        return asdict(self)


def _heard(r: dict) -> bool:
    return r["found"] in (1, "1", True) and num(r["first_heard_s"]) is not None


def record_end_s(r: dict) -> float:
    return num(r["time_zero_s"]) + num(r["track_len_s"]) / num(r["rate"])


def window_for(a: dict, b: dict, mix_len_s: float) -> tuple[float, float]:
    """The bounded window in mix seconds. Start: B cannot sound before its time zero, and no earlier
    than REACH_S before it was first heard. End: no later than REACH_S after A was last heard, and no
    later than A's own end unless A was heard past it (a loop). For a cut with a gap the span runs
    from A's last heard to B's first heard. PAD_S either side, clipped to the mix."""
    b_first, a_last = num(b["first_heard_s"]), num(a["last_heard_s"])
    start = max(min(num(b["time_zero_s"]), b_first), b_first - REACH_S)
    end = min(max(record_end_s(a), a_last), a_last + REACH_S)
    t0 = min(start, a_last) - PAD_S
    t1 = max(end, b_first) + PAD_S
    return max(0.0, t0), min(mix_len_s, t1)


def pairs_for_mix(rows: list[dict], mix_len_s: float) -> list[Pair]:
    """`rows` are one mix's plays rows (controls excluded). Returns every consecutive pair in audio
    order among the records that were heard."""
    own = [r for r in rows if r["is_control"] not in (1, "1", True)]
    heard = sorted(
        (r for r in own if _heard(r) and r["play_type"] != "simultaneous"),
        key=lambda r: num(r["first_heard_s"]),
    )
    overlays = [r for r in own if _heard(r) and r["play_type"] == "simultaneous"]
    by_order = {num(r["order_listed"]): r for r in own}
    out = []
    for a, b in zip(heard, heard[1:]):
        oa, ob = num(a["order_listed"]), num(b["order_listed"])
        lo, hi = min(oa, ob), max(oa, ob)
        between = [by_order[o] for o in range(lo + 1, hi) if o in by_order]
        unlocated = [r for r in between if not _heard(r)]
        gap = round(num(b["first_heard_s"]) - num(a["last_heard_s"]), 1)
        usable, reason = 1, ""
        if gap > MAX_GAP_S:
            usable, reason = 0, f"gap {gap:.0f} s between A last heard and B first heard"
        t0, t1 = window_for(a, b, mix_len_s)
        others = [
            r
            for r in heard + overlays
            if r is not a
            and r is not b
            and num(r["first_heard_s"]) < t1
            and num(r["last_heard_s"]) > t0
        ]
        out.append(
            Pair(
                seam_id=f"{a['mix_id']}_{a['track_id']}_{b['track_id']}",
                mix_id=a["mix_id"],
                dj=a["dj"],
                genre=a["genre"],
                track_a=a["track_id"],
                track_b=b["track_id"],
                order_a=oa,
                order_b=ob,
                order_ok=int(ob == oa + 1),
                listed_between=len(between),
                unlocated_between=len(unlocated),
                gap_s=gap,
                usable=usable,
                reason=reason,
                a_confidence=a["confidence"],
                b_confidence=b["confidence"],
                a_time_zero_s=num(a["time_zero_s"]),
                a_rate=num(a["rate"]),
                a_first_heard_s=num(a["first_heard_s"]),
                a_last_heard_s=num(a["last_heard_s"]),
                a_end_s=round(record_end_s(a), 1),
                b_time_zero_s=num(b["time_zero_s"]),
                b_rate=num(b["rate"]),
                b_first_heard_s=num(b["first_heard_s"]),
                b_last_heard_s=num(b["last_heard_s"]),
                third_records=" ".join(r["track_id"] for r in others),
                window_t0=round(t0, 1),
                window_t1=round(t1, 1),
                window_s=round(t1 - t0, 1),
            )
        )
    return out
