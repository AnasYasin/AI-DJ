"""Which audio file is which show. Settled on play order, never on vote count alone.

Why order (2026-09-21, Fred's tour). He plays the same USB every night, so a record from one show's
tracklist really is in another show's audio, and every test record cleared the floor in every segment.
A record that repeats elsewhere on the tour lands at an unrelated time, so a wrong file scatters, while
the right one finds the records in the listed order.

Per list and file: the records found (the locate stage's own floor from three controls by other DJs),
and the longest chain of them whose mix times rise with the listed order (`in_order`). A chain can
also rise by chance, and more so the more records a wrong file happens to hold: twenty scattered hits
give a chain of about nine. So the chance is measured on the same rows, never assumed: the found
records' times are shuffled `SHUFFLES` times and `p_order` is the share of shuffles whose chain is as
long as the real one. A file proves a list when the chain is at least `MIN_IN_ORDER` long and
`p_order` is at most `MAX_P_ORDER`.

The locate itself is `locate.locate_mix`, unchanged. This module only reads its rows.
"""

from bisect import bisect_left
import random

MIN_IN_ORDER = 3  # the rule agreed 2026-09-25: three or more records found in listed order
MAX_P_ORDER = 0.01  # at most one shuffle in a hundred may give a chain this long
SHUFFLES = 2000


def longest_rising(times: list[float]) -> list[int]:
    """Indices of the longest strictly rising run of `times`, taken in list order (patience sort)."""
    tails, tail_idx, prev = [], [], [-1] * len(times)
    for i, t in enumerate(times):
        k = bisect_left(tails, t)
        if k == len(tails):
            tails.append(t)
            tail_idx.append(i)
        else:
            tails[k] = t
            tail_idx[k] = i
        prev[i] = tail_idx[k - 1] if k else -1
    out, i = [], tail_idx[-1] if tail_idx else -1
    while i != -1:
        out.append(i)
        i = prev[i]
    return out[::-1]


def order_chance(times: list[float], observed: int, seed: str, shuffles: int = SHUFFLES) -> tuple:
    """(median chain length under shuffling, share of shuffles reaching `observed`)."""
    if not times:
        return 0, 1.0
    rng = random.Random(seed)
    work = list(times)
    lengths = []
    for _ in range(shuffles):
        rng.shuffle(work)
        lengths.append(len(longest_rising(work)))
    lengths.sort()
    reach = sum(1 for n in lengths if n >= observed)
    return lengths[len(lengths) // 2], reach / shuffles


def summarise(list_id: str, file_id: str, rows: list[dict], listed: int) -> dict:
    """One row per list and file from that pair's located records. `rows` carry order_listed,
    time_zero_s, found, confidence, floor, control_max; `listed` counts the list's records, on disk
    or not."""
    found = sorted((r for r in rows if r["found"]), key=lambda r: r["order_listed"])
    times = [r["time_zero_s"] for r in found]
    chain = [found[i] for i in longest_rising(times)]
    chance, p = order_chance(times, len(chain), seed=f"{list_id}|{file_id}")
    proved = len(chain) >= MIN_IN_ORDER and p <= MAX_P_ORDER
    first = rows[0] if rows else {}
    # where a list carries the time each record is heard inside this file (the USB002 app does), how
    # far the found record's first heard sits from it: a report column, not a gate
    offsets = sorted(
        abs(r["first_heard_s"] - 60 * r["listed_min"])
        for r in found
        if r.get("listed_min") is not None and r.get("first_heard_s") is not None
    )
    return {
        "list_id": list_id,
        "file_id": file_id,
        "listed": listed,
        "on_disk": len(rows),
        "found": len(found),
        "confident": sum(1 for r in found if r["confidence"] == "confident"),
        "in_order": len(chain),
        "chance_in_order": chance,
        "p_order": round(p, 4),
        "proved": int(proved),
        "chain_orders": " ".join(str(r["order_listed"]) for r in chain),
        "chain_from_min": round(chain[0]["time_zero_s"] / 60, 2) if chain else None,
        "chain_to_min": round(chain[-1]["time_zero_s"] / 60, 2) if chain else None,
        "time_checked": len(offsets),
        "time_median_off_s": round(offsets[len(offsets) // 2], 1) if offsets else None,
        "time_within_60s": sum(1 for o in offsets if o <= 60) if offsets else None,
        "floor": first.get("floor"),
        "control_max": first.get("control_max"),
    }
