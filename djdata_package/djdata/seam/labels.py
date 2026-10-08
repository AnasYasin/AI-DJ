"""The measured moves of a seam, in bars and as fractions of the overlap. These are the features a
model trains on; the type a DJ would name is read from them in `types.py`.

Bars come from the outgoing record's tempo at the speed it was played (`tempo.py`); with no tempo the
seam is `unmeasured`, never guessed. Fractions run from the moment the incoming record is first heard
(0) to the moment the outgoing is last heard (1), so a long and a short transition compare directly.

    overlap_bars      how long the two records were both heard
    swap_pos          where the bass swapped
    b_high_pos ...    where each band of the incoming record arrived
    a_high_pos ...    where each band of the outgoing record left
    bass_cut_bars     the longest run with no bass at all while a record with bass played
    bass_both_bars    the longest run with both basses together
    sweep_out_bars    outgoing lost low, then mid, then high: the span, else None
    sweep_in_bars     incoming gained high, then mid, then low: the span, else None
    tension           1 when the bass was held back for TENSION_MIN_BARS or more
    loop              1 when the outgoing record stood still before the change (`loops.py`)

Until 2026-10-09 this file also named nine types by length thresholds (long blend over 8 bars, and so
on). Length is no part of a type, so those are gone; the moves stay.
"""

from dataclasses import asdict, dataclass

from . import loops

SWEEP_MIN_BARS = 8.0
TENSION_MIN_BARS = 8.0
BANDS = ("high", "mid", "low")


@dataclass
class Label:
    seam_id: str
    measured: int
    bar_s: float | None
    overlap_bars: float | None
    swap_pos: float | None
    b_high_pos: float | None
    b_mid_pos: float | None
    b_low_pos: float | None
    a_high_pos: float | None
    a_mid_pos: float | None
    a_low_pos: float | None
    bass_cut_bars: float | None
    bass_both_bars: float | None
    sweep_out_bars: float | None
    sweep_in_bars: float | None
    tension: int | None
    loop: int | None
    loop_steps: int | None

    def as_row(self) -> dict:
        return asdict(self)


def _bars(seconds, bar_s):
    return None if seconds is None or bar_s is None else round(seconds / bar_s, 1)


def sweep_span(first: dict, order: tuple) -> float | None:
    """Seconds from the first to the last band in `order` when the bands went in that order, else None."""
    if any(first.get(b) is None for b in order):
        return None
    times = [first[b] for b in order]
    if times[0] < times[1] < times[2]:
        return times[2] - times[0]
    return None


def position(t, in_s, out_s) -> float | None:
    """Where a moment sits in the overlap, 0 at the incoming's entry and 1 at the outgoing's exit."""
    if t is None or in_s is None or out_s is None or out_s <= in_s:
        return None
    return round((t - in_s) / (out_s - in_s), 2)


def label(seam_id: str, m: dict, bar_s: float | None) -> Label:
    """`m` is the measures row (numbers, not strings). `bar_s` is seconds per bar for this seam."""
    steps = int(m.get("loop_steps") or 0)
    if bar_s is None or m.get("overlap_s") is None:
        return Label(seam_id, 0, bar_s, *([None] * 14), loop_steps=steps)
    in_s, out_s = m.get("in_s"), m.get("out_s")
    cut_bars = (m.get("bass_cut_longest_s") or 0.0) / bar_s
    out_span = sweep_span({b: m.get(f"{b}_out_s") for b in BANDS}, ("low", "mid", "high"))
    in_span = sweep_span({b: m.get(f"{b}_in_s") for b in BANDS}, ("high", "mid", "low"))
    return Label(
        seam_id=seam_id,
        measured=1,
        bar_s=round(bar_s, 3),
        overlap_bars=round(m["overlap_s"] / bar_s, 1),
        swap_pos=position(m.get("bass_swap_s"), in_s, out_s),
        b_high_pos=position(m.get("high_in_s"), in_s, out_s),
        b_mid_pos=position(m.get("mid_in_s"), in_s, out_s),
        b_low_pos=position(m.get("low_in_s"), in_s, out_s),
        a_high_pos=position(m.get("high_out_s"), in_s, out_s),
        a_mid_pos=position(m.get("mid_out_s"), in_s, out_s),
        a_low_pos=position(m.get("low_out_s"), in_s, out_s),
        bass_cut_bars=round(cut_bars, 1),
        bass_both_bars=round((m.get("bass_both_longest_s") or 0.0) / bar_s, 1),
        sweep_out_bars=_bars(out_span, bar_s),
        sweep_in_bars=_bars(in_span, bar_s),
        tension=int(cut_bars >= TENSION_MIN_BARS),
        loop=int(loops.is_loop(steps)),
        loop_steps=steps,
    )
