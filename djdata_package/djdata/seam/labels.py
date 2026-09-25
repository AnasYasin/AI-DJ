"""The transition type of a measured seam, by threshold rules on the measured columns.

Nine types, first match in `ORDER` wins. The rules are the labeller's as written on 2026-09-18 and
heard by Anas on Raveform; they are carried here unchanged so a DJ-corpus label means the same thing.
A seam gets every label whose rule holds (`labels`) and one main label. Bars come from the record's
tempo (`tempo.py`); with no tempo the seam is `unmeasured`, never guessed.

    edit_or_talk   the incoming record was heard more than 2 bars before the outgoing left (an edit,
                   or the fingerprint read speech as neither)
    loop           the outgoing record stood still for at least 3 chunks before the change
    tension        the mix had no bass for 8 bars or more while a record with bass was playing
    sweep_out      the outgoing record lost low, then mid, then high, over 8 bars or more (a high pass out)
    sweep_in       the incoming record gained high, then mid, then low, over 8 bars or more (a high pass in)
    cut            overlap between -2 and 1 bars
    layer          both records' bass together for 16 bars or more
    long_blend     overlap over 8 bars
    short_blend    overlap over 1 and up to 8 bars
"""

from dataclasses import asdict, dataclass

from . import loops

ORDER = [
    "loop",
    "edit_or_talk",
    "tension",
    "sweep_out",
    "sweep_in",
    "cut",
    "layer",
    "long_blend",
    "short_blend",
]
SWEEP_MIN_BARS = 8.0
TENSION_MIN_BARS = 8.0
LAYER_MIN_BARS = 16.0
CUT_MIN_BARS, CUT_MAX_BARS = -2.0, 1.0
SHORT_BLEND_MAX_BARS = 8.0


@dataclass
class Label:
    seam_id: str
    label: str
    labels: str
    bar_s: float | None
    overlap_bars: float | None
    bass_cut_bars: float | None
    bass_both_bars: float | None
    sweep_out_bars: float | None
    sweep_in_bars: float | None
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


def label(seam_id: str, m: dict, bar_s: float | None) -> Label:
    """`m` is the measures row (numbers, not strings). `bar_s` is seconds per bar for this seam."""
    if bar_s is None or m.get("overlap_s") is None:
        return Label(
            seam_id, "unmeasured", "", bar_s, None, None, None, None, None, m.get("loop_steps")
        )
    overlap = m["overlap_s"] / bar_s
    cut_bars = m.get("bass_cut_longest_s", 0.0) / bar_s
    both_bars = m.get("bass_both_longest_s", 0.0) / bar_s
    out_span = sweep_span(
        {b: m.get(f"{b}_out_s") for b in ("low", "mid", "high")}, ("low", "mid", "high")
    )
    in_span = sweep_span(
        {b: m.get(f"{b}_in_s") for b in ("high", "mid", "low")}, ("high", "mid", "low")
    )
    steps = int(m.get("loop_steps") or 0)
    got = []
    if overlap < CUT_MIN_BARS:
        got.append("edit_or_talk")
    if loops.is_loop(steps):
        got.append("loop")
    if cut_bars >= TENSION_MIN_BARS:
        got.append("tension")
    if out_span is not None and out_span / bar_s >= SWEEP_MIN_BARS:
        got.append("sweep_out")
    if in_span is not None and in_span / bar_s >= SWEEP_MIN_BARS:
        got.append("sweep_in")
    if CUT_MIN_BARS <= overlap <= CUT_MAX_BARS:
        got.append("cut")
    if both_bars >= LAYER_MIN_BARS:
        got.append("layer")
    if overlap > SHORT_BLEND_MAX_BARS:
        got.append("long_blend")
    elif CUT_MAX_BARS < overlap <= SHORT_BLEND_MAX_BARS:
        got.append("short_blend")
    main = next((name for name in ORDER if name in got), "none")
    return Label(
        seam_id=seam_id,
        label=main,
        labels=" ".join(got),
        bar_s=round(bar_s, 3),
        overlap_bars=round(overlap, 1),
        bass_cut_bars=round(cut_bars, 1),
        bass_both_bars=round(both_bars, 1),
        sweep_out_bars=_bars(out_span, bar_s),
        sweep_in_bars=_bars(in_span, bar_s),
        loop_steps=steps,
    )
