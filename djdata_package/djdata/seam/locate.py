"""Where inside a mix each listed record plays, at what speed, and which part of it was heard.

The mix is fingerprinted once (`fingerprint.mix_fingerprint`) and every record is looked up in that one
table. Nothing narrows the search to the listed minute: the whole mix is searched for every record, and
the listed minute is only reported as drift.

Per record, in order:

1. speed: the nine coarse speeds by whole-record votes, early accept at `EARLY_ACCEPT` votes at speed
   1.0, then `FINE_RATE_STEP` steps within `FINE_RATE_SPAN` of the best;
2. time zero: the whole-record best offset, then each `SECTION_S` section of the record searched within
   `LOCAL_FRAMES` of it, and the vote-weighted median of the sections' offsets becomes the record's time
   zero. A loop-shifted pick moves to where most of the record agrees. `sections_agree` counts the
   sections within `AGREE_FRAMES` of that;
3. presence: in every `SWEEP_WIN_S` window of the mix, stepping `SWEEP_HOP_S`, the record's best
   offset in that window, whatever it is, is counted. A window is present when that count reaches
   the sweep floor, twice the loudest window any control record scored in this mix. First and last
   present window are where the record was first and last heard, and `played_from_s` /
   `played_to_s` are the record times at those two windows' own offsets.

   Why any offset and not the record's one time zero (2026-09-25). Solomun loops: in the gaps
   between his records, record A matched at four and five offsets shifted by tens of seconds with
   thousands of votes each against a control of 35. A single-offset sweep heard him for 2.5 min of a
   nine minute play and set 21 of his 23 seams aside. The section consensus still gives one time
   zero for the cut; presence follows the DJ.

The floor is measured, never assumed: control records from other mixes go through the identical path and
`floors.from_controls` sets the floor at twice the loudest of them. Acceptance is loose on purpose
(Anas, 2026-09-24): found = votes at or above the floor OR at least `MIN_SECTIONS_AGREE` sections agree.
`confidence` says which.

Time zero is the mix time the record's own 0:00 would fall on. It can sit before the mix starts when the
DJ cues in deep. The part of the record playing at mix time T is track time (T - time_zero) * rate.

Checked 2026-09-24: a synthetic record played 3 % fast at 90.00 s comes back at 90.00 s, rate 1.03;
against the run's old locate on 35 tracks the time zero agrees to a median 0.6 s; six clips heard.
"""

from dataclasses import asdict, dataclass
import logging
import time

import numpy as np

from . import floors
from .fingerprint import (
    FRAME_S,
    SR,
    at_mix_speed,
    fingerprint,
    load,
    match,
    mix_fingerprint,
    pairs,
)

log = logging.getLogger("djdata.seam.locate")

COARSE_RATES = (1.0, 0.99, 1.01, 0.98, 1.02, 0.97, 1.03, 0.96, 1.04)
EARLY_ACCEPT = 400  # votes at speed 1.0 that make the other speeds pointless
FINE_RATE_STEP = 0.0005  # 0.05 % steps; the 0.25 % grid drifted 56 ms across 45 s
FINE_RATE_SPAN = 0.0025
FINE_MIN_VOTES = 10  # below this the coarse pick is noise and refining it is wasted work
SECTION_S = 45.0  # the record is matched in sections this long
LOCAL_FRAMES = (
    1290  # a section is searched within 15 s of the whole-record offset: room for a loop shift
)
SECTION_MIN_VOTES = 5  # a section with fewer votes has nothing to say
AGREE_FRAMES = 43  # sections agree when within half a second
MIN_SECTIONS_AGREE = 3  # this many sections on one offset is presence, whatever the vote total
SWEEP_WIN_S = (
    30.0  # presence is read in 30 s windows: a 10 s window matches a techno loop anywhere
)
SWEEP_HOP_S = 10.0
SWEEP_MIN_HASHES = 5  # the least a window can hold and count as present, whatever the controls say
OFFSET_TOLERANCE = 2  # frames: a pair agrees with the offset within this
EDGE_PAIRS_PER_S = (
    3  # first and last heard: the first and last second holding this many agreeing pairs
)


@dataclass
class Located:
    """One record looked up in one mix. Times in seconds of mix time unless the name says track."""

    track_id: str
    votes: int
    rate: float
    time_zero_s: float
    sections: int
    sections_agree: int
    first_heard_s: float | None
    last_heard_s: float | None
    played_from_s: float | None
    played_to_s: float | None
    sweep_votes: int
    sweep_floor: int
    track_len_s: float
    seconds: float

    def as_row(self) -> dict:
        return asdict(self)


def speed_and_offset(track: np.ndarray, mix_fp: dict) -> dict:
    """The speed the mix plays the record at and its whole-record best offset, by votes."""
    best = {"votes": 0, "rate": 1.0, "offset": 0}
    for rate in COARSE_RATES:
        votes, offset = match(fingerprint(at_mix_speed(track, rate)), mix_fp)
        if votes > best["votes"]:
            best = {"votes": votes, "rate": rate, "offset": offset}
        if rate == 1.0 and votes >= EARLY_ACCEPT:
            break
    if best["votes"] < FINE_MIN_VOTES:
        return best
    steps = int(round(FINE_RATE_SPAN / FINE_RATE_STEP))
    for k in range(1, steps + 1):
        for rate in (best["rate"] - k * FINE_RATE_STEP, best["rate"] + k * FINE_RATE_STEP):
            votes, offset = match(fingerprint(at_mix_speed(track, rate)), mix_fp)
            if votes > best["votes"]:
                best = {"votes": votes, "rate": rate, "offset": offset}
    return best


def section_offsets(track_at_speed: np.ndarray, mix_fp: dict, near: int) -> list[tuple[int, int]]:
    """(votes, offset) per SECTION_S section, each searched only within LOCAL_FRAMES of `near`.

    A free search per section drowned in a two-hour mix's coincidences (2026-09-24). Searched locally a
    section either confirms the offset, moves it by a loop, or has nothing to say."""
    out = []
    step = int(SECTION_S * SR)
    for start in range(0, len(track_at_speed) - step // 2, step):
        piece = track_at_speed[start : start + step]
        shift = int(round(start / SR / FRAME_S))
        _, offsets = pairs(fingerprint(piece), mix_fp)
        offsets = offsets - shift
        local = offsets[np.abs(offsets - near) <= LOCAL_FRAMES]
        if not len(local):
            out.append((0, near))
            continue
        values, counts = np.unique(local, return_counts=True)
        best = int(np.argmax(counts))
        out.append((int(counts[best]), int(values[best])))
    return out


def consensus(sections: list[tuple[int, int]], near: int) -> tuple[int, int]:
    """(sections agreeing, refined offset): the vote-weighted median offset of the sections with votes,
    and how many sections sit within AGREE_FRAMES of it."""
    live = sorted((offset, votes) for votes, offset in sections if votes >= SECTION_MIN_VOTES)
    if not live:
        return 0, near
    total = sum(v for _, v in live)
    acc, refined = 0, near
    for offset, votes in live:
        acc += votes
        if acc * 2 >= total:
            refined = offset
            break
    agree = sum(1 for offset, _ in live if abs(offset - refined) <= AGREE_FRAMES)
    return agree, refined


def presence(
    track_fp: dict, mix_fp: dict, floor: int = SWEEP_MIN_HASHES
) -> tuple[float | None, float | None, int, int, int]:
    """(first heard, last heard, best window count, offset at first, offset at last), mix seconds and
    frames. Per SWEEP_WIN_S window the record's best offset in that window is counted, so a looped or
    edited record is present wherever any part of it plays. The times are the first and last shared
    pair inside the first and last present window. None when nothing is shared."""
    mix_frames, offsets = pairs(track_fp, mix_fp)
    if not len(mix_frames):
        return None, None, 0, 0, 0
    order = np.argsort(mix_frames, kind="stable")
    mix_frames, offsets = mix_frames[order], offsets[order]
    win, hop = int(SWEEP_WIN_S / FRAME_S), int(SWEEP_HOP_S / FRAME_S)
    starts = np.arange(max(int(mix_frames[0]) - win, 0), int(mix_frames[-1]) + hop, hop)
    lo = np.searchsorted(mix_frames, starts, side="left")
    hi = np.searchsorted(mix_frames, starts + win, side="left")
    counts = np.zeros(len(starts), dtype=np.int64)
    best_offset = np.zeros(len(starts), dtype=np.int64)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a < SWEEP_MIN_HASHES:
            continue
        values, n = np.unique(offsets[a:b], return_counts=True)
        k = int(np.argmax(n))
        counts[i], best_offset[i] = n[k], values[k]
    best = int(counts.max())
    present = np.nonzero(counts >= max(SWEEP_MIN_HASHES, floor))[0]
    if not len(present):
        return None, None, best, 0, 0
    i0, i1 = present[0], present[-1]
    first = _edge(mix_frames[lo[i0] : hi[i0]], offsets[lo[i0] : hi[i0]], best_offset[i0], True)
    last = _edge(mix_frames[lo[i1] : hi[i1]], offsets[lo[i1] : hi[i1]], best_offset[i1], False)
    return (
        float(first * FRAME_S),
        float(last * FRAME_S),
        best,
        int(best_offset[i0]),
        int(best_offset[i1]),
    )


def _edge(mix_frames: np.ndarray, offsets: np.ndarray, offset: int, first: bool) -> int:
    """The first (or last) second inside one window that holds EDGE_PAIRS_PER_S pairs at the
    window's own offset. One coincidental pair at that offset must not move the edge."""
    agreeing = np.sort(mix_frames[np.abs(offsets - offset) <= OFFSET_TOLERANCE])
    per_s = int(round(1.0 / FRAME_S))
    seconds = agreeing // per_s
    values, counts = np.unique(seconds, return_counts=True)
    dense = values[counts >= EDGE_PAIRS_PER_S]
    if not len(dense):
        return int(agreeing[0] if first else agreeing[-1])
    sec = dense[0] if first else dense[-1]
    inside = agreeing[seconds == sec]
    return int(inside[0] if first else inside[-1])


def locate_track(
    track_id: str, track: np.ndarray, mix_fp: dict, sweep_floor: int = SWEEP_MIN_HASHES
) -> Located:
    """One record against one mix table. `track` is the record's audio at the fingerprint rate.
    `sweep_floor` is the presence floor from this mix's controls (`sweep_floor_from`)."""
    t0 = time.time()
    best = speed_and_offset(track, mix_fp)
    at_speed = at_mix_speed(track, best["rate"])
    sections = section_offsets(at_speed, mix_fp, best["offset"])
    n_agree, offset = consensus(sections, best["offset"])
    first, last, sweep_best, off_first, off_last = presence(
        fingerprint(at_speed), mix_fp, sweep_floor
    )
    time_zero = offset * FRAME_S
    rate = round(best["rate"], 4)
    track_len = len(track) / SR
    played_from = played_to = None
    if first is not None:
        # record time at each end, along that end's own offset, so a loop reads as the part looped
        played_from = round(min(max((first - off_first * FRAME_S) * rate, 0.0), track_len), 1)
        played_to = round(min(max((last - off_last * FRAME_S) * rate, 0.0), track_len), 1)
    return Located(
        track_id=track_id,
        votes=best["votes"],
        rate=rate,
        time_zero_s=round(time_zero, 2),
        sections=len(sections),
        sections_agree=n_agree,
        first_heard_s=None if first is None else round(first, 1),
        last_heard_s=None if last is None else round(last, 1),
        played_from_s=played_from,
        played_to_s=played_to,
        sweep_votes=sweep_best,
        sweep_floor=sweep_floor,
        track_len_s=round(track_len, 1),
        seconds=round(time.time() - t0, 1),
    )


def confidence(votes: int, floor: floors.Floor, sections_agree: int) -> str:
    """confident | by votes | by sections | not found."""
    by_floor = floor.clears(votes)
    by_sections = sections_agree >= MIN_SECTIONS_AGREE
    if by_floor and by_sections:
        return "confident"
    if by_floor:
        return "by votes"
    if by_sections:
        return "by sections"
    return "not found"


def sweep_floor_from(controls: list[Located]) -> int:
    """The presence floor: twice the loudest 30 s window any control scored, at least SWEEP_MIN_HASHES."""
    return max(SWEEP_MIN_HASHES, floors.from_controls([c.sweep_votes for c in controls]).floor)


def locate_mix(mix_path, tracks: list[dict], controls: list[dict]) -> dict:
    """Fingerprint the mix once, then every record and every control against it.

    `tracks` and `controls` are [{track_id, path}]. Returns {"mix_len_s", "floor", "records": [Located],
    "controls": [Located]}. The floor is from the controls' votes, and the caller decides found and
    confidence per record with `confidence`."""
    t0 = time.time()
    mix_fp, mix_len = mix_fingerprint(mix_path)
    log.info(
        "locate %s: mix fingerprinted, %.1f min, %.0f s", mix_path, mix_len / 60, time.time() - t0
    )
    ctrl = [locate_track(c["track_id"], load(c["path"]), mix_fp) for c in controls]
    sweep_floor = sweep_floor_from(ctrl)
    records = [locate_track(t["track_id"], load(t["path"]), mix_fp, sweep_floor) for t in tracks]
    floor = floors.from_controls([c.votes for c in ctrl])
    found = sum(1 for r in records if confidence(r.votes, floor, r.sections_agree) != "not found")
    log.info(
        "locate %s: %d of %d records found, floor %d, sweep floor %d, from %d controls, %.0f s",
        mix_path,
        found,
        len(records),
        floor.floor,
        sweep_floor,
        floor.control_n,
        time.time() - t0,
    )
    return {"mix_len_s": mix_len, "floor": floor, "records": records, "controls": ctrl}
