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

from dataclasses import asdict, dataclass, field
import logging
import time

import librosa
import numpy as np
import pyrubberband

from . import floors
from . import tempo as tempo_mod
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
# A list that carries the exact time a record is heard (the USB002 app, 2026-09-28) is searched only
# here around that time. The noise of a whole two hour mix buried Fred's one to two minute plays: over
# the whole file the controls scored up to 135, inside such a window at most 64 (window check,
# 2026-09-28, 739 lookups on ten solo segments), and 40 to 60 % of his records were found instead of 1 in 5.
NEAR_BEFORE_S = 180.0
NEAR_AFTER_S = 300.0
# Speed from the tempo the mix plays the record at (2026-10-08/09). The resampling search above cannot
# see a record played with key lock (tempo changed, pitch kept), and reads those near 1.00: on 40
# Raveform records with Raveform's placement as truth, it had 46 % of the off-speed ones within 0.1 %.
# Measuring the played tempo over the heard span, dividing by the record's own BPM, and then matching
# the record stretched both ways (resampled and key-locked) with a vote search around that speed had
# 92 %, and 20 of the 40 matched best with pitch kept. The candidate with more votes wins, so a record
# without a tempo, or one the tempo meter misreads, keeps the old search's answer.
REFINE_COARSE_STEP, REFINE_COARSE_SPAN = 0.002, 0.01  # ±1 % in 0.2 % steps, both modes
REFINE_FINE_STEP, REFINE_FINE_SPAN = 0.0005, 0.002  # then ±0.2 % in 0.05 % steps, the winner only
REFINE_PAD_S = 20.0  # record cropped to its heard span plus this on each side for the speed search
REFINE_MIN_SPAN_S = 30.0  # a shorter heard span has too few bars for the tempo meter
REFINE_MAX_RATE_OFF = (
    0.2  # a played/own tempo ratio further from 1 than this is an octave or a bad hint
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
    # "whole" or "window"; in a window search the decision votes, the window's control floor and the
    # listed time the window is centred on
    search: str = "whole"
    near_s: float | None = None
    window_votes: int | None = None
    window_floor: int | None = None
    window_control_max: int | None = None
    # every 30 s window where the record beats the loudest control window of this mix:
    # (window start in mix s, votes, offset in frames); the layer timeline is built from these
    windows: list = field(default_factory=list, repr=False)
    # how the record was stretched to the speed that matched: "resample" (pitch moved with tempo) or
    # "keylock" (pitch kept); the tempo the mix played it at, when the refinement ran
    mode: str = "resample"
    played_bpm: float | None = None

    def as_row(self) -> dict:
        row = asdict(self)
        row.pop("windows")
        return row


def coarse_fps(track: np.ndarray) -> dict:
    """The record fingerprinted at each of the coarse speeds, {rate: table}."""
    return {rate: fingerprint(at_mix_speed(track, rate)) for rate in COARSE_RATES}


def speed_and_offset(
    track: np.ndarray, mix_fp: dict, coarse: dict | None = None, span: tuple | None = None
) -> dict:
    """The speed the mix plays the record at and its whole-record best offset, by votes. `coarse` is
    `coarse_fps(track)` when the caller already has it; `span` limits the votes to those mix frames."""
    best = {"votes": 0, "rate": 1.0, "offset": 0}
    for rate in COARSE_RATES:
        fp = coarse[rate] if coarse is not None else fingerprint(at_mix_speed(track, rate))
        votes, offset = match(fp, mix_fp, span)
        if votes > best["votes"]:
            best = {"votes": votes, "rate": rate, "offset": offset}
        if rate == 1.0 and votes >= EARLY_ACCEPT:
            break
    if best["votes"] < FINE_MIN_VOTES:
        return best
    steps = int(round(FINE_RATE_SPAN / FINE_RATE_STEP))
    for k in range(1, steps + 1):
        for rate in (best["rate"] - k * FINE_RATE_STEP, best["rate"] + k * FINE_RATE_STEP):
            votes, offset = match(fingerprint(at_mix_speed(track, rate)), mix_fp, span)
            if votes > best["votes"]:
                best = {"votes": votes, "rate": rate, "offset": offset}
    return best


def keylocked(track: np.ndarray, rate: float) -> np.ndarray:
    """The record `rate` times faster with its pitch kept, the mixer's R3 engine."""
    if abs(rate - 1.0) < 1e-6:
        return track
    return pyrubberband.time_stretch(track, SR, rate, rbargs={"-3": ""}).astype(np.float32)


def at_speed(track: np.ndarray, rate: float, mode: str) -> np.ndarray:
    return at_mix_speed(track, rate) if mode == "resample" else keylocked(track, rate)


def played_tempo(mix_path, first_s: float, last_s: float, bpm: float) -> float | None:
    """The tempo the mix runs at between `first_s` and `last_s`, the record's own BPM as the hint."""
    y, _ = librosa.load(
        str(mix_path), sr=tempo_mod.mixer.SR, mono=True, offset=first_s, duration=last_s - first_s
    )
    return tempo_mod.mixer._measure_tempo(y, bpm)


def tempo_refine(
    track: np.ndarray,
    mix_path,
    mix_fp: dict,
    bpm: float | None,
    first_s: float | None,
    last_s: float | None,
    span: tuple | None = None,
) -> dict | None:
    """Speed and offset from the played tempo, refined by votes in whichever stretch mode matches.
    Returns {"votes", "rate", "offset", "mode", "played_bpm"} with whole-record votes, comparable with
    `speed_and_offset`, or None when there is no tempo to go on."""
    if bpm is None or first_s is None or last_s is None or last_s - first_s < REFINE_MIN_SPAN_S:
        return None
    played = played_tempo(mix_path, first_s, last_s, bpm)
    if played is None:
        return None
    rate0 = played / bpm
    if abs(rate0 - 1.0) > REFINE_MAX_RATE_OFF:
        return None
    whole = {
        m: match(fingerprint(at_speed(track, rate0, m)), mix_fp, span)
        for m in ("resample", "keylock")
    }
    mode0 = max(whole, key=lambda m: whole[m][0])
    if whole[mode0][0] < FINE_MIN_VOTES:
        return None
    # the part of the record that sounds in the heard span, in the record's own clock
    zero = whole[mode0][1] * FRAME_S
    start = max(0.0, (first_s - zero - REFINE_PAD_S) * rate0)
    stop = (last_s - zero + REFINE_PAD_S) * rate0
    crop = track[int(start * SR) : int(stop * SR)]
    if len(crop) < SR * REFINE_MIN_SPAN_S:
        return None
    best = {}
    for mode in ("resample", "keylock"):
        best[mode] = (0, rate0)
        for r in np.arange(
            rate0 - REFINE_COARSE_SPAN, rate0 + REFINE_COARSE_SPAN + 1e-9, REFINE_COARSE_STEP
        ):
            v, _ = match(fingerprint(at_speed(crop, float(r), mode)), mix_fp, span)
            if v > best[mode][0]:
                best[mode] = (v, float(r))
    mode = max(best, key=lambda m: best[m][0])
    votes, rate = best[mode]
    for r in np.arange(rate - REFINE_FINE_SPAN, rate + REFINE_FINE_SPAN + 1e-9, REFINE_FINE_STEP):
        v, _ = match(fingerprint(at_speed(crop, float(r), mode)), mix_fp, span)
        if v > votes:
            votes, rate = v, float(r)
    rate = round(rate, 5)
    votes, offset = match(fingerprint(at_speed(track, rate, mode)), mix_fp, span)
    return {
        "votes": votes,
        "rate": rate,
        "offset": offset,
        "mode": mode,
        "played_bpm": round(float(played), 3),
    }


def section_offsets(
    track_at_speed: np.ndarray, mix_fp: dict, near: int, span: tuple | None = None
) -> list[tuple[int, int]]:
    """(votes, offset) per SECTION_S section, each searched only within LOCAL_FRAMES of `near`.

    A free search per section drowned in a two-hour mix's coincidences (2026-09-24). Searched locally a
    section either confirms the offset, moves it by a loop, or has nothing to say."""
    out = []
    step = int(SECTION_S * SR)
    for start in range(0, len(track_at_speed) - step // 2, step):
        piece = track_at_speed[start : start + step]
        shift = int(round(start / SR / FRAME_S))
        mix_frames, offsets = pairs(fingerprint(piece), mix_fp)
        if span is not None:
            offsets = offsets[(mix_frames >= span[0]) & (mix_frames < span[1])]
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


def window_counts(track_fp: dict, mix_fp: dict) -> dict:
    """Per SWEEP_WIN_S window stepping SWEEP_HOP_S: the record's best offset in that window and the
    pairs on it. Returns {starts, counts, offsets (per window), mix_frames, pair_offsets, lo, hi}, or
    None when nothing is shared."""
    mix_frames, offsets = pairs(track_fp, mix_fp)
    if not len(mix_frames):
        return None
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
    return {
        "starts": starts,
        "counts": counts,
        "offsets": best_offset,
        "mix_frames": mix_frames,
        "pair_offsets": offsets,
        "lo": lo,
        "hi": hi,
    }


def windows_above(w: dict | None, level: int) -> list:
    """(window start in mix s, votes, offset in frames) for every window with more than `level` votes."""
    if w is None:
        return []
    keep = np.nonzero(w["counts"] > level)[0]
    return [
        (round(float(w["starts"][i] * FRAME_S), 1), int(w["counts"][i]), int(w["offsets"][i]))
        for i in keep
    ]


def presence(
    track_fp: dict, mix_fp: dict, floor: int = SWEEP_MIN_HASHES, w: dict | None = None
) -> tuple[float | None, float | None, int, int, int]:
    """(first heard, last heard, best window count, offset at first, offset at last), mix seconds and
    frames. Per SWEEP_WIN_S window the record's best offset in that window is counted, so a looped or
    edited record is present wherever any part of it plays. The times are the first and last shared
    pair inside the first and last present window. None when nothing is shared. `w` is
    `window_counts(track_fp, mix_fp)` when the caller already has it."""
    if w is None:
        w = window_counts(track_fp, mix_fp)
    if w is None:
        return None, None, 0, 0, 0
    counts, best_offset = w["counts"], w["offsets"]
    mix_frames, offsets, lo, hi = w["mix_frames"], w["pair_offsets"], w["lo"], w["hi"]
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


def near_span(near_s: float) -> tuple[int, int]:
    """The mix frames a record listed at `near_s` is searched in."""
    return (
        int(max(0.0, near_s - NEAR_BEFORE_S) / FRAME_S),
        int((near_s + NEAR_AFTER_S) / FRAME_S),
    )


def restrict(mix_fp: dict, span: tuple) -> dict:
    """The mix table cut down to the frames of `span`. Matching against it gives exactly the votes
    `match(..., span)` gives on the whole table, at a fraction of the cost, because only the hashes
    inside the window are paired (2026-09-28: the span filter alone left a windowed lookup slower than
    a whole-mix one, Fred's NY1 set at about 2 min a record)."""
    lo, hi = span
    out = {}
    for key, frames in mix_fp.items():
        inside = frames[(frames >= lo) & (frames < hi)]
        if len(inside):
            out[key] = inside
    return out


def window_control_votes(control_coarse: list[dict], mix_fp: dict, span: tuple) -> list[int]:
    """Each control's best votes over the coarse speeds inside one span: the identical path a windowed
    record's decision votes take."""
    return [max(match(fp, mix_fp, span)[0] for fp in c.values()) for c in control_coarse]


def locate_track(
    track_id: str,
    track: np.ndarray,
    mix_fp: dict,
    sweep_floor: int = SWEEP_MIN_HASHES,
    weak_level: int | None = None,
    near_s: float | None = None,
    control_coarse: list[dict] | None = None,
    mix_path=None,
    bpm: float | None = None,
) -> Located:
    """One record against one mix table. `track` is the record's audio at the fingerprint rate.
    With `mix_path` and the record's own `bpm`, the speed is refined from the played tempo
    (`tempo_refine`) once the record's heard span is known, and the better-voted answer stands.
    `sweep_floor` is the presence floor from this mix's controls (`sweep_floor_from`), `weak_level` the
    loudest control window (`weak_level_from`): windows above it are kept for the layer timeline.

    With `near_s` (a list that carries the exact time the record is heard) the speed, offset and
    sections are searched only in `near_span(near_s)`, and the decision votes are the best over the
    coarse speeds there, beside the same number for every control in `control_coarse`. Presence is
    always read over the whole mix, so a record used again far from its listed time is still seen."""
    t0 = time.time()
    span = near_span(near_s) if near_s is not None else None
    coarse = coarse_fps(track) if span is not None else None
    extra, search_fp = {}, mix_fp
    if span is not None:
        sub = restrict(mix_fp, span)
        wf = floors.from_controls(window_control_votes(control_coarse or [], sub, None))
        window_votes = max(match(fp, sub)[0] for fp in coarse.values())
        extra = {
            "search": "window",
            "near_s": round(near_s, 1),
            "window_votes": window_votes,
            "window_floor": wf.floor,
            "window_control_max": wf.control_max,
        }
        # found in its window: speed, offset and sections come from the window. Not found there: the
        # whole-mix search as for any record, so its speed is real and presence can hear it where it
        # does play (test_layers: a record listed at 520 s that plays at 60 s)
        if wf.clears(window_votes):
            search_fp = sub
    best = speed_and_offset(track, search_fp, coarse)
    mode, played = "resample", None
    audio = at_mix_speed(track, best["rate"])
    w = window_counts(fingerprint(audio), mix_fp)
    first, last, sweep_best, off_first, off_last = presence(None, mix_fp, sweep_floor, w)
    refined = (
        tempo_refine(track, mix_path, search_fp, bpm, first, last, span) if mix_path else None
    )
    if refined is not None and refined["votes"] > best["votes"]:
        best, mode, played = refined, refined["mode"], refined["played_bpm"]
        audio = at_speed(track, best["rate"], mode)
        w = window_counts(fingerprint(audio), mix_fp)
        first, last, sweep_best, off_first, off_last = presence(None, mix_fp, sweep_floor, w)
    sections = section_offsets(audio, search_fp, best["offset"])
    n_agree, offset = consensus(sections, best["offset"])
    time_zero = offset * FRAME_S
    rate = round(best["rate"], 5)
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
        windows=windows_above(w, weak_level) if weak_level is not None else [],
        mode=mode,
        played_bpm=played,
        **extra,
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


def weak_level_from(controls: list[Located]) -> int:
    """The loudest 30 s window any control scored in this mix. A record window above it is `weak`
    evidence, above twice it (the sweep floor) `present`. Several records weak at one place is a
    signal of layering (Anas, 2026-09-28)."""
    return max([c.sweep_votes for c in controls] + [SWEEP_MIN_HASHES - 1])


def locate_mix(mix_path, tracks: list[dict], controls: list[dict]) -> dict:
    """Fingerprint the mix once, then every record and every control against it.

    `tracks` and `controls` are [{track_id, path}], a track may carry `near_s` (see `locate_track`).
    Returns {"mix_len_s", "floor", "weak_level", "records": [Located], "controls": [Located]}. The floor
    is from the controls' votes, and the caller decides found and confidence per record with
    `confidence` (a windowed record with its own `window_floor`)."""
    t0 = time.time()
    mix_fp, mix_len = mix_fingerprint(mix_path)
    log.info(
        "locate %s: mix fingerprinted, %.1f min, %.0f s", mix_path, mix_len / 60, time.time() - t0
    )
    control_audio = [load(c["path"]) for c in controls]
    ctrl = [locate_track(c["track_id"], a, mix_fp) for c, a in zip(controls, control_audio)]
    sweep_floor = sweep_floor_from(ctrl)
    weak_level = weak_level_from(ctrl)
    control_coarse = (
        [coarse_fps(a) for a in control_audio]
        if any(t.get("near_s") is not None for t in tracks)
        else None
    )
    records = []
    for i, t in enumerate(tracks, 1):
        records.append(
            locate_track(
                t["track_id"],
                load(t["path"]),
                mix_fp,
                sweep_floor,
                weak_level,
                t.get("near_s"),
                control_coarse,
                mix_path=mix_path,
                bpm=t.get("bpm"),
            )
        )
        if i % 10 == 0 or i == len(tracks):
            log.info(
                "locate %s: %d of %d records, %.0f s", mix_path, i, len(tracks), time.time() - t0
            )
    floor = floors.from_controls([c.votes for c in ctrl])
    found = sum(1 for r in records if confidence(r.votes, floor, r.sections_agree) != "not found")
    log.info(
        "locate %s: %d of %d records found on the whole-mix floor, floor %d, sweep floor %d, "
        "weak level %d, from %d controls, %.0f s",
        mix_path,
        found,
        len(records),
        floor.floor,
        sweep_floor,
        weak_level,
        floor.control_n,
        time.time() - t0,
    )
    return {
        "mix_len_s": mix_len,
        "floor": floor,
        "weak_level": weak_level,
        "records": records,
        "controls": ctrl,
    }
