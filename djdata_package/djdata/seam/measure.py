"""Measure one seam window: per band when each record enters and leaves, the bass swap, the bass words,
full-band presence, the outgoing record's loop, with every floor from controls run on that window.

The job carries paths and alignment only, so it crosses a process boundary cleanly:
    {seam_id, window_path, window_t0, a: {path, time_zero_s, rate}, b: {...}, controls: [path, ...]}

Each record is resampled to the speed the mix plays it and laid on the window's clock. Only the stretch
that can sound in the window is fingerprinted, `SLICE_PAD_S` either side, because a whole record in the
high band made 280,000 hashes and ten workers holding five each stalled the VM (2026-09-21). Controls
go through the identical slice rule at B's alignment, so the floor is a fair one.

Times in the row are seconds into the window. `measured` is the one judgement made here: both records
beat their own floor somewhere in some band. Everything else is a number with its floor beside it.
"""

import numpy as np

from . import bands, bass, floors, loops, presence
from .fingerprint import SR, at_mix_speed, load

SLICE_PAD_S = 60.0


def usable_slice(track_at_speed: np.ndarray, track_t_at_window_start: float, window_len: float):
    """The part of the record that can sound in the window plus the pad. Returns (slice, its start in
    the record's mix-speed clock)."""
    first = max(0.0, track_t_at_window_start - SLICE_PAD_S)
    last = track_t_at_window_start + window_len + SLICE_PAD_S
    return track_at_speed[int(first * SR) : int(last * SR)], first


def lay_out(window_t0: float, window_len: float, rec: dict, audio: np.ndarray):
    """A record on the window's clock: (slice at mix speed, anchor). The anchor is (slice time, window
    time) of one common instant, the window's start. At mix speed the record's clock advances with the
    mix, so its position at the window start is window_t0 minus its time zero."""
    at_speed = at_mix_speed(audio, rec["rate"])
    piece, shift = usable_slice(at_speed, window_t0 - rec["time_zero_s"], window_len)
    if len(piece) < SR:
        # the window lies outside the record at its time zero (heard past its own length, a loop):
        # a second of silence keeps every reading at zero instead of crashing the seam
        piece = np.zeros(SR, dtype=np.float32)
    return piece, (window_t0 - rec["time_zero_s"] - shift, 0.0)


def measure_one(job: dict) -> dict:
    window = load(job["window_path"])
    window_len = len(window) / SR
    t0 = float(job["window_t0"])
    row = {
        "seam_id": job["seam_id"],
        "window_s": round(window_len, 1),
        "control_n": len(job["controls"]),
    }

    tracks, anchors = {}, {}
    for name, rec in (("A", job["a"]), ("B", job["b"])):
        tracks[name], anchors[name] = lay_out(t0, window_len, rec, load(rec["path"]))
    for i, path in enumerate(job["controls"], 1):
        tracks[f"C{i}"], anchors[f"C{i}"] = lay_out(t0, window_len, job["b"], load(path))
    control_names = [n for n in tracks if n.startswith("C")]
    row["track_slice_s"] = round(min(len(a) / SR for a in tracks.values()), 1)

    # bands: when each record's band is first and last seen
    for band in bands.BANDS:
        fps = {name: bands.band_fp(audio, band) for name, audio in tracks.items()}
        curves = bands.band_votes(window, band, fps, anchors)
        floor = floors.from_controls([int(curves[n].max()) for n in control_names])
        _, a_last = bands.first_last(curves, "A", floor.floor)
        b_first, _ = bands.first_last(curves, "B", floor.floor)
        row.update(
            {
                f"{band}_out_s": None if a_last is None else round(a_last, 1),
                f"{band}_in_s": None if b_first is None else round(b_first, 1),
                f"{band}_a_votes": int(curves.A.max()),
                f"{band}_b_votes": int(curves.B.max()),
                f"{band}_floor": floor.floor,
                f"{band}_control_max": floor.control_max,
                f"{band}_a_separation": floors.separation(curves.A.max(), floor),
                f"{band}_b_separation": floors.separation(curves.B.max(), floor),
                f"{band}_floor_trusted": floor.trusted,
            }
        )

    # bass: the swap, from the fine low-band fingerprint
    low_fps = {name: bass.low_band_fp(audio) for name, audio in tracks.items()}
    low = bass.low_band_votes(window, low_fps, anchors)
    bass_floor = floors.from_controls([int(low[n].max()) for n in control_names])
    row.update(
        {
            "bass_swap_s": bass.bass_swap(low, bass_floor.floor),
            "bass_a_votes": int(low.A.max()),
            "bass_b_votes": int(low.B.max()),
            "bass_floor": bass_floor.floor,
            "bass_control_max": bass_floor.control_max,
            "bass_a_separation": floors.separation(low.A.max(), bass_floor),
            "bass_b_separation": floors.separation(low.B.max(), bass_floor),
            "bass_floor_trusted": bass_floor.trusted,
        }
    )

    # full-band presence: entry, exit, overlap, and the outgoing record's loop
    full_fps = {
        name: presence.fingerprint(audio, presence.PERCENTILE) for name, audio in tracks.items()
    }
    pres = presence.curves(window, full_fps, anchors)
    pres_floor = floors.from_controls([int(pres[f"{n}_anch"].max()) for n in control_names])
    free_floor = floors.from_controls([int(pres[n].max()) for n in control_names])
    b_in = presence.first_present(pres, "B", pres_floor.floor)
    a_out = presence.last_present(pres, "A", pres_floor.floor)
    row.update(
        {
            "in_s": None if b_in is None else round(b_in, 1),
            "out_s": None if a_out is None else round(a_out, 1),
            "overlap_s": None if b_in is None or a_out is None else round(a_out - b_in, 1),
            "presence_floor": pres_floor.floor,
            "presence_a_votes": int(pres.A_anch.max()),
            "presence_b_votes": int(pres.B_anch.max()),
            "loop_steps": loops.loop_steps(pres, "A", free_floor.floor, b_in),
            "loop_floor": free_floor.floor,
        }
    )

    # how loud each record's own band is across the overlap, against the loud parts of the slice: a
    # band the record does not carry here cannot be heard arriving or leaving, whatever the EQ did.
    # On a pair cued at a sparse intro the incoming read 3 to 7 dB under and its moments came 7 to 17
    # bars late; on a steady pair 0 to 2.5 dB under and the moments were on time (2026-10-09).
    for name in ("A", "B"):
        track_t, mix_t = anchors[name]
        if b_in is not None and a_out is not None and a_out > b_in:
            start, n_ov = track_t + (b_in - mix_t), max(int((a_out - b_in) / bands.HOP_S), 1)
        else:
            start, n_ov = None, 0
        n_all = max(int(len(tracks[name]) / SR / bands.HOP_S) - 4, 1)
        for band in bands.BANDS:
            whole = bands.band_level_db(tracks[name], band, 0.0, n_all)
            level = (
                bands.band_level_db(tracks[name], band, start, n_ov) if start is not None else []
            )
            carried = (
                None
                if not len(level) or np.all(np.isnan(level))
                else round(float(np.nanmedian(level) - np.nanpercentile(whole, 80)), 1)
            )
            row[f"{band}_{name.lower()}_carried_db"] = carried

    # bass words along the window
    times = low.t.to_numpy()
    n = len(times)
    mix_low = bass.low_level_db(window, 0.0, n)
    mix_has = bass.has_bass(mix_low, float(np.nanpercentile(mix_low, 80)))
    has, here = {}, {}
    for name in ("A", "B"):
        track_t, _ = anchors[name]
        level = bass.low_level_db(tracks[name], track_t, n)
        whole = bass.low_level_db(
            tracks[name], 0.0, max(int(len(tracks[name]) / SR / bands.HOP_S) - 4, 1)
        )
        has[name] = bass.has_bass(level, float(np.nanpercentile(whole, 80)))
        here[name] = presence.here_at(pres, name, pres_floor.floor, times)
    word_list = bass.words(
        mix_has,
        here["A"],
        here["B"],
        has["A"],
        has["B"],
        (low.A >= bass_floor.floor).to_numpy(),
        (low.B >= bass_floor.floor).to_numpy(),
    )
    for word, seconds in bass.seconds_per_word(word_list).items():
        row[f"bass_{word}_s"] = seconds
    row["bass_cut_longest_s"] = bass.longest_run(times, word_list, "cut")
    row["bass_both_longest_s"] = bass.longest_run(times, word_list, "both")

    a_sep = max(row[f"{b}_a_separation"] for b in bands.BANDS)
    b_sep = max(row[f"{b}_b_separation"] for b in bands.BANDS)
    row["a_best_separation"] = a_sep
    row["b_best_separation"] = b_sep
    row["measured"] = int(a_sep >= 1.0 and b_sep >= 1.0)
    row["control_trusted"] = int(row["low_floor_trusted"] and row["bass_floor_trusted"])
    return row, {"bands": None, "low": low, "presence": pres, "words": word_list}
