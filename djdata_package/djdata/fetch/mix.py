"""Download one full mix, cut one window per seam, record what part of each track was played, delete the mix.

Two sources, two orders.

Raveform: the alignment is given, so the windows are cut straight from it and the tracks are fetched
afterwards. Time-based partial downloads measured 5.6 s off, so the whole file is fetched and cut locally.

1001tracklists: nothing is given but the play order and, usually, a start time to the minute. So the
tracks are fetched FIRST and the mix second; then every track is fingerprinted against the mix audio
(seam/locate.py) to find the mix time its time zero falls on and the speed it was played at, and the
windows are cut from those. Cutting from the listed minutes instead was the old way and could miss the
transition by up to a minute, with the mix already deleted by the time anyone noticed.

The cut is a stream copy with `-ss`/`-to` AFTER `-i`, which seeks accurately, to one codec packet.

Putting them BEFORE `-i` is the fast form, and it is what this did until 2026-09-20. With `-c copy` that
seeks to a container boundary, not to the time asked for. On mp3 a boundary is one frame, about 26 ms,
so the Raveform corpus came out right and the error went unnoticed. These mixes are webm, where the
boundary is a cluster: 1,375 of 1,475 windows were webm, only 7.2% came out the length requested, 35.3%
were exactly 10.0 s long, and the excess sat at the FRONT. Sample zero of a window was therefore up to
10 s earlier than every placement calculation assumed, which is why an audit that looked for the records
inside the cut windows found nothing while locate was finding them correctly in the full mix.
Measured on one seam: asked for 240.000 s, the old form began at 230.02, the new form at 240.00.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import logging
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from ..state import State
from . import yt

log = logging.getLogger("djdata.fetch.mix")

WINDOW_BEFORE_S = 90.0     # kept before the best guess of the transition
WINDOW_AFTER_S = 210.0     # and after it
MIN_TRACKS_LOCATED = 2     # a mix with fewer located tracks yields no seam


def download_full(cfg, url: str, dest_stem: Path) -> Path:
    with tempfile.TemporaryDirectory(dir=dest_stem.parent) as td:
        got = yt.download(cfg, url, cfg.download["mix_format"], Path(td))
        dest = dest_stem.with_suffix(got.suffix)
        shutil.move(str(got), dest)
    return dest


def cut_window(src: Path, t0: float, t1: float, dest_stem: Path) -> Path:
    dest = dest_stem.with_suffix(src.suffix)
    # -ss and -to AFTER -i. Before it they seek to a container boundary, not to the time asked for.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-ss", f"{max(t0, 0):.3f}", "-to", f"{t1:.3f}", "-c", "copy", str(dest)], check=True)
    return dest


def duration_s(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def process_mix(cfg, state: State, mix: dict) -> int:
    """Download the mix, cut every seam window of it, then delete the mix unless we are keeping it.

    `download.keep_mixes` keeps the full file on disk. Where the window positions come from the audio
    rather than from a given alignment, a bad locate leaves wrong windows and, once the mix is gone,
    no way to recut them without downloading it again. 139 mixes is about 17 GB, which is cheap next
    to losing one (Anas, 2026-09-19)."""
    full = download_full(cfg, mix["url"], cfg.dirs["mixes_tmp"] / mix["mix_id"])
    keep = bool(cfg.download.get("keep_mixes"))
    try:
        if cfg.source == "tracklists":
            return _windows_by_locate(cfg, state, mix, full)
        return _windows_from_alignment(state, mix, full, cfg)
    finally:
        if keep:
            log.info("  kept the mix at %s (%.0f MB)", full, full.stat().st_size / 1e6 if full.exists() else 0)
        else:
            full.unlink(missing_ok=True)


def _windows_from_alignment(state: State, mix: dict, full: Path, cfg) -> int:
    n = 0
    for seam in state.seams_of_mix(mix["mix_id"]):
        if seam["status"] != "pending":
            continue
        t0, t1 = json.loads(seam["coarse"])["window"]
        window = cut_window(full, t0, t1, cfg.dirs["windows"] / seam["seam_id"])
        state.set_seam(seam["seam_id"], "ready", window_path=str(window))
        n += 1
    return n


def _windows_by_locate(cfg, state: State, mix: dict, full: Path) -> int:
    """Fingerprint every track of this mix against the mix audio, then cut the windows from where they
    actually played. Writes out/played/<mix_id>.json: the part of each record the DJ used."""
    from ..legacy.locate_slice import locate_all

    mix_id = mix["mix_id"]
    seams = [s for s in state.seams_of_mix(mix_id) if s["status"] == "pending"]
    if not seams:
        return 0
    mix_len = duration_s(full)

    # every distinct track of this mix, with its listed time and the file the track worker fetched
    entries, listed = {}, {}
    for seam in seams:
        coarse = json.loads(seam["coarse"])
        for side, track_id in (("A", seam["a"]), ("B", seam["b"])):
            row = state.track(track_id)
            if not row or row["status"] != "ready" or not row["path"]:
                continue
            listed_t = coarse.get(side, {}).get("listed_mix_t")
            listed[track_id] = listed_t
            entries[track_id] = {"track_id": track_id, "track_path": row["path"], "listed_mix_t": listed_t,
                                 "mix_duration_s": mix_len}
    if len(entries) < MIN_TRACKS_LOCATED:
        raise RuntimeError(f"only {len(entries)} of this mix's tracks are on disk")

    # A track with no listed time is searched against the whole mix, and a whole-mix fingerprint is
    # heavy: twelve of them at once would exhaust the machine's memory. Those mixes get a small pool.
    untimed = any(entry["listed_mix_t"] is None for entry in entries.values())
    workers = int(cfg.workers.get("locate_untimed", 3) if untimed else cfg.workers.get("locate", 8))
    budget = float(cfg.workers.get("locate_timeout_s", 900))
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx)
    try:
        located = locate_all(full, list(entries.values()), pool=pool, timeout_s=budget)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    found = {k: v for k, v in located.items() if v["found"]}
    votes = sorted(v["votes"] for v in found.values())
    rates = sorted(v["rate"] for v in found.values())
    log.info("LOCATE %s%s: %d/%d tracks found in %.0fs | votes min/median/max %s/%s/%s | rate %s..%s | mix %.0fs",
             mix_id, " (no listed times)" if untimed else "", len(found), len(entries), time.time() - t0,
             votes[0] if votes else "-", votes[len(votes) // 2] if votes else "-", votes[-1] if votes else "-",
             rates[0] if rates else "-", rates[-1] if rates else "-", mix_len)

    share = len(found) / max(len(entries), 1)
    ordered = [v["mix_t"] for _, v in sorted(found.items(), key=lambda kv: kv[1]["mix_t"])]
    spacing = [round(b - a) for a, b in zip(ordered, ordered[1:])]
    log.info("  QUALITY %s: %.0f%% of tracks located, spacings %s",
             mix_id, 100 * share, spacing[:12])
    if share < 0.5:
        log.warning("  QUALITY %s: only %.0f%% located — check this mix before trusting its windows",
                    mix_id, 100 * share)

    _write_played(cfg, mix_id, found, listed, mix_len)

    n = 0
    for seam in seams:
        a, b = found.get(seam["a"]), found.get(seam["b"])
        if not a or not b:
            missing = [s for s, r in (("A", a), ("B", b)) if not r]
            reason = f"locate: {','.join(missing)} not found in the mix"
            state.set_seam(seam["seam_id"], "failed", error=reason)
            log.info("  seam %s skipped: %s", seam["seam_id"], reason)
            continue
        # the page order says these two records are neighbours; the audio has to agree. Where the
        # tracklist carried no times the order is the page's guess alone, so this is the only check.
        if b["mix_t"] > _track_end(a) or _track_end(b) < a["mix_t"]:
            state.set_seam(seam["seam_id"], "failed",
                           error=f"locate: records do not overlap (A {a['mix_t']:.0f}-{_track_end(a):.0f}s, "
                                 f"B {b['mix_t']:.0f}-{_track_end(b):.0f}s)")
            log.info("  seam %s skipped: records do not overlap", seam["seam_id"])
            continue
        window = _window_for(a, b, listed.get(seam["b"]), mix_len)
        if window is None:
            state.set_seam(seam["seam_id"], "failed", error="locate: window shorter than 60 s")
            continue
        start, end = window
        coarse = {
            "needs_locate": False, "source": "tracklists",
            "A": {"orig_t": 0.0, "mix_t": a["mix_t"], "rate": a["rate"], "votes": a["votes"],
                  "duration": a.get("track_len_s")},
            "B": {"orig_t": 0.0, "mix_t": b["mix_t"], "rate": b["rate"], "votes": b["votes"],
                  "duration": b.get("track_len_s")},
            "overlap_start_mix_t": max(a["mix_t"], b["mix_t"]),
            "overlap_end_mix_t": _track_end(a),
            "b_track_start_mix_t": b["mix_t"], "a_track_end_mix_t": _track_end(a),
            "window": [start, end],
        }
        path = cut_window(full, start, end, cfg.dirs["windows"] / seam["seam_id"])
        state.set_seam_coarse(seam["seam_id"], coarse)
        state.set_seam(seam["seam_id"], "ready", window_path=str(path))
        log.info("  window %s: %.0f-%.0f s (A at %.0f rate %.3f, B at %.0f rate %.3f)",
                 seam["seam_id"], start, end, a["mix_t"], a["rate"], b["mix_t"], b["rate"])
        n += 1
    return n


def _track_end(result: dict) -> float:
    return result["mix_t"] + (result.get("track_len_s") or 0.0) / result["rate"]


def _window_for(a: dict, b: dict, listed_b, mix_len: float):
    """A window that contains the transition with margin. The best guess of where it happens is the
    listed start of the incoming record, clamped to the span in which the outgoing one is still running."""
    a_end = _track_end(a)
    guess = listed_b if listed_b is not None else b["mix_t"]
    guess = min(max(guess, a["mix_t"] + 30.0), a_end)
    start = max(0.0, guess - WINDOW_BEFORE_S)
    end = min(mix_len, guess + WINDOW_AFTER_S)
    return (start, end) if end - start >= 60 else None


def _write_played(cfg, mix_id: str, found: dict, listed: dict, mix_len: float):
    """What part of each record the DJ used: the span in mix time, bounded by the neighbours, and the
    same span in the record's own time. This is the per-DJ record the profiles are built from."""
    order = sorted(found.items(), key=lambda kv: kv[1]["mix_t"])
    rows = []
    for i, (track_id, result) in enumerate(order):
        starts = result["mix_t"]
        ends = _track_end(result)
        next_start = order[i + 1][1]["mix_t"] if i + 1 < len(order) else mix_len
        audible_to = min(ends, max(next_start, starts) + 600, mix_len)
        rows.append({
            "track_id": track_id, "listed_mix_t": listed.get(track_id),
            "mix_t_of_track_zero": round(result["mix_t"], 2), "rate": result["rate"], "votes": result["votes"],
            "track_len_s": result.get("track_len_s"),
            "played_mix_from_s": round(max(starts, 0.0), 1), "played_mix_to_s": round(audible_to, 1),
            "played_track_from_s": round(max(0.0, (max(starts, 0.0) - result["mix_t"]) * result["rate"]), 1),
            "played_track_to_s": round(max(0.0, (audible_to - result["mix_t"]) * result["rate"]), 1),
        })
    out_dir = cfg.dirs["out"] / "played"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{mix_id}.json").write_text(json.dumps({"mix_id": mix_id, "mix_len_s": round(mix_len, 1),
                                                        "tracks": rows}, indent=1))
