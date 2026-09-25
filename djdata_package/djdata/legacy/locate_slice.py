"""The run's original locate, kept for the old fetch path and for `fred_match2.py`. Superseded.

It searches a seven-minute slice around the listed time, fingerprints the mix again for every track, and
its `mix_t` multiplies the offset by the rate after the record was already resampled, which reads 92.7 s
where the answer is 90.00 s (measured 2026-09-24). `djdata.seam.locate` replaces it. Do not build on this.
"""

from concurrent.futures import FIRST_COMPLETED, wait
from collections import defaultdict
import logging
import time

import librosa
import numpy as np
from scipy.ndimage import maximum_filter

from .track_fetcher import (
    DT_MAX_FRAMES, DT_MIN_FRAMES, FAN_OUT, PEAK_NEIGHBOURHOOD, PEAK_PERCENTILE, VERIFY_HOP, VERIFY_NFFT, VERIFY_SR,
)

log = logging.getLogger("djdata.legacy.locate_slice")

FRAME_S = VERIFY_HOP / VERIFY_SR
SEARCH_PAD_S = 210.0        # half-width of the mix slice searched around a listed start time
COARSE_RATES = (1.0, 0.99, 1.01, 0.98, 1.02, 0.97, 1.03, 0.96, 1.04)
WHOLE_MIX_RATES = COARSE_RATES
WHOLE_MIX_MAX_S = 3 * 3600                         # a set longer than this is not worth the search
FINE_STEP = 0.0025
MIN_VOTES = 40              # a wrong record scored at most 21 on the fetcher and 14 on seam windows
EARLY_ACCEPT = 400          # votes at the first rate that make the other rates pointless


def fingerprint(audio: np.ndarray) -> dict:
    """The fetcher's _fingerprint, on an array instead of a path. Same constants, same maths."""
    spec_db = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=VERIFY_NFFT, hop_length=VERIFY_HOP)))
    local_max = maximum_filter(spec_db, size=PEAK_NEIGHBOURHOOD, mode="nearest")
    freqs, frames = np.nonzero((spec_db == local_max) & (spec_db > np.percentile(spec_db, PEAK_PERCENTILE)))
    order = np.argsort(frames)
    freqs, frames = freqs[order], frames[order]
    table = defaultdict(list)
    n = len(frames)
    for i in range(n):
        for j in range(i + 1, min(i + 1 + FAN_OUT, n)):
            gap = int(frames[j] - frames[i])
            if gap < DT_MIN_FRAMES:
                continue
            if gap > DT_MAX_FRAMES:
                break
            table[(freqs[i] // 2, freqs[j] // 2, gap)].append(int(frames[i]))
    return table


def best_offset(slice_fp: dict, track_fp: dict) -> tuple[int, int]:
    """(votes at the best offset, that offset in frames = track_frame - slice_frame)."""
    offsets = [tf - sf for h, slice_frames in slice_fp.items() if h in track_fp
               for sf in slice_frames for tf in track_fp[h]]
    if not offsets:
        return 0, 0
    values, counts = np.unique(np.array(offsets), return_counts=True)
    best = int(np.argmax(counts))
    return int(counts[best]), int(values[best])


def load(path, offset=0.0, duration=None) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=VERIFY_SR, mono=True, offset=offset, duration=duration)
    return audio.astype(np.float32)


def at_mix_speed(track: np.ndarray, rate: float) -> np.ndarray:
    """The record as the mix plays it: `rate` times faster, every frequency `rate` times higher."""
    if abs(rate - 1.0) < 1e-6:
        return track
    return librosa.resample(track, orig_sr=int(round(VERIFY_SR * rate)), target_sr=VERIFY_SR)


def locate_one(job: dict) -> dict:
    """One (mix, track) search. Module level and self-contained so a process pool can run it.

    job: {track_id, track_path, mix_path, listed_mix_t or None, mix_duration_s}
    """
    listed = job.get("listed_mix_t")
    mix_len = job.get("mix_duration_s")
    if listed is not None and mix_len and not (0 <= listed <= mix_len):
        listed = None          # a time outside the mix is a scrape error, not a hint
    if listed is None:
        slice_start, slice_duration = 0.0, min(mix_len or WHOLE_MIX_MAX_S, WHOLE_MIX_MAX_S)
        rates = WHOLE_MIX_RATES
    else:
        slice_start = max(0.0, listed - SEARCH_PAD_S)
        slice_duration = 2 * SEARCH_PAD_S
        rates = COARSE_RATES
    result = {"track_id": job["track_id"], "listed_mix_t": listed, "found": False, "votes": 0}
    try:
        mix_slice = load(job["mix_path"], offset=slice_start, duration=slice_duration)
        if len(mix_slice) < 30 * VERIFY_SR:
            result["reason"] = "mix slice too short"
            return result
        slice_fp = fingerprint(mix_slice)
        track = load(job["track_path"])
        result["track_len_s"] = round(len(track) / VERIFY_SR, 1)

        best = {"votes": 0, "rate": 1.0, "offset": 0}
        for rate in rates:
            votes, offset = best_offset(slice_fp, fingerprint(at_mix_speed(track, rate)))
            if votes > best["votes"]:
                best = {"votes": votes, "rate": rate, "offset": offset}
            if rate == 1.0 and votes >= EARLY_ACCEPT:
                break
        if best["votes"] >= MIN_VOTES:
            for step in (-FINE_STEP, FINE_STEP):
                votes, offset = best_offset(slice_fp, fingerprint(at_mix_speed(track, best["rate"] + step)))
                if votes > best["votes"]:
                    best = {"votes": votes, "rate": best["rate"] + step, "offset": offset}

        result["votes"] = best["votes"]
        if best["votes"] < MIN_VOTES:
            result["reason"] = f"{best['votes']} votes, floor {MIN_VOTES}"
            return result
        # KNOWN BUG, kept as it ran: the offset is already in resampled-track frames, the rate factor is wrong
        result.update(found=True, rate=round(best["rate"], 4),
                      mix_t=round(slice_start - best["offset"] * FRAME_S * best["rate"], 2))
    except Exception as error:  # a single unreadable track must not lose the whole mix
        result["reason"] = f"{type(error).__name__}: {error}"
    return result


def locate_all(mix_path, entries: list[dict], pool=None, timeout_s: float | None = None) -> dict:
    """entries: [{track_id, track_path, listed_mix_t}] in play order. Returns {track_id: result}."""
    jobs = [{"mix_path": str(mix_path), **entry} for entry in entries]
    if pool is None:
        results = [locate_one(job) for job in jobs]
    else:
        futures = {pool.submit(locate_one, job): job for job in jobs}
        deadline = time.time() + timeout_s if timeout_s else None
        pending = set(futures)
        while pending:
            remaining = None if deadline is None else deadline - time.time()
            if remaining is not None and remaining <= 0:
                break
            _, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
        results = []
        for future, job in futures.items():
            if future.done() and not future.cancelled():
                try:
                    results.append(future.result())
                    continue
                except Exception as error:  # noqa: BLE001
                    results.append({"track_id": job["track_id"], "found": False, "votes": 0,
                                    "reason": f"{type(error).__name__}: {error}"})
                    continue
            future.cancel()
            results.append({"track_id": job["track_id"], "found": False, "votes": 0,
                            "reason": "timed out: the mix budget ran out before this track was searched"})
    found = sum(1 for r in results if r["found"])
    log.info("locate %s: %d of %d tracks found", mix_path, found, len(results))
    for r in results:
        if r["found"]:
            log.debug("  %s at mix %.1f s, rate %.4f, %d votes", r["track_id"], r["mix_t"], r["rate"], r["votes"])
        else:
            log.info("  %s NOT found: %s", r["track_id"], r.get("reason"))
    return {r["track_id"]: r for r in results}
