"""The resumable runner: mix workers, N track workers, M seam analysis processes, one download gate.

The mix worker waits while `max_pending_mixes` downloaded mixes still have unanalysed seams, so the
disk never fills with windows nobody is ready to use. Track workers take tracks in the order their
mixes landed. Seam workers take any seam whose window and both tracks exist, so analysis starts as
soon as the first mix's first pair of tracks is on disk. Every claim is an atomic sqlite update,
so a killed run resumes by restarting the command. Track downloads are paced through the Gate; a
YouTube block (fetch.yt.Blocked) puts the item back to pending and closes the gate until a probe
succeeds, so a block costs time, never a track.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import logging
from pathlib import Path
import threading
import time

from .config import Config
from .fetch import mix as fetch_mix
from .fetch import track as fetch_track
from .fetch.yt import Blocked, probe
from .gate import Gate
from .seam.analyse import analyse
from .seam.coarse import fingerprint_anchors
from .state import State

log = logging.getLogger("djdata.run")
POLL_S = 3.0


def _mix_worker(cfg: Config, state: State, stop: threading.Event, gate: Gate):
    name = threading.current_thread().name
    tiers = cfg.run_tiers
    while not stop.is_set():
        if state.pending_unanalysed_mixes(tiers) >= cfg.workers["max_pending_mixes"]:
            time.sleep(POLL_S)
            continue
        mix = state.claim_mix(tiers, name)
        if mix is None:
            time.sleep(POLL_S)
            continue
        t0 = time.time()
        try:
            if not mix["url"]:
                raise RuntimeError("no audio url (run media-links first)")
            n = fetch_mix.process_mix(cfg, state, mix)
            state.set_mix(mix["mix_id"], "windows_ready")
            log.info("mix done %s: %d windows in %.1fs", mix["mix_id"], n, time.time() - t0)
        except Blocked as e:
            state.set_mix(mix["mix_id"], "pending", error=str(e)[:500])
            gate.blocked(str(e))
            gate.wait_open()
        except Exception as e:  # one bad mix must not stop the run; its seams are marked with the reason
            state.set_mix(mix["mix_id"], "failed", error=str(e)[:500])
            for s in state.seams_of_mix(mix["mix_id"]):
                state.set_seam(s["seam_id"], "failed", error=f"mix: {str(e)[:200]}")
            log.error("mix FAILED %s after %.1fs: %s", mix["mix_id"], time.time() - t0, e)


def _track_worker(cfg: Config, state: State, stop: threading.Event, gate: Gate):
    name = threading.current_thread().name
    tiers = cfg.run_tiers
    while not stop.is_set():
        gate.wait_open()
        track = state.claim_track(tiers, name)
        if track is None:
            time.sleep(POLL_S)
            continue
        gate.pace()
        t0 = time.time()
        try:
            path, dur = fetch_track.fetch(cfg, track)
            state.set_track(track["track_id"], "ready", path=str(path), duration=dur)
            log.info("track done %s (%.0fs audio) in %.1fs", track["track_id"], dur, time.time() - t0)
        except Blocked as e:
            state.set_track(track["track_id"], "pending", error=str(e)[:500])
            log.warning("track %s blocked, back to pending: %s", track["track_id"], e)
            gate.blocked(str(e))
        except Exception as e:
            state.set_track(track["track_id"], "failed", error=str(e)[:500])
            log.error("track FAILED %s after %.1fs: %s", track["track_id"], time.time() - t0, e)
            _fail_seams_of_track(state, track["track_id"], str(e))


def _fail_seams_of_track(state: State, track_id: str, reason: str):
    c = state._conn()
    c.execute("UPDATE seams SET status='failed', error=?, updated=? WHERE (a=? OR b=?) AND status NOT IN ('done','failed')",
              (f"track: {reason[:200]}", time.time(), track_id, track_id))


def _analyse_job(seam: dict, cfg_an: dict, curves_dir: str) -> dict:
    """Runs in a worker process. Returns the result row; writes the curves file."""
    coarse = json.loads(seam["coarse"])
    if coarse.get("needs_fingerprint"):
        coarse = fingerprint_anchors(Path(seam["window_path"]), Path(seam["a_path"]), Path(seam["b_path"]), coarse)
    res = analyse(seam["window_path"], seam["a_path"], seam["b_path"], coarse, cfg_an)
    import csv

    rows = res["curves"]
    with open(Path(curves_dir) / f"{seam['seam_id']}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return res["row"]


def _seam_scheduler(cfg: Config, state: State, stop: threading.Event):
    tiers = cfg.run_tiers
    n_workers = cfg.workers["seams"]
    out_path = cfg.dirs["out"] / "seams.jsonl"
    in_flight = {}
    # spawn, not fork: forked children inherit the parent's thread locks (logging, sqlite, BLAS) and
    # can deadlock at exit, which once left the pool's shutdown hanging for the whole join timeout.
    pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        while not stop.is_set() or in_flight:
            while len(in_flight) < n_workers and not stop.is_set():
                seam = state.claim_seam(tiers, "seam-pool")
                if seam is None:
                    break
                fut = pool.submit(_analyse_job, seam, cfg.analysis, str(cfg.dirs["curves"]))
                in_flight[fut] = (seam, time.time())
                log.info("seam start %s", seam["seam_id"])
            done = [f for f in in_flight if f.done()]
            for f in done:
                seam, t0 = in_flight.pop(f)
                try:
                    row = f.result()
                    row.update({"seam_id": seam["seam_id"], "mix_id": seam["mix_id"], "a": seam["a"], "b": seam["b"],
                                "tier": seam["tier"], "source": seam["source"]})
                    with open(out_path, "a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    state.set_seam(seam["seam_id"], "done")
                    log.info("seam done %s in %.1fs: overlap %s bars, swap %s", seam["seam_id"], time.time() - t0,
                             row.get("overlap_bars"), row.get("bass_swap_t"))
                except Exception as e:
                    state.set_seam(seam["seam_id"], "failed", error=str(e)[:500])
                    log.error("seam FAILED %s after %.1fs: %s", seam["seam_id"], time.time() - t0, e)
            if not done:
                time.sleep(POLL_S)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        log.info("seam scheduler stopped")


def run(cfg: Config) -> dict:
    state = State(cfg.db_path)
    state.release_stale()
    tiers = cfg.run_tiers
    log.info("run start: tiers %s, workers %s, counts %s", tiers, cfg.workers, state.counts(tiers))
    stop = threading.Event()
    gate = Gate(cfg.download["min_interval_s"], cfg.download["block_wait_s"],
                lambda: probe(cfg, cfg.dirs["tracks"]), stop)
    threads = [threading.Thread(target=_mix_worker, args=(cfg, state, stop, gate), name=f"mix-{i}", daemon=True)
               for i in range(cfg.workers["mix"])]
    threads += [threading.Thread(target=_track_worker, args=(cfg, state, stop, gate), name=f"track-{i}", daemon=True)
                for i in range(cfg.workers["tracks"])]
    sched = threading.Thread(target=_seam_scheduler, args=(cfg, state, stop), name="seam-sched", daemon=True)
    for t in threads:
        t.start()
    sched.start()
    last = 0.0
    try:
        while state.work_left(tiers):
            time.sleep(POLL_S)
            if time.time() - last > 60:
                log.info("progress %s", state.counts(tiers))
                last = time.time()
    except KeyboardInterrupt:
        log.warning("interrupted; claimed items are released on the next start")
    finally:
        stop.set()
        sched.join(timeout=600)
    counts = state.counts(tiers)
    counts["youtube_blocks"] = gate.blocks
    log.info("run end: %s", counts)
    return counts
