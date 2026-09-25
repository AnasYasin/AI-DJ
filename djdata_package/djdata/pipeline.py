"""The stages, in order, each one function, each resumable, each writing its own table under out/.

    locate    every listed record found in its mix, with a per-mix control floor      -> plays.csv
    pairs     which record follows which, from what was heard, with the window bounds -> seams.csv
    cut       one window per usable seam, stream copied, audited at both ends         -> cuts.csv
    measure   per band entry and exit, bass swap and words, presence, loop            -> measures.csv
    tempo     BPM per record from its own audio                                       -> tempos.csv
    label     the nine transition types, in bars                                      -> labels.csv
    export    one flat seams table joining all of the above                           -> seams_index.csv

Every stage takes the config and returns a small dict of counts that the CLI prints. A stage skips the
keys its table already holds, so a killed run is resumed with the same command. Work that needs a
process per item runs in a spawn pool, and the BLAS thread count is pinned to one before the pool
starts: twelve workers once made 198 threads on 16 cores and the run made no progress for 97 minutes.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
import os
from pathlib import Path
import time

from . import logs
from .config import Config
from .seam import cut as cut_mod
from .seam import eartest
from .seam import labels as labels_mod
from .seam import locate as locate_mod
from .seam import measure as measure_mod
from .seam import pairs as pairs_mod
from .seam import tempo as tempo_mod
from .seam.fingerprint import load as load_audio
from .sources import raveform, tracklists
from .store import tables
from .store.tables import num

log = logging.getLogger("djdata.pipeline")

N_CONTROLS = 3


def _one_thread_blas():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")


def _pool(cfg: Config, workers: int) -> ProcessPoolExecutor:
    """Spawned workers start with no logging, so each one sets up the same console and file log."""
    _one_thread_blas()
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=logs.setup,
        initargs=(cfg.dirs["logs"], cfg.log_level),
    )


def corpus(cfg: Config, only: list[str] | None = None) -> list[dict]:
    """The mixes this config's corpus holds on disk, with their records."""
    if cfg.source == "tracklists":
        return tracklists.mixes_on_disk(cfg.root, cfg.tracklists_csv, only)
    if cfg.source == "raveform":
        return raveform.mixes_on_disk(cfg, only)
    raise ValueError(f"no corpus reader for source {cfg.source!r}")


# ---------------------------------------------------------------- locate


def _locate_job(job: dict) -> list[dict]:
    """One mix in one process. Returns plays rows, records and controls alike."""
    got = locate_mod.locate_mix(job["path"], job["tracks"], job["controls"])
    floor = got["floor"]
    rows = []
    for entry, found in zip(job["tracks"] + job["controls"], got["records"] + got["controls"]):
        is_control = entry.get("is_control", 0)
        conf = (
            "control"
            if is_control
            else locate_mod.confidence(found.votes, floor, found.sections_agree)
        )
        listed_min = entry.get("listed_min")
        drift = None
        if listed_min is not None and not is_control:
            drift = round(found.time_zero_s / 60 - listed_min, 2)
        rows.append(
            {
                "mix_id": job["mix_id"],
                "dj": job["dj"],
                "genre": job["genre"],
                "mix_title": job["mix_title"],
                "mix_minutes": round(got["mix_len_s"] / 60, 1),
                "track_id": entry["track_id"],
                "title": entry.get("title", ""),
                "is_control": is_control,
                "order_listed": entry.get("order_listed", ""),
                "listed_min": listed_min,
                "play_type": entry.get("play_type", ""),
                "overlay_parent": entry.get("overlay_parent", ""),
                "time_zero_s": found.time_zero_s,
                "rate": found.rate,
                "votes": found.votes,
                "floor": floor.floor,
                "control_max": floor.control_max,
                "control_n": floor.control_n,
                "sections": found.sections,
                "sections_agree": found.sections_agree,
                "confidence": conf,
                "found": int(conf not in ("control", "not found")),
                "drift_min": drift,
                "first_heard_s": found.first_heard_s,
                "last_heard_s": found.last_heard_s,
                "played_from_s": found.played_from_s,
                "played_to_s": found.played_to_s,
                "sweep_votes": found.sweep_votes,
                "sweep_floor": found.sweep_floor,
                "track_len_s": found.track_len_s,
                "seconds": found.seconds,
            }
        )
    return rows


def locate(cfg: Config, workers: int = 1, only: list[str] | None = None) -> dict:
    """Stage 1. One process per mix, the mix fingerprinted once, every listed record and three control
    records from other mixes looked up in it. Writes plays.csv, one row per record and per control."""
    table = tables.plays(cfg.dirs["out"])
    done = {k[0] for k in table.keys()}
    if cfg.source == "raveform":
        rows = [r for r in raveform.plays_rows(cfg, only) if r["mix_id"] not in done]
        table.append(rows)
        n_mixes = len({r["mix_id"] for r in rows})
        log.info(
            "locate: raveform alignment stands in, %d records in %d mixes", len(rows), n_mixes
        )
        return {
            "mixes": n_mixes,
            "done_before": len(done),
            "ran": 0,
            "failed": 0,
            "records": len(rows),
            "seconds": 0.0,
        }
    mixes = corpus(cfg, only)
    pool_tracks = tracklists.track_pool(mixes)
    jobs = []
    for m in mixes:
        if m["mix_id"] in done:
            continue
        if not m["tracks"]:
            log.warning("locate %s: no track audio on disk, skipped", m["mix_id"])
            continue
        controls = [
            {**c, "is_control": 1} for c in tracklists.controls_for(m, pool_tracks, N_CONTROLS)
        ]
        jobs.append({**m, "controls": controls})
    log.info(
        "locate: %d mixes on disk, %d already done, %d to run at %d workers",
        len(mixes),
        len(done),
        len(jobs),
        workers,
    )
    t0 = time.time()
    counts = {"mixes": len(mixes), "done_before": len(done), "ran": 0, "failed": 0, "records": 0}
    with _pool(cfg, workers) as pool:
        futures = {pool.submit(_locate_job, j): j["mix_id"] for j in jobs}
        for fut in as_completed(futures):
            mix_id = futures[fut]
            try:
                rows = fut.result()
            except Exception as error:  # one bad mix must not stop the run
                counts["failed"] += 1
                log.error("locate FAILED %s: %s: %s", mix_id, type(error).__name__, error)
                continue
            table.append(rows)
            own = [r for r in rows if not r["is_control"]]
            found = sum(r["found"] for r in own)
            counts["ran"] += 1
            counts["records"] += len(own)
            log.info(
                "locate done %s %s: %d of %d found, floor %s, %.0f s",
                mix_id,
                rows[0]["dj"],
                found,
                len(own),
                rows[0]["floor"],
                sum(r["seconds"] for r in rows),
            )
    counts["seconds"] = round(time.time() - t0, 1)
    return counts


# ---------------------------------------------------------------- pairs


def _by_mix(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        out.setdefault(r["mix_id"], []).append(r)
    return out


def pairs(cfg: Config) -> dict:
    """Stage 2. From plays.csv, every consecutive pair of heard records per mix, in audio order, with
    the adjacency flags and the bounded window. Writes seams.csv. A mix already in it is skipped."""
    plays_t = tables.plays(cfg.dirs["out"])
    seams_t = tables.seams(cfg.dirs["out"])
    done = {r["mix_id"] for r in seams_t.rows()}
    counts = {"mixes": 0, "seams": 0, "usable": 0}
    for mix_id, rows in _by_mix(plays_t.rows()).items():
        if mix_id in done:
            continue
        mix_len = num(rows[0]["mix_minutes"]) * 60.0
        got = pairs_mod.pairs_for_mix(rows, mix_len)
        if cfg.source == "raveform":
            got = _adopt_raveform_windows(cfg, got)
        seams_t.append([p.as_row() for p in got])
        counts["mixes"] += 1
        counts["seams"] += len(got)
        counts["usable"] += sum(p.usable for p in got)
        log.info(
            "pairs done %s: %d seams, %d usable", mix_id, len(got), sum(p.usable for p in got)
        )
    return counts


def _adopt_raveform_windows(cfg: Config, got: list) -> list:
    """Raveform seams keep the window the manifest cut, when that file is on disk. A pair with no such
    window is kept as a row, not usable, with the reason."""
    windows = raveform.seam_windows(cfg.db_path)
    for p in got:
        w = windows.get(p.seam_id)
        if w is None or cut_mod.existing_window(cfg.dirs["windows"], p.seam_id) is None:
            p.usable, p.reason = 0, "no raveform window on disk"
            continue
        p.window_t0, p.window_t1 = round(w["window_t0"], 1), round(w["window_t1"], 1)
        p.window_s = round(p.window_t1 - p.window_t0, 1)
    return got


# ---------------------------------------------------------------- cut


def _track_paths(cfg: Config) -> dict:
    return {t["track_id"]: t["path"] for m in corpus(cfg) for t in m["tracks"]}


def _cut_job(job: dict) -> dict:
    seam = job["seam"]
    t0, t1 = num(seam["window_t0"]), num(seam["window_t1"])
    if job["mix_path"] is None:
        dest = cut_mod.existing_window(job["dest_stem"].parent, seam["seam_id"])
        status = "adopted"
    else:
        dest = cut_mod.cut_window(job["mix_path"], t0, t1, job["dest_stem"])
        status = "ok"
    actual = cut_mod.duration_s(dest)
    row = {
        "seam_id": seam["seam_id"],
        "window_file": dest.name,
        "asked_s": round(t1 - t0, 2),
        "actual_s": round(actual, 2),
        "cut_error_s": round(actual - (t1 - t0), 3),
        "status": status,
    }
    a = {"time_zero_s": num(seam["a_time_zero_s"]), "rate": num(seam["a_rate"])}
    b = {"time_zero_s": num(seam["b_time_zero_s"]), "rate": num(seam["b_rate"])}
    row.update(
        cut_mod.audit(
            dest,
            t0,
            a,
            b,
            load_audio(job["a_path"]),
            load_audio(job["b_path"]),
            [load_audio(c) for c in job["controls"]],
        )
    )
    return row


def cut(cfg: Config, workers: int = 1) -> dict:
    """Stage 3. One window per usable seam from seams.csv, stream copied into windows/, then the ten
    second audit at both ends. Writes cuts.csv."""
    seams_t = tables.seams(cfg.dirs["out"])
    cuts_t = tables.cuts(cfg.dirs["out"])
    done = cuts_t.keys()
    mixes = {m["mix_id"]: m for m in corpus(cfg)}
    paths = _track_paths(cfg)
    pool_tracks = tracklists.track_pool(list(mixes.values()))
    jobs = []
    for seam in seams_t.rows():
        if seam["seam_id"] in done or seam["usable"] != "1":
            continue
        mix = mixes.get(seam["mix_id"])
        if mix is None or seam["track_a"] not in paths or seam["track_b"] not in paths:
            log.warning("cut %s: audio missing, skipped", seam["seam_id"])
            continue
        if (
            mix["path"] is None
            and cut_mod.existing_window(cfg.dirs["windows"], seam["seam_id"]) is None
        ):
            log.warning("cut %s: no full mix and no window on disk, skipped", seam["seam_id"])
            continue
        controls = tracklists.controls_for(mix, pool_tracks, N_CONTROLS)
        jobs.append(
            {
                "seam": seam,
                "mix_path": mix["path"],
                "dest_stem": cfg.dirs["windows"] / seam["seam_id"],
                "a_path": paths[seam["track_a"]],
                "b_path": paths[seam["track_b"]],
                "controls": [c["path"] for c in controls],
            }
        )
    log.info("cut: %d done before, %d to cut at %d workers", len(done), len(jobs), workers)
    counts = {"done_before": len(done), "cut": 0, "failed": 0, "start_ok": 0, "end_ok": 0}
    t0 = time.time()
    with _pool(cfg, workers) as pool:
        futures = {pool.submit(_cut_job, j): j["seam"]["seam_id"] for j in jobs}
        for fut in as_completed(futures):
            seam_id = futures[fut]
            try:
                row = fut.result()
            except Exception as error:
                counts["failed"] += 1
                log.error("cut FAILED %s: %s: %s", seam_id, type(error).__name__, error)
                continue
            cuts_t.append([row])
            counts["cut"] += 1
            counts["start_ok"] += row["audit_start_ok"]
            counts["end_ok"] += row["audit_end_ok"]
            log.info(
                "cut done %s: %.0f s, start %s end %s",
                seam_id,
                row["actual_s"],
                "ok" if row["audit_start_ok"] else "NOT A alone",
                "ok" if row["audit_end_ok"] else "NOT B alone",
            )
    counts["seconds"] = round(time.time() - t0, 1)
    return counts


# ---------------------------------------------------------------- measure


def _measure_job(job: dict) -> dict:
    t0 = time.time()
    row, _ = measure_mod.measure_one(job)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def measure(cfg: Config, workers: int = 1) -> dict:
    """Stage 4. Every cut window measured: per band entry and exit, bass swap and words, full-band
    presence and loop, every floor from three controls on that window. Writes measures.csv."""
    seams = {r["seam_id"]: r for r in tables.seams(cfg.dirs["out"]).rows()}
    cuts_t = tables.cuts(cfg.dirs["out"])
    measures_t = tables.measures(cfg.dirs["out"])
    done = measures_t.keys()
    mixes = {m["mix_id"]: m for m in corpus(cfg)}
    paths = _track_paths(cfg)
    pool_tracks = tracklists.track_pool(list(mixes.values()))
    jobs = []
    for c in cuts_t.rows():
        seam = seams.get(c["seam_id"])
        if seam is None or c["seam_id"] in done or c["status"] not in ("ok", "adopted"):
            continue
        mix = mixes.get(seam["mix_id"])
        if mix is None or seam["track_a"] not in paths or seam["track_b"] not in paths:
            continue
        controls = tracklists.controls_for(mix, pool_tracks, N_CONTROLS)
        jobs.append(
            {
                "seam_id": seam["seam_id"],
                "window_path": cfg.dirs["windows"] / c["window_file"],
                "window_t0": num(seam["window_t0"]),
                "a": {
                    "path": paths[seam["track_a"]],
                    "time_zero_s": num(seam["a_time_zero_s"]),
                    "rate": num(seam["a_rate"]),
                },
                "b": {
                    "path": paths[seam["track_b"]],
                    "time_zero_s": num(seam["b_time_zero_s"]),
                    "rate": num(seam["b_rate"]),
                },
                "controls": [c["path"] for c in controls],
            }
        )
    log.info("measure: %d done before, %d to measure at %d workers", len(done), len(jobs), workers)
    counts = {"done_before": len(done), "measured": 0, "failed": 0, "both_clear": 0}
    t0 = time.time()
    with _pool(cfg, workers) as pool:
        futures = {pool.submit(_measure_job, j): j["seam_id"] for j in jobs}
        for fut in as_completed(futures):
            seam_id = futures[fut]
            try:
                row = fut.result()
            except Exception as error:
                counts["failed"] += 1
                log.error("measure FAILED %s: %s: %s", seam_id, type(error).__name__, error)
                continue
            measures_t.append([row])
            counts["measured"] += 1
            counts["both_clear"] += row["measured"]
            log.info(
                "measure done %s: sep A %.1f B %.1f, in %s out %s, bass swap %s, %.0f s",
                seam_id,
                row["a_best_separation"],
                row["b_best_separation"],
                row["in_s"],
                row["out_s"],
                row["bass_swap_s"],
                row["seconds"],
            )
    counts["seconds"] = round(time.time() - t0, 1)
    return counts


# ---------------------------------------------------------------- tempo


def _tempo_job(job: dict) -> dict:
    t0 = time.time()
    return {
        "track_id": job["track_id"],
        "bpm": tempo_mod.bpm_of(job["path"]),
        "seconds": round(time.time() - t0, 1),
    }


def tempo(cfg: Config, workers: int = 1) -> dict:
    """Stage 5. BPM for every record that appears in a seam. Writes tempos.csv."""
    seams_t = tables.seams(cfg.dirs["out"])
    tempos_t = tables.tempos(cfg.dirs["out"])
    done = tempos_t.keys()
    paths = _track_paths(cfg)
    wanted = set()
    for s in seams_t.rows():
        wanted.update((s["track_a"], s["track_b"]))
    jobs = [
        {"track_id": t, "path": paths[t]} for t in sorted(wanted) if t in paths and t not in done
    ]
    log.info("tempo: %d done before, %d to measure at %d workers", len(done), len(jobs), workers)
    counts = {"done_before": len(done), "measured": 0, "no_tempo": 0, "failed": 0}
    with _pool(cfg, workers) as pool:
        futures = {pool.submit(_tempo_job, j): j["track_id"] for j in jobs}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as error:
                counts["failed"] += 1
                log.error("tempo FAILED %s: %s: %s", futures[fut], type(error).__name__, error)
                continue
            tempos_t.append([row])
            counts["measured"] += 1
            counts["no_tempo"] += row["bpm"] is None
    return counts


# ---------------------------------------------------------------- label


def label(cfg: Config) -> dict:
    """Stage 6. The nine transition types per measured seam, in bars from the outgoing record's tempo
    at the rate it was played (the incoming record's when the outgoing has none). Writes labels.csv."""
    seams = {r["seam_id"]: r for r in tables.seams(cfg.dirs["out"]).rows()}
    bpm = {r["track_id"]: num(r["bpm"]) for r in tables.tempos(cfg.dirs["out"]).rows()}
    labels_t = tables.labels(cfg.dirs["out"])
    done = labels_t.keys()
    counts = {"labelled": 0, "unmeasured": 0, "by_label": {}}
    for m in tables.measures(cfg.dirs["out"]).rows():
        if m["seam_id"] in done or m["seam_id"] not in seams:
            continue
        seam = seams[m["seam_id"]]
        bar_s = tempo_mod.bar_seconds(bpm.get(seam["track_a"]), num(seam["a_rate"]))
        if bar_s is None:
            bar_s = tempo_mod.bar_seconds(bpm.get(seam["track_b"]), num(seam["b_rate"]))
        numbers = {k: num(v) for k, v in m.items()}
        got = labels_mod.label(m["seam_id"], numbers, bar_s)
        labels_t.append([got.as_row()])
        counts["labelled"] += 1
        counts["unmeasured"] += got.label == "unmeasured"
        counts["by_label"][got.label] = counts["by_label"].get(got.label, 0) + 1
    return counts


# ---------------------------------------------------------------- export


def export(cfg: Config) -> dict:
    """Stage 7. One flat row per seam joining seams, cuts, measures and labels, for the notebooks.
    Rewritten every time. Writes seams_index.csv."""
    import csv

    out = cfg.dirs["out"]
    joined = {}
    for t, prefix in (
        (tables.seams(out), ""),
        (tables.cuts(out), ""),
        (tables.measures(out), ""),
        (tables.labels(out), ""),
    ):
        for r in t.rows():
            joined.setdefault(r["seam_id"], {}).update({f"{prefix}{k}": v for k, v in r.items()})
    columns = []
    for cols in (tables.SEAMS, tables.CUTS, tables.MEASURES, tables.LABELS):
        columns += [c for c in cols if c not in columns]
    path = out / "seams_index.csv"
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=columns)
        w.writeheader()
        for r in joined.values():
            w.writerow({c: r.get(c, "") for c in columns})
    log.info("export: %d seams to %s", len(joined), path)
    return {"seams": len(joined), "file": str(path)}


# ---------------------------------------------------------------- ear test


def ear_test(cfg: Config, out_dir, n: int = 10, label: str | None = None, seed: int = 25) -> dict:
    """Clips for Anas from the tables: `n` measured seams at random, or `n` per `label`. Left ear the
    window, right ear both records placed as located. Writes the clips and MARKS.csv to `out_dir`."""
    import random

    out_dir = Path(out_dir)
    seams = {r["seam_id"]: r for r in tables.seams(cfg.dirs["out"]).rows()}
    cuts = {r["seam_id"]: r for r in tables.cuts(cfg.dirs["out"]).rows()}
    measures = {r["seam_id"]: r for r in tables.measures(cfg.dirs["out"]).rows()}
    labels = {r["seam_id"]: r for r in tables.labels(cfg.dirs["out"]).rows()}
    paths = _track_paths(cfg)
    pool = [sid for sid in measures if sid in cuts and sid in seams]
    if label is not None:
        pool = [sid for sid in pool if labels.get(sid, {}).get("label") == label]
    rng = random.Random(seed)
    rng.shuffle(pool)
    rows = []
    for i, sid in enumerate(pool[:n], 1):
        seam, c, m = seams[sid], cuts[sid], measures[sid]
        name = f"{i:02d}_{seam['dj'].replace(' ', '')}_{sid}.wav"
        a = {
            "path": paths[seam["track_a"]],
            "time_zero_s": num(seam["a_time_zero_s"]),
            "rate": num(seam["a_rate"]),
        }
        b = {
            "path": paths[seam["track_b"]],
            "time_zero_s": num(seam["b_time_zero_s"]),
            "rate": num(seam["b_rate"]),
        }
        length = eartest.clip(
            cfg.dirs["windows"] / c["window_file"], num(seam["window_t0"]), a, b, out_dir / name
        )
        rows.append(
            {
                "n": i,
                "file": name,
                "seam_id": sid,
                "dj": seam["dj"],
                "track_a": seam["track_a"],
                "track_b": seam["track_b"],
                "window_s": round(length, 1),
                "in_s": m["in_s"],
                "out_s": m["out_s"],
                "bass_swap_s": m["bass_swap_s"],
                "label": labels.get(sid, {}).get("label", ""),
                "a_confidence": seam["a_confidence"],
                "b_confidence": seam["b_confidence"],
                "audit_start_ok": c["audit_start_ok"],
                "audit_end_ok": c["audit_end_ok"],
            }
        )
        log.info("ear test %s written, %.0f s", name, length)
    eartest.write_marks(out_dir, rows)
    return {"clips": len(rows), "dir": str(out_dir)}
