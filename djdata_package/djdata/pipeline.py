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
from .seam import eartest, floors
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
        mixes = tracklists.mixes_on_disk(
            cfg.root, cfg.tracklists_csv, only, cfg.raw.get("extra_tracklists")
        )
        excluded = tracklists.excluded_mixes(cfg.raw.get("excluded_mixes_csv"))
        return [m for m in mixes if m["mix_id"] not in excluded]
    if cfg.source == "raveform":
        return raveform.mixes_on_disk(cfg, only)
    raise ValueError(f"no corpus reader for source {cfg.source!r}")


def _control_pool(cfg: Config, mixes: list[dict]) -> list[dict]:
    """Where a corpus's control records come from. `control_djs` in the config names other DJs whose
    records serve; without it the corpus's own other mixes do."""
    control_djs = cfg.raw.get("control_djs")
    if not control_djs or cfg.source != "tracklists":
        return tracklists.track_pool(mixes)
    own = {m["dj"] for m in mixes}
    return tracklists.other_dj_pool(cfg.root, cfg.tracklists_csv, control_djs, own)


def _own_dj_exclusions(cfg: Config) -> dict:
    """dj -> every record that DJ lists anywhere, for the DJs named in `controls_exclude_own_dj`.

    Why (2026-09-28). A DJ who plays his own records night after night (Fred again..'s tour USB) can
    get a control that is really in the mix, and the floor rises over true plays. Off for every DJ not
    named, so the DJs tested before keep their controls exactly."""
    djs = cfg.raw.get("controls_exclude_own_dj") or []
    extra = cfg.raw.get("extra_tracklists")
    return {dj: tracklists.dj_records(cfg.tracklists_csv, [dj], extra) for dj in djs}


def _controls(mix: dict, pool: list[dict], exclusions: dict) -> list[dict]:
    return tracklists.controls_for(mix, pool, N_CONTROLS, exclusions.get(mix["dj"]))


# ---------------------------------------------------------------- locate


def _probe(path) -> dict:
    """codec, profile, bitrate, sample rate, channels and length of an audio file, by ffprobe."""
    import json
    import subprocess

    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,profile,bit_rate,sample_rate,channels:format=duration,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        info = json.loads(out.stdout)
    except ValueError:
        return {}
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format", {})
    return {
        "codec": stream.get("codec_name", ""),
        "profile": stream.get("profile", ""),
        "bitrate": stream.get("bit_rate") or fmt.get("bit_rate", ""),
        "sample_rate": stream.get("sample_rate", ""),
        "channels": stream.get("channels", ""),
        "duration_s": round(float(fmt["duration"]), 1) if fmt.get("duration") else "",
    }


def _locate_job(job: dict) -> dict:
    """One mix in one process. Returns {plays, presence, mix}: plays rows for records and controls,
    presence rows (every window where a record beats the loudest control window), one mixes row."""
    got = locate_mod.locate_mix(job["path"], job["tracks"], job["controls"])
    floor = got["floor"]
    rows = []
    for entry, found in zip(job["tracks"] + job["controls"], got["records"] + got["controls"]):
        is_control = entry.get("is_control", 0)
        if is_control:
            conf = "control"
        elif found.search == "window":
            # a windowed record is judged on its window votes against the controls in that window
            wf = floors.Floor(
                floor=found.window_floor,
                control_max=found.window_control_max,
                control_n=len(job["controls"]),
                control_votes=(),
                margin=floors.MARGIN,
                trusted=len(job["controls"]) >= floors.MIN_CONTROLS,
            )
            # sections count only when they were searched in the window, which is when the votes
            # there cleared the window floor; a record not in its window is not found, whatever it
            # scores elsewhere in the mix (its presence rows still show where it plays)
            conf = (
                locate_mod.confidence(found.window_votes, wf, found.sections_agree)
                if wf.clears(found.window_votes)
                else "not found"
            )
        else:
            conf = locate_mod.confidence(found.votes, floor, found.sections_agree)
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
                "search": found.search,
                "near_s": found.near_s,
                "window_votes": found.window_votes,
                "window_floor": found.window_floor,
                "window_control_max": found.window_control_max,
            }
        )
    presence_rows, seen = [], set()
    # the presence floor the records were judged on (a control itself is located with the minimum)
    sweep_floor = got["records"][0].sweep_floor if got["records"] else None
    for found in got["records"]:
        if found.track_id in seen:
            continue  # a record listed twice is one record in the audio
        seen.add(found.track_id)
        for start_s, votes, offset in found.windows:
            presence_rows.append(
                {
                    "mix_id": job["mix_id"],
                    "track_id": found.track_id,
                    "window_start_s": start_s,
                    "votes": votes,
                    "state": "present" if votes >= found.sweep_floor else "weak",
                    # the part of the record playing at the window start, along that window's offset
                    "record_at_s": round((start_s - offset * locate_mod.FRAME_S) * found.rate, 1),
                    "sweep_floor": found.sweep_floor,
                    "weak_level": got["weak_level"],
                }
            )
    mix_row = {
        "mix_id": job["mix_id"],
        "dj": job["dj"],
        "file": Path(job["path"]).name,
        **_probe(job["path"]),
        "floor": floor.floor,
        "sweep_floor": sweep_floor,
        "weak_level": got["weak_level"],
        "control_n": floor.control_n,
    }
    return {"plays": rows, "presence": presence_rows, "mix": mix_row}


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
    pool_tracks = _control_pool(cfg, mixes)
    exclusions = _own_dj_exclusions(cfg)
    jobs = []
    for m in mixes:
        if m["mix_id"] in done:
            continue
        if not m["tracks"]:
            log.warning("locate %s: no track audio on disk, skipped", m["mix_id"])
            continue
        controls = [{**c, "is_control": 1} for c in _controls(m, pool_tracks, exclusions)]
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
                got = fut.result()
            except Exception as error:  # one bad mix must not stop the run
                counts["failed"] += 1
                log.error("locate FAILED %s: %s: %s", mix_id, type(error).__name__, error)
                continue
            rows = got["plays"]
            tables.presence(cfg.dirs["out"]).append(got["presence"])
            tables.mixes(cfg.dirs["out"]).append([got["mix"]])
            table.append(rows)  # last, so a mix counts as done only once all three are written
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


# ---------------------------------------------------------------- prove


def lists_1001(cfg: Config, djs: list[str]) -> list[dict]:
    """PROOF_LISTS rows for every 1001 mix by these DJs, one list per mix id, track paths left to the
    glob in tracks/."""
    out = []
    for mix_id, rows in tracklists.listed(cfg.tracklists_csv).items():
        if rows[0]["dj"] not in djs:
            continue
        for r in rows:
            out.append(
                {
                    "list_id": mix_id,
                    "dj": r["dj"],
                    "source": "1001",
                    "order_listed": r["order_listed"],
                    "track_id": r["track_id"],
                    "title": r["title"],
                    "listed_min": r["listed_min"],
                    "track_path": "",
                }
            )
    return out


def _prove_job(job: dict) -> list[dict]:
    """One candidate file in one process: every record of the lists paired with it, and the controls."""
    got = locate_mod.locate_mix(job["path"], job["tracks"], job["controls"])
    floor = got["floor"]
    by_track = {t["track_id"]: f for t, f in zip(job["tracks"], got["records"])}
    base = {
        "file_id": job["file_id"],
        "floor": floor.floor,
        "control_max": floor.control_max,
        "control_n": floor.control_n,
        "file_minutes": round(got["mix_len_s"] / 60, 1),
    }

    def row(located, **extra):
        return {
            **base,
            **extra,
            "votes": located.votes,
            "rate": located.rate,
            "time_zero_s": located.time_zero_s,
            "sections": located.sections,
            "sections_agree": located.sections_agree,
            "first_heard_s": located.first_heard_s,
            "last_heard_s": located.last_heard_s,
            "sweep_floor": located.sweep_floor,
            "seconds": located.seconds,
        }

    rows = []
    for entry in job["entries"]:
        found = by_track[entry["track_id"]]
        conf = locate_mod.confidence(found.votes, floor, found.sections_agree)
        rows.append(
            row(
                found,
                list_id=entry["list_id"],
                order_listed=entry["order_listed"],
                track_id=entry["track_id"],
                title=entry["title"],
                is_control=0,
                confidence=conf,
                found=int(conf != "not found"),
            )
        )
    for c, found in zip(job["controls"], got["controls"]):
        rows.append(
            row(
                found,
                list_id="",
                order_listed="",
                track_id=c["track_id"],
                title="",
                is_control=1,
                confidence="control",
                found=0,
            )
        )
    return rows


def prove(cfg: Config, candidates_csv, lists_csv, pairs_csv=None, workers: int = 1) -> dict:
    """Which candidate file is which show. Each file is fingerprinted once and every record of every
    list paired with it is looked up, with three controls by other DJs (`control_djs`). Pairs default
    to every list against every file. Writes proofs.csv, one row per file, list and listed record, and
    rewrites proof_summary.csv, one row per list and file (see `seam/prove.py` for the rule)."""
    import csv

    from .seam import prove as prove_mod

    candidates = list(csv.DictReader(open(candidates_csv, newline="")))
    entries = list(csv.DictReader(open(lists_csv, newline="")))
    for e in entries:
        e["order_listed"] = int(e["order_listed"])
    lists = {}
    for e in entries:
        lists.setdefault(e["list_id"], []).append(e)
    if pairs_csv:
        wanted = {
            (r["file_id"], r["list_id"]) for r in csv.DictReader(open(pairs_csv, newline=""))
        }
    else:
        wanted = {(c["file_id"], lid) for c in candidates for lid in lists}

    own_djs = {e["dj"] for e in entries}
    pool = tracklists.other_dj_pool(
        cfg.root, cfg.tracklists_csv, cfg.raw.get("control_djs", []), own_djs
    )
    listed_anywhere = {e["track_id"] for e in entries}
    pool = [c for c in pool if c["track_id"] not in listed_anywhere]

    table = tables.proofs(cfg.dirs["out"])
    done = {k[0] for k in table.keys()}
    jobs, missing = [], set()
    for c in candidates:
        if c["file_id"] in done:
            continue
        mine = [e for lid, es in lists.items() if (c["file_id"], lid) in wanted for e in es]
        tracks, on_disk = {}, []
        for e in mine:
            path = (
                Path(e["track_path"])
                if e.get("track_path")
                else tracklists.audio_file(cfg.dirs["tracks"], e["track_id"])
            )
            if path is None or not Path(path).exists():
                missing.add(e["track_id"])
                continue
            tracks.setdefault(e["track_id"], {"track_id": e["track_id"], "path": path})
            on_disk.append(e)
        if not tracks:
            log.warning("prove %s: no listed record on disk, skipped", c["file_id"])
            continue
        controls = tracklists.controls_for(
            {"mix_id": c["file_id"], "tracks": []}, pool, N_CONTROLS
        )
        if len(controls) < N_CONTROLS:
            log.warning("prove %s: only %d controls in the pool", c["file_id"], len(controls))
        jobs.append(
            {
                "file_id": c["file_id"],
                "path": c["path"],
                "tracks": list(tracks.values()),
                "entries": on_disk,
                "controls": controls,
            }
        )
    log.info(
        "prove: %d files, %d done before, %d to run at %d workers, %d listed records not on disk",
        len(candidates),
        len(done),
        len(jobs),
        workers,
        len(missing),
    )
    counts = {"files": len(candidates), "done_before": len(done), "ran": 0, "failed": 0}
    t0 = time.time()
    with _pool(cfg, workers) as pool_x:
        futures = {pool_x.submit(_prove_job, j): j["file_id"] for j in jobs}
        for fut in as_completed(futures):
            file_id = futures[fut]
            try:
                rows = fut.result()
            except Exception as error:
                counts["failed"] += 1
                log.error("prove FAILED %s: %s: %s", file_id, type(error).__name__, error)
                continue
            table.append(rows)
            counts["ran"] += 1
            log.info("prove done %s: %d rows, floor %s", file_id, len(rows), rows[0]["floor"])

    summary = []
    by_pair = {}
    listed_at = {(e["list_id"], e["order_listed"]): num(e.get("listed_min")) for e in entries}
    for r in table.rows():
        if r["is_control"] == "1":
            continue
        by_pair.setdefault((r["list_id"], r["file_id"]), []).append(
            {
                "order_listed": int(r["order_listed"]),
                "time_zero_s": float(r["time_zero_s"]),
                "first_heard_s": num(r["first_heard_s"]),
                "listed_min": listed_at.get((r["list_id"], int(r["order_listed"]))),
                "found": r["found"] == "1",
                "confidence": r["confidence"],
                "floor": r["floor"],
                "control_max": r["control_max"],
            }
        )
    for (list_id, file_id), rows in sorted(by_pair.items()):
        summary.append(prove_mod.summarise(list_id, file_id, rows, len(lists.get(list_id, []))))
    out = tables.proof_summary(cfg.dirs["out"])
    if out.path.exists():
        out.path.unlink()
    out.append(summary)
    counts["proved"] = sum(s["proved"] for s in summary)
    counts["seconds"] = round(time.time() - t0, 1)
    return counts


# ---------------------------------------------------------------- layers


LAYER_GRID_S = (
    10.0  # every record's presence windows snap to one 10 s grid so they can be compared
)
MAX_SPAN_S = 300.0  # a long stack is read in pieces this long, so one job stays small


def _presence_by_mix(cfg: Config) -> dict:
    by = {}
    for r in tables.presence(cfg.dirs["out"]).rows():
        by.setdefault(r["mix_id"], []).append(r)
    return by


def _grid(t: float) -> float:
    return round(round(float(t) / LAYER_GRID_S) * LAYER_GRID_S, 1)


SAME_AUDIO_T0_S = 1.0  # two ids whose time zero agrees this closely ...
SAME_AUDIO_OVERLAP = 0.8  # ... and who are present in mostly the same windows are one recording


def same_recording(a_path, b_path, control_paths: list) -> bool:
    """Are two record files the same recording? A's fingerprint is matched against B's, and against
    each control, and the floor is twice the loudest control (the one rule, `floors.from_controls`).
    Time zero and windows alone cannot decide it: two different records started on the same second
    look the same there (test_layers, r2 and r3)."""
    from .seam.fingerprint import as_arrays, fingerprint, match

    a_fp = fingerprint(load_audio(a_path))
    votes = match(a_fp, as_arrays(fingerprint(load_audio(b_path))))[0]
    ctrl = [match(a_fp, as_arrays(fingerprint(load_audio(c))))[0] for c in control_paths]
    return floors.from_controls(ctrl).clears(votes)


def same_audio(plays_rows: list[dict], pres: list[dict], is_same=None) -> dict:
    """track_id -> the one id it is counted under. One recording listed under two ids (NY6 lists two
    Skepta & PlaqueBoyMax versions, found at 425.31 s and 425.27 s with the same votes, 2026-09-28)
    would read as two records playing together. A pair is a candidate when their time zero agrees
    within SAME_AUDIO_T0_S and their present windows overlap by SAME_AUDIO_OVERLAP of the smaller set;
    `is_same(a, b)` then decides on the audio (`same_recording`). Without it nothing is merged."""
    t0 = {
        r["track_id"]: num(r["time_zero_s"])
        for r in plays_rows
        if r["is_control"] == "0" and num(r["time_zero_s"]) is not None
    }
    windows = {}
    for r in pres:
        if r["state"] == "present":
            windows.setdefault(r["track_id"], set()).add(_grid(r["window_start_s"]))
    ids = sorted(k for k in windows if k in t0)
    canon = {k: k for k in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if abs(t0[a] - t0[b]) > SAME_AUDIO_T0_S:
                continue
            small = min(len(windows[a]), len(windows[b]))
            if not small or len(windows[a] & windows[b]) < SAME_AUDIO_OVERLAP * small:
                continue
            if is_same is not None and is_same(a, b):
                canon[b] = canon[a]
    return canon


def _stack(rows: list[dict], canon: dict | None = None) -> dict:
    """grid time -> {track_id: state}, a record present in any window snapped there counts present.
    With `canon`, ids of one recording are counted once, under its first id."""
    out = {}
    canon = canon or {}
    for r in rows:
        tid = canon.get(r["track_id"], r["track_id"])
        cell = out.setdefault(_grid(r["window_start_s"]), {})
        if cell.get(tid) != "present":
            cell[tid] = r["state"]
    return out


def _aliases(canon: dict) -> dict:
    """canonical id -> "id=other id" for every recording counted under more than one id."""
    groups = {}
    for k, v in canon.items():
        groups.setdefault(v, []).append(k)
    return {v: "=".join(sorted(ks)) for v, ks in groups.items() if len(ks) > 1}


def _layers_job(job: dict) -> list[dict]:
    """One mix's layer rows: which ids are one recording (checked on the audio), then per grid time
    the present and weak records."""
    canon = same_audio(job["rows"], job["presence"], _is_same_paths(job["paths"], job["rows"]))
    alias = _aliases(canon)
    out = []
    for t, cell in sorted(_stack(job["presence"], canon).items()):
        present = sorted(k for k, v in cell.items() if v == "present")
        weak = sorted(k for k, v in cell.items() if v == "weak")
        out.append(
            {
                "mix_id": job["mix_id"],
                "dj": job["dj"],
                "window_start_s": t,
                "n_present": len(present),
                "n_weak": len(weak),
                "present": " ".join(present),
                "weak": " ".join(weak),
                "same_audio": " ".join(alias[k] for k in present + weak if k in alias),
            }
        )
    return out


def _is_same_paths(paths: dict, plays_rows: list[dict]):
    """`is_same` from a ready path map, so it crosses a process boundary as plain data."""
    controls = [
        paths[r["track_id"]]
        for r in plays_rows
        if r["is_control"] == "1" and r["track_id"] in paths
    ]

    def is_same(a, b):
        if a not in paths or b not in paths or not controls:
            return False
        return same_recording(paths[a], paths[b], controls)

    return is_same


def layers(cfg: Config, workers: int = 1) -> dict:
    """Stage 7a. The layer timeline: per mix and per 10 s grid time, which records are present (at or
    above the sweep floor, twice the loudest control window) and which are weak (above the loudest
    control window, under twice it). Several weak records at one place is kept as a signal of its own.
    Two ids of one recording, checked on the audio, are counted once and named in `same_audio`.
    One mix per worker. Rewrites layers.csv from presence.csv."""
    out = tables.layers(cfg.dirs["out"])
    plays_rows = tables.plays(cfg.dirs["out"]).rows()
    djs = {r["mix_id"]: r["dj"] for r in plays_rows}
    by_mix = {}
    for r in plays_rows:
        by_mix.setdefault(r["mix_id"], []).append(r)
    paths = {k: str(v) for k, v in _track_paths(cfg).items()}
    jobs = [
        {
            "mix_id": mix_id,
            "dj": djs.get(mix_id, ""),
            "rows": by_mix.get(mix_id, []),
            "presence": pres,
            "paths": {
                r["track_id"]: paths[r["track_id"]]
                for r in by_mix.get(mix_id, [])
                if r["track_id"] in paths
            },
        }
        for mix_id, pres in sorted(_presence_by_mix(cfg).items())
    ]
    log.info("layers: %d mixes at %d workers", len(jobs), workers)
    rows, t0 = {}, time.time()
    with _pool(cfg, workers) as pool:
        futures = {pool.submit(_layers_job, j): j["mix_id"] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            rows[futures[fut]] = fut.result()
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                log.info(
                    "layers: %d of %d mixes, %.0f s, about %.0f s left",
                    i,
                    len(jobs),
                    el,
                    el / i * (len(jobs) - i),
                )
    flat = [r for mix_id in sorted(rows) for r in rows[mix_id]]
    if out.path.exists():
        out.path.unlink()
    out.append(flat)
    counts = {
        "mixes": len(rows),
        "windows": len(flat),
        "two_or_more_present": sum(1 for r in flat if r["n_present"] >= 2),
        "three_or_more_present_or_weak": sum(1 for r in flat if r["n_present"] + r["n_weak"] >= 3),
    }
    log.info("layers: %s", counts)
    return counts


def _canon_from_layers(cfg: Config) -> dict:
    """mix_id -> {track_id: the id it is counted under}, from layers.csv's same_audio column, so
    layer-bands does not check the audio a second time."""
    out = {}
    for r in tables.layers(cfg.dirs["out"]).rows():
        for group in r.get("same_audio", "").split():
            ids = group.split("=")
            m = out.setdefault(r["mix_id"], {})
            for k in ids:
                m[k] = ids[0]
    return out


def _adjacent_pairs(plays_rows: list[dict]) -> set:
    """Consecutive records in audio order (by first heard): the ordinary seams the measure stage reads."""
    heard = sorted(
        (num(r["first_heard_s"]), r["track_id"])
        for r in plays_rows
        if r["is_control"] == "0" and r["found"] == "1" and num(r["first_heard_s"]) is not None
    )
    ids = [t for _, t in heard]
    return {frozenset(p) for p in zip(ids, ids[1:])}


def layer_spans(stack: dict, adjacent: set) -> list[dict]:
    """Runs of grid times whose PRESENT records are three or more, or two that are not a consecutive
    pair in audio order. [{start_s, end_s, records: {track_id: "present"}}], each at most MAX_SPAN_S.

    Weak records never make or join a span (2026-09-28). A weak window is the best of many chance
    offsets, and on Black Coffee, who does not stack records, one record read weak at eight places over
    two hours; the band floor, from controls given no such best-of choice, then passed chance as a
    band. Weak stays in layers.csv, where Fred's weak stacks can be set against a non-stacking DJ's."""
    spans, cur = [], None
    for t in sorted(stack):
        cell = {k: v for k, v in stack[t].items() if v == "present"}
        ids = frozenset(cell)
        qualifies = len(ids) >= 3 or (len(ids) == 2 and ids not in adjacent)
        if (
            qualifies
            and cur is not None
            and t - cur["last"] <= LAYER_GRID_S + 0.1
            and (t - cur["start_s"] < MAX_SPAN_S)
        ):
            cur["last"] = t
            for k, v in cell.items():
                if cur["records"].get(k) != "present":
                    cur["records"][k] = v
        elif qualifies:
            cur = {"start_s": t, "last": t, "records": dict(cell)}
            spans.append(cur)
        else:
            cur = None
    for sp in spans:
        sp["end_s"] = sp.pop("last") + locate_mod.SWEEP_WIN_S
    return spans


def _layer_band_job(job: dict) -> list[dict]:
    import subprocess
    import tempfile

    from .seam import bands
    from .seam.fingerprint import SR

    t0, t1 = job["start_s"], job["end_s"]
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "span.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{t0:.2f}",
                "-t",
                f"{t1 - t0:.2f}",
                "-i",
                str(job["mix_path"]),
                "-ac",
                "1",
                "-ar",
                str(SR),
                str(wav),
            ],
            check=True,
        )
        window = load_audio(wav)
    span_len = len(window) / SR
    tracks, anchors = {}, {}
    for rec in job["records"]:
        tracks[rec["track_id"]], anchors[rec["track_id"]] = measure_mod.lay_out(
            t0, span_len, rec, load_audio(rec["path"])
        )
    first = job["records"][0]
    names = []
    for i, path in enumerate(job["controls"], 1):
        name = f"__control_{i}"
        tracks[name], anchors[name] = measure_mod.lay_out(t0, span_len, first, load_audio(path))
        names.append(name)
    rows = []
    for band in bands.BANDS:
        fps = {name: bands.band_fp(audio, band) for name, audio in tracks.items()}
        curves = bands.band_votes(window, band, fps, anchors)
        floor = floors.from_controls([int(curves[n].max()) for n in names])
        for rec in job["records"]:
            tid = rec["track_id"]
            seen = curves[tid] >= floor.floor
            f, last = bands.first_last(curves, tid, floor.floor)
            rows.append(
                {
                    "mix_id": job["mix_id"],
                    "dj": job["dj"],
                    "span_id": job["span_id"],
                    "span_start_s": round(t0, 1),
                    "span_end_s": round(t1, 1),
                    "n_records": len(job["records"]),
                    "records": " ".join(r["track_id"] for r in job["records"]),
                    "track_id": tid,
                    "state": rec["state"],
                    "band": band,
                    "votes": int(curves[tid].max()),
                    "floor": floor.floor,
                    "control_max": floor.control_max,
                    "control_n": floor.control_n,
                    "separation": floors.separation(curves[tid].max(), floor),
                    "seen_s": round(float(seen.sum()) * bands.HOP_S, 1),
                    "first_seen_s": None if f is None else round(t0 + f, 1),
                    "last_seen_s": None if last is None else round(t0 + last, 1),
                }
            )
    return rows


def layer_bands(cfg: Config, workers: int = 1) -> dict:
    """Stage 7b. For every span where records stack beyond an ordinary seam (three or more present or
    weak, or two that are not consecutive in audio order), per record and per band: votes at the
    record's own alignment in that span, the band floor from three controls laid on the same span,
    and the seconds it is seen. The two-record seam measure is left as it is. Writes layer_bands.csv."""
    out = tables.layer_bands(cfg.dirs["out"])
    done = {(k[0], k[1]) for k in out.keys()}
    plays_rows = tables.plays(cfg.dirs["out"]).rows()
    by_mix = {}
    for r in plays_rows:
        by_mix.setdefault(r["mix_id"], []).append(r)
    mixes = {m["mix_id"]: m for m in corpus(cfg)}
    paths = _track_paths(cfg)
    if not tables.layers(cfg.dirs["out"]).exists():
        raise RuntimeError("layer-bands reads layers.csv: run djdata layers first")
    canon_by_mix = _canon_from_layers(cfg)
    jobs = []
    for mix_id, pres in sorted(_presence_by_mix(cfg).items()):
        mix = mixes.get(mix_id)
        if mix is None or mix["path"] is None:
            continue
        rows = by_mix.get(mix_id, [])
        rate = {r["track_id"]: num(r["rate"]) for r in rows if r["is_control"] == "0"}
        canon = canon_by_mix.get(mix_id, {})
        controls = [
            paths[r["track_id"]] for r in rows if r["is_control"] == "1" and r["track_id"] in paths
        ]
        stack = _stack(pres, canon)
        adjacent = {frozenset(canon.get(t, t) for t in p) for p in _adjacent_pairs(rows)}
        for k, sp in enumerate(layer_spans(stack, adjacent), 1):
            span_id = f"{mix_id}_L{k:04d}"
            if (mix_id, span_id) in done:
                continue
            records = []
            for tid, state in sorted(sp["records"].items()):
                if tid not in paths or tid not in rate:
                    continue
                inside = [
                    num(r["window_start_s"]) - num(r["record_at_s"]) / rate[tid]
                    for r in pres
                    if r["track_id"] == tid
                    and r["state"] == "present"
                    and sp["start_s"] - 5 <= _grid(r["window_start_s"]) < sp["end_s"]
                ]
                if not inside:
                    continue
                inside.sort()
                records.append(
                    {
                        "track_id": tid,
                        "path": paths[tid],
                        "rate": rate[tid],
                        "time_zero_s": inside[len(inside) // 2],
                        "state": state,
                    }
                )
            if len(records) < 2:
                continue
            jobs.append(
                {
                    "mix_id": mix_id,
                    "dj": mix["dj"],
                    "span_id": span_id,
                    "mix_path": mix["path"],
                    "start_s": sp["start_s"],
                    "end_s": sp["end_s"],
                    "records": records,
                    "controls": controls,
                }
            )
    log.info(
        "layer-bands: %d spans to read at %d workers, %d done before",
        len(jobs),
        workers,
        len(done),
    )
    counts = {"spans": len(jobs), "done_before": len(done), "ran": 0, "failed": 0}
    t0 = time.time()
    with _pool(cfg, workers) as pool:
        futures = {pool.submit(_layer_band_job, j): j["span_id"] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                out.append(fut.result())
                counts["ran"] += 1
            except Exception as error:
                counts["failed"] += 1
                log.error(
                    "layer-bands FAILED %s: %s: %s", futures[fut], type(error).__name__, error
                )
            if i % 20 == 0 or i == len(jobs):
                el = time.time() - t0
                log.info(
                    "layer-bands: %d of %d spans, %.0f s, about %.0f s left",
                    i,
                    len(jobs),
                    el,
                    el / i * (len(jobs) - i),
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
    pool_tracks = _control_pool(cfg, list(mixes.values()))
    exclusions = _own_dj_exclusions(cfg)
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
        controls = _controls(mix, pool_tracks, exclusions)
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
    pool_tracks = _control_pool(cfg, list(mixes.values()))
    exclusions = _own_dj_exclusions(cfg)
    jobs = []
    for c in cuts_t.rows():
        seam = seams.get(c["seam_id"])
        if seam is None or c["seam_id"] in done or c["status"] not in ("ok", "adopted"):
            continue
        mix = mixes.get(seam["mix_id"])
        if mix is None or seam["track_a"] not in paths or seam["track_b"] not in paths:
            continue
        controls = _controls(mix, pool_tracks, exclusions)
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
