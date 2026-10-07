"""Command line. Every subcommand is one stage function, so an Airflow task can call the same thing.

  djdata manifest   --config config.yaml            build the seam manifest into the state db
  djdata run        --config config.yaml            fetch mixes and tracks, analyse seams (resumable)
  djdata status     --config config.yaml            counts per status
  djdata archivable --config config.yaml            files safe to move to S3 (json)
  djdata archived   --config config.yaml FILE...    record files that were moved
  djdata export     --config config.yaml            seams.csv, curves.csv, qa.csv
  djdata probe      --config config.yaml [--n 20]   YouTube download test from this machine
  djdata retry      --config config.yaml --match T  failed tracks whose error contains T back to pending
  djdata media-links --config config.yaml           1001 mix pages → audio urls (needs a display: xvfb-run -a)
  djdata scrape-tracklists --dj URL --genre G        legacy 1001 scraper (needs a display)

The profiling pipeline, one stage per subcommand, each resumable (see pipeline.py):
  djdata prove      --config config_fred.yaml --candidates C --lists L [--pairs P] [--workers N]
                    which audio file is which show, by play order -> out/proofs.csv, out/proof_summary.csv
  djdata prove-lists --config config_fred.yaml --djs Fredagain.. --out L   the 1001 lists for prove
  djdata locate     --config config_djs.yaml [--workers N] [--mixes ID...]   records found in mixes -> out/plays.csv
  djdata pairs      --config config_djs.yaml                                  which record follows which -> out/seams.csv
  djdata cut        --config config_djs.yaml [--workers N]                    windows cut and audited -> out/cuts.csv
  djdata measure    --config config_djs.yaml [--workers N]                    bands, bass, presence, loop -> out/measures.csv
  djdata tempo      --config config_djs.yaml [--workers N]                    BPM per record -> out/tempos.csv
  djdata label      --config config_djs.yaml                                  transition types in bars -> out/labels.csv
  djdata export-seams --config config_djs.yaml                                one flat table -> out/seams_index.csv
  djdata layers     --config config_djs.yaml                                  the layer timeline -> out/layers.csv
  djdata layer-bands --config config_djs.yaml [--workers N]                   bands of stacked records -> out/layer_bands.csv
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time

from . import config as config_mod
from . import logs


def _cfg(args):
    cfg = config_mod.load(args.config)
    logs.setup(cfg.dirs["logs"], cfg.log_level)
    return cfg


def cmd_manifest(args):
    from .manifest import raveform, tracklists
    from .state import State

    cfg = _cfg(args)
    state = State(cfg.db_path)
    if cfg.source == "raveform":
        print(json.dumps(raveform.build(cfg, state)))
    else:
        tier = cfg.tiers.get("tracklists", {})
        timed = args.djs or tier.get("djs_timed", [])
        every = args.djs_all or tier.get("djs_all", [])
        print(
            json.dumps(
                tracklists.build(
                    cfg,
                    state,
                    timed,
                    every,
                    min_tracks=tier.get("min_tracks", 8),
                    min_timed=tier.get("min_timed", 0.95),
                )
            )
        )


def cmd_run(args):
    from .run import run

    print(json.dumps(run(_cfg(args))))


def cmd_status(args):
    from .state import State

    cfg = _cfg(args)
    st = State(cfg.db_path)
    out = st.counts(cfg.run_tiers)
    out["failures"] = st.failure_summary(cfg.run_tiers)
    gate = cfg.dirs["logs"] / "gate.json"
    if gate.exists():
        out["downloads"] = json.loads(gate.read_text())
    print(json.dumps(out, indent=1))


def cmd_retry(args):
    from .state import State

    cfg = _cfg(args)
    print(json.dumps(State(cfg.db_path).retry_tracks(args.match)))


def cmd_archivable(args):
    from .state import State

    cfg = _cfg(args)
    print(json.dumps(State(cfg.db_path).archivable(cfg.run_tiers), indent=1))


def cmd_archived(args):
    from .state import State

    cfg = _cfg(args)
    state = State(cfg.db_path)
    stems = [Path(f).stem for f in args.files]
    tracks = [s for s in stems if "_" not in s or len(s) <= 12]
    seams = [s for s in stems if s not in tracks]
    state.mark_archived(tracks, seams)
    print(json.dumps({"tracks": len(tracks), "seams": len(seams)}))


def cmd_export(args):
    from .export import export

    cfg = _cfg(args)
    print(json.dumps(export(cfg.dirs["out"])))


def cmd_probe(args):
    """Twenty track downloads from the manifest: does YouTube serve this machine?"""
    from .fetch.track import download_by_url
    from .state import State

    cfg = _cfg(args)
    rows = (
        State(cfg.db_path)
        ._conn()
        .execute("SELECT track_id, url FROM tracks WHERE url IS NOT NULL")
        .fetchall()
    )
    random.seed(0)
    sample = random.sample(rows, min(args.n, len(rows)))
    ok, fail, t0 = 0, [], time.time()
    probe_dir = cfg.root / "probe"
    probe_dir.mkdir(exist_ok=True)
    for r in sample:
        t = time.time()
        try:
            p, dur = download_by_url(cfg, r["url"], probe_dir / r["track_id"])
            ok += 1
            print(f"ok   {r['track_id']} {dur:.0f}s in {time.time() - t:.1f}s")
            p.unlink()
        except Exception as e:
            fail.append((r["track_id"], str(e)[:120]))
            print(f"FAIL {r['track_id']} {str(e)[:120]}")
    print(
        json.dumps(
            {
                "ok": ok,
                "failed": len(fail),
                "seconds": round(time.time() - t0, 1),
                "errors": fail[:5],
            }
        )
    )


def cmd_media_links(args):
    from .fetch.media_link import fill_media_links
    from .state import State

    cfg = _cfg(args)
    print(json.dumps({"filled": fill_media_links(cfg, State(cfg.db_path))}))


def cmd_locate(args):
    from . import pipeline

    cfg = _cfg(args)
    print(json.dumps(pipeline.locate(cfg, workers=args.workers, only=args.mixes)))


def cmd_prove(args):
    from . import pipeline

    cfg = _cfg(args)
    print(
        json.dumps(
            pipeline.prove(
                cfg, args.candidates, args.lists, pairs_csv=args.pairs, workers=args.workers
            )
        )
    )


def cmd_prove_lists(args):
    import csv

    from . import pipeline
    from .store import tables

    cfg = _cfg(args)
    rows = pipeline.lists_1001(cfg, args.djs)
    with open(args.out, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=tables.PROOF_LISTS)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"lists": len({r["list_id"] for r in rows}), "records": len(rows)}))


def cmd_usb002_tracklist(args):
    import csv

    from .sources import usb002

    rows = usb002.tracklist_rows(args.app, args.tracks, args.segments)
    with open(args.out, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=usb002.TRACKLIST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"lists": len({r["mix_id"] for r in rows}), "records": len(rows)}))


def cmd_stage(args):
    from . import pipeline

    cfg = _cfg(args)
    stage = getattr(pipeline, args.stage)
    kwargs = (
        {"workers": args.workers}
        if args.stage in ("cut", "measure", "tempo", "layers", "layer_bands")
        else {}
    )
    if args.stage == "tempo":
        kwargs["all_tracks"] = args.all_tracks
    print(json.dumps(stage(cfg, **kwargs)))


def cmd_ear_test(args):
    from . import pipeline

    cfg = _cfg(args)
    print(json.dumps(pipeline.ear_test(cfg, args.out, n=args.n, label=args.label, seed=args.seed)))


def cmd_scrape_tracklists(args):
    import asyncio

    from .legacy.tracklists1001_client import scrape_mixes_from_djs

    asyncio.run(scrape_mixes_from_djs([{"url": u, "genre": args.genre} for u in args.dj]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="djdata")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("manifest", cmd_manifest),
        ("run", cmd_run),
        ("status", cmd_status),
        ("archivable", cmd_archivable),
        ("archived", cmd_archived),
        ("export", cmd_export),
        ("probe", cmd_probe),
        ("media-links", cmd_media_links),
        ("retry", cmd_retry),
    ):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.set_defaults(fn=fn)
        if name == "manifest":
            sp.add_argument(
                "--djs",
                nargs="*",
                default=None,
                help="tracklists source: DJs whose mixes must carry start times (default: config)",
            )
            sp.add_argument(
                "--djs-all",
                nargs="*",
                default=None,
                help="tracklists source: DJs where every mix with a tracklist is taken, timed or not",
            )
        if name == "archived":
            sp.add_argument("files", nargs="+")
        if name == "probe":
            sp.add_argument("--n", type=int, default=20)
        if name == "retry":
            sp.add_argument(
                "--match",
                required=True,
                help="substring of the track error, e.g. 'HTTP Error 403'",
            )
    sp = sub.add_parser("locate")
    sp.add_argument("--config", required=True)
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument(
        "--mixes", nargs="*", default=None, help="only these mix ids (default: every mix on disk)"
    )
    sp.set_defaults(fn=cmd_locate)
    sp = sub.add_parser("prove")
    sp.add_argument("--config", required=True)
    sp.add_argument("--candidates", required=True, help="csv: file_id, path")
    sp.add_argument("--lists", required=True, help="csv with the PROOF_LISTS columns")
    sp.add_argument("--pairs", default=None, help="csv: file_id, list_id (default: every pair)")
    sp.add_argument("--workers", type=int, default=1)
    sp.set_defaults(fn=cmd_prove)
    sp = sub.add_parser("usb002-tracklist")
    sp.add_argument("--app", required=True, help="the app's data.json, saved")
    sp.add_argument("--tracks", required=True, help="usb002_solo_tracks.csv")
    sp.add_argument("--segments", required=True, help="the marathon segments.csv")
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_usb002_tracklist)
    sp = sub.add_parser("prove-lists")
    sp.add_argument("--config", required=True)
    sp.add_argument("--djs", nargs="+", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_prove_lists)
    for name, stage in (
        ("pairs", "pairs"),
        ("cut", "cut"),
        ("measure", "measure"),
        ("tempo", "tempo"),
        ("label", "label"),
        ("export-seams", "export"),
        ("layers", "layers"),
        ("layer-bands", "layer_bands"),
    ):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.add_argument("--workers", type=int, default=1)
        if name == "tempo":
            sp.add_argument(
                "--all-tracks",
                action="store_true",
                help="every audio file in tracks/, not only records that sit in a seam",
            )
        sp.set_defaults(fn=cmd_stage, stage=stage)
    sp = sub.add_parser("ear-test")
    sp.add_argument("--config", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--n", type=int, default=10)
    sp.add_argument("--label", default=None, help="only seams with this main label")
    sp.add_argument("--seed", type=int, default=25)
    sp.set_defaults(fn=cmd_ear_test)
    sp = sub.add_parser("scrape-tracklists")
    sp.add_argument("--dj", nargs="+", required=True, help="1001tracklists DJ page urls")
    sp.add_argument("--genre", required=True)
    sp.set_defaults(fn=cmd_scrape_tracklists)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
