"""Command line. Every subcommand is one stage function, so an Airflow task can call the same thing.

    djdata manifest   --config config.yaml            build the seam manifest into the state db
    djdata run        --config config.yaml            fetch mixes and tracks, analyse seams (resumable)
    djdata status     --config config.yaml            counts per status
    djdata archivable --config config.yaml            files safe to move to S3 (json)
    djdata archived   --config config.yaml FILE...    record files that were moved
    djdata export     --config config.yaml            seams.csv, curves.csv, qa.csv
    djdata probe      --config config.yaml [--n 20]   YouTube download test from this machine
    djdata media-links --config config.yaml           1001 mix pages → audio urls (needs a display: xvfb-run -a)
    djdata scrape-tracklists --dj URL --genre G        legacy 1001 scraper (needs a display)
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
        print(json.dumps(tracklists.build(cfg, state, args.djs)))


def cmd_run(args):
    from .run import run

    print(json.dumps(run(_cfg(args))))


def cmd_status(args):
    from .state import State

    cfg = _cfg(args)
    print(json.dumps(State(cfg.db_path).counts(cfg.run_tiers), indent=1))


def cmd_archivable(args):
    from .state import State

    cfg = _cfg(args)
    print(json.dumps(State(cfg.db_path).archivable(cfg.run_tiers), indent=1))


def cmd_archived(args):
    from .state import State

    cfg = _cfg(args)
    state = State(cfg.db_path)
    stems = [Path(f).stem for f in args.files]
    tracks = [s for s in stems if not "_" in s or len(s) <= 12]
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
    rows = State(cfg.db_path)._conn().execute("SELECT track_id, url FROM tracks WHERE url IS NOT NULL").fetchall()
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
    print(json.dumps({"ok": ok, "failed": len(fail), "seconds": round(time.time() - t0, 1), "errors": fail[:5]}))


def cmd_media_links(args):
    from .fetch.media_link import fill_media_links
    from .state import State

    cfg = _cfg(args)
    print(json.dumps({"filled": fill_media_links(cfg, State(cfg.db_path))}))


def cmd_scrape_tracklists(args):
    import asyncio

    from .legacy.tracklists1001_client import scrape_mixes_from_djs

    asyncio.run(scrape_mixes_from_djs([{"url": u, "genre": args.genre} for u in args.dj]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="djdata")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("manifest", cmd_manifest), ("run", cmd_run), ("status", cmd_status), ("archivable", cmd_archivable),
                     ("archived", cmd_archived), ("export", cmd_export), ("probe", cmd_probe), ("media-links", cmd_media_links)):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.set_defaults(fn=fn)
        if name == "manifest":
            sp.add_argument("--djs", nargs="*", default=["Black Coffee"], help="tracklists source only")
        if name == "archived":
            sp.add_argument("files", nargs="+")
        if name == "probe":
            sp.add_argument("--n", type=int, default=20)
    sp = sub.add_parser("scrape-tracklists")
    sp.add_argument("--dj", nargs="+", required=True, help="1001tracklists DJ page urls")
    sp.add_argument("--genre", required=True)
    sp.set_defaults(fn=cmd_scrape_tracklists)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
