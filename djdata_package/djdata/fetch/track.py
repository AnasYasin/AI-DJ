"""Fetch one track. Raveform seams carry a YouTube id, so the download is direct. Tracklist seams
carry only artist and title; those go through the legacy fetch_track (YouTube search, duration gate,
fingerprint check against the 30 s preview)."""

import logging
from pathlib import Path
import shutil
import subprocess
import tempfile

from . import yt

log = logging.getLogger("djdata.fetch.track")


def _duration_s(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def download_by_url(cfg, url: str, dest_stem: Path) -> tuple[Path, float]:
    with tempfile.TemporaryDirectory(dir=dest_stem.parent) as td:
        got = yt.download(cfg, url, cfg.download["track_format"], Path(td))
        dest = dest_stem.with_suffix(got.suffix)
        shutil.move(str(got), dest)
    dur = _duration_s(dest)
    lo, hi = cfg.download["min_track_s"], cfg.download["max_track_s"]
    if not lo <= dur <= hi:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"duration {dur:.0f}s outside [{lo}, {hi}]")
    return dest, dur


def download_by_name(cfg, title: str, track_id: str, dest_dir: Path) -> tuple[Path, float]:
    """Tracklist path: 'Artist - Title' → legacy search + fingerprint verification."""
    from ..legacy.track_fetcher import fetch_track, preview_paths

    artist, _, name = title.partition(" - ")
    res = fetch_track(artist, name, track_id, dest_dir, preview_path=preview_paths().get(track_id))
    if res["status"] not in ("ok", "cached"):
        raise RuntimeError(f"fetch_track {res['status']} for {title}")
    p = Path(res["path"])
    return p, _duration_s(p)


def fetch(cfg, track: dict) -> tuple[Path, float]:
    dest_dir = cfg.dirs["tracks"]
    existing = list(dest_dir.glob(f"{track['track_id']}.*"))
    if existing:
        return existing[0], _duration_s(existing[0])
    if track.get("url"):
        return download_by_url(cfg, track["url"], dest_dir / track["track_id"])
    return download_by_name(cfg, track["title"], track["track_id"], dest_dir)
