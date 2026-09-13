"""Fetch one track. Raveform seams carry a YouTube id, so the download is direct. Tracklist seams
carry only artist and title; those go through the legacy fetch_track (YouTube search, duration gate,
fingerprint check against the 30 s preview)."""

import logging
from pathlib import Path
import shutil
import subprocess
import tempfile

import yt_dlp

log = logging.getLogger("djdata.fetch.track")


def _duration_s(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def download_by_url(cfg, url: str, dest_stem: Path) -> tuple[Path, float]:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True,
            "format": cfg.download["track_format"], "retries": 3}
    if cfg.download.get("cookies_file"):
        opts["cookiefile"] = cfg.download["cookies_file"]
    if cfg.download.get("youtube_player_client"):
        # with cookies, YouTube's default tv client returns UNPLAYABLE (yt-dlp issue 17389); web_embedded works
        opts["extractor_args"] = {"youtube": {"player_client": cfg.download["youtube_player_client"].split(",")}}
    with tempfile.TemporaryDirectory(dir=dest_stem.parent) as td:
        opts["outtmpl"] = str(Path(td) / "t.%(ext)s")
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
        files = [f for f in Path(td).iterdir() if f.is_file()]
        if not files:
            raise RuntimeError(f"yt-dlp produced no file for {url}")
        dest = dest_stem.with_suffix(files[0].suffix)
        shutil.move(str(files[0]), dest)
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
