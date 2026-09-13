"""Download one full mix, cut one window per seam without re-encoding, delete the full mix.

Time-based partial downloads were measured 5.6 s off, so the whole file is fetched. The cut is
`ffmpeg -ss -to -c copy`, accurate to one codec frame (about 26 ms); the analyser realigns from the
window audio itself, so the window's nominal start is only a coarse anchor.
"""

import logging
from pathlib import Path
import shutil
import subprocess
import tempfile

import yt_dlp

from ..state import State

log = logging.getLogger("djdata.fetch.mix")


def _ydl_opts(cfg, outtmpl: str) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True,
            "format": cfg.download["mix_format"], "outtmpl": outtmpl, "retries": 3}
    if cfg.download.get("cookies_file"):
        opts["cookiefile"] = cfg.download["cookies_file"]
    if cfg.download.get("youtube_player_client"):
        # with cookies, YouTube's default tv client returns UNPLAYABLE (yt-dlp issue 17389); web_embedded works
        opts["extractor_args"] = {"youtube": {"player_client": cfg.download["youtube_player_client"].split(",")}}
    return opts


def download_full(cfg, url: str, dest_stem: Path) -> Path:
    with tempfile.TemporaryDirectory(dir=dest_stem.parent) as td:
        with yt_dlp.YoutubeDL(_ydl_opts(cfg, str(Path(td) / "mix.%(ext)s"))) as ydl:
            ydl.extract_info(url, download=True)
        files = [f for f in Path(td).iterdir() if f.is_file()]
        if not files:
            raise RuntimeError(f"yt-dlp produced no file for {url}")
        dest = dest_stem.with_suffix(files[0].suffix)
        shutil.move(str(files[0]), dest)
    return dest


def cut_window(src: Path, t0: float, t1: float, dest_stem: Path) -> Path:
    dest = dest_stem.with_suffix(src.suffix)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(t0, 0):.3f}", "-to", f"{t1:.3f}",
                    "-i", str(src), "-c", "copy", str(dest)], check=True)
    return dest


def process_mix(cfg, state: State, mix: dict) -> int:
    """Download the mix, cut every seam window of it, delete the mix. Returns the number of windows."""
    full = download_full(cfg, mix["url"], cfg.dirs["mixes_tmp"] / mix["mix_id"])
    n = 0
    try:
        for seam in state.seams_of_mix(mix["mix_id"]):
            if seam["status"] != "pending":
                continue
            import json
            t0, t1 = json.loads(seam["coarse"])["window"]
            w = cut_window(full, t0, t1, cfg.dirs["windows"] / seam["seam_id"])
            state.set_seam(seam["seam_id"], "ready", window_path=str(w))
            n += 1
    finally:
        full.unlink(missing_ok=True)
    return n
