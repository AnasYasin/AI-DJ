"""One yt-dlp download, with the two things every fetcher needs from it.

Cookies. YouTube rotates account cookies on every request and yt-dlp writes the rotated values back
to the cookie file when it closes. On 2026-09-13 five workers shared one file, each wrote back its
own rotation state, and after 11 minutes YouTube treated the conflicting states as a stolen session
and logged the account out everywhere (every request then returned HTTP 403). One lock now serialises
every download that carries cookies, so there is exactly one rotation chain, and yt-dlp is the only
writer of the file. Cookies are only sent to YouTube; SoundCloud and Mixcloud downloads run outside
the lock.

Blocks. A 403, a 429 or "Sign in to confirm" means YouTube refused because of who is asking, not what
was asked for. yt-dlp reports the 403 as a warning and then fails with "Requested format is not
available", so warnings are captured and the failure is raised as Blocked, which the runner treats
as "pause and retry", never as "this track is bad".
"""

import contextlib
import tempfile
from pathlib import Path
import threading

import yt_dlp

_COOKIE_LOCK = threading.Lock()
BLOCK_MARKERS = ("HTTP Error 403", "HTTP Error 429", "not a bot")   # not "Sign in to confirm": the age gate says "Sign in to confirm your age"
# 2026-09-14/15: an address YouTube had finished with kept answering page requests but every media
# download timed out connecting to googlevideo.com (170 s each, then "Giving up after 3 retries");
# zero tracks completed on it for a day while the gate stayed open. That is a block of the address.
CDN_TIMEOUT_MARKER = "googlevideo timeout"


class Blocked(RuntimeError):
    """YouTube refused because of who is asking, not what was asked for."""


class _Warnings:
    """yt-dlp logger that keeps warnings and drops everything else."""

    def __init__(self):
        self.messages = []

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        self.messages.append(msg)

    def error(self, msg):
        pass


def is_block(error: str, warnings: list[str]) -> str | None:
    """The block marker that explains this failure, or None."""
    for text in [error, *warnings]:
        for m in BLOCK_MARKERS:
            if m in text:
                return m
        if "googlevideo" in text and ("timed out" in text or "Network is unreachable" in text or "Failed to establish" in text):
            return CDN_TIMEOUT_MARKER
    return None


def uses_cookies(cfg, url: str) -> bool:
    return bool(cfg.download.get("cookies_file")) and ("youtube.com" in url or "youtu.be" in url)


def download(cfg, url: str, fmt: str, workdir: Path, simulate: bool = False) -> Path | None:
    """Fetch `url` into `workdir`. Returns the file, or None when simulating. Raises Blocked on a block."""
    warnings = _Warnings()
    opts = {"quiet": True, "noprogress": True, "logger": warnings, "format": fmt, "retries": 3,
            "outtmpl": str(workdir / "dl.%(ext)s"), "simulate": simulate,
            # 2026-09-15: a download on a dead socket sat for hours holding the cookie lock, and the probe
            # waiting for that lock never ran; the gate stayed "rotated, probing" with no result.
            "socket_timeout": 30,
            # 2026-09-15: googlevideo hosts resolve to IPv6 too and the VM has no IPv6 route ("[Errno 101]
            # Network is unreachable"); force IPv4 like yt-dlp -4.
            "source_address": "0.0.0.0"}
    with_cookies = uses_cookies(cfg, url)
    if with_cookies:
        opts["cookiefile"] = cfg.download["cookies_file"]
        if cfg.download.get("youtube_player_client"):
            opts["extractor_args"] = {"youtube": {"player_client": cfg.download["youtube_player_client"].split(",")}}
    with _COOKIE_LOCK if with_cookies else contextlib.nullcontext():
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=not simulate)
        except Exception as e:
            marker = is_block(str(e), warnings.messages)
            if marker:
                raise Blocked(f"{marker}: {str(e)[:200]}") from e
            raise
    if simulate:
        return None
    files = [f for f in workdir.iterdir() if f.is_file()]
    if not files:
        raise RuntimeError(f"yt-dlp produced no file for {url}")
    return files[0]


def probe(cfg, workdir: Path) -> bool:
    """One real download of the configured probe video (a few MB, deleted at once). True when YouTube
    serves it. A simulated request is not enough: it never touches the media servers, and a blocked
    address fails exactly there (googlevideo connect timeouts) while the page requests still pass."""
    try:
        with tempfile.TemporaryDirectory(dir=workdir) as tmp:
            download(cfg, cfg.download["block_probe_url"], cfg.download["track_format"], Path(tmp))
        return True
    except Blocked:
        return False
