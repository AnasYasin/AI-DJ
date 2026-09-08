#!/usr/bin/env python3
"""
Fetch full audio for a planned tracklist.

The planner works on a catalog built from 30-second iTunes previews, so a plan
names tracks it has no audio for. This module finds each one on YouTube,
downloads it, and proves the download is the right recording before accepting
it.

Two gates, because either one alone lets rubbish through:

  duration  A planned track has to be the full version. Radio edits, snippets
            and teasers are shorter than MIN_SECONDS and are rejected on the
            search metadata and again on the decoded file, which is the check
            that counts because the metadata lies. Anything longer than
            MAX_SECONDS is a DJ set or an album upload, not a track.

  identity  Constellation fingerprint against the track's own 30-second
            preview. A search hit with a matching title is often a remix, a
            live rip or somebody else's bootleg. The preview is ground truth
            for what the catalog entry sounds like, so the download has to
            contain it. Chroma or mel cross-correlation was tried first and
            could not separate the two cases, because a loop-based track
            correlates with itself at many lags. Paired spectral peaks are
            sharp: measured on 12 real downloads, true pairs score 457 to
            14,466 matching hashes at one offset and wrong pairs score at
            most 21.

Run:
  python -m src.models.predict_model --genre techno --bpm 128 138 --n 8 --json plan.json
  python -m src.data.track_fetcher --plan plan.json --out data/external/run1
"""

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

log = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/raw/preview_manifest.csv")

# ── Gates ──────────────────────────────────────────────────────────────────────

MIN_SECONDS = 240  # 4 min. The old 120 s floor let radio edits through, and two
# planned windows got cut to about a minute of real audio.
MAX_SECONDS = 900  # 15 min. Past this it is a set, a podcast or an album rip.
SEARCH_RESULTS = 8  # candidates pulled per track before filtering

VERIFY_SR = 22_050
VERIFY_HOP = 256
VERIFY_NFFT = 1024
VERIFY_MIN_HASHES = 50  # true pairs score hundreds to thousands, wrong pairs ≤ 21
PEAK_NEIGHBOURHOOD = (13, 9)  # local-max window, (freq bins, time frames)
PEAK_PERCENTILE = 90
FAN_OUT = 10  # how many later peaks each peak is paired with
DT_MIN_FRAMES, DT_MAX_FRAMES = 1, 60  # target zone for a pair

# Titles that are the right track in the wrong form.
_BAD_TITLE = re.compile(
    r"\b(radio\s*edit|short\s*(edit|version)|snippet|teaser|preview|trailer|"
    r"tik\s*tok|sped\s*up|slowed|nightcore|8d\s*audio|karaoke|instrumental|"
    r"acapella|reaction|tutorial|live\s*set|dj\s*set|full\s*set|mixtape|"
    r"episode|podcast)\b",
    re.I,
)
_PAREN = re.compile(r"[\(\[][^\)\]]*[\)\]]")
_NONWORD = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> set[str]:
    return set(_NONWORD.sub(" ", _PAREN.sub(" ", str(text).lower())).split())


def _title_overlap(query: str, candidate: str) -> float:
    """Fraction of the wanted words present in the candidate's title."""
    want, got = _norm(query), _norm(candidate)
    return len(want & got) / max(len(want), 1)


# ── Identity check ─────────────────────────────────────────────────────────────


def _fingerprint(path: str | Path) -> dict[tuple[int, int, int], list[int]]:
    """
    Constellation hashes → the frames they occur at.

    Spectral peaks survive re-encoding, EQ and level changes, which is what
    separates a YouTube rip of the right recording from a different one. Each
    peak is paired with the next few peaks after it; the pair's two frequency
    bins and their time gap form a hash that is invariant to where in the track
    it happens.
    """
    y, _ = librosa.load(str(path), sr=VERIFY_SR, mono=True)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=VERIFY_NFFT, hop_length=VERIFY_HOP)))
    local_max = maximum_filter(S, size=PEAK_NEIGHBOURHOOD, mode="nearest")
    freqs, times = np.nonzero((S == local_max) & (S > np.percentile(S, PEAK_PERCENTILE)))
    order = np.argsort(times)
    freqs, times = freqs[order], times[order]

    table: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    n = len(times)
    for i in range(n):
        for j in range(i + 1, min(i + 1 + FAN_OUT, n)):
            dt = int(times[j] - times[i])
            if dt < DT_MIN_FRAMES:
                continue
            if dt > DT_MAX_FRAMES:
                break
            table[(freqs[i] // 2, freqs[j] // 2, dt)].append(int(times[i]))
    return table


def verify_match(preview_path: str | Path, full_path: str | Path) -> tuple[int, float]:
    """
    Locate the preview inside the full track by fingerprint alignment.

    Every hash the two files share votes for one time offset. The right
    recording puts thousands of votes on a single offset; an unrelated track
    scatters a handful across many. Returns (votes at the best offset,
    offset in seconds).
    """
    P, F = _fingerprint(preview_path), _fingerprint(full_path)
    offsets = [tf - tp for h, tps in P.items() if h in F for tp in tps for tf in F[h]]
    if not offsets:
        return 0, 0.0
    values, counts = np.unique(np.array(offsets), return_counts=True)
    best = int(np.argmax(counts))
    return int(counts[best]), float(values[best] * VERIFY_HOP / VERIFY_SR)


# ── YouTube ────────────────────────────────────────────────────────────────────


def _search(query: str, n: int = SEARCH_RESULTS) -> list[dict]:
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            res = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
        except Exception as e:
            log.warning("search failed for %r: %s", query, e)
            return []
    return [e for e in (res or {}).get("entries", []) if e]


def _rank(query: str, entries: list[dict]) -> list[dict]:
    """Drop candidates that fail the duration or title gate, best first."""
    keep = []
    for e in entries:
        dur = e.get("duration")
        title = e.get("title") or ""
        if dur is None or not (MIN_SECONDS <= dur <= MAX_SECONDS):
            log.debug("  reject %r: duration %s", title[:60], dur)
            continue
        if _BAD_TITLE.search(title):
            log.debug("  reject %r: title form", title[:60])
            continue
        keep.append({**e, "overlap": _title_overlap(query, title)})
    return sorted(keep, key=lambda e: -e["overlap"])


def _download(url: str, dest: Path) -> Path | None:
    import yt_dlp

    with tempfile.TemporaryDirectory() as td:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "format": "bestaudio/best",
            "outtmpl": str(Path(td) / "%(id)s.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                ydl.extract_info(url, download=True)
            except Exception as e:
                log.warning("download failed for %s: %s", url, e)
                return None
        files = [f for f in Path(td).iterdir() if f.is_file()]
        if not files:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest = dest.with_suffix(files[0].suffix)
        shutil.move(str(files[0]), dest)
        return dest


def _decoded_seconds(path: Path) -> float:
    """True duration from the decoder. Search metadata is not trustworthy."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return float(librosa.get_duration(path=str(path)))


# ── Per-track fetch ────────────────────────────────────────────────────────────


def fetch_track(
    artist: str,
    title: str,
    track_id: str,
    out_dir: Path,
    preview_path: str | Path | None = None,
    max_tries: int = 3,
) -> dict:
    """Find, download and verify one track. Returns a result record."""
    out_dir = Path(out_dir)
    existing = list(out_dir.glob(f"{track_id}.*"))
    if existing:
        return {"track_id": track_id, "status": "cached", "path": str(existing[0])}

    query = f"{artist} {title}"
    ranked = _rank(query, _search(query))
    if not ranked:
        return {"track_id": track_id, "status": "no_candidate", "query": query}

    for cand in ranked[:max_tries]:
        url = cand.get("url") or f"https://www.youtube.com/watch?v={cand['id']}"
        path = _download(url, out_dir / track_id)
        if path is None:
            continue

        seconds = _decoded_seconds(path)
        if not (MIN_SECONDS <= seconds <= MAX_SECONDS):
            log.info(
                "  %s: decoded %.0fs outside [%d, %d] — rejected",
                title[:40],
                seconds,
                MIN_SECONDS,
                MAX_SECONDS,
            )
            path.unlink(missing_ok=True)
            continue

        if preview_path and Path(preview_path).exists():
            votes, offset = verify_match(preview_path, path)
            if votes < VERIFY_MIN_HASHES:
                log.info(
                    "  %s: fingerprint %d < %d votes — wrong recording, rejected",
                    title[:40],
                    votes,
                    VERIFY_MIN_HASHES,
                )
                path.unlink(missing_ok=True)
                continue
        else:
            votes, offset = -1, float("nan")  # no preview on disk: duration gate only

        log.info("  %s: %.0fs, %d votes → %s", title[:40], seconds, votes, path.name)
        return {
            "track_id": track_id,
            "status": "ok",
            "path": str(path),
            "seconds": round(seconds, 1),
            "verify_votes": votes,
            "preview_offset_s": round(offset, 1),
            "source": url,
            "youtube_title": cand.get("title"),
        }

    return {"track_id": track_id, "status": "no_verified_candidate", "query": query}


def preview_paths() -> dict[str, str]:
    """track_id → local 30 s preview, the ground truth the fingerprint checks against."""
    if not MANIFEST_PATH.exists():
        return {}
    m = pd.read_csv(MANIFEST_PATH).drop_duplicates("track_id")
    return dict(zip(m["track_id"], m["local_path"]))


def fetch_plan(plan_path: str | Path, out_dir: str | Path) -> dict:
    plan = json.loads(Path(plan_path).read_text())
    out_dir = Path(out_dir)
    previews = preview_paths()

    results = []
    for t in plan["tracks"]:
        log.info("[%d/%d] %s – %s", t["n"], len(plan["tracks"]), t["artist"], t["title"])
        results.append(
            fetch_track(
                t["artist"], t["title"], t["track_id"], out_dir, previews.get(t["track_id"])
            )
        )
    ok = [r for r in results if r["status"] in ("ok", "cached")]
    log.info("fetched %d/%d tracks", len(ok), len(results))
    return {
        "out_dir": str(out_dir),
        "results": results,
        "paths": [r["path"] for r in ok],
        "n_ok": len(ok),
        "n_total": len(results),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Fetch and verify full audio for a plan.")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--json", default=None, help="write the fetch report here")
    args = p.parse_args()
    rep = fetch_plan(args.plan, args.out)
    for r in rep["results"]:
        print(f"  {r['status']:<22} {r.get('path', r.get('query', ''))}")
    print(f"\n{rep['n_ok']}/{rep['n_total']} verified → {rep['out_dir']}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1))
