"""Coarse anchors from the audio for seams that come without alignment (the 1001tracklists path).

The landmark fingerprint from the legacy track fetcher locates a 60 s piece of the window inside each
original track: `verify_match(piece, track) → (votes, track_time_of_piece_start)`. That is accurate
to about a bar in loop-based music, which is what the fine alignment expects. Rate is taken as 1.0
and refined by the aligner's residual scan (it corrects up to about 0.5 %); larger tempo changes
are caught by the tempo check in the analyser and the seam is marked failed with the reason.
"""

import logging
from pathlib import Path
import tempfile

import soundfile as sf

from ..legacy.track_fetcher import VERIFY_MIN_HASHES, verify_match
from .align import SR, load_mono

log = logging.getLogger("djdata.seam.coarse")


def fingerprint_anchors(window_path: Path, a_path: Path, b_path: Path, coarse: dict) -> dict:
    """Fill A/B anchors in `coarse` (window-relative mix times) from the audio. Raises if a record is not found."""
    t0, t1 = coarse["window"]
    w = load_mono(window_path)
    ov = coarse["overlap_start_mix_t"] - t0
    pieces = {"A": (max(ov - 70.0, 0.0), max(ov - 10.0, 60.0)), "B": (min(ov + 30.0, len(w) / SR - 60.0), None)}
    out = dict(coarse)
    for tag, path in (("A", a_path), ("B", b_path)):
        p0 = pieces[tag][0]
        p1 = pieces[tag][1] if pieces[tag][1] is not None else p0 + 60.0
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, w[int(p0 * SR): int(p1 * SR)], SR)
            votes, track_t = verify_match(tmp.name, str(path))
        if votes < VERIFY_MIN_HASHES:
            raise RuntimeError(f"fingerprint could not place record {tag}: {votes} votes")
        out[tag] = {"orig_t": float(track_t), "mix_t": float(t0 + p0), "rate": 1.0, "votes": int(votes)}
        log.info("fingerprint %s: %d votes, track_t %.1f at window %.1f s", tag, votes, track_t, p0)
    out["needs_fingerprint"] = False
    return out
