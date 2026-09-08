"""Kick-hit alignment of every regression pair at the shift the mixer applied (numbers only).

Prepares A and B exactly as the render did (same target BPM, cue pins), cuts the overlap at the
mixer's out_bar / in_bar plus the applied shift, finds ONE strong kick per beat in each record
(loudest rising kick-band frame within ±0.6 beat, > 6x median) and reports, per 4-bar window, the
median offset of B's nearest kick to each A kick (+ = B late), plus the whole-overlap median.
"""

import json
import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, ".")
logging.disable(logging.WARNING)
from src.audio import audio_mixer as M  # noqa: E402
from src.audio.audio_mixer import (  # noqa: E402
    SR,
    _kick_envelope,
    _prepare_track,
    structure_regime,
)

man = {p["id"]: p for p in json.loads(Path("tests/regression/manifest.json").read_text())["pairs"]}
res = json.loads(Path("data/external/regression_set/results.json").read_text())
only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
hop = M.ONSET_HOP


def kicks(mono, beat):
    from scipy.ndimage import maximum_filter1d

    env = _kick_envelope(mono)
    size = 2 * int(0.6 * beat * SR / hop) + 1
    is_max = (env == maximum_filter1d(env, size=size, mode="nearest")) & (env > np.median(env) * 6)
    t = np.flatnonzero(is_max) * hop / SR
    keep, last = [], -1e9
    for x in t:
        if x - last >= 0.35 * beat / 0.465:
            keep.append(x)
            last = x
    return np.array(keep)


for pid, p in man.items():
    if only and pid not in only:
        continue
    r = res[pid]
    target = float(r["target_bpm"])
    bar = 4 * 60 / target
    beat = bar / 4
    bt = M.GENRE_BARS.get(p["genre"], M.DEFAULT_BARS)
    max_tail = int(np.median(list(bt.values())))
    phrase = M.GENRE_PHRASE_BARS.get(p["genre"], M.DEFAULT_PHRASE_BARS)
    play_min = M.GENRE_PLAY_MINUTES.get(p["genre"], M.DEFAULT_PLAY_MINUTES)
    target_bars = max(int(round(play_min * 60 / bar / phrase)) * phrase, M.MIN_BODY_BARS)
    et = p.get("energy_targets") or [None, None]
    A = _prepare_track(
        p["A"]["path"],
        target,
        max_tail,
        target_bars,
        et[0],
        phrase,
        structure_regime(et[1]) == "low",
        tuple(p["A"]["cue"]) if p["A"].get("cue") else None,
    )
    B = _prepare_track(
        p["B"]["path"],
        target,
        max_tail,
        target_bars,
        et[1],
        phrase,
        False,
        tuple(p["B"]["cue"]) if p["B"].get("cue") else None,
    )
    a0 = int(A["bars"][r["out_bar"]] * SR)
    applied_ms = r.get("seam_shift_total_ms", r["seam_offset_before_ms"])
    b0 = int(B["bars"][r["in_bar"]] * SR) + int(round(applied_ms / 1000 * SR))
    n_ov = int(r["bars"] * bar * SR)
    tail = M._to_mono(A["audio"][a0 : a0 + n_ov])
    head = M._to_mono(B["audio"][b0 : b0 + n_ov])
    ka, kb = kicks(tail, beat), kicks(head, beat)
    win = 4 * bar
    rows, alld = [], []
    for w in range(max(r["bars"] // 4, 1)):
        t0, t1 = w * win, (w + 1) * win
        a = ka[(ka >= t0) & (ka < t1)]
        b = kb[(kb >= t0) & (kb < t1)]
        if len(a) < 4 or len(b) < 4:
            rows.append("   n/a")
            continue
        d = np.array([b[np.argmin(np.abs(b - x))] - x for x in a]) * 1000
        d = d[np.abs(d) < beat * 1000 / 2]
        if len(d) < 4:
            rows.append("   n/a")
            continue
        rows.append(f"{np.median(d):+6.0f}")
        alld.extend(d.tolist())
    med = np.median(alld) if alld else float("nan")
    iqr = (np.percentile(alld, 75) - np.percentile(alld, 25)) if alld else float("nan")
    print(
        f"{pid} {p['seam_at'] // 60:3d}:00 {r['type']:5s} {r['bars']:2d}b applied {applied_ms:+5.0f} ms "
        f"({r['seam_band'][:10]:10s}) | kick-hit residual median {med:+6.0f} ms  IQR {iqr:4.0f}  "
        f"A kicks/beat {len(ka) / (r['bars'] * 4):.2f} B {len(kb) / (r['bars'] * 4):.2f} | windows: {' '.join(rows)}",
        flush=True,
    )
