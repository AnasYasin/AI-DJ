import logging
import sys
import time

import numpy as np

logging.disable(logging.WARNING)
sys.path.insert(0, ".")
import librosa

from src.audio.audio_mixer import render_mix
from src.data import audio_segmenter as seg

paths = sys.argv[2:]
do_render = sys.argv[1] == "render"
for p in paths:
    t0 = time.time()
    info = seg.segment(p)
    dt = time.time() - t0
    # what the OLD kick-phase heuristic would have picked, for comparison
    y, sr = librosa.load(p, sr=seg.SR, mono=True)
    _, kb, kconf = seg._kick_phase_grid(y, sr, info["bpm"])
    beats = np.array(info["beats"])
    bars = np.array(info["bars"])
    old_phase_on_new_grid = int(np.argmin(np.abs(beats - kb[len(kb) // 2]))) % 4
    new_phase = int(np.searchsorted(beats, bars[0])) % 4
    print(
        f"{p.split('/')[-1][:30]:30s} {dt:5.1f}s  downbeats={info['downbeat_source']} conf={info['downbeat_confidence']:.3f} "
        f"phrase_offset={info['phrase_offset']}  kick-phase pick agrees={old_phase_on_new_grid == new_phase} (kick conf {kconf:.3f})"
    )
    print("    sections:", [(s["label"], s["bars"]) for s in info["sections"]])
    bounds = sorted({s["bars"][0] for s in info["sections"]} - {0})
    print(
        "    boundaries mod 8:",
        [b % 8 for b in bounds],
        " (phrase offset",
        info["phrase_offset"],
        ")",
    )
if do_render:
    out = "/tmp/claude-1000/-home-anas-yasin-projects-AI-DJ/7b53d0e9-c755-4a45-ae4b-a83b3ac18d8e/scratchpad/verify_%s.flac"
    for force in (None, "blend"):
        rep = render_mix(paths[:2], out % (force or "auto"), genre="techno", force_type=force)
        tr = rep["transitions"][0]
        print(
            f"\nRENDER force={force}: type={tr['type']} bars={tr['bars']} at={tr['at']} end={tr['end']} lead_swap={tr['lead_swap']}"
        )
        print(
            f"   out_bar(A)={tr['out_bar']} in_bar(B)={tr['in_bar']} phrase_offsets={tr['phrase_offsets']} downbeats={tr['downbeats']}"
        )
        print(
            f"   seam avg {tr['seam_offset_ms']} ms (before {tr['seam_offset_before_ms']}), drift (median window) {tr['seam_drift_ms']} ms, windows off {tr['seam_windows_off']}"
        )
        pa, pb = tr["phrase_offsets"]
        print(
            f"   A cue-out on A's phrase grid: {(tr['out_bar'] - pa) % 8 == 0}   B entry on B's phrase grid: {(tr['in_bar'] - pb) % 8 == 0}   overlap whole phrases: {tr['bars'] % 8 == 0}"
        )
