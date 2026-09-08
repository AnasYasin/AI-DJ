import logging
import sys

import numpy as np

logging.disable(logging.WARNING)
sys.path.insert(0, ".")
from src.audio.audio_mixer import SR, _prepare_track, measure_seam_offset

target = 128.5
A = _prepare_track("data/external/transition_tests/tracks/4f33cf8656f2.webm", target, 32, 64)
B = _prepare_track("data/external/transition_tests/tracks/967eeec2621a.webm", target, 32, 64)
bar = 4 * 60 / target
n_bars = 64
a0 = int(A["bars"][A["cue_out_bar"]] * SR)
b0 = int(B["bars"][B["cue_in_bar"]] * SR)
n_ov = int(n_bars * bar * SR)
tail = A["audio"][a0 : a0 + n_ov]
head = B["audio"][b0 : b0 + n_ov]
n_ov = min(len(tail), len(head))
tail, head = tail[:n_ov], head[:n_ov]
print(
    f"A rate={A['rate']:.4f} (bpm est {60 / np.median(np.diff(A['beat_times'])) * A['rate']:.2f})  B rate={B['rate']:.4f}"
)
full = measure_seam_offset(tail, head, target)
print(f"whole-overlap seam offset (what render_mix reports): {full * 1000:+.1f} ms")
# apply the same single shift render_mix would
b0 += int(round(full * SR))
head = B["audio"][b0 : b0 + n_ov]
print(
    "after render_mix-style single shift, offset measured per 4-bar window across the 64-bar overlap:"
)
w = int(4 * bar * SR)
for k in range(0, n_ov - w, w * 2):
    off = measure_seam_offset(tail[k : k + w], head[k : k + w], target) * 1000
    print(
        f"  bars {k // int(bar * SR):3d}-{(k + w) // int(bar * SR):3d}  t={k / SR:6.1f}s  beat offset {off:+7.1f} ms  ({abs(off) / (1000 * 60 / target):.2f} beat)"
    )
