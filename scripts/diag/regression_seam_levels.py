"""Numbers-only check of a rendered regression set: level continuity at every seam.

For each pair: RMS of the last bar before the seam vs the first overlap bar (step at the seam),
last overlap bar vs first bar of B alone (step at the end), and the quietest bar inside the
overlap relative to the seam level (the hole). Compares shift/gains with a previous results file.
"""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data/external/regression_set")
prev_path = Path(sys.argv[2]) if len(sys.argv) > 2 else D / "results_before_edge_unity.json"
SR = 44100
res = json.loads((D / "results.json").read_text())
prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
man = json.loads(Path("tests/regression/manifest.json").read_text())


def load(t0, t1):
    cmd = [
        "ffmpeg",
        "-v",
        "quiet",
        "-ss",
        f"{t0:.3f}",
        "-t",
        f"{t1 - t0:.3f}",
        "-i",
        str(D / "regression_set.flac"),
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-f",
        "f32le",
        "-",
    ]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.float32)


def db(x):
    return 20 * np.log10(np.sqrt((x**2).mean()) + 1e-12)


print(
    f"{'pair':4s} {'seam':>5s} {'bars':>4s} {'step@seam':>9s} {'step@end':>8s} {'hole':>6s} | shift now/before   gains now / before"
)
for p in man["pairs"]:
    r = res[p["id"]]
    seam = p["seam_at"]
    bar = 4 * 60 / r["target_bpm"]
    ov_len = r["end_s"] - r["at_s"]
    y = load(seam - 5 * bar, seam + ov_len + 5 * bar)
    n = int(bar * SR)
    i_seam = int(5 * bar * SR)
    i_end = int((5 * bar + ov_len) * SR)
    # medians of 4 bars on each side: a one-bar fill or drop-out before a phrase
    # line is content, not a step, and must not trip the check
    med = lambda a, b: float(np.median([db(y[i : i + n]) for i in range(a, b - n + 1, n)]))  # noqa: E731
    last_a = med(i_seam - 4 * n, i_seam)
    first_ov = med(i_seam, i_seam + 4 * n)
    last_ov = med(i_end - 4 * n, i_end)
    first_b = med(i_end, i_end + 4 * n)
    inside = [db(y[i : i + n]) for i in range(i_seam, i_end - n, n)]
    hole = min(inside) - first_ov if inside else 0.0
    q = prev.get(p["id"], {})
    flag = " <-- step" if first_ov - last_a > 1.5 or abs(first_b - last_ov) > 1.5 else ""
    print(
        f"{p['id']:4s} {seam // 60:3d}:00 {r['bars']:4d} {first_ov - last_a:+9.1f} {first_b - last_ov:+8.1f} {hole:+6.1f} | "
        f"{r['seam_offset_before_ms']:+5.0f}/{q.get('seam_offset_before_ms', float('nan')):+5.0f}   "
        f"{r['overlap_gain_db']} / {q.get('overlap_gain_db')}{flag}"
    )
