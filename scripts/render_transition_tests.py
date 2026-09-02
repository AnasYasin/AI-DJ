"""
Render the same two tracks once per transition type and report the numbers.

Run from the repo root:
  python -m scripts.render_transition_tests trackA.m4a trackB.m4a
  python -m scripts.analyse_transition_tests
"""

import json
import logging
from pathlib import Path
import sys

from src.audio.audio_mixer import render_mix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
DEFAULT_DIR = Path("data/external/transition_tests/tracks")
if len(sys.argv) >= 3:
    A, B = Path(sys.argv[1]), Path(sys.argv[2])
else:  # the pair the recipes were tuned against
    A = next(DEFAULT_DIR.glob("4f33cf8656f2.*"))  # Something More (Voorn Remix), 129 bpm
    B = next(DEFAULT_DIR.glob("967eeec2621a.*"))  # Drum Death, 128 bpm, intro→buildup→drop
OUT = Path("data/external/transition_tests")
OUT.mkdir(parents=True, exist_ok=True)

reports = {}
for ttype in ["blend", "melt", "fade", "wave", "rise", "slam", "drop"]:
    print(f"\n{'=' * 70}\n  {ttype.upper()}\n{'=' * 70}", flush=True)
    reports[ttype] = render_mix(
        [A, B],
        OUT / f"{ttype}.flac",
        genre="techno",
        play_minutes=2.5,
        force_type=ttype,
        drop_align=False,
    )

# the automatic path: rules → intent gate → drop promotion, nothing forced
print(f"\n{'=' * 70}\n  AUTO (curve=build, drop_align on)\n{'=' * 70}", flush=True)
reports["auto"] = render_mix(
    [A, B], OUT / "auto.flac", genre="techno", play_minutes=2.5, curve="build"
)

Path(OUT / "report.json").write_text(json.dumps(reports, indent=1))

hdr = f"{'render':<8}{'type':>7}{'bars':>6}{'seam before':>14}{'after':>10}{'LUFS':>8}{'peak':>7}{'len':>7}"
print("\n\n" + hdr)
print("-" * len(hdr))
for t, r in reports.items():
    tr = r["transitions"][0]
    print(
        f"{t:<8}{tr['type']:>7}{tr['bars']:>6}{tr['seam_offset_before_ms']:>12.1f}ms"
        f"{tr['seam_offset_ms']:>8.1f}ms{r['lufs']:>8.1f}{r['peak']:>7.2f}{r['duration_s'] / 60:>6.1f}m"
    )
