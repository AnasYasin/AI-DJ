"""
Print a timeline of a rendered mix: what is playing when, and what happens at
each seam. Reads the plan and the render report produced by `render_plan`.

Run:
  python -m scripts.describe_mix data/external/mixes/stereo_report.json
"""

import argparse
import json
from pathlib import Path


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def to_seconds(stamp: str) -> float:
    m, s = stamp.split(":")
    return int(m) * 60 + int(s)


def describe(report: dict, plan: dict) -> None:
    tracks = plan["tracks"]
    trans = report["transitions"]

    print(f"  {Path(report['out']).name}")
    print(
        f"  {report['duration_s'] / 60:.2f} min | {report['target_bpm']:.0f} BPM | "
        f"{report.get('channels', 1)} ch | {report['lufs']} LUFS | peak {report['peak']}"
    )
    print(f"  genre {plan['genre']} | energy curve {plan['curve']}\n")

    # Each track runs from the end of the previous overlap to the start of the next.
    starts = [0.0] + [to_seconds(t["end"]) for t in trans]
    ends = [to_seconds(t["at"]) for t in trans] + [report["duration_s"]]

    for i, t in enumerate(tracks[: len(starts)]):
        print(f"  {mmss(starts[i])}–{mmss(ends[i])}  {t['artist']} – {t['title']}")
        print(
            f"           {t['bpm']:.0f}→{report['target_bpm']:.0f} BPM, {t['key']}, "
            f"energy {t['energy']:.2f}, curve target {t.get('energy_target01', float('nan')):.2f}"
        )
        if i < len(trans):
            tr = trans[i]
            demoted = (
                f", rules said {tr['measured'].upper()}" if tr["measured"] != tr["type"] else ""
            )
            print(
                f"  {tr['at']}–{tr['end']}  ── {tr['type'].upper()}, {tr['bars']} bars"
                f"{demoted}, seam {tr['seam_offset_ms']:+.1f} ms"
                f" (was {tr['seam_offset_before_ms']:+.0f} ms)"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Describe a rendered mix as a timeline.")
    p.add_argument("report", help="JSON: either one render report or {label: report}")
    p.add_argument("--plan", action="append", default=[], help="plan JSON, once per mix, in order")
    args = p.parse_args()

    data = json.loads(Path(args.report).read_text())
    reports = data if "transitions" not in data else {"MIX": data}
    for (label, rep), plan_path in zip(reports.items(), args.plan):
        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
        describe(rep, json.loads(Path(plan_path).read_text()))
