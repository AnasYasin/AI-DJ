"""
The same tracks rendered once with the mixer's rules and once per style group, for the ear.

    PYTHONPATH=. python scripts/experiments/ear_test_groups.py data/external/ear_test/groups_2026-11-18 \\
        data/external/ear_test/sets_2026-10-15/work/techno "techno blend" "techno cut"

The work folder holds `plan.json` and `tracks/` from an earlier `make_mix` or ear-test run, so nothing
is planned or fetched here. The plan's curve is dropped for all mixes: a chill curve forbids every cut,
which vetoed all four planned cuts of `techno cut` on 2026-11-18, and the test is of the styles. Writes rules.flac and <group>.flac, and `seams.csv` with one row per mix
and seam: the planned type, the recipe the mixer rendered, whether the rules let the plan stand, the
overlap in bars, where the seam is in the mix and the bars the incoming track plays.
"""

import json
import logging
from pathlib import Path
import sys

import pandas as pd

from src.audio.audio_mixer import render_plan
from src.models.set_plan import set_plan


def main(out_dir: Path, work: Path, groups: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "render.log"), logging.StreamHandler()],
        force=True,
    )
    n_tracks = len(list((work / "tracks").iterdir()))
    plan_path = out_dir / "plan.json"
    plan_path.write_text(
        json.dumps(json.loads((work / "plan.json").read_text()) | {"curve": None})
    )
    rows = []
    for name in ["rules", *groups]:
        style = [{"group": name, "dj": None, "weight": 1.0}]
        plan = None if name == "rules" else set_plan(style, n_tracks)
        out = out_dir / f"{name.replace(' ', '_')}.flac"
        report = render_plan(plan_path, work / "tracks", out, seam_plan=plan)
        for i, seam in enumerate(report["transitions"]):
            keep = ("planned", "type", "plan_check", "measured", "bars", "at", "in_bar", "out_bar")
            wanted = {f"plan_{k}": v for k, v in plan[i].items() if k != "type"} if plan else {}
            rows.append({"mix": name, "seam": i + 1} | {k: seam.get(k) for k in keep} | wanted)
        pd.DataFrame(rows).to_csv(out_dir / "seams.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
