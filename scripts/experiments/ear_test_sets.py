"""
Render one mellow set per genre from held audio for an ear test.

    PYTHONPATH=. python scripts/experiments/ear_test_sets.py data/external/ear_test/sets_2026-10-15

Chill curve, 6 records, pool = records whose full audio is held (nothing searched on
YouTube), play length = the genre's measured median. Writes <out>/<genre>.flac, the plan
with render times next to it, and a judgement table `summary.csv` (energy percentile of
each pick in its pool, key clashes per join, BPM climb) for listening with numbers in hand.
"""

import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scripts.make_mix import add_times

from src.audio.audio_mixer import render_plan
from src.audio.key_shift import CLASH_DISTANCE
from src.data.dataset import load_tracks
from src.data.track_fetcher import fetch_plan
from src.models.predict_model import plan_mix

# BPM windows sit on the slower side of each genre's held pool (its 25th to 50th percentile).
BPM = {
    "house": (120, 126),
    "techno": (126, 132),
    "tech house": (122, 127),
    "melodic house": (121, 125),
    "afro house": (119, 124),
    "trance": (132, 138),
    "drum and bass": (170, 175),
}
# ponytail: the same recording under two artist spellings (Bongi Dude / Bongi Dube, cos 1.0);
# the artist veto cannot see it. 31 such pairs among 12,153 held records, dedupe is a data job.
EXCLUDE = {"afro house": {"d3231e8dc492"}}


def main(out_dir: Path, genres: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "render.log"), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("ear_test")
    tracks = load_tracks()
    full = tracks[tracks.audio == "full"]
    rows = []
    for genre in genres:
        lo, hi = BPM[genre]
        pool = full[(full.genre == genre) & full.bpm.between(lo, hi)]
        ex = EXCLUDE.get(genre, set())
        ex_ids = set(pool[pool.source_ref.isin(ex)].track_id)
        plan = plan_mix(genre, (lo, hi), 6, "chill", audio="full", exclude_ids=ex_ids or None)
        work = out_dir / "work" / genre.replace(" ", "_")
        work.mkdir(parents=True, exist_ok=True)
        plan_path = work / "plan.json"
        for i, t in enumerate(plan["tracks"]):
            t["n"] = i + 1
        plan_path.write_text(json.dumps(plan, indent=1))
        log.info(
            "== %s: %s", genre, " | ".join(f"{t['artist']} – {t['title']}" for t in plan["tracks"])
        )
        fetched = fetch_plan(plan_path, work / "tracks")
        bad = [r for r in fetched["results"] if r["status"] not in ("ok", "cached")]
        if bad:
            log.error("%s: %d record(s) not fetched, skipping the set: %s", genre, len(bad), bad)
            continue
        out = out_dir / f"{genre.replace(' ', '_')}.flac"
        report = render_plan(plan_path, work / "tracks", out)
        plan = add_times(plan, report)
        (out_dir / f"{genre.replace(' ', '_')}.plan.json").write_text(json.dumps(plan, indent=1))
        e = np.array([t["energy"] for t in plan["tracks"]])
        bpm = np.array([t["bpm"] for t in plan["tracks"]])
        pct = [float((pool.energy_mean < v).mean()) for v in e]
        cam = [t.get("cam_dist") for t in plan["tracks"][1:]]
        rows.append(
            {
                "genre": genre,
                "records": len(e),
                "pool": len(pool),
                "minutes": plan["render"]["duration_min"],
                "energy_pct_min": round(min(pct), 2),
                "energy_pct_median": round(float(np.median(pct)), 2),
                "energy_pct_max": round(max(pct), 2),
                "clash_share": round(
                    float(np.mean([c is not None and c > CLASH_DISTANCE for c in cam])), 2
                ),
                "same_key_share": round(float(np.mean([c == 0 for c in cam])), 2),
                "bpm_first_to_last_pct": round(100 * (bpm[-1] / bpm[0] - 1), 1),
                "transitions": " ".join(t["transition"] or "-" for t in plan["tracks"]),
            }
        )
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
        log.info("%s done: %s", genre, rows[-1])
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    out = Path(sys.argv[1])
    main(out, sys.argv[2:] or list(BPM))
