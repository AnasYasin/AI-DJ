"""
One command from intent to a finished mix.

    plan → fetch and verify → replan around whatever could not be fetched → render

The replan step is the reason this exists. The catalog is built from 30-second
previews, so a plan routinely names a track with no findable full version, and a
set that renders three tracks short is not the set that was planned. Each round
drops the tracks that failed and asks the planner for a fresh set of the right
length, keeping the energy curve intact.

Run:
  python -m scripts.make_mix --genre techno --bpm 128 136 --curve build \\
      --n 3 --minutes 10 --out data/external/mixes/techno.flac
"""

import argparse
import json
import logging
from pathlib import Path

from src.audio.audio_mixer import render_plan
from src.data.track_fetcher import fetch_plan
from src.models.predict_model import CURVES, plan_mix

log = logging.getLogger(__name__)


def make_mix(
    genre: str,
    bpm: tuple[float, float],
    out_path: str | Path,
    n_tracks: int = 3,
    curve: str = "build",
    minutes: float | None = None,
    work_dir: str | Path = "data/interim/mixes",
    max_rounds: int = 4,
    min_energy_pct: float | None = None,
    compat_weight: float = 0.0,
) -> dict:
    out_path = Path(out_path)
    work_dir = Path(work_dir) / out_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    excluded: set[str] = set()
    for attempt in range(1, max_rounds + 1):
        plan = plan_mix(
            genre,
            bpm,
            n_tracks,
            curve,
            exclude_ids=excluded,
            min_energy_pct=min_energy_pct,
            compat_weight=compat_weight,
        )
        plan_path = work_dir / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=1))
        log.info(
            "round %d: %s",
            attempt,
            " | ".join(f"{t['artist']} – {t['title']}" for t in plan["tracks"]),
        )

        report = fetch_plan(plan_path, work_dir / "tracks")
        failed = {
            plan["tracks"][i]["track_id"]
            for i, r in enumerate(report["results"])
            if r["status"] not in ("ok", "cached")
        }
        if not failed:
            break
        excluded |= failed
        log.warning(
            "round %d: %d of %d tracks unfetchable, replanning", attempt, len(failed), n_tracks
        )
    else:
        log.warning("gave up replanning after %d rounds — rendering what verified", max_rounds)

    # Each track contributes its own window to the mix, so the target length
    # divides across the set.
    play_minutes = minutes / n_tracks if minutes else None
    return render_plan(plan_path, work_dir / "tracks", out_path, play_minutes=play_minutes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Plan, fetch and render a mix in one go.")
    p.add_argument("--genre", required=True)
    p.add_argument("--bpm", nargs=2, type=float, required=True, metavar=("LO", "HI"))
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--curve", choices=list(CURVES), default="build")
    p.add_argument("--minutes", type=float, default=None, help="target mix length")
    p.add_argument("--work-dir", default="data/interim/mixes")
    p.add_argument(
        "--min-energy-pct", type=float, default=None, help="drop the quietest N%% of the pool"
    )
    p.add_argument(
        "--compat-weight", type=float, default=0.0, help="raise to plan for long overlaps"
    )
    args = p.parse_args()

    rep = make_mix(
        args.genre,
        tuple(args.bpm),
        args.out,
        args.n,
        args.curve,
        args.minutes,
        args.work_dir,
        min_energy_pct=args.min_energy_pct,
        compat_weight=args.compat_weight,
    )
    print(
        f"\n{rep['out']}  {rep['duration_s'] / 60:.1f} min @ {rep['target_bpm']:.0f} bpm"
        f"  {rep['lufs']:.1f} LUFS"
    )
    for tr in rep["transitions"]:
        print(
            f"  {tr['from']}  →  {tr['to']}   {tr['type'].upper()} {tr['bars']} bars"
            f"  seam {tr['seam_offset_ms']:+.1f} ms"
        )
