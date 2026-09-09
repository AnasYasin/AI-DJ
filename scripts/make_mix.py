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
from src.data.track_fetcher import fetch_plan, fetch_track, preview_paths
from src.models.predict_model import CURVES, REPAIR_TRIES, plan_mix, repair_candidates

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
    plan_path = work_dir / "plan.json"
    tracks_dir = work_dir / "tracks"
    previews = preview_paths()

    def write_plan(plan: dict) -> None:
        for i, t in enumerate(plan["tracks"]):
            t["n"] = i + 1
        plan_path.write_text(json.dumps(plan, indent=1))

    def repair_slot(plan: dict, slot: int) -> bool:
        """Fill one empty slot against its verified neighbours. Never touches the rest."""
        for cand in repair_candidates(
            plan, slot, bpm, excluded, REPAIR_TRIES, compat_weight, min_energy_pct
        ):
            log.info("  slot %d: trying %s – %s", slot + 1, cand["artist"], cand["title"])
            res = fetch_track(
                cand["artist"],
                cand["title"],
                cand["track_id"],
                tracks_dir,
                previews.get(cand["track_id"]),
            )
            if res["status"] in ("ok", "cached"):
                next_link = cand.pop("next_link", None)
                plan["tracks"][slot] = cand
                if next_link and slot + 1 < len(plan["tracks"]):
                    plan["tracks"][slot + 1].update(next_link)
                return True
            excluded.add(cand["track_id"])
        return False

    plan = plan_mix(
        genre, bpm, n_tracks, curve, min_energy_pct=min_energy_pct, compat_weight=compat_weight
    )
    write_plan(plan)
    for attempt in range(1, max_rounds + 1):
        log.info(
            "round %d: %s",
            attempt,
            " | ".join(f"{t['artist']} – {t['title']}" for t in plan["tracks"]),
        )
        report = fetch_plan(plan_path, tracks_dir)
        failed = [
            i for i, r in enumerate(report["results"]) if r["status"] not in ("ok", "cached")
        ]
        if not failed:
            break
        excluded |= {plan["tracks"][i]["track_id"] for i in failed}
        # Keep every verified track. Ask the model for a replacement that fits the
        # empty slot's neighbours, try up to REPAIR_TRIES of them, and only if
        # none can be fetched fall back to replanning the whole set.
        unrepaired = [i for i in failed if not repair_slot(plan, i)]
        write_plan(plan)
        if not unrepaired:
            log.info("round %d: %d slot(s) repaired, set kept", attempt, len(failed))
            break
        log.warning(
            "round %d: slot(s) %s could not be repaired after %d candidates each — replanning the set",
            attempt,
            [i + 1 for i in unrepaired],
            REPAIR_TRIES,
        )
        plan = plan_mix(
            genre,
            bpm,
            n_tracks,
            curve,
            exclude_ids=excluded,
            min_energy_pct=min_energy_pct,
            compat_weight=compat_weight,
        )
        write_plan(plan)
    else:
        log.warning("gave up after %d rounds — rendering what verified", max_rounds)

    # Each track contributes its own window to the mix, so the target length
    # divides across the set.
    play_minutes = minutes / n_tracks if minutes else None
    report = render_plan(
        plan_path, work_dir / "tracks", out_path, play_minutes=play_minutes, total_minutes=minutes
    )
    write_plan(add_times(plan, report))
    return report


def add_times(plan: dict, report: dict) -> dict:
    """
    Write the render's timing back into the plan's track entries: `start` (when
    the track enters), `transition` and `bars` (the move that brought it in)
    and `fully_in` (when the overlap ends). The planner cannot know these; the
    windows and overlaps are decided at render time.
    """
    by_id = {t["track_id"]: t for t in plan["tracks"]}
    first = plan["tracks"][0]
    first.update({"start": "0:00", "transition": None, "bars": None, "fully_in": None})
    for tr in report.get("transitions", []):
        t = by_id.get(tr["to"])
        if t is not None:
            t.update(
                {
                    "start": tr["at"],
                    "transition": tr["type"],
                    "bars": tr["bars"],
                    "fully_in": tr["end"],
                    "seam_shift_ms": tr.get("seam_shift_total_ms", tr.get("seam_offset_before_ms")),
                    "seam_decided_by": tr.get("seam_band"),
                }
            )
    if report:
        plan["render"] = {
            "out": report.get("out"),
            "duration_min": round(report.get("duration_s", 0) / 60, 2),
            "target_bpm": report.get("target_bpm"),
            "lufs": report.get("lufs"),
            "true_peak_dbtp": report.get("true_peak_dbtp"),
        }
    return plan


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
