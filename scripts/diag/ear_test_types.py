"""Type sweep: per genre, the best-matched planner pair rendered once per transition type."""

import json
import logging
from pathlib import Path
import sys
import time
import traceback

import soundfile as sf

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from src.audio.audio_mixer import RECIPES, SR, render_mix

OUT = Path("data/external/ear_test_2026-09-07")
TRACKS = Path("data/external/test_tracks")
GENRES = {"tech house": "tech_house", "techno": "techno", "drum and base": "drum_and_base"}
CURVE = "arc"
PAD_S = 120.0
report = json.loads((OUT / "report.json").read_text())


def mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


rows = []
for genre, gdir in GENRES.items():
    done = [e for e in report if e["genre"] == genre and "error" not in e]
    if not done:
        logging.error("%s: no rendered pairs to sweep", genre)
        continue
    best = max(done, key=lambda e: e["compatibility"])
    a_id, b_id = best["file"].split("_")[-1].replace(".flac", "").split("-")
    plan = json.loads((OUT / gdir / "plan.json").read_text())
    by_id = {t["track_id"]: t for t in plan["tracks"]}
    targets = [by_id[a_id]["energy_target01"], by_id[b_id]["energy_target01"]]
    paths = [TRACKS / gdir / f"{a_id}.m4a", TRACKS / gdir / f"{b_id}.m4a"]
    tdir = OUT / gdir / "types"
    tdir.mkdir(exist_ok=True)
    logging.info(
        "=== %s: sweeping types on pair %d (%s → %s, compatibility %.2f)",
        genre,
        best["pair"],
        a_id,
        b_id,
        best["compatibility"],
    )
    for ttype in RECIPES:
        tmp = tdir / f"_full_{ttype}.flac"
        t0 = time.time()
        try:
            rep = render_mix(
                paths, tmp, genre=genre, curve=None, energy_targets=None, force_type=ttype
            )
        except Exception as e:
            logging.error("%s %s failed: %s", genre, ttype, e)
            traceback.print_exc()
            rows.append({"genre": genre, "type": ttype, "error": str(e)})
            continue
        tr = rep["transitions"][0]
        at = int(tr["at"].split(":")[0]) * 60 + int(tr["at"].split(":")[1])
        end = int(tr["end"].split(":")[0]) * 60 + int(tr["end"].split(":")[1])
        y, _ = sf.read(tmp)
        c0, c1 = max(int((at - PAD_S) * SR), 0), min(int((end + PAD_S) * SR), len(y))
        name = f"{ttype}_{a_id}-{b_id}.flac"
        sf.write(tdir / name, y[c0:c1], SR)
        tmp.unlink()
        rows.append(
            {
                "genre": genre,
                "type": ttype,
                "file": f"{gdir}/types/{name}",
                "pair": best["pair"],
                "A": best["A"],
                "B": best["B"],
                "bars": tr["bars"],
                "lands_on_drop": tr["lands_on_drop"],
                "A_tail_section": tr["tail_section"],
                "B_head_section": tr["head_section"],
                "overlap_in_clip": f"{mmss(at - c0 / SR)} – {mmss(end - c0 / SR)}",
                "seam_ms": tr["seam_offset_ms"],
                "seam_band": tr["seam_band"],
                "drift_ms": tr["seam_drift_ms"],
                "overlap_gain_db": tr["overlap_gain_db"],
                "render_s": round(time.time() - t0),
            }
        )
        logging.info("%s %s done: %d bars, %.0fs", genre, ttype, tr["bars"], rows[-1]["render_s"])
        (OUT / "report_types.json").write_text(json.dumps(rows, indent=1))
lines = [
    "",
    "## Type sweep",
    "",
    "Same pair per genre (the best-matched of the four above), rendered once per transition type with no curve and no plan,",
    "so the only difference between clips is the transition. `drop` only lands on a drop when B has one at least a phrase past its cue-in.",
    "",
    "| genre | clip | bars | lands on drop | A tail | B head | overlap in clip | seam ms (band) | drift ms | gain dB |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    if "error" in r:
        lines.append(f"| {r['genre']} | {r['type']} FAILED: {r['error'][:60]} | | | | | | | | |")
        continue
    lines.append(
        f"| {r['genre']} | {r['file']} | {r['bars']} | {r['lands_on_drop']} | {r['A_tail_section']} | {r['B_head_section']} | {r['overlap_in_clip']} | {r['seam_ms']} ({r['seam_band']}) | {r['drift_ms']} | {r['overlap_gain_db']} |"
    )
with (OUT / "README.md").open("a") as f:
    f.write("\n".join(lines) + "\n")
logging.info("TYPES DONE: %d clips", sum("error" not in r for r in rows))
