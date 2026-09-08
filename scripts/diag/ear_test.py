"""Ear-test clips: 4 planner-chosen pairs per genre, system-chosen transition, trimmed to
2 min before the overlap + overlap + 2 min after."""

import json
import logging
from pathlib import Path
import sys
import time
import traceback

import pandas as pd
import soundfile as sf

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("numba", "matplotlib", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
from src.audio.audio_mixer import SR, render_mix
from src.data.audio_segmenter import segment
from src.models.predict_model import plan_mix

OUT = Path("data/external/ear_test_2026-09-07")
OUT.mkdir(parents=True, exist_ok=True)
TRACKS = Path("data/external/test_tracks")
GENRES = {"tech house": "tech_house", "techno": "techno", "drum and base": "drum_and_base"}
CURVE = "arc"
PAD_S = 120.0
manifest = pd.read_csv(TRACKS / "manifest.csv")
summary = []


def mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def section_at(info, bar):
    for s in info["sections"]:
        if s["bars"][0] <= bar < s["bars"][1]:
            return s["label"]
    return None


for genre, gdir in GENRES.items():
    rows = manifest[manifest["genre"] == genre]
    ids = [i for i in rows["track_id"] if (TRACKS / gdir / f"{i}.m4a").exists()]
    lo, hi = float(rows["bpm"].min()) - 1, float(rows["bpm"].max()) + 1
    logging.info("=== %s: %d local tracks, bpm %.0f-%.0f", genre, len(ids), lo, hi)
    plan = plan_mix(genre, (lo, hi), n_tracks=8, curve=CURVE, track_ids=ids, compat_weight=5.0)
    tracks = plan["tracks"]
    logging.info("plan: %s", [(t["track_id"], t.get("bpm"), t["energy_target01"]) for t in tracks])
    (OUT / gdir).mkdir(exist_ok=True)
    (OUT / gdir / "plan.json").write_text(json.dumps(plan, indent=1))
    pairs = [(k, k + 1) for k in range(0, len(tracks) - 1, 2)][:4]
    for n, (i, j) in enumerate(pairs, 1):
        a, b = tracks[i], tracks[j]
        paths = [TRACKS / gdir / f"{a['track_id']}.m4a", TRACKS / gdir / f"{b['track_id']}.m4a"]
        targets = [a["energy_target01"], b["energy_target01"]]
        tmp = OUT / gdir / f"_full_pair{n}.flac"
        t0 = time.time()
        try:
            rep = render_mix(paths, tmp, genre=genre, curve=CURVE, energy_targets=targets)
        except Exception as e:
            logging.error("pair %d failed: %s", n, e)
            traceback.print_exc()
            summary.append({"genre": genre, "pair": n, "error": str(e)})
            continue
        tr = rep["transitions"][0]
        at = int(tr["at"].split(":")[0]) * 60 + int(tr["at"].split(":")[1])
        end = int(tr["end"].split(":")[0]) * 60 + int(tr["end"].split(":")[1])
        y, _ = sf.read(tmp)
        c0, c1 = max(int((at - PAD_S) * SR), 0), min(int((end + PAD_S) * SR), len(y))
        name = f"pair{n}_{tr['type']}_{a['track_id']}-{b['track_id']}.flac"
        sf.write(OUT / gdir / name, y[c0:c1], SR)
        tmp.unlink()
        infoA, infoB = segment(paths[0]), segment(paths[1])
        entry = {
            "genre": genre,
            "pair": n,
            "file": name,
            "A": f"{a['artist']} – {a['title']}",
            "B": f"{b['artist']} – {b['title']}",
            "A_bpm": a.get("bpm"),
            "B_bpm": b.get("bpm"),
            "target_bpm": rep["target_bpm"],
            "curve_values": targets,
            "regime": tr["regime"],
            "transition": tr["type"],
            "measured": tr["measured"],
            "bars": tr["bars"],
            "lands_on_drop": tr["lands_on_drop"],
            "A_tail_section": tr["tail_section"],
            "B_head_section": tr["head_section"],
            "overlap_in_clip": f"{mmss(at - c0 / SR)} – {mmss(end - c0 / SR)}",
            "lead_swap_in_clip": mmss(
                int(tr["lead_swap"].split(":")[0]) * 60
                + int(tr["lead_swap"].split(":")[1])
                - c0 / SR
            )
            if tr["lead_swap"]
            else None,
            "seam_ms": tr["seam_offset_ms"],
            "seam_band": tr["seam_band"],
            "drift_ms": tr["seam_drift_ms"],
            "overlap_gain_db": tr["overlap_gain_db"],
            "compatibility": tr["compatibility"],
            "keys": [None, None],
            "downbeat_conf": [infoA.get("downbeat_confidence"), infoB.get("downbeat_confidence")],
            "clip_s": round((c1 - c0) / SR, 1),
            "render_s": round(time.time() - t0),
        }
        summary.append(entry)
        logging.info(
            "pair %d done: %s %d bars, clip %.0fs, %.0fs render",
            n,
            tr["type"],
            tr["bars"],
            entry["clip_s"],
            entry["render_s"],
        )
        (OUT / "report.json").write_text(json.dumps(summary, indent=1))

lines = [
    "# Ear test clips — 2026-09-07",
    "",
    "Each clip: 2 min of track A, the overlap, 2 min of track B. The overlap starts at 2:00 in every clip",
    "unless A's window was shorter than 2 min (then see `overlap_in_clip`). Pairs come from the planner",
    f"(8-track `{CURVE}` set from the 20 local verified tracks per genre), transition type chosen by the mixer.",
    "",
    "| genre | clip | transition | bars | regime | lands on drop | A tail | B head | overlap in clip | seam ms | drift ms | gain dB |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for e in summary:
    if "error" in e:
        lines.append(
            f"| {e['genre']} | pair {e['pair']} | FAILED: {e['error'][:60]} | | | | | | | | | |"
        )
        continue
    lines.append(
        f"| {e['genre']} | {e['file']} | {e['transition']} (measured {e['measured']}) | {e['bars']} | {e['regime']} | {e['lands_on_drop']} | {e['A_tail_section']} | {e['B_head_section']} | {e['overlap_in_clip']} | {e['seam_ms']} | {e['drift_ms']} | {e['overlap_gain_db']} |"
    )
lines += ["", "Tracks per clip:", ""]
for e in summary:
    if "error" not in e:
        lines.append(
            f"- {e['genre']} pair {e['pair']}: **A** {e['A']} ({e['A_bpm']} bpm) → **B** {e['B']} ({e['B_bpm']} bpm), set {e['target_bpm']:.1f} bpm, curve values {e['curve_values']}"
        )
(OUT / "README.md").write_text("\n".join(lines) + "\n")
logging.info("ALL DONE: %d clips in %s", sum("error" not in e for e in summary), OUT)
