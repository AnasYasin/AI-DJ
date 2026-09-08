"""A/B the seam alignment: old (A-vs-B envelope xcorr) vs new (own-grid residuals) on two pairs."""

import json
import logging
from pathlib import Path
import sys
import time

import pandas as pd
import soundfile as sf

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from src.audio import audio_mixer as am
from src.models.predict_model import plan_mix

OUT = Path("data/external/ear_test_2026-09-07/seamfix")
OUT.mkdir(parents=True, exist_ok=True)
TRACKS = Path("data/external/test_tracks")
PAD = 120.0
SR = am.SR
manifest = pd.read_csv(TRACKS / "manifest.csv")


def mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


jobs = []
techno = json.loads(Path("data/external/ear_test_2026-09-07/techno/plan.json").read_text())[
    "tracks"
]
a, b = techno[2], techno[3]
jobs.append(
    (
        "techno_pair2",
        "techno",
        "arc",
        [TRACKS / "techno" / f"{a['track_id']}.m4a", TRACKS / "techno" / f"{b['track_id']}.m4a"],
        [a["energy_target01"], b["energy_target01"]],
        f"{a['artist']} – {a['title']}",
        f"{b['artist']} – {b['title']}",
    )
)
rows = manifest[manifest["genre"] == "melodic house"]
ids = [i for i in rows["track_id"] if (TRACKS / "melodic_house" / f"{i}.m4a").exists()]
plan = plan_mix(
    "melodic house",
    (float(rows["bpm"].min()) - 1, float(rows["bpm"].max()) + 1),
    n_tracks=4,
    curve="chill",
    track_ids=ids,
    compat_weight=5.0,
)
(OUT / "melodic_house_plan.json").write_text(json.dumps(plan, indent=1))
a, b = plan["tracks"][0], plan["tracks"][1]
logging.info(
    "melodic house chill plan: %s",
    [(t["track_id"], t.get("bpm"), t["energy_target01"]) for t in plan["tracks"]],
)
jobs.append(
    (
        "house_low",
        "melodic house",
        "chill",
        [
            TRACKS / "melodic_house" / f"{a['track_id']}.m4a",
            TRACKS / "melodic_house" / f"{b['track_id']}.m4a",
        ],
        [a["energy_target01"], b["energy_target01"]],
        f"{a['artist']} – {a['title']}",
        f"{b['artist']} – {b['title']}",
    )
)

report = []
for name, genre, curve, paths, targets, an, bn in jobs:
    for method in ("xcorr", "grid"):
        am.SEAM_METHOD = method
        tag = "OLD_xcorr" if method == "xcorr" else "NEW_grid"
        existing = [f for f in OUT.glob(f"{name}_{tag}_*.flac")]
        prev = (
            {r["file"]: r for r in json.loads((OUT / "report.json").read_text())}
            if (OUT / "report.json").exists()
            else {}
        )
        if existing and existing[0].name in prev:
            report.append(prev[existing[0].name])
            logging.info("kept %s", existing[0].name)
            continue
        for p in paths:  # the calibration depends on the method, so the segment cache is fine but the prepared grid is not cached anyway
            pass
        tmp = OUT / f"_full_{name}_{method}.flac"
        t0 = time.time()
        rep = am.render_mix(paths, tmp, genre=genre, curve=curve, energy_targets=targets)
        tr = rep["transitions"][0]
        at = int(tr["at"].split(":")[0]) * 60 + int(tr["at"].split(":")[1])
        end = int(tr["end"].split(":")[0]) * 60 + int(tr["end"].split(":")[1])
        y, _ = sf.read(tmp)
        c0, c1 = max(int((at - PAD) * SR), 0), min(int((end + PAD) * SR), len(y))
        fname = f"{name}_{'OLD_xcorr' if method == 'xcorr' else 'NEW_grid'}_{tr['type']}.flac"
        sf.write(OUT / fname, y[c0:c1], SR)
        tmp.unlink()
        row = {
            "pair": name,
            "method": method,
            "file": fname,
            "A": an,
            "B": bn,
            "type": tr["type"],
            "bars": tr["bars"],
            "regime": tr["regime"],
            "overlap_in_clip": f"{mmss(at - c0 / SR)} – {mmss(end - c0 / SR)}",
            "seam_before_ms": tr["seam_offset_before_ms"],
            "seam_after_ms": tr["seam_offset_ms"],
            "seam_band": tr["seam_band"],
            "drift_ms": tr["seam_drift_ms"],
            "windows": tr["seam_windows_off"],
            "render_s": round(time.time() - t0),
        }
        report.append(row)
        logging.info("DONE %s", row)
        (OUT / "report.json").write_text(json.dumps(report, indent=1))
lines = [
    "# Seam alignment A/B — old (envelope xcorr) vs new (own-grid residuals)",
    "",
    "| pair | method | clip | type | bars | regime | overlap in clip | shift applied ms | band | drift ms | windows |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for r in report:
    lines.append(
        f"| {r['pair']} | {r['method']} | {r['file']} | {r['type']} | {r['bars']} | {r['regime']} | {r['overlap_in_clip']} | {r['seam_before_ms']} | {r['seam_band']} | {r['drift_ms']} | {r['windows']} |"
    )
lines += ["", *[f"- {r['pair']}: A {r['A']} → B {r['B']}" for r in report[::2]]]
(OUT / "README.md").write_text("\n".join(lines) + "\n")
logging.info("SEAMFIX DONE")
