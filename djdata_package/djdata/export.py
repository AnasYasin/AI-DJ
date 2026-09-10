"""Flatten out/seams.jsonl and out/curves/*.csv into three CSV files: seams.csv (one row per seam,
scalar parameters), curves.csv (one row per beat), qa.csv (diagnostics to filter bad seams)."""

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("djdata.export")

QA_COLS = ["seam_id", "tier", "tempo_bpm", "rate_A", "rate_B", "bar_fix_A", "bar_fix_B", "onset_shift_A_s", "onset_shift_B_s",
           "absent_region_A_beats", "absent_region_B_beats", "unexplained_mean_overlap"]


def export(out_dir: Path) -> dict:
    rows = [json.loads(l) for l in open(out_dir / "seams.jsonl")] if (out_dir / "seams.jsonl").exists() else []
    rows = list({r["seam_id"]: r for r in rows}.values())   # append-only log: a re-analysed seam keeps its last row
    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        for tag in ("A", "B"):
            for b, c in r.get(f"{tag}_crossings", {}).items():
                f[f"{tag}_b{b}_m6_t"], f[f"{tag}_b{b}_m12_t"] = c["m6"], c["m12"]
        f["n_volume_cuts"] = len(r.get("volume_cuts", []))
        f["n_unexplained_spikes"] = len(r.get("unexplained_spikes", []))
        f["volume_cuts"] = json.dumps(r.get("volume_cuts", []))
        f["unexplained_spikes"] = json.dumps(r.get("unexplained_spikes", []))
        f["match_rate_A"], f["match_rate_B"] = (r.get("match_rate") or [None, None])
        for k, v in r.get("ambiguous_frac_overlap", {}).items():
            f[f"ambiguous_{k}"] = v
        for k, v in r.get("quiet_frac_overlap", {}).items():
            f[f"quiet_{k}"] = v
        f["time_total_s"] = r.get("timing_s", {}).get("total")
        flat.append(f)
    seams = pd.DataFrame(flat)
    seams.to_csv(out_dir / "seams.csv", index=False)
    qa = seams[[c for c in seams.columns if c in QA_COLS or c.startswith("ambiguous_") or c.startswith("quiet_")]] if len(seams) else seams
    qa.to_csv(out_dir / "qa.csv", index=False)
    parts = []
    for p in sorted((out_dir / "curves").glob("*.csv")):
        d = pd.read_csv(p)
        d.insert(0, "seam_id", p.stem)
        parts.append(d)
    curves = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    curves.to_csv(out_dir / "curves.csv", index=False)
    log.info("export: %d seams, %d beat rows → %s", len(seams), len(curves), out_dir)
    return {"seams": len(seams), "beats": len(curves)}
