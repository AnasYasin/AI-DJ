"""Build the seam regression set: one long file of independent pairs in fixed slots.

Each pair in the manifest is rendered in full with the CURRENT mixer (render_mix), then the
piece from LEAD_S before the overlap to TAIL_S after it is cut out and placed so the overlap
starts exactly on the pair's `seam_at` (whole minutes). Silence fills the rest of each slot.
Tracks, analysis and stretches are cached, so a regeneration after a mixer change costs
render time only.

    python -m scripts.regression_set                 # manifest tests/regression/manifest.json → data/external/regression_set/
    python -m scripts.regression_set --manifest M --out-dir D --only p03,p07

Outputs (in --out-dir): regression_set.flac, README.md, results.json. `results.json` holds every
number the render reported per pair (shift, band, drift, gains, bars, type, sections). Anas's
verdicts go into the manifest's `approved` block by hand; tests/test_regression_set.py compares a
fresh render's numbers with them.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
from src.audio.audio_mixer import SR, render_mix  # noqa: E402

log = logging.getLogger("regression_set")
LEAD_S = 10.0  # outgoing track alone before the seam
TAIL_S = 20.0  # incoming track alone after the overlap
GAP_S = 2.0  # silence between a slot's audio and the next slot


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def hms(s: float) -> str:
    return mmss(s)


def render_pair(pair: dict, work: Path) -> tuple[dict, np.ndarray, float, float]:
    """Render A→B in full; return (transition report, audio, overlap start s, overlap end s)."""
    a, b = pair["A"], pair["B"]
    tmp = work / f"_full_{pair['id']}.flac"
    cues = [
        tuple(a["cue"]) if a.get("cue") else None,
        tuple(b["cue"]) if b.get("cue") else None,
    ]
    rep = render_mix(
        [a["path"], b["path"]],
        tmp,
        target_bpm=pair.get("target_bpm"),
        genre=pair.get("genre"),
        curve=pair.get("curve"),
        energy_targets=pair.get("energy_targets"),
        force_type=pair.get("force_type"),
        force_bars=pair.get("force_bars"),
        cue_overrides=cues,
    )
    tr = rep["transitions"][0]
    at, end = float(tr["at_s"]), float(tr["end_s"])
    y, _ = sf.read(tmp, dtype="float32")
    tmp.unlink()
    tr = dict(tr)
    tr["target_bpm"] = rep["target_bpm"]
    tr["track_gain_db"] = [t.get("gain_db") for t in rep.get("tracks", [])]
    return tr, y, float(at), float(end)


def build(manifest_path: Path, out_dir: Path, only: set[str] | None) -> None:
    man = json.loads(manifest_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    pairs = man["pairs"]
    total_s = max(p["seam_at"] for p in pairs) + 240.0
    mix = np.zeros((int(total_s * SR), 2), dtype=np.float32)
    for pair in pairs:
        if only and pair["id"] not in only:
            continue
        t0 = time.time()
        try:
            tr, y, at, end = render_pair(pair, out_dir)
        except Exception as e:  # noqa: BLE001
            log.error("%s failed: %s", pair["id"], e)
            traceback.print_exc()
            results[pair["id"]] = {"error": str(e)}
            continue
        c0 = max(int((at - LEAD_S) * SR), 0)
        c1 = min(int((end + TAIL_S) * SR), len(y))
        clip = y[c0:c1]
        if clip.ndim == 1:
            clip = np.stack([clip, clip], axis=1)
        start = int((pair["seam_at"] - (at - c0 / SR)) * SR)
        if start < 0:
            raise ValueError(f"{pair['id']}: lead-in longer than the slot allows")
        n = min(len(clip), len(mix) - start)
        mix[start : start + n] = clip[:n]
        results[pair["id"]] = {
            **{k: tr.get(k) for k in tr if k not in ("measured",)},
            "measured_type": tr.get("measured"),
            "slot": {
                "clip_start": mmss(start / SR),
                "seam": mmss(pair["seam_at"]),
                "lead_swap": mmss(pair["seam_at"] + _sec(tr.get("lead_swap")) - at)
                if tr.get("lead_swap")
                else None,
                "fully_in": mmss(pair["seam_at"] + (end - at)),
                "clip_end": mmss((start + n) / SR),
            },
            "render_s": round(time.time() - t0),
        }
        results_path.write_text(json.dumps(results, indent=1, default=str))
        log.info(
            "%s done: %s %d bars, seam %s, shift %+.0f ms (%s), gains %s, %ds",
            pair["id"],
            tr["type"],
            tr["bars"],
            mmss(pair["seam_at"]),
            tr["seam_offset_before_ms"],
            tr["seam_band"],
            tr["overlap_gain_db"],
            results[pair["id"]]["render_s"],
        )
    # keep audio of pairs not re-rendered this run
    prev = out_dir / "regression_set.flac"
    if only and prev.exists():
        old, _ = sf.read(prev, dtype="float32")
        for pair in pairs:
            if (
                pair["id"] in only
                or pair["id"] not in results
                or "slot" not in results[pair["id"]]
            ):
                continue
            s0 = int(_sec(results[pair["id"]]["slot"]["clip_start"]) * SR)
            s1 = int(_sec(results[pair["id"]]["slot"]["clip_end"]) * SR)
            s1 = min(s1, len(old), len(mix))
            mix[s0:s1] = old[s0:s1]
    sf.write(prev, mix, SR)
    write_readme(man, results, out_dir)
    log.info("wrote %s (%.1f min) and README.md", prev, len(mix) / SR / 60)


def _sec(mmss_str: str) -> float:
    m, s = mmss_str.split(":")
    return int(m) * 60 + int(s)


def write_readme(man: dict, results: dict, out_dir: Path) -> None:
    lines = [
        f"# {man.get('title', 'Seam regression set')}",
        "",
        man.get("description", ""),
        "",
        f"Each slot: {LEAD_S:.0f} s of A alone, the overlap starting exactly at `seam`, then {TAIL_S:.0f} s of B alone.",
        "Times are positions in `regression_set.flac`. `shift` is the seam correction the mixer applied,",
        "`gains` the overlap gain ride in/out (dB). Verdict column is for Anas.",
        "",
        "| # | seam | fully in | lead swap | edge case | genre | A → B | type | bars | shift ms (band) | drift ms | gains dB | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in man["pairs"]:
        r = results.get(p["id"], {})
        if "slot" not in r:
            lines.append(
                f"| {p['id']} | {mmss(p['seam_at'])} | | | {p['edge_case']} | {p['genre']} | {p['A']['name']} → {p['B']['name']} | FAILED {r.get('error', '')} | | | | | |"
            )
            continue
        verdict = (p.get("approved") or {}).get("verdict", "")
        lines.append(
            f"| {p['id']} | {r['slot']['seam']} | {r['slot']['fully_in']} | {r['slot']['lead_swap'] or ''} | {p['edge_case']} | {p['genre']} | "
            f"{p['A']['name']} → {p['B']['name']} | {r['type']} | {r['bars']} | {r['seam_offset_before_ms']:+.0f} ({r['seam_band']}) | "
            f"{r['seam_drift_ms']} | {r['overlap_gain_db']} | {verdict} |"
        )
    lines += ["", "## Why each pair is in", ""]
    for p in man["pairs"]:
        lines.append(f"- **{p['id']}** {p['A']['name']} → {p['B']['name']}: {p['reason']}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="tests/regression/manifest.json")
    ap.add_argument("--out-dir", default="data/external/regression_set")
    ap.add_argument("--only", default=None, help="comma-separated pair ids to re-render")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("numba", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    build(
        Path(args.manifest), Path(args.out_dir), set(args.only.split(",")) if args.only else None
    )


if __name__ == "__main__":
    main()
