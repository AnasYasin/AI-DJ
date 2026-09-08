"""Scan local tracks for the properties that make a seam hard, for picking regression pairs.

Per 8-bar block (source time, on the track's own beat grid):
  kick_share  kick-band onset energy / full-band onset energy
  offlow      share of low (30-1000 Hz) onset energy that is NOT on a beat (±12% of a beat)
  hat_off8    share of high (5-12 kHz) onset energy that is NOT on an 8th (±10% of a beat)
  ghost       extra kick-band peaks per beat beyond one (doubled / ghost kicks)
  kpb         strong kick peaks per bar (4 = four on the floor, ~2 = half-time)
Track summary: medians over drop/groove blocks, minima/maxima, intro/outro kick share, breakdowns,
downbeat confidence, precise tempo, integrated loudness.
"""

import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
logging.disable(logging.WARNING)
from src.audio import audio_mixer as M  # noqa: E402
from src.audio.audio_mixer import SR, _band_envelopes, _level_db, load_audio  # noqa: E402
from src.data.audio_segmenter import segment  # noqa: E402

ROOTS = {
    "tech house": "data/external/test_tracks/tech_house",
    "techno": "data/external/test_tracks/techno",
    "drum and base": "data/external/test_tracks/drum_and_base",
    "afro house": "data/external/test_tracks/afro_house",
    "melodic house": "data/external/test_tracks/melodic_house",
    "trance": "data/external/test_tracks/trance",
}
LISTEN = {  # sets rendered for Anas; genre from the set name
    "techno": ["work_techno_build/techno_build_15min", "work_techno_high/techno_high_15min"],
    "afro house": ["work_afro/afro_20min"],
    "melodic house": ["work_house_low/house_low_15min", "work_minimal_mid/minimal_mid_15min"],
}
OUT_CSV = Path("data/interim/regression_scan.csv")
OUT_JSON = Path("data/interim/regression_scan_blocks.json")
hop = M.ONSET_HOP


def peaks(env, thr_mult, min_gap_s, neigh_s=None):
    """Times of local maxima above thr_mult x median that are the largest within ±neigh_s,
    at least min_gap_s apart (vectorised; the frame loop was 20 s per track)."""
    from scipy.ndimage import maximum_filter1d

    thr = np.median(env) * thr_mult
    size = 2 * (int(neigh_s * SR / hop) if neigh_s else 1) + 1
    is_max = (env == maximum_filter1d(env, size=size, mode="nearest")) & (env > thr)
    idx = np.flatnonzero(is_max)
    times = idx * hop / SR
    keep, last = [], -1e9
    for t in times:
        if t - last >= min_gap_s:
            keep.append(t)
            last = t
    return np.array(keep)


def scan(path: Path, genre: str, meta: dict) -> tuple[dict, list]:
    info = segment(path)
    y = load_audio(path)
    mono = M._to_mono(y)
    env = _band_envelopes(mono)
    beats = np.asarray(info["beats"])
    bars = np.asarray(info["bars"])
    beat = 60.0 / info["bpm"]
    n_frames = len(env["full"])
    t = np.arange(n_frames) * hop / SR
    # phase of every frame inside its beat, from the tracker's beats
    idx = np.clip(np.searchsorted(beats, t) - 1, 0, len(beats) - 2)
    phase = (t - beats[idx]) / np.maximum(beats[idx + 1] - beats[idx], 1e-3)
    on_beat = np.minimum(phase, 1 - phase) < 0.12
    on_8th = np.minimum(np.abs(phase - 0.5), np.minimum(phase, 1 - phase)) < 0.10
    low = env["kick"] + env["lowmid"]
    kick_pk = peaks(env["kick"], 4.0, 0.12)
    strong_pk = peaks(env["kick"], 6.0, 0.35, 0.6 * beat)
    labels = {}
    for s in info["sections"]:
        for b in range(s["bars"][0], s["bars"][1]):
            labels[b] = s["label"]
    blocks = []
    for b0 in range(0, len(bars) - 8, 8):
        t0, t1 = bars[b0], bars[min(b0 + 8, len(bars) - 1)]
        f = (t >= t0) & (t < t1)
        full = env["full"][f].sum() + 1e-9
        kick = env["kick"][f].sum()
        lo = low[f].sum() + 1e-9
        hi = env["high"][f].sum() + 1e-9
        n_beats = max((t1 - t0) / beat, 1)
        blocks.append(
            {
                "bar": int(b0),
                "label": labels.get(b0, "?"),
                "kick_share": round(float(kick / full), 4),
                "offlow": round(float(low[f & ~on_beat].sum() / lo), 3),
                "hat_off8": round(float(env["high"][f & ~on_8th].sum() / hi), 3),
                "ghost": round(
                    float(max(((kick_pk >= t0) & (kick_pk < t1)).sum() / n_beats - 1, 0)), 2
                ),
                "kpb": round(
                    float(((strong_pk >= t0) & (strong_pk < t1)).sum() / (n_beats / 4)), 2
                ),
                "level_db": round(float(M._rms_db(mono[int(t0 * SR) : int(t1 * SR)])), 1),
            }
        )
    body = [b for b in blocks if b["label"] in ("drop", "groove")] or blocks
    breakdowns = [s for s in info["sections"] if s["label"] in ("breakdown", "buildup")]
    summ = {
        "track_id": path.stem,
        "genre": genre,
        "artist": meta.get("artist", ""),
        "title": meta.get("title", ""),
        "path": str(path),
        "bpm": info["bpm"],
        "tempo": round(M._precise_bpm(info, y), 3),
        "duration_min": round(info["duration"] / 60, 2),
        "n_bars": info["n_bars"],
        "downbeat_source": info.get("downbeat_source"),
        "downbeat_conf": round(float(info.get("downbeat_confidence") or 0), 3),
        "lufs": round(_level_db(y), 1),
        "sections": " ".join(f"{s['label'][:5]}{s['bars'][0]}" for s in info["sections"]),
        "n_breakdown_bars": int(sum(s["bars"][1] - s["bars"][0] for s in breakdowns)),
        "kick_share_body": round(float(np.median([b["kick_share"] for b in body])), 3),
        "kick_share_min": round(float(min(b["kick_share"] for b in blocks)), 3),
        "kick_share_intro": blocks[0]["kick_share"] if blocks else None,
        "kick_share_outro": blocks[-1]["kick_share"] if blocks else None,
        "offlow_body": round(float(np.median([b["offlow"] for b in body])), 3),
        "offlow_max": round(float(max(b["offlow"] for b in body)), 3),
        "hat_off8_body": round(float(np.median([b["hat_off8"] for b in body])), 3),
        "ghost_body": round(float(np.median([b["ghost"] for b in body])), 2),
        "kpb_body": round(float(np.median([b["kpb"] for b in body])), 2),
    }
    return summ, blocks


def main():
    shard, n_shards = (int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0/1").split("/"))
    man = pd.read_csv("data/external/test_tracks/manifest.csv").drop_duplicates("track_id")
    meta = {r.track_id: {"artist": r.artist, "title": r.title} for r in man.itertuples()}
    jobs = []
    for genre, root in ROOTS.items():
        for p in sorted(Path(root).glob("*.*")):
            if p.suffix in (".m4a", ".webm", ".mp3", ".opus"):
                jobs.append((p, genre))
    for genre, dirs in LISTEN.items():
        for d in dirs:
            plan = json.loads((Path("data/external/listen") / d / "plan.json").read_text())
            for tr in plan["tracks"]:
                meta.setdefault(tr["track_id"], {"artist": tr["artist"], "title": tr["title"]})
            for p in sorted((Path("data/external/listen") / d / "tracks").glob("*.*")):
                jobs.append((p, genre))
    jobs = [j for k, j in enumerate(jobs) if k % n_shards == shard]
    mine = {j[0].stem for j in jobs}
    out_csv = OUT_CSV if n_shards == 1 else OUT_CSV.with_suffix(f".{shard}.csv")
    out_json = OUT_JSON if n_shards == 1 else OUT_JSON.with_suffix(f".{shard}.json")
    seen, rows, blocks_all = set(), [], {}
    for csv_p, json_p in ((OUT_CSV, OUT_JSON), (out_csv, out_json)):  # resume from earlier runs
        if csv_p.exists():
            for r in pd.read_csv(csv_p).to_dict("records"):
                if r["track_id"] in mine and r["track_id"] not in seen:
                    rows.append(r)
                    seen.add(r["track_id"])
            if json_p.exists():
                blocks_all.update(
                    {k: v for k, v in json.loads(json_p.read_text()).items() if k in mine}
                )
    for n, (p, genre) in enumerate(jobs, 1):
        if p.stem in seen:
            continue
        try:
            summ, blocks = scan(p, genre, meta.get(p.stem, {}))
        except Exception as e:  # noqa: BLE001
            print(f"[{n}/{len(jobs)}] {p.name} FAILED: {e}", flush=True)
            continue
        rows.append(summ)
        blocks_all[p.stem] = blocks
        seen.add(p.stem)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        out_json.write_text(json.dumps(blocks_all))
        print(
            f"[{n}/{len(jobs)}] {genre:13s} {summ['artist'][:22]:22s} – {summ['title'][:28]:28s} "
            f"bpm {summ['tempo']:7.2f} conf {summ['downbeat_conf']:.2f} kick {summ['kick_share_body']:.2f} "
            f"offlow {summ['offlow_body']:.2f} hats {summ['hat_off8_body']:.2f} ghost {summ['ghost_body']:.2f} "
            f"kpb {summ['kpb_body']:.1f} lufs {summ['lufs']:.1f}",
            flush=True,
        )
    print("SCAN DONE", len(rows))


if __name__ == "__main__":
    main()
