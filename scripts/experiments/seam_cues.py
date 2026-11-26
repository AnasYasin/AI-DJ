"""
Where in each track a measured seam happens: one row per measured seam, per source.

    PYTHONPATH=. python scripts/experiments/seam_cues.py     -> data/djdata/<source>/out/seam_cues.csv

Mix time comes from the seam window reading (`window_t0 + in_s`, `out_s`, `bass_swap_s`), which was
checked on mixer renders on 2026-10-09 (DATASET_STATE.md). Track time = (mix time - time_zero_s) * rate.
Bars, phrase grid and sections come from data/djdata/<source>/segments/<ref>.json, levels from
curves/<ref>.npz (both written by scripts/vm/track_structure.py).

  b_in_*    where the incoming track is first heard       a_out_*   where the outgoing track is last heard
  *_swap_*  where each track is at the bass swap
  _s track seconds, _bar bar index (negative = before the first bar), _phrase_pos bar inside the 8-bar
  phrase, _section the section label there, _edge_bars bars to the nearest section boundary,
  _rms_rel_db / _low_rel_db level over the 8 s after the entry (before the exit, around the swap)
  minus the track's own 90th percentile. a_bars_left = bars of the outgoing track after its exit.
`export-dataset` joins the file onto the seams table by seam_id.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import load_seams, load_tracks

ROOT = Path("data/djdata")
GRID_S = 0.25  # the curves' time step
SPAN_S = 8.0


def position(seg: dict, t: float) -> dict:
    """Bar index, place in the phrase, section and bars to the nearest section boundary at track time t."""
    bars = np.asarray(seg["bars"])
    bar_s = float(np.median(np.diff(bars)))
    # outside the grid (before the first bar, after the last) the bar length extends it
    bar = (
        int(np.floor((t - bars[0]) / bar_s))
        if t < bars[0] or t >= bars[-1] + bar_s
        else int(np.searchsorted(bars, t, side="right") - 1)
    )
    label = next((s["label"] for s in seg["sections"] if s["start"] <= t < s["end"]), "")
    edges = sorted({b for s in seg["sections"] for b in s["bars"]})
    return {
        "bar": bar,
        "phrase_pos": (bar - seg["phrase_offset"]) % seg["phrase_bars"],
        "section": label,
        "edge_bars": min(abs(bar - e) for e in edges),
    }


def level(curve, t0: float, t1: float) -> dict:
    """Mean level between two track times, against the track's own 90th percentile (dB)."""
    i, j = max(int(t0 / GRID_S), 0), int(t1 / GRID_S)
    out = {}
    for name, key in (("rms_rel_db", "rms_db"), ("low_rel_db", "low_db")):
        x = curve[key]
        out[name] = (
            round(float(x[i:j].mean() - np.percentile(x, 90)), 2)
            if j > i and i < len(x)
            else np.nan
        )
    return out


def main():
    tracks = load_tracks().set_index("track_id")[["source", "source_ref"]]
    seams = load_seams()
    seams = seams[seams["measured"] == 1]
    cache = {}

    def track(tid):
        if tid not in cache:
            src, ref = tracks.loc[tid]
            seg, cur = (
                ROOT / src / "segments" / f"{ref}.json",
                ROOT / src / "curves" / f"{ref}.npz",
            )
            cache[tid] = (
                (json.loads(seg.read_text()), dict(np.load(cur)))
                if seg.exists() and cur.exists()
                else None
            )
        return cache[tid]

    rows = []
    for s in seams.itertuples():
        row = {"seam_id": s.seam_id, "source": s.source}
        mix = {
            "in": s.window_t0 + s.in_s,
            "out": s.window_t0 + s.out_s,
            "swap": s.window_t0 + s.bass_swap_s,
        }
        for side, tid, zero, rate, cues in (
            ("a", s.track_a, s.a_time_zero_s, s.a_rate, ("out", "swap")),
            ("b", s.track_b, s.b_time_zero_s, s.b_rate, ("in", "swap")),
        ):
            held = track(tid)
            if held is None:
                continue
            seg, cur = held
            row |= {
                f"{side}_n_bars": seg["n_bars"],
                f"{side}_duration_s": seg["duration"],
                f"{side}_downbeat_source": seg["downbeat_source"],
                f"{side}_downbeat_confidence": round(seg["downbeat_confidence"], 3),
            }
            for cue in cues:
                t = (mix[cue] - zero) * rate
                if not np.isfinite(t):
                    continue
                span = {
                    "in": (t, t + SPAN_S),
                    "out": (t - SPAN_S, t),
                    "swap": (t - SPAN_S / 2, t + SPAN_S / 2),
                }[cue]
                found = {"s": round(t, 2)} | position(seg, t) | level(cur, *span)
                row |= {f"{side}_{cue}_{k}": v for k, v in found.items()}
            if side == "a" and "a_out_bar" in row:
                row["a_bars_left"] = seg["n_bars"] - 1 - row["a_out_bar"]
        rows.append(row)
    out = pd.DataFrame(rows)
    for src, part in out.groupby("source"):
        part.drop(columns="source").to_csv(ROOT / src / "out" / "seam_cues.csv", index=False)
        print(
            src,
            len(part),
            "seams,",
            int(part["b_in_bar"].notna().sum()),
            "with the incoming track held,",
            int(part["a_out_bar"].notna().sum()),
            "with the outgoing track held",
        )


def demo():
    seg = {
        "bars": [1.0, 3.0, 5.0, 7.0], "phrase_offset": 1, "phrase_bars": 8,
        "sections": [{"label": "intro", "start": 1.0, "end": 5.0, "bars": [0, 2]},
                     {"label": "drop", "start": 5.0, "end": 9.0, "bars": [2, 4]}],
    }  # fmt: skip
    assert position(seg, 5.5) == {"bar": 2, "phrase_pos": 1, "section": "drop", "edge_bars": 0}
    assert position(seg, 0.5)["bar"] == -1 and position(seg, 11.5)["bar"] == 5
    cur = {"rms_db": np.r_[np.zeros(40), np.full(40, -6.0)], "low_db": np.zeros(80)}
    assert level(cur, 10, 18)["rms_rel_db"] == -6.0


if __name__ == "__main__":
    demo()
    main()
