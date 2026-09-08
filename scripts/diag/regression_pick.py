"""Rank scanned tracks per edge case and print candidate pairs with the block to cue on."""

import json
from pathlib import Path
import sys

import pandas as pd

rows, blocks = [], {}
for p in sorted(Path("data/interim").glob("regression_scan*.csv")):
    rows.append(pd.read_csv(p))
for p in sorted(Path("data/interim").glob("regression_scan_blocks*.json")):
    blocks.update(json.loads(p.read_text()))
df = pd.concat(rows).drop_duplicates("track_id").reset_index(drop=True)
print(f"{len(df)} tracks scanned: {df.groupby('genre').size().to_dict()}\n")


def peak_block(tid, key, fn=max, labels=("drop", "groove")):
    bl = [b for b in blocks.get(tid, []) if b["label"] in labels] or blocks.get(tid, [])
    if not bl:
        return None
    b = fn(bl, key=lambda x: x[key])
    return f"bar {b['bar']} ({b['label']}, {key}={b[key]})"


def show(title, sub, cols, n=6, block_key=None, block_fn=max):
    print(f"== {title}")
    for r in sub.head(n).itertuples():
        extra = f" | {peak_block(r.track_id, block_key, block_fn)}" if block_key else ""
        vals = " ".join(f"{c}={getattr(r, c)}" for c in cols)
        print(
            f"  {r.genre:13s} {r.track_id} {str(r.artist)[:22]:22s} – {str(r.title)[:30]:30s} {vals}{extra}"
        )
    print()


which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "offlow"):
    show(
        "syncopated low end (offlow_body high)",
        df.sort_values("offlow_body", ascending=False),
        ["tempo", "offlow_body", "kick_share_body", "downbeat_conf"],
        block_key="offlow",
    )
if which in ("all", "hats"):
    show(
        "off-beat / shuffled hats (hat_off8_body high)",
        df.sort_values("hat_off8_body", ascending=False),
        ["tempo", "hat_off8_body", "kick_share_body"],
        block_key="hat_off8",
    )
if which in ("all", "weak"):
    show(
        "weak kick in body (kick_share_body low)",
        df.sort_values("kick_share_body"),
        ["tempo", "kick_share_body", "kick_share_intro", "downbeat_conf"],
        block_key="kick_share",
        block_fn=min,
    )
if which in ("all", "conf"):
    show(
        "low downbeat confidence",
        df.sort_values("downbeat_conf"),
        ["tempo", "downbeat_conf", "downbeat_source", "kick_share_body"],
    )
if which in ("all", "ghost"):
    show(
        "ghost / doubled kicks (ghost_body high)",
        df.sort_values("ghost_body", ascending=False),
        ["tempo", "ghost_body", "kpb_body"],
        block_key="ghost",
    )
if which in ("all", "half"):
    show(
        "half-time kick (kpb_body low)",
        df.sort_values("kpb_body"),
        ["tempo", "kpb_body", "kick_share_body"],
        block_key="kpb",
        block_fn=min,
    )
if which in ("all", "break"):
    show(
        "most breakdown/buildup bars",
        df.sort_values("n_breakdown_bars", ascending=False),
        ["tempo", "n_breakdown_bars", "sections"],
    )
if which in ("all", "loud"):
    print("== loudness extremes per genre (lufs)")
    for g, sub in df.groupby("genre"):
        lo, hi = sub.loc[sub.lufs.idxmin()], sub.loc[sub.lufs.idxmax()]
        print(
            f"  {g:13s} quiet {lo.track_id} {str(lo.artist)[:20]} {lo.lufs} | loud {hi.track_id} {str(hi.artist)[:20]} {hi.lufs}"
        )
    print()
if which in ("all", "tempo"):
    print("== tempo spread per genre (for a 5-6 % stretch pair)")
    for g, sub in df.groupby("genre"):
        s = sub.sort_values("tempo")
        print(f"  {g:13s} {s.tempo.min():.1f} .. {s.tempo.max():.1f}  ({len(s)} tracks)")
    print()
if which in ("all", "plain"):
    plain = df[(df.downbeat_conf > 0.8) & (df.kick_share_body > df.kick_share_body.median())]
    show(
        "plain controls (clean kick, confident grid, low offlow)",
        plain.sort_values("offlow_body"),
        ["tempo", "offlow_body", "hat_off8_body", "kick_share_body", "downbeat_conf"],
        n=10,
    )
