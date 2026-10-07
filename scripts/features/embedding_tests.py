"""Phase 3 embedding checks on the Raveform sample (EMBEDDINGS.md). Raw cosine, no training.

    python embedding_tests.py --seams sample_seams.csv --features seam_tracks_features.parquet \
        --patches seam_track_patches --plays plays.csv --previews sample_previews.parquet --out report.json

Spans per track in a seam: whole (mean of all patches), played (patches inside played_from_s..played_to_s
of that mix), edge (A: last EDGE_BARS bars of its played span; B: first EDGE_BARS bars), preview (the
30 s iTunes clip's mean). Patches are one per second of record time (essentia's default hop).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EDGE_BARS = 32
FALLBACK_EDGE_S = 60.0
RNG = np.random.default_rng(0)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def span_mean(patches, t0, t1):
    a, b = max(0, int(t0)), min(len(patches), int(np.ceil(t1)))
    if b - a < 1:
        return None
    return unit(patches[a:b].astype(np.float32).mean(axis=0))


def main(a):
    seams = pd.read_csv(a.seams)
    feats = pd.read_parquet(a.features).set_index("track_id")
    plays = pd.read_csv(a.plays)
    plays = plays[plays.is_control == 0].set_index(["mix_id", "track_id"])
    prev = pd.read_parquet(a.previews).set_index("track_id") if a.previews else None
    bpm = feats["bpm"].to_dict()

    def edge_s(tid):
        b = bpm.get(tid)
        return EDGE_BARS * 4 * 60.0 / b if b and np.isfinite(b) else FALLBACK_EDGE_S

    vec = {"whole": {}, "played": {}, "edge_out": {}, "edge_in": {}, "preview": {}}
    patches_cache = {}

    def patches(tid):
        if tid not in patches_cache:
            patches_cache[tid] = np.load(Path(a.patches) / f"{tid}.npy")
        return patches_cache[tid]

    rows = []
    for s in seams.itertuples():
        ok = True
        for role, tid in (("a", s.track_a), ("b", s.track_b)):
            if tid not in feats.index:
                ok = False
                break
            P = patches(tid)
            vec["whole"][tid] = unit(feats.loc[tid, "embedding"])
            try:
                p = plays.loc[(s.mix_id, tid)]
                f, t = float(p.played_from_s), float(p.played_to_s)
            except KeyError:
                f, t = 0.0, len(P)
            vec["played"][(s.seam_id, tid)] = span_mean(P, f, t)
            e = edge_s(tid)
            if role == "a":
                vec["edge_out"][(s.seam_id, tid)] = span_mean(P, max(f, t - e), t)
            else:
                vec["edge_in"][(s.seam_id, tid)] = span_mean(P, f, min(t, f + e))
            if prev is not None and tid in prev.index:
                vec["preview"][tid] = unit(prev.loc[tid, "embedding"])
        if ok:
            rows.append(s)
    seams = pd.DataFrame(rows)
    genre = seams.set_index("seam_id").genre.fillna("").str.split().str[0].to_dict()

    def pair_vecs(s, span):
        if span == "whole":
            return vec["whole"].get(s.track_a), vec["whole"].get(s.track_b)
        if span == "preview":
            return vec["preview"].get(s.track_a), vec["preview"].get(s.track_b)
        if span == "played":
            return vec["played"].get((s.seam_id, s.track_a)), vec["played"].get(
                (s.seam_id, s.track_b)
            )
        return vec["edge_out"].get((s.seam_id, s.track_a)), vec["edge_in"].get(
            (s.seam_id, s.track_b)
        )

    def negatives(s, same_genre):
        pool = [
            o
            for o in seams.itertuples()
            if o.seam_id != s.seam_id and o.track_b != s.track_b and o.track_b != s.track_a
        ]
        if same_genre:
            pool = [o for o in pool if genre[o.seam_id] == genre[s.seam_id]] or pool
        return pool

    report = {"n_seams": int(len(seams))}
    for span in ("whole", "played", "edge", "preview"):
        pos, neg_same, neg_other = [], [], []
        for s in seams.itertuples():
            va, vb = pair_vecs(s, span)
            if va is None or vb is None:
                continue
            pos.append(float(va @ vb))
            for same, store in ((True, neg_same), (False, neg_other)):
                o = negatives(s, same)[RNG.integers(0, 10**9) % max(1, len(negatives(s, same)))]
                if span == "edge":
                    vo = vec["edge_in"].get((o.seam_id, o.track_b))
                else:
                    _, vo = pair_vecs(o, span)
                if vo is not None:
                    store.append(float(va @ vo))
        if not pos:
            continue
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg_same))]
        y2 = np.r_[np.ones(len(pos)), np.zeros(len(neg_other))]
        report[span] = {
            "n": len(pos),
            "auc_vs_same_genre": round(float(roc_auc_score(y, np.r_[pos, neg_same])), 3),
            "auc_vs_other_genre": round(float(roc_auc_score(y2, np.r_[pos, neg_other])), 3),
            "cos_pair_median": round(float(np.median(pos)), 3),
            "cos_same_genre_random_median": round(float(np.median(neg_same)), 3),
            "cos_other_genre_random_median": round(float(np.median(neg_other)), 3),
        }
    if prev is not None:
        both = [t for t in vec["preview"] if t in vec["whole"]]
        d = [1 - float(vec["preview"][t] @ vec["whole"][t]) for t in both]
        same = [
            1 - float(vec["whole"][t] @ vec["whole"][u])
            for t, u in zip(both, RNG.permutation(both))
            if t != u
        ]
        report["preview_vs_full"] = {
            "n_tracks": len(both),
            "cos_dist_same_track_median": round(float(np.median(d)), 3),
            "cos_dist_same_track_p90": round(float(np.percentile(d, 90)), 3),
            "cos_dist_random_other_track_median": round(float(np.median(same)), 3),
            "share_same_track_farther_than_random_median": round(
                float(np.mean(np.array(d) > np.median(same))), 3
            ),
        }
    print(json.dumps(report, indent=1))
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("seams", "features", "patches", "plays", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--previews", default=None)
    main(ap.parse_args())
