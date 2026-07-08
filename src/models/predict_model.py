"""
Phase 8 — mix planner: intent → ordered tracklist.

Pipeline per step of the beam search:
  1. Candidate pool: genre + BPM-range filter over features.parquet.
  2. Hard rules mask: |log BPM ratio| ≤ 0.05 vs current track, Camelot
     distance ≤ 2, track not already used.
  3. Score = w_ctx·z(Model-B context similarity)
           + w_gbm·EdgeScorer probability
           + w_energy·(energy-curve fit at this position)
  4. Beam search (width 8) keeps the best partial sets; best complete set wins.

Energy curves ("build", "peak", "wave", "chill", "arc") are mapped to target
energy values via the pool's own energy quantiles.

Run:
  python -m src.models.predict_model --genre techno --bpm 126 136 --n 10 --curve build
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.edge_scorer import EdgeScorer, TrackArrays, camelot_distance, pair_features
from src.models.train_sequence import TOKEN_EXTRA, SequenceModel

log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")
TRACKLIST_PATH = Path("data/processed/tracklist_clean.csv")
SEQ_MODEL_PATH = Path("models/sequence_model.pt")

BEAM_WIDTH = 8
MAX_BPM_LOG_RATIO = 0.05  # ≈ ±5% — pyrubberband stretches this cleanly
MAX_CAMELOT_DIST = 2.0
W_CTX, W_GBM, W_ENERGY = 1.0, 1.0, 0.7

CURVES = {
    "build": lambda t: t,
    "peak": lambda t: np.where(t <= 0.85, np.minimum(t / 0.7, 1.0), 1 - (t - 0.85) / 0.3),
    "wave": lambda t: 0.5 + 0.5 * np.sin(2 * np.pi * t - np.pi / 2),
    "chill": lambda t: 0.15 + 0.1 * t,
    "arc": lambda t: np.sin(np.pi * t),
}


# ── Model B wrapper ────────────────────────────────────────────────────────────


class ContextModel:
    def __init__(self, path: Path = SEQ_MODEL_PATH):
        ckpt = torch.load(path, weights_only=False)
        self.model = SequenceModel(ckpt["genres"], ckpt["hparams"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.max_len = ckpt["hparams"]["max_len"]

    @torch.no_grad()
    def next_vectors(self, beams_tokens: list[np.ndarray], genre: str) -> np.ndarray:
        """List of (L_i, 132) token matrices → (n_beams, 128) context vectors."""
        L = max(t.shape[0] for t in beams_tokens)
        B = len(beams_tokens)
        tokens = np.zeros((B, L, 128 + TOKEN_EXTRA), dtype=np.float32)
        pad = np.ones((B, L), dtype=bool)
        for i, t in enumerate(beams_tokens):
            tokens[i, : t.shape[0]] = t
            pad[i, : t.shape[0]] = False
        gi = torch.tensor([self.model.genre_to_idx[genre]] * B)
        preds = self.model(torch.from_numpy(tokens), gi, torch.from_numpy(pad))
        last = torch.tensor([t.shape[0] - 1 for t in beams_tokens])
        return preds[torch.arange(B), last].numpy()


def make_tokens(A: TrackArrays, track_idxs: list[int], n_total: int) -> np.ndarray:
    """Token matrix for a partial set (planner-side mirror of seq_tensors)."""
    idx = np.array(track_idxs)
    emb = A.proj[idx]
    b = A.bpm[idx]
    e = A.energy[idx]
    d_bpm = np.r_[0.0, np.diff(b)] / 20.0
    d_energy = np.r_[0.0, np.diff(e)] * 5.0
    pos = np.arange(len(idx), dtype=np.float32) / max(n_total - 1, 1)
    extra = np.stack([pos, b / 200.0, d_bpm, d_energy], axis=1)
    return np.concatenate([emb, extra], axis=1).astype(np.float32)


# ── Planner ────────────────────────────────────────────────────────────────────


def plan_mix(
    genre: str,
    bpm_range: tuple[float, float],
    n_tracks: int = 10,
    curve: str = "build",
    seed_track: str | None = None,
    track_ids: list[str] | None = None,
) -> dict:
    f = pd.read_parquet(FEATURES_PATH).drop_duplicates("track_id")
    meta = (
        pd.read_csv(TRACKLIST_PATH, usecols=["track_id", "genre", "artist_name", "track_name"])
        .drop_duplicates("track_id")
        .set_index("track_id")
    )
    if track_ids is not None:  # explicit pool (e.g. locally available audio)
        f = f[f["track_id"].isin(track_ids)]
    else:
        f = f[f["track_id"].map(meta["genre"]) == genre]
        f = f[(f["bpm"] >= bpm_range[0]) & (f["bpm"] <= bpm_range[1])]
        if len(f) < n_tracks * 5:
            raise ValueError(f"only {len(f)} candidates for {genre} {bpm_range} — widen the range")
    if len(f) < n_tracks:
        raise ValueError(f"pool has only {len(f)} tracks")
    A = TrackArrays(f)
    artist_code = pd.factorize(A.df["track_id"].map(meta["artist_name"]).fillna(""))[0]
    log.info("pool: %d %s tracks in %.0f–%.0f BPM", len(f), genre, *bpm_range)

    scorer = EdgeScorer.load()
    ctx_model = ContextModel()

    # energy targets from the pool's own distribution
    shape = CURVES[curve](np.arange(n_tracks) / max(n_tracks - 1, 1))
    e_lo, e_hi = np.quantile(A.energy, 0.15), np.quantile(A.energy, 0.85)
    targets = e_lo + shape * (e_hi - e_lo)
    e_scale = A.energy.std() + 1e-9

    # seed beams: tracks nearest the opening energy target
    if seed_track is not None:
        seeds = [A.idx[seed_track]]
    else:
        seeds = list(np.argsort(np.abs(A.energy - targets[0]))[:BEAM_WIDTH])
    beams = [([int(s)], 0.0) for s in seeds]

    for step in range(1, n_tracks):
        ctx = ctx_model.next_vectors(
            [make_tokens(A, tracks, n_tracks) for tracks, _ in beams], genre
        )  # (n_beams, 128)
        expansions = []
        for bi, (tracks, score) in enumerate(beams):
            last = tracks[-1]
            mask = (np.abs(np.log(A.bpm[last] / A.bpm)) <= MAX_BPM_LOG_RATIO) & (
                camelot_distance(A.cam_pos[last], A.cam_mode[last], A.cam_pos, A.cam_mode)
                <= MAX_CAMELOT_DIST
            )
            mask[tracks] = False
            mask &= ~np.isin(artist_code, artist_code[tracks])  # one track per artist
            cand = np.flatnonzero(mask)
            if len(cand) == 0:
                continue
            ctx_sim = A.proj[cand] @ ctx[bi]
            gbm = scorer.score(pair_features(A, last, cand))
            e_fit = 1.0 - np.abs(A.energy[cand] - targets[step]) / (2 * e_scale)
            z = (ctx_sim - ctx_sim.mean()) / (ctx_sim.std() + 1e-9)
            total = W_CTX * z + W_GBM * gbm + W_ENERGY * e_fit
            order = np.argsort(-total)[:BEAM_WIDTH]
            for c, s in zip(cand[order], total[order]):
                expansions.append((tracks + [int(c)], score + float(s)))
        if not expansions:
            log.warning("beam search dead-ended at step %d — relax rules or widen pool", step)
            break
        expansions.sort(key=lambda x: -x[1])
        seen, beams = set(), []
        for tracks, score in expansions:  # dedupe identical unordered sets
            key = frozenset(tracks)
            if key not in seen:
                seen.add(key)
                beams.append((tracks, score))
            if len(beams) >= BEAM_WIDTH:
                break

    tracks, score = beams[0]
    out = {"genre": genre, "curve": curve, "score": round(score, 2), "tracks": []}
    for i, ti in enumerate(tracks):
        tid = A.tid[ti]
        m = meta.loc[tid]
        entry = {
            "n": i + 1,
            "track_id": tid,
            "artist": m["artist_name"],
            "title": m["track_name"],
            "bpm": round(float(A.bpm[ti]), 1),
            "key": str(A.df["key"].iloc[ti]),
            "energy": round(float(A.energy[ti]), 3),
            "target_energy": round(float(targets[i]), 3),
        }
        if i > 0:
            prev = tracks[i - 1]
            entry["d_bpm"] = round(float(A.bpm[ti] - A.bpm[prev]), 1)
            entry["cam_dist"] = float(
                camelot_distance(
                    A.cam_pos[prev],
                    A.cam_mode[prev],
                    np.array([A.cam_pos[ti]]),
                    np.array([A.cam_mode[ti]]),
                )[0]
            )
        out["tracks"].append(entry)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Plan a DJ set from intent.")
    p.add_argument("--genre", required=True)
    p.add_argument("--bpm", nargs=2, type=float, required=True, metavar=("LO", "HI"))
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--curve", choices=list(CURVES), default="build")
    p.add_argument("--seed-track", default=None)
    p.add_argument("--json", default=None, help="write full plan to this path")
    args = p.parse_args()

    plan = plan_mix(args.genre, tuple(args.bpm), args.n, args.curve, args.seed_track)
    print(f"\n{plan['genre']} | curve={plan['curve']} | beam score {plan['score']}")
    for t in plan["tracks"]:
        trans = f"  Δbpm {t['d_bpm']:+.1f} cam {t['cam_dist']:.1f}" if t["n"] > 1 else ""
        print(
            f"  {t['n']:>2}. {t['artist']} – {t['title']}"
            f"  [{t['bpm']:.0f} bpm, {t['key']}, e={t['energy']:.2f}→{t['target_energy']:.2f}]{trans}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(plan, indent=1))
        print(f"\nplan → {args.json}")
