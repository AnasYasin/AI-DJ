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
import re

import numpy as np
import pandas as pd
import torch

from src.data.preprocess_tracklist import is_unidentified
from src.features.compatibility import compatibility
from src.models.edge_scorer import EdgeScorer, TrackArrays, camelot_distance, pair_features
from src.models.key_rules import clash_allowed, is_clash, key_term, sequence_allowed
from src.models.train_sequence import TOKEN_EXTRA, SequenceModel

log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")
TRACKLIST_PATH = Path("data/processed/tracklist_clean.csv")
SEQ_MODEL_PATH = Path("models/sequence_model.pt")

BEAM_WIDTH = 8
MAX_BPM_LOG_RATIO = 0.05  # ≈ ±5% — pyrubberband stretches this cleanly
W_CTX, W_GBM, W_ENERGY = 1.0, 1.0, 0.7

# Weight on pair compatibility, which is what decides how LONG the mixer is
# allowed to hold two records together. The default is 0: without it the planner
# optimises for what fits next, not for what can be blended slowly. Raise it to
# plan a set built for long overlaps.
W_COMPAT_DEFAULT = 0.0

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


# ── Artists ────────────────────────────────────────────────────────────────────

# Credit strings join collaborators in a dozen ways. Splitting on all of them is
# what makes "one track per artist" mean one track per *person*.
# Symbols may sit tight against the names; words must be surrounded by spaces,
# and take any trailing period with them ("pres." must not leave a "." behind).
_ARTIST_SPLIT = re.compile(
    r"\s*[&+/,]\s*"
    r"|\s+(?:x|vs\.?|with|and|feat\.?|ft\.?|pres\.?|presents|meets)\s+",
    re.I,
)


def split_artists(credit: str) -> frozenset[str]:
    """ "A & B pres. C" → {"a", "b", "c"}. Empty credits yield an empty set."""
    parts = (p.strip().lower() for p in _ARTIST_SPLIT.split(str(credit)))
    return frozenset(p for p in parts if p)


# ── Planner ────────────────────────────────────────────────────────────────────


def _load_pool(
    genre: str,
    bpm_range: tuple[float, float],
    n_tracks: int,
    track_ids: list[str] | None,
    exclude_ids: set[str] | None,
    min_energy_pct: float | None,
) -> dict:
    """Candidate pool + per-track artist sets, shared by plan_mix and repair_candidates."""
    f = pd.read_parquet(FEATURES_PATH).drop_duplicates("track_id")
    meta = (
        pd.read_csv(TRACKLIST_PATH, usecols=["track_id", "genre", "artist_name", "track_name"])
        .drop_duplicates("track_id")
        .set_index("track_id")
    )
    # "Massano – ID" is a placeholder for an unreleased track. It has no
    # searchable name, so it can never be fetched or played, and it must not
    # reach a plan.
    named = ~meta.apply(lambda r: is_unidentified(r["artist_name"], r["track_name"]), axis=1)
    n_before = len(f)
    f = f[f["track_id"].map(named).fillna(False)]
    if n_before != len(f):
        log.info("dropped %d unidentified (ID/Unknown) tracks from the pool", n_before - len(f))

    if min_energy_pct is not None:
        floor = float(np.percentile(f["energy_mean"], min_energy_pct))
        n_before = len(f)
        f = f[f["energy_mean"] >= floor]
        log.info(
            "energy floor p%.0f = %.3f: %d → %d tracks", min_energy_pct, floor, n_before, len(f)
        )

    if exclude_ids:
        n_before = len(f)
        f = f[~f["track_id"].isin(exclude_ids)]
        log.info("excluded %d previously unfetchable tracks", n_before - len(f))

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

    # One track per artist, counting every artist on a collaboration. Comparing
    # the credit string as a whole let "Reinier Zonneveld & Miro" and "Reinier
    # Zonneveld" both into the same three-track set.
    artist_sets = [split_artists(a) for a in A.df["track_id"].map(meta["artist_name"]).fillna("")]
    artist_tracks: dict[str, list[int]] = {}
    for i, names in enumerate(artist_sets):
        for name in names:
            artist_tracks.setdefault(name, []).append(i)
    log.info("pool: %d %s tracks in %.0f–%.0f BPM", len(f), genre, *bpm_range)
    return {
        "f": f,
        "meta": meta,
        "A": A,
        "artist_sets": artist_sets,
        "artist_tracks": artist_tracks,
    }


def plan_mix(
    genre: str,
    bpm_range: tuple[float, float],
    n_tracks: int = 10,
    curve: str = "build",
    seed_track: str | None = None,
    track_ids: list[str] | None = None,
    exclude_ids: set[str] | None = None,
    min_energy_pct: float | None = None,
    compat_weight: float = W_COMPAT_DEFAULT,
) -> dict:
    """
    Plan a set.

    `exclude_ids` drops tracks a previous attempt could not fetch, so the caller
    can replan around them instead of rendering a short set.

    `min_energy_pct` (0-100) drops the quietest tail of the pool before planning.
    The energy curve is mapped onto the pool's own quantiles, so it shapes the
    set relative to whatever is in the pool; raising the floor is what makes a
    set loud in absolute terms.

    `compat_weight` adds pair compatibility to the beam score. The mixer caps
    overlap length by that same measure, so a set planned without it will be
    rendered with short transitions however smooth the ordering looks.
    """
    pool = _load_pool(genre, bpm_range, n_tracks, track_ids, exclude_ids, min_energy_pct)
    meta, A, artist_sets, artist_tracks = (
        pool["meta"],
        pool["A"],
        pool["artist_sets"],
        pool["artist_tracks"],
    )

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
            mask = np.abs(np.log(A.bpm[last] / A.bpm)) <= MAX_BPM_LOG_RATIO
            cam = camelot_distance(A.cam_pos[last], A.cam_mode[last], A.cam_pos, A.cam_mode)
            if not clash_allowed(_join_distances(A, tracks), n_tracks - 1):
                mask &= ~is_clash(cam)
            mask[tracks] = False
            for t in tracks:  # one track per artist, collaborations included
                for name in artist_sets[t]:
                    mask[artist_tracks[name]] = False
            cand = np.flatnonzero(mask)
            if len(cand) == 0:
                continue
            ctx_sim = A.proj[cand] @ ctx[bi]
            gbm = scorer.score(pair_features(A, last, cand))
            e_fit = 1.0 - np.abs(A.energy[cand] - targets[step]) / (2 * e_scale)
            z = (ctx_sim - ctx_sim.mean()) / (ctx_sim.std() + 1e-9)
            total = W_CTX * z + W_GBM * gbm + W_ENERGY * e_fit + key_term(genre, cam[cand])
            if compat_weight:
                total = total + compat_weight * compatibility(
                    0.0,  # key is the planner's soft term above, not a compatibility veto
                    np.abs(A.energy[last] - A.energy[cand]),
                    np.abs(A.loud[last] - A.loud[cand]),
                    np.abs(np.log(A.bpm[last] / A.bpm[cand])) * 100,
                )
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
    out = {
        "genre": genre,
        "curve": curve,
        "score": round(score, 2),
        "n_clashes": int(is_clash(_join_distances(A, tracks)).sum()),
        "tracks": [],
    }
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
            # The curve value itself, 0-1, before it is mapped onto the pool's
            # energy range. This is what the mixer needs to pick which window of
            # the track to play; `target_energy` above is in absolute energy
            # units and is for display only. Passing that one to the mixer
            # silently flattens the curve.
            "energy_target01": round(float(shape[i]), 3),
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


def _join_distances(A: TrackArrays, tracks: list[int]) -> list[float]:
    """Camelot distance of every join in a track sequence."""
    return [
        float(
            camelot_distance(
                A.cam_pos[a], A.cam_mode[a], np.array([A.cam_pos[b]]), np.array([A.cam_mode[b]])
            )[0]
        )
        for a, b in zip(tracks, tracks[1:])
    ]


REPAIR_TRIES = 5  # candidates tried per empty slot before a full replan


def _plan_entry(A: TrackArrays, meta, ti: int, n: int, target_energy: float, e01: float) -> dict:
    tid = A.tid[ti]
    m = meta.loc[tid]
    return {
        "n": n,
        "track_id": tid,
        "artist": m["artist_name"],
        "title": m["track_name"],
        "bpm": round(float(A.bpm[ti]), 1),
        "key": str(A.df["key"].iloc[ti]),
        "energy": round(float(A.energy[ti]), 3),
        "target_energy": round(float(target_energy), 3),
        "energy_target01": round(float(e01), 3),
    }


def _link(A: TrackArrays, prev_ti: int, ti: int) -> dict:
    return {
        "d_bpm": round(float(A.bpm[ti] - A.bpm[prev_ti]), 1),
        "cam_dist": float(
            camelot_distance(
                A.cam_pos[prev_ti],
                A.cam_mode[prev_ti],
                np.array([A.cam_pos[ti]]),
                np.array([A.cam_mode[ti]]),
            )[0]
        ),
    }


def repair_candidates(
    plan: dict,
    slot: int,
    bpm_range: tuple[float, float],
    exclude_ids: set[str],
    n: int = REPAIR_TRIES,
    compat_weight: float = W_COMPAT_DEFAULT,
    min_energy_pct: float | None = None,
    track_ids: list[str] | None = None,
) -> list[dict]:
    """
    Replacements for one slot of a plan, best first, fitted to the slot's
    verified neighbours. Every other track in the plan stays where it is.

    A candidate must pass the hard rules against BOTH neighbours (BPM within
    MAX_BPM_LOG_RATIO) and must not repeat a
    track or an artist already in the plan. It is scored the way plan_mix
    scores a step: Model B context from the tracks before the slot, the GBM
    edge from the previous track, energy fit to the slot's target, and pair
    compatibility with each neighbour. At the first or last slot there is one
    neighbour and the fit is against that one plus the energy target.
    """
    tracks = plan["tracks"]
    others = [t for i, t in enumerate(tracks) if i != slot]
    pool = _load_pool(
        plan["genre"], bpm_range, len(tracks), track_ids, exclude_ids, min_energy_pct
    )
    A, meta, artist_tracks = pool["A"], pool["meta"], pool["artist_tracks"]

    mask = np.ones(len(A.tid), dtype=bool)
    for t in others:
        if t["track_id"] in A.idx:
            mask[A.idx[t["track_id"]]] = False
        for name in split_artists(t["artist"]):
            for j in artist_tracks.get(name, []):
                mask[j] = False
    neighbours = []
    for k in (slot - 1, slot + 1):
        if 0 <= k < len(tracks) and tracks[k]["track_id"] in A.idx:
            neighbours.append((k, A.idx[tracks[k]["track_id"]]))
    for _, ni in neighbours:
        mask &= np.abs(np.log(A.bpm[ni] / A.bpm)) <= MAX_BPM_LOG_RATIO
    cand = np.flatnonzero(mask)
    # the repaired sequence must still respect the clash cap and never clash twice in a row
    fixed = [A.idx.get(t["track_id"]) for t in tracks]
    keep = []
    for ci in cand:
        seq = [ci if k == slot else ti for k, ti in enumerate(fixed)]
        known = [ti for ti in seq if ti is not None]
        keep.append(sequence_allowed(_join_distances(A, known), len(tracks) - 1))
    cand = cand[np.array(keep, dtype=bool)] if len(cand) else cand
    if len(cand) == 0:
        return []

    target = float(tracks[slot]["target_energy"])
    e_scale = A.energy.std() + 1e-9
    total = W_ENERGY * (1.0 - np.abs(A.energy[cand] - target) / (2 * e_scale))

    before = [A.idx[t["track_id"]] for t in tracks[:slot] if t["track_id"] in A.idx]
    prev_ti = (
        A.idx[tracks[slot - 1]["track_id"]]
        if slot > 0 and tracks[slot - 1]["track_id"] in A.idx
        else None
    )
    if before:
        ctx = ContextModel().next_vectors([make_tokens(A, before, len(tracks))], plan["genre"])[0]
        ctx_sim = A.proj[cand] @ ctx
        total = total + W_CTX * (ctx_sim - ctx_sim.mean()) / (ctx_sim.std() + 1e-9)
    if prev_ti is not None:
        total = total + W_GBM * EdgeScorer.load().score(pair_features(A, prev_ti, cand))
    for _, ni in neighbours:
        cam = camelot_distance(A.cam_pos[ni], A.cam_mode[ni], A.cam_pos[cand], A.cam_mode[cand])
        total = total + key_term(plan["genre"], cam) / len(neighbours)
        c = compatibility(
            0.0,  # key is the soft term above
            np.abs(A.energy[ni] - A.energy[cand]),
            np.abs(A.loud[ni] - A.loud[cand]),
            np.abs(np.log(A.bpm[ni] / A.bpm[cand])) * 100,
        )
        total = total + max(compat_weight, 1.0) * c / len(neighbours)

    order = np.argsort(-total)[:n]
    out = []
    for ci in cand[order]:
        entry = _plan_entry(A, meta, int(ci), slot + 1, target, tracks[slot]["energy_target01"])
        if prev_ti is not None:
            entry.update(_link(A, prev_ti, int(ci)))
        if slot + 1 < len(tracks) and tracks[slot + 1]["track_id"] in A.idx:
            entry["next_link"] = _link(A, int(ci), A.idx[tracks[slot + 1]["track_id"]])
        out.append(entry)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Plan a DJ set from intent.")
    p.add_argument("--genre", required=True)
    p.add_argument("--bpm", nargs=2, type=float, required=True, metavar=("LO", "HI"))
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--curve", choices=list(CURVES), default="build")
    p.add_argument("--seed-track", default=None)
    p.add_argument(
        "--min-energy-pct", type=float, default=None, help="drop the quietest N%% of the pool"
    )
    p.add_argument(
        "--compat-weight",
        type=float,
        default=W_COMPAT_DEFAULT,
        help="weight on pair compatibility — raise it to plan for long overlaps",
    )
    p.add_argument("--json", default=None, help="write full plan to this path")
    args = p.parse_args()

    plan = plan_mix(
        args.genre,
        tuple(args.bpm),
        args.n,
        args.curve,
        args.seed_track,
        min_energy_pct=args.min_energy_pct,
        compat_weight=args.compat_weight,
    )
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
