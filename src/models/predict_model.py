"""
Phase 8 — mix planner: intent → ordered tracklist.

Pipeline per step of the beam search:
  1. Candidate pool: genre + BPM-range filter over the dataset's tracks (data/dataset/<date>/).
  2. Hard rules mask: |log BPM ratio| ≤ 0.05 vs current track, track and artist
     not already used. No key rule: the pair model sees the Camelot distance.
  3. Score = w_pair·z(pair model logit: this record after the current one, given the set so far)
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

from src.audio.key_shift import CLASH_DISTANCE
from src.data.dataset import dj_key, load_plays, load_tracks
from src.data.preprocess_tracklist import is_unidentified
from src.features.compatibility import compatibility
from src.models.pair_model import PairScorer
from src.models.track_arrays import TrackArrays, camelot_distance

log = logging.getLogger(__name__)

BEAM_WIDTH = 8
MAX_BPM_LOG_RATIO = 0.05  # ≈ ±5% — pyrubberband stretches this cleanly
W_PAIR, W_ENERGY = 1.0, 0.7

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
    audio: str | None = None,
    dj: str | None = None,
) -> dict:
    """Candidate pool + per-track artist sets, shared by plan_mix and repair_candidates.

    `audio="full"` keeps only records whose full audio is held (no search, no wrong recording at
    fetch). `dj` keeps only records that DJ has played, the first piece of a profile. The energy
    floor is a percentile of the pool after the genre and BPM filters, so "drop the quietest 65 %"
    means the quietest of this pool.
    """
    f = load_tracks().dropna(subset=["bpm"]).reset_index(drop=True)
    if audio is not None:
        f = f[f["audio"] == audio]
    if dj is not None:
        plays = load_plays(columns=["dj_key", "track_id"])
        mine = set(plays.loc[plays["dj_key"] == dj_key(dj), "track_id"])
        f = f[f["track_id"].isin(mine)]
        log.info("dj %s: %d records with features", dj, len(f))
    meta = f.set_index("track_id")[["genre", "artist", "title", "source", "source_ref", "audio"]]
    # "Massano – ID" is a placeholder for an unreleased track. It has no
    # searchable name, so it can never be fetched or played, and it must not
    # reach a plan.
    named = ~meta.apply(lambda r: is_unidentified(r["artist"], r["title"]), axis=1)
    n_before = len(f)
    f = f[f["track_id"].map(named).fillna(False)]
    if n_before != len(f):
        log.info("dropped %d unidentified (ID/Unknown) tracks from the pool", n_before - len(f))

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
    if min_energy_pct is not None and len(f):
        floor = float(np.percentile(f["energy_mean"], min_energy_pct))
        n_before = len(f)
        f = f[f["energy_mean"] >= floor]
        log.info(
            "energy floor p%.0f = %.3f: %d → %d tracks", min_energy_pct, floor, n_before, len(f)
        )
    if len(f) < n_tracks:
        raise ValueError(f"pool has only {len(f)} tracks")
    A = TrackArrays(f)

    # One track per artist, counting every artist on a collaboration. Comparing
    # the credit string as a whole let "Reinier Zonneveld & Miro" and "Reinier
    # Zonneveld" both into the same three-track set.
    artist_sets = [split_artists(a) for a in A.df["track_id"].map(meta["artist"]).fillna("")]
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
    audio: str | None = None,
    dj: str | None = None,
) -> dict:
    """
    Plan a set.

    `audio="full"` plans only from records whose full audio is held; `dj` only from records that
    DJ has played (see `_load_pool`).

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
    pool = _load_pool(
        genre, bpm_range, n_tracks, track_ids, exclude_ids, min_energy_pct, audio, dj
    )
    meta, A, artist_sets, artist_tracks = (
        pool["meta"],
        pool["A"],
        pool["artist_sets"],
        pool["artist_tracks"],
    )

    scorer = PairScorer(A)

    # energy targets from the pool's own distribution
    shape = CURVES[curve](np.arange(n_tracks) / max(n_tracks - 1, 1))
    e_lo, e_hi = np.quantile(A.energy, 0.15), np.quantile(A.energy, 0.85)
    targets = e_lo + shape * (e_hi - e_lo)
    e_scale = A.energy.std() + 1e-9

    # seed beams: from the slower half of the pool, the tracks nearest the opening energy target.
    # Real sets open in the slower half of their own BPM range 71 % of the time (chance 50 %) and in
    # the lower energy half 59 %; the tempo then climbs 1 % over the set (trance 4 %, techno 2 %).
    if seed_track is not None:
        seeds = [A.idx[seed_track]]
    else:
        slow = np.flatnonzero(A.bpm <= np.median(A.bpm))
        seeds = list(slow[np.argsort(np.abs(A.energy[slow] - targets[0]))[:BEAM_WIDTH]])
    beams = [([int(s)], 0.0) for s in seeds]

    for step in range(1, n_tracks):
        expansions = []
        for tracks, score in beams:
            last = tracks[-1]
            mask = np.abs(np.log(A.bpm[last] / A.bpm)) <= MAX_BPM_LOG_RATIO
            mask[tracks] = False
            for t in tracks:  # one track per artist, collaborations included
                for name in artist_sets[t]:
                    mask[artist_tracks[name]] = False
            cand = np.flatnonzero(mask)
            if len(cand) == 0:
                continue
            e_fit = 1.0 - np.abs(A.energy[cand] - targets[step]) / (2 * e_scale)
            total = W_PAIR * _z(scorer.score(last, tracks, cand)) + W_ENERGY * e_fit
            if compat_weight:
                total = total + compat_weight * compatibility(
                    0.0,  # key is the pair model's to weigh, not a compatibility veto
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
        for tracks, score in expansions:  # dedupe: same records AND same last record
            key = (frozenset(tracks), tracks[-1])  # same records in another order are another beam
            if key not in seen:
                seen.add(key)
                beams.append((tracks, score))
            if len(beams) >= BEAM_WIDTH:
                break

    tracks, score = beams[0]
    out = {
        "genre": genre,
        "curve": curve,
        "audio": audio,
        "dj": dj,
        "score": round(score, 2),
        "n_clashes": int((np.array(_join_distances(A, tracks)) > CLASH_DISTANCE).sum()),
        "tracks": [],
    }
    for i, ti in enumerate(tracks):
        tid = A.tid[ti]
        m = meta.loc[tid]
        entry = {
            "n": i + 1,
            "track_id": tid,
            "artist": m["artist"],
            "title": m["title"],
            "source": m["source"],
            "source_ref": m["source_ref"],
            "audio": m["audio"],
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


def _z(x: np.ndarray) -> np.ndarray:
    """Standardised within the candidate set, so the weights read against the other terms."""
    return (x - x.mean()) / (x.std() + 1e-9)


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
        "artist": m["artist"],
        "title": m["title"],
        "source": m["source"],
        "source_ref": m["source_ref"],
        "audio": m["audio"],
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
    verified neighbours, from the same pool the plan came from (its `audio` and `dj`).
    verified neighbours. Every other track in the plan stays where it is.

    A candidate must pass the hard rules against BOTH neighbours (BPM within
    MAX_BPM_LOG_RATIO) and must not repeat a
    track or an artist already in the plan. It is scored the way plan_mix
    scores a step: the pair model from the previous track given the tracks
    before the slot (at the first slot, the next track scored after the
    candidate), energy fit to the slot's target, and pair compatibility with
    each neighbour. At the first or last slot there is one neighbour.
    """
    tracks = plan["tracks"]
    others = [t for i, t in enumerate(tracks) if i != slot]
    pool = _load_pool(
        plan["genre"],
        bpm_range,
        len(tracks),
        track_ids,
        exclude_ids,
        min_energy_pct,
        plan.get("audio"),
        plan.get("dj"),
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
    next_ti = (
        A.idx[tracks[slot + 1]["track_id"]]
        if slot + 1 < len(tracks) and tracks[slot + 1]["track_id"] in A.idx
        else None
    )
    if prev_ti is not None:
        total = total + W_PAIR * _z(PairScorer(A).score(prev_ti, before, cand))
    elif (
        next_ti is not None
    ):  # an opening slot: the candidate is scored as what the next record follows
        total = total + W_PAIR * _z(PairScorer(A).score_before(next_ti, cand))
    for _, ni in neighbours:
        c = compatibility(
            0.0,  # key is the pair model's to weigh
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
    p.add_argument(
        "--audio", choices=["full"], default=None, help="only records whose full audio is held"
    )
    p.add_argument("--dj", default=None, help="only records this DJ has played")
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
        audio=args.audio,
        dj=args.dj,
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
