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
from src.data.dataset import (
    dj_key,
    load_djs,
    load_groups,
    load_plays,
    load_structure,
    load_tracks,
)
from src.data.preprocess_tracklist import is_unidentified
from src.features.compatibility import compatibility
from src.models.pair_model import PairScorer
from src.models.set_plan import join_fit, pick_styles, set_plan
from src.models.track_arrays import TrackArrays, camelot_distance

log = logging.getLogger(__name__)

BEAM_WIDTH = 8
MAX_BPM_LOG_RATIO = 0.05  # ≈ ±5% — pyrubberband stretches this cleanly
STYLE_BPM_SPAN = 0.04  # a styled set's pool stays this far either side of the style's median tempo
SIGNATURE_SETS = (
    3  # a named DJ's track played in this many of their sets is one of their own sound
)
W_PAIR, W_ENERGY = 1.0, 0.7
W_SLOT = 0.7  # weight of "this seam suits this join" (set_plan.join_fit); never tuned
W_PLAYED = 0.7  # weight of "the style's DJs play this track in many sets"; never tuned

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
    dj: str | list[str] | None = None,
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
        keys = {dj_key(d) for d in ([dj] if isinstance(dj, str) else dj)}
        mine = set(plays.loc[plays["dj_key"].isin(keys), "track_id"])
        f = f[f["track_id"].isin(mine)]
        log.info("dj %s: %d records with features", ", ".join(sorted(keys)), len(f))
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
    genre: str | None,
    bpm_range: tuple[float, float] | None,
    n_tracks: int = 10,
    curve: str = "build",
    seed_track: str | None = None,
    track_ids: list[str] | None = None,
    exclude_ids: set[str] | None = None,
    min_energy_pct: float | None = None,
    compat_weight: float = W_COMPAT_DEFAULT,
    audio: str | None = None,
    dj: str | list[str] | None = None,
    group: str | list[str] | None = None,
    seed: int = 0,
    style: list[str] | None = None,
    energy: float | None = None,
) -> dict:
    """
    Plan a set.

    `audio="full"` plans only from records whose full audio is held; `dj` only from records that
    DJ has played (see `_load_pool`).

    `group`, a profiled `dj` and `style` plan in the style of real DJs (`pick_styles`): one or more
    style groups by name, one or more DJs in their own style, or `style` words, where "auto" takes
    every group of the genre by weight. Several styles share the set by weight. `energy` (0 to 1)
    says where in the styles' own energy band the set sits, 1 for a peak-time set. The set then
    gets a budget of seams (`set_plan`), the pool is what the styles' DJs play (the named DJs' own
    records when there are enough), `bpm_range` defaults to the range of the styles' sets and the
    audio has to be held. At every step each candidate takes
    the seam left in the budget that suits its join best (`join_fit`), so a cut lands where the two
    tracks can carry one and the long blends where the tempos are close. The budget is used up by
    the last join. `seams` in the result is in set order. `seed` draws the play times.

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
    as_list = lambda x: [x] if isinstance(x, str) else list(x or [])  # noqa: E731
    styles, seams, pool_dj = [], None, dj
    if group or dj or style is not None:
        genre, styles = pick_styles(genre, as_list(dj), style, as_list(group), bpm_range)
    if styles:
        cards = load_groups().set_index("group").loc[[s["group"] for s in styles]].reset_index()
        # a named DJ is matched against their own tempo, energy and bass-free limits
        own = load_djs().set_index(["group", "dj_key"])
        for i, s in enumerate(styles):
            if (s["group"], s["dj"]) in own.index:
                mine = own.loc[(s["group"], s["dj"])].dropna()
                cards.loc[i, mine.index] = mine.to_numpy()
        weight = np.array([s["weight"] for s in styles])
        audio = "full"  # a slot fit needs the track's segment file
        # The mixer stretches every track to the set's median tempo and refuses more than 8 %, and
        # the planner's own tempo rule only joins neighbours. A style's sets can span more (house
        # cut 122 to 138 BPM), so the pool is the style's range cut to a span around its median.
        mid = float(cards["bpm"] @ weight)
        low = max(float(cards["bpm_low"].min()), mid * (1 - STYLE_BPM_SPAN))
        high = min(float(cards["bpm_high"].max()), mid * (1 + STYLE_BPM_SPAN))
        # half a BPM of room: a track made at 140 measures 139.98 or 140.04
        bpm_range = bpm_range or (round(low) - 0.5, round(high) + 0.5)
        seams = set_plan(styles, n_tracks, seed)
        group_djs = sorted({d for djs in cards["djs"] for d in djs})
        # a named DJ's own records, a group's DJs for a style that names none
        pool_dj = sorted(
            {d for s, djs in zip(styles, cards["djs"]) for d in ([s["dj"]] if s["dj"] else djs)}
        )
        # Tracks emptier than three quarters of what the styles' DJs really play are left out. Heard
        # 2026-11-23 (Anas): "many buildups, they lack kick and main beats". A `techno cut` set had no
        # bass for 33 % of its time, the group's real plays for 11 % (middle half 5 to 22 %).
        structure = load_structure()
        limit = float(cards["no_bass_high"] @ weight)
        exclude_ids = (exclude_ids or set()) | set(
            structure.loc[structure["no_bass"] > limit, "track_id"]
        )
        # A named DJ's signature tracks stay in whatever their bass-free share: the DJ makes them
        # work. Fred again plays his own tracks on 53 % of his plays, and the limits left 20 of his
        # 114 (63 over the bass-free limit, their loudness at 0.02 or 0.98 against a band of 0.42 to
        # 0.77), so a set of his had one of them in six.
        plays = load_plays(columns=["mix_id", "track_id", "dj_key"])
        named = plays[plays["dj_key"].isin([s["dj"] for s in styles if s["dj"]])]
        in_sets = named.groupby(["dj_key", "track_id"])["mix_id"].nunique()
        signature = set(in_sets[in_sets >= SIGNATURE_SETS].index.get_level_values("track_id"))
        exclude_ids -= signature
        one_offs = set(in_sets[in_sets == 1].index.get_level_values("track_id")) - set(
            in_sets[in_sets > 1].index.get_level_values("track_id")
        )
        band = [float(cards["energy_low"] @ weight), float(cards["energy_high"] @ weight)]
        if energy is not None:  # half the band, centred where the request says
            centre, half = band[0] + energy * (band[1] - band[0]), (band[1] - band[0]) / 4
            band = [max(centre - half, 0.0), min(centre + half, 1.0)]
            # An asked energy is a floor on the tracks, not one more score: a peak-time house set
            # scored on it alone got tracks at the genre's 0th to 32nd percentile (2026-11-24).
            every = load_tracks(
                audio="full", columns=["track_id", "genre", "audio", "energy_mean"]
            )
            every = every[every["genre"] == genre]
            too_calm = every["energy_mean"] < every["energy_mean"].quantile(band[0])
            exclude_ids = exclude_ids | set(every.loc[too_calm, "track_id"])
    if bpm_range is None:
        raise ValueError(
            f"no BPM range given and {dj or 'nobody'} is not a profiled DJ in {genre}"
        )
    # A named DJ's pool is what they play again: 182 of Fred again's 234 held tracks were in one set
    # only, and with those in the pool a set of his held one of his own tracks in six. When what is
    # left is too small the one-offs come back, and after that the DJs of the styles' groups.
    tries = [(exclude_ids, pool_dj)]
    if styles:
        tries = [
            (exclude_ids | one_offs, pool_dj),
            (exclude_ids, pool_dj),
            (exclude_ids, group_djs),
        ]
    for k, (leave_out, pool_dj) in enumerate(tries):
        try:
            pool = _load_pool(
                genre, bpm_range, n_tracks, track_ids, leave_out, min_energy_pct, audio, pool_dj
            )
            break
        except ValueError:
            if k == len(tries) - 1:
                raise
            log.info("pool too small, widening it (%d of %d)", k + 1, len(tries) - 1)
    meta, A, artist_sets, artist_tracks = (
        pool["meta"],
        pool["A"],
        pool["artist_sets"],
        pool["artist_tracks"],
    )

    scorer = PairScorer(A)
    structure = structure.set_index("track_id").reindex(A.tid) if seams else None
    # The artist rule does not hold a named DJ back from their own tracks. It let one Fred again
    # track into a Fred again set, where he plays his own on 53 % of his plays: this, not the tempo
    # or the bass-free limit, was why his sets came out as other people's tracks (2026-11-25).
    named_keys = {s["dj"] for s in styles if s["dj"]}
    own_names = {name for names in artist_sets for name in names if dj_key(name) in named_keys}
    played = None
    if styles:
        # How many sets of the style's DJs a track is in. Without it a named DJ's set was built from
        # tracks played once: Fred again gave Darude and Dr. Alban (1 set each) where his own tracks
        # sit in 6 to 11 of his 29 sets, and 182 of his 234 held tracks were played in one set only.
        mine = plays[plays["dj_key"].isin(pool_dj)].groupby("track_id")["mix_id"].nunique()
        played = np.log1p(mine.reindex(A.tid).fillna(0).to_numpy(dtype=float))
        is_signature = np.isin(A.tid, list(signature))

    # energy targets from the pool's own distribution
    shape = CURVES[curve](np.arange(n_tracks) / max(n_tracks - 1, 1))
    e_lo, e_hi = np.quantile(A.energy, 0.15), np.quantile(A.energy, 0.85)
    e01 = shape
    if styles:
        # A group's set moves inside the energy band its own sets sit in (10th to 90th percentile of
        # the group's mixes, on the genre's percentile scale), so it no longer opens on the quietest
        # tracks of the pool. Heard 2026-11-23 (Anas): sets planned on the pool's 15th to 85th
        # percentile "seem slow", their first tracks sat at the genre's 7th to 15th percentile. Real
        # sets do not build: end minus start energy has a median of 0.00 over 3,413 mixes.
        every = load_tracks(audio="full", columns=["genre", "audio", "energy_mean"])
        e01 = band[0] + shape * (band[1] - band[0])
        e_lo, e_hi = np.quantile(
            every.loc[every["genre"] == genre, "energy_mean"], [e01.min(), e01.max()]
        )
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
    beams = [([int(s)], 0.0, ()) for s in seeds]  # tracks, score, the budget seams used so far
    grid = structure["downbeat_confidence"].fillna(1.0).to_numpy() if seams else None

    for step in range(1, n_tracks):
        expansions = []
        for tracks, score, used in beams:
            last = tracks[-1]
            mask = np.abs(np.log(A.bpm[last] / A.bpm)) <= MAX_BPM_LOG_RATIO
            mask[tracks] = False
            for t in tracks:  # one track per artist, collaborations included
                for name in artist_sets[t] - own_names:
                    mask[artist_tracks[name]] = False
            cand = np.flatnonzero(mask)
            if len(cand) == 0:
                continue
            e_fit = 1.0 - np.abs(A.energy[cand] - targets[step]) / (2 * e_scale)
            # In a styled plan every term is standardised within the candidates, so the weights
            # compare. Left raw, the energy fit spans about 0 to 1 against several units of a z-score
            # and lost every time: a techno blend set took tracks at the genre's 9th percentile.
            total = W_PAIR * _z(scorer.score(last, tracks, cand))
            if styles:  # a signature track is never the wrong loudness for its own DJ
                e_fit = np.where(is_signature[cand], e_fit.max(), e_fit)
            total = total + W_ENERGY * (_z(e_fit) if styles else e_fit)
            chosen = np.zeros(len(cand), dtype=int)
            if played is not None:
                total = total + W_PLAYED * _z(played[cand])
            if seams:
                # one seam of each type still in the budget; each candidate takes the one that fits
                left = {seams[k]["type"]: k for k in reversed(range(len(seams))) if k not in used}
                gap = np.abs(np.log(A.bpm[last] / A.bpm[cand])) * 100
                fits = np.stack(
                    [
                        join_fit(structure.iloc[cand], seams[k], gap, grid[last])
                        for k in left.values()
                    ]
                )
                chosen = np.array(list(left.values()))[fits.argmax(axis=0)]
                total = total + W_SLOT * _z(fits.max(axis=0))
            if compat_weight:
                total = total + compat_weight * compatibility(
                    0.0,  # key is the pair model's to weigh, not a compatibility veto
                    np.abs(A.energy[last] - A.energy[cand]),
                    np.abs(A.loud[last] - A.loud[cand]),
                    np.abs(np.log(A.bpm[last] / A.bpm[cand])) * 100,
                )
            order = np.argsort(-total)[:BEAM_WIDTH]
            for c, s, k in zip(cand[order], total[order], chosen[order]):
                expansions.append((tracks + [int(c)], score + float(s), (*used, int(k))))
        if not expansions:
            log.warning("beam search dead-ended at step %d — relax rules or widen pool", step)
            break
        expansions.sort(key=lambda x: -x[1])
        seen, beams = set(), []
        for tracks, score, used in expansions:  # dedupe: same records AND same last record
            key = (frozenset(tracks), tracks[-1])  # same records in another order are another beam
            if key not in seen:
                seen.add(key)
                beams.append((tracks, score, used))
            if len(beams) >= BEAM_WIDTH:
                break

    tracks, score, used = beams[0]
    seams = [seams[k] for k in used] if seams else None  # in set order
    out = {
        "genre": genre,
        "curve": curve,
        "audio": audio,
        "dj": pool_dj,  # the pool, for a slot repair: one DJ, or the group's DJs
        "bpm_range": list(bpm_range),
        "styles": styles,
        "seams": seams,
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
            "energy_target01": round(float(e01[i]), 3),
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
    p.add_argument("--genre", default=None, help="needed unless --group is given")
    p.add_argument("--bpm", nargs=2, type=float, default=None, metavar=("LO", "HI"))
    p.add_argument("--group", nargs="*", default=None, help='style groups, e.g. "techno blend"')
    p.add_argument("--style", nargs="*", default=None, help='"auto", or words: blend cut fast ...')
    p.add_argument("--energy", type=float, default=None, help="0 to 1 inside the styles' own band")
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
    p.add_argument(
        "--dj", nargs="*", default=None, help="DJs: their records, and style if profiled"
    )
    p.add_argument("--json", default=None, help="write full plan to this path")
    args = p.parse_args()

    plan = plan_mix(
        args.genre,
        tuple(args.bpm) if args.bpm else None,
        args.n,
        args.curve,
        args.seed_track,
        min_energy_pct=args.min_energy_pct,
        compat_weight=args.compat_weight,
        audio=args.audio,
        dj=args.dj,
        group=args.group,
        style=args.style,
        energy=args.energy,
    )
    print(f"\n{plan['genre']} | curve={plan['curve']} | beam score {plan['score']}")
    for s in plan["styles"]:
        print(f"  {s['weight']:.0%} {s['group']}" + (f", as {s['dj']}" if s["dj"] else ""))
    if plan["seams"]:
        print("  " + ", ".join(f"{s['type']} ({s['group']})" for s in plan["seams"]))
    for t in plan["tracks"]:
        trans = f"  Δbpm {t['d_bpm']:+.1f} cam {t['cam_dist']:.1f}" if t["n"] > 1 else ""
        print(
            f"  {t['n']:>2}. {t['artist']} – {t['title']}"
            f"  [{t['bpm']:.0f} bpm, {t['key']}, e={t['energy']:.2f}→{t['target_energy']:.2f}]{trans}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(plan, indent=1))
        print(f"\nplan → {args.json}")
