# Transition type experiments

What decides the transition type a DJ uses between two records, measured on the seams the pipeline
measured. Every test here, its numbers, how they were obtained, and what it means for the mixer. The
mixer's rules are described in `CLAUDE.md`; the seam measurements in `DATA_PIPELINE.md`.

## The labels

A measured seam (`measured == 1`, 6,731 in `data/dataset/2026-10-14/seams.parquet`) carries two labels.

- `signature`, what the seam reader can tell apart from where the bass swapped and how long the records
  overlapped. This is the label used here. Counts over the seams whose two records both have features:

  | signature | seams | the mixer's types it covers |
  |---|---|---|
  | centre swap | 1,858 | fade, melt, wave |
  | cut | 1,123 | slam |
  | tension | 1,038 | (none; both basses off for a run of bars) |
  | end swap | 688 | rise, drop |
  | late swap | 415 | blend |
  | loop | 63 | (none) |
  | gap | 1,280 | not a transition |

- `type`, the nearest of the mixer's seven recipes by templates measured on one techno pair. It cannot
  separate the members of a signature group (`djdata_package/djdata/seam/types.py`), so it is not used
  as a training label.

Anas, 2026-10-15: the measurements are trusted as they are. The bass swap, loop, tension and the band
entries and exits were checked by ear before.

## Test 1. Whole-record features, 2026-10-15

`scripts/experiments/signature_learnability.py`, output `reports/signature_learnability_2026-10-15.txt`.
5,122 seams with five signatures (cut, centre swap, late swap, end swap, tension). Train on the train
DJs, score on the validation DJs (1,061 seams, never trained on). Inputs: both records' BPM, energy,
loudness, onset, spectral centroid, energy spread, release year, listeners, their differences, Camelot
distance, effnet cosine, genre.

| who chooses | accuracy | macro F1 |
|---|---|---|
| always "centre swap" | 0.41 | |
| the mixer's rules, mapped to signatures | 0.25 | |
| logistic regression, pair features and genre | 0.41 | 0.23 |
| GBM, pair features and genre | 0.37 | 0.23 |
| GBM, pair features and both effnet embeddings | 0.39 | 0.24 |

Within DJs (a random 20 % of each DJ's seams held out, the same DJs on both sides):

| who chooses | accuracy | macro F1 |
|---|---|---|
| always the most common type | 0.36 | |
| the DJ's own most common type | 0.51 | |
| logistic, pair features and the DJ | 0.51 | 0.39 |

Overlap length in bars (non-cut seams, median 29): R² −0.04 from the pair on unseen DJs, 0.23 with the
DJ known.

The rules pick blend on 51 % of seams, where real DJs late-swap on 8 %. They pick a slam on 4 %, where
real DJs cut on 22 %. They never produce tension, which is 20 % of real seams. That is why the ear-test
sets below came out as blend after blend.

## Test 2. The state of both records at the cue points, 2026-10-15

Anas's question: read how record A sounded in the bars before the DJ started leaving it and how record
B sounded where the DJ brought it in, and see whether DJs pick a transition type for that situation.

`scripts/experiments/seam_situation.py`. Run on techno and house only, as a test of the method. The
records are the full files the pipeline measured the seams on, pulled from S3 into the normal track
folders (techno 1,022 files, house 1,124). Per record, loudness, lows (under 150 Hz), mids (150 Hz to
2 kHz), highs (over 2 kHz) and onset on a 0.25 s grid, cached as `data/interim/situation/<ref>.v2.npz`.
Per seam, the cue-out in A and the cue-in in B from the seams table (`a_last_heard_s`, `b_first_heard_s`
in record time through `time_zero_s` and `rate`). 42 numbers per seam, all relative to the record's own
median: each band's level over the 8 bars at the cue, the change from the 8 bars before (a build-up or
a fall), the change over 32 bars, how repetitive the 8 bars are (correlation with the 8 before), mids
against lows and highs, position in the record, bars to or since full level.

Before reading any pattern, a signal check: (1) do the situations vary at all, (2) does the type depend
on the situation cell, by chi-square against a copy with the types shuffled, overall and within each DJ
with 80 or more seams, (3) does it hold on unseen DJs.

**Techno**, 823 seams, `reports/seam_situation_techno_2026-10-15.txt`.

- The situations vary. Lows run 7 dB under to 1.6 dB over the record's median at the cue, highs 6 dB
  under to 2.6 over, the 32-bar build from −3 to +4 dB (10th to 90th percentile).
- The type does not follow them. Every cut of the table gives the overall shares within a few points
  (centre swap 0.36, cut 0.24, tension 0.17, end swap 0.15, late swap 0.08). Chi-square p 0.87 overall,
  0.72 within Adam Beyer (284 seams), 0.69 within Amelie Lens (238), the same as the shuffled copy.
- Unseen DJs (120 seams): situation models 0.33 to 0.38, the majority 0.43. Within DJs, adding the
  situation to the DJ drops accuracy from 0.48 to 0.41.
- Where techno DJs cut: A at 90 % of its length, B at 12 %, both at their normal level. Body into body
  in about 85 % of seams.

**House**, 779 seams, `reports/seam_situation_house_2026-10-15.txt`.

- The situations vary more. B enters more than 3 dB under its median on 31 % of seams (techno 15 %).
- Across DJs the type depends on the situation. Chi-square p < 0.0001, Cramér's V 0.18 against 0.11
  shuffled. The clearest cut is B's entry level:

  | situation | centre swap | cut | end swap | late swap | tension | seams |
  |---|---|---|---|---|---|---|
  | A loud, B loud | 0.46 | 0.17 | 0.15 | 0.12 | 0.10 | 420 |
  | A loud, B quiet | 0.25 | 0.28 | 0.19 | 0.09 | 0.18 | 193 |
  | A quiet, B loud | 0.54 | 0.17 | 0.10 | 0.06 | 0.13 | 117 |
  | A quiet, B quiet | 0.39 | 0.22 | 0.18 | 0.08 | 0.12 | 49 |

  A quiet incoming record gets fewer long centre swaps and more cuts and tension.
- Within Claptone (259 seams) there is no dependence, p 0.48.
- Unseen DJs (73 seams): logistic on the situation 0.47, majority 0.32, top-3 hit 0.74. Within DJs,
  adding the situation does not help (0.37 against 0.39 without it, per-DJ majority 0.43).

**What it means.** In techno the situation adds nothing. In house it carries a signal across DJs, and
none inside the one DJ with enough seams to test, so it may be that DJs who bring records in quiet are
also the DJs who cut more. 73 validation seams are too few to separate the two. Not settled; the
other genres would settle it, and they need their records pulled the same way.

## The design that follows, 2026-10-15

Discussed with Anas and built between 2026-11-18 and 2026-11-25. One command now plans, fetches and mixes
a set in the style of real DJs: `make_mix --dj "fred again"`, `--genre house --style auto --energy 0.9`,
`--genre techno --style blend cut`, `--group "techno blend"`. Without `--dj`, `--style` or `--group` the
legacy system runs, unchanged. Nothing reads a prompt yet. The arguments are the fields a model will fill.

How it runs. `pick_styles` (`src/models/set_plan.py`) turns the request into one or more styles with
weights. `set_plan` gives the set a budget of seams in the styles' shares, each seam whole from one style,
with the style's ranges for overlap and start bar. `plan_mix` picks tracks from what the styles' DJs play
and gives each join the seam of the budget that suits it (`join_fit`). The mixer renders each planned type
with its own recipe (`seam_plan` in `render_mix`), inside the ranges, with its rules as the check.

The legacy mixer is kept apart (Anas, 2026-11-25: design changes go beside the legacy code, by flags, and do
not overwrite it). Planned-set behaviour runs only when a seam has a plan. The legacy lines are marked
`LEGACY` in `src/audio/audio_mixer.py`, the window of a planned track is its own function
(`_choose_planned_window`), and a test fails if the legacy path reaches a planned-set rule. Proof: a legacy
techno set rendered on 2026-11-25 is sample-identical to the one of 2026-11-18 (106,872,186 samples, 0
differ), checked three times that day.

Decisions of 2026-11-18 (Anas):

- The unit is a style group of DJs per genre, with a DJ's own row for a request that names the DJ. 15
  groups exist. In techno and tech house the same two styles appear: long blends with a centre swap
  (`techno blend`, median overlap 38 bars, 0.07 cuts) and cuts with short overlaps (`techno cut`, 6 bars,
  0.57 cuts).
- No seam is copied onto other tracks. A seam was made for its two tracks. The plan for a set holds how
  many transitions of each type, roughly how each is done, and for each place a rough play time and
  cue-in. The tracks are then chosen to fit, with the pair model first.
- A group's share of a type and its recipe weigh each DJ by the square root of their seams. In `tech
  house cut` tension is 0.50 by seams, 0.29 with every DJ equal and 0.38 weighted. Solomun has 429 of
  that group's 661 typed seams.
- A group with a type on 10 % of its seams gets it on about 10 % of the set's seams.
- The mixer renders the planned type and its rules stay as the check. The type mapping is the table
  under "The labels".
- Tension needs a recipe in the mixer. On 1,038 tension seams the overlap is a median 28 bars (middle
  half 14 to 48), both basses are off for a median 12.5 bars (9.5 to 17), and a sweep is used on 14 to
  21 %. At the swap both tracks have their own bass playing on about half of them, so the DJ removed it.
- The group is picked from a DJ name, or from words about the mixing (smooth, quick cuts, fast), or by
  one question to the user. The music asked for picks the group only where it is measured to differ:
  `techno fast` (Amelie Lens) plays sets at 134 to 145 BPM.

Heard by Anas on the rendered sets, what was measured, and what it changed. Each rule is for planned sets.

- 2026-11-23, "sets seem slow". The first tracks sat at the genre's 7th to 15th loudness percentile,
  because the build curve starts at the bottom of the pool. Real sets do not build (end minus start
  energy has a median of 0.00 over 3,413 mixes). A styled set now moves inside the energy band of its
  style's own sets and holds its level unless `--curve` is given (`level` curve).
- 2026-11-23, "many buildups, they lack kick and main beats". A `techno cut` set had no bass for 33 % of
  its time, the group's real plays for 11 %, the legacy set for 11 %. The planner had picked tracks from
  the emptiest quarter of the pool (whole-track share without bass 0.29 against a pool median of 0.15).
  `structure.parquet` now carries `no_bass` per track and the planner leaves out tracks emptier than the
  style's 75th percentile of real plays. The same set measured 7 % after. Two wrong explanations came
  first (a loudness check, then a 48-bar bass check after each entry) and both matched real DJs.
- 2026-11-23, the first cut "felt wrong". The incoming track's `downbeat_confidence` was 0.20. A planned
  cut is vetoed under 0.5 on either track (`CUT_MIN_GRID_CONFIDENCE`, untuned) and the planner keeps such
  tracks out of cut seams.
- 2026-11-25, a Fred again set held one of his own tracks in six. The one-track-per-artist rule blocked
  the rest. A named DJ's own name is now exempt, their pool is the tracks they play in more than one set
  (182 of his 234 held tracks were one-offs), and they are matched against their own card (`djs.parquet`).
  He plays his own tracks on 53 % of his plays. Three fixes were stacked before the artist rule was found.
- 2026-11-25, "lyrics to cut and diff vibe", cuts mostly wrong. Cuts had been placed on the pairs
  furthest apart in tempo. Three cuts in four now go to pairs alike in tempo and loudness
  (`CUT_CONTRAST_SHARE` = 0.25 keeps the old way its room, Anas: "not 100 % every time").
- 2026-11-25, a cut "a bit before" the drop, then "should have been here [2:54] or waited a bit more".
  2:54 was the bar the outgoing track's main drop begins. A planned cut now ends the outgoing track on
  the bar one of its drops begins, the main one when several are in reach, else where a drop ends, and
  still exists when the track offers neither (Anas: "lets not make cut difficult to exist"). The slam
  recipe also ran the outgoing track two bars past its end before cutting. In a planned set the cut now
  lands on the end bar.
- 2026-11-25, long blends over a breakdown and an intro "slow the music too much", unless wanted. A track
  that leaves on a blend ends where its tail keeps a kick for three quarters of the overlap
  (`TAIL_BEAT_SHARE`), unless the set's own curve makes the next slot calm (under `STRUCTURE_LOW`).
- 2026-11-25, the end of a track. It is aimed at the style's bars left, counted from the track's own end,
  and may move 16 bars to a section change. Tracks are heard longer since (techno cut 5 to 6 minutes
  each, a six-track set 31 minutes against 24).
- 2026-11-25, "add one long blend" to the Fred again set, then "both lyrics going on, not good". A named
  DJ's own overlap length is used from 3 seams of a type (his centre swaps overlap 62 to 97 bars, his
  group's 23 to 35). In a planned blend of 32 bars or more the supporting track's mids go down 18 dB and
  swap at the handover (`VOICE_DIP_DB`, untuned). Of his 35 joins with 16 or more bars of overlap, 20 are
  one song against its own remix or second version and 7 more are a related pair, so his long blend is
  mostly a version pair. The planner does not pair versions yet.

Tried and dropped.

- A score penalty for missing the planned start bar (0.03 per bar). It lost to the energy match, which
  runs on z-scores, and 2 of 15 tracks started at bar 0. Hard ranges and filters replaced soft terms:
  the same lesson came up for the energy fit and for the played-in-many-sets term, which now sit on one
  scale (`_z`).
- Looking up the most similar real seam for a join. Ruled out by Anas (a seam was made for its two
  tracks) and by Test 2 above (the situation does not predict the DJ's choice).
- Reading the group from the music alone. Right on 0.11 to 0.68 of mixes by genre for unseen DJs.
- Key distance and energy step as reasons for a type. On 3,646 real seams only the tempo gap differs by
  type (cuts 1.63 %, centre swaps 0.97 %).

Open, in the order agreed or proposed.

1. Fred again's gaps. 53 of his 110 measured joins are a stop and a restart (median 22 bars apart). The
   mixer has no such type and his shares are counted over the 31 typed joins, so his cuts are overstated
   (39 % against 11 % of all joins).
2. Version pairs for a named DJ's long blend.
3. Vocals per bar for all held tracks (a vocal separator on the bigger VM), so two voices never overlap.
4. The style's own bass-swap position (`new_bass_at`, `old_bass_out` are in the tables, the mixer uses
   its fixed ones).
5. Beat matching where one track has no kick yet. Styles start tracks around bars 9 to 18 and the mixer
   aligns on kicks. For planned seams the low layers decide when the layers split half a beat apart
   (`_layer_consensus`), measured on one seam.
6. The opener and the closer follow no plan. A frozen test over genres, DJs and curves has not been run:
   only techno, house, melodic house and two named DJs were rendered, each after a change.
7. The request as JSON checked by a pydantic model, then a model that fills it from a prompt (Anas: no
   LLM yet, arguments by hand). The critic of TODO item 7.

Ear-test files: `data/external/ear_test/system_2026-11-18/` to `system_2026-11-25f/`, one folder per
round, each with the set, its log and `work/<name>/plan.json`. Anas called `system_2026-11-23/
techno_blend.flac` "nice".

1. Per DJ and per genre, the measured shares of the five transition types, and per type the recipe from
   the same seams (overlap bars, bass swap bar, the bar each band of each record enters or leaves, bass
   cut bars, sweep bars). Profiles carry the whole distribution, never a median (Anas, 2026-10-12), so a
   recipe is drawn from that DJ's own seams of that type.
2. A mood mix uses the genre row, cut by set energy and by club or radio where the data shows those
   change the shares (not measured yet). A profile mix puts the DJ row in front.
3. The types for the whole set are drawn first in the row's proportions, then each type is placed on a
   seam that can carry it. The mixer's rules veto: BPM gap, key clash for a melt, the curve gate (a chill
   set never slams). The cue points are then chosen for the type, using the window search the mixer
   already has for section boundaries and drop landings. A type no seam can carry is swapped for the
   next in the row.
4. The mixer renders every type its own way (Anas: the mixer's recipes are refined by ear and stay; a
   melt from the data is the mixer's melt).
5. Judged by ear: real seams of one DJ next to the mixer rendering the same records with that DJ's row,
   and whether it sounds better than the rules. Fred again is one of the DJs to profile (33 measured
   house seams).

Open before building: whether consecutive seams in a mix depend on each other (two cuts in a row, tension
after a long blend), a few minutes on the seams table. Answered 2026-11-13 under Test 3: they do
not. Genre and set energy, left unmeasured in point 2, are answered there too.

## Ear test, chill sets, 2026-10-15

`scripts/experiments/ear_test_sets.py`. One chill set per genre, six records, held full audio only (every
file from S3, the first live use of that path), play length the genre median. Output in
`data/external/ear_test/sets_2026-10-15/`: one flac per genre, the plan with each record's entry time,
`summary.csv`, the render log `driver.out`. Each pair of numbers is the planned set, then the real sets
of that genre (sets of six or more records; clash means Camelot distance over 2, as the planner reports).

| genre | minutes | clash share | same key | opener energy | set energy | transitions |
|---|---|---|---|---|---|---|
| house | 24.9 | 0.80 / 0.60 | 0.00 / 0.08 | p20 / p38 | p19 / p56 | blend ×4, melt |
| techno | 23.2 | 1.00 / 0.62 | 0.00 / 0.07 | p21 / p38 | p24 / p53 | blend ×5 |
| tech house | 23.2 | 0.20 / 0.62 | 0.20 / 0.06 | p22 / p53 | p24 / p56 | blend ×4, melt |
| melodic house | 21.8 | 0.00 / 0.49 | 0.60 / 0.13 | p20 / p51 | p31 / p56 | blend, fade, blend, fade, melt |
| afro house | 27.5 | 0.00 / 0.58 | 0.20 / 0.09 | p25 / p48 | p32 / p50 | blend ×5 |
| trance | 26.4 | 0.40 / 0.50 | 0.00 / 0.14 | p22 / p51 | p12 / p52 | blend ×3, melt ×2 |
| drum and bass | 12.7 | 0.40 / 0.61 | 0.40 / 0.09 | p21 / p43 | p28 / p52 | blend ×5 |

Energy percentiles are within the genre's pool of held records. Real sets: the opener's percentile is
the median over 3,709 sets, the set's is the median of each set's median.

Anas on melodic house:

- Track selection is fine.
- Every set starts too low on energy and is not interesting from the start. The chill curve opens at
  0.15 of the pool's 15th to 85th percentile range.
- 6:33 to 7:04, Human Oscillator into Ravel, sounded wrong. The rules chose a fade because the energy
  drops over the join (0.265 to 0.209), and a fade filters A's mids and highs away. Same key, same BPM.
  Anas would land on the drop, slam into a drop, or melt long. The rules forbid a melt there (energy gap
  over 0.05) and the chill curve forbids drop landings.
- 9:34 to 10:05, Ravel into Memories, sounded like a track change. A 16-bar blend, cued into Memories at
  bar 48, past a minute and a half of its intro, with the overlap level stepping down 8.5 dB on the way
  in and the seam decided by the consensus band after a 242 ms shift.
- 13:40 to 14:11, Memories into Paper Walls, sounded perfect. Also a 16-bar fade, a 4 dB step, the seam
  decided by the kick with no shift.

Found in the render logs:

- The mixer spells the genre "drum and base" and the dataset "drum and bass", so the drum and bass play
  length, phrase bars and halved overlaps never apply.
- The drum and bass tempo was read at half (84 BPM for 170 BPM records), so its windows came out 23 to 56
  bars and the set ran 12.7 minutes.

## Club or radio, 2026-10-15

A title rule splits the real mixes (a show name, "radio", "podcast", an episode number means radio; "@",
a venue, "festival", "club", "live" means live). Script inline in the session, table in
`data/external/ear_test/sets_2026-10-15/mix_type_by_genre.txt`. Over 3,937 mixes: radio 713, live
1,270, both 351, unknown 1,603 (mostly the DJ data, whose mix title is the DJ's name). Clash share,
same-key share, energy and BPM climb are the same on both sides. Hold length per record differs:

| genre | radio | live |
|---|---|---|
| house | 4.40 min | 2.70 |
| techno | 4.41 | 3.47 |
| trance | 3.77 | 2.97 |
| tech house | 4.38 | 3.95 |
| drum and bass | 2.92 | 3.20 |

The DJ data report found earlier (2026-10-09) that club against radio explains 0 to 4 % of the variance
in the transition signatures, and most of that is Adam Beyer.

## Test 3. Genre, set energy and the previous seam, 2026-11-13

On the 5,122 measured seams with one of the five signatures (centre swap, cut, tension, end swap, late
swap), 412 DJs, 1,092 mixes. The script was a scratch file and was not kept.

- **Genre.** The type shares differ by genre. Cramér's V is 0.171 against 0.034 with the genres shuffled.
  Trance has tension on 0.03 of its seams and cuts on 0.34. DJs inside one genre differ more than the
  genres do. The standard deviation of a type's share across the DJs of a genre is 0.10 to 0.27.
- **Set energy.** Inside the same DJ, the energy of the set does not change the type. 45 DJs with 613
  mixes, p 0.66, V 0.028.
- **The previous seam.** Consecutive seams are independent. The same type twice in a row occurs on 0.427
  of consecutive pairs against 0.419 with the types shuffled inside each mix, p 0.13.

- **The kind of set, 2026-11-18.** Each mix got `set_kind` from its energy inside its genre (warm-up,
  middle, peak-time, `mixes.parquet`). Across DJs the type shares differ by it in four of five genres
  (p under 0.01, trance 0.07), and the direction flips between genres. Peak-time techno has a median
  overlap of 10 bars against 27 for warm-up, peak-time tech house 32 against 8. Inside one DJ nothing
  consistent moves: of the 5 DJs with 15 seams of both kinds, 2 have the longer overlap in warm-up, 2 the
  higher cut share in peak-time, 3 the earlier entry in warm-up. So the difference comes from which DJs
  play which kind of set.
- **Setting, 2026-11-18.** The title names a setting on 354 of 849 measured mixes. Techno festival sets
  (20) run at 135.5 BPM with a track change every 3.9 minutes and a median overlap of 11 bars, techno
  radio sets (49) at 128.0 BPM, 4.5 minutes and 39 bars. Club and festival sets do not play louder tracks
  (energy 0.36 and 0.47 against 0.52 for radio). Higher energy goes with faster changes inside a genre
  (rank correlation -0.24).

**What it means.** The DJ row of the design above carries the type. The genre row is a weaker stand-in for
a mood mix. Nothing has to be modelled about the order of types inside a set or about the set's energy.
A DJ mixes the same way in a calm set and in a loud one, so the kind of set says which DJs to follow.

## Where DJs cue in and out, 2026-11-14

From the cue columns of the 2026-11-14 dataset (`b_in_bar`, `a_bars_left`, the sections; how they are
computed and their limits are in `DATA_ARCHITECTURE.md` and `DATASET_STATE.md`). All 6,731 measured seams,
425 DJs, 1,172 mixes. Bars are the track's own, from Beat This!.

Where the incoming track is first heard, share of the genre's seams by bar:

| genre | seams | before bar 0 | 0 to 8 | 8 to 16 | 16 to 48 | past 48 |
|---|---|---|---|---|---|---|
| house | 1,004 | 0.21 | 0.34 | 0.18 | 0.23 | 0.04 |
| drum and bass | 777 | 0.20 | 0.31 | 0.19 | 0.26 | 0.04 |
| tech house | 2,585 | 0.11 | 0.29 | 0.20 | 0.33 | 0.07 |
| trance | 751 | 0.17 | 0.25 | 0.16 | 0.35 | 0.07 |
| techno | 1,187 | 0.06 | 0.21 | 0.19 | 0.38 | 0.16 |

"Before bar 0" is an entry read more than 2 bars before the track's first bar. It is a looped intro, a
different version or a wrong time zero, and the three are not separated. Counted with the first 8 bars it
gives the 2026-11-13 figures (house 0.59, techno 0.31).

How many bars of the outgoing track are left when it is last heard:

| genre | past its end | last 8 | 8 to 16 | 16 to 48 | over 48 |
|---|---|---|---|---|---|
| house | 0.11 | 0.32 | 0.13 | 0.29 | 0.14 |
| trance | 0.07 | 0.30 | 0.14 | 0.43 | 0.07 |
| techno | 0.04 | 0.27 | 0.18 | 0.36 | 0.15 |
| tech house | 0.05 | 0.26 | 0.16 | 0.36 | 0.17 |
| drum and bass | 0.11 | 0.21 | 0.12 | 0.31 | 0.25 |

- **Sections.** The incoming track is first heard in a drop on 0.32 of seams, in its intro on 0.25, in a
  buildup on 0.15, and before its start on 0.20. The outgoing track is last heard in a drop on 0.51 and in
  its outro on 0.27. At the bass swap the outgoing track is in a drop on 0.55 and the incoming one in a
  drop on 0.47, a buildup on 0.26 and its intro on 0.17.
- **Level.** Over the 8 s after its entry the incoming track sits a median 4.4 dB under its own loud
  level (quartiles -9.7 and -2.2) and its lows 5.6 dB under (-15.4 and -2.4). Over the 8 s before its exit
  the outgoing track sits 3.0 dB under (-6.8 and -1.7).
- **By type.** A cut enters past bar 48 on 0.09 of its seams, the swaps and tension on 0.01 to 0.04.
- **Cue-out against cue-in.** The bars left on the outgoing track and the entry bar of the incoming one
  are close to unrelated. Rank correlation 0.11 over all seams, house 0.23, techno 0.17, drum and bass 0.01.
- **The track or the DJ.** The same track brought in by two different DJs enters a mean 15.6 bars apart
  (median 11.5, 302 pairs) against 21.1 for random pairs (median 14.5). The track sets part of the entry
  point and the DJ the rest.
- **One DJ.** Claptone, 269 seams. The incoming track is heard before bar 8 on 0.80 of them and the
  outgoing one has under 8 bars left on 0.80.
- **What explains the entry bar.** Over 5,837 entries the mix explains 0.40 of its variance (inflated, a
  mix has about 6 seams and no shuffled baseline was computed), the DJ 0.23 (0.01 with DJs shuffled), the
  transition type 0.13 and the genre 0.04. The bar of the incoming track's first drop has a rank correlation
  of 0.02 with the entry bar, and 9 % of entries sit within 2 bars of it. DJs apply a habit. They do not
  aim at the drop.
- **How much of a track is played.** On 3,678 plays with both ends measured, a track is heard for 0.80 to
  0.89 of its length. That is 5.5 of 6.3 minutes in house, 5.4 of 6.8 in tech house, 5.1 of 6.6 in trance,
  4.7 of 6.2 in techno and 4.0 of 5.3 in drum and bass. The median play skips 11 bars at the start and leaves 15 at the end. This
  is first heard to last heard and includes the overlaps, so it is a different quantity from the mixer's
  `GENRE_PLAY_MINUTES`, which is the time between track changes.
- **Styles.** Adam Beyer overlaps a median 49 bars and plays the outgoing track into its last 8 bars on
  0.47 of seams. Amelie Lens overlaps 3 bars and brings the next track in past bar 32 on 0.52.
- **Before the start.** The entry reads more than 5 s before second 0 on 12 % of seams placed in resample
  mode and 3 % in key-lock mode, so most of those readings are a placement fault.
- **The drop label.** On a quarter of the tracks in measured seams the segmenter's first drop starts at
  bar 0. In techno it mostly means the loud body, so numbers against the drop hold best outside techno.

**What it means for the mixer.** Real DJs bring a track in within its first 16 bars on about 6 seams in 10
and past bar 48 on 1 in 13 (techno 1 in 6). They play the outgoing track into its last 16 bars on about
half. Not verified: the mixer's 4-minute play window on tracks of about 6.5 minutes trims more than this,
which would explain entries at bar 48 that were heard as bad. The check is to put the mixer's cue-ins on
the 35 ear-test seams next to these tables.

**Not answerable from this table.** Whether DJs enter on a phrase line. The cue times sit on a 2.5 s grid,
which is more than one bar.

## Rerun

```bash
PYTHONPATH=. python scripts/experiments/signature_learnability.py
PYTHONPATH=. python scripts/experiments/seam_situation.py curves --procs 6   # needs the records on disk
PYTHONPATH=. python scripts/experiments/seam_situation.py features
PYTHONPATH=. python scripts/experiments/seam_situation.py test --genre house
PYTHONPATH=. python scripts/experiments/seam_cues.py                         # cue table, needs curves/ and segments/
python -m src.data.profiles                                                  # mixes, groups and the two type tables
PYTHONPATH=. python scripts/experiments/ear_test_sets.py data/external/ear_test/sets_<date>
```
