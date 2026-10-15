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

Discussed with Anas, not built.

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
after a long blend), a few minutes on the seams table.

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

## Rerun

```bash
PYTHONPATH=. python scripts/experiments/signature_learnability.py
PYTHONPATH=. python scripts/experiments/seam_situation.py curves --procs 6   # needs the records on disk
PYTHONPATH=. python scripts/experiments/seam_situation.py features
PYTHONPATH=. python scripts/experiments/seam_situation.py test --genre house
PYTHONPATH=. python scripts/experiments/ear_test_sets.py data/external/ear_test/sets_<date>
```
