# Dataset state, 2026-09-25

## The chain

Nothing here is independent. Each thing needs the one below it.

```
feature extraction
   needs  transition reading
             needs  band reading, bass reading, gimmick reading
                       needs  the transition correctly identified
                              AND the correct records used in it
                                 needs  the record located in the mix by its anchor
                                        needs  the right audio file for that record
```

**Raveform is resolved to the top of that chain.**
**DJ profiling breaks at the bottom**, at finding the record inside the mix.

**2026-09-24, what it turned out to be.** Neither the track files nor the playback speed was the main fault.
The MIX files were: every DJ mix had been chosen by a YouTube title search that was never checked, and many
were other sets, radio shows or clips, so no track could be found in them. That is fixed (281 mixes from
their own 1001 page links, see the DJ PROFILING note below and `DATA_ARCHITECTURE.md` section 0). The
speed theory and the quad-fingerprint work in **THE BLOCKER** below are kept as a record of what was tested;
the speed part is superseded, the quad fingerprint remains a valid second opinion.

---

# 2026-09-25, the profiling pipeline is in the package, checked on synthetic audio, three passes on
# five real mixes, and heard

**Heard.** Anas listened to the pass-three clip sets in `data/external/ear_test/pipeline_2026-09-25_all_djs`
(twelve seams across Amelie Lens, DJ Tennis, Solomun and Black Coffee), `_loops` (the three loop
seams) and `_tension` (six tension seams), each with the window in the left ear and both records placed
as located in the right, and the measured entry, exit, bass swap and label in MARKS.csv. His verdict on
2026-09-25: the ear test is right and usable. So the locate, the pairing, the cut and the labels of
this pass hold on real DJ audio by ear, which is the only proof that counts here.

Read `PLAN.md` "DATA PIPELINE" for the plan and `DATA_ARCHITECTURE.md` section 9 for the layout.
Everything below was built and run on 2026-09-25 on the laptop, on the development sample
(`data/interim/dev_sample_files.csv`, four DJ mixes, Fred's Vancouver segment, 20 Raveform seams).

## What is in `djdata_package/djdata` now

```
seam/fingerprint.py  the landmark fingerprint, imported from the fetcher, the vote arithmetic
seam/floors.py       the control floor rule, moved from scripts/diag unchanged
seam/locate.py       the whole-mix locator, ported from scripts/diag/mix_locate_once.py
seam/pairs.py        which record follows which, from what was heard; the bounded window
seam/cut.py          ffmpeg stream copy and the ten second audit at both ends
seam/bands.py        per band entry and exit (fp_bands library functions)
seam/bass.py         the bass swap and the bass words (fp_bass, fp_bass_timeline)
seam/presence.py     full-band presence along the window (fp_presence_batch3 rule)
seam/loops.py        the outgoing record standing still (the labeller's rule)
seam/measure.py      one seam measured, every floor from three controls on that window
seam/tempo.py        BPM per record, librosa hint then the mixer's kick autocorrelation
seam/labels.py       the nine transition types, in bars, first match wins
seam/eartest.py      clips for Anas from the tables, window left, records right
sources/tracklists.py, sources/raveform.py   what a corpus is and where its audio is
store/tables.py      the dataset tables, append only CSVs: plays, seams, cuts, measures, tempos, labels
pipeline.py          the stages, one function each, resumable, spawn pools with BLAS pinned to one thread
cli.py               djdata locate | pairs | cut | measure | tempo | label | export-seams | ear-test
```

The old run's locate moved to `legacy/locate_slice.py` with its bug noted. `fetch/mix.py` still
imports it for the old fetch path and is untouched. Nothing in `scripts/diag` was deleted yet.

## Tests: 88 on synthetic audio, all green, in `make ci`

`tests/synth.py` builds records from random chords with a known answer and mixes with known time
zero, speed, deep cue, band gates and a bass swap. Every module has its own test file. What they
prove: a record played 3 % fast at 90.00 s comes back at 90.00 s and rate 1.03; a deep cue's played
part is right to 5 s; an absent record sits under a floor set by three controls; pairs follow the
audio order and flag the tracklist; a window that starts inside the overlap or ends before A left
fails the audit; band entries and exits land on their gates to one step; the bass swap lands on the
swap; the words read outgoing, both, cut, incoming; a loop is a record standing still; a synthetic
record at 127.5 BPM reads 127.5 within 0.2; Raveform's alignment stands in for locate and its windows
for the cut through to a label.

Two things the synthetic tests exposed and the code now carries:

- **The presence sweep loses a record early when the other record dominates the peaks.** On a
  synthetic seam with both records at equal level, A's last heard came 80 s before A stopped. So
  `pairs.REACH_S` is 120 s past last heard, the cut audit catches a window that still ends too soon,
  and the measure stage's anchored per-band exit is the precise number. Play length must come from
  the measured exit, not the sweep, and that comparison is still to be made on real seams.
- **A record shorter than three 45 s sections can never be "confident".** Real records are five to
  eight minutes; a two minute edit will read "by votes" at best.

## Tempo, checked against the catalog on the development tracks

`data/interim/tempo_check_dev_2026-09-25.csv`, 48 tracks that have a catalog BPM (DeepRhythm, integer).

```
within 1 %   39 of 48      within 2 %   40 of 48      median error   0.18 %
off by more than 2 %:  1 an octave, 2 spoken word (Chaplin, Funkadelic), 5 real disagreements (3 to 20 %)
```

The catalog is not ground truth either. Bars from this tempo are used, and a seam's label in bars
is only as good as this on about one record in eight. Not fixed.

## The development sample located: five mixes, 126 listed records, 1 h 48 min at two laptop workers

```
mix          DJ               min  listed found confident by votes not found  floor  audio order = listed
be77589ff6   Amelie Lens       57     15    15         6        9         0    140   14 of 14
ead5fc8098   Amelie Lens NYE  125     35    27         7       20         8    176   26 of 26
3ec220759a   Solomun          223     31    24        23        1         7    126   23 of 23
74a372440e   Black Coffee     120     17    14         7        6         3    292   13 of 13
b945ffd03d   DJ Tennis NYE    175     28    26        23        3         2    214   23 of 25
```

106 of 126 found (84 %), 66 confident (52 %). Audio order agrees with the listed order on 99 of 101
consecutive found pairs; the two disagreements are in the DJ Tennis set that held the repeated-record
case of the 2026-09-25 ear test. Each record is heard a median 2.5 to 3.2 min by the sweep. The
counts match the 2026-09-24 sample run on the mixes both runs share (Solomun 24 of 31, DJ Tennis
26 of 28). Amelie Lens techno reads "by votes" on most records: two to four 45 s sections per record
and loops that make sections disagree.

## The development sample through every stage (first pass, sweep at 15 % share)

```
dj             seams usable  cut  start ok  end ok  measured  both clear  window med  sep A med  sep B med  overlap med  bass swap
Amelie Lens       40     31   31        31      27        31          31       176 s        8.8        7.4         10 s     26 of 31
Solomun           23      2    2         2       2         2           2       184 s       85.8      114.9         14 s      2 of 2
Black Coffee      12      4    4         3       4         4           4       154 s       87.2       80.7         45 s      4 of 4
DJ Tennis         25     15   15        15      15        15          15       170 s       62.8       43.7         25 s     12 of 15
labels over 52    long_blend 15  short_blend 13  edit_or_talk 9  tension 4  loop 4  sweep_in 3  sweep_out 3  cut 1
```

All 52 cut windows measured with both records over their floors. The end audit failed on 4 of 52
(A still there at the end of the window) and the start audit on 1. Solomun and DJ Tennis separations
are ten times Amelie Lens's: house records with melodic content fingerprint far better than looped
techno. Every rule fired at least once on real DJ audio.

**48 of 100 seams were set aside, all for one reason: a gap of 128 to 1,199 s between A last heard and
B first heard.** Solomun kept 2 of 23. He plays each record about nine minutes and the sweep heard a
median 2.5, because a window only counted as present at 15 % of the record's own loudest window. That
rule was replaced the same day: the presence floor is now twice the loudest window any control record
scored in the mix, the same control rule as everywhere else, and it travels in `plays.csv` as
`sweep_votes` and `sweep_floor`. Re-run on the sample below.

## Second pass, sweep at the control floor: almost no change, so the share was not the cause

```
                     pass 1 (15 % share)      pass 2 (control floor)
records found            106 of 126               106 of 126
heard median             2.6 min                  2.7 min   (Solomun 2.5 -> 2.8, Black Coffee 2.7 -> 5.2)
seams usable             52 of 100                53 of 98  (Solomun 2 -> 4 of 23)
cut audits               start 51/52 end 48/52    start 51/53 end 47/53
measured, both clear     52 of 52                 53 of 53
labels                   long 15 short 13 edit 9 tension 4 loop 4 sweeps 6 cut 1
                                                  long 20 short 9 edit 8 tension 5 loop 3 sweeps 7 cut 1
```

Records are matchable for about 2.5 to 3 minutes whichever threshold is used, on every DJ. The
sweep counts pairs within 2 frames (23 ms) of one offset. The speed grid is 0.05 %, so a residual
error of 0.025 % drifts 23 ms in about 90 s, and presence can only reach a minute or two either side
of the anchor. That is the median heard. Checked next on Solomun's records section by section; the
fix, if it holds, is to read presence per 45 s section at each section's own offset, which the
locator already computes and which tolerates 15 s of drift.

**Checked, and it is not drift.** Solomun's three strongest records, section by section against
the mix: every 45 s section sits at +0.00 s from the consensus offset across up to nine minutes, with
1,500 to 7,700 votes each, and the sweep's extent equals the sections' extent on all three (6.3, 8.7
and 4.1 min heard). The fingerprint stays locked. What the 27 unusable listed-neighbour seams have in
common is 2 to 19 min of mix between A last heard and B first heard with both records confident,
for example Solomun's record 26, a four minute record played whole, then 3.6 min before record 27.
That is looping or edits, which one offset cannot follow, or unlisted audio. Tested next by a free
match of both records and a control against the gap region.

**It is loops.** Four Solomun gaps, both records and a wrong-record control matched freely against
the gap audio, best offsets and their votes:

```
seam 7->8   gap 11.8 min   A at four offsets 431, 337, 306, 196 s before the region: 637, 625, 550, 507 votes
                           B at 598 s (its located place) 752 votes            control 35
seam 26->27 gap  3.6 min   A at 2, 41, 9, 17 s: 2,755, 2,629, 2,583, 2,515 votes   B absent (27)   control 35
seam 8->9   gap 10.1 min   A at 241 s (three bins) and 272 s: 3,318 to 2,969   B at 379, 389, 401, 413 s: 1,724 to 1,649
seam 14->15 gap 11.2 min   A at 189, 158, 181, 165 s: 312 to 274                  B at 665, 667, 663, 655 s: 830 to 698
```

The DJ repeats sections. Each repeat shifts the fingerprint offset by the loop length, and the sweep
at one offset heard none of it. So presence is now read per 30 s window at the record's best offset
in that window, whatever it is, against the controls' per-window floor; the section consensus keeps
one time zero for the cut. A record heard past its own end (looped) bounds its window by where it was
last heard, not by its length. Third pass on the sample follows.

## Third pass, presence per window at any offset: the loops are heard

```
                        pass 2 (one offset)       pass 3 (any offset per window)
heard median            2.7 min                   4.7 min   (Solomun 2.8 -> 5.8, DJ Tennis 2.7 -> 5.7, Black Coffee 5.2 -> 6.7)
gap median A to B       106 s                     1 s
seams usable            53 of 98                  70 of 98  (Solomun 4 -> 11, DJ Tennis 13 -> 19, Amelie Lens 29 -> 32)
cut audits              start 51/53 end 47/53     start 66/70 end 64/70
measured, both clear    53 of 53                  66 of 70, the 2 that crashed on an empty slice now read as unmeasured
bass swap found         47                        55
labels                  long 23 short 15 edit 8 tension 6 sweep_in 5 sweep_out 3 loop 3 cut 1 unmeasured 4
locate time             5 mixes, 61 min at four laptop workers
```

The 28 seams still set aside are gaps the records themselves do not cover, records not found, or
listed records between the pair. They stay as rows with the reason. Everything measured here is
unheard; the clip sets in `data/external/ear_test/pipeline_2026-09-25*` are rebuilt from this pass.

**A limit the loops leave in the measure stage.** Locate now follows a looped record, but measure
still lays each record on the window at its one time zero and counts anchored votes there. When the
outgoing record is looped into the window, it is absent at that offset, its exit reads as nothing,
and the seam is `unmeasured` (6 of 70, two of them crashed on an empty slice before the guarded zero
reading). The honest fix is a loop-tolerant measure, per band free matching of the record slice per
step inside the window, and it is not built. Those seams stay unmeasured rather than guessed.

## The first real mix through every stage: Amelie Lens be77589ff6 (57 min, 15 tracks)

```
locate    15 of 15 found, floor 140 from controls at 51 to 70 votes, 6 confident, 9 by votes
          audio order equals the listed order on all 15; listed minutes sit about 2 min late throughout
pairs     14 seams, 13 usable; one flagged: 6 min between record 1 last heard and record 2 first heard
cut       13 of 13, windows 107 to 237 s, start audit A alone 13 of 13, end audit B alone 13 of 13
```

```
measure   13 of 13, both records over their floors on all 13; separations 1.3 to 304, median about 6
          the bass swap agrees with the low band's entry within a few seconds on 11 of 13 (two
          independent fingerprints); mids enter before lows on 9 of the 11 seams that have both
          about 110 s per seam at two laptop workers
tempo     15 of 15 records, 120 to 155 BPM, bars 1.55 to 2.0 s
label     long_blend 5, short_blend 4, edit_or_talk 4 (overlap -8 to -27 bars), no loop, no sweep, no tension
clips     data/external/ear_test/pipeline_2026-09-25/  ten seams, window left ear, both records right ear,
          MARKS.csv with in, out, bass swap and label per clip. NOT YET HEARD.
```

The sweep against the measured exit on these 13. The locator's last heard came 30 to 70 s before
the measured full-band exit on four seams and 20 to 55 s after it on two, so it is not a play-length
measurement. The measured exit inside the window is. The four `edit_or_talk` labels are seams where
the incoming record's presence starts 20 to 50 s after the outgoing record's ends, which is odd for
a techno set and is the first thing to listen for in the clips: either the presence floor misses a
quiet tail or filtered intro, or those are real edits.


---

# RAVEFORM — good

5,062 usable seams, 919 mixes, 6,579 tracks. On the VM at `~/AI-DJ/data/djdata/raveform` and on S3 at
`s3://aidj-1/djdata/`.

## How we find the tracks

We do not search for them. Raveform ships a YouTube video id per record, and **the track id IS the video
id**. `s2OWD0_5Fkw` is both the row key and `youtube.com/watch?v=s2OWD0_5Fkw`. Download is by url, never
by name, so there is no wrong-recording risk.

Position in the mix comes from Raveform's own DTW alignment, shipped in `alignments/*.jsonl`. Per record
it gives `mixin_time_mix`, `mixin_time_track`, `mixout_time_mix`, `mixout_time_track` and the playback
rate. **Our `locate.py` never runs on this corpus.**

How good that alignment is, measured on 60 seams, one per mix, all five genres, wrong-record control on
every seam:

```
both records found in the window        56 / 60   93%
a record not found at all                4 / 60    7%
Raveform's anchor vs the measured position, median lag    A 0.06 s   B 0.07 s
free fingerprint votes, median          A 994   B 880
wrong-record control, median            13
control failures                        0
```

Anas ear-checked placement on 45 files across two sets. **40 of 45 right.** The 2 marked off were loops,
where the record stands still and the fingerprint reads it as not advancing. That is not a wrong placement.

## How we cut

`djdata_package/djdata/manifest/raveform.py::_coarse`, then `fetch/mix.py::cut_window`.

```python
b_track0_mix   = b.mixin_time_mix  - b.mixin_time_track / rb      # where B's own 0:00 falls in the mix
a_trackend_mix = a.mixout_time_mix + (a_dur - a.mixout_time_track) / ra   # where A must be finished

t0 = max(a.mixin_time_mix,  min(b_track0_mix   - pad_s, ov_start - alone_s))
t1 = min(b.mixout_time_mix, max(a_trackend_mix + pad_s, ov_end   + alone_s))
```

Params `pad_s` and `alone_s` in `djdata_package/config.yaml` under `window`.

The rule is that B cannot be audible before its own time zero and A cannot be audible after its own end,
so cutting that span holds the transition by construction.

Measured on a random 300 windows across every container:

```
cut accuracy   mp3 and m4a (291 of 300)     within 0.05 s
               webm (9 of 300)              median 5.9 s of excess, at the FRONT
the overlap sits inside the file            290 / 300   96.7%
margin before the overlap                   median 53 s, minimum 45 s
margin after                                median 62 s, minimum 45 s
```

248 of the 6,218 windows are webm and carry that front excess, because the old cut put `-ss` before `-i`
with `-c copy`. It does not matter, because the aligner re-derives position from the window audio rather
than trusting the nominal start, and all 248 have at least 40.6 s of head margin.

## Band, bass and transition reading

`scripts/diag/measure_seam.py` is the runner. Bands are low 20 to 200 Hz, mid 200 to 3000 Hz, high 3000
to 14000 Hz, defined as `BANDS` in `fp_bands.py`. Bass is the low band on a 2.7 Hz grid in `fp_bass.py`.

Measured on 30 seams, all genres, three wrong-record controls each, every floor derived from those
controls:

```
band   floor   separation, median   records clearing their own floor
low        4                 14.7                            57/60
mid       10                 67.1                            58/60
high       8                 24.2                            45/60
bass       5                 25.0                            58/60

measured (both records clear somewhere)                       30/30
```

Separation is how many times the real record beats its own noise floor. At 15 to 67 these are settled
measurements, not close calls.

Earlier on the 25-seam set: the wrong-record control never exceeded 7 votes in any band across 75 rows,
the bass swap was found on 24 of 25 with a control of 3 or less, and the EQ order fell out of it. Mids
enter before bass on 13 of 16, bass leaves before highs on 13 of 18, highs outlast bass by a median 10.5 s.

## Transition labels

`scripts/diag/transition_labels.py` turns those measured columns into types by threshold rules. It has
run on **124 seams across 79 DJs and 5 genres**, output in
`data/external/ear_test/transition_labels_all_seams.csv`.

```
long_blend 64   tension 12   short_blend 11   sweep_in 10   sweep_out 9
edit_or_talk 6  unmeasured 6  loop 5   cut 1
```

Nine types with thresholds written: loop, edit_or_talk, tension, sweep_out, sweep_in, cut, layer,
long_blend, short_blend.

## What Raveform still needs

**It has never been measured at scale.** `measure_seam` has run on 30 of 5,062. `djdata/out/` holds only
the 1,181 rows from the dead NNLS gain fit, which failed its wrong-record control and is discarded.
Running the full corpus needs nothing new. It is about six hours at twelve workers.

**The measured times have never been checked by ear.** In `ear_test/raveform_mapping/MARKS.csv` and
`raveform_mapping_genres/MARKS.csv` the columns `your_incoming_in`, `your_outgoing_out` and
`your_bass_swap` are empty on all 45 files. There is a third set, `ear_test/label_check/` with 93 seams,
whose marks I have not examined. So placement is ear-verified and timing is not.

---

# DJ PROFILING — the cut works, finding the record does not

> **2026-09-24.** The section below describes the corpus as it was before the mix audio was redone. What changed:
> the mix files were the problem, not the tracks. All 135 fetched mixes had come from a YouTube title search that
> was never checked, and many were other sets or clips. Every mix's audio now comes from the link on its own 1001
> page: 281 mixes, 4,414 tracks on disk, 25 tracks with no upload anywhere. Radio shows and mixes with no page link
> or a dead one are on the exclusion list, 279 mixes. The fingerprint-once whole-mix locate (`mix_locate_once.py`)
> found 29 of 35 page-link mixes to be their set with the listed minute right to within about a minute. What is
> still unmeasured is the tracks against their mixes, and no window is cut until that has run and been heard.
> Details in `DATA_ARCHITECTURE.md` section 0.

1,475 ready seams, 137 mixes, 3,158 tracks, on the VM at `~/AI-DJ/data/djdata/djs`.

```
Amelie Lens  695     DJ Tennis       52
Solomun      367     Fred again..    49
Black Coffee 288     Sultan+Shepard  24
                     Roman Flügel     0
```

## 2026-09-24/25 — the track locator for the DJ corpus, what was tried, how, and where it stands

Read this before touching the DJ corpus. Everything below was run on the VM (`aidj-data`, 16 cores) unless
it says laptop. All scripts are in `scripts/diag/`, all configs in `djdata_package/config_djs.yaml`.

### The question

For every record a DJ played, do we hold the right file, where in the mix does its own 0:00 fall, at what
speed, and which part of it played. Nothing downstream (window cut, band and bass reading, features) is
worth running until those four are trusted, and Anas's rule is that only his ears count as proof.

### The instrument: `scripts/diag/mix_locate_once.py`

Fingerprints a mix ONCE (the run's landmark fingerprint, `djdata.seam.locate.fingerprint`, unchanged: 22,050 Hz,
hop 256, peak pairs) in ten-minute chunks, then looks every listed track up in that one table. Per track:

1. speed: the nine coarse speeds of `locate.COARSE_RATES` by whole-record votes, early-accept at 400 votes at
   speed 1.0, then `FINE_RATE_STEP` = 0.05% steps within 0.25% of the best;
2. offset: the whole-record best, then each `SECTION_S` = 45 s section of the record searched within
   `LOCAL_FRAMES` = 15 s of it, and the vote-weighted median of the sections' offsets becomes the record's
   time zero (a loop-shifted pick moves to where most of the record agrees); `sections_agree` counts the
   sections within 0.5 s of that;
3. presence: the hashes agreeing with that offset are counted in `SWEEP_WIN_S` = 30 s windows stepping 10 s;
   the first and last windows at 15% of the record's best window (and at least 5 hashes) are `entry_min` and
   `exit_min`, and `played_from_s`/`played_to_s` = (entry/exit - time zero) x rate is the part of the record played;
4. floor: three control records from other mixes go through the identical path per mix and the floor is
   `floors.from_controls` (twice the loudest control). Acceptance is LOOSE on purpose (Anas, 2026-09-24):
   `found` = votes >= floor OR `sections_agree` >= 3; `confidence` says which ("confident" = both, "by votes",
   "by sections", "not found").

Output: one row per record and per control, `data/interim/mix_locate_*.csv`. Row columns: mix_id, dj,
mix_title, mix_minutes, play_order, track_id, title, is_control, listed_min, found_min (time zero, mix minutes),
drift_min (found - listed), votes, rate, control_max, floor, sections, sections_agree, found, confidence,
entry_min, exit_min, played_from_s, played_to_s, seconds.

Run: `python -m scripts.diag.mix_locate_once --mixes <ids> --workers 14 --out data/interim/<name>.csv`
(`--page-link` takes every mix with a 1001 page link). Resumable: rows already in `--out` are skipped.
Memory: 14 workers used 10 GB of 30 on 12 mixes including three-hour sets. Time: about 20 min per mix, so
the full 281 is ~7 h; the speed search on a section instead of the whole record would cut that to 2-3 h and
has NOT been done, so the set that was heard is the set that ran.

### How the instrument was checked, in order

1. **Synthetic, known answer** (`--selftest A B`, laptop): a record played 3% fast placed at 90.00 s comes back
   at 90.00 s, rate 1.03, every version. The production `locate.py::locate_one` returns 92.7 s on the same
   input: it multiplies the offset by the rate after the record was already resampled. That bug is NOT fixed
   in `locate.py`; everything here avoids it.
2. **Against the run's own locate** on the Amelie Lens NYE mix (`ead5fc8098`, 35 tracks the run had located,
   laptop): 29 found, time zero agrees to a median 0.6 s, p90 3.3 s (was 7.4 s before the section step; the
   6-9 s disagreements were loop shifts). One 55-minute disagreement is a record at 147 votes over a 142 floor
   with 1 of 8 sections agreeing: a vote-only acceptance, flagged as such.
3. **Ear test 1, one record at a time** (`scripts/diag/locate_lr_test.py`, 10 clips, record left ear, mix right,
   drawn across the floor): Anas heard six. 01 close, 06 match, 05 match but drifting (the 0.25% speed grid,
   now 0.05%) and a true match UNDER its floor, 02 same track wrong segment (loop shift, now the section
   step), 03 talk in the mix (correct negative), 04 same track badly placed at 49 votes (correct negative).
   Clips 07-10 were not heard. Folder `data/external/ear_test/locate_lr_2026-09-24/`.
4. **Ear test 2, the whole thing at once** (`scripts/diag/seam_ear_test.py`): every consecutive pair of a
   tracklist with both records found is a seam; the bounded window it would be cut on is written per seam;
   one 52-90 s clip across the crossing, mix left ear, BOTH records right ear placed at their located time
   zero and speed. 12 clips from the 12-mix sample, 8 with both records confident, 4 with a vote-only one.
   Folder `data/external/ear_test/seams_lr_2026-09-24/`, verdict columns in its MARKS.csv. **Not yet heard.**
   Clips 05, 06 and 10 have B's entry and A's exit 26-57 min apart, so the presence sweep is wrong on one
   record there and the clip may miss the crossing.

### The sample and its numbers (12 random mixes, seed 24, ids in the CSV)

```
tracks listed 276    located 204 (74%): confident 117, by votes 85, by sections 2    not found 72
consecutive pairs 262    both records located 160 (61%): both confident 63, a vote-only record 97
per DJ, both found:  DJ Tennis 77/133   Solomun 26/64   Amelie Lens 27/28   Sultan + Shepard 17/21   Black Coffee 13/16
```

Scaled to the 281 mixes that is roughly 1,500-1,800 seams with both records. Solomun is the outlier every
time (his own edits, festival recordings); nothing was done about him.

### What went wrong on the way, so it is not repeated

- The first section rule searched each section freely across the whole mix and drowned in coincidences
  (1-2 of 8 sections agreeing on strong records). Sections must be searched near the whole-record offset.
- The database still carried `data/djdata_djs/...` paths after the layout move; 159 of 268 sampled tracks were
  silently skipped until the paths were repointed (both copies, 2026-09-24). Glob by id, do not trust `path`.
- A 10-second presence window is too short for techno (Anas): a loop matches anywhere. 30 s it is.
- The run's own locate (`claim_track_before_mixes`) only claims tracks named by seams in tier `tracklists`
  that are not failed; 1,676 seams were still tagged `skipped_cut` from the September trim and 1,073 were
  `failed` from old track failures, so the track fetch sat idle until they were retagged and reset.

### Where it stands and what comes next

Waiting on Anas's marks in `seams_lr_2026-09-24/MARKS.csv` (A aligned, B aligned, transition inside). If the
confident seams pass and the vote-only ones are mixed, the plan is: the speed-search speed-up, then
`mix_locate_once --page-link` over all 281 mixes (one CSV, the seam index), then the second ear test Anas asked
for on the seam mechanics (alignment shift and stretch, bands, bass swap, sweeps) using `measure_seam.py` on
windows cut by `recut_windows_bounded.py` from that CSV, and only then features. No window has been cut from
the new locate and nothing in `out/played/` should be read.

## How we find the tracks — two separate steps, and the first one is the suspect

**Step 1, get the audio file.** The 1001 scrape gives "Artist - Title" and nothing else. `fetch/track.py`
searches YouTube for that text, ranks the results by name, downloads the top candidate, and **accepts it
on the name score alone when that score clears 2.6**, before it ever looks at the audio.

```
accepted on the title text only          2,911 of 3,158
no preview existed to check against        175
actually verified by fingerprint            72
```

**Step 2, locate it in the mix.** `djdata_package/djdata/seam/locate.py`. The track is resampled to each
of nine candidate speeds, fingerprinted, and matched against a slice of the mix. Every shared peak pair
votes for one time offset. The best offset must clear 40 votes. It returns `rate` and `mix_t`, which is
**the mix time the record's own time zero falls on**, not where the DJ brought it in.

If step 1 fetched the wrong recording, step 2 cannot find it, and the seam has no usable anchor.

How often step 2 succeeds, over all 2,950 records in ready seams:

```
under  60 locate votes     18%      (the floor is 40)
under 100 locate votes     31%
```

## How we cut

`scripts/diag/recut_windows_bounded.py`. Same rule as Raveform, but the inputs come from our locate
rather than from a DTW alignment.

```python
t0 = max(0,       b.mix_t - PAD_S)
t1 = min(mix_len, a.mix_t + a.duration / a.rate + PAD_S)

PAD_S = 25.0    MIN_WINDOW_S = 60    MAX_WINDOW_S = 900
```

Output goes to `data/djdata/djs/windows_v3`. Measured on 149 windows: **149 of 149 within 0.026 s, and
the span the transition must lie in is inside the file 149 of 149.** Window length median 177 s, p90
440 s, max 744 s.

**Only 149 of 1,475 have been cut.** The rest is minutes of ffmpeg and about 5 GB.

## The cut is fine. Finding the record decides everything

Same audit on both corpora, wrong-record control on every seam:

```
                                    both records found   clean crossing   record not found
DJ, bounded cut, votes >= 150              92%            84% of those          8%
Raveform, its own windows                  93%            77% of those          7%
DJ, bounded cut, iTunes-verified           68%            79% of those         32%
```

**Once both records are found, the DJ cut is as good as Raveform's.**

```
cut                                       transition inside
old, listed 1001 time, 90 s before 210 after    never audited
b.mix_t +/- 120 s                               20 / 60   33%
bounded span, any tracks                        26 / 60   43%
bounded span, iTunes-verified tracks            27 / 50   54%
bounded span, locate votes >= 150               38 / 50   76%
Raveform, identical audit                       43 / 60   72%
```

Gating on both records scoring at least 150 locate votes keeps **719 seams in 95 mixes**, and 76% of
those have a clean crossing. So about **550 usable DJ seams today**.

## Why the two earlier cuts were wrong

| where | what it centred on | what that is |
|---|---|---|
| the run, `fetch/mix.py::_window_for` | the 1001 page's listed start time | accurate to the minute, rolls over hourly |
| the recut, `scripts/diag/recut_windows.py` | `coarse["B"]["mix_t"]` ± 120 s | where the record's own 0:00 falls, not the entry |

A record cued in 90 s deep enters 90 s after that window's centre. Measured: the incoming record first
sounds a median **22.5 s after the centre**, 30 of 35 after it, worst +102.5 s. And the outgoing record's
theoretical end falls inside that window in only **643 of 1,475 seams, 44%**.

The manifest's "transition inside 98.7%" was arithmetic. It cut at `entry - 120` and reported
`entry - t0`, which is 120 by construction.

**The same geometry is copied into seven scripts.** Anyone running `recut_windows.py`,
`seam_window_check.py`, `ear_test_djseams.py`, `align_windows.py`, `seam_reconstruct.py`,
`fp_bands_dj.py` or `seam_rebuild.py` recreates it.

## The measurement gap this causes

Same code, same controls, both corpora:

```
                                   measured    low    mid   high   bass
Raveform, 30 seams                   30/30    14.7   67.1   24.2   25.0
DJ, vote-gated, 49 seams             39/49     2.0    3.2    1.3    2.2
DJ, iTunes-verified, 50 seams        29/50     1.0    1.5    0.7    1.0
```

The floors the two corpora derive are almost identical, so the noise is the same. The difference is how
far above it the real records sit. A separation of 1.3 means half the DJ records scrape past their own
noise, which is luck and not evidence.

---

# CLOSED — do not try these again

## The iTunes check is not a gate. Stop using it to judge DJ tracks

`verify_match` against a track's 30 s iTunes preview answers "is this file the commercial release of
this title". That is not the question. The question is "is this the audio that plays in this mix", and
the check does not answer it.

Three independent measurements, all 2026-09-21/22:

```
on Raveform, where tracks are downloaded by id and cannot be wrong,
iTunes still calls them wrong                                        11%
on records that locate strongly in their mix, iTunes calls wrong      8%
on 100 records that iTunes calls CORRECT, invisible in their window  33%
```

That last one is decisive. A clean sample of 100 iTunes-verified records, and a third of them cannot be
found in the mix they belong to. iTunes verification carries no information about findability.

It also measures worse as a gate. Seams selected by iTunes verification give a median separation of 1.0.
Seams selected by locate votes give 2.0 on the same code.

**Use locate votes.** The iTunes preview has no further role in judging the DJ corpus.

## Refining the playback rate does not help. It slightly hurts

`align_record` was wired into `measure_seam --refine-rate` and run against an identical plain run on the
same 50 seams.

```
separation        mean    median          per record
plain            11.52      2.38          improved by >20%   11
refined           9.81      2.25          worse    by >20%   16

of the 34 records whose rate actually moved: improved 8, worse 16
shifts applied: median 0.22%, max 1.23%
```

The shifts were the size that theory said would matter, 0.22% against the 0.4% that costs a factor of
five in votes. They were applied and nothing improved. **Playback rate is closed as an explanation for
the weak DJ readings.**

A second result from the same run is more useful. **The envelope fit could not lock on to 66 of 100
records at all.** The envelope uses onset energy and the fingerprint uses spectral peaks, so they fail
for different reasons. When both fail on the same record, the simplest reading is that the record is not
in that audio.

---
# THE BLOCKER — it is our instrument, not the tracks  (2026-09-22; superseded 2026-09-24: the mix files were the fault, see the top of this file)

## Everything tried, and where each one stands

```
TRIED AND FAILED
  iTunes preview check as a gate      ruled out. 11% false alarms on a corpus where
                                      tracks cannot be wrong
  playback rate refinement            ruled out. 34 records moved by a median 0.22%,
                                      8 improved and 16 got worse
  refetching wrong tracks             0 of 8 recovered, the uploads are restricted
  a fresh cookie file                 identical failures, identical successes

TRIED AND WORKING
  the bounded span window cut         84% clean crossings once a record is found,
                                      against Raveform's 77%
  gating on locate votes              719 seams kept, 76% with a clean crossing
  control-derived floors              every threshold now carries its own evidence
  the quad fingerprint                speed-invariant to 5% up and 3% down, exact
                                      position, speed to four decimals
  the clean-region test               all six controls positive, so a null means
                                      something

PROVEN AS A GROUP, NOT PER RECORD
  the low-vote records are wrong      21 of 21 fail their own clean-region test, which
  files                               clears 11 of 30 records known to be present
  the high-vote records are ours      3 of 7 clear a strict floor, up to 43.7x
  to fix, not the files'

BELIEVED, NOT YET PROVEN
  walking outward from a clean        would give the overlap edges. Designed, not built
  match to find the seam edges

NOT TRIED
  tuning any quad parameter           all are the paper's, unchanged
  searching the whole mix rather      would say where a record really is, independent
  than our window                     of our window
```

## What we thought

A third of DJ records cannot be found in the window cut for them, while the same measurement on Raveform
gives separations of 15 to 67. The working assumption all week was that the audio files were wrong,
because they were fetched by searching YouTube for "Artist - Title" and accepting the top result on its
name alone, for 2,911 of 3,158 tracks.

## What we tested, in order, and what each one cost

**The track files, against iTunes.** `verify_dj_tracks.py`, all 2,949 checkable tracks. It said 15% were
the wrong recording. Then the test itself failed its own control: on Raveform, where tracks are
downloaded by YouTube id and cannot be wrong, it still calls 11% of them wrong. It answers "is this the
commercial release", and a DJ plays the extended mix. **Ruled out as a gate.**

**The playback rate.** `align_record` from `align_windows.py`, envelope correlation with a Theil-Sen fit,
verified exact to five decimals on synthetic drift. Wired in behind `measure_seam --refine-rate` and run
against an identical plain run on the same 50 seams.

```
separation        mean    median          per record
plain            11.52      2.38          improved by >20%   11
refined           9.81      2.25          worse    by >20%   16
```

34 of 100 records had their rate moved, by a median of 0.22%, which is the size theory said would matter.
Nothing improved. **Ruled out.**

**A second, independent instrument.** The same run showed the envelope fit could not lock on to 66 of 100
records either. The envelope uses onset energy and the fingerprint uses spectral peaks, so they fail for
different reasons. Two instruments failing on the same records pointed at something shared between them.

## Why we stopped blaming the tracks

The shared thing is that both instruments assume the record plays at a speed we already know. Our
fingerprint hashes a **pair** of spectral peaks as `(freq1, freq2, frame_gap)`. Stretch the audio and the
frame gap changes, so the hash misses. `locate.py` works around it by resampling to nine candidate speeds
with a 0.25% refinement, and against the envelope fit that stored rate is still out by a median 0.28% and
up to 0.62%.

The literature had already measured this. Sonnleitner, Arzt and Widmer tested three fingerprinting
families on real DJ mixes at ISMIR 2016. Ours is the one they call Audfprint.

```
method                    accuracy   precision   specificity
Audfprint  (pairs, ours)     0.637       0.680         0.255
Panako     (triples)         0.360       0.432         0.349
Qfp        (quads)           0.876       0.959         0.927
```

Specificity is the ability to say a track is **not** there. Ours is 0.255, so three quarters of the time
a track is absent the method claims it anyway.

## What we built

`scripts/diag/quadfp.py`. Written from the DAFx-14 and ISMIR-16 papers, not from Panako's AGPL source, so
no licence enters the repo.

Hash **four** peaks instead of two. Translate and scale the constellation so A sits at (0,0) and B at
(1,1), then store only where C and D fall inside that unit square. Stretching moves all four peaks
together, so C and D keep their relative positions. The idea comes from blind astrometry, matching star
fields at unknown rotation and zoom.

```
audio          mono, 16 kHz
STFT           Hann 1024 (64 ms), hop 128 (8 ms), magnitude
peaks          max filter 65 bins x 91 frames, min filter 3 x 3 to reject silence
quad           A root, B furthest in time, C and D inside the rectangle
hash           (C'x, C'y, D'x, D'y) after normalising A to (0,0), B to (1,1)
reference      centre 4 s, width 2 s, 2 quads per root, 5 candidates    sparse
query          centre 4 s, width 7.9 s, 500 per root, 8 candidates      dense
matching       epsilon 0.012 range search in a 4-D KD-tree
speed          s_time = (Bq.x - Aq.x) / (Br.x - Ar.x), from the unnormalised points
```

The asymmetry is deliberate. A stretched query's peaks drift out of a narrow grouping window, so the
query needs a wider one and many more quads to still overlap the sparse reference set.

## The test, with a known answer

`scripts/diag/quadfp_test.py`. Cut a 30 s piece from 120 s into a real record, stretch it by a known
amount, ask both methods to find it back in the untouched record. A wrong record goes through the
identical path at every speed.

```
speed   QUAD votes  wrong   speed found   offset err   PAIR votes  wrong
1.000        1661     37        1.0000         0.00         18902     10
1.005         127      9        0.9947         0.00           303     11
1.010         215     10        0.9904         0.00            65     10
1.020         261      9        0.9800         0.00            31     11
1.030         149      8        0.9714         0.01            17     11
1.050         187     16        0.9519         0.00            16      9
0.970         193      9        1.0312         0.01            17     10
```

**The pair method dies beyond 1%.** At 2% it scores 31 against a control of 11, at 3% it scores 17
against 11. Below the floor of 40 in `locate.py` and indistinguishable from a record that is not there.
DJs routinely pitch further than this.

**The quad method holds to 5% up and 3% down**, at 127 to 261 votes against a control of 8 to 16. It
never confuses the wrong record at any speed.

**Position is exact.** The piece was cut at 120 s and it reports 120 s at every speed.

**Speed comes back to four decimals**, which we currently run a separate grid search for and still get
wrong by up to 0.62%.

## Two bugs the test caught, both in the position arithmetic

The matching and the speed were right throughout. Recorded because both would have passed silently
without a known answer to check against.

**Wrong frame of reference.** `(qa_x - ra_x * s_time)` should be `(ra_x - qa_x / s_time)`. Exact at speed
1.000 and wrong everywhere else, which is the worst kind of bug to find by eye.

**Binning before filtering.** One query quad matches many reference quads and most are noise, each with
its own wrong speed and therefore its own wrong offset. Binning all the offsets together let the noise
win whenever the true matches thinned out. At a 0.5% stretch the speed came back correct at 0.9951 while
the offset was 204 s out. The paper settles the scale first and then histograms the offsets among the
pairs that agreed on it.

## Run 1, the quad method against our own windows. Two populations

`scripts/diag/quadfp_invisible.py`. Each of the 30 invisible records against the window we cut for it,
with an unrelated record through the identical path as the control. Stopped at 17 of 30 once the split
was unambiguous.

```
locate votes high, quad finds something
  392    QUAD 214   ctrl 12    17x     speed 1.0437
 2520    QUAD  58   ctrl 12     5x     speed 0.9908
  185    QUAD  31   ctrl 15     2x
  148    QUAD  24   ctrl 12     2x

locate votes low, quad finds nothing
  13 records, locate 41 to 58, quad 5 to 22 against controls of 10 to 16
```

Every record locate found strongly also shows quad signal. Every record locate found weakly shows none,
now with a second instrument that is immune to the speed problem.

**The limit of this run.** It uses our own window as the query, so it inherits any window fault. It
cannot distinguish "the file is wrong" from "the window is in the wrong place".

## Run 2, Anas's clean-region test. The distinction the first run could not make

Between where A is listed as starting and where B is listed as starting, A plays. The second half of
that stretch is the cleanest part: the previous record has mixed out and B has not arrived. Cut that
and search it inside track A's own audio.

```
mix:   ...A starts........................B starts...
                          |<-- query -->|
```

It never touches our window, so it cannot inherit a bad one. It uses only the two listed start times,
which the tracklist got right and we did not compute. And it is one record alone, no second record's
peaks competing. `scripts/diag/clean_region.py`.

A positive control was run through the identical path: records the current method finds clearly, whose
presence is least in doubt. Without it a null result is unreadable, because it could mean the files are
wrong or it could mean the test does not work on real mix audio.

```
CONTROLS, records we know are present          INVISIBLE
  locate   QUAD  ctrl   ratio                    locate   QUAD  ctrl
   40389     87    13    6.7x                         45     18    34   below
   29298     24    12    2.0x                         54     12    12   equal
   23035     94     6   15.7x                         54     12    18   below
    4186     35    18    1.9x                         44     10    10   equal
    3842     38    24    1.6x                       2520     83    15    5.5x
   31803    102     8   12.8x
```

**All six controls beat their control record**, so the test works and a null result carries information.

**The four low-vote records score at or below a wrong record**, playing alone, for 90 seconds, with a
speed-invariant method. Those files are not the records in those mixes.

**Nick Acid, the 2,520-vote record, comes back at 5.5x**, 40 seconds before the clean region starts. That
record is in that mix and our window is cut in the wrong place for it.

## Where this leaves the DJ corpus

Two separate faults that had been treated as one.

```
wrong file           the low-vote records. Refetching is the fix
misplaced window     the high-vote records. The cut is the fix
```

Neither is about playback speed and neither is about iTunes. Both of those are closed above.

## Honest limits of the quad method

**It is far less sensitive than `locate`.** A record scoring 40,389 votes under the pair method gives 87
under this one. Another at 29,298 gives 24 against a control of 12. On real mix audio a true match
reached only 1.6x its control at the weakest, while an absent record sits at 1.0x. The gap between
present and absent is narrow.

**It is a second opinion, not a replacement.** Use it where the primary method fails, which is how it was
used here.

**Nothing has been tuned.** Every parameter is the paper's, unchanged. Their reported accuracy is 0.876
and our margins are thin. That gap could be the implementation, our 130 kbps audio, or parameters that do
not suit this material. Those have not been separated.

## Run 3, both groups at full size. The verdict

30 known-present records and 28 invisible ones, identical code, 90 second queries, a wrong record
through the same path on every row.

```
                 n    quad votes   control   ratio median   p10
KNOWN PRESENT   30            47        12           4.17   1.00
INVISIBLE       28            12        12           1.00   0.00
```

Split the invisible group by how strongly `locate` found each record in the whole mix:

```
locate >= 100     7 records   ratio median 3.9   3 clear a strict floor
locate <  100    21 records   ratio median 1.0   0 clear it, none above 2.2x
```

**Not one of the 21 low-vote records shows a real match in its own clean stretch.** The three that do
clear the floor are all high-locate records: Chelonis R. Jones at 43.7x, Manoo at 6.7x, Nick Acid at 5.5x.

**The verdict.** The 21 low-vote files are not the records that play in those mixes. The 7 high-vote ones
are the right files in the wrong window.

**The limit of this test, stated plainly.** With the floor at twice the loudest wrong record, only 11 of
30 known-present records clear it, and the known-present p10 ratio is 1.00. So the test gives a confident
yes and a soft no. A record that scores well is definitely there. A record that scores nothing is
probably absent, and no single track should be condemned on one reading.

What is safe is the group conclusion. 21 of 21 low-vote records failed the same test that clears 11 of 30
records we know are present. That does not happen by chance.

## Final verdict on the quad fingerprint

**Keep it, as a second opinion. Do not make it the primary.**

What it is for. When `locate` says a record is weakly present and we cannot tell whether that is a bad
file or a bad measurement, this answers it, because it fails for different reasons. That is the whole
job it did here and it did it.

```
WHERE IT IS BETTER
  speed                    survives 5% up and 3% down. The pair method dies past 1%,
                           scoring 31 against a control of 11 at 2%
  speed as an output       returns it to four decimals, free, from the match itself.
                           `locate` runs a separate 0.25% grid search and is out by up to 0.62%
  position                 exact. 0.00 s error against a known cut at every speed tested
  independence             different failure modes from the pair method, which is what makes it
                           usable as a check on it

WHERE IT IS WORSE
  sensitivity              a record scoring 40,389 pair votes gives 87 quad votes. Another at
                           29,298 gives 24 against a control of 12
  margin on real audio     a true match reached only 1.6x its control at the weakest, while an
                           absent record sits at 1.0x
  soft negatives           at a strict floor only 11 of 30 known-present records clear it. A
                           confident yes, a soft no
  cost                     about 90 s per record against seconds for the pair method
```

**How to read its output.** A high score means the record is there and you can trust it. A zero score
means probably absent, and never condemn a single record on one reading. Use it on groups.

**What has not been done to it.** No tuning of any kind. Every parameter is the paper's. Their reported
accuracy on DJ mixes is 0.876 and our margins are thinner than that suggests, and the cause has not been
separated between the implementation, our 130 kbps audio, and parameters that may not suit this material.

**What it does not replace.** `locate` stays the primary. It was right about presence all along, on every
record both methods judged. The thing that was wrong was our use of `mix_t`, which is where a record's
0:00 would fall rather than where it starts.

## What to do with it

```
21 low-vote records    refetch. The file is not the record in the mix
 7 high-vote records   recut. The file is right and the window is wrong
```

Both lists are in `data/interim/clean_region_all.csv` with the votes, the control and the speed per row.

The 21 are packaged for a manual Shazam check in `data/external/ear_test/shazam_wrong_files/`, each with
the downloaded file, 45 s of the mix where it should be playing, and a MARKS.csv sorted weakest first.
Numbers 13, 14 and 21 scored zero against zero, so the fingerprint read nothing either way, which usually
means quiet or sparse audio rather than a mismatch.

Neither repair is about playback speed and neither is about iTunes. Both of those are closed above.

# FEATURE EXTRACTION — what we found, and the open question

## What was tested

Two tests on 2026-09-18, both on seam-corpus tracks, in `data/external/ear_test/d4_check/` and
`d4_pairs/`.

**Does a full track represent the same thing as its 30 s preview.** 184 tracks with both embedded,
1,280 dimensions, and the crucial split is by whether the download actually matched its preview:

```
                    n     cosine to own preview    own preview is nearest of 21
right recording   164                     0.962                     146   89%
wrong recording    20                     0.720                       6   30%

unrelated tracks, for scale             median 0.549, p90 0.806      chance is 5%
```

**Does it change how pairs rank.** 100 real pairs, the same pairs scored both ways:

```
score                  preview    full    full minus preview
raw_embedding_cosine     0.701   0.718              +0.017
gbm_edge                 0.597   0.628              +0.031
loudness_closeness       0.515   0.541              +0.025
onset_closeness          0.515   0.538              +0.023
model_a_cosine           0.600   0.614              +0.014
compatibility            0.631   0.634              +0.003
tempo_closeness          0.673   0.666              -0.007
energy_closeness         0.569   0.541              -0.028
key_closeness            0.614   0.582              -0.032
```

Also measured on 60 tracks: bpm identical on 45 of 60, key identical on 35 of 60, loudness difference
a median of −0.1 LUFS.

## What that settles

**Full audio and previews sit in the same space.** 0.962 against 0.549 for unrelated tracks is not a
close call. A model trained on preview features can read full-track features.

**Neither is better for ranking pairs.** The differences are small and, more tellingly, they point both
ways across features that all derive from the same audio. If full audio carried real extra information
the audio-derived scores would move together. They do not. That is stronger evidence than the sample
size allows any single number to be.

This matters because the old dataset's 28,460 tracks will never have full audio. It does **not** mean we
should use previews where full audio exists.

## Do not use iTunes previews for Raveform

**About 11% of them are a different version of the record that is actually in the mix.**

The 20 "wrong recordings" in that table are seam-corpus tracks, and Raveform tracks are downloaded by
YouTube id with no search, so the fetcher cannot have fetched the wrong record. Something in each of
those pairs is a different version, and it is not the full track.

The full track is right, and the mix proves it. On 60 seams, one per mix, both records were found inside
their own window with a median of 900 votes against a wrong-record control of 13, and Anas ear-checked
placement on 45 files with 40 right. The record on disk is demonstrably the one playing in the mix.
`seam_previews_fetch.py` looks the same track up on iTunes by artist and title, and iTunes has no idea
which of the extended mix, the radio edit or a remaster the DJ played.

So for the 6,579 seam tracks the preview is not a cheaper substitute for the full audio. It is a lossier
copy that is the wrong record 11% of the time, and we already hold 41 GB of the right ones.

Two consequences.

**For Raveform, drop the previews.** Extract from the full audio. The 5,989 files in
`data/interim/seam_previews/audio` stop being needed.

**For the DJ corpus, this is a measurement of the instrument.** On a corpus where the tracks are known
correct, the iTunes check still calls 11% of them wrong. That is its false-alarm rate, and it agrees with
the 8% measured independently from the locate side. The DJ corpus's 15% "wrong download" figure therefore
carries roughly 11 points of instrument error. The real wrong-download rate is much lower than 15%.

## Decision

**Extract the 6,579 seam tracks from the full audio.** 6,917 files, 41 GB, already on the VM, downloaded
by id. The 5,989 iTunes seam previews stop being needed.

## The open question — which features, over which span

There are two different needs and they have been treated as one.

**Track-level features, for choosing which record follows which.** Whole-track is right. That is what the
old dataset holds and what Models A, B and the GBM were trained on. Nothing changes except the source.

**Section-level features, at the seam.** A transition model does not need the average of a seven-minute
record. It needs what the outgoing record was doing in its last bars and what the incoming one was doing
in its first. A record with a quiet intro and a hard drop has a whole-track energy that describes neither
end of it. Nobody has built these.

**Hypothesis.** Whole-track features are too coarse to predict a transition, and the section actually
played is what carries the signal. The alignment already tells us which part of each record sits in the
window, so the played span is addressable for the first time.

**Why it is open.** It has never been measured. We do not know whether section features beat whole-track
features at predicting anything, and we do not know what span to take. Candidates: a fixed 30 s either
side of the seam, the whole overlap, the last and first phrase, or the cue-out and cue-in sections from
the segmenter.

**What would test it.** Take the seams where the measurement is solid, compute both feature sets, and see
which better predicts a held-out measured property of the same seam, for example overlap length or
whether the bass swapped. If whole-track does as well, the simpler thing wins and section features are
not worth building.

**The constraint.** Section features depend on the alignment being right. That is settled for Raveform and
is exactly what is unsettled for the DJ corpus, so this is a Raveform job first.

## Status

Extraction itself is not blocked. The audio, the discogs-effnet model and the extractors are on the VM.
GPU quota is 0, so it runs on CPU. What is undecided is the span, not the ability to run it.

# HOW THRESHOLDS WORK NOW

Every floor used by the measurement scripts is derived from controls run on that seam, in that band, on
that audio. `scripts/diag/floors.py`.

```
MIN_CONTROLS = 3     one control is a single draw of the noise
MARGIN       = 2.0   a real match must beat the loudest wrong record by this factor
ABSOLUTE_MIN = 4
```

Every row carries `control_n`, `control_max`, `floor` and `separation`, so the evidence for a threshold
travels with the number it produced.

This exists because the old constants were picked, not fitted. On Fred's 48 kbps audio three control
records by a different DJ scored 43, 47 and 58 while `MIN_VOTES` was 40, so the floor sat under the noise
and everything between 30 and 80 passed. On Raveform's audio the same control sits at 13.

**The constants inside `locate.py` and `fetch/track.py` have not been changed.** Locate still uses a flat
40 and the fetcher still uses a name score of 2.6.

---

# WHICH SCRIPT TO USE

| job | use | do not use |
|---|---|---|
| cut a DJ window | `recut_windows_bounded.py` | `recut_windows.py`, `seam_window_check.py` |
| cut a Raveform window | the pipeline, `manifest/raveform.py` | — |
| check a window holds the transition | `audit_recut.py`, `audit_raveform.py` | `audit_dj_seams.py` |
| measure bands, bass, presence | `measure_seam.py` | `fp_bands.py::set_test`, `fp_bass.py::main`, `fp_bands_dj.py`, `fp_bands_keylock.py` |
| refine playback rate in a window | `align_windows.py::align_record`, via `measure_seam --refine-rate` | `seam_rate_envelope.py` for window work |
| label a transition | `transition_labels.py` | — |
| check a DJ track is the right recording | `verify_dj_tracks.py` | the fetcher's own name-score path |
| fetch previews | `dj_previews_fetch.py`, `seam_previews_fetch.py` | — |
| match a Fred segment to a show | `fred_match2.py`, by play order | `fred_match_segments.py`, by votes, proven not to work |

**Libraries, still live, imported by the runners above:** `fp_presence.py`, `fp_presence_batch2.py`,
`fp_presence_batch3.py`, `fp_bands.py`, `fp_bass.py`, `fp_bass_timeline.py`, `align_windows.py`.

**Dead, do not revive:** `djdata/seam/gains.py`, `seam_rebuild.py`, `seam_reconstruct.py`. The gain fit
failed its wrong-record control and the rebuild metric scored a wrong record as well as a right one in
6 of 12.

**Built and does not work:** `refetch_wrong_tracks.py`.

---

# RUNNING IT

VM `aidj-data`, c6i.4xlarge, 16 vCPU, 30 GB, Elastic IP 98.91.107.37 so the `aidj` alias survives a stop.

```
cutting        up to 16 workers   ffmpeg stream copy, IO bound
measurement    12 workers         peak worker 1,176 MB, 8 GB of 30 used, load 9.9
```

12 only holds because `measure_seam` fingerprints each record over the stretch that can appear in the
window rather than its whole length. Before that a worker held gigabytes of hash tables and ten workers
stalled the box.

---

# FRED, separate job

Parked until the rest of the DJ data works. He is the hardest case and needs his own attention.

- Four sources. **Three of them, including the 108.53 h marathon, are 48.7 kbps HE-AAC**, the lowest
  YouTube serves. The normal fetch gets about 130 kbps opus.
- 42 segments cut, every cut within 0.046 s. **18 solo, 24 b2b. Only the 18 solo go into profiling.**
- **Vancouver is proved.** Four of his own records found in that segment in exact play order, 406 to
  3,154 votes. `c55b5a562e` is `fred_19_vancouver_fred.m4a`. The other ten cities are unproved.
- Vote count cannot separate his shows, because the tour repeats its own USB. **Play order can.**
- Commercial records never match, because he plays his own edits. Four of ten tracks per show is the
  realistic yield, and his wrong-download rate is 38%.
- **Zero seams exist for his eleven tour mixes.** All eleven are `pending` and the segments are joined to
  nothing in the database.
