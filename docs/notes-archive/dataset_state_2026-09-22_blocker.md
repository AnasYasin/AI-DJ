# DATASET_STATE, the 2026-09-22 blocker section, moved whole to the archive on 2026-09-27

Superseded on 2026-09-24: the mix files were the fault, not the instrument. Kept whole because it records the quad fingerprint work, its two bugs, its three runs and its final verdict, so none of it is tried again.

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

