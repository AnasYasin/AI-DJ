# AI-DJ, how the system works today (written 2026-10-13, planner section rewritten 2026-10-15)

This page is the system as it runs now. Every stage has the same three parts, in this order: what was planned at the
start of the project, what is there right now, and what is planned next. Where a stage is a rule, it says rule; where it
is a model, it says what the model is, what it was trained on and what it scored. The ten-step plan is `PLAN.md`; the
gotchas and the constants are in `CLAUDE.md`; the data is in `DATASET_STATE.md`.

## The flow in one picture

```
prompt (today: command-line flags)
  -> pool        the catalog filtered to the genre and the tempo range
  -> planner     beam search over the pool, scored by Model B, the GBM, energy fit and compatibility
  -> fetcher     the full recording of each picked record from YouTube, verified against its 30-second preview
  -> analysis    per record at render time: beats, downbeats, sections, exact tempo, stretch, loudness
  -> per seam    the transition type, its length, the kick alignment, which record leads; all rules
  -> render      EQ, filter and fader per record, -14 LUFS, a true-peak limiter, a flac
```

The example used throughout is the command in `CLAUDE.md`:

```
python -m scripts.make_mix --genre techno --bpm 134 142 --curve peak --n 6 --minutes 15 \
    --min-energy-pct 65 --compat-weight 5 --out mix.flac
```

## 1. The prompt

### Planned at the start

Phase 10: a natural-language intent parser, one Claude call, behind a FastAPI service. A prompt like "fast driving techno
for a 2 am peak" becomes the fields below. Never built.

### What is there now

Ten flags on the command line.

| flag | example | what it means |
|---|---|---|
| `--genre` | techno | one of techno, tech house, house, trance, drum and bass, melodic house. A record's genre is the genre of the 1001tracklists set it was first listed in; 882 records sit in sets of two genres and carry one (`DATASET_STATE.md`, TO DO of 2026-10-09). |
| `--bpm` | 134 142 | a range in beats per minute. Only records whose catalog BPM lies inside it enter the pool. The range is kept about 6 % wide because the mixer stretches every record to one tempo and refuses a stretch over 8 %. |
| `--curve` | peak | the energy shape of the set, a formula over the slot position 0 to 1: build rises straight, peak rises to 0.85 of the set and falls, wave is a sine, chill stays at 0.15 to 0.25, arc rises and falls. Its value at a slot is the slot's energy target, 0 to 1. |
| `--n` | 6 | records in the set |
| `--minutes` | 15 | the last record plays on until the mix reaches this length, never past its own final bar |
| `--min-energy-pct` | 65 | drop the quietest 65 % of the pool, a percentile taken after the genre and BPM filters (since 2026-10-13; before that it was taken over the whole table). |
| `--audio` | full | only records whose full audio is held (12,153 of 37,612); the fetcher then copies the file and searches nothing. Default any. |
| `--dj` | Solomun | only records that DJ has played, from the plays table (Solomun 2,245 with features). The first piece of a DJ profile. Default none. |
| `--compat-weight` | 5 | how much the planner prefers pairs the mixer can overlap for long. Default 0, a leftover; 5 is right until Step 2 sets it per slot. |
| `--out` | mix.flac | flac or wav lossless, mp3 at 320k |

Why the energy floor matters. The curve is relative: "top of the pool" is the pool's own 85th percentile and "bottom" its 15th, so a peak curve
over a quiet pool gives a quiet set. The floor is the one lever that makes a set loud in absolute terms. BPM is not one;
techno energy medians sit at 0.32 in every band from 128 to 150.

Why compat-weight matters. The mixer caps every overlap by a pair-compatibility score it computes itself (section 6). The
planner does not use that score unless told to, so at 0 it can order records that look right to Model B and the GBM and
then render as a string of 16-second cuts. At 5 the planner adds the same score to each candidate and picks records the
mixer will blend for long. The one cost is sameness: records in the same key at the same energy make a flat set.

### Planned next

Step 9: one LLM call reads the prompt and writes the same fields, plus three new ones. A DJ profile name, so the planner
takes that DJ's priors. Named records to pin into slots, fuzzy-matched to the catalog, or brought in by name through an
iTunes preview lookup, then features, then the fetcher. A surprise knob, 0 to 1, that lets lower-scored candidates
through the beam. Everything after the parser stays deterministic.

## 2. The pool

### Planned at the start

A catalog of every record listed in the scraped 1001tracklists sets, known only through 30-second iTunes previews, with
discogs-effnet embeddings and librosa features measured on the preview. discogs-effnet was frozen as the embedding after
a 200-record signal-ratio test against MERT (1.04 times worse), MuQ (1.175) and CLAP. DeepRhythm for BPM, essentia edma
for key.

### What is there now

`data/dataset/<date>/tracks.parquet`, 37,612 records from the three sources (`DATA_ARCHITECTURE.md`): 25,459 old
records measured on the 30 s preview (`audio = preview`) and 12,153 Raveform and DJ-data records measured on the full
track (`audio = full`; a record in both keeps the full row). Per record: a 1,280-dimensional discogs-effnet embedding, BPM
with its source, key normalised to sharps, energy, loudness in LUFS, onset strength (stored raw, compared normalised, see
`CLAUDE.md`), 13 MFCCs, play and DJ counts, release year and artist listeners where the enrichment passes found them. The
pool for the example, counted on the old rows before the new ones joined (1,506 techno records in range after):

| filter | records |
|---|---|
| genre techno | 5,060 |
| BPM 134 to 142 | 1,461 |
| above the energy floor | 840 (catalog-wide percentile, as the code does it); 512 if the pool's own |

A record named "ID" or "Unknown" is dropped (238 catalog-wide) because it can never be fetched.

Beside the catalog, since October: 12,211 records with full audio from the two seam corpora, embedded over the whole
track with one vector per second kept as patches, tempo measured by the mixer's own kick autocorrelation and checked
against Raveform's beat annotations, key on the full track (`EMBEDDINGS.md`). On 150 old records fetched in full, the
preview BPM was within 2 % on 81 %, the key right on 69 %, loudness and energy off by half the spread between records.
These rows are in the pool since 2026-10-13.

### Planned next

Any threshold derived from features is measured on full-track rows. The old record's genre is still the first set it
was listed in (882 records sit in sets of two genres), and the Raveform tag mapping has not been checked on a sample.

## 3. The planner, which record comes next

### Planned at the start

Three learned parts and a fourth that was dropped. Model A, a contrastive encoder that makes a "mixability" embedding
from real adjacency. Model B, a sequence model over the set so far that predicts the next record. A gradient-boosted edge
scorer on pair features. A transition classifier trained on rule labels (dropped, section 6). A beam search joins them
with energy fit and hard vetoes on BPM, key and repeats.

### What is there now

**The pair model** (`src/models/pair_model.py`, `models/pair_model.pt`, trained 2026-10-14). One model replaces Model A,
Model B and the GBM edge scorer, which were deleted on 2026-10-14 with the key rules (their description is in
`docs/notes-archive/system_overview_planner_2026-10-13.md`). It scores how well candidate record B follows record A,
given the set so far.

- Record encoder, one per record: fully connected 1,305 to 512, batch normalisation, ReLU, dropout 0.5, fully connected
  512 to 128. The input is the effnet embedding (1,280, unit length, times 8), eight z-scored scalars (BPM, energy mean,
  loudness, onset, spectral centroid, energy spread, release year, log listeners), 13 z-scored MFCCs and the key as
  Camelot sine, cosine, mode and an unknown flag.
- Context vector: the mean of the encoded records of the set so far, the last 16.
- Pair head: fully connected 799 to 256, ReLU, dropout 0.5, fully connected 256 to 1, over A, B, their product, their
  absolute difference, 31 z-scored gap features, the context vector and its product with B.
- Training: softmax ranking loss over the record the DJ played next and 16 random feasible candidates (same genre, BPM
  within 5 %), AdamW, up to 20 epochs with early stopping on 41 development DJs, 5 seeds averaged. Data is every
  source's plays from `data/dataset/2026-10-14/`, split by whole DJ.
- On 58 validation DJs never seen in training, 6,305 pairs, 100 feasible candidates each: AUC 0.876, the true next record
  in the top 10 for 0.649 of pairs, median rank 5. The raw effnet cosine gives 0.310 in the top 10, the old Model A
  retrained 0.285, a GBM on rich pair features 0.572. Every model tried is in `PAIRING_EXPERIMENTS.md`.
- The model is only valid inside the feasible set it was trained on. Given candidates of any BPM it puts far-BPM records
  first (4.6 % of its top 10 within 5 % BPM, against 46.8 % of the candidates), so the BPM veto is part of the model.

**The search** (`plan_mix` in `src/models/predict_model.py`). Beam search, width 8. Seeds are the 8 records nearest the
first slot's energy target among the slower half of the pool by BPM, because 71 % of real sets open in their slower BPM
half. No model chooses the opening record. Hard vetoes: absolute log BPM ratio over 0.05, record already used, artist
already used. There is no key rule. Real sets clash on 58 % of joins and put two clashes in a row in 84 to 90 % of
mixes, and the pair model sees the Camelot distance itself. The score:

```
1.0 x z-score of the pair model logit, standardised within the candidate set
+ 0.7 x energy fit to the slot's target
+ compat_weight x pair compatibility (the mixer's own measure, section 6; 0 by default)
```

The weights 1.0 and 0.7 were never tuned. The beam dedupes on (set of records, last record). Slot repair after a failed
fetch uses the same score, and at the first slot asks the model how well the second record follows the candidate.

**Heard on 2026-10-15.** Seven chill sets of six records from held audio, one per genre
(`data/external/ear_test/sets_2026-10-15/`). Anas: track selection is fine. Every set starts too low on energy. The
chill curve opens at about the pool's 20th energy percentile and stays there; real sets open at the 38th to 53rd
percentile and sit at the 50th to 56th. The transitions and cue points need work (section 6).

### Planned next

The chill curve's range moved up toward where real sets open, waiting on Anas's numbers. A club or radio setting on the
request, measured on the mix titles (`TODO.md` item 2). Per-DJ profiles as planner input, after the transition tables
exist (`TODO.md` item 4). Profiles carry the DJ's whole distribution, never a median (Anas, 2026-10-12).

## 4. The fetcher

### Planned at the start

Nothing. The plan had previews only; full audio came in with the mixer.

### What is there now

`src/data/track_fetcher.py`. A plan entry says where its record comes from (`source`, `source_ref`, `audio`), and the
fetcher tries, in order: the held file on this laptop or on S3 for a record with `audio = full` (copied, nothing
searched or verified); the known YouTube link for a Raveform record (downloaded by id, no search, no fingerprint, the
id is the recording); and last a YouTube search by name, yt-dlp, best audio with no re-encode, native Opus at 130 to
145 kbps or AAC 128k, through three gates:

1. Duration 240 to 900 seconds on the search metadata, and again on the decoded file, because metadata lies.
2. Title words rejected: radio edit, snippet, sped up, live set and the like.
3. Identity: a constellation fingerprint of the download against the record's own 30-second preview. The right recording
   scores 457 to 14,466 votes, a wrong one 21 or fewer; the threshold is 50. Chroma and mel correlation both failed here.

A record fetched by link or by search is then registered as held full audio (`src/data/register_track.py`): the file
goes under `data/djdata/old/tracks/`, its full-track features, effnet mean and measured BPM are appended to the old
source's `track_features.parquet`, and the next export marks it `audio = full`, so no later mix looks for it again.
About 15 s per record.

If every candidate for a slot fails, slot repair (`repair_candidates`, Anas's design, 8 September) ranks replacements
that fit both neighbours, BPM within 5 %, no repeated record or artist, scored by Model B, the GBM, energy fit and
compatibility, from the same pool as the plan (its `audio` and `dj`), fetches up to five, and only if none verifies
replans the whole set. yt-dlp must stay current or every download is an HTTP 403.

### Planned next

The 240 s duration gate goes once the play length is predicted per record, because a 3.5-minute record is a normal
choice in drum and bass. The rest of the old records get fetched and registered the same way once the models are judged
on the new data (Anas, 2026-10-13). The YouTube source is the ceiling on quality and nothing in the code lowers it.

## 5. Analysis at render

### Planned at the start

Phase 7: a structural segmenter on full tracks, and tempo read from the beat grid.

### What is there now

Per record, in `src/audio/audio_mixer.py` with the segmenter in `src/data/audio_segmenter.py`:

- **Segmenter.** Cached on path, size and modified time (20 seconds cold, 0 warm). Beats, downbeats from Beat This!
  (CPU, 20 to 40 seconds per record), sections from novelty peaks. The raw downbeats fire on every beat in some passages,
  so `regular_bars()` anchors on the longest run of clean one-bar gaps and extends a regular four-beat grid;
  `downbeat_confidence` is the share of tracker downbeats on that grid, 0.50 to 0.98 on real records. Section boundaries
  snap to half-phrase lines (four bars) of that grid.
- **Tempo** is measured from the audio, not read from the beat grid, because the grid rounds to 23.2 ms frames and reads
  136.05 for records at 134, 135 and 137, which drifted half a beat inside one overlap. `_measure_tempo` runs a
  kick-envelope autocorrelation at the bar lag over the middle 180 seconds with a parabolic peak.
- **Stretch** to the set's median BPM with rubberband R3, which keeps pitch; 18.6 seconds per record and 80 % of render
  time; warned above 4 %, refused above 8 %. Then `_calibrate_grid_phase` shifts the grid onto the real kicks.
- **Play window** (`_choose_window`) from the genre median, 4.08 minutes for techno, measured over 24,831 real changes in
  1,665 sets (afro 5.17, melodic 4.63, trance 4.42, tech house 4.00, dnb 2.58). The individual set explains 42 % of the
  variance, the DJ 22 %, the genre 12.5 %, record features 0.2 %, so no energy term. The window flexes 16 bars either way
  to land its cue-out on a section line, cue-out bonus 0.25 over cue-in 0.10; the slot's regime (section 6) can prefer a
  quiet tail. `max_tail` reserves the median transition, not the maximum.
- **Loudness** matched to the set median, capped at 6 dB.

### Planned next

Step 2, roles. The curve gives each slot climb, hold, peak, release, first or last; the role sets the play-length range
and the transition-length range from the measured numbers instead of one genre average, and a cue-out may not begin a
breakdown unless the role asks (the Farrago failure). Step 3 beside it: the finished mix is measured per record for
energy and loudness against the plan and flagged when a slot misses.

## 6. Per seam, how to mix

### Planned at the start

Phase 4: six transition classes (slam, rise, fade, melt, wave, blend) defined by rules on the two records' features, a
classifier trained on those labels, and the recipes applied at render. The classifier was dropped; the rules apply
directly.

### What is there now

Four decisions, all rules, constants in `src/data/transition_labeler.py` and imported by the mixer, never copied.

1. **Type.** `measured_transition`, six rules in order on the two records' measured BPM ratio, energy change, Camelot
   distance, loudness change and onset: slam when BPM within 3 % and energy jumps over 0.15, or the keys clash and energy
   rises; rise when energy rises over 0.08, BPM within 7 % and keys compatible; fade when energy falls over 0.08; melt
   when BPM tight, keys perfect, energy and loudness nearly equal; wave when BPM tight and both records have strong
   onsets (normalised onset over 0.35); blend otherwise. Then `gate_transition` demotes to the set's intent: chill turns
   slam to melt and rise to blend, build turns slam to rise before the half. A rise becomes a drop when the incoming
   record has a usable drop at least a phrase past its cue-in. The slot's `energy_target01` also picks a regime: under
   0.35 never lands on a drop and prefers a quiet tail; over 0.65 ends the overlap on the incoming drop with the bass
   swap in the last bar.
2. **Length.** `pair_compatibility`, 0.35 key plus 0.30 energy difference plus 0.20 BPM difference plus 0.15 loudness
   difference, calibrated on 43,073 real pairs, AUC 0.643, into ceilings of 16, 32, 60 or 90 seconds at its p40, p75 and
   p90, taken in whole phrases. Genre defaults in bars: slam 4, rise 32, fade 32, melt 64, wave 16, blend 16, drop 16;
   tech house, melodic and afro house use blend 32, rise 24, fade 16; drum and bass halves everything. Stretchable types
   may run to twice their default; slam and drop never stretch.
3. **Alignment.** `measure_seam_offset` cross-correlates the two kick envelopes within half a beat with a parabolic
   peak, shifts and re-measures up to four passes; seams start 180 to 215 ms out and end under 0.3 ms. `seam_decision`
   compares the kick reading with the consensus of bass, mids, highs and full band: agree within 60 ms, the kick;
   disagree by half a beat, the kick (off-beat layers); disagree otherwise, the consensus; no kick, the consensus.
   Settled by two A/B listens. `SEAM_METHOD = "xcorr"` stays; a "grid" method measured better and sounded messier.
4. **Lead.** `LEAD`: the outgoing record in front, the incoming enters at -5 to -8 dB with mids cut, the lead swaps over
   a fixed bar count so the ambiguous zone does not grow with the overlap (163 s to 35 s across all types).

**The ground truth that exists for it.** On 9 September the roadmap put Step 5 first, a seam pipeline that measures
real transitions. It is done. Raveform (KAIST's dataset of mix audio with YouTube ids and an alignment per record) gives
5,060 measured seams; the DJ corpus (seven DJs' 1001 sets, records located by fingerprint with a tempo-refined,
key-lock-aware placement since 9 October) gives 2,617. Per seam the tables hold the overlap in bars, where the bass
swapped, when each band of each record arrived and left as a fraction of the overlap, bass-both and bass-cut bars, loop
and loop steps, the playback speed and mode, and each record's full features and patch vectors. Since 9 October a seam
carries a signature the reader can tell apart (cut, centre swap, late swap, end swap, tension, loop) and the nearest
mixer recipe with a distance; fade, melt and wave, and rise and drop, are told apart by the nearest recipe only. The
signatures have not been checked by ear; 24 clips wait for that.

### Planned next

**Measured 2026-10-15, before building it** (`TRANSITION_EXPERIMENTS.md`). From the two records' whole-track features
the transition type is not predictable on unseen DJs: 0.41 accuracy, the same as always guessing the most common type,
and the mixer's rules match the DJ's real choice on 0.25 of seams. With the DJ known, the DJ's own most common type
gives 0.51. The state of both records at the cue points (band levels, build-up, repetition, mids) adds nothing in
techno and a weak signal in house. So the first version is a per-DJ and per-genre table of type shares and recipe
distributions, with the mixer's rules as the veto. The step as written before that measurement:

Step 6, the transition model. Input per seam: the two records' edge patch vectors pooled over the last bars of the
outgoing and the first bars of the incoming, their features, DJ and genre. Output: the signature, the overlap in bars,
the bass swap point, the band order, later the speed change. Trained on the seam tables with `measured == 1` and the
band separations as the filter, split by set and DJ. Judged by bar error and signature agreement on held-out seams
against the rule table, then a blind listen against the seven rule types on the same pairs. The rules stay as the
floor. The mixer gains a per-seam tempo ramp within the stretch limits. Then Step 7: a loop at the cue-out, or both
basses off for a phrase and back on the drop, as one designed move, decided from role and audio, reported, rare.

## 7. The render

### Planned at the start

A mono render with a scalar loudness guard and a one-pass EQ. Rendering in mono discarded the side channel of every
record, 10 to 17 dB under the mid, and was the largest quality defect.

### What is there now

Stereo throughout, audio as (samples, channels) from decode to write. Each record runs EQ, then filter, then fader. The
low band is a gain envelope, so the bass swap is a one-bar EQ kill on a half-phrase line (`_snap_low_swap`). Rise, wave
and drop carry `A_sweep` and `B_sweep` for `_swept_filter`, a real moving corner that crossfades between fixed zero-phase
filters on a third-octave grid, verified tracking 4 kHz to 25 Hz. EQ uses `sosfiltfilt`, because a one-pass butterworth
put the low band 2.38 ms behind the highs. The gain across the overlap (`_overlap_gain_ride`) is anchored at both ends on
the louder of the last or first real bars, excluding drop-outs more than 6 dB under the loudest reference bar, linear in
dB between, clamped to 9 dB; the interior is left alone so an incoming drop still hits. Then the whole mix to -14 LUFS
and `_true_peak_limiter`, four times oversampled, 5 ms lookahead, 100 ms release, ceiling -1 dBTP; on a ten-minute render
0.75 dB of reduction on 0.18 % of samples. A 15.6-minute mix renders in 192 seconds with everything local. Every piece
was rebuilt 6 to 9 September and checked by Anas's ear.

### Planned next

Step 10, gimmicks: spinback, echo out, stutter, a cappella drop, bass drop-out, as options on any recipe, role-gated.
Last.

## The data, and why the plan changed

At the start the only data was text: 2,834 tracklists from 1001tracklists, 29,393 plays, and the previews. That is
enough to learn which record follows which and says nothing about how a DJ mixes. So on 9 September the roadmap put
Step 5 first, the seam pipeline. 10 September brought Raveform, with mix audio, YouTube ids and an alignment per record.
25 September the djdata package, eight stages from mix audio to labelled seams. 29 and 30 September both corpora ran.
7 October the tempos were checked against Raveform's beats and 1,056 seams relabelled, and every record got full-track
features and embeddings. 8 October the DJ-corpus speeds failed a beat-match check; key lock was the cause; fixed 9
October and both corpora re-run, landing 11 October with a beat-match median of 0.05 %. The nine type names are gone,
replaced by measured moves plus a signature. Every seam row now carries what Step 6 trains on. Nothing in the chooser or
the render changed for it.

## Agreed order of the next work (2026-10-12)

Step 6 first, Step 4 beside it when a retrain comes up, then Steps 2 and 3, then 7, 8, 9, 10. The DJ data report
(`DJ_DATA_REPORT.md`) is the analysis that precedes the training decisions.
