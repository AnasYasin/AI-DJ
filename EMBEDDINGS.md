# Track features and embeddings for the seam corpora (plan, 2026-10-07)

The two seam corpora (Raveform, 5,060 measured seams, and the DJ corpus, 2,459) have full audio for
12,211 tracks on the VM and S3, and no track features or embeddings yet. The old dataset has features
and embeddings for 28,460 tracks, all from 30 s iTunes previews, and Models A, B and the GBM were
trained on those. This file is the plan for computing the new features, checking what already exists,
and deciding how the old and new data fit together. Numbers land here as each phase runs.

## What is settled before this plan

- Encoder. discogs-effnet, frozen, 1,280 numbers per 2.05 s patch of 16 kHz audio. It beat MERT, MuQ and
  CLAP on a 200-track test (`CLAUDE.md`, Models). We train only a small head on top (Model A). An
  encoder trained from scratch, or effnet fine-tuned end to end, is not on the table at 15,000 pairs.
- Key. essentia `edma` through `normalise_key()`. Root right about 3 of 4, mode unreliable in every
  detector tested. Planner uses key as a soft term, mixer ignores it. Not reopened without a new ear
  result (`PLAN.md`, "Key, settled").
- Tempo instrument. `audio_mixer._measure_tempo`, kick autocorrelation at the bar lag with a parabolic
  peak, checked by ear on real renders. DeepRhythm is a hint only.
- Loudness, energy, onset. pyloudnorm integrated LUFS on mono doubled at 22,050; librosa RMS mean; raw
  onset mean, normalised by `normalise_onset()` at the point of use. Same conventions as
  `audio_mixer._analyse_body`.
- A positive pair is two records adjacent in one mix, one pair per occurrence. False negatives (two
  records that would mix well but never met) are accepted.
- Previews are not a substitute for full audio where full audio exists: 11 % of Raveform's previews are a
  different version of the record (`DATASET_STATE.md`, "Do not use iTunes previews for Raveform").

## Phase 1. What exists, and whether it is right

**Tempo.** `data/djdata/<corpus>/out/tempos.csv`: 6,905 Raveform and 2,959 DJ records, one BPM each,
from `djdata/seam/tempo.py`. The refining step is the mixer's `_measure_tempo`. The hint is librosa's
tempo over the middle 60 s, where the mixer uses DeepRhythm. The refinement searches within 6 % of the
hint, so a hint at the wrong octave gives a wrong tempo and the code cannot tell. The values were never
heard. On the DJ development sample 40 of 48 agreed with the catalog within 2 %.

Check: Raveform ships beat annotations per track in `data/raw/raveform/raveform/beats/`. The median
beat interval gives an independent BPM. Compare with `tempos.csv` on all 6,905: share within 2 %, share
at exactly half or double, the rest. Octave errors are corrected toward the annotation, the correction
recorded per row. The DJ corpus has no annotations and inherits the error rate this shows.

**Key, loudness, energy, onset, centroid, MFCC.** Nothing computed for the new corpora. Methods are the
settled ones. Key cannot be checked for accuracy beyond what was done; a consistency check (full track
against preview on the 5,989 tracks that have both) comes free in phase 3.

**Embeddings.** None for the new corpora.

Output: a tempo report here, and a corrected BPM column with its source.

**Done 2026-10-07.** The check, the re-measure and the relabel, with their numbers, are in
`DATASET_STATE.md` under the 2026-09-30 Raveform run ("Tempo check and relabel"). Outcome for this file:
`tempos.csv` holds the corrected BPM, 6,767 of 6,905 Raveform records trusted, 138 null; the DJ corpus is
unchecked.

## Phase 2. What to extract, and how

Per full track, both corpora, 12,211 files.

- discogs-effnet at 16 kHz mono. Every 2.05 s patch vector kept, float16, one `.npy` per track under `data/djdata/<corpus>/embeddings/`
  (a 7-minute track is about 940 KB at essentia's one-vector-per-second hop, both corpora about 11.5 GB). The whole-track mean goes in the table. The
  patches make the played-span and seam-edge tests possible without a second model run.
- librosa at 22,050: key (edma via `normalise_key()`), LUFS, RMS energy, raw onset (same scale and
  column name as the old table), spectral centroid, 13 MFCCs.
- BPM joined from the corrected `tempos.csv`. DeepRhythm not run. `bpm_source` is `measured` for new
  rows, `deeprhythm` for old ones.
- Output: `data/djdata/<corpus>/out/track_features.parquet`, one row per track, one table per corpus like
  the other tables there. The old `features.parquet` is untouched. Whether the two join is decided in phase 3.

Code: `build_features.py` gets a manifest built from the two `tracks/` folders, a patch-saving branch
in the embedder, the `normalise_key()` call in place of its own flat-to-sharp map, and a configurable
output path. Each track is decoded once for both the embedding and the librosa features.

## Phase 3. The embedding checks

On a sample first: about 500 measured Raveform seams, both tracks, roughly 1,000 tracks, chosen so most
have a preview on the laptop. Raw cosine on the frozen vectors, no training.

1. Preview against full track. Preview vectors computed on the laptop (essentia installed, 5,989
   preview files in `data/interim/seam_previews/audio`). Per track the cosine distance between the two
   vectors; per seam the adjacency AUC of each. Decides whether the old 28,460 preview rows can train in
   one space with the new rows.
2. Whole track against played span against seam edge. Played span from `plays.csv` (`played_from_s`,
   `played_to_s`), seam edge the last and first 32 bars from the measures and tempos. Each is a mean over
   a patch range. Adjacency AUC for each. Decides what the chooser's head trains on and what the
   transition model (Step 6) gets.
3. Adjacent pairs against same-genre random pairs against other-genre random pairs, cosine per group,
   per span. Checks the closeness requirement directly.

Baseline: raw full-track adjacency AUC against the 0.622 the previews gave. If it does not move, the
pairing signal is weak in the audio and no head fixes it.

Decision at the end of this phase: the span the head trains on; whether old and new data join; whether
patches stay on disk for all tracks or only the means are kept.

**Sample run, 2026-10-07.** 500 measured seams, one per mix where possible, 977 tracks (tech house 113,
techno 108, trance 87, house 79, drum and bass 79, deep house 20). Full tracks on the VM, 6 spawn
processes, 1,588 s, 1.6 s per track. Previews on the laptop, 3 processes, 1.0 s per track. Outputs:
`data/interim/embedding_sample/` (report, sample table, seam list), the test script `scripts/features/embedding_tests.py`,
`s3://aidj-1/djdata/features/` (the 977 patch files, the sample table, the report).

**Results, raw cosine on frozen discogs-effnet, no training.** Negatives are one random track per
seam from another seam, same genre or other genre. 500 positives, so the AUC's standard error is
about 0.02.

```
span          AUC vs same-genre   AUC vs other-genre   cos pair   cos same-genre random   cos other-genre random
whole track        0.665               0.869              0.843          0.788                  0.614
played span        0.657               0.860              0.837          0.778                  0.628
seam edge 32 bars  0.655               0.840              0.795          0.737                  0.575
preview 30 s       0.609               0.774              0.753          0.694                  0.557

preview against full track, same track, 977 tracks
  cosine distance median 0.053, p90 0.248; random other track median 0.384
  share of tracks whose own preview is farther than the random median: 5.7 %
```

What the numbers say.
- Whole track is the best span, and played span and seam edge are within noise of it (0.008 and
  0.010 under). The pairing signal does not improve by looking at the part actually played.
- The raw whole-track vector from full audio (0.665) equals the trained Model A head on previews
  (0.663) and beats the raw preview vector (0.609 here, 0.622 on the old split). The audio source was
  the limit, not the head.
- Pairs are closer than same-genre random pairs, which are closer than other-genre pairs, on every
  span. The closeness requirement holds.
- A preview sits close to its own full track (0.053 against 0.384 for a stranger), so the two live in
  one space geometrically. Its pairing signal is weaker by 0.056 AUC, more than the noise. The 5.7 %
  of previews far from their own track is in line with the 11 % different-version share.

## Decisions on using the old and the new data

Moved to `DATASET_STATE.md`, top section "How the old and the new data are used" (2026-10-07), because
it covers the planner, the mixer and the transition model, not only the embeddings.

## Phase 4. Where to run what

- Phase 3 tests are cosine on vectors: laptop.
- The sample and the full 12,211: the CPU VM (c6i.4xlarge), `essentia-tensorflow` and `pyloudnorm`
  installed in its `aidj` environment on 2026-10-07. The 20-track timing below put a full track at
  4.8 s in one process, so the full run is 4 to 6 h at 4 to 8 processes and the GPU path
  (`DiscogsEmbedderGPU`) is not needed.
- Audio reaches any VM from S3 (`s3://aidj-1/djdata/tracks/`, `s3://aidj-1/djdata_djs/tracks/`), never
  from the laptop. Previews stay on the laptop and are embedded there.

**20-track timing, 2026-10-07, CPU VM, one process.** Mean track 6.2 min. Decode to 16 kHz 1.1 s,
discogs-effnet 2.0 s, librosa features 1.7 s, 4.8 s per track in all. essentia's default patch hop
gives one 1,280-vector per second of audio, 938 KB per track in float16, about 11.5 GB for all 12,211.
Full run about 16 h in one process, 4 to 6 h at 4 to 8 processes. The GPU is not needed; the quota
request (8 vCPUs G, id 9869c3de, 2026-10-07) can stay pending or be withdrawn. The 20 readings were
sane: keys plausible, LUFS -7.7 to -17.2, raw onset 1.1 to 2.5.

## Phase 5. The full run

**Started 2026-10-07 01:06 UTC** on the CPU VM, `~/AI-DJ/run_track_features_2026-10-07.sh` in tmux
`trackfeat`: Raveform then the DJ corpus, 6 spawn processes, the 977 sample tracks resumed. One table
per corpus at `data/djdata/<corpus>/out/track_features.parquet`, vectors at
`data/djdata/<corpus>/embeddings/`. At the end the script syncs to `s3://aidj-1/djdata/` and
`s3://aidj-1/djdata_djs/` (same layout) and shuts the VM down, which the seam-run scripts did not.
Expected about 5.5 h. The sample's S3 copy under `djdata/features/` is superseded by this and can go
once the sync is checked.

**Done 2026-10-07 06:11 UTC.** Raveform 01:06 to 03:48 (6,917 tracks, 977 resumed), DJ corpus 03:48 to
06:11 (5,294). No failures. Synced and counted on S3: 6,917 and 5,294 vector files, the two tables
(56 and 43 MB). The VM shut itself down at 06:14. Tables pulled to the laptop.

```
                   raveform   djs
rows                 6,917   5,294
bpm present          6,767   2,959   (djs: only records that sit in a seam have a tempos.csv row)
key unreadable           0       0
minutes, median       6.27    6.00
vectors per track      376     361   (one per second of audio)
LUFS, median          -9.6   -10.1
raw onset, median     1.70    1.71   (old table range 1.1 to 2.5)
```

**BPM for every track, 2026-10-07 08:15 to 08:23 UTC.** The 2,335 DJ tracks and 12 Raveform tracks
with no `tempos.csv` row (fetched, never in a seam) were measured with the tempo stage's instrument
(`tempo.py::bpm_of`, librosa hint, 16 workers, 477 s and 9 s), appended to each corpus's `tempos.csv`
(the earlier files kept as `tempos_before_2026-10-07.csv` and `tempos_before_fill_2026-10-07.csv`),
and re-joined into both tables. Now: Raveform 6,779 of 6,917 with a BPM (138 uncertain, null), DJ
corpus 5,286 of 5,294 (8 with no periodic kick, null). The new DJ values carry the librosa hint's
known weakness above 150 BPM; the DJ corpus has no drum and bass. Tables and tempos on S3 and the laptop.

## Order

1. Tempo check against the Raveform beats (laptop). Done 2026-10-07.
2. GPU quota request. Sent 2026-10-07, then found unnecessary by the timing.
3. VM install, 20-track timing. Done 2026-10-07.
4. The extraction code, `src/features/track_features.py`, with `tests/test_track_features.py`;
   `build_features.py` loads DeepRhythm lazily and keys go through `normalise_key()`. CI green
   2026-10-07. The pool must be spawn: forked children deadlocked on the first model call because the
   parent had imported essentia's TensorFlow runtime (both runs hung 15 min).
5. Sample run (500 seams, 977 tracks, `--only`), the three tests, decision.
6. Full run. Not before the decision.

## Unverified at the time of writing

Overlap between the old dataset's tracks and Raveform's by title, which matters for any shared
validation split. Whether `data/raw/previews` still exists on the VM or only on S3.
