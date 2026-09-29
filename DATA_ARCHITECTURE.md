# Data architecture as it stands, 2026-09-24

**This is a description, not a design.** Nothing here was architected. It grew one script at a time while
the measurement method was still being worked out, and it shows. The stores are split across three places
by accident of where a job happened to run, the same information lives in several forms, and a third of
the CSVs are one-off diagnostic output that nothing reads.

**The rework waits on the blockage.** The scripts that measure and mine the features are not finished.
Until the DJ track question is settled and `measure_seam` has run across a corpus, we do not know which
columns the seam table actually needs, so designing storage around it now would be guessing. Once that is
clear, this gets rebuilt as something robust and maintainable: one seam index per corpus, one place per
kind of audio, and the sqlite files demoted to what they are, which is a resume log.

Everything below was verified by listing the files on 2026-09-22 and 2026-09-24, not from memory.

---

## 0. Layout since 2026-09-23

One directory for everything the seam pipeline produces, one sub-directory per corpus. The three trial
roots `djdata_1001`, `djdata_test` and `djdata_tier1` were deleted the same day: an early 1001 trial, a
pipeline test, and the export of the discarded NNLS measurement.

```
data/djdata/
  raveform/          Raveform corpus: state.sqlite, tracks/, windows/, mixes_tmp/, out/, logs/   (config.yaml)
  djs/               DJ profiling:    state.sqlite, tracks/, windows*/, mixes_tmp/, out/, logs/   (config_djs.yaml)
  djs/lists/         the DJ-profiling lists, moved out of data/interim:
                       dj_mixes_all_1001_links.csv   one row per mix: 1001 page url, club_or_radio, audio source, audio url
                       dj_mixes_need_audio.csv       the 184 whose audio came from a title search and must be refetched
                       dj_mixes_excluded.csv         mixes dropped for good; the manifest skips them (see below)
                       dj_mix_media_links.csv, dj_mix_audio_found.csv, dj_profiling_*.csv, untimed_mixes_deferred.csv
  djs/fred/          Fred again, handled separately: his audio on the VM, his lists under fred/lists/
data/interim/        only what is interim by design: tracklist.csv, mix_urls.csv, and the caches
                     (segments/, stretch/, dj_previews/, seam_previews/)
data/raw/dj_mixes_1001_links.csv   dj, 1001 page url, mix link if found, for every mix in the DJ database
```

**Exclusions.** `djs/lists/dj_mixes_excluded.csv` holds mix ids Anas has dropped from DJ profiling, with the
reason and date. 208 radio shows on 2026-09-23: all 42 of Solomun's, 74 of Amelie Lens's keeping her 34
fetched plus the 6 most recent, 92 of Sultan + Shepard's keeping the 40 most recent. `manifest/tracklists.py`
reads the file and skips those mixes on every rebuild. `scripts/diag/apply_exclusions.py` removes them from a
database that already holds them, with their seams and the tracks nothing else uses. Run on both machines. After the dead-link round on 2026-09-24 the exclusion list holds 279 mixes and the database 285.

**Mix audio, refetched 2026-09-24.** Every DJ mix's audio now comes from the link on its own 1001 page, never
from a title search. `scripts/diag/mix_media_links.py` read 312 pages (no Cloudflare challenge in the whole
run): 302 gave a link, 39 said no recording exists, 11 had no player we recognise. `scripts/diag/fetch_mix_links.py`
then fetched them on the VM: 197 fetched, 21 dead links (19 Mixcloud accounts gone, 2 YouTube videos removed),
84 were already on disk from the same link. The 50 without a link and the 21 dead are on the exclusion list.

```
mixes with audio from their own 1001 page     281     Solomun 87, Amelie Lens 64, Sultan + Shepard 43,
                                                      DJ Tennis 36, Roman Flügel 31 (16 without a tracklist yet), Black Coffee 20
flagged in dj_mixes_fetched.csv               long 6, short 6, two DJs in the title 23   (check by hand)
old title-search files                        data/djdata/djs/mixes_tmp/old_searched/ on the VM, not deleted
tracks named in those 265 tracklists          4,063: on disk 4,414 after the 2026-09-24 fetch, 25 with no upload anywhere
```

**Tracks, refetched 2026-09-24.** `djdata run` in fetch-only mode fetched the 1,761 tracks missing from the 281
mixes' tracklists, with the name rules in `fetch/names.py` (both sources searched, "title remixer" query when the
title has brackets, one listed artist or the remixer in the upload's title or channel, closest title of the top
results wins, a relaxed pass on length and 60% of the title words when the strict pass finds nothing, 60 s floor,
no iTunes preview check). ID tracks were deleted from the database with the seams that named them
(`lists/dj_tracks_unidentified.csv`, `lists/dj_seams_unidentified.csv`).

```
tracks in the DJ database        4,414 ready (file on the VM)    25 no upload anywhere    198 in the 8 deferred untimed mixes
fetched this round with a url    1,659 YouTube, 118 SoundCloud; 13 on the relaxed pass (log "relaxed pass")
url stored per track             tracks.url, so the source of every track fetched from now on is known
```

The 25 are unreleased edits, mostly DJ Tennis's own, plus three SoundCloud uploads that refused to download; Anas
could not find them by hand either. None of the 4,414 has yet been checked against its mix: that is the whole-mix
locate, `scripts/diag/mix_locate_once.py`, with a listening test first (`scripts/diag/locate_lr_test.py`).

**YouTube from the VM needs two things now.** A PO-token server (bgutil, Deno, tmux session `pot`, port 4416)
and an address YouTube has not tied to the SABR-only experiment. On 2026-09-24 the old address got only the
360p fallback for every long upload, which then 403'd; rotating the Elastic IP (`rotate.py`)
fixed it; the address has changed again since, and the gate rotates it by itself on a block, so never write it down: `scripts/vm/vmssh` finds it. `fetch_mix_links.py` probes before declaring a block.

**Paths inside the databases.** The layout move left `data/djdata_djs/...` in `tracks.path`, `mixes.path` and
`seams.window_path`. Repointed in both copies on 2026-09-24 (159 of 268 sampled tracks had been skipped by the
locate for that reason). Anything that reads a path from the database should fall back to a glob by id.

**S3 keys are unchanged**: `s3://aidj-1/djdata/` and `s3://aidj-1/djdata_djs/`.

**The VM has the same layout** since 2026-09-24, and the same exclusions applied to its database.

## 0b. Changes on 2026-09-28/29

The three old DJ window folders (`djs/windows` listed-time cut, `windows_v2` virtual-zero cut,
`windows_v3` bounded sample, 12.4 GB) were deleted for good on Anas's word; `djs/windows/` now holds only
the 2,459 windows of the 2026-09-28 run. `djs/mixes_tmp/` also holds 17 links `usb_fred_*.m4a` to Fred's
solo segments and `51f57bd5cc.m4a` to `fred/bWUsbsTUKV4.m4a`; the two wrong title-search files
40caea7a7d and e3249f53e6 are in `mixes_tmp/old_searched/`. New tables in `djs/out/`: `presence.csv`,
`layers.csv`, `layer_bands.csv`, `mixes.csv` (`DATA_PIPELINE.md` 2026-09-28). The corpus reads
`djs/lists/dj_mixes_excluded.csv` and `djs/lists/usb002_solo_tracklist.csv` through `config_djs.yaml`.
On S3 the corpus mirrors the VM: the final tables in `s3://aidj-1/djdata_djs/out/`, the run logs in `s3://aidj-1/djdata_djs/logs/`, beside `mixes_tmp/` and `tracks/`.
**VM cleaned for Raveform, 2026-09-29.** Everything of the DJ profiling run was uploaded to
`s3://aidj-1/djdata_djs/` (mixes_tmp, tracks, windows, fred, lists, logs, out, state.sqlite, and Fred's
check outputs in `fred/checks/`), checked file by file (name and size, 0 missing), then deleted from the
VM except `djs/tracks/` (5,294 files, 31 GB, kept for feature extraction), the tables in `djs/out/`,
`lists/`, `logs/` and `state.sqlite`. Deleted without upload: `mixes_tmp/old_searched/` (wrong
title-search mixes), the audio of the excluded mixes, `out/curves`, `out/played`, the pre-fix-up table
backup, old `state.sqlite.bak-*` copies, the 2026-09-28 test root. Raveform's data was not touched.
VM disk after: 61 GB free of 193 GB. To work on DJ mixes or windows again, sync them back from S3.
Sections 2 to 7 below predate this and describe the September state.

## 1. The id scheme

`seam_id = mix_id + "_" + a + "_" + b` on both corpora. Verified true on samples from each.

**Raveform.** The track id **is** the YouTube video id.

```
seam_id     mix1323_s2OWD0_5Fkw_kX-C1ADo8r0
mix_id      mix1323
a, b        s2OWD0_5Fkw, kX-C1ADo8r0
track url   https://www.youtube.com/watch?v=s2OWD0_5Fkw
track file  data/djdata/raveform/tracks/s2OWD0_5Fkw.m4a
window file data/djdata/raveform/windows/mix1323_s2OWD0_5Fkw_kX-C1ADo8r0.mp3
```

So a Raveform seam id names its own two videos. There is nothing to look up.

**DJ profiling.** The ids are 12 hex characters, minted by the 1001tracklists scrape.

```
seam_id     00be6be61d_a1a6b806f640_0b64b6df0cb1
mix_id      00be6be61d
a, b        a1a6b806f640, 0b64b6df0cb1
track url   NULL            the record was found by searching, not by a known url
track file  data/djdata/djs/tracks/a1a6b806f640.m4a
window file data/djdata/djs/windows/00be6be61d_a1a6b806f640_0b64b6df0cb1.mp3
```

The DJ ids join cleanly back to the scrape. Verified: **all 7,849 track ids and all 564 mix ids appear in
`data/interim/tracklist.csv`.**

The old dataset shares the same 12-hex track id, which is why 1,716 DJ tracks already had a preview.

---

## 2. The two databases

Both are `state.sqlite` with the same three tables.

```
mixes    mix_id, title, url, source, year, genres, status, path, error, worker, updated
tracks   track_id, title, url, duration, status, path, error, worker, updated, archived
seams    seam_id, mix_id, a, b, tier, source, coarse, status, window_path, error, worker, updated, archived
```

`coarse` is a JSON blob and it is where all the alignment lives. Its shape differs by corpus and by
whether locate has run.

```
raveform          A, B, a_track_end_mix_t, b_track_start_mix_t, match_rate,
                  overlap_start_mix_t, overlap_end_mix_t, window
  per record      duration, mix_t, orig_t, rate            (no votes: not from our locate)

dj, located       A, B, a_track_end_mix_t, b_track_start_mix_t, needs_locate, source,
                  overlap_start_mix_t, overlap_end_mix_t, window
  per record      duration, mix_t, orig_t, rate, votes
                  orig_t is ALWAYS 0, so mix_t is the record's virtual time zero

dj, not located   A, B, needs_locate: true
  per record      track_id, listed_mix_t                   nothing else exists yet
```

**Row counts, and the laptop copy is stale.**

```
                   LAPTOP djdata            VM djdata (the real one)
mixes    2,084     all pending              939 windows_ready, 85 failed, 1,060 pending
tracks  16,842     all pending              6,917 ready, 817 failed, 9,108 pending
seams   14,080     all pending              1,181 done, 3,931 ready, 1,743 failed, 7,225 pending
```

The laptop's `data/djdata/raveform/state.sqlite` is the manifest as built, before the run. Querying it returns
zeros for everything. **The VM copy is the only real one.**

`djdata/djs/state.sqlite` is identical on both machines.

```
mixes      564    137 windows_ready, 38 downloaded, 34 pending, 30 deferred_untimed, 8 failed, 317 skipped_cut
tracks   7,849    3,158 ready, 270 failed, 4,421 skipped_cut
seams    9,872    1,475 ready, 1,393 failed, 7,004 pending
```

(Those are the 2026-09-22 counts. After the 2026-09-23/24 exclusions, the ID-track removal and the retag of seam
tiers, the VM copy holds 285 mixes, about 5,750 seams and 4,414 ready tracks; see section 0.)

After the 2026-09-23 exclusions the laptop copy of `djdata/djs/state.sqlite` holds 356 mixes, 7,034 seams and
5,528 tracks. The VM copy still has the 564 until `apply_exclusions.py` runs there.

Usable Raveform seams, meaning a window plus both tracks present, is 5,062. That is `done` plus `ready`
minus the ones missing a record.

---

## 3. Audio stores

```
                                        LAPTOP              VM                 S3
djdata/raveform/windows                        none                6,218   15.5 GB    yes
djdata/raveform/tracks                         none                6,917   40.9 GB    yes
djdata/djs/windows        (old cut)     none                1,475    7.1 GB    no
djdata/djs/windows_v2     (wrong cut)   none                1,471    5.5 GB    no
djdata/djs/windows_v3     (bounded)     none                  149    0.5 GB    no
djdata/djs/tracks                       35                  3,157   19.1 GB    yes
djdata/djs/mixes_tmp      (full mixes)  1                     175   13.7 GB    partial
djdata/djs/fred                         none                   59    4.0 GB    no
djdata/djs/fred/segments                none                   43    1.4 GB    no
raw/raveform    (the source dataset)    74,417   1.8 GB     74,416   1.3 GB    partial
raw/dj_mixes    (28 mixes, unused)      29       2.3 GB     none                no
```

Three window directories exist for the DJ corpus and only the third is correct. `windows` was cut on the
1001 listed time, `windows_v2` on the record's virtual zero, `windows_v3` on the bounded span. **The
database still points at `windows`.**

---

## 4. Preview stores, three of them

```
data/raw/previews                  28,720 laptop / 29,936 VM    iTunes, old dataset, keyed by 12-hex track_id
data/interim/seam_previews/audio    5,989 laptop / none on VM   iTunes, Raveform seam tracks, keyed by YouTube id
data/interim/seam_previews/played_part 132 laptop / 551 VM      cut from the mix where iTunes had nothing
data/interim/dj_previews/audio        457 laptop / 1,233 VM     iTunes, DJ tracks, fetched 2026-09-21
```

`data/raw/preview_manifest.csv` indexes the first one: 28,460 track ids with artist, track name, source
and local path. The other two have no manifest, only a fetch log.

**Preview coverage of the 3,158 DJ tracks on disk**, verified by joining the ids:

```
have an old-dataset preview      1,716
have a preview fetched today     1,233
covered by either                2,949
no preview anywhere                209
```

---

## 5. The CSVs

### The scrape, which everything joins back to

```
data/interim/tracklist.csv            131,869 rows   the raw 1001 scrape. 58,893 track ids, 5,911 mix ids
                                                     mix_id, mix_title, dj_name, genre, url, track_id,
                                                     starting_time, track_name, artist_name, play_type, overlay_parent
data/processed/tracklist_clean.csv     47,390 rows   the cleaned version used by the models
data/interim/mix_urls.csv               5,692 rows   url, dj_name, genre, complete
```

**`starting_time` is minutes past the hour and rolls over hourly.** It is not seconds and not an offset
from the mix start. Values look like `2.5`, `11.333`. This is the field the original DJ window cut was
built on, and it is why that cut failed.

### The old dataset

```
data/raw/preview_manifest.csv         28,460 rows   track_id, artist, track_name, source, local_path
data/processed/features.parquet       28,460 rows   embedding, bpm, key, energy, loudness, onset, mfcc
data/processed/split_mixes.csv         2,834 rows   mix_id, genre, split
data/processed/transition_labels.csv       64 rows   the hand-labelled transitions
```

### Raveform

```
data/external/ear_test/genre_labels/raveform_genre_labels.csv  1,176 rows   mix_id, dj, label, seams, tier, club, keep
data/interim/raveform_webm_check.csv      248 rows   the webm cut audit
data/interim/raveform_seam_audit.csv       60 rows   presence audit with controls
data/interim/measure_rav.csv               30 rows   bands, bass, floors, separations
```

### DJ profiling

```
data/djdata/djs/lists/dj_profiling_tracks.csv     3,347 rows   artist_name, track_name, track_id, dj_name, mix_id
data/djdata/djs/lists/dj_profiling_mixes_skipped.csv 317 rows  what was cut from the run and why
data/djdata/djs/lists/untimed_mixes_deferred.csv    248 rows   mixes with no usable start times
data/djdata/djs/lists/dj_mixes_missing_url.csv       74 rows   mixes with no audio link
data/djdata/djs/lists/dj_mix_media_links.csv         43 rows   links found by scraping
data/djdata/djs/lists/dj_mix_links_missing.csv       56 rows   and the ones still missing
data/djdata/djs/lists/dj_previews_todo.csv        1,442 rows   the preview fetch work list
data/djdata/djs/lists/dj_track_identity_full.csv  2,949 rows   track_id, title, votes, verdict
data/interim/recut_manifest_v3_*.csv        50 rows   the bounded cut, one per sample
data/djdata/djs/lists/dj_audit_votes.csv             49 rows   presence audit, vote-gated sample
data/interim/measure_dj_votes.csv           49 rows   bands, bass, floors, separations
```

### Fred

```
data/djdata/djs/fred/lists/fred_manual_tracklists.csv      232 rows   show, marathon_time, title, note
data/djdata/djs/fred/lists/fred_tracks_resolved_spotify.csv 59 rows   the Spotify resolution
data/djdata/djs/fred/lists/fred_marathon_segments.csv       44 rows   parsed from the video description
data/djdata/djs/fred/segments/segments.csv    42 rows   file, show, artists, start, end, length, cut_error, is_b2b
data/djdata/djs/fred/lists/fred_segment_order.csv           39 rows   the segment match by play order
```

### Diagnostic output that nothing reads

`seam_rate_measure`, `seam_rate_scan`, `seam_rate_full_mix`, `seam_rebuild`, `seam_reconstruct`,
`fp_bands_dj`, `fp_bands_keylock`, `fp_check_smoke`, `recut_check_smoke`, `dj_recut_audit_smoke`,
`raveform_seam_audit_smoke`, `measure_smoke_rav`, `dj_seam_audit`, `dj_rate_sweep`, `dj_refetch_report`.
All one-off, all superseded, all still on disk.

---

## 6. What each record was used for, and why it is wrong

`data/djdata/djs/out/played/<mix_id>.json`, 137 files on the VM, one per mix.

```json
{"mix_id": ..., "mix_len_s": ..., "tracks": [
  {"track_id", "listed_mix_t", "mix_t_of_track_zero", "rate", "votes", "track_len_s",
   "played_mix_from_s", "played_mix_to_s", "played_track_from_s", "played_track_to_s"}]}
```

This is meant to be the per-DJ record the profiles are built from. **Two of its four played fields are
not measurements.**

```
played_mix_from_s    = mix_t, the record's virtual time zero, so played_track_from_s is 0
                       for every track. It says every DJ starts every record at 0:00.
played_mix_to_s      = min(theoretical_end, next_start + 600, mix_len). Against the envelope
                       measurement it overstates the cut by a median 15 s over 16 records.
```

Nothing should read these until they are replaced with measured entry and exit.

---

## 7. Where everything lives, and what is at risk

```
S3  s3://aidj-1/djdata/        14,326 objects   56.5 GB
    s3://aidj-1/djdata_djs/     3,526 objects   36.8 GB
    plus processed/, models/, raw/previews/ from the old dataset
```

**On the laptop only, on no backup:**

```
data/interim/seam_previews/audio     5,989 files   6.4 GB   the Raveform 30 s previews
data/raw/dj_mixes                       29 files   2.3 GB   28 mixes with no tracklists, unused
data/processed                          21 files   1.4 GB   features, split, the good embeddings
data/external/ear_test               2,298 files   7.6 GB   every ear test and its results
```

The ear tests are the only human ground truth in the project and they exist in one place.

**On the VM only:** `windows_v2`, `windows_v3`, `mixes_tmp` (the full mix audio), the whole Fred
directory, and the 1,233 previews fetched today.

**Stale on the laptop:** `data/djdata/raveform/state.sqlite`, which is the pre-run manifest.

---

## 8. What is wrong with this architecture

Listed so the rework has a target.

1. **The dataset lives inside a resume log.** `state.sqlite` has to be sqlite because workers claim rows
   transactionally, but the seam definitions, the mix ids and the alignment are only readable by opening
   a database. There should be one `seams_index.csv` per corpus next to the audio.
2. **The alignment is a JSON blob in a column**, with a different shape per corpus and per stage. Nothing
   can query it.
3. **Three window directories**, only the third correct, and the database points at the first.
4. **Three preview stores**, two with no manifest, split differently across the two machines.
5. **The played record is derived, not measured**, and its field names read as though it were measured.
6. **Fifteen diagnostic CSVs** that are superseded and indistinguishable from live ones by name.
7. **No single index** that answers "which DJ, which mix, which two records, which part of each, and
   where is the audio" without joining a database, two CSVs and a directory listing.

---

## 9. The planned structure, agreed 2026-09-25

This is the layout the data pipeline moves to. It is a plan. Each line says whether it exists today.
The reasoning and the order of work are in `PLAN.md` under "DATA PIPELINE".

### One package, one CLI, one config per corpus

```
djdata_package/djdata/
  sources/        what a corpus is and where its links come from
    raveform.py       exists as manifest/raveform.py
    tracklists1001.py exists as manifest/tracklists.py plus fetch/media_link.py
  fetch/          audio in, mixes and tracks pulled in parallel by one runner
    mix.py            exists. _window_for and the locate call inside it are dropped
    track.py          exists
    yt.py             exists, the shared yt-dlp wrapper
  seam/
    locate.py         to build, from scripts/diag/mix_locate_once.py. The current locate.py is dead
    pairs.py          to build, the adjacency rule and the third-record flag
    cut.py            to build, from recut_windows_bounded.py, reading the locate table, no cap
    floors.py         to move from scripts/diag/floors.py, unchanged
    bands.py          to move from scripts/diag/fp_bands.py, library functions only
    bass.py           to move from fp_bass.py and fp_bass_timeline.py, library functions only
    presence.py       to move from fp_presence.py and fp_presence_batch3.py, library functions only
    measure.py        to build around measure_seam.measure_one
    labels.py         to build from transition_labels.py, rules only, no folder-bound input
    loops.py          to build from transition_labels.loop_steps
    tempo.py          to build, the mixer's tempo function on a track, checked first
  pipeline.py     the stages in order, each resumable, each one function
  store/
    state.py          exists, the resume log workers claim rows from
    tables.py         to build, the dataset tables below
  export.py       exists, rewritten for the tables below
  cli.py          exists, gains one subcommand per stage
djdata_package/configs/
  raveform.yaml   exists as config.yaml
  djs.yaml        exists as config_djs.yaml
  fred.yaml       to write
djdata_package/tests/   synthetic audio only, no network, run by make ci (not in make ci today)
notebooks/              read out/ tables only
```

### One folder per corpus, the same inside

```
data/djdata/<corpus>/
  mixes_tmp/      full mix audio, named by mix id, or Fred's solo segments
  tracks/         full track audio, named by track id
  windows/        one cut per seam, named by seam id
  out/            the tables below, flat CSVs beside them
  logs/
  state.sqlite    the resume log, not the dataset
```

Corpora. `raveform`, `djs`, `fred`. Fred is a corpus whose mixes are his solo marathon segments. The old
text-and-preview dataset joins later as a corpus with no mixes, tracks or windows.

### The dataset tables, one shape for every corpus

`plays`, one row per track play, from the locate stage.
```
mix_id, dj, genre, track_id, order_listed, order_heard, first_heard, last_heard, rate, votes,
floor, control_max, confidence, listed_min, drift_min, played_from_s, played_to_s
```

`seams`, one row per seam, from pairs, cut and audit.
```
seam_id, mix_id, dj, genre, track_a, track_b, order_ok, gap_flag, third_record, window_file,
window_t0, window_t1, window_s, audit_start_ok, audit_end_ok
```

`measures`, one row per seam, from measure, label, loops and bars.
```
seam_id,
low_in, low_out, mid_in, mid_out, high_in, high_out, per band votes, floor, separation,
bass_swap_s, bass_words (both, outgoing, incoming, cut, break, with seconds each),
overlap_s, overlap_bars, bar_s, tempo_a, tempo_b,
sweep_in, sweep_out, loop_steps, label, labels_all, measured, control_trusted
```

Times are seconds into the mix for `plays`, seconds into the window for `measures`. Every row carries the
control floor that judged it. Ear verdicts stay in `data/external/ear_test/*/MARKS.csv` and are joined by
seam id, never copied in.

### What exists today, and what does not (updated 2026-09-25 evening)

Built and in the package, with tests in `make ci`. Every module under `seam/` above except that the
files are named as listed there (`locate.py` is the whole-mix locator; the old slice locate is
`legacy/locate_slice.py`), `sources/tracklists.py`, `sources/raveform.py`, `store/tables.py`,
`pipeline.py`, the CLI stages, `config_fred.yaml`. The tables above exist with these differences:
`plays` also carries `sweep_votes` and `sweep_floor`; the cut and its audit are one table `cuts`;
`measures` and `labels` are separate tables; `tempos` is a table of its own; `export-seams` writes
`seams_index.csv` joining them. `manifest/` and `fetch/` are untouched and still serve the old fetch path.

Run on real audio once, the development sample, see `DATASET_STATE.md` 2026-09-25. Not yet heard.

Does not exist. The old `scripts/diag` code is not deleted yet. `fetch/mix.py::_window_for` and its
locate call are still there, unused with `fetch_only: true`. No notebook reads the tables yet.

Dead, unchanged. The gain fit and everything that depends on it, rate refinement, the nine scripts on
the 120 s geometry, the run's `locate_one`.

### Rules the layout follows

- The dataset is the tables in `out/`. `state.sqlite` is a working file nobody reads.
- Anything that reads a path from the database falls back to a glob by id.
- Adding a DJ is a config row. Adding a source is one file in `sources/`.
- A stage is one function. Airflow calls the same thing the CLI calls.
- Every number travels with the control floor and separation that produced it.
