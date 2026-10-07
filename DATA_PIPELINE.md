# DATA_PIPELINE, how to run the seam pipeline and use its tables

Moved from START_HERE.md on 2026-09-27, content unchanged. What was built, what it is proved by, how to run it, where the data lands, how to add data, and the watch list for a run. The numbers and the findings behind it are in DATASET_STATE.md; the layout in DATA_ARCHITECTURE.md; the code in djdata_package with its own README.

## Added 2026-09-30: the Raveform run

Numbers in `DATASET_STATE.md` 2026-09-30. Same stages as the DJ run under `djdata_package/config.yaml`,
from the repo root on the VM:

```
C=djdata_package/config.yaml
PYTHONPATH=djdata_package:. python -m djdata.cli locate --config $C          # alignment → plays, 2 min
PYTHONPATH=djdata_package:. python -m djdata.cli pairs --config $C
PYTHONPATH=djdata_package:. python -m djdata.cli cut --config $C --workers 16   # adopts windows, audits
PYTHONPATH=djdata_package:. python -m djdata.cli measure --config $C --workers 14
PYTHONPATH=djdata_package:. python -m djdata.cli tempo --config $C --workers 16
PYTHONPATH=djdata_package:. python -m djdata.cli label --config $C
PYTHONPATH=djdata_package:. python -m djdata.cli export-seams --config $C
```

The run script is `data/djdata/raveform/run_full_2026-09-30.sh` on the VM (copy in
`s3://aidj-1/djdata/logs/`): stops at the first stage with a failure, logs memory every 30 s, no S3 sync,
leaves the VM on. `layers` and `layer-bands` do not apply, since Raveform has no full mix audio.

**Watch list, added 2026-09-30.**

- Test a corpus on a sample root first. `data/djdata/raveform_sample/` links `tracks/`, `windows/` and
  `state.sqlite` to the real root and writes its own `out/` and `logs/`; no stage writes to
  `state.sqlite` (the only connection is read-only). A 10-mix sample took 19 min end to end.
- Adopted windows can be longer than asked. `cuts.csv` has `asked_s` and `actual_s`;
  `seam/cut.py::file_start` reads the file from its real start. Check `cut_error_s` by container on any
  new corpus of adopted windows before trusting measure.
- `measured == 1` needs only one band to clear somewhere, so a misplaced seam can pass it at a
  separation of 1.0. Filter on the per-band separations, not the flag alone.
- S3: a new run writes tables with the old names into an `out/` that may hold an older run. Copy the old
  tables into a dated folder first and check the count, then sync, then delete old top-level files that
  the new run does not write (done for Raveform: `out/old_gainfit_2026-09-14/`, 1,185 files).
- The workers' stderr log reaches about 370,000 lines on the full run; no line holds "error" or
  "traceback".
- A process pool that forks after essentia's TensorFlow runtime is loaded in the parent deadlocks on
  the children's first model call, with load 0 and no error (two runs hung 15 min, 2026-10-07). Use a
  spawn context, as `djdata/pipeline.py` and `src/features/track_features.py` do.
- The VM's `aidj` environment had no `pyarrow`, so pandas could not write parquet; a run started while
  pip was still installing it also failed. Check `python -c "import pyarrow"` before a parquet-writing
  job, and never start a job during a pip install.

## Added 2026-09-28: Fred again.. in the DJ corpus, layer tables, exact-time lookup

Commits 3a563ab, 2908560 and 241f24c on `origin/dev`. The numbers behind each choice are in
`DATASET_STATE.md` under 2026-09-28.

**Fred is part of the DJ run, not a corpus of his own.** His 17 solo sets from the USB002 marathon that
have a tracklist are mixes named `usb_<segment>` (for example `usb_fred_07_madrid_fred`), each a link in
`data/djdata/djs/mixes_tmp/` to `data/djdata/djs/fred/segments/<segment>.m4a`. Their tracklists come
from usb002-tracklist.app, saved as `data/djdata/djs/fred/lists/usb002_app_2026-09-28.json`, turned into
tracklist rows with

```
PYTHONPATH=djdata_package python -m djdata.cli usb002-tracklist \
  --app data/djdata/djs/fred/lists/usb002_app_2026-09-28.json \
  --tracks data/djdata/djs/fred/lists/usb002_solo_tracks.csv \
  --segments data/djdata/djs/fred/segments/segments.csv \
  --out data/djdata/djs/lists/usb002_solo_tracklist.csv
```

and read beside the 1001 scrape through `extra_tracklists` in `config_djs.yaml`. Their tracks are in
`djs/tracks/`: 1001 ids where title and an artist agree, `usb_<hash>` ids otherwise
(`usb002_solo_tracks.csv` maps them). His three 1001 shows proved on their own audio are ordinary mixes
(452b91b4c4, 4560f420f5, and 51f57bd5cc linked to `fred/bWUsbsTUKV4.m4a`). The two 1001 shows whose
audio came from a wrong title search (40caea7a7d, e3249f53e6) are moved to `mixes_tmp/old_searched/`.

**What changed in locate, and what did not.** A tracklist row whose `exact_time` column is 1 carries
the exact minute the record is heard inside the file. Such a record is searched only from 3 min before
to 5 min after it (`locate.NEAR_BEFORE_S`, `NEAR_AFTER_S`), on the mix table cut to that window, with the
three controls matched in the same window. A record not in its window is `not found` there, and its
presence rows still show where it plays. Every 1001 list keeps the whole-mix search. On two dev mixes
the new locate gives all 38 plays rows identical to pass three.

`controls_exclude_own_dj: [Fredagain..]` keeps every record he lists anywhere out of his controls; his
tour repeats its records every night. Every other DJ keeps the controls he had.

**New stages and tables.** None of the tables above changed its columns except `plays.csv`, which gains
`search`, `near_s`, `window_votes`, `window_floor`, `window_control_max` at the end.

```
djdata locate       also writes out/presence.csv and out/mixes.csv
djdata layers       the layer timeline per 10 s                                           -> out/layers.csv
djdata layer-bands  which record carries which band where records stack                   -> out/layer_bands.csv
```

`presence.csv` is one row per record per 30 s window where it beats the loudest control window of the
mix: `present` at twice that (the sweep floor), `weak` between the two, with `record_at_s`, the part of
the record playing there. `layers.csv` counts present and weak records per 10 s and names them;
`same_audio` names two ids of one recording, which are counted once. `layer_bands.csv` is per record and
per band (low, mid, high) for every span of three or more present records, or two present records that
are not a consecutive pair: votes at the record's own alignment, the band floor from three controls on
that span, seconds seen. `mixes.csv` is codec, profile, bitrate, sample rate, length, and the mix's
floors.

**The run, every stage, from the repo root on the VM.**

```
C=djdata_package/config_djs.yaml
PYTHONPATH=djdata_package:. python -m djdata.cli locate --config $C --workers 14
PYTHONPATH=djdata_package:. python -m djdata.cli pairs --config $C
PYTHONPATH=djdata_package:. python -m djdata.cli cut --config $C --workers 16
PYTHONPATH=djdata_package:. python -m djdata.cli measure --config $C --workers 16
PYTHONPATH=djdata_package:. python -m djdata.cli tempo --config $C --workers 16
PYTHONPATH=djdata_package:. python -m djdata.cli label --config $C
PYTHONPATH=djdata_package:. python -m djdata.cli export-seams --config $C
PYTHONPATH=djdata_package:. python -m djdata.cli layers --config $C
PYTHONPATH=djdata_package:. python -m djdata.cli layer-bands --config $C --workers 16
```

`.` on `PYTHONPATH` is needed since `seam/tempo.py` imports the mixer from `src/`.

**The full run, 2026-09-28/29.** Done: `profiling_run.sh` for all stages, then `fixup_2026-09-29.py`
(both in the VM's home folder) for the four things the run left, see `DATASET_STATE.md` 2026-09-29.
The VM's swap is `/swapfile`, 10 GB, not in /etc/fstab: after a reboot run `sudo swapon /swapfile`.
A run script that should leave the VM on must not end in `shutdown`; on 2026-09-29 a tmux session
`noshut` cancelled the fix-up's scheduled shutdown every 10 s for three hours.

**S3 from the VM, 2026-09-29.** The VM's role `aidj-eip` can write to S3 but not delete from it;
deletes go from the laptop with the `talhanonstatic` profile. `aws s3 sync` follows links: syncing
`data/djdata/fred/` uploaded its `tracks` link as 31 GB of duplicate tracks. Use `--no-follow-symlinks`.

**Watch list, added 2026-09-28.**

- Weak is chance-level evidence. A weak window is the best of many offsets, so on Black Coffee, who does
  not stack records, one record read weak at eight places in two hours. Weak never makes a span and is
  never band-read. Judge Fred's weak stacks only against a non-stacking DJ's rate in the same run.
- Fred's cut audits fail more by nature: on NY6 1 of 5 windows started with A alone and 2 of 5 ended
  with B alone, because two records often play together. Read his seams with that in mind.
- One recording can sit under two ids (NY6's two Skepta & PlaqueBoyMax versions). The layer tables
  count them once; `plays.csv` keeps both rows.
- A windowed Fred record costs about 70 s, like a whole-mix one, because a record not in its window
  gets the whole-mix speed search so its presence is read at its real speed.
- Resuming after a crash. Every stage skips what its table already holds: locate by mix (a mix's rows
  are written when it finishes), pairs by mix, cut, measure and label by seam, tempo by record,
  layer-bands by span; layers is rebuilt whole. A crash loses only the items in progress. To resume,
  start the VM and run the same `~/profiling_run.sh` in tmux.
- The run script's weak point (2026-09-28). An out-of-memory kill of one locate worker breaks the
  whole process pool, so every mix still waiting in that run is marked failed and not written. The
  locate stage still ends normally, and `profiling_run.sh` goes on through pairs, cut, measure and the
  rest with the partial set, then syncs and shuts down. No data is lost, only time: after any run, read
  `failed` in the locate line of `logs/profiling_run_2026-09-28.log`, and if it is not 0, rerun the
  script. The next run script stops at the first stage that reports a failed item.
- Memory, measured 2026-09-28 on the VM. A mix's fingerprint table barely grows with its length: 0.68 GB
  for a 1 h mix, 0.77 GB for 2.5 h, peak 1.08 and 1.43 GB while fingerprinting. 14 locate workers held
  13 to 15 GB of the 30 on one to two hour mixes. The corpus's longest mixes are 10.1 h (Solomun) and
  9.75 h (DJ Tennis), and its heaviest stretch is 14 DJ Tennis mixes in a row with 40.6 h of audio,
  estimated about 22 GB. A 10 GB swap file (`/swapfile`, made 2026-09-28) stands behind it, so a spike
  slows the VM instead of killing a worker. It does not survive a reboot unless added to /etc/fstab.
- Storage, 2026-09-28. The VM disk is 193 GB. The old September cuts (`djs/windows`, `windows_v2`,
  `windows_v3`, 12.4 GB, all wrong cuts) were deleted for good on Anas's word, leaving 27 GB free.
  The new seam windows need about 8 to 10 GB (about 3,100 seams, median window about 170 s, 128 kbps).
- 14 locate workers on the VM's 16 vCPUs (8 physical cores, two threads each) keep the load near 14 to
  17, and a record lookup then takes about 85 s instead of the 55 s one worker gives. More workers
  would not help.

## State on 2026-09-25

**What exists.** `djdata_package/djdata` holds the whole chain from mix and track audio to a labelled
seam table, as eight stages:

```
djdata locate       every listed record found in its mix at its speed, presence per 30 s window at
                    any offset (so loops are followed), floor from three control records   -> out/plays.csv
djdata pairs        which record follows which, in audio order, flags against the tracklist,
                    the bounded window, nothing dropped silently                           -> out/seams.csv
djdata cut          one window per usable seam, ffmpeg stream copy, ten second audit at
                    both ends against a control floor                                        -> out/cuts.csv
djdata measure      per band entry and exit, bass swap, bass words, presence, loop, every
                    floor from three controls on that window                                 -> out/measures.csv
djdata tempo        BPM per record from its own audio                                        -> out/tempos.csv
djdata label        the nine transition types, in bars                                       -> out/labels.csv
djdata export-seams one flat table joining all of the above                                  -> out/seams_index.csv
djdata ear-test     clips from the tables, window left ear, records right ear, MARKS.csv
```

**What is proved.** 88 tests on synthetic audio with a known answer (`djdata_package/tests`, run by
`make ci`). Three passes on a five mix development sample on the laptop, numbers in `DATASET_STATE.md`.
And your ear on 2026-09-25: the clip sets from pass three were right, so locate, pairing, cut and labels
hold on real DJ audio. Tempo agrees with the catalog within 2 % on 40 of 48 tracks. The three things the
real passes found, the sweep share, no drift, loops, are written up in `DATASET_STATE.md` in order.

**Known limits.** Measure lays each record at one offset, so a looped outgoing record reads as
`unmeasured` (6 of 70 on the sample). Tempo is off on about one record in eight below 150 BPM and on
most records above it (the hint, see the 2026-10-07 watch-list entry). A record shorter than
135 s cannot be "confident". All three are in the rows, none is guessed over.

**How to run it.** One config per corpus, every stage the same command, every stage resumable: kill it
and rerun the same command. Run from the repo root.

```
PYTHONPATH=djdata_package python -m djdata.cli locate  --config djdata_package/config_djs.yaml --workers 14
PYTHONPATH=djdata_package python -m djdata.cli pairs   --config djdata_package/config_djs.yaml
PYTHONPATH=djdata_package python -m djdata.cli cut     --config djdata_package/config_djs.yaml --workers 16
PYTHONPATH=djdata_package python -m djdata.cli measure --config djdata_package/config_djs.yaml --workers 16
PYTHONPATH=djdata_package python -m djdata.cli tempo   --config djdata_package/config_djs.yaml --workers 16
PYTHONPATH=djdata_package python -m djdata.cli label   --config djdata_package/config_djs.yaml
PYTHONPATH=djdata_package python -m djdata.cli export-seams --config djdata_package/config_djs.yaml
PYTHONPATH=djdata_package python -m djdata.cli ear-test --config djdata_package/config_djs.yaml --out data/external/ear_test/<name> --n 12 [--label loop]
```

Configs: `config_djs.yaml` the DJ corpus, `config_fred.yaml` Fred, `config.yaml` Raveform (its alignment
stands in for locate and its cut windows for the cut; the same commands). On the VM the same commands
run in tmux with the `aidj` conda python; a locate worker holds about 1.5 GB, so locate runs at 14 workers
and the rest at 16. Logs go to `<root>/logs/djdata.log`.

**Where the data is.** The dataset is the tables under `data/djdata/<corpus>/out/`. `plays.csv` is one
row per record play: mix, DJ, order listed, first and last heard, played from and to in record time,
time zero, rate, votes and floor, confidence. `seams.csv` is one row per consecutive pair with the
flags and the window. `cuts.csv` the window file and the audit. `measures.csv` the numbers per seam
with the floor beside each. `tempos.csv`, `labels.csv`. `seams_index.csv` joins them for a notebook:

```python
import pandas as pd
seams = pd.read_csv("data/djdata/djs/out/seams_index.csv")
plays = pd.read_csv("data/djdata/djs/out/plays.csv")
```

Times are seconds into the mix in `plays.csv` and seconds into the window everywhere else. Every
number sits beside the control floor that judged it, so a filter like `measured == 1` or
`a_best_separation > 3` is the first thing to apply. `state.sqlite` is a resume log, not the dataset.
The window audio is `data/djdata/<corpus>/windows/<seam_id>.<ext>`.

**How to add data.** A new DJ is rows in `data/interim/tracklist.csv` plus the mix and track audio in
`mixes_tmp/` and `tracks/` named by id; the stages pick up what is on disk and skip what is done. A new
corpus is a config with its own `root`. A proved Fred segment is a link in
`data/djdata/fred/mixes_tmp/<mix_id>.m4a` to the segment file.

**Where the code is.** Commit ef8c946 on `origin/dev` holds all of it, history cleaned of every
co-author line. `main` is protected (no force-push, PR with lint-and-test required), so its 25 old
commits keep their lines until the rule is relaxed once; the new work reaches main by a PR from dev.
`data/interim/tracklist.csv` (34.5 MB) is still versioned and should be untracked in a later commit;
the pipeline reads it from disk and the VM and S3 have copies.

**Watch list for the VM run, noted while building.**

- Memory. A locate worker holds about 1.5 GB (a two hour mix's hash table). Start locate at 14 on the
  30 GB box and read `free -g` in the first ten minutes; everything else at 16.
- Time. The laptop did 10 h of mix audio in 61 min at 4 workers. The 281 mixes are about 420 h.
- The VM's clone must be reset to the rewritten history before anything runs:
  `git fetch origin && git reset --hard origin/dev`. Then `pip install -e djdata_package` is not
  needed; run with `PYTHONPATH=djdata_package`.
- Partial runs pick other controls (found 2026-09-28). Each mix's three controls are drawn from the
  records of the other mixes in the same run. `locate --mixes A B` builds that pool from A and B only,
  so its controls, floors and first and last heard differ from a full run's. Measured: Amelie Lens
  be77589ff6 got floor 136 instead of 140 and three first or last heard times moved; with the full
  five-mix pool all 38 rows matched pass three exactly. Rerun a subset only into a fresh root, and
  never mix its rows with a full run's.
- Floors. Every row carries `control_n` and the `*_floor_trusted` flags. A corpus with few mixes gives
  thin floors; filter on them before using a number. Controls are chosen per mix by a seed from the
  mix id, from the corpus's own records.
- Seams set aside. `seams.csv` keeps every pair with `usable` and `reason`. On the sample 28 of 98
  were gaps over 120 s with both records confident, records not found, or listed records between.
  Read the reasons before changing `pairs.MAX_GAP_S`.
- Cut audits. `audit_end_ok` failed on 6 of 70 windows on the sample: the outgoing record was still
  there at the end. Look at those rows first when checking a run.
- Loops in measure. A looped outgoing record is `unmeasured` (6 of 70). A loop-tolerant measure, per
  band free matching of the record slice per step inside the window, is the next improvement.
- Tempo. The hint is the weak part, not the instrument: librosa's tempo with `start_bpm=128` reads
  drum and bass at half or two thirds, and the refinement cannot leave a 6 % band around the hint
  (Raveform check 2026-10-07, `EMBEDDINGS.md`: 28 % right above 150 BPM, 97 % in 115 to 135). On a
  corpus with fast genres, give `tempo.py` a better hint or check against an annotation.
- Short records. Under 135 s a record cannot be "confident" (three 45 s sections cannot agree).
- Presence constants. Windows of 30 s stepping 10 s, floor twice the loudest control window, first and
  last heard at the first and last second holding 3 agreeing pairs. All in `seam/locate.py`.
- This laptop's network drops long uploads (a push sat a day on a dead connection, scp to the VM ran at
  6 KB/s). Move anything large through S3, and expect a stalled push to need a retry.
- Fred on the VM needs the same links as the laptop: `data/djdata/fred/mixes_tmp/<mix_id>.m4a` to each
  proved segment and `data/djdata/fred/tracks` to `../djs/tracks`.
- The old untracked `scripts/diag` code was deleted on 2026-09-30 and archived at
  `s3://aidj-1/archive/scripts_diag_untracked_2026-09-30.tar.gz`. `fetch/mix.py` still imports the
  legacy locate.

**What is next.** Fred's segments proved by locate on the VM, then the full run over the 281 DJ mixes
and Raveform on the VM, then the old `scripts/diag` code goes. The section below is the state before this.

