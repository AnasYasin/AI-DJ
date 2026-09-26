# DATA_PIPELINE, how to run the seam pipeline and use its tables

Moved from START_HERE.md on 2026-09-27, content unchanged. What was built, what it is proved by, how to run it, where the data lands, how to add data, and the watch list for a run. The numbers and the findings behind it are in DATASET_STATE.md; the layout in DATA_ARCHITECTURE.md; the code in djdata_package with its own README.

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
`unmeasured` (6 of 70 on the sample). Tempo is off on about one record in eight. A record shorter than
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
- Tempo. One record in eight disagrees with the catalog by more than 2 %, spoken word among them.
  Bars on those seams are wrong; `tempos.csv` has the number per record.
- Short records. Under 135 s a record cannot be "confident" (three 45 s sections cannot agree).
- Presence constants. Windows of 30 s stepping 10 s, floor twice the loudest control window, first and
  last heard at the first and last second holding 3 agreeing pairs. All in `seam/locate.py`.
- This laptop's network drops long uploads (a push sat a day on a dead connection, scp to the VM ran at
  6 KB/s). Move anything large through S3, and expect a stalled push to need a retry.
- Fred on the VM needs the same links as the laptop: `data/djdata/fred/mixes_tmp/<mix_id>.m4a` to each
  proved segment and `data/djdata/fred/tracks` to `../djs/tracks`.
- The old `scripts/diag` code is untracked and untouched; `fetch/mix.py` still imports the legacy
  locate. Both go after the VM run has been heard.

**What is next.** Fred's segments proved by locate on the VM, then the full run over the 281 DJ mixes
and Raveform on the VM, then the old `scripts/diag` code goes. The section below is the state before this.

