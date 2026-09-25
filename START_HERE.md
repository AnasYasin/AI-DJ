# AI-DJ, start here, 2026-09-25

A prompt turns into a real, beat-matched DJ mix. The mixer is deterministic and ear-verified. The learned
parts choose the track, the length, the move and its parameters, trained on how real DJs actually mix.
Nothing learned replaces a rule until it beats it in a blind listen.

---

## Where to look for what

| you want | read |
|---|---|
| the state of the data, what is measured, what is blocked | **`DATASET_STATE.md`** |
| every store, every CSV, how the ids join, what lives where | **`DATA_ARCHITECTURE.md`** |
| the mixer, the models, the transition rules, the gotchas | **`CLAUDE.md`** |
| the roadmap and its ten steps | **`PLAN.md`** |
| how to run the thing end to end | **`README.md`** |
| Fred again, a separate job | **`FRED_AGAIN_SETS.md`** |
| how to set the VM up, cookies, yt-dlp, the token server, the changing address | **`djdata_package/VM_SETUP.md`** |
| ssh to the VM whatever its address is today | `scripts/vm/vmssh` |
| old session records and superseded notes | `../docs/notes-archive/` |

---

## The three datasets

| | what it is | state |
|---|---|---|
| **old dataset** | 2,852 mixes scraped from 1001tracklists, a 30 s iTunes preview per track, features and models built from those previews | complete, on S3 |
| **Raveform** | 5,062 seams with the mix audio and both full records, 6,579 tracks | **good.** Cut, alignment, bands and bass all verified against controls. Never measured at scale |
| **DJ profiling** | six DJs Raveform lacks, plus Fred again on his own track. 281 mixes, 4,414 tracks | **audio settled, nothing measured yet.** Every mix now comes from the link on its own 1001 page. The tracks have not yet been checked against their mixes |

Never call the old dataset "the catalog". Raveform's tier 1 and tier 2 are one dataset; the split was run
order only.

---

## The chain, and where it breaks

Nothing is independent. Each thing needs the one below it.

```
feature extraction
   needs  transition reading
             needs  band reading, bass reading, gimmick reading
                       needs  the transition correctly identified
                              AND the correct records used in it
                                 needs  the record located in the mix by its anchor
                                        needs  the right audio file for that record
```

**Raveform is resolved to the top.** Its track id is the YouTube video id, so there is no wrong-recording
risk, and its position comes from its own DTW alignment, which lands within 0.06 s. Measured on 30 seams,
records beat their own noise floor by 15 to 67 times.

**DJ profiling broke one level lower than the chain shows, and that is fixed.** The MIX files were the fault.
All of them had been chosen by a YouTube title search that was never checked, and many were other sets,
radio shows or clips. Since 2026-09-24 every DJ mix's audio comes from the link on its own 1001 page, 281
mixes. On a 35-mix sample of those, 29 were their set with the listed minute right to within about a minute.
The 4,414 track files were fetched by name with the rules in `djdata/fetch/names.py` and have NOT yet been
checked against their mixes. That check is the whole-mix locate, and it is where work stands.

---

## 2026-09-25: the profiling pipeline is built, run, and heard

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

**What is next.** Fred's segments proved by locate on the VM, then the full run over the 281 DJ mixes
and Raveform on the VM, then the old `scripts/diag` code goes. The section below is the state before this.

## The next thing to settle (as of 2026-09-24)

**Are the DJ track files the records in their mixes.** `scripts/diag/mix_locate_once.py` fingerprints a mix once
and looks every listed track up in it, with the vote floor derived from three control records per mix. Its
positions agree with the run's locate to 0.5 s and it is exact on a synthetic 3% stretch. Before it runs over
all 281 mixes, `scripts/diag/locate_lr_test.py` renders ten clips from a random 12-mix sample, mix in the left
ear and the located record in the right, spanning records above, at and below their floor. Anas listens.
A "not found" in sync means the floor is too high; a "found" out of sync means it is too low. Clips land in
`data/external/ear_test/locate_lr_2026-09-24/` with a MARKS.csv.

Only after that verdict: locate over all 281, then the bounded recut, then measurement. No window is cut before.
The full account of the locator, its checks, its numbers and its pitfalls is the 2026-09-24/25 section of
`DATASET_STATE.md` under DJ PROFILING. Read it before touching the DJ corpus.

## Behind it, in order

```
1  the locate ear test                    ten clips, needs Anas
2  locate over the 281 DJ mixes           ~6 h at 8 workers on the VM
3  D6, the ear marks                      blocks everything learned, needs Anas
4  run the measurement on Raveform        needs nothing, about six hours
5  bounded recut of the DJ seams          minutes of ffmpeg, after 2
6  the 25 tracks nobody can find, Fred    parked
```

**D6 is the real blocker for anything learned.** Entry, exit and bass swap have measured values on both
corpora and nothing to score them against. In `ear_test/raveform_mapping/MARKS.csv` and
`raveform_mapping_genres/MARKS.csv` the mark columns are empty on all 45 files. There is a third set,
`ear_test/label_check/` with 93 seams, not yet examined.

---

## How Anas works, follow this

- **Say what you are about to do and wait for a go.** Even in auto mode, even inside a job he approved.
  Running things to "see if they work" has cost real time and got a site to block us.
- **Measure first, act second.** Never the reverse. A threshold is a measurement, not a constant.
- Report results in chat. He writes PLAN.md himself unless he asks for a section.
- Plain words in everything he reads: column names, files, variables. No `rf_b_in_s`. Times as 1:45.
- His ear is ground truth. No automatic check counts as proof.
- Bulk audio work runs on the VM where the files are, not over his home connection.
- Do not lose data: the sets were already trimmed once, so anything skipped gets written down and a list
  saved.
