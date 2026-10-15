# START_HERE as it stood on 2026-09-24, moved whole to the archive on 2026-09-27

Superseded by DATASET_STATE.md and DATA_PIPELINE.md. Kept because it records how the DJ corpus stood before the pipeline was built.

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

