# TODO (2026-10-15)

What is agreed to build, in order. How each one is built gets decided when it is started, not here.
Numbers next to an item say what was measured when it was written down and how.

## 1. What to play, how long, from where

The measured sources hold `played_from_s` and `played_to_s` per play. Start as a measurement: hold
length relative to the record, where the cue-in lands, per genre and per DJ. Then a predicted play
window per record replaces the genre constant in the mixer and the fetcher's 240 s duration floor.

## 2. Club / festival / live versus radio setting

A flag on the mix request. Background listening gets the radio data in the pool, the planner and the
mixer settings; a party set gets the club and festival data. Measured 2026-10-15 over 3,937 real
mixes, split by a title rule (show name or number means radio; "@", venue, festival, club means live):
radio 713, live 1,270, both 351, unknown 1,603. The unknown ones are mostly the djs source, whose
mix title is just the DJ's name; its YouTube title would settle them. Key clashes, same-key share,
energy level and BPM climb are the same on both sides. Hold length per record is shorter on live
sets within every house and techno genre (house 2.70 min live against 4.40 radio, techno 3.47
against 4.41, trance 2.97 against 3.77, tech house 3.95 against 4.38; drum and bass the other way,
3.20 against 2.92). Table and the rule: `data/external/ear_test/sets_2026-10-15/mix_type_by_genre.txt`.
Pieces: a `mix_type` column on plays and seams at export, a pool filter like `--dj`, play length per
genre per setting, transition type per setting once item 3 exists, the pair model only if radio and
live pairs rank differently.

## 3. Transition type from data

Same shape as the pairing work: a yardstick on the measured seams, the rule table as the reference
row, a small model beside it, then compared with the mixer's rules. First look at what the seams and
layers tables record per seam before promising what it can learn.

Measured 2026-10-15, all numbers in `TRANSITION_EXPERIMENTS.md`. From the two records' features the
transition type is not predictable on unseen DJs (0.41, the majority), the mixer's rules match the
DJ's real choice on 0.25 of seams, and the DJ's own most common type gives 0.51. The state of both
records at the cue points adds nothing in techno and a signal across DJs in house that does not hold
inside Claptone. Agreed shape: per-DJ and per-genre tables of type shares and recipe distributions
(never medians), types drawn for the whole set and placed on seams that can carry them, the mixer's
rules as the veto, the mixer's own rendering. Open: whether consecutive seams depend on each other.
Tension is 20 % of measured seams and loop 63, so item 5's first question has an answer in the data.

## 4. DJ profiling

The per-DJ version of items 1, 3 and the pairing model, so it comes after they exist as measurable
things. Likely shape: a learned DJ vector into the pair model, per-DJ distributions for length and
transition type, genre as the fallback for a DJ with few sets. Most DJs have a handful of sets, so
profiles will be thin.

## 5. Gimmicks (loops for tension, bass, how much and when)

Starts with a question, not a build: can loops and tension builds be detected in the DJ mixes held?
If the data cannot show when DJs do it, nothing can be learned from it.

## 6. The planner's fetch uses the data pipeline's fetch

Anas, 2026-10-15: the fetch the data pipeline ran for its tracks is the more robust one, so the
planner and mixer use its logic when they fetch full audio. Today `src/data/track_fetcher.py`
searches YouTube once by one query, ranks on title overlap, then gates on duration, bad title words
and the fingerprint. `djdata_package/djdata/fetch/track.py` builds on the same gates and adds:
several search queries per record (`names.search_queries`), SoundCloud searched next to YouTube,
name matching that checks the artist is present and the remix or version agrees
(`names.same_record`, `artist_present`), the closest title among the top three results first, and
the block gate with cookies and the player client (`fetch/yt.py`) that got the VM through YouTube's
datacenter block. The held and link paths in the planner's fetcher stay. The search branch of
`fetch_track` calls `download_by_name` with the pipeline's config, and the fetcher's own search,
rank and download code goes.

## Ear test 2026-10-15: the cases items 1 and 3 must get right

Seven chill sets from held audio, `data/external/ear_test/sets_2026-10-15/` (one flac and one
plan.json with entry times per genre, the mixer's log in `driver.out`). Anas on melodic house:
track selection fine, the cue-in points and the transition types need work, and every set starts
too low on energy.

- **6:33 to 7:04, Human Oscillator → Ravel, heard as wrong.** The rules chose `fade` because the
  energy drops over the join (0.265 → 0.209), and a fade filters the outgoing record's mids and
  highs away. Same key (C#m → C#m), same BPM, the loudest record of the set (−8.8 LUFS) played for
  only 80 bars. Anas: land on the drop, slam into a drop, or a long melt. A melt needs |Δe| < 0.05
  under the rules, so it was not allowed.
- **9:34 to 10:05, Ravel → Memories, heard as a track change.** A 16-bar blend (the chill gate
  demoted the measured `rise`), cued into Memories at bar 48, past 1.5 min of its intro, with the
  overlap gain ride stepping −8.5 dB in and the seam decided by the consensus band with a 242 ms
  shift and 62 ms of drift.
- **13:40 to 14:11, Memories → Paper Walls, heard as perfect.** Also a 16-bar `fade`, −4.2 dB in,
  kick-decided seam with no shift. So the type is not wrong on its own; the record, the cue points
  and the level ride decide it.
- **Starting energy too low.** The chill curve starts at 0.15 of the pool's 15th to 85th percentile
  range, so the opener sits at about the pool's 15th to 20th energy percentile. Real sets open at the
  pool's 38th to 53rd percentile by genre (median over 3,709 sets; lower quartile 13 to 28), with the
  set's median at the 50th to 56th and its loudest record at the 89th to 95th. A mellow set in the
  data is a set that opens near the pool's middle and peaks lower, not one that stays in the bottom
  fifth.

## Data fixes found by the ear-test plans (2026-10-15)

- **Duplicate records under two artist spellings.** "Go Around" is held as Bongi Dude and Bongi Dube,
  identical audio (cosine 1.0). 31 such pairs among the 12,153 held records (same title, same BPM,
  embedding cosine over 0.97). The artist veto cannot see them. Dedupe at export.
- **Radio-show prefixes in titles.** 706 held records read like "Listener's Choice: Lustral –
  Everytime" or "[11:36] Billie Ray Martin", raveform house and trance mostly; one artist field is
  mis-split. Audio is right, metadata is dirty. A title cleaner at export.
- **The mixer spells the genre "drum and base"**, the dataset says "drum and bass", so the dnb play
  length, phrase bars and halved overlaps never apply. One line in `src/audio/audio_mixer.py`.
