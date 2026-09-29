# Fred again.. sets — sources found 2026-09-20 and how to process them

Anas found these by hand after the 1001 scrape came back empty for all 11 Fred mixes. They do not fit the
normal pipeline, which expects one mix page carrying one audio link. These need cutting and matching
before they become seams. They are worth it: Fred is the weakest DJ in the profiling set and these
recover most of him.

**Do this at the end.** Order agreed 2026-09-20: seam cutting and track marking on the DJ-profiling data
first, then these sets, then the grand feature extraction.

## What we hold for Fred already

439 of 532 track endpoints are on disk (82.5%). Per mix:

| mix_id | date | tracks | on disk | seams | set |
|---|---|---|---|---|---|
| 999b60e3e1 | 2025-10-09 | 25 | 16 | 11 | GIMIC Radio + Bar, Bruxelles |
| cea5b407bc | 2025-10-24 | 76 | 69 | 65 | Halle Tony Garnier, Lyon |
| e67ef9dfaf | 2025-11-01 | 50 | 47 | 44 | RDS Simmonscourt, Dublin |
| 51f57bd5cc | 2025-11-14 | 13 | 9 | 9 | YZD Hanger 5 Toronto (Twitch) |
| 56bf970778 | 2025-11-14 | 33 | 29 | 28 | YZD Hanger 5 Toronto |
| f0015906a0 | 2025-11-15 | 70 | 63 | 60 | YZD Hanger 5 Toronto, w/ Four Tet |
| 9b01f9fa29 | 2025-11-21 | 60 | 47 | 43 | Navy Pier Chicago, w/ Sammy Virji |
| e7f707d059 | 2025-11-21 | 16 | 14 | 14 | Twitch, w/ Sammy Virji |
| c55b5a562e | 2025-11-28 | 60 | 52 | 49 | Vancouver Convention Centre, w/ Skream & Benga |
| 355372ab9f | 2025-12-05 | 97 | 75 | 61 | Cow Palace, w/ Oppidan & Hamdi |
| c339b11625 | 2025-12-05 | 32 | 18 | 14 | HydeFM (Twitch) |

The tracklists and timings are already in `data/interim/tracklist.csv`. Only the mix audio was missing.

## The four sources

### 1. Two standalone sets, tracklist not yet matched

```
https://www.youtube.com/watch?v=3J74qwNuMyo
https://www.youtube.com/watch?v=bWUsbsTUKV4
```

Which 1001 mix each belongs to is **not established**. Match by fingerprinting two or three tracks from a
candidate tracklist inside the audio, the same proof used everywhere else. Do not match on the title.

### 2. Vancouver, tracklist matched, chapters marked

```
video     https://www.youtube.com/watch?v=j9EkAYdouyM
1001      https://www.1001tracklists.com/tracklist/2dy4m0j9/fred-again..-skream-benga-usb002-vancouver-convention-centre-canada-2025-11-28.html
mix_id    c55b5a562e     60 tracks, 52 on disk, 49 seams
```

This one is ready to use. The video also carries its own chapter marks, which are a second, independent
view of where the set changes gear:

```
0:00     Live stream intro            44:29    Peak set and technical focus
1:58     Opening set                  59:35    Melodic tracks and fan shoutouts
7:19     High energy club mix         1:10:09  High energy buildup
15:05    Genre blending and samples   1:20:35  Final track selection
26:04    Live mixing and vocals       1:33:51  Closing the set
33:48    Drum and bass session        1:47:33  Stream wrap up
```

Those are YouTube's own auto-generated section names, so treat them as a hint about structure, never as
ground truth. The fingerprint stays the measurement.

### 3. The USB002 marathon — the big one

```
https://www.youtube.com/watch?v=GiXKukOtmeE     about 107 hours
```

A single continuous upload of the whole USB002 run, city by city, with every artist's segment timestamped
in the description. Fred plays roughly 40 of those segments, alone or b2b. Everyone else's segments are
not wanted.

**Why this matters more than it first looks.** The city markers line up with the very 1001 tracklists whose
audio we could not find. If that holds, this one video is the missing audio for most of the eleven.

| marker in the video | starts | likely 1001 mix | mix_id |
|---|---|---|---|
| BRUSSELS | 4:48:00 | GIMIC Bruxelles 2025-10-09 | 999b60e3e1 |
| LYON | 17:50:00 | Halle Tony Garnier 2025-10-24 | cea5b407bc |
| DUBLIN | 22:47:00 | RDS Simmonscourt 2025-11-01 | e67ef9dfaf |
| TORONTO 1 | 28:04:00 | YZD Hanger 5 2025-11-14 | 56bf970778 |
| TORONTO 2 | 33:37:00 | YZD Hanger 5 w/ Four Tet 2025-11-15 | f0015906a0 |
| CHICAGO | 38:08:00 | Navy Pier w/ Sammy Virji 2025-11-21 | 9b01f9fa29 |
| VANCOUVER | 43:02:00 | Vancouver Convention Centre 2025-11-28 | c55b5a562e |
| SAN FRAN | 49:07:00 | Cow Palace w/ Oppidan & Hamdi 2025-12-05 | 355372ab9f |

**This mapping is a hypothesis, not a fact.** The description names cities, not dates, and the USB002 tour
may have played a city more than once. It is settled the same way as everything else, by fingerprinting a
few tracks from the candidate tracklist inside that stretch of audio. GLASGOW, MADRID and the six NY and
four LDN blocks have no 1001 tracklist in our list at all.

## How to process the marathon

**Audio only.** At roughly 130 kbps the whole 107 hours is about 6 GB, which fits the VM's free space
several times over. The same run as video would be hundreds of GB. That figure is an estimate from the
bitrate the fetcher normally gets, not a measured size; check it with `yt-dlp -F` before starting.

**One download, not forty.** `yt-dlp --download-sections` can pull a single stretch, but forty ranged
requests against one video is both fragile and conspicuous. Fetch the audio once with `-c` so it resumes,
then cut locally with ffmpeg. Cutting is free once the file is down.

**Take the description with it.** `--write-description` gives the exact timestamps from the source. The
segment list below was transcribed from screenshots and **must be replaced by that file before any cut is
made.** A minute of transcription error puts every seam in the wrong place.

**Then the usual proof.** For each Fred segment, run `seam/locate.py` over the tracks we already hold for
the matching 1001 tracklist. A track that hits gives its position and speed, and two adjacent hits give a
seam. Segments with no matching tracklist can still be mined the other way: search our whole Fred track
library inside them and keep what lands.

**What is different about these, and why the normal path does not fit.** Everywhere else one mix page
gives one audio file and the tracklist covers all of it. Here one file holds dozens of sets by different
artists, the tracklist covers only part of it, and the offset between the tracklist's own clock and the
video's clock is unknown until the fingerprint finds it. So the order is cut first, then locate, then
seam. Never assume the tracklist's `starting_time` and the video's clock share an origin.

## Fred's segments in the marathon — PROVISIONAL, transcribed from screenshots

End of each segment is the start of the next entry in the full list. Verify against the downloaded
description before use.

```
0:00:00   GLASGOW - Fred                                   43:02:00  VANCOUVER (Lou Nour)
0:15:25   Fred b2b Rick Salad                              45:23:00  Fred b2b Skream & Benga
2:42:00   Fred b2b HAAi b2b ¥ØU$UK€ ¥UK1MAT$U              46:53:00  Fred
3:10:00   Fred                                             52:15:00  Fred b2b Oppidan b2b Hamdi
7:44:40   Fred                                             53:07:00  Fred
13:59:00  Skin On Skin b2b Fred                            55:33:00  NY1 - Fred b2b Romy b2b HAAi
15:02:00  Fred                                             57:32:00  Fred
17:50:00  LYON - Fred b2b Floating Points b2b Caribou      59:29:00  Fred b2b Romy b2b HAAi
20:54:00  Fred                                             62:27:00  Fred again.. b2b Oppidan b2b Hamdi
21:52:00  Fred b2b Floating Points b2b Caribou             63:57:00  Fred
25:00:00  Fred                                             68:13:00  X3butterfly b2b Doctor Jeep b2b Sister Zo b2b Fred
28:04:00  TORONTO 1 - Fred                                 69:18:00  Fred
30:53:00  Fred                                             73:59:00  Fred again.. b2b Caribou
33:37:00  TORONTO 2 - Fred b2b Four Tet                    75:59:00  Fred
38:08:00  CHICAGO - Fred b2b Sammy Virji                   76:56:00  Fred again.. b2b Caribou
40:34:00  Fred                                             79:34:00  Fred
42:02:00  Fred b2b Sammy Virji                             82:15:00  NY6 - Fred again.. b2b Ben UFO b2b Caribou
                                                           85:13:00  Fred
87:41:00  Hamdi b2b Oppidan b2b Fred                       86:21:00  Fred again.. b2b Ben UFO b2b Caribou
89:23:00  Fred b2b Nia Archives                            98:39:00  LDN 3 - Lou Nour b2b Fred
90:14:00  Fred                                             99:42:00  Fred again.. b2b Skream b2b Benga b2b Coki b2b Mala
92:42:00  LDN 2 - MPH b2b JOY b2b Femi b2b Fred            101:36:00 Fred
104:12:00 LDN 4 - Fred                                     106:33:00 Fred b2b Thomas Bangalter
```

## The exact steps to start the marathon download

Agreed 2026-09-20 and deliberately deferred, because the seam work on the DJ-profiling data comes first.
Run these when picking it back up.

1. **Start the VM.**
   `aws ec2 start-instances --region us-east-1 --instance-ids i-023e6cd5518feff75`
   Its Elastic IP changes on every YouTube block (the gate rotates it), so use `scripts/vm/vmssh`, which looks the
   current address up and fixes the `aidj` alias before connecting (2026-09-24).
   The instance is a c6i.4xlarge and bills while it runs, so the download and the seam work should share
   the same uptime rather than each paying for its own.

2. **Check the real size before committing to it.**
   `yt-dlp -F "https://www.youtube.com/watch?v=GiXKukOtmeE"`
   One request, nothing downloaded. Read the bitrate of the best audio-only format and multiply by the
   duration. The 6 GB figure in this file is an estimate and has not been measured.

3. **Download audio only, resumable, in tmux.** Use the project's cookie file and the usual player client,
   the same as `fetch/yt.py` does, because a datacenter address without cookies gets refused.
   `-f bestaudio -c --write-description`. No re-encode.

4. **Cut from the downloaded description, never from the transcribed table above.**

A 107-hour download is long enough that it should be started first and left alone. It needs no attention
while it runs, which is the whole point of doing it beside the seam work.

## Open questions

- Which 1001 mixes `3J74qwNuMyo` and `bWUsbsTUKV4` are. Unmatched.
- Whether the city markers really are the eleven tour dates. Unproven.
- Whether a b2b segment is usable at all. Two DJs alternating is not one DJ's phrasing, and the transition
  model is meant to learn how one DJ moves between records. Fred's solo segments are the safe ones. Anas
  to decide whether b2b stretches go in the training set or only the solo ones.

---

# What was measured 2026-09-21/22

Everything above this line is as written on 2026-09-20 and parts of it are now settled or wrong.

## Vancouver is proved. The other ten cities are not

`c55b5a562e` (Vancouver Convention Centre, w/ Skream & Benga) **is** `fred_19_vancouver_fred.m4a`.

Ten tracks from that 1001 tracklist were tested against three marathon segments, with three Roman Flügel
tracks as a wrong-record control. Four of Fred's own records landed in the Vancouver segment **in exact
play order**:

```
play order   #24      #28       #32       #36
found at    4.1 min  18.8 min  31.1 min  57.1 min
votes         464     2,645     3,154       406
```

Dublin and Lyon each hold one or two of the same records at unrelated times, which is the tour repeating
its own USB.

## Vote count cannot map his shows. Play order can

The first attempt matched by vote count and **failed**: every test track cleared the floor in every
segment, including segments it had no business being in. `scripts/diag/fred_match_segments.py` is the
script that does this and it should not be used.

`scripts/diag/fred_match2.py` judges by play order instead, and that is the one that works. A record that
merely repeats elsewhere on the tour lands at an unrelated time, so a wrong segment scatters while the
right one rises.

## The vote floor in the code sits under the noise on his audio

The three Roman Flügel control tracks, which Fred never played, scored **43, 47 and 58 votes** in every
segment. `MIN_VOTES` in `locate.py` is 40. So on his material everything between 30 and 80 votes is
noise that the code would accept.

On Raveform's audio the same control sits at 13. Any measurement of Fred needs a floor derived from a
control on his own audio, which is what `scripts/diag/floors.py` does.

## Three of the four sources are 48.7 kbps

```
GiXKukOtmeE  USB002 marathon   108.53 h   48,689 bps  HE-AAC   format_id 18, asr 22050
3J74qwNuMyo  Week 4 DJ set       4.88 h   97,379 bps  AAC-LC
j9EkAYdouyM  HydeFM SF           1.88 h   48,690 bps  HE-AAC
bWUsbsTUKV4  Canada Twitch       1.29 h   48,692 bps  HE-AAC
```

That is the lowest audio YouTube serves. The project's normal fetch gets about 130 kbps opus, and 126 of
the 175 DJ-profiling mixes are at that quality. The `yt-dlp -F` check in step 2 of the download
instructions above would have caught this before 108 hours were pulled.

Whether 48.7 kbps is good enough to measure bands and bass on is **not known**. The control floor being
four times higher than normal is the first sign that it may not be.

## Only the solo segments go into profiling

Anas, 2026-09-21. `segments.csv` carries `is_b2b`. **18 solo, 24 b2b.** Two DJs alternating is not one
DJ's phrasing, so the b2b segments are excluded from his profile. They stay on disk and can go into the
general transition pool.

## His tracks are the hardest of any DJ

```
wrong-download rate, Fred          38%
everyone else                      4 to 17%
```

The cause is that he plays his own edits. In the segment test the commercial releases never matched:
Prodigy, Charli xcx, Usher, Clipse and Lil Yachty all sat in the noise, while only Fred's own releases
scored. **Four of ten tracks per show is the realistic yield** from the fingerprint path.

## Nothing has been built yet

**Zero seams exist for the eleven tour mixes.** All eleven are `pending` in `state.sqlite`. The 42
segments are loose files in `fred/segments/` joined to nothing in the database. The 49 Fred seams that do
exist come from four other mixes.

Start times: 10 of the 11 have them on 78 to 99% of tracks. **Dublin `e67ef9dfaf` has 6%**, so it is the
worst one to test against, which I did not know when I picked it first. `starting_time` is minutes past
the hour and rolls over hourly, not seconds from the mix start.

## Order of work, when he is picked up

1. Prove each remaining city by play order against a wrong-record control, `fred_match2.py`. Minutes each.
2. Derive the vote floor from that control rather than using the flat 40.
3. Register the proved segments as mix audio in `state.sqlite` so the seam stage can see them.
4. Cut with the bounded rule, solo segments only.
5. Expect to lose most of the commercial titles and keep his own records.

---

# 2026-09-28, where he stands

Everything above this line is as written before and parts of it are superseded here. The numbers are in
`DATASET_STATE.md` under 2026-09-28, the code and the run in `DATA_PIPELINE.md` under the same date.

- **His tracklists are the USB002 app's, not 1001's.** usb002-tracklist.app lists every set of the
  marathon with the second each track is heard, and its set times equal our segment cuts. Most 1001
  city lists do not describe the marathon's solo audio; only Toronto 1, Toronto 2 and the Lyon b2b match.
- **All 17 solo segments with a list are proved** by the windowed check, 26 to 70 % of their records
  found at the app time against controls in the same windows. San Francisco's solo set has no list.
- **Proved on their own audio:** 452b91b4c4, 4560f420f5, and 51f57bd5cc (the Canada Twitch video
  `bWUsbsTUKV4`). **Wrong audio:** 40caea7a7d and e3249f53e6, moved out of the corpus.
- **The b2b segments are not in the profile**, as decided on 2026-09-21.
- **He layers two and more records at once** (Anas's Shazam check of ten misses). His profile is read
  from the layer tables as much as from the seams.
- **He is part of the DJ profiling run**, as mixes `usb_<segment>` in `djs/`, not a corpus of his own;
  `config_fred.yaml` and the separate `fred` root are no longer used for the run.

**2026-09-29.** In the full DJ run: 649 of his listed records on disk, 376 found (57 %), 266 seams, 165
usable, 113 measured with both records clear, 46 stacked spans band-read. Numbers in `DATASET_STATE.md`.
