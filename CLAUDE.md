# AI DJ — Claude Context

## Where things are documented

START_HERE.md is the index. It gives the basic state of the project and points to the md file for
each kind of work: the data files for data work, this file and PLAN.md for the mixer and planner,
and any new file for new work, for example model fine-tuning or retraining, which gets its own md
linked from START_HERE.md. Read START_HERE.md first, then the file it points to for the work at hand.
Detail lives in the specific file, never in START_HERE.md.

## Logging work

Whatever the session works on gets recorded in the md file for that area, dated, with the numbers
and how they were obtained. A new area gets a new md and a line in START_HERE.md. The plan in
PLAN.md is Anas's and is not rewritten; a dated status paragraph under the relevant section is fine.
Ask before touching any md file, saying what will be added or removed.

## Keep the md files lean

An md file is read into context at the start of work, so stale notes cost context and mislead.
When a section is superseded, delete it or move it to docs/notes-archive/ with its date. Keep
history only where it stops a mistake being repeated, for example a closed list of what was tried
and failed. Numbers stay with how they were obtained. Nothing is summarised away: move it whole
rather than shorten it. Ask before deleting or moving.

## Memory notes for the next session

Memory is how the next session knows what this one learned. Write a note whenever something is
settled that the code and the docs do not carry, and always when Anas says "save the session".
Each note is one fact, dated, in the memory folder, with a line in MEMORY.md. What goes in:

- What was tried, what worked and how, and what did not work and why, so it is not tried again.
- Decisions Anas made and the reason he gave.
- What is running or pending, where its output lands, and what to do with it when it finishes.
- What to check first next time: rows to inspect, numbers to compare, limits to keep in mind.
- Pitfalls hit this session, in the tools, the machine, the network or the data.
- Pointers: which md file holds the detail, which folder holds the clips or the tables.

Not what the repo already records. If it is in the code, the git history or an md file, point to it.

## When Anas says "save the session"

He will continue in a new session. In order, reporting each step:

1. Update the md file for the area worked on, and the pointer or status line in START_HERE.md if the
   state changed. Ask first, saying what will be added or removed.
2. Write the memory notes as above, and update MEMORY.md.
3. Say what is still running, what is uncommitted, and whether the VM is stopped.

Natural-language prompt → a real, beat-matched, stereo DJ mix.
Architecture diagrams: `docs/architecture/ai-dj-component-map.html`. Working notes: `PLAN.md`.

## Environment
`conda activate aidj` (Python 3.10). EC2 for GPU feature extraction (g4dn/g5.xlarge, T4).

## One command
```bash
python -m scripts.make_mix --genre techno --bpm 134 142 --curve peak \
    --n 6 --minutes 15 --min-energy-pct 65 --compat-weight 5 --out mix.flac
```
plan → fetch + verify → replan around unfetchable tracks → render. Stages are also usable alone:
`predict_model` (--json plan.json) → `track_fetcher` (--plan) → `audio_mixer` (--plan --tracks-dir).

## Implementation status
```
Phase 0-6  ✅ scrape, previews, features, ChromaDB, transition labels, Airflow DAG
Phase 7    ✅ src/data/audio_segmenter.py    structural segmentation
Phase 8    ✅ src/models/predict_model.py    beam-search planner
Phase 8a   ✅ src/data/track_fetcher.py      plan → verified full audio
Phase 8b   ✅ src/audio/audio_mixer.py       stereo render
Phase 9    ▶  mix_profiler.py done; dj_profiler.py + dj_profiles.json NOT built
Phase 10      intent parser + FastAPI — NOT built. No NL input; everything is CLI args.
```
Also missing: Model C transition critic, mix-audio corpus (5,652 URLs collected, unused),
learned cue points, overlays ("w/" tracks).

## Key data files
- `data/processed/tracklist_clean.csv` — 2,834 mixes, 29,393 tracks, 6 genres, `starting_time`
- `data/processed/features.parquet` — 28,460 tracks: `embedding`[1280 discogs-effnet],
  `embedding_proj`[128, written back by train_model], bpm, key, energy_mean, loudness_lufs,
  onset_strength (RAW — see below), mfcc_0..12
- `data/raw/previews/` — 28,460 × 30 s iTunes m4a. Ground truth for fetch verification.
- `data/processed/split_mixes.csv` — mix-level 15% val, frozen
- `data/interim/segments/` — segment JSON cache, keyed on path+size+mtime (20.1 s cold → 0 s warm)
- `models/` — contrastive_encoder.pt + encoder_norm.npz, sequence_model.pt, edge_gbm.pkl
- `data/processed/chromadb/` — built by vector_store.py, **not read at inference**

## Models
| | | |
|---|---|---|
| A | `train_model.py` | MLP 1287→256→128, NT-Xent. Val adjacency AUC **0.663** (raw 0.622, BPM 0.624). Inference must z-score with `encoder_norm.npz`. |
| B | `train_sequence.py` | Causal transformer 947k, genre token, InfoNCE. **+17% MRR** over context-free. |
| GBM | `edge_scorer.py` | 7 pair features. Val AUC **0.69**. |

discogs-effnet frozen (beat MERT 1.04x / MuQ 1.175x / CLAP on a 200-track signal-ratio test).
BPM from DeepRhythm. Key from essentia `edma`. Transition classifier dropped — the rules are applied
directly at render time.

## Feature extraction (split GPU/CPU, both resumable)
```bash
python src/features/build_features.py --mode discogs-only   # EC2 GPU → embeddings.parquet
python src/features/build_features.py --mode librosa-only   # local CPU → features.parquet
```

## GOTCHA — onset is on two scales
`build_features` stores librosa's **raw** mean onset_strength (catalog range 1.1–2.5). Every threshold
is written for the **normalised** scale `raw / ONSET_SCALE` (5.0), where the catalog runs 0.22–0.51 and
`ONSET_HIGH_MIN = 0.35` sits near the median. `normalise_onset()` in `transition_labeler.py` is the one
definition — call it before comparing, always. Comparing raw against 0.35 passes for 100% of tracks.

Rule constants live in `transition_labeler.py` and are **imported** by the mixer, not copied.
Duplicating them is how the scales drifted apart.

## Transition rules (Phase 4, first match wins)
`BPM_TIGHT=.03 LOOSE=.07 · ENERGY_RISE=.08 FALL=-.08 SLAM=.15 MELT=.05 · HARM_PERFECT=1 COMPATIBLE=2 CLASH=5 · LOUD_MELT=3.0 · ONSET_HIGH=0.35`

| | class | condition |
|---|---|---|
| 1 | slam | (bpm_tight AND Δe > SLAM) OR (harm ≥ CLASH AND Δe > RISE) |
| 2 | rise | Δe > RISE AND bpm_loose AND harm ≤ COMPATIBLE |
| 3 | fade | Δe < FALL |
| 4 | melt | bpm_tight AND harm ≤ PERFECT AND \|Δe\| < MELT AND \|Δloud\| < LOUD_MELT |
| 5 | wave | bpm_tight AND onset_a > ONSET_HIGH AND onset_b > ONSET_HIGH |
| 6 | blend | default |

## Planner (`plan_mix`)
Pool = genre + BPM range, minus `is_unidentified()` (238 "ID"/"Unknown" tracks), minus
`exclude_ids`, optionally above `min_energy_pct`. Then beam search, width 8:

- **Hard veto**: \|log bpm ratio\| ≤ 0.05 · camelot ≤ 2 · unused · artist unused
  (`split_artists()` splits &, x, vs., feat., pres. — a collaboration blocks the solo artist)
- **Score**: `1.0·z(proj·ctx) + 1.0·GBM + 0.7·energy_fit + compat_weight·compatibility`
- Seeds: 8 tracks nearest `curve[0]` energy — **no model involved in the opening track**
- Dedupes on `frozenset` — **two orderings of the same set collide, ordering is never compared**
- `W_CTX / W_GBM / W_ENERGY = 1.0 / 1.0 / 0.7` were **never tuned**

Curves `build peak wave chill arc` map onto the **pool's own** energy quantiles, so a peak curve over a
mixed pool is only a relative peak — `min_energy_pct` is what makes a set loud in absolute terms.
BPM is not a lever: techno energy medians sit at 0.32 across every band from 128 to 150.

Plan JSON carries `target_energy` (absolute, display only) and **`energy_target01`** (raw 0-1 curve
value — this is the one the mixer needs; passing the other silently flattens the curve).

## Fetcher (`fetch_plan`) — three gates
1. **duration** 240–900 s on search metadata, re-checked on the decoded file (metadata lies)
2. **title** rejects radio edit / snippet / sped up / live set / …
3. **identity** constellation fingerprint vs the track's own 30 s preview. Right recording scores
   457–14,466 votes, wrong ≤ 21; threshold 50. Chroma and mel correlation both failed here.

Failing all candidates → **slot repair** (2026-09-08, Anas's design): the verified tracks stay, `repair_candidates`
in `predict_model.py` ranks replacements for the empty slot against BOTH neighbours (BPM ≤5%, Camelot ≤2, no
repeated track or artist; scored by Model B context, GBM edge from the previous track, energy fit to the slot's
target, compatibility with each neighbour; a first/last slot fits its one neighbour), `make_mix` fetches up to
`REPAIR_TRIES`=5 of them, and only if none verifies does it replan the whole set (`max_rounds`). In a 20-track
pool almost nothing fits between two neighbours; the catalog pools (thousands) give 5 candidates easily.
`yt-dlp` must stay current or every download returns HTTP 403.

## Mixer (`render_mix`)
**Stereo.** Audio is `(samples, channels)` from decode to write, time on axis 0. Analysis (`_to_mono`)
runs on the mono sum — beats, sections, key, seam correlation. Rendering in mono discarded the side
channel of every record (−10 to −17 dB rel. mid) and was the largest quality defect.

Per track: `segment()` → `_precise_bpm` (`_measure_tempo`: kick-envelope autocorrelation at the bar
lag over the middle 180 s, parabolic peak) → stretch to the set median BPM with rubberband →
`_calibrate_grid_phase` (shift grid to real kicks) → `_choose_window` → `_analyse_body`.

**GOTCHA — the beat grid cannot give a tempo.** The segmenter's beats are rounded to 23.2 ms frames
(hop 512 @ 22,050 Hz), so a median inter-beat interval snaps to a few values: 134, 135 and 137 BPM
tracks all read 136.05. Two tracks stretched from that drifted half a beat inside one overlap while
the seam report said 0.0 ms (it averages over the whole overlap). Tempo is measured from the audio.
Key names: essentia returns flats (Eb, Ab, Bb); `normalise_key()` in `transition_labeler.py` is the
one definition, used by the mixer. Unnormalised flats scored Camelot distance 2.5 (unknown key).

**Downbeats and the phrase grid.** The segmenter's downbeats come from Beat This! (`beat_this`,
CPU, ~20-40 s per track, checkpoint cached in `~/.cache/torch/hub/checkpoints/`). Its raw downbeats
fire on every beat in some passages, so `regular_bars()` anchors on the longest run of clean 1-bar
gaps and extends a regular 4-beat grid from there; `downbeat_confidence` is the share of tracker
downbeats on that grid (0.50-0.98 on real tracks). The old kick-phase pick stays as fallback and is
a coin toss on 4/4 (`downbeat_source` says which ran). `phrase_offset` is the residue mod 8 that the
novelty peaks land on; section boundaries snap to half-phrase lines of that grid. The mixer then
puts every move on the grid: cue-in/out (`_choose_window`), overlap length in whole phrases
(`_whole_phrases`), the bass swap as a 1-bar move starting on a half-phrase line (`_snap_low_swap`),
the lead handover (`_lead_envelopes(grid_bars)`), and a slam cut on the phrase line. Report fields
`in_bar`, `out_bar`, `phrase_offsets`, `downbeats`. `seam_drift_ms` is the MEDIAN per-4-bar-window
offset (drift moves most windows); `seam_windows_off` counts windows over 20 ms, which on real tracks
are breakdown passages where one record's kick is off the beat, not drift.
Then all tracks LUFS-matched to the set median (±6 dB cap).

**Play length** from `GENRE_PLAY_MINUTES`, measured over 24,831 real track changes in 1,665 sets
(afro 5.17, melodic 4.63, trance 4.42, techno 4.08, tech house 4.00, dnb 2.58; default 4.17).
Variance explained: individual set 0.42, DJ 0.22, genre 0.125, position ~0, track features 0.0022 —
so **do not add an energy term**. `_choose_window` flexes ±16 bars to land on section boundaries
(cue-out bonus 0.25 > cue-in 0.10). `max_tail` reserves the MEDIAN transition, not the max.

Per seam, four independent decisions:
1. **type** — `measured_transition` rules, then `gate_transition(curve)` demotes to the set's intent
   (chill: slam→melt, rise→blend; build: slam→rise before 50%), then rise→drop if the incoming
   track has a usable drop
2. **length** — `pair_compatibility` (0.35 key + 0.30 Δenergy + 0.20 Δbpm + 0.15 Δloud, calibrated on
   43,073 real pairs, AUC 0.643) → ceiling 16/32/60/90 s at p40/p75/p90. `STRETCHABLE` types may run
   to 2× their genre default; `slam` and `drop` never stretch.
3. **alignment** — `measure_seam_offset` cross-correlates kick envelopes ±½ beat with parabolic peak,
   shifts and re-measures up to 4 passes. Seams land 180–215 ms out and end ≤ 0.3 ms.
4. **lead** — `LEAD`: A in front, B enters at −5 to −8 dB with mids cut, lead swaps over a fixed **bar**
   count so the ambiguous zone doesn't grow with the overlap (163 s → 35 s across all types).

**Overlap level (2026-09-07).** No duck and no scalar loudness guard. `_overlap_gain_ride` anchors the
overlap's gain at both ends: A's last 4 body bars going in, B's first 4 body bars coming out, linear in dB
between, clamped ±9 dB (`overlap_gain_db` in the report). Measured: step into the overlap −5 dB → −0.4 dB,
step out +4.5 dB → −1.2 dB on the real test pair; synthetic constant-level render continuous within 0.2 dB.
The interior is left alone so an incoming drop still hits. Anchors (2026-09-08, after Anas heard a step at the seam):
each anchor is the LOUDER of the last/first 2 bars that are not drop-outs (`_edge_level_db`, drop-out =
more than 6 dB under the loudest of the 8 reference bars), and the overlap's own edges are judged within
their first/last 8 bars only. Measured on Materium→Vale: the 4-bar median anchored at −23 dB against the
−17 the ear had just heard (two one-bar breaks in the last four bars); on Farrago – Sinner the whole-overlap
filter discarded the quiet opening and gained −1.4 dB where +7 was needed.

**Seam decision (`seam_decision`, 2026-09-08, settled by two A/B listens).** Kick offset vs the consensus of
bass/mids/highs/full (median, ≥2 readable bands): agree within 60 ms → kick; disagree by half a beat ±20 % →
kick (off-beat layers: Sin Sin → Nova, kick +103 vs others −116 at 127.5 BPM, Anas preferred the kick);
disagree otherwise → consensus (Wigbert → Joyhauser, kick +75 corr 0.12 vs others +194, a quarter beat,
Anas preferred the consensus); no kick → consensus. `seam_band` reports which. Genre-agnostic.

**Seam alignment: keep `SEAM_METHOD = "xcorr"`.** A "grid" method (each record's residual against its own
tracker grid, kick+bass over ±½ beat, upper bands within ±40 ms) measured better on Sin Sin – Break Down
but Anas heard it as messier: the tracker grid there is most likely half a beat off, and no band analysis
can prove a grid right. His ear overrules. "grid" stays in the code for experiments only. Stretch: rubberband **R3** (`-3`), ~40% slower
than R2, sharper kicks by measurement. `_check_stretch_rate` warns above 4% and raises above 8%. The swept
filter crossfades between fixed zero-phase filters on a 1/3-octave grid (exact at a constant cutoff; the
old per-block design had a 20%-of-peak edge error on white noise, which was measurable but I could not
show it was audible: −96 dB on tones).

**Structure by intent (2026-09-07, Anas's design).** The slot's `energy_target01` picks the regime
(`structure_regime`): **low** < 0.35 never lands on a drop and the OUTGOING window prefers a cue-out whose
tail is breakdown/outro/intro (`quiet_tail`, +0.15 bonus, driven by the NEXT slot's regime); **mid** keeps
the old behaviour (only a measured rise is re-aimed at a drop); **high** ≥ 0.65 ends the overlap on the
incoming drop when it is at least a phrase past the cue-in (`_land_on_drop`, whole phrases, capped at 2×
the type's default), with the bass swap in the last bar (`_end_anchored_swap`) and the handover at the
end (LEAD["drop"]). So a chill curve gets zero drop landings, build gets them late, peak almost everywhere.
Report: `regime`, `lands_on_drop`, `tail_section`, `head_section`. No plan → mid.

**Total length.** `render_mix(total_minutes=…)` (passed by `make_mix --minutes`): the LAST track plays past its
window until the mix reaches the requested length, on its own phrase grid, never past its final bar
(`_last_track_end_bar`). Earlier windows keep the genre length. Before this, 15-minute requests rendered 13.1-13.6.

**Final stage.** −14 LUFS then `_true_peak_limiter` (4× oversampled, 5 ms lookahead, 100 ms release,
ceiling −1 dBTP; `limiter` and `true_peak_dbtp` in the report). Measured on a 10-min real render: −14.0
LUFS, −1.0 dBTP, max 0.75 dB reduction on 0.18% of samples; the old scalar method lost the full overshoot.
**Seam measurement is band-aware** (`_seam_envelopes`): kick band when both records have a beat-regular
kick (`_envelope_periodicity` ≥ 0.12 AND kick share of onset energy ≥ 0.02, real sections 0.05-0.45), else the
full-band onset envelope, else NO shift (grids trusted;
`seam_band` = kick|full|none). Measured on a breakdown→drop seam: kick alone read −72 ms (corr 0.03, no
outgoing kick) while low-mid/mid/full band agreed at −15 to −17 ms; hats alone read +220 ms (off-beat).
`seam_windows_off` = [off, judged, total]: only windows with a periodic pulse in both records are judged,
`seam_drift_ms` is their median and is null below 2 judged windows.
Source quality: fetcher is `bestaudio` with no re-encode (native Opus ~130-145 kbps or AAC 128k), render is
lossless; the YouTube source is the ceiling and nothing in the code lowers it.

Signal path per channel: **EQ → filter → fader**. Low band is a gain envelope (bass swap is an EQ kill);
`rise`/`wave`/`drop` carry `A_sweep`/`B_sweep` for `_swept_filter`, a real moving corner (verified
tracking 4 kHz → 25 Hz). EQ uses `sosfiltfilt` — one-pass butterworth put the low band 2.38 ms behind
the highs. Output −14 LUFS then peak-limited; `.flac`/`.wav` lossless or `.mp3` at 320k.

Genre overlap defaults (bars): `{slam 4, rise 32, fade 32, melt 64, wave 16, blend 16, drop 16}`;
tech/melodic/afro house use blend 32, rise 24, fade 16; dnb halves everything.

## Performance
Per 6.5 min track: segment 0 s cached / 20.1 s cold · decode 2.1 s · **stretch 18.6 s** · grid 1.1 s ·
body 1.5 s. A 15.6 min mix = 192 s with everything local. Stretch is 80% and depends only on
(track, target_bpm) so it is cacheable; `_prepare_track` is serial but independent so parallelisable.

## Services
MLflow :5000 · Airflow :8080 (airflow/airflow) · ChromaDB :8000 · W&B · S3 `aidj-1` (us-east-1, account talhanonstatic; the old ai-dj-data bucket was deleted 2026-09-12)

## Notes
- `starting_time` rolls over hourly: if time_B < time_A → gap = (60−time_A)+time_B; skip pairs > 15 min
- `tests/conftest.py` — `tmp_audio_file` fixture (sine wave WAV). 214 tests, ruff clean.
- S3 sync: `aws s3 sync data/processed/ s3://aidj-1/processed/`
