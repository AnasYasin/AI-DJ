# AI-DJ — Plan (rev. 2026-07-07)

> ORIGINAL PLAN, restored verbatim 2026-09-02. Everything above the divider is
> the pre-session plan as written on 2026-07-07/09. Do not edit it — it is the
> baseline for measuring drift. Session work is recorded below the divider.

## What happened
`embeddings.parquet` was corrupt: `DiscogsEmbedderGPU._mel_patches` fed `log10(mel)`
to discogs-effnet instead of essentia's `log10(1+10000·mel)` frontend. All signal
probes and the failed contrastive training ran on garbage. **Fixed + re-embedded
all 28,460 previews** (validated: cosine 0.9991 vs essentia wrapper).

## Signal (fixed embeddings, full data)
| Probe | corrupt | fixed |
|---|---|---|
| genre same/cross | 0.503 | **0.686** |
| next-track (raw cosine, 40k pairs) | 0.50 | **0.630** |
| next-track (MLP probe) | 0.52 | **0.688** |
| same-mix vs cross-mix | 0.515 | **0.602** |
| same-DJ (in-genre) | 0.517 | **0.569** |

Baselines to beat: BPM 0.624, combined scalars 0.597, MLP probe 0.688.

## Models
1. **A — Track Tower (now):** MLP(1287→256→128), NT-Xent, consecutive pairs =
   positives, genre-blocked in-batch negatives + ChromaDB semi-hard negatives
   (band recalibrated to [0.15, 0.40]). CPU-trainable ~1-2h.
2. **B — Sequence Model (next):** small transformer over mix history
   (A-embeddings + position + deltas, genre/DJ tokens) → InfoNCE next-track
   embedding prediction. Context tower of the two-tower retrieval.
3. **C — Transition Critic (v2):** real transitions (cut from downloaded mixes
   via validated timestamp alignment) vs synthetic splices. Judges rendered audio;
   also the evaluation metric vs real DJ mixes.
4. **D — DJ/style profiles (cheap, ongoing):** per-DJ BPM/energy-arc/transition
   stats → ~10-15 style archetypes (only 48 DJs have ≥2 mixes).

## Inference pipeline (unchanged)
prompt → intent (genre, energy curve, BPM, profile) → ChromaDB candidates →
A+B ranking + hard rules (BPM ±3%, Camelot, phrase alignment) → beam search →
segmenter picks played section per energy curve → mixer renders.

## Segmenter (Phase 7, inference on full tracks)
madmom beats → bar/phrase grid → per-bar RMS/onset/flux/kick features →
SSM novelty boundaries snapped to phrases → intro/buildup/drop/breakdown/outro
labels → energy curve selects sections (minimal = skip drops, long buildups).
v2: learn cue-point priors from aligned mix audio.

## Overlays ("w/" tracks)
4,615 simultaneous events in raw scrape (currently filtered out). v2: mine rules
(P(overlay|genre,position), key-delta≈0, phrase-aligned entry); rendering needs
stem separation — park until core loop ships.

## Data decisions
- Previews: sufficient for A + B. Full original tracks: never needed.
- Mix audio (yt-dlp search + duration check + preview-DTW verification — all
  validated): phase 2, for critic + played-segment fine-tune.
- Val split: `data/processed/split_mixes.csv` — mix-level, 15% per genre, frozen.

## Status (2026-07-08)
- [x] embeddings fixed & re-embedded (`embeddings.parquet` = fixed; corrupt kept
      as `*_corrupt_backup.parquet`); ChromaDB re-indexed
- [x] features merged (28,460 × [embedding + bpm/key/energy/…]); split frozen
- [x] **Model A** trained (z-scored inputs + scalar branch): val adjacency AUC
      0.663 vs raw 0.622 / BPM 0.624. 50 epochs is the sweet spot (150 overfits).
      Inference must apply `models/encoder_norm.npz`.
- [x] **Model B** (train_sequence.py, causal transformer + genre token, InfoNCE):
      val next-track MRR 0.0428 vs 0.0365 context-free baseline (+17%),
      median rank 870 vs 1138 (~4,700 candidates)
- [x] **Edge scorer** (edge_scorer.py → models/edge_gbm.pkl): val AUC ~0.69
- [x] **Planner** (predict_model.py): beam search over Model B context + GBM +
      hard rules (BPM ≤5%, Camelot ≤2, one track per artist) + energy curves
- [x] **Segmenter** (audio_segmenter.py): phrase-aligned sections, tested on 4 tracks
- [x] **Mixer** (audio_mixer.py): first mix rendered — 6.6 min @ 117 BPM,
      16-bar bass-swap transitions (data/external/first_mix.mp3)

## Next
- [ ] Listen-test the mix; tune segmenter/mixer from what the ears say
- [ ] Wire transition-type → overlap length (genre params in CLAUDE.md)
- [ ] Jamendo full-audio source → connect planner to mixer end-to-end
- [ ] Claude intent parsing + FastAPI (Phase 10)
- [ ] DJ style archetypes (48 multi-mix DJs → ~10-15 clusters)
- [ ] v2: mix-audio corpus (downloader/aligner validated), transition critic,
      learned cue-point priors

## Known issues — listening tests (2026-07-09, collecting before fixing)
1. **Transition classifier doesn't know the set's intent.** It fires on measured
   pair features only — a chill mix got a SLAM (ΔE 0.18 between windows, rule 1
   fired). Fix direction: pass curve/mood into classify_transition and gate
   types (chill → demote slam/rise to blend/melt; peak → allow).
2. **Audio quality is not good.** Suspects, in likely order of impact:
   - time-stretch is librosa phase vocoder (rubberband CLI not installed) —
     smears transients/kick; install rubberband-cli or pin stretch ≤ ~3%
   - double lossy: YouTube AAC (~128k) → decode → 192k MP3 re-encode
   - RMS gain matching (not LUFS) can pump quiet/loud tracks
   - band-split EQ uses butterworth sosfilt (non-linear phase) — possible
     smearing around crossover during transitions
   - beat grid: librosa beat tracker phase errors → sloppy bar alignment in
     overlaps
3. **Short YouTube edits slip through**: duration filter is >120s; planned
   tracks need ≥ ~4 min (two windows got cut to ~1 min: Digweed remix, VIP
   Business).
4. **"ID" tracks reach the planner** (e.g. "Massano – ID") — filter
   track_name == ID/Unknown at pool build.
5. **Beat misalignment** — melodic_house_chill.mp3, transition 1
   (Slove → Mosca, FADE 16 bars, ~3:10): beats audibly not locked during the
   overlap. Likely causes: per-track downbeat/beat-grid phase error (grids are
   estimated independently, never cross-checked at the seam), beat-tracker
   drift across the window, constant-ratio stretch vs real tempo drift inside
   the track. Fix direction: micro-align at each overlap start via onset-
   envelope cross-correlation (±½ beat) between tail and head, on top of the
   bar-grid placement.
6. (collecting more from listening…)

---

# Session record — 2026-09-02

Everything above is the original plan, untouched. This section is what
actually happened since, so the two can be compared.

## Drift from the original plan
- **ChromaDB is not used at inference.** The plan routed candidates through it;
  `predict_model` brute-forces the full genre+BPM pool instead. Better quality
  (no approximate-retrieval loss), cheap at ~2k candidates, but Model A's trained
  role as a *retrieval* space is unused — it is only a scoring feature now.
- **Jamendo was never used.** Full audio comes from YouTube via `track_fetcher.py`,
  gated on duration and a fingerprint match against the track's own preview.
- **Hard rules loosened**: BPM ±3% in the plan, ±5% in code (`MAX_BPM_LOG_RATIO`).
- **Model C and the mix-audio corpus are still not started.** The 5,652 mix URLs
  remain unused, so there is still no learned judgement of a rendered transition
  and no evaluation metric against real DJ mixes.
- **Model D (DJ profiles) still not started**, and it is now the best-measured
  available win: DJ explains 0.22 of play-length variance against genre's 0.125.
- **Overlays still parked**, as planned.
- **Added, not in the original plan**: pair-compatibility gating of overlap length,
  the lead/support model, real swept filters, stereo rendering, an energy floor,
  and a compatibility weight in the beam score.

## Where it stands

Full pipeline works end to end, one command:

```bash
python -m scripts.make_mix --genre techno --bpm 134 142 --curve peak \
    --n 6 --minutes 15 --min-energy-pct 65 --compat-weight 5 --out mix.flac
```

plan → fetch + verify → replan around failures → render. 214 tests, ruff clean.
Architecture diagrams: `docs/architecture/ai-dj-component-map.html`.

| | |
|---|---|
| Catalog | 2,834 mixes, 28,460 tracks with features |
| Model A | contrastive encoder, val adjacency AUC 0.663 |
| Model B | sequence transformer, +17% MRR |
| GBM | edge scorer, val AUC 0.69 |
| Compatibility | 4-term score, AUC 0.643 vs random pairs |

## Fixed this session (all measured, all tested)

1. **Transition types ignored set intent** → `gate_transition(curve)` demotes; a chill set never slams.
2. **Mixer rendered MONO.** The largest audio defect. Side channel of every record (−10 to −17 dB rel. mid) was discarded by `librosa.load(mono=True)`. Audio path is now (samples, channels) stereo end to end; analysis still runs on the mono sum. YouTube was NOT the cause — source and render matched within 1-2 dB at every frequency.
3. **EQ smeared the kick.** One-pass butterworth put the low band 2.38 ms behind the highs. `sosfiltfilt` → 0.00 ms.
4. **RMS gain → LUFS**, measured on the played window, overlaps held to their louder neighbour, mix to −14 LUFS.
5. **Seams were 180-215 ms out.** Grid calibration alone wasn't enough. `measure_seam_offset` cross-correlates kick envelopes, corrects and re-measures → ≤0.3 ms everywhere.
6. **Short YouTube edits** → rebuilt `track_fetcher.py` (the old one was never committed and was lost). Duration 240-900 s checked on the decoded file, plus a constellation fingerprint against the track's own preview (real 457-14,466 votes, wrong ≤21).
7. **"ID" tracks reached the planner** → `is_unidentified()`, drops 238 of 29,393.
8. **Overlaps were mush.** Both records sat within 3 dB for 163 s across the seven types. `LEAD` model: A in front, B at −5 to −8 dB with mids cut, lead swaps over a fixed BAR count → 35 s.
9. **Overlap length ignored the pair.** `pair_compatibility` (calibrated on 43,073 real pairs) caps at 16/32/60/90 s; STRETCHABLE types may run to 2× their default.
10. **"Filters" were not filters.** Three fixed band gains at 180 Hz / 3 kHz. `_swept_filter` now moves a real corner, verified tracking 4 kHz → 25 Hz.
11. **Onset was on two scales.** Catalog raw (1.1-2.5), thresholds written for raw/5. The labeler compared raw against 0.35 which 100% of tracks exceed, so the `wave` rule was a no-op. `normalise_onset()` is now the one definition. Re-labelling 43,073 pairs: wave 28.2% → 9.1%, 19.1% of pairs change label.
12. **One-track-per-artist missed collaborations.** "Reinier Zonneveld & Miro" ≠ "Reinier Zonneveld". `split_artists()` fixes it.
13. **Planner never planned for long overlaps** → `compat_weight` in the beam score (default 0).
14. **`min_energy_pct`** — the curve maps onto pool quantiles, so a peak curve over a mixed pool is only a relative peak.

## Open points

**Missing entirely**
- Intent parser (Phase 10) and the API service. No NL input; everything is CLI args.
- Model C transition critic, mix-audio corpus (5,652 URLs collected, unused), learned cue points.
- DJ profiles: `mix_profiler.py` exists, `dj_profiler.py` / `dj_profiles.json` do not. 48 DJs have 5+ mixes. Explains 0.22 of play-length variance vs genre's 0.125.
- Overlays ("w/" tracks). Requests for a named artist, era, or a mood outside the five curves.

**Known weak**
- Beam dedupes on `frozenset(tracks)` → two orderings of the same set collide, ordering is never compared.
- Opening track chosen by energy distance alone, no model.
- `W_CTX / W_GBM / W_ENERGY = 1.0 / 1.0 / 0.7` never tuned.
- A failed fetch replans the WHOLE set, not the failed slot. Discards good plans.
- Exact duration is not a control; lands within ~10%.
- ChromaDB is built but never read at inference.

**Performance (measured, per 6.5 min track)**
segmentation 0.0 s cached / 20.1 s cold · decode 2.1 s · **time-stretch 18.6 s** · grid phase 1.1 s · body 1.5 s.
A 15.6 min mix = 192 s with everything local. Stretch is 80% of it and depends only on (track, target_bpm), so it is cacheable; `_prepare_track` is serial but independent, so parallelisable.

**For a live demo**: prefetch a verified pool to S3 (`ai-dj-data`, eu-north-1) and restrict the planner with `plan_mix(track_ids=...)`. That removes the ~6 min fetch, the replan and the yt-dlp 403 risk, but not the 192 s render. Pre-stretch + parallel prepare should get it under a minute.

## Next
- [ ] Listen to `data/external/mixes/*_v4.flac` and `techno_peak_15min.flac`
- [ ] Replace only the failed slot on replan
- [ ] DJ profiles (Phase 9) — best available win on play length
- [ ] Intent parser + FastAPI (Phase 10)
- [ ] Tune the beam weights against something real

---

# Session record — 2026-09-06 / 2026-09-07

All changes are on disk and UNCOMMITTED (git status: requirements.txt, src/audio/audio_mixer.py,
src/data/audio_segmenter.py, src/features/transition_labeler.py, tests/conftest.py,
tests/test_audio_mixer.py modified; scripts/diag/ new). 250 tests pass, ruff clean.
Full technical detail of every change is in CLAUDE.md (Mixer section). Short version:

## Fixed and measured
1. Tempo: `_measure_tempo` from the kick envelope (grid was 23 ms quantised; 134/135/137 BPM all read 136.05).
2. Key names: essentia flats normalised (`normalise_key`), ~20% of tracks were "unknown key" in the mixer.
3. Downbeats: `beat_this` tracker + `regular_bars()`; phrase offset from novelty peaks; cue points, overlap
   lengths, bass swaps, handovers and slam cuts all on the phrase grid.
4. Sound quality (Anas's choices): two-point overlap gain ride (no duck, no scalar guard), rubberband R3,
   stretch warn 4%/error 8%, 1/3-octave grid swept filter, true-peak limiter −1 dBTP.
5. Structure by intent (Anas: "not every cue-in on the drop"): energy_target01 < 0.35 low / ≥ 0.65 high.
6. Seam measurement band-aware + kick energy floor; drift = median of judged 4-bar windows.

## Ear-test clips (for Anas to listen to)
`data/external/ear_test_2026-09-07/` — 32 FLAC clips, README.md with tables, report*.json.
11 system-chosen pairs (tech house 4, techno 3, dnb 4) + 21 type-sweep clips (7 types × 3 genres, `types/`).
Each clip = 2 min A + overlap + 2 min B; overlap starts at 2:00 unless README says otherwise.
Tech house type sweep is clamped to 8 bars (pair 1 has little tail); techno sweep has full lengths.

## OPEN — found by diagnosis, NOT fixed (needs Anas's go-ahead)
Techno pair 2 (Sin Sin – Break Down (Tom Laws Remix) → Mathias Kaden – Nova) is misaligned ~150 ms:
- Track A has a weak syncopated low end and off-beat open hats. Its beat_this grid is right (bass on the
  grid), but `_calibrate_grid_phase` (kick band only, ±¼ beat) moved the grid +64 ms late.
- At the seam the kick correlation read +103 ms (corr 0.12, A kick periodicity 0.12) while lowmid/mid/hats/
  full all said −116 (corr up to 0.67). The mixer trusted the kick (it passed the 0.12 gate by a hair).
- Evidence: scripts/diag/seam_diag2.txt, seam_diag1.txt, grid_phase_diag.py output in the session log.
Proposed fix (design change, not applied): drop A-vs-B audio cross-correlation for the seam. Measure each
record's residual against its OWN tracker grid within ±40 ms using the band with the best correlation, shift
by the difference. ±40 ms can never produce a half-beat jump; the tracker becomes the arbiter of "the beat".
Same for `_calibrate_grid_phase` (multi-band, ±40 ms, only when downbeat_source == beat_this).
Other clips with drift > 60 ms (tech house pairs 1, 2, 4; techno pair 1) probably share the cause.

## Next (Anas's order)
1. Anas listens to the clips, especially techno pair 2 and pair 1, and reports.
2. Fix the seam/calibration as above if confirmed; re-render the affected clips.
3. Key shift mechanism (none exists; time stretch is pitch-locked). Design to be agreed first.
4. "Finalize beat mixing" and test end to end.
5. Later: model leakage + track-level re-split (review found Model A/B/GBM inflated by leakage).

## Diagnostics (copied from the session scratchpad into scripts/diag/)
tempo_check.py, drift_check.py, verify_phrase.py, seam_diag.py, grid_phase_diag.py, ear_test.py,
ear_test_types.py. Run from repo root in the `aidj` env. beat_this checkpoint is cached in
~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt (81 MB; the JKU server is slow, ~50 KB/s single stream).

## 2026-09-08 (resumed after reboot)
- Anas listened: beat matching "much better", techno pair 2 slightly off but tolerable, sound quality fine
  (his BT headphones were the earlier problem). Loudness step at the seam on the house clip: real, fixed.
- Grid-based seam method built and tested; he heard it as WORSE on techno pair 2 → default stays xcorr
  (`SEAM_METHOD`). Lesson recorded in CLAUDE.md: ear overrules measurement; tracker grids can be half a beat off.
- Gain-ride anchors: drop-out-aware, edge-based (last/first 2 real bars), overlap edges judged locally.
  Verified on Materium→Vale (step gone) and on the worst case Farrago→Heil (render in progress at time of writing).
- Listening mixes: data/external/listen/house_low_15min.flac (13.6 min, 4 tracks, all verified first pass);
  minimal_mid_15min.flac rendering (tech house 122-128, arc, energy floor p40; one FISHER edit and one
  African Roots copy rejected by the fingerprint, replanned).
- Next: Anas listens to the listening mixes; then key shift mechanism (design first), then finalize beat mixing.
- Slot repair implemented (Anas: never discard verified tracks; fit the empty slot to its neighbours, one
  neighbour at the ends). Last track now runs on to the requested total length (±1 phrase). Both tested
  with targeted tests only; Anas asked not to run the full suite (time). Edge-anchor worst case
  (Farrago → Heil) verified: overlap opens at -16.4 dB against A's last real bar -16.9.
- Next: key shift mechanism (design first), then finalize beat mixing; later rebuild start/end slot logic.
- Seam decision rule (kick vs consensus, half-beat test) applied after A/B on Wigbert→Joyhauser (Anas: B/consensus
  sounded good). Verified on numbers only, both real seams decide as his ear did. Listening sets: house_low_15min,
  minimal_mid_15min, techno_high_15min (14.7 min, slot repair fired and worked), afro_20min (20.1 min).
- Key redesign agreed in principle (drop veto, per-genre soft weight, cap floor(joins/2), no consecutive clashes,
  ≤2 semitone shift in the mixer); not built yet — Anas to say go.

## State at session end, 2026-09-08 (Anas restarting the session)
All changes UNCOMMITTED. Targeted tests only (Anas: no full suite). Nothing running.

Done today, all in code:
- Seam decision rule (kick vs consensus of bass/mids/highs/full; half-beat test) — applied, verified on the
  numbers of both A/B seams. NOT yet heard in a full set; Anas asked to generate the techno set again to test,
  then stopped the session before the render started. Next session: render
  `techno_high_15min_v2.flac` with the same command as techno_high (tracks cached in
  data/external/listen/work_techno_high) and let him listen around 4:59.
- Loudness anchors: edge bars, drop-out aware, overlap edges judged locally (Anas: "loudness is fixed").
- Slot repair in make_mix (fired live on techno_high: Luis Miranda failed 3×, Wigbert repaired slot 1).
- Last track runs on to the requested total length (techno 14.7 min vs 15 asked; afro 20.1 vs 20).
- plan.json now carries per-track start / transition / bars / fully_in and a render block. No extra files.
- SEAM_METHOD default back to "xcorr" (grid method kept for experiments; he heard it as worse on pair 2).

Listening sets (all rendered BEFORE the seam decision rule except none): data/external/listen/
  house_low_15min (13.6), minimal_mid_15min (13.1), techno_high_15min (14.7, seam at 4:14 is the Wigbert
  case — now fixed in code, not re-rendered), afro_20min (20.1, all D#m, all 16-bar overlaps).
A/B clips: data/external/ear_test_2026-09-07/seam_techno_high/ (A kick, B consensus — Anas: B sounds good).

Agreed, not built: KEY REDESIGN — drop the Camelot veto in the planner, take key out of the planning
score, add a per-genre soft key weight tuned to real clash shares (melodic house/trance ~42 %, techno/tech
house/dnb ~53-55 %), cap clashing joins at floor(joins/2) with no two in a row, mixer shifts the incoming
track ≤2 semitones when that lands within 1 Camelot step (about 1 join in 3), otherwise treats the pair as a
clash (short overlap, early bass swap). Data: real DJ joins 14 % same key / 36 % 1-2 steps / 50 % >2 steps;
our current sets 62 % same key. Anas asked for the final verdict and got it; he has not said go.

Then: finalize beat mixing; later rebuild start/end slot logic; model leakage/re-split.

Working style Anas asked for: explain a code-changing command before running it; no full test suite;
do not re-render clips he has not heard; ear overrules numbers; ask on design decisions.

## Hand-off, end of 2026-09-08 (second restart)
- New since the last hand-off: seam decision rule verified on a NEW set, techno_build_15min.flac (15.0 min,
  Amelie Lens / Mikael Jonasson / Tiger Stripes / The Reason Y, seams shifted +121 / −24 / −115 ms). Anas has
  NOT reported on it yet; ask what he heard at 3:43–4:42 and 11:22–12:22 first. plan.json entries now carry
  `seam_shift_ms` and `seam_decided_by` (from the next render on).
- Committed and pushed: 4cb3741 on dev. After that commit, one small uncommitted change: scripts/make_mix.py
  (seam fields in add_times). Commit it with the key work.

## KEY REDESIGN — full design, agreed with Anas, NOT built. Implement next.
Evidence (2026-09-08, 43,073 real consecutive pairs vs 36k random same-genre pairs, keys from essentia on
30 s previews): real 14 % same key / 36 % 1–2 Camelot steps / 50 % >2 steps; random 9 / 34 / 57. DJs treat
key as a weak preference. Our planner today: 62 % same key, 0 % >2 steps → monotone sets. With the veto off
but compat_weight kept: still 65 % same key (the key term inside `compatibility()` dominates). With the veto
off and key out of the planning score: 15 / 44 / 41 — matches the DJs.

1. Planner (`src/models/predict_model.py`, `plan_mix` and `repair_candidates`):
   a. Remove the hard veto `camelot_distance(...) <= MAX_CAMELOT_DIST` from the candidate mask (keep the BPM veto).
   b. Planning compatibility must NOT include key: call `compatibility()` with the key distance replaced by 0,
      or add a `key_weight` argument to `src/features/compatibility.py` and pass 0 from the planner. The MIXER
      keeps the full compatibility (key included) for overlap length — a clash still gets a short overlap.
   c. Per-genre soft key penalty added to the beam score: `− KEY_SOFT[genre] · max(cam_dist − 2, 0)`, tuned so
      planned sets land near the genre's real clash share: melodic house 42 %, trance 43 %, techno 53 %,
      tech house 55 %, drum and base 54 %, afro house 51 %. Tune by running plan_mix over the 6 genres × 3
      curves (the script in the session log did exactly this) and adjusting KEY_SOFT until the >2-step share
      is within ±5 % of the target. Start at 0.3 and sweep.
   d. Cap per set inside the beam expansion: a beam may not add a candidate if it would make
      (clashing joins) > floor(n_joins / 2), or if the previous join already clashed (never two in a row).
      Clash = cam_dist > 2.
   e. Plan JSON: write `cam_dist` for every join (already there) and a set-level `n_clashes`.
2. Mixer (`src/audio/audio_mixer.py`, per transition, after `measured_transition`):
   a. Compute Camelot distance A→B from the played windows (already done: `_camelot_dist(ta["key"], tb["key"])`).
   b. If distance > 1: find the semitone shift s ∈ {−2, −1, +1, +2} that minimises the Camelot distance of
      B's key shifted by s (a shift of k semitones moves the Camelot number by 7k mod 12; +1 st = +7 = −5 steps,
      +2 st = +2 steps). If the best shift brings the distance to ≤ 1, apply it to B's audio with rubberband
      pitch shift (`pyrubberband.pitch_shift(y, SR, s, rbargs={"-3": ""})`, or fold it into the existing
      time_stretch call: rubberband CLI `-p <semitones>` alongside `-t`), BEFORE `_analyse_body` so the key
      used downstream is the shifted one. Record `key_shift_semitones` in the track dict and the report.
   c. If no shift within ±2 fixes it: no shift; the pair is a clash. The compatibility ceiling already shortens
      the overlap; additionally move the bass swap to the FIRST half-phrase of the overlap (early swap) so the
      two harmonies overlap as little as possible. Report `clash: true`.
   d. Tests: Camelot arithmetic (5 steps → 1 st, 2 steps → 2 st, 3/4/6 steps → no fix), shift applied only when
      it lands ≤ 1 step, key string updated after shift, planner cap (no two clashes in a row, ≤ floor(n/2)),
      per-genre share within tolerance on the real pool (skipif no catalog).
3. Verification for Anas: render one set that contains at least one shifted pair and one unshifted clash
   (the plan will say which), and let him listen. Expect ~1 join in 3 shifted, ~1 in 5 a clash mix.
4. Open constants to revisit by ear: ±2 semitone cap, the 60/40 clash cap for short sets, KEY_SOFT per genre.

## 2026-09-08 (third session) — overlap loudness anchor
Anas: techno_build seams at 3:43–4:42 and 11:22–12:22 are fine. Complaint: loudness "goes up and down
suddenly", e.g. around 4:43. Measured (per-bar RMS, both sources aligned to the mix): both records sat at
-10/-11 dB through the second half of overlap 1, the mix slid from -12.5 (4:11) to -20.0 (4:42), then
Mosaic's own 8-bar breakdown (-21..-25) and its drop at 4:57 came up 17 dB in one bar. Cause: the ride's end
anchor read Mosaic's first 8 body bars, which are all breakdown, so it aimed the overlap at -16.7.
Fix (src/audio/audio_mixer.py `_overlap_gain_ride`): anchors are now each record's BODY level over its
whole played window (`_anchor_level_db`, break-skipping median), not the 8 bars beside the seam. Call site
passes prev's played window and cur's full body. Cap left at ±9 dB (a measured case, Farrago, needs +7 in).
Re-render (same plan.json, cached tracks): overlap gains +5.0/0.0, +6.7/0.0, +5.4/0.0. Overlap 1 second
half now -9.7..-11.4 (was -15.7..-20.0); break and drop are the record's own (-21 → -11, 10 dB).
Side effect to judge by ear: the +5 dB start lift (A's cue-out lands in A's 8-bar break) now decays
linearly to 0 across 32 bars, so A's drop inside the overlap (3:56–4:05) plays ~3 dB above body, was ~1.
Files: data/external/listen/techno_build_seam1_loudfix_3m30-5m10.flac (clip for Anas),
techno_build_15min_v2.flac (full, for later), work_techno_build/techno_build_15min/render_v2.log.
Tests: 11 targeted gain-ride tests pass; new test `test_gain_ride_anchors_to_the_body_not_to_a_breakdown_after_the_seam`;
Farrago test expectation 7→6 dB (body median vs max-of-two-bars anchor). UNCOMMITTED with make_mix seam fields.
Lesson: for one seam, render the pair with scripts/diag/ear_test.py's route, not the whole set.

### Seam 1 of techno_build: NOT drift, a wrong single shift (measured 2026-09-08, not fixed)
Anas heard beats not matched near 4:20 in the loudness-fixed render. Measured:
- Tempo: Amelie Lens 131.00 BPM and Mosaic 130.00 BPM constant across their whole files (kick ACF at 8/16-bar
  lags, per minute); stretched tails read 129.000 / 128.998. No tempo drift.
- Per-window xcorr readings after the +121 ms shift swing -75/+43/0/+158/+38 ms; grid readings differ again.
  Both read different layers per window. Neither is a measurement of the kicks.
- One-strong-kick-per-beat onsets (kick band, one peak per beat): after the mixer's +121 ms shift Mosaic's kicks
  sit 105-117 ms EARLY against Amelie Lens in all 8 windows (IQR a few ms). With shift 0: +4 vs -8 ms.
  With +16 ms: -3 to +12 ms. The seam decision ("kick(offbeat layers)", +121 ms) moved an aligned pair a 16th
  note out. Same shift was in the original render; the old 8 dB downward tilt hid it.
- Diagnostic: scratchpad kicks_seam1.py (prepare A/B as the render does, kick onsets one per beat, per-window
  nearest-kick offset + phase against the 129 BPM lattice).
Proposed fix, awaiting Anas: after the seam decision, verify the residual with the one-kick-per-beat onset
phase (A vs B) and, if it exceeds ~30 ms, use the onset-phase shift instead. Candidate to replace the
xcorr/consensus decision entirely if it holds on the other seams.

## Regression set (agreed 2026-09-08) and the things it must be able to test later
Anas's future list, kept here so the set and its manifest are shaped for them (each gets its own design pass later):
- Mixing different BPMs like DJs do; a model-compatible pair must not be dropped only because tempo is out of range.
- Overall loudness of the mix must not change (level stays flat across the set).
- Overall energy curve of the mix — how it is handled and verified.
- Model retrain (after the leakage/re-split).
- DJ profiling (recipes, lengths and choices that follow a DJ's style).
- Play length per track: tracks are ~3 min windows today; a mechanism to decide how long each track plays.
- Key correction (KEY REDESIGN above).
- Looping parts of tracks the way DJs do.
- Transition recipe chosen per pair, style, genre, DJ profile and tracks, not per type alone.
Consequences for the set: one long file of independent pairs in fixed slots, seams on round minutes, 10 s lead-in,
~20 s after; per pair the manifest stores track ids, cue bars, overlap bars, transition type, slot, applied shift,
gains, and Anas's verdict + approved shift once heard. Recipe/type/length are manifest fields so the same pairs
can be re-rendered under a different recipe, loop, tempo rule or profile. New real failures are added as pairs.
Plain controls (clean four-on-the-floor) at the end of the file.

### Regression set — build progress (2026-09-08, in flight)
Done and verified:
- Stretch cache: `_stretch` caches rubberband output by audio content hash + rate + engine args + version
  (data/interim/stretch/*.npy). Verified identical to uncached; hit 0.3 s vs 38 s. 40 of 45 s per track prep was the stretch.
- Test hooks in `render_mix`: `force_bars` (final say on overlap length, clamped to audio) and `cue_overrides`
  (per track (cue_in_bar, cue_out_bar) or None) → `_prepare_track(cue=...)`. Report now carries `at_s`/`end_s`.
- `scripts/regression_set.py`: manifest → renders each pair in full, cuts 10 s + overlap + 20 s, places the seam exactly on
  the pair's `seam_at` (whole minutes), writes regression_set.flac + README.md + results.json; `--only ids` re-renders
  some pairs and keeps the others' audio. Verified on a 2-pair mini manifest: seams at 1:00.000 and 3:00.000, lead-in
  10.00 s; Amelie→Mosaic reproduces the +121 ms fault exactly.
- `tests/test_regression_set.py` (marker `regression`, excluded by default in pyproject addopts): numbers-only check of
  every pair with an `approved` block (shift ±20 ms, gains ±1.5 dB, bars equal) + manifest well-formedness.
- `scripts/diag/regression_scan.py` (sharded, resumable) measures per 8-bar block: kick_share, offlow (low-band onset
  energy off the beat), hat_off8, ghost, kpb (kick peaks per bar), level; per track: tempo, downbeat conf, LUFS, sections.
  `scripts/diag/regression_pick.py` ranks candidates per edge case with the block to cue on.
Known pairs to include (ids): Amelie c9a1c5963da5→Mosaic 7f0290e636d0 (listen/work_techno_build); Sin Sin dc549025561c→Nova
98ccb1974095; Farrago 2ac6c5e5d611→Heil b96e58ca1b40 (test_tracks/techno); Wigbert 4900eb25d876→Joyhauser af90f79242e8
(listen/work_techno_high); Materium 7eb26f5ddeb6→Vale 1ab15822e184 (test_tracks/melodic_house); DnB ear-test pairs
3ef523169490→06b288f1a9f9, 7462d65de9f6→ed13dcecc9ab.
Next: finish scan (running, ~144 tracks), pick 10 hard + 4 controls with cue blocks, write tests/regression/manifest.json,
render to data/external/regression_set/ (README for Anas), then run the kick-hit verifier idea against the set.

### Regression set v1 — BUILT 2026-09-08, awaiting Anas's verdicts
- Files: data/external/regression_set/{regression_set.flac (33.0 min, ~19 min of audio), README.md, results.json}.
  Manifest: tests/regression/manifest.json (15 pairs: p01–p11 hard, c01–c04 plain controls; seams at 1:00, 3:00 … 29:00,
  10 s lead-in, ~20 s after). Rebuild: `python -m scripts.regression_set` (all, ~10 min with caches) or `--only p03,c01`.
- Pins used (cue overrides) are in the manifest with the reason appended. Verified: every seam starts at exactly N:00.000
  with a 10.00 s lead-in; all overlaps have the requested bar counts.
- First-render numbers worth his ear: p01 +121 (the known false shift), p02 +103 kick(offbeat), p03 −59 consensus (was +203 on
  the mixer's own window), p04 gain +9.0 in (cap) breakdown cue-out, p05 gain +9.0 in, p06 drop-aligned +78 consensus into
  Safar's break, p07 gains +5.2/+3.2 weak kicks, p08 −143 consensus DnB, p09 −180 consensus DnB ghost kicks, c04 −94 consensus
  on a "plain" trance pair (suspicious for a control). Controls c01–c03 within ±14 ms.
- Workflow from here: Anas listens, writes verdict + approved shift/gains into each pair's `approved` block (or tells me and I
  write it); then `pytest -m regression tests/test_regression_set.py` guards those numbers (renders each approved pair to tmp).
  The kick-hit verifier (one strong kick per beat, scratchpad kicks_seam1.py) is the first method to test against this set.
- Not yet committed. Files touched today: src/audio/audio_mixer.py (loudness anchors, stretch cache, force_bars, cue_overrides,
  at_s/end_s), scripts/make_mix.py, scripts/regression_set.py, scripts/diag/regression_scan.py, scripts/diag/regression_pick.py,
  tests/test_audio_mixer.py, tests/test_regression_set.py, tests/regression/manifest.json, pyproject.toml (regression marker).

### Gain ride v3 (2026-09-08 evening, agreed with Anas): start = last real bars + 1.5 dB continuity cap, end = unity
- Why: the morning's body-level anchor lifted the Farrago → Heil overlap +9 dB (cap) while Farrago had gone into its break
  2 bars before the seam → a 4.6 dB step exactly at 7:00 of the regression set. The end anchor on B's post-overlap bars was
  the Mosaic fault. Every recipe has A at 0 volume by the last bar, so the overlap end IS B at its matched gain: g1 = 0.
- Rule now in `_overlap_gain_ride`: g0 = A's edge level (louder of the last two non-drop-out bars, counted back from the
  seam) minus the overlap opening's edge level, clipped ±9 dB, and if positive also capped so the first overlap bar is
  ≤ last-heard bar + SEAM_STEP_MAX_DB (1.5). g1 = 0 always. b_body/anchor_end kept in the signature, unused.
- Tests: 11 targeted gain tests pass; `test_overlap_gain_anchors_to_both_bodies` now expects g1 = 0;
  new `test_gain_ride_opens_at_the_last_bars_not_the_body` (Farrago shape, g0 ≈ +6.5 by the cap).
- Regeneration time with caches: 15 pairs in 8 min (was 21 min first build); Mosaic 15-min set 4 min.
- Numbers after regen (scripts/diag/regression_seam_levels.py): all end gains 0; Farrago lift 9.0 → 5.4 (seam step +2.8,
  cap missed because a_body bars were chunked from the window start; fixed by chunking from the seam, p04 re-rendered);
  Materium 9.0 → 0.0; p07 5.2/3.2 → 1.0/0. Flagged "steps" that are CONTENT, not gain: c02/c03 one-bar drop-out before
  the phrase line (correct), c01 seam on RSquared's own drop (+7.5 → control replaced by Andruss → Binga with measured-steady
  cue bars), c03 re-pinned (A 152 / B 64), p06 A's outro starts at the seam (cue choice), end steps p01/p02/p03/p10 = B's
  section change on the phrase after the overlap (natural).
- Anas has NOT yet listened to v3. Files: data/external/regression_set/ (flac + mp3 + README + results.json,
  results_before_edge_unity.json = numbers before v3); listen/techno_build_15min_v2.flac and the 3:30–5:10 clip are v3.
- Final v3 numbers (4-bar medians each side, scripts/diag/regression_seam_levels.py): no gain-caused step remains. Controls
  c01–c04 within ±0.4 dB at the seam. Content steps to ear-test: p04 7:00 overlap sits 8–12 dB under body (Farrago cue-out in
  its break, cue problem); p10 19:00 Katoff's own drop hits at the seam (+6, my cue pin at the drop start); p06 11:00 −2 dB
  into the seam (Nduduzo outro at cue-out); end-of-overlap jumps p02 4:00 (+8), p03 6:10 (+11), p09 17:45 (−10), p01 1:59
  (−14, Mosaic break) = B's section change on the phrase after the overlap. Beat suspicion (not loudness): p08 −143, p09 −180,
  c04 −94 ms consensus shifts. c01 replaced (Andruss → Binga), c03 re-pinned; both by measured-steady cue bars.
- DJ mixes downloaded for analysis: data/raw/dj_mixes/ (28 sets, 2.2 GB, manifest.csv; scripts/data/fetch_dj_mixes.py).
  Two are shared bills: Fred again.. & Thomas Bangalter (USB002), Sub Focus/Dimension/Culture Shock/1991 livestream.

### Anas's verdicts on regression set v3 (2026-09-08 night) + kick-hit measurement
- Loudness: "fixed now" (gain ride v3 approved by ear). Beat matching "pretty good"; 11:00 (p06, drop type) "so good".
  21:00 (p11) "off", and he heard misalignment "around 20:40" — 20:40 is silence in the file (p10 audio ends 20:19, p11
  starts 20:50); asked him which seam he meant. 29:00 (c04): seam quieter than the music before/after — measured: Factor B's
  first 16 bars are 3–5 dB under its own later body and the LUFS match uses the whole window, AND the blend hands A over at
  60 % while B sits at −6 dB support until 100 % → bars 10–12 of the overlap are B alone at −6 dB (−20 dB hole at 29:18).
- Transition TYPE choice needs work (his note; some seams flawless, some types inappropriate) — later.
- Kick-hit check over all 15 pairs at the applied shifts (scripts/diag/regression_kick_hits.py, output kick_hits_v3.txt):
  decisive (IQR ≤ 40 ms, ≥ 0.8 kicks/beat both sides) on p01 −104 (false shift confirmed), p04 +41, p07 0, c03 −3, and p10
  −174..−192 in 6/8 windows; unreliable (IQR > 100) on p02, p03, p05, p06, p08, p09, p11, c01, c02, c04 — sparse/irregular
  kick detection, DnB. Conclusion: usable as a gated VERIFIER (confidence gate), not a replacement for the correlation.
- 21:40 (p11 PARAFRAME → Koyah, corrected from "20:40"): kick hits are unusable on this pair (PARAFRAME kick share 0.19,
  Koyah low end syncopated → detections scattered in every window). Hat hits: A's loudest hat sits −190 ms off its own
  grid (off-8th pattern) for bars 0–18, then moves ON the beat from bar 20 (21:37); Koyah's hat placement wanders bar to
  bar (+96, −86, +204, +211 …). From 21:41–21:52 the two records' loudest hats sit half a beat apart in 21 of 21 hits.
  So what Anas hears at 21:40 is the hat patterns of two off-8th-hat records changing against each other while B leads,
  not a shift error that one number fixes; a grid residual of ~70 ms cannot be excluded (hats in the mid section cluster
  at −60..−80 ms) but cannot be proven with kicks. Fix direction: recipe/EQ (take A's highs out earlier) or pair choice.
  Scripts: scratchpad p11_detail.py / p11_hats.py.

### Kick-hit verifier — BUILT 2026-09-08 night (Anas: "do it")
- `kick_hit_residual(tail, head, bpm)` in audio_mixer: one strong kick per beat per record (loudest rising kick-band frame
  within ±0.6 beat, > 6× median, ≥ 0.75 beat apart), nearest-neighbour offsets per 4-bar window. Gate: ≥ 0.8 kicks/beat both
  sides, ≥ 4 hits per window, window medians within 40 ms, |residual| ≤ beat/3. Applied after the correlation's shift passes:
  if the gate passes and |residual| > 30 ms, b0 moves by the residual (+ = head late → earlier); report `kick_verify_ms`,
  `seam_shift_total_ms`, seam_band gets "+kick-hits". 4 synthetic tests (late head, sparse gate, > beat/3 refused, aligned).
- Regression set re-rendered: only p01 corrected (+121 → total +14 ms; kicks now within ±6 ms in 7/8 windows); p07 and c03
  confirmed; the other 12 untouched (gate closed). p04 (+41 ms, two break windows at +150 break the 40 ms agreement) and
  p10 (−180 in 6/8 windows, two at −84) stay uncorrected by design — candidates for a "majority of windows" gate later,
  after Anas hears p01. Levels unchanged. results_before_kickverify.json keeps the previous numbers.
- Offline check scripts/diag/regression_kick_hits.py now measures at `seam_shift_total_ms`.

## KEY REDESIGN — BUILT 2026-09-08/09 (policy changed from the written design, agreed with Anas)
Two facts changed the design. (1) One semitone moves a key 7 Camelot places, two semitones 2 places, so ±2 st can fix EVERY
same-mode distance; the "about 1 join in 3 shifted" estimate was wrong. (2) The GBM edge model already sees Camelot distance,
so with the veto off the planned clash share is 22-41 % by genre at weight 0 (measured, 27 joins per genre).
Policy agreed: shift ONE semitone only, and only when it brings the pair within one step (fixes distances 4-6, e.g. Cm→F#m
becomes Cm→Gm, the move Anas saw a DJ make); distances 2-3 and major/minor mismatches are not shifted; beyond 2 steps an
unshifted join is a clash (short overlap via the compatibility tiers + bass swap in the first half-phrase).
Code (small modules, per Anas's rule):
- src/audio/key_shift.py: camelot_distance, shift_key, choose_shift (ties → up), plan_joins (a shifted key carries forward).
  CLASH_DISTANCE = 2.0 is the single definition of "clash" (key_rules imports it).
- audio_mixer: `_stretch(..., semitones=)` does stretch + pitch in one rubberband pass (cache key includes semitones;
  rate-1 case uses pyrubberband.pitch_shift because time_stretch skips the engine at rate 1); `_apply_key_joins(tracks)`
  runs after prepare and before loudness matching, `_pitch_shift_track` re-renders B from source; `_start_anchored_swap`
  for clashes; report carries keys, cam_dist, key_shift_semitones, key_clash. The mixer's own _CAMELOT table is gone.
- src/models/key_rules.py: REAL_CLASH_SHARE per genre, KEY_WEIGHT (−0.2 all genres), key_term, max_clashes = joins // 2,
  clash_allowed / sequence_allowed (never two in a row). predict_model: Camelot veto removed (MAX_CAMELOT_DIST gone), key
  taken out of the compatibility term (0.0), soft term added, cap enforced in the beam and in repair_candidates; plan JSON
  has n_clashes. make_mix plan.json rows carry key_shift_semitones / key_clash.
- Sweep (scripts/diag/key_weight_sweep.py): weight 0 → 30/41/41/33/37/22 %; any negative weight → 44 % (the cap, 4 of 9);
  positive → 0-15 %. Real: 42/43/53/55/54/51. Cap binds for techno, tech house, DnB, afro (7-11 points under) — raising the
  cap for those genres is an open by-ear decision.
- Tests: tests/test_key_shift.py (7), tests/test_key_rules.py (3), planner test updated for the veto removal; render tests pass.
- Regression set re-rendered with the policy (catalog keys predict: p11 21:00 and c01 23:00 get +1 st shifts, p07 13:00 and
  p10 19:00 become clashes; the mixer measures its own keys so the render decides). results_before_key.json = numbers before.

### Key detection verification (2026-09-09) — no detector is trustworthy on MODE; root is ~3/4
- Ground truth: data/interim/key_ground_truth.csv (21 tracks, published keys from Beatport/Tunebat via search snippets — themselves
  algorithmic and unverified; two conflicting). Compare script: scripts/diag/key_detector_compare.py (--cnn for madmom).
- Results (exact or relative key): essentia edma 14/21, krumhansl 14/21, bgate 13, catalog(preview) 10, madmom CNN 8 (13.9 s/track),
  MusicalKeyCNN (a1ex90) 8 (2 s/track, calls 20/21 minor). Root note right ~15/21 for all; mode is the disagreement.
  Green/amber/red zoning (essentia+CNN agree / same root / differ) does NOT reach 90 % in green (8/13): the two agree on the wrong
  mode together.
- Ear tests: sine-chord major/minor test was useless for a non-musician (Anas). Mix test works: render the disputed track into a
  trusted partner unshifted and shifted so each mode hypothesis predicts "in tune" for one clip
  (data/external/key_ear_test/mix_test/, scratchpad mode_mix_test.py). Peetu S — Mental Component: Anas first said B minor, then said it was a guess — WITHDRAWN
  (CNNs right; Beatport, essentia edma and the spectral third-hint wrong). n = 1.
- madmom installed by Anas (pip, needs gcc + cython, --no-build-isolation) and patched for py3.10/numpy 1.26
  (collections.abc.MutableSequence, np.float/np.int) — site-packages only, not in the repo. MusicalKeyCNN clone lives in scratch.
- Key policy code (planner + mixer, uncommitted) is on hold until the detector question is settled. Standing recommendation:
  decide on the ROOT only, mode unknown. Regression-set render with the key policy was stopped; results.json restored.

### Fred again.. style set (2026-09-09, Anas's request) — data/external/listen/fred_20min.flac (17.4 min, 7 tracks, 126 BPM)
Pool: Fred's own catalog tracks (21 after de-duplicating remix versions, 122-131 BPM) + 14 tracks other DJs play right beside
his (his own tracklists are NOT in our data; the Boiler Room London 2022 list from set79.com had none of its non-Fred tracks in
the catalog). Planner run with the one-track-per-artist rule lifted for Fred only (monkeypatch in scratchpad fred_set.py), curve
peak, key step disabled (unverified detector), play_minutes 2.5, total 20 (one fetch failed → 7 tracks → 17.4 min).
Seams: 2:17 blend, 4:49 wave, 6:51 blend, 9:31 fade, 11:41 slam, 14:02 fade. Not implemented and visible here: per-track play
length variation (his 90 s cuts), DJ-specific pool/tracklists (Phase 9 dj_profiler). Awaiting Anas's ear.

### TODO before the model retrain: add Fred again.. to the tracklist scraper
`src/data/tracklists1001_client.py` hardcodes 52 DJ pages; Fred again.. is not one (only a shared Fisher radio session mentions
him). His page: https://www.1001tracklists.com/dj/fredagain../index.html. Add the entry, scrape his sets, fetch the tracks he
plays that the catalog lacks, then retrain (Model A/B/GBM) so his style is learnable. Anas, 2026-09-09: "we will add him before
retraining". Same applies to any other DJ he wants profiled (Phase 9).

### Harmonic fit from audio (2026-09-09) — built as an experiment in src/audio/key_shift.py, NOT validated, NOT wired in
Idea (agreed): decide shifts from the two records' chroma over the overlap, not from key labels. Tested on 11 real overlaps
(scratchpad fit_test.py / fit_test2.py): plain cosine of mean chroma is useless (0.83-0.99 for every shift); an interval-consonance
score is flat (0.08-0.11); centred correlation discriminates (-0.4..+0.8) but picks −1 st on Peetu→Koyah where Anas heard +1 in
tune, and would shift 2 of the 4 plain controls. Anas's Peetu verdict compared different partners (Factor B vs Koyah), so it is
not clean either → rendering Peetu S into Koyah at −1/0/+1 st (data/external/key_ear_test/shift_test/) for a clean ear test.
Until a fit measure agrees with clean ear tests, no key action in the mixer. The chroma helpers in key_shift.py are unused.

### KEY — final state 2026-09-09 (Anas's decision after the −1/0/+1 ear test: "I cannot feel any difference")
Mixer: NO key action. `_apply_key_joins`, `_pitch_shift_track`, `_start_anchored_swap`, the semitone option of `_stretch` and the
chroma helpers are removed; `src/audio/key_shift.py` keeps only `camelot_distance` + `CLASH_DISTANCE`. Report still carries
`keys` and `cam_dist` (estimated labels). Planner KEPT: veto removed, per-genre soft key term (KEY_WEIGHT −0.2), clash cap
(≤ half the joins, never two in a row), key out of the compatibility term (src/models/key_rules.py). Caveat Anas raised and I
accept: a wrong label on a melodic pair far apart can still let a grinding overlap through — watched by ear on melodic/trance
sets; the compatibility tiers (label-based) still shorten far-key overlaps. Tests: 44 targeted pass. UNCOMMITTED (Anas commits).

## LOCKED PLAN — "Data-driven DJ mimic" (agreed 2026-09-09, discussion only so far, nothing built)
Order of work, each step verified before the next:
1. Analytics notebook on the data we already hold (tracklist_clean.csv: 2,834 mixes with start times; features.parquet: 28,460
   tracks; play counts; mix_profiler curve shapes). Questions: play length by energy trend around the change (rising / flat /
   falling), by DJ, by genre, by fame (play count), first/last slot; curve-shape distribution per DJ. No downloads needed.
2. Role table from the notebook: slot roles derived at PLAN time from the curve's movement (climb / hold / peak / release);
   each role → play-length range + transition length. Roles come from the prompt-driven curve, not from per-track precompute.
3. Energy check on rendered output: measured per-track energy of the mix vs the planned targets, in the render report; run on
   the existing listening sets first.
4. Loops + tension bass cut as ONE designed move, mixer-decided from role + audio, reported, ear-tested on the regression set:
   loop = 4/8-bar phrase with identity (vocal/melodic, low kick share) at the outgoing cue-out, repeated 2–4× with movement
   (filter/bass cut/level), ends on a phrase line where the incoming drop/downbeat lands; bass of both records off during
   tension, back on the drop; rarely (well under half the seams), role- and profile-gated. Side benefit: stationary tail →
   predictable seams on messy cue-outs. Recipe unchanged (loop is tail material; extra high-pass on the loop).
5. Intent parser (Phase 10): a capable LLM maps a GENERAL vibe prompt to a small structured plan (genres, tempo range, length,
   energy curve as numbers, mood words → roles/recipes, optional DJ reference); everything downstream deterministic; template
   prompts are examples of the same schema; gaps filled by data defaults (genre-typical curve/length/tempo).
Test points: notebook findings reviewed by Anas before the role table; role table checked against real sets by number; energy
check on existing renders; loops/tension on regression pairs by ear; intent parser on template + vague prompts by plan diff.
Still to discuss before building: transition type/recipe choice by pair/style/genre/profile; realistic evaluation of the models
(A, B, GBM) beyond our own loss/AUC; DJ profiling (Fred again.. as the first profile); Fred in the data + retrain; DJ gimmicks.

### Prior art for step 5/6 (found 2026-09-09)
- Kim, Yang, Nam (KAIST): "A Computational Analysis of Real-World DJ Mixes using Mix-To-Track Subsequence Alignment" (ISMIR 2020,
  1001Tracklists mixes → cue points/transition lengths) and "Reverse-Engineering The Transition Regions of Real-World DJ Mixes using
  Sub-band Analysis with Convex Optimization" (NIME 2021; per-band gain curves of each record; validated by reconstruction + 14-person
  listening test; beats linear crossfade). Code: https://github.com/mir-aidj/transition-analysis — reuse for the seam decomposition.
- DJtransGAN (Chen et al., ICASSP 2022): GAN sets EQ+fader from real mixes; listening tests "competitive with baselines" (= matched good
  rules, did not beat them). https://github.com/ChenPaulYu/DJtransGAN
- Vande Veire & De Bie 2018 (rule-based DnB auto-DJ): high quality when MIR analysis correct (91 % of tracks); quality limited by
  analysis, not rules. https://github.com/lenvdv/dnb-autodj-3
- Raveform (TISMIR): metrical/functional structure annotations of EDM tracks in DJ mixes — candidate benchmark for our segmenter.
Conclusion agreed: keep the deterministic mixer as the engine; learn the DECISIONS (track, length, overlap, swap timing) from measured
real seams; every learned part must beat the rule baseline in a blind listen before it replaces it.

## MASTER ROADMAP — "Data-driven DJ mimic" (final, agreed 2026-09-09; supersedes the short locked plan above)

**Principle.** The mixer's signal path stays deterministic and ear-verified. The learned parts make the decisions: which track, how
long, which move, what parameters. Every learned part must beat the rule it replaces in a blind listen before it replaces it; the rule
stays as the floor. Working order for every feature: verify the input measurement → unit tests → one pair → full set. Small modules,
one clear path, no fallbacks.

**The regression set is the standing test bed.** Every real failure becomes a pair; loop pairs, tempo-ramp pairs and future features get
pairs; the numbers-only test runs before any mixer change; its tracks are excluded from all training. Open item: the kick verifier's
gate is deliberately strict (a majority-of-windows gate would correct 7:00 and 19:00) — Anas's ear decides.

**Key, settled.** Planner keeps veto-off, soft key term (KEY_WEIGHT −0.2) and clash cap. Mixer does nothing with key. Root readings right
~3/4, mode unreliable in every detector (essentia 14/21, CNNs 8/21 vs published keys, themselves algorithmic); a ±1 st shift was inaudible
to Anas. Melodic/trance sets watched by ear for far-key overlaps (compatibility tiers still shorten them). Not reopened without a new ear
result.

**Step 1 — Analytics notebook (text data only).** Sources: 2,834 tracklists (1,443 with start times on ≥95 % of tracks), catalog 28,460
tracks with features, play counts, mix_profiler curve shapes, DJ/genre list. Questions: play length when energy rises/holds/falls around
the change, by DJ, genre, fame, first vs last slot; which tracks get long/short plays (famous, own, remixes); curve shapes per DJ/genre;
key relations DJs use; tempo drift within sets; tracks per hour per DJ. Output: notebook Anas reviews + findings table read by steps 2
and 8. Part two when the seam table arrives: transition behaviour by role/genre/DJ.

**Step 2 — Roles, play length and window choice (planner).** At plan time the curve gives each slot a target; the role is how it moves:
climb / hold / peak / release (+ first, last). Role → play-length range + transition-length range, numbers from step 1 (later step 5).
Nothing per track precomputed; `_choose_window` takes its length from the role instead of GENRE_PLAY_MINUTES and still flexes to section
lines. Farrago rule: a cue-out must not begin a breakdown unless the role asks for one (= the start/end slot logic rebuild). Test: role
statistics vs real sets; one rendered set with visibly short and long slots.

**Step 3 — Energy and loudness check on rendered output.** Per-track measured energy and loudness of the finished mix vs planned target and
set level, in the render report and regression README; flag when a slot misses by a set margin or the level moves. Run first on the five
listening sets and the Fred set.

**Step 4 — Clean evaluation before any retrain.** Split by mix AND DJ AND track; regression tracks excluded. Every model number next to
baselines on the same held-out data: random, popularity, BPM+key heuristic. Metrics: next-track hit@k / MRR (Model B), pair AUC (GBM,
Model A) vs heuristic, set-level share of planned pairs adjacent in real sets vs chance. New pair feature: rhythmic compatibility (the
21:00 hat-pattern clash). Blind listening protocol: same pool, three orders (model / rules / a real DJ's), same mixer, Anas picks. Known:
Model A raw 0.622 vs BPM-only 0.624; B's +17 % MRR is relative; old split leaked.

**Step 5 — Seam pipeline (real transition ground truth), local first, then VM.** One resumable command: mix pages from the 1001tracklists
list (they have start times) → download mix audio from the page's link → fetch + verify the two tracks around each seam (fetcher +
fingerprint) → mix-to-track alignment → per-band gain decomposition with the two known sources (KAIST: Kim et al. ISMIR 2020, Kim/Yang/Nam
NIME 2021; code mir-aidj/transition-analysis). One row per seam: overlap bars, cue-out/cue-in on each track's structure, bass swap timing,
filter/EQ moves, level ride, cut vs blend, tempo change, DJ, genre, slot role. Test locally on 5–10 mixes of DJs we hold tracks for; run
on the VM (raw audio stays there, seam table + alignments come back); first 50 seams checked by ear before anything trains on them. The
28 mixes downloaded 2026-09-08 (data/raw/dj_mixes) only if their tracklists get scraped; swap the two shared bills. Adding DJs later is a
re-run, Fred first.

**Step 6 — Learned transitions and tempo ramps replace the type rules.** Recipes become parameters (overlap bars, bass-swap point, EQ
shape, level ride, cut/blend, tempo change) predicted from pair features + rhythmic compatibility + slot role + DJ profile, trained on
the seam table, evaluated on held-out seams (bar error, cut/blend agreement), then blind-listened against the seven rule types. Mixer
gains a per-seam tempo ramp within the stretch limits so a model-liked pair is not dropped for tempo alone. Style = conditioning on
genre/DJ. Rules stay as baseline until beaten. Prior art: DJtransGAN (ICASSP 2022) end-to-end only matched good rules → learn parameters,
not audio.

**Step 7 — Loops + tension bass cut (one designed move).** Mixer-decided from role + audio, reported, rare, role/profile-gated. Loop: 4/8-bar
phrase with identity (vocal/melodic over low kick share; block features + a vocal/melodic detector, a sub-task) at the outgoing cue-out,
repeated 2–4× with movement (filter opening, bass cut, level), ending on a phrase line where the incoming drop lands. Tension: both
records' bass off for a phrase or two, back together on the drop. Recipe unchanged (loop = tail material + high-pass). Stationary tails
tame messy seams. Loop pairs join the regression set; frequency by ear.

**Step 8 — DJ profiles, Fred in the data, retrain.** `dj_profiler.py` + `dj_profiles.json`: per DJ — pool (own tracks and share, artists,
labels), tempo, curve shapes, play length per role, key behaviour, transition parameters, move frequency; planner priors, role lengths,
transition conditioning; a prompt naming a DJ selects it. Fred: add https://www.1001tracklists.com/dj/fredagain../index.html to
src/data/tracklists1001_client.py (52 DJs now), scrape, fetch the tracks he plays that the catalog lacks, run his mixes through step 5.
Retrain A/B/GBM on the clean split with the re-keyed catalog (root only; key override table from full-track readings for tracks we hold,
growing with every render). Tests: profile statistics of a generated set vs the DJ's real sets; Anas's blind name-the-DJ test.
Expectation: fans say "in his style"; one-off edits/mashups are not reproducible.

**Step 9 — Intent parser (Phase 10).** A capable LLM maps a GENERAL vibe prompt to a small structured plan: genres, tempo range, length,
energy curve as numbers, mood words → roles/recipes, optional DJ profile, named tracks to include, a surprise knob (how often less likely
choices are taken). Everything downstream deterministic; templates are examples of the same schema; gaps filled from data defaults.
Named tracks fuzzy-matched to the catalog and pinned into slots; tracks outside the catalog via add-track-by-name (iTunes preview lookup →
features → fetch). Test: plan diffs template vs vague; one prompt with two profiles → visibly different plans.

**Step 10 — Gimmicks.** Small library of moves (volume cut/stutter, spinback, echo-out, filter sweep, a cappella drop, bass drop-out) as
options on any recipe, role/profile-gated, measured from mixes where step 5 can see them, otherwise ear-tuned. Last.

**Sequencing.** Build the step-5 pipeline first (long pole) and start the VM run; do steps 1–4 while it runs; extend the notebook when the
seam table lands; then 6–10. Steps 1–4 are not rebuilt by the seam data, they gain columns. Step 6 waits for data.

**Confidence.** High that learned decisions match real DJs better than rules on numbers; moderate that the first pass sounds better by ear
(measurement noise → the 50-seam ear check); low risk of regressing (rule floor). Prior art: KAIST mix analysis + transition
reverse-engineering (open code); DJtransGAN matched but did not beat rules; 2018 DnB auto-DJ: quality limited by analysis, not rules.
