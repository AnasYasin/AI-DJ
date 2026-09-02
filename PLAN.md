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
