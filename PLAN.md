# AI-DJ — Plan (rev. 2026-07-07)

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

## Status
- [x] embeddings fixed & re-embedded (`embeddings.parquet` = fixed; corrupt kept
      as `*_corrupt_backup.parquet`)
- [x] features merged (28,460 × [embedding + bpm/key/energy/…])
- [x] split file written
- [x] train_model.py: 7 bugs fixed (default dataset, false-neg masking,
      genre-blocked batches, genre-filtered mining, recalibrated band, cached
      inputs, vectorised metrics)
- [ ] ChromaDB re-index (`python src/features/vector_store.py` after `rm -rf data/processed/chromadb`)
- [ ] Train Model A: `conda activate aidj && python src/models/train_model.py`
- [ ] Eval vs 0.688 probe / 0.624 BPM baselines → then Model B
