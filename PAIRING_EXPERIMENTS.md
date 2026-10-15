# Pairing experiments: which kind of model learns what DJs play next

Every model tried for the pairing question, what it is, what it saw, and its numbers on the same
validation DJs, so nothing is tried twice and nothing is kept on a leaky number. The question is
Anas's (2026-10-13): "we want to know what type of model is capable of learning what DJs pick as next
track or playing together". Script: `scripts/experiments/pairing_yardstick.py`, one run about 2.5 min
on the laptop; tables in `reports/pairing_yardstick_2026-10-13*.csv`.

## The yardstick

**Data.** `data/dataset/2026-10-14/`: 37,612 tracks with features, 65,513 plays from 459 DJs. Pairs are
two consecutive plays in one mix, same genre, both with features. Training pairs come from the
training DJs (60,000 sampled); validation pairs from the 58 validation DJs, whole DJs the models never
saw (`split`, `DATA_ARCHITECTURE.md`): 6,297 pairs, of which 2,594 from the measured sources (Raveform,
DJ data), where the pair is a seam that was heard.

**Negatives.** Random records the planner could have chosen instead: same genre, BPM within 5 %, another
artist. So every number is "inside the feasible set", which is where the planner's score acts; BPM and
genre cannot do the work.

**Two tasks.**
- `pair`: a chosen pair against one random feasible pair, AUC.
- `rank`: the true next record among 99 random feasible candidates, hit at 10 and MRR, which is what
  the beam search does at every step. Random gives 0.10.

## The models, and what each saw

| model | what it is | trained on |
|---|---|---|
| random, bpm closeness, key closeness | the trivial scores | nothing |
| popularity | log play count of the candidate | training plays |
| compat formula | `src/features/compatibility.py`, the hand formula the mixer caps overlaps with | calibrated once on the old data |
| raw effnet cosine | cosine of the two whole-track discogs-effnet means | nothing (frozen embedding) |
| Model A cosine | the saved contrastive head, `models/contrastive_encoder.pt` | run 3: the OLD split, which held these validation DJs' old-data mixes: leaky, kept as a warning. Retrained on the new split below |
| crate context | cosine between the candidate and the mean vector of the set so far | nothing |
| GBM, 6 pair features | `HistGradientBoosting`, 300 trees: log BPM ratio, Camelot distance, energy, loudness and onset gaps, raw cosine | training pairs, 2 feasible negatives each |
| GBM, rich pair features | the same plus each record's own scalars and their gaps (bpm, energy, loudness, onset, centroid, energy spread, release year, artist listeners), 31 features | the same |
| pair MLP (one record encoder per record, dot product) | one ReLU network per record, 1,288 to 512 to 128, dot product, 8 epochs, no seed | the same pairs and negatives |
| Model B | the saved context transformer, `models/sequence_model.pt`, candidates ranked by its context vector | the OLD split: leaky on old-data pairs |

## Results, run 3 (clean), 2026-10-13

All 6,297 validation pairs:

| model | AUC | hit@10 | MRR | median rank of 100 |
|---|---|---|---|---|
| random | 0.503 | 0.107 | 0.055 | 49 |
| bpm closeness | 0.446 | 0.222 | 0.184 | 45 |
| key closeness | 0.545 | 0.211 | 0.141 | 39 |
| popularity (training plays) | 0.243 | 0.124 | 0.061 | 82 |
| compat formula | 0.543 | 0.166 | 0.082 | 43 |
| raw effnet cosine | 0.614 | 0.310 | 0.151 | 24 |
| Model A cosine (leaky) | 0.732 | 0.508 | 0.308 | 10 |
| crate context | 0.655 | 0.343 | 0.164 | 20 |
| **GBM, 6 pair features** | 0.795 | 0.521 | 0.379 | 9 |
| **GBM, rich pair features** | **0.818** | **0.572** | **0.417** | **7** |
| pair MLP (one record encoder per record, dot product) | 0.656 | 0.284 | 0.132 | 25 |
| Model B (leaky) | | 0.414 | 0.298 | 22 |

The 2,594 pairs from the measured sources only (no model has seen these DJs in any form):

| model | AUC | hit@10 | MRR | median rank |
|---|---|---|---|---|
| random | 0.509 | 0.099 | 0.050 | 51 |
| bpm closeness | 0.521 | 0.168 | 0.095 | 41 |
| key closeness | 0.544 | 0.216 | 0.137 | 38 |
| popularity | 0.226 | 0.140 | 0.058 | 84 |
| compat formula | 0.601 | 0.199 | 0.094 | 35 |
| raw effnet cosine | 0.678 | 0.390 | 0.186 | 16 |
| Model A cosine | 0.632 | 0.299 | 0.144 | 24 |
| crate context | 0.700 | 0.396 | 0.176 | 15 |
| **GBM, 6 pair features** | 0.801 | 0.505 | 0.321 | 10 |
| **GBM, rich pair features** | **0.837** | **0.567** | **0.356** | **8** |
| pair MLP (one record encoder per record, dot product) | 0.685 | 0.308 | 0.150 | 21 |
| Model B | | 0.187 | 0.088 | 52 |

## What it says

1. **Trees on pair features learn it; nothing else comes close.** The rich GBM puts the true next
   record in the top 10 of 100 feasible candidates 57 % of the time on DJs it never saw, median rank 7,
   AUC 0.82 to 0.84. The 6-feature GBM is 4 points behind it. Both are seconds to train.
2. **The frozen embedding is the floor, not the ceiling.** Raw cosine alone gives 0.61 to 0.68; the
   trees add 0.15 to 0.20 of AUC over it with the five scalar gaps.
3. **Model A adds nothing honest.** On the measured sources, where it never trained, it is under the raw
   cosine (0.632 against 0.678). Its 0.732 on all pairs is memory of the validation DJs' old mixes.
4. **Model B is near random on unseen DJs.** Hit at 10 of 0.187 on the measured sources, against 0.099
   random and 0.396 for a plain mean of the set so far. The 0.414 on all pairs is the same memory as
   Model A's. A 947k-parameter transformer on 2,400 sets does not generalise to a new DJ.
5. **A small ReLU network does not beat trees here.** 0.66 to 0.75 AUC across runs (it moves 0.05
   between runs, no seed, 8 epochs), always under the 6-feature GBM. With 60,000 pairs and tabular
   gaps this is the expected order, and it answers the "why not deep learning" question for this stage.
6. **Context helps a little.** The mean of the set so far beats the last record alone (0.655 against
   0.614, 0.700 against 0.678). Worth one feature in the pair model, not a model of its own.
7. **The compat formula is weak as a chooser** (0.54 to 0.60). It was calibrated to separate real pairs
   from random same-genre pairs, which BPM mostly does; inside the feasible set it has little left.
8. **Record popularity cannot be used for unseen DJs.** 92 % of records are played by one DJ, so a new
   DJ's next record has no training plays, and popularity is anti-correlated (AUC 0.24) by construction.
   Artist fame (Last.fm listeners) is external and stays a feature.

## Model A retrained on the new split, 2026-10-13

`train_model.py` as it is (two branches, NT-Xent, 50 epochs, best epoch by its own top-1 retrieval,
epoch 20) on the plays with the new `split`. The old encoder is kept beside it as
`models/contrastive_encoder_old_split_2026-07-07.pt`. Yardstick rerun with the new one
(`reports/pairing_yardstick_2026-10-13_modelA_newsplit.csv`, the other rows unchanged):

| set | AUC | hit@10 | MRR | median rank | raw effnet cosine on the same set |
|---|---|---|---|---|---|
| all validation pairs | 0.611 | 0.285 | 0.145 | 26 | 0.614 / 0.310 |
| measured sources only | 0.689 | 0.375 | 0.188 | 17 | 0.678 / 0.390 |

Honest, it equals the frozen embedding. Its own validation top-10 retrieval fell from 0.054 at
epoch 1 to 0.039 at epoch 50. Training did not change the vector in a way that helps on unseen DJs.

## Neural sweep, 2026-10-13: Model A shapes and Model B shapes, sized and judged for fit

Anas: "try different arch, models for model A and B, bigger or smaller models based on overfitting
or underfitting, more features etc". Script `scripts/experiments/neural_sweep.py`, results
`reports/neural_sweep_2026-10-13.csv`, log `logs/neural_sweep_2026-10-13.out`.

**Setup.** Same pairs, same feasible negatives and same validation DJs as the yardstick. The training
DJs are split again: 41 of them (10 %) are a dev set for early stopping, so the validation DJs are
never looked at while training. 32,990 training pairs, 10,082 dev pairs, 6,305 validation pairs.
Loss is BCE on the positive against 2 fresh feasible negatives per epoch (rotating through 6 sampled),
AdamW 1e-3, weight decay 1e-4, batch 1,024, up to 20 epochs, patience 3 on dev AUC, seed 0.
Inputs per record: `base` = unit effnet mean (scaled by 8 so the block is not swamped) + the 8
yardstick scalars z-scored; `rich` = plus the 13 MFCCs and the key as sin/cos on the Camelot wheel,
mode and an unknown-key flag. `gap` = the 31 rich pair features, z-scored on the training pairs.
"train AUC" is on pairs the model saw (so a gap to dev is expected); "fit" is my label:
train minus dev over 0.06 is "overfits", train under 0.75 is "underfits".

| config | what it is | params | best ep | train AUC | dev AUC | val AUC | hit@10 | MRR | rank |
|---|---|---|---|---|---|---|---|---|---|
| dot S base | record encoder 256→64, dot product (Model A's shape) | 347k | 6 | 0.769 | 0.675 | 0.710 | 0.364 | 0.169 | 19 |
| dot M base | record encoder 512→128 | 727k | 2 | 0.722 | 0.673 | 0.703 | 0.351 | 0.166 | 20 |
| dot L base | record encoder 1024→256, dropout 0.3 | 1.58M | 1 | 0.724 | 0.680 | 0.705 | 0.351 | 0.163 | 20 |
| dot M rich | record encoder 512→128, rich inputs | 735k | 6 | 0.805 | 0.692 | 0.730 | 0.370 | 0.174 | 17 |
| gap MLP | MLP 31→256→1 on the gap features only, no record encoder (the GBM's inputs) | 4k | 15 | 0.775 | 0.691 | 0.796 | 0.516 | 0.355 | 10 |
| pair S rich | record encoders 256→64, head MLP over [zi, zj, zi·zj, \|zi−zj\|, gap] | 388k | 3 | 0.851 | 0.727 | 0.826 | 0.528 | 0.316 | 9 |
| pair M rich | record encoders 512→128, dropout 0.3 | 875k | 4 | 0.872 | 0.735 | 0.833 | 0.544 | 0.334 | 9 |
| pair L rich | record encoders 1024→256, dropout 0.4 | 2.14M | 7 | 0.881 | 0.723 | 0.829 | 0.535 | 0.336 | 9 |
| **mean ctx M rich** | pair M + the mean of the set-so-far record vectors as context (head also sees ctx and ctx·zj) | 940k | 9 | 0.910 | 0.783 | **0.867** | **0.602** | **0.368** | **7** |
| gru ctx M rich | pair M + a GRU over the last 16 records | 1.04M | 9 | 0.913 | 0.748 | 0.836 | 0.548 | 0.343 | 8 |
| tfm ctx M rich | pair M + a 1-layer, 4-head transformer over the last 16, mean-pooled | 1.07M | 3 | 0.876 | 0.770 | 0.856 | 0.576 | 0.344 | 8 |
| tfm ctx L rich | the same at 1024→256, dropout 0.4 | 2.94M | 9 | 0.909 | 0.774 | 0.863 | 0.583 | 0.352 | 7 |

For reference on the same validation pairs: rich GBM 0.818 / 0.572 / 0.417 / 7; raw cosine 0.614 / 0.310.

**What it says.**

1. **Model A's shape is the limit, not its size.** A dot product of two record vectors cannot see the
   gap between two records. Small, medium, large, base or rich inputs: 0.70 to 0.73, and the large one
   peaks at epoch 1. Retrained for 50 epochs with the contrastive loss it equals the raw embedding.
2. **Give the network the pair and it matches the trees.** Same record encoders, but a head that sees both
   vectors, their product, their difference and the gap features: 0.826 to 0.833 AUC, 34 to 61 s of
   training. Size S/M/L changes 0.007. On the gap features alone the MLP (0.796) is under the trees
   (0.818), so the net's edge over the gap-only model is the record encoder over the embedding.
3. **Give it the set so far and it beats the trees.** The mean of the previous records' record vectors
   as context: 0.867 / 0.602, the first model over the rich GBM on hit at 10. Dev AUC 0.783 against
   0.735 without context, so it is signal on unseen DJs, not memory.
4. **Order within the set does not add.** GRU (0.836) and transformer (0.856 medium, 0.863 large, 35 min of
   training) over the sequence are under the plain mean. At 33k pairs a sequence model has nothing to learn beyond "what is in this
   set".
5. **Every model overfits by epoch 3 to 9** (train 0.87 to 0.91 against dev 0.73 to 0.78), and size
   does not change it, so it comes from the data: 92 % of records are one DJ's, a network can memorise
   which records go together. The dev DJs (41 random training DJs) are a harder set than the validation
   DJs for every model (0.78 against 0.867 for the best), so the "overfits" label overstates it.
6. **Recommendation.** One model for the planner, the pair head with mean context, in place of Model A,
   Model B and the GBM. DJ-agnostic, and it reads the plan so far, which is what the beam search has at
   every step. Not built yet.

**Follow-ups:** the five items (ranking loss, regularisation, seeds, candidate-aware context, extra
signals) are sweep 2 below. Not run: trees plus net rank average. Fine-tuning the effnet backbone on
audio is the only "bigger model" step left; GPU days, and 33k pairs do not justify it.

## Preview embeddings in training: does the old source hurt? 2026-10-14

Anas: "old data has preview embeddings, using them mixes for training is not causing any issues?"
Two measurements.

**How far a preview is from the full record.** 3,001 records have both a 30 s preview embedding
(`data/processed/features.parquet`) and a full-track embedding (the measured sources), unit cosine:

| | cosine |
|---|---|
| preview vs full, same record, median | 0.932 |
| preview vs full, same record, 10th percentile | 0.718 |
| preview vs full, different records, 90th percentile | 0.776 |

The preview picks its own full record first among the 3,001 only 67 % of the time (top 10: 76 %).
Scalars drift as well: BPM differs by 12 or more on 10 % of records (half or double tempo readings on
30 s), onset is lower on the full track (median −0.21, breakdowns are in it), loudness differs by
5.5 dB or more on 10 %. Inside the old mixes 8.5 % of plays now have full-audio features and the
rest preview features, so a gap feature there can be a measurement artefact.

**Does it hurt the model.** The best model (pair head + mean context) trained twice, scored on the
old (preview) and the measured (full-audio) validation pairs separately
(`neural_sweep.py --train-sources`, `reports/neural_sweep_2026-10-14_audio_*.csv`):

| trained on | train pairs | val AUC | hit@10 all | hit@10 old | hit@10 measured |
|---|---|---|---|---|---|
| all three sources | 32,990 | 0.865 | 0.605 | 0.590 | **0.628** |
| measured sources only | 7,199 | 0.745 | 0.408 | 0.312 | 0.546 |

The preview pairs help, even on the full-audio validation pairs: 0.628 against 0.546 without them.
Five times the pairs outweigh the measurement mismatch. So the old source stays in training. The
mismatch is still real (the model scores 4 points higher where the audio is full), and two things
would close it: full audio for the old records (Anas's plan, fetch the rest once the model is judged),
and, cheaper, an `audio` flag per record as an input so the model can discount preview-derived gaps.
Not tried.

## Sweep 2, 2026-10-14: loss, negatives, regularisation, seeds, context, signals

Anas: "try that" on the five follow-ups. All on the sweep 1 winner (pair head + mean context, medium,
rich inputs), same data, dev and validation as sweep 1. `neural_sweep.py --sweep 2`, results
`reports/neural_sweep2_2026-10-14.csv`. The first four ran under heavy swap (PyCharm and Chrome,
10.7 GB in swap), so their `seconds` are not comparable.

Reference, sweep 1 best (BCE, 2 random negatives, dropout 0.3, one seed): 0.867 / 0.602 / 0.368 / 7,
hit@10 old 0.590, measured 0.628.

| config | what changes | best ep | train AUC | dev AUC | val AUC | hit@10 | MRR | rank | hit@10 old | hit@10 measured |
|---|---|---|---|---|---|---|---|---|---|---|
| listwise 8 random | softmax cross-entropy over the positive and 8 random feasible negatives | 3 | 0.864 | 0.768 | 0.847 | 0.597 | 0.369 | 7 | 0.585 | 0.615 |
| **listwise 16 random** | the same with 16 | 5 | 0.878 | 0.781 | 0.858 | **0.615** | **0.392** | **6** | 0.595 | **0.643** |
| listwise 16 hard | 8 closest to A by raw cosine + 8 random | 12 | 0.915 | 0.757 | 0.836 | 0.558 | 0.367 | 8 | 0.543 | 0.578 |
| bce 8 hard | BCE, 4 hardest + 4 random | 9 | 0.915 | 0.760 | 0.841 | 0.530 | 0.342 | 9 | 0.524 | 0.539 |
| reg dropout .5 | dropout 0.3 → 0.5 | 10 | 0.900 | 0.786 | **0.875** | 0.607 | 0.377 | 7 | 0.596 | 0.622 |
| reg wd 1e-2 | weight decay 1e-4 → 1e-2 | 6 | 0.900 | 0.782 | 0.869 | 0.601 | 0.367 | 7 | 0.589 | 0.619 |
| reg input dropout .1 | dropout on the 1,305 input features | 5 | 0.888 | 0.772 | 0.863 | 0.589 | 0.359 | 7 | 0.579 | 0.603 |
| reg smooth .1 | label smoothing 0.1 | 4 | 0.883 | 0.827 | 0.866 | 0.595 | 0.351 | 7 | 0.590 | 0.602 |
| reg all | the four together | 6 | 0.878 | 0.773 | 0.863 | 0.594 | 0.356 | 7 | 0.592 | 0.595 |
| **5 seeds** | five initialisations, mean logit | 8 | 0.903 | 0.779 | 0.873 | **0.617** | 0.381 | **6** | **0.602** | 0.638 |
| attn ctx | the candidate attends over the set so far (query B, keys and values the set) | 5 | 0.896 | 0.770 | 0.856 | 0.571 | 0.351 | 8 | 0.557 | 0.592 |
| signals position + genre | position in the set and a learned genre embedding into the head | 6 | 0.901 | 0.779 | 0.867 | 0.596 | 0.367 | 7 | 0.578 | 0.620 |
| seam hold weight | measured pairs weighted by the incoming record's play length relative to the DJ's median, clipped [0.5, 2] (6,405 training pairs weighted) | 6 | 0.895 | 0.774 | 0.865 | 0.594 | 0.362 | 7 | 0.584 | 0.608 |

**What it says.**

1. **Training on the ranking helps.** Softmax over 16 random candidates: hit@10 0.615, MRR 0.392,
   median rank 6, and 0.643 on the full-audio pairs. 8 candidates is not enough (0.597). AUC drops a
   little (0.858) because the loss no longer optimises pairs.
2. **Hard negatives hurt, in both losses.** The closest feasible candidates by embedding push train
   AUC to 0.915 and validation down to 0.53 to 0.56. The hardest candidates are records a DJ could just
   as well have played; teaching the model to reject them is teaching it noise.
3. **Seed averaging is worth 1.5 points.** 0.617 / 0.873 from five initialisations. Pure variance
   reduction, so it stacks with everything else.
4. **Regularisation buys at most 1 point.** Dropout 0.5 gives the best AUC (0.875) and 0.607; weight
   decay, input dropout, label smoothing and all four together do nothing or hurt. The train-dev gap is
   the data (92 % of records belong to one DJ), not the optimiser.
5. **Attention over the set loses to the mean again** (0.571), like the GRU and transformer in sweep 1.
6. **Position, genre and the seam hold length add nothing.** The feasible set already fixes the genre,
   the context mean carries the position, and the hold weight moves 6,405 of 33k pairs by at most 2×.

**The combination** (ranking loss with 16 random negatives + dropout 0.5 + 5 seeds) is the final
model, next section.

**Pitfalls, so they are not repeated.** Piping the log through `grep -v` into a file buffers it until
the process exits (the system grep is ugrep, large buffer); write the log directly and filter when
reading. With PyCharm and Chrome open the laptop swapped 10 GB and one config took 2 h 14 m instead of
35 min; check `free -m` before a long run.

## The final model, 2026-10-14: the planner's one learned score

Anas: "finalise this model and change the code replacing the old model." `src/models/pair_model.py`,
saved as `models/pair_model.pt` (18.9 MB, gitignored like the other model files), trained by
`python -m src.models.pair_model` in 5 h 42 min on the laptop (log
`logs/pair_model_train_2026-10-14.log` is empty: an imported module installs the logging handler
first, fixed with `force=True` for the next run). The planner (`predict_model.py`) scores with it
in place of Model A, Model B and the GBM, in `plan_mix` and `repair_candidates`; the Model A
projection is no longer computed for a pool.

```
record A ──► record encoder ──► zA (128) ─────────────────────────┐
record B ──► record encoder ──► zB (128) ─────────────────────────┤
set so far ─► record encoder ──► z₁..zₖ ──► mean ──► c (128) ──────┤
A, B ───────► 31 gap features, z-scored ──────────────────────────┤
                                                                   ▼
                         [zA, zB, zA⊙zB, |zA−zB|, gaps, c, c⊙zB]  (799)
                         FC 799 → 256, ReLU, dropout 0.5, FC 256 → 1 ──► logit

record encoder: FC 1,305 → 512, batch normalisation, ReLU, dropout 0.5, FC 512 → 128
input per record: unit effnet mean × 8 (1,280), 8 scalars z-scored, 13 MFCCs z-scored,
                  key as sin/cos on the Camelot wheel, mode, unknown-key flag
```

Training: softmax cross-entropy over the true next record and 16 random feasible candidates
(same genre, BPM within 5 %, another artist; 32 sampled per positive, rotated per epoch), AdamW
1e-3, weight decay 1e-4, batch 60 groups, up to 20 epochs, early stopping with patience 3 on the
dev AUC (41 of the 401 training DJs), five initialisations averaged at inference (mean logit).
32,990 training pairs. The z-scoring statistics (per-record scalars and MFCCs over the records,
gap features over the training pairs) are saved in the checkpoint and applied to every pool;
a pool is never normalised on itself.

Judged on the 58 validation DJs, the same 6,305 pairs and feasible candidates as the sweeps:

| model | val AUC | hit@10 | MRR | median rank | hit@10 old | hit@10 measured |
|---|---|---|---|---|---|---|
| raw effnet cosine | 0.614 | 0.310 | 0.151 | 24 | | |
| Model A, retrained | 0.611 | 0.285 | 0.145 | 26 | | |
| rich GBM | 0.818 | 0.572 | 0.417 | 7 | | |
| sweep 1 winner | 0.867 | 0.602 | 0.368 | 7 | 0.590 | 0.628 |
| best single change (5 seeds) | 0.873 | 0.617 | 0.381 | 6 | 0.602 | 0.638 |
| **final (ranking loss 16 + dropout 0.5 + 5 seeds)** | **0.876** | **0.649** | **0.420** | **5** | **0.640** | **0.662** |

Dev AUC 0.783 (mean of the five seeds, best epochs 16, 12, 5, 10, 13). The three changes stack:
0.649 against 0.615 and 0.617 alone, and above the 0.62 to 0.63 expected. The true next record is in
the top 10 of 100 feasible candidates two times in three on DJs the model never saw, median rank 5.

**In the planner.** Score per step is `W_PAIR · z(logit) + W_ENERGY · energy fit + key term`
(`+ compat_weight · compatibility` when asked), the logit standardised within the candidate set as
the context term was before, `W_PAIR = 1.0` in the slot of the old `W_CTX = 1.0` and `W_GBM = 1.0`
together. The weights have not been tuned, as before. The set so far is the beam's own tracks, the
current record included, the last 16 at most. In `repair_candidates` the pair term uses the records
before the slot; a first slot has no previous record and keeps energy and compatibility only.

Anas, 2026-10-14: the old models go. Deleted `train_model.py` (Model A), `train_sequence.py` (Model B),
`edge_scorer.py` (the GBM) with `models/contrastive_encoder.pt`, `encoder_norm.npz`, `sequence_model.pt`,
`edge_gbm.pkl`, and `key_rules.py` with its sweep script and tests. `TrackArrays` and the Camelot helpers
moved to `src/models/track_arrays.py`. The yardstick script lost its Model A and Model B rows; their
numbers stay in the tables above and in `reports/pairing_yardstick_2026-10-13*.csv`. `W_PAIR = 1.0`
stays untuned.

## The planner's hard rules against the real plays, 2026-10-14

Anas: "the hard rules at the end do we need them or not check that too." At every step the planner
vetoes a candidate that is more than 5 % off the current record's BPM, a record already in the set,
a record by an artist already in the set, and a key clash (Camelot distance over 2) when the set has
hit the cap (`key_rules.py`: at most half the joins, never two in a row). How often the real plays
break each rule, over every mix with two or more records with features (3,929 mixes, 54,306 joins)
and over the measured sources alone (1,219 mixes, 13,045 joins, BPM from the full audio):

| rule broken | all sources | measured sources |
|---|---|---|
| join with a BPM gap over 5 % (half/double readings are 0.9 % of joins) | 22.5 % of joins | 17.4 % |
| join with a BPM gap over 7 % | 17.1 % | 11.4 % |
| mix with any join over 5 % | 70 % of mixes | 49 % |
| next record by the same artist | 9.0 % of joins | 4.4 % |
| mix that plays an artist twice anywhere | 60 % of mixes | 40 % |
| mix that plays a record twice | 4.1 % | 2.1 % |
| key clash | 57.5 % of joins | 58.9 % |
| mix over the clash cap (more than half the joins) | 65 % of mixes | 64 % |
| mix with two clashes in a row | 90 % | 84 % |

Per genre, all sources: a BPM gap over 5 % on 16 % of afro house joins, 39 % drum and bass, 23 %
house, 17 % melodic house, 20 % tech house, 21 % techno, 16 % trance. Two clashes in a row in 81 to
97 % of mixes in every genre; melodic house and trance clash least (49 to 50 % of joins), tech house
and techno most (62 %).

1. **BPM.** A DJ pitches, so the records' own BPMs differ by more than 5 % on one join in six. The 5 %
   is the mixer's stretch limit (rubberband warns above 4 %, raises above 8 %), not a DJ habit. It
   stays as a mixer constraint; whether the model would keep to it on its own is measured below.
2. **One record per artist.** Broken somewhere in 40 to 60 % of real mixes, but only on 4 to 9 % of
   joins, and a real mix is 10 to 13 records long. A taste rule, cheap inside a pool of thousands.
3. **The key clash cap** is contradicted by the data. Real sets put two clashes in a row in 84 to 90 %
   of mixes and go past "half the joins" in 64 %. The cap was written so a short set is not clashes
   end to end; the real sets mostly are. The pair model sees the Camelot distance as an input and
   the −0.2 key term exists only to reach the cap. Whether both can go depends on the clash share the
   model plans on its own: measured below, after training.

**Does the model keep to the BPM and artist rules on its own?** The 6,305 validation pairs scored
again with the final model, the true record among 99 random same-genre candidates of ANY BPM and
ANY artist, the set so far as context:

| | share of candidates | share of the model's top 10 |
|---|---|---|
| within 5 % BPM of the current record | 46.8 % | **4.6 %** |
| by the same artist as the current record | 0.11 % | 0.19 % |

hit@10 in that wider set: 0.136 (random 0.10; 0.649 inside the feasible set). The model does not
keep to the BPM rule; it inverts it. It was trained only inside the feasible set, so it has never
seen a far-BPM candidate, and the gap features of one (z-scored on within-5 % pairs) are far
outside anything the head was fitted on. A network extrapolates there, and here it extrapolates
upward. So the BPM veto is not optional for this model: the feasible set is its training
distribution, and the model is only valid inside it. The same holds for the artist rule in a
weaker form (the model leans toward the same artist, 2× the base rate, on tiny numbers).

**What clash share does the model plan on its own?** 54 sets planned with the final model
(6 genres × build, peak, chill × 10 records, the planner's default pool for each genre's band),
clash share of the 9 joins per set, against the real share per genre from the table above:

| planner | techno | tech house | melodic | trance | dnb | afro | all | sets with two clashes in a row |
|---|---|---|---|---|---|---|---|---|
| real DJs | 0.62 | 0.62 | 0.49 | 0.50 | 0.61 | 0.58 | 0.57 | 84 to 90 % |
| as is: cap + key term −0.2 | 0.44 | 0.44 | 0.44 | 0.15 | 0.15 | 0.44 | 0.35 | 0 of 18 |
| cap off, key term −0.2 kept | 0.96 | 1.00 | 0.78 | 1.00 | 0.33 | 1.00 | 0.85 | 16 of 18 |
| cap off, key term 0: the model alone | 0.41 | 0.52 | 0.41 | 0.00 | 0.33 | 0.56 | 0.37 | 12 of 18 |

The −0.2 key term without the cap plans clashes almost everywhere (0.85): it was sized to push
against the cap and is wrong on its own. The model alone lands where the cap and the term together
land (0.37 against 0.35), within 2 to 10 points of the real share in four genres, and it puts two
clashes in a row in two sets of three, as DJs do. Trance at 0.00 and drum and bass at 0.33 are 27
joins each and under in every setting, so they are the pool or the model, not the rule. So the cap
and the key term can go together, with the model's clash share as the number to watch, and they
cannot go one without the other.

**Why trance and drum and bass plan so few clashes.** Two measurements, 2026-10-14. (1) It is not the
pool: every pool (3 BPM bands per genre, 600 to 2,344 records, no unknown keys, the top 3 keys hold a
third of the records) offers a clash on 0.65 of random feasible pairs, yet over 15 planned sets per
genre the model plans 0.00 to 0.07 in trance, 0.18 to 0.24 in drum and bass, 0.42 to 0.51 in techno.
(2) On the validation pairs, among the true next record and 99 feasible candidates:

| genre | real next record | candidates | model's top 10 | model's top 1 |
|---|---|---|---|---|
| all | 0.59 | 0.64 | 0.49 | 0.35 |
| trance | 0.48 | 0.64 | 0.48 | 0.13 |
| drum and bass | 0.58 | 0.64 | 0.49 | 0.38 |
| techno | 0.63 | 0.65 | 0.50 | 0.42 |
| afro house | 0.59 | 0.63 | 0.37 | 0.27 |

DJs clash a little less than chance (0.59 against 0.64). The model's top 10 is close to that (0.49, in
trance exactly 0.48), its top 1 is far below it (0.35, trance 0.13). So the key preference it learned
is mild and about right, and the beam search amplifies it: the best of 1,500 candidates is nearly
always an in-key record, because in-key is a small but consistent plus. A real DJ does not take the
best of 1,500. The same amplification applies to every mild preference the model has, so planned
sets are more typical than a typical set. Not a model fault; a search property. Options, not built:
sample each expansion from the softmax of the scores instead of taking the top 8; or accept it,
since the mixer does not act on key. To decide on after the whole-set judgement (same-key share,
energy shape, BPM climb of planned against real sets).

Anas, 2026-10-14: remove both. The planner has no key rule now; `n_clashes`
and `cam_dist` stay in the plan as reports. The number to watch on every retrain is the planned clash
share against the real 0.57.

## What was wrong in runs 1 and 2, kept so it is not repeated

- **Run 1** (`..._leaky_first_run.csv`): rich GBM AUC 0.898, hit at 10 0.713. Two leaks. `play_count`
  from the export counts every play including the validation play itself, so the true next record
  always had one; and the Model A cosine was a feature, carrying its memory of the validation DJs'
  old mixes. Popularity alone scored 0.416 hit at 10 on that run.
- **Run 2** (`..._run2_popularity_inverted.csv`): popularity from training plays only and no Model A
  feature. Popularity flipped to AUC 0.24 and the rich GBM collapsed to 0.49, because on training pairs
  the positive always had plays and the trees learned "no plays, no pair", which inverts on unseen
  DJs. Record popularity left the features for run 3.

## Retraining with more and better data: what to try, in order

Every number above was set by the data, so each item says which number to look at again when the
data grows. The yardstick stays the judge: a change is kept only when it beats the saved model's
validation hit@10 on the same validation DJs (`python -m src.models.pair_model` prints it; the
sweep scripts take `--only NAME` for one idea at a time). New DJs go into the split by whole DJ.

1. **Look at the train-dev gap first.** Every net overfits by epoch 3 to 9 (train AUC 0.90, dev
   0.78) because 92 % of records belong to one DJ. More DJs is worth more than any change to the
   model. If the gap closes, the bigger shapes (pair L, 2.1M params, 0.829 today) get a second look.
2. **Full audio for the old records.** The model scores 4 points higher where the audio is full
   (0.643 against 0.595 hit@10), and preview BPM is off by 12 or more on 10 % of records. Until the
   old records are fetched, the cheap version is an `audio` flag per record as an input. Not tried.
3. **A sequence model over the set.** The transformer context was the runner-up (0.583 against
   0.602 for the mean) and is the shape that gains from more sets. Rerun
   `neural_sweep.py --only "tfm ctx M rich"` when the sets have grown by about 3×.
4. **Hard negatives.** They hurt now (0.53 to 0.56) because the closest feasible candidates are
   records the DJ could just as well have played. With more plays per record they become real
   negatives. Retry at 100k pairs or more.
5. **Trees and net averaged by rank.** Not run. The rich GBM is 0.572; a 50/50 rank average with the
   net is a 10-minute check in `neural_sweep.py`.
6. **Not worth it before the data is 10×:** fine-tuning the effnet embedding (GPU days), wider or
   deeper record encoders (S/M/L differ by 0.007), order-aware context (every variant lost to the
   mean), weight decay, input dropout, label smoothing (nothing), position, genre and hold-length
   signals (nothing).
7. **Still open from the yardstick:** kick periodicity and kick share per record as pair features
   (the mixer's own measures, not computed per full track yet).

## Rerun

```
PYTHONPATH=. python scripts/experiments/pairing_yardstick.py --out reports/pairing_yardstick_<date>.csv
```
