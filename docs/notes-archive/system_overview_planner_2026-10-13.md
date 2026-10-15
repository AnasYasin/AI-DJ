# SYSTEM_OVERVIEW.md section 3, the planner, as written 2026-10-13 (superseded 2026-10-14)

Moved whole on 2026-10-15. Model A, Model B, the GBM edge scorer and the key rules it describes were deleted on 2026-10-14 and replaced by the pair model (PAIRING_EXPERIMENTS.md). Kept because it records what those models were and how they were scored.

### What is there now

**Model A, the pair embedding** (`src/models/train_model.py`). Two branches, the 1,280 effnet numbers to 256 and the
seven scalars to 64, fused to 128 on the unit sphere. The scalars are BPM over 200, the key as sine and cosine on the
Camelot circle plus its mode, energy, onset over 5, loudness. Loss NT-Xent at temperature 0.07. A positive pair is two
records played one after the other in a real set; negatives are the rest of a batch of 128, all one genre, plus semi-hard
negatives found by cosine within the genre every 5 epochs. 50 epochs. `project_tracks()` z-scores the input with
`models/encoder_norm.npz` and projects every track at load time; the vectors are never stored. Validation adjacency AUC
0.663; BPM alone gives 0.624, the raw embedding 0.622.

**Model B, the sequence model** (`src/models/train_sequence.py`). A causal transformer, 3 layers, 4 heads, width 192,
feed-forward 384, dropout 0.2, 947k parameters. Each record in the set so far is a token: its Model A vector plus four
numbers, position in the set, BPM, change in BPM, change in energy. A learned genre token is prefixed. At every position
the model outputs a 128 vector trained with InfoNCE to match the Model A vector of the record the DJ actually played
next, against the batch plus 256 random records of the same genre. Windows of up to 24 records, batches of 16 sets. Plus
17 % MRR over the context-free baseline, which is the last record's own vector.

**The GBM, the edge scorer** (`src/models/edge_scorer.py`). sklearn HistGradientBoosting, 300 iterations, learning rate
0.08, on seven pair features: absolute log BPM ratio, Camelot distance, absolute differences in energy, loudness and
onset, the raw embedding dot product, the Model A dot product. Positives are consecutive pairs in training sets,
negatives random same-genre pairs. Validation AUC 0.69.

**The search** (`plan_mix` in `src/models/predict_model.py`). Beam search, width 8. Seeds are the 8 pool records nearest
the first slot's energy target; no model in the opening record. At each slot every pool record is a candidate. Hard
vetoes first: absolute log BPM ratio over 0.05, record already used, artist already used, where
artists are split on &, x, vs, feat and pres so a collaboration blocks the solo artist. Key is not vetoed. A clash, Camelot distance over 2, is allowed on at
most half the joins and never two in a row (`clash_allowed` in `src/models/key_rules.py`). Then the score:

```
1.0 x z-score of (Model B output . candidate's Model A vector)
+ 1.0 x GBM probability
+ 0.7 x energy fit, 1 - |candidate energy - slot target|, scaled
+ compat_weight x pair compatibility (the mixer's own measure, section 6)
+ key term, +0.2 per Camelot step beyond 2 (KEY_WEIGHT -0.2, negated), so clashes are pushed up to the cap
```

The weights 1.0, 1.0, 0.7 were set once and never tuned. The beam dedupes on the set of records, so two orders of the
same six collide and ordering is never compared. The plan is a JSON with six records, each with its slot's
`energy_target01` (0 to 1, the one the mixer needs) and `target_energy` (absolute, display only).

**Training data.** The saved models were trained on 2,834 sets, 29,393 plays, scraped from 1001tracklists, with 15 % of
sets per genre held out in a split that leaked, because the same records appeared on both sides; the +17 % and the 0.663
are measured on it. Since 2026-10-13 the training scripts read `data/dataset/<date>/` (every source's plays, 65,513 of
them, folds by DJ with fold 0 held out); no model has been retrained on it yet.

**Changed since the start.** The key veto came off in September and a soft term remains, after Anas could not hear a
plus or minus one semitone shift; root readings are right about three in four, mode is unreliable in every detector. On
7 October it was decided that the chooser trains on both datasets' pairs, since it only needs order and the old set has
five times the pairs.

### Planned next

Step 4: a split by set, by DJ and by record, with the regression-set records excluded. Every number beside three
baselines, random, popularity, BPM plus key. Metrics: next-record hit at k and MRR for Model B, pair AUC for the GBM and
Model A, and the share of planned pairs that are adjacent in real sets against chance. Then Model A retrained twice on
that split, once on the full-audio rows only, once on full plus preview rows with a source column, both scored on the
same held-out full-audio seams; the higher AUC wins, a tie within 0.02 goes to the simpler. The planner switches encoder
only if adjacency AUC beats 0.663 and a render is listened to.

Step 8, later: per-DJ profiles (pool and own-music share, tempo, curve shapes, play length per role, move frequencies)
as planner priors a prompt can name. `dj_profiler.py` and `dj_profiles.json` do not exist yet. Profiles carry the DJ's
whole distribution, never a median (Anas, 2026-10-12).
