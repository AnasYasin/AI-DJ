"""Quick MuQ signal check: 5 consecutive + 5 random pairs per genre, 3 genres."""
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from muq import MuQ

PREVIEWS_DIR   = Path("data/raw/previews")
GENRES         = ["techno", "tech house", "trance"]
N_PAIRS        = 5
SEED           = 42

print("Loading MuQ …", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").eval().to(device)
print(f"Ready on {device}\n", flush=True)

df   = pd.read_parquet("data/processed/features.parquet")
tc   = pd.read_csv("data/processed/tracklist_clean.csv")
feat_ids = set(df["track_id"])
tname = tc[["track_id","artist_name"]].drop_duplicates("track_id").set_index("track_id")

# build consecutive pairs per genre
tc_sorted = tc.sort_values(["mix_id","starting_time"])
pairs_by_genre = defaultdict(list)
for _, grp in tc_sorted.groupby("mix_id"):
    genre = grp["genre"].iloc[0]
    if genre not in GENRES:
        continue
    tids = grp["track_id"].tolist()
    for i in range(len(tids)-1):
        a, b = tids[i], tids[i+1]
        if a not in feat_ids or b not in feat_ids or a == b:
            continue
        try:
            if tname.loc[a,"artist_name"] == tname.loc[b,"artist_name"]:
                continue
        except KeyError:
            pass
        if (PREVIEWS_DIR/f"{a}.m4a").exists() and (PREVIEWS_DIR/f"{b}.m4a").exists():
            pairs_by_genre[genre].append((a, b))

def embed(tid):
    y, _ = librosa.load(str(PREVIEWS_DIR/f"{tid}.m4a"), sr=24000, mono=True)
    x = torch.tensor(y).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x, output_hidden_states=False)
    return out.last_hidden_state[0].mean(0).cpu().numpy()

def cos_dist(a, b):
    a /= np.linalg.norm(a) + 1e-8
    b /= np.linalg.norm(b) + 1e-8
    return float(1 - np.dot(a, b))

rng = np.random.default_rng(SEED)
results = []

for genre in GENRES:
    pairs = pairs_by_genre[genre]
    print(f"--- {genre} ({len(pairs)} pairs available) ---", flush=True)

    # pick N_PAIRS consecutive pairs
    chosen = [pairs[i] for i in rng.choice(len(pairs), N_PAIRS, replace=False)]

    # collect all needed track IDs + extras for random baseline
    all_ids = list({tid for a,b in pairs for tid in (a,b)})
    extra   = [all_ids[i] for i in rng.choice(len(all_ids), N_PAIRS*2, replace=False)]
    to_embed = list({tid for a,b in chosen for tid in (a,b)} | set(extra))

    print(f"Embedding {len(to_embed)} tracks …", flush=True)
    emb = {}
    for k, tid in enumerate(to_embed, 1):
        t0 = time.time()
        emb[tid] = embed(tid)
        print(f"  {k}/{len(to_embed)}  {time.time()-t0:.1f}s", flush=True)

    pos_d  = [cos_dist(emb[a], emb[b]) for a,b in chosen if a in emb and b in emb]
    rand_d = []
    ids = list(emb)
    for _ in range(N_PAIRS * 5):
        i, j = rng.integers(0, len(ids), 2)
        if ids[i] != ids[j]:
            rand_d.append(cos_dist(emb[ids[i]].copy(), emb[ids[j]].copy()))
        if len(rand_d) >= N_PAIRS:
            break

    pm, rm = np.mean(pos_d), np.mean(rand_d)
    print(f"  consecutive dist: {pm:.4f}", flush=True)
    print(f"  random dist:      {rm:.4f}", flush=True)
    print(f"  ratio (rand/pos): {rm/pm:.3f}\n", flush=True)
    results.append((genre, pm, rm, rm/pm))

print("="*44)
print(f"{'Genre':<14} {'pos':>7} {'rand':>7} {'ratio':>7}")
print("-"*44)
for g, pm, rm, ratio in results:
    print(f"{g:<14} {pm:>7.4f} {rm:>7.4f} {ratio:>7.3f}")
