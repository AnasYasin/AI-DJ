"""Column arrays for a pool of records, shared by the planner and the pair model, and the Camelot
distance on arrays."""

import numpy as np
import pandas as pd

from src.features.transition_labeler import _CAMELOT


def camelot_arrays(keys: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Key strings → (wheel position 1-12, mode 0/1) arrays. Unknown key → pos 0."""
    pos = keys.map(_CAMELOT).fillna(0).to_numpy(dtype=np.float32)
    mode = (~keys.astype(str).str.endswith("m")).to_numpy(dtype=np.float32)
    return pos, mode


def camelot_distance(pa, ma, pb, mb) -> np.ndarray:
    """Wheel distance + 0.5 mode penalty; unknown keys (pos 0) → neutral 2.5."""
    d = np.abs(pa - pb)
    d = np.minimum(d, 12 - d) + 0.5 * (ma != mb)
    return np.where((pa == 0) | (pb == 0), 2.5, d)


class TrackArrays:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.tid = self.df["track_id"].to_numpy()
        self.idx = {t: i for i, t in enumerate(self.tid)}
        self.bpm = self.df["bpm"].to_numpy(dtype=np.float32)
        self.energy = self.df["energy_mean"].to_numpy(dtype=np.float32)
        self.loud = self.df["loudness_lufs"].to_numpy(dtype=np.float32)
        self.onset = self.df["onset_strength"].to_numpy(dtype=np.float32)
        self.cam_pos, self.cam_mode = camelot_arrays(self.df["key"])
        raw = np.stack([np.asarray(e, dtype=np.float32) for e in self.df["embedding"]])
        self.raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
