import logging
import sys

import librosa
import numpy as np

logging.disable(logging.WARNING)
sys.path.insert(0, ".")
from src.audio.audio_mixer import SR, _precise_bpm, _to_mono, load_audio
from src.data.audio_segmenter import segment


def true_bpm(y, bpm_hint):
    """Fine tempo: autocorrelation of kick-onset envelope, hop 128 (2.9 ms), parabolic peak."""
    hop = 128
    S = np.abs(librosa.stft(_to_mono(y), n_fft=1024, hop_length=hop))
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    env = np.maximum(np.diff(S[(f >= 30) & (f < 130)].sum(0), prepend=0), 0)
    env = env - env.mean()
    # autocorr via FFT, search +-6% around 4 beats (one bar) for precision
    n = len(env)
    N = 1 << (2 * n - 1).bit_length()
    ac = np.fft.irfft(np.abs(np.fft.rfft(env, N)) ** 2)[:n]
    bar = 4 * 60 / bpm_hint * SR / hop
    lo, hi = int(bar * 0.94), int(bar * 1.06)
    k = lo + int(np.argmax(ac[lo:hi]))
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return 4 * 60 / (k * hop / SR)


def downbeat_conf(y, beats):
    S = np.abs(librosa.stft(_to_mono(y), n_fft=2048, hop_length=512))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    kick = S[(f >= 20) & (f < 150)].sum(0)
    fr = np.clip((beats * SR / 512).astype(int), 0, len(kick) - 1)
    kb = kick[fr]
    phases = np.array([kb[p::4].sum() for p in range(4)])
    phases = phases / phases.max()
    return np.sort(phases)[::-1]


rows = []
for path in sys.argv[1:]:
    info = segment(path)
    y = load_audio(path)
    dr, med = info["bpm"], _precise_bpm(info)
    tb = true_bpm(y, dr)
    beats = np.array(info["beats"])
    # regression slope of beat times (uses all beats, quantisation averages out)
    idx = np.arange(len(beats))
    slope = np.polyfit(idx, beats, 1)[0]
    reg = 60 / slope
    conf = downbeat_conf(y, beats)
    print(
        f"{path.split('/')[-1][:32]:32s} deeprhythm={dr:6.1f} mixer_precise={med:7.2f} regression={reg:7.2f} autocorr_true={tb:7.2f}  "
        f"err_mixer={100 * (med - tb) / tb:+.2f}%  downbeat_phases(norm)={np.round(conf, 3)}"
    )
