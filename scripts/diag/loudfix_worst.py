"""Worst case for the edge anchor: techno pair 1, whose last two bars before the cue-out dip 3-6 dB."""
import json, logging, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from src.audio import audio_mixer as am
OUT = Path("data/external/ear_test_2026-09-07/loudfix"); TRACKS = Path("data/external/test_tracks/techno"); PAD = 120.0; SR = am.SR
plan = json.loads(Path("data/external/ear_test_2026-09-07/techno/plan.json").read_text())["tracks"]
a, b = plan[0], plan[1]
paths = [TRACKS / f"{a['track_id']}.m4a", TRACKS / f"{b['track_id']}.m4a"]
tmp = OUT / "_full_techno1b.flac"; t0 = time.time()
rep = am.render_mix(paths, tmp, genre="techno", curve="arc", energy_targets=[a["energy_target01"], b["energy_target01"]])
tr = rep["transitions"][0]
def mmss(s): return f"{int(s // 60)}:{int(s % 60):02d}"
at = int(tr["at"].split(":")[0]) * 60 + int(tr["at"].split(":")[1]); end = int(tr["end"].split(":")[0]) * 60 + int(tr["end"].split(":")[1])
y, _ = sf.read(tmp); c0, c1 = max(int((at - PAD) * SR), 0), min(int((end + PAD) * SR), len(y))
fname = f"techno_pair1_edge2_{tr['type']}.flac"; sf.write(OUT / fname, y[c0:c1], SR); tmp.unlink()
bar = 4 * 60 / rep["target_bpm"]
def db(t0_, t1_): s = y[int(t0_ * SR):int(t1_ * SR)]; return round(20 * np.log10(np.sqrt((s ** 2).mean()) + 1e-12), 1)
row = {"pair": "techno_pair1_edge2", "file": fname, "A": f"{a['artist']} – {a['title']}", "B": f"{b['artist']} – {b['title']}", "type": tr["type"], "bars": tr["bars"],
       "overlap_in_clip": f"{mmss(at - c0 / SR)} – {mmss(end - c0 / SR)}", "overlap_gain_db": tr["overlap_gain_db"],
       "A_last_8_bars_db": [db(at - (8 - k) * bar, at - (7 - k) * bar) for k in range(8)],
       "overlap_first_8_bars_db": [db(at + k * bar, at + (k + 1) * bar) for k in range(8)],
       "overlap_last_4_bars_db": [db(end - (4 - k) * bar, end - (3 - k) * bar) for k in range(4)],
       "B_first_4_bars_db": [db(end + k * bar, end + (k + 1) * bar) for k in range(4)], "render_s": round(time.time() - t0)}
r = json.loads((OUT / "report.json").read_text()) if (OUT / "report.json").exists() else []
r.append(row); (OUT / "report.json").write_text(json.dumps(r, indent=1)); logging.info("DONE %s", json.dumps(row)); logging.info("WORST DONE")
